"""Native Structured Decisions & System One endpoints (`/v1/decisions`, `/v1/systemone`, `/v1/decide`).

Implements AnyJev's L0 Decider natively inside FreeToken using `SamplingParams.logprob_token_ids`:
1. OpenAI Structured Decisions (`POST /v1/decisions`): `predicate`, `choice` (string or boolean values),
   and `score` (ordered `levels`) compatible with `client.decisions.create` and LiteLLM `mode: decisions`.
2. Jev / System One (`POST /v1/systemone` and `POST /v1/decide`): `noul`, `choice`, and `score`
   keyed by question name, supporting `gzip`/`deflate` request bodies and full-vocabulary
   `label_mass` / `argmax_is_label` diagnostics.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import string
import time
import uuid
import zlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from freetoken.core import SamplingParams
from freetoken.message import TokenizeMsg

from .generation import GenerationError, send_new_request
from .logprobs import reply_logprob_pairs
from .openai_api import (
    _await_watching_disconnect,
    _maintenance_gate,
    create_error_response,
)
from .request_logger import log_request

_LETTERS = string.ascii_uppercase
MAX_OPTIONS = 26
MAX_QUESTIONS = 64
MAX_BODY = 32 << 20

DEFAULT_SYSTEM = (
    "You are a decision function. You will be given a state and one question. "
    "Reply with the answer label only: no words, no punctuation, no explanation."
)
DEFAULT_PROBES: tuple[str, ...] = ("N/A", "", "[MASK]")
DEFAULT_LOG_MARGIN = 8.5
ADAPTIVE_MIN_SHIFTS = 2
ADAPTIVE_WAVE = 2
_EPS_PRIOR = 1e-8
_EPS_LOG = 1e-12
_CF_CACHE_LIMIT = 1024

# Bounded in-memory cache of content-free priors keyed by (id(tokenizer), q.key, num_perms).
_CF_PRIOR_CACHE: dict[tuple[int, str, int], list[list[float]]] = {}


class DecisionRequestError(ValueError):
    def __init__(
        self,
        message: str,
        param: str | None = None,
        status: int = 400,
        err_type: str | None = None,
        code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.param = param
        self.status = status
        self.err_type = err_type or (
            "upstream_error" if status == 502 else "invalid_request_error"
        )
        self.code = code


@dataclass(frozen=True)
class _QuestionSpec:
    kind: str  # "noul" | "choice" | "score"
    text: str
    options: tuple[str, ...]
    name: str
    scale: tuple[float, float] = (0.0, 1.0)
    ordered: bool = False
    centers: tuple[float, ...] | None = None

    @property
    def k(self) -> int:
        return len(self.options)

    @property
    def is_binary_noul(self) -> bool:
        return self.kind == "noul" and self.options == ("Yes", "No")

    @property
    def key(self) -> str:
        payload = json.dumps(
            {
                "kind": self.kind,
                "text": self.text,
                "options": list(self.options),
                "scale": list(self.scale),
                "centers": list(self.centers) if self.centers is not None else None,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def bin_centers(self) -> list[float]:
        if self.centers is not None:
            return list(self.centers)
        lo, hi = self.scale
        n = self.k
        return [lo + (hi - lo) * (i + 0.5) / n for i in range(n)]


@dataclass
class _EvaluatedDecision:
    question: _QuestionSpec
    probs: list[float]
    level: str
    label_mass: float
    argmax_is_label: bool

    @property
    def confidence(self) -> float:
        return float(max(self.probs)) if self.probs else 0.0

    @property
    def p_true(self) -> float:
        return float(self.probs[0])

    @property
    def score_value(self) -> float:
        centers = self.question.bin_centers()
        return float(sum(p * c for p, c in zip(self.probs, centers, strict=True)))


def register_decisions_routes(
    app: FastAPI,
    get_state: Callable[[], Any],
    get_model_sampling: Callable[[], dict[str, Any]] | None = None,
) -> None:
    del get_model_sampling

    @app.post("/v1/decisions")
    async def v1_decisions(request: Request):
        state = get_state()
        if (gate := _maintenance_gate(state)) is not None:
            return gate
        raw_req = await _read_json_body(request)
        if isinstance(raw_req, JSONResponse):
            return raw_req
        log_request("/v1/decisions", raw_req, request)
        return await handle_decisions(raw_req, request, state)

    @app.post("/v1/systemone")
    @app.post("/v1/decide")
    async def v1_systemone(request: Request):
        state = get_state()
        if (gate := _maintenance_gate(state)) is not None:
            return gate
        raw_req = await _read_json_body(request)
        if isinstance(raw_req, JSONResponse):
            return raw_req
        log_request(request.url.path, raw_req, request)
        return await handle_systemone(raw_req, request, state)


async def _read_json_body(request: Request) -> dict[str, Any] | JSONResponse:
    raw = await request.body()
    if len(raw) > MAX_BODY:
        return create_error_response("request body too large", status_code=413)
    enc = (request.headers.get("content-encoding") or "identity").strip().lower()
    try:
        if enc in ("gzip", "x-gzip"):
            raw = zlib.decompress(raw, 16 + zlib.MAX_WBITS)
        elif enc == "deflate":
            raw = zlib.decompress(raw, zlib.MAX_WBITS)
        elif enc not in ("", "identity"):
            return create_error_response(
                f"unsupported Content-Encoding {enc!r} (identity, gzip, deflate)",
                status_code=415,
            )
        parsed = json.loads(raw or b"{}")
    except (json.JSONDecodeError, zlib.error, UnicodeDecodeError) as exc:
        return create_error_response(f"invalid request body: {exc}", status_code=400)
    if not isinstance(parsed, dict):
        return create_error_response(
            "the request body must be a JSON object", status_code=400
        )
    return parsed


# ---- Input & Question Parsing (OpenAI Decisions + Jev / SystemOne) --------


def _extract_text(value: Any, param: str) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        out: list[str] = []
        for i, part in enumerate(value):
            kind = part.get("type") if isinstance(part, dict) else None
            if kind in ("input_text", "output_text", "text"):
                out.append(str(part.get("text", "")))
            elif kind == "input_json":
                out.append(json.dumps(part.get("data"), ensure_ascii=False))
            elif kind == "refusal":
                out.append(str(part.get("refusal", "")))
            elif kind in ("input_image", "image", "image_url"):
                raise DecisionRequestError(
                    "images are not supported by this decision model",
                    f"{param}[{i}]",
                )
            else:
                raise DecisionRequestError(
                    f"unsupported content part type {kind!r}",
                    f"{param}[{i}]",
                )
        return "\n".join(out)
    raise DecisionRequestError("must be a string or an array of content parts", param)


def _option_label(value: Any, description: Any) -> str:
    return f"{value}: {description}" if description else str(value)


def _render_state(state_val: Any) -> str:
    if state_val is None:
        return ""
    if isinstance(state_val, str):
        return state_val
    if (
        isinstance(state_val, list)
        and state_val
        and all(
            isinstance(m, dict) and "role" in m and "content" in m for m in state_val
        )
    ):
        return "\n".join(f"{m['role']}: {m['content']}" for m in state_val)
    try:
        return json.dumps(state_val, indent=2, ensure_ascii=False, default=str)
    except TypeError:
        return str(state_val)


def _parse_openai_state(value: Any, instructions: str | None = None) -> str:
    if isinstance(value, str):
        rendered = value
    elif isinstance(value, list):
        messages: list[dict[str, str]] = []
        for i, item in enumerate(value):
            if not isinstance(item, dict):
                raise DecisionRequestError(
                    "each input item must be an object", f"input[{i}]"
                )
            kind = item.get("type", "message")
            if kind == "message" or "role" in item:
                role = item.get("role")
                if role not in ("user", "developer", "system", "assistant"):
                    raise DecisionRequestError(
                        f"unsupported role {role!r}", f"input[{i}].role"
                    )
                messages.append(
                    {
                        "role": role,
                        "content": _extract_text(
                            item.get("content", ""), f"input[{i}].content"
                        ),
                    }
                )
            elif kind in ("function_call", "custom_tool_call"):
                args = item.get("arguments", item.get("input", ""))
                messages.append(
                    {
                        "role": "assistant",
                        "content": f"[tool call {item.get('name')}] {args}",
                    }
                )
            elif kind in ("function_call_output", "custom_tool_call_output"):
                out = item.get("output", "")
                messages.append(
                    {
                        "role": "user",
                        "content": f"[tool result] {_extract_text(out, f'input[{i}].output')}",
                    }
                )
            else:
                raise DecisionRequestError(
                    f"unsupported input item type {kind!r}", f"input[{i}].type"
                )
        rendered = _render_state(messages)
    else:
        raise DecisionRequestError(
            "input must be a string or an array of items", "input"
        )

    if instructions:
        inst_text = _extract_text(instructions, "instructions")
        if inst_text:
            rendered = f"{inst_text}\n\n{rendered}" if rendered else inst_text
    return rendered


def _make_choice_spec(
    text: str, labels: list[str], name: str, param: str
) -> _QuestionSpec:
    if len(labels) < 2:
        raise DecisionRequestError("choice needs at least 2 options", param)
    if len(labels) > MAX_OPTIONS:
        raise DecisionRequestError(
            f"at most {MAX_OPTIONS} options are supported here (AnyJev letter readout)",
            param,
        )
    if len(set(labels)) != len(labels):
        raise DecisionRequestError("options must be unique", param)
    return _QuestionSpec(kind="choice", text=text, options=tuple(labels), name=name)


def _parse_openai_questions(
    raw: Any,
) -> tuple[list[_QuestionSpec], list[dict[str, Any]]]:
    if not isinstance(raw, list) or not (1 <= len(raw) <= 400):
        raise DecisionRequestError(
            "questions must be an array of 1 to 400 items", "questions"
        )
    if len(raw) > MAX_QUESTIONS:
        raise DecisionRequestError(
            f"at most {MAX_QUESTIONS} questions per request are allowed here",
            "questions",
        )
    questions: list[_QuestionSpec] = []
    metas: list[dict[str, Any]] = []
    for i, q in enumerate(raw):
        p = f"questions[{i}]"
        if not isinstance(q, dict):
            raise DecisionRequestError("each question must be an object", p)
        kind = q.get("type")
        name = q.get("name") if "name" in q else q.get("id")
        raw_inst = (
            q.get("instructions") if "instructions" in q else q.get("question", "")
        )
        text = _extract_text(raw_inst, f"{p}.instructions")
        qname = str(name) if name is not None else f"q{i}"

        if kind == "predicate":
            questions.append(
                _QuestionSpec(kind="noul", text=text, options=("Yes", "No"), name=qname)
            )
            metas.append({"type": "predicate", "name": name})
        elif kind == "choice":
            if "choices" in q or "options" not in q:
                choices = q.get("choices")
                if not isinstance(choices, list) or not (2 <= len(choices) <= 255):
                    raise DecisionRequestError(
                        "choices must be an array of 2 to 255 items", f"{p}.choices"
                    )
                values = [
                    c.get("value") if isinstance(c, dict) else None for c in choices
                ]
                if any(not isinstance(v, (str, bool)) for v in values):
                    raise DecisionRequestError(
                        "each choice needs a string or boolean value", f"{p}.choices"
                    )
                if len({(type(v), v) for v in values}) != len(values):
                    raise DecisionRequestError(
                        "choice values must be unique", f"{p}.choices"
                    )
                labels = [
                    _option_label(c["value"], c.get("description")) for c in choices
                ]
                questions.append(_make_choice_spec(text, labels, qname, f"{p}.choices"))
                metas.append({"type": "choice", "name": name, "values": values})
            else:
                opts = q.get("options")
                if isinstance(opts, dict):
                    values = [str(k) for k in opts]
                    labels = [_option_label(k, opts[k]) for k in opts]
                elif isinstance(opts, list):
                    values = [v if isinstance(v, (str, bool)) else str(v) for v in opts]
                    labels = [str(v) for v in opts]
                else:
                    raise DecisionRequestError(
                        "options must be a list or object", f"{p}.choices"
                    )
                if len({(type(v), v) for v in values}) != len(values):
                    raise DecisionRequestError(
                        "choice values must be unique", f"{p}.choices"
                    )
                questions.append(_make_choice_spec(text, labels, qname, f"{p}.choices"))
                metas.append({"type": "choice", "name": name, "values": values})
        elif kind == "score":
            if "levels" in q or not any(k in q for k in ("min", "max", "scale")):
                levels = q.get("levels")
                if not isinstance(levels, list) or not (2 <= len(levels) <= 10):
                    raise DecisionRequestError(
                        "levels must be an array of 2 to 10 items", f"{p}.levels"
                    )
                if any(
                    not isinstance(lv, dict) or not isinstance(lv.get("label"), str)
                    for lv in levels
                ):
                    raise DecisionRequestError(
                        "each level needs a string label", f"{p}.levels"
                    )
                opt_labels = [
                    _option_label(lv["label"], lv.get("description")) for lv in levels
                ]
                level_names = [lv["label"] for lv in levels]
                n_lv = len(opt_labels)
                questions.append(
                    _QuestionSpec(
                        kind="score",
                        text=text,
                        options=tuple(opt_labels),
                        name=qname,
                        scale=(0.0, float(n_lv - 1)),
                        ordered=True,
                        centers=tuple(float(idx) for idx in range(n_lv)),
                    )
                )
                metas.append(
                    {
                        "type": "score",
                        "name": name,
                        "labels": level_names,
                        "values": list(range(n_lv)),
                    }
                )
            else:
                if q.get("scale") is not None:
                    sc = q["scale"]
                    if not isinstance(sc, (list, tuple)) or len(sc) != 2:
                        raise DecisionRequestError(
                            "score scale must be [min, max]", f"{p}.levels"
                        )
                    lo, hi = int(sc[0]), int(sc[1])
                else:
                    lo = int(q.get("min", 0))
                    hi = int(q.get("max", 9))
                n_lv = hi - lo + 1
                if not (2 <= n_lv <= 10):
                    raise DecisionRequestError(
                        "score range must have 2 to 10 levels", f"{p}.levels"
                    )
                anchors = dict(q.get("anchors") or {})
                opt_labels = [
                    anchors.get(str(val), f"Score {val} (range {lo} to {hi})")
                    for val in range(lo, hi + 1)
                ]
                level_names = [str(val) for val in range(lo, hi + 1)]
                vals = list(range(lo, hi + 1))
                questions.append(
                    _QuestionSpec(
                        kind="score",
                        text=text,
                        options=tuple(opt_labels),
                        name=qname,
                        scale=(float(lo), float(hi)),
                        ordered=True,
                        centers=tuple(float(val) for val in vals),
                    )
                )
                metas.append(
                    {
                        "type": "score",
                        "name": name,
                        "labels": level_names,
                        "values": vals,
                    }
                )
        else:
            raise DecisionRequestError(
                f"unsupported question type {kind!r} (predicate, choice or score)",
                f"{p}.type",
            )
    return questions, metas


def _parse_jev_questions(
    raw: Any,
) -> tuple[list[_QuestionSpec], list[dict[str, Any]]]:
    if not isinstance(raw, dict) or not raw:
        raise DecisionRequestError(
            "questions must be a non-empty object of named questions", "questions"
        )
    if len(raw) > MAX_QUESTIONS:
        raise DecisionRequestError(
            f"at most {MAX_QUESTIONS} questions per request are allowed here",
            "questions",
        )
    questions: list[_QuestionSpec] = []
    metas: list[dict[str, Any]] = []
    for name, q in raw.items():
        p = f"questions.{name}"
        if not isinstance(q, dict):
            raise DecisionRequestError("each question must be an object", p)
        kind = q.get("type")
        raw_inst = (
            q.get("instructions") if "instructions" in q else q.get("question", "")
        )
        text = _extract_text(raw_inst, f"{p}.instructions")
        criteria = q.get("criteria")
        opts_fallback = q.get("options")

        if kind == "noul":
            if criteria is None and opts_fallback is None:
                questions.append(
                    _QuestionSpec(
                        kind="noul", text=text, options=("Yes", "No"), name=name
                    )
                )
                metas.append({"type": "noul", "name": name, "binary": True})
            else:
                src = criteria if criteria is not None else opts_fallback
                if isinstance(src, dict):
                    keys = [str(k) for k in src]
                    labels = [_option_label(k, src[k]) for k in keys]
                elif isinstance(src, list):
                    keys = [str(v) for v in src]
                    labels = [str(v) for v in src]
                else:
                    raise DecisionRequestError(
                        "noul criteria/options must be an object or array",
                        f"{p}.criteria",
                    )
                keys.append("none_of_the_above")
                labels.append("none_of_the_above: None of the above")
                questions.append(_make_choice_spec(text, labels, name, f"{p}.criteria"))
                metas.append(
                    {"type": "noul", "name": name, "binary": False, "keys": keys}
                )
        elif kind == "choice":
            src = criteria if criteria is not None else opts_fallback
            if not isinstance(src, dict) or len(src) < 2:
                raise DecisionRequestError(
                    "choice criteria must map at least 2 labels to descriptions",
                    f"{p}.criteria",
                )
            keys = [str(k) for k in src]
            labels = [_option_label(k, src[k]) for k in keys]
            questions.append(_make_choice_spec(text, labels, name, f"{p}.criteria"))
            metas.append({"type": "choice", "name": name, "keys": keys})
        elif kind == "score":
            if (
                not isinstance(criteria, list)
                or not (2 <= len(criteria) <= 10)
                or not all(isinstance(c, str) for c in criteria)
            ):
                raise DecisionRequestError(
                    "score criteria must be an ordered array of 2 to 10 strings",
                    f"{p}.criteria",
                )
            n_lv = len(criteria)
            questions.append(
                _QuestionSpec(
                    kind="score",
                    text=text,
                    options=tuple(criteria),
                    name=name,
                    scale=(0.0, float(n_lv - 1)),
                    ordered=True,
                    centers=tuple(float(idx) for idx in range(n_lv)),
                )
            )
            metas.append({"type": "score", "name": name, "legend": list(criteria)})
        else:
            raise DecisionRequestError(
                f"unsupported question type {kind!r} (choice, score or noul)",
                f"{p}.type",
            )
    return questions, metas


# ---- AnyJev Readout, Calibration, and Permutation Marginalization ----------


def _softmax(scores: Sequence[float]) -> list[float]:
    if not scores:
        return []
    hi = max(scores)
    exps = [math.exp(max(float(s) - hi, -80.0)) for s in scores]
    tot = sum(exps) or 1.0
    return [e / tot for e in exps]


def _spread_order(k: int) -> list[int]:
    order, seen = [0], {0}
    d = 2
    while len(order) < k and d <= 4 * k:
        for num in range(1, d, 2):
            s = int(k * num / d) % k
            if s not in seen:
                seen.add(s)
                order.append(s)
        d *= 2
    order.extend(s for s in range(k) if s not in seen)
    return order[:k]


def _perms_for_question(
    q: _QuestionSpec, level: str, max_permutations: int | None
) -> list[list[int]]:
    if level == "raw" or q.ordered:
        return [list(range(q.k))]
    if q.is_binary_noul:
        n = 2 if max_permutations is None else max(1, min(2, max_permutations))
        return [[0, 1], [1, 0]][:n]
    # Canonical ordering by option text makes partial and full shift budgets order-invariant.
    canon = sorted(range(q.k), key=lambda i: (str(q.options[i]), i))
    n = q.k if max_permutations is None else max(1, min(q.k, max_permutations))
    return [[canon[(j + s) % q.k] for j in range(q.k)] for s in range(n)]


def _map_label_tokens(tokenizer: Any, labels: Sequence[str]) -> list[int]:
    vocab_bound = len(tokenizer)
    ids: list[int] = []
    for lab in labels:
        found: int | None = None
        for cand in (lab, " " + lab):
            toks = tokenizer.encode(cand, add_special_tokens=False)
            if len(toks) == 1 and 0 <= int(toks[0]) < vocab_bound:
                found = int(toks[0])
                break
        if found is None:
            raise ValueError(
                f"label {lab!r} is not a single token for this tokenizer; "
                f"got {tokenizer.encode(lab, add_special_tokens=False)}"
            )
        ids.append(found)
    if len(set(ids)) != len(ids):
        raise ValueError(f"label tokens collide: {dict(zip(labels, ids, strict=True))}")
    return ids


def _resolve_labels_sync(
    tokenizer: Any, q: _QuestionSpec
) -> tuple[list[str], list[int]]:
    if q.is_binary_noul:
        labels = ["Yes", "No"]
        try:
            return labels, _map_label_tokens(tokenizer, labels)
        except ValueError:
            letters = list(_LETTERS[: q.k])
            return letters, _map_label_tokens(tokenizer, letters)
    if q.kind == "score":
        digits = [str(i + 1) for i in range(q.k)]
        try:
            return digits, _map_label_tokens(tokenizer, digits)
        except ValueError:
            letters = list(_LETTERS[: q.k])
            return letters, _map_label_tokens(tokenizer, letters)
    letters = list(_LETTERS[: q.k])
    return letters, _map_label_tokens(tokenizer, letters)


def _label_ids_for_perm(
    q: _QuestionSpec, ids: Sequence[int], perm: Sequence[int], labels: Sequence[str]
) -> list[int]:
    # Non-positional only when Yes/No words themselves move with `perm`
    if q.is_binary_noul and list(labels) == ["Yes", "No"]:
        return [int(ids[i]) for i in perm]
    return [int(x) for x in ids]


def _build_prompt_messages(
    state_text: str, q: _QuestionSpec, perm: Sequence[int], labels: Sequence[str]
) -> list[dict[str, str]]:
    prefix = [
        "State:",
        state_text if state_text else "(empty)",
        "",
        f"Question: {q.text}",
    ]
    suffix: list[str] = []
    if q.is_binary_noul and list(labels) == ["Yes", "No"]:
        order = [labels[i] for i in perm]
        suffix.append(f"Answer {order[0]} or {order[1]}.")
    elif q.is_binary_noul:
        prefix.append("Options:")
        for j, i in enumerate(perm):
            suffix.append(f"{labels[j]}. {q.options[i]}")
        suffix.append("Answer with the letter only.")
    elif q.kind == "score":
        if q.centers is not None:
            prefix.append("Pick the level that applies (the levels are ordered):")
        else:
            lo, hi = q.scale
            prefix.append(
                f"Answer on a scale from {lo:g} to {hi:g} by picking the closest bin:"
            )
        for j, i in enumerate(perm):
            suffix.append(f"{labels[j]}. {q.options[i]}")
        suffix.append(
            "Answer with the number only."
            if labels[0].isdigit()
            else "Answer with the letter only."
        )
    else:
        prefix.append("Options:")
        for j, i in enumerate(perm):
            suffix.append(f"{labels[j]}. {q.options[i]}")
        suffix.append("Answer with the letter only.")
    user_body = "\n".join(prefix) + "\n" + "\n".join(suffix)
    return [
        {"role": "system", "content": DEFAULT_SYSTEM},
        {"role": "user", "content": user_body},
    ]


def _content_free_prior(probe_probs: Sequence[Sequence[float]]) -> list[float]:
    k = len(probe_probs[0])
    n_c = max(1, len(probe_probs))
    mean = [max(sum(row[j] for row in probe_probs) / n_c, _EPS_PRIOR) for j in range(k)]
    tot = sum(mean) or 1.0
    return [v / tot for v in mean]


def _apply_contextual(
    p_raw: Sequence[float], prior: Sequence[float] | None
) -> list[float]:
    if prior is None:
        return list(p_raw)
    ratio = [
        max(float(p) / max(float(pr), _EPS_PRIOR), _EPS_PRIOR)
        for p, pr in zip(p_raw, prior, strict=True)
    ]
    tot = sum(ratio) or 1.0
    return [v / tot for v in ratio]


def _marginalize_logmean(
    p_by_perm: Sequence[Sequence[float]], perms: Sequence[Sequence[int]], k: int
) -> list[float]:
    n_p = max(1, len(perms))
    mean_log = [0.0] * k
    for p_pos, perm in zip(p_by_perm, perms, strict=True):
        for j, opt_idx in enumerate(perm):
            mean_log[opt_idx] += math.log(max(float(p_pos[j]), _EPS_LOG))
    mean_log = [v / n_p for v in mean_log]
    return _softmax(mean_log)


def _log_margin(marginal: Sequence[float]) -> float:
    if len(marginal) < 2:
        return float("inf")
    ordered = sorted((float(x) for x in marginal), reverse=True)
    return math.log(max(ordered[0], _EPS_LOG)) - math.log(max(ordered[1], _EPS_LOG))


async def _run_batch_label_reads(
    state: Any,
    request: Request | None,
    prompts_and_targets: list[tuple[list[dict[str, str]], list[int]]],
) -> tuple[list[tuple[int, float, dict[int, float]]], int, int, int] | None:
    if not prompts_and_targets:
        return [], 0, 0, 0
    uids: list[int] = []
    ctk = {
        "enable_thinking": False,
        "thinking": False,
        "thinking_mode": "chat",
    }
    for messages, target_ids in prompts_and_targets:
        uid = state.new_user()
        sp = SamplingParams(
            temperature=0.0,
            max_tokens=1,
            logprobs=True,
            top_logprobs=min(len(target_ids) + 1, 32),
            logprob_token_ids=list(target_ids),
        )
        msg = TokenizeMsg(
            uid=uid,
            text=messages,
            sampling_params=sp,
            chat_template_kwargs=ctk,
        )
        try:
            await send_new_request(state, uid, msg)
        except Exception:
            if uids and hasattr(state, "abort_user"):
                await asyncio.gather(
                    *(state.abort_user(u) for u in uids), return_exceptions=True
                )
            raise
        uids.append(uid)

    prompt_tokens = 0
    completion_tokens = 0
    cached_tokens = 0

    async def _drain_single(u: int) -> tuple[int, float, dict[int, float]]:
        nonlocal prompt_tokens, completion_tokens, cached_tokens
        sampled_id = -1
        sampled_lp = -9999.0
        lp_map: dict[int, float] = {}
        async for ack in state.wait_for_ack(u):
            if getattr(ack, "error", None):
                raise GenerationError(ack.error, getattr(ack, "error_code", None))
            prompt_tokens += ack.prompt_tokens_delta
            completion_tokens += ack.completion_tokens_delta
            cached_tokens += ack.cached_tokens
            for entry, _ in reply_logprob_pairs(ack):
                sampled_id = int(entry["token_id"])
                sampled_lp = float(entry["logprob"])
                lp_map[sampled_id] = sampled_lp
                for cand in entry.get("top") or ():
                    cid = int(cand["token_id"])
                    clp = float(cand["logprob"])
                    if cid not in lp_map or clp > lp_map[cid]:
                        lp_map[cid] = clp
            if ack.finished:
                break
        return sampled_id, sampled_lp, lp_map

    gathered = await _await_watching_disconnect(
        asyncio.gather(*(_drain_single(u) for u in uids)),
        request,
        state,
        uids,
    )
    if gathered is None:
        return None
    return list(gathered), prompt_tokens, completion_tokens, cached_tokens


async def _evaluate_specs(
    state: Any,
    request: Request | None,
    state_text: str,
    questions: list[_QuestionSpec],
    *,
    calibration: str = "content_free",
    level: str = "L0",
    rotations_override: int | None = None,
) -> tuple[list[_EvaluatedDecision], dict[str, int]] | JSONResponse:
    build_tok = getattr(state, "frontend_tokenizer", None)
    if build_tok is None:
        return create_error_response(
            "Structured decisions require a server tokenizer",
            status_code=503,
            err_type="server_error",
        )
    manager = await asyncio.to_thread(build_tok)
    tokenizer = manager.tokenizer
    tok_id = id(tokenizer)

    resolved = await asyncio.to_thread(
        lambda: [_resolve_labels_sync(tokenizer, q) for q in questions]
    )

    total_pt = 0
    total_ct = 0
    total_cached = 0

    want_cf = level == "L0" and calibration == "content_free"
    want_ctx = level == "L0" and calibration == "contextual"

    # Build per-question execution plans
    q_plans: list[dict[str, Any]] = []
    cf_specs: list[tuple[list[dict[str, str]], list[int]]] = []
    cf_pending: list[tuple[tuple[int, str, int], int, int, int]] = []

    for q, (labels, base_ids) in zip(questions, resolved, strict=True):
        perms = _perms_for_question(q, level, rotations_override)
        perm_ids = [_label_ids_for_perm(q, base_ids, p, labels) for p in perms]
        cache_key = (tok_id, q.key, len(perms))
        if want_cf and cache_key not in _CF_PRIOR_CACHE:
            start_idx = len(cf_specs)
            for perm, pids in zip(perms, perm_ids, strict=True):
                for probe in DEFAULT_PROBES:
                    cf_specs.append(
                        (_build_prompt_messages(probe, q, perm, labels), pids)
                    )
            cf_pending.append((cache_key, start_idx, len(perms), len(DEFAULT_PROBES)))

        adaptive = (
            level == "L0"
            and rotations_override is None
            and q.kind == "choice"
            and not q.ordered
            and q.k >= 3
        )
        shift_order = _spread_order(len(perms)) if adaptive else list(range(len(perms)))
        q_plans.append(
            {
                "q": q,
                "labels": labels,
                "perms": perms,
                "perm_ids": perm_ids,
                "cache_key": cache_key,
                "adaptive": adaptive,
                "shift_order": shift_order,
                "used_shifts": [],
                "p_pos_raw": [],
                "masses": [],
                "argmax_flags": [],
                "ctx_prior": None,
                "decided": False,
            }
        )

    # 1. Populate uncached content-free priors in a single batch (matching AnyJev's separate CF batch)
    if cf_specs:
        cf_out = await _run_batch_label_reads(state, request, cf_specs)
        if cf_out is None:
            return create_error_response(
                "client disconnected before the response was ready",
                status_code=499,
                err_type="client_disconnected",
            )
        cf_reads, pt, ct, cached = cf_out
        total_pt += pt
        total_ct += ct
        total_cached += cached
        for cache_key, start_idx, n_perms, n_probes in cf_pending:
            priors_by_perm: list[list[float]] = []
            for pi in range(n_perms):
                probe_rows: list[list[float]] = []
                for ci in range(n_probes):
                    read_idx = start_idx + pi * n_probes + ci
                    _, _, lp_map = cf_reads[read_idx]
                    pids = cf_specs[read_idx][1]
                    probe_rows.append(
                        _softmax([lp_map.get(tid, -9999.0) for tid in pids])
                    )
                priors_by_perm.append(_content_free_prior(probe_rows))
            if len(_CF_PRIOR_CACHE) >= _CF_CACHE_LIMIT:
                _CF_PRIOR_CACHE.pop(next(iter(_CF_PRIOR_CACHE)))
            _CF_PRIOR_CACHE[cache_key] = priors_by_perm

    def _prior_for(plan: dict[str, Any], sidx: int) -> list[float] | None:
        if want_cf:
            return _CF_PRIOR_CACHE[plan["cache_key"]][sidx]
        if want_ctx and plan["ctx_prior"] is not None:
            return plan["ctx_prior"][sidx]
        return None

    def _corrected_rows(plan: dict[str, Any]) -> list[list[float]]:
        return [
            _apply_contextual(p_raw, _prior_for(plan, sidx))
            for sidx, p_raw in zip(plan["used_shifts"], plan["p_pos_raw"], strict=True)
        ]

    # Optional contextual prior (empty-question probe on the given state) if explicitly requested
    if want_ctx:
        ctx_specs: list[tuple[list[dict[str, str]], list[int]]] = []
        ctx_slices: list[tuple[int, int]] = []
        for plan in q_plans:
            q_empty = _QuestionSpec(
                kind=plan["q"].kind,
                text="",
                options=tuple("[Option]" for _ in range(plan["q"].k))
                if not plan["q"].is_binary_noul
                else ("Yes", "No"),
                name=plan["q"].name,
                scale=plan["q"].scale,
                ordered=plan["q"].ordered,
                centers=plan["q"].centers,
            )
            st = len(ctx_specs)
            for perm, pids in zip(plan["perms"], plan["perm_ids"], strict=True):
                ctx_specs.append(
                    (
                        _build_prompt_messages(
                            state_text, q_empty, perm, plan["labels"]
                        ),
                        pids,
                    )
                )
            ctx_slices.append((st, len(ctx_specs)))
        ctx_out = await _run_batch_label_reads(state, request, ctx_specs)
        if ctx_out is None:
            return create_error_response(
                "client disconnected before the response was ready",
                status_code=499,
                err_type="client_disconnected",
            )
        ctx_reads, pt, ct, cached = ctx_out
        total_pt += pt
        total_ct += ct
        total_cached += cached
        for plan, (st, en) in zip(q_plans, ctx_slices, strict=True):
            plan["ctx_prior"] = [
                _softmax(
                    [ctx_reads[idx][2].get(tid, -9999.0) for tid in ctx_specs[idx][1]]
                )
                for idx in range(st, en)
            ]

    # 2. Run real state reads across all questions in synchronized waves:
    #    Wave 0 runs ALL non-adaptive reads + the first ADAPTIVE_WAVE shifts of adaptive choices together!
    max_shifts = max((len(p["shift_order"]) for p in q_plans), default=0)
    wave_cursor = 0
    while wave_cursor < max_shifts:
        wave_specs: list[tuple[list[dict[str, str]], list[int]]] = []
        wave_owners: list[tuple[int, int]] = []

        for qi, plan in enumerate(q_plans):
            if plan["decided"]:
                continue
            order: list[int] = plan["shift_order"]
            if not plan["adaptive"]:
                if wave_cursor > 0:
                    continue
                batch_shifts = order
            else:
                if wave_cursor >= ADAPTIVE_MIN_SHIFTS and plan["used_shifts"]:
                    pc = _corrected_rows(plan)
                    used_perms = [plan["perms"][s] for s in plan["used_shifts"]]
                    marg = _marginalize_logmean(pc, used_perms, plan["q"].k)
                    if _log_margin(marg) >= DEFAULT_LOG_MARGIN:
                        plan["decided"] = True
                        continue
                batch_shifts = order[wave_cursor : wave_cursor + ADAPTIVE_WAVE]

            for sidx in batch_shifts:
                perm = plan["perms"][sidx]
                pids = plan["perm_ids"][sidx]
                wave_specs.append(
                    (
                        _build_prompt_messages(
                            state_text, plan["q"], perm, plan["labels"]
                        ),
                        pids,
                    )
                )
                wave_owners.append((qi, sidx))

        if not wave_specs:
            break

        wave_out = await _run_batch_label_reads(state, request, wave_specs)
        if wave_out is None:
            return create_error_response(
                "client disconnected before the response was ready",
                status_code=499,
                err_type="client_disconnected",
            )
        wave_reads, pt, ct, cached = wave_out
        total_pt += pt
        total_ct += ct
        total_cached += cached

        for (qi, sidx), (sampled_id, _, lp_map) in zip(
            wave_owners, wave_reads, strict=True
        ):
            plan = q_plans[qi]
            pids = plan["perm_ids"][sidx]
            raw_lps = [lp_map.get(tid, -9999.0) for tid in pids]
            mass = min(1.0, sum(math.exp(max(lp, -80.0)) for lp in raw_lps))
            plan["used_shifts"].append(sidx)
            plan["p_pos_raw"].append(_softmax(raw_lps))
            plan["masses"].append(mass)
            plan["argmax_flags"].append(sampled_id in set(pids))

        wave_cursor += ADAPTIVE_WAVE

    # 3. Assemble final calibrated distributions
    results: list[_EvaluatedDecision] = []
    achieved_level = "L0" if level == "L0" else "raw"
    for plan in q_plans:
        q: _QuestionSpec = plan["q"]
        used_perms = [plan["perms"][s] for s in plan["used_shifts"]]
        if level == "raw":
            probs = _marginalize_logmean(plan["p_pos_raw"][:1], used_perms[:1], q.k)
        else:
            pc = _corrected_rows(plan)
            probs = _marginalize_logmean(pc, used_perms, q.k)
        mean_mass = sum(plan["masses"]) / max(1, len(plan["masses"]))
        results.append(
            _EvaluatedDecision(
                question=q,
                probs=probs,
                level=achieved_level,
                label_mass=round(mean_mass, 6),
                argmax_is_label=all(plan["argmax_flags"]),
            )
        )

    usage = {
        "input_tokens": total_pt,
        "completion_tokens": total_ct,
        "cached_tokens": min(total_cached, total_pt),
    }
    return results, usage


# ---- Route Handlers --------------------------------------------------------


def _format_openai_answer(
    dec: _EvaluatedDecision, meta: dict[str, Any]
) -> dict[str, Any]:
    probs = [float(x) for x in dec.probs]
    kind = meta["type"]
    if kind == "predicate":
        return {
            "type": "predicate",
            "name": meta["name"],
            "probability": dec.p_true,
        }
    if kind == "choice":
        best = max(range(len(probs)), key=probs.__getitem__)
        return {
            "type": "choice",
            "name": meta["name"],
            "choice": meta["values"][best],
            "confidence": dec.confidence,
            "probabilities": [
                {"value": v, "probability": p}
                for v, p in zip(meta["values"], probs, strict=True)
            ],
        }
    return {
        "type": "score",
        "name": meta["name"],
        "score": dec.score_value,
        "confidence": dec.confidence,
        "probabilities": [
            {"value": val, "label": lb, "probability": p}
            for val, lb, p in zip(meta["values"], meta["labels"], probs, strict=True)
        ],
    }


def _format_jev_answer(dec: _EvaluatedDecision, meta: dict[str, Any]) -> dict[str, Any]:
    probs = [float(x) for x in dec.probs]
    kind = meta["type"]
    if kind == "noul" and meta.get("binary", True):
        return {"type": "noul", "noul": dec.p_true}
    if kind in ("choice", "noul"):
        keys: list[str] = meta["keys"]
        best = max(range(len(probs)), key=probs.__getitem__)
        return {
            "type": kind,
            "choice": keys[best],
            "confidence": dec.confidence,
            "probabilities": dict(zip(keys, probs, strict=True)),
        }
    return {
        "type": "score",
        "score": dec.score_value,
        "confidence": dec.confidence,
        "probabilities": {str(i): p for i, p in enumerate(probs)},
        "legend": {str(i): label for i, label in enumerate(meta["legend"])},
    }


def _resolve_exec_knobs(raw_req: dict[str, Any]) -> tuple[str, str, int | None]:
    calibration = str(raw_req.get("calibration") or "content_free")
    if calibration not in ("content_free", "contextual", "none"):
        raise DecisionRequestError(
            f"unsupported calibration {calibration!r} (content_free, contextual, none)",
            "calibration",
        )
    level = str(raw_req.get("level") or "L0")
    if level not in ("L0", "raw"):
        raise DecisionRequestError(f"unsupported level {level!r} (L0, raw)", "level")
    rot_raw = raw_req.get("rotations")
    rotations: int | None = None
    if rot_raw is not None:
        if (
            isinstance(rot_raw, bool)
            or not isinstance(rot_raw, int)
            or not (1 <= rot_raw <= 26)
        ):
            raise DecisionRequestError(
                "rotations must be an integer between 1 and 26", "rotations"
            )
        rotations = rot_raw
    return calibration, level, rotations


def _served_model_name(raw_req: dict[str, Any], state: Any) -> str:
    req_model = raw_req.get("model")
    if isinstance(req_model, str) and req_model.strip():
        return req_model
    return (
        getattr(state.config, "served_model_name", None)
        or getattr(state.config, "model_path", None)
        or "default"
    )


async def handle_decisions(
    raw_req: dict[str, Any], request: Request | None, state: Any
) -> JSONResponse:
    try:
        if raw_req.get("stream"):
            raise DecisionRequestError(
                "streaming is not supported for decisions", "stream"
            )
        if "input" not in raw_req:
            raise DecisionRequestError("input is required", "input")
        calibration, level, rotations = _resolve_exec_knobs(raw_req)
        state_text = _parse_openai_state(raw_req["input"], raw_req.get("instructions"))
        questions, metas = _parse_openai_questions(raw_req.get("questions"))
        evaluated = await _evaluate_specs(
            state,
            request,
            state_text,
            questions,
            calibration=calibration,
            level=level,
            rotations_override=rotations,
        )
    except DecisionRequestError as exc:
        return create_error_response(
            str(exc),
            status_code=exc.status,
            err_type=exc.err_type,
            param=exc.param,
            code=exc.code,
        )
    except GenerationError as exc:
        code = getattr(exc, "code", None)
        status = 400 if code == "invalid_request_error" else 502
        err_type = "invalid_request_error" if status == 400 else "upstream_error"
        return create_error_response(
            f"decision backend failed: {exc}",
            status_code=status,
            err_type=err_type,
            code=code,
        )
    except (ValueError, TypeError, RuntimeError) as exc:
        return create_error_response(
            f"decision backend failed: {exc}",
            status_code=502,
            err_type="upstream_error",
        )

    if isinstance(evaluated, JSONResponse):
        return evaluated
    decisions, usage_raw = evaluated
    in_tok = usage_raw["input_tokens"]
    cached_tok = usage_raw["cached_tokens"]
    levels = [d.level for d in decisions]
    body = {
        "id": f"dec_{uuid.uuid4().hex}",
        "model": _served_model_name(raw_req, state),
        "answers": [
            _format_openai_answer(d, m) for d, m in zip(decisions, metas, strict=True)
        ],
        "usage": {
            "input_tokens": in_tok,
            "input_tokens_details": {
                "cached_tokens": cached_tok,
                "cache_write_tokens": 0,
            },
            "output_tokens": 0,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": in_tok,
        },
    }
    return JSONResponse(
        content=body,
        status_code=200,
        headers={"X-AnyJev-Levels": ",".join(levels)},
    )


async def handle_systemone(
    raw_req: dict[str, Any], request: Request | None, state: Any
) -> JSONResponse:
    try:
        if raw_req.get("stream"):
            raise DecisionRequestError(
                "streaming is not supported for decisions", "stream"
            )
        if "state" not in raw_req:
            raise DecisionRequestError("state is required", "state")
        for media_key in ("assets", "images", "audio", "videos"):
            if raw_req.get(media_key):
                raise DecisionRequestError(
                    f"{media_key} are not supported by this decision model",
                    media_key,
                )
        calibration, level, rotations = _resolve_exec_knobs(raw_req)
        state_text = _render_state(raw_req["state"])
        if raw_req.get("instructions"):
            inst = _extract_text(raw_req["instructions"], "instructions")
            if inst:
                state_text = f"{inst}\n\n{state_text}" if state_text else inst
        questions, metas = _parse_jev_questions(raw_req.get("questions"))
        evaluated = await _evaluate_specs(
            state,
            request,
            state_text,
            questions,
            calibration=calibration,
            level=level,
            rotations_override=rotations,
        )
    except DecisionRequestError as exc:
        return create_error_response(
            str(exc),
            status_code=exc.status,
            err_type=exc.err_type,
            param=exc.param,
            code=exc.code,
        )
    except GenerationError as exc:
        code = getattr(exc, "code", None)
        status = 400 if code == "invalid_request_error" else 502
        err_type = "invalid_request_error" if status == 400 else "upstream_error"
        return create_error_response(
            f"decision backend failed: {exc}",
            status_code=status,
            err_type=err_type,
            code=code,
        )
    except (ValueError, TypeError, RuntimeError) as exc:
        return create_error_response(
            f"decision backend failed: {exc}",
            status_code=502,
            err_type="upstream_error",
        )

    if isinstance(evaluated, JSONResponse):
        return evaluated
    decisions, usage_raw = evaluated
    in_tok = usage_raw["input_tokens"]
    levels = [d.level for d in decisions]
    answers = {
        m["name"]: _format_jev_answer(d, m)
        for d, m in zip(decisions, metas, strict=True)
    }
    diagnostics = {
        m["name"]: {
            "label_mass": d.label_mass,
            "argmax_is_label": d.argmax_is_label,
        }
        for d, m in zip(decisions, metas, strict=True)
    }
    body: dict[str, Any] = {
        "id": f"dec_{uuid.uuid4().hex}",
        "object": "structured_decision",
        "created": int(time.time()),
        "model": _served_model_name(raw_req, state),
        "answers": answers,
        "usage": {
            "input_tokens": in_tok,
            "output_tokens": 0,
            "cost_usd": 0.0,
        },
        "diagnostics": diagnostics,
    }
    return JSONResponse(
        content=body,
        status_code=200,
        headers={"X-AnyJev-Levels": ",".join(levels)},
    )
