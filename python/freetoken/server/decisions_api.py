"""Native Structured Decisions & System One endpoints (`/v1/decisions`, `/v1/systemone`, `/v1/decide`).

Implements both:
1. OpenAI Structured Decisions (`POST /v1/decisions`): ``predicate``, ``choice``, and ``score`` questions
   with AnyJev L0 cyclic-shift permutation marginalization and contextual/content-free prior calibration.
2. Jev / vLLM / llama.cpp System One (`POST /v1/systemone` and `POST /v1/decide`): ``choice``, ``score``,
   ``noul``, and ``predicate`` questions with full-vocabulary ``label_mass`` and ``argmax_is_label``
   diagnostics powered by ``SamplingParams.logprob_token_ids``.
"""

from __future__ import annotations

import asyncio
import math
import time
import uuid
from collections.abc import Callable
from typing import Any, Literal

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from freetoken.core import SamplingParams
from freetoken.message import TokenizeMsg

from .generation import GenerationError, render_messages, send_new_request
from .logprobs import reply_logprob_pairs
from .openai_api import (
    _await_watching_disconnect,
    _maintenance_gate,
    create_error_response,
)
from .request_logger import log_request

_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_MAX_QUESTIONS = 128


class DecisionQuestion(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str | None = None
    type: Literal["predicate", "choice", "score", "noul"] = "choice"
    question: str | None = None
    instructions: str | None = None
    options: list[str] | dict[str, str] | None = None
    criteria: dict[str, str] | None = None
    min: int = 0
    max: int = 10
    scale: list[int] | tuple[int, int] | None = None
    anchors: dict[str, str] | None = None


class DecisionsRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    input: str | list[dict[str, Any]]
    instructions: str | None = None
    questions: list[DecisionQuestion] = Field(min_length=1, max_length=_MAX_QUESTIONS)
    calibration: Literal["contextual", "content_free", "none"] = "contextual"
    rotations: int | None = Field(default=None, ge=1, le=26)


class SystemOneRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str | None = None
    state: str | list[dict[str, Any]]
    instructions: str | None = None
    questions: dict[str, DecisionQuestion]
    calibration: Literal["contextual", "content_free", "none"] = "contextual"
    rotations: int | None = Field(default=None, ge=1, le=26)
    diagnostics: bool = False


def register_decisions_routes(
    app: FastAPI,
    get_state: Callable[[], Any],
    get_model_sampling: Callable[[], dict[str, Any]] | None = None,
) -> None:
    del get_model_sampling

    @app.post("/v1/decisions")
    async def v1_decisions(req: DecisionsRequest, request: Request):
        log_request("/v1/decisions", req, request)
        state = get_state()
        if (gate := _maintenance_gate(state)) is not None:
            return gate
        return await handle_decisions(req, request, state)

    @app.post("/v1/systemone")
    @app.post("/v1/decide")
    async def v1_systemone(req: SystemOneRequest, request: Request):
        log_request("/v1/systemone", req, request)
        state = get_state()
        if (gate := _maintenance_gate(state)) is not None:
            return gate
        return await handle_systemone(req, request, state)


def _normalize_state_messages(
    raw_input: str | list[dict[str, Any]], instructions: str | None
) -> list[dict[str, Any]]:
    msgs: list[dict[str, Any]] = []
    if instructions:
        msgs.append({"role": "system", "content": instructions})
    if isinstance(raw_input, str):
        if not raw_input.strip() and not instructions:
            raise ValueError("input/state must not be empty")
        msgs.append({"role": "user", "content": raw_input})
        return render_messages(msgs)
    if not raw_input:
        raise ValueError("input/state must not be empty")
    for item in raw_input:
        if not isinstance(item, dict):
            raise TypeError("input items must be objects")
        role = item.get("role", "user")
        if role == "developer":
            role = "system"
        content = item.get("content")
        if isinstance(content, list):
            parts = []
            for p in content:
                if isinstance(p, dict) and p.get("type") in (
                    "input_text",
                    "output_text",
                ):
                    parts.append({"type": "text", "text": p.get("text") or ""})
                else:
                    parts.append(p)
            content = parts
        msgs.append({**item, "role": role, "content": content})
    return render_messages(msgs)


def _softmax(scores: list[float]) -> list[float]:
    if not scores:
        return []
    hi = max(scores)
    exps = [math.exp(max(s - hi, -80.0)) for s in scores]
    tot = sum(exps) or 1.0
    return [e / tot for e in exps]


def _cyclic_shift(items: list[tuple[str, str]], offset: int) -> list[tuple[str, str]]:
    k = len(items)
    if k <= 1:
        return list(items)
    off = offset % k
    return items[off:] + items[:off]


async def _resolve_label_token_ids(
    tokenizer: Any, labels: list[str]
) -> dict[str, list[int]]:
    """Resolve single-token IDs (both bare ``'A'`` and space-prefixed ``' A'``) for each label."""

    def _lookup() -> dict[str, list[int]]:
        vocab_bound = len(tokenizer)
        out: dict[str, list[int]] = {}
        for lbl in labels:
            cands: list[int] = []
            for variant in (lbl, f" {lbl}"):
                ids = tokenizer.encode(variant, add_special_tokens=False)
                if len(ids) == 1 and 0 <= int(ids[0]) < vocab_bound:
                    tid = int(ids[0])
                    if tid not in cands:
                        cands.append(tid)
            if not cands:
                # Fallback: first token of bare encoding
                ids = tokenizer.encode(lbl, add_special_tokens=False)
                if not ids or not (0 <= int(ids[0]) < vocab_bound):
                    raise ValueError(
                        f"Could not resolve single token id for label {lbl!r}"
                    )
                cands.append(int(ids[0]))
            out[lbl] = cands
        return out

    return await asyncio.to_thread(_lookup)


def _build_question_options(
    q: DecisionQuestion, *, default_score_max: int = 10
) -> tuple[str, list[tuple[str, str]], list[float] | None]:
    """Return ``(q_kind, [(option_key, option_text), ...], numeric_values_or_None)``."""
    qtype = q.type
    if qtype == "predicate":
        return (
            "predicate",
            [("true", "Yes / True"), ("false", "No / False")],
            [1.0, 0.0],
        )

    if qtype == "noul":
        base: list[tuple[str, str]] = []
        if isinstance(q.options, dict):
            base = [(str(k), str(v)) for k, v in q.options.items()]
        elif isinstance(q.options, list):
            base = [(str(v), str(v)) for v in q.options]
        elif isinstance(q.criteria, dict):
            base = [(str(k), str(v)) for k, v in q.criteria.items()]
        base.append(
            ("none_of_the_above", "None of the above / insufficient information")
        )
        if len(base) < 2 or len(base) > len(_LETTERS):
            raise ValueError(f"noul question requires 1..{len(_LETTERS) - 1} options")
        return "noul", base, None

    if qtype == "choice":
        opts: list[tuple[str, str]] = []
        if isinstance(q.options, dict):
            opts = [(str(k), str(v)) for k, v in q.options.items()]
        elif isinstance(q.options, list):
            opts = [(str(v), str(v)) for v in q.options]
        elif isinstance(q.criteria, dict):
            opts = [(str(k), str(v)) for k, v in q.criteria.items()]
        if len(opts) < 2:
            raise ValueError("choice question requires at least 2 options")
        if len(opts) > len(_LETTERS):
            raise ValueError(
                f"choice question supports at most {len(_LETTERS)} options"
            )
        return "choice", opts, None

    if qtype == "score":
        if q.scale is not None:
            if len(q.scale) != 2:
                raise ValueError("score scale must be [min, max]")
            lo, hi = int(q.scale[0]), int(q.scale[1])
        else:
            lo = int(q.min)
            hi = int(q.max) if q.max is not None else default_score_max
        if hi <= lo:
            raise ValueError(f"score max ({hi}) must exceed min ({lo})")
        if hi - lo + 1 > len(_LETTERS):
            raise ValueError(
                f"score range [{lo}, {hi}] has {hi - lo + 1} levels (max {len(_LETTERS)})"
            )
        anchors = dict(q.anchors or q.criteria or {})
        levels: list[tuple[str, str]] = []
        nums: list[float] = []
        for val in range(lo, hi + 1):
            sval = str(val)
            desc = anchors.get(sval, f"Score {val} out of {hi} (range {lo}..{hi})")
            levels.append((sval, desc))
            nums.append(float(val))
        return "score", levels, nums

    raise ValueError(f"Unsupported question type: {qtype!r}")


def _format_read_prompt_suffix(
    question_text: str,
    rotated_options: list[tuple[str, str]],
    letters: list[str],
    *,
    neutral: bool = False,
) -> tuple[str, str]:
    """Return ``(user_question_block, assistant_prefill)``."""
    if neutral:
        lines = ["Select one option label:"]
        for letter in letters:
            lines.append(f"{letter}) [Option]")
        lines.append(f"Reply with only a single letter ({', '.join(letters)}).")
        return "\n".join(lines), "Answer: "
    lines = []
    if question_text.strip():
        lines.append(f"Question: {question_text.strip()}")
    lines.append("Options:")
    for letter, (_, desc) in zip(letters, rotated_options, strict=True):
        lines.append(f"{letter}) {desc}")
    lines.append(
        f"Reply with only the single letter ({', '.join(letters)}) of the best option."
    )
    return "\n".join(lines), "Answer: "


def _append_probe_to_messages(
    base_messages: list[dict[str, Any]], user_suffix: str, assistant_prefill: str
) -> list[dict[str, Any]]:
    copied = [dict(m) for m in base_messages]
    if (
        copied
        and copied[-1].get("role") == "user"
        and isinstance(copied[-1].get("content"), str)
    ):
        prev = copied[-1]["content"]
        copied[-1]["content"] = f"{prev}\n\n{user_suffix}" if prev else user_suffix
    else:
        copied.append({"role": "user", "content": user_suffix})
    copied.append({"role": "assistant", "content": assistant_prefill})
    return copied


async def _run_batch_label_reads(
    state: Any,
    request: Request | None,
    prompts_and_targets: list[tuple[list[dict[str, Any]], list[int]]],
) -> tuple[list[tuple[int, float, dict[int, float]]], int, int, int] | None:
    """Submit all label reads together and return ``([(sampled_id, sampled_lp, {tid: lp}), ...], pt, ct, cached)``."""
    uids: list[int] = []
    ctk = {
        "enable_thinking": False,
        "thinking": False,
        "thinking_mode": "chat",
        "continue_final_message": True,
    }
    for messages, target_ids in prompts_and_targets:
        uid = state.new_user()
        sp = SamplingParams(
            temperature=0.0,
            max_tokens=1,
            logprobs=True,
            top_logprobs=min(len(target_ids) + 1, 32),
            logprob_token_ids=target_ids,
        )
        msg = TokenizeMsg(
            uid=uid,
            text=messages,
            sampling_params=sp,
            chat_template_kwargs=ctk,
        )
        try:
            await send_new_request(state, uid, msg)
        except GenerationError:
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


def _logsumexp(vals: list[float]) -> float:
    if not vals:
        return -9999.0
    m = max(vals)
    if not math.isfinite(m):
        return -9999.0
    return m + math.log(sum(math.exp(max(v - m, -80.0)) for v in vals))


async def _evaluate_questions(
    state: Any,
    request: Request | None,
    base_messages: list[dict[str, Any]],
    questions: list[tuple[str, DecisionQuestion]],
    *,
    calibration: str,
    rotations_override: int | None,
    default_score_max: int,
) -> (
    tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, int]]
    | JSONResponse
):
    build_tok = getattr(state, "frontend_tokenizer", None)
    if build_tok is None:
        return create_error_response(
            "Structured decisions require a server tokenizer",
            status_code=503,
            err_type="server_error",
        )
    manager = await asyncio.to_thread(build_tok)

    # Plan all reads across all questions up front so the scheduler batches them together.
    plans: list[dict[str, Any]] = []
    needed_letters: set[str] = set()
    for qid, q in questions:
        q_kind, opts, nums = _build_question_options(
            q, default_score_max=default_score_max
        )
        k = len(opts)
        letters = list(_LETTERS[:k])
        needed_letters.update(letters)
        if rotations_override is not None:
            n_rot = min(rotations_override, k)
        elif q_kind == "score":
            n_rot = min(2, k)
        else:
            n_rot = min(k, 8)
        rot_indices = (
            [0, k - 1]
            if (q_kind == "score" and n_rot == 2 and k > 2)
            else list(range(n_rot))
        )
        q_text = q.question or q.instructions or ""
        plans.append(
            {
                "id": qid,
                "kind": q_kind,
                "opts": opts,
                "nums": nums,
                "letters": letters,
                "rot_indices": rot_indices,
                "q_text": q_text,
            }
        )

    letter_to_tids = await _resolve_label_token_ids(
        manager.tokenizer, sorted(needed_letters)
    )

    read_specs: list[tuple[list[dict[str, Any]], list[int]]] = []
    for plan in plans:
        letters = plan["letters"]
        target_ids: list[int] = []
        for ch in letters:
            for tid in letter_to_tids[ch]:
                if tid not in target_ids:
                    target_ids.append(tid)
        plan["target_ids"] = target_ids
        plan["read_slice_start"] = len(read_specs)
        for r_idx in plan["rot_indices"]:
            if q_is_score_reverse := (
                plan["kind"] == "score"
                and r_idx == len(plan["opts"]) - 1
                and len(plan["opts"]) > 2
            ):
                rotated = list(reversed(plan["opts"]))
            else:
                rotated = _cyclic_shift(plan["opts"], r_idx)
            del q_is_score_reverse
            user_sfx, prefill = _format_read_prompt_suffix(
                plan["q_text"], rotated, letters, neutral=False
            )
            read_specs.append(
                (
                    _append_probe_to_messages(base_messages, user_sfx, prefill),
                    target_ids,
                )
            )
        plan["read_slice_end"] = len(read_specs)
        if calibration in ("contextual", "content_free"):
            prior_base = (
                base_messages
                if calibration == "contextual"
                else [{"role": "user", "content": ""}]
            )
            u_sfx, prefill = _format_read_prompt_suffix(
                "", plan["opts"], letters, neutral=True
            )
            plan["prior_idx"] = len(read_specs)
            read_specs.append(
                (_append_probe_to_messages(prior_base, u_sfx, prefill), target_ids)
            )
        else:
            plan["prior_idx"] = None

    batch_out = await _run_batch_label_reads(state, request, read_specs)
    if batch_out is None:
        return create_error_response(
            "client disconnected before the response was ready",
            status_code=499,
            err_type="client_disconnected",
        )
    reads, pt, ct, _ = batch_out

    answers: list[dict[str, Any]] = []
    diagnostics: dict[str, dict[str, Any]] = {}

    for plan in plans:
        qid = plan["id"]
        q_kind = plan["kind"]
        opts: list[tuple[str, str]] = plan["opts"]
        nums: list[float] | None = plan["nums"]
        letters: list[str] = plan["letters"]
        target_set = set(plan["target_ids"])
        k = len(opts)

        prior_lps = [0.0] * k
        if plan["prior_idx"] is not None:
            _, _, p_map = reads[plan["prior_idx"]]
            raw_prior = [
                _logsumexp([p_map.get(tid, -9999.0) for tid in letter_to_tids[ch]])
                for ch in letters
            ]
            p_probs = _softmax(raw_prior)
            prior_lps = [math.log(max(p, 1e-12)) for p in p_probs]

        agg_log_scores = [0.0] * k
        rot_reads = reads[plan["read_slice_start"] : plan["read_slice_end"]]
        masses: list[float] = []
        argmax_flags: list[bool] = []

        for r_idx, (sampled_id, _, lp_map) in zip(
            plan["rot_indices"], rot_reads, strict=True
        ):
            letter_lps = [
                _logsumexp([lp_map.get(tid, -9999.0) for tid in letter_to_tids[ch]])
                for ch in letters
            ]
            mass = min(1.0, sum(math.exp(max(lp, -80.0)) for lp in letter_lps))
            masses.append(mass)
            argmax_flags.append(sampled_id in target_set)

            calibrated_letter = [
                lp - p_lp for lp, p_lp in zip(letter_lps, prior_lps, strict=True)
            ]
            norm_probs = _softmax(calibrated_letter)
            is_rev = q_kind == "score" and r_idx == k - 1 and k > 2
            for slot_pos, prob in enumerate(norm_probs):
                orig_idx = (k - 1 - slot_pos) if is_rev else ((slot_pos + r_idx) % k)
                agg_log_scores[orig_idx] += math.log(max(prob, 1e-12))

        n_r = max(1, len(rot_reads))
        mean_log = [s / n_r for s in agg_log_scores]
        final_probs = _softmax(mean_log)
        dist = {opts[i][0]: round(final_probs[i], 6) for i in range(k)}
        best_idx = max(range(k), key=lambda i: final_probs[i])

        diagnostics[qid] = {
            "label_mass": round(sum(masses) / max(1, len(masses)), 6),
            "argmax_is_label": all(argmax_flags),
        }

        if q_kind == "predicate":
            expected = round(final_probs[0], 6)
            answers.append(
                {
                    "id": qid,
                    "type": "predicate",
                    "expected": expected,
                    "choice": opts[best_idx][0],
                    "distribution": dist,
                }
            )
        elif q_kind == "score":
            assert nums is not None
            expected = round(
                sum(p * val for p, val in zip(final_probs, nums, strict=True)), 6
            )
            answers.append(
                {
                    "id": qid,
                    "type": "score",
                    "expected": expected,
                    "choice": opts[best_idx][0],
                    "distribution": dist,
                }
            )
        else:
            answers.append(
                {
                    "id": qid,
                    "type": q_kind,
                    "choice": opts[best_idx][0],
                    "distribution": dist,
                }
            )

    usage = {
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "total_tokens": pt + ct,
    }
    return answers, diagnostics, usage


async def handle_decisions(
    req: DecisionsRequest, request: Request | None, state: Any
) -> JSONResponse | dict[str, Any]:
    try:
        base_messages = _normalize_state_messages(req.input, req.instructions)
        q_pairs = [(q.id or f"q_{idx}", q) for idx, q in enumerate(req.questions)]
        evaluated = await _evaluate_questions(
            state,
            request,
            base_messages,
            q_pairs,
            calibration=req.calibration,
            rotations_override=req.rotations,
            default_score_max=10,
        )
    except (ValueError, TypeError, GenerationError) as exc:
        code = getattr(exc, "code", None)
        return create_error_response(str(exc), code=code)

    if isinstance(evaluated, JSONResponse):
        return evaluated
    answers, _, usage = evaluated
    return {
        "id": f"dec_{uuid.uuid4().hex}",
        "object": "decision",
        "created": int(time.time()),
        "model": req.model,
        "answers": answers,
        "usage": usage,
    }


async def handle_systemone(
    req: SystemOneRequest, request: Request | None, state: Any
) -> JSONResponse | dict[str, Any]:
    if not req.questions:
        return create_error_response("questions must not be empty", param="questions")
    if len(req.questions) > _MAX_QUESTIONS:
        return create_error_response(
            f"questions may hold at most {_MAX_QUESTIONS} entries", param="questions"
        )
    try:
        base_messages = _normalize_state_messages(req.state, req.instructions)
        q_pairs = list(req.questions.items())
        evaluated = await _evaluate_questions(
            state,
            request,
            base_messages,
            q_pairs,
            calibration=req.calibration,
            rotations_override=req.rotations,
            default_score_max=5,
        )
    except (ValueError, TypeError, GenerationError) as exc:
        code = getattr(exc, "code", None)
        return create_error_response(str(exc), code=code)

    if isinstance(evaluated, JSONResponse):
        return evaluated
    answers_list, diagnostics, usage = evaluated
    served_model = (
        req.model
        or getattr(state.config, "served_model_name", None)
        or getattr(state.config, "model_path", "default")
    )
    ans_map: dict[str, Any] = {}
    for item in answers_list:
        qid = item["id"]
        ans_map[qid] = {k: v for k, v in item.items() if k != "id"}
    resp: dict[str, Any] = {
        "id": f"sys1_{uuid.uuid4().hex}",
        "object": "systemone",
        "created": int(time.time()),
        "model": served_model,
        "answers": ans_map,
        "usage": usage,
    }
    if req.diagnostics:
        resp["diagnostics"] = diagnostics
    return resp
