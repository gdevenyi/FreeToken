#!/usr/bin/env python3
"""Decision gateway for chatdnp: the OpenAI Decisions API and the Jev API, answered by AnyJev's Decider (L0).

    POST /v1/decisions                 OpenAI Decisions (client.decisions.create): input + questions[] -> answers[]
    POST /v1/decide, /v1/systemone     Jev: state + questions{name: ...} -> answers{name: ...}
    GET  /v1/models, /health

The Decider reads label logprobs from FreeToken's /v1/completions (allowed_token_ids, max_tokens=1), so the
server must run the AnyJev logprobs port. LiteLLM fronts this gateway with authenticated pass-through routes;
the gateway itself checks only an optional shared secret (env var named by --key-env, sent by LiteLLM as
`X-Gateway-Key: <secret>` or `Authorization: Bearer <secret>`).

Decisions run one at a time (one shared Decider, whose content-free priors are cached per question), and the
backend runs --workers reads at once, so FreeToken never sees more than --workers reads from here.

usage: gateway.py --tokenizer <snapshot dir> [--upstream http://127.0.0.1:8000] [--host 127.0.0.1] [--port 8100]
"""

from __future__ import annotations

import argparse
import http.server
import json
import os
import threading
import time
import uuid
import zlib
from typing import Any

from anyjev import Decider
from anyjev.question import MAX_OPTIONS, Question, QuestionError

MAX_BODY = 32 << 20
MAX_QUESTIONS = 32  # one request holds the single Decider; --max-questions
DEFAULT_MODEL = "anyjev-qwen3.8-flash-next"


class RequestError(ValueError):
    def __init__(self, message: str, param: str | None = None, status: int = 400):
        super().__init__(message)
        self.param, self.status = param, status


# ---- engine --------------------------------------------------------------------------------------


class Engine:
    """One Decider behind a lock; counts the prompt tokens each decision sends to the backend."""

    def __init__(self, backend, model: str = DEFAULT_MODEL, price_per_mtok: float = 0.0, **decider_kwargs):
        self.model, self.price = model, price_per_mtok
        self.decider = Decider(backend, **decider_kwargs)
        self.lock = threading.Lock()
        self._tokens = 0
        tokenizer, read = backend.tokenizer, backend.next_token_logprobs

        def counted(prompts, token_ids):
            # the server counts a raw completions prompt exactly like this (probe P1)
            self._tokens += sum(len(tokenizer.encode(p, add_special_tokens=False)) for p in prompts)
            return read(prompts, token_ids)

        backend.next_token_logprobs = counted

    def decide(self, state: Any, questions: list[Question]):
        with self.lock:
            self._tokens = 0
            try:
                decisions = list(self.decider.decide(state, questions))
            except (RequestError, QuestionError):
                raise
            except Exception as e:
                raise RequestError(f"decision backend failed: {e}", status=502) from e
            return decisions, self._tokens


# ---- shared parsing ------------------------------------------------------------------------------


def _text(value: Any, param: str) -> str:
    """A string, or OpenAI content parts (input_text / input_json); images are refused."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        out = []
        for i, part in enumerate(value):
            kind = part.get("type") if isinstance(part, dict) else None
            if kind in ("input_text", "output_text", "text"):
                out.append(str(part.get("text", "")))
            elif kind == "input_json":
                out.append(json.dumps(part.get("data"), ensure_ascii=False))
            elif kind == "refusal":
                out.append(str(part.get("refusal", "")))
            elif kind == "input_image":
                raise RequestError("images are not supported by this decision model", f"{param}[{i}]")
            else:
                raise RequestError(f"unsupported content part type {kind!r}", f"{param}[{i}]")
        return "\n".join(out)
    raise RequestError("must be a string or an array of content parts", param)


def _choice(text: str, labels: list[str], name: str, param: str) -> Question:
    if len(labels) > MAX_OPTIONS:
        raise RequestError(f"at most {MAX_OPTIONS} options are supported here (AnyJev letter readout)", param)
    if len(set(labels)) != len(labels):
        raise RequestError("options must be unique", param)
    return Question.choice(text, labels, name=name)


def _option_label(value: Any, description: Any) -> str:
    return f"{value}: {description}" if description else str(value)


# ---- OpenAI Decisions ----------------------------------------------------------------------------


def openai_state(value: Any) -> Any:
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        raise RequestError("input must be a string or an array of items", "input")
    messages = []
    for i, item in enumerate(value):
        if not isinstance(item, dict):
            raise RequestError("each input item must be an object", f"input[{i}]")
        kind = item.get("type", "message")
        if kind == "message" or "role" in item:
            role = item.get("role")
            if role not in ("user", "developer", "system", "assistant"):
                raise RequestError(f"unsupported role {role!r}", f"input[{i}].role")
            messages.append({"role": role, "content": _text(item.get("content", ""), f"input[{i}].content")})
        elif kind in ("function_call", "custom_tool_call"):
            args = item.get("arguments", item.get("input", ""))
            messages.append({"role": "assistant", "content": f"[tool call {item.get('name')}] {args}"})
        elif kind in ("function_call_output", "custom_tool_call_output"):
            out = item.get("output", "")
            messages.append({"role": "user", "content": f"[tool result] {_text(out, f'input[{i}].output')}"})
        else:
            raise RequestError(f"unsupported input item type {kind!r}", f"input[{i}].type")
    return messages


def openai_questions(raw: Any) -> tuple[list[Question], list[dict]]:
    if not isinstance(raw, list) or not 1 <= len(raw) <= 400:
        raise RequestError("questions must be an array of 1 to 400 items", "questions")
    if len(raw) > MAX_QUESTIONS:
        raise RequestError(f"at most {MAX_QUESTIONS} questions per request are allowed here", "questions")
    questions, metas = [], []
    for i, q in enumerate(raw):
        p = f"questions[{i}]"
        if not isinstance(q, dict):
            raise RequestError("each question must be an object", p)
        kind, name = q.get("type"), q.get("name")
        text = _text(q.get("instructions", ""), f"{p}.instructions")
        qname = name or f"q{i}"
        if kind == "predicate":
            questions.append(Question.noul(text, name=qname))
            metas.append({"type": kind, "name": name})
        elif kind == "choice":
            choices = q.get("choices")
            if not isinstance(choices, list) or not 2 <= len(choices) <= 255:
                raise RequestError("choices must be an array of 2 to 255 items", f"{p}.choices")
            values = [c.get("value") if isinstance(c, dict) else None for c in choices]
            if any(not isinstance(v, (str, bool)) for v in values):
                raise RequestError("each choice needs a string or boolean value", f"{p}.choices")
            if len({(type(v), v) for v in values}) != len(values):
                raise RequestError("choice values must be unique", f"{p}.choices")
            labels = [_option_label(c["value"], c.get("description")) for c in choices]
            questions.append(_choice(text, labels, qname, f"{p}.choices"))
            metas.append({"type": kind, "name": name, "values": values})
        elif kind == "score":
            levels = q.get("levels")
            if not isinstance(levels, list) or not 2 <= len(levels) <= 10:
                raise RequestError("levels must be an array of 2 to 10 items", f"{p}.levels")
            if any(not isinstance(lv, dict) or not isinstance(lv.get("label"), str) for lv in levels):
                raise RequestError("each level needs a string label", f"{p}.levels")
            questions.append(
                Question.score(
                    text, levels=[_option_label(lv["label"], lv.get("description")) for lv in levels], name=qname
                )
            )
            metas.append({"type": kind, "name": name, "labels": [lv["label"] for lv in levels]})
        else:
            raise RequestError(f"unsupported question type {kind!r} (predicate, choice or score)", f"{p}.type")
    return questions, metas


def openai_answer(d, meta: dict) -> dict:
    probs = [float(x) for x in d.probs]
    if meta["type"] == "predicate":
        return {"type": "predicate", "name": meta["name"], "probability": d.p_true}
    if meta["type"] == "choice":
        best = max(range(len(probs)), key=probs.__getitem__)
        return {
            "type": "choice",
            "name": meta["name"],
            "choice": meta["values"][best],
            "confidence": d.confidence,
            "probabilities": [{"value": v, "probability": p} for v, p in zip(meta["values"], probs)],
        }
    return {
        "type": "score",
        "name": meta["name"],
        "score": d.value,
        "confidence": d.confidence,
        "probabilities": [
            {"value": i, "label": lb, "probability": p} for i, (lb, p) in enumerate(zip(meta["labels"], probs))
        ],
    }


def handle_openai(engine: Engine, req: dict) -> tuple[dict, list[str]]:
    if req.get("stream"):
        raise RequestError("streaming is not supported for decisions", "stream")
    if "input" not in req:
        raise RequestError("input is required", "input")
    state = openai_state(req["input"])
    questions, metas = openai_questions(req.get("questions"))
    decisions, tokens = engine.decide(state, questions)
    body = {
        "id": f"dec_{uuid.uuid4().hex}",
        "model": engine.model,
        "answers": [openai_answer(d, m) for d, m in zip(decisions, metas)],
        "usage": {
            "input_tokens": tokens,
            "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
            "output_tokens": 0,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": tokens,
        },
    }
    return body, [d.level for d in decisions]


# ---- Jev ------------------------------------------------------------------------------------------


def jev_questions(raw: Any) -> tuple[list[Question], list[dict]]:
    if not isinstance(raw, dict) or not raw:
        raise RequestError("questions must be a non-empty object of named questions", "questions")
    if len(raw) > MAX_QUESTIONS:
        raise RequestError(f"at most {MAX_QUESTIONS} questions per request are allowed here", "questions")
    questions, metas = [], []
    for name, q in raw.items():
        p = f"questions.{name}"
        if not isinstance(q, dict):
            raise RequestError("each question must be an object", p)
        kind, text, criteria = q.get("type"), _text(q.get("instructions", ""), f"{p}.instructions"), q.get("criteria")
        if kind == "noul":
            questions.append(Question.noul(text, name=name))
            metas.append({"type": kind, "name": name})
        elif kind == "choice":
            if not isinstance(criteria, dict) or len(criteria) < 2:
                raise RequestError("choice criteria must map at least 2 labels to descriptions", f"{p}.criteria")
            keys = [str(k) for k in criteria]
            questions.append(_choice(text, [_option_label(k, criteria[k]) for k in keys], name, f"{p}.criteria"))
            metas.append({"type": kind, "name": name, "keys": keys})
        elif kind == "score":
            if (
                not isinstance(criteria, list)
                or not 2 <= len(criteria) <= 10
                or not all(isinstance(c, str) for c in criteria)
            ):
                raise RequestError("score criteria must be an ordered array of 2 to 10 strings", f"{p}.criteria")
            questions.append(Question.score(text, levels=criteria, name=name))
            metas.append({"type": kind, "name": name, "legend": list(criteria)})
        else:
            raise RequestError(f"unsupported question type {kind!r} (choice, score or noul)", f"{p}.type")
    return questions, metas


def jev_answer(d, meta: dict) -> dict:
    probs = [float(x) for x in d.probs]
    if meta["type"] == "noul":
        return {"type": "noul", "noul": d.p_true}
    if meta["type"] == "choice":
        best = max(range(len(probs)), key=probs.__getitem__)
        return {
            "type": "choice",
            "choice": meta["keys"][best],
            "confidence": d.confidence,
            "probabilities": dict(zip(meta["keys"], probs)),
        }
    return {
        "type": "score",
        "score": d.value,
        "confidence": d.confidence,
        "probabilities": {str(i): p for i, p in enumerate(probs)},
        "legend": {str(i): label for i, label in enumerate(meta["legend"])},
    }


def handle_jev(engine: Engine, req: dict) -> tuple[dict, list[str]]:
    if "state" not in req:
        raise RequestError("state is required", "state")
    if req.get("assets"):
        raise RequestError("assets (images) are not supported by this decision model", "assets")
    questions, metas = jev_questions(req.get("questions"))
    decisions, tokens = engine.decide(req["state"], questions)
    body = {
        "id": f"dec_{uuid.uuid4().hex}",
        "model": engine.model,
        "answers": {m["name"]: jev_answer(d, m) for d, m in zip(decisions, metas)},
        "usage": {"input_tokens": tokens, "cost_usd": round(tokens * engine.price / 1e6, 9)},
    }
    return body, [d.level for d in decisions]


# ---- HTTP -----------------------------------------------------------------------------------------


def make_server(engine: Engine, host: str = "127.0.0.1", port: int = 8100, key: str | None = None):
    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _send(self, code: int, obj: dict, levels: list[str] | None = None):
            body = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            if levels:
                self.send_header("X-AnyJev-Levels", ",".join(levels))
            self.end_headers()
            self.wfile.write(body)

        def _error(self, openai: bool, status: int, message: str, param: str | None = None, code: str | None = None):
            if openai:
                kind = {401: "authentication_error", 502: "upstream_error"}.get(status, "invalid_request_error")
                return self._send(status, {"error": {"message": message, "type": kind, "param": param, "code": code}})
            return self._send(
                status, {"error": message, **({"code": code} if code else {}), **({"param": param} if param else {})}
            )

        def _authorized(self) -> bool:
            if not key:
                return True
            return self.headers.get("X-Gateway-Key") == key or self.headers.get("Authorization") == f"Bearer {key}"

        def do_GET(self):
            if self.path == "/health":
                return self._send(200, {"ok": True})
            if not self._authorized():
                return self._error(True, 401, "invalid gateway key", code="invalid_api_key")
            if self.path == "/v1/models":
                return self._send(
                    200, {"object": "list", "data": [{"id": engine.model, "object": "model", "owned_by": "anyjev"}]}
                )
            return self._error(True, 404, "not found")

        def do_POST(self):
            path = self.path.split("?", 1)[0]
            openai = path == "/v1/decisions"
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY:
                self.close_connection = True  # the unread body would be parsed as the next request
                return self._error(openai, 413, "request body too large")
            # read the body before any reply: LiteLLM reuses keep-alive connections
            raw = self.rfile.read(length)
            if path not in ("/v1/decisions", "/v1/decide", "/v1/systemone"):
                return self._error(True, 404, "not found")
            if not self._authorized():
                return self._error(openai, 401, "invalid gateway key", code="invalid_api_key")
            enc = (self.headers.get("Content-Encoding") or "identity").lower()
            t0 = time.time()
            try:
                if enc in ("gzip", "deflate"):
                    raw = zlib.decompress(raw, 16 + zlib.MAX_WBITS if enc == "gzip" else zlib.MAX_WBITS)
                elif enc != "identity":
                    return self._error(openai, 415, f"unsupported Content-Encoding {enc!r} (identity, gzip, deflate)")
                req = json.loads(raw or b"{}")
                if not isinstance(req, dict):
                    raise RequestError("the request body must be a JSON object")
                body, levels = (handle_openai if openai else handle_jev)(engine, req)
            except (json.JSONDecodeError, zlib.error) as e:
                return self._error(openai, 400, f"invalid request body: {e}")
            except QuestionError as e:
                return self._error(openai, 400, str(e), "questions")
            except RequestError as e:
                return self._error(openai, e.status, str(e), e.param)
            print(
                f"{time.strftime('%F %T')} {path} 200 {len(levels)} questions {body['usage']['input_tokens']} tokens "
                f"{time.time() - t0:.2f} s levels {','.join(sorted(set(levels)))}",
                flush=True,
            )
            return self._send(200, body, levels)

        def log_message(self, fmt, *args):
            pass

    server = http.server.ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server


def main() -> None:
    global MAX_QUESTIONS
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tokenizer", required=True, help="local checkpoint dir with the served model's tokenizer")
    ap.add_argument("--upstream", default="http://127.0.0.1:8000", help="FreeToken root URL")
    ap.add_argument("--upstream-model", default="Qwen3.8-Flash-Next", help="FreeToken's served model name")
    ap.add_argument("--model-name", default=DEFAULT_MODEL, help="the model name this gateway reports")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--workers", type=int, default=4, help="concurrent reads sent to FreeToken")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--prior", default="content_free", choices=("content_free", "batch", "none"))
    ap.add_argument("--price-per-mtok", type=float, default=0.0, help="Jev usage.cost_usd per million input tokens")
    ap.add_argument(
        "--max-questions", type=int, default=MAX_QUESTIONS, help="questions per request (OpenAI allows 400)"
    )
    ap.add_argument("--key-env", default="ANYJEV_GW_KEY", help="env var holding the shared secret (unset: no check)")
    a = ap.parse_args()
    MAX_QUESTIONS = a.max_questions
    from anyjev.backends.vllm import VLLMBackend

    backend = VLLMBackend(
        a.upstream, a.upstream_model, tokenizer_name=a.tokenizer, workers=a.workers, timeout=a.timeout
    )
    engine = Engine(
        backend, a.model_name, a.price_per_mtok, level="L0", prior=a.prior, adaptive_shifts=True, canonical_order=True
    )
    key = os.environ.get(a.key_env) or None
    server = make_server(engine, a.host, a.port, key)
    print(
        f"decision gateway on http://{a.host}:{a.port} -> {a.upstream} ({a.upstream_model}); key check {'on' if key else 'off'}",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
