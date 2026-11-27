"""OpenAI Responses API (`/v1/responses`) — the endpoint codex talks to.

Adapted from vLLM's ``OpenAIServingResponses`` (the *simple* rail; the gpt-oss
Harmony rail is not ported). Like the Anthropic adapter, this drives the shared
protocol-neutral primitive (:func:`freetoken.server.openai_api.submit_generation` +
``generate_events`` / ``generate_full``) and formats its semantic events into
Responses typed items + stream events using the real ``openai.types.responses``
SDK models. It consumes typed events, not a re-parsed OpenAI stream.

Scope: a STATELESS subset (``store``/``previous_response_id``/``background`` are
ignored; ``GET /v1/responses/{id}`` and ``/cancel`` are stubbed) — enough to drive
codex, which resends full context as ``input`` when not storing. Reasoning streams
as a first-class ``reasoning`` item (``response.reasoning_text.delta``, the shape
vLLM emits and codex renders); reasoning items replayed as ``input`` are folded
back into their assistant turn as ``reasoning_content``.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator, Callable
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from openai.types.responses import (
    Response,
    ResponseCompletedEvent,
    ResponseContentPartAddedEvent,
    ResponseContentPartDoneEvent,
    ResponseCreatedEvent,
    ResponseError,
    ResponseFailedEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseFunctionToolCall,
    ResponseIncompleteEvent,
    ResponseInProgressEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseReasoningItem,
    ResponseReasoningTextDeltaEvent,
    ResponseReasoningTextDoneEvent,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
)
from openai.types.responses.response import IncompleteDetails
from openai.types.responses.response_reasoning_item import (
    Content as ReasoningTextContent,
)
from openai.types.responses.response_usage import (
    InputTokensDetails,
    OutputTokensDetails,
    ResponseUsage,
)
from pydantic import BaseModel, ConfigDict

from .generation import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    KEEPALIVE,
    ContentDelta,
    GenDone,
    GenerationError,
    GenResult,
    GenSpec,
    GenUsage,
    ReasoningDelta,
    ToolCallArgsDelta,
    ToolCallsDelta,
    ToolCallStart,
    count_prompt_tokens,
    generate_events,
    generate_full,
    render_messages,
    resolve_sampling,
    split_tool_lists,
    submit_generation,
    timings_block,
    with_keepalive,
)
from .logprobs import MAX_CHAT_LOGPROBS, responses_logprobs_content
from .openai_api import (
    _ACCEPTED_EFFORTS,
    _await_watching_disconnect,
    _model_context_length,
    _response_format_error,
)
from .request_logger import log_request

# Seconds of event silence before a keep-alive frame is emitted on the stream.
# codex's stream-idle timeout (default 300s) only resets on a data-bearing SSE
# frame, so long queue/prefill or decode gaps must be bridged with real events.
KEEPALIVE_INTERVAL_S = 10.0


class ResponsesRequest(BaseModel):
    """The subset of the Responses request FreeToken honors (extra fields allowed)."""

    model_config = ConfigDict(extra="allow")

    model: str
    input: str | list[dict[str, Any]]
    instructions: str | None = None
    max_output_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    stream: bool = False
    # llama.cpp's field: every delta event carries the timings block (effective only with
    # --enable-timings-report; extra="allow" used to swallow it).
    timings_per_token: bool = False
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any | None = None
    reasoning: dict[str, Any] | None = None
    text: dict[str, Any] | None = None
    include: list[str] | None = None
    top_logprobs: int | None = None
    truncation: str | None = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float | None = None
    min_p: float | None = None
    logit_bias: dict[str, float] | None = None
    stop: str | list[str] | None = None
    stop_token_ids: list[int] | None = None
    seed: Any | None = None
    # Stateful features are accepted but ignored in this subset.
    store: bool = False
    previous_response_id: str | None = None
    background: bool = False
    metadata: dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None


class ResponsesInputTokensRequest(BaseModel):
    """Request body for ``POST /v1/responses/input_tokens``."""

    model_config = ConfigDict(extra="allow")

    model: str | None = None
    input: str | list[dict[str, Any]] | None = None
    instructions: str | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any | None = None
    reasoning: dict[str, Any] | None = None
    previous_response_id: str | None = None
    parallel_tool_calls: bool | None = None


def register_responses_routes(
    app: FastAPI,
    get_state: Callable[[], Any],
    get_model_sampling: Callable[[], dict[str, Any]],
) -> None:
    @app.post("/v1/responses")
    async def v1_responses(req: ResponsesRequest, request: Request):
        log_request("/v1/responses", req, request)
        state = get_state()
        mstate = getattr(state, "maintenance_state", "serving")
        if mstate != "serving":
            detail = (
                "model is still loading"
                if mstate == "loading"
                else "cache rebuild in progress"
            )
            return _error_response(503, detail, err_type="server_error")
        if req.background:
            return _error_response(
                400, "background mode is not supported", param="background"
            )
        if req.previous_response_id:
            return _error_response(
                400,
                "previous_response_id is not supported (stateless server); "
                "resend full context in 'input'",
                param="previous_response_id",
            )
        return await handle_responses(req, request, state, get_model_sampling())

    @app.post("/v1/responses/input_tokens")
    async def v1_responses_input_tokens(
        req: ResponsesInputTokensRequest, request: Request
    ):
        log_request("/v1/responses/input_tokens", req, request)
        return await handle_responses_input_tokens(req, get_state())

    @app.get("/v1/responses/{response_id}")
    async def v1_responses_get(response_id: str):
        return _error_response(
            404,
            f"response {response_id!r} not found (stateless server)",
            err_type="not_found_error",
        )

    @app.delete("/v1/responses/{response_id}")
    async def v1_responses_delete(response_id: str):
        return _error_response(
            404,
            f"response {response_id!r} not found (stateless server)",
            err_type="not_found_error",
        )

    @app.post("/v1/responses/{response_id}/cancel")
    async def v1_responses_cancel(response_id: str):
        return _error_response(
            404,
            f"response {response_id!r} not found (stateless server)",
            err_type="not_found_error",
        )


async def handle_responses(
    req: ResponsesRequest,
    request: Request | None,
    state: Any,
    model_sampling: dict[str, Any],
):
    response_id = f"resp_{uuid.uuid4().hex}"
    created = int(time.time())
    if req.max_output_tokens is not None and req.max_output_tokens < 1:
        return _error_response(
            400,
            "max_output_tokens must be a positive integer",
            param="max_output_tokens",
        )
    if req.truncation not in (None, "disabled", "auto"):
        return _error_response(
            400,
            f"truncation must be 'disabled' or 'auto', got {req.truncation!r}",
            param="truncation",
        )
    if req.top_logprobs is not None and not 0 <= req.top_logprobs <= MAX_CHAT_LOGPROBS:
        return _error_response(
            400,
            f"top_logprobs must be between 0 and {MAX_CHAT_LOGPROBS}",
            param="top_logprobs",
        )
    if isinstance(req.reasoning, dict) and req.reasoning.get("effort") is not None:
        eff = str(req.reasoning.get("effort")).strip().lower()
        if eff and eff not in _ACCEPTED_EFFORTS:
            return _error_response(
                400,
                f"reasoning.effort must be one of {', '.join(_ACCEPTED_EFFORTS)}; got {req.reasoning.get('effort')!r}",
                param="reasoning.effort",
            )
    default_max = (
        getattr(state.config, "max_output_tokens", None) or DEFAULT_MAX_OUTPUT_TOKENS
    )
    try:
        spec = convert_responses_to_genspec(
            req,
            model_sampling,
            default_max_tokens=default_max,
            reasoning_parser=getattr(state.config, "reasoning_parser", None),
            # stamp per-token only when the server opted into timings too; llama.cpp
            # publishes partial timings on streams, so non-stream requests never pay
            timings_per_token=bool(
                req.stream
                and req.timings_per_token
                and getattr(state.config, "enable_timings_report", False)
            ),
            thinking=getattr(state.config, "thinking", None),
        )
        if req.truncation == "auto":
            await _auto_truncate_spec(spec, state)
        uid = await submit_generation(spec, state)
    except GenerationError as exc:
        return _error_response(400, str(exc), exc.code)
    except ValueError as exc:
        return _error_response(400, str(exc), param=getattr(exc, "param", None))

    cache_report = getattr(state.config, "enable_cache_report", False)
    if req.stream:
        # shared with generate_events: its ack loop stamps usage.timings before yielding
        # each event, so a delta reads the snapshot of the ack behind it
        usage = GenUsage()
        events = responses_stream_generator(
            generate_events(uid, spec, state, source="/v1/responses", usage=usage),
            req,
            response_id,
            created,
            cache_report=cache_report,
            state=state,
            usage=usage,
        )
        if request is not None:
            events = state.stream_with_cancellation(events, request, uid)
        return StreamingResponse(events, media_type="text/event-stream")

    try:
        # an abandoned request must not keep decoding to max_tokens (#222's watcher)
        result = await _await_watching_disconnect(
            generate_full(uid, spec, state, source="/v1/responses"), request, state, uid
        )
    except GenerationError as exc:
        return _error_response(400, str(exc), exc.code)
    if result is None:
        return _error_response(499, "client disconnected before the response was ready")
    response = build_responses_response(
        result, req, response_id, created, cache_report=cache_report, state=state
    )
    return JSONResponse(content=response.model_dump(mode="json"))


# --------------------------------------------------------------------------- #
# Request conversion: Responses -> GenSpec (the neutral generation spec; built
# directly with the Responses' OWN max-tokens default — so the OpenAI-chat 16-token
# floor can never leak in, exactly as vLLM keeps a per-protocol to_sampling_params).
# --------------------------------------------------------------------------- #
def convert_responses_prompt(
    req: ResponsesRequest | ResponsesInputTokensRequest,
    reasoning_parser: str | None = None,
    thinking: str | None = None,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]] | None,
    list[dict[str, Any]] | None,
    dict[str, Any],
]:
    """(messages, template_tools, parser_tools, chat_template_kwargs) shared by
    ``/v1/responses`` and ``/v1/responses/input_tokens``."""
    del reasoning_parser
    # Collect every system/developer text — the top-level `instructions` PLUS any
    # system/developer-role input items (codex sends both: a system prompt as `instructions`
    # and a `developer` permissions message) — into ONE leading system message. Strict chat
    # templates (e.g. Qwen3.5) reject a second system message with "System message must be at
    # the beginning"; merging mirrors the Anthropic adapter (which is why Qwen3.5 already works
    # through Claude Code).
    system_texts: list[str] = []
    if req.instructions:
        system_texts.append(req.instructions)

    other: list[dict[str, Any]] = []
    if req.input is None:
        pass
    elif isinstance(req.input, str):
        other.append({"role": "user", "content": req.input})
    else:
        deferred_tool_images: list[dict[str, Any]] = []
        for item in req.input:
            converted = _convert_input_item(item)
            for m in converted:
                if m.get("_deferred_tool_images"):
                    deferred_tool_images.extend(m["content"])
                    continue
                if deferred_tool_images and m.get("role") != "tool":
                    other.append(
                        {"role": "user", "content": list(deferred_tool_images)}
                    )
                    deferred_tool_images.clear()
                if m.get("role") == "system":
                    system_texts.append(m.get("content") or "")
                else:
                    other.append(m)
        if deferred_tool_images:
            other.append({"role": "user", "content": list(deferred_tool_images)})
        other = _merge_assistant_run(other)

    messages: list[dict[str, Any]] = []
    system_text = "\n\n".join(t for t in system_texts if t)
    if system_text:
        messages.append({"role": "system", "content": system_text})
    messages.extend(other)

    raw_tools = _convert_tools(req.tools) or []
    template_tools, parser_tools, _ = _resolve_responses_tool_choice(
        raw_tools, req.tool_choice
    )

    from .model_meta import effort_toggle_kwargs, thinking_override_kwargs

    ctk = dict(getattr(req, "chat_template_kwargs", None) or {})
    if req.reasoning:
        ctk = effort_toggle_kwargs(req.reasoning.get("effort"), ctk)
    ctk = thinking_override_kwargs(thinking, ctk)
    if req.parallel_tool_calls is False:
        ctk = {**ctk, "parallel_tool_calls": False}
    return render_messages(messages), template_tools, parser_tools, ctk


async def handle_responses_input_tokens(req: ResponsesInputTokensRequest, state: Any):
    if req.previous_response_id:
        return _error_response(
            400,
            "previous_response_id is not supported (stateless server); "
            "resend full context in 'input'",
        )
    try:
        messages, template_tools, _, ctk = convert_responses_prompt(
            req,
            reasoning_parser=getattr(state.config, "reasoning_parser", None),
            thinking=getattr(state.config, "thinking", None),
        )
    except ValueError as exc:
        return _error_response(400, str(exc))
    if not messages:
        return _error_response(400, "input: no tokenizable content")
    try:
        n_tokens = await count_prompt_tokens(messages, template_tools, ctk, state)
    except GenerationError as exc:
        return _error_response(400, str(exc), exc.code)
    except Exception as exc:  # noqa: BLE001 — tokenizer init / other failure -> server error
        return _error_response(
            500, f"token counting failed: {exc}", err_type="server_error"
        )
    return JSONResponse(
        content={"object": "response.input_tokens", "input_tokens": n_tokens}
    )


def _resolve_responses_tool_choice(
    raw_tools: list[dict[str, Any]], tc: Any
) -> tuple[list[dict[str, Any]] | None, list[dict[str, Any]] | None, str | None]:
    tool_names = {(t.get("function") or {}).get("name") for t in raw_tools}
    if tc is None or tc == "auto":
        return (raw_tools or None), (raw_tools or None), None
    if tc == "none":
        return None, None, None
    if tc == "required":
        if not raw_tools:
            raise ValueError("tool_choice='required' requires non-empty tools")
        return raw_tools, raw_tools, "required"
    if isinstance(tc, dict):
        tc_type = tc.get("type")
        if tc_type in ("function", "custom"):
            selected = tc.get("name") or (tc.get("function") or {}).get("name")
            if not selected or selected not in tool_names:
                raise ValueError(f"tool_choice {selected!r} was not found in tools")
            tmpl, parser = split_tool_lists(raw_tools, selected)
            return tmpl, parser, f"named:{selected}"
        if tc_type == "allowed_tools":
            inner = (
                tc.get("allowed_tools")
                if isinstance(tc.get("allowed_tools"), dict)
                else tc
            )
            mode = inner.get("mode", "auto")
            if mode not in ("auto", "required", "none"):
                raise ValueError(
                    f"allowed_tools.mode must be 'auto', 'required', or 'none', got {mode!r}"
                )
            if mode == "none":
                return None, None, None
            refs = inner.get("tools") or []
            allowed = {
                (r.get("function") or {}).get("name") or r.get("name")
                for r in refs
                if isinstance(r, dict)
            } - {None, ""}
            if not allowed:
                raise ValueError("allowed_tools requires at least one tool")
            missing = sorted(allowed - tool_names)
            if missing:
                raise ValueError(
                    f"allowed_tools references unknown tool(s): {', '.join(missing)}"
                )
            subset = [
                t for t in raw_tools if (t.get("function") or {}).get("name") in allowed
            ]
            if mode == "required" and len(subset) == 1:
                only = (subset[0].get("function") or {}).get("name")
                return subset, raw_tools, f"named:{only}"
            return subset, raw_tools, ("required" if mode == "required" else None)
    raise ValueError(f"Unsupported tool_choice: {tc!r}")


def _extract_responses_format(text_cfg: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(text_cfg, dict):
        return None
    fmt = text_cfg.get("format")
    if not isinstance(fmt, dict):
        return None
    ftype = fmt.get("type")
    if ftype in (None, "text"):
        return {"type": "text"}
    if ftype == "json_object":
        return {"type": "json_object"}
    if ftype == "json_schema":
        # Responses API puts schema directly on format ({"type": "json_schema", "name": ..., "schema": ...})
        nested = (
            fmt.get("json_schema") if isinstance(fmt.get("json_schema"), dict) else fmt
        )
        rf = {"type": "json_schema", "json_schema": nested}
        if (err := _response_format_error(rf)) is not None:
            raise ValueError(err)
        return rf
    raise ValueError(
        f"text.format.type {ftype!r} is not supported (supported: text, json_object, json_schema)"
    )


async def _auto_truncate_spec(spec: GenSpec, state: Any) -> None:
    """Drop oldest non-system conversation turns when ``truncation='auto'`` exceeds the context window."""
    ctx_limit = _model_context_length(state)
    if (
        not ctx_limit
        or len(spec.messages) <= 1
        or getattr(state, "frontend_tokenizer", None) is None
    ):
        return
    budget = max(1, ctx_limit - min(spec.sampling_params.max_tokens, ctx_limit // 2))
    while len(spec.messages) > 1:
        try:
            count = await count_prompt_tokens(
                spec.messages, spec.template_tools, spec.chat_template_kwargs, state
            )
        except Exception:  # noqa: BLE001
            return
        if count <= budget:
            return
        drop_idx = 1 if spec.messages[0].get("role") == "system" else 0
        if drop_idx >= len(spec.messages) - 1:
            return
        del spec.messages[drop_idx]


def convert_responses_to_genspec(
    req: ResponsesRequest,
    model_sampling: dict[str, Any],
    default_max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    reasoning_parser: str | None = None,
    timings_per_token: bool = False,
    thinking: str | None = None,
) -> GenSpec:
    messages, template_tools, parser_tools, ctk = convert_responses_prompt(
        req, reasoning_parser=reasoning_parser, thinking=thinking
    )
    raw_tools = _convert_tools(req.tools) or []
    _, _, tool_choice_mode = _resolve_responses_tool_choice(raw_tools, req.tool_choice)
    response_format = _extract_responses_format(req.text)
    want_lps = bool(req.include and "message.output_text.logprobs" in req.include)
    sp = resolve_sampling(
        temperature=req.temperature,
        top_k=req.top_k,
        top_p=req.top_p,
        max_tokens=req.max_output_tokens or default_max_tokens,
        ignore_eos=False,
        model_sampling=model_sampling,
        stop=req.stop,
        presence_penalty=req.presence_penalty,
        frequency_penalty=req.frequency_penalty,
        repetition_penalty=req.repetition_penalty,
        min_p=req.min_p,
        logit_bias=req.logit_bias,
        stop_token_ids=req.stop_token_ids,
        seed=req.seed,
        timings_per_token=timings_per_token,
    )
    if want_lps:
        sp.logprobs = True
        sp.top_logprobs = req.top_logprobs if req.top_logprobs is not None else 0
    return GenSpec(
        messages=messages,
        sampling_params=sp,
        chat_template_kwargs=ctk,
        template_tools=template_tools,
        parser_tools=parser_tools,
        include_logprobs=want_lps,
        disable_parallel_tool_use=bool(req.parallel_tool_calls is False),
        tool_choice_mode=tool_choice_mode,
        response_format=response_format,
    )


def _convert_input_item(item: dict[str, Any]) -> list[dict[str, Any]]:
    itype = item.get("type", "message")
    if itype == "message" or ("role" in item and "type" not in item):
        # OpenAI's "developer" role is their spelling of "system"; chat templates only
        # know system/user/assistant/tool, so map it before handing off.
        role = item.get("role", "user")
        if role == "developer":
            role = "system"
        return [{"role": role, "content": _input_content(item.get("content"))}]
    if itype in ("function_call", "custom_tool_call"):
        args_val = (
            json.dumps({"input": item.get("input", "")})
            if itype == "custom_tool_call" and "arguments" not in item
            else (item.get("arguments", "") or "")
        )
        return [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": item.get("call_id")
                        or item.get("id")
                        or f"call_{uuid.uuid4().hex[:8]}",
                        "type": "function",
                        "function": {
                            "name": item.get("name", ""),
                            "arguments": args_val,
                        },
                    }
                ],
            }
        ]
    if itype in ("function_call_output", "custom_tool_call_output"):
        output = item.get("output")
        content = (
            _input_content(output) if isinstance(output, list) else _stringify(output)
        )
        tool_msg = {
            "role": "tool",
            "tool_call_id": item.get("call_id", ""),
            "content": content,
        }
        if isinstance(content, str):
            return [tool_msg]
        # Chat templates render tool messages as text, so the images ride on a single user
        # turn emitted after the contiguous tool-output run (preserving parallel tool_calls).
        tool_msg["content"] = "".join(p["text"] for p in content if p["type"] == "text")
        images = [p for p in content if p["type"] == "image"]
        if not images:
            return [tool_msg]
        return [tool_msg, {"_deferred_tool_images": True, "content": images}]
    if itype == "reasoning":
        # Folded into its assistant turn by _merge_assistant_run; summary-only /
        # encrypted items carry no recoverable raw reasoning text.
        text = "".join(
            c.get("text") or ""
            for c in (item.get("content") or [])
            if isinstance(c, dict) and c.get("type") in ("reasoning_text", "text")
        )
        if text:
            return [{"role": "assistant", "reasoning_content": text}]
        return []
    if itype == "item_reference":
        raise ValueError(
            "item_reference is not supported on this stateless server; resend full item content"
        )
    raise ValueError(f"Unsupported input item type: {itype!r}")


def _merge_assistant_run(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Coalesce one assistant turn's items (reasoning / message / function_call)
    into a single message — separate messages would render as extra assistant
    turns. Slot conflicts start a new message: a reasoning item always opens a
    turn, a message item only fills a turn with no content/tool_calls yet, so
    distinct assistant messages never merge."""
    merged: list[dict[str, Any]] = []
    for m in messages:
        prev = merged[-1] if merged else None
        mergeable = (
            prev is not None
            and m.get("role") == "assistant"
            and prev.get("role") == "assistant"
            and not m.get("reasoning_content")
            and (
                m.get("tool_calls")
                or (
                    m.get("content")
                    and not prev.get("content")
                    and not prev.get("tool_calls")
                )
            )
        )
        if not mergeable:
            merged.append(dict(m))
            continue
        if m.get("content"):
            prev["content"] = m["content"]
        if m.get("tool_calls"):
            prev["tool_calls"] = (prev.get("tool_calls") or []) + m["tool_calls"]
    return merged


def _input_content(content: Any) -> str | list[dict[str, Any]]:
    """Like _input_text, but keeps input_image parts as template-ready image parts."""
    if not isinstance(content, list):
        return _input_text(content)
    parts: list[dict[str, Any]] = []
    has_image = False
    for part in content:
        if isinstance(part, dict) and part.get("type") == "input_image":
            url = part.get("image_url") or part.get("url")
            if isinstance(url, dict):
                url = url.get("url")
            if not url:
                # an input_image without a url (e.g. a file_id) is not servable; fail rather than answer text-only
                raise ValueError("input_image without image_url is not supported")
            parts.append(
                {"type": "image", "freetoken_ref": {"kind": "url", "data": url}}
            )
            has_image = True
            continue
        parts.append({"type": "text", "text": _input_text([part])})
    if not has_image:
        return "".join(p["text"] for p in parts)
    return parts


def _input_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for part in content:
        if isinstance(part, dict):
            if (
                part.get("type") in ("input_text", "output_text", "text")
                or "text" in part
            ):
                parts.append(part.get("text") or "")
        else:
            parts.append(str(part))
    return "".join(parts)


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    return str(value)


def _convert_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    if not tools:
        return None
    converted: list[dict[str, Any]] = []
    for tool in tools:
        ttype = tool.get("type")
        if ttype == "custom":
            converted.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.get("name", ""),
                        "description": tool.get("description"),
                        "parameters": {
                            "type": "object",
                            "properties": {"input": {"type": "string"}},
                            "required": ["input"],
                        },
                    },
                }
            )
            continue
        if ttype not in (None, "function"):
            # Built-in server tools (web_search, code_interpreter, ...) are unsupported; skip.
            continue
        fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        converted.append(
            {
                "type": "function",
                "function": {
                    "name": fn.get("name", ""),
                    "description": fn.get("description"),
                    "parameters": fn.get("parameters") or {"type": "object"},
                },
            }
        )
    return converted or None


# --------------------------------------------------------------------------- #
# Non-streaming response assembly (GenResult -> Response)
# --------------------------------------------------------------------------- #
def build_responses_response(
    result: GenResult,
    req: ResponsesRequest,
    response_id: str,
    created: int,
    cache_report: bool = False,
    state: Any = None,
) -> Response:
    truncated = result.finish_reason == "length"
    want_lps = bool(req.include and "message.output_text.logprobs" in req.include)
    lp_items = responses_logprobs_content(result.logprobs or []) if want_lps else None
    output: list[Any] = []
    if result.reasoning:
        output.append(
            ResponseReasoningItem(
                id=f"rs_{uuid.uuid4().hex}",
                type="reasoning",
                summary=[],
                content=[
                    ReasoningTextContent(type="reasoning_text", text=result.reasoning)
                ],
                status="incomplete"
                if (truncated and not result.content and not result.tool_calls)
                else "completed",
            )
        )
    if result.content or (not result.reasoning and not result.tool_calls):
        msg_kwargs: dict[str, Any] = {}
        if lp_items is not None:
            msg_kwargs["logprobs"] = lp_items
        output.append(
            ResponseOutputMessage(
                id=f"msg_{uuid.uuid4().hex}",
                role="assistant",
                status="incomplete"
                if (truncated and not result.tool_calls)
                else "completed",
                type="message",
                content=[
                    ResponseOutputText(
                        type="output_text",
                        text=result.content or "",
                        annotations=[],
                        **msg_kwargs,
                    )
                ],
            )
        )
    for idx, call in enumerate(result.tool_calls):
        is_last = idx == len(result.tool_calls) - 1
        output.append(
            ResponseFunctionToolCall(
                type="function_call",
                id=f"fc_{uuid.uuid4().hex}",
                call_id=f"call_{uuid.uuid4().hex[:24]}",
                name=call.name or "",
                arguments=call.parameters or "",
                status="incomplete" if (truncated and is_last) else "completed",
            )
        )

    return _response_obj(
        response_id,
        created,
        req.model,
        output,
        status="incomplete" if truncated else "completed",
        usage=_usage(
            result.prompt_tokens,
            result.completion_tokens,
            result.cached_tokens if cache_report else 0,
            reasoning_tokens=result.reasoning_tokens,
        ),
        incomplete_reason="max_output_tokens" if truncated else None,
        timings=timings_block(state, result.timings),
        req=req,
    )


def _response_obj(
    response_id: str,
    created: int,
    model: str,
    output: list[Any],
    *,
    status: str,
    usage: ResponseUsage | None,
    error: ResponseError | None = None,
    incomplete_reason: str | None = None,
    timings: dict[str, Any] | None = None,
    req: ResponsesRequest | None = None,
) -> Response:
    # Response allows extra fields, so timings rides along without forking the schema.
    extra: dict[str, Any] = {"timings": timings} if timings is not None else {}
    if status == "completed":
        extra["completed_at"] = float(int(time.time()))
    if req is not None:
        if req.instructions is not None:
            extra["instructions"] = req.instructions
        if req.metadata is not None:
            extra["metadata"] = req.metadata
        if req.temperature is not None:
            extra["temperature"] = req.temperature
        if req.top_p is not None:
            extra["top_p"] = req.top_p
        if req.max_output_tokens is not None:
            extra["max_output_tokens"] = req.max_output_tokens
        if req.truncation in ("auto", "disabled"):
            extra["truncation"] = req.truncation
    parallel = (
        bool(req.parallel_tool_calls)
        if (req is not None and req.parallel_tool_calls is not None)
        else True
    )
    tc_val: Any = "auto"
    if req is not None and isinstance(req.tool_choice, str):
        tc_val = req.tool_choice
    return Response(
        id=response_id,
        created_at=created,
        model=model,
        object="response",
        output=output,
        status=status,
        usage=usage,
        error=error,
        incomplete_details=IncompleteDetails(reason=incomplete_reason)
        if incomplete_reason
        else None,
        parallel_tool_calls=parallel,
        tool_choice=tc_val,
        tools=[],
        **extra,
    )


def _usage(
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: int = 0,
    reasoning_tokens: int = 0,
) -> ResponseUsage:
    # input_tokens stays inclusive of the cached prefix (OpenAI semantics);
    # cached_tokens is nonzero only under --enable-cache-report.
    return ResponseUsage(
        input_tokens=prompt_tokens,
        output_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        input_tokens_details=InputTokensDetails(
            cached_tokens=cached_tokens, cache_write_tokens=0
        ),
        output_tokens_details=OutputTokensDetails(reasoning_tokens=reasoning_tokens),
    )


# --------------------------------------------------------------------------- #
# Streaming: GenEvent stream -> Responses semantic events
# --------------------------------------------------------------------------- #
async def responses_stream_generator(
    events: AsyncIterator[Any],
    req: ResponsesRequest,
    response_id: str,
    created: int,
    cache_report: bool = False,
    state: Any = None,
    usage: GenUsage | None = None,
) -> AsyncIterator[str]:
    seq = _Seq()

    def timings_extra() -> dict[str, Any]:
        # per-token: every delta carries the ack's snapshot. Request-side gate: the
        # terminal snapshot rides every request, so post-terminal residue must stay clean.
        if not req.timings_per_token or usage is None:
            return {}
        t = timings_block(state, usage.timings)
        return {"timings": t} if t is not None else {}

    def snapshot(status, output, usage=None, incomplete_reason=None):
        return _response_obj(
            response_id,
            created,
            req.model,
            output,
            status=status,
            usage=usage,
            incomplete_reason=incomplete_reason,
            req=req,
        )

    yield _sse(
        ResponseCreatedEvent(
            type="response.created",
            sequence_number=seq.next(),
            response=snapshot("in_progress", []),
        )
    )
    yield _sse(
        ResponseInProgressEvent(
            type="response.in_progress",
            sequence_number=seq.next(),
            response=snapshot("in_progress", []),
        )
    )

    output_items: list[Any] = []
    output_index = 0
    current: dict[str, Any] | None = None
    finish_reason = "stop"
    usage_pt = 0
    usage_ct = 0
    usage_cached = 0
    usage_reasoning = 0

    want_lps = bool(req.include and "message.output_text.logprobs" in req.include)

    def close_current() -> list[str]:
        nonlocal current, output_index
        frames: list[str] = []
        if current is None:
            return frames
        is_trunc = finish_reason == "length"
        if current["kind"] == "reasoning":
            item_id, text = current["id"], current["text"]
            frames.append(
                _sse(
                    ResponseReasoningTextDoneEvent(
                        type="response.reasoning_text.done",
                        sequence_number=seq.next(),
                        item_id=item_id,
                        output_index=output_index,
                        content_index=0,
                        text=text,
                    )
                )
            )
            done_item = ResponseReasoningItem(
                id=item_id,
                type="reasoning",
                summary=[],
                content=[ReasoningTextContent(type="reasoning_text", text=text)],
                status="incomplete" if is_trunc else "completed",
            )
        elif current["kind"] == "message":
            item_id, text = current["id"], current["text"]
            all_lps = current.get("logprobs") or []
            frames.append(
                _sse(
                    ResponseTextDoneEvent(
                        type="response.output_text.done",
                        sequence_number=seq.next(),
                        item_id=item_id,
                        output_index=output_index,
                        content_index=0,
                        text=text,
                        logprobs=all_lps,
                    )
                )
            )
            part_kwargs: dict[str, Any] = {"logprobs": all_lps} if want_lps else {}
            frames.append(
                _sse(
                    ResponseContentPartDoneEvent(
                        type="response.content_part.done",
                        sequence_number=seq.next(),
                        item_id=item_id,
                        output_index=output_index,
                        content_index=0,
                        part=ResponseOutputText(
                            type="output_text",
                            text=text,
                            annotations=[],
                            **part_kwargs,
                        ),
                    )
                )
            )
            # finish_reason is only non-"stop" once GenDone has arrived, so only the final
            # (truncated) message item is marked incomplete; items closed mid-stream stay completed.
            done_item = ResponseOutputMessage(
                id=item_id,
                role="assistant",
                status="incomplete" if is_trunc else "completed",
                type="message",
                content=[
                    ResponseOutputText(
                        type="output_text",
                        text=text,
                        annotations=[],
                        **part_kwargs,
                    )
                ],
            )
        else:  # function_call
            item_id, args = current["id"], current["args"]
            frames.append(
                _sse(
                    ResponseFunctionCallArgumentsDoneEvent(
                        type="response.function_call_arguments.done",
                        sequence_number=seq.next(),
                        item_id=item_id,
                        output_index=output_index,
                        name=current["name"],
                        arguments=args,
                    )
                )
            )
            done_item = ResponseFunctionToolCall(
                type="function_call",
                id=item_id,
                call_id=current["call_id"],
                name=current["name"],
                arguments=args,
                status="incomplete" if is_trunc else "completed",
            )
        frames.append(
            _sse(
                ResponseOutputItemDoneEvent(
                    type="response.output_item.done",
                    sequence_number=seq.next(),
                    output_index=output_index,
                    item=done_item,
                )
            )
        )
        output_items.append(done_item)
        output_index += 1
        current = None
        return frames

    def open_function_call(
        name: str, ordinal: int | None, *, stable: bool = True
    ) -> list[str]:
        nonlocal current
        frames = close_current()
        item_id = f"fc_{uuid.uuid4().hex}"
        call_id = f"call_{uuid.uuid4().hex[:24]}"
        current = {
            "kind": "function_call",
            "id": item_id,
            "call_id": call_id,
            "name": name,
            "args": "",
            "ordinal": ordinal,
            "stable": stable,
        }
        frames.append(
            _sse(
                ResponseOutputItemAddedEvent(
                    type="response.output_item.added",
                    sequence_number=seq.next(),
                    output_index=output_index,
                    item=ResponseFunctionToolCall(
                        type="function_call",
                        id=item_id,
                        call_id=call_id,
                        name=name,
                        arguments="",
                        status="in_progress",
                    ),
                )
            )
        )
        return frames

    def args_delta_frame(fragment: str) -> str:
        return _sse(
            ResponseFunctionCallArgumentsDeltaEvent(
                type="response.function_call_arguments.delta",
                sequence_number=seq.next(),
                item_id=current["id"],
                output_index=output_index,
                delta=fragment,
                **timings_extra(),
            )
        )

    events = with_keepalive(events, KEEPALIVE_INTERVAL_S)
    try:
        async for ev in events:
            if ev is KEEPALIVE:
                # Data-bearing frame codex ignores but whose arrival resets its
                # stream-idle countdown (SSE comments would not).
                yield _sse(
                    ResponseInProgressEvent(
                        type="response.in_progress",
                        sequence_number=seq.next(),
                        response=snapshot("in_progress", output_items),
                    )
                )

            elif isinstance(ev, ReasoningDelta):
                # Streamed as a first-class reasoning item (codex renders
                # response.reasoning_text.delta natively).
                if ev.text == "":
                    continue
                if current is None or current["kind"] != "reasoning":
                    for f in close_current():
                        yield f
                    item_id = f"rs_{uuid.uuid4().hex}"
                    current = {"kind": "reasoning", "id": item_id, "text": ""}
                    yield _sse(
                        ResponseOutputItemAddedEvent(
                            type="response.output_item.added",
                            sequence_number=seq.next(),
                            output_index=output_index,
                            item=ResponseReasoningItem(
                                id=item_id,
                                type="reasoning",
                                summary=[],
                                status="in_progress",
                            ),
                        )
                    )
                current["text"] += ev.text
                yield _sse(
                    ResponseReasoningTextDeltaEvent(
                        type="response.reasoning_text.delta",
                        sequence_number=seq.next(),
                        item_id=current["id"],
                        output_index=output_index,
                        content_index=0,
                        delta=ev.text,
                        **timings_extra(),
                    )
                )

            elif isinstance(ev, ContentDelta):
                if ev.text == "":
                    continue
                if current is None or current["kind"] != "message":
                    for f in close_current():
                        yield f
                    item_id = f"msg_{uuid.uuid4().hex}"
                    current = {
                        "kind": "message",
                        "id": item_id,
                        "text": "",
                        "logprobs": [],
                    }
                    yield _sse(
                        ResponseOutputItemAddedEvent(
                            type="response.output_item.added",
                            sequence_number=seq.next(),
                            output_index=output_index,
                            item=ResponseOutputMessage(
                                id=item_id,
                                role="assistant",
                                status="in_progress",
                                type="message",
                                content=[],
                            ),
                        )
                    )
                    yield _sse(
                        ResponseContentPartAddedEvent(
                            type="response.content_part.added",
                            sequence_number=seq.next(),
                            item_id=item_id,
                            output_index=output_index,
                            content_index=0,
                            part=ResponseOutputText(
                                type="output_text", text="", annotations=[]
                            ),
                        )
                    )
                current["text"] += ev.text
                delta_lps = (
                    responses_logprobs_content(ev.logprobs or []) if want_lps else []
                )
                if delta_lps:
                    current["logprobs"].extend(delta_lps)
                yield _sse(
                    ResponseTextDeltaEvent(
                        type="response.output_text.delta",
                        sequence_number=seq.next(),
                        item_id=current["id"],
                        output_index=output_index,
                        content_index=0,
                        delta=ev.text,
                        logprobs=delta_lps,
                        **timings_extra(),
                    )
                )

            elif isinstance(ev, ToolCallStart):
                for f in open_function_call(
                    ev.name or "", ev.tool_index, stable=ev.args_prefix_stable
                ):
                    yield f

            elif isinstance(ev, ToolCallArgsDelta):
                if (
                    current is None
                    or current["kind"] != "function_call"
                    or not current.get("stable", True)
                ):
                    continue  # defensive: fragment without an open call or unstable prefix
                current["args"] += ev.fragment
                yield args_delta_frame(ev.fragment)

            elif isinstance(ev, ToolCallsDelta):
                for call in ev.calls:
                    if (
                        current is not None
                        and current["kind"] == "function_call"
                        and current.get("ordinal") == call.tool_index
                    ):
                        # Close of a ToolCallStart-opened call: the final arguments
                        # are authoritative — top up whatever wasn't streamed yet.
                        final = call.parameters or ""
                        if final.startswith(current["args"]):
                            remainder = final[len(current["args"]) :]
                            if remainder:
                                yield args_delta_frame(remainder)
                        current["args"] = final
                        for f in close_current():
                            yield f
                        continue
                    # Standalone complete call (buffered fallback path): open,
                    # deliver the whole arguments, close immediately so codex
                    # persists the item even if the stream dies later.
                    for f in open_function_call(call.name or "", call.tool_index):
                        yield f
                    args = call.parameters or ""
                    current["args"] = args
                    yield args_delta_frame(args)
                    for f in close_current():
                        yield f

            elif isinstance(ev, GenDone):
                if (
                    want_lps
                    and ev.logprobs
                    and current is not None
                    and current["kind"] == "message"
                ):
                    current["logprobs"].extend(responses_logprobs_content(ev.logprobs))
                finish_reason = ev.finish_reason
                usage_pt, usage_ct = ev.prompt_tokens, ev.completion_tokens
                usage_cached = ev.cached_tokens if cache_report else 0
                usage_reasoning = ev.reasoning_tokens
                final_timings = timings_block(state, ev.timings)
                for f in close_current():
                    yield f
                if finish_reason == "length":
                    yield _sse(
                        ResponseIncompleteEvent(
                            type="response.incomplete",
                            sequence_number=seq.next(),
                            response=snapshot(
                                "incomplete",
                                output_items,
                                usage=_usage(
                                    usage_pt, usage_ct, usage_cached, usage_reasoning
                                ),
                                incomplete_reason="max_output_tokens",
                            ),
                            **(
                                {"timings": final_timings}
                                if final_timings is not None
                                else {}
                            ),
                        )
                    )
                else:
                    yield _sse(
                        ResponseCompletedEvent(
                            type="response.completed",
                            sequence_number=seq.next(),
                            response=snapshot(
                                "completed",
                                output_items,
                                usage=_usage(
                                    usage_pt, usage_ct, usage_cached, usage_reasoning
                                ),
                            ),
                            **(
                                {"timings": final_timings}
                                if final_timings is not None
                                else {}
                            ),
                        )
                    )
    except GenerationError as exc:
        # Request failed before/instead of producing output (template render error, over-length
        # prompt). Emit the protocol's terminal failure event so codex sees a clean error rather
        # than a stalled stream that trips its 300s idle timeout into "Reconnecting".
        for f in close_current():
            yield f
        yield _sse(
            ResponseFailedEvent(
                type="response.failed",
                sequence_number=seq.next(),
                response=_response_obj(
                    response_id,
                    created,
                    req.model,
                    output_items,
                    status="failed",
                    usage=_usage(usage_pt, usage_ct, usage_cached, usage_reasoning),
                    # codex reads this code to tell a blown context window from a generic failure.
                    # model_construct because ResponseError.code is a closed Literal in the SDK.
                    error=ResponseError.model_construct(
                        code=exc.code or "server_error", message=str(exc)
                    ),
                ),
            )
        )
    except Exception as exc:  # noqa: BLE001 — never leave the client without a terminal event
        for f in close_current():
            yield f
        yield _sse(
            ResponseFailedEvent(
                type="response.failed",
                sequence_number=seq.next(),
                response=_response_obj(
                    response_id,
                    created,
                    req.model,
                    output_items,
                    status="failed",
                    usage=_usage(usage_pt, usage_ct, usage_cached, usage_reasoning),
                    error=ResponseError(code="server_error", message=str(exc)),
                ),
            )
        )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
class _Seq:
    def __init__(self):
        self._n = -1

    def next(self) -> int:
        self._n += 1
        return self._n


def _sse(event) -> str:
    return f"event: {event.type}\ndata: {event.model_dump_json()}\n\n"


def _error_response(
    status_code: int,
    message: str,
    code: str | None = None,
    *,
    err_type: str = "invalid_request_error",
    param: str | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": err_type,
                "param": param,
                "code": code,
            }
        },
    )
