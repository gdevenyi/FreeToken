from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator, Callable
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from freetoken.core import SamplingParams
from freetoken.message import TokenizeMsg
from freetoken.tokenizer.effort import EFFORT_SCALE, KNOWN_REASONING_EFFORTS

from .api_models import (
    ChatCompletionRequest,
    CompletionRequest,
    DetokenizeRequest,
    ModelCard,
    ModelList,
    TokenizeRequest,
    ToolChoiceObject,
)
from .function_call_parser import ToolCallItem
from .logprobs import completion_logprobs_error, completions_logprobs
from .request_logger import log_request
from .generation import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    ContentDelta,
    GenDone,
    GenerationError,
    GenSpec,
    GenUsage,
    ReasoningDelta,
    ToolCallArgsDelta,
    ToolCallsDelta,
    ToolCallStart,
    generate_events,
    generate_full,
    prerender_error,
    public_request_id,
    render_messages,
    resolve_sampling,
    send_new_request,
    submit_generation,
    timings_block,
)

logger = logging.getLogger(__name__)

#: The wire superset plus "off", DeepSeek's disable synonym that
#: effort_toggle_kwargs has always honored.
_ACCEPTED_EFFORTS = (*KNOWN_REASONING_EFFORTS, "off")


def _thinking_type(req: Any) -> str | None:
    """The DeepSeek-wire thinking toggle, or None for absent/foreign shapes
    (which stay ignored, as extra="allow" ignored them before the field existed)."""
    if isinstance(req.thinking, dict):
        value = req.thinking.get("type")
        if value in ("enabled", "disabled"):
            return value
    return None



def chat_request_to_genspec(
    req: ChatCompletionRequest,
    model_sampling: dict[str, Any],
    default_max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    timings_per_token: bool = False,
    thinking: str | None = None,
) -> GenSpec:
    """OpenAI ChatCompletionRequest -> GenSpec (the OpenAI 'to_sampling_params')."""
    from .model_meta import effort_toggle_kwargs, thinking_override_kwargs

    ctk = req.chat_template_kwargs
    thinking_type = _thinking_type(req)
    if req.reasoning_effort or thinking_type:
        ctk = effort_toggle_kwargs(req.reasoning_effort, ctk, thinking_type=thinking_type)
    ctk = thinking_override_kwargs(thinking, ctk)
    if req.continue_final_message:
        ctk = {**ctk, "continue_final_message": True}
    return GenSpec(
        messages=render_messages([m.model_dump(exclude_none=True) for m in req.messages]),
        sampling_params=_resolve_sampling(
            req,
            model_sampling,
            default_max_tokens=default_max_tokens,
            timings_per_token=timings_per_token,
        ),
        chat_template_kwargs=ctk,
        template_tools=_tools_for_template(req),
        parser_tools=(_all_tool_dicts(req.tools) if _should_parse_tools(req) else None),
    )


def _all_tool_dicts(tools) -> list[dict[str, Any]]:
    return [t.model_dump(exclude_none=True) for t in (tools or [])]


def _maintenance_gate(state: Any) -> JSONResponse | None:
    """503 while the engine is not serving. Distinguishes the startup "loading" phase from a
    runtime cache "rebuild"/"failed" so clients (and the desktop) get an actionable message.
    None when serving."""
    mstate = getattr(state, "maintenance_state", "serving")
    if mstate == "serving":
        return None
    if mstate == "loading":
        msg = "model is still loading"
    elif mstate == "failed":
        msg = "server unavailable: maintenance failed (restart required)"
    else:
        msg = "server unavailable: cache rebuild in progress"
    return JSONResponse({"error": msg}, status_code=503)


def register_openai_routes(
    app: FastAPI,
    get_state: Callable[[], Any],
    get_model_sampling: Callable[[], dict[str, Any]],
) -> None:
    @app.api_route("/v1", methods=["GET", "POST", "HEAD", "OPTIONS"])
    async def v1_root():
        return {"status": "ok"}

    @app.post("/v1/chat/completions")
    async def v1_chat_completions(req: ChatCompletionRequest, request: Request):
        log_request("/v1/chat/completions", req, request)
        state = get_state()
        if (gate := _maintenance_gate(state)) is not None:
            return gate
        return await handle_chat_completion(req, request, state, get_model_sampling())

    @app.post("/v1/completions")
    async def v1_completions(req: CompletionRequest, request: Request):
        log_request("/v1/completions", req, request)
        state = get_state()
        if (gate := _maintenance_gate(state)) is not None:
            return gate
        return await handle_completion(req, request, state, get_model_sampling())

    @app.post("/tokenize")
    @app.post("/v1/tokenize")
    async def tokenize(req: TokenizeRequest):
        return await handle_tokenize(req, get_state())

    @app.post("/detokenize")
    @app.post("/v1/detokenize")
    async def detokenize(req: DetokenizeRequest):
        return await handle_detokenize(req, get_state())

    @app.get("/v1/models")
    async def v1_models():
        state = get_state()
        model_id = _served_model_name(state)
        ctx = _model_context_length(state)
        efforts, default_effort = await _effort_fields(state)
        return ModelList(data=[ModelCard(
            id=model_id,
            root=state.config.model_path,
            max_model_len=ctx,
            context_length=ctx,
            supported_reasoning_efforts=efforts,
            default_reasoning_effort=default_effort,
        )])


# How often a non-streaming handler looks at the transport while the engine generates: one
# second bounds an abandoned request's extra decode work without measurable polling overhead.
_DISCONNECT_POLL_SECONDS = 1.0


async def _await_watching_disconnect(awaitable, request: Request | None, state: Any, uid: int):
    """Run an in-flight generation awaitable while watching the transport - the
    non-streaming twin of stream_with_cancellation. Returns the awaitable's result,
    or None when the client disconnected first: the same shielded abort_user is
    delivered so an abandoned request stops burning decode slots instead of running
    to max_tokens (with --max-running-requests 1 that is a full outage for its
    remaining budget)."""
    gen = asyncio.ensure_future(awaitable)
    if request is None:
        return await gen
    try:
        while True:
            done, _ = await asyncio.wait({gen}, timeout=_DISCONNECT_POLL_SECONDS)
            if done:
                return gen.result()
            if await request.is_disconnected():
                break
    except asyncio.CancelledError:
        # The handler itself was cancelled (server shutdown): deliver the abort,
        # then let the cancellation propagate - mirrors stream_with_cancellation.
        try:
            await asyncio.shield(state.abort_user(uid))
        except Exception:  # noqa: BLE001
            logger.exception("Failed to deliver abort for user %s", uid)
        gen.cancel()
        raise
    # Client gone. Deliver the AbortMsg first (the decode stops sooner), then
    # wind down the drain task; its result is undeliverable either way.
    try:
        await asyncio.shield(state.abort_user(uid))
    except Exception:  # noqa: BLE001
        logger.exception("Failed to deliver abort for user %s", uid)
    gen.cancel()
    try:
        await gen
    except asyncio.CancelledError:
        pass
    except Exception:  # noqa: BLE001 - the response is undeliverable, only cleanup matters
        pass
    return None


def _client_disconnected_response() -> JSONResponse:
    # 499 - nginx's "client closed request". The client is gone; this status is
    # for the access log, not the wire.
    return create_error_response(
        "client disconnected before the response was ready",
        status_code=499,
        err_type="client_disconnected",
    )


async def handle_chat_completion(
    req: ChatCompletionRequest,
    request: Request | None,
    state: Any,
    model_sampling: dict[str, Any],
):
    if req.function_call is not None:
        return create_error_response("function_call is not supported; use tools/tool_choice instead")
    if _response_format_unsupported(req.response_format):
        rtype = (req.response_format or {}).get("type")
        return create_error_response(
            f"response_format.type {rtype!r} is not supported "
            "(supported: text, json_object, json_schema)",
            param="response_format",
        )
    if req.n != 1:
        return create_error_response("Only n=1 is supported", param="n")
    # Case/whitespace and the "off" disable synonym stay accepted here because
    # effort_toggle_kwargs normalizes and honors them downstream.
    effort = req.reasoning_effort.strip().lower() if isinstance(req.reasoning_effort, str) else None
    if effort and effort not in _ACCEPTED_EFFORTS:
        return create_error_response(
            f"reasoning_effort must be one of {', '.join(_ACCEPTED_EFFORTS)}; "
            f"got {req.reasoning_effort!r}",
            param="reasoning_effort",
        )
    if isinstance(req.thinking, dict):
        thinking_type = req.thinking.get("type")
        if thinking_type is not None and thinking_type not in ("enabled", "disabled"):
            return create_error_response(
                f"thinking.type must be 'enabled' or 'disabled'; got {thinking_type!r}",
                param="thinking",
            )

    try:
        default_max_tokens = (
            getattr(state.config, "max_output_tokens", None) or DEFAULT_MAX_OUTPUT_TOKENS
        )
        spec = chat_request_to_genspec(
            req,
            model_sampling,
            default_max_tokens=default_max_tokens,
            # stamp per-token only when the server opted into timings too; llama.cpp
            # publishes partial timings on streams, so non-stream requests never pay
            timings_per_token=bool(
                req.stream
                and req.timings_per_token
                and getattr(state.config, "enable_timings_report", False)
            ),
            thinking=_thinking_policy(state),
        )
    except ValueError as exc:
        return create_error_response(str(exc))

    if req.stream:
        # Non-stream requests already surface render failures as a clean 400
        # through GenerationError; only the stream path needs the pre-check.
        err = await prerender_error(spec, state)
        if err is not None:
            return create_error_response(str(err), code=err.code)

    try:
        uid = await submit_generation(spec, state)
    except GenerationError as exc:
        return create_error_response(str(exc), code=exc.code)

    if req.stream:
        chunks = stream_chat_completion_chunks(uid, req, state, spec)
        if request is not None:
            chunks = state.stream_with_cancellation(chunks, request, uid)
        return StreamingResponse(chunks, media_type="text/event-stream")

    try:
        result = await _await_watching_disconnect(
            generate_full(uid, spec, state, source="/v1/chat/completions"), request, state, uid
        )
    except GenerationError as exc:
        return create_error_response(str(exc), code=exc.code)
    if result is None:
        return _client_disconnected_response()
    message: dict[str, Any] = {"role": "assistant", "content": result.content}
    if result.reasoning:
        message["reasoning_content"] = result.reasoning
    if result.tool_calls:
        message["tool_calls"] = _tool_calls_to_openai(result.tool_calls)

    response: dict[str, Any] = {
        "id": _response_id("chatcmpl", req, public_request_id(uid)),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "system_fingerprint": None,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": result.finish_reason,
            }
        ],
        "usage": _usage(
            result.prompt_tokens,
            result.completion_tokens,
            _reported_cached(state, result.cached_tokens),
            reasoning_tokens=result.reasoning_tokens,
        ),
    }
    if (timings := timings_block(state, result.timings)) is not None:
        response["timings"] = timings
    return response


async def stream_chat_completion_chunks(
    uid: int,
    req: ChatCompletionRequest,
    state: Any,
    spec: GenSpec | None = None,
) -> AsyncIterator[bytes]:
    """Format generate_events() into the OpenAI chat.completion.chunk SSE stream."""
    if spec is None:
        spec = chat_request_to_genspec(req, {})
    yield _sse(
        _chat_chunk(
            req,
            uid,
            [{"delta": {"role": "assistant", "content": ""}, "index": 0, "finish_reason": None}],
        )
    )

    timings: dict[str, Any] | None = None
    # timings ride the stream's LAST chunk only: the usage chunk when the client asked
    # for one, otherwise the finish chunk
    include_usage = bool(req.stream_options and req.stream_options.include_usage)
    continuous_usage = include_usage and bool(req.stream_options.continuous_usage_stats)
    # the final usage chunk reads it too, so a stream cut by an error reports what it produced
    usage = GenUsage()

    def emit(chunk: dict[str, Any]) -> bytes:
        if continuous_usage:
            chunk["usage"] = _reported_usage(state, usage)
        # per-token: post-ack chunks carry the ack's snapshot. Request-side gate: the
        # terminal snapshot rides every request, so post-terminal residue must stay clean.
        if req.timings_per_token and (t := timings_block(state, usage.timings)) is not None:
            chunk["timings"] = t
        return _sse(chunk)

    tool_calls_sent = 0
    open_tool: dict[str, Any] | None = None
    events = generate_events(uid, spec, state, source="/v1/chat/completions", usage=usage)
    while True:
        try:
            ev = await events.__anext__()
        except StopAsyncIteration:
            break
        except GenerationError as exc:
            # Request failed before producing output — emit an error chunk + [DONE] so the
            # client gets a terminal signal instead of a stalled stream.
            yield _sse(
                {"error": {"message": str(exc), "type": "invalid_request_error", "code": exc.code}}
            )
            break
        if isinstance(ev, ReasoningDelta):
            yield emit(
                _chat_chunk(
                    req,
                    uid,
                    [{"delta": {"reasoning_content": ev.text}, "index": 0, "finish_reason": None}],
                )
            )
        elif isinstance(ev, ContentDelta):
            yield emit(
                _chat_chunk(
                    req,
                    uid,
                    [{"delta": {"content": ev.text}, "index": 0, "finish_reason": None}],
                )
            )
        elif isinstance(ev, ToolCallStart):
            open_tool = {
                "index": tool_calls_sent,
                "ordinal": ev.tool_index,
                "sent": "",
                "stable": ev.args_prefix_stable,
            }
            yield emit(
                _chat_chunk(
                    req, uid,
                    [{
                        "delta": {"tool_calls": [{
                            "index": open_tool["index"],
                            "id": _tool_call_id(ev.name, open_tool["index"]),
                            "type": "function",
                            "function": {"name": ev.name, "arguments": ""},
                        }]},
                        "index": 0, "finish_reason": None,
                    }],
                )
            )
        elif isinstance(ev, ToolCallArgsDelta):
            # Clients concatenate argument fragments, so only prefix-stable
            # fragments stream; otherwise the full arguments arrive at close.
            if open_tool is not None and open_tool["stable"] and ev.fragment:
                open_tool["sent"] += ev.fragment
                yield emit(
                    _chat_chunk(
                        req, uid,
                        [{
                            "delta": {"tool_calls": [{
                                "index": open_tool["index"],
                                "function": {"arguments": ev.fragment},
                            }]},
                            "index": 0, "finish_reason": None,
                        }],
                    )
                )
        elif isinstance(ev, ToolCallsDelta):
            for call in ev.calls:
                if open_tool is not None and open_tool["ordinal"] == call.tool_index:
                    # Close of a ToolCallStart-opened call: send whatever of the
                    # final (authoritative) arguments wasn't streamed yet.
                    final = call.parameters or ""
                    remainder = (
                        final[len(open_tool["sent"]):]
                        if final.startswith(open_tool["sent"])
                        else final if not open_tool["sent"] else ""
                    )
                    if remainder:
                        yield emit(
                            _chat_chunk(
                                req, uid,
                                [{
                                    "delta": {"tool_calls": [{
                                        "index": open_tool["index"],
                                        "function": {"arguments": remainder},
                                    }]},
                                    "index": 0, "finish_reason": None,
                                }],
                            )
                        )
                    open_tool = None
                    tool_calls_sent += 1
                    continue
                # Standalone complete call (buffered fallback path).
                for delta in _tool_call_deltas([call], start_index=tool_calls_sent):
                    yield emit(
                        _chat_chunk(
                            req, uid,
                            [{"delta": {"tool_calls": [delta]}, "index": 0, "finish_reason": None}],
                        )
                    )
                tool_calls_sent += 1
        elif isinstance(ev, GenDone):
            timings = ev.timings
            done = _chat_chunk(req, uid, [{"delta": {}, "index": 0, "finish_reason": ev.finish_reason}])
            if not include_usage and (t := timings_block(state, timings)) is not None:
                done["timings"] = t
            yield emit(done)

    if include_usage:
        final: dict[str, Any] = {**_chat_chunk(req, uid, []), "usage": _reported_usage(state, usage)}
        if (t := timings_block(state, timings)) is not None:
            final["timings"] = t
        yield _sse(final)

    yield b"data: [DONE]\n\n"


async def handle_completion(
    req: CompletionRequest,
    request: Request | None,
    state: Any,
    model_sampling: dict[str, Any],
):
    unsupported = _completion_unsupported_reason(req)
    if unsupported is not None:
        return create_error_response(unsupported)
    if req.n != 1:
        return create_error_response("Only n=1 is supported", param="n")
    try:  # surfaces an out-of-range value as a 400 rather than a 500 from the worker
        default_max_tokens = (
            getattr(state.config, "max_output_tokens", None) or DEFAULT_MAX_OUTPUT_TOKENS
        )
        _resolve_sampling(req, model_sampling, default_max_tokens=default_max_tokens)
    except ValueError as exc:
        return create_error_response(str(exc))

    # stamp per-token only when the server opted into timings too; llama.cpp publishes
    # partial timings on streams, so non-stream requests never pay for the snapshots
    timings_pt = bool(
        req.stream
        and req.timings_per_token
        and getattr(state.config, "enable_timings_report", False)
    )
    prompts = [req.prompt] if isinstance(req.prompt, str) else req.prompt
    assert isinstance(prompts, list)
    if not prompts:
        return create_error_response("prompt must not be empty", param="prompt")
    # echo returns what the client sent, so take it before the FIM wrap below
    echo_texts = list(prompts) if req.echo else [""] * len(prompts)
    if req.suffix is not None:
        markers = await _fim_markers(state)
        if markers is None:
            return create_error_response(
                "suffix needs a model with fill-in-the-middle tokens "
                "(<|fim_prefix|> / <|fim_suffix|> / <|fim_middle|>)",
                param="suffix",
            )
        prompts = [_fim_prompt(p, req.suffix, markers) for p in prompts]
    if req.stream:
        if len(prompts) != 1:
            return create_error_response("Streaming completions only support a single text prompt")
        uid = state.new_user()
        try:
            await send_new_request(
                state,
                uid,
                TokenizeMsg(uid=uid, text=prompts[0], sampling_params=_resolve_sampling(
                    req, model_sampling, default_max_tokens=default_max_tokens,
                    timings_per_token=timings_pt,
                )),
            )
        except GenerationError as exc:
            return create_error_response(str(exc))
        chunks = stream_completion_chunks(uid, req, state, echo_text=echo_texts[0])
        if request is not None:
            chunks = state.stream_with_cancellation(chunks, request, uid)
        return StreamingResponse(chunks, media_type="text/event-stream")

    choices: list[dict[str, Any]] = []
    prompt_tokens = 0
    completion_tokens = 0
    cached_tokens = 0
    # with several prompts the timings are the FIRST prompt's own spans, as llama.cpp
    # returns arr[0]; one clock across the sequential loop would mix the later prompts'
    # queue and prefill into the rates
    first_timings: dict[str, Any] | None = None
    for index, prompt in enumerate(prompts):
        uid = state.new_user()
        try:
            await send_new_request(
                state,
                uid,
                TokenizeMsg(
                    uid=uid,
                    text=prompt,
                    sampling_params=_resolve_sampling(
                        req, model_sampling, default_max_tokens=default_max_tokens,
                        timings_per_token=timings_pt,
                    ),
                ),
            )
        except GenerationError as exc:
            return create_error_response(str(exc))

        async def _drain_one(uid: int = uid, index: int = index):
            nonlocal prompt_tokens, completion_tokens, cached_tokens, first_timings
            text = ""
            entries: list[dict] = []
            offsets: list[int] = []
            finish_reason = "stop"
            async for ack in state.wait_for_ack(uid):
                if getattr(ack, "error", None):
                    return create_error_response(ack.error, code=getattr(ack, "error_code", None))
                prompt_tokens += ack.prompt_tokens_delta
                completion_tokens += ack.completion_tokens_delta
                cached_tokens += ack.cached_tokens
                if req.logprobs is not None and ack.logprobs is not None:
                    entries.append(ack.logprobs)
                    offsets.append(ack.text_offset)
                text += ack.incremental_output
                if ack.finished:
                    if index == 0:
                        first_timings = getattr(ack, "timings", None)
                    finish_reason = getattr(ack, "finish_reason", None) or "stop"
                    break
            return {
                "index": index,
                "text": echo_texts[index] + text,
                "finish_reason": finish_reason,
                "logprobs": (
                    completions_logprobs(entries, offsets, len(text)) if req.logprobs is not None else None
                ),
            }

        choice = await _await_watching_disconnect(_drain_one(), request, state, uid)
        if choice is None:
            return _client_disconnected_response()
        if isinstance(choice, JSONResponse):
            return choice
        choices.append(choice)

    response: dict[str, Any] = {
        "id": _response_id("cmpl", req, uuid.uuid4().hex),
        "object": "text_completion",
        "created": int(time.time()),
        "model": req.model,
        "system_fingerprint": None,
        "choices": choices,
        "usage": _usage(prompt_tokens, completion_tokens, _reported_cached(state, cached_tokens)),
    }
    if (timings := timings_block(state, first_timings)) is not None:
        response["timings"] = timings
    return response


def _completion_chunk(req: CompletionRequest, uid: Any, choices: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "id": _response_id("cmpl", req, public_request_id(uid)),
        "object": "text_completion.chunk",
        "created": int(time.time()),
        "model": req.model,
        "choices": choices,
    }


async def stream_completion_chunks(
    uid: int, req: CompletionRequest, state: Any, echo_text: str = ""
) -> AsyncIterator[bytes]:
    usage = GenUsage()
    timings: dict[str, Any] | None = None
    include_usage = bool(req.stream_options and req.stream_options.include_usage)
    continuous_usage = include_usage and bool(req.stream_options.continuous_usage_stats)

    def emit(chunk: dict[str, Any]) -> bytes:
        if continuous_usage:
            chunk["usage"] = _reported_usage(state, usage)
        # per-token: same request-side check as the chat stream - the terminal ack carries
        # timings for every request, only the flag decides who gets a block on each chunk
        if req.timings_per_token and (t := timings_block(state, usage.timings)) is not None:
            chunk["timings"] = t
        return _sse(chunk)

    finish_reason = "stop"
    sent_len = 0
    # Logprob entries of acks with no visible text yet (held partial UTF-8 or stop prefix),
    # carried to the next chunk that has text.
    pending: list[dict] = []
    pending_offsets: list[int] = []
    if echo_text:
        # echo: the prompt leads the stream (vLLM sends it in the first chunk)
        yield _sse(_completion_chunk(req, uid, [{"text": echo_text, "index": 0, "finish_reason": None, "logprobs": None}]))
    async for ack in state.wait_for_ack(uid):
        if getattr(ack, "error", None):
            yield _sse({"error": {"message": ack.error, "type": "invalid_request_error", "code": getattr(ack, "error_code", None)}})
            yield b"data: [DONE]\n\n"
            return
        usage.prompt_tokens += ack.prompt_tokens_delta
        usage.completion_tokens += ack.completion_tokens_delta
        usage.cached_tokens += ack.cached_tokens
        if getattr(ack, "timings", None) is not None:
            timings = ack.timings
            usage.timings = timings
        if req.logprobs is not None and ack.logprobs is not None:
            pending.append(ack.logprobs)
            pending_offsets.append(ack.text_offset)
        if ack.incremental_output:
            sent_len += len(ack.incremental_output)
            yield emit(
                _completion_chunk(
                    req, uid,
                    [{
                        "text": ack.incremental_output,
                        "index": 0,
                        "finish_reason": None,
                        # Mid-stream, a stamp past sent_len is held-back text still to come.
                        "logprobs": (
                            completions_logprobs(
                                pending, pending_offsets, sent_len if ack.finished else None
                            )
                            if pending
                            else None
                        ),
                    }],
                )
            )
            pending, pending_offsets = [], []
        if ack.finished:
            finish_reason = getattr(ack, "finish_reason", None) or "stop"
            break

    # Entries whose text never showed (EOS, a stop string, a trailing partial character)
    # ride the finish chunk, so the joined stream matches the non-stream list.
    done_chunk = _completion_chunk(
        req, uid,
        [{
            "text": "",
            "index": 0,
            "finish_reason": finish_reason,
            "logprobs": (
                completions_logprobs(pending, pending_offsets, sent_len) if pending else None
            ),
        }],
    )
    if not include_usage and (t := timings_block(state, timings)) is not None:
        # as on the chat stream: the finish chunk is the last one only when no usage
        # chunk follows
        done_chunk["timings"] = t
    yield emit(done_chunk)
    if include_usage:
        final: dict[str, Any] = {**_completion_chunk(req, uid, []), "usage": _reported_usage(state, usage)}
        if (t := timings_block(state, timings)) is not None:
            final["timings"] = t
        yield _sse(final)
    yield b"data: [DONE]\n\n"


_FIM_TOKENS = ("<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>")


async def _fim_markers(state: Any) -> tuple[str, str, str] | None:
    """The model's fill-in-the-middle marker strings when its vocabulary has them
    (Qwen / DeepSeek-Coder style), else None."""
    build = getattr(state, "frontend_tokenizer", None)
    if build is None:
        return None
    try:
        manager = await asyncio.to_thread(build)
        tok = manager.tokenizer
        unk = getattr(tok, "unk_token_id", None)
        for name in _FIM_TOKENS:
            tid = tok.convert_tokens_to_ids(name)
            if not isinstance(tid, int) or tid < 0 or tid == unk:
                return None
    except Exception:  # noqa: BLE001 -- no tokenizer here means no FIM
        return None
    return _FIM_TOKENS


def _fim_prompt(prompt: str, suffix: str, markers: tuple[str, str, str]) -> str:
    """The prefix-suffix-middle prompt of OpenAI's ``suffix`` (llama.cpp's /infill)."""
    prefix_tok, suffix_tok, middle_tok = markers
    return f"{prefix_tok}{prompt}{suffix_tok}{suffix}{middle_tok}"


async def handle_tokenize(req: TokenizeRequest, state: Any):
    """``POST /tokenize``: the vLLM / SGLang shape. A raw ``prompt`` is encoded as is; chat
    ``messages`` are rendered through the model's chat template first (the exact prompt a
    generation would tokenize)."""
    if (req.prompt is None) == (req.messages is None):
        return create_error_response("pass exactly one of prompt or messages")
    build = getattr(state, "frontend_tokenizer", None)
    if build is None:
        return create_error_response("no tokenizer on this server", status_code=503, err_type="api_error")
    try:
        manager = await asyncio.to_thread(build)
        if req.prompt is not None:
            ids = await asyncio.to_thread(
                manager.tokenizer.encode, req.prompt, add_special_tokens=req.add_special_tokens
            )
        else:
            messages = render_messages([m.model_dump(exclude_none=True) for m in req.messages])
            ctk = dict(req.chat_template_kwargs)
            if not req.add_generation_prompt:
                ctk["continue_final_message"] = True
            msg = TokenizeMsg(uid=-1, text=messages, sampling_params=SamplingParams(), chat_template_kwargs=ctk)
            text = await asyncio.to_thread(manager.render_prompt, msg)
            ids = await asyncio.to_thread(manager.tokenizer.encode, text, add_special_tokens=False)
    except ValueError as exc:
        return create_error_response(str(exc))
    body: dict[str, Any] = {
        "count": len(ids),
        "max_model_len": _model_context_length(state),
        "tokens": [int(t) for t in ids],
    }
    if req.return_token_strs:
        body["token_strs"] = manager.tokenizer.convert_ids_to_tokens(ids)
    return body


async def handle_detokenize(req: DetokenizeRequest, state: Any):
    build = getattr(state, "frontend_tokenizer", None)
    if build is None:
        return create_error_response("no tokenizer on this server", status_code=503, err_type="api_error")
    manager = await asyncio.to_thread(build)
    text = await asyncio.to_thread(
        manager.tokenizer.decode, req.tokens, skip_special_tokens=req.skip_special_tokens
    )
    return {"prompt": text}


def create_error_response(
    message: str,
    status_code: int = 400,
    err_type: str = "invalid_request_error",
    param: str | None = None,
    code: str | None = None,
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


def _thinking_policy(state: Any) -> str | None:
    """The operator's --thinking policy ("auto" | "on" | "off")."""
    return getattr(state.config, "thinking", None)


def _resolve_sampling(
    req: ChatCompletionRequest | CompletionRequest,
    model_sampling: dict[str, Any],
    default_max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    timings_per_token: bool = False,
) -> SamplingParams:
    sampling_params = resolve_sampling(
        temperature=req.temperature,
        top_k=req.top_k,
        top_p=req.top_p,
        max_tokens=req.max_tokens,
        ignore_eos=req.ignore_eos,
        model_sampling=model_sampling,
        stop=req.stop,
        default_max_tokens=default_max_tokens,
        min_p=req.min_p,
        presence_penalty=req.presence_penalty,
        frequency_penalty=req.frequency_penalty,
        repetition_penalty=req.repetition_penalty,
        logit_bias=req.logit_bias,
        min_tokens=req.min_tokens,
        stop_token_ids=req.stop_token_ids,
        include_stop_str_in_output=req.include_stop_str_in_output,
        skip_special_tokens=req.skip_special_tokens,
        timings_per_token=timings_per_token,
    )
    # completions only: on chat, extra="allow" leaves a client's logprobs readable but unhonoured
    if isinstance(req, CompletionRequest) and req.logprobs is not None:
        sampling_params.logprobs = True
        sampling_params.top_logprobs = req.logprobs
    return sampling_params


def _response_id(prefix: str, req: Any, fallback: Any) -> str:
    """``request_id`` from the client when given (vLLM request_id / SGLang rid), else the
    server's own id."""
    rid = getattr(req, "request_id", None)
    return f"{prefix}-{rid}" if rid else f"{prefix}-{fallback}"


def _tools_for_template(req: ChatCompletionRequest) -> list[dict[str, Any]] | None:
    if not _should_parse_tools(req):
        return None

    tools = req.tools or []
    if isinstance(req.tool_choice, ToolChoiceObject):
        selected = req.tool_choice.function.name
        tools = [tool for tool in tools if tool.function.name == selected]

    return [tool.model_dump(exclude_none=True) for tool in tools]


def _should_parse_tools(req: ChatCompletionRequest) -> bool:
    return bool(req.tools) and req.tool_choice != "none"


def _tool_calls_to_openai(calls: list[ToolCallItem]) -> list[dict[str, Any]]:
    result = []
    for index, call in enumerate(calls):
        result.append(
            {
                "id": _tool_call_id(call.name, index),
                "index": index,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": call.parameters,
                },
            }
        )
    return result


def _tool_call_deltas(calls: list[ToolCallItem], start_index: int = 0) -> list[dict[str, Any]]:
    """OpenAI tool_calls stream deltas. ``start_index`` offsets the slot index so
    calls arriving across multiple ToolCallsDelta events (streamed one per call as
    each closes) don't all collapse into slot 0."""
    deltas: list[dict[str, Any]] = []
    for offset, call in enumerate(calls):
        index = start_index + offset
        call_id = _tool_call_id(call.name, index)
        deltas.append(
            {
                "index": index,
                "id": call_id,
                "type": "function",
                "function": {"name": call.name, "arguments": ""},
            }
        )
        deltas.append({"index": index, "function": {"arguments": call.parameters}})
    return deltas


def _tool_call_id(name: str | None, index: int) -> str:
    prefix = (name or "tool").replace("_", "-")[:24]
    return f"call_{prefix}_{index}_{uuid.uuid4().hex[:8]}"


def _chat_chunk(req: ChatCompletionRequest, uid: int, choices: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "id": _response_id("chatcmpl", req, public_request_id(uid)),
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": req.model,
        "system_fingerprint": None,
        "choices": choices,
    }


def _sse(payload: dict[str, Any]) -> bytes:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode()


def _reported_cached(state: Any, cached_tokens: int) -> int:
    """The prefix-cache hit to report; 0 unless --enable-cache-report is set."""
    return cached_tokens if getattr(state.config, "enable_cache_report", False) else 0


def _reported_usage(state: Any, usage: GenUsage) -> dict[str, Any]:
    return _usage(
        usage.prompt_tokens,
        usage.completion_tokens,
        _reported_cached(state, usage.cached_tokens),
        reasoning_tokens=usage.reasoning_tokens,
    )


def _usage(
    prompt_tokens: int, completion_tokens: int, cached_tokens: int = 0, reasoning_tokens: int = 0
) -> dict[str, Any]:
    usage: dict[str, Any] = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
    # sglang convention: the details object appears only for a nonzero hit, so a
    # disabled report and a 0-token hit serialize identically.
    if cached_tokens > 0:
        usage["prompt_tokens_details"] = {"cached_tokens": cached_tokens}
    # OpenAI's completion_tokens_details.reasoning_tokens: tokens up to and including the
    # reasoning end tag, only when the model emitted one.
    if reasoning_tokens > 0:
        usage["completion_tokens_details"] = {"reasoning_tokens": reasoning_tokens}
    return usage


def _response_format_unsupported(response_format: dict[str, Any] | None) -> bool:
    # No constrained decoding, but clients that ask for JSON in the prompt still need the
    # wire field accepted instead of a hard 400. Unknown types stay refused.
    if response_format is None:
        return False
    return response_format.get("type") not in (None, "text", "json_object", "json_schema")


def _completion_unsupported_reason(req: CompletionRequest) -> str | None:
    if _is_token_prompt(req.prompt):
        return "OpenAI token-id prompt inputs are not supported; pass text prompt strings instead"
    logprobs_error = completion_logprobs_error(req)
    if logprobs_error is not None:
        return logprobs_error
    if _response_format_unsupported(req.response_format):
        rtype = (req.response_format or {}).get("type")
        return (
            f"response_format.type {rtype!r} is not supported "
            "(supported: text, json_object, json_schema)"
        )
    return None


def _is_token_prompt(prompt: Any) -> bool:
    return (
        isinstance(prompt, list)
        and bool(prompt)
        and (
            all(isinstance(item, int) for item in prompt)
            or all(isinstance(item, list) and all(isinstance(token, int) for token in item) for item in prompt)
        )
    )


async def _effort_fields(state: Any) -> tuple[list[str] | None, str | None]:
    """The checkpoint's probed effort vocabulary for /v1/models, or (None, None)
    when there is no frontend tokenizer, it fails to build, or the model has no
    effort knob — a metadata route must never 500 over this."""
    build = getattr(state, "frontend_tokenizer", None)
    if build is None:
        return None, None
    try:
        manager = await asyncio.to_thread(build)
        profile = await asyncio.to_thread(manager.effort_profile)
    except Exception:  # noqa: BLE001 -- metadata only; the generation path reports real faults
        return None, None
    from freetoken.tokenizer.effort import effective_efforts

    served = effective_efforts(profile)
    if not served:
        return None, None
    ordered = sorted(served, key=lambda name: -EFFORT_SCALE.get(name, 0.0))
    return ordered, profile.default


def _served_model_name(state: Any) -> str:
    return getattr(state.config, "served_model_name", None) or state.config.model_path


def _model_context_length(state: Any) -> int | None:
    """The usable context: ``min(model ceiling, allocated KV tokens)``.

    This is the clamp the engine already applies to its own ``max_seq_len`` (Engine.__init__,
    and again in _refresh_seq_state after a rebuild), and therefore the limit the scheduler
    admits against. The frontend process holds an unclamped ServerArgs copy, so publishing
    ``config.max_seq_len`` here advertised the model's positional ceiling instead -- e.g. 262144
    against a 178176-token pool, so the engine and this route disagreed about one quantity.

    `ft launch` reads this to size each agent's context window (opencode's ``limit.context``,
    codex's ``context_window``, ``CLAUDE_CODE_MAX_CONTEXT_TOKENS``), so overstating it stops
    those agents compacting before the KV pool runs out.
    """
    try:  # never 500 a metadata route: max_seq_len walks into the HF config on some builds
        value = int(state.config.max_seq_len)
    except Exception:  # noqa: BLE001
        return None
    if value <= 0:
        return None
    try:
        # Local import: api_server imports this module, so a module-level one would cycle.
        from .api_server import kv_pool_geometry

        num_pages, page_size = kv_pool_geometry(state)
        kv_tokens = num_pages * page_size
    except Exception:  # noqa: BLE001
        kv_tokens = 0
    # 0 before the ("meta", …) ack lands; the ceiling is the best available answer until then.
    return min(value, kv_tokens) if kv_tokens > 0 else value
