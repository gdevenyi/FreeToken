from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator, Callable, Sequence
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from freetoken.core import SamplingParams
from freetoken.message import TokenizeMsg
from freetoken.tokenizer.effort import EFFORT_SCALE, KNOWN_REASONING_EFFORTS

from .api_models import (
    MAX_N,
    MAX_PROMPTS,
    DetokenizeRequest,
    TokenizeRequest,
    ChatCompletionRequest,
    CompletionRequest,
    ModelCard,
    ModelList,
    ToolChoiceObject,
)
from .function_call_parser import ToolCallItem
from .request_logger import log_request
from .logprobs import (
    chat_logprobs_error,
    chat_logprobs_object,
    completion_logprobs_error,
    completions_logprobs,
    untrimmed,
)
from .generation import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    ContentDelta,
    GenDone,
    GenerationError,
    GenSpec,
    GenTimings,
    ReasoningDelta,
    ToolCallArgsDelta,
    ToolCallsDelta,
    ToolCallStart,
    build_metrics,
    chat_logprob_markers,
    chat_logprobs_supported,
    generate_events,
    generate_full,
    metrics_enabled,
    prerender_error,
    render_messages,
    resolve_sampling,
    submit_generation,
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
    default_thinking_mode: str | None = None,
) -> GenSpec:
    """OpenAI ChatCompletionRequest -> GenSpec (the OpenAI 'to_sampling_params')."""
    from .model_meta import apply_default_thinking_mode, effort_toggle_kwargs

    ctk = req.chat_template_kwargs
    thinking_type = _thinking_type(req)
    if req.reasoning_effort or thinking_type:
        ctk = effort_toggle_kwargs(req.reasoning_effort, ctk, thinking_type=thinking_type)
    ctk = apply_default_thinking_mode(ctk, default_thinking_mode)
    if req.continue_final_message:
        ctk = {**ctk, "continue_final_message": True}
    sampling_params = _resolve_sampling(req, model_sampling, default_max_tokens=default_max_tokens)
    sampling_params.logprobs = bool(req.logprobs)
    sampling_params.top_logprobs = req.top_logprobs or 0
    return GenSpec(
        messages=render_messages([m.model_dump(exclude_none=True) for m in req.messages]),
        sampling_params=sampling_params,
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

    @app.get("/v1/models/{model_id:path}")
    async def v1_model(model_id: str):
        """Retrieve metadata for the currently served model.

        OpenAI-compatible clients may use this endpoint to validate a model
        name before sending completion requests.
        """
        if not model_id:
            # The :path converter also matches "/v1/models/", which used to redirect to the list.
            return await v1_models()
        state = get_state()
        served_model_id = _served_model_name(state)

        if model_id != served_model_id:
            return create_error_response(
                f"The model '{model_id}' does not exist",
                status_code=404,
                err_type="invalid_request_error",
                param="model",
                code="model_not_found",
            )

        ctx = _model_context_length(state)
        efforts, default_effort = await _effort_fields(state)
        return ModelCard(
            id=served_model_id,
            root=state.config.model_path,
            max_model_len=ctx,
            context_length=ctx,
            supported_reasoning_efforts=efforts,
            default_reasoning_effort=default_effort,
        )

# How often a non-streaming handler looks at the transport while the engine
# generates. The streaming path checks per chunk; one second bounds an abandoned
# request's extra decode work without measurable polling overhead.
_DISCONNECT_POLL_SECONDS = 1.0


async def _await_watching_disconnect(
    awaitable, request: Request | None, state: Any, uids: Sequence[int]
):
    """Run an in-flight generation awaitable while watching the transport — the
    non-streaming twin of stream_with_cancellation. Returns the awaitable's result,
    or None when the client disconnected first: the same shielded abort_user is
    delivered for every uid (an ``n > 1`` fan-out) so an abandoned request stops
    burning decode slots instead of running to max_tokens (with
    --max-running-requests 1 that is a full outage for its remaining budget)."""
    gen = asyncio.ensure_future(awaitable)
    if request is None:
        return await gen

    async def abort_all():
        for uid in uids:
            try:
                await asyncio.shield(state.abort_user(uid))
            except Exception:  # noqa: BLE001
                logger.exception("Failed to deliver abort for user %s", uid)

    try:
        while True:
            done, _ = await asyncio.wait({gen}, timeout=_DISCONNECT_POLL_SECONDS)
            if done:
                return gen.result()
            if await request.is_disconnected():
                break
    except asyncio.CancelledError:
        # The handler itself was cancelled (server shutdown): deliver the abort,
        # then let the cancellation propagate — mirrors stream_with_cancellation.
        await abort_all()
        gen.cancel()
        raise
    # Client gone. Deliver the AbortMsg first (the decode stops sooner), then
    # wind down the drain task; its result is undeliverable either way.
    await abort_all()
    gen.cancel()
    try:
        await gen
    except asyncio.CancelledError:
        pass
    except Exception:  # noqa: BLE001 — the response is undeliverable, only cleanup matters
        pass
    return None


def _client_disconnected_response() -> JSONResponse:
    # 499 — nginx's "client closed request". The client is gone; this status is
    # for the access log, not the wire.
    return create_error_response(
        "client disconnected before the response was ready",
        status_code=499,
        err_type="client_disconnected",
    )


async def _chat_logprobs_conflict(req: ChatCompletionRequest, state: Any) -> str | None:
    """Refuse chat logprobs when the server's parsers can hide or reclassify generated
    tokens in a way the logprob segmenter cannot follow: entries must cover message
    content only, never reasoning or tool-call tokens. The raw /v1/completions path has
    no semantic layer and keeps full logprobs support."""
    if not req.logprobs:
        return None
    markers = await chat_logprob_markers(state)
    if not chat_logprobs_supported(state, False, markers):
        return (
            "logprobs on /v1/chat/completions are not supported with the "
            f"{state.config.reasoning_parser!r} reasoning parser: reasoning tokens cannot "
            "be told apart from message content. Use /v1/completions for raw token logprobs."
        )
    if not chat_logprobs_supported(state, _should_parse_tools(req), markers):
        return (
            "logprobs with tool parsing on /v1/chat/completions are not supported for the "
            f"{state.config.tool_call_parser!r} tool-call parser: tool-call tokens cannot be "
            "told apart from message content. Send tool_choice='none' or use "
            "/v1/completions for raw token logprobs."
        )
    return None


async def handle_chat_completion(
    req: ChatCompletionRequest,
    request: Request | None,
    state: Any,
    model_sampling: dict[str, Any],
):
    if req.function_call is not None:
        return create_error_response("function_call is not supported; use tools/tool_choice instead")
    logprobs_error = chat_logprobs_error(req)
    if logprobs_error is not None:
        return create_error_response(logprobs_error, param="top_logprobs")
    logprobs_conflict = await _chat_logprobs_conflict(req, state)
    if logprobs_conflict is not None:
        return create_error_response(logprobs_conflict, param="logprobs")
    if _response_format_unsupported(req.response_format):
        rtype = (req.response_format or {}).get("type")
        return create_error_response(
            f"response_format.type {rtype!r} is not supported "
            "(supported: text, json_object, json_schema)",
            param="response_format",
        )
    if not 1 <= req.n <= MAX_N:
        return create_error_response(f"n must be between 1 and {MAX_N}", param="n")
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
            default_thinking_mode=getattr(state.config, "default_thinking_mode", "auto"),
        )
    except ValueError as exc:
        return create_error_response(str(exc))

    if req.stream:
        # Non-stream requests already surface render failures as a clean 400
        # through GenerationError; only the stream path needs the pre-check.
        err = await prerender_error(spec, state)
        if err is not None:
            return create_error_response(str(err), code=err.code)

    # n > 1: one generation per choice, submitted together (the prefix cache serves the
    # shared prompt after the first prefill).
    uids: list[int] = []
    try:
        for _ in range(req.n):
            uids.append(await submit_generation(spec, state))
    except GenerationError as exc:
        for u in uids:  # choices already queued before a later one failed
            await state.abort_user(u)
        return create_error_response(str(exc), code=exc.code)
    uid = uids[0]

    if req.stream:
        if req.n == 1:
            chunks = stream_chat_completion_chunks(uid, req, state, spec)
        else:
            chunks = _merge_streams(
                [
                    stream_chat_completion_chunks(u, req, state, spec, index=i, terminal=False)
                    for i, u in enumerate(uids)
                ],
                lambda: _chat_chunk(req, uid, []),
                include_usage=bool(req.stream_options and req.stream_options.include_usage),
                state=state,
            )
        if request is not None:
            chunks = state.stream_with_cancellation(chunks, request, uids)
        return StreamingResponse(chunks, media_type="text/event-stream")

    tasks = [
        asyncio.ensure_future(generate_full(u, spec, state, source="/v1/chat/completions"))
        for u in uids
    ]
    try:
        results = await _await_watching_disconnect(asyncio.gather(*tasks), request, state, uids)
    except GenerationError as exc:
        # gather does not cancel the other choices; the aborts below drop the events they
        # wait on, so without the cancel they would stay pending forever
        for task in tasks:
            task.cancel()
        for u in uids:
            await state.abort_user(u)
        return create_error_response(str(exc), code=exc.code)
    if results is None:
        return _client_disconnected_response()
    choices: list[dict[str, Any]] = []
    for index, result in enumerate(results):
        message: dict[str, Any] = {"role": "assistant", "content": result.content}
        if result.reasoning:
            message["reasoning_content"] = result.reasoning
        if result.tool_calls:
            message["tool_calls"] = _tool_calls_to_openai(result.tool_calls)
        choices.append({
            "index": index,
            "message": message,
            # OpenAI always sends the key, null when logprobs were not requested
            "logprobs": chat_logprobs_object(result.logprobs) if req.logprobs else None,
            "finish_reason": result.finish_reason,
        })

    first = results[0]
    response: dict[str, Any] = {
        "id": _response_id("chatcmpl", req, uid),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "system_fingerprint": None,
        "choices": choices,
        "usage": _usage(
            first.prompt_tokens,
            sum(r.completion_tokens for r in results),
            _reported_cached(state, max(r.cached_tokens for r in results)),
            reasoning_tokens=sum(r.reasoning_tokens for r in results),
        ),
    }
    if metrics_enabled(state):
        response["metrics"] = build_metrics(
            prompt_tokens=first.prompt_tokens,
            completion_tokens=first.completion_tokens,
            cached_tokens=first.cached_tokens,
            timings=first.timings,
        )
    return response


async def stream_chat_completion_chunks(
    uid: int,
    req: ChatCompletionRequest,
    state: Any,
    spec: GenSpec | None = None,
    index: int = 0,
    terminal: bool = True,
) -> AsyncIterator[bytes]:
    """Format generate_events() into the OpenAI chat.completion.chunk SSE stream.

    ``index`` is the choice index (``n > 1`` runs one of these per choice); with
    ``terminal=False`` the usage chunk and ``[DONE]`` are left to the merger, which reads
    this stream's usage from the ``_StreamUsage`` yielded last."""
    if spec is None:
        spec = chat_request_to_genspec(
            req,
            {},
            default_thinking_mode=getattr(state.config, "default_thinking_mode", "auto"),
        )
    yield _sse(
        _chat_chunk(
            req,
            uid,
            [{"delta": {"role": "assistant", "content": ""}, "index": index, "finish_reason": None}],
        )
    )

    prompt_tokens = 0
    completion_tokens = 0
    cached_tokens = 0
    reasoning_tokens = 0
    timings = GenTimings()
    tool_calls_sent = 0
    open_tool: dict[str, Any] | None = None
    events = generate_events(uid, spec, state, source="/v1/chat/completions")
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
            yield _sse(
                _chat_chunk(
                    req,
                    uid,
                    [{"delta": {"reasoning_content": ev.text}, "index": index, "finish_reason": None}],
                )
            )
        elif isinstance(ev, ContentDelta):
            choice: dict[str, Any] = {"delta": {"content": ev.text}, "index": index, "finish_reason": None}
            if ev.logprobs:
                choice["logprobs"] = chat_logprobs_object(ev.logprobs)
            yield _sse(_chat_chunk(req, uid, [choice]))
        elif isinstance(ev, ToolCallStart):
            open_tool = {
                "index": tool_calls_sent,
                "ordinal": ev.tool_index,
                "sent": "",
                "stable": ev.args_prefix_stable,
            }
            yield _sse(
                _chat_chunk(
                    req, uid,
                    [{
                        "delta": {"tool_calls": [{
                            "index": open_tool["index"],
                            "id": _tool_call_id(ev.name, open_tool["index"]),
                            "type": "function",
                            "function": {"name": ev.name, "arguments": ""},
                        }]},
                        "index": index, "finish_reason": None,
                    }],
                )
            )
        elif isinstance(ev, ToolCallArgsDelta):
            # Clients concatenate argument fragments, so only prefix-stable
            # fragments stream; otherwise the full arguments arrive at close.
            if open_tool is not None and open_tool["stable"] and ev.fragment:
                open_tool["sent"] += ev.fragment
                yield _sse(
                    _chat_chunk(
                        req, uid,
                        [{
                            "delta": {"tool_calls": [{
                                "index": open_tool["index"],
                                "function": {"arguments": ev.fragment},
                            }]},
                            "index": index, "finish_reason": None,
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
                        yield _sse(
                            _chat_chunk(
                                req, uid,
                                [{
                                    "delta": {"tool_calls": [{
                                        "index": open_tool["index"],
                                        "function": {"arguments": remainder},
                                    }]},
                                    "index": index, "finish_reason": None,
                                }],
                            )
                        )
                    open_tool = None
                    tool_calls_sent += 1
                    continue
                # Standalone complete call (buffered fallback path).
                for delta in _tool_call_deltas([call], start_index=tool_calls_sent):
                    yield _sse(
                        _chat_chunk(
                            req, uid,
                            [{"delta": {"tool_calls": [delta]}, "index": index, "finish_reason": None}],
                        )
                    )
                tool_calls_sent += 1
        elif isinstance(ev, GenDone):
            prompt_tokens = ev.prompt_tokens
            completion_tokens = ev.completion_tokens
            cached_tokens = ev.cached_tokens
            reasoning_tokens = ev.reasoning_tokens
            done: dict[str, Any] = {"delta": {}, "index": index, "finish_reason": ev.finish_reason}
            if ev.logprobs:
                done["logprobs"] = chat_logprobs_object(ev.logprobs)
            yield _sse(_chat_chunk(req, uid, [done]))
            timings = ev.timings

    usage = _usage(
        prompt_tokens, completion_tokens, _reported_cached(state, cached_tokens),
        reasoning_tokens=reasoning_tokens,
    )
    if not terminal:
        yield _StreamUsage(usage)
        return
    if req.stream_options and req.stream_options.include_usage:
        final: dict[str, Any] = {**_chat_chunk(req, uid, []), "usage": usage}
        if metrics_enabled(state):
            final["metrics"] = build_metrics(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cached_tokens=cached_tokens,
                timings=timings,
            )
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
    if not 1 <= req.n <= MAX_N:
        return create_error_response(f"n must be between 1 and {MAX_N}", param="n")
    num_prompts = 1 if isinstance(req.prompt, str) else len(req.prompt)
    if num_prompts * req.n > MAX_PROMPTS:
        return create_error_response(
            f"prompts x n must be at most {MAX_PROMPTS}, got {num_prompts * req.n}", param="prompt"
        )
    try:  # surfaces an out-of-range value as a 400 rather than a 500 from the worker
        default_max_tokens = (
            getattr(state.config, "max_output_tokens", None) or DEFAULT_MAX_OUTPUT_TOKENS
        )
        sampling = _resolve_sampling(req, model_sampling, default_max_tokens=default_max_tokens)
    except ValueError as exc:
        return create_error_response(str(exc))
    if sampling.allowed_token_ids:
        # the tokenizer worker checks again; this one answers before anything is queued
        vocab = await _vocab_len(state)
        if vocab is not None and sampling.allowed_token_ids[-1] >= vocab:
            return create_error_response(
                f"allowed_token_ids must be below the vocabulary size {vocab}",
                param="allowed_token_ids",
            )

    prompts = [req.prompt] if isinstance(req.prompt, str) else req.prompt
    assert isinstance(prompts, list)
    if req.suffix is not None:
        markers = await _fim_markers(state)
        if markers is None:
            return create_error_response(
                "suffix needs a model with fill-in-the-middle tokens "
                "(<|fim_prefix|> / <|fim_suffix|> / <|fim_middle|>)",
                param="suffix",
            )
        prompts = [_fim_prompt(p, req.suffix, markers) for p in prompts]
    echo_texts = list(prompts) if req.echo else [""] * len(prompts)
    if req.suffix is not None and req.echo:
        # echo returns what the client sent, not the FIM-wrapped prompt
        echo_texts = [req.prompt] if isinstance(req.prompt, str) else list(req.prompt)

    # choices are ordered prompt-major: prompt i, sample k -> index i * n + k (OpenAI)
    uids: list[int] = []
    for prompt in prompts:
        for _ in range(req.n):
            uid = state.new_user()
            await state.send_one(TokenizeMsg(uid=uid, text=prompt, sampling_params=sampling))
            uids.append(uid)

    if req.stream:
        if len(uids) == 1:
            chunks = stream_completion_chunks(uids[0], req, state, echo_text=echo_texts[0])
        else:
            chunks = _merge_streams(
                [
                    stream_completion_chunks(
                        u, req, state, index=i, terminal=False, echo_text=echo_texts[i // req.n]
                    )
                    for i, u in enumerate(uids)
                ],
                lambda: _completion_chunk(req, uids[0], []),
                include_usage=bool(req.stream_options and req.stream_options.include_usage),
                state=state,
            )
        if request is not None:
            chunks = state.stream_with_cancellation(chunks, request, uids)
        return StreamingResponse(chunks, media_type="text/event-stream")

    async def _drain_one(uid: int, index: int) -> dict[str, Any] | JSONResponse:
        text = ""
        entries: list[dict] = []
        finish_reason = "stop"
        prompt_tokens = completion_tokens = cached_tokens = 0
        async for ack in state.wait_for_ack(uid):
            if getattr(ack, "error", None):
                return create_error_response(ack.error, code=getattr(ack, "error_code", None))
            prompt_tokens += ack.prompt_tokens_delta
            completion_tokens += ack.completion_tokens_delta
            cached_tokens += ack.cached_tokens
            text += ack.incremental_output
            if req.logprobs is not None:
                entries.extend(untrimmed(ack.logprobs))
            if ack.finished:
                finish_reason = getattr(ack, "finish_reason", None) or "stop"
                break
        return {
            "index": index,
            "text": echo_texts[index // req.n] + text,
            "finish_reason": finish_reason,
            "logprobs": (
                completions_logprobs(entries, len(text), req.return_tokens_as_token_ids)
                if req.logprobs is not None
                else None
            ),
            "_usage": (prompt_tokens, completion_tokens, cached_tokens),
        }

    drained = await _await_watching_disconnect(
        asyncio.gather(*(_drain_one(u, i) for i, u in enumerate(uids))), request, state, uids
    )
    if drained is None:
        return _client_disconnected_response()
    choices: list[dict[str, Any]] = []
    prompt_tokens = completion_tokens = cached_tokens = 0
    for choice in drained:
        if isinstance(choice, JSONResponse):
            return choice
        p, c, cached = choice.pop("_usage")
        # the prompt is counted once per prompt, not once per sample of it
        if choice["index"] % req.n == 0:
            prompt_tokens += p
        completion_tokens += c
        cached_tokens = max(cached_tokens, cached)
        choices.append(choice)

    return {
        "id": _response_id("cmpl", req, uuid.uuid4().hex),
        "object": "text_completion",
        "created": int(time.time()),
        "model": req.model,
        "system_fingerprint": None,
        "choices": choices,
        "usage": _usage(prompt_tokens, completion_tokens, _reported_cached(state, cached_tokens)),
    }


def _completion_chunk(req: CompletionRequest, uid: Any, choices: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "id": _response_id("cmpl", req, uid),
        "object": "text_completion.chunk",
        "created": int(time.time()),
        "model": req.model,
        "choices": choices,
    }


async def stream_completion_chunks(
    uid: int,
    req: CompletionRequest,
    state: Any,
    index: int = 0,
    terminal: bool = True,
    echo_text: str = "",
) -> AsyncIterator[bytes]:
    prompt_tokens = 0
    sent_len = 0
    # Logprob entries of acks with no visible text yet (a held partial UTF-8 character, an
    # EOS), carried to the next chunk that has text; a reply can bring several at once.
    pending: list[dict] = []
    completion_tokens = 0
    cached_tokens = 0
    finish_reason = "stop"
    if echo_text:
        # echo: the prompt leads the stream (vLLM sends it in the first chunk)
        yield _sse(_completion_chunk(req, uid, [{"text": echo_text, "index": index, "finish_reason": None, "logprobs": None}]))
    async for ack in state.wait_for_ack(uid):
        if getattr(ack, "error", None):
            error = {"message": ack.error, "type": "invalid_request_error", "code": getattr(ack, "error_code", None)}
            yield _sse({"error": error})
            if terminal:
                yield b"data: [DONE]\n\n"
            return
        prompt_tokens += ack.prompt_tokens_delta
        completion_tokens += ack.completion_tokens_delta
        cached_tokens += ack.cached_tokens
        if req.logprobs is not None:
            pending.extend(untrimmed(ack.logprobs))
        if ack.incremental_output:
            sent_len += len(ack.incremental_output)
            yield _sse(
                _completion_chunk(
                    req, uid,
                    [{
                        "text": ack.incremental_output,
                        "index": index,
                        "finish_reason": None,
                        # Mid-stream, a stamp past sent_len is held-back text still to come.
                        "logprobs": (
                            completions_logprobs(
                                pending,
                                sent_len if ack.finished else None,
                                req.return_tokens_as_token_ids,
                            )
                            if pending
                            else None
                        ),
                    }],
                )
            )
            pending = []
        if ack.finished:
            finish_reason = getattr(ack, "finish_reason", None) or "stop"
            break

    # Entries whose text never showed (EOS, a stop string, a trailing partial character)
    # ride the finish chunk, so the joined stream matches the non-stream list.
    yield _sse(
        _completion_chunk(
            req, uid,
            [{
                "text": "",
                "index": index,
                "finish_reason": finish_reason,
                "logprobs": (
                    completions_logprobs(pending, sent_len, req.return_tokens_as_token_ids)
                    if pending
                    else None
                ),
            }],
        )
    )
    usage = _usage(prompt_tokens, completion_tokens, _reported_cached(state, cached_tokens))
    if not terminal:
        yield _StreamUsage(usage)
        return
    if req.stream_options and req.stream_options.include_usage:
        yield _sse({**_completion_chunk(req, uid, []), "usage": usage})
    yield b"data: [DONE]\n\n"


class _StreamUsage:
    """Yielded last by a non-terminal sub-stream: its usage dict for the merger."""

    __slots__ = ("usage",)

    def __init__(self, usage: dict[str, Any]) -> None:
        self.usage = usage


async def _merge_streams(
    streams: list[AsyncIterator[Any]],
    envelope: Callable[[], dict[str, Any]],
    *,
    include_usage: bool,
    state: Any,
) -> AsyncIterator[bytes]:
    """Interleave the SSE chunks of several choice streams into one response (``n > 1``),
    then one usage chunk (prompt counted once, completions summed) and one ``[DONE]``."""
    queue: asyncio.Queue[tuple[int, Any]] = asyncio.Queue()
    usages: dict[int, dict[str, Any]] = {}

    async def pump(slot: int, stream: AsyncIterator[Any]) -> None:
        try:
            async for item in stream:
                if isinstance(item, _StreamUsage):
                    usages[slot] = item.usage
                else:
                    await queue.put((slot, item))
        finally:
            await queue.put((slot, None))

    tasks = [asyncio.create_task(pump(i, st)) for i, st in enumerate(streams)]
    try:
        open_slots = len(streams)
        while open_slots:
            slot, item = await queue.get()
            if item is None:
                open_slots -= 1
                continue
            yield item
    finally:
        for task in tasks:
            task.cancel()
    if include_usage and usages:
        first = usages[min(usages)]
        merged = _usage(
            first["prompt_tokens"],
            sum(u["completion_tokens"] for u in usages.values()),
            max(u.get("prompt_tokens_details", {}).get("cached_tokens", 0) for u in usages.values()),
            reasoning_tokens=sum(
                u.get("completion_tokens_details", {}).get("reasoning_tokens", 0) for u in usages.values()
            ),
        )
        yield _sse({**envelope(), "usage": merged})
    yield b"data: [DONE]\n\n"


_FIM_TOKENS = ("<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>")


async def _vocab_len(state: Any) -> int | None:
    """``len(tokenizer)`` (added tokens included), or None without a frontend tokenizer."""
    build = getattr(state, "frontend_tokenizer", None)
    if build is None:
        return None
    try:
        manager = await asyncio.to_thread(build)
        return len(manager.tokenizer)
    except Exception:  # noqa: BLE001 -- the tokenizer worker still range-checks
        return None


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


def _resolve_sampling(
    req: ChatCompletionRequest | CompletionRequest,
    model_sampling: dict[str, Any],
    default_max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
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
        allowed_token_ids=req.allowed_token_ids if isinstance(req, CompletionRequest) else None,
    )
    if isinstance(req, CompletionRequest) and req.logprobs is not None:
        sampling_params.logprobs = True
        sampling_params.top_logprobs = req.logprobs
        if sampling_params.allowed_token_ids:
            _check_restricted_logprobs(sampling_params)
    return sampling_params


def _check_restricted_logprobs(sp: SamplingParams) -> None:
    """Logprobs restricted to allowed_token_ids are vLLM's processed_logprobs at temperature 0:
    log_softmax over the allowed ids of the unprocessed logits. Anything vLLM would also fold
    into them is refused rather than silently left out."""
    if sp.temperature > 0:
        raise ValueError("allowed_token_ids with logprobs needs temperature=0")
    if sp.presence_penalty or sp.frequency_penalty or sp.repetition_penalty != 1.0:
        raise ValueError("allowed_token_ids with logprobs does not support penalties")
    if sp.min_tokens > 0:
        raise ValueError("allowed_token_ids with logprobs does not support min_tokens")
    if sp.logit_bias:
        raise ValueError("allowed_token_ids with logprobs does not support logit_bias")


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
        "id": _response_id("chatcmpl", req, uid),
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
    # Soft-accept OpenAI json_object/json_schema. FreeToken has no constrained
    # decoding; clients that already ask for JSON in the prompt still need the
    # wire field accepted (HTTP 200) instead of a hard 400. Unknown types 400.
    if response_format is None:
        return False
    return response_format.get("type") not in (None, "text", "json_object", "json_schema")


# vLLM /v1/completions fields that change the result and are not implemented here: a set
# value is a 400 rather than silently dropped by extra="allow".
_UNIMPLEMENTED_COMPLETION_FIELDS = (
    "logprob_token_ids",
    "prompt_logprobs",
    "guided_json",
    "guided_regex",
    "guided_choice",
    "guided_grammar",
    "structured_outputs",
    "bad_words",
)


def _is_set(value: Any) -> bool:
    # not truthiness: prompt_logprobs=0 is a real request, and 0 == False
    if value is None or value is False:
        return False
    return not (isinstance(value, (str, list, dict)) and not value)


def _empty_prompt(req: CompletionRequest) -> bool:
    prompt = req.prompt
    if isinstance(prompt, str):
        # suffix-only fill-in-the-middle is a real request
        return not prompt and req.suffix is None
    if not prompt:
        return True
    return any(
        (isinstance(item, list) and not item)
        or (isinstance(item, str) and not item and req.suffix is None)
        for item in prompt
    )


def _completion_unsupported_reason(req: CompletionRequest) -> str | None:
    for name in _UNIMPLEMENTED_COMPLETION_FIELDS:
        if _is_set(getattr(req, name, None)):
            return f"{name} is not supported by this server"
    if _empty_prompt(req):
        return "prompt must not be empty"
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
