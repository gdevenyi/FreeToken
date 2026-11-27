from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import replace
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from freetoken.core import SamplingParams
from freetoken.message import TokenizeMsg
from freetoken.tokenizer.effort import EFFORT_SCALE, KNOWN_REASONING_EFFORTS

from .api_models import (
    MAX_N_CHOICES,
    MAX_PROMPTS,
    ChatCompletionRequest,
    CompletionRequest,
    DetokenizeRequest,
    ModelCard,
    ModelList,
    TokenizeRequest,
    ToolChoiceObject,
)
from .function_call_parser import ToolCallItem
from .generation import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    MAX_TOKEN_ID,
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
from .logprobs import (
    MAX_CHAT_LOGPROBS,
    chat_logprobs_content,
    completion_logprobs_error,
    completions_logprobs,
    reply_logprob_pairs,
)
from .request_logger import log_request

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
    reasoning_parser: str | None = None,
    tool_call_parser: str | None = None,
) -> GenSpec:
    """OpenAI ChatCompletionRequest -> GenSpec (the OpenAI 'to_sampling_params')."""
    from freetoken.grammar.schema import build_structured_output_spec

    from .model_meta import effort_toggle_kwargs, thinking_override_kwargs

    ctk = req.chat_template_kwargs
    thinking_type = _thinking_type(req)
    if req.reasoning_effort or thinking_type:
        ctk = effort_toggle_kwargs(
            req.reasoning_effort, ctk, thinking_type=thinking_type
        )
    ctk = thinking_override_kwargs(thinking, ctk)
    if req.continue_final_message:
        ctk = {**ctk, "continue_final_message": True}
    if req.parallel_tool_calls is False:
        ctk = {**ctk, "parallel_tool_calls": False}
    template_tools, parser_tools, tool_choice_mode = _resolve_chat_tools(req)
    disable_parallel = bool(req.parallel_tool_calls is False)
    sp = _resolve_sampling(
        req,
        model_sampling,
        default_max_tokens=default_max_tokens,
        timings_per_token=timings_per_token,
    )
    sp.structured_output = build_structured_output_spec(
        response_format=req.response_format,
        guided_json=req.guided_json,
        guided_regex=req.guided_regex,
        guided_choice=req.guided_choice,
        guided_grammar=req.guided_grammar,
        structured_outputs=req.structured_outputs,
        tools=template_tools,
        tool_choice_mode=tool_choice_mode,
        tool_call_parser=tool_call_parser,
        disable_parallel_tool_use=disable_parallel,
        reasoning_parser=reasoning_parser,
        chat_template_kwargs=ctk,
    )
    return GenSpec(
        messages=render_messages(
            [m.model_dump(exclude_none=True) for m in req.messages]
        ),
        sampling_params=sp,
        chat_template_kwargs=ctk,
        template_tools=template_tools,
        parser_tools=parser_tools,
        include_logprobs=bool(req.logprobs),
        disable_parallel_tool_use=disable_parallel,
        tool_choice_mode=tool_choice_mode,
        response_format=req.response_format,
    )


def _all_tool_dicts(tools) -> list[dict[str, Any]]:
    return [t.model_dump(exclude_none=True) for t in (tools or [])]


async def _build_model_card(state: Any) -> ModelCard:
    model_id = _served_model_name(state)
    ctx = _model_context_length(state)
    efforts, default_effort = await _effort_fields(state)
    return ModelCard(
        id=model_id,
        root=state.config.model_path,
        max_model_len=ctx,
        context_length=ctx,
        supported_reasoning_efforts=efforts,
        default_reasoning_effort=default_effort,
    )


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
    return create_error_response(msg, status_code=503, err_type="server_error")


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
        card = await _build_model_card(get_state())
        return ModelList(data=[card])

    @app.get("/v1/models/{model_id:path}")
    async def v1_model_detail(model_id: str):
        state = get_state()
        served = _served_model_name(state)
        if model_id not in (served, state.config.model_path):
            return create_error_response(
                f"The model {model_id!r} does not exist",
                status_code=404,
                err_type="not_found_error",
                param="model",
                code="model_not_found",
            )
        return await _build_model_card(state)


# How often a non-streaming handler looks at the transport while the engine generates: one
# second bounds an abandoned request's extra decode work without measurable polling overhead.
_DISCONNECT_POLL_SECONDS = 1.0


async def _await_watching_disconnect(
    awaitable, request: Request | None, state: Any, uid: int | Sequence[int]
):
    """Run an in-flight generation awaitable while watching the transport - the
    non-streaming twin of stream_with_cancellation. Returns the awaitable's result,
    or None when the client disconnected first: the same shielded abort_user is
    delivered for every uid (a completion's several prompts) so an abandoned request
    stops burning decode slots instead of running to max_tokens (with
    --max-running-requests 1 that is a full outage for its remaining budget)."""
    gen = asyncio.ensure_future(awaitable)
    if request is None:
        return await gen
    uids = [uid] if isinstance(uid, int) else list(uid)

    async def abort_all():
        results = await asyncio.gather(
            *(asyncio.shield(state.abort_user(u)) for u in uids),
            return_exceptions=True,
        )
        for u, res in zip(uids, results):
            if isinstance(res, Exception):
                logger.error("Failed to deliver abort for user %s: %r", u, res)

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
        return create_error_response(
            "function_call is not supported; use tools/tool_choice instead",
            param="function_call",
        )
    for name in _UNIMPLEMENTED_CHAT_FIELDS:
        if _is_set(getattr(req, name, None)):
            return create_error_response(
                f"{name} is not supported by this server", param=name
            )
    if (rf_err := _response_format_error(req.response_format)) is not None:
        rf_code = (
            "invalid_json_schema"
            if isinstance(req.response_format, dict)
            and req.response_format.get("type") == "json_schema"
            else None
        )
        return create_error_response(rf_err, param="response_format", code=rf_code)
    if not 1 <= req.n <= MAX_N_CHOICES:
        return create_error_response(
            f"n must be between 1 and {MAX_N_CHOICES}, got {req.n}", param="n"
        )
    if req.top_logprobs is not None:
        if not req.logprobs:
            return create_error_response(
                "top_logprobs requires logprobs=true", param="top_logprobs"
            )
        if not 0 <= req.top_logprobs <= MAX_CHAT_LOGPROBS:
            return create_error_response(
                f"top_logprobs must be between 0 and {MAX_CHAT_LOGPROBS}",
                param="top_logprobs",
            )
    if req.logprob_token_ids is not None and not req.logprobs:
        return create_error_response(
            "logprob_token_ids requires logprobs=true", param="logprob_token_ids"
        )
    # Case/whitespace and the "off" disable synonym stay accepted here because
    # effort_toggle_kwargs normalizes and honors them downstream.
    effort = (
        req.reasoning_effort.strip().lower()
        if isinstance(req.reasoning_effort, str)
        else None
    )
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
            getattr(state.config, "max_output_tokens", None)
            or DEFAULT_MAX_OUTPUT_TOKENS
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
            reasoning_parser=getattr(state.config, "reasoning_parser", None),
            tool_call_parser=getattr(state.config, "tool_call_parser", None),
        )
    except ValueError as exc:
        return create_error_response(
            str(exc),
            param=getattr(exc, "param", None),
            code=getattr(exc, "code", None),
        )

    chat_id_fields = [
        (name, ids)
        for name, ids in spec.sampling_params.client_token_ids()
        if name in ("allowed_token_ids", "logprob_token_ids")
    ]
    if (bad := await _token_id_error(state, chat_id_fields)) is not None:
        return bad

    if req.stream:
        # Non-stream requests already surface render failures as a clean 400
        # through GenerationError; only the stream path needs the pre-check.
        err = await prerender_error(spec, state)
        if err is not None:
            return create_error_response(str(err), code=err.code)

    specs = _fan_out_specs(spec, req.n)
    uids: list[int] = []
    for sample_spec in specs:
        try:
            uids.append(await submit_generation(sample_spec, state))
        except GenerationError as exc:
            if uids and hasattr(state, "abort_user"):
                await asyncio.gather(
                    *(state.abort_user(u) for u in uids), return_exceptions=True
                )
            return create_error_response(str(exc), code=exc.code)

    if req.stream:
        if len(uids) == 1:
            chunks = stream_chat_completion_chunks(uids[0], req, state, specs[0])
            if request is not None:
                chunks = state.stream_with_cancellation(chunks, request, uids[0])
        else:
            chunks = _stream_multi_chat_chunks(uids, req, state, specs, request)
        return StreamingResponse(chunks, media_type="text/event-stream")

    try:
        if len(uids) == 1:
            single = await _await_watching_disconnect(
                generate_full(uids[0], specs[0], state, source="/v1/chat/completions"),
                request,
                state,
                uids[0],
            )
            results = None if single is None else [single]
        else:
            results = await _await_watching_disconnect(
                asyncio.gather(
                    *(
                        generate_full(u, s, state, source="/v1/chat/completions")
                        for u, s in zip(uids, specs, strict=True)
                    )
                ),
                request,
                state,
                uids,
            )
    except GenerationError as exc:
        if len(uids) > 1 and hasattr(state, "abort_user"):
            await asyncio.gather(
                *(state.abort_user(u) for u in uids), return_exceptions=True
            )
        return create_error_response(str(exc), code=exc.code)
    if results is None:
        return _client_disconnected_response()

    choices: list[dict[str, Any]] = []
    for idx, res in enumerate(results):
        message: dict[str, Any] = {"role": "assistant", "content": res.content}
        if res.reasoning:
            message["reasoning_content"] = res.reasoning
        if res.tool_calls:
            message["tool_calls"] = _tool_calls_to_openai(res.tool_calls)
        choice: dict[str, Any] = {
            "index": idx,
            "message": message,
            "finish_reason": res.finish_reason,
        }
        if req.logprobs:
            choice["logprobs"] = {
                "content": chat_logprobs_content(
                    res.logprobs or [], req.return_tokens_as_token_ids
                ),
                "refusal": None,
            }
        choices.append(choice)

    first = results[0]
    response: dict[str, Any] = {
        "id": _response_id("chatcmpl", req, public_request_id(uids[0])),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "system_fingerprint": None,
        "choices": choices,
        "usage": _usage(
            first.prompt_tokens,
            sum(r.completion_tokens for r in results),
            _reported_cached(state, first.cached_tokens),
            reasoning_tokens=sum(r.reasoning_tokens for r in results),
        ),
    }
    if (timings := timings_block(state, first.timings)) is not None:
        response["timings"] = timings
    return response


async def stream_chat_completion_chunks(
    uid: int,
    req: ChatCompletionRequest,
    state: Any,
    spec: GenSpec | None = None,
    *,
    choice_index: int = 0,
    shared_id_uid: int | None = None,
    emit_done_sentinel: bool = True,
    emit_usage_chunk: bool | None = None,
    usage_out: GenUsage | None = None,
) -> AsyncIterator[bytes]:
    """Format generate_events() into the OpenAI chat.completion.chunk SSE stream."""
    if spec is None:
        spec = chat_request_to_genspec(req, {})
    id_uid = uid if shared_id_uid is None else shared_id_uid
    role_choice: dict[str, Any] = {
        "delta": {"role": "assistant", "content": ""},
        "index": choice_index,
        "finish_reason": None,
    }
    if req.logprobs:
        role_choice["logprobs"] = None
    yield _sse(_chat_chunk(req, id_uid, [role_choice]))

    timings: dict[str, Any] | None = None
    # timings ride the stream's LAST chunk only: the usage chunk when the client asked
    # for one, otherwise the finish chunk
    include_usage = (
        bool(req.stream_options and req.stream_options.include_usage)
        if emit_usage_chunk is None
        else emit_usage_chunk
    )
    continuous_usage = include_usage and bool(
        req.stream_options and req.stream_options.continuous_usage_stats
    )
    # the final usage chunk reads it too, so a stream cut by an error reports what it produced
    usage = usage_out if usage_out is not None else GenUsage()

    def _lp_obj(entries: list[dict[str, Any]] | None) -> dict[str, Any] | None:
        if not req.logprobs:
            return None
        if not entries:
            return None
        return {
            "content": chat_logprobs_content(entries, req.return_tokens_as_token_ids),
            "refusal": None,
        }

    def emit(chunk: dict[str, Any]) -> bytes:
        if continuous_usage:
            chunk["usage"] = _reported_usage(state, usage)
        # per-token: post-ack chunks carry the ack's snapshot. Request-side gate: the
        # terminal snapshot rides every request, so post-terminal residue must stay clean.
        if (
            req.timings_per_token
            and (t := timings_block(state, usage.timings)) is not None
        ):
            chunk["timings"] = t
        return _sse(chunk)

    tool_calls_sent = 0
    open_tool: dict[str, Any] | None = None
    events = generate_events(
        uid, spec, state, source="/v1/chat/completions", usage=usage
    )
    while True:
        try:
            ev = await events.__anext__()
        except StopAsyncIteration:
            break
        except GenerationError as exc:
            # Request failed before producing output — emit an error chunk + [DONE] so the
            # client gets a terminal signal instead of a stalled stream.
            yield _sse(
                {
                    "error": {
                        "message": str(exc),
                        "type": "invalid_request_error",
                        "param": None,
                        "code": exc.code,
                    }
                }
            )
            break
        if isinstance(ev, ReasoningDelta):
            c_dict: dict[str, Any] = {
                "delta": {"reasoning_content": ev.text},
                "index": choice_index,
                "finish_reason": None,
            }
            if req.logprobs:
                c_dict["logprobs"] = _lp_obj(ev.logprobs)
            yield emit(_chat_chunk(req, id_uid, [c_dict]))
        elif isinstance(ev, ContentDelta):
            c_dict = {
                "delta": {"content": ev.text},
                "index": choice_index,
                "finish_reason": None,
            }
            if req.logprobs:
                c_dict["logprobs"] = _lp_obj(ev.logprobs)
            yield emit(_chat_chunk(req, id_uid, [c_dict]))
        elif isinstance(ev, ToolCallStart):
            open_tool = {
                "index": tool_calls_sent,
                "ordinal": ev.tool_index,
                "sent": "",
                "stable": ev.args_prefix_stable,
            }
            yield emit(
                _chat_chunk(
                    req,
                    id_uid,
                    [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": open_tool["index"],
                                        "id": _tool_call_id(
                                            ev.name, open_tool["index"]
                                        ),
                                        "type": "function",
                                        "function": {"name": ev.name, "arguments": ""},
                                    }
                                ]
                            },
                            "index": choice_index,
                            "finish_reason": None,
                        }
                    ],
                )
            )
        elif isinstance(ev, ToolCallArgsDelta):
            # Clients concatenate argument fragments, so only prefix-stable
            # fragments stream; otherwise the full arguments arrive at close.
            if open_tool is not None and open_tool["stable"] and ev.fragment:
                open_tool["sent"] += ev.fragment
                yield emit(
                    _chat_chunk(
                        req,
                        id_uid,
                        [
                            {
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": open_tool["index"],
                                            "function": {"arguments": ev.fragment},
                                        }
                                    ]
                                },
                                "index": choice_index,
                                "finish_reason": None,
                            }
                        ],
                    )
                )
        elif isinstance(ev, ToolCallsDelta):
            for call in ev.calls:
                if open_tool is not None and open_tool["ordinal"] == call.tool_index:
                    # Close of a ToolCallStart-opened call: send whatever of the
                    # final (authoritative) arguments wasn't streamed yet.
                    final = call.parameters or ""
                    remainder = (
                        final[len(open_tool["sent"]) :]
                        if final.startswith(open_tool["sent"])
                        else final
                        if not open_tool["sent"]
                        else ""
                    )
                    if remainder:
                        yield emit(
                            _chat_chunk(
                                req,
                                id_uid,
                                [
                                    {
                                        "delta": {
                                            "tool_calls": [
                                                {
                                                    "index": open_tool["index"],
                                                    "function": {
                                                        "arguments": remainder
                                                    },
                                                }
                                            ]
                                        },
                                        "index": choice_index,
                                        "finish_reason": None,
                                    }
                                ],
                            )
                        )
                    open_tool = None
                    tool_calls_sent += 1
                    continue
                # Standalone complete call (buffered fallback path).
                for delta in _tool_call_deltas([call], start_index=tool_calls_sent):
                    yield emit(
                        _chat_chunk(
                            req,
                            id_uid,
                            [
                                {
                                    "delta": {"tool_calls": [delta]},
                                    "index": choice_index,
                                    "finish_reason": None,
                                }
                            ],
                        )
                    )
                tool_calls_sent += 1
        elif isinstance(ev, GenDone):
            timings = ev.timings
            done_choice: dict[str, Any] = {
                "delta": {},
                "index": choice_index,
                "finish_reason": ev.finish_reason,
            }
            if req.logprobs:
                done_choice["logprobs"] = _lp_obj(ev.logprobs)
            done = _chat_chunk(req, id_uid, [done_choice])
            if (
                emit_done_sentinel
                and not include_usage
                and (t := timings_block(state, timings)) is not None
            ):
                done["timings"] = t
            yield emit(done)

    if include_usage and emit_done_sentinel:
        final: dict[str, Any] = {
            **_chat_chunk(req, id_uid, []),
            "usage": _reported_usage(state, usage),
        }
        if (t := timings_block(state, timings)) is not None:
            final["timings"] = t
        yield _sse(final)

    if emit_done_sentinel:
        yield b"data: [DONE]\n\n"


async def _stream_multi_chat_chunks(
    uids: list[int],
    req: ChatCompletionRequest,
    state: Any,
    specs: list[GenSpec],
    request: Request | None,
) -> AsyncIterator[bytes]:
    """Merge ``n > 1`` chat streams concurrently onto one SSE stream."""
    usages = [GenUsage() for _ in uids]
    queue: asyncio.Queue[bytes | BaseException | None] = asyncio.Queue()

    async def _pump(idx: int) -> None:
        try:
            async for chunk in stream_chat_completion_chunks(
                uids[idx],
                req,
                state,
                specs[idx],
                choice_index=idx,
                shared_id_uid=uids[0],
                emit_done_sentinel=False,
                emit_usage_chunk=False,
                usage_out=usages[idx],
            ):
                await queue.put(chunk)
        except BaseException as exc:  # noqa: BLE001
            await queue.put(exc)
        finally:
            await queue.put(None)

    tasks = [asyncio.create_task(_pump(i)) for i in range(len(uids))]
    finished = 0
    try:
        while finished < len(uids):
            if request is not None and await request.is_disconnected():
                break
            item = await queue.get()
            if item is None:
                finished += 1
            elif isinstance(item, BaseException):
                raise item
            else:
                yield item
        if finished == len(uids):
            if req.stream_options and req.stream_options.include_usage:
                merged = GenUsage(
                    prompt_tokens=usages[0].prompt_tokens,
                    completion_tokens=sum(u.completion_tokens for u in usages),
                    cached_tokens=usages[0].cached_tokens,
                    reasoning_tokens=sum(u.reasoning_tokens for u in usages),
                    timings=usages[0].timings,
                )
                final: dict[str, Any] = {
                    **_chat_chunk(req, uids[0], []),
                    "usage": _reported_usage(state, merged),
                }
                if (t := timings_block(state, usages[0].timings)) is not None:
                    final["timings"] = t
                yield _sse(final)
            yield b"data: [DONE]\n\n"
    finally:
        for t in tasks:
            if not t.done():
                t.cancel()
        if finished < len(uids) and hasattr(state, "abort_user"):
            await asyncio.gather(
                *(asyncio.shield(state.abort_user(u)) for u in uids),
                return_exceptions=True,
            )


async def handle_completion(
    req: CompletionRequest,
    request: Request | None,
    state: Any,
    model_sampling: dict[str, Any],
):
    unsupported = _completion_unsupported_reason(req)
    if unsupported is not None:
        reason, param = unsupported
        code = (
            "invalid_json_schema"
            if param == "response_format"
            and isinstance(req.response_format, dict)
            and req.response_format.get("type") == "json_schema"
            else None
        )
        return create_error_response(reason, param=param, code=code)
    if not 1 <= req.n <= MAX_N_CHOICES:
        return create_error_response(
            f"n must be between 1 and {MAX_N_CHOICES}, got {req.n}", param="n"
        )
    try:  # surfaces an out-of-range value as a 400 rather than a 500 from the worker
        from freetoken.grammar.schema import build_structured_output_spec

        default_max_tokens = (
            getattr(state.config, "max_output_tokens", None)
            or DEFAULT_MAX_OUTPUT_TOKENS
        )
        sampling = _resolve_sampling(
            req, model_sampling, default_max_tokens=default_max_tokens
        )
        completion_so = build_structured_output_spec(
            response_format=req.response_format,
            guided_json=req.guided_json,
            guided_regex=req.guided_regex,
            guided_choice=req.guided_choice,
            guided_grammar=req.guided_grammar,
            structured_outputs=req.structured_outputs,
        )
        sampling.structured_output = completion_so
    except ValueError as exc:
        return create_error_response(
            str(exc),
            param=getattr(exc, "param", None),
            code=getattr(exc, "code", None),
        )
    token_prompts = _is_token_prompt(req.prompt)
    prompts = _prompt_list(req.prompt)
    max_prompts = int(
        getattr(state.config, "max_completion_prompts", None) or MAX_PROMPTS
    )
    if len(prompts) > max_prompts:
        return create_error_response(
            f"prompt must hold at most {max_prompts} prompts, got {len(prompts)}",
            param="prompt",
        )
    id_fields = sampling.client_token_ids()
    if token_prompts:
        id_fields += [("prompt", ids) for ids in prompts]
    if (bad := await _token_id_error(state, id_fields)) is not None:
        return bad

    # stamp per-token only when the server opted into timings too; llama.cpp publishes
    # partial timings on streams, so non-stream requests never pay for the snapshots
    timings_pt = bool(
        req.stream
        and req.timings_per_token
        and getattr(state.config, "enable_timings_report", False)
    )

    def _tokenize_msg(
        uid: int, prompt: str | list[int], sample_idx: int = 0
    ) -> TokenizeMsg:
        sampling_params = _resolve_sampling(
            req,
            model_sampling,
            default_max_tokens=default_max_tokens,
            timings_per_token=timings_pt,
        )
        sampling_params.structured_output = completion_so
        if sample_idx > 0 and sampling_params.seed is not None:
            sampling_params.seed = sampling_params.seed + sample_idx
        if token_prompts:
            return TokenizeMsg(
                uid=uid,
                text="",
                input_ids=prompt,
                sampling_params=sampling_params,
                response_format=req.response_format,
            )
        return TokenizeMsg(
            uid=uid,
            text=prompt,
            sampling_params=sampling_params,
            add_special_tokens=req.add_special_tokens,
            response_format=req.response_format,
        )

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

    jobs: list[tuple[int, int, str | list[int], str]] = [
        (p_idx * req.n + s_idx, s_idx, prompt, echo_texts[p_idx])
        for p_idx, prompt in enumerate(prompts)
        for s_idx in range(req.n)
    ]

    if req.stream:
        if len(jobs) == 1:
            uid = state.new_user()
            try:
                await send_new_request(state, uid, _tokenize_msg(uid, prompts[0], 0))
            except GenerationError as exc:
                return create_error_response(str(exc), code=exc.code)
            chunks = stream_completion_chunks(uid, req, state, echo_text=echo_texts[0])
            if request is not None:
                chunks = state.stream_with_cancellation(chunks, request, uid)
            return StreamingResponse(chunks, media_type="text/event-stream")

        stream_uids: list[int] = []
        for _, s_idx, prompt, _ in jobs:
            uid = state.new_user()
            try:
                await send_new_request(state, uid, _tokenize_msg(uid, prompt, s_idx))
            except GenerationError as exc:
                if stream_uids and hasattr(state, "abort_user"):
                    await asyncio.gather(
                        *(state.abort_user(u) for u in stream_uids),
                        return_exceptions=True,
                    )
                return create_error_response(str(exc), code=exc.code)
            stream_uids.append(uid)
        chunks = _stream_multi_completion_chunks(stream_uids, jobs, req, state, request)
        return StreamingResponse(chunks, media_type="text/event-stream")

    choices: list[dict[str, Any]] = []
    prompt_tokens = 0
    completion_tokens = 0
    cached_tokens = 0
    # with several prompts the timings are the FIRST prompt's own spans, as llama.cpp
    # returns arr[0]; one clock across the prompts would mix the later prompts' queue and
    # prefill into the rates
    first_timings: dict[str, Any] | None = None
    # every prompt is queued before any is drained, so the scheduler can prefill them together
    uids: list[int] = []
    for _, s_idx, prompt, _ in jobs:
        uid = state.new_user()
        try:
            await send_new_request(state, uid, _tokenize_msg(uid, prompt, s_idx))
        except GenerationError as exc:
            # nothing would drain the prompts already queued
            if uids:
                await asyncio.gather(
                    *(state.abort_user(sent) for sent in uids),
                    return_exceptions=True,
                )
            return create_error_response(str(exc), code=exc.code)
        uids.append(uid)

    drain_tasks: list[asyncio.Task] = []

    async def _abort_siblings(failed_uid: int) -> None:
        siblings = [u for u in uids if u != failed_uid]
        if siblings:
            await asyncio.gather(
                *(asyncio.shield(state.abort_user(u)) for u in siblings),
                return_exceptions=True,
            )
        for t in drain_tasks:
            if not t.done() and t is not asyncio.current_task():
                t.cancel()

    async def _drain_one(
        uid: int, index: int, sample_idx: int, echo_prefix: str
    ) -> dict[str, Any] | JSONResponse:
        nonlocal prompt_tokens, completion_tokens, cached_tokens, first_timings
        text = ""
        entries: list[dict] = []
        offsets: list[int] = []
        finish_reason = "stop"
        async for ack in state.wait_for_ack(uid):
            if getattr(ack, "error", None):
                await _abort_siblings(uid)
                return create_error_response(
                    ack.error, code=getattr(ack, "error_code", None)
                )
            if sample_idx == 0:
                prompt_tokens += ack.prompt_tokens_delta
                cached_tokens += ack.cached_tokens
            completion_tokens += ack.completion_tokens_delta
            if req.logprobs is not None:
                for lp_entry, lp_off in reply_logprob_pairs(ack):
                    entries.append(lp_entry)
                    offsets.append(lp_off)
            text += ack.incremental_output
            if ack.finished:
                if index == 0:
                    first_timings = getattr(ack, "timings", None)
                finish_reason = getattr(ack, "finish_reason", None) or "stop"
                break
        return {
            "index": index,
            "text": echo_prefix + text,
            "finish_reason": finish_reason,
            "logprobs": (
                completions_logprobs(
                    entries, offsets, len(text), req.return_tokens_as_token_ids
                )
                if req.logprobs is not None
                else None
            ),
        }

    async def _gather_drains():
        drain_tasks.extend(
            asyncio.create_task(_drain_one(u, choice_idx, s_idx, echo_prefix))
            for u, (choice_idx, s_idx, _, echo_prefix) in zip(uids, jobs, strict=True)
        )
        results = await asyncio.gather(*drain_tasks, return_exceptions=True)
        for res in results:
            if isinstance(res, JSONResponse):
                return [res]
        for res in results:
            if isinstance(res, BaseException):
                raise res
        return results

    drained = await _await_watching_disconnect(_gather_drains(), request, state, uids)
    if drained is None:
        return _client_disconnected_response()
    for choice in drained:
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
        "usage": _usage(
            prompt_tokens, completion_tokens, _reported_cached(state, cached_tokens)
        ),
    }
    if (timings := timings_block(state, first_timings)) is not None:
        response["timings"] = timings
    return response


def _completion_chunk(
    req: CompletionRequest, uid: Any, choices: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "id": _response_id("cmpl", req, public_request_id(uid)),
        "object": "text_completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": choices,
    }


async def stream_completion_chunks(
    uid: int,
    req: CompletionRequest,
    state: Any,
    echo_text: str = "",
    *,
    choice_index: int = 0,
    shared_id_uid: int | None = None,
    emit_done_sentinel: bool = True,
    emit_usage_chunk: bool | None = None,
    usage_out: GenUsage | None = None,
) -> AsyncIterator[bytes]:
    id_uid = uid if shared_id_uid is None else shared_id_uid
    usage = usage_out if usage_out is not None else GenUsage()
    timings: dict[str, Any] | None = None
    include_usage = (
        bool(req.stream_options and req.stream_options.include_usage)
        if emit_usage_chunk is None
        else emit_usage_chunk
    )
    continuous_usage = include_usage and bool(
        req.stream_options and req.stream_options.continuous_usage_stats
    )

    def emit(chunk: dict[str, Any]) -> bytes:
        if continuous_usage:
            chunk["usage"] = _reported_usage(state, usage)
        # per-token: same request-side check as the chat stream - the terminal ack carries
        # timings for every request, only the flag decides who gets a block on each chunk
        if (
            req.timings_per_token
            and (t := timings_block(state, usage.timings)) is not None
        ):
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
        yield _sse(
            _completion_chunk(
                req,
                id_uid,
                [
                    {
                        "text": echo_text,
                        "index": choice_index,
                        "finish_reason": None,
                        "logprobs": None,
                    }
                ],
            )
        )
    async for ack in state.wait_for_ack(uid):
        if getattr(ack, "error", None):
            yield _sse(
                {
                    "error": {
                        "message": ack.error,
                        "type": "invalid_request_error",
                        "param": None,
                        "code": getattr(ack, "error_code", None),
                    }
                }
            )
            if emit_done_sentinel:
                yield b"data: [DONE]\n\n"
            return
        usage.prompt_tokens += ack.prompt_tokens_delta
        usage.completion_tokens += ack.completion_tokens_delta
        usage.cached_tokens += ack.cached_tokens
        if getattr(ack, "timings", None) is not None:
            timings = ack.timings
            usage.timings = timings
        if req.logprobs is not None:
            for lp_entry, lp_off in reply_logprob_pairs(ack):
                pending.append(lp_entry)
                pending_offsets.append(lp_off)
        if ack.incremental_output:
            sent_len += len(ack.incremental_output)
            yield emit(
                _completion_chunk(
                    req,
                    id_uid,
                    [
                        {
                            "text": ack.incremental_output,
                            "index": choice_index,
                            "finish_reason": None,
                            # Mid-stream, a stamp past sent_len is held-back text still to come.
                            "logprobs": (
                                completions_logprobs(
                                    pending,
                                    pending_offsets,
                                    sent_len if ack.finished else None,
                                    req.return_tokens_as_token_ids,
                                )
                                if pending
                                else None
                            ),
                        }
                    ],
                )
            )
            pending, pending_offsets = [], []
        if ack.finished:
            finish_reason = getattr(ack, "finish_reason", None) or "stop"
            break

    # Entries whose text never showed (EOS, a stop string, a trailing partial character)
    # ride the finish chunk, so the joined stream matches the non-stream list.
    done_chunk = _completion_chunk(
        req,
        id_uid,
        [
            {
                "text": "",
                "index": choice_index,
                "finish_reason": finish_reason,
                "logprobs": (
                    completions_logprobs(
                        pending,
                        pending_offsets,
                        sent_len,
                        req.return_tokens_as_token_ids,
                    )
                    if pending
                    else None
                ),
            }
        ],
    )
    if (
        emit_done_sentinel
        and not include_usage
        and (t := timings_block(state, timings)) is not None
    ):
        # as on the chat stream: the finish chunk is the last one only when no usage
        # chunk follows
        done_chunk["timings"] = t
    yield emit(done_chunk)
    if include_usage and emit_done_sentinel:
        final: dict[str, Any] = {
            **_completion_chunk(req, id_uid, []),
            "usage": _reported_usage(state, usage),
        }
        if (t := timings_block(state, timings)) is not None:
            final["timings"] = t
        yield _sse(final)
    if emit_done_sentinel:
        yield b"data: [DONE]\n\n"


async def _stream_multi_completion_chunks(
    uids: list[int],
    jobs: list[tuple[int, int, str | list[int], str]],
    req: CompletionRequest,
    state: Any,
    request: Request | None,
) -> AsyncIterator[bytes]:
    """Merge multi-prompt or ``n > 1`` completion streams onto one SSE stream."""
    usages = [GenUsage() for _ in uids]
    queue: asyncio.Queue[bytes | BaseException | None] = asyncio.Queue()

    async def _pump(idx: int) -> None:
        choice_idx, _, _, echo_prefix = jobs[idx]
        try:
            async for chunk in stream_completion_chunks(
                uids[idx],
                req,
                state,
                echo_text=echo_prefix,
                choice_index=choice_idx,
                shared_id_uid=uids[0],
                emit_done_sentinel=False,
                emit_usage_chunk=False,
                usage_out=usages[idx],
            ):
                await queue.put(chunk)
        except BaseException as exc:  # noqa: BLE001
            await queue.put(exc)
        finally:
            await queue.put(None)

    tasks = [asyncio.create_task(_pump(i)) for i in range(len(uids))]
    finished = 0
    try:
        while finished < len(uids):
            if request is not None and await request.is_disconnected():
                break
            item = await queue.get()
            if item is None:
                finished += 1
            elif isinstance(item, BaseException):
                raise item
            else:
                yield item
        if finished == len(uids):
            if req.stream_options and req.stream_options.include_usage:
                prompt_tot = sum(
                    u.prompt_tokens
                    for u, (_, s_idx, _, _) in zip(usages, jobs, strict=True)
                    if s_idx == 0
                )
                cached_tot = sum(
                    u.cached_tokens
                    for u, (_, s_idx, _, _) in zip(usages, jobs, strict=True)
                    if s_idx == 0
                )
                merged = GenUsage(
                    prompt_tokens=prompt_tot,
                    completion_tokens=sum(u.completion_tokens for u in usages),
                    cached_tokens=cached_tot,
                    timings=usages[0].timings,
                )
                final: dict[str, Any] = {
                    **_completion_chunk(req, uids[0], []),
                    "usage": _reported_usage(state, merged),
                }
                if (t := timings_block(state, usages[0].timings)) is not None:
                    final["timings"] = t
                yield _sse(final)
            yield b"data: [DONE]\n\n"
    finally:
        for t in tasks:
            if not t.done():
                t.cancel()
        if finished < len(uids) and hasattr(state, "abort_user"):
            await asyncio.gather(
                *(asyncio.shield(state.abort_user(u)) for u in uids),
                return_exceptions=True,
            )


_FIM_TOKENS = ("<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>")


async def _token_id_error(
    state: Any, fields: list[tuple[str, list[int]]]
) -> JSONResponse | None:
    """A 400 for the first client token id outside the vocabulary, before anything is
    queued. Without a frontend tokenizer the bound is the engine's int32; the tokenizer
    worker checks the completions-only fields against its own tokenizer either way."""
    if not fields:
        return None
    bound = await _vocab_len(state) or MAX_TOKEN_ID + 1
    for name, ids in fields:
        if not all(0 <= t < bound for t in ids):
            what = "prompt token ids" if name == "prompt" else name
            return create_error_response(f"{what} must be in [0, {bound})", param=name)
    return None


async def _vocab_len(state: Any) -> int | None:
    """Upper bound on valid client token IDs (`min(len(tokenizer), model_config.vocab_size)`), or None."""
    bounds: list[int] = []
    try:
        model_vocab = getattr(
            getattr(state.config, "model_config", None), "vocab_size", None
        )
        if isinstance(model_vocab, int) and model_vocab > 0:
            bounds.append(model_vocab)
    except Exception:  # noqa: BLE001
        pass
    build = getattr(state, "frontend_tokenizer", None)
    if build is not None:
        try:
            manager = await asyncio.to_thread(build)
            bounds.append(len(manager.tokenizer))
        except Exception:  # noqa: BLE001 -- the tokenizer worker still range-checks
            pass
    return min(bounds) if bounds else None


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
    if req.continue_final_message and not req.messages:
        return create_error_response(
            "continue_final_message requires messages", param="continue_final_message"
        )
    if (
        req.continue_final_message
        and req.messages
        and req.messages[-1].role != "assistant"
    ):
        return create_error_response(
            "continue_final_message needs a final assistant message",
            param="continue_final_message",
        )
    build = getattr(state, "frontend_tokenizer", None)
    if build is None:
        return create_error_response(
            "no tokenizer on this server", status_code=503, err_type="api_error"
        )
    try:
        manager = await asyncio.to_thread(build)
        if req.prompt is not None:
            ids = await asyncio.to_thread(
                manager.tokenizer.encode,
                req.prompt,
                add_special_tokens=req.add_special_tokens,
            )
        else:
            messages = render_messages(
                [m.model_dump(exclude_none=True) for m in req.messages]
            )
            ctk = dict(req.chat_template_kwargs)
            if req.continue_final_message:
                ctk["continue_final_message"] = True
            tool_dicts = (
                [t.model_dump(exclude_none=True) for t in req.tools]
                if req.tools
                else None
            )
            if not req.add_generation_prompt and not req.continue_final_message:
                # Render directly with add_generation_prompt=False when the caller asked
                # not to append a generation prompt without continuing an assistant message.
                apply_fn = getattr(manager.tokenizer, "apply_chat_template", None)
                if (
                    callable(apply_fn)
                    and getattr(manager, "_dsv4_encoder", None) is None
                ):
                    tpl_kwargs = dict(ctk)
                    if tool_dicts is not None:
                        tpl_kwargs["tools"] = tool_dicts
                    text = await asyncio.to_thread(
                        apply_fn,
                        messages,
                        tokenize=False,
                        add_generation_prompt=False,
                        **tpl_kwargs,
                    )
                else:
                    msg = TokenizeMsg(
                        uid=-1,
                        text=messages,
                        sampling_params=SamplingParams(),
                        chat_template_kwargs={**ctk, "continue_final_message": True},
                        tools=tool_dicts,
                    )
                    text = await asyncio.to_thread(manager.render_prompt, msg)
            else:
                msg = TokenizeMsg(
                    uid=-1,
                    text=messages,
                    sampling_params=SamplingParams(),
                    chat_template_kwargs=ctk,
                    tools=tool_dicts,
                )
                text = await asyncio.to_thread(manager.render_prompt, msg)
            ids = await asyncio.to_thread(
                manager.tokenizer.encode, text, add_special_tokens=False
            )
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
        return create_error_response(
            "no tokenizer on this server", status_code=503, err_type="api_error"
        )
    if (bad := await _token_id_error(state, [("tokens", req.tokens)])) is not None:
        return bad
    manager = await asyncio.to_thread(build)
    text = await asyncio.to_thread(
        manager.tokenizer.decode,
        req.tokens,
        skip_special_tokens=req.skip_special_tokens,
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


def _fan_out_specs(spec: GenSpec, n: int) -> list[GenSpec]:
    if n <= 1:
        return [spec]
    out: list[GenSpec] = []
    for idx in range(n):
        sp = replace(spec.sampling_params)
        if idx > 0 and sp.seed is not None:
            sp.seed = sp.seed + idx
        out.append(replace(spec, sampling_params=sp))
    return out


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
        allowed_token_ids=req.allowed_token_ids,
        logprob_token_ids=req.logprob_token_ids,
        seed=req.seed,
    )
    if isinstance(req, CompletionRequest) and req.logprobs is not None:
        sampling_params.logprobs = True
        sampling_params.top_logprobs = req.logprobs
        if sampling_params.allowed_token_ids:
            _check_restricted_logprobs(sampling_params)
    elif isinstance(req, ChatCompletionRequest) and req.logprobs:
        sampling_params.logprobs = True
        sampling_params.top_logprobs = (
            req.top_logprobs if req.top_logprobs is not None else 0
        )
        if sampling_params.allowed_token_ids:
            _check_restricted_logprobs(sampling_params)
    return sampling_params


def _check_restricted_logprobs(sp: SamplingParams) -> None:
    """Logprobs restricted to allowed_token_ids are vLLM's processed_logprobs at temperature 0:
    log_softmax over the allowed ids of the unprocessed logits. Anything vLLM would also fold
    into them is refused rather than silently left out."""
    if sp.temperature > 0:
        raise ValueError("allowed_token_ids with logprobs needs temperature=0")
    if sp.min_p > 0:
        raise ValueError("allowed_token_ids with logprobs does not support min_p")
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


def _resolve_chat_tools(
    req: ChatCompletionRequest,
) -> tuple[list[dict[str, Any]] | None, list[dict[str, Any]] | None, str | None]:
    """Validate ``tool_choice`` against ``tools`` and return ``(template_tools, parser_tools, tool_choice_mode)``."""
    tc = req.tool_choice
    all_tools = _all_tool_dicts(req.tools)
    tool_names = {(t.get("function") or {}).get("name") for t in all_tools}
    if tc is None or tc == "auto":
        if not all_tools:
            return None, None, None
        return all_tools, all_tools, None
    if tc == "none":
        return None, None, None
    if tc == "required":
        if not all_tools:
            raise ValueError("tool_choice='required' requires non-empty tools")
        return all_tools, all_tools, "required"
    if isinstance(tc, ToolChoiceObject):
        selected = tc.function.name
        if not all_tools or selected not in tool_names:
            raise ValueError(
                f"tool_choice function {selected!r} was not found in tools"
            )
        filtered = [
            t for t in all_tools if (t.get("function") or {}).get("name") == selected
        ]
        return filtered, all_tools, f"named:{selected}"
    if isinstance(tc, dict):
        tc_type = tc.get("type")
        if tc_type == "function":
            fn = tc.get("function")
            selected = fn.get("name") if isinstance(fn, dict) else tc.get("name")
            if not selected or not all_tools or selected not in tool_names:
                raise ValueError(
                    f"tool_choice function {selected!r} was not found in tools"
                )
            filtered = [
                t
                for t in all_tools
                if (t.get("function") or {}).get("name") == selected
            ]
            return filtered, all_tools, f"named:{selected}"
        if tc_type == "allowed_tools":
            spec_obj = tc.get("allowed_tools")
            inner = spec_obj if isinstance(spec_obj, dict) else tc
            mode = inner.get("mode", "auto")
            if mode not in ("auto", "required", "none"):
                raise ValueError(
                    f"allowed_tools.mode must be 'auto', 'required', or 'none', got {mode!r}"
                )
            if mode == "none":
                return None, None, None
            refs = inner.get("tools") or []
            allowed_names: set[str] = set()
            for r in refs:
                if isinstance(r, dict):
                    rname = (r.get("function") or {}).get("name") or r.get("name")
                    if rname:
                        allowed_names.add(rname)
            if not allowed_names:
                raise ValueError("allowed_tools requires at least one tool")
            missing = sorted(allowed_names - tool_names)
            if missing:
                raise ValueError(
                    f"allowed_tools references unknown tool(s): {', '.join(missing)}"
                )
            subset = [
                t
                for t in all_tools
                if (t.get("function") or {}).get("name") in allowed_names
            ]
            if mode == "required" and len(subset) == 1:
                only = (subset[0].get("function") or {}).get("name")
                return subset, all_tools, f"named:{only}"
            return subset, all_tools, ("required" if mode == "required" else None)
    raise ValueError(f"Unsupported tool_choice: {tc!r}")


def _tools_for_template(req: ChatCompletionRequest) -> list[dict[str, Any]] | None:
    return _resolve_chat_tools(req)[0]


def _should_parse_tools(req: ChatCompletionRequest) -> bool:
    return _resolve_chat_tools(req)[1] is not None


def _tool_calls_to_openai(calls: list[ToolCallItem]) -> list[dict[str, Any]]:
    result = []
    for index, call in enumerate(calls):
        result.append(
            {
                "id": _tool_call_id(call.name, index),
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": call.parameters,
                },
            }
        )
    return result


def _tool_call_deltas(
    calls: list[ToolCallItem], start_index: int = 0
) -> list[dict[str, Any]]:
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


def _chat_chunk(
    req: ChatCompletionRequest, uid: int, choices: list[dict[str, Any]]
) -> dict[str, Any]:
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
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: int = 0,
    reasoning_tokens: int = 0,
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


def _response_format_error(response_format: dict[str, Any] | None) -> str | None:
    if response_format is None:
        return None
    rtype = response_format.get("type")
    if rtype not in (None, "text", "json_object", "json_schema"):
        return (
            f"response_format.type {rtype!r} is not supported "
            "(supported: text, json_object, json_schema)"
        )
    if rtype == "json_schema":
        from freetoken.grammar.schema import SchemaValidationError, validate_json_schema

        js = response_format.get("json_schema")
        container = js if isinstance(js, dict) else response_format
        schema = container.get("schema")
        if not isinstance(schema, dict):
            return "response_format with type 'json_schema' requires a json_schema.schema object"
        try:
            validate_json_schema(schema, strict=bool(container.get("strict", False)))
        except SchemaValidationError as exc:
            return str(exc)
    return None


def _response_format_unsupported(response_format: dict[str, Any] | None) -> bool:
    return _response_format_error(response_format) is not None


# vLLM /v1/completions and /v1/chat/completions fields that change the result and are not
# implemented here: a set value is a 400 rather than silently dropped by extra="allow".
_UNIMPLEMENTED_COMPLETION_FIELDS = (
    "prompt_logprobs",
    "bad_words",
)
_UNIMPLEMENTED_CHAT_FIELDS = _UNIMPLEMENTED_COMPLETION_FIELDS


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


def _completion_unsupported_reason(
    req: CompletionRequest,
) -> tuple[str, str | None] | None:
    for name in _UNIMPLEMENTED_COMPLETION_FIELDS:
        if _is_set(getattr(req, name, None)):
            return f"{name} is not supported by this server", name
    if _empty_prompt(req):
        return "prompt must not be empty", "prompt"
    if _is_token_prompt(req.prompt):
        if req.echo:
            return "echo is not supported with token-id prompts", "echo"
        if req.suffix is not None:
            return "suffix is not supported with token-id prompts", "suffix"
    logprobs_error = completion_logprobs_error(req)
    if logprobs_error is not None:
        return logprobs_error, "logprobs"
    if (rf_err := _response_format_error(req.response_format)) is not None:
        return rf_err, "response_format"
    return None


def _prompt_list(prompt: Any) -> list:
    """One entry per prompt: a string or a token-id list."""
    if isinstance(prompt, str) or (prompt and all(isinstance(t, int) for t in prompt)):
        return [prompt]
    return list(prompt)


def _is_token_prompt(prompt: Any) -> bool:
    return (
        isinstance(prompt, list)
        and bool(prompt)
        and (
            all(isinstance(item, int) for item in prompt)
            or all(
                isinstance(item, list) and all(isinstance(token, int) for token in item)
                for item in prompt
            )
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
        spec_headroom = max(
            0, int(getattr(state.config, "decode_rows_per_req", 1) or 1) - 1
        )
        if kv_tokens > spec_headroom:
            kv_tokens -= spec_headroom
    except Exception:  # noqa: BLE001
        kv_tokens = 0
    # 0 before the ("meta", …) ack lands; the ceiling is the best available answer until then.
    return min(value, kv_tokens) if kv_tokens > 0 else value
