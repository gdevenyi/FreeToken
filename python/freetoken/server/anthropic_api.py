"""Anthropic Messages API (`/v1/messages`) — the endpoint Claude Code talks to.

Adapted from vLLM's ``AnthropicServingMessages``. FreeToken's serving layer is
function-based, so instead of subclassing the chat server this module drives the
shared protocol-neutral generation primitive
(:func:`freetoken.server.openai_api.submit_generation` + ``generate_events`` /
``generate_full``) and formats its semantic events into the Anthropic wire shape.
It consumes typed events, not a re-parsed OpenAI stream.
"""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import AsyncIterator, Callable
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exception_handlers import (
    http_exception_handler,
    request_validation_exception_handler,
)
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from .anthropic_models import (
    AnthropicContentBlock,
    AnthropicCountTokensRequest,
    AnthropicCountTokensResponse,
    AnthropicDelta,
    AnthropicError,
    AnthropicErrorResponse,
    AnthropicMessagesRequest,
    AnthropicMessagesResponse,
    AnthropicStreamEvent,
    AnthropicUsage,
)
from .generation import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    KEEPALIVE,
    ContentDelta,
    GenDone,
    GenerationError,
    GenResult,
    GenSpec,
    PromptAdmitted,
    ReasoningDelta,
    ToolCallArgsDelta,
    ToolCallsDelta,
    ToolCallStart,
    count_prompt_tokens,
    generate_events,
    generate_full,
    prerender_error,
    public_request_id,
    render_messages,
    resolve_sampling,
    submit_generation,
    with_keepalive,
)
from .openai_api import (
    _ACCEPTED_EFFORTS,
    _await_watching_disconnect,
)
from .request_logger import log_request

# Emit a protocol-native `ping` event after this many seconds of stream silence,
# bridging long queue/prefill/decode gaps for clients with stream-idle timeouts.
KEEPALIVE_INTERVAL_S = 10.0

# OpenAI finish_reason -> Anthropic stop_reason (vLLM's stop_reason_map).
STOP_REASON_MAP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
}


def _anthropic_stop(
    finish_reason: str | None, matched_stop: str | None
) -> tuple[str | None, str | None]:
    """(stop_reason, stop_sequence) with vLLM/Anthropic priority:
    max_tokens > tool_use > stop_sequence > end_turn."""
    if finish_reason == "length":
        return "max_tokens", None
    if finish_reason == "tool_calls":
        return "tool_use", None
    if matched_stop is not None:
        return "stop_sequence", matched_stop
    return STOP_REASON_MAP.get(finish_reason or "stop"), None


def _maintenance_detail(mstate: str) -> str:
    if mstate == "loading":
        return "model is still loading"
    if mstate == "stopping":
        return "server is shutting down"
    if mstate == "failed":
        return "engine failed"
    return "cache rebuild in progress"


def register_anthropic_routes(
    app: FastAPI,
    get_state: Callable[[], Any],
    get_model_sampling: Callable[[], dict[str, Any]],
) -> None:
    @app.post("/v1/messages")
    async def v1_messages(req: AnthropicMessagesRequest, request: Request):
        log_request("/v1/messages", req, request)
        state = get_state()
        mstate = getattr(state, "maintenance_state", "serving")
        if mstate != "serving":
            return _anthropic_error_response(
                503, "overloaded_error", _maintenance_detail(mstate)
            )
        return await handle_anthropic_messages(
            req, request, state, get_model_sampling()
        )

    @app.post("/v1/messages/count_tokens")
    async def v1_messages_count_tokens(
        req: AnthropicCountTokensRequest, request: Request
    ):
        log_request("/v1/messages/count_tokens", req, request)
        return await handle_anthropic_count_tokens(req, get_state())

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError):
        # Anthropic clients parse errors as {"type": "error", "error": {...}}; FastAPI's
        # default 422 {"detail": [...]} is unreadable to them. Scoped to /v1/messages*
        # so the OpenAI-protocol routes keep their existing wire behavior.
        if request.url.path.startswith("/v1/messages"):
            return _anthropic_error_response(
                400, "invalid_request_error", _validation_error_message(exc)
            )
        return await request_validation_exception_handler(request, exc)

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException):
        if request.url.path.startswith("/v1/messages"):
            err_type = {
                401: "authentication_error",
                403: "permission_error",
                404: "not_found_error",
                413: "request_too_large",
                429: "rate_limit_error",
                503: "overloaded_error",
                529: "overloaded_error",
            }.get(
                exc.status_code,
                "api_error" if exc.status_code >= 500 else "invalid_request_error",
            )
            detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
            return _anthropic_error_response(exc.status_code, err_type, detail)
        return await http_exception_handler(request, exc)


async def handle_anthropic_messages(
    req: AnthropicMessagesRequest,
    request: Request | None,
    state: Any,
    model_sampling: dict[str, Any],
):
    if not req.messages:
        return _anthropic_error_response(
            400, "invalid_request_error", "messages: at least one message is required"
        )
    try:
        spec = convert_anthropic_to_genspec(
            req,
            model_sampling,
            reasoning_parser=getattr(state.config, "reasoning_parser", None),
            tool_call_parser=getattr(state.config, "tool_call_parser", None),
            default_max_tokens=(
                getattr(state.config, "max_output_tokens", None)
                or DEFAULT_MAX_OUTPUT_TOKENS
            ),
            inline_system_policy=getattr(
                state.config, "anthropic_inline_system", "auto"
            ),
            thinking=getattr(state.config, "thinking", None),
        )
    except GenerationError as exc:
        status_code, err_type = _classify_generation_error(exc)
        return _anthropic_error_response(
            status_code, err_type, _sanitize_error_message(str(exc))
        )
    except ValueError as exc:
        return _anthropic_error_response(400, "invalid_request_error", str(exc))

    if req.stream:
        err = await prerender_error(spec, state)
        if err is not None:
            status_code, err_type = _classify_generation_error(err)
            return _anthropic_error_response(
                status_code, err_type, _sanitize_error_message(str(err))
            )

    try:
        uid = await submit_generation(spec, state)
    except GenerationError as exc:
        status_code, err_type = _classify_generation_error(exc)
        return _anthropic_error_response(
            status_code, err_type, _sanitize_error_message(str(exc))
        )
    except ValueError as exc:
        return _anthropic_error_response(400, "invalid_request_error", str(exc))

    cache_report = getattr(state.config, "enable_cache_report", False)
    if req.stream:
        events = anthropic_event_stream(
            generate_events(uid, spec, state, source="/v1/messages"),
            req.model,
            uid,
            cache_report=cache_report,
        )
        if request is not None:
            events = state.stream_with_cancellation(events, request, uid)
        return StreamingResponse(events, media_type="text/event-stream")

    try:
        # an abandoned request must not keep decoding to max_tokens (#222's watcher)
        result = await _await_watching_disconnect(
            generate_full(uid, spec, state, source="/v1/messages"), request, state, uid
        )
    except GenerationError as exc:
        status_code, err_type = _classify_generation_error(exc)
        return _anthropic_error_response(
            status_code, err_type, _sanitize_error_message(str(exc))
        )
    except Exception as exc:  # noqa: BLE001 — preserve Anthropic JSON error envelope
        return _anthropic_error_response(
            500, "api_error", _sanitize_error_message(str(exc))
        )
    if result is None:
        return _anthropic_error_response(
            499, "api_error", "client disconnected before the response was ready"
        )
    response = anthropic_full_response(
        result, req.model, uid, cache_report=cache_report
    )
    return JSONResponse(content=_dump_anthropic_message(response))


async def handle_anthropic_count_tokens(req: AnthropicCountTokensRequest, state: Any):
    if not req.messages:
        return _anthropic_error_response(
            400, "invalid_request_error", "messages: at least one message is required"
        )
    # Client-input validation (400) is kept strictly separate from tokenizer execution (500):
    # a checkpoint whose chat template raises ValueError is a server fault, not a bad request,
    # so it must not fall into the convert/empty-prompt ValueError branch.
    try:
        messages, template_tools, _, ctk = convert_anthropic_prompt(
            req,
            reasoning_parser=getattr(state.config, "reasoning_parser", None),
            thinking=getattr(state.config, "thinking", None),
        )
    except GenerationError as exc:
        status_code, err_type = _classify_generation_error(exc)
        return _anthropic_error_response(
            status_code, err_type, _sanitize_error_message(str(exc))
        )
    except ValueError as exc:
        return _anthropic_error_response(400, "invalid_request_error", str(exc))
    if not messages:
        # Non-empty on the wire but nothing survived conversion (e.g. image-only blocks on
        # this text-only server) — a client error, not a tokenizer fault.
        return _anthropic_error_response(
            400, "invalid_request_error", "messages: no tokenizable content"
        )
    try:
        n_tokens = await count_prompt_tokens(
            messages,
            template_tools,
            ctk,
            state,
            inline_system_policy=getattr(
                state.config, "anthropic_inline_system", "auto"
            ),
        )
    except GenerationError as exc:
        # The chat template could not render this conversation (bad role ordering, an unmatched
        # tool_result, ...) — classify identically to /v1/messages.
        status_code, err_type = _classify_generation_error(exc)
        return _anthropic_error_response(
            status_code, err_type, _sanitize_error_message(str(exc))
        )
    except Exception as exc:  # noqa: BLE001 — tokenizer init / other failure -> server error
        return _anthropic_error_response(
            500, "api_error", f"token counting failed: {exc}"
        )
    response = AnthropicCountTokensResponse(input_tokens=n_tokens)
    return JSONResponse(content=response.model_dump())


# --------------------------------------------------------------------------- #
# Request conversion: Anthropic Messages -> GenSpec (the neutral generation spec;
# built directly, like vLLM's Responses path — no ChatCompletionRequest pivot).
# --------------------------------------------------------------------------- #
_CCH_TOKEN_RE = re.compile(r"\bcch=[0-9a-fA-F]+;?[ \t]*")


def strip_anthropic_billing_cch(text: str) -> str:
    """Strip per-request ``cch=<hex>;`` tokens from leading ``x-anthropic-billing-header:``
    system blocks so Claude Code turns reuse the cached system-prompt prefix."""
    if not text.lstrip().startswith("x-anthropic-billing-header:"):
        return text
    out: list[str] = []
    for line in text.split("\n"):
        if line.lstrip().startswith("x-anthropic-billing-header:"):
            line = _CCH_TOKEN_RE.sub("", line).rstrip()
        out.append(line)
    return "\n".join(out)


_VALID_THINKING_TYPES = frozenset(
    {
        "enabled",
        "disabled",
        "adaptive",
        "auto",
        "between_tools",
        "on",
        "off",
        "true",
        "false",
        "none",
    }
)
_ANTHROPIC_EFFORTS = frozenset({*_ACCEPTED_EFFORTS, "max"})


def _parse_anthropic_output_format(
    fmt: Any, field_name: str
) -> dict[str, Any] | None:
    if fmt is None:
        return None
    if not isinstance(fmt, dict):
        raise ValueError(f"{field_name} must be an object")
    ftype = fmt.get("type")
    if ftype == "json_schema":
        nested = (
            fmt.get("json_schema")
            if isinstance(fmt.get("json_schema"), dict)
            else fmt
        )
        schema_val = nested.get("schema")
        if not isinstance(schema_val, dict):
            raise GenerationError(
                f"{field_name} with type 'json_schema' requires a 'schema' object",
                code="invalid_json_schema",
            )
        from freetoken.grammar.schema import SchemaValidationError, validate_json_schema

        strict = bool(nested.get("strict", False))
        try:
            validate_json_schema(schema_val, strict=strict, param=field_name)
        except SchemaValidationError as exc:
            raise GenerationError(str(exc), code=exc.code) from exc
        return {"type": "json_schema", "json_schema": nested}
    if ftype == "json_object":
        return {"type": "json_object"}
    if ftype not in (None, "text"):
        raise ValueError(f"Unsupported {field_name}.type: {ftype!r}")
    return None


def _validate_anthropic_controls(
    req: AnthropicMessagesRequest | AnthropicCountTokensRequest,
) -> dict[str, Any] | None:
    """Validate ``thinking``, ``output_config``, and beta ``output_format``, returning ``response_format``."""
    if req.thinking is not None:
        if not isinstance(req.thinking, dict):
            raise ValueError("thinking must be an object")
        ttype = req.thinking.get("type")
        if ttype is not None:
            if not isinstance(ttype, str):
                raise ValueError(f"Unsupported thinking.type: {ttype!r}")
            norm = ttype.strip().lower()
            if norm and norm not in _VALID_THINKING_TYPES:
                raise ValueError(f"Unsupported thinking.type: {ttype!r}")
        budget = req.thinking.get("budget_tokens")
        if budget is not None:
            if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
                raise ValueError("thinking.budget_tokens must be a positive integer")

    cfg_fmt: dict[str, Any] | None = None
    if req.output_config is not None:
        if not isinstance(req.output_config, dict):
            raise ValueError("output_config must be an object")
        eff = req.output_config.get("effort")
        if eff is not None:
            if not isinstance(eff, str):
                raise ValueError(f"Unsupported output_config.effort: {eff!r}")
            norm_eff = eff.strip().lower()
            if norm_eff and norm_eff not in _ANTHROPIC_EFFORTS:
                raise ValueError(f"Unsupported output_config.effort: {eff!r}")
        cfg_fmt = _parse_anthropic_output_format(
            req.output_config.get("format"), "output_config.format"
        )

    beta_fmt = _parse_anthropic_output_format(
        getattr(req, "output_format", None), "output_format"
    )
    if cfg_fmt is not None and beta_fmt is not None and cfg_fmt != beta_fmt:
        raise ValueError(
            "output_config.format and output_format cannot specify conflicting formats"
        )
    return cfg_fmt if cfg_fmt is not None else beta_fmt


def _tool_to_openai_dict(tool: Any) -> dict[str, Any]:
    if isinstance(tool, dict):
        ttype = tool.get("type")
        name = tool.get("name") or ""
        if ttype not in (None, "custom", "function"):
            raise ValueError(
                f"Server tool type {ttype!r} (tool {name!r}) is not supported; only custom client tools are supported"
            )
        if "function" in tool and isinstance(tool["function"], dict):
            fn = dict(tool["function"])
            fn["description"] = fn.get("description") or ""
            if "strict" not in fn and tool.get("strict") is not None:
                fn["strict"] = bool(tool["strict"])
            return {"type": "function", "function": fn}
        schema = dict(tool.get("input_schema") or {"type": "object"})
        if "type" not in schema:
            schema["type"] = "object"
        fn_dict: dict[str, Any] = {
            "name": name,
            "description": tool.get("description") or "",
            "parameters": schema,
        }
        if tool.get("strict") is not None:
            fn_dict["strict"] = bool(tool["strict"])
        return {"type": "function", "function": fn_dict}
    if tool.type not in (None, "custom"):
        raise ValueError(
            f"Server tool type {tool.type!r} (tool {tool.name!r}) is not supported; only custom client tools are supported"
        )
    schema = dict(tool.input_schema or {"type": "object"})
    if "type" not in schema:
        schema["type"] = "object"
    fn_dict = {
        "name": tool.name,
        "description": tool.description if tool.description is not None else "",
        "parameters": schema,
    }
    if tool.strict is not None:
        fn_dict["strict"] = bool(tool.strict)
    return {"type": "function", "function": fn_dict}


def _document_text(block: AnthropicContentBlock) -> str:
    src = block.source or {}
    stype = src.get("type")
    if stype == "text":
        body = src.get("data") or ""
    elif stype == "content":
        raw_c = src.get("content")
        if isinstance(raw_c, str):
            body = raw_c
        elif isinstance(raw_c, list):
            body = "\n".join(
                item.get("text") or ""
                for item in raw_c
                if isinstance(item, dict) and item.get("type") == "text"
            )
        else:
            body = ""
    else:
        raise ValueError(
            f"Unsupported document source type {stype!r}; only 'text' and 'content' documents are supported"
        )
    parts: list[str] = []
    if block.title:
        parts.append(f"[Document: {block.title}]")
    if block.context:
        parts.append(block.context)
    if body:
        parts.append(body)
    return "\n".join(parts)


def convert_anthropic_prompt(
    req: AnthropicMessagesRequest | AnthropicCountTokensRequest,
    reasoning_parser: str | None = None,
    thinking: str | None = None,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]] | None,
    list[dict[str, Any]] | None,
    dict[str, Any],
]:
    """(messages, template_tools, parser_tools, chat_template_kwargs) — the prompt
    side of the conversion, shared by /v1/messages and /v1/messages/count_tokens so
    a counted prompt is exactly the prompt a generation would tokenize."""
    del reasoning_parser
    _validate_anthropic_controls(req)

    # Build active tool registry (supporting defer_loading, tool_addition, tool_removal, tool_reference).
    tool_order: list[str] = []
    known_tools: dict[str, dict[str, Any]] = {}
    deferred_names: set[str] = set()
    referenced_names: set[str] = set()
    removed_names: set[str] = set()
    for t in req.tools or []:
        t_dict = _tool_to_openai_dict(t)
        t_name = t_dict["function"]["name"]
        if t_name not in known_tools:
            tool_order.append(t_name)
        known_tools[t_name] = t_dict
        if getattr(t, "defer_loading", False):
            deferred_names.add(t_name)

    # Hoisting turn-specific reminders rewrites the cached prompt prefix.
    # Preserve their position; the tokenizer adapts them to the model's roles.
    system_texts: list[str] = []
    if req.system:
        if isinstance(req.system, str):
            stripped = strip_anthropic_billing_cch(req.system)
            if stripped:
                system_texts.append(stripped)
        else:
            joined = "".join(
                strip_anthropic_billing_cch(b.text)
                for b in req.system
                if b.type == "text" and b.text
            )
            if joined:
                system_texts.append(joined)

    other: list[dict[str, Any]] = []
    for msg in req.messages:
        if msg.role == "system":
            if isinstance(msg.content, list):
                for b in msg.content:
                    if b.type == "tool_addition":
                        for raw_add in b.tools or []:
                            added = _tool_to_openai_dict(raw_add)
                            aname = added["function"]["name"]
                            if aname not in known_tools:
                                tool_order.append(aname)
                            known_tools[aname] = added
                            removed_names.discard(aname)
                            if isinstance(raw_add, dict) and raw_add.get(
                                "defer_loading"
                            ):
                                deferred_names.add(aname)
                    elif b.type == "tool_removal":
                        names_to_drop = list(b.tool_names or [])
                        if b.tool_name:
                            names_to_drop.append(b.tool_name)
                        if b.name:
                            names_to_drop.append(b.name)
                        removed_names.update(names_to_drop)
            text = strip_anthropic_billing_cch(_content_text(msg.content))
            if not other:
                if text:
                    system_texts.append(text)
            elif text:
                other.append({"role": "system", "content": text})
            continue

        if isinstance(msg.content, str):
            if (
                msg.role == "user"
                and other
                and other[-1].get("role") == "user"
                and isinstance(other[-1].get("content"), str)
            ):
                prev_c = other[-1]["content"]
                other[-1]["content"] = (
                    f"{prev_c}\n\n{msg.content}"
                    if (prev_c and msg.content)
                    else (prev_c or msg.content)
                )
            else:
                other.append({"role": msg.role, "content": msg.content})
            continue

        content_parts: list[dict[str, Any]] = []
        tool_calls: list[dict[str, Any]] = []
        thinking_parts: list[str] = []
        saw_tool_result = False
        for block in msg.content:
            if block.type == "text" and block.text:
                content_parts.append({"type": "text", "text": block.text})
            elif block.type == "document":
                doc_txt = _document_text(block)
                if doc_txt:
                    content_parts.append({"type": "text", "text": doc_txt})
            elif block.type == "thinking" and block.thinking:
                # -> reasoning_content; redacted_thinking stays skipped (opaque payload).
                thinking_parts.append(block.thinking)
            elif block.type == "image":
                content_parts.append(_image_part(block.source))
                continue
            elif block.type == "tool_use":
                tool_calls.append(
                    {
                        "id": block.id or f"call_{uuid.uuid4().hex[:8]}",
                        "type": "function",
                        "function": {
                            "name": block.name or "",
                            "arguments": json.dumps(block.input or {}),
                        },
                    }
                )
            elif block.type == "tool_result":
                saw_tool_result = True
                text, images, refs = _tool_result_parts(block.content, known_tools)
                referenced_names.update(refs)
                if block.is_error and not text:
                    text = "Tool execution failed"
                if msg.role == "user":
                    other.append(
                        {
                            "role": "tool",
                            "tool_call_id": block.tool_use_id or block.id or "",
                            "content": text,
                        }
                    )
                    # Chat templates render tool messages as text, so the images ride on the user turn that follows the tool messages (vLLM does the same).
                    content_parts.extend(images)
                else:
                    if images:
                        raise ValueError(
                            "images inside a tool_result are only accepted in a user message"
                        )
                    content_parts.append(
                        {"type": "text", "text": f"Tool result: {text}"}
                    )

        openai_msg: dict[str, Any] = {"role": msg.role}
        if thinking_parts:
            openai_msg["reasoning_content"] = "\n\n".join(thinking_parts)
        if tool_calls:
            openai_msg["tool_calls"] = tool_calls
        if content_parts:
            if len(content_parts) == 1 and content_parts[0]["type"] == "text":
                openai_msg["content"] = content_parts[0]["text"]
            else:
                openai_msg["content"] = content_parts
        elif not tool_calls and not thinking_parts:
            if msg.role == "assistant":
                # Preserve empty / redacted_thinking-only assistant turns so adjacent user turns never merge.
                openai_msg["content"] = ""
            elif saw_tool_result:
                continue
            else:
                continue
        other.append(openai_msg)

    messages: list[dict[str, Any]] = []
    system_text = "\n\n".join(t for t in system_texts if t)
    if system_text:
        messages.append({"role": "system", "content": system_text})
    messages.extend(other)

    active_all_tools = [
        known_tools[name] for name in tool_order if name not in removed_names
    ]
    visible_tools = [
        known_tools[name]
        for name in tool_order
        if name not in removed_names
        and (name not in deferred_names or name in referenced_names)
    ]

    # Validate and apply Anthropic tool_choice ("none", "auto", "any", "tool").
    if req.tool_choice and req.tool_choice.type == "none":
        template_tools, parser_tools = None, None
    else:
        selected: str | None = None
        if req.tool_choice:
            if req.tool_choice.type == "any" and not active_all_tools:
                raise ValueError("tool_choice.type='any' requires non-empty tools")
            if req.tool_choice.type == "tool":
                selected = req.tool_choice.name
                if not selected:
                    raise ValueError("tool_choice.type='tool' requires a tool name")
                active_names = {t["function"]["name"] for t in active_all_tools}
                if selected not in active_names:
                    raise ValueError(
                        f"tool_choice tool {selected!r} was not found in tools"
                    )
        if selected:
            template_tools = [
                t for t in active_all_tools if t["function"]["name"] == selected
            ]
            parser_tools = active_all_tools or None
        else:
            template_tools = visible_tools or None
            parser_tools = active_all_tools or None

    # Native extended-thinking toggle -> template kwargs, broadcast in every
    # spelling the ecosystem's templates read (a bare enable_thinking bool is
    # inert for templates that read a different knob, e.g. M3's thinking_mode).
    # output_config.effort folds in as the Responses API's reasoning.effort does;
    # an explicit thinking type still decides on/off ("disabled" wins over effort).
    from .model_meta import (
        _DISABLE_THINKING_TYPES,
        effort_toggle_kwargs,
        thinking_override_kwargs,
        thinking_toggle_kwargs,
    )

    ctk: dict[str, Any] = {}
    raw_thinking_type = (
        req.thinking.get("type") if isinstance(req.thinking, dict) else None
    )
    thinking_type = (
        raw_thinking_type.strip().lower()
        if isinstance(raw_thinking_type, str) and raw_thinking_type.strip()
        else None
    )
    output_config = req.output_config if isinstance(req.output_config, dict) else {}
    effort = output_config.get("effort")
    if isinstance(effort, str) and effort.strip():
        ctk = effort_toggle_kwargs(effort, ctk, thinking_type=thinking_type)
    elif thinking_type in _DISABLE_THINKING_TYPES:
        ctk = thinking_toggle_kwargs(False)
    elif thinking_type == "auto":
        pass
    elif thinking_type is not None:
        ctk = thinking_toggle_kwargs(True)
    ctk = thinking_override_kwargs(thinking, ctk)

    if req.tool_choice and req.tool_choice.disable_parallel_tool_use:
        ctk["parallel_tool_calls"] = False

    rendered_msgs = render_messages(messages, hoist_system=False)
    if (
        rendered_msgs
        and rendered_msgs[-1].get("role") == "assistant"
        and not rendered_msgs[-1].get("tool_calls")
    ):
        ctk["continue_final_message"] = True

    # late system reminders stay in place: the tokenizer folds them instead (prefix reuse)
    return (
        rendered_msgs,
        template_tools,
        parser_tools,
        ctk,
    )


def convert_anthropic_to_genspec(
    req: AnthropicMessagesRequest,
    model_sampling: dict[str, Any],
    reasoning_parser: str | None = None,
    default_max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    inline_system_policy: str = "auto",
    thinking: str | None = None,
    tool_call_parser: str | None = None,
) -> GenSpec:
    from freetoken.grammar.schema import (
        SchemaValidationError,
        build_structured_output_spec,
    )

    response_format = _validate_anthropic_controls(req)
    messages, template_tools, parser_tools, ctk = convert_anthropic_prompt(
        req, reasoning_parser=reasoning_parser, thinking=thinking
    )
    tool_choice_mode: str | None = None
    if req.tool_choice:
        if req.tool_choice.type == "any":
            tool_choice_mode = "required"
        elif req.tool_choice.type == "tool" and req.tool_choice.name:
            tool_choice_mode = f"named:{req.tool_choice.name}"
    disable_parallel = bool(
        req.tool_choice and req.tool_choice.disable_parallel_tool_use
    )
    sp = resolve_sampling(
        temperature=req.temperature,
        top_k=req.top_k,
        top_p=req.top_p,
        max_tokens=req.max_tokens,
        ignore_eos=False,
        model_sampling=model_sampling,
        stop=req.stop_sequences,
        default_max_tokens=default_max_tokens,
    )
    try:
        sp.structured_output = build_structured_output_spec(
            response_format=response_format,
            tools=template_tools,
            tool_choice_mode=tool_choice_mode,
            tool_call_parser=tool_call_parser,
            disable_parallel_tool_use=disable_parallel,
            reasoning_parser=reasoning_parser,
            chat_template_kwargs=ctk,
        )
    except SchemaValidationError as exc:
        raise GenerationError(str(exc), code=exc.code) from exc
    return GenSpec(
        messages=messages,
        sampling_params=sp,
        chat_template_kwargs=ctk,
        template_tools=template_tools,
        parser_tools=parser_tools,
        inline_system_policy=inline_system_policy,
        disable_parallel_tool_use=disable_parallel,
        tool_choice_mode=tool_choice_mode,
        response_format=response_format,
    )


def _content_text(content) -> str:
    """Plain text of an Anthropic message content (str or list of blocks)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "".join(
        b.text for b in content if getattr(b, "type", None) == "text" and b.text
    )


def _image_part(source: dict[str, Any] | None) -> dict[str, Any]:
    src = source or {}
    stype = src.get("type")
    if stype not in ("base64", "url"):
        raise ValueError(f"unsupported image source type: {stype!r}")
    data = src.get("data") if stype == "base64" else src.get("url")
    if not data:
        # an unsupported image source must fail the request, not degrade to a text-only answer
        raise ValueError(f"image source of type {stype!r} is missing data/url")
    return {
        "type": "image",
        "freetoken_ref": {"kind": "b64" if stype == "base64" else "url", "data": data},
    }


def _tool_result_parts(
    content: Any, known_tools: dict[str, dict[str, Any]] | None = None
) -> tuple[str, list[dict[str, Any]], list[str]]:
    """The text of a tool_result, its image blocks, and any referenced deferred tool names."""
    if content is None:
        return "", [], []
    if isinstance(content, str):
        return content, [], []
    texts: list[str] = []
    images: list[dict[str, Any]] = []
    referenced: list[str] = []
    loaded_defs: list[dict[str, Any]] = []
    for item in content:
        if isinstance(item, dict):
            itype = item.get("type")
            if itype == "image":
                images.append(_image_part(item.get("source")))
            elif itype == "tool_reference":
                ref_name = item.get("tool_name") or item.get("name") or ""
                if ref_name:
                    referenced.append(ref_name)
                    if known_tools and ref_name in known_tools:
                        loaded_defs.append(known_tools[ref_name]["function"])
            elif itype == "document":
                doc_block = AnthropicContentBlock.model_validate(item)
                doc_txt = _document_text(doc_block)
                if doc_txt:
                    texts.append(doc_txt)
            else:
                txt = item.get("text")
                if txt:
                    texts.append(str(txt))
        else:
            texts.append(str(item))
    if loaded_defs:
        texts.append(
            "<additional_tools>\n"
            + json.dumps(loaded_defs, ensure_ascii=False)
            + "\n</additional_tools>"
        )
    return "\n".join(texts), images, referenced


# --------------------------------------------------------------------------- #
# Output formatting: GenResult / GenEvent -> Anthropic response / events
# --------------------------------------------------------------------------- #
def anthropic_full_response(
    result: GenResult, model: str, uid: int, cache_report: bool = False
) -> AnthropicMessagesResponse:
    content: list[AnthropicContentBlock] = []
    if result.reasoning:
        content.append(
            AnthropicContentBlock(
                type="thinking",
                thinking=result.reasoning,
                signature=uuid.uuid4().hex,
            )
        )
    if result.content or (not result.reasoning and not result.tool_calls):
        content.append(AnthropicContentBlock(type="text", text=result.content or ""))
    for index, call in enumerate(result.tool_calls):
        content.append(
            AnthropicContentBlock(
                type="tool_use",
                id=_tool_use_id(call.name, index),
                name=call.name,
                input=_parse_json_args(call.parameters),
            )
        )
    stop_reason, stop_sequence = _anthropic_stop(
        result.finish_reason, result.matched_stop
    )
    return AnthropicMessagesResponse(
        id=f"msg_{public_request_id(uid)}",
        content=content,
        model=model,
        stop_reason=stop_reason,
        stop_sequence=stop_sequence,
        usage=_anthropic_usage(
            result.prompt_tokens,
            result.completion_tokens,
            result.cached_tokens,
            cache_report,
        ),
    )


def _anthropic_usage(
    prompt_tokens: int, completion_tokens: int, cached_tokens: int, cache_report: bool
) -> AnthropicUsage:
    """Anthropic billing semantics under --enable-cache-report: input_tokens EXCLUDES
    the cached prefix and cache_read_input_tokens carries it (absent when 0, matching
    sglang). With the flag off, input_tokens is the full prompt length."""
    cached = cached_tokens if cache_report else 0
    extra = {"cache_read_input_tokens": cached} if cached > 0 else {}
    return AnthropicUsage(
        input_tokens=max(prompt_tokens - cached, 0),
        output_tokens=completion_tokens,
        **extra,
    )


async def anthropic_event_stream(
    events: AsyncIterator[Any], model: str, uid: int, cache_report: bool = False
) -> AsyncIterator[str]:
    """Format the protocol-neutral GenEvent stream into Anthropic SSE events.

    No re-parsing of an OpenAI stream: this consumes ReasoningDelta / ContentDelta /
    ToolCallsDelta / GenDone directly. The terminal message_delta + message_stop are
    driven by GenDone (always present), so there is no usage-chunk dependency.
    """
    block_index = 0
    block_open: str | None = (
        None  # "text" | "thinking" (tool_use blocks open+close atomically)
    )

    def _open_text() -> str:
        nonlocal block_open
        block_open = "text"
        return _event(
            AnthropicStreamEvent(
                type="content_block_start",
                index=block_index,
                content_block=AnthropicContentBlock(type="text", text=""),
            )
        )

    def _open_thinking() -> str:
        nonlocal block_open
        block_open = "thinking"
        return _event(
            AnthropicStreamEvent(
                type="content_block_start",
                index=block_index,
                content_block=AnthropicContentBlock(
                    type="thinking", thinking="", signature=""
                ),
            )
        )

    def _stop_block() -> list[str]:
        nonlocal block_open, block_index
        frames: list[str] = []
        if block_open == "thinking":
            # Emit a non-empty opaque signature so Claude Code / Anthropic SDKs preserve
            # the thinking block across multi-turn replays.
            frames.append(
                _event(
                    AnthropicStreamEvent(
                        type="content_block_delta",
                        index=block_index,
                        delta=AnthropicDelta(
                            type="signature_delta", signature=uuid.uuid4().hex
                        ),
                    )
                )
            )
        frames.append(
            _event(AnthropicStreamEvent(type="content_block_stop", index=block_index))
        )
        block_open = None
        block_index += 1
        return frames

    started = False

    def _message_start(usage: AnthropicUsage) -> str:
        nonlocal started
        started = True
        return _event(
            AnthropicStreamEvent(
                type="message_start",
                message=AnthropicMessagesResponse(
                    id=f"msg_{public_request_id(uid)}",
                    content=[],
                    model=model,
                    stop_reason=None,
                    stop_sequence=None,
                    usage=usage,
                ),
            )
        )

    # message_start waits for the prompt's admission so it can carry the real input/cache
    # usage; anything that must go out first (a keepalive, an error) sends it with zeros.
    zero_usage = AnthropicUsage(input_tokens=0, output_tokens=0)

    def _open_tool(
        name: str | None, ordinal: int | None, stable: bool = True
    ) -> list[str]:
        nonlocal block_open, tool_args_sent, tool_ordinal, tool_stable
        frames: list[str] = []
        if block_open:
            frames.extend(_stop_block())
        frames.append(
            _event(
                AnthropicStreamEvent(
                    type="content_block_start",
                    index=block_index,
                    content_block=AnthropicContentBlock(
                        type="tool_use",
                        id=_tool_use_id(name, block_index),
                        name=name,
                        input={},
                    ),
                )
            )
        )
        block_open = "tool"
        tool_args_sent = ""
        tool_ordinal = ordinal
        tool_stable = stable
        return frames

    def _tool_args_delta(fragment: str) -> str:
        nonlocal tool_args_sent
        tool_args_sent += fragment
        return _event(
            AnthropicStreamEvent(
                type="content_block_delta",
                index=block_index,
                delta=AnthropicDelta(type="input_json_delta", partial_json=fragment),
            )
        )

    tool_args_sent = ""
    tool_ordinal: int | None = None
    tool_stable = True
    events = with_keepalive(events, KEEPALIVE_INTERVAL_S)
    try:
        async for ev in events:
            if isinstance(ev, PromptAdmitted):
                if not started:
                    yield _message_start(
                        _anthropic_usage(
                            ev.prompt_tokens, 0, ev.cached_tokens, cache_report
                        )
                    )
                continue
            if not started:
                yield _message_start(zero_usage)
            if ev is KEEPALIVE:
                yield _event(AnthropicStreamEvent(type="ping"))
                continue

            if isinstance(ev, ReasoningDelta):
                # Streamed as a native thinking block (Claude Code renders these).
                if ev.text == "":
                    continue
                if block_open != "thinking":
                    if block_open:
                        for f in _stop_block():
                            yield f
                    yield _open_thinking()
                yield _event(
                    AnthropicStreamEvent(
                        type="content_block_delta",
                        index=block_index,
                        delta=AnthropicDelta(type="thinking_delta", thinking=ev.text),
                    )
                )

            elif isinstance(ev, ContentDelta):
                if ev.text == "":
                    continue
                if block_open != "text":
                    if block_open:
                        for f in _stop_block():
                            yield f
                    yield _open_text()
                yield _event(
                    AnthropicStreamEvent(
                        type="content_block_delta",
                        index=block_index,
                        delta=AnthropicDelta(type="text_delta", text=ev.text),
                    )
                )

            elif isinstance(ev, ToolCallStart):
                for f in _open_tool(ev.name, ev.tool_index, ev.args_prefix_stable):
                    yield f

            elif isinstance(ev, ToolCallArgsDelta):
                if block_open != "tool" or not ev.fragment:
                    continue  # defensive: fragment without an open tool block or empty fragment
                if not tool_stable:
                    continue  # fragments don't concatenate cleanly: full args at close
                yield _tool_args_delta(ev.fragment)

            elif isinstance(ev, ToolCallsDelta):
                for call in ev.calls:
                    if block_open == "tool" and tool_ordinal == call.tool_index:
                        # Close of a ToolCallStart-opened block: top up whatever of
                        # the final (authoritative) arguments wasn't streamed yet.
                        final = _canonical_json_args_str(call.parameters)
                        if not tool_args_sent:
                            yield _tool_args_delta(final)
                        elif final.startswith(tool_args_sent):
                            remainder = final[len(tool_args_sent) :]
                            if remainder:
                                yield _tool_args_delta(remainder)
                        for f in _stop_block():
                            yield f
                        continue
                    # Standalone complete call (buffered fallback path).
                    for f in _open_tool(call.name, call.tool_index):
                        yield f
                    yield _tool_args_delta(_canonical_json_args_str(call.parameters))
                    for f in _stop_block():
                        yield f

            elif isinstance(ev, GenDone):
                if block_open:
                    for f in _stop_block():
                        yield f
                stop_reason, stop_sequence = _anthropic_stop(
                    ev.finish_reason, ev.matched_stop
                )
                yield _event(
                    AnthropicStreamEvent(
                        type="message_delta",
                        delta=AnthropicDelta(
                            stop_reason=stop_reason, stop_sequence=stop_sequence
                        ),
                        usage=_anthropic_usage(
                            ev.prompt_tokens,
                            ev.completion_tokens,
                            ev.cached_tokens,
                            cache_report,
                        ),
                    )
                )
                yield _event(AnthropicStreamEvent(type="message_stop"))
                # Anthropic streams terminate on message_stop — no OpenAI-style
                # `data: [DONE]` sentinel (it would be an unknown event to strict clients).
        if not started:
            yield _message_start(zero_usage)
    except GenerationError as exc:
        # Request-side failure (template rejection / over-length prompt) or classified capacity/engine fault.
        _, err_type = _classify_generation_error(exc)
        if not started:
            yield _message_start(zero_usage)
        if block_open:
            for f in _stop_block():
                yield f
        yield _event(
            AnthropicStreamEvent(
                type="error",
                error=AnthropicError(
                    type=err_type, message=_sanitize_error_message(str(exc))
                ),
            )
        )
    except Exception as exc:  # noqa: BLE001 — surface as an Anthropic error event
        if not started:
            yield _message_start(zero_usage)
        if block_open:
            for f in _stop_block():
                yield f
        yield _event(
            AnthropicStreamEvent(
                type="error",
                error=AnthropicError(
                    type="api_error", message=_sanitize_error_message(str(exc))
                ),
            )
        )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _dump_anthropic_message(msg: AnthropicMessagesResponse) -> dict[str, Any]:
    """Serialize ``AnthropicMessagesResponse`` keeping required nullable ``stop_reason`` and ``stop_sequence``."""
    d = msg.model_dump(exclude_none=True)
    d["stop_reason"] = msg.stop_reason
    d["stop_sequence"] = msg.stop_sequence
    return d


def _event(event: AnthropicStreamEvent) -> str:
    """Encode an event as a named Anthropic SSE frame: ``event: <type>\\ndata: <json>``."""
    d = event.model_dump(exclude_none=True)
    if event.type == "message_start" and event.message is not None:
        d["message"] = _dump_anthropic_message(event.message)
    elif event.type == "message_delta" and event.delta is not None:
        d.setdefault("delta", {})
        d["delta"]["stop_reason"] = event.delta.stop_reason
        d["delta"]["stop_sequence"] = event.delta.stop_sequence
    return f"event: {event.type}\ndata: {json.dumps(d, ensure_ascii=False)}\n\n"


def _canonical_json_args_str(arguments: Any) -> str:
    """Return a non-empty valid JSON object string so ``anthropic-sdk-python``'s ``jiter.from_json`` never crashes."""
    if isinstance(arguments, str) and arguments.strip():
        try:
            parsed = json.loads(arguments)
            if isinstance(parsed, dict):
                return arguments
        except json.JSONDecodeError:
            pass
    return json.dumps(_parse_json_args(arguments), ensure_ascii=False)


def _tool_use_id(name: str | None, index: int) -> str:
    prefix = (name or "tool").replace("_", "-")[:24]
    return f"toolu_{prefix}_{index}_{uuid.uuid4().hex[:8]}"


def _parse_json_args(arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str) and arguments:
        try:
            parsed = json.loads(arguments)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _validation_error_message(exc: RequestValidationError) -> str:
    """First pydantic error as an Anthropic-style 'field.path: message' line."""
    try:
        err = exc.errors()[0]
        loc = ".".join(str(p) for p in err.get("loc", ()) if p != "body")
        msg = err.get("msg") or "invalid request"
        return f"{loc}: {msg}" if loc else msg
    except Exception:  # noqa: BLE001 — never let error formatting raise
        return "invalid request"


def _sanitize_error_message(message: str) -> str:
    """Strip CR/NUL control characters so multiline or raw error messages never break SSE framing."""
    return (
        message.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n").strip()
        or "error"
    )


def _classify_generation_error(exc: GenerationError) -> tuple[int, str]:
    """Map ``GenerationError.code`` to ``(http_status, anthropic_error_type)``."""
    code = (getattr(exc, "code", None) or "").strip().lower()
    if code in {"overloaded", "server_busy", "capacity_exceeded"}:
        return 529, "overloaded_error"
    if code in {"rate_limit", "rate_limit_exceeded"}:
        return 429, "rate_limit_error"
    if code in {"internal_error", "engine_error"}:
        return 500, "api_error"
    return 400, "invalid_request_error"


def _anthropic_error_response(
    status_code: int, err_type: str, message: str
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=AnthropicErrorResponse(
            error=AnthropicError(type=err_type, message=message)
        ).model_dump(),
    )
