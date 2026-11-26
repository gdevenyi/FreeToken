"""Pydantic models for the Anthropic Messages API.

Adapted from vLLM's ``vllm/entrypoints/anthropic/protocol.py`` (Apache-2.0,
Copyright contributors to the vLLM project). Kept close to the upstream shapes
so the conversion logic in :mod:`freetoken.server.anthropic_api` mirrors vLLM's
``AnthropicServingMessages``.
"""

from __future__ import annotations

import time
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class AnthropicError(BaseModel):
    type: str
    message: str


class AnthropicErrorResponse(BaseModel):
    type: Literal["error"] = "error"
    error: AnthropicError


class AnthropicUsage(BaseModel):
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int | None = None
    cache_read_input_tokens: int | None = None


class AnthropicContentBlock(BaseModel):
    """A content block in a message. ``type`` is left open (str) rather than a
    Literal: real clients (Claude Code) send blocks beyond text/thinking/image/
    tool_use/tool_result (e.g. ``redacted_thinking``); the converter handles the
    known types and skips the rest instead of rejecting the whole request with a 422."""

    model_config = ConfigDict(extra="allow")

    type: str
    text: str | None = None
    # For thinking blocks.
    thinking: str | None = None
    signature: str | None = None
    # For image / document content.
    source: dict[str, Any] | None = None
    title: str | None = None
    context: str | None = None
    # For tool use / result / deferred tool reference.
    id: str | None = None
    tool_use_id: str | None = None
    name: str | None = None
    tool_name: str | None = None
    input: dict[str, Any] | None = None
    content: str | list[dict[str, Any]] | None = None
    is_error: bool | None = None
    tools: list[dict[str, Any]] | None = None
    tool_names: list[str] | None = None


class AnthropicMessage(BaseModel):
    # Anthropic's spec is user/assistant only, but Claude Code sends system-role
    # messages inside the array (not just the top-level `system` field); accept it
    # so the converter can render it as a system message instead of 422-ing.
    model_config = ConfigDict(extra="allow")

    role: Literal["user", "assistant", "system"]
    content: str | list[AnthropicContentBlock]


class AnthropicTool(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str | None = None
    name: str
    description: str | None = None
    input_schema: dict[str, Any] | None = None
    defer_loading: bool | None = None
    strict: bool | None = None

    @model_validator(mode="after")
    def validate_tool_schema(self) -> "AnthropicTool":
        if self.type not in (None, "custom"):
            # Built-in server tools (web_search_*, bash_*, text_editor_*) omit input_schema;
            # let them reach the handler so it returns a specific unsupported-server-tool 400.
            return self
        if not isinstance(self.input_schema, dict):
            raise ValueError("input_schema must be a dictionary")
        if "type" not in self.input_schema:
            self.input_schema["type"] = "object"
        return self


class AnthropicToolChoice(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: Literal["auto", "any", "tool", "none"]
    name: str | None = None
    disable_parallel_tool_use: bool | None = None


class AnthropicMessagesRequest(BaseModel):
    """Anthropic Messages API request (https://docs.anthropic.com/en/api/messages)."""

    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[AnthropicMessage]
    max_tokens: int
    metadata: dict[str, Any] | None = None
    stop_sequences: list[str] | None = None
    stream: bool | None = False
    system: str | list[AnthropicContentBlock] | None = None
    temperature: float | None = None
    # Native extended-thinking toggle: {"type": "enabled"|"disabled", "budget_tokens": N}.
    # budget_tokens is accepted but not enforced (stateless subset).
    thinking: dict[str, Any] | None = None
    # {"effort": "low"|"medium"|"high"|"max", "format": ...}.
    output_config: dict[str, Any] | None = None
    tool_choice: AnthropicToolChoice | None = None
    tools: list[AnthropicTool] | None = None
    top_k: int | None = None
    top_p: float | None = None

    @field_validator("model")
    @classmethod
    def validate_model_non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("model must not be empty")
        return v

    @field_validator("max_tokens")
    @classmethod
    def validate_max_tokens(cls, v):
        if v <= 0:
            raise ValueError("max_tokens must be positive")
        return v


class AnthropicCountTokensRequest(BaseModel):
    """Anthropic count_tokens request (POST /v1/messages/count_tokens): the input
    side of a Messages request — no max_tokens/stream/sampling fields."""

    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[AnthropicMessage]
    system: str | list[AnthropicContentBlock] | None = None
    thinking: dict[str, Any] | None = None
    output_config: dict[str, Any] | None = None
    tool_choice: AnthropicToolChoice | None = None
    tools: list[AnthropicTool] | None = None

    @field_validator("model")
    @classmethod
    def validate_model_non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("model must not be empty")
        return v


class AnthropicCountTokensResponse(BaseModel):
    input_tokens: int


class AnthropicDelta(BaseModel):
    """Delta payload for streaming events."""

    type: (
        Literal["text_delta", "input_json_delta", "thinking_delta", "signature_delta"]
        | None
    ) = None
    text: str | None = None
    partial_json: str | None = None
    thinking: str | None = None
    signature: str | None = None

    # message_delta carries the terminal stop reason.
    stop_reason: (
        Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"] | None
    ) = None
    stop_sequence: str | None = None


class AnthropicMessagesResponse(BaseModel):
    id: str
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    content: list[AnthropicContentBlock]
    model: str
    stop_reason: (
        Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"] | None
    ) = None
    stop_sequence: str | None = None
    usage: AnthropicUsage | None = None

    def model_post_init(self, __context):
        if not self.id:
            self.id = f"msg_{int(time.time() * 1000)}"


class AnthropicStreamEvent(BaseModel):
    type: Literal[
        "message_start",
        "message_delta",
        "message_stop",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "ping",
        "error",
    ]
    message: Optional[AnthropicMessagesResponse] = None
    delta: AnthropicDelta | None = None
    content_block: AnthropicContentBlock | None = None
    index: int | None = None
    error: AnthropicError | None = None
    usage: AnthropicUsage | None = None
