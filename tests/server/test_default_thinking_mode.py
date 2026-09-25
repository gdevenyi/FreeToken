"""Unit tests for the --default-thinking-mode server flag merge logic."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

from freetoken.server import anthropic_api as A
from freetoken.server.anthropic_models import AnthropicCountTokensRequest, AnthropicMessagesRequest
from freetoken.server.api_models import ChatCompletionRequest
from freetoken.server.model_meta import apply_default_thinking_mode
from freetoken.server.openai_api import chat_request_to_genspec
from freetoken.server.responses_api import ResponsesRequest, convert_responses_to_genspec

OFF = {"enable_thinking": False, "thinking_mode": "disabled"}
ON = {"enable_thinking": True, "thinking_mode": "enabled"}


def test_auto_is_noop():
    assert apply_default_thinking_mode(None, "auto") is None
    assert apply_default_thinking_mode({"enable_thinking": True}, "auto") == {
        "enable_thinking": True
    }
    assert apply_default_thinking_mode(None, None) is None


def test_chat_fills_unset_kwargs():
    assert apply_default_thinking_mode(None, "chat") == OFF
    assert apply_default_thinking_mode({}, "chat") == OFF
    assert apply_default_thinking_mode({"other": 1}, "chat") == {"other": 1, **OFF}


def test_thinking_fills_unset_kwargs():
    assert apply_default_thinking_mode(None, "thinking") == ON
    assert apply_default_thinking_mode({}, "thinking") == ON


def test_explicit_request_value_wins():
    # Explicit per-request values override the server default, both directions.
    assert apply_default_thinking_mode({"enable_thinking": True}, "chat") == {
        "enable_thinking": True
    }
    assert apply_default_thinking_mode({"enable_thinking": False}, "thinking") == {
        "enable_thinking": False
    }
    assert apply_default_thinking_mode({"thinking": False}, "chat") == {"thinking": False}
    assert apply_default_thinking_mode({"thinking_mode": "thinking"}, "chat") == {
        "thinking_mode": "thinking"
    }
    # A raw graded-effort request asks for thinking too.
    assert apply_default_thinking_mode({"reasoning_effort": "high"}, "chat") == {
        "reasoning_effort": "high"
    }


def test_original_dict_not_mutated():
    original = {"other": 1}
    result = apply_default_thinking_mode(original, "chat")
    assert result == {"other": 1, **OFF}
    assert original == {"other": 1}


def test_unknown_mode_is_noop():
    assert apply_default_thinking_mode({"other": 1}, "bogus") == {"other": 1}


def _chat(**extra):
    body = {"model": "m", "messages": [{"role": "user", "content": "hi"}], **extra}
    return chat_request_to_genspec(
        ChatCompletionRequest.model_validate(body), {}, default_thinking_mode="chat"
    ).chat_template_kwargs


def test_chat_default_fills_a_request_without_thinking_controls():
    assert _chat() == OFF


def test_chat_default_keeps_protocol_level_thinking_requests():
    assert _chat(reasoning_effort="high")["enable_thinking"] is True
    assert _chat(thinking={"type": "enabled"})["enable_thinking"] is True
    assert _chat(reasoning_effort="none")["enable_thinking"] is False


def test_responses_chat_default_keeps_reasoning_effort():
    def ctk(**extra):
        req = ResponsesRequest.model_validate({"model": "m", "input": "hi", **extra})
        return convert_responses_to_genspec(req, {}, default_thinking_mode="chat").chat_template_kwargs

    assert ctk() == OFF
    on = ctk(reasoning={"effort": "medium"})
    assert on["enable_thinking"] is True and on["reasoning_effort"] == "medium"


def _anthropic(**extra):
    return AnthropicMessagesRequest.model_validate(
        {"model": "claude-x", "max_tokens": 64, "messages": [{"role": "user", "content": "hi"}], **extra}
    )


def test_anthropic_chat_default_applies_only_without_a_thinking_block():
    def ctk(**extra):
        return A.convert_anthropic_to_genspec(
            _anthropic(**extra), {}, default_thinking_mode="chat"
        ).chat_template_kwargs

    assert ctk() == OFF
    assert ctk(thinking={"type": "enabled", "budget_tokens": 1024}) == ON
    assert ctk(thinking={"type": "adaptive"}) == {}


def test_anthropic_count_tokens_renders_the_generation_prompt_under_the_default(monkeypatch):
    captured = {}

    async def fake_count(messages, template_tools, ctk, state):
        captured["ctk"] = ctk
        return 1

    monkeypatch.setattr(A, "count_prompt_tokens", fake_count)
    state = SimpleNamespace(config=SimpleNamespace(reasoning_parser=None, default_thinking_mode="chat"))
    body = {"model": "claude-x", "messages": [{"role": "user", "content": "hi"}]}
    asyncio.run(A.handle_anthropic_count_tokens(AnthropicCountTokensRequest.model_validate(body), state))

    generation = A.convert_anthropic_to_genspec(_anthropic(), {}, default_thinking_mode="chat")
    assert captured["ctk"] == generation.chat_template_kwargs == OFF
