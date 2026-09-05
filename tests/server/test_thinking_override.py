"""--thinking {auto,on,off}: operator policy must beat every client spelling of the knob.

Per-request chat_template_kwargs works, but generic OpenAI/Anthropic clients send their own
thinking keys, so a per-request setting can override an operator's intended server policy.
Upstream issue #302.
"""

from freetoken.server.model_meta import thinking_override_kwargs
from freetoken.tokenizer.effort import THINKING_OFF_KWARGS, THINKING_ON_KWARGS


def test_auto_and_none_leave_the_request_alone():
    for policy in ("auto", None):
        assert thinking_override_kwargs(policy, {"enable_thinking": True}) == {
            "enable_thinking": True
        }
        assert thinking_override_kwargs(policy, None) == {}


def test_off_beats_every_client_spelling():
    for asked in ({"enable_thinking": True}, {"thinking": True}, {"thinking_mode": "on"},
                  {"reasoning_effort": "xhigh"}, dict(THINKING_ON_KWARGS)):
        got = thinking_override_kwargs("off", asked)
        assert got == dict(THINKING_OFF_KWARGS), (asked, got)


def test_on_beats_a_client_that_asked_for_off():
    assert thinking_override_kwargs("on", dict(THINKING_OFF_KWARGS)) == dict(THINKING_ON_KWARGS)


def test_unrelated_kwargs_ride_along():
    got = thinking_override_kwargs("off", {"enable_thinking": True, "custom_flag": 7})
    assert got["custom_flag"] == 7
    assert all(got[k] == v for k, v in THINKING_OFF_KWARGS.items())


def test_the_three_protocol_converters_all_honour_it():
    from freetoken.server.anthropic_api import (
        AnthropicMessagesRequest, convert_anthropic_to_genspec,
    )
    from freetoken.server.openai_api import ChatCompletionRequest, chat_request_to_genspec

    msgs = [{"role": "user", "content": "hi"}]
    # OpenAI: a client asking for high effort still gets the operator's "off"
    chat = ChatCompletionRequest(model="m", messages=msgs, reasoning_effort="high")
    ctk = chat_request_to_genspec(chat, {}, None, "off").chat_template_kwargs
    assert all(ctk[k] == v for k, v in THINKING_OFF_KWARGS.items())
    assert "reasoning_effort" not in ctk

    # Anthropic: an explicit thinking.type=enabled still gets "off"
    anth = AnthropicMessagesRequest(
        model="m", max_tokens=8, messages=msgs, thinking={"type": "enabled"}
    )
    ctk = convert_anthropic_to_genspec(anth, {}, thinking="off").chat_template_kwargs
    assert all(ctk[k] == v for k, v in THINKING_OFF_KWARGS.items())

    # and "auto" leaves that same request alone
    ctk = convert_anthropic_to_genspec(anth, {}, thinking="auto").chat_template_kwargs
    assert all(ctk[k] == v for k, v in THINKING_ON_KWARGS.items())


def test_the_flag_is_parsed_and_defaults_to_auto():
    from freetoken.server.args import ServerArgs

    assert ServerArgs.thinking == "auto"
