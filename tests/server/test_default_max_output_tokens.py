"""--max-output-tokens must reach every protocol, not just Responses.

The flag is documented as the default for requests that omit a limit, but
``resolve_sampling`` used to hardcode DEFAULT_MAX_OUTPUT_TOKENS, so only the Responses
API (which passes its own resolved default) honoured it. Upstream issue #395.
"""

from types import SimpleNamespace

import pytest
from freetoken.server.generation import DEFAULT_MAX_OUTPUT_TOKENS, resolve_sampling


def _resolve(max_tokens, default_max_tokens):
    return resolve_sampling(
        temperature=None, top_k=None, top_p=None, max_tokens=max_tokens,
        ignore_eos=False, model_sampling={}, default_max_tokens=default_max_tokens,
    )


def test_configured_default_fills_an_omitted_limit():
    assert _resolve(None, 4096).max_tokens == 4096


def test_an_explicit_request_limit_still_wins():
    assert _resolve(77, 4096).max_tokens == 77


def test_no_configuration_keeps_the_documented_constant():
    assert _resolve(None, None).max_tokens == DEFAULT_MAX_OUTPUT_TOKENS


@pytest.mark.parametrize("bad", [0, -1])
def test_a_non_positive_explicit_limit_is_still_a_client_error(bad):
    with pytest.raises(ValueError):
        _resolve(bad, 4096)


def test_openai_and_anthropic_converters_thread_the_default_through():
    from freetoken.server.anthropic_api import convert_anthropic_to_genspec
    from freetoken.server.anthropic_api import AnthropicMessagesRequest
    from freetoken.server.openai_api import ChatCompletionRequest, chat_request_to_genspec

    chat = ChatCompletionRequest(model="m", messages=[{"role": "user", "content": "hi"}])
    assert chat.max_tokens is None
    assert chat_request_to_genspec(chat, {}, 1234).sampling_params.max_tokens == 1234
    assert chat_request_to_genspec(chat, {}).sampling_params.max_tokens == DEFAULT_MAX_OUTPUT_TOKENS

    # Anthropic requires max_tokens in the schema, so the default only shows through when a
    # caller passes None explicitly; the wiring is what this asserts.
    anth = AnthropicMessagesRequest(
        model="m", max_tokens=55, messages=[{"role": "user", "content": "hi"}]
    )
    spec = convert_anthropic_to_genspec(anth, {}, default_max_tokens=1234)
    assert spec.sampling_params.max_tokens == 55


def test_state_helper_reads_the_server_config():
    from freetoken.server.openai_api import _default_max_tokens

    assert _default_max_tokens(SimpleNamespace(config=SimpleNamespace(max_output_tokens=999))) == 999
    assert _default_max_tokens(SimpleNamespace(config=SimpleNamespace())) is None
