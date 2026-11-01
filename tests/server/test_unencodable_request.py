"""CPU-only: a request field msgpack cannot pack gets a 4xx and leaves no bookkeeping (#681)."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from freetoken.message.tokenizer import BaseTokenizerMsg
from freetoken.server.anthropic_api import handle_anthropic_messages
from freetoken.server.anthropic_models import AnthropicMessagesRequest
from freetoken.server.api_models import ChatCompletionRequest, CompletionRequest
from freetoken.server.api_server import FrontendManager
from freetoken.server.openai_api import handle_chat_completion, handle_completion
from freetoken.server.responses_api import ResponsesRequest, handle_responses
from freetoken.utils.mp import ZmqAsyncPushQueue

HUGE = 2**64
MESSAGES = [{"role": "user", "content": "hi"}]


def _nested(depth: int, wrap):
    value = 1
    for _ in range(depth):
        value = wrap(value)
    return value


def _state() -> FrontendManager:
    # the real pack path, up to the socket
    push = ZmqAsyncPushQueue.__new__(ZmqAsyncPushQueue)
    push.encoder = BaseTokenizerMsg.encoder
    sent: list[bytes] = []

    async def send(event, copy=False):
        sent.append(event)

    push.socket = SimpleNamespace(send=send)
    state = FrontendManager(
        config=SimpleNamespace(
            served_model_name="m", tool_call_parser=None, reasoning_parser=None, max_seq_len=4096
        ),
        send_tokenizer=push,
        recv_tokenizer=None,
        initialized=True,
        maintenance_state="serving",
    )
    state.sent = sent
    return state


def _chat(**fields):
    return handle_chat_completion, ChatCompletionRequest(model="m", messages=MESSAGES, **fields)


def _completion(**fields):
    return handle_completion, CompletionRequest(model="m", prompt="hi", **fields)


def _messages(**fields):
    return handle_anthropic_messages, AnthropicMessagesRequest(
        model="m", max_tokens=8, messages=MESSAGES, **fields
    )


def _responses(**fields):
    return handle_responses, ResponsesRequest(model="m", input="hi", **fields)


@pytest.mark.parametrize(
    "build",
    [
        lambda: _chat(chat_template_kwargs={"x": HUGE}),
        lambda: _chat(chat_template_kwargs={"x": -(2**63) - 1}),
        lambda: _chat(chat_template_kwargs={"x": _nested(2000, lambda v: [v])}),
        lambda: _chat(chat_template_kwargs={"x": _nested(2000, lambda v: {"k": v})}),
        lambda: _chat(top_k=HUGE),
        lambda: _chat(max_tokens=HUGE),
        lambda: _chat(stop=["\ud800"]),
        lambda: _chat(logit_bias={str(HUGE): 1.0}),
        lambda: _chat(stop_token_ids=[HUGE]),
        lambda: _chat(top_k=HUGE, stream=True),
        lambda: _completion(max_tokens=HUGE),
        lambda: _completion(top_k=HUGE),
        lambda: _completion(logit_bias={str(HUGE): 1.0}),
        lambda: _completion(stop_token_ids=[HUGE]),
        lambda: _completion(top_k=HUGE, stream=True),
        lambda: _messages(top_k=HUGE),
        lambda: _messages(stop_sequences=["\ud800"]),
        lambda: _responses(top_k=HUGE),
    ],
)
def test_unencodable_field_is_a_400_with_no_bookkeeping_left(build):
    handler, req = build()
    state = _state()
    # a field that does pack would wait for an ack forever
    response = asyncio.run(asyncio.wait_for(handler(req, None, state, {}), 10))
    assert response.status_code == 400
    assert "request cannot be encoded" in json.loads(response.body)["error"]["message"]
    assert state.stats.active == 0
    assert state.ack_map == {} and state.event_map == {}
    assert state.sent == []


def test_a_large_id_that_packs_is_still_sent():
    # ids in [2**63, 2**64) fit the wire; the sampler and the scheduler ignore them
    handler, req = _chat(stop_token_ids=[2**63], logit_bias={str(2**63): 1.0}, stream=True)
    state = _state()
    response = asyncio.run(handler(req, None, state, {}))
    assert response.status_code == 200
    assert len(state.sent) == 1
    assert state.stats.active == 1
