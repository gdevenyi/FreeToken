from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from freetoken.message import TokenizeMsg, UserReply
from freetoken.server.openai_api import (
    ChatCompletionRequest,
    CompletionRequest,
    handle_chat_completion,
    handle_completion,
    stream_completion_chunks,
)


def run(coro):
    return asyncio.run(coro)


class FakeState:
    def __init__(
        self,
        replies: list[UserReply],
        reasoning_parser: str | None = None,
        tool_call_parser: str = "llama3",
    ) -> None:
        self.config = SimpleNamespace(
            mm=SimpleNamespace(text_model_only=False, disabled_encoders=frozenset()),
            model_path="/models/unit-model",
            served_model_name="unit-model",
            tool_call_parser=tool_call_parser,
            reasoning_parser=reasoning_parser,
        )
        self.replies = replies
        self.sent: TokenizeMsg | None = None

    def new_user(self) -> int:
        return 42

    async def send_one(self, msg):
        self.sent = msg

    async def wait_for_ack(self, uid: int):
        assert uid == 42
        for reply in self.replies:
            yield reply


def reply(text: str, *, finished: bool = False, logprobs: dict | None = None) -> UserReply:
    return UserReply(
        uid=42,
        incremental_output=text,
        finished=finished,
        prompt_tokens_delta=3 if not text else 0,
        completion_tokens_delta=1 if text else 0,
        cached_tokens=0,
        logprobs=logprobs,
    )


def entry(token_id: int, token: str, logprob: float) -> dict:
    return {
        "token_id": token_id,
        "token": token,
        "bytes": list(token.encode("utf-8")),
        "logprob": logprob,
        "top": [
            {
                "token_id": token_id,
                "token": token,
                "bytes": list(token.encode("utf-8")),
                "logprob": logprob,
            },
            {
                "token_id": token_id + 1,
                "token": "x",
                "bytes": [120],
                "logprob": logprob - 1,
            },
        ],
    }


def chat_request(**kwargs) -> ChatCompletionRequest:
    payload = {
        "model": "client-model",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 8,
    }
    payload.update(kwargs)
    return ChatCompletionRequest(**payload)


def parse_sse(chunks: list[bytes]) -> list[dict | str]:
    events: list[dict | str] = []
    for chunk in chunks:
        for line in chunk.decode().splitlines():
            if line.startswith("data: "):
                data = line.removeprefix("data: ")
                events.append(data if data == "[DONE]" else json.loads(data))
    return events


async def _collect(iterator):
    return [chunk async for chunk in iterator]


# The completions fields this port implements, set on a chat request as a proxy or a chat UI
# can send them. extra="allow" keeps them readable as attributes, so a shared helper can leak them.
CHAT_EXTRAS = {
    "logprobs": True,
    "top_logprobs": 2,
    "allowed_token_ids": [32, 33],
    "return_tokens_as_token_ids": True,
    "prompt_logprobs": 1,
}
WEATHER_TOOL = {
    "type": "function",
    "function": {"name": "get_weather", "parameters": {"type": "object"}},
}


@pytest.mark.parametrize("tools", [False, True])
@pytest.mark.parametrize("stream", [False, True])
def test_chat_ignores_logprobs_and_vllm_extras(stream, tools) -> None:
    # Production runs both parsers; #224 refuses chat logprobs under them, 2bfdae17 refuses
    # the extras. Chat clients must keep getting a 200 with no logprobs, and the engine no request for them.
    state = FakeState(
        [reply(""), reply("Hi", finished=True)],
        reasoning_parser="qwen3",
        tool_call_parser="qwen3_coder",
    )
    req = chat_request(
        stream=stream, tools=[WEATHER_TOOL] if tools else None, **CHAT_EXTRAS
    )
    resp = run(handle_chat_completion(req, None, state, {}))

    assert getattr(resp, "status_code", 200) == 200
    if stream:
        events = parse_sse(run(_collect(resp.body_iterator)))
        chunks = [e for e in events if isinstance(e, dict)]
        assert chunks and not any("error" in e for e in chunks)
        choices = [c for e in chunks for c in e["choices"]]
    else:
        choices = resp["choices"]
    assert choices and not any("logprobs" in c for c in choices)
    sp = state.sent.sampling_params
    assert (
        getattr(sp, "logprobs", False) is False and getattr(sp, "top_logprobs", 0) == 0
    )
    assert getattr(sp, "allowed_token_ids", None) is None


def _status(resp) -> int:
    return getattr(resp, "status_code", 200)


def test_completion_logprobs_non_stream_and_stream() -> None:
    first = entry(1, "Hi", -0.1)
    second = entry(2, "!", -0.2)
    req = CompletionRequest(model="client-model", prompt="hello", logprobs=2, max_tokens=8)
    state = FakeState([reply("Hi", logprobs=first), reply("!", finished=True, logprobs=second)])

    result = run(handle_completion(req, None, state, {}))
    assert _status(result) == 200
    assert state.sent.sampling_params.logprobs is True
    assert state.sent.sampling_params.top_logprobs == 2
    assert result["choices"][0]["logprobs"] == {
        "tokens": ["Hi", "!"],
        "token_logprobs": [-0.1, -0.2],
        "top_logprobs": [
            {"Hi": -0.1, "x": -1.1},
            {"!": -0.2, "x": -1.2},
        ],
        "text_offset": [0, 2],
    }

    stream_req = CompletionRequest(model="client-model", prompt="hello", logprobs=2, max_tokens=8, stream=True)
    events = parse_sse(
        run(_collect(stream_completion_chunks(42, stream_req, FakeState([reply("Hi", finished=True, logprobs=first)]))))
    )
    chunk = next(event for event in events if isinstance(event, dict) and event["choices"][0]["text"])
    assert chunk["choices"][0]["logprobs"] == {
        "tokens": ["Hi"],
        "token_logprobs": [-0.1],
        "top_logprobs": [{"Hi": -0.1, "x": -1.1}],
        "text_offset": [0],
    }


def test_completion_without_logprobs_asks_the_engine_for_none() -> None:
    state = FakeState([reply("Hi", finished=True, logprobs=entry(1, "Hi", -0.1))])
    result = run(handle_completion(CompletionRequest(model="client-model", prompt="hello"), None, state, {}))
    assert result["choices"][0]["logprobs"] is None
    assert state.sent.sampling_params.logprobs is False


@pytest.mark.parametrize("extra", [{"logprobs": 33}, {"logprobs": -1}, {"echo": True, "logprobs": 1}])
def test_completion_logprobs_validation_errors(extra) -> None:
    state = FakeState([])
    resp = run(handle_completion(CompletionRequest(model="client-model", prompt="hello", **extra), None, state, {}))
    assert resp.status_code == 400
    assert state.sent is None
