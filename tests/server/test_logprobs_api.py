from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from freetoken.message import TokenizeMsg, UserReply
from freetoken.server.openai_api import (
    ChatCompletionRequest,
    CompletionRequest,
    chat_request_to_genspec,
    handle_chat_completion,
    handle_completion,
    stream_chat_completion_chunks,
    stream_completion_chunks,
)


def run(coro):
    return asyncio.run(coro)


class FakeState:
    def __init__(self, replies: list[UserReply], reasoning_parser: str | None = None) -> None:
        self.config = SimpleNamespace(
            model_path="/models/unit-model",
            served_model_name="unit-model",
            tool_call_parser="llama3",
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


def test_chat_non_stream_logprobs() -> None:
    first = entry(1, "Hello", -0.1)
    second = entry(2, "!", -0.2)
    result = run(
        handle_chat_completion(
            chat_request(logprobs=True, top_logprobs=2),
            None,
            FakeState([reply("Hello", logprobs=first), reply("!", finished=True, logprobs=second)]),
            {},
        )
    )

    content = result["choices"][0]["logprobs"]["content"]
    assert content == [
        {
            "token": "Hello",
            "logprob": -0.1,
            "bytes": [72, 101, 108, 108, 111],
            "top_logprobs": [
                {"token": "Hello", "logprob": -0.1, "bytes": [72, 101, 108, 108, 111]},
                {"token": "x", "logprob": -1.1, "bytes": [120]},
            ],
        },
        {
            "token": "!",
            "logprob": -0.2,
            "bytes": [33],
            "top_logprobs": [
                {"token": "!", "logprob": -0.2, "bytes": [33]},
                {"token": "x", "logprob": -1.2, "bytes": [120]},
            ],
        },
    ]

    without_logprobs = run(
        handle_chat_completion(
            chat_request(),
            None,
            FakeState([reply("Hello", finished=True, logprobs=first)]),
            {},
        )
    )
    # OpenAI sends the key with null rather than leaving it out
    assert "logprobs" in without_logprobs["choices"][0]
    assert without_logprobs["choices"][0]["logprobs"] is None


def test_chat_stream_logprobs_follow_content_deltas() -> None:
    first = entry(1, "Hello", -0.1)
    state = FakeState(
        [
            reply("Hello", logprobs=first),
            reply(" world", finished=True),
        ]
    )
    req = chat_request(stream=True, logprobs=True, top_logprobs=2)
    spec = chat_request_to_genspec(req, {})
    events = parse_sse(run(_collect(stream_chat_completion_chunks(42, req, state, spec))))

    content_choices = [
        event["choices"][0]
        for event in events
        if isinstance(event, dict)
        and event["choices"]
        and event["choices"][0]["delta"].get("content")
    ]
    assert content_choices[0]["logprobs"]["content"][0]["token"] == "Hello"
    assert "logprobs" not in content_choices[1]


def test_completion_logprobs_non_stream_and_stream() -> None:
    first = entry(1, "Hi", -0.1)
    second = entry(2, "!", -0.2)
    req = CompletionRequest(model="client-model", prompt="hello", logprobs=2, max_tokens=8)

    result = run(
        handle_completion(
            req,
            None,
            FakeState([reply("Hi", logprobs=first), reply("!", finished=True, logprobs=second)]),
            {},
        )
    )
    assert result["choices"][0]["logprobs"] == {
        "tokens": ["Hi", "!"],
        "token_logprobs": [-0.1, -0.2],
        "top_logprobs": [
            {"Hi": -0.1, "x": -1.1},
            {"!": -0.2, "x": -1.2},
        ],
        "text_offset": [0, 2],
    }

    events = parse_sse(
        run(
            _collect(
                stream_completion_chunks(
                    42,
                    CompletionRequest(
                        model="client-model",
                        prompt="hello",
                        logprobs=2,
                        max_tokens=8,
                        stream=True,
                    ),
                    FakeState([reply("Hi", finished=True, logprobs=first)]),
                )
            )
        )
    )
    chunk = next(event for event in events if isinstance(event, dict) and event["choices"][0]["text"])
    assert chunk["choices"][0]["logprobs"] == {
        "tokens": ["Hi"],
        "token_logprobs": [-0.1],
        "top_logprobs": [{"Hi": -0.1, "x": -1.1}],
        "text_offset": [0],
    }


def _crab_replies() -> list[UserReply]:
    # 🦀 is three byte-level tokens: the first two acks carry no visible text yet.
    pieces = [
        reply("", logprobs=entry(9008, "\ufffd", -0.1)),
        reply("", logprobs=entry(99, "\ufffd", -0.2)),
        reply("🦀", logprobs=entry(222, "\ufffd", -0.3)),
        reply(" x", logprobs=entry(1, " x", -0.4)),
        reply("", finished=True, logprobs=entry(2, "<|im_end|>", -0.5)),
    ]
    return pieces


def test_completion_text_offset_counts_visible_text() -> None:
    req = CompletionRequest(model="client-model", prompt="hello", logprobs=1, max_tokens=8)
    result = run(handle_completion(req, None, FakeState(_crab_replies()), {}))
    logprobs = result["choices"][0]["logprobs"]
    assert result["choices"][0]["text"] == "🦀 x"
    # The split character's pieces all point at where it starts, not at
    # len("\ufffd") steps that drift past the real text.
    assert logprobs["text_offset"] == [0, 0, 0, 1, 3]


def test_completion_top_logprobs_key_collision_keeps_most_likely() -> None:
    # Partial-UTF-8 candidates all decode to U+FFFD and collide on the dict key.
    colliding = entry(9008, "\ufffd", -1.0)
    colliding["top"] = [
        {"token_id": 9008, "token": "\ufffd", "bytes": [240, 159], "logprob": -1.0},
        {"token_id": 9009, "token": "\ufffd", "bytes": [240, 160], "logprob": -2.0},
        {"token_id": 5, "token": "a", "bytes": [97], "logprob": -3.0},
    ]
    req = CompletionRequest(model="client-model", prompt="hello", logprobs=3, max_tokens=8)
    result = run(handle_completion(req, None, FakeState([reply("x", finished=True, logprobs=colliding)]), {}))
    assert result["choices"][0]["logprobs"]["top_logprobs"] == [{"\ufffd": -1.0, "a": -3.0}]


def test_completion_stream_logprobs_match_non_stream() -> None:
    # Acks with no visible text must not lose their entries: they ride the next chunk
    # with text (or the finish chunk), so the joined stream equals the non-stream list.
    req = CompletionRequest(model="client-model", prompt="hello", logprobs=1, max_tokens=8)
    expected = run(handle_completion(req, None, FakeState(_crab_replies()), {}))
    stream_req = CompletionRequest(model="client-model", prompt="hello", logprobs=1, max_tokens=8, stream=True)
    events = parse_sse(run(_collect(stream_completion_chunks(42, stream_req, FakeState(_crab_replies())))))
    joined: dict[str, list] = {"tokens": [], "token_logprobs": [], "top_logprobs": [], "text_offset": []}
    for event in events:
        if isinstance(event, dict) and event["choices"] and event["choices"][0]["logprobs"]:
            for key in joined:
                joined[key] += event["choices"][0]["logprobs"][key]
    assert joined == expected["choices"][0]["logprobs"]
    assert len(joined["tokens"]) == 5


def test_logprobs_validation_errors() -> None:
    chat_top_out_of_range = run(
        handle_chat_completion(chat_request(logprobs=True, top_logprobs=25), None, FakeState([]), {})
    )
    assert chat_top_out_of_range.status_code == 400

    chat_missing_flag = run(
        handle_chat_completion(chat_request(top_logprobs=1), None, FakeState([]), {})
    )
    assert chat_missing_flag.status_code == 400

    completion_out_of_range = run(
        handle_completion(
            CompletionRequest(model="client-model", prompt="hello", logprobs=7),
            None,
            FakeState([]),
            {},
        )
    )
    assert completion_out_of_range.status_code == 400

    completion_echo = run(
        handle_completion(
            CompletionRequest(model="client-model", prompt="hello", echo=True, logprobs=1),
            None,
            FakeState([]),
            {},
        )
    )
    assert completion_echo.status_code == 400


def test_chat_logprobs_fail_closed_under_semantic_parsing() -> None:
    # A server-side reasoning parser hides <think> tokens from message content, so
    # logprob entries cannot be aligned with it: reject up front, stream and
    # non-stream alike, before any engine work is submitted.
    with_parser = FakeState([], reasoning_parser="qwen3")
    for stream in (False, True):
        resp = run(
            handle_chat_completion(chat_request(logprobs=True, stream=stream), None, with_parser, {})
        )
        assert resp.status_code == 400
        assert json.loads(resp.body)["error"]["param"] == "logprobs"

    # Tool parsing consumes tokens into tool_calls -- same conflict.
    tool = {"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}
    tools_resp = run(
        handle_chat_completion(chat_request(logprobs=True, tools=[tool]), None, FakeState([]), {})
    )
    assert tools_resp.status_code == 400
    assert json.loads(tools_resp.body)["error"]["param"] == "logprobs"

    # tool_choice="none" disables parsing, so logprobs stay available.
    ok = run(
        handle_chat_completion(
            chat_request(logprobs=True, tools=[tool], tool_choice="none"),
            None,
            FakeState([reply("Hi", finished=True, logprobs=entry(1, "Hi", -0.1))]),
            {},
        )
    )
    assert ok["choices"][0]["logprobs"]["content"][0]["token"] == "Hi"


def test_reasoning_logprob_is_not_carried_to_content_delta() -> None:
    # The generation-layer guard behind the API fail-close: even when a caller
    # bypasses request validation, a hidden reasoning token's entry is dropped,
    # never attached to a later visible delta (where its token string would leak).
    state = FakeState(
        [
            reply("<think>thought</think>", logprobs=entry(1, "thought", -0.1)),
            reply("answer", finished=True),
        ],
        reasoning_parser="qwen3",
    )
    req = chat_request(stream=True, logprobs=True, top_logprobs=2)
    spec = chat_request_to_genspec(req, {})
    events = parse_sse(run(_collect(stream_chat_completion_chunks(42, req, state, spec))))

    content_choice = next(
        event["choices"][0]
        for event in events
        if isinstance(event, dict)
        and event["choices"]
        and event["choices"][0]["delta"].get("content") == "answer"
    )
    assert "logprobs" not in content_choice


async def _collect(iterator):
    return [chunk async for chunk in iterator]
