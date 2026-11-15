from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from freetoken.message import TokenizeMsg, UserReply
from freetoken.server import openai_api
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
        tokenizer: object | None = None,
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
        if tokenizer is not None:
            manager = SimpleNamespace(tokenizer=tokenizer)
            self.frontend_tokenizer = lambda: manager

    def new_user(self) -> int:
        return 42

    async def send_one(self, msg):
        self.sent = msg

    async def wait_for_ack(self, uid: int):
        assert uid == 42
        for reply in self.replies:
            yield reply


def reply(
    text: str,
    *,
    finished: bool = False,
    logprobs: dict | None = None,
    text_offset: int = 0,
) -> UserReply:
    return UserReply(
        uid=42,
        incremental_output=text,
        finished=finished,
        prompt_tokens_delta=3 if not text else 0,
        completion_tokens_delta=1 if text else 0,
        cached_tokens=0,
        logprobs=logprobs,
        text_offset=text_offset,
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
    req = CompletionRequest(
        model="client-model", prompt="hello", logprobs=2, max_tokens=8
    )
    state = FakeState(
        [
            reply("Hi", logprobs=first),
            reply("!", finished=True, logprobs=second, text_offset=2),
        ]
    )

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

    stream_req = CompletionRequest(
        model="client-model", prompt="hello", logprobs=2, max_tokens=8, stream=True
    )
    events = parse_sse(
        run(
            _collect(
                stream_completion_chunks(
                    42,
                    stream_req,
                    FakeState([reply("Hi", finished=True, logprobs=first)]),
                )
            )
        )
    )
    chunk = next(
        event
        for event in events
        if isinstance(event, dict) and event["choices"][0]["text"]
    )
    assert chunk["choices"][0]["logprobs"] == {
        "tokens": ["Hi"],
        "token_logprobs": [-0.1],
        "top_logprobs": [{"Hi": -0.1, "x": -1.1}],
        "text_offset": [0],
    }


def test_completion_without_logprobs_asks_the_engine_for_none() -> None:
    state = FakeState([reply("Hi", finished=True, logprobs=entry(1, "Hi", -0.1))])
    result = run(
        handle_completion(
            CompletionRequest(model="client-model", prompt="hello"), None, state, {}
        )
    )
    assert result["choices"][0]["logprobs"] is None
    assert state.sent.sampling_params.logprobs is False


@pytest.mark.parametrize(
    "extra", [{"logprobs": 33}, {"logprobs": -1}, {"echo": True, "logprobs": 1}]
)
def test_completion_logprobs_validation_errors(extra) -> None:
    state = FakeState([])
    resp = run(
        handle_completion(
            CompletionRequest(model="client-model", prompt="hello", **extra),
            None,
            state,
            {},
        )
    )
    assert resp.status_code == 400
    assert state.sent is None


def _crab_replies() -> list[UserReply]:
    # 🦀 is three byte-level tokens: the first two acks carry no visible text yet.
    return [
        reply("", logprobs=entry(9008, "\ufffd", -0.1)),
        reply("", logprobs=entry(99, "\ufffd", -0.2)),
        reply("🦀", logprobs=entry(222, "\ufffd", -0.3)),
        reply(" x", logprobs=entry(1, " x", -0.4), text_offset=1),
        reply("", finished=True, logprobs=entry(2, "<|im_end|>", -0.5), text_offset=3),
    ]


def test_completion_text_offset_counts_visible_text() -> None:
    req = CompletionRequest(
        model="client-model", prompt="hello", logprobs=1, max_tokens=8
    )
    result = run(handle_completion(req, None, FakeState(_crab_replies()), {}))
    logprobs = result["choices"][0]["logprobs"]
    assert result["choices"][0]["text"] == "🦀 x"
    # The split character's pieces all point at where it starts, not at
    # len("\ufffd") steps that drift past the real text.
    assert logprobs["text_offset"] == [0, 0, 0, 1, 3]


def _holdback_replies() -> list[UserReply]:
    # "Answer: 4\nQuestion 2" under stop=["\n\n"]: the "\n" ack sends nothing (it could
    # start the stop), and "Question" releases it along with its own text.
    acks = [
        ("Answer", "Answer", 0),
        (":", ":", 6),
        (" ", " ", 7),
        ("4", "4", 8),
        ("\n", "", 9),
        ("Question", "\nQuestion", 10),
        (" ", " ", 18),
        ("2", "2", 19),
    ]
    return [
        reply(
            text,
            finished=i == len(acks) - 1,
            logprobs=entry(i, token, -0.1),
            text_offset=offset,
        )
        for i, (token, text, offset) in enumerate(acks)
    ]


def _joined_stream_logprobs(replies: list[UserReply], **kwargs) -> dict[str, list]:
    req = CompletionRequest(
        model="client-model", prompt="hello", max_tokens=8, stream=True, **kwargs
    )
    events = parse_sse(
        run(_collect(stream_completion_chunks(42, req, FakeState(replies))))
    )
    joined: dict[str, list] = {
        "tokens": [],
        "token_logprobs": [],
        "top_logprobs": [],
        "text_offset": [],
    }
    for event in events:
        if (
            isinstance(event, dict)
            and event["choices"]
            and event["choices"][0]["logprobs"]
        ):
            for key in joined:
                joined[key] += event["choices"][0]["logprobs"][key]
    return joined


def test_completion_text_offset_survives_stop_prefix_holdback() -> None:
    req = CompletionRequest(
        model="client-model", prompt="hello", logprobs=0, max_tokens=8, stop=["\n\n"]
    )
    choice = run(handle_completion(req, None, FakeState(_holdback_replies()), {}))[
        "choices"
    ][0]
    logprobs = choice["logprobs"]
    assert choice["text"] == "Answer: 4\nQuestion 2"
    assert logprobs["text_offset"] == [0, 6, 7, 8, 9, 10, 18, 19]
    for token, offset in zip(logprobs["tokens"], logprobs["text_offset"], strict=True):
        assert choice["text"][offset:].startswith(token)
    assert (
        _joined_stream_logprobs(_holdback_replies(), logprobs=0, stop=["\n\n"])
        == logprobs
    )


def test_completion_text_offset_clipped_to_stop_trimmed_text() -> None:
    # "a\n\n" under stop=["\n\n"]: the second "\n" completes the stop and both are trimmed,
    # so their offsets cannot point past the returned text.
    replies = [
        reply("a", logprobs=entry(1, "a", -0.1)),
        reply("", logprobs=entry(2, "\n", -0.2), text_offset=1),
        reply("", finished=True, logprobs=entry(2, "\n", -0.3), text_offset=2),
    ]
    req = CompletionRequest(
        model="client-model", prompt="hello", logprobs=0, max_tokens=8, stop=["\n\n"]
    )
    choice = run(handle_completion(req, None, FakeState(replies), {}))["choices"][0]
    assert choice["text"] == "a"
    assert choice["logprobs"]["text_offset"] == [0, 1, 1]
    assert (
        _joined_stream_logprobs(replies, logprobs=0, stop=["\n\n"])
        == choice["logprobs"]
    )


def test_completion_stream_text_offset_ahead_of_sent_text() -> None:
    # stop=["aab"] over "aaax": two "a"s stay held while the first is sent, so the held
    # tokens' offsets run past the text sent so far; that is where they land, not an overrun.
    def replies() -> list[UserReply]:
        acks = [("a", "", 0), ("a", "", 1), ("a", "a", 2), ("x", "aax", 3)]
        return [
            reply(
                text,
                finished=i == 3,
                logprobs=entry(i, token, -0.1),
                text_offset=offset,
            )
            for i, (token, text, offset) in enumerate(acks)
        ]

    req = CompletionRequest(
        model="client-model", prompt="hello", logprobs=0, max_tokens=8, stop=["aab"]
    )
    choice = run(handle_completion(req, None, FakeState(replies()), {}))["choices"][0]
    assert choice["text"] == "aaax"
    assert choice["logprobs"]["text_offset"] == [0, 1, 2, 3]
    assert (
        _joined_stream_logprobs(replies(), logprobs=0, stop=["aab"])
        == choice["logprobs"]
    )


def test_completion_top_logprobs_key_collision_keeps_most_likely() -> None:
    # Partial-UTF-8 candidates all decode to U+FFFD and collide on the dict key.
    colliding = entry(9008, "\ufffd", -1.0)
    colliding["top"] = [
        {"token_id": 9008, "token": "\ufffd", "bytes": [240, 159], "logprob": -1.0},
        {"token_id": 9009, "token": "\ufffd", "bytes": [240, 160], "logprob": -2.0},
        {"token_id": 5, "token": "a", "bytes": [97], "logprob": -3.0},
    ]
    req = CompletionRequest(
        model="client-model", prompt="hello", logprobs=3, max_tokens=8
    )
    result = run(
        handle_completion(
            req, None, FakeState([reply("x", finished=True, logprobs=colliding)]), {}
        )
    )
    assert result["choices"][0]["logprobs"]["top_logprobs"] == [
        {"\ufffd": -1.0, "a": -3.0}
    ]


def test_completion_stream_logprobs_match_non_stream() -> None:
    # Acks with no visible text must not lose their entries: they ride the next chunk
    # with text (or the finish chunk), so the joined stream equals the non-stream list.
    req = CompletionRequest(
        model="client-model", prompt="hello", logprobs=1, max_tokens=8
    )
    expected = run(handle_completion(req, None, FakeState(_crab_replies()), {}))
    joined = _joined_stream_logprobs(_crab_replies(), logprobs=1)
    assert joined == expected["choices"][0]["logprobs"]
    assert len(joined["tokens"]) == 5


def _completion(**kwargs) -> CompletionRequest:
    return CompletionRequest(
        **{"model": "client-model", "prompt": "hello", "max_tokens": 1, **kwargs}
    )


def _one_token_state() -> FakeState:
    return FakeState([reply("A", finished=True, logprobs=entry(32, "A", -0.1))])


# Result-changing vLLM fields this server does not implement: a set value must be a 400,
# never a 200 computed as if the field were absent.
UNIMPLEMENTED_COMPLETION_FIELDS = [
    ("return_tokens_as_token_ids", True),
    ("logprob_token_ids", [32]),
    ("prompt_logprobs", 0),
    ("guided_json", {"type": "object"}),
    ("guided_regex", "[AB]"),
    ("guided_choice", ["A", "B"]),
    ("guided_grammar", 'root ::= "A"'),
    ("structured_outputs", {"choice": ["A"]}),
    ("bad_words", ["B"]),
]


@pytest.mark.parametrize(("field", "value"), UNIMPLEMENTED_COMPLETION_FIELDS)
def test_completion_rejects_unimplemented_fields_when_set(field, value) -> None:
    state = _one_token_state()
    resp = run(
        handle_completion(_completion(logprobs=2, **{field: value}), None, state, {})
    )
    assert _status(resp) == 400
    assert field in json.loads(resp.body)["error"]["message"]
    assert state.sent is None


@pytest.mark.parametrize("value", [None, False, [], {}, ""])
def test_completion_accepts_unimplemented_fields_left_unset(value) -> None:
    # clients send the vLLM defaults explicitly; not truthiness: prompt_logprobs=0 is set
    resp = run(
        handle_completion(
            _completion(prompt_logprobs=value), None, _one_token_state(), {}
        )
    )
    assert resp["choices"][0]["text"] == "A"


@pytest.mark.parametrize("prompt", ["", [], [""], ["ok", ""], [[]]])
def test_completion_rejects_empty_prompts(prompt) -> None:
    state = _one_token_state()
    resp = run(handle_completion(_completion(prompt=prompt), None, state, {}))
    assert _status(resp) == 400
    assert state.sent is None  # nothing was queued


def test_completion_suffix_only_fim_prompt_is_not_empty() -> None:
    reason = openai_api._completion_unsupported_reason(
        _completion(prompt="", suffix="tail")
    )
    assert reason is None


@pytest.mark.parametrize(("num_prompts", "status"), [(256, 200), (257, 400)])
def test_completion_caps_the_prompt_count(num_prompts, status) -> None:
    state = FakeState([reply("A", finished=True)])
    resp = run(
        handle_completion(_completion(prompt=["p"] * num_prompts), None, state, {})
    )
    assert _status(resp) == status


class VocabTokenizer:
    """The frontend tokenizer's length: Qwen3.8's 248077 ids, added tokens included."""

    def __len__(self) -> int:
        return 248077


def test_completion_logprobs_cap_is_32() -> None:
    # AnyJev reads up to 26 letter labels; its vLLM recipe runs --max-logprobs 32
    assert (
        _status(
            run(
                handle_completion(
                    _completion(logprobs=32), None, _one_token_state(), {}
                )
            )
        )
        == 200
    )
    assert (
        _status(
            run(
                handle_completion(
                    _completion(logprobs=33), None, _one_token_state(), {}
                )
            )
        )
        == 400
    )


def test_completion_allowed_token_ids_reach_the_engine_sorted_and_unique() -> None:
    state = _one_token_state()
    req = _completion(allowed_token_ids=[33, 32, 33], logprobs=2, temperature=0)
    assert (
        _status(run(handle_completion(req, None, state, {"temperature": 0.6}))) == 200
    )
    assert state.sent.sampling_params.allowed_token_ids == [32, 33]


@pytest.mark.parametrize(
    ("ids", "fragment"),
    [
        ([], "1 to 1024"),
        (list(range(1025)), "1 to 1024"),
        ([-1, 3], "non-negative"),
        ([248077], "vocabulary"),
    ],
)
def test_completion_allowed_token_ids_out_of_range(ids, fragment) -> None:
    state = FakeState([], tokenizer=VocabTokenizer())
    resp = run(handle_completion(_completion(allowed_token_ids=ids), None, state, {}))
    assert (
        _status(resp) == 400 and fragment in json.loads(resp.body)["error"]["message"]
    )
    assert state.sent is None


def test_completion_allowed_token_ids_at_the_vocab_edge_pass() -> None:
    state = FakeState([reply("A", finished=True)], tokenizer=VocabTokenizer())
    assert (
        _status(
            run(
                handle_completion(
                    _completion(allowed_token_ids=[0, 248076]), None, state, {}
                )
            )
        )
        == 200
    )


@pytest.mark.parametrize(
    ("extra", "model_sampling"),
    [
        ({"temperature": 0.7}, {}),
        (
            {},
            {"temperature": 0.6},
        ),  # the checkpoint default counts, not just the request
        ({"temperature": 0, "presence_penalty": 0.5}, {}),
        ({"temperature": 0, "frequency_penalty": 0.5}, {}),
        ({"temperature": 0, "repetition_penalty": 1.1}, {}),
        ({"temperature": 0}, {"repetition_penalty": 1.05}),
        ({"temperature": 0, "min_tokens": 1}, {}),
        ({"temperature": 0, "logit_bias": {"32": 1.0}}, {}),
    ],
)
def test_restricted_logprobs_refuse_what_they_would_not_fold_in(
    extra, model_sampling
) -> None:
    # vLLM's processed_logprobs would fold these in; the raw logits here cannot, so a 400
    # beats numbers computed as if they were absent (the checkpoint's temperature is 1.0).
    req = _completion(allowed_token_ids=[32, 33], logprobs=2, **extra)
    assert (
        _status(run(handle_completion(req, None, _one_token_state(), model_sampling)))
        == 400
    )


def test_allowed_token_ids_without_logprobs_may_sample() -> None:
    req = _completion(allowed_token_ids=[32, 33], temperature=0.7, presence_penalty=0.5)
    assert _status(run(handle_completion(req, None, _one_token_state(), {}))) == 200


def test_non_finite_logprobs_are_clamped_to_the_vllm_floor() -> None:
    sampled = entry(32, "A", float("-inf"))
    sampled["top"][1]["logprob"] = float("nan")
    state = FakeState([reply("A", finished=True, logprobs=sampled)])
    result = run(handle_completion(_completion(logprobs=2), None, state, {}))
    lp = result["choices"][0]["logprobs"]
    assert lp["token_logprobs"] == [-9999.0] and lp["top_logprobs"] == [
        {"A": -9999.0, "x": -9999.0}
    ]
    json.dumps(result, allow_nan=False)
