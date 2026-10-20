from __future__ import annotations

import asyncio
import json
import os
import random
from types import SimpleNamespace

import pytest

from freetoken.message import TokenizeMsg, UserReply
from freetoken.server import openai_api
from freetoken.server.generation import generate_full
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


THINK, END_THINK, TOOL_CALL, END_TOOL_CALL = 248068, 248069, 248058, 248059
IM_END, ENDOFTEXT = 248046, 248044
_MARKERS = {
    "<think>": THINK,
    "</think>": END_THINK,
    "<tool_call>": TOOL_CALL,
    "</tool_call>": END_TOOL_CALL,
    "<|im_end|>": IM_END,
    "<|endoftext|>": ENDOFTEXT,
}


class QwenTokenizer:
    """Qwen3.8's marker ids; any other text encodes one id per character."""

    all_special_ids = [ENDOFTEXT, IM_END]

    def __init__(self, single_id_markers: bool = True) -> None:
        self.single = single_id_markers

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        if self.single and text in _MARKERS:
            return [_MARKERS[text]]
        return [ord(ch) for ch in text]

    def decode(self, ids: list[int]) -> str:
        names = {v: k for k, v in _MARKERS.items()}
        return "".join(names.get(i, chr(i) if i < 0x110000 else "") for i in ids)


def token_replies(
    tokens: list[str], *, finish_reason: str = "stop", texts: list[str] | None = None
) -> list[UserReply]:
    """One reply per sampled token, each carrying that token's entry (EOS appended for a
    "stop" finish). ``texts`` overrides the text each reply releases, the way the
    detokenizer holds back partial characters and stop prefixes."""
    tokens = tokens + (["<|im_end|>"] if finish_reason == "stop" else [])
    if texts is None:
        texts = ["" if t in ("<|im_end|>", "<|endoftext|>") else t for t in tokens]
    replies = [UserReply(uid=42, incremental_output="", finished=False, prompt_tokens_delta=3)]
    for i, (token, text) in enumerate(zip(tokens, texts, strict=True)):
        last = i == len(tokens) - 1
        replies.append(
            UserReply(
                uid=42,
                incremental_output=text,
                finished=last,
                finish_reason=finish_reason if last else None,
                completion_tokens_delta=1,
                logprobs=[entry(_MARKERS.get(token, 1000 + i), token, -0.01 * (i + 1))],
            )
        )
    return replies


class FakeState:
    def __init__(
        self,
        replies: list[UserReply],
        reasoning_parser: str | None = None,
        tool_call_parser: str = "llama3",
        tokenizer: QwenTokenizer | None = None,
    ) -> None:
        self.config = SimpleNamespace(
            model_path="/models/unit-model",
            served_model_name="unit-model",
            tool_call_parser=tool_call_parser,
            reasoning_parser=reasoning_parser,
        )
        self.replies = replies
        self.sent: TokenizeMsg | None = None
        if tokenizer is not None:
            self.frontend_tokenizer = lambda: SimpleNamespace(tokenizer=tokenizer)

    def new_user(self) -> int:
        return 42

    async def send_one(self, msg):
        self.sent = msg

    async def wait_for_ack(self, uid: int):
        assert uid == 42
        for reply in self.replies:
            yield reply


def reply(text: str, *, finished: bool = False, logprobs: list[dict] | None = None) -> UserReply:
    return UserReply(
        uid=42,
        incremental_output=text,
        finished=finished,
        prompt_tokens_delta=3 if not text else 0,
        completion_tokens_delta=1 if text else 0,
        cached_tokens=0,
        logprobs=logprobs,
    )


def entry(token_id: int, token: str, logprob: float, *, text_offset: int = 0) -> dict:
    return {
        "token_id": token_id,
        "token": token,
        "text_offset": text_offset,
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
            FakeState([reply("Hello", logprobs=[first]), reply("!", finished=True, logprobs=[second])]),
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
            FakeState([reply("Hello", finished=True, logprobs=[first])]),
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
            reply("Hello", logprobs=[first]),
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
    second = entry(2, "!", -0.2, text_offset=2)
    req = CompletionRequest(model="client-model", prompt="hello", logprobs=2, max_tokens=8)

    result = run(
        handle_completion(
            req,
            None,
            FakeState([reply("Hi", logprobs=[first]), reply("!", finished=True, logprobs=[second])]),
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
                    FakeState([reply("Hi", finished=True, logprobs=[first])]),
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
        reply("", logprobs=[entry(9008, "�", -0.1)]),
        reply("", logprobs=[entry(99, "�", -0.2)]),
        reply("🦀", logprobs=[entry(222, "�", -0.3)]),
        reply(" x", logprobs=[entry(1, " x", -0.4, text_offset=1)]),
        reply("", finished=True, logprobs=[entry(2, "<|im_end|>", -0.5, text_offset=3)]),
    ]
    return pieces


def test_completion_text_offset_counts_visible_text() -> None:
    req = CompletionRequest(model="client-model", prompt="hello", logprobs=1, max_tokens=8)
    result = run(handle_completion(req, None, FakeState(_crab_replies()), {}))
    logprobs = result["choices"][0]["logprobs"]
    assert result["choices"][0]["text"] == "🦀 x"
    # The split character's pieces all point at where it starts, not at
    # len("�") steps that drift past the real text.
    assert logprobs["text_offset"] == [0, 0, 0, 1, 3]


def _offset_replies(acks: list[tuple[str, list[tuple[str, int]]]]) -> list[UserReply]:
    """One reply per (text, [(token, text_offset), ...]) ack, entries in generation order."""
    replies, n = [], 0
    for i, (text, tokens) in enumerate(acks):
        entries = []
        for token, offset in tokens:
            entries.append(entry(n, token, -0.1, text_offset=offset))
            n += 1
        replies.append(reply(text, finished=i == len(acks) - 1, logprobs=entries or None))
    return replies


# "Answer: 4\nQuestion 2" under stop=["\n\n"]: the "\n" ack sends nothing (it could start
# the stop), and "Question" releases it along with its own text. "own" gives every ack
# its own entry; "held" is what the detokenizer sends, holding the "\n" entry until the
# reply that releases its text, which then carries two entries.
_HOLDBACK_ACKS = {
    "own": [("Answer", [("Answer", 0)]), (":", [(":", 6)]), (" ", [(" ", 7)]), ("4", [("4", 8)]),
            ("", [("\n", 9)]), ("\nQuestion", [("Question", 10)]), (" ", [(" ", 18)]),
            ("2", [("2", 19)])],
    "held": [("Answer", [("Answer", 0)]), (":", [(":", 6)]), (" ", [(" ", 7)]), ("4", [("4", 8)]),
             ("", []), ("\nQuestion", [("\n", 9), ("Question", 10)]), (" ", [(" ", 18)]),
             ("2", [("2", 19)])],
}


def _joined_stream_logprobs(replies: list[UserReply], **kwargs) -> dict[str, list]:
    req = CompletionRequest(model="client-model", prompt="hello", max_tokens=8, stream=True, **kwargs)
    events = parse_sse(run(_collect(stream_completion_chunks(42, req, FakeState(replies)))))
    joined: dict[str, list] = {"tokens": [], "token_logprobs": [], "top_logprobs": [], "text_offset": []}
    for event in events:
        if isinstance(event, dict) and event["choices"] and event["choices"][0]["logprobs"]:
            for key in joined:
                joined[key] += event["choices"][0]["logprobs"][key]
    return joined


@pytest.mark.parametrize("shape", ["own", "held"])
def test_completion_text_offset_survives_stop_prefix_holdback(shape: str) -> None:
    req = CompletionRequest(model="client-model", prompt="hello", logprobs=0, max_tokens=8, stop=["\n\n"])
    replies = _offset_replies(_HOLDBACK_ACKS[shape])
    choice = run(handle_completion(req, None, FakeState(replies), {}))["choices"][0]
    logprobs = choice["logprobs"]
    assert choice["text"] == "Answer: 4\nQuestion 2"
    assert logprobs["tokens"] == ["Answer", ":", " ", "4", "\n", "Question", " ", "2"]
    assert logprobs["text_offset"] == [0, 6, 7, 8, 9, 10, 18, 19]
    for token, offset in zip(logprobs["tokens"], logprobs["text_offset"], strict=True):
        assert choice["text"][offset:].startswith(token)
    assert _joined_stream_logprobs(replies, logprobs=0, stop=["\n\n"]) == logprobs


def test_completion_text_offset_clipped_to_stop_trimmed_text() -> None:
    # "a\n\n" under stop=["\n\n"]: the second "\n" completes the stop and both are trimmed.
    # Unmarked, their offsets are clipped so they cannot point past the returned text.
    replies = [
        reply("a", logprobs=[entry(1, "a", -0.1)]),
        reply("", logprobs=[entry(2, "\n", -0.2, text_offset=1)]),
        reply("", finished=True, logprobs=[entry(2, "\n", -0.3, text_offset=2)]),
    ]
    req = CompletionRequest(model="client-model", prompt="hello", logprobs=0, max_tokens=8, stop=["\n\n"])
    choice = run(handle_completion(req, None, FakeState(replies), {}))["choices"][0]
    assert choice["text"] == "a"
    assert choice["logprobs"]["text_offset"] == [0, 1, 1]
    assert _joined_stream_logprobs(replies, logprobs=0, stop=["\n\n"]) == choice["logprobs"]

    # As the detokenizer sends them: held with their text, then marked and dropped.
    trimmed = [
        {**entry(2, "\n", -0.2, text_offset=1), "stop_trimmed": True},
        {**entry(2, "\n", -0.3, text_offset=2), "stop_trimmed": True},
    ]
    marked = [replies[0], reply("", logprobs=None), reply("", finished=True, logprobs=trimmed)]
    choice = run(handle_completion(req, None, FakeState(marked), {}))["choices"][0]
    assert choice["logprobs"]["tokens"] == ["a"]
    assert choice["logprobs"]["text_offset"] == [0]
    assert _joined_stream_logprobs(marked, logprobs=0, stop=["\n\n"]) == choice["logprobs"]


# stop=["aab"] over "aaax": two "a"s stay held while the first is sent. "own": the held
# tokens' entries ride their own acks, so their offsets run past the text sent so far;
# that is where they land, not an overrun. "held": the detokenizer's shape, each entry
# waiting for its text, so the finished reply carries three.
_AHEAD_ACKS = {
    "own": [("", [("a", 0)]), ("", [("a", 1)]), ("a", [("a", 2)]), ("aax", [("x", 3)])],
    "held": [("", []), ("", []), ("a", [("a", 0)]), ("aax", [("a", 1), ("a", 2), ("x", 3)])],
}


@pytest.mark.parametrize("shape", ["own", "held"])
def test_completion_stream_text_offset_ahead_of_sent_text(shape: str) -> None:
    req = CompletionRequest(model="client-model", prompt="hello", logprobs=0, max_tokens=8, stop=["aab"])
    replies = _offset_replies(_AHEAD_ACKS[shape])
    choice = run(handle_completion(req, None, FakeState(replies), {}))["choices"][0]
    assert choice["text"] == "aaax"
    assert choice["logprobs"]["tokens"] == ["a", "a", "a", "x"]
    assert choice["logprobs"]["text_offset"] == [0, 1, 2, 3]
    assert _joined_stream_logprobs(replies, logprobs=0, stop=["aab"]) == choice["logprobs"]


def test_completion_text_offset_real_tokenizer_with_stops() -> None:
    # The real Qwen3.8 tokenizer through the real detokenizer: every token's offset must
    # point at its own text, stream and non-stream alike, with entries held for their text
    # and those of stop-trimmed tokens dropped.
    path = os.path.expanduser("~/models/Qwen3.8-Flash-Next-NVFP4-FTW-fp8")
    if not os.path.exists(os.path.join(path, "tokenizer.json")):
        pytest.skip("local Qwen3.8 tokenizer not present")
    from transformers import AutoTokenizer

    from freetoken.message import DetokenizeMsg
    from freetoken.tokenizer.detokenize import DetokenizeManager, build_logprobs_entry

    tokenizer = AutoTokenizer.from_pretrained(path)
    cases = [
        ("Answer: 4\nQuestion 2", "\n\n", None),
        ("Q: hi\nA: Question time 🦀 ok", "Question:", None),
        # the stop matches on the last token: the "\n\n" token it trimmed loses its entry
        ("Answer: 4\n\n", "\n\n", "\n\n"),
        # the stop cuts inside " hello": that entry stays, " world" (trimmed) does not
        ("Say hello world", "lo w", "lo w"),
    ]
    held = False
    for text, stop, matched in cases:
        ids = tokenizer.encode(text, add_special_tokens=False)
        manager = DetokenizeManager(tokenizer, frozenset({248046}))
        replies = []
        for i, tid in enumerate(ids):
            last = i == len(ids) - 1
            msg = DetokenizeMsg(
                uid=42, next_token=tid, finished=last, stop_strs=[stop],
                matched_stop=matched if last else None,
            )
            (out,), _, (released,) = manager.detokenize_with_logprobs(
                [msg], [build_logprobs_entry(tokenizer, tid, -0.1, [], [])]
            )
            held |= len(released) > 1
            replies.append(reply(out, finished=last, logprobs=released or None))
        visible = text[: text.find(matched)] if matched else text
        req = CompletionRequest(model="client-model", prompt="p", logprobs=0, max_tokens=64, stop=[stop])
        choice = run(handle_completion(req, None, FakeState(replies), {}))["choices"][0]
        assert choice["text"] == visible
        tokens, offsets = choice["logprobs"]["tokens"], choice["logprobs"]["text_offset"]
        assert len(tokens) == len(ids) - (1 if matched else 0)
        for token, offset in zip(tokens, offsets, strict=True):
            if "\ufffd" not in token:
                # a token's text sits at its offset; only one the stop cut ends the text early
                seg = visible[offset : offset + len(token)]
                assert token.startswith(seg), (text, token, offset)
                assert seg == token or offset + len(seg) == len(visible), (text, token, offset)
        assert offsets == sorted(offsets)
        assert _joined_stream_logprobs(replies, logprobs=0, stop=[stop]) == choice["logprobs"]
    assert held, "no reply carried more than one entry"


def test_completion_top_logprobs_key_collision_keeps_most_likely() -> None:
    # Partial-UTF-8 candidates all decode to U+FFFD and collide on the dict key.
    colliding = entry(9008, "�", -1.0)
    colliding["top"] = [
        {"token_id": 9008, "token": "�", "bytes": [240, 159], "logprob": -1.0},
        {"token_id": 9009, "token": "�", "bytes": [240, 160], "logprob": -2.0},
        {"token_id": 5, "token": "a", "bytes": [97], "logprob": -3.0},
    ]
    req = CompletionRequest(model="client-model", prompt="hello", logprobs=3, max_tokens=8)
    result = run(handle_completion(req, None, FakeState([reply("x", finished=True, logprobs=[colliding])]), {}))
    assert result["choices"][0]["logprobs"]["top_logprobs"] == [{"�": -1.0, "a": -3.0}]


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


def _stop_trimmed_replies() -> list[UserReply]:
    # "Hi", then a stop string "##" sampled as "#" "#": the detokenizer withheld both
    # entries with the text and releases them, marked, on the finished reply.
    trimmed = [{**entry(3, "#", -0.3), "stop_trimmed": True}, {**entry(3, "#", -0.4), "stop_trimmed": True}]
    return [
        reply("Hi", logprobs=[entry(1, "Hi", -0.1)]),
        reply("", logprobs=None),
        reply("", finished=True, logprobs=trimmed),
    ]


def test_completion_logprobs_drop_stop_trimmed_entries() -> None:
    req = CompletionRequest(model="client-model", prompt="hello", logprobs=1, max_tokens=8, stop=["##"])
    result = run(handle_completion(req, None, FakeState(_stop_trimmed_replies()), {}))
    assert result["choices"][0]["logprobs"]["tokens"] == ["Hi"]

    stream_req = req.model_copy(update={"stream": True})
    events = parse_sse(run(_collect(stream_completion_chunks(42, stream_req, FakeState(_stop_trimmed_replies())))))
    tokens = [
        token
        for event in events
        if isinstance(event, dict) and event["choices"] and event["choices"][0]["logprobs"]
        for token in event["choices"][0]["logprobs"]["tokens"]
    ]
    assert tokens == ["Hi"]


def test_chat_logprobs_drop_stop_trimmed_entries() -> None:
    req = chat_request(logprobs=True, stop=["##"])
    result = run(handle_chat_completion(req, None, FakeState(_stop_trimmed_replies()), {}))
    assert [e["token"] for e in result["choices"][0]["logprobs"]["content"]] == ["Hi"]

    stream_req = chat_request(logprobs=True, stop=["##"], stream=True)
    spec = chat_request_to_genspec(stream_req, {})
    events = parse_sse(
        run(_collect(stream_chat_completion_chunks(42, stream_req, FakeState(_stop_trimmed_replies()), spec)))
    )
    tokens = [
        e["token"]
        for event in events
        if isinstance(event, dict) and event["choices"] and event["choices"][0].get("logprobs")
        for e in event["choices"][0]["logprobs"]["content"]
    ]
    assert tokens == ["Hi"]


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
            CompletionRequest(model="client-model", prompt="hello", logprobs=33),
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


TOOL = {
    "type": "function",
    "function": {
        "name": "read",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
    },
}
CALL = [
    "<tool_call>", "\n", "<", "function", "=read", ">", "\n", "<", "parameter", "=path", ">",
    "\n", "a", ".py", "\n", "</", "parameter", ">", "\n", "</", "function", ">", "\n",
    "</tool_call>",
]


def qwen_state(replies: list[UserReply], **kwargs) -> FakeState:
    """The production stack: qwen3 reasoning parser and qwen3_coder tool parser."""
    return FakeState(
        replies,
        reasoning_parser="qwen3",
        tool_call_parser="qwen3_coder",
        tokenizer=kwargs.pop("tokenizer", QwenTokenizer()),
        **kwargs,
    )


def _chat_both(replies: list[UserReply], **req_kwargs) -> tuple[dict, list[dict]]:
    """The non-streaming response and the streamed chunk choices for the same replies."""
    req = chat_request(logprobs=True, **req_kwargs)
    result = run(handle_chat_completion(req, None, qwen_state(replies), {}))
    req = chat_request(logprobs=True, stream=True, **req_kwargs)
    spec = chat_request_to_genspec(req, {})
    chunks = stream_chat_completion_chunks(42, req, qwen_state(replies), spec)
    events = parse_sse(run(_collect(chunks)))
    choices = [e["choices"][0] for e in events if isinstance(e, dict) and e.get("choices")]
    return result, choices


def _streamed_tokens(choices: list[dict]) -> list[str]:
    return [e["token"] for c in choices if c.get("logprobs") for e in c["logprobs"]["content"]]


def _content_tokens(result: dict) -> list[str]:
    return [e["token"] for e in result["choices"][0]["logprobs"]["content"]]


def test_chat_logprobs_exclude_reasoning() -> None:
    replies = token_replies(["thought", " more", "</think>", "\n\n", "Hello", " world"])
    result, choices = _chat_both(replies)

    assert result["choices"][0]["message"]["reasoning_content"] == "thought more"
    assert result["choices"][0]["logprobs"] == {
        "content": result["choices"][0]["logprobs"]["content"],
        "refusal": None,
    }
    assert _content_tokens(result) == ["Hello", " world"]
    assert _streamed_tokens(choices) == ["Hello", " world"]
    for choice in choices:
        if choice["delta"].get("reasoning_content"):
            assert "logprobs" not in choice
        if choice.get("logprobs"):
            assert choice["logprobs"]["refusal"] is None


def test_chat_logprobs_exclude_tool_call_markup() -> None:
    replies = token_replies(["plan", "</think>", "\n\n", "Sure", "."] + CALL, finish_reason="stop")
    result, choices = _chat_both(replies, tools=[TOOL])

    assert result["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "read"
    assert _content_tokens(result) == ["Sure", "."]
    assert _streamed_tokens(choices) == ["Sure", "."]
    for choice in choices:
        if choice["delta"].get("tool_calls"):
            assert "logprobs" not in choice


def test_chat_logprobs_bare_function_call_is_not_content() -> None:
    # The qwen3_coder parser takes a declared <function=...> without its <tool_call>
    # wrapper as a call; its markup tokens are not content either.
    call = CALL[2:-2]
    replies = token_replies(["plan", "</think>", "Doing", " it", ".\n"] + call + ["\n", "ok"])
    result, choices = _chat_both(replies, tools=[TOOL])

    assert result["choices"][0]["message"]["tool_calls"]
    assert result["choices"][0]["message"]["content"] == "Doing it.\n\nok"
    assert _content_tokens(result) == ["Doing", " it", ".\n", "\n", "ok"]
    assert _streamed_tokens(choices) == _content_tokens(result)


def test_chat_logprobs_text_merged_into_a_bare_call_closer_is_content() -> None:
    # Qwen3.8 merges the closing ">" with the next character (">n"): the emitted "n"
    # sits at the end of "</function>n", not on the "n" inside "function".
    call = CALL[2:-3] + [">n"]
    replies = token_replies(["plan", "</think>"] + call)
    result, choices = _chat_both(replies, tools=[TOOL])

    assert result["choices"][0]["message"]["tool_calls"]
    assert result["choices"][0]["message"]["content"] == "n"
    assert _content_tokens(result) == [">n"]
    assert _streamed_tokens(choices) == _content_tokens(result)


def test_chat_logprobs_leading_space_merged_into_a_bare_call_opener_is_not_content() -> None:
    # The reasoning parser strips content's leading whitespace before the tool parser
    # sees it, so the space of " <" is never message text.
    call = [" <"] + CALL[3:-2]
    replies = token_replies(["x", "</think>"] + call)
    result, choices = _chat_both(replies, tools=[TOOL])

    assert result["choices"][0]["message"]["tool_calls"]
    assert not result["choices"][0]["message"]["content"]
    assert _content_tokens(result) == []
    assert _streamed_tokens(choices) == []


def _stop_cut_replies(tokens: list[str], visible: int) -> list[UserReply]:
    """A stop string matched inside the last token: only its first ``visible`` chars
    were released, and the detokenizer says so on its entry."""
    texts = tokens[:-1] + [tokens[-1][:visible]]
    replies = token_replies(tokens, finish_reason="length", texts=texts)
    replies[-1].finish_reason = "stop"
    replies[-1].logprobs = [{**replies[-1].logprobs[0], "stop_visible": visible}]
    return replies


def test_chat_logprobs_follow_the_visible_part_of_a_token_a_stop_string_cut() -> None:
    # Only the trailing space of " !" survives the stop "!", and content trimming drops it.
    result, choices = _chat_both(_stop_cut_replies(["x", "</think>", "Hi", " !"], 1), stop=["!"])
    assert result["choices"][0]["message"]["content"] == "Hi"
    assert _content_tokens(result) == ["Hi"]
    assert _streamed_tokens(choices) == ["Hi"]

    # A stop "_c" leaves "<tool" of <tool_call> as text, not a call opener.
    replies = _stop_cut_replies(["x", "</think>", "Use", " ", "<tool_call>"], 5)
    result, choices = _chat_both(replies, stop=["_c"], tools=[TOOL])
    assert result["choices"][0]["message"]["content"] == "Use <tool"
    assert _content_tokens(result) == ["Use", " ", "<tool_call>"]
    assert _streamed_tokens(choices) == _content_tokens(result)


def test_chat_logprobs_literal_markers_in_content_stay_content() -> None:
    # Only a marker that switches state is markup: a later </think> or a <tool_call>
    # named in prose is message text, and so is its entry.
    replies = token_replies(
        ["plan", "</think>", "Use", " </think>", " and", " <tool_call>", " tags", "."]
    )
    result, choices = _chat_both(replies, tools=[TOOL])

    content = result["choices"][0]["message"]["content"]
    assert content == "Use </think> and <tool_call> tags."
    assert "".join(_content_tokens(result)) == content
    assert _streamed_tokens(choices) == _content_tokens(result)


def test_chat_logprobs_ride_the_delta_that_releases_a_held_marker_prefix() -> None:
    # The tool parser holds the "<" of " <" as a possible <tool_call>: the token's
    # entry waits until all its text is out, then rides that delta, in order.
    replies = token_replies(["</think>", "a", " <", " b"])
    _, choices = _chat_both(replies, tools=[TOOL])

    content_chunks = [c for c in choices if c["delta"].get("content")]
    assert [c["delta"]["content"] for c in content_chunks] == ["a", " ", "< b"]
    assert [
        [e["token"] for e in c["logprobs"]["content"]] if c.get("logprobs") else None
        for c in content_chunks
    ] == [["a"], None, [" <", " b"]]


def test_chat_logprobs_trailing_whitespace_is_dropped() -> None:
    # Content after </think> is trimmed at both ends, and so are its whitespace entries;
    # whitespace between words is kept.
    replies = token_replies(["</think>", "\n\n", "Hi", "\n", "\n", "there", "\n\n"])
    result, choices = _chat_both(replies)

    assert result["choices"][0]["message"]["content"] == "Hi\n\nthere"
    assert _content_tokens(result) == ["Hi", "\n", "\n", "there"]
    assert _streamed_tokens(choices) == _content_tokens(result)


@pytest.mark.parametrize("seed", range(12))
def test_chat_logprobs_stream_matches_non_stream_for_any_chunking(seed: int) -> None:
    # The detokenizer can release a token's text on a later reply (a partial character,
    # a stop prefix), leaving replies with no text at all: the streamed entries, joined
    # across content chunks and the finish chunk, must still equal the one-shot list.
    tokens = (
        ["  ", "<think>", "plan", " it", "</think>", "\n\n", "Hi", " ", "<", "b", ">", "\n\n"]
        + CALL
        + ["\n", "Then", " more", "\n"]
    )
    full = "".join(tokens)
    rng = random.Random(seed)
    cuts, end, prev = [], 0, 0
    for token in tokens:
        end += len(token)
        prev = rng.randint(prev, end)
        cuts.append(prev)
    cuts[-1] = len(full)
    texts = [full[a:b] for a, b in zip([0] + cuts[:-1], cuts, strict=True)] + [""]
    replies = token_replies(tokens, texts=texts)

    result, choices = _chat_both(replies, tools=[TOOL])
    assert result["choices"][0]["message"]["content"] == "Hi <b>\n\n\nThen more"
    assert _content_tokens(result) == ["Hi", " ", "<", "b", ">", "\n\n", "\n", "Then", " more"]
    assert _streamed_tokens(choices) == _content_tokens(result)


def test_chat_logprobs_start_in_content_when_thinking_is_off() -> None:
    replies = token_replies(["  ", "Hi", " </think>", " there"])
    result, choices = _chat_both(replies, chat_template_kwargs={"enable_thinking": False})

    assert result["choices"][0]["message"]["content"] == "  Hi </think> there"
    assert _content_tokens(result) == ["  ", "Hi", " </think>", " there"]
    assert _streamed_tokens(choices) == _content_tokens(result)


def test_chat_logprobs_truncated_reasoning_has_no_content_entries() -> None:
    replies = token_replies(["still", " thinking"], finish_reason="length")
    result, choices = _chat_both(replies)

    assert result["choices"][0]["finish_reason"] == "length"
    assert result["choices"][0]["logprobs"] == {"content": [], "refusal": None}
    assert _streamed_tokens(choices) == []


def test_chat_logprobs_drop_the_eos_entry() -> None:
    # The end-of-turn token is not message text; non-streaming used to list it.
    replies = token_replies(["Hi", "!"])
    state = FakeState(replies, tokenizer=QwenTokenizer())
    result = run(handle_chat_completion(chat_request(logprobs=True), None, state, {}))
    assert _content_tokens(result) == ["Hi", "!"]


def test_chat_logprobs_unsupported_parser_families_still_400() -> None:
    def status(state: FakeState, **kwargs) -> int | None:
        resp = run(handle_chat_completion(chat_request(logprobs=True, **kwargs), None, state, {}))
        if isinstance(resp, dict):
            return None
        assert json.loads(resp.body)["error"]["param"] == "logprobs"
        return resp.status_code

    # A reasoning parser whose split the segmenter does not mirror.
    assert status(FakeState([], reasoning_parser="glm", tokenizer=QwenTokenizer())) == 400
    # qwen3 whose think markers are not single tokens, or with no tokenizer to tell.
    multi = QwenTokenizer(single_id_markers=False)
    assert status(FakeState([], reasoning_parser="qwen3", tokenizer=multi)) == 400
    assert status(FakeState([], reasoning_parser="qwen3")) == 400
    for stream in (False, True):
        assert status(FakeState([], reasoning_parser="qwen3"), stream=stream) == 400
    # A tool-call parser the segmenter cannot shadow...
    assert status(FakeState([], tokenizer=QwenTokenizer()), tools=[TOOL]) == 400
    # ...unless tool parsing is off for the request.
    ok = run(
        handle_chat_completion(
            chat_request(logprobs=True, tools=[TOOL], tool_choice="none"),
            None,
            FakeState(token_replies(["Hi"]), tokenizer=QwenTokenizer()),
            {},
        )
    )
    assert _content_tokens(ok) == ["Hi"]


def test_chat_logprobs_are_dropped_when_the_segmenter_cannot_build() -> None:
    # The generation-layer guard behind the API check: a caller that bypasses it still
    # never gets a reasoning token's entry on a visible delta.
    state = FakeState(
        token_replies(["thought", "</think>", "answer"]), reasoning_parser="qwen3"
    )
    req = chat_request(stream=True, logprobs=True, top_logprobs=2)
    spec = chat_request_to_genspec(req, {})
    events = parse_sse(run(_collect(stream_chat_completion_chunks(42, req, state, spec))))

    choices = [e["choices"][0] for e in events if isinstance(e, dict) and e.get("choices")]
    assert any(c["delta"].get("content") == "answer" for c in choices)
    assert _streamed_tokens(choices) == []


def test_generate_full_ignores_entries_when_logprobs_were_not_requested() -> None:
    # The protocol-neutral result carries entries only for a request that asked for them,
    # so an adapter reading it never sees unsegmented (reasoning) entries.
    state = FakeState(
        token_replies(["thought", "</think>", "answer"]), reasoning_parser="qwen3"
    )
    spec = chat_request_to_genspec(chat_request(), {})
    result = run(generate_full(42, spec, state))
    assert result.content == "answer"
    assert result.logprobs == []


async def _collect(iterator):
    return [chunk async for chunk in iterator]


def _completion(**kwargs) -> CompletionRequest:
    return CompletionRequest(**{"model": "client-model", "prompt": "hello", "max_tokens": 1, **kwargs})


def _one_token_state() -> FakeState:
    return FakeState([reply("A", finished=True, logprobs=[entry(32, "A", -0.1)])])


# Result-changing vLLM fields this server does not implement: a set value must be a 400,
# never a 200 computed as if the field were absent.
UNIMPLEMENTED_COMPLETION_FIELDS = [
    ("logprob_token_ids", [32]),
    ("prompt_logprobs", 0),
    ("guided_json", {"type": "object"}),
    ("guided_regex", "[AB]"),
    ("guided_choice", ["A", "B"]),
    ("guided_grammar", "root ::= \"A\""),
    ("structured_outputs", {"choice": ["A"]}),
    ("bad_words", ["B"]),
]


@pytest.mark.parametrize(("field", "value"), UNIMPLEMENTED_COMPLETION_FIELDS)
def test_completion_rejects_unimplemented_fields_when_set(field, value) -> None:
    resp = run(handle_completion(_completion(logprobs=2, **{field: value}), None, _one_token_state(), {}))
    assert getattr(resp, "status_code", 200) == 400
    assert field in json.loads(resp.body)["error"]["message"]


@pytest.mark.parametrize("value", [None, False, [], {}, ""])
@pytest.mark.parametrize("field", [f for f, _ in UNIMPLEMENTED_COMPLETION_FIELDS])
def test_completion_accepts_unimplemented_fields_left_unset(field, value) -> None:
    resp = run(handle_completion(_completion(**{field: value}), None, _one_token_state(), {}))
    assert resp["choices"][0]["text"] == "A"


@pytest.mark.parametrize("prompt", ["", [], [""], ["ok", ""], [[]]])
def test_completion_rejects_empty_prompts(prompt) -> None:
    state = _one_token_state()
    resp = run(handle_completion(_completion(prompt=prompt), None, state, {}))
    assert resp.status_code == 400
    assert state.sent is None  # nothing was queued


def test_completion_suffix_only_fim_prompt_is_not_empty() -> None:
    reason = openai_api._completion_unsupported_reason(_completion(prompt="", suffix="tail"))
    assert reason is None


@pytest.mark.parametrize(("num_prompts", "n", "status"), [(256, 1, 200), (257, 1, 400), (86, 3, 400)])
def test_completion_caps_prompts_times_n(num_prompts, n, status) -> None:
    replies = [reply("A", finished=True)]
    resp = run(handle_completion(_completion(prompt=["p"] * num_prompts, n=n), None, FakeState(replies), {}))
    assert getattr(resp, "status_code", 200) == status


def _too_long() -> UserReply:
    return UserReply(
        uid=42, incremental_output="", finished=True,
        error="prompt is too long: 9 tokens > 8 maximum", error_code="context_length_exceeded",
    )


def test_completion_passes_the_context_length_code_through() -> None:
    resp = run(handle_completion(_completion(), None, FakeState([_too_long()]), {}))
    error = json.loads(resp.body)["error"]
    assert resp.status_code == 400
    assert error["code"] == "context_length_exceeded"
    assert error["message"].startswith("prompt is too long: ")

    events = parse_sse(run(_collect(stream_completion_chunks(42, _completion(stream=True), FakeState([_too_long()])))))
    assert events[0]["error"]["code"] == "context_length_exceeded"


class VocabTokenizer(QwenTokenizer):
    def __len__(self) -> int:
        return 248077


def _status(resp) -> int:
    return getattr(resp, "status_code", 200)


def test_completion_logprobs_cap_is_32() -> None:
    assert _status(run(handle_completion(_completion(logprobs=32), None, _one_token_state(), {}))) == 200
    assert _status(run(handle_completion(_completion(logprobs=33), None, _one_token_state(), {}))) == 400


def test_completion_allowed_token_ids_reach_the_engine_sorted_and_unique() -> None:
    state = _one_token_state()
    req = _completion(allowed_token_ids=[33, 32, 33], logprobs=2, temperature=0)
    assert _status(run(handle_completion(req, None, state, {"temperature": 0.6}))) == 200
    assert state.sent.sampling_params.allowed_token_ids == [32, 33]


@pytest.mark.parametrize(
    ("ids", "fragment"),
    [([], "1 to 1024"), (list(range(1025)), "1 to 1024"), ([-1, 3], "non-negative"), ([248077], "vocabulary")],
)
def test_completion_allowed_token_ids_out_of_range(ids, fragment) -> None:
    state = FakeState([], tokenizer=VocabTokenizer())
    resp = run(handle_completion(_completion(allowed_token_ids=ids), None, state, {}))
    assert resp.status_code == 400 and fragment in json.loads(resp.body)["error"]["message"]
    assert state.sent is None


def test_completion_allowed_token_ids_at_the_vocab_edge_pass() -> None:
    state = FakeState([reply("A", finished=True)], tokenizer=VocabTokenizer())
    assert _status(run(handle_completion(_completion(allowed_token_ids=[0, 248076]), None, state, {}))) == 200


@pytest.mark.parametrize(
    ("extra", "model_sampling"),
    [
        ({"temperature": 0.7}, {}),
        ({}, {"temperature": 0.6}),  # the checkpoint default counts, not just the request
        ({"temperature": 0, "presence_penalty": 0.5}, {}),
        ({"temperature": 0, "frequency_penalty": 0.5}, {}),
        ({"temperature": 0, "repetition_penalty": 1.1}, {}),
        ({"temperature": 0}, {"repetition_penalty": 1.05}),
        ({"temperature": 0, "min_tokens": 1}, {}),
        ({"temperature": 0, "logit_bias": {"32": 1.0}}, {}),
    ],
)
def test_restricted_logprobs_refuse_what_they_would_not_fold_in(extra, model_sampling) -> None:
    req = _completion(allowed_token_ids=[32, 33], logprobs=2, **extra)
    assert _status(run(handle_completion(req, None, _one_token_state(), model_sampling))) == 400


def test_allowed_token_ids_without_logprobs_may_sample() -> None:
    req = _completion(allowed_token_ids=[32, 33], temperature=0.7, presence_penalty=0.5)
    assert _status(run(handle_completion(req, None, _one_token_state(), {}))) == 200


def test_non_finite_logprobs_are_clamped_to_the_vllm_floor() -> None:
    inf = float("inf")
    sampled = entry(32, "A", -inf)
    sampled["top"][1]["logprob"] = float("nan")
    state = FakeState([reply("A", finished=True, logprobs=[sampled])])
    result = run(handle_completion(_completion(logprobs=2), None, state, {}))
    lp = result["choices"][0]["logprobs"]
    assert lp["token_logprobs"] == [-9999.0] and lp["top_logprobs"] == [{"A": -9999.0, "x": -9999.0}]
    json.dumps(result, allow_nan=False)

    chat = run(
        handle_chat_completion(
            chat_request(logprobs=True, top_logprobs=2), None,
            FakeState([reply("A", finished=True, logprobs=[sampled])]), {},
        )
    )
    content = chat["choices"][0]["logprobs"]["content"][0]
    assert content["logprob"] == -9999.0
    json.dumps(chat, allow_nan=False)


def test_completion_return_tokens_as_token_ids_keys_by_id() -> None:
    # two ids that decode to the same text stay two entries
    a = entry(32, "\ufffd", -0.1)
    a["top"][1].update(token_id=900, token="\ufffd")
    req = _completion(logprobs=2, return_tokens_as_token_ids=True)
    result = run(handle_completion(req, None, FakeState([reply("A", finished=True, logprobs=[a])]), {}))
    expected = {
        "tokens": ["token_id:32"],
        "token_logprobs": [-0.1],
        "top_logprobs": [{"token_id:32": -0.1, "token_id:900": -1.1}],
        "text_offset": [0],
    }
    assert result["choices"][0]["logprobs"] == expected

    stream_req = _completion(logprobs=2, return_tokens_as_token_ids=True, stream=True)
    events = parse_sse(
        run(_collect(stream_completion_chunks(42, stream_req, FakeState([reply("A", finished=True, logprobs=[a])]))))
    )
    streamed = [e["choices"][0]["logprobs"] for e in events if isinstance(e, dict) and e["choices"][0]["logprobs"]]
    assert streamed == [expected]
