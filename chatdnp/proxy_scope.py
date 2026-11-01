"""Option-by-option check of the LiteLLM proxy in front of FreeToken.

OpenAI side: /v1/models, /v1/chat/completions, /v1/completions (model Qwen3.8-Flash-Next).
Anthropic side: /v1/messages, /v1/messages/count_tokens (model Qwen3.8-Flash-Next-anthropic).
The proxy URL comes from $CHATDNP_BASE. The key comes from $CHATDNP_API_KEY and is never printed.

usage: python3 proxy_scope.py [name-filter]
"""

import base64
import json
import os
import struct
import sys
import time
import urllib.error
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor

BASE = os.environ["CHATDNP_BASE"].rstrip("/")
KEY = os.environ["CHATDNP_API_KEY"]
OA = "Qwen3.8-Flash-Next"
AN = "Qwen3.8-Flash-Next-anthropic"
NOTHINK = {"chat_template_kwargs": {"enable_thinking": False}}
AHDR = {"anthropic-version": "2023-06-01"}


def post(path, body, headers=None, key=KEY, stream=False, timeout=300):
    h = {"Content-Type": "application/json"}
    if key:
        h["Authorization"] = f"Bearer {key}"
    h.update(headers or {})
    req = urllib.request.Request(BASE + path, json.dumps(body).encode(), h)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode()
            status = r.status
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:400]
    if stream:
        return status, raw
    return status, json.loads(raw)


def get(path):
    req = urllib.request.Request(BASE + path, headers={"Authorization": f"Bearer {KEY}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.status, json.loads(r.read().decode())


def sse(raw):
    """[(event, data-dict)] from an SSE body; event is None for OpenAI-style streams."""
    out, ev = [], None
    for line in raw.splitlines():
        if line.startswith("event: "):
            ev = line[7:]
        elif line.startswith("data: "):
            d = line[6:]
            out.append((ev, None if d == "[DONE]" else json.loads(d)))
    return out


def png(rgb, size=64):
    row = b"\x00" + bytes(rgb) * size
    raw = zlib.compress(row * size)

    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", raw) + chunk(b"IEND", b"")


RED_B64 = base64.b64encode(png((220, 20, 20))).decode()
WEATHER_OA = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for one city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}
TIME_OA = {
    "type": "function",
    "function": {
        "name": "get_time",
        "description": "Get the local time in one city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}
WEATHER_AN = {
    "name": "get_weather",
    "description": "Get the current weather for one city.",
    "input_schema": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
}
TIME_AN = {
    "name": "get_time",
    "description": "Get the local time in one city.",
    "input_schema": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
}


def chat(body, **kw):
    return post("/v1/chat/completions", {"model": OA, **body}, **kw)


def msg(body, **kw):
    return post("/v1/messages", {"model": AN, **body}, headers=AHDR, **kw)


def m0(d):
    return d["choices"][0]["message"]


def texts(d):
    return "".join(c.get("text", "") for c in d["content"] if c["type"] == "text")


def user(t):
    return [{"role": "user", "content": t}]


TESTS = {}


def test(name):
    def deco(fn):
        TESTS[name] = fn
        return fn

    return deco


# ---------------------------------------------------------------- OpenAI: models, chat
@test("oa.models")
def _():
    s, d = get("/v1/models")
    ids = [m["id"] for m in d["data"]]
    return s == 200 and OA in ids, ids


@test("oa.chat.basic")
def _():
    s, d = chat({"messages": user("Say OK."), "max_tokens": 50, **NOTHINK})
    return s == 200 and "OK" in (m0(d)["content"] or ""), m0(d)["content"]


@test("oa.chat.default_thinking")
def _():
    s, d = chat({"messages": user("What is 12*12? Answer with the number."), "max_tokens": 2000})
    r = m0(d).get("reasoning_content") or ""
    return s == 200 and len(r) > 0 and "144" in (
        m0(d)["content"] or ""
    ), f"reasoning {len(r)} chars, content {m0(d)['content']!r}"


@test("oa.chat.thinking_off")
def _():
    s, d = chat(
        {
            "messages": user("What is 12*12? Answer with the number."),
            "max_tokens": 200,
            **NOTHINK,
        }
    )
    r = m0(d).get("reasoning_content") or ""
    return s == 200 and not r and "144" in m0(d)["content"], f"reasoning {len(r)} chars, content {m0(d)['content']!r}"


@test("oa.chat.reasoning_effort_levels")
def _():
    # the checkpoint knows low / medium / xhigh (default); FreeToken maps "high" to xhigh
    q = user("How many prime numbers are there below 60? Answer with the number.")
    out = {e: chat({"messages": q, "max_tokens": 8000, "reasoning_effort": e})[1] for e in ("low", "medium", "xhigh")}
    toks = {e: d["usage"]["completion_tokens"] for e, d in out.items()}
    return all("17" in m0(d)["content"] for d in out.values()), f"completion tokens {toks}"


@test("oa.chat.system_and_multiturn")
def _():
    s, d = chat(
        {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a pirate. Always end your reply with the word ARR.",
                },
                {"role": "user", "content": "My name is Gabriel."},
                {"role": "assistant", "content": "Ahoy Gabriel! ARR"},
                {"role": "user", "content": "What is my name? One short sentence."},
            ],
            "max_tokens": 80,
            **NOTHINK,
        }
    )
    c = m0(d)["content"]
    return s == 200 and "Gabriel" in c and "ARR" in c.upper(), c


@test("oa.chat.stream_usage")
def _():
    s, raw = chat(
        {
            "messages": user("Count 1 to 5."),
            "max_tokens": 60,
            "stream": True,
            "stream_options": {"include_usage": True},
            **NOTHINK,
        },
        stream=True,
    )
    ev = sse(raw)
    text = "".join((d["choices"][0]["delta"].get("content") or "") for _, d in ev if d and d.get("choices"))
    usage = [d["usage"] for _, d in ev if d and d.get("usage")]
    done = ev and ev[-1][1] is None
    return (
        s == 200 and "5" in text and usage and done,
        f"{len(ev)} chunks, usage {usage[-1] if usage else None}, [DONE] {done}",
    )


@test("oa.chat.stream_reasoning")
def _():
    s, raw = chat(
        {"messages": user("What is 7*8?"), "max_tokens": 1500, "stream": True},
        stream=True,
    )
    ev = [d for _, d in sse(raw) if d and d.get("choices")]
    r = "".join((d["choices"][0]["delta"].get("reasoning_content") or "") for d in ev)
    t = "".join((d["choices"][0]["delta"].get("content") or "") for d in ev)
    return (
        s == 200 and r and "56" in t,
        f"reasoning {len(r)} chars, content {t.strip()[:40]!r}",
    )


@test("oa.chat.max_tokens_length")
def _():
    s, d = chat(
        {
            "messages": user("Write a long story about a ship."),
            "max_tokens": 10,
            **NOTHINK,
        }
    )
    return s == 200 and d["choices"][0]["finish_reason"] == "length" and d["usage"]["completion_tokens"] == 10, (
        d["choices"][0]["finish_reason"],
        d["usage"]["completion_tokens"],
    )


@test("oa.chat.max_completion_tokens")
def _():
    s, d = chat(
        {
            "messages": user("Write a long story about a ship."),
            "max_completion_tokens": 12,
            **NOTHINK,
        }
    )
    return s == 200 and d["usage"]["completion_tokens"] <= 12, (
        d["choices"][0]["finish_reason"],
        d["usage"]["completion_tokens"],
    )


@test("oa.chat.stop")
def _():
    s, d = chat(
        {
            "messages": user("List the numbers one to ten in words, comma separated."),
            "max_tokens": 80,
            "stop": ["five"],
            **NOTHINK,
        }
    )
    c = m0(d)["content"]
    return s == 200 and "five" not in c and "six" not in c and d["choices"][0]["finish_reason"] == "stop", c


@test("oa.chat.temperature0_deterministic")
def _():
    b = {
        "messages": user("Write one sentence about the sea."),
        "max_tokens": 40,
        "temperature": 0,
        **NOTHINK,
    }
    a, c = m0(chat(b)[1])["content"], m0(chat(b)[1])["content"]
    return a == c, a


@test("oa.chat.seed_reproducible")
def _():
    b = {
        "messages": user("Invent a name for a cat."),
        "max_tokens": 20,
        "temperature": 1.0,
        "seed": 1234,
        **NOTHINK,
    }
    a, c = m0(chat(b)[1])["content"], m0(chat(b)[1])["content"]
    return a == c, f"{a!r} vs {c!r}"


@test("oa.chat.sampling_params")
def _():
    s, d = chat(
        {
            "messages": user("Say hello."),
            "max_tokens": 30,
            "temperature": 0.7,
            "top_p": 0.9,
            "presence_penalty": 0.5,
            "frequency_penalty": 0.5,
            "top_k": 20,
            "min_p": 0.05,
            "repetition_penalty": 1.05,
            **NOTHINK,
        }
    )
    return s == 200 and m0(d)["content"], m0(d)["content"] if s == 200 else d


@test("oa.chat.n2")
def _():
    s, d = chat(
        {
            "messages": user("Invent a name for a dog."),
            "max_tokens": 20,
            "n": 2,
            "temperature": 1.0,
            **NOTHINK,
        }
    )
    return s == 200 and len(d["choices"]) == 2, [c["message"]["content"] for c in d["choices"]] if s == 200 else d


@test("oa.chat.logprobs")
def _():
    s, d = chat(
        {
            "messages": user("Say yes."),
            "max_tokens": 5,
            "logprobs": True,
            "top_logprobs": 3,
            **NOTHINK,
        }
    )
    lp = (d["choices"][0].get("logprobs") or {}).get("content") if s == 200 else None
    return bool(lp) and len(lp[0].get("top_logprobs", [])) == 3, lp[0] if lp else d


@test("oa.chat.json_object")
def _():
    s, d = chat(
        {
            "messages": user("Give a JSON object with keys city and country for Montreal."),
            "max_tokens": 80,
            "response_format": {"type": "json_object"},
            **NOTHINK,
        }
    )
    o = json.loads(m0(d)["content"])
    return isinstance(o, dict) and "city" in o, o


@test("oa.chat.json_schema")
def _():
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name", "age"],
        "additionalProperties": False,
    }
    s, d = chat(
        {
            "messages": user("Make up a person."),
            "max_tokens": 80,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "person", "schema": schema, "strict": True},
            },
            **NOTHINK,
        }
    )
    o = json.loads(m0(d)["content"])
    return set(o) == {"name", "age"} and isinstance(o["age"], int), o


@test("oa.chat.tools_auto")
def _():
    s, d = chat(
        {
            "messages": user("What is the weather in Montreal?"),
            "tools": [WEATHER_OA],
            "max_tokens": 200,
            **NOTHINK,
        }
    )
    tc = m0(d).get("tool_calls") or []
    ok = tc and tc[0]["function"]["name"] == "get_weather" and "Montreal" in tc[0]["function"]["arguments"]
    return ok and d["choices"][0]["finish_reason"] == "tool_calls", [
        (t["function"]["name"], t["function"]["arguments"]) for t in tc
    ]


@test("oa.chat.tools_required")
def _():
    s, d = chat(
        {
            "messages": user("Hello!"),
            "tools": [WEATHER_OA],
            "tool_choice": "required",
            "max_tokens": 200,
            **NOTHINK,
        }
    )
    tc = m0(d).get("tool_calls") or []
    return bool(tc), [(t["function"]["name"], t["function"]["arguments"]) for t in tc] or m0(d)["content"]


@test("oa.chat.tools_named")
def _():
    s, d = chat(
        {
            "messages": user("What is the weather in Paris?"),
            "tools": [WEATHER_OA, TIME_OA],
            "tool_choice": {"type": "function", "function": {"name": "get_time"}},
            "max_tokens": 200,
            **NOTHINK,
        }
    )
    tc = m0(d).get("tool_calls") or []
    return bool(tc) and tc[0]["function"]["name"] == "get_time", [
        (t["function"]["name"], t["function"]["arguments"]) for t in tc
    ]


@test("oa.chat.tools_none")
def _():
    s, d = chat(
        {
            "messages": user("What is the weather in Montreal?"),
            "tools": [WEATHER_OA],
            "tool_choice": "none",
            "max_tokens": 100,
            **NOTHINK,
        }
    )
    return not m0(d).get("tool_calls") and m0(d)["content"], m0(d)["content"]


@test("oa.chat.tools_parallel")
def _():
    s, d = chat(
        {
            "messages": user("Get the weather in Montreal and in Toronto. Call the tool for both cities at once."),
            "tools": [WEATHER_OA],
            "max_tokens": 300,
            **NOTHINK,
        }
    )
    tc = m0(d).get("tool_calls") or []
    return len(tc) >= 2, [t["function"]["arguments"] for t in tc]


@test("oa.chat.tool_roundtrip")
def _():
    msgs = user("What is the weather in Montreal?")
    s, d = chat({"messages": msgs, "tools": [WEATHER_OA], "max_tokens": 200, **NOTHINK})
    call = m0(d)["tool_calls"][0]
    msgs = msgs + [
        {"role": "assistant", "content": None, "tool_calls": [call]},
        {
            "role": "tool",
            "tool_call_id": call["id"],
            "content": '{"temp_c": -7, "sky": "snow"}',
        },
    ]
    s, d = chat({"messages": msgs, "tools": [WEATHER_OA], "max_tokens": 120, **NOTHINK})
    c = m0(d)["content"] or ""
    return s == 200 and ("-7" in c or "7" in c) and "snow" in c.lower(), c


@test("oa.chat.tools_stream")
def _():
    s, raw = chat(
        {
            "messages": user("What is the weather in Montreal?"),
            "tools": [WEATHER_OA],
            "max_tokens": 200,
            "stream": True,
            **NOTHINK,
        },
        stream=True,
    )
    ev = [d for _, d in sse(raw) if d and d.get("choices")]
    args = "".join(
        (tc.get("function", {}).get("arguments") or "")
        for d in ev
        for tc in (d["choices"][0]["delta"].get("tool_calls") or [])
    )
    fin = [d["choices"][0].get("finish_reason") for d in ev if d["choices"][0].get("finish_reason")]
    return "Montreal" in args and fin[-1:] == ["tool_calls"], f"args {args!r}, finish {fin}"


@test("oa.chat.vision")
def _():
    s, d = chat(
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What colour is this image? One word.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{RED_B64}"},
                        },
                    ],
                }
            ],
            "max_tokens": 20,
            **NOTHINK,
        }
    )
    c = m0(d)["content"] if s == 200 else d
    return s == 200 and "red" in c.lower(), c


@test("oa.chat.user_and_metadata_fields")
def _():
    s, d = chat({"messages": user("Say OK."), "max_tokens": 10, "user": "scope-test", **NOTHINK})
    return s == 200, m0(d)["content"] if s == 200 else d


# ---------------------------------------------------------------- OpenAI: text completions
def comp(body, **kw):
    return post("/v1/completions", {"model": OA, **body}, **kw)


@test("oa.comp.basic")
def _():
    s, d = comp({"prompt": "The capital of France is", "max_tokens": 5, "temperature": 0})
    return s == 200 and "Paris" in d["choices"][0]["text"], d["choices"][0]["text"]


@test("oa.comp.stream")
def _():
    s, raw = comp(
        {"prompt": "1, 2, 3,", "max_tokens": 10, "temperature": 0, "stream": True},
        stream=True,
    )
    ev = [d for _, d in sse(raw) if d and d.get("choices")]
    t = "".join(d["choices"][0].get("text") or "" for d in ev)
    return s == 200 and "4" in t, f"{len(ev)} chunks: {t!r}"


@test("oa.comp.stop")
def _():
    s, d = comp({"prompt": "1, 2, 3, 4,", "max_tokens": 30, "temperature": 0, "stop": [" 7"]})
    t = d["choices"][0]["text"]
    return "7" not in t and "6" in t, t


@test("oa.comp.prompt_list")
def _():
    s, d = comp(
        {
            "prompt": ["The capital of France is", "The capital of Italy is"],
            "max_tokens": 4,
            "temperature": 0,
        }
    )
    return s == 200 and len(d.get("choices", [])) == 2, [c["text"] for c in d["choices"]] if s == 200 else d


@test("oa.comp.n2")
def _():
    s, d = comp(
        {
            "prompt": "A good name for a cat is",
            "max_tokens": 5,
            "n": 2,
            "temperature": 1.0,
        }
    )
    return s == 200 and len(d.get("choices", [])) == 2, [c["text"] for c in d["choices"]] if s == 200 else d


@test("oa.comp.logprobs")
def _():
    s, d = comp(
        {
            "prompt": "The capital of France is",
            "max_tokens": 3,
            "temperature": 0,
            "logprobs": 2,
        }
    )
    lp = d["choices"][0].get("logprobs") if s == 200 else None
    return bool(lp and lp.get("tokens")), lp if lp else d


@test("oa.comp.echo")
def _():
    s, d = comp({"prompt": "Hello world", "max_tokens": 3, "temperature": 0, "echo": True})
    t = d["choices"][0]["text"] if s == 200 else d
    return s == 200 and t.startswith("Hello world"), t


@test("oa.comp.seed")
def _():
    b = {"prompt": "A name for a ship:", "max_tokens": 8, "temperature": 1.0, "seed": 7}
    a, c = comp(b)[1]["choices"][0]["text"], comp(b)[1]["choices"][0]["text"]
    return a == c, f"{a!r} vs {c!r}"


# ---------------------------------------------------------------- Anthropic
@test("an.basic")
def _():
    s, d = msg({"max_tokens": 400, "messages": user("Say OK.")})
    return s == 200 and d["stop_reason"] == "end_turn" and "OK" in texts(d), [c["type"] for c in d["content"]]


@test("an.x_api_key_header")
def _():
    h = {**AHDR, "x-api-key": KEY}
    s, d = post(
        "/v1/messages",
        {"model": AN, "max_tokens": 300, "messages": user("Say OK.")},
        headers=h,
        key="",
    )
    return s == 200 and "OK" in texts(d), texts(d) if s == 200 else d


@test("an.system_string")
def _():
    s, d = msg(
        {
            "max_tokens": 400,
            "system": "Reply only in French.",
            "messages": user("Good morning! One short sentence."),
        }
    )
    return s == 200 and texts(d), texts(d)


@test("an.system_blocks_cache_control")
def _():
    sysb = [
        {
            "type": "text",
            "text": "You are terse. " * 200,
            "cache_control": {"type": "ephemeral"},
        }
    ]
    s1, d1 = msg({"max_tokens": 300, "system": sysb, "messages": user("Say OK.")})
    s2, d2 = msg({"max_tokens": 300, "system": sysb, "messages": user("Say OK.")})
    u = d2.get("usage", {})
    return s1 == s2 == 200 and (u.get("cache_read_input_tokens") or 0) > 0, u


@test("an.multiturn")
def _():
    s, d = msg(
        {
            "max_tokens": 400,
            "messages": [
                {"role": "user", "content": "My favourite colour is teal."},
                {"role": "assistant", "content": "Noted: teal."},
                {"role": "user", "content": "What is my favourite colour? One word."},
            ],
        }
    )
    return "teal" in texts(d).lower(), texts(d)


@test("an.thinking_enabled")
def _():
    s, d = msg(
        {
            "max_tokens": 2000,
            "thinking": {"type": "enabled", "budget_tokens": 1500},
            "messages": user("What is 17*23? Number only."),
        }
    )
    th = [c for c in d["content"] if c["type"] == "thinking"]
    return th and "signature" in th[0] and "391" in texts(
        d
    ), f"thinking {len(th[0]['thinking']) if th else 0} chars, text {texts(d)!r}"


@test("an.thinking_disabled")
def _():
    s, d = msg(
        {
            "max_tokens": 200,
            "thinking": {"type": "disabled"},
            "messages": user("What is 17*23? Number only."),
        }
    )
    th = [c for c in d["content"] if c["type"] == "thinking" and c.get("thinking")]
    return not th and "391" in texts(d), f"thinking blocks {len(th)}, text {texts(d)!r}"


@test("an.thinking_roundtrip_signature")
def _():
    q = user("Pick a random fruit. One word.")
    s, d = msg(
        {
            "max_tokens": 1500,
            "thinking": {"type": "enabled", "budget_tokens": 1000},
            "messages": q,
        }
    )
    msgs = q + [
        {"role": "assistant", "content": d["content"]},
        {"role": "user", "content": "Which fruit did you pick? One word."},
    ]
    s2, d2 = msg(
        {
            "max_tokens": 1500,
            "thinking": {"type": "enabled", "budget_tokens": 1000},
            "messages": msgs,
        }
    )
    return s2 == 200 and texts(d2), f"turn1 {texts(d)!r} turn2 {texts(d2)!r}"


@test("an.max_tokens_stop")
def _():
    s, d = msg(
        {
            "max_tokens": 10,
            "thinking": {"type": "disabled"},
            "messages": user("Write a long story about a ship."),
        }
    )
    return d["stop_reason"] == "max_tokens", (d["stop_reason"], d["usage"])


@test("an.stop_sequences")
def _():
    s, d = msg(
        {
            "max_tokens": 200,
            "thinking": {"type": "disabled"},
            "stop_sequences": ["five"],
            "messages": user("List the numbers one to ten in words, comma separated."),
        }
    )
    return d["stop_reason"] == "stop_sequence" and d.get("stop_sequence") == "five" and "six" not in texts(d), (
        d["stop_reason"],
        d.get("stop_sequence"),
        texts(d),
    )


@test("an.sampling_params")
def _():
    s, d = msg(
        {
            "max_tokens": 50,
            "thinking": {"type": "disabled"},
            "temperature": 0.5,
            "top_p": 0.9,
            "top_k": 40,
            "messages": user("Say hello."),
        }
    )
    return s == 200 and texts(d), texts(d) if s == 200 else d


@test("an.temperature0_deterministic")
def _():
    b = {
        "max_tokens": 40,
        "thinking": {"type": "disabled"},
        "temperature": 0,
        "messages": user("Write one sentence about the sea."),
    }
    a, c = texts(msg(b)[1]), texts(msg(b)[1])
    return a == c, a


@test("an.metadata")
def _():
    s, d = msg(
        {
            "max_tokens": 300,
            "metadata": {"user_id": "scope-test"},
            "messages": user("Say OK."),
        }
    )
    return s == 200, texts(d) if s == 200 else d


@test("an.stream")
def _():
    s, raw = msg(
        {
            "max_tokens": 1500,
            "stream": True,
            "thinking": {"type": "enabled", "budget_tokens": 1000},
            "messages": user("What is 6*7?"),
        },
        stream=True,
    )
    ev = sse(raw)
    names = [e for e, _ in ev]
    kinds = [d["content_block"]["type"] for e, d in ev if e == "content_block_start"]
    sig = any(d.get("delta", {}).get("type") == "signature_delta" for e, d in ev if e == "content_block_delta")
    text = "".join(d["delta"].get("text", "") for e, d in ev if e == "content_block_delta")
    md = [d for e, d in ev if e == "message_delta"]
    ok = (
        names[0] == "message_start"
        and names[-1] == "message_stop"
        and kinds == ["thinking", "text"]
        and "42" in text
        and md
    )
    return (
        ok,
        f"{len(ev)} events, blocks {kinds}, signature_delta {sig}, stop {md[-1]['delta'].get('stop_reason') if md else None}, usage {md[-1].get('usage') if md else None}",
    )


@test("an.tools_auto")
def _():
    s, d = msg(
        {
            "max_tokens": 600,
            "tools": [WEATHER_AN],
            "messages": user("What is the weather in Montreal?"),
        }
    )
    tu = [c for c in d["content"] if c["type"] == "tool_use"]
    return d["stop_reason"] == "tool_use" and tu and tu[0]["input"].get("city") == "Montreal", [
        (t["name"], t["input"]) for t in tu
    ]


@test("an.tools_any")
def _():
    s, d = msg(
        {
            "max_tokens": 600,
            "tools": [WEATHER_AN],
            "tool_choice": {"type": "any"},
            "messages": user("Hello!"),
        }
    )
    tu = [c for c in d["content"] if c["type"] == "tool_use"]
    return bool(tu), [(t["name"], t["input"]) for t in tu] or texts(d)


@test("an.tools_named")
def _():
    s, d = msg(
        {
            "max_tokens": 600,
            "tools": [WEATHER_AN, TIME_AN],
            "tool_choice": {"type": "tool", "name": "get_time"},
            "messages": user("What is the weather in Paris?"),
        }
    )
    tu = [c for c in d["content"] if c["type"] == "tool_use"]
    return bool(tu) and tu[0]["name"] == "get_time", [(t["name"], t["input"]) for t in tu]


@test("an.tools_none")
def _():
    s, d = msg(
        {
            "max_tokens": 600,
            "tools": [WEATHER_AN],
            "tool_choice": {"type": "none"},
            "messages": user("What is the weather in Montreal?"),
        }
    )
    tu = [c for c in d["content"] if c["type"] == "tool_use"]
    return s == 200 and not tu and texts(d), texts(d) if s == 200 else d


@test("an.tools_parallel")
def _():
    s, d = msg(
        {
            "max_tokens": 800,
            "tools": [WEATHER_AN],
            "messages": user("Get the weather in Montreal and in Toronto. Call the tool for both cities at once."),
        }
    )
    tu = [c for c in d["content"] if c["type"] == "tool_use"]
    return len(tu) >= 2, [t["input"] for t in tu]


@test("an.tool_result_roundtrip")
def _():
    q = user("What is the weather in Montreal?")
    s, d = msg({"max_tokens": 600, "tools": [WEATHER_AN], "messages": q})
    tu = [c for c in d["content"] if c["type"] == "tool_use"][0]
    msgs = q + [
        {"role": "assistant", "content": d["content"]},
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tu["id"],
                    "content": '{"temp_c": -7, "sky": "snow"}',
                }
            ],
        },
    ]
    s, d = msg({"max_tokens": 600, "tools": [WEATHER_AN], "messages": msgs})
    t = texts(d)
    return s == 200 and "7" in t and "snow" in t.lower(), t


@test("an.tool_result_is_error")
def _():
    q = user("What is the weather in Montreal?")
    s, d = msg({"max_tokens": 600, "tools": [WEATHER_AN], "messages": q})
    tu = [c for c in d["content"] if c["type"] == "tool_use"][0]
    msgs = q + [
        {"role": "assistant", "content": d["content"]},
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tu["id"],
                    "is_error": True,
                    "content": "service unavailable",
                }
            ],
        },
    ]
    s, d = msg({"max_tokens": 600, "tools": [WEATHER_AN], "messages": msgs})
    return s == 200, (d["stop_reason"], texts(d)[:120]) if s == 200 else d


@test("an.tools_stream")
def _():
    s, raw = msg(
        {
            "max_tokens": 600,
            "stream": True,
            "tools": [WEATHER_AN],
            "messages": user("What is the weather in Montreal?"),
        },
        stream=True,
    )
    ev = sse(raw)
    js = "".join(d["delta"].get("partial_json", "") for e, d in ev if e == "content_block_delta")
    stop = [d["delta"].get("stop_reason") for e, d in ev if e == "message_delta"]
    return "Montreal" in js and stop[-1:] == ["tool_use"], f"input_json {js!r}, stop {stop}"


@test("an.image_base64")
def _():
    s, d = msg(
        {
            "max_tokens": 400,
            "thinking": {"type": "disabled"},
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": RED_B64,
                            },
                        },
                        {
                            "type": "text",
                            "text": "What colour is this image? One word.",
                        },
                    ],
                }
            ],
        }
    )
    return s == 200 and "red" in texts(d).lower(), texts(d) if s == 200 else d


@test("an.count_tokens")
def _():
    s, d = post(
        "/v1/messages/count_tokens",
        {"model": AN, "messages": user("Hello there")},
        headers=AHDR,
    )
    return s == 200 and d.get("input_tokens", 0) > 0, d


@test("an.count_tokens_tools_system")
def _():
    s, d = post(
        "/v1/messages/count_tokens",
        {
            "model": AN,
            "system": "Be brief.",
            "tools": [WEATHER_AN],
            "messages": user("Hello there"),
        },
        headers=AHDR,
    )
    return s == 200 and d.get("input_tokens", 0) > 9, d


# ---------------------------------------------------------------- errors
@test("err.bad_key")
def _():
    s, d = chat({"messages": user("hi"), "max_tokens": 5}, key="sk-wrong")
    return s in (400, 401), (s, str(d)[:120])


@test("err.unknown_model")
def _():
    s, d = post(
        "/v1/chat/completions",
        {"model": "no-such-model", "messages": user("hi"), "max_tokens": 5},
    )
    return s in (400, 404), (s, str(d)[:120])


@test("err.an_missing_max_tokens")
def _():
    s, d = post("/v1/messages", {"model": AN, "messages": user("hi")}, headers=AHDR)
    return True, (s, str(d)[:160])


def run(name):
    t0 = time.time()
    try:
        ok, info = TESTS[name]()
    except Exception as e:  # a crash is a failure with its reason
        ok, info = False, f"{type(e).__name__}: {e}"
    return name, bool(ok), time.time() - t0, info


if __name__ == "__main__":
    names = [n for n in TESTS if len(sys.argv) < 2 or sys.argv[1] in n]
    with ThreadPoolExecutor(6) as pool:
        results = list(pool.map(run, names))
    for name, ok, dt, info in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name:38s} {dt:6.1f}s  {str(info)[:260]}")
    print(f"\n{sum(r[1] for r in results)}/{len(results)} passed")
