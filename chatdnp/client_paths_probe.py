"""Live checks of two client paths: Anthropic /v1/messages streaming with thinking (Claude Code
through LiteLLM) and an OpenAI tool call whose schema uses $ref/$defs and a nullable anyOf."""

import json
import os

import requests

BASE = os.environ.get("FT_BASE", "http://127.0.0.1:8000")
MODEL = "Qwen3.8-Flash-Next"
fails = 0

# 1. Anthropic streaming with thinking
body = {
    "model": MODEL,
    "max_tokens": 400,
    "stream": True,
    "thinking": {"type": "enabled", "budget_tokens": 300},
    "messages": [{"role": "user", "content": "What is 2+2? Answer with one number."}],
}
events, starts = [], []
with requests.post(
    BASE + "/v1/messages",
    json=body,
    headers={"anthropic-version": "2023-06-01"},
    stream=True,
    timeout=300,
) as r:
    status = r.status_code
    for line in r.iter_lines():
        if line.startswith(b"event: "):
            events.append(line[7:].decode())
        elif line.startswith(b"data: "):
            d = json.loads(line[6:])
            if d.get("type") == "content_block_start":
                starts.append(d["content_block"])
thinking = [b for b in starts if b.get("type") == "thinking"]
ok1 = status == 200 and thinking and "signature" in thinking[0] and events and events[-1] == "message_stop"
fails += not ok1
print(
    "anthropic stream:",
    "ok " if ok1 else "BAD",
    {
        "status": status,
        "blocks": [b.get("type") for b in starts],
        "thinking_start_keys": sorted(thinking[0]) if thinking else None,
        "last_event": events[-1] if events else None,
    },
)

# 2. tool call with $ref / $defs / nullable anyOf
tools = [
    {
        "type": "function",
        "function": {
            "name": "set_alarm",
            "description": "Set an alarm.",
            "parameters": {
                "type": "object",
                "$defs": {
                    "Time": {
                        "type": "object",
                        "properties": {"hour": {"type": "integer"}, "minute": {"type": "integer"}},
                        "required": ["hour", "minute"],
                    }
                },
                "properties": {
                    "time": {"$ref": "#/$defs/Time"},
                    "daily": {"anyOf": [{"type": "boolean"}, {"type": "null"}]},
                },
                "required": ["time"],
            },
        },
    }
]
r = requests.post(
    BASE + "/v1/chat/completions",
    json={
        "model": MODEL,
        "messages": [{"role": "user", "content": "Set an alarm for 7:30 every day."}],
        "tools": tools,
        "temperature": 0,
        "max_tokens": 300,
        "chat_template_kwargs": {"enable_thinking": False},
    },
    timeout=300,
)
calls = (r.json()["choices"][0]["message"].get("tool_calls") or []) if r.status_code == 200 else []
args = json.loads(calls[0]["function"]["arguments"]) if calls else None
t = (args or {}).get("time") if isinstance(args, dict) else None
ok2 = (
    isinstance(t, dict)
    and isinstance(t.get("hour"), int)
    and isinstance(t.get("minute"), int)
    and (args.get("daily") is None or isinstance(args.get("daily"), bool))
)
fails += not ok2
print("tool call $ref:", "ok " if ok2 else "BAD", {"status": r.status_code, "arguments": args})
print("CLIENT_PATHS_PROBE", "PASS" if fails == 0 else f"FAIL ({fails})")
