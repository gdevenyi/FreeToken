"""Latency of a returning agent turn (cached prefix + about 20 new tokens), and prefix retention.

Do returning agent sessions still hit the prefix cache after other sessions ran? (GDN snapshot retention)

Phase 1 sends turn 1 of S sessions, each with its own long prefix (T tokens). Phase 2 revisits every
session in the same order with turn 2 (turn 1 + the reply + a short new question). Only the GDN
snapshots that survived phase 1 let turn 2 skip the prefix, so the cached fraction of turn 2 shows
how many resume points the snapshot pool kept, and turn2_latency_s is the wall time of that turn. Needs the server started with --enable-cache-report
(usage.prompt_tokens_details.cached_tokens).

usage: revisit_probe.py [--sessions 16] [--tokens 24000] [--concurrency 4] [--out f.json]
"""

import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

import requests

ap = argparse.ArgumentParser()
ap.add_argument("--base-url", default="http://127.0.0.1:8000")
ap.add_argument("--model", default="Qwen3.8-Flash-Next")
ap.add_argument("--sessions", type=int, default=16)
ap.add_argument("--tokens", type=int, default=24000, help="prefix tokens per session (about)")
ap.add_argument("--concurrency", type=int, default=4)
ap.add_argument("--out", default="")
a = ap.parse_args()
url = a.base_url + "/v1/chat/completions"


def prefix(i):
    # distinct from the first token on, so sessions share nothing but the chat template head
    words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel"]
    line = f"Session {i} log entry: " + " ".join(words[(i + k) % 8] for k in range(12)) + ". "
    return f"[session {i}] " + line * int(a.tokens * 5.0 / len(line))


def ask(messages):
    t0 = time.monotonic()
    r = requests.post(
        url,
        json={
            "model": a.model,
            "messages": messages,
            "max_tokens": 8,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=1800,
    )
    dt = time.monotonic() - t0
    r.raise_for_status()
    j = r.json()
    u = j.get("usage") or {}
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
    return j["choices"][0]["message"].get("content") or "", u.get("prompt_tokens", 0), cached, dt


def turn1(i):
    msgs = [{"role": "user", "content": prefix(i) + "\nReply with the word READY."}]
    reply, pt, cached, _ = ask(msgs)
    return i, msgs, reply, pt, cached


with ThreadPoolExecutor(a.concurrency) as ex:
    first = sorted(ex.map(turn1, range(a.sessions)))


def turn2(item):
    i, msgs, reply, pt1, _ = item
    msgs = msgs + [{"role": "assistant", "content": reply}, {"role": "user", "content": "Now reply with the word BYE."}]
    _, pt, cached, dt = ask(msgs)
    return i, pt1, pt, cached, dt


with ThreadPoolExecutor(a.concurrency) as ex:
    second = sorted(ex.map(turn2, first))

rows = [
    {
        "session": i,
        "turn1_prompt": p1,
        "turn2_prompt": p,
        "turn2_cached": c,
        "hit": round(c / max(p1, 1), 3),
        "turn2_latency_s": round(dt, 3),
    }
    for i, p1, p, c, dt in second
]
hits = [r["hit"] for r in rows]
latencies = [r["turn2_latency_s"] for r in rows]
res = {
    "sessions": a.sessions,
    "tokens": a.tokens,
    "mean_hit_of_turn1_prefix": round(sum(hits) / len(hits), 3),
    "sessions_hit_ge_90pct": sum(h >= 0.9 for h in hits),
    "turn2_latency_median_s": round(statistics.median(latencies), 3),
    "turn2_latency_max_s": max(latencies),
    "turn2_new_tokens_median": statistics.median(r["turn2_prompt"] - r["turn2_cached"] for r in rows),
    "rows": rows,
}
print("REVISIT_PROBE", json.dumps({k: v for k, v in res.items() if k != "rows"}))
if a.out:
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
