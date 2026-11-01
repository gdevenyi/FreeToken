"""How long does a long prefill freeze another user's stream? (the #484 decode-interleave question)

One streaming request decodes (ignore_eos, thinking off) while a second request sends a long
prompt. We record the arrival time of every streamed chunk and report the largest gap inside
the prefill window, the stream's rate outside it, and the long request's time to first token.

usage: stall_probe.py [--tokens 200000] [--base-url http://127.0.0.1:8000] [--out f.json]
"""

import os
import argparse
import itertools
import json
import threading
import time

import requests

ap = argparse.ArgumentParser()
ap.add_argument("--base-url", default=os.environ.get("FT_BASE", "http://127.0.0.1:8000"))
ap.add_argument("--model", default="Qwen3.8-Flash-Next")
ap.add_argument("--tokens", type=int, default=200000, help="long prompt, tokens (about)")
ap.add_argument("--stream-tokens", type=int, default=6000)
ap.add_argument("--lead", type=float, default=4.0, help="seconds of streaming before the long prompt")
ap.add_argument("--out", default="")
a = ap.parse_args()
url = a.base_url + "/v1/chat/completions"

arrivals = []
stream_done = threading.Event()


def streamer():
    body = {
        "model": a.model,
        "messages": [{"role": "user", "content": "Count upward from 1, one number per line."}],
        "max_tokens": a.stream_tokens,
        "stream": True,
        "ignore_eos": True,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    with requests.post(url, json=body, stream=True, timeout=900) as r:
        for line in r.iter_lines():
            if line.startswith(b"data: ") and line != b"data: [DONE]":
                arrivals.append(time.monotonic())
            if stream_done.is_set():
                break


t = threading.Thread(target=streamer, daemon=True)
t.start()
time.sleep(a.lead)
# ~5.27 characters per token for this filler on the Qwen3.8 tokenizer
unit = "The quick brown fox jumps over the lazy dog near the quiet river bank. "
filler = unit * int(a.tokens * 5.27 / len(unit))
t0 = time.monotonic()
r = requests.post(
    url,
    json={
        "model": a.model,
        "messages": [{"role": "user", "content": filler + "\nReply with the single word OK."}],
        "max_tokens": 4,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    },
    timeout=1800,
)
t1 = time.monotonic()
usage = r.json().get("usage", {})
time.sleep(3)
stream_done.set()
t.join(timeout=30)

inside = [b - x for x, b in itertools.pairwise(arrivals) if x >= t0 and b <= t1 + 0.5]
before = [x for x in arrivals if x < t0]
rate_before = (len(before) - 1) / (before[-1] - before[0]) if len(before) > 2 else 0.0
res = {
    "long_prompt_tokens": usage.get("prompt_tokens"),
    "long_request_s": round(t1 - t0, 2),
    "stream_chunks_total": len(arrivals),
    "stream_chunks_during_prefill": sum(1 for x in arrivals if t0 <= x <= t1),
    "max_gap_during_prefill_s": round(max(inside), 2) if inside else round(t1 - t0, 2),
    "stream_rate_before_chunks_per_s": round(rate_before, 1),
}
print("STALL_PROBE", json.dumps(res))
if a.out:
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
