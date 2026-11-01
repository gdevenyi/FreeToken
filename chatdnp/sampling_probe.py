"""Sampling parameters the 2026-09-26 build must accept or refuse, over HTTP.

Refused with 400: temperature -1, top_p 0, top_p 1.5, and with 422 from upstream #527 pydantic bounds: presence_penalty 9 (outside [-2, 2]).
Accepted (200): top_k 0 and top_k -1 (both mean "disabled"), min_p, presence/frequency penalty,
repetition_penalty, logit_bias, min_tokens, stop_token_ids, and a plain request.
Also checks usage.completion_tokens_details.reasoning_tokens on a short thinking turn.
"""

import os
import argparse
import json

import requests

ap = argparse.ArgumentParser()
ap.add_argument("--base-url", default=os.environ.get("FT_BASE", "http://127.0.0.1:8000"))
ap.add_argument("--model", default="Qwen3.8-Flash-Next")
a = ap.parse_args()
url = a.base_url + "/v1/chat/completions"


def call(extra, thinking=False, max_tokens=8):
    body = {
        "model": a.model,
        "messages": [{"role": "user", "content": "Say hello in one word."}],
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": thinking},
    }
    body.update(extra)
    r = requests.post(url, json=body, timeout=300)
    return r.status_code, r


cases = [
    ({"temperature": -1}, 400),
    ({"top_p": 0}, 400),
    ({"top_p": 1.5}, 400),
    ({"presence_penalty": 9}, 422),
    ({}, 200),
    ({"top_k": 0}, 200),
    ({"top_k": -1}, 200),
    ({"top_k": 20, "top_p": 0.9, "temperature": 0.7}, 200),
    ({"min_p": 0.05}, 200),
    ({"presence_penalty": 0.5, "frequency_penalty": 0.3}, 200),
    ({"repetition_penalty": 1.1}, 200),
    ({"logit_bias": {"9707": -100}}, 200),
    ({"min_tokens": 4}, 200),
    ({"stop_token_ids": [248046]}, 200),
]
fails = 0
for extra, want in cases:
    got, r = call(extra)
    ok = got == want
    fails += not ok
    tail = "" if ok else " " + r.text[:160].replace("\n", " ")
    print(f"{'ok ' if ok else 'BAD'} {json.dumps(extra):58s} -> {got} (want {want}){tail}")

got, r = call({}, thinking=True, max_tokens=256)
det = (r.json().get("usage") or {}).get("completion_tokens_details") if got == 200 else None
print(f"reasoning_tokens usage detail: {det}")
print("SAMPLING_PROBE", "PASS" if fails == 0 else f"FAIL ({fails})")
