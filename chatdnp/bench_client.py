"""Decode rate, aggregate rate and TTFT of a FreeToken server, with deterministic prompts.

Single stream: streaming chat completion, thinking off, temperature 0, ignore_eos, 256 output tokens.
The rate is usage.completion_tokens / (last token time - first token time). N repeats, median.
Aggregate: C streams at the same time, 512 output tokens each. The rate is the sum of the tokens
divided by the wall time from the first start to the last end.
TTFT: about 1000 prompt tokens (calibrated with usage.prompt_tokens), several repeats, median.
Each request has its own prompt from --seed, so the same seed gives the same prompts. The same seed
on the same running server hits the prefix cache and skews TTFT: use a new --seed for each run. The
tool warns if a TTFT sample reports cached tokens.

The final stdout line is JSON. Its lists have one element per run of the tool, so that parsers of
several runs can concatenate them. --out has the raw samples.

usage: bench_client.py --base-url URL --repeats N --concurrent C --out F [--seed S] [--ttft-repeats R]
"""

import argparse
import json
import random
import statistics
import sys
import threading
import time

import requests

WORDS = [
    "river",
    "stone",
    "garden",
    "window",
    "paper",
    "silver",
    "market",
    "bridge",
    "winter",
    "summer",
    "forest",
    "engine",
    "candle",
    "harbor",
    "valley",
    "planet",
    "orchard",
    "copper",
    "lantern",
    "meadow",
    "signal",
    "mirror",
    "castle",
    "thunder",
    "pencil",
    "ladder",
    "blanket",
    "compass",
    "mountain",
    "whistle",
    "shadow",
    "feather",
    "anchor",
    "basket",
    "cotton",
    "dragon",
    "marble",
    "needle",
    "pocket",
    "rabbit",
    "saddle",
    "ticket",
    "velvet",
    "wagon",
    "yellow",
    "zipper",
    "almond",
    "bottle",
    "cactus",
]


def make_words(kind, seed, index, count):
    """Text that depends only on the arguments. The first word is unique to this request."""
    rng = random.Random(f"{seed}:{kind}:{index}")
    return f"{rng.getrandbits(64):016x} " + " ".join(rng.choice(WORDS) for _ in range(count))


def decode_prompt(kind, seed, index):
    return make_words(kind, seed, index, 40) + "\nContinue these words as a long story."


def stream_chat(a, prompt, max_tokens):
    """One streaming completion. Returns the times of the first and last tokens and the usage."""
    body = {
        "model": a.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    first = last = usage = None
    start = time.monotonic()
    with requests.post(a.base_url + "/v1/chat/completions", json=body, stream=True, timeout=1800) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line.startswith(b"data: ") or line == b"data: [DONE]":
                continue
            now = time.monotonic()
            chunk = json.loads(line[6:])
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                if delta.get("content") or delta.get("reasoning_content"):
                    first = first if first is not None else now
                    last = now
    end = time.monotonic()
    if usage is None or first is None:
        raise RuntimeError(f"stream without usage or tokens: usage={usage} first={first}")
    details = usage.get("prompt_tokens_details") or {}
    return {
        "start": start,
        "end": end,
        "ttft_s": first - start,
        "first_to_last_s": last - first,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "cached_tokens": details.get("cached_tokens", 0),  # the server omits the details for 0
    }


def public(sample):
    """A sample without the absolute clock values."""
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in sample.items() if k not in ("start", "end")}


def single_stream(a):
    samples = []
    for i in range(a.repeats):
        s = stream_chat(a, decode_prompt("single", a.seed, i), a.single_tokens)
        s["tps"] = s["completion_tokens"] / s["first_to_last_s"]
        samples.append(s)
    return statistics.median(s["tps"] for s in samples), [public(s) for s in samples]


def aggregate(a):
    results = [None] * a.concurrent
    errors = []
    barrier = threading.Barrier(a.concurrent)

    def worker(i):
        barrier.wait()
        try:
            results[i] = stream_chat(a, decode_prompt("aggregate", a.seed, i), a.aggregate_tokens)
        except Exception as e:  # noqa: BLE001 - the main thread reports it after the join
            errors.append(repr(e))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(a.concurrent)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise RuntimeError(f"aggregate streams failed: {errors}")
    wall = max(r["end"] for r in results) - min(r["start"] for r in results)
    tokens = sum(r["completion_tokens"] for r in results)
    return tokens / wall, {"wall_s": round(wall, 4), "tokens": tokens, "streams": [public(r) for r in results]}


def ttft_1k(a):
    """Calibrate the word count on its own prompt, then measure on unique prompts of that length."""
    calib_words = 700
    calib = stream_chat(a, make_words("calibrate", a.seed, 0, calib_words), 1)
    words = max(1, round(calib_words * a.ttft_prompt_tokens / calib["prompt_tokens"]))
    samples = [
        stream_chat(a, make_words("ttft", a.seed, i, words), a.ttft_output_tokens) for i in range(a.ttft_repeats)
    ]
    for s in samples:
        if s["cached_tokens"] > 0.1 * s["prompt_tokens"]:
            print(f"WARNING: TTFT sample has {s['cached_tokens']} cached tokens: use a new --seed", file=sys.stderr)
    return (
        statistics.median(s["ttft_s"] for s in samples),
        statistics.median(s["prompt_tokens"] for s in samples),
        {"calibration_prompt_tokens": calib["prompt_tokens"], "words": words, "samples": [public(s) for s in samples]},
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="Qwen3.8-Flash-Next")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--concurrent", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--single-tokens", type=int, default=256)
    ap.add_argument("--aggregate-tokens", type=int, default=512)
    ap.add_argument("--ttft-repeats", type=int, default=5)
    ap.add_argument("--ttft-prompt-tokens", type=int, default=1000)
    ap.add_argument("--ttft-output-tokens", type=int, default=4)
    a = ap.parse_args()
    a.base_url = a.base_url.rstrip("/")

    status = requests.get(a.base_url + "/v1/cache/status", timeout=30)
    status.raise_for_status()
    status = status.json()
    if status.get("state") != "serving":
        sys.exit(f"server state is {status.get('state')}, not serving")
    geo = status["geometry"]
    residency = geo["moe_cache_size"] / (geo["num_experts"] * geo["num_moe_layers"])
    kv_tokens = geo["num_pages"] * geo["page_size"]

    single_tps, single_raw = single_stream(a)
    agg_tps, agg_raw = aggregate(a)
    ttft_s, ttft_tokens, ttft_raw = ttft_1k(a)

    final = {
        "out": a.out,
        "single_median_tps": [round(single_tps, 2)],
        "agg_tps": round(agg_tps, 2),
        "residency": [round(residency, 4)],
        "kv_tokens": [kv_tokens],
        "ttft_1k_s": [round(ttft_s, 4)],
        "ttft_1k_prompt_tokens": [ttft_tokens],
        "mamba_slots": [geo["num_mamba_slots"]],
    }
    with open(a.out, "w") as f:
        json.dump(
            {
                "args": vars(a),
                "state": status["state"],
                "geometry": geo,
                "summary": final,
                "single": single_raw,
                "aggregate": agg_raw,
                "ttft_1k": ttft_raw,
            },
            f,
            indent=1,
        )
    print(json.dumps(final))


if __name__ == "__main__":
    main()
