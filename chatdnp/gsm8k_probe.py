"""Accuracy on the first N questions of GSM8K test.jsonl, greedy.

The data file is downloaded once to ~/.cache/freetoken-tools/. The prompt asks for '#### <number>';
the answer is the last number after the last marker, and numbers are compared numerically. A reply
without a marker counts as wrong. Prints one JSON line and writes per-question results to --out.
Exit status is 1 if a request failed.

usage: gsm8k_probe.py --n 300 --concurrency 8 --max-tokens 1536 --label L --out F [--thinking]
"""

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction

import requests

DATA_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
INSTRUCTION = (
    "Solve the problem step by step. End your reply with the final answer on its own line, as '#### <number>'."
)


def load_questions(n):
    cache = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    path = os.path.join(cache, "freetoken-tools", "gsm8k_test.jsonl")
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        r = requests.get(DATA_URL, timeout=120)
        r.raise_for_status()
        with open(path + ".tmp", "w") as f:
            f.write(r.text)
        os.replace(path + ".tmp", path)
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()][:n]


def to_number(text):
    """The numeric value of a number string, or None."""
    try:
        return Fraction(text.replace(",", "").rstrip("."))
    except (ValueError, ZeroDivisionError):
        return None


def extract(text):
    """The string of the last number after the last '####', or None."""
    if not text or "####" not in text:
        return None
    found = NUMBER.findall(text.rsplit("####", 1)[1])
    return found[-1] if found else None


def ask(a, index, item):
    gold = extract(item["answer"])
    body = {
        "model": a.model,
        "messages": [{"role": "user", "content": f"{item['question']}\n\n{INSTRUCTION}"}],
        "max_tokens": a.max_tokens,
        "temperature": a.temperature,
        "chat_template_kwargs": {"enable_thinking": a.thinking},
    }
    row = {"index": index, "question": item["question"], "gold": gold, "pred": None, "correct": False}
    row.update(finish_reason=None, completion_tokens=0, text=None, error=None)
    try:
        r = requests.post(a.base_url + "/v1/chat/completions", json=body, timeout=3600)
        r.raise_for_status()
        reply = r.json()
    except (requests.RequestException, ValueError) as e:
        row["error"] = repr(e)
        return row
    choice = reply["choices"][0]
    row["text"] = choice["message"].get("content") or ""
    row["finish_reason"] = choice.get("finish_reason")
    row["completion_tokens"] = (reply.get("usage") or {}).get("completion_tokens", 0)
    row["pred"] = extract(row["text"])
    if row["pred"] is not None:
        row["correct"] = to_number(row["pred"]) == to_number(gold)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default=os.environ.get("FT_BASE", "http://127.0.0.1:8000"))
    ap.add_argument("--model", default="Qwen3.8-Flash-Next")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=1536)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--label", default="gsm8k")
    ap.add_argument("--out", required=True)
    ap.add_argument("--thinking", action="store_true")
    a = ap.parse_args()
    a.base_url = a.base_url.rstrip("/")

    items = load_questions(a.n)
    t0 = time.monotonic()
    with ThreadPoolExecutor(a.concurrency) as ex:
        rows = list(ex.map(lambda pair: ask(a, *pair), enumerate(items)))
    wall = time.monotonic() - t0

    correct = sum(r["correct"] for r in rows)
    errors = sum(r["error"] is not None for r in rows)
    summary = {
        "label": a.label,
        "n": len(rows),
        "accuracy": round(correct / max(len(rows), 1), 4),
        "correct": correct,
        "truncated": sum(r["finish_reason"] == "length" for r in rows),
        "mean_tokens": round(sum(r["completion_tokens"] for r in rows) / max(len(rows), 1), 1),
        "wall_s": round(wall, 2),
        "thinking": a.thinking,
        "temperature": a.temperature,
        "errors": errors,
    }
    with open(a.out, "w") as f:
        json.dump({"summary": summary, "results": rows}, f, indent=1)
    print(json.dumps(summary))
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
