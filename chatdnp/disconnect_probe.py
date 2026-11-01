"""Does the server stop generating when the client closes the socket?

For a streamed and a non-streaming completion (ignore_eos, --max-tokens tokens): open a socket, wait
until the server generates, close the socket, then poll GET /v1/stats. The request is gone when
requests.active is back at the value from before the request, and at least --margin s before the
request could have finished by itself. A short request after that must also return quickly.
With FT_LOG set, the new part of that log must have an "Aborting request" line for each disconnect.
Other clients of the same server can delay the return to the old value and cause a false FAIL.

Ends with DISCONNECT_PROBE PASS or FAIL.

usage: [FT_LOG=~/logs/ft.log] disconnect_probe.py [--base-url URL] [--max-tokens 2000] [--watch 10]
"""

import argparse
import json
import os
import socket
import sys
import time
from urllib.parse import urlsplit

import requests


def stats(a):
    return requests.get(a.base_url + "/v1/stats", timeout=10).json()


def active(a):
    return stats(a)["requests"]["active"]


def open_request(a, stream):
    body = json.dumps(
        {
            "model": a.model,
            "messages": [{"role": "user", "content": "Count upward from 1, one number per line."}],
            "max_tokens": a.max_tokens,
            "temperature": 0,
            "ignore_eos": True,
            "stream": stream,
            "chat_template_kwargs": {"enable_thinking": False},
        }
    ).encode()
    url = urlsplit(a.base_url)
    head = (
        f"POST /v1/chat/completions HTTP/1.1\r\nHost: {url.netloc}\r\nContent-Type: application/json\r\n"
        f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n"
    ).encode()
    sock = socket.create_connection((url.hostname, url.port or 80), timeout=60)
    sock.sendall(head + body)
    return sock


def drop(a, stream, baseline):
    """One disconnect. Returns (ok, facts)."""
    sock = open_request(a, stream)
    t0 = time.monotonic()
    received = b""
    if stream:
        while received.count(b"data: ") < a.chunks:
            part = sock.recv(65536)
            if not part:
                sock.close()
                return False, {
                    "error": "the stream ended before the disconnect",
                    "head": received[:300].decode("latin-1"),
                }
            received += part
    else:
        # no bytes come before the end: wait until the request is active, then let it generate
        deadline = t0 + 30
        while active(a) <= baseline and time.monotonic() < deadline:
            time.sleep(0.05)
        time.sleep(a.hold)
    during = stats(a)
    sock.close()
    t1 = time.monotonic()

    back = None
    while time.monotonic() - t1 < a.watch:
        if active(a) <= baseline:
            back = time.monotonic() - t1
            break
        time.sleep(0.05)
    after = stats(a)

    t2 = time.monotonic()
    r = requests.post(
        a.base_url + "/v1/chat/completions",
        json={
            "model": a.model,
            "messages": [{"role": "user", "content": "Say hi."}],
            "max_tokens": 1,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=120,
    )
    short_s = time.monotonic() - t2

    facts = {
        "held_s": round(t1 - t0, 2),
        "active_during": during["requests"]["active"],
        "active_back_after_s": None if back is None else round(back, 2),
        "active_after": after["requests"]["active"],
        "kv_pages_during": (during.get("kv") or {}).get("used_pages"),
        "kv_pages_after": (after.get("kv") or {}).get("used_pages"),
        "decode_tps_after": after["throughput"]["decode_tps"],
        "short_request_s": round(short_s, 2),
        "short_status": r.status_code,
    }
    ok = facts["active_during"] > baseline and back is not None and r.status_code == 200 and short_s < a.short_limit
    return ok, facts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="Qwen3.8-Flash-Next")
    ap.add_argument("--max-tokens", type=int, default=2000)
    ap.add_argument("--watch", type=float, default=10.0, help="seconds to wait for the request to go away")
    ap.add_argument("--chunks", type=int, default=50, help="stream: chunks to read before the disconnect")
    ap.add_argument("--hold", type=float, default=2.0, help="non-streaming: seconds of generation before it")
    ap.add_argument("--short-limit", type=float, default=10.0, help="limit for the short request after it, s")
    a = ap.parse_args()
    a.base_url = a.base_url.rstrip("/")

    log = os.environ.get("FT_LOG")
    log = os.path.expanduser(log) if log else None
    offset = os.path.getsize(log) if log else 0

    all_ok = True
    drops = 0
    for name, stream in (("streaming", True), ("non-streaming", False)):
        baseline = active(a)
        ok, facts = drop(a, stream, baseline)
        drops += 1
        all_ok &= ok
        print(f"{'pass' if ok else 'FAIL'} {name}: baseline active {baseline}, {json.dumps(facts)}")

    if log:
        time.sleep(1)
        with open(log, errors="replace") as f:
            f.seek(offset)
            new = f.read()
        aborts = new.count("Aborting request")
        ok = aborts >= drops
        all_ok &= ok
        print(f"{'pass' if ok else 'FAIL'} FT_LOG {log}: {aborts} 'Aborting request' lines, want >= {drops}")
    print("DISCONNECT_PROBE", "PASS" if all_ok else "FAIL")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
