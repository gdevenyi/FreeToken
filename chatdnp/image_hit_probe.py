"""Prefix cache and images: repeated images must hit, a different image must not reuse image tokens.

Needs a server with --enable-cache-report (usage.prompt_tokens_details.cached_tokens). The server
omits that object for 0 cached tokens, so a server without the flag fails checks 2, 4 and 6.
Each run has its own nonce in the images and the text, so a run on a warm server is still cold.

 1 cold        image A + text, nothing cached
 2 turn 2      turn 1 + the reply + new text: the turn 1 prefix must be cached
 3 other image image B + the same text: must NOT reuse image tokens, and must read image B
 4 repeat      request 1 again: must be cached
 5 long cold   long text (--long-tokens, chunked prefill) + image A + question: nothing cached, reads image A
 6 long turn 2 request 5 + the reply + new text: the prefix must be cached, reads the image

Requests 2 and 6 extend the earlier prompt, like an agent session. A prompt that replaces the last
text is a different case: the hybrid cache resumes only at its snapshots, so such a prompt can get
0 cached tokens with a long shared head, with or without an image.

"Image tokens" = prompt tokens of request 1 less those of the same text without the image.
Ends with IMAGE_HIT_PROBE PASS or FAIL.

usage: image_hit_probe.py [--base-url URL] [--model NAME] [--long-tokens 23000] [--nonce N]
"""

import argparse
import base64
import io
import json
import os
import sys
import time

import requests
from PIL import Image, ImageDraw, ImageFont

HIT_FRACTION = 0.8
UNIT = "The quick brown fox jumps over the lazy dog near the quiet river bank. "
CHARS_PER_TOKEN = 4.71  # this filler on the Qwen3.8 tokenizer, measured with usage.prompt_tokens


def png_data_url(word, number, rgb, nonce):
    img = Image.new("RGB", (768, 432), "white")
    draw = ImageDraw.Draw(img)
    draw.text((40, 60), f"{word} {number}", fill="black", font=ImageFont.load_default(size=80))
    draw.rectangle((40, 220, 728, 400), fill=rgb)
    # nonce pixels: no earlier run has this exact image
    for bit in range(32):
        if (nonce >> bit) & 1:
            draw.rectangle((40 + bit * 21, 410, 58 + bit * 21, 428), fill="black")
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


class Probe:
    def __init__(self, a):
        self.a = a
        self.ok = True

    def chat(self, messages, max_tokens=24):
        t0 = time.monotonic()
        r = requests.post(
            self.a.base_url + "/v1/chat/completions",
            json={
                "model": self.a.model,
                "messages": messages,
                "temperature": 0,
                "max_tokens": max_tokens,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=900,
        )
        dt = time.monotonic() - t0
        if r.status_code != 200:
            sys.exit(f"IMAGE_HIT_PROBE FAIL: HTTP {r.status_code}: {r.text[:300]}")
        j = r.json()
        u = j.get("usage") or {}
        return {
            "reply": j["choices"][0]["message"].get("content") or "",
            "prompt": u.get("prompt_tokens", 0),
            "cached": (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
            "dt": dt,
        }

    def check(self, name, res, ok, why):
        self.ok &= bool(ok)
        print(
            f"{'pass' if ok else 'FAIL'} {name}: prompt={res['prompt']} cached={res['cached']} "
            f"{res['dt']:.2f} s, want {why}; reply={res['reply'].strip()!r}"
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="Qwen3.8-Flash-Next")
    ap.add_argument("--long-tokens", type=int, default=23000)
    ap.add_argument("--nonce", type=int, default=int(time.time() * 1000) ^ (os.getpid() << 16))
    a = ap.parse_args()
    a.base_url = a.base_url.rstrip("/")
    nonce = a.nonce & 0xFFFFFFFF
    print(f"nonce {nonce}")

    p = Probe(a)
    number_a, number_b = 100 + nonce % 450, 550 + nonce % 449
    image_a = {"type": "image_url", "image_url": {"url": png_data_url("CEDAR", number_a, (200, 40, 40), nonce)}}
    image_b = {"type": "image_url", "image_url": {"url": png_data_url("MAPLE", number_b, (40, 40, 200), nonce)}}
    tag = f"Request {nonce}. "
    text = {"type": "text", "text": tag + "What word and number does this image show? Reply with them only."}

    base = p.chat([{"role": "user", "content": text["text"]}], max_tokens=1)
    turn1 = [{"role": "user", "content": [image_a, text]}]
    cold = p.chat(turn1)
    image_tokens = cold["prompt"] - base["prompt"]
    print(f"image tokens {image_tokens} (text-only prompt {base['prompt']}, cached {base['cached']})")
    if image_tokens <= 0:
        sys.exit("IMAGE_HIT_PROBE FAIL: the image adds no prompt tokens")
    limit = image_tokens / 2
    p.check(
        "1 cold",
        cold,
        cold["cached"] < limit and "cedar" in cold["reply"].lower(),
        f"cached < {limit:.0f}, reads CEDAR",
    )

    follow = turn1 + [
        {"role": "assistant", "content": cold["reply"]},
        {"role": "user", "content": "Now reply with the word DONE."},
    ]
    res = p.chat(follow)
    need = HIT_FRACTION * cold["prompt"]
    p.check("2 turn 2, same image + new text", res, res["cached"] >= need, f"cached >= {need:.0f}")

    res = p.chat([{"role": "user", "content": [image_b, text]}])
    low = res["reply"].lower()
    p.check(
        "3 other image, same text",
        res,
        res["cached"] < limit and "maple" in low and "cedar" not in low,
        f"cached < {limit:.0f}, reads MAPLE and not CEDAR",
    )

    res = p.chat(turn1)
    p.check("4 identical repeat", res, res["cached"] >= need, f"cached >= {need:.0f}")

    filler = f"[{nonce}] " + UNIT * int(a.long_tokens * CHARS_PER_TOKEN / len(UNIT))
    long_head = {"type": "text", "text": filler}

    long_first = [{"role": "user", "content": [long_head, image_a, text]}]
    res = p.chat(long_first)
    low = res["reply"].lower()
    long_prompt = res["prompt"]
    p.check("5 long cold", res, res["cached"] < 0.1 * long_prompt and "cedar" in low, "cached < 10%, reads CEDAR")
    question = "What colour is the rectangle in this image? Reply with the colour only."
    long_next = long_first + [{"role": "assistant", "content": res["reply"]}, {"role": "user", "content": question}]
    res = p.chat(long_next)
    need = HIT_FRACTION * long_prompt
    p.check(
        "6 long turn 2", res, res["cached"] >= need and "red" in res["reply"].lower(), f"cached >= {need:.0f}, says red"
    )

    print(json.dumps({"image_tokens": image_tokens, "long_prompt_tokens": long_prompt}))
    print("IMAGE_HIT_PROBE", "PASS" if p.ok else "FAIL")
    sys.exit(0 if p.ok else 1)


if __name__ == "__main__":
    main()
