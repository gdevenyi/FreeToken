"""Does the served model see images? Synthetic PNGs made with Pillow, plus a text-only check.

Image 1: text HELLO 42 with a blue circle. Image 2: text GREEN TEA with a green circle.
The reply has the form "TEXT: ...; CIRCLE: ...". The text part must read the text and the circle
part must name the colour and not the colour of the other image.
Prints one line per check with timing and usage, and ends with VISION_PROBE PASS or FAIL.

usage: vision_probe.py [--base-url URL] [--model NAME] [--max-tokens N]
"""

import os
import argparse
import base64
import io
import json
import sys
import time

import requests
from PIL import Image, ImageDraw, ImageFont

CHECKS = [
    ("HELLO 42", "blue", (30, 60, 220), "green"),
    ("GREEN TEA", "green", (20, 170, 60), "blue"),
]
QUESTION = (
    "Read the text in this image and name the colour of the circle. "
    "Reply in exactly this form, with nothing else: TEXT: <the text>; CIRCLE: <the colour>"
)


def png_data_url(text, rgb):
    img = Image.new("RGB", (640, 360), "white")
    draw = ImageDraw.Draw(img)
    draw.text((30, 40), text, fill="black", font=ImageFont.load_default(size=72))
    draw.ellipse((230, 160, 410, 340), fill=rgb, outline="black")
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def chat(a, content):
    t0 = time.monotonic()
    r = requests.post(
        a.base_url + "/v1/chat/completions",
        json={
            "model": a.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0,
            "max_tokens": a.max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=600,
    )
    dt = time.monotonic() - t0
    if r.status_code != 200:
        return r.status_code, r.text[:300], {}, dt
    j = r.json()
    return 200, j["choices"][0]["message"].get("content") or "", j.get("usage") or {}, dt


def report(name, ok, reply, usage, dt):
    print(f"{'pass' if ok else 'FAIL'} {name} {dt:.2f} s reply={reply.strip()!r}")
    print(f"     usage {json.dumps(usage)}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default=os.environ.get("FT_BASE", "http://127.0.0.1:8000"))
    ap.add_argument("--model", default="Qwen3.8-Flash-Next")
    ap.add_argument("--max-tokens", type=int, default=64)
    a = ap.parse_args()
    a.base_url = a.base_url.rstrip("/")

    ok = True
    for text, colour, rgb, other in CHECKS:
        content = [
            {"type": "image_url", "image_url": {"url": png_data_url(text, rgb)}},
            {"type": "text", "text": QUESTION},
        ]
        status, reply, usage, dt = chat(a, content)
        read, _, circle = reply.lower().partition("circle:")
        good = status == 200 and text.lower() in read and colour in circle and other not in circle
        ok &= report(f"image {text!r} {colour} circle", good, reply, usage, dt)

    status, reply, usage, dt = chat(a, "What is 2+2? Reply with only the number.")
    ok &= report("text only", status == 200 and "4" in reply, reply, usage, dt)
    print("VISION_PROBE", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
