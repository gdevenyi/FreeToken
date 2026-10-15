"""OpenAI logprobs formatting: neutral engine entries -> wire shapes."""

from __future__ import annotations

from typing import Any


def chat_logprobs_error(req: Any) -> str | None:
    if req.top_logprobs is not None and not req.logprobs:
        return "top_logprobs requires logprobs=true"
    if req.top_logprobs is not None and not 0 <= req.top_logprobs <= 20:
        return "top_logprobs must be between 0 and 20"
    return None


def completion_logprobs_error(req: Any) -> str | None:
    if req.logprobs is not None and not 0 <= req.logprobs <= 5:
        return "logprobs must be between 0 and 5"
    if req.echo and req.logprobs is not None:
        return "echo with logprobs is not supported"
    return None


def chat_content_entry(entry: dict) -> dict:
    return {
        "token": entry["token"],
        "logprob": entry["logprob"],
        "bytes": entry["bytes"],
        "top_logprobs": [
            {
                "token": candidate["token"],
                "logprob": candidate["logprob"],
                "bytes": candidate["bytes"],
            }
            for candidate in entry["top"]
        ],
    }


def completions_logprobs(entries: list[dict], offsets: list[int], text_len: int) -> dict:
    """``offsets[i]`` is where entry i's token starts in the generated text (the pieces of
    a split multi-byte character share its start). Offsets are clipped to ``text_len``,
    the text sent so far, since a stop string trims tokens off the end."""
    tokens: list[str] = []
    token_logprobs: list[float] = []
    top_logprobs: list[dict[str, float]] = []

    for entry in entries:
        tokens.append(entry["token"])
        token_logprobs.append(entry["logprob"])
        top: dict[str, float] = {}
        for candidate in entry["top"]:
            # The legacy shape keys alternatives by text, and partial-UTF-8 candidates
            # all decode to U+FFFD: keep the most likely one per key.
            key, logprob = candidate["token"], candidate["logprob"]
            if key not in top or logprob > top[key]:
                top[key] = logprob
        top_logprobs.append(top)

    return {
        "tokens": tokens,
        "token_logprobs": token_logprobs,
        "top_logprobs": top_logprobs,
        "text_offset": [min(offset, text_len) for offset in offsets],
    }
