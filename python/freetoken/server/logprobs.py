"""OpenAI logprobs formatting: neutral engine entries -> the legacy completions wire shape."""

from __future__ import annotations

from typing import Any

# AnyJev reads up to 26 letter labels; its vLLM recipe runs --max-logprobs 32
MAX_COMPLETION_LOGPROBS = 32
# vLLM's stand-in for -inf: Starlette's JSON encoder refuses non-finite floats
LOGPROB_FLOOR = -9999.0


def _clamped(logprob: float) -> float:
    # not max(): max(nan, floor) is nan, and a NaN must clamp too
    return logprob if logprob >= LOGPROB_FLOOR else LOGPROB_FLOOR  # noqa: FURB136


def completion_logprobs_error(req: Any) -> str | None:
    if req.logprobs is not None and not 0 <= req.logprobs <= MAX_COMPLETION_LOGPROBS:
        return f"logprobs must be between 0 and {MAX_COMPLETION_LOGPROBS}"
    if req.echo and req.logprobs is not None:
        return "echo with logprobs is not supported"
    return None


def completions_logprobs(
    entries: list[dict], offsets: list[int], text_len: int | None = None
) -> dict:
    """``offsets[i]`` is where entry i's token starts in the generated text (the pieces of
    a split multi-byte character share its start). Given the finished text's ``text_len``,
    offsets are clipped to it, since a matched stop string trims tokens off the end."""
    tokens: list[str] = []
    token_logprobs: list[float] = []
    top_logprobs: list[dict[str, float]] = []

    for entry in entries:
        tokens.append(entry["token"])
        token_logprobs.append(_clamped(entry["logprob"]))
        top: dict[str, float] = {}
        for candidate in entry["top"]:
            # The legacy shape keys alternatives by text, and partial-UTF-8 candidates
            # all decode to U+FFFD: keep the most likely one per key.
            key, logprob = candidate["token"], _clamped(candidate["logprob"])
            if key not in top or logprob > top[key]:
                top[key] = logprob
        top_logprobs.append(top)

    return {
        "tokens": tokens,
        "token_logprobs": token_logprobs,
        "top_logprobs": top_logprobs,
        "text_offset": [
            offset if text_len is None else min(offset, text_len) for offset in offsets
        ],
    }
