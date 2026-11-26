"""OpenAI logprobs formatting: neutral engine entries -> the legacy completions wire shape."""

from __future__ import annotations

from typing import Any

# AnyJev reads up to 26 letter labels; its vLLM recipe runs --max-logprobs 32
MAX_COMPLETION_LOGPROBS = 32
MAX_CHAT_LOGPROBS = 20
# vLLM's stand-in for -inf: Starlette's JSON encoder refuses non-finite floats
LOGPROB_FLOOR = -9999.0


def _clamped(logprob: float) -> float:
    # not max(): max(nan, floor) is nan, and a NaN must clamp too
    return logprob if logprob >= LOGPROB_FLOOR else LOGPROB_FLOOR  # noqa: FURB136


def reply_logprob_pairs(ack: Any) -> list[tuple[dict, int]]:
    """Extract ``[(entry, text_offset), ...]`` from a ``UserReply`` (single-token or multi-token spec)."""
    lp_list = getattr(ack, "logprobs_list", None)
    if lp_list:
        offs = getattr(ack, "text_offsets", None)
        if not offs or len(offs) != len(lp_list):
            base = int(getattr(ack, "text_offset", 0))
            offs = [base] * len(lp_list)
        return list(zip(lp_list, offs, strict=True))
    single = getattr(ack, "logprobs", None)
    if single is not None:
        return [(single, int(getattr(ack, "text_offset", 0)))]
    return []


def completion_logprobs_error(req: Any) -> str | None:
    if req.logprobs is not None and not 0 <= req.logprobs <= MAX_COMPLETION_LOGPROBS:
        return f"logprobs must be between 0 and {MAX_COMPLETION_LOGPROBS}"
    if getattr(req, "logprob_token_ids", None) is not None and req.logprobs is None:
        return "logprob_token_ids requires logprobs to be set"
    if req.echo and req.logprobs is not None:
        return "echo with logprobs is not supported"
    return None


def chat_logprobs_content(
    entries: list[dict], as_token_ids: bool = False
) -> list[dict[str, Any]]:
    """Format neutral engine logprob entries into OpenAI ``ChatCompletionTokenLogprob`` dicts."""

    def name(item: dict) -> str:
        return f"token_id:{item['token_id']}" if as_token_ids else item["token"]

    out: list[dict[str, Any]] = []
    for entry in entries:
        out.append(
            {
                "token": name(entry),
                "logprob": _clamped(entry["logprob"]),
                "bytes": entry["bytes"],
                "top_logprobs": [
                    {
                        "token": name(cand),
                        "logprob": _clamped(cand["logprob"]),
                        "bytes": cand["bytes"],
                    }
                    for cand in entry["top"]
                ],
            }
        )
    return out


def responses_logprobs_content(entries: list[dict]) -> list[dict[str, Any]]:
    """Format neutral engine logprob entries into OpenAI Responses ``output_text`` logprob dicts."""
    return [
        {
            "token": entry["token"],
            "logprob": _clamped(entry["logprob"]),
            "bytes": entry["bytes"],
            "top_logprobs": [
                {
                    "token": cand["token"],
                    "logprob": _clamped(cand["logprob"]),
                    "bytes": cand["bytes"],
                }
                for cand in entry["top"]
            ],
        }
        for entry in entries
    ]


def completions_logprobs(
    entries: list[dict],
    offsets: list[int],
    text_len: int | None = None,
    as_token_ids: bool = False,
) -> dict:
    """``offsets[i]`` is where entry i's token starts in the generated text (the pieces of
    a split multi-byte character share its start). Given the finished text's ``text_len``,
    offsets are clipped to it, since a matched stop string trims tokens off the end.
    ``as_token_ids`` (vLLM return_tokens_as_token_ids) names every token ``token_id:N``
    instead of by its text."""

    def name(item: dict) -> str:
        return f"token_id:{item['token_id']}" if as_token_ids else item["token"]

    tokens: list[str] = []
    token_logprobs: list[float] = []
    top_logprobs: list[dict[str, float]] = []

    for entry in entries:
        tokens.append(name(entry))
        token_logprobs.append(_clamped(entry["logprob"]))
        top: dict[str, float] = {}
        for candidate in entry["top"]:
            # The legacy shape keys alternatives by text, and partial-UTF-8 candidates
            # all decode to U+FFFD: keep the most likely one per key.
            key, logprob = name(candidate), _clamped(candidate["logprob"])
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
