"""Scheduler-side stop-string matching (Scheduler._match_stop_str) against the same
incrementally decoded text the frontend streams."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from freetoken.core import Req, SamplingParams
from freetoken.scheduler.scheduler import Scheduler

EOS, SPECIAL = 0, 1


class _ByteTok:
    """Byte-level ids like Qwen's BPE fallback: a 4-byte emoji is four tokens. 0 is EOS,
    1 a special marker that skip_special_tokens drops."""

    def decode(self, ids, skip_special_tokens=False):
        def piece(t):
            if t == EOS:
                return b""
            if t == SPECIAL:
                return b"" if skip_special_tokens else b"<s>"
            return bytes([t])

        return b"".join(piece(t) for t in ids).decode("utf-8", errors="replace")


def _req(prompt: list[int], output_len: int, **sp) -> Req:
    return Req(
        input_ids=torch.tensor(prompt, dtype=torch.int32),
        table_idx=0,
        cached_len=0,
        output_len=output_len,
        uid=1,
        sampling_params=SamplingParams(**sp),
        cache_handle=None,
    )


def _match(req: Req) -> str | None:
    stub = SimpleNamespace(tokenizer=_ByteTok(), eos_token_ids={EOS})
    return Scheduler._match_stop_str(stub, req)


def _feed(req: Req, ids: list[int]) -> list[str | None]:
    seen = []
    for t in ids:
        req.append_host(torch.tensor([t], dtype=torch.int32))
        req.complete_one()
        seen.append(_match(req))
    return seen


def test_a_stop_spanning_more_tokens_than_characters_is_matched():
    dna = list("\N{DNA DOUBLE HELIX}".encode())  # one character, four byte tokens
    req = _req([65, 66], 32, stop_strs=["\N{DNA DOUBLE HELIX}"])
    text = list(b"Here is DNA ") + dna + list(b" and more")
    seen = _feed(req, text)
    first = next(i for i, m in enumerate(seen) if m is not None)
    assert first == len(b"Here is DNA ") + len(dna) - 1


def test_an_eos_short_of_the_delivered_limit_stays_in_the_matched_text():
    # Overlap scheduling advances device_len ahead of the host: the budget is spent on the
    # device (can_decode is False) while the host holds one token less than the limit. That
    # EOS is not terminal, so the frontend decodes it and the scheduler must too.
    req = _req([65], 3, stop_strs=["zz"], ignore_eos=True)
    for t in (65, EOS):
        req.append_host(torch.tensor([t], dtype=torch.int32))
    while req.can_decode:
        req.complete_one()
    assert req.input_ids.numel() < req.max_device_len
    assert _match(req) is None
    assert req.stop_decode_status.decoded_ids == [65, EOS]


def test_stops_match_the_text_decoded_with_the_requests_skip_special_tokens():
    # the frontend streams "ab" for a request that skips special tokens; the scheduler has
    # to match the stop in that same text, not in "a<s>b"
    req = _req([65], 8, stop_strs=["ab"], skip_special_tokens=True)
    assert _feed(req, [ord("a"), SPECIAL, ord("b")]) == [None, None, "ab"]
    req = _req([65], 8, stop_strs=["ab"])
    assert _feed(req, [ord("a"), SPECIAL, ord("b")]) == [None, None, None]
