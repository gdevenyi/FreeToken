"""Shared driver for the hybrid (GDN) snapshot-placement tests: one prompt's chunked prefill,
scheduled and drained as the real scheduler does, on CPU managers; and, for the host-bank tests, a
stand-in GDN state, the restore check and a copier that copies late. Holds no tests itself."""

from __future__ import annotations

import zlib
from collections.abc import Callable
from types import SimpleNamespace
from typing import NamedTuple

import torch
from freetoken.attention.linear import build_fla_metadata
from freetoken.core import Batch, Req
from freetoken.scheduler.prefill import ChunkedReq


class PrefillRun(NamedTuple):
    final: Req                         # the final prompt chunk, committed with finished=False
    cached: int                        # the prefix-cache hit at admission
    tracks: list[tuple[int, int]]      # (tracked boundary, end of the chunk that tracked it)


def drive_prefill(cm, pm, pool, pending, *, budget: int, overlap: bool,
                  on_drain: Callable[[Req], None] | None = None,
                  on_launch: Callable[[Batch], None] | None = None,
                  on_forward: Callable[[Req, bool], None] | None = None) -> PrefillRun:
    """Prefill ``pending`` chunk by chunk: allocate pages, build the GDN track metadata (which
    advances the track cursor as a forward does), complete the chunk, then drain it as
    Scheduler._process_last_data does (commit_chunk for an intermediate chunk, cache_req for the
    final one). ``overlap`` schedules the next chunk before the previous one drains (no spec);
    otherwise each chunk drains before the next is scheduled (normal_loop: DFlash and MTP).
    ``on_launch(batch)`` runs where the scheduler restores GDN states (after the allocation, before
    the forward); ``on_forward(req, tracked)`` stands in for the forward's writes to the GDN slots."""
    pm.pending_list = [pending]
    tracks, cached, final, waiting = [], None, None, None
    while pm.runnable or waiting is not None:
        batch = pm.schedule_next_batch(budget)
        if batch is not None:
            batch.padded_reqs = batch.reqs
            cm.allocate_paged(batch.reqs)
            (req,) = batch.reqs
            if on_launch is not None:
                on_launch(batch)
            tracked = build_fla_metadata(batch, torch.device("cpu"), pool).track_dst is not None
            if tracked:
                tracks.append((req.mamba_last_track_seqlen, req.device_len))
            if on_forward is not None:
                on_forward(req, tracked)
            if batch.prompt_admissions:
                cached = batch.prompt_admissions[0][2]
            req.complete_one()
        drained, waiting = ([waiting], batch) if overlap else ([batch], None)
        for done in filter(None, drained):
            (r,) = done.reqs
            if isinstance(r, ChunkedReq):
                cm.commit_chunk(r)
            else:
                cm.cache_req(r, finished=False)
                final = r
            if on_drain is not None:
                on_drain(r)
    return PrefillRun(final, cached, tracks)


def gdn_state(ids: torch.Tensor) -> float:
    """A stand-in GDN state: a function of the tokens it has consumed, exact in fp32."""
    return float(zlib.crc32(ids.numpy().tobytes()) % 1_000_003)


def forward_writes(pool):
    """``on_forward`` for drive_prefill: the forward leaves the live slot at the state of the whole
    chunk and, when it tracks, the frozen ping-pong slot at the state of the tracked boundary."""
    def write(req, tracked):
        pool.recurrent_states[:, req.linear_slot_idx] = gdn_state(req.input_ids[: req.device_len])
        if tracked:
            frozen = req.mamba_ping_pong[1 - req.mamba_next_track_idx]
            pool.recurrent_states[:, frozen] = gdn_state(req.input_ids[: req.mamba_last_track_seqlen])
    return write


def restore_and_check(pool, admitted: list):
    """``on_launch`` for drive_prefill: run the scheduler's restore, then check that the live slot of
    each resumed request holds the state of exactly its cached prefix, from either tier."""
    from freetoken.scheduler.scheduler import Scheduler

    stub = SimpleNamespace(engine=SimpleNamespace(linear_state_pool=pool))

    def launch(batch):
        kinds = [("device" if r.mamba_restore_src is not None else
                  "host" if r.mamba_restore_host is not None else None) for r in batch.reqs]
        Scheduler._restore_linear_states(stub, batch)
        for r, kind in zip(batch.reqs, kinds):
            if r.cached_len == r.cache_handle.cached_len and r.cached_len > 0 and kind is not None:
                assert pool.recurrent_states[0, r.linear_slot_idx, 0, 0, 0].item() == gdn_state(
                    r.input_ids[: r.cached_len]), f"{kind} restore of {r.cached_len} tokens holds another state"
                admitted.append((r.uid, r.cached_len, kind))
            elif r.cached_len == r.cache_handle.cached_len:
                admitted.append((r.uid, r.cached_len, kind))
    return launch


class CopiesAtTheFence:
    """A host-bank copier whose device->host copies run only at the next fence: the latest point
    a real stream could run them. A slot written before its copy ran would reach the bank wrong."""

    def __init__(self):
        self.queue = []

    def to_host(self, pairs):
        self.queue.extend(pairs)

    def from_host(self, pairs):
        for dst, src in pairs:
            dst.copy_(src)

    def fence(self):
        for dst, src in self.queue:
            dst.copy_(src)
        self.queue.clear()
