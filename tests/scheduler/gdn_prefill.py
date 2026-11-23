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


def forward_writes(pool, cm=None):
    """``on_forward`` for drive_prefill: the forward leaves the live slot at the state of the whole
    chunk and, when it tracks, the frozen ping-pong slot at the state of the tracked boundary. With
    a KV host tier on ``cm`` it also writes a stand-in KV payload (token, position) into the rope rows
    of the chunk's slots, so a restore can show that each page holds its own tokens."""
    def write(req, tracked):
        pool.recurrent_states[:, req.linear_slot_idx] = gdn_state(req.input_ids[: req.device_len])
        if tracked:
            frozen = req.mamba_ping_pong[1 - req.mamba_next_track_idx]
            pool.recurrent_states[:, frozen] = gdn_state(req.input_ids[: req.mamba_last_track_seqlen])
        if cm is not None and cm.kv_host is not None:
            slots = cm.page_table[req.table_idx, req.cached_len : req.device_len].long()
            rope = cm.kv_host._rope_positions
            rope[slots, 0] = req.input_ids[req.cached_len : req.device_len]
            rope[slots, 1] = torch.arange(req.cached_len, req.device_len, dtype=torch.int32)
    return write


def restore_and_check(pool, admitted: list, cm=None):
    """``on_launch`` for drive_prefill: run the scheduler's restore, then check that the live slot of
    each resumed request holds the state of exactly its cached prefix, from either tier. With a KV
    host tier on ``cm`` the page-table row of the prefix must hold the payload of its own tokens too,
    and a KV restore from the host adds "+kv" to the kind."""
    from freetoken.scheduler.scheduler import Scheduler

    kv = None if cm is None else cm.kv_host
    stub = SimpleNamespace(engine=SimpleNamespace(linear_state_pool=pool, kv_cache=kv))

    def launch(batch):
        kinds = [("device" if r.mamba_restore_src is not None else
                  "host" if r.mamba_restore_host is not None else None) for r in batch.reqs]
        kinds = [k if getattr(r, "kv_restore", None) is None else f"{k}+kv" for r, k in zip(batch.reqs, kinds)]
        Scheduler._restore_linear_states(stub, batch)
        for r, kind in zip(batch.reqs, kinds):
            if kv is not None and r.cached_len == r.cache_handle.cached_len and r.cached_len > 0:
                rows = kv._rope_positions[cm.page_table[r.table_idx, : r.cached_len].long()]
                assert torch.equal(rows[:, 0], r.input_ids[: r.cached_len]) and torch.equal(
                    rows[:, 1], torch.arange(r.cached_len, dtype=torch.int32)
                ), f"{kind} restore of {r.cached_len} tokens: a page holds other tokens"
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
        self.copied = 0                # copies taken, so a test can tell the copier was used

    def to_host(self, pairs):
        self.queue.extend(pairs)
        self.copied += len(pairs)

    def from_host(self, pairs):
        for dst, src in pairs:
            dst.copy_(src)

    def fence(self):
        for dst, src in self.queue:
            dst.copy_(src)
        self.queue.clear()
