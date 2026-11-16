"""Shared driver for the hybrid (GDN) snapshot-placement tests: one prompt's chunked prefill,
scheduled and drained as the real scheduler does, on CPU managers. Holds no tests itself."""

from __future__ import annotations

from collections.abc import Callable
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
