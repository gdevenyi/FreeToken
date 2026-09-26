# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The FlashLib Authors
# Adapted from flashlib 0.3.0 (kernels/slot_cache/triton/lru_ensure.py: _phase1, _stats,
# _lru_ensure_kernel, _num_warps_for), https://github.com/FlashML-org/flashlib
"""Scored-victim slot-cache admission for the MoE expert cache (``--moe-cache-policy``).

flashlib's sequential-argmin ``lru_ensure`` with the victim key made pluggable. POLICY 0 is
LRU and matches flashlib bit for bit; the others evict the lowest packed score key:

- 1 ``kd``: ``-(k + beta * d / L)``. ``k`` is the tokens the expert has been passed over
  since its last use, ``d`` the layer-steps until its layer comes up again.
- 2 ``kdfb``: kd plus ``w * log2(decayed use count)`` with a ``halflife``-token decay.
- 3 ``rule``: kdfb plus the near-miss refresh: this layer's experts whose router logit is
  within a margin of the top-k-th count as used one token ago (``last_tok``), unpinned.

The per-(layer, expert) state survives eviction; slot-indexed mirrors of it (tagged with the
owning id) let the victim scan read it coalesced instead of gathering by id. Scores and
counts are int32 Q16 fixed point and the count update ``log2(2^x + 1)`` reads a host-built
table, so a CPU reference can reproduce every victim exactly (``tl.exp2``/``tl.log2`` are
approximate).
"""
from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

POLICY_IDS = {"lru": 0, "kd": 1, "kdfb": 2, "rule": 3}

# Replay-tuned on sep5_s01 at 1650 slots; the plateau is flat for beta 1-8, w 2-4, halflife 32-128.
BETA = 4.0
W = 3.0
HALFLIFE = 64

Q = 16  # fraction bits of scores and counts
LAST_TOK_NEVER = -1_000_000
LC_NEVER = -(1 << 30)
LC_FLOOR = -(30 << Q)
K_MAX = 1 << 14  # tokens; with the caps on beta and w below, a score fits int32
MAX_BETA, MAX_W = 64.0, 16.0
DECAY_SHIFT = 8
SCORE_BIAS = 1 << 31
MAX_SLOT_BITS = 30
# g(u) = log2(1 + 2^-u) sampled every 2^-G_STEP_BITS for u in [0, G_RANGE]; zero beyond
G_STEP_BITS = 6
G_RANGE = 24
G_LAST = G_RANGE << G_STEP_BITS
# jit code may only read globals that are constexpr
_Q = tl.constexpr(Q)
_LC_NEVER = tl.constexpr(LC_NEVER)
_LC_FLOOR = tl.constexpr(LC_FLOOR)
_K_MAX = tl.constexpr(K_MAX)
_DECAY_SHIFT = tl.constexpr(DECAY_SHIFT)
_SCORE_BIAS = tl.constexpr(SCORE_BIAS)
_G_SHIFT = tl.constexpr(Q - G_STEP_BITS)
_G_LAST = tl.constexpr(G_LAST)


def softplus2_table(device=None) -> torch.Tensor:
    """Q16 ``log2(1 + 2^-u)`` at ``u = i / 64``, two entries past ``G_LAST`` for the interpolation."""
    step = 1 << G_STEP_BITS
    vals = [round(math.log2(1.0 + 2.0 ** (-i / step)) * (1 << Q)) for i in range(G_LAST + 2)]
    return torch.tensor(vals, dtype=torch.int32, device=device)


def score_params(beta: float, w: float, halflife: int, num_layers: int) -> tuple[int, int, int, int]:
    """``(BETA_STEP, W_Q4, DECAY_MUL, DT_MAX)`` shared by the kernel and the CPU reference.

    ``beta * d / L`` is ``d * BETA_STEP`` in Q16; the decay over ``dt`` layer-steps is
    ``(dt * DECAY_MUL) >> DECAY_SHIFT``, with ``dt`` capped at 64 half-lives (past the -30 floor).
    """
    assert 0 <= beta <= MAX_BETA and 0 <= w <= MAX_W and halflife >= 1
    span = num_layers * int(halflife)
    decay_mul = round((1 << (Q + DECAY_SHIFT)) / span)
    dt_max = 64 * span
    assert dt_max * decay_mul < (1 << 31)
    return round(beta * (1 << Q) / num_layers), round(w * 16), decay_mul, dt_max


@triton.jit
def _phase1(query_ptr, slot_of_id_ptr, lru_usage_ptr, num_copy_ptr, step, K,
            BLOCK_K: tl.constexpr, id_base):
    """Vendored unchanged: dedup the query, split hit/miss, rank the misses, bump the hits."""
    k = tl.arange(0, BLOCK_K)
    kmask = k < K
    q = tl.load(query_ptr + k, mask=kmask, other=-1) + id_base
    s = tl.load(slot_of_id_ptr + q, mask=kmask, other=-1)
    hit = kmask & (s >= 0)
    miss = kmask & (s == -1)
    same = (
        (q[:, None] == q[None, :])
        & (k[:, None] > k[None, :])
        & kmask[:, None]
        & kmask[None, :]
    )
    first = kmask & (tl.sum(same.to(tl.int32), axis=1) == 0)
    first_miss = miss & first
    smaller = (q[None, :] < q[:, None]) & first_miss[None, :]
    rank = tl.sum(smaller.to(tl.int32), axis=1)
    num_missing = tl.sum(first_miss.to(tl.int32))
    tl.store(num_copy_ptr, num_missing.to(tl.int64))
    # Duplicated hits write the same value to the same slot -- idempotent.
    tl.store(lru_usage_ptr + s, step, mask=hit)
    return q, kmask, miss, first, first_miss, rank, num_missing, tl.where(hit, s, -1)


@triton.jit
def _prefetch_phase1(query_ptr, slot_of_id_ptr, K, BLOCK_K: tl.constexpr, id_base):
    """_phase1 for a prefetch: -1 entries are padding, hits are left alone (a prefetch is not an
    access), and the misses rank in query order so the select's best candidates install first."""
    k = tl.arange(0, BLOCK_K)
    raw = tl.load(query_ptr + k, mask=k < K, other=-1)
    kmask = raw >= 0
    q = tl.where(kmask, raw + id_base, 0)
    s = tl.load(slot_of_id_ptr + q, mask=kmask, other=0)
    miss = kmask & (s == -1)
    same = (q[:, None] == q[None, :]) & (k[:, None] > k[None, :]) & kmask[:, None] & kmask[None, :]
    first_miss = miss & (tl.sum(same.to(tl.int32), axis=1) == 0)
    rank = tl.cumsum(first_miss.to(tl.int32), axis=0) - 1
    return q, first_miss, rank, tl.sum(first_miss.to(tl.int32))


@triton.jit
def _stats(stats_ptr, first, num_missing):
    # One vectorized atomic over 3 lanes, not three scalar ones: a scalar atomic serializes the CTA.
    si = tl.arange(0, 4)
    v = tl.where(si == 0, tl.sum(first.to(tl.int32)), tl.where(si == 1, num_missing, 1))
    # relaxed: only later kernels read the counters; flashlib's acq_rel waits on this call's stores (~0.3 us)
    tl.atomic_add(stats_ptr + si, v.to(tl.int64), mask=si < 3, sem="relaxed")


@triton.jit
def _decayed(lc, ct, now, DECAY_MUL: tl.constexpr, DT_MAX: tl.constexpr):
    """int32 Q16 log2 count decayed from ``ct`` to ``now`` (int64 layer-steps)."""
    dt = tl.minimum(tl.maximum(now - ct, 0), DT_MAX).to(tl.int32)
    return lc - ((dt * DECAY_MUL) >> _DECAY_SHIFT)


@triton.jit
def _softplus2(x, g_ptr, mask):
    """Q16 ``log2(2^x + 1)`` = ``max(x, 0) + g(|x|)``, g linearly interpolated from the table."""
    u = tl.abs(x)
    i = tl.minimum(u >> _G_SHIFT, _G_LAST)
    frac = u & ((1 << _G_SHIFT) - 1)
    g0 = tl.load(g_ptr + i, mask=mask, other=0)
    g1 = tl.load(g_ptr + i + 1, mask=mask, other=0)
    g = tl.where(i < _G_LAST, g0 + (((g1 - g0) * frac) >> _G_SHIFT), 0)
    return tl.maximum(x, 0) + g


@triton.jit
def _owner_state(mirror_ptr, state_ptr, c, oid, mirrored, stale, other):
    """The slot owner's state: its coalesced mirror, or a gather by id where the mirror is stale."""
    return tl.where(
        mirrored,
        tl.load(mirror_ptr + c, mask=mirrored, other=other),
        tl.load(state_ptr + oid, mask=stale, other=other),
    )


@triton.jit(do_not_specialize=["K", "num_cached", "id_base", "nm_topk", "nm_stride", "pf_rows"])
def _scored_ensure_kernel(
    query_ptr, slot_of_id_ptr, id_of_slot_ptr, lru_usage_ptr, lru_step_ptr,
    out_ptr, src_ptr, dst_ptr, num_copy_ptr, stats_ptr,
    tok_ptr, last_tok_ptr, lc_ptr, ct_ptr, g_ptr, logits_ptr,
    owner_ptr, slot_last_ptr, slot_lc_ptr, slot_ct_ptr, pin_ptr,
    pf_stats_ptr, pf_ready_ptr, pf_num_ptr,
    K, num_cached, id_base, nm_topk, nm_stride, nm_thr, pf_rows,
    BLOCK_K: tl.constexpr, BLOCK_C: tl.constexpr, BLOCK_E: tl.constexpr, BLOCK_TOPK: tl.constexpr,
    NM_ROWS: tl.constexpr,
    USAGE_MAX: tl.constexpr, COLLECT_STATS: tl.constexpr,
    POLICY: tl.constexpr, BUMP_TOK: tl.constexpr, UPDATE_STATE: tl.constexpr, PIN_SINCE: tl.constexpr,
    NUM_LAYERS: tl.constexpr, NUM_EXPERTS: tl.constexpr,
    BETA_STEP: tl.constexpr, W_Q4: tl.constexpr, DECAY_MUL: tl.constexpr, DT_MAX: tl.constexpr,
    SLOT_BITS: tl.constexpr,
    LOWPRI: tl.constexpr, PF_COUNT: tl.constexpr, PREFETCH: tl.constexpr,
):
    if PREFETCH:
        # not an access: the clock stays, so usage == step still pins the last demand call's slots
        # (at step 0, right after a reset, every slot reads pinned: nothing installs before a demand call)
        step = tl.load(lru_step_ptr)
        tl.store(num_copy_ptr, tl.zeros([], tl.int64))
        tl.store(pf_ready_ptr, 0)
    else:
        step = tl.load(lru_step_ptr) + 1
        tl.store(lru_step_ptr, step)
    if POLICY != 0:
        tok = tl.load(tok_ptr)
        if BUMP_TOK:
            tok = tok + 1
            tl.store(tok_ptr, tok)
        layer = id_base // NUM_EXPERTS
        if PREFETCH:
            # score from where the pass is, right after layer - 1's demand call: the target layer's
            # residents are next up, not a whole pass away (else it evicts what the layer is about to route)
            layer = layer - 1
        now = tok * NUM_LAYERS + layer
        if POLICY == 3:
            if UPDATE_STATE:
                # Routed ids qualify too; they are pinned now and set to tok below, so that is harmless.
                ex = tl.arange(0, BLOCK_E)
                j = tl.arange(0, BLOCK_TOPK)
                prev = tl.load(last_tok_ptr + id_base + ex, mask=ex < NUM_EXPERTS, other=0)
                held_at = tl.load(slot_of_id_ptr + id_base + ex, mask=ex < NUM_EXPERTS, other=-1)
                near = ex < 0
                for r in tl.static_range(NM_ROWS):
                    routed = tl.load(query_ptr + r * nm_topk + j, mask=j < nm_topk, other=-1)
                    row_ptr = logits_ptr + r * nm_stride
                    kth = tl.min(tl.load(row_ptr + routed, mask=routed >= 0, other=float("inf")).to(tl.float32), axis=0)
                    row = tl.load(row_ptr + ex, mask=ex < NUM_EXPERTS, other=float("-inf")).to(tl.float32)
                    near = near | (row >= kth - nm_thr)
                refreshed = tl.maximum(prev, tok - 1)
                tl.store(last_tok_ptr + id_base + ex, refreshed, mask=near)
                tl.store(slot_last_ptr + held_at, refreshed, mask=near & (held_at >= 0))
                # the victim scan and the routed update below read and rewrite what this wrote
                tl.debug_barrier()
    if PREFETCH:
        q, first_miss, rank, num_missing = _prefetch_phase1(query_ptr, slot_of_id_ptr, K, BLOCK_K, id_base)
    else:
        if PF_COUNT:
            kk = tl.arange(0, BLOCK_K)
            sk = tl.load(slot_of_id_ptr + tl.load(query_ptr + kk, mask=kk < K, other=0) + id_base, mask=kk < K, other=-1)
            # a hit on a held slot with usage 0 consumes a prefetch; read it before _phase1 bumps it
            was_pf = (sk >= 0) & (tl.load(lru_usage_ptr + sk, mask=sk >= 0, other=1) == 0)
            tl.debug_barrier()
        q, kmask, miss, first, first_miss, rank, num_missing, out = _phase1(
            query_ptr, slot_of_id_ptr, lru_usage_ptr, num_copy_ptr, step, K, BLOCK_K,
            id_base)

    if num_missing > 0:
        # REQUIRED (flashlib): the reload below must see _phase1's hit bump, or a hit is evicted.
        tl.debug_barrier()
        c = tl.arange(0, BLOCK_C)
        cmask = c < num_cached
        u = tl.load(lru_usage_ptr + c, mask=cmask, other=USAGE_MAX)
        if POLICY == 0:
            umax = tl.full([BLOCK_C], USAGE_MAX, u.dtype)
            if LOWPRI:
                held0 = cmask & (tl.load(id_of_slot_ptr + c, mask=cmask, other=-1) >= 0)
                # empty slots, then unconsumed prefetches (held, usage 0), then residents by recency
                u = tl.where((u == step) | (~cmask), umax, tl.where(held0, u + 1, 0))
            else:
                u = tl.where((u == step) | (~cmask), umax, u)
            n_free = tl.sum((u != umax).to(tl.int32))
        else:
            oid = tl.load(id_of_slot_ptr + c, mask=cmask, other=-1)
            held = cmask & (oid >= 0)
            owner = tl.load(owner_ptr + c, mask=cmask, other=-1)
            mirrored = held & (owner == oid)
            stale = held & (owner != oid)
            lk = oid // NUM_EXPERTS
            ahead = lk > layer
            d = tl.where(ahead, lk - layer, NUM_LAYERS - layer + lk)
            last = _owner_state(slot_last_ptr, last_tok_ptr, c, oid, mirrored, stale, 0)
            k = tl.minimum(tl.maximum(tl.where(ahead, tok - 1, tok) - last, 0), _K_MAX).to(tl.int32)
            # int32 throughout: |score| < 2^31 by the caps on k, beta and w
            score = -((k << _Q) + d * BETA_STEP)
            if POLICY >= 2:
                lc = _owner_state(slot_lc_ptr, lc_ptr, c, oid, mirrored, stale, _LC_NEVER)
                ct = _owner_state(slot_ct_ptr, ct_ptr, c, oid, mirrored, stale, 0)
                lcv = _decayed(lc, ct, now, DECAY_MUL, DT_MAX)
                lcv = tl.where(lc == _LC_NEVER, _LC_FLOOR, tl.maximum(lcv, _LC_FLOOR))
                score += (W_Q4 * lcv) >> 4
            # Every key is distinct and ties go to the lowest slot; an empty slot keys below any held one.
            key = tl.where(held, ((score.to(tl.int64) + _SCORE_BIAS) << SLOT_BITS) | c, c.to(tl.int64))
            if LOWPRI:
                # an unconsumed prefetch (held, usage 0) keys above every empty slot and below every
                # resident: the caps keep score + 2^31 > 2^30 - 2^25 - 2^22, far above this band's 1
                key = tl.where(held & (u == 0), (1 << SLOT_BITS) | c.to(tl.int64), key)
            evictable = cmask & (u != step)
            if PIN_SINCE:
                # the score ignores recency, so slots an earlier call touched since *pin_ptr stay pinned too
                evictable = evictable & (u <= tl.load(pin_ptr))
            # the pinned key's slot bits are 0, so an all-pinned scan falls back to slot 0 like flashlib's argmin
            pinned_key = 0x7FFFFFFFFFFFFFFF ^ ((1 << SLOT_BITS) - 1)
            key = tl.where(evictable, key, pinned_key)
            n_free = tl.sum(evictable.to(tl.int32))
        if PREFETCH:
            # never the all-pinned fallback: that slot may be under the running GEMM
            n_iter = tl.minimum(num_missing, n_free)
            tl.store(num_copy_ptr, n_iter.to(tl.int64))
            tl.store(pf_stats_ptr, tl.load(pf_stats_ptr) + n_iter)
        else:
            n_iter = num_missing
        for i in tl.range(n_iter):
            if POLICY == 0:
                victim = tl.argmin(u, axis=0).to(tl.int32)
            else:
                victim = (tl.min(key, axis=0) & ((1 << SLOT_BITS) - 1)).to(tl.int32)
            # Scalar load: victims are distinct, so no earlier iteration wrote this slot.
            old = tl.load(id_of_slot_ptr + victim)
            if old >= 0:
                tl.store(slot_of_id_ptr + old, -1)
            e = tl.sum(tl.where((rank == i) & first_miss, q, 0))
            tl.store(id_of_slot_ptr + victim, e)
            tl.store(slot_of_id_ptr + e, victim)
            if PREFETCH:
                # usage 0 on a held slot marks it low priority until a demand call touches it
                tl.store(lru_usage_ptr + victim, 0)
                if POLICY != 0:
                    tl.store(owner_ptr + victim, -1)
            else:
                tl.store(lru_usage_ptr + victim, step)
            tl.store(dst_ptr + i, victim)
            tl.store(src_ptr + i, e - id_base)  # back to the caller's id space
            if not PREFETCH:
                out = tl.where((rank == i) & miss, victim, out)
            if POLICY == 0:
                u = tl.where(c == victim, umax, u)  # claim in-register
            else:
                key = tl.where(c == victim, pinned_key, key)

    if not PREFETCH:
        # Written from registers, never re-read from slot_of_id, so out_ptr may alias query_ptr.
        tl.store(out_ptr + tl.arange(0, BLOCK_K), out, mask=kmask)
    if PF_COUNT:
        # one writer per column: pf_ensure adds ISSUED (column 0) on its own stream, ordered before this
        pn = tl.load(pf_num_ptr)
        late = (pn > 0) & (tl.load(pf_ready_ptr, volatile=True) == 0)
        col = tl.arange(0, 8)
        add = tl.where(col == 1, tl.sum((first & was_pf).to(tl.int32)).to(tl.int64), 0)
        add = tl.where(col == 3, 1, add)
        add = tl.where(col == 4, pf_rows, add)
        add = tl.where(col == 5, num_missing.to(tl.int64), add)
        add = tl.where(col == 6, late.to(tl.int64), add)
        add = tl.where(col == 7, (pn > 0).to(tl.int64), add)
        cols = (col == 1) | (col >= 3)
        tl.store(pf_stats_ptr + col, tl.load(pf_stats_ptr + col, mask=cols, other=0) + add, mask=cols)
    if POLICY != 0:
        if UPDATE_STATE:
            # distinct ids only, so a duplicate at bs > 1 counts once per call; out is each id's final slot
            tl.store(last_tok_ptr + q, tok, mask=first)
            tl.store(owner_ptr + out, q, mask=first)
            tl.store(slot_last_ptr + out, tok, mask=first)
            if POLICY >= 2:
                lc = tl.load(lc_ptr + q, mask=first, other=_LC_NEVER)
                x = _decayed(lc, tl.load(ct_ptr + q, mask=first, other=0), now, DECAY_MUL, DT_MAX)
                lc_new = tl.where(lc == _LC_NEVER, 0, _softplus2(x, g_ptr, first))
                tl.store(lc_ptr + q, lc_new, mask=first)
                tl.store(ct_ptr + q, now, mask=first)
                tl.store(slot_lc_ptr + out, lc_new, mask=first)
                tl.store(slot_ct_ptr + out, now, mask=first)
        elif not PREFETCH:
            # a slot filled without a state update reads the per-id state until a decode hit re-mirrors it
            tl.store(owner_ptr + out, -1, mask=first_miss)
    if COLLECT_STATS:
        _stats(stats_ptr, first, num_missing)


def _num_warps_for(block_c: int) -> int:
    # flashlib's rule: 8 warps from 2048 slots, more only past ~32 elements per thread.
    warps = 8 if block_c >= 2048 else 4
    return max(warps, min(triton.next_power_of_2(max(block_c // 1024, 1)), 32))


def scored_ensure(
    query: torch.Tensor,
    slot_of_id: torch.Tensor,
    id_of_slot: torch.Tensor,
    lru_usage: torch.Tensor,
    lru_step: torch.Tensor,
    out_indices: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    num_copy: torch.Tensor,
    *,
    policy: int,
    num_layers: int,
    num_experts: int,
    tok: torch.Tensor | None = None,
    last_tok: torch.Tensor | None = None,
    lc: torch.Tensor | None = None,
    ct: torch.Tensor | None = None,
    g_table: torch.Tensor | None = None,
    slot_owner: torch.Tensor | None = None,
    slot_last_tok: torch.Tensor | None = None,
    slot_lc: torch.Tensor | None = None,
    slot_ct: torch.Tensor | None = None,
    router_logits: torch.Tensor | None = None,
    near_miss_thr: float = 0.0,
    stats: torch.Tensor | None = None,
    id_base: int = 0,
    bump_tok: bool = False,
    update_state: bool = True,
    pin_since: torch.Tensor | None = None,
    beta: float = BETA,
    w: float = W,
    halflife: int = HALFLIFE,
    lowpri: bool = False,
    pf_stats: torch.Tensor | None = None,
    pf_ready: torch.Tensor | None = None,
    pf_num: torch.Tensor | None = None,
    pf_rows: int = 0,
    prefetch: bool = False,
) -> None:
    """``flashlib.lru_ensure`` (sequential strategy) with the victim picked by ``policy``.

    Same contract as ``lru_ensure``; ``policy`` > 0 also reads and updates the per-(layer,
    expert) state ``last_tok``/``lc``/``ct`` (flat ``layer * num_experts + expert``) against
    the device token counter ``tok``, and keeps its ``[num_cached]`` mirrors ``slot_last_tok``/
    ``slot_lc``/``slot_ct`` valid where ``slot_owner`` equals ``id_of_slot`` (-1 marks a stale
    mirror). ``bump_tok`` advances the counter first (once per decode step);
    ``update_state=False`` leaves the state untouched (small-prefill ensures).
    ``pin_since`` (a ``lru_step`` value) also pins every slot touched after it, so a small
    prefill's chunked calls cannot evict each other's experts; policy > 0 only.
    Policy 3 refreshes the experts whose ``router_logits`` (``[rows, num_experts]``, rows of
    ``query``) are within ``near_miss_thr`` of their row's lowest routed logit.

    FREETOKEN_MOE_PREFETCH=on: ``lowpri`` keys held slots with usage 0 (unconsumed prefetches)
    between the empty slots and every resident. ``pf_stats`` (a ``[8]`` int64 prefetch stats row)
    also counts this demand call: hits on such slots, the call, ``pf_rows``, the misses, and whether
    the prefetch plan ``pf_num`` had rows whose copy had not set ``pf_ready`` yet.
    ``prefetch=True`` is :func:`prefetch_ensure`'s install mode instead.
    """
    if prefetch:
        assert lowpri and pf_stats is not None and pf_ready is not None and stats is None
        assert not update_state and not bump_tok and pin_since is None and router_logits is None
    if pf_stats is not None:
        assert lowpri and pf_stats.dtype == torch.int64 and pf_stats.numel() == 8 and pf_stats.is_contiguous()
        assert pf_ready is not None and pf_ready.dtype == torch.int32 and pf_ready.numel() == 1
        assert prefetch or (pf_num is not None and pf_num.dtype == torch.int64 and pf_num.numel() == 1)
    k = query.numel()
    num_cached = id_of_slot.numel()
    assert query.dtype == torch.int32 and query.is_contiguous()
    assert slot_of_id.dtype == torch.int32 and id_of_slot.dtype == torch.int32
    assert out_indices.dtype == torch.int32 and out_indices.numel() == k
    assert lru_usage.dtype == lru_step.dtype and lru_usage.numel() == num_cached
    plan = min(k, num_cached)
    assert src_indices.numel() >= plan and dst_indices.numel() >= plan
    block_c = triton.next_power_of_2(num_cached)
    slot_bits = max(block_c.bit_length() - 1, 1)
    beta_step, w_q4, decay_mul, dt_max = score_params(beta, w, halflife, num_layers)
    if policy != 0:
        assert slot_bits <= MAX_SLOT_BITS, f"{num_cached} slots overflow the packed score key"
        assert tok is not None and last_tok is not None
        assert last_tok.dtype == torch.int64 and last_tok.numel() == num_layers * num_experts
        assert slot_owner is not None and slot_last_tok is not None
        assert slot_owner.dtype == torch.int32 and slot_owner.numel() == num_cached
        assert slot_last_tok.dtype == torch.int64 and slot_last_tok.numel() == num_cached
    if pin_since is not None:
        assert policy != 0 and pin_since.dtype == lru_step.dtype and pin_since.numel() == 1
    if policy >= 2:
        assert lc is not None and ct is not None and g_table is not None
        assert lc.dtype == torch.int32 and ct.dtype == torch.int64
        assert slot_lc is not None and slot_lc.dtype == torch.int32 and slot_lc.numel() == num_cached
        assert slot_ct is not None and slot_ct.dtype == torch.int64 and slot_ct.numel() == num_cached
    nm_rows = nm_topk = nm_stride = 0
    if policy == 3 and router_logits is not None:
        assert router_logits.ndim == 2 and router_logits.shape[1] == num_experts
        assert router_logits.stride(1) == 1 and k % router_logits.shape[0] == 0
        nm_rows, nm_stride = router_logits.shape[0], router_logits.stride(0)
        nm_topk = k // nm_rows
    # unused pointers still need a tensor argument
    dummy = lru_step
    _scored_ensure_kernel[(1,)](
        query, slot_of_id, id_of_slot, lru_usage, lru_step,
        out_indices, src_indices, dst_indices, num_copy, stats,
        dummy if tok is None else tok,
        dummy if last_tok is None else last_tok,
        dummy if lc is None else lc,
        dummy if ct is None else ct,
        dummy if g_table is None else g_table,
        dummy if nm_rows == 0 else router_logits,
        dummy if slot_owner is None else slot_owner,
        dummy if slot_last_tok is None else slot_last_tok,
        dummy if slot_lc is None else slot_lc,
        dummy if slot_ct is None else slot_ct,
        dummy if pin_since is None else pin_since,
        dummy if pf_stats is None else pf_stats,
        dummy if pf_ready is None else pf_ready,
        dummy if pf_num is None else pf_num,
        k, num_cached, id_base, nm_topk, nm_stride, float(near_miss_thr), int(pf_rows),
        BLOCK_K=triton.next_power_of_2(k),
        BLOCK_C=block_c,
        BLOCK_E=triton.next_power_of_2(num_experts),
        BLOCK_TOPK=triton.next_power_of_2(max(nm_topk, 1)),
        NM_ROWS=nm_rows,
        USAGE_MAX=torch.iinfo(lru_usage.dtype).max,
        COLLECT_STATS=stats is not None,
        POLICY=policy,
        BUMP_TOK=bool(bump_tok) and policy != 0,
        UPDATE_STATE=bool(update_state) and policy != 0,
        PIN_SINCE=pin_since is not None,
        NUM_LAYERS=num_layers,
        NUM_EXPERTS=num_experts,
        BETA_STEP=beta_step,
        W_Q4=w_q4,
        DECAY_MUL=decay_mul,
        DT_MAX=dt_max,
        SLOT_BITS=slot_bits,
        LOWPRI=bool(lowpri),
        PF_COUNT=pf_stats is not None and not prefetch,
        PREFETCH=bool(prefetch),
        num_warps=_num_warps_for(block_c),
    )


def prefetch_ensure(
    query: torch.Tensor,
    slot_of_id: torch.Tensor,
    id_of_slot: torch.Tensor,
    lru_usage: torch.Tensor,
    lru_step: torch.Tensor,
    dst_slots: torch.Tensor,
    src_rows: torch.Tensor,
    num_copy: torch.Tensor,
    pf_stats: torch.Tensor,
    pf_ready: torch.Tensor,
    **state,
) -> None:
    """Install the ids of ``query`` (-1 = padding) that are not resident as low-priority slots
    (held, usage 0) and write their copy plan to ``dst_slots``/``src_rows``/``num_copy``.

    Unlike a demand call it bumps no clock, touches no hit and no per-id state, and so keeps
    every slot of the last demand call (usage == step) pinned; victims follow the demand key
    (empty, then earlier unconsumed prefetches, then the coldest residents), scored as of the
    previous layer's call, the position the prefetch runs at. With fewer
    evictable slots than misses it installs only the first ones, in query order. Adds the
    installs to ``pf_stats[0]`` and clears ``pf_ready``.
    """
    scored_ensure(
        query, slot_of_id, id_of_slot, lru_usage, lru_step, query, src_rows, dst_slots, num_copy,
        update_state=False, lowpri=True, pf_stats=pf_stats, pf_ready=pf_ready, prefetch=True, **state,
    )
