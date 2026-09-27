"""Expert slot audit (``FREETOKEN_MOE_SLOT_AUDIT=N``): a debug detector for PERSISTENT slot corruption.

Two parts, both on the device and off by default:

- writer history: every writer of the slot maps (demand and prefetch ensures, prefill ensures, the
  whole-layer materialize, prefill buffer invalidation, reset) and every writer of the slot banks
  (demand, prefetch and prefill copies, the prefill double buffers) is followed on its own stream by
  a tiny kernel keyed on the same plan (or slot range) that appends ``(seq, kind, decode step, lru
  step, layer, expert row, ...)`` to a per-slot ring of the last ``DEPTH`` metadata events and one of
  the last ``DEPTH`` byte events. ``seq`` is one global atomic counter, so the two rings merge into
  one cross-stream order. Clears are recorded just before their writer, so they carry the old owner.
  Fixed shapes and no host sync: the recorders capture into the decode graph.
- audit: every N decode steps, after the forward on the compute stream, every held slot is checked:
  its bytes in every bank against the host row of the (layer, expert) its ``id_of_slot`` names
  (host rows gathered with the fused copy into a scratch of fewer than 64 MiB, one layer chunk at a
  time), ``slot_for_id[id_of_slot[s]] == s``, and, over every id, ``id_of_slot[slot_for_id[id]] == id``
  (a stale entry routes that id to another expert's bytes). The first ``RING`` bad (slot, owner)
  pairs are kept with both history rings; the engine logs them and the totals every
  MOE_STATS_INTERVAL decode steps and at shutdown, with a diagnosis read from the history.
"""
from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

from freetoken.env import ENV
from freetoken.utils import init_logger

logger = init_logger(__name__)

# events kept per slot and ring (metadata / bytes)
DEPTH = 4
# event fields
(E_SEQ, E_KIND, E_DSTEP, E_LSTEP, E_LAYER, E_ROW, E_AUX, E_OWNER, E_USAGE, E_POS) = range(10)
EVENT_FIELDS = 10
# metadata kinds (installs name the new owner; clears the old one, recorded before the writer)
DEMAND_INSTALL, PREFETCH_INSTALL, PREFILL_INSTALL, MATERIALIZE_INSTALL = 1, 2, 3, 4
MATERIALIZE_CLEAR, INVALIDATE, RESET = 5, 6, 7
# byte kinds
DEMAND_COPY, PREFETCH_COPY, PREFILL_COPY, MATERIALIZE_COPY, PAGEABLE_COPY = 11, 12, 13, 14, 15
PREFILL_BUFFER, PREFILL_SPLIT_H2D, PREFILL_HIT_D2D, OTHER_COPY = 16, 17, 18, 19
KIND_NAMES = {
    DEMAND_INSTALL: "demand install", PREFETCH_INSTALL: "prefetch install", PREFILL_INSTALL: "prefill ensure install",
    MATERIALIZE_INSTALL: "materialize install", MATERIALIZE_CLEAR: "materialize clear",
    INVALIDATE: "prefill buffer invalidate", RESET: "reset",
    DEMAND_COPY: "demand copy", PREFETCH_COPY: "prefetch copy", PREFILL_COPY: "prefill ensure copy",
    MATERIALIZE_COPY: "materialize copy", PAGEABLE_COPY: "pageable materialize copy",
    PREFILL_BUFFER: "prefill buffer copy", PREFILL_SPLIT_H2D: "prefill buffer H2D (misses and small banks)",
    PREFILL_HIT_D2D: "prefill buffer D2D", OTHER_COPY: "copy",
}
INSTALLS = (DEMAND_INSTALL, PREFETCH_INSTALL, PREFILL_INSTALL, MATERIALIZE_INSTALL)
# the copy each install expects next
COPY_FOR = {
    DEMAND_INSTALL: DEMAND_COPY, PREFETCH_INSTALL: PREFETCH_COPY, PREFILL_INSTALL: PREFILL_COPY,
    MATERIALIZE_INSTALL: MATERIALIZE_COPY,
}
# recorder row sources: the plan's src rows, slot - row_base, the range index, the old owner
ROW_PLAN, ROW_FROM_SLOT, ROW_INDEX, ROW_OLD_OWNER = range(4)
# recorder predicates: every slot, held slots, slots >= E held by the layer
PRED_ALL, PRED_HELD, PRED_LAYER_ABOVE_E = range(3)
# totals: audits, held slots seen, slots byte-checked, held slots not byte-checked, bad slots found
# (summed over audits), of them with bad bytes / a forward map miss / named by a stale slot_for_id /
# an owner out of range, slot_for_id entries past the cache, new records, repeat findings, events
# recorded, plan entries naming no slot
(T_AUDITS, T_HELD, T_BYTES_CHECKED, T_UNCHECKED, T_BAD, T_BYTES, T_FWD, T_STALE, T_OWNER_RANGE, T_MAP_RANGE,
 T_NEW, T_STILL, T_EVENTS, T_BAD_PLAN) = range(14)
NUM_TOTALS = 16
# per-slot audit flags
F_BYTES, F_FWD, F_STALE, F_OWNER_RANGE = 1, 2, 4, 8
# record header, then the metadata ring and the byte ring as stored ([DEPTH, EVENT_FIELDS] each)
(H_AUDIT, H_DSTEP, H_LSTEP, H_SLOT, H_OWNER, H_MAPPED, H_USAGE, H_FLAGS, H_BANKS, H_FIRST_BANK, H_FIRST_BYTE,
 H_BAD_WORDS, H_STALE_ID, H_STALE_N, H_META_N, H_BYTES_N) = range(16)
HEADER = 16
RING_EVENTS = DEPTH * EVENT_FIELDS
RECORD_FIELDS = HEADER + 2 * RING_EVENTS
RING = 32
SCRATCH_BYTES = 64 << 20
MAX_CHUNK_ROWS = 64
COMPARE_WORDS = 4096
_NO_BAD = tl.constexpr(0x7FFFFFFF)
_D = tl.constexpr(DEPTH)
_F = tl.constexpr(EVENT_FIELDS)


def resolve_interval(interval: int | None = None) -> int:
    value = ENV.MOE_SLOT_AUDIT.value if interval is None else interval
    value = int(value or 0)
    if value < 0:
        raise ValueError(f"FREETOKEN_MOE_SLOT_AUDIT={value}: expected 0 (off) or the decode steps between audits")
    return value


@triton.jit
def _put(vals, f, idx: tl.constexpr, v):
    return tl.where(f[None, :] == idx, v.to(tl.int64)[:, None], vals)


@triton.jit
def _bump_step_kernel(dstep_ptr):
    tl.store(dstep_ptr, tl.load(dstep_ptr) + 1)


@triton.jit(do_not_specialize=["kind", "layer", "start", "row_base", "width", "num_slots"])
def _record_kernel(
    hist_ptr, cur_ptr, seq_ptr, dstep_ptr, lstep_ptr, id_of_slot_ptr, usage_ptr, totals_ptr,
    dst_ptr, src_ptr, aux_ptr, num_ptr,
    kind, layer, start, row_base, width, num_slots,
    PLAN: tl.constexpr, ROW: tl.constexpr, HAS_AUX: tl.constexpr, HAS_NUM: tl.constexpr, PRED: tl.constexpr,
    E: tl.constexpr, BLOCK: tl.constexpr,
):
    """Append one event per written slot to that slot's ring: plan entries ``i < min(*num, width)``
    (``PLAN``) or the slots ``start + i`` for ``i < width``."""
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    n = width
    if HAS_NUM:
        n = tl.minimum(tl.load(num_ptr).to(tl.int32), width)
    valid = i < n
    if PLAN:
        slot = tl.load(dst_ptr + i, mask=valid, other=-1)
    else:
        slot = start + i
    ok = valid & (slot >= 0) & (slot < num_slots)
    bad_plan = valid & ~ok
    owner = tl.load(id_of_slot_ptr + slot, mask=ok, other=-3)
    usage = tl.load(usage_ptr + slot, mask=ok, other=-3)
    if PRED == 1:
        ok = ok & (owner >= 0)
    if PRED == 2:
        ok = ok & (owner >= layer * E) & (owner < layer * E + E) & (slot >= E)
    lay = tl.zeros([BLOCK], tl.int32) + layer
    aux = tl.zeros([BLOCK], tl.int32) - 1
    if ROW == 0:
        row = tl.load(src_ptr + i, mask=ok, other=-1)
    elif ROW == 1:
        row = slot - row_base
    elif ROW == 2:
        row = i
    else:
        row = tl.where(owner >= 0, owner % E, -1)
        lay = tl.where(owner >= 0, owner // E, -1)
        aux = aux + 1 + layer
    if HAS_AUX:
        aux = tl.load(aux_ptr + i, mask=ok, other=-1)
    cnt = tl.sum(ok.to(tl.int64), axis=0)
    seq = tl.atomic_add(seq_ptr, cnt) + tl.cumsum(ok.to(tl.int64), axis=0) - 1
    pos = tl.atomic_add(cur_ptr + slot, 1, mask=ok)
    f = tl.arange(0, 16)
    z = tl.zeros([BLOCK], tl.int64)
    vals = tl.zeros([BLOCK, 16], tl.int64) - 1
    vals = _put(vals, f, 0, seq)
    vals = _put(vals, f, 1, z + kind)
    vals = _put(vals, f, 2, z + tl.load(dstep_ptr))
    vals = _put(vals, f, 3, z + tl.load(lstep_ptr))
    vals = _put(vals, f, 4, lay)
    vals = _put(vals, f, 5, row)
    vals = _put(vals, f, 6, aux)
    vals = _put(vals, f, 7, owner)
    vals = _put(vals, f, 8, usage)
    vals = _put(vals, f, 9, i)
    at = (slot.to(tl.int64) * _D + (pos % _D)) * _F
    tl.store(hist_ptr + at[:, None] + f[None, :], vals, mask=ok[:, None] & (f[None, :] < _F))
    t = tl.arange(0, 2)
    add = tl.where(t == 0, cnt, tl.sum(bad_plan.to(tl.int64), axis=0))
    tl.atomic_add(totals_ptr + 12 + t, add, sem="relaxed")


@triton.jit(do_not_specialize=["num_slots", "num_ids"])
def _audit_meta_kernel(
    id_of_slot_ptr, slot_for_id_ptr, layer_ok_ptr, count_ptr, rows_ptr, slots_ptr,
    flags_ptr, stale_id_ptr, stale_n_ptr, totals_ptr, num_slots, num_ids,
    E: tl.constexpr, CAP: tl.constexpr, BLOCK: tl.constexpr,
):
    """Both map checks, and each byte-checkable held slot appended to its layer's list."""
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    ms = i < num_slots
    o = tl.load(id_of_slot_ptr + i, mask=ms, other=-1)
    held = ms & (o >= 0)
    in_range = held & (o < num_ids)
    mapped = tl.load(slot_for_id_ptr + o, mask=in_range, other=-2)
    fwd = in_range & (mapped != i)
    out_of_range = held & ~in_range
    fl = tl.where(fwd, 2, 0) | tl.where(out_of_range, 8, 0)
    tl.atomic_or(flags_ptr + i, fl, mask=ms & (fl != 0))
    layer = tl.where(in_range, o // E, 0)
    ok = in_range & (tl.load(layer_ok_ptr + layer, mask=in_range, other=0) != 0)
    p = tl.atomic_add(count_ptr + layer, 1, mask=ok)
    take = ok & (p < CAP)
    tl.store(rows_ptr + layer * CAP + p, o - layer * E, mask=take)
    tl.store(slots_ptr + layer * CAP + p, i, mask=take)
    mi = i < num_ids
    s = tl.load(slot_for_id_ptr + i, mask=mi, other=-1)
    live = mi & (s >= 0)
    s_ok = live & (s < num_slots)
    owner = tl.load(id_of_slot_ptr + s, mask=s_ok, other=-1)
    stale = s_ok & (owner != i)
    tl.atomic_or(flags_ptr + s, 4, mask=stale)
    tl.store(stale_id_ptr + s, i, mask=stale)
    tl.atomic_add(stale_n_ptr + s, 1, mask=stale)
    t = tl.arange(0, 16)
    add = tl.where(t == 1, tl.sum(held.to(tl.int64), axis=0), 0)
    add = tl.where(t == 2, tl.sum(take.to(tl.int64), axis=0), add)
    add = tl.where(t == 3, tl.sum((held & ~take).to(tl.int64), axis=0), add)
    add = tl.where(t == 6, tl.sum(fwd.to(tl.int64), axis=0), add)
    add = tl.where(t == 7, tl.sum(stale.to(tl.int64), axis=0), add)
    add = tl.where(t == 8, tl.sum(out_of_range.to(tl.int64), axis=0), add)
    add = tl.where(t == 9, tl.sum((live & ~s_ok).to(tl.int64), axis=0), add)
    add = tl.where((t == 0) & (tl.program_id(0) == 0), 1, add)
    tl.atomic_add(totals_ptr + t, add, mask=add != 0, sem="relaxed")


@triton.jit
def _audit_nums_kernel(count_ptr, nums_ptr, L: tl.constexpr, C: tl.constexpr, R: tl.constexpr, BLOCK: tl.constexpr):
    j = tl.arange(0, BLOCK)
    m = j < L * C
    cnt = tl.load(count_ptr + j // C, mask=m, other=0)
    tl.store(nums_ptr + j, tl.minimum(tl.maximum(cnt - (j % C) * R, 0), R).to(tl.int64), mask=m)


@triton.jit
def _audit_compare_kernel(cache_ptrs, ref_ptrs, words_ptr, slots_ptr, num_ptr, bad_cnt_ptr, bad_first_ptr,
                          NB: tl.constexpr, BLOCK: tl.constexpr):
    """Program (row, bank, chunk): one chunk of a listed slot's bank row against its gathered host row."""
    r = tl.program_id(0)
    if r < tl.load(num_ptr):
        b = tl.program_id(1)
        words = tl.load(words_ptr + b)
        off = tl.program_id(2).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        m = off < words
        slot = tl.load(slots_ptr + r).to(tl.int64)
        got = tl.load(tl.load(cache_ptrs + b).to(tl.pointer_type(tl.int32)) + slot * words + off, mask=m, other=0)
        want = tl.load(tl.load(ref_ptrs + b).to(tl.pointer_type(tl.int32)) + r * words + off, mask=m, other=0)
        bad = m & (got != want)
        nbad = tl.sum(bad.to(tl.int32), axis=0)
        if nbad > 0:
            tl.atomic_add(bad_cnt_ptr + slot * NB + b, nbad)
            tl.atomic_min(bad_first_ptr + slot * NB + b, tl.min(tl.where(bad, off, _NO_BAD), axis=0).to(tl.int32))


@triton.jit(do_not_specialize=["num_slots", "num_ids"])
def _audit_report_kernel(
    id_of_slot_ptr, slot_for_id_ptr, usage_ptr, lstep_ptr, dstep_ptr,
    flags_ptr, stale_id_ptr, stale_n_ptr, bad_cnt_ptr, bad_first_ptr, reported_ptr,
    meta_hist_ptr, meta_cur_ptr, bytes_hist_ptr, bytes_cur_ptr,
    ring_ptr, cursor_ptr, totals_ptr, count_ptr,
    num_slots, num_ids,
    NB: tl.constexpr, NB_P: tl.constexpr, L: tl.constexpr, L_P: tl.constexpr, RING_N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Keep each new bad (slot, owner) with its history, count the findings and clear the work arrays."""
    s = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    ms = s < num_slots
    b = tl.arange(0, NB_P)
    cell = ms[:, None] & (b < NB)[None, :]
    cnt = tl.load(bad_cnt_ptr + s[:, None] * NB + b[None, :], mask=cell, other=0)
    first = tl.load(bad_first_ptr + s[:, None] * NB + b[None, :], mask=cell, other=_NO_BAD)
    bank_bad = cnt > 0
    bytes_bad = tl.sum(bank_bad.to(tl.int32), axis=1) > 0
    bank_mask = tl.sum(tl.where(bank_bad, 1 << b[None, :], 0), axis=1)
    first_bank = tl.min(tl.where(bank_bad, b[None, :], NB_P), axis=1)
    first_word = tl.sum(tl.where(b[None, :] == first_bank[:, None], first, 0), axis=1)
    fl = tl.load(flags_ptr + s, mask=ms, other=0) | tl.where(bytes_bad, 1, 0)
    bad = ms & (fl != 0)
    owner = tl.load(id_of_slot_ptr + s, mask=ms, other=-1)
    sid = tl.load(stale_id_ptr + s, mask=ms, other=-1)
    key = (owner.to(tl.int64) + 1) * (num_ids + 2) + sid + 1
    new = bad & (tl.load(reported_ptr + s, mask=ms, other=-1) != key)
    tl.store(reported_ptr + s, tl.where(bad, key, -1), mask=ms)
    pos = tl.atomic_add(cursor_ptr + s * 0, 1, mask=new)
    keep = new & (pos < RING_N)
    in_range = ms & (owner >= 0) & (owner < num_ids)
    f = tl.arange(0, 16)
    z = tl.zeros([BLOCK], tl.int64)
    vals = tl.zeros([BLOCK, 16], tl.int64) - 1
    vals = _put(vals, f, 0, z + tl.load(totals_ptr) - 1)
    vals = _put(vals, f, 1, z + tl.load(dstep_ptr))
    vals = _put(vals, f, 2, z + tl.load(lstep_ptr))
    vals = _put(vals, f, 3, s)
    vals = _put(vals, f, 4, owner)
    vals = _put(vals, f, 5, tl.load(slot_for_id_ptr + owner, mask=in_range, other=-2))
    vals = _put(vals, f, 6, tl.load(usage_ptr + s, mask=ms, other=-1))
    vals = _put(vals, f, 7, fl)
    vals = _put(vals, f, 8, bank_mask)
    vals = _put(vals, f, 9, tl.where(bytes_bad, first_bank, -1))
    vals = _put(vals, f, 10, tl.where(bytes_bad, first_word.to(tl.int64) * 4, -1))
    vals = _put(vals, f, 11, tl.sum(cnt, axis=1))
    vals = _put(vals, f, 12, sid)
    vals = _put(vals, f, 13, tl.load(stale_n_ptr + s, mask=ms, other=0))
    vals = _put(vals, f, 14, tl.load(meta_cur_ptr + s, mask=ms, other=0))
    vals = _put(vals, f, 15, tl.load(bytes_cur_ptr + s, mask=ms, other=0))
    rec = ring_ptr + pos.to(tl.int64) * (16 + 2 * _D * _F)
    tl.store(rec[:, None] + f[None, :], vals, mask=keep[:, None])
    j = tl.arange(0, 64)
    jm = keep[:, None] & (j < _D * _F)[None, :]
    src = s.to(tl.int64)[:, None] * (_D * _F) + j[None, :]
    tl.store(rec[:, None] + 16 + j[None, :], tl.load(meta_hist_ptr + src, mask=jm, other=-1), mask=jm)
    tl.store(rec[:, None] + 16 + _D * _F + j[None, :], tl.load(bytes_hist_ptr + src, mask=jm, other=-1), mask=jm)
    t = tl.arange(0, 16)
    add = tl.where(t == 4, tl.sum(bad.to(tl.int64), axis=0), 0)
    add = tl.where(t == 5, tl.sum((ms & bytes_bad).to(tl.int64), axis=0), add)
    add = tl.where(t == 10, tl.sum(new.to(tl.int64), axis=0), add)
    add = tl.where(t == 11, tl.sum((bad & ~new).to(tl.int64), axis=0), add)
    tl.atomic_add(totals_ptr + t, add, mask=add != 0, sem="relaxed")
    tl.store(flags_ptr + s, tl.zeros([BLOCK], tl.int32), mask=ms)
    tl.store(stale_id_ptr + s, tl.zeros([BLOCK], tl.int32) - 1, mask=ms)
    tl.store(stale_n_ptr + s, tl.zeros([BLOCK], tl.int32), mask=ms)
    tl.store(bad_cnt_ptr + s[:, None] * NB + b[None, :], tl.zeros([BLOCK, NB_P], tl.int32), mask=cell)
    tl.store(bad_first_ptr + s[:, None] * NB + b[None, :], tl.zeros([BLOCK, NB_P], tl.int32) + _NO_BAD, mask=cell)
    if tl.program_id(0) == 0:
        ll = tl.arange(0, L_P)
        tl.store(count_ptr + ll, tl.zeros([L_P], tl.int32), mask=ll < L)


class SlotAuditor:
    """Per-slot writer history (recorded by ``record_*`` on the writer's stream) and the periodic
    audit (``end_decode_step`` -> ``scan`` on the compute stream). ``bind`` sizes everything for a
    cache's slots and banks; a rebuild rebinds and starts a new history, the totals and records stay.
    Holds no reference to its cache: every call takes it."""

    def __init__(self, num_layers: int, num_experts: int, device: torch.device, *, interval: int,
                 ring: int | None = None, scratch_bytes: int = SCRATCH_BYTES) -> None:
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.device = device
        self.interval = interval
        self.ring_n = RING if ring is None else ring
        self.scratch_bytes = scratch_bytes
        ring = self.ring_n
        self.dstep = torch.zeros((1,), dtype=torch.int64, device=device)
        self.seq = torch.zeros((1,), dtype=torch.int64, device=device)
        self.totals = torch.zeros((NUM_TOTALS,), dtype=torch.int64, device=device)
        self.ring = torch.full((ring, RECORD_FIELDS), -1, dtype=torch.int64, device=device)
        self.cursor = torch.zeros((1,), dtype=torch.int64, device=device)
        self.count = torch.zeros((num_layers,), dtype=torch.int32, device=device)
        self.num_slots = 0
        self.bank_names: list[str] = []
        self._feats: list[int] = []
        self._steps = 0
        self._printed = 0
        self._last = torch.zeros((NUM_TOTALS,), dtype=torch.int64)
        self._dummy = torch.zeros((1,), dtype=torch.int32, device=device)
        logger.info(f"MoE slot audit: recording every slot writer, auditing every held slot each {interval} decode steps")

    # ------------------------------------------------------------------ setup

    def bind(self, cache) -> None:
        """Size the history and audit arrays for ``cache``'s slots and banks (a rebuild starts a new
        history) and point the byte checks at its banks."""
        caches = [bank for _, bank in cache.banks]
        feats = [math.prod(bank.shape[1:]) * bank.element_size() for bank in caches]
        assert all(f % 4 == 0 for f in feats), feats
        if self.bank_names:
            assert list(cache.bank_schema) == self.bank_names and feats == self._feats, "the bank layout changed"
        else:
            self.bank_names = list(cache.bank_schema)
            self._feats = feats
            self._alloc_scratch(cache, feats)
        self.cache_ptrs = torch.tensor([bank.data_ptr() for bank in caches], dtype=torch.int64, device=self.device)
        S, dev = cache.cache_size, self.device
        self.num_slots = S
        self.meta_hist = torch.full((S, DEPTH, EVENT_FIELDS), -1, dtype=torch.int64, device=dev)
        self.bytes_hist = torch.full((S, DEPTH, EVENT_FIELDS), -1, dtype=torch.int64, device=dev)
        self.meta_cur = torch.zeros((S,), dtype=torch.int32, device=dev)
        self.bytes_cur = torch.zeros((S,), dtype=torch.int32, device=dev)
        self.flags = torch.zeros((S,), dtype=torch.int32, device=dev)
        self.stale_id = torch.full((S,), -1, dtype=torch.int32, device=dev)
        self.stale_n = torch.zeros((S,), dtype=torch.int32, device=dev)
        self.reported = torch.full((S,), -1, dtype=torch.int64, device=dev)
        nb = len(feats)
        self.bad_cnt = torch.zeros((S * nb,), dtype=torch.int32, device=dev)
        self.bad_first = torch.full((S * nb,), 0x7FFFFFFF, dtype=torch.int32, device=dev)
        # a correct cache holds at most num_experts slots of a layer; more are counted unchecked
        per_layer = min(self.num_experts, S)
        self.chunks = triton.cdiv(per_layer, self.rows)
        self.cap = self.chunks * self.rows
        self.list_rows = torch.zeros((self.num_layers, self.cap), dtype=torch.int32, device=dev)
        self.list_slots = torch.zeros((self.num_layers, self.cap), dtype=torch.int32, device=dev)
        self.nums = torch.zeros((self.num_layers * self.chunks,), dtype=torch.int64, device=dev)
        # layers without a device alias (LOCKED/PAGEABLE host banks) cannot be gathered
        self.layer_ok_host = [layer not in cache._unpinned_layers for layer in range(self.num_layers)]
        self.layer_ok = torch.tensor(self.layer_ok_host, dtype=torch.int32, device=dev)
        self.count.zero_()

    def _alloc_scratch(self, cache, feats: list[int]) -> None:
        row = sum(feats)
        self.rows = max(1, min(MAX_CHUNK_ROWS, (self.scratch_bytes - 1) // row))
        rows = self.rows
        self.scratch = torch.empty((rows * row,), dtype=torch.uint8, device=self.device)
        offsets = [rows * sum(feats[:b]) for b in range(len(feats))]
        self.scratch_views = [
            self.scratch[o : o + rows * f].view(bank.dtype).view(rows, *bank.shape[1:])
            for o, f, (_, bank) in zip(offsets, feats, cache.banks)
        ]
        self.scratch_ptrs = torch.tensor([self.scratch.data_ptr() + o for o in offsets], dtype=torch.int64, device=self.device)
        self.words = torch.tensor([f // 4 for f in feats], dtype=torch.int64, device=self.device)
        self.iota = torch.arange(rows, dtype=torch.int32, device=self.device)
        self._word_chunks = triton.cdiv(max(feats) // 4, COMPARE_WORDS)
        logger.info(f"MoE slot audit scratch: {self.scratch.numel() / 2**20:.1f} MiB, {rows} experts a chunk")

    # ------------------------------------------------------------------ recording (graph-capturable)

    def begin_step(self) -> None:
        """Advance the decode-step clock (the first GPU layer's decode ensure calls this)."""
        _bump_step_kernel[(1,)](self.dstep)

    def _launch(self, cache, kind: int, layer: int, *, meta: bool, width: int, dst=None, src=None, aux=None,
                num=None, start: int = 0, row: int = ROW_PLAN, row_base: int = 0, pred: int = PRED_ALL) -> None:
        if width <= 0 or not self.num_slots:
            return
        block = min(triton.next_power_of_2(width), 128)
        d = self._dummy
        _record_kernel[(triton.cdiv(width, block),)](
            self.meta_hist if meta else self.bytes_hist, self.meta_cur if meta else self.bytes_cur,
            self.seq, self.dstep, cache.step, cache.id_of_slot, cache.usage, self.totals,
            d if dst is None else dst, d if src is None else src, d if aux is None else aux, d if num is None else num,
            kind, layer, start, row_base, width, self.num_slots,
            PLAN=dst is not None, ROW=row, HAS_AUX=aux is not None, HAS_NUM=num is not None, PRED=pred,
            E=self.num_experts, BLOCK=block, num_warps=4,
        )

    def record_plan(self, cache, kind: int, layer: int, dst: torch.Tensor, src: torch.Tensor | None, num: torch.Tensor,
                    *, meta: bool, width: int | None = None, aux: torch.Tensor | None = None, row_base: int = 0) -> None:
        """The writer just wrote slots ``dst[:num]`` (rows ``src[:num]``, else ``dst - row_base``)."""
        width = dst.numel() if width is None else min(width, dst.numel())
        self._launch(cache, kind, layer, meta=meta, width=width, dst=dst, src=src, aux=aux, num=num,
                     row=ROW_PLAN if src is not None else ROW_FROM_SLOT, row_base=row_base)

    def record_range(self, cache, kind: int, layer: int, start: int, n: int, *, meta: bool) -> None:
        """The writer just wrote expert ``i`` of ``layer`` into slot ``start + i`` for ``i < n``."""
        self._launch(cache, kind, layer, meta=meta, width=n, start=start, row=ROW_INDEX)

    def record_clear(self, cache, kind: int, context_layer: int, start: int, n: int, *, layer_above_e: bool = False) -> None:
        """Before a writer clears held slots in ``[start, start + n)`` (with ``layer_above_e``: only
        those >= E held by ``context_layer``); the event names the old owner."""
        self._launch(cache, kind, context_layer, meta=True, width=n, start=start, row=ROW_OLD_OWNER,
                     pred=PRED_LAYER_ABOVE_E if layer_above_e else PRED_HELD)

    # ------------------------------------------------------------------ audit

    def end_decode_step(self, cache) -> bool:
        """After a decode forward, on the compute stream: audit every ``interval`` steps."""
        self._steps += 1
        if self._steps % self.interval:
            return False
        self.scan(cache)
        return True

    def scan(self, cache) -> None:
        """Audit every held slot now (enqueued on the current stream; no host sync)."""
        S, num_ids = self.num_slots, self.num_layers * self.num_experts
        assert S == cache.cache_size and cache.id_of_slot.numel() == S, "bind after a rebuild"
        block = 1024
        _audit_meta_kernel[(triton.cdiv(max(S, num_ids), block),)](
            cache.id_of_slot, cache.slot_for_id.view(-1), self.layer_ok, self.count, self.list_rows, self.list_slots,
            self.flags, self.stale_id, self.stale_n, self.totals, S, num_ids,
            E=self.num_experts, CAP=self.cap, BLOCK=block, num_warps=4,
        )
        n_lists = self.num_layers * self.chunks
        _audit_nums_kernel[(1,)](self.count, self.nums, L=self.num_layers, C=self.chunks, R=self.rows,
                                 BLOCK=triton.next_power_of_2(n_lists))
        nb = len(self.bank_names)
        R = self.rows
        for layer in range(self.num_layers):
            if not self.layer_ok_host[layer]:
                continue
            for c in range(self.chunks):
                rows = self.list_rows[layer, c * R : (c + 1) * R]
                slots = self.list_slots[layer, c * R : (c + 1) * R]
                num = self.nums[layer * self.chunks + c : layer * self.chunks + c + 1]
                self._gather(cache, layer, rows, num)
                _audit_compare_kernel[(R, nb, self._word_chunks)](
                    self.cache_ptrs, self.scratch_ptrs, self.words, slots, num, self.bad_cnt, self.bad_first,
                    NB=nb, BLOCK=COMPARE_WORDS, num_warps=8,
                )
        rblock = 256
        _audit_report_kernel[(triton.cdiv(S, rblock),)](
            cache.id_of_slot, cache.slot_for_id.view(-1), cache.usage, cache.step, self.dstep,
            self.flags, self.stale_id, self.stale_n, self.bad_cnt, self.bad_first, self.reported,
            self.meta_hist, self.meta_cur, self.bytes_hist, self.bytes_cur,
            self.ring, self.cursor, self.totals, self.count, S, num_ids,
            NB=nb, NB_P=triton.next_power_of_2(nb), L=self.num_layers, L_P=triton.next_power_of_2(self.num_layers),
            RING_N=self.ring_n, BLOCK=rblock, num_warps=4,
        )

    def _gather(self, cache, layer: int, rows: torch.Tensor, num: torch.Tensor) -> None:
        # the plain fused copy, not the slim one prefetch copies use: a slim-kernel bug cannot hide itself
        if cache._copy_fused_ok:
            from freetoken.kernel.fast_index_copy import fast_index_copy_multi_jit

            fast_index_copy_multi_jit(
                self.scratch_ptrs, cache._copy_src_ptrs[layer], cache._copy_feat_bytes, self.iota, rows, num,
            )
            return
        from freetoken.kernel import fast_index_copy_jit

        for (per_layer, _), view in zip(cache.banks, self.scratch_views):
            fast_index_copy_jit(view, self.iota, per_layer[layer], rows, num)

    # ------------------------------------------------------------------ reporting (host syncs)

    def take_window(self) -> tuple[list[int], list[int], list[list[int]], int]:
        """(totals now, totals of this window, records kept since the last call, records in all)."""
        totals = self.totals.cpu()
        window = (totals - self._last).tolist()
        self._last = totals
        cursor = int(self.cursor.item())
        upto = min(cursor, self.ring_n)
        records = self.ring[self._printed : upto].cpu().tolist() if upto > self._printed else []
        self._printed = max(self._printed, upto)
        return totals.tolist(), window, records, cursor

    def report_window(self, steps: int) -> list[tuple[str, str]]:
        """(level, line) pairs for one MOE_STATS_INTERVAL window; nothing when no audit ran."""
        totals, window, records, cursor = self.take_window()
        if not window[T_AUDITS] and not records:
            return []
        bad = bool(window[T_BAD] or window[T_MAP_RANGE] or window[T_BAD_PLAN] or records)
        lines = [("warning" if bad else "info",
                  f"MoE slot audit ({steps} decode steps, every {self.interval}): {self.format_counts(window)}; "
                  f"session: {totals[T_AUDITS]} audits, {totals[T_NEW]} bad (slot, owner) pairs")]
        lines += [("warning", "MoE slot audit record: " + self.format_record(r)) for r in records]
        if records and cursor > self.ring_n:
            lines.append(("warning", f"MoE slot audit: {cursor} bad (slot, owner) pairs in all, the first {self.ring_n} are kept"))
        return lines

    def report_session(self) -> list[tuple[str, str]]:
        """Session totals and every kept record (shutdown)."""
        totals = self.totals.cpu().tolist()
        cursor = int(self.cursor.item())
        if not totals[T_AUDITS]:
            return []
        bad = bool(totals[T_BAD] or totals[T_MAP_RANGE] or totals[T_BAD_PLAN])
        lines = [("warning" if bad else "info", f"MoE slot audit (session, every {self.interval} decode steps): "
                                                  f"{self.format_counts(totals)}")]
        records = self.ring[: min(cursor, self.ring_n)].cpu().tolist()
        lines += [("warning", "MoE slot audit record: " + self.format_record(r)) for r in records]
        if cursor > self.ring_n:
            lines.append(("warning", f"MoE slot audit: {cursor} bad (slot, owner) pairs in all, the first {self.ring_n} are kept"))
        return lines

    @staticmethod
    def format_counts(t: list[int]) -> str:
        audits = t[T_AUDITS]
        per = (lambda v: f"{v / audits:.0f}") if audits else str
        text = (
            f"{audits} audits, {per(t[T_HELD])} held slots an audit ({per(t[T_BYTES_CHECKED])} byte-checked), "
            f"bad slots={t[T_BAD]} (bytes={t[T_BYTES]}, map={t[T_FWD]}, stale slot_for_id={t[T_STALE]}, "
            f"owner out of range={t[T_OWNER_RANGE]}), new={t[T_NEW]}, repeat={t[T_STILL]}, events={t[T_EVENTS]}"
        )
        if t[T_UNCHECKED]:
            text += f", held slots not byte-checked={t[T_UNCHECKED]}"
        if t[T_MAP_RANGE]:
            text += f", slot_for_id entries past the cache={t[T_MAP_RANGE]}"
        if t[T_BAD_PLAN]:
            text += f", plan entries naming no slot={t[T_BAD_PLAN]}"
        return text

    # ------------------------------------------------------------------ record decoding

    def _name(self, flat: int) -> str:
        e = self.num_experts
        return f"L{flat // e}/e{flat % e}" if flat >= 0 else str(flat)

    @staticmethod
    def events(rec: list[int]) -> list[dict]:
        """Both rings of a record as event dicts in ``seq`` order (``ring`` 'meta' or 'bytes')."""
        out = []
        for ring, base in (("meta", HEADER), ("bytes", HEADER + RING_EVENTS)):
            for d in range(DEPTH):
                ev = rec[base + d * EVENT_FIELDS : base + (d + 1) * EVENT_FIELDS]
                if ev[E_SEQ] >= 0:
                    out.append({
                        "ring": ring, "seq": ev[E_SEQ], "kind": ev[E_KIND], "dstep": ev[E_DSTEP], "lstep": ev[E_LSTEP],
                        "layer": ev[E_LAYER], "row": ev[E_ROW], "aux": ev[E_AUX], "owner": ev[E_OWNER],
                        "usage": ev[E_USAGE], "pos": ev[E_POS],
                    })
        return sorted(out, key=lambda ev: ev["seq"])

    def _flat(self, ev: dict) -> int:
        return ev["layer"] * self.num_experts + ev["row"] if ev["layer"] >= 0 and ev["row"] >= 0 else -1

    def format_event(self, ev: dict) -> str:
        e = self.num_experts
        text = f"#{ev['seq']} {KIND_NAMES.get(ev['kind'], ev['kind'])} L{ev['layer']}/e{ev['row']}"
        if ev["kind"] == PREFILL_HIT_D2D:
            text += f" from slot {ev['aux']}"
        elif ev["kind"] in (MATERIALIZE_CLEAR, INVALIDATE) and ev["aux"] >= 0:
            text += f" (for layer {ev['aux']})"
        seen = "slot held" if ev["ring"] == "bytes" else "then held"
        if ev["kind"] in (MATERIALIZE_CLEAR, INVALIDATE, RESET):
            seen = "held"
        owner = f"L{ev['owner'] // e}/e{ev['owner'] % e}" if ev["owner"] >= 0 else "nothing"
        return f"{text} @step {ev['dstep']}/lru {ev['lstep']} ({seen} {owner}, usage {ev['usage']})"

    def _kind(self, ev: dict) -> str:
        return KIND_NAMES.get(ev["kind"], str(ev["kind"]))

    def _pending_install(self, copy: dict, meta: list[dict], byte: list[dict], before: dict) -> dict | None:
        """The earlier install (before ``before``) that ``copy`` completes: an install of the id it
        copied with no copy of that id between them."""
        flat = self._flat(copy)
        for inst in reversed([ev for ev in meta if ev["kind"] in INSTALLS and ev["seq"] < before["seq"]]):
            if self._flat(inst) == flat:
                done = any(self._flat(ev) == flat and inst["seq"] < ev["seq"] < copy["seq"] for ev in byte)
                return None if done else inst
        return None

    def _bytes_story(self, owner: int, meta: list[dict], byte: list[dict]) -> str:
        insts = [ev for ev in meta if ev["kind"] in INSTALLS and self._flat(ev) == owner]
        if not insts:
            return (f"no recorded install of {self._name(owner)} in the kept history "
                    f"(an unrecorded map writer, or it is older than the last {DEPTH} events)")
        inst = insts[-1]
        name = self._kind(inst)
        after = [ev for ev in byte if ev["seq"] > inst["seq"]]
        own = [ev for ev in after if self._flat(ev) == owner]
        if own:
            last = after[-1]
            if self._flat(last) == owner:
                return (f"{name} and {self._kind(last)} of {self._name(owner)} (#{last['seq']}), then the bytes changed "
                        f"with no recorded write: overwritten after the last recorded write, or that copy did not land")
            text = (f"{name} and {self._kind(own[-1])} of {self._name(owner)} (#{own[-1]['seq']}), then "
                    f"{self._kind(last)} of {self._name(self._flat(last))} (#{last['seq']}) overwrote it")
            prior = self._pending_install(last, meta, byte, inst)
            if prior is not None:
                text += f": the late copy of the earlier {self._kind(prior)} of {self._name(self._flat(prior))} (#{prior['seq']})"
            return text
        text = f"{name}, no {KIND_NAMES[COPY_FOR[inst['kind']]]}"
        if not after:
            return text
        last = after[-1]
        prior = self._pending_install(last, meta, byte, inst)
        if prior is not None:
            return (f"{text}; the last write, {self._kind(last)} of {self._name(self._flat(last))} (#{last['seq']}), "
                    f"is the copy of the earlier {self._kind(prior)} (#{prior['seq']}) and landed after this install")
        where = (f"from layer {last['layer']}'s source" if last["row"] == owner % self.num_experts
                 else f"of {self._name(self._flat(last))}")
        return f"{name} of {self._name(owner)}, then {self._kind(last)} {where} ({self._name(self._flat(last))}, #{last['seq']})"

    def diagnose(self, rec: list[int]) -> str:
        """What the history says about a bad slot, in words."""
        owner, fl = rec[H_OWNER], rec[H_FLAGS]
        evs = self.events(rec)
        meta = [ev for ev in evs if ev["ring"] == "meta"]
        byte = [ev for ev in evs if ev["ring"] == "bytes"]
        parts = []
        if fl & F_OWNER_RANGE:
            parts.append(f"id_of_slot={owner} names no expert")
        if fl & F_BYTES and owner >= 0:
            parts.append(self._bytes_story(owner, meta, byte))
        if fl & F_FWD:
            parts.append(f"slot_for_id[{self._name(owner)}]={rec[H_MAPPED]}, not this slot")
        if fl & F_STALE:
            sid = rec[H_STALE_ID]
            text = (f"stale slot_for_id: {self._name(sid)} -> this slot, which holds {self._name(owner)}"
                    + (f" ({rec[H_STALE_N]} such ids)" if rec[H_STALE_N] > 1 else ""))
            gone = [ev for ev in meta if ev["kind"] in INSTALLS and self._flat(ev) == sid]
            if gone:
                later = [ev for ev in meta if ev["seq"] > gone[-1]["seq"]]
                text += (f"; {self._name(sid)} was installed here by {self._kind(gone[-1])} #{gone[-1]['seq']}"
                         + (f" and replaced by {self._kind(later[0])} #{later[0]['seq']} without clearing "
                            f"its map entry" if later else ""))
            parts.append(text)
        # an install's recorder reads the slot right after the install on the same stream
        for ev in meta:
            if ev["kind"] in INSTALLS and ev["owner"] != self._flat(ev):
                parts.append(f"right after the {self._kind(ev)} of {self._name(self._flat(ev))} (#{ev['seq']}) the slot "
                             f"named {self._name(ev['owner'])}: another map writer ran concurrently")
        return "; ".join(parts) or "clean"

    def format_record(self, rec: list[int]) -> str:
        fl = rec[H_FLAGS]
        text = (f"audit={rec[H_AUDIT]} step={rec[H_DSTEP]} lru={rec[H_LSTEP]} slot={rec[H_SLOT]} "
                f"holds {self._name(rec[H_OWNER])} (slot_for_id={rec[H_MAPPED]}, usage={rec[H_USAGE]})")
        if fl & F_BYTES:
            banks = [self.bank_names[b] if b < len(self.bank_names) else str(b)
                     for b in range(max(len(self.bank_names), 1)) if rec[H_BANKS] >> b & 1]
            first = self.bank_names[rec[H_FIRST_BANK]] if 0 <= rec[H_FIRST_BANK] < len(self.bank_names) else rec[H_FIRST_BANK]
            text += (f" | bytes differ in {', '.join(banks)} (first at {first}+{rec[H_FIRST_BYTE]}, "
                     f"{rec[H_BAD_WORDS]} bad words)")
        text += f" | diagnosis: {self.diagnose(rec)}"
        history = "; ".join(self.format_event(ev) for ev in self.events(rec))
        return text + f" | history ({rec[H_META_N]} map / {rec[H_BYTES_N]} byte events in all): {history or 'none'}"
