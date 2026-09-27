"""Decode expert-cache verifier (``FREETOKEN_MOE_PREFETCH_VERIFY``): a debug detector for slot races.

For every GPU decode layer call it checks, on the compute stream and in its order, what the
grouped GEMM is about to read and has just read:

- meta (before and after the GEMM): each routed entry's slot maps to the expert the router chose,
  in both slot maps (``id_of_slot[s] == layer * E + id`` and ``slot_for_id[layer, id] == s``); the
  router's ids are copied before ensure rewrites them in place;
- bytes before the GEMM (``full``): every routed slot's row in every bank equals the host bank row
  of that expert for that layer, gathered into a scratch buffer with the fused copy;
- bytes after the GEMM (``full``): the same slots against the same scratch, so a slot overwritten
  while the GEMM ran shows up even when a later copy restores it.

Nothing here records or waits on an event: the prefetch streams keep exactly the edges they have
without it. The checks do delay the compute stream (``full`` gathers every routed expert over
PCIe), which moves where the other streams land relative to it. ``meta`` skips the byte checks and
costs a few one-CTA launches per layer. Counts accumulate per layer on the device and the first
``RING`` offending records are kept with the slot state seen at check time; the engine reports
them every MOE_STATS_INTERVAL decode steps and at shutdown.
"""
from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

from freetoken.env import ENV
from freetoken.utils import init_logger

logger = init_logger(__name__)

MODES = ("off", "meta", "full")
# counters[layer, col]: checked calls, offending records per check, routed experts left unchecked
CHECKS, META_BAD, PRE_BAD, POST_BAD, POST_META_BAD, UNCHECKED = range(6)
NUM_COLS = 6
COL_NAMES = ("checks", "meta_bad", "pre_bad", "post_bad", "post_meta_bad", "unchecked")
META_PRE, BYTES_PRE, META_POST, BYTES_POST = range(4)
KINDS = ("meta_pre", "bytes_pre", "meta_post", "bytes_post")
# record fields; owner / mapped / usage / step are read when the check runs
(R_CALL, R_LAYER, R_ROW, R_EXPERT, R_SLOT, R_BANK, R_KIND, R_OWNER, R_MAPPED, R_USAGE, R_STEP,
 R_SOURCE, R_FIRST_BYTE, R_BAD_WORDS) = range(14)
NUM_FIELDS = 14
# R_SOURCE bits: the slot is in this call's demand copy plan, this layer's prefetch copy plan, or
# the next layer's prefetch plan (installed earlier on the compute stream)
SRC_DEMAND, SRC_PREFETCH, SRC_NEXT_PREFETCH = 1, 2, 4
RING = 64
# routed entries a call may have (bs * top_k); wider eager batches are counted unchecked
MAX_ENTRIES = 64
CHUNK_WORDS = 4096
_NF = tl.constexpr(NUM_FIELDS)
_NC = tl.constexpr(NUM_COLS)
_NO_BAD = tl.constexpr(0x7FFFFFFF)


def resolve_mode(mode: str | None = None) -> str:
    raw = (ENV.MOE_PREFETCH_VERIFY.value if mode is None else mode).strip().lower()
    if raw in ("", "0", "off", "false", "no"):
        return "off"
    if raw in ("1", "on", "true", "yes", "full"):
        return "full"
    if raw == "meta":
        return "meta"
    raise ValueError(f"FREETOKEN_MOE_PREFETCH_VERIFY={raw!r}: expected 0, 1 (full) or meta")


@triton.jit
def _put(vals, f, idx: tl.constexpr, v):
    return tl.where(f[None, :] == idx, v.to(tl.int64)[:, None], vals)


@triton.jit
def _in_plan(s, plan_ptr, num_ptr, W: tl.constexpr, W_P: tl.constexpr):
    j = tl.arange(0, W_P)
    plan = tl.load(plan_ptr + j, mask=j < tl.minimum(tl.load(num_ptr), W), other=-3)
    return tl.sum((s[:, None] == plan[None, :]).to(tl.int32), axis=1) > 0


@triton.jit
def _source(s, dem_ptr, dem_num_ptr, pf_ptr, pf_num_ptr, nx_ptr, nx_num_ptr,
            DEM_W: tl.constexpr, DEM_P: tl.constexpr, PF_W: tl.constexpr, PF_P: tl.constexpr,
            HAS_PF: tl.constexpr, HAS_NX: tl.constexpr):
    src = tl.where(_in_plan(s, dem_ptr, dem_num_ptr, DEM_W, DEM_P), 1, 0)
    if HAS_PF:
        src += tl.where(_in_plan(s, pf_ptr, pf_num_ptr, PF_W, PF_P), 2, 0)
    if HAS_NX:
        src += tl.where(_in_plan(s, nx_ptr, nx_num_ptr, PF_W, PF_P), 4, 0)
    return src


@triton.jit
def _write(ring_ptr, cursor, bad, vals, RING_N: tl.constexpr):
    """Append the ``bad`` rows of ``vals`` after ``cursor`` while the ring has room; the new cursor."""
    f = tl.arange(0, 16)
    pos = cursor + tl.cumsum(bad.to(tl.int64), axis=0) - 1
    keep = bad & (pos < RING_N)
    tl.store(ring_ptr + pos[:, None] * _NF + f[None, :], vals, mask=keep[:, None] & (f[None, :] < _NF))
    return cursor + tl.sum(bad.to(tl.int64), axis=0)


@triton.jit
def _meta(ids_ptr, slots_ptr, id_of_slot_ptr, slot_for_id_ptr, usage_ptr, step_ptr,
          dem_ptr, dem_num_ptr, pf_ptr, pf_num_ptr, nx_ptr, nx_num_ptr,
          ring_ptr, cursor, counters_ptr, col: tl.constexpr, kind: tl.constexpr, call, layer, num_slots,
          N: tl.constexpr, N_P: tl.constexpr, E: tl.constexpr,
          DEM_W: tl.constexpr, DEM_P: tl.constexpr, PF_W: tl.constexpr, PF_P: tl.constexpr,
          HAS_PF: tl.constexpr, HAS_NX: tl.constexpr, RING_N: tl.constexpr):
    """Check every routed entry's slot against both maps; returns the new ring cursor."""
    i = tl.arange(0, N_P)
    m = i < N
    ids = tl.load(ids_ptr + i, mask=m, other=0)
    s = tl.load(slots_ptr + i, mask=m, other=-1)
    ok_s = m & (s >= 0) & (s < num_slots)
    ok_id = m & (ids >= 0) & (ids < E)
    fid = layer * E + ids
    owner = tl.load(id_of_slot_ptr + s, mask=ok_s, other=-2)
    mapped = tl.load(slot_for_id_ptr + fid, mask=ok_id, other=-2)
    bad = m & ~(ok_s & ok_id & (owner == fid) & (mapped == s))
    usage = tl.load(usage_ptr + s, mask=ok_s, other=-1)
    src = _source(s, dem_ptr, dem_num_ptr, pf_ptr, pf_num_ptr, nx_ptr, nx_num_ptr, DEM_W, DEM_P, PF_W, PF_P, HAS_PF, HAS_NX)
    f = tl.arange(0, 16)
    z = tl.zeros([N_P], tl.int64)
    vals = tl.zeros([N_P, 16], tl.int64) - 1
    vals = _put(vals, f, 0, z + call)
    vals = _put(vals, f, 1, z + layer)
    vals = _put(vals, f, 2, i)
    vals = _put(vals, f, 3, ids)
    vals = _put(vals, f, 4, s)
    vals = _put(vals, f, 6, z + kind)
    vals = _put(vals, f, 7, owner)
    vals = _put(vals, f, 8, mapped)
    vals = _put(vals, f, 9, usage)
    vals = _put(vals, f, 10, z + tl.load(step_ptr))
    vals = _put(vals, f, 11, src)
    nbad = tl.sum(bad.to(tl.int64), axis=0)
    tl.store(counters_ptr + col, tl.load(counters_ptr + col) + nbad)
    return _write(ring_ptr, cursor, bad, vals, RING_N)


@triton.jit(do_not_specialize=["layer", "num_slots"])
def _plan_kernel(
    ids_ptr, slots_ptr, id_of_slot_ptr, slot_for_id_ptr, usage_ptr, step_ptr,
    dem_ptr, dem_num_ptr, pf_ptr, pf_num_ptr, nx_ptr, nx_num_ptr,
    ring_ptr, cursor_ptr, counters_ptr, calls_ptr,
    u_ids_ptr, u_slots_ptr, u_rows_ptr, u_num_ptr, slots_copy_ptr,
    layer, num_slots,
    N: tl.constexpr, N_P: tl.constexpr, E: tl.constexpr, CAP: tl.constexpr,
    DEM_W: tl.constexpr, DEM_P: tl.constexpr, PF_W: tl.constexpr, PF_P: tl.constexpr,
    HAS_PF: tl.constexpr, HAS_NX: tl.constexpr, BYTES: tl.constexpr, RING_N: tl.constexpr,
):
    """Before the GEMM: count the call, check the maps, keep the slots for the after-GEMM checks and
    list the distinct (expert, slot) pairs to byte-check (first CAP of them)."""
    call = tl.load(calls_ptr + layer)
    tl.store(calls_ptr + layer, call + 1)
    row = counters_ptr + layer * _NC
    tl.store(row + 0, tl.load(row + 0) + 1)
    cursor = tl.load(cursor_ptr)
    cursor = _meta(ids_ptr, slots_ptr, id_of_slot_ptr, slot_for_id_ptr, usage_ptr, step_ptr,
                   dem_ptr, dem_num_ptr, pf_ptr, pf_num_ptr, nx_ptr, nx_num_ptr,
                   ring_ptr, cursor, row, 1, 0, call, layer, num_slots,
                   N, N_P, E, DEM_W, DEM_P, PF_W, PF_P, HAS_PF, HAS_NX, RING_N)
    tl.store(cursor_ptr, cursor)
    i = tl.arange(0, N_P)
    m = i < N
    ids = tl.load(ids_ptr + i, mask=m, other=-1)
    s = tl.load(slots_ptr + i, mask=m, other=-1)
    tl.store(slots_copy_ptr + i, s, mask=m)
    if BYTES:
        # an unreadable slot or id is already a meta record; a duplicate pair shares its first row
        ok = m & (s >= 0) & (s < num_slots) & (ids >= 0) & (ids < E)
        same = (ids[:, None] == ids[None, :]) & (s[:, None] == s[None, :]) & (i[None, :] < i[:, None]) & ok[None, :]
        first = ok & (tl.sum(same.to(tl.int32), axis=1) == 0)
        rank = tl.cumsum(first.to(tl.int32), axis=0) - 1
        take = first & (rank < CAP)
        tl.store(u_ids_ptr + rank, ids, mask=take)
        tl.store(u_slots_ptr + rank, s, mask=take)
        tl.store(u_rows_ptr + rank, i, mask=take)
        total = tl.sum(first.to(tl.int64), axis=0)
        kept = tl.minimum(total, CAP)
        tl.store(u_num_ptr, kept)
        tl.store(row + 5, tl.load(row + 5) + total - kept)


@triton.jit
def _compare_kernel(cache_ptrs, ref_ptrs, words_ptr, u_slots_ptr, u_num_ptr, bad_cnt_ptr, bad_first_ptr,
                    NB: tl.constexpr, BLOCK: tl.constexpr):
    """Program (row, bank, chunk): compare one chunk of a routed slot's row with its host copy."""
    r = tl.program_id(0)
    b = tl.program_id(1)
    words = tl.load(words_ptr + b)
    off = tl.program_id(2).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = (r < tl.load(u_num_ptr)) & (off < words)
    slot = tl.load(u_slots_ptr + r).to(tl.int64)
    got = tl.load(tl.load(cache_ptrs + b).to(tl.pointer_type(tl.int32)) + slot * words + off, mask=m, other=0)
    want = tl.load(tl.load(ref_ptrs + b).to(tl.pointer_type(tl.int32)) + r * words + off, mask=m, other=0)
    bad = m & (got != want)
    nbad = tl.sum(bad.to(tl.int32), axis=0)
    if nbad > 0:
        tl.atomic_add(bad_cnt_ptr + r * NB + b, nbad)
        tl.atomic_min(bad_first_ptr + r * NB + b, tl.min(tl.where(bad, off, _NO_BAD), axis=0).to(tl.int32))


@triton.jit(do_not_specialize=["layer", "num_slots"])
def _tally_kernel(
    ids_ptr, slots_copy_ptr, id_of_slot_ptr, slot_for_id_ptr, usage_ptr, step_ptr,
    dem_ptr, dem_num_ptr, pf_ptr, pf_num_ptr, nx_ptr, nx_num_ptr,
    ring_ptr, cursor_ptr, counters_ptr, calls_ptr,
    u_ids_ptr, u_slots_ptr, u_rows_ptr, u_num_ptr, bad_cnt_ptr, bad_first_ptr,
    layer, num_slots,
    N: tl.constexpr, N_P: tl.constexpr, E: tl.constexpr, CAP: tl.constexpr, CAP_P: tl.constexpr,
    NB: tl.constexpr, NB_P: tl.constexpr,
    DEM_W: tl.constexpr, DEM_P: tl.constexpr, PF_W: tl.constexpr, PF_P: tl.constexpr,
    HAS_PF: tl.constexpr, HAS_NX: tl.constexpr, BYTES: tl.constexpr, POST: tl.constexpr,
    BYTES_KIND: tl.constexpr, BYTES_COL: tl.constexpr, RING_N: tl.constexpr,
):
    """Turn the compare flags into counts and records (and clear them); after the GEMM also recheck
    the maps of the slots it read."""
    call = tl.load(calls_ptr + layer) - 1
    row = counters_ptr + layer * _NC
    cursor = tl.load(cursor_ptr)
    if POST:
        cursor = _meta(ids_ptr, slots_copy_ptr, id_of_slot_ptr, slot_for_id_ptr, usage_ptr, step_ptr,
                       dem_ptr, dem_num_ptr, pf_ptr, pf_num_ptr, nx_ptr, nx_num_ptr,
                       ring_ptr, cursor, row, 4, 2, call, layer, num_slots,
                       N, N_P, E, DEM_W, DEM_P, PF_W, PF_P, HAS_PF, HAS_NX, RING_N)
    if BYTES:
        j = tl.arange(0, CAP_P * NB_P)
        r = j // NB_P
        b = j % NB_P
        cell = (r < CAP) & (b < NB)
        live = cell & (r < tl.load(u_num_ptr))
        cnt = tl.load(bad_cnt_ptr + r * NB + b, mask=live, other=0)
        first = tl.load(bad_first_ptr + r * NB + b, mask=live, other=_NO_BAD)
        tl.store(bad_cnt_ptr + r * NB + b, tl.zeros([CAP_P * NB_P], tl.int32), mask=cell)
        tl.store(bad_first_ptr + r * NB + b, tl.zeros([CAP_P * NB_P], tl.int32) + _NO_BAD, mask=cell)
        bad = live & (cnt > 0)
        ids = tl.load(u_ids_ptr + r, mask=live, other=0)
        s = tl.load(u_slots_ptr + r, mask=live, other=-1)
        ok_s = live & (s >= 0) & (s < num_slots)
        src = _source(s, dem_ptr, dem_num_ptr, pf_ptr, pf_num_ptr, nx_ptr, nx_num_ptr, DEM_W, DEM_P, PF_W, PF_P, HAS_PF, HAS_NX)
        f = tl.arange(0, 16)
        z = tl.zeros([CAP_P * NB_P], tl.int64)
        vals = tl.zeros([CAP_P * NB_P, 16], tl.int64) - 1
        vals = _put(vals, f, 0, z + call)
        vals = _put(vals, f, 1, z + layer)
        vals = _put(vals, f, 2, tl.load(u_rows_ptr + r, mask=live, other=-1))
        vals = _put(vals, f, 3, ids)
        vals = _put(vals, f, 4, s)
        vals = _put(vals, f, 5, b)
        vals = _put(vals, f, 6, z + BYTES_KIND)
        vals = _put(vals, f, 7, tl.load(id_of_slot_ptr + s, mask=ok_s, other=-2))
        vals = _put(vals, f, 8, tl.load(slot_for_id_ptr + layer * E + ids, mask=live, other=-2))
        vals = _put(vals, f, 9, tl.load(usage_ptr + s, mask=ok_s, other=-1))
        vals = _put(vals, f, 10, z + tl.load(step_ptr))
        vals = _put(vals, f, 11, src)
        vals = _put(vals, f, 12, first.to(tl.int64) * 4)
        vals = _put(vals, f, 13, cnt)
        tl.store(row + BYTES_COL, tl.load(row + BYTES_COL) + tl.sum(bad.to(tl.int64), axis=0))
        cursor = _write(ring_ptr, cursor, bad, vals, RING_N)
    tl.store(cursor_ptr, cursor)


class ExpertVerifier:
    """Device counters, the record ring and (``full``) the scratch the byte checks compare against.

    ``begin`` copies the router's ids before ensure rewrites them, ``before_gemm`` runs after the
    joins and the demand copy, ``after_gemm`` right after the GEMM; all three enqueue on the current
    (compute) stream with fixed shapes and no host sync, so they capture into the decode graph.
    ``bind`` sizes the scratch from the banks once and refreshes the slot bank addresses in place
    (a rebuild reallocates the banks); counters survive rebuilds and resets.
    """

    def __init__(self, num_layers: int, num_experts: int, device: torch.device, *, mode: str = "full",
                 rows: int | None = None) -> None:
        from freetoken.kernel.fast_index_copy import _skip_fast_index_copy_enabled

        if mode == "full" and _skip_fast_index_copy_enabled():
            logger.warning(
                "FREETOKEN_MOE_PREFETCH_VERIFY=1 with FREETOKEN_SKIP_FAST_INDEX_COPY: the host rows cannot be "
                "gathered, so only the slot maps are checked (meta)"
            )
            mode = "meta"
        self.mode = mode
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.device = device
        self.rows = ENV.MOE_PREFETCH_VERIFY_ROWS.value if rows is None else int(rows)
        if self.rows < 1:
            raise ValueError(f"FREETOKEN_MOE_PREFETCH_VERIFY_ROWS={self.rows} must be >= 1")
        self.counters = torch.zeros((num_layers, NUM_COLS), dtype=torch.int64, device=device)
        self.totals = torch.zeros((num_layers, NUM_COLS), dtype=torch.int64)
        self.calls = torch.zeros((num_layers,), dtype=torch.int64, device=device)
        self.ring = torch.full((RING, NUM_FIELDS), -1, dtype=torch.int64, device=device)
        self.cursor = torch.zeros((1,), dtype=torch.int64, device=device)
        self.ids = torch.full((MAX_ENTRIES,), -1, dtype=torch.int32, device=device)
        self.slots = torch.full((MAX_ENTRIES,), -1, dtype=torch.int32, device=device)
        self.u_ids = torch.zeros((self.rows,), dtype=torch.int32, device=device)
        self.u_slots = torch.zeros((self.rows,), dtype=torch.int32, device=device)
        self.u_rows = torch.zeros((self.rows,), dtype=torch.int32, device=device)
        self.u_num = torch.zeros((1,), dtype=torch.int64, device=device)
        self.iota = torch.arange(self.rows, dtype=torch.int32, device=device)
        self.bank_names: list[str] = []
        self.scratch: torch.Tensor | None = None
        self._printed = 0
        self._top_k = 1
        self._call = None  # (layer, entries, this layer's prefetch plan or None) between begin and after_gemm
        self._next = None
        logger.info(
            f"MoE verify {self.mode}: checking every GPU decode layer call's slot maps"
            + (f" and slot bytes before and after the GEMM (up to {self.rows} experts a call)" if self.mode == "full" else "")
        )

    def bind(self, cache) -> None:
        """Size the scratch for ``cache``'s banks (first call) or point the checks at its new banks."""
        caches = [bank for _, bank in cache.banks]
        feats = [math.prod(bank.shape[1:]) * bank.element_size() for bank in caches]
        if self.bank_names:
            assert list(cache.bank_schema) == self.bank_names and feats == self._feats, "the bank layout changed"
        else:
            assert all(f % 4 == 0 for f in feats), feats
            self.bank_names = list(cache.bank_schema)
            self._feats = feats
            self.cache_ptrs = torch.zeros((len(feats),), dtype=torch.int64, device=self.device)
            if self.mode == "full":
                self._alloc_scratch(cache, feats)
        self.cache_ptrs.copy_(torch.tensor([bank.data_ptr() for bank in caches], dtype=torch.int64))

    def _alloc_scratch(self, cache, feats: list[int]) -> None:
        rows = self.rows
        self.scratch = torch.empty((rows * sum(feats),), dtype=torch.uint8, device=self.device)
        offsets = [rows * sum(feats[:b]) for b in range(len(feats))]
        self.scratch_views = [
            self.scratch[o : o + rows * f].view(bank.dtype).view(rows, *bank.shape[1:])
            for o, f, (_, bank) in zip(offsets, feats, cache.banks)
        ]
        self.scratch_ptrs = torch.tensor([self.scratch.data_ptr() + o for o in offsets], dtype=torch.int64, device=self.device)
        self.words = torch.tensor([f // 4 for f in feats], dtype=torch.int64, device=self.device)
        self.bad_cnt = torch.zeros((rows * len(feats),), dtype=torch.int32, device=self.device)
        self.bad_first = torch.full((rows * len(feats),), 0x7FFFFFFF, dtype=torch.int32, device=self.device)
        self._chunks = triton.cdiv(max(feats) // 4, CHUNK_WORDS)
        logger.info(f"MoE verify scratch: {self.scratch.numel() / 2**20:.1f} MiB for {rows} experts")

    def begin(self, cache, layer_id: int, topk_ids: torch.Tensor, plan: tuple | None) -> None:
        """Copy the router's ids of ``layer_id`` (``plan``: this layer's prefetch copy plan, if forked)."""
        assert topk_ids.dtype == torch.int32 and topk_ids.is_contiguous()
        n = topk_ids.numel()
        self._top_k = topk_ids.shape[-1]
        if n > MAX_ENTRIES:
            self.counters[layer_id, UNCHECKED] += n
            self._call = None
            return
        self.ids[:n].copy_(topk_ids.view(-1))
        self._call = (layer_id, n, plan)

    def _plans(self, cache) -> tuple:
        _, n, plan = self._call
        pf, nx = plan, self._next
        dummy = (self.u_slots, self.u_num)
        width = (pf or nx or dummy)[0].numel()
        return (
            cache.evict_slots, cache.num_indices, *(pf or dummy), *(nx or dummy),
            dict(DEM_W=n, DEM_P=triton.next_power_of_2(n), PF_W=width, PF_P=triton.next_power_of_2(width),
                 HAS_PF=pf is not None, HAS_NX=nx is not None),
        )

    def before_gemm(self, cache, slots: torch.Tensor, next_plan: tuple | None) -> None:
        """Check the maps and (``full``) the bytes of the slots ``slots`` the GEMM is about to read
        (``next_plan``: the next layer's prefetch plan, if one was issued)."""
        if self._call is None:
            return
        layer, n, _ = self._call
        self._next = next_plan
        *plans, plan_kw = self._plans(cache)
        full = self.mode == "full"
        _plan_kernel[(1,)](
            self.ids, slots, cache.id_of_slot, cache.slot_for_id.view(-1), cache.usage, cache.step, *plans,
            self.ring, self.cursor, self.counters, self.calls,
            self.u_ids, self.u_slots, self.u_rows, self.u_num, self.slots,
            layer, cache.cache_size,
            N=n, N_P=triton.next_power_of_2(n), E=self.num_experts, CAP=self.rows,
            BYTES=full, RING_N=RING, num_warps=4, **plan_kw,
        )
        if not full:
            return
        self._gather(cache, layer)
        self._compare_and_tally(cache, post=False)

    def after_gemm(self, cache) -> None:
        """Recheck the maps and (``full``) the bytes of the slots the GEMM just read."""
        if self._call is None:
            return
        if self.mode == "full":
            self._compare_and_tally(cache, post=True)
        else:
            self._tally(cache, post=True, bytes_=False)
        self._call = self._next = None

    def _gather(self, cache, layer: int) -> None:
        if cache._copy_fused_ok:
            from freetoken.kernel.fast_index_copy import fast_index_copy_multi_jit

            fast_index_copy_multi_jit(
                self.scratch_ptrs, cache._copy_src_ptrs[layer], cache._copy_feat_bytes, self.iota, self.u_ids, self.u_num,
            )
            return
        from freetoken.kernel import fast_index_copy_jit

        for (per_layer, _), view in zip(cache.banks, self.scratch_views):
            fast_index_copy_jit(view, self.iota, per_layer[layer], self.u_ids, self.u_num)

    def _compare_and_tally(self, cache, *, post: bool) -> None:
        nb = len(self.bank_names)
        _compare_kernel[(self.rows, nb, self._chunks)](
            self.cache_ptrs, self.scratch_ptrs, self.words, self.u_slots, self.u_num, self.bad_cnt, self.bad_first,
            NB=nb, BLOCK=CHUNK_WORDS, num_warps=8,
        )
        self._tally(cache, post=post, bytes_=True)

    def _tally(self, cache, *, post: bool, bytes_: bool) -> None:
        layer, n, _ = self._call
        *plans, plan_kw = self._plans(cache)
        nb = max(len(self.bank_names), 1)
        _tally_kernel[(1,)](
            self.ids, self.slots, cache.id_of_slot, cache.slot_for_id.view(-1), cache.usage, cache.step, *plans,
            self.ring, self.cursor, self.counters, self.calls,
            self.u_ids, self.u_slots, self.u_rows, self.u_num,
            self.bad_cnt if bytes_ else self.u_rows, self.bad_first if bytes_ else self.u_rows,
            layer, cache.cache_size,
            N=n, N_P=triton.next_power_of_2(n), E=self.num_experts, CAP=self.rows,
            CAP_P=triton.next_power_of_2(self.rows), NB=nb, NB_P=triton.next_power_of_2(nb),
            BYTES=bytes_, POST=post, BYTES_KIND=BYTES_POST if post else BYTES_PRE, BYTES_COL=POST_BAD if post else PRE_BAD,
            RING_N=RING, num_warps=4, **plan_kw,
        )

    # ------------------------------------------------------------------ reporting (host syncs)

    def take_window(self) -> tuple[torch.Tensor, list[list[int]], int]:
        """This window's ``[num_layers, NUM_COLS]`` counts (added to ``totals``; the device ones
        restart), the records kept since the last call, and how many records there were in all."""
        window = self.counters.cpu()
        self.counters.zero_()
        self.totals += window
        cursor = int(self.cursor.item())
        upto = min(cursor, RING)
        records = self.ring[self._printed : upto].cpu().tolist() if upto > self._printed else []
        self._printed = max(self._printed, upto)
        return window, records, cursor

    def report_window(self, steps: int) -> list[tuple[str, str]]:
        """(level, line) pairs for one MOE_STATS_INTERVAL window; nothing when no call was checked."""
        window, records, cursor = self.take_window()
        if not int(window[:, CHECKS].sum()) and not records:
            return []
        bad = self._bad(window) or bool(records)
        session = self._bad(self.totals)
        lines = [("warning" if bad else "info",
                  f"MoE verify {self.mode} ({steps} decode steps): {self.format_counts(window)}; session bad={session}")]
        lines += [("warning", "MoE verify record: " + self.format_record(r)) for r in records]
        if bad and cursor > RING:
            lines.append(("warning", f"MoE verify: {cursor} bad records in all, the first {RING} are kept"))
        return lines

    def report_session(self) -> list[tuple[str, str]]:
        """Session counts and every kept record (shutdown)."""
        totals = self.totals + self.counters.cpu()
        cursor = int(self.cursor.item())
        if not int(totals[:, CHECKS].sum()):
            return []
        bad = self._bad(totals)
        lines = [("warning" if bad else "info", f"MoE verify {self.mode} (session): {self.format_counts(totals)}")]
        records = self.ring[: min(cursor, RING)].cpu().tolist()
        lines += [("warning", "MoE verify record: " + self.format_record(r)) for r in records]
        if cursor > RING:
            lines.append(("warning", f"MoE verify: {cursor} bad records in all, the first {RING} are kept"))
        return lines

    @staticmethod
    def _bad(counts: torch.Tensor) -> int:
        return int(counts[:, [META_BAD, PRE_BAD, POST_BAD, POST_META_BAD]].sum())

    def format_counts(self, counts: torch.Tensor) -> str:
        total = counts.sum(0).tolist()
        text = f"{total[CHECKS]} layer calls checked, " + ", ".join(
            f"{COL_NAMES[c]}={total[c]}" for c in (META_BAD, PRE_BAD, POST_BAD, POST_META_BAD)
        )
        if total[UNCHECKED]:
            text += f", unchecked experts={total[UNCHECKED]}"
        per_layer = [
            f"L{layer}=" + "/".join(str(row[c]) for c in (META_BAD, PRE_BAD, POST_BAD, POST_META_BAD))
            for layer, row in enumerate(counts.tolist())
            if any(row[c] for c in (META_BAD, PRE_BAD, POST_BAD, POST_META_BAD))
        ]
        if per_layer:
            text += " (meta/pre/post/post_meta by layer: " + ", ".join(per_layer) + ")"
        return text

    def format_record(self, rec: list[int]) -> str:
        e = self.num_experts
        kind = KINDS[rec[R_KIND]] if 0 <= rec[R_KIND] < len(KINDS) else str(rec[R_KIND])
        owner = rec[R_OWNER]
        owner_text = f"L{owner // e}/e{owner % e}" if owner >= 0 else str(owner)
        src = [name for bit, name in ((SRC_DEMAND, "demand copy"), (SRC_PREFETCH, "prefetch copy"),
                                       (SRC_NEXT_PREFETCH, "next layer's prefetch plan")) if rec[R_SOURCE] & bit]
        row = rec[R_ROW]
        text = (
            f"{kind} call={rec[R_CALL]} layer={rec[R_LAYER]} row={row} (token {row // self._top_k}, rank {row % self._top_k}) "
            f"expert={rec[R_EXPERT]} slot={rec[R_SLOT]}"
        )
        if rec[R_BANK] >= 0:
            bank = self.bank_names[rec[R_BANK]] if rec[R_BANK] < len(self.bank_names) else rec[R_BANK]
            text += f" bank={bank} first_bad_byte={rec[R_FIRST_BYTE]} bad_words={rec[R_BAD_WORDS]}"
        return text + (
            f" | slot now holds {owner_text}, slot_for_id={rec[R_MAPPED]}, usage={rec[R_USAGE]}, lru_step={rec[R_STEP]}, "
            f"slot in: {', '.join(src) or 'no copy plan (a resident hit)'}"
        )
