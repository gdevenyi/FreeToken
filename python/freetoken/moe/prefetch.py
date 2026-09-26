"""Cross-layer expert prefetch for GPU decode (``FREETOKEN_MOE_PREFETCH``).

Stage A, ``measure``: a training-free router lookahead predicts layer L's experts from the
tensor layer L-1's router consumed (``W_gate[L] . x_{L-1}``, top-K), keeps the first BUDGET
candidates that are not resident (what a prefetch would copy), and after layer L's ensure counts
how many of them layer L actually routed. Nothing is copied and the cache is never touched.

Streams, per decode step and layer L >= 1 (layer 0 has no predecessor):

    compute: ... ensure(L-1) -fork-> copy(L-1) GEMM(L-1) ... ensure(L) copy(L) -join-> count(L) GEMM(L)
    predict:                  \\-> router_L(x_{L-1}) -> top-K -> select(L) -/

The predictor reads ``slot_for_id[L]`` without ordering against ensure(L); it finishes long
before ensure(L) starts (~225 us of layer compute between them), and a late read only skews the
counters, never the model. At bs 2 both rows' top-K lists merge rank by rank (row 0's first,
then row 1's first, ...), duplicates dropped, under one budget; a kept id is useful once if
either row routes it.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from freetoken.env import ENV
from freetoken.utils import init_logger

logger = init_logger(__name__)

MODES = ("off", "measure", "on")
# non-resident candidates per target layer: GDN layers leave a shorter copy window than QSA ones
GDN_BUDGET = 3
ATTN_BUDGET = 4
# counters[layer, col]
ISSUED, USEFUL, RESIDENT_HITS, CALLS, ROWS, MISSES = range(6)
NUM_COLS = 6
# the select kernel's dedup tile is (rows * k)^2; larger (eager-only) decode batches skip the prediction
MAX_ROWS = 8


def resolve_mode(mode: str | None = None) -> str:
    mode = (ENV.MOE_PREFETCH.value if mode is None else mode).strip().lower()
    if mode not in MODES:
        raise ValueError(f"FREETOKEN_MOE_PREFETCH={mode!r}: expected one of {', '.join(MODES)}")
    if mode == "on":
        raise NotImplementedError("FREETOKEN_MOE_PREFETCH=on (prefetch copies) is not built yet; use measure")
    return mode


def default_budget(before_linear: bool) -> int:
    """Non-resident candidates kept for a target layer, by the target's attention kind."""
    return GDN_BUDGET if before_linear else ATTN_BUDGET


def dedicated_stream(device: torch.device) -> torch.cuda.ExternalStream:
    # not one of torch's 32 pooled streams: a pooled one can alias the engine or capture stream
    from cuda.bindings import runtime as cudart

    with torch.cuda.device(device):
        err, handle = cudart.cudaStreamCreateWithFlags(cudart.cudaStreamNonBlocking)
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"cudaStreamCreateWithFlags failed: {err}")
    return torch.cuda.ExternalStream(int(handle), device=device)


@triton.jit(do_not_specialize=["stride", "budget"])
def _lookahead_select_kernel(
    logits_ptr, stride, resident_ptr, sel_ptr, res_ptr, budget,
    ROWS: tl.constexpr, ROWS_P: tl.constexpr, E: tl.constexpr, E_P: tl.constexpr,
    K: tl.constexpr, N_P: tl.constexpr, W: tl.constexpr, W_P: tl.constexpr,
):
    b = tl.arange(0, ROWS_P)
    e = tl.arange(0, E_P)
    live = (b < ROWS)[:, None] & (e < E)[None, :]
    v = tl.load(logits_ptr + b[:, None] * stride + e[None, :], mask=live, other=float("-inf")).to(tl.float32)
    # candidates in merged order: position r * ROWS + b holds row b's rank-r id
    j = tl.arange(0, N_P)
    cand = tl.full([N_P], -1, tl.int32)
    for r in tl.static_range(K):
        top = tl.argmax(v, axis=1).to(tl.int32)  # ties go to the lowest id
        at = (j[:, None] == r * ROWS + b[None, :]) & (b < ROWS)[None, :]
        cand = tl.where(tl.sum(at.to(tl.int32), axis=1) > 0, tl.sum(tl.where(at, top[None, :], 0), axis=1), cand)
        v = tl.where(e[None, :] == top[:, None], float("-inf"), v)
    valid = j < ROWS * K
    earlier = (cand[:, None] == cand[None, :]) & (j[None, :] < j[:, None]) & valid[None, :]
    first = valid & (tl.sum(earlier.to(tl.int32), axis=1) == 0)
    held = tl.load(resident_ptr + cand, mask=first, other=-1) >= 0
    pick = first & (~held)
    rank = tl.cumsum(pick.to(tl.int32), axis=0) - 1
    kept = first & held & (tl.cumsum(first.to(tl.int32), axis=0) <= budget)
    krank = tl.cumsum(kept.to(tl.int32), axis=0) - 1
    p = tl.arange(0, W_P)
    m_sel = pick[None, :] & (rank[None, :] == p[:, None]) & (p < budget)[:, None]
    m_res = kept[None, :] & (krank[None, :] == p[:, None])
    # at most one candidate matches each output position
    sel = tl.sum(tl.where(m_sel, cand[None, :] + 1, 0), axis=1) - 1
    res = tl.sum(tl.where(m_res, cand[None, :] + 1, 0), axis=1) - 1
    tl.store(sel_ptr + p, sel, mask=p < W)
    tl.store(res_ptr + p, res, mask=p < W)


@triton.jit(do_not_specialize=["id_base"])
def _prefetch_count_kernel(
    slots_ptr, id_of_slot_ptr, misses_ptr, sel_ptr, res_ptr, counters_ptr, id_base,
    N: tl.constexpr, N_P: tl.constexpr, W: tl.constexpr, W_P: tl.constexpr, ROWS: tl.constexpr,
):
    i = tl.arange(0, N_P)
    slot = tl.load(slots_ptr + i, mask=i < N, other=-1)
    # ensure rewrote the routed ids to slots, all pinned to this layer's ids until the next ensure
    routed = tl.where(slot >= 0, tl.load(id_of_slot_ptr + slot, mask=slot >= 0, other=0) - id_base, -1)
    p = tl.arange(0, W_P)
    sel = tl.load(sel_ptr + p, mask=p < W, other=-1)
    res = tl.load(res_ptr + p, mask=p < W, other=-1)
    sel_hit = tl.sum(((sel[:, None] == routed[None, :]) & (i < N)[None, :]).to(tl.int32), axis=1) > 0
    res_hit = tl.sum(((res[:, None] == routed[None, :]) & (i < N)[None, :]).to(tl.int32), axis=1) > 0
    issued = tl.sum((sel >= 0).to(tl.int32))
    useful = tl.sum((sel_hit & (sel >= 0)).to(tl.int32))
    res_hits = tl.sum((res_hit & (res >= 0)).to(tl.int32))
    misses = tl.load(misses_ptr)
    c = tl.arange(0, 8)
    add = tl.where(c == 0, issued, tl.where(c == 1, useful, tl.where(c == 2, res_hits, tl.where(c == 3, 1, ROWS))))
    add = tl.where(c == 5, misses, add.to(tl.int64))
    # one writer per layer row, stream-ordered: no atomics needed
    old = tl.load(counters_ptr + c, mask=c < 6, other=0)
    tl.store(counters_ptr + c, old + add, mask=c < 6)


def lookahead_select(
    logits: torch.Tensor, resident: torch.Tensor, sel: torch.Tensor, res: torch.Tensor, *, k: int, budget: int
) -> None:
    """Top-``k`` of each ``logits`` row, merged rank by rank; ``sel`` gets the first ``budget``
    distinct ids with ``resident[id] < 0`` and ``res`` the resident ids among the first ``budget``
    distinct ones, both in merge order and padded with -1. One CTA, no host sync."""
    rows, num_experts = logits.shape
    width = sel.numel()
    assert logits.stride(1) == 1 and resident.numel() == num_experts and resident.dtype == torch.int32
    assert sel.dtype == res.dtype == torch.int32 and res.numel() == width and 1 <= k <= num_experts
    _lookahead_select_kernel[(1,)](
        logits, logits.stride(0), resident, sel, res, min(budget, width),
        ROWS=rows, ROWS_P=triton.next_power_of_2(rows),
        E=num_experts, E_P=triton.next_power_of_2(num_experts),
        K=k, N_P=triton.next_power_of_2(rows * k),
        W=width, W_P=triton.next_power_of_2(width),
        num_warps=4,
    )


def prefetch_count(
    slots: torch.Tensor, id_of_slot: torch.Tensor, misses: torch.Tensor,
    sel: torch.Tensor, res: torch.Tensor, counters: torch.Tensor, *, id_base: int, rows: int,
) -> None:
    """Add one call's issued / useful / resident-hit / call / row / miss counts to ``counters``."""
    assert slots.dtype == torch.int32 and slots.is_contiguous() and counters.dtype == torch.int64
    n, width = slots.numel(), sel.numel()
    _prefetch_count_kernel[(1,)](
        slots, id_of_slot, misses, sel, res, counters, id_base,
        N=n, N_P=triton.next_power_of_2(n), W=width, W_P=triton.next_power_of_2(width), ROWS=rows,
        num_warps=1,
    )


class ExpertPrefetcher:
    """Predictor stream, per-layer buffers and device counters for the router lookahead.

    ``fork`` runs layer L's prediction on the dedicated stream right after layer L-1's ensure;
    ``join_and_count`` waits for it after layer L's copy is enqueued and counts on the compute
    stream. The events are created up front, the shapes are fixed and nothing syncs the host,
    so both sides capture into the decode graph. Nothing here is cache_size-shaped: a rebuild
    keeps it and only resets the counters.
    """

    def __init__(
        self, num_layers: int, num_experts: int, device: torch.device, *,
        mode: str = "measure", k: int | None = None, budget: int | None = None,
    ) -> None:
        self.mode = mode
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.device = device
        k = ENV.MOE_PREFETCH_K.value if k is None else int(k)
        if k < 1:
            raise ValueError(f"FREETOKEN_MOE_PREFETCH_K={k} must be >= 1")
        self.k = min(k, num_experts)
        # 0 = the per-layer default the model wires in
        self.budget_override = ENV.MOE_PREFETCH_BUDGET.value if budget is None else int(budget)
        if self.budget_override < 0:
            raise ValueError(f"FREETOKEN_MOE_PREFETCH_BUDGET={self.budget_override} must be >= 0")
        self.stream = dedicated_stream(device)
        self._events = [(torch.cuda.Event(), torch.cuda.Event()) for _ in range(num_layers)]
        # torch creates the CUDA event on its first record: do that now, never inside a capture
        for pair in self._events:
            for event in pair:
                event.record(self.stream)
        width = 2 * self.k  # room for the whole bs-2 union
        self.sel = torch.full((num_layers, width), -1, dtype=torch.int32, device=device)
        self.res = torch.full((num_layers, width), -1, dtype=torch.int32, device=device)
        self.counters = torch.zeros((num_layers, NUM_COLS), dtype=torch.int64, device=device)
        self.totals = torch.zeros((num_layers, NUM_COLS), dtype=torch.int64)
        # the router input of each in-flight prediction, alive until its join is enqueued so the
        # allocator cannot hand its block to later compute-stream work the predictor still reads
        self._inflight: list[torch.Tensor | None] = [None] * num_layers
        logger.info(
            f"MoE prefetch {mode}: router lookahead top-{self.k}, budget "
            + (str(self.budget_override) if self.budget_override else f"{GDN_BUDGET} before GDN / {ATTN_BUDGET} before attention layers")
        )

    def budget_for(self, wired: int) -> int:
        return min(self.budget_override or wired, self.sel.shape[1])

    def fork(self, target: int, x: torch.Tensor, gate, resident: torch.Tensor, budget: int) -> None:
        """Predict ``target``'s experts from ``x`` (the previous layer's router input) on the
        predictor stream, ordered after the work already on the current stream."""
        if x.shape[0] > MAX_ROWS:
            return
        forked, done = self._events[target]
        forked.record(torch.cuda.current_stream(self.device))
        self.stream.wait_event(forked)
        with torch.cuda.stream(self.stream):
            logits = gate.forward(x)
            lookahead_select(logits, resident, self.sel[target], self.res[target], k=self.k, budget=self.budget_for(budget))
            done.record(self.stream)
        self._inflight[target] = x

    def join_and_count(self, layer_id: int, slots: torch.Tensor, id_of_slot: torch.Tensor, misses: torch.Tensor) -> None:
        """Join ``layer_id``'s prediction (if one was forked this step) and count it against the
        routed slots ``slots`` of this layer's ensure."""
        x = self._inflight[layer_id]
        if x is None:
            return
        torch.cuda.current_stream(self.device).wait_event(self._events[layer_id][1])
        prefetch_count(
            slots.view(-1), id_of_slot, misses, self.sel[layer_id], self.res[layer_id], self.counters[layer_id],
            id_base=layer_id * self.num_experts, rows=x.shape[0],
        )
        self._inflight[layer_id] = None

    def reset_counters(self) -> None:
        self.counters.zero_()

    def reset(self) -> None:
        """Counters and in-flight state; the stream, events and buffers stay."""
        self.counters.zero_()
        self.totals.zero_()
        self._inflight = [None] * self.num_layers

    def take_window(self) -> torch.Tensor:
        """This window's counters on the host (one sync), added to ``totals``; the device ones restart."""
        window = self.counters.cpu()
        self.counters.zero_()
        self.totals += window
        return window


def summarize(counters: torch.Tensor) -> dict:
    """Aggregate ``[num_layers, NUM_COLS]`` counters into per-layer-call and per-token rates."""
    total = counters.sum(0).tolist()
    calls, issued, useful = total[CALLS], total[ISSUED], total[USEFUL]
    tokens = int(counters[:, ROWS].max()) if counters.numel() else 0
    return {
        "layer_calls": calls,
        "tokens": tokens,
        "issued_per_layer": issued / calls if calls else 0.0,
        "useful_per_layer": useful / calls if calls else 0.0,
        "resident_hits_per_layer": total[RESIDENT_HITS] / calls if calls else 0.0,
        "misses_per_layer": total[MISSES] / calls if calls else 0.0,
        "precision": useful / issued if issued else 0.0,
        "coverage": useful / total[MISSES] if total[MISSES] else 0.0,
        "useful_per_token": useful / tokens if tokens else 0.0,
    }


def format_summary(stats: dict) -> str:
    return (
        f"issued/layer={stats['issued_per_layer']:.2f}, useful/layer={stats['useful_per_layer']:.2f}, "
        f"precision={stats['precision']:.3f}, useful/token={stats['useful_per_token']:.1f}, "
        f"misses/layer={stats['misses_per_layer']:.2f}, coverage={stats['coverage']:.3f}, "
        f"resident_hits/layer={stats['resident_hits_per_layer']:.2f}"
    )


def format_per_layer(counters: torch.Tensor) -> str:
    parts = []
    for layer, row in enumerate(counters.tolist()):
        if row[CALLS]:
            c = row[CALLS]
            parts.append(f"L{layer}={row[USEFUL] / c:.2f}/{row[ISSUED] / c:.2f}/{row[MISSES] / c:.2f}")
    return ", ".join(parts)
