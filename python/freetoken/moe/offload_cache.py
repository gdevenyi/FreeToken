from __future__ import annotations

import math
import os
import weakref
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator

import torch
from flashlib.kernels.slot_cache import N_STATS, Stat

# Fuse the per-bank expert copies into a single multi-bank launch (one per copy_missing
# instead of one per bank). Set FREETOKEN_FUSED_COPY=0 to force the legacy per-bank path
# (kept for A/B profiling). Falls back to per-bank automatically if a bank's row bytes or
# base address are not 16-byte aligned.
_FUSED_COPY = os.getenv("FREETOKEN_FUSED_COPY", "1").strip().lower() not in {"0", "false", "no", "off"}
# Opt-in: run the fused copy_missing on the slim kernel (a few small CTAs, leaves most SMs to
# concurrent compute) instead of the 8-CTAs-per-bank one. Off by default.
_SLIM_COPY = os.getenv("FREETOKEN_MOE_SLIM_COPY", "0").strip().lower() in {"1", "true", "yes", "on"}

# cudaMemcpyBatchAsync silently degrades to a SYNCHRONOUS copy when a batch mixes
# large entries with sub-~256KB entries on registered host memory (H100 + CUDA 13.0,
# empirically bisected: a single 5-22KB entry beside one large entry blocks the
# calling thread for the full transfer; >=253KB entries never do). A synchronous
# call still moves bytes at full PCIe rate but stalls the host, which un-hides the
# GEMM under the copy in transition-zone workloads (gpt-oss 2048tok: -22% e2e).
# Banks whose rows are smaller than this ship as ONE whole-layer entry (their
# whole layer is tiny) and are excluded from the hit gather, so every per-run
# entry the batch sees is >= this size.
_SMALL_BANK_FEAT_BYTES = 256 * 1024

from freetoken.kernel.fast_index_copy import default_blocks_per_bank
from freetoken.env import ENV
from freetoken.utils import init_logger

if TYPE_CHECKING:
    from freetoken.moe.offload_kernels import HybridRoutes

logger = init_logger(__name__)

# quant_format -> bank names, in registration order: the single place a format's bank
# layout is declared. The cache machinery (copy_missing, the prefill double buffers,
# bank_views) iterates banks in this order, the layers' kernel dispatch unpacks views
# in this order, and set_bank_sources validates against it.
_BANK_SCHEMAS: dict[str, tuple[str, ...]] = {
    # dense bf16 expert weights
    "bf16": ("gate_up", "down"),
    # DeepSeek-V3-style 128x128 block-fp8 experts (Qwen3.5-FP8): fp8-e4m3 weights +
    # bf16 per-block weight_scale_inv. gate_up [L*E, 2I, H] fp8 + gate_up_scale
    # [L*E, 2I//128, H//128] bf16; down [L*E, H, I] fp8 + down_scale [L*E, H//128, I//128].
    # Half the host/cache footprint of bf16; the grouped GEMM (kernel/triton/fp8_blockscale_moe)
    # reads the routed fp8 rows directly and dequantizes in the K-loop (no bf16 materialization).
    "fp8_block": ("gate_up", "gate_up_scale", "down", "down_scale"),
    # GGUF experts in their own ggml type per layer: gate_up [L*E, 2I, row_bytes] (gate rows, then up rows), down [L*E, H, row_bytes]; float experts keep their dtype
    "gguf": ("gate_up", "down"),
    # GGUF experts whose NVFP4 tensors carry per-expert scales: global [L*E, 4] fp32 (gate, up, down, 1) weight_scale_2
    "gguf_scaled": ("gate_up", "down", "global"),
    # native ModelOpt rows for the Triton inline-dequant kernels: packed e2m1 codes +
    # fp8-e4m3 per-16 block scales + per-output-row fp32 globals (w1/w3 carry distinct
    # globals, and folding them into the e4m3 block scales would underflow)
    "nvfp4": (
        "gate_up_packed",
        "gate_up_scale",
        "gate_up_global",
        "down_packed",
        "down_scale",
        "down_global",
    ),
    # pre-tiled layouts for the borrowed kernels; the globals are folded into the
    # block scales at repack time and collapse to [L*E] GPU-resident alpha vectors
    # (set_alphas), so they are not banks
    "nvfp4_marlin": ("gate_up_packed", "gate_up_scale", "down_packed", "down_scale"),
    "nvfp4_b12x": ("gate_up_packed", "gate_up_scale", "down_packed", "down_scale"),
    # gpt-oss mxfp4, transposed split-K layout (N innermost): per-expert blocks_t
    # [K//2, N] (uint8), scales_t [K//32, N] (uint8 e8m0), bias [N]. No folded alphas
    # (scales are a bank); split-K GEMV decode + transposed _t grouped prefill.
    "mxfp4_triton": (
        "gate_up_blocks",
        "gate_up_scales",
        "gate_up_bias",
        "down_blocks",
        "down_scales",
        "down_bias",
    ),
    # DeepSeek-V4 FP4: packed e2m1 codes + e8m0 per-32 block scales, no global scale
    # (4 banks). Read by DeepSeek-V4's own DS-FP4 grouped GEMV kernels via pool_bank_views().
    "ds_fp4": ("gate_up_packed", "gate_up_scale", "down_packed", "down_scale"),
}

# lives in kernel/aot_models.py: the AOT row table shares it and must stay importable in the torch-only kernel-cache build env, which cannot import freetoken.moe
from freetoken.kernel.aot_models import fp8_block_scale_pad


# bytes per (expert, layer) as f(hidden, moe_intermediate), from the bank shapes above; keep in sync with _BANK_SCHEMAS
# keyed by the config-time format tag (expert_quant / moe_weight_format), not quant_format: "mxfp4" sizes the mxfp4_triton banks, "nvfp4" also covers its repacked variants
_BANK_BYTES_PER_EXPERT = {
    "bf16": lambda H, I: 3 * I * H * 2,
    "fp8_block": lambda H, I: 3 * I * H + (
        (2 * I // 128) * fp8_block_scale_pad(2 * I // 128, H // 128)
        + (H // 128) * fp8_block_scale_pad(H // 128, I // 128)
    ) * 2,
    "nvfp4": lambda H, I: 2 * I * (H // 2 + H // 16 + 4) + H * (I // 2 + I // 16 + 4),
    "mxfp4": lambda H, I: 2 * I * (H // 2 + H // 32 + 2) + H * (I // 2 + I // 32 + 2),
    "ds_fp4": lambda H, I: 2 * I * (H // 2 + H // 32) + H * (I // 2 + I // 32),
}

# vLLM's marlin grouped-GEMM hands the full [cache_size] slot cache as its expert
# dimension; moe_align_block_size requires round_up(experts, 32) < 1024, i.e. <= 992.
MARLIN_MAX_CACHE_SIZE = 992


def placeholder_bank(num_experts: int) -> torch.Tensor:
    """The source of a bank whose role a layer's expert kernel does not read (experts of several formats): zero bytes per row."""
    return torch.empty((num_experts, 0), dtype=torch.uint8)


def bank_layouts(sources: dict[str, list[torch.Tensor]], layer_tags: list | None = None) -> list[tuple]:
    """Each layer's layout key: its quant tag and every bank's (dtype, row shape). A slot pool holds layers of one key."""
    num_layers = len(next(iter(sources.values())))
    tags = layer_tags if layer_tags is not None else [None] * num_layers
    assert len(tags) == num_layers, (len(tags), num_layers)
    return [(tags[l], tuple((t[l].dtype, tuple(t[l].shape[1:])) for t in sources.values())) for l in range(num_layers)]


@dataclass
class OffloadMoeCache:
    num_layers: int
    num_experts: int
    cache_size: int
    device: torch.device
    cache_policy: str = "lru"
    prefill_overlap: bool = False
    # Prefill hit/miss split: experts already resident in the slot cache (slots
    # >= 2 * num_experts) are gathered device-side into the double buffer instead
    # of re-crossing PCIe; only the misses are H2D'd (one cudaMemcpyBatchAsync of
    # coalesced runs). Requires prefill_overlap, cache_size > 2 * num_experts and
    # the fused copy plan; silently falls back to the full-layer copy otherwise.
    prefill_hit_d2d: bool = False
    # "bf16" (default, dense expert weights) or one of the NVFP4 bank layouts:
    # "nvfp4" (native ModelOpt rows, FreeToken Triton kernels), "nvfp4_marlin"
    # (Marlin-tiled, vLLM W4A16 GEMM, sm_80-99) or "nvfp4_b12x" (flashinfer SM12x
    # W4A16); or "mxfp4_triton" (gpt-oss transposed split-K GEMV decode + _t grouped
    # prefill). The format names its bank layout (_BANK_SCHEMAS) and which kernels
    # may read the banks; the cache machinery itself is layout-agnostic.
    quant_format: str = "bf16"
    # Decode mode + bank layout; per-layer CPU routing is cpu_layer_ids. "gpu":
    # GPU-tiled banks, all decode on GPU (stream misses over PCIe into the slot
    # cache, GEMM on GPU). "cpu": native (CPU-readable) banks + a CPU executor;
    # decode computes experts on the CPU (the slot cache only backs the prefill
    # double buffer). "hybrid": native banks + a CPU executor + a full slot cache;
    # each layer fetches a capped subset of its misses over PCIe (``hybrid_max_fetch``
    # / ``hybrid_fetch_fraction`` below; the GPU computes those plus the hits) and the
    # CPU absorbs the overflow misses, then the partials merge. The CPU executor is
    # attached (set_cpu_executor) for cpu/hybrid, set whenever >=1 layer decodes on the CPU.
    decode_target: str = "gpu"
    # hybrid only: max experts fetched over PCIe per (layer, decode step); the rest
    # of that step's misses are computed on the CPU. 0 -> never fetch (CPU does every
    # miss, the GPU cache stays cold); large -> behaves like pure offload.
    hybrid_max_fetch: int = 1
    # hybrid only, one per MoE layer: when > 0, replaces the fixed cap with a per-step fraction -- fetch
    # ~fraction * misses experts over PCIe (rounded to whichever integer balances the
    # overlap best), the CPU computes the rest. The engine sets it to the benched
    # pcie_bw / cpu_bw ratio so the PCIe fetch and the CPU overflow GEMV take equal
    # time (perfect overlap): fetched : cpu = pcie : cpu - pcie.
    hybrid_fetch_fraction: list[float] | None = None
    # bank layout from the expert kernel (a BankSpec per role); when given it replaces the _BANK_SCHEMAS lookup and the slot cap comes from max_slots
    layout: dict | None = None
    max_slots: int | None = None
    # Per-layer kernel layouts for experts of several formats (replaces ``layout``). The banks are the union of their roles;
    # a layer whose kernel reads no such role carries a zero-byte [num_experts, 0] uint8 placeholder (see placeholder_bank).
    layer_layouts: list[dict] | None = None
    # Each layer belongs to one pool: a contiguous slot range with its own LRU scope; pool p owns slots [sum(pool_sizes[:p]), +pool_sizes[p]). Slot ids in slot_for_id are local to the pool. None = one pool for all layers.
    pool_of_layer: list[int] | None = None
    pool_sizes: list[int] | None = None
    # GPU decode only: the caller's work that is independent of the routed experts (qwen4_exp: the
    # shared expert) runs on decode_copy_stream beside the miss copy. None = FREETOKEN_MOE_COPY_OVERLAP.
    decode_copy_overlap: bool | None = None
    # GPU decode only: cross-layer expert prefetch (moe/prefetch.py). None = FREETOKEN_MOE_PREFETCH.
    prefetch_mode: str | None = None
    # GPU decode only: debug slot checks (moe/verify.py). None = FREETOKEN_MOE_PREFETCH_VERIFY.
    verify_mode: str | None = None
    # GPU decode only: debug slot writer history + audit every N decode steps (moe/slot_audit.py).
    # None = FREETOKEN_MOE_SLOT_AUDIT, 0 = off.
    slot_audit: int | None = None

    def __post_init__(self) -> None:
        from freetoken.moe.scored_ensure import POLICY_IDS

        assert self.cache_policy in POLICY_IDS, self.cache_policy
        assert self.decode_target in ("gpu", "cpu", "hybrid"), self.decode_target
        if self.cache_policy != "lru" and self.decode_target != "gpu":
            # the hybrid and CPU decode paths run their own kernels and would silently stay LRU
            raise ValueError(
                f"--moe-cache-policy {self.cache_policy} needs GPU decode, not {self.decode_target!r}"
            )
        if self.layer_layouts is not None:
            assert self.layout is None, "pass layout or layer_layouts, not both"
            assert len(self.layer_layouts) == self.num_layers, (len(self.layer_layouts), self.num_layers)
            if any(spec.resident for layout in self.layer_layouts for spec in layout.values()):
                raise ValueError("experts of several formats cannot use a kernel with GPU-resident alphas (marlin, b12x)")
        elif self.layout is None:
            assert self.quant_format in _BANK_SCHEMAS, f"unknown quant_format {self.quant_format!r}"
        if self.pool_sizes is None:
            self.pool_of_layer = [0] * self.num_layers
            self.pool_sizes = [self.cache_size]
        self.pool_of_layer = list(self.pool_of_layer)
        self.pool_sizes = list(self.pool_sizes)
        assert len(self.pool_of_layer) == self.num_layers, (len(self.pool_of_layer), self.num_layers)
        assert sorted(set(self.pool_of_layer)) == list(range(len(self.pool_sizes))), self.pool_of_layer
        assert sum(self.pool_sizes) == self.cache_size and min(self.pool_sizes) > 0, (self.pool_sizes, self.cache_size)
        # Attached by the engine for decode_target == "cpu" (CpuMoeExecutor); None
        # for the GPU decode path.
        self.cpu_executor = None
        # MoE layer ids whose decode runs on the CPU executor; the rest use the GPU
        # offload/PCIe path. Set by the engine after construction (empty = all-GPU,
        # all layers = the plain --moe-strategy cpu case).
        self.cpu_layer_ids: frozenset = frozenset()
        # Bank row layouts, set by set_bank_sources: layouts[g][k] is (dtype, row shape) of bank k (layouts of two quant tags may be equal), layer_layout[l] is layer l's g.
        # With one (dtype, row shape) per bank a bank cache is typed [cache_size, *row]; with several, or with pads, it is a byte segment each pool views with its own layout.
        self.layouts: list[tuple[tuple[torch.dtype, tuple[int, ...]], ...]] = []
        self.layer_layout: list[int] = [0] * self.num_layers
        # _pad_bytes[g][k]: zeroed bytes after the last row of every pool and staging buffer of layout g in bank k, for kernels that read past it (16-aligned)
        self._pad_bytes: list[list[int]] = []
        self._raw_banks = False
        # num_experts floor + nvfp4_marlin slot cap, shared with the runtime-rebuild path.
        self.validate_rebuild(self.cache_size, self.pool_sizes)
        assert not self.prefill_overlap or self.cache_size >= 2 * self.num_experts, (
            "Prefill overlap borrows two full expert-layer buffers from the unified MoE "
            "cache, so cache_size must be at least 2 * num_experts "
            "(raise moe_cache_size or disable moe_prefill_overlap)"
        )
        self.cache_policy_id = POLICY_IDS[self.cache_policy]
        # rule: an unrouted expert within this many logits of the top-k-th is a near miss
        self.near_miss_thr = (
            float(os.getenv("FREETOKEN_MOE_NEAR_MISS_THR", "0.25")) if self.cache_policy == "rule" else None
        )
        self.slot_for_id = torch.full(
            (self.num_layers, self.num_experts),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        # Reverse map, in the flat id space flashlib's slot_cache works in:
        # id == layer_id * num_experts + expert, so one array replaces the (layer,
        # expert) pair and evicting a slot needs no decode.
        self.id_of_slot = torch.full(
            (self.cache_size,),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        self.usage = torch.zeros((self.cache_size,), dtype=torch.int64, device=self.device)
        self.step = torch.zeros((), dtype=torch.int64, device=self.device)
        self.active_mask = torch.zeros((self.num_experts,), dtype=torch.int32, device=self.device)
        # lru_ensure validates these against plan = min(batch * top_k, cache_size), so num_experts elements would under-size them
        plan_slots = max(self.num_experts, self.cache_size)
        self.evict_slots = torch.empty((plan_slots,), dtype=torch.int32, device=self.device)
        self.src_indices = torch.empty((plan_slots,), dtype=torch.int32, device=self.device)
        self.num_indices = torch.zeros((1,), dtype=torch.int64, device=self.device)
        # hybrid only: full missing count BEFORE the per-step fetch cap (num_indices holds
        # the capped count that copy_missing actually fetches). The difference is what the
        # CPU computes this step. Written by the hybrid ensure kernel.
        self.num_missing_full = torch.zeros((1,), dtype=torch.int64, device=self.device)
        # hybrid only: per-(layer, expert) last-active decode step (LRU on the expert), -1
        # if never active. The hybrid ensure kernel reads it to pick which capped misses to
        # fetch (most-recently active first) and bumps it for every active expert.
        self.expert_recency = torch.full(
            (self.num_layers, self.num_experts), -1, dtype=torch.int64, device=self.device
        )
        self._init_evict_state()
        # Host source banks (one [num_experts, ...] tensor per layer, so layers can
        # carry independent host attributes -- see layer_residency) and their GPU
        # slot caches, keyed by the format's bank schema (attached by
        # set_bank_sources). The GPU slot cache stays one unified pool per bank.
        if self.layer_layouts is not None:
            self.bank_schema = tuple(dict.fromkeys(
                role for layout in self.layer_layouts for role, spec in layout.items() if not spec.resident
            ))
        elif self.layout is not None:
            self.bank_schema = tuple(role for role, spec in self.layout.items() if not spec.resident)
        else:
            self.bank_schema = _BANK_SCHEMAS[self.quant_format]
        self.bank_sources: dict[str, list[torch.Tensor]] = {}
        self.bank_caches: dict[str, torch.Tensor] = {}
        # per-layer host residency: the GPU movement paths require "pinned"; LOCKED/PAGEABLE layers decode on the CPU executor and prefill via copy_missing's pageable branch
        # _unpinned_layers is the derived id set the hot paths test against
        self.layer_residency: list[str] = []
        self._unpinned_layers: frozenset = frozenset()
        # marlin/b12x per-expert global scales ([L*E], GPU resident, see set_alphas).
        self.gate_up_alpha: torch.Tensor | None = None
        self.down_alpha: torch.Tensor | None = None
        # Opt-in decode miss-rate instrumentation. Accumulated on-device (no per-step host
        # sync); read via ``decode_miss_stats``. Graph-safe: the ``+=`` is captured into the
        # decode graph and re-executes with each replay's REAL routing (record_decode_stats
        # must be enabled before capture — see engine graph setup). The only graph artifact
        # is a one-off warm-up increment at capture time (<0.1% over a session).
        self.collect_stats = False
        # [num_layers, N_STATS] -- ensure_experts passes lru_stats[layer_id] straight to
        # the kernel, which accumulates in the same launch. The stat_* tensors below are
        # the hybrid path's: views of one buffer its kernel accumulates into the same way.
        self.lru_stats = torch.zeros(
            (self.num_layers, N_STATS), dtype=torch.int64, device=self.device
        )
        # (missing, fetched, active, calls), then the same per layer
        stat_buf = torch.zeros(4 * (self.num_layers + 1), dtype=torch.int64, device=self.device)
        self.hybrid_stats = (stat_buf[:4], stat_buf[4:])
        # hybrid only: stat_fetched counts the experts actually fetched over PCIe (<= stat_missing).
        # The CPU computes stat_missing - stat_fetched of them.
        self.stat_missing, self.stat_fetched, self.stat_active, self.stat_calls = stat_buf[:4].unbind()
        # Per-layer counterparts of the scalars above (indexed by MoE-layer id). Same
        # device-side accumulation (graph-safe: layer_id is a static index per graph node),
        # so one req's per-layer miss rate is readable via decode_miss_stats_per_layer().
        per_layer = stat_buf[4:].view(self.num_layers, 4)
        self.stat_missing_layer, self.stat_fetched_layer = per_layer[:, 0], per_layer[:, 1]
        self.stat_active_layer, self.stat_steps_layer = per_layer[:, 2], per_layer[:, 3]
        # Opt-in decode routing histogram (per layer, per expert) for cache-skew
        # analysis. Accumulated in ``ensure_experts`` from the raw expert ids before the
        # kernel rewrites them to slots. Only accurate with CUDA graphs disabled (the
        # captured graph would not re-run this host-side scatter on replay).
        self.collect_decode_freq = False
        self.decode_freq = torch.zeros(
            (self.num_layers, self.num_experts), dtype=torch.int64, device=self.device
        )
        # (per-layer sources, cache) per bank, in schema order. Every piece of cache
        # machinery that moves bank bytes (copy_missing, the prefill double buffers,
        # the pool and staging views) iterates this list, so the slot cache is bank-count agnostic.
        self.banks: list[tuple[list[torch.Tensor], torch.Tensor]] = []
        # Fused multi-bank copy descriptor (built by set_bank_sources/_build_copy_plan).
        # Source pointers are per layer (_copy_src_ptrs[layer_id] -> [num_banks] device
        # tensor); dst pointers are per pool and row bytes per layout.
        self._copy_fused_ok = False
        self._copy_blocks_per_bank = 8
        self._copy_dst_ptrs: torch.Tensor | None = None
        self._copy_dst_ptrs_pool: list[torch.Tensor] | None = None
        self._copy_src_ptrs: list[torch.Tensor] | None = None
        self._copy_feat_bytes: list[torch.Tensor] | None = None
        # The layer whose misses ensure_experts/materialize_layer staged last; consumed
        # by copy_missing to pick the per-layer source (part of the same pending-copy
        # state as evict_slots/src_indices/num_indices).
        # _pending_whole_layer records WHICH staged it: the pageable branch is only sound after materialize_layer
        self._pending_src_layer: int | None = None
        self._pending_whole_layer = False
        # The two prefill staging buffers are the first 2 * _stage_bytes[k] bytes of every bank, on top of the first pools.
        # _stage_views[g][b] are buffer b's per-bank [num_experts, ...] views for layout g; _inval[b] are the global slots it overwrites.
        self._stage_bytes: list[int] = []
        self._stage_views: list[list[tuple[torch.Tensor, ...] | None]] = []
        # _stage_tails[g][b]: the pad after buffer b's rows in layout g
        self._stage_tails: list[list[list[torch.Tensor]]] = []
        self._inval: list[slice | torch.Tensor] = []
        # the slot audit's label for the staged copy, and the most entries its plan can have
        self._pending_kind = 0
        self._pending_width: int | None = None
        self.prefill_copy_stream: torch.cuda.Stream | None = None
        self.prefill_begin_event: torch.cuda.Event | None = None
        self.prefill_ready_events: list[torch.cuda.Event] = []
        self.prefill_release_events: list[torch.cuda.Event] = []
        self._prefill_buffer_layer: list[int | None] = [None, None]
        self._prefill_buffer_released: list[bool] = [True, True]
        self._prefill_buffer_has_release_event: list[bool] = [False, False]
        # hit-D2D split state: pinned begin-of-chunk snapshot of slot_for_id (the
        # classification input; frozen for the chunk -- no decode runs inside one,
        # and buffer invalidation only clears slots under the staging buffers, which classify as
        # miss regardless), the lazily resolved batch-memcpy entry point (False =
        # unavailable), and row counters for cache reports.
        self._prefill_slot_snapshot: torch.Tensor | None = None
        self._prefill_snapshot_np = None
        self._prefill_hit_d2d_active = False
        self._hit_d2d_fallback_logged = False
        self._batch_memcpy = None
        self.prefill_hit_rows = 0
        self.prefill_total_rows = 0
        self._slice_pools()
        # Decode copy overlap: one side stream plus a (forked, done) event pair per layer.
        # Neither is cache_size-shaped, so rebuild() keeps them.
        if self.decode_copy_overlap is None:
            self.decode_copy_overlap = bool(ENV.MOE_COPY_OVERLAP)
        self.decode_copy_stream: torch.cuda.Stream | None = None
        self._decode_copy_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        if self.decode_copy_overlap and self.device.type == "cuda" and self.decode_target == "gpu":
            self._init_decode_copy_overlap()
        self.prefetch = self._init_prefetch()
        self.verify = self._init_verify()
        self.audit = self._init_audit()

    def _init_audit(self):
        from freetoken.kernel.fast_index_copy import _skip_fast_index_copy_enabled
        from freetoken.moe.slot_audit import SlotAuditor, resolve_interval

        from_env = self.slot_audit is None
        self.slot_audit = resolve_interval(self.slot_audit)
        if not self.slot_audit or self.device.type != "cuda":
            return None
        if self.decode_target != "gpu":
            if not from_env:
                raise ValueError(f"FREETOKEN_MOE_SLOT_AUDIT={self.slot_audit} needs GPU decode, not {self.decode_target!r}")
            logger.warning(f"FREETOKEN_MOE_SLOT_AUDIT={self.slot_audit} ignored: decode target is {self.decode_target!r}")
            self.slot_audit = 0
            return None
        if _skip_fast_index_copy_enabled():
            logger.warning("FREETOKEN_MOE_SLOT_AUDIT ignored: FREETOKEN_SKIP_FAST_INDEX_COPY makes every copy a no-op")
            self.slot_audit = 0
            return None
        return SlotAuditor(self.num_layers, self.num_experts, self.device, interval=self.slot_audit)

    def _init_verify(self):
        from freetoken.moe.verify import ExpertVerifier, resolve_mode

        from_env = self.verify_mode is None
        self.verify_mode = resolve_mode(self.verify_mode)
        if self.verify_mode == "off" or self.device.type != "cuda":
            return None
        if self.decode_target != "gpu":
            if not from_env:
                raise ValueError(f"FREETOKEN_MOE_PREFETCH_VERIFY={self.verify_mode} needs GPU decode, not {self.decode_target!r}")
            logger.warning(f"FREETOKEN_MOE_PREFETCH_VERIFY={self.verify_mode} ignored: decode target is {self.decode_target!r}")
            self.verify_mode = "off"
            return None
        return ExpertVerifier(self.num_layers, self.num_experts, self.device, mode=self.verify_mode)

    def _init_prefetch(self):
        from freetoken.moe.prefetch import ExpertPrefetcher, resolve_mode

        from_env = self.prefetch_mode is None
        self.prefetch_mode = resolve_mode(self.prefetch_mode)
        if self.prefetch_mode == "off":
            return None
        if self.decode_target != "gpu":
            if not from_env:
                raise ValueError(f"FREETOKEN_MOE_PREFETCH={self.prefetch_mode} needs GPU decode, not {self.decode_target!r}")
            # the process-wide flag skips hybrid/CPU-decoded caches, like FREETOKEN_MOE_COPY_OVERLAP
            logger.warning(f"FREETOKEN_MOE_PREFETCH={self.prefetch_mode} ignored: decode target is {self.decode_target!r}")
            self.prefetch_mode = "off"
            return None
        if self.device.type != "cuda":
            return None
        prefetch = ExpertPrefetcher(self.num_layers, self.num_experts, self.device, mode=self.prefetch_mode)
        # a proxy, not a cycle: dropping the cache must free its slot banks without waiting for gc
        prefetch.cache = weakref.proxy(self)
        return prefetch

    @property
    def prefetch_on(self) -> bool:
        """FREETOKEN_MOE_PREFETCH=on: every ensure keys prefetched slots low priority."""
        return self.prefetch is not None and self.prefetch.mode == "on"

    def _pool_layout(self, pool: int) -> int:
        return self.layer_layout[self.pool_of_layer.index(pool)]

    def _bank_rows(self, k: int, layout: int, offset: int, nrows: int) -> torch.Tensor:
        """``nrows`` rows of bank ``k`` in ``layout``, starting ``offset`` bytes into the bank cache."""
        cache = self.banks[k][1]
        row = self._row_bytes[layout][k]
        if not self._raw_banks:
            first = offset // row
            return cache if first == 0 and nrows == cache.shape[0] else cache[first : first + nrows]
        dtype, shape = self.layouts[layout][k]
        return cache[offset : offset + nrows * row].view(dtype).view(nrows, *shape)

    def _bank_bytes(self, k: int, pool_sizes: list[int]) -> int:
        """Bytes of raw bank ``k``: each pool's slots and pad, plus the largest pad, so two staging buffers fit wherever two layers of slots do."""
        layouts = [self._pool_layout(p) for p in range(len(pool_sizes))]
        slots = sum(s * self._row_bytes[g][k] + self._pad_bytes[g][k] for g, s in zip(layouts, pool_sizes))
        return slots + max(pads[k] for pads in self._pad_bytes)

    def _stage_fits(self, buffers: int) -> bool:
        return all(c.numel() * c.element_size() >= buffers * s for (_, c), s in zip(self.banks, self._stage_bytes))

    def _slice_pools(self) -> None:
        E = self.num_experts
        bases = [0]
        for size in self.pool_sizes[:-1]:
            bases.append(bases[-1] + size)
        self.pool_bases = bases
        spans = list(zip(bases, self.pool_sizes))

        def cut(t: torch.Tensor, b: int, s: int) -> torch.Tensor:
            return t if s == t.shape[0] else t[b : b + s]

        self._pool_id_of_slot = [cut(self.id_of_slot, b, s) for b, s in spans]
        self._pool_usage = [cut(self.usage, b, s) for b, s in spans]
        # rows[k][p]: bytes of one slot of pool p in bank k; before the banks exist, one unit per slot with a stage of E
        if self.banks:
            rows = [[self._row_bytes[self._pool_layout(p)][k] for p in range(len(spans))] for k in range(len(self.banks))]
            pads = [[self._pad_bytes[self._pool_layout(p)][k] for p in range(len(spans))] for k in range(len(self.banks))]
            stage = self._stage_bytes
        else:
            rows, pads, stage = [[1] * len(spans)], [[0] * len(spans)], [E]
        offsets = []
        for row, pad in zip(rows, pads):
            offsets.append([0])
            for s, r, d in zip(self.pool_sizes[:-1], row, pad):
                offsets[-1].append(offsets[-1][-1] + s * r + d)
        self._pool_offsets = offsets
        # local slots below this index overlap the prefill staging area in some bank; a pool's zero-byte rows (a role its kernel does not read) overlap nothing
        self._pool_stable_from = [
            max((min(max(-(-(2 * stage[k] - offsets[k][p]) // rows[k][p]), 0), s) for k in range(len(rows)) if rows[k][p]), default=0)
            for p, s in enumerate(self.pool_sizes)
        ]
        self._inval = []
        for b in range(2):
            mask = torch.zeros(self.cache_size, dtype=torch.bool)
            for k in range(len(rows)):
                lo, hi = b * stage[k], (b + 1) * stage[k]
                for p, (base, s) in enumerate(spans):
                    if not rows[k][p]:
                        continue
                    first = max((lo - offsets[k][p]) // rows[k][p], 0)
                    end = min(-(-(hi - offsets[k][p]) // rows[k][p]), s)
                    mask[base + first : base + max(first, end)] = True
            idx = mask.nonzero().flatten()
            if idx.numel() == 0 or int(idx[-1] - idx[0]) + 1 == idx.numel():
                self._inval.append(slice(int(idx[0]), int(idx[-1]) + 1) if idx.numel() else slice(0, 0))
            else:
                self._inval.append(idx.to(self.device))
        self._stage_dirty = [False, False]
        if not self.banks:
            self._pool_bank_views, self._stage_views, self._stage_tails = [], [], []
            return
        self._pool_bank_views = [
            tuple(self._bank_rows(k, self._pool_layout(p), offsets[k][p], s) for k in range(len(self.banks)))
            for p, s in enumerate(self.pool_sizes)
        ]
        self._stage_views = [
            [
                tuple(self._bank_rows(k, g, b * stage[k], E) for k in range(len(self.banks))) if self._stage_fits(b + 1) else None
                for b in range(2)
            ]
            for g in range(len(self.layouts))
        ]
        self._stage_tails = [
            [[self.banks[k][1][b * stage[k] + E * row[k] :][: pad[k]] for k in range(len(self.banks)) if pad[k] and self._stage_fits(b + 1)] for b in range(2)]
            for row, pad in zip(self._row_bytes, self._pad_bytes)
        ]

    def stage_views(self, layer_id: int, buffer_id: int) -> tuple[torch.Tensor, ...]:
        """Per-bank ``[num_experts, ...]`` views of prefill staging buffer ``buffer_id`` in ``layer_id``'s layout; row == expert id."""
        views = self._stage_views[self.layer_layout[layer_id]][buffer_id]
        assert views is not None, f"the slot cache is too small for staging buffer {buffer_id}"
        return views

    def pool_state(self, layer_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        """``(id_of_slot, usage)`` views of the pool that serves ``layer_id``."""
        p = self.pool_of_layer[layer_id]
        return self._pool_id_of_slot[p], self._pool_usage[p]

    def pool_slots(self, layer_id: int) -> int:
        return self.pool_sizes[self.pool_of_layer[layer_id]]

    def pool_bank_views(self, layer_id: int) -> tuple[torch.Tensor, ...]:
        """Per-bank slot views of the pool that serves ``layer_id``, indexed by its local slot ids."""
        assert self.banks, "set_bank_sources must register the banks first"
        return self._pool_bank_views[self.pool_of_layer[layer_id]]
    def _init_decode_copy_overlap(self) -> None:
        # a dedicated stream, not one of torch's 32 pooled ones: a pooled stream can alias the engine
        # or capture stream, and the side work would then run serially (correct, but no overlap)
        from cuda.bindings import runtime as cudart

        with torch.cuda.device(self.device):
            err, handle = cudart.cudaStreamCreateWithFlags(cudart.cudaStreamNonBlocking)
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"cudaStreamCreateWithFlags failed: {err}")
        self.decode_copy_stream = torch.cuda.ExternalStream(int(handle), device=self.device)
        self._decode_copy_events = [
            (torch.cuda.Event(), torch.cuda.Event()) for _ in range(self.num_layers)
        ]
        # torch creates the CUDA event on its first record: do that now, never inside a capture
        for pair in self._decode_copy_events:
            for event in pair:
                event.record(self.decode_copy_stream)
        logger.info("MoE decode copy overlap on: expert misses copy on a side stream")

    def _init_evict_state(self) -> None:
        """Scored policies' per-(layer, expert) state (moe/scored_ensure.py); None under LRU.

        It is not slot-shaped: it survives eviction and prefill invalidation, and a rebuild
        only resets it. ``evict_tok`` is the decode-step clock the first ensure of each step bumps.
        """
        from freetoken.moe import scored_ensure as se

        self.evict_tok = self.evict_last_tok = self.evict_lc = self.evict_ct = None
        self.evict_g_table = None
        n, dev = self.num_layers * self.num_experts, self.device
        if self.cache_policy_id >= 1:
            self.evict_tok = torch.zeros((), dtype=torch.int64, device=dev)
            self.evict_last_tok = torch.full((n,), se.LAST_TOK_NEVER, dtype=torch.int64, device=dev)
        if self.cache_policy_id >= 2:
            self.evict_lc = torch.full((n,), se.LC_NEVER, dtype=torch.int32, device=dev)
            self.evict_ct = torch.zeros((n,), dtype=torch.int64, device=dev)
            self.evict_g_table = se.softplus2_table(dev)
        self._alloc_evict_mirrors()

    def _alloc_evict_mirrors(self) -> None:
        """Slot-shaped copies of each slot owner's state, read coalesced by the victim scan.

        A mirror is valid while ``evict_slot_owner`` equals ``id_of_slot``; the kernel keeps it
        so, and anything else that installs an id into a slot must set the owner to -1.
        """
        self.evict_slot_owner = self.evict_slot_last_tok = self.evict_slot_lc = self.evict_slot_ct = None
        if self.evict_tok is None:
            return
        size, dev = self.cache_size, self.device
        self.evict_slot_owner = torch.full((size,), -1, dtype=torch.int32, device=dev)
        self.evict_slot_last_tok = torch.zeros((size,), dtype=torch.int64, device=dev)
        if self.evict_lc is not None:
            self.evict_slot_lc = torch.zeros((size,), dtype=torch.int32, device=dev)
            self.evict_slot_ct = torch.zeros((size,), dtype=torch.int64, device=dev)

    def reset_evict_state(self) -> None:
        if self.evict_tok is None:
            return
        from freetoken.moe import scored_ensure as se

        self.evict_tok.zero_()
        self.evict_last_tok.fill_(se.LAST_TOK_NEVER)
        self.evict_slot_owner.fill_(-1)
        if self.evict_lc is not None:
            self.evict_lc.fill_(se.LC_NEVER)
            self.evict_ct.zero_()

    def evict_clock_layer(self) -> int:
        """The first layer that ensures on the GPU each decode step; its call bumps ``evict_tok``."""
        if not self.cpu_layer_ids:
            return 0
        return min(set(range(self.num_layers)) - self.cpu_layer_ids, default=0)

    def set_bank_sources(
        self,
        sources: dict[str, list[torch.Tensor]],
        layer_residency: list[str] | None = None,
        layer_tags: list | None = None,
        row_tail_bytes: dict[str, list[int]] | None = None,
    ) -> None:
        """Attach the host (CPU pinned) expert source banks and allocate a GPU slot
        cache per bank, following the format's bank schema.

        Every bank is a list of ``num_layers`` tensors, one ``[num_experts, ...]``
        per layer (independent allocations, so each layer can carry its own host
        attributes); each slot cache mirrors the bank's row shape and dtype as one
        unified GPU pool. The row layouts are produced by the weight loaders /
        repackers (see ``_BANK_SCHEMAS`` and :mod:`freetoken.layers.quantization.moe.nvfp4`)
        -- the cache machinery is layout-agnostic and just moves rows.

        ``layer_residency`` labels each layer with a ``HostResidency`` value (default: all pinned).
        Non-pinned (LOCKED/PAGEABLE) layers have no device address: they must already be routed to the CPU executor (``cpu_layer_ids``, set BEFORE this call), the copy plan skips their rows, and their only movement is ``copy_missing``'s whole-layer pageable prefill branch -- which is why prefill overlap is incompatible with them.

        ``layer_tags`` names each layer's quant (see :func:`bank_layouts`); layers of different tags never share a pool.

        ``row_tail_bytes`` gives, per bank and layer, how many bytes past the last row the kernel reads; every pool and staging buffer is then followed by that many zero bytes.
        """
        from freetoken.moe.legacy_format import canonical_role
        from freetoken.moe.host_banks import HostResidency

        # loaders and FTW files may still name the banks the old way (gate_up_packed, ...)
        by_role = {canonical_role(name): per_layer for name, per_layer in sources.items()}
        if set(by_role) != {canonical_role(n) for n in self.bank_schema}:
            raise AssertionError(
                f"banks {sorted(sources)} do not match the {self.quant_format!r} schema {self.bank_schema}"
            )
        sources = {name: by_role[canonical_role(name)] for name in self.bank_schema}
        residency = layer_residency or [HostResidency.PINNED.value] * self.num_layers
        assert len(residency) == self.num_layers, (len(residency), self.num_layers)
        unpinned = frozenset(
            i for i, r in enumerate(residency) if r != HostResidency.PINNED.value
        )
        if unpinned:
            if not unpinned <= self.cpu_layer_ids:
                raise ValueError(
                    f"non-pinned layers {sorted(unpinned - self.cpu_layer_ids)} are not in "
                    f"cpu_layer_ids: a layer without a device address can only decode on "
                    f"the CPU executor (set cache.cpu_layer_ids before set_bank_sources)"
                )
            if self.prefill_overlap:
                raise ValueError(
                    "prefill overlap DMAs from registered banks; it must be disabled "
                    "when any layer is LOCKED/PAGEABLE (the engine does this)"
                )
        self._unpinned_layers = unpinned
        self.layer_residency = list(residency)
        for name in self.bank_schema:
            per_layer = sources[name]
            assert len(per_layer) == self.num_layers, (name, len(per_layer))
            for layer_id, source in enumerate(per_layer):
                assert source.is_contiguous(), f"bank {name!r} layer {layer_id} must be contiguous"
                assert source.size(0) == self.num_experts, (name, layer_id, source.shape)
                spec = self.layout[name] if self.layout is not None else None
                if self.layer_layouts is not None:
                    spec = self.layer_layouts[layer_id].get(name)
                    if spec is None and source.numel() != 0:
                        raise ValueError(f"layer {layer_id}'s expert kernel reads no {name!r} bank, so it takes a zero-byte placeholder")
                if spec is not None:
                    if tuple(source.shape[1:]) != tuple(spec.shape) or source.dtype != spec.dtype:
                        raise ValueError(
                            f"bank {name!r} rows are {tuple(source.shape[1:])} {source.dtype} but the expert kernel's layout "
                            f"wants {tuple(spec.shape)} {spec.dtype}; the banks were packed for another kernel"
                        )
            self.bank_sources[name] = list(per_layer)
        keys = bank_layouts(sources, layer_tags)
        unique = list(dict.fromkeys(keys))
        self.layouts = [banks for _, banks in unique]
        self.layer_layout = [unique.index(key) for key in keys]
        for p in range(len(self.pool_sizes)):
            mixed = {self.layer_layout[l] for l, q in enumerate(self.pool_of_layer) if q == p}
            assert len(mixed) == 1, f"slot pool {p} holds layers of {len(mixed)} layouts; a pool needs one"
        self._pad_bytes = [[0] * len(self.bank_schema) for _ in self.layouts]
        if row_tail_bytes is not None:
            tails = {canonical_role(name): per_layer for name, per_layer in row_tail_bytes.items()}
            for k, name in enumerate(self.bank_schema):
                for layer_id, tail in enumerate(tails[canonical_role(name)]):
                    g = self.layer_layout[layer_id]
                    self._pad_bytes[g][k] = max(self._pad_bytes[g][k], -(-tail // 16) * 16)
        self._raw_banks = len(set(self.layouts)) > 1 or any(map(any, self._pad_bytes))
        self._row_bytes = [[math.prod(shape) * dtype.itemsize for dtype, shape in layout] for layout in self.layouts]
        if self._raw_banks:
            # every pool view and staging buffer starts on a row boundary of its own layout
            assert all(r % 16 == 0 for rows in self._row_bytes for r in rows), self._row_bytes
        self._stage_bytes = [
            self.num_experts * max(rows[k] for rows in self._row_bytes) + max(pads[k] for pads in self._pad_bytes)
            for k in range(len(self.bank_schema))
        ]
        self._alloc_banks()
        if not self._stage_fits(1):
            raise ValueError(f"MoE cache of {self.cache_size} slots cannot stage one whole layer of experts")
        if self.prefill_overlap and not self._stage_fits(2):
            logger.warning("Disabling MoE prefill overlap: the slot cache cannot hold two whole layers of experts")
            self.prefill_overlap = False
        self._slice_pools()
        self._build_copy_plan()
        if self.verify is not None:
            self.verify.bind(self)
        if self.audit is not None:
            self.audit.bind(self)
        if self.prefill_overlap:
            self._init_prefill_overlap_buffers()

    def _alloc_banks(self) -> None:
        for k, name in enumerate(self.bank_schema):
            if self._raw_banks:
                self.bank_caches[name] = torch.zeros(self._bank_bytes(k, self.pool_sizes), dtype=torch.uint8, device=self.device)
            else:
                dtype, shape = self.layouts[0][k]
                self.bank_caches[name] = torch.zeros((self.cache_size, *shape), dtype=dtype, device=self.device)
        self.banks = [(self.bank_sources[n], self.bank_caches[n]) for n in self.bank_schema]

    def _build_copy_plan(self) -> None:
        self._build_fused_copy_plan()
        if self._copy_fused_ok or self.device.type != "cuda" or not self.banks:
            return
        for k, name in enumerate(self.bank_schema):
            for rows in self._row_bytes:
                if rows[k] % 128:
                    raise RuntimeError(
                        f"MoE bank {name!r} rows are {rows[k]} bytes (not a multiple of 128): "
                        f"only the fused multi-bank copy can move them, but it is disabled"
                    )

    def _build_fused_copy_plan(self) -> None:
        """Precompute the fused multi-bank copy descriptor (base addrs + per-row bytes).

        Built once here (and on :meth:`rebuild`, which reallocates the slot caches);
        the addresses are fixed for the cache's lifetime so the descriptor tensors are
        CUDA-graph safe. Disabled (-> per-bank fallback) if any bank's row bytes or base
        address is not 16-byte aligned, or via FREETOKEN_FUSED_COPY=0.
        """
        self._copy_fused_ok = False
        self._copy_dst_ptrs = None
        self._copy_dst_ptrs_pool = None
        self._copy_src_ptrs = None
        self._copy_feat_bytes = None
        self._copy_dst_ptrs_host: list[int] = []
        self._copy_src_ptrs_host: list[list[int]] = []
        self._gather_bank_ids: list[list[int]] = []
        self._gather_feat_bytes: list[torch.Tensor | None] = []
        self._gather_stage_ptrs: list[list[torch.Tensor] | None] = []
        self._gather_pool_ptrs: list[torch.Tensor | None] = []
        if not _FUSED_COPY or self.device.type != "cuda" or not self.banks:
            return
        from freetoken.kernel.pinned import device_ptr

        aligned = [self._stage_bytes] + self._row_bytes + self._pool_offsets
        if any(n % 16 for ns in aligned for n in ns) or any(c.data_ptr() % 16 for _, c in self.banks):
            return  # leave fused disabled; copy_missing uses the per-bank path
        dst_ptrs = [cache.data_ptr() for _, cache in self.banks]
        layer_src_ptrs = [[] for _ in range(self.num_layers)]
        for per_layer, _ in self.banks:
            for layer_id, source in enumerate(per_layer):
                if layer_id in self._unpinned_layers:
                    # unregistered layer: no device alias exists, and the row is never consumed (CPU decode; pageable prefill)
                    # a 0 placeholder keeps the descriptor shape
                    layer_src_ptrs[layer_id].append(0)
                    continue
                if source.numel() == 0:
                    layer_src_ptrs[layer_id].append(0)  # placeholder bank: zero bytes per row, never dereferenced
                    continue
                # The kernel dereferences these on the GPU, so store each host bank's
                # device alias (== data_ptr() under UVA identity; differs on
                # Windows/WDDM).
                src_dev = device_ptr(source)
                if src_dev % 16 != 0:
                    return
                layer_src_ptrs[layer_id].append(src_dev)

        def ptrs(values: list[int]) -> torch.Tensor:
            return torch.tensor(values, dtype=torch.int64, device=self.device)

        self._copy_dst_ptrs = ptrs(dst_ptrs)
        self._copy_dst_ptrs_pool = [
            self._copy_dst_ptrs if all(o[p] == 0 for o in self._pool_offsets)
            else ptrs([d + o[p] for d, o in zip(dst_ptrs, self._pool_offsets)])
            for p in range(len(self.pool_sizes))
        ]
        self._copy_src_ptrs = [ptrs(values) for values in layer_src_ptrs]
        self._copy_feat_bytes = [ptrs(rows) for rows in self._row_bytes]
        self._copy_dst_ptrs_host = dst_ptrs
        self._copy_src_ptrs_host = layer_src_ptrs
        # hit-D2D gather serves only the big banks; small banks are whole-layer
        # H2D entries (see _SMALL_BANK_FEAT_BYTES), so their rows never need D2D.
        # The gather copies cache rows of the layer's pool into its staging buffer, row == expert id.
        self._gather_bank_ids = [[k for k, r in enumerate(rows) if r >= _SMALL_BANK_FEAT_BYTES] for rows in self._row_bytes]
        for g, ids in enumerate(self._gather_bank_ids):
            self._gather_feat_bytes.append(ptrs([self._row_bytes[g][k] for k in ids]) if ids else None)
            self._gather_stage_ptrs.append(
                [ptrs([dst_ptrs[k] + b * self._stage_bytes[k] for k in ids]) for b in range(2)] if ids else None
            )
        for p in range(len(self.pool_sizes)):
            ids = self._gather_bank_ids[self._pool_layout(p)]
            self._gather_pool_ptrs.append(ptrs([dst_ptrs[k] + self._pool_offsets[k][p] for k in ids]) if ids else None)
        self._copy_blocks_per_bank = default_blocks_per_bank(self.device)
        self._copy_fused_ok = True

    def validate_rebuild(self, cache_size: int, pool_sizes: list[int] | None = None) -> None:
        """Pure geometry validation of a rebuild target (no GPU side effects).

        Raises ``ValueError`` if ``cache_size`` is below the ``num_experts`` floor or
        a pool is above the expert kernel's slot cap, or the bank byte segments cannot stage one layer. Called by :meth:`rebuild` and by the engine's
        pre-teardown check, so an invalid target rejects with the old cache intact
        (no destructive free first).
        """
        if cache_size < self.num_experts:
            raise ValueError(f"cache_size {cache_size} < num_experts {self.num_experts}")
        # decode kernels see one pool's slots, so the cap applies per pool
        largest = cache_size if pool_sizes is None else max(pool_sizes)
        what = f"moe_cache_size={cache_size}" if largest == cache_size else f"a MoE slot pool of {largest} slots"
        if self.max_slots is not None and largest > self.max_slots:
            raise ValueError(
                f"{what} exceeds the expert kernel's slot limit of {self.max_slots}; "
                f"pass --moe-cache-size {self.max_slots} or less, or let the default kernel serve the experts"
            )
        if self.layout is None and self.quant_format == "nvfp4_marlin" and largest > MARLIN_MAX_CACHE_SIZE:
            raise ValueError(
                f"{what} exceeds the marlin backend's slot limit of "
                f"{MARLIN_MAX_CACHE_SIZE} (vLLM moe_align_block_size caps padded experts at "
                "1024); reduce moe_cache_size or force --quant-backend moe.nvfp4=triton"
            )
        if self._raw_banks and pool_sizes is not None:
            for k, stage in enumerate(self._stage_bytes):
                if self._bank_bytes(k, pool_sizes) < stage:
                    raise ValueError(f"moe_cache_size={cache_size} cannot stage one whole layer of experts")

    def rebuild(self, cache_size: int, pool_sizes: list[int] | None = None) -> None:
        """Resize the GPU slot cache + bookkeeping to ``cache_size`` IN PLACE.

        Keeps the CPU/pinned ``bank_sources`` and the GPU-resident alphas; never
        reloads banks. Tears down prefill-overlap buffers first (their views alias
        the old ``bank_caches``), frees the old GPU tensors, then reallocates. Slots
        cold-start after rebuild. Object identity is preserved so attached layers and
        ``ctx.moe_offload_cache`` stay valid.
        """
        assert self.bank_sources, "set_bank_sources must run before rebuild"
        sizes = [cache_size] if pool_sizes is None else list(pool_sizes)
        assert len(sizes) == len(self.pool_sizes) and sum(sizes) == cache_size and min(sizes) > 0, (sizes, cache_size)
        self.validate_rebuild(cache_size, sizes)
        # 1. Tear down prefill-overlap (its buffer views alias the old bank_caches).
        self.prefill_copy_stream = None
        self.prefill_begin_event = None
        self.prefill_ready_events = []
        self.prefill_release_events = []
        self._prefill_buffer_layer = [None, None]
        self._prefill_buffer_released = [True, True]
        self._prefill_buffer_has_release_event = [False, False]
        # 2. Drop old GPU tensors (free-before-alloc).
        self.banks = []
        self.bank_caches = {}
        self._pool_bank_views = []
        self._stage_views = []
        self._stage_tails = []
        self.cache_size = cache_size
        self.pool_sizes = sizes
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()
        # 3. Reallocate the slot cache from the retained host sources.
        self._alloc_banks()
        # 4. Reallocate cache_size-shaped bookkeeping; reset the slot map (cold start).
        self.slot_for_id.fill_(-1)
        self.id_of_slot = torch.full((cache_size,), -1, dtype=torch.int32, device=self.device)
        self.usage = torch.zeros((cache_size,), dtype=torch.int64, device=self.device)
        self._slice_pools()
        self._build_copy_plan()  # slot caches were reallocated -> refresh fused-copy addrs
        if self.verify is not None:
            self.verify.bind(self)
        plan_slots = max(self.num_experts, cache_size)
        self.evict_slots = torch.empty((plan_slots,), dtype=torch.int32, device=self.device)
        self.src_indices = torch.empty((plan_slots,), dtype=torch.int32, device=self.device)
        self.step.zero_()
        self.active_mask.zero_()
        self.num_indices.zero_()
        self.num_missing_full.zero_()
        self.expert_recency.fill_(-1)
        self._alloc_evict_mirrors()
        self.reset_evict_state()
        self.stat_missing.zero_()
        self.stat_active.zero_()
        self.stat_calls.zero_()
        self.stat_fetched.zero_()
        self.stat_missing_layer.zero_()
        # a rebuild is a cold start for the cache; carrying pre-rebuild hit/miss counts over would skew every post-rebuild stats report
        self.lru_stats.zero_()
        self.stat_active_layer.zero_()
        self.stat_fetched_layer.zero_()
        self.stat_steps_layer.zero_()
        self.decode_freq.zero_()
        if self.prefetch is not None:
            self.prefetch.reset()
        if self.audit is not None:
            self.audit.bind(self)  # new slot arrays: a new history
        self.prefill_hit_rows = 0
        self.prefill_total_rows = 0
        self._hit_d2d_fallback_logged = False  # geometry changed; re-log if still unusable
        # 5. Re-evaluate prefill overlap against the new size.
        if self.prefill_overlap and not self._stage_fits(2):
            logger.warning(
                f"Disabling MoE prefill overlap on rebuild: cache_size {cache_size} "
                f"cannot hold two whole layers of experts (2*num_experts {2 * self.num_experts} slots of one layout)."
            )
            self.prefill_overlap = False
        if self.prefill_overlap:
            self._init_prefill_overlap_buffers()

    def set_alphas(
        self, gate_up_alpha: torch.Tensor | None, down_alpha: torch.Tensor | None
    ) -> None:
        """Attach the marlin/b12x per-expert global scales (``[L*E]``, GPU resident).

        These are kernel-preprocessed scalars, far too small to bother offloading;
        the forward path looks them up per slot with :meth:`alphas_for_slots` /
        :meth:`alphas_for_layer` (pure device-side lookups, CUDA-graph safe).
        ``(None, None)`` is a no-op so callers can pass a format's (possibly
        absent) alphas through unconditionally.
        """
        if gate_up_alpha is None and down_alpha is None:
            return
        assert gate_up_alpha is not None and down_alpha is not None
        total = self.num_layers * self.num_experts
        assert gate_up_alpha.shape == down_alpha.shape == (total,)
        self.gate_up_alpha = gate_up_alpha.to(self.device)
        self.down_alpha = down_alpha.to(self.device)

    def set_cpu_executor(self, executor) -> None:
        """Attach the CPU MoE executor (``decode_target`` in {"cpu", "hybrid"}).

        The executor owns the persistent worker pool, the pinned activation/result
        IO buffers, and the ``cudaLaunchHostFunc`` submit/sync plumbing. It reads
        experts straight from this cache's host ``bank_sources`` (no extra copy).
        """
        assert self.decode_target in ("cpu", "hybrid"), (
            "set_cpu_executor requires decode_target in {'cpu','hybrid'}"
        )
        self.cpu_executor = executor

    def is_cpu_layer(self, layer_id: int) -> bool:
        """Whether ``layer_id`` decodes on the CPU executor (vs the GPU offload path)."""
        return layer_id in self.cpu_layer_ids

    def is_unpinned_layer(self, layer_id: int) -> bool:
        """Whether ``layer_id``'s host banks have no device address (LOCKED/PAGEABLE): the GPU slot-gather paths cannot serve it.
        ``copy_missing`` takes the whole-layer pageable branch, which presumes materialize's position == expert id (never ``ensure_experts``'s LRU slot remap)."""
        return layer_id in self._unpinned_layers

    def alphas_for_slots(self, layer_id: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Per-slot global scales for a decode call, or ``None`` when the format
        keeps no GPU-resident alphas (bf16 / triton-nvfp4). Slots of other layers
        yield garbage values, but only slots routed to -- and those belong to
        ``layer_id`` -- are ever read by the grouped GEMM."""
        if self.gate_up_alpha is None:
            return None
        id_of_slot, _ = self.pool_state(layer_id)
        idx = layer_id * self.num_experts + (
            id_of_slot.clamp(min=0).long() % self.num_experts
        )
        return self.gate_up_alpha[idx], self.down_alpha[idx]

    def alphas_for_layer(self, layer_id: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Global scales for a full-layer prefill (overlap or materialize), where
        position == expert id (contiguous slices, no gather); ``None`` when the
        format keeps no GPU-resident alphas."""
        if self.gate_up_alpha is None:
            return None
        lo = layer_id * self.num_experts
        hi = lo + self.num_experts
        return self.gate_up_alpha[lo:hi], self.down_alpha[lo:hi]

    def _init_prefill_overlap_buffers(self) -> None:
        assert self.banks, "set_bank_sources must register the banks first"
        self._prefill_buffer_layer = [None, None]
        self._prefill_buffer_released = [True, True]
        self._prefill_buffer_has_release_event = [False, False]
        # the double buffers are the staging views (_slice_pools), on top of the first pools' slots
        if self.device.type == "cuda":
            self.prefill_copy_stream = torch.cuda.Stream(device=self.device)
            self.prefill_ready_events = [torch.cuda.Event() for _ in range(2)]
            self.prefill_release_events = [torch.cuda.Event() for _ in range(2)]
            self.prefill_begin_event = torch.cuda.Event()
        if self.prefill_hit_d2d and self.device.type == "cuda":
            self._prefill_slot_snapshot = torch.empty(
                (self.num_layers, self.num_experts), dtype=torch.int32, pin_memory=True
            )
            self._prefill_snapshot_np = self._prefill_slot_snapshot.numpy()
            self._prefill_hit_dst = torch.empty(
                (self.num_experts,), dtype=torch.int32, device=self.device
            )
            self._prefill_hit_src = torch.empty(
                (self.num_experts,), dtype=torch.int32, device=self.device
            )
            self._prefill_hit_num = torch.zeros((1,), dtype=torch.int64, device=self.device)

    def _invalidate_prefill_buffer(self, buffer_id: int, layer_id: int) -> None:
        """Take buffer ``buffer_id``'s slots for staging ``layer_id``."""
        slots = self._inval[buffer_id]
        if isinstance(slots, slice):
            if self.audit is not None:
                from freetoken.moe.slot_audit import INVALIDATE

                self.audit.record_clear(self, INVALIDATE, layer_id, slots.start, slots.stop - slots.start)
            # One fixed-shape launch. The boolean-mask index
            # (slot_for_id[old_ids[old_ids >= 0]] = -1) has a data-dependent shape, so every
            # call hid a device-to-host sync -- and with two buffer reuses per chunk x 48
            # layers the host waited, per layer, for ALL enqueued GPU work (the cached-context
            # attention included), serializing the prefill-overlap pipeline.
            from freetoken.kernel.triton.moe import invalidate_prefill_slots

            if slots.stop > slots.start:
                invalidate_prefill_slots(
                    self.id_of_slot, self.slot_for_id, self.usage, slots.start, slots.stop - slots.start
                )
        else:
            old_ids = self.id_of_slot[slots]
            self.slot_for_id.view(-1)[old_ids[old_ids >= 0].long()] = -1
            self.id_of_slot[slots] = -1
            # usage=0 makes these slots the oldest, so the argmin(usage) victim selection in
            # ensure_experts evicts them first.
            self.usage[slots] = 0
        self._stage_dirty[buffer_id] = True
        # the pad may hold rows of a wider layout or of a pool
        for tail in self._stage_tails[self.layer_layout[layer_id]][buffer_id]:
            tail.zero_()

    def begin_prefill(self) -> None:
        if not self.prefill_overlap:
            return
        self._prefill_buffer_layer = [None, None]
        self._prefill_buffer_released = [True, True]
        if self.prefill_copy_stream is not None:
            # Fence this prefill's copy-stream work behind everything already enqueued
            # on the compute stream. The release/ready events only order against the
            # *previous prefill*; under overlap scheduling a new prefill can be enqueued
            # while the preceding decode batch is still running, and that decode may
            # have loaded experts into the slots the buffers borrow -- without this
            # fence the first prefetch would stomp bytes a running GEMM is reading.
            self.prefill_begin_event.record(torch.cuda.current_stream(self.device))
            self.prefill_copy_stream.wait_event(self.prefill_begin_event)
        self._prefill_hit_d2d_active = self.prefill_hit_d2d and self._hit_d2d_usable()
        if self._prefill_hit_d2d_active:
            # The copy stream is fenced behind the previous decode, so the snapshot
            # observes its final slot map; one host sync per chunk, then per-layer
            # classification is pure host math.
            with torch.cuda.stream(self.prefill_copy_stream):
                self._prefill_slot_snapshot.copy_(self.slot_for_id, non_blocking=True)
            self.prefill_copy_stream.synchronize()

    def prefetch_prefill_layer(self, layer_id: int) -> None:
        if not self.prefill_overlap or layer_id >= self.num_layers:
            return
        if layer_id < 0:
            raise ValueError(f"Invalid prefill layer id: {layer_id}")

        assert self.banks and self._stage_views

        buffer_id = layer_id % 2
        if self._prefill_buffer_layer[buffer_id] == layer_id:
            return
        if self._prefill_buffer_layer[buffer_id] is not None:
            assert self._prefill_buffer_released[buffer_id], (
                "Prefill overlap buffer is being reused before release"
            )

        def copy() -> None:
            self._invalidate_prefill_buffer(buffer_id, layer_id)
            for (per_layer, _), view in zip(self.banks, self.stage_views(layer_id, buffer_id)):
                view.copy_(per_layer[layer_id], non_blocking=True)
            if self.audit is not None:
                from freetoken.moe.slot_audit import PREFILL_BUFFER

                self.audit.record_range(self, PREFILL_BUFFER, layer_id, buffer_id * self.num_experts, self.num_experts, meta=False)

        if self._prefill_hit_d2d_active:
            self._prefetch_split(layer_id, buffer_id)
        elif self.prefill_copy_stream is None:
            copy()
        else:
            with torch.cuda.stream(self.prefill_copy_stream):
                if self._prefill_buffer_has_release_event[buffer_id]:
                    self.prefill_copy_stream.wait_event(self.prefill_release_events[buffer_id])
                copy()
                self.prefill_ready_events[buffer_id].record(self.prefill_copy_stream)

        self._prefill_buffer_layer[buffer_id] = layer_id
        self._prefill_buffer_released[buffer_id] = False

    def _hit_d2d_usable(self) -> bool:
        """Whether the hit-D2D split can serve this prefill; logs the first fallback.

        The flag is an auto-fallback optional: any unusable condition must degrade
        to the legacy full-layer copy AND say so once in the server log, so a
        configuration that silently runs the legacy path is visible.
        """
        from freetoken.kernel.fast_index_copy import _skip_fast_index_copy_enabled

        if self._prefill_slot_snapshot is None or self.prefill_copy_stream is None:
            reason = "prefill overlap buffers are not initialized for this device"
        elif _skip_fast_index_copy_enabled():
            reason = "FREETOKEN_SKIP_FAST_INDEX_COPY is set (the hit gather would be a no-op)"
        elif not self._copy_fused_ok:
            reason = "the fused copy plan is unavailable (bank alignment or FREETOKEN_FUSED_COPY=0)"
        elif all(stable >= s for stable, s in zip(self._pool_stable_from, self.pool_sizes)):
            reason = (
                f"cache_size {self.cache_size} leaves no hit region "
                f"(every slot lies under the prefill staging buffers of {2 * self.num_experts} experts)"
            )
        elif not self._resolve_batch_memcpy():
            reason = "cudaMemcpyBatchAsync is unavailable"  # resolve logged the specifics
        else:
            return True
        if not self._hit_d2d_fallback_logged:
            logger.warning(
                f"MoE prefill hit-D2D requested but unavailable ({reason}); "
                "falling back to full-layer copies"
            )
            self._hit_d2d_fallback_logged = True
        return False

    def _resolve_batch_memcpy(self) -> bool:
        if self._batch_memcpy is None:
            try:
                from freetoken.kernel.batch_memcpy import load_batch_memcpy

                self._batch_memcpy = load_batch_memcpy()
            except Exception as exc:  # noqa: BLE001 -- any build/runtime gap => legacy path
                logger.warning(f"MoE prefill hit-D2D disabled ({exc}); using full-layer copies")
                self._batch_memcpy = False
        return self._batch_memcpy is not False

    def _prefetch_split(self, layer_id: int, buffer_id: int) -> None:
        """Hit/miss-split prefetch of one expert layer into the double buffer.

        Resident experts are gathered cache -> buffer on the CURRENT stream, fully
        device-side: a one-launch compaction reads the LIVE slot_for_id row into
        fixed-shape gather indices (no host round trip), then fast_index_copy_multi
        moves the rows. Serializing the gather before this layer's GEMMs costs its
        plain duration instead of nondeterministic SM contention. Misses cross
        PCIe as ONE cudaMemcpyBatchAsync of coalesced expert-id runs on the copy
        stream, under the existing release/ready event discipline; its host-built
        run list comes from the begin-of-chunk snapshot because the batch API
        takes HOST pointer arrays. Live-vs-snapshot cannot disagree: the only
        chunk-internal writer (buffer invalidation) rewrites slots already below
        the pool's staging threshold, and those slots (and -1) are misses on both sides
        -- the buffers own those slots, so their bytes are volatile within the
        chunk. Hit and miss row sets are disjoint, so the streams need no
        ordering against each other.
        """
        import numpy as np

        from freetoken.kernel.fast_index_copy import fast_index_copy_multi_jit
        from freetoken.moe.offload_kernels import prefill_hit_compact

        E = self.num_experts
        layout = self.layer_layout[layer_id]
        pool = self.pool_of_layer[layer_id]
        snap = self._prefill_snapshot_np[layer_id]
        hit_mask = snap >= self._pool_stable_from[pool]
        self.prefill_hit_rows += int(hit_mask.sum())
        self.prefill_total_rows += E
        if self._gather_feat_bytes[layout] is not None:
            prefill_hit_compact(self, layer_id)
            # blocks_per_bank=64 vs the PCIe-tuned default of 8: HBM D2D needs the
            # wider grid (~22 GB/s per 1024-thread block on H100).
            fast_index_copy_multi_jit(
                self._gather_stage_ptrs[layout][buffer_id],
                self._gather_pool_ptrs[pool],
                self._gather_feat_bytes[layout],
                self._prefill_hit_dst,
                self._prefill_hit_src,
                self._prefill_hit_num,
                blocks_per_bank=64,
            )
            if self.audit is not None:
                from freetoken.moe.slot_audit import PREFILL_HIT_D2D

                self.audit.record_plan(self, PREFILL_HIT_D2D, layer_id, self._prefill_hit_dst, None, self._prefill_hit_num,
                                       meta=False, aux=self._prefill_hit_src, row_base=buffer_id * E)
        miss = np.nonzero(~hit_mask)[0]
        with torch.cuda.stream(self.prefill_copy_stream):
            if self._prefill_buffer_has_release_event[buffer_id]:
                self.prefill_copy_stream.wait_event(self.prefill_release_events[buffer_id])
            self._invalidate_prefill_buffer(buffer_id, layer_id)
            if miss.size:
                run_starts = np.concatenate(([0], np.nonzero(np.diff(miss) != 1)[0] + 1))
                starts = miss[run_starts]
                lengths = np.diff(np.concatenate((run_starts, [miss.size])))
            dst, src, nbytes = [], [], []
            for b, feat in enumerate(self._row_bytes[layout]):
                stage = self._copy_dst_ptrs_host[b] + buffer_id * self._stage_bytes[b]
                if feat == 0:
                    continue
                if feat < _SMALL_BANK_FEAT_BYTES:
                    # Whole layer as one entry, EVEN with zero misses: it keeps every
                    # batch entry above the driver's async floor and covers the hit
                    # rows the gather skips for these banks.
                    dst.append(stage)
                    src.append(self._copy_src_ptrs_host[layer_id][b])
                    nbytes.append(E * feat)
                elif miss.size:
                    dst.extend(stage + starts * feat)
                    src.extend(self._copy_src_ptrs_host[layer_id][b] + starts * feat)
                    nbytes.extend(lengths * feat)
            if dst:
                self._batch_memcpy(
                    torch.tensor(dst, dtype=torch.int64),
                    torch.tensor(src, dtype=torch.int64),
                    torch.tensor(nbytes, dtype=torch.int64),
                    torch.cuda.current_stream(self.device).cuda_stream,
                )
                if self.audit is not None:
                    from freetoken.moe.slot_audit import PREFILL_SPLIT_H2D

                    # every row gets at least its small banks from the host, the misses all of them
                    self.audit.record_range(self, PREFILL_SPLIT_H2D, layer_id, buffer_id * E, E, meta=False)
            self.prefill_ready_events[buffer_id].record(self.prefill_copy_stream)

    def wait_prefill_layer(self, layer_id: int) -> tuple[torch.Tensor, ...]:
        """Full-layer ``[num_experts, ...]`` bank views for ``layer_id``, one per
        registered bank in registration order: bf16 ``(gate_up, down)``; nvfp4
        marlin/b12x ``(gate_up_packed, gate_up_scale, down_packed, down_scale)``;
        nvfp4 native adds the two global banks after each scale bank."""
        assert self.prefill_overlap
        assert self._stage_views
        self.prefetch_prefill_layer(layer_id)
        buffer_id = layer_id % 2
        assert self._prefill_buffer_layer[buffer_id] == layer_id
        if self.prefill_ready_events:
            torch.cuda.current_stream(self.device).wait_event(self.prefill_ready_events[buffer_id])
        return self.stage_views(layer_id, buffer_id)

    def release_prefill_layer(self, layer_id: int) -> None:
        if layer_id == self.num_layers - 1:
            # the staged slots were invalidated and nothing registers them within the forward, so their bytes are free to clear
            for b in range(2):
                if self._stage_dirty[b]:
                    # staged rows left there may be shifted or of another quant, and kernels still read them: MMQ past a slot's last row, weight-0 routes on slot 0
                    for (_, cache), stage in zip(self.banks, self._stage_bytes):
                        cache.view(-1).view(torch.uint8)[b * stage : (b + 1) * stage].zero_()
                    self._stage_dirty[b] = False
        if not self.prefill_overlap:
            return
        buffer_id = layer_id % 2
        if self._prefill_buffer_layer[buffer_id] != layer_id:
            return
        if self.prefill_release_events:
            self.prefill_release_events[buffer_id].record(torch.cuda.current_stream(self.device))
            self._prefill_buffer_has_release_event[buffer_id] = True
        self._prefill_buffer_released[buffer_id] = True

    def ensure_experts(
        self,
        layer_id: int,
        expert_ids: torch.Tensor,
        *,
        update_state: bool = True,
        router_logits: torch.Tensor | None = None,
        pin_since: torch.Tensor | None = None,
    ) -> None:
        """``update_state=False`` (small-prefill ensures) keeps a scored policy's state and
        decode-step clock still; ``router_logits`` (``[rows, num_experts]``) feed the ``rule``
        policy's near misses; ``pin_since`` (a ``step`` value) keeps every slot touched after it
        (a small prefill's earlier chunks). All are ignored under LRU, whose recency order
        already keeps those slots."""
        from freetoken.moe.offload_kernels import ensure_experts, ensure_experts_scored

        if self.collect_decode_freq:
            # ``expert_ids`` still holds raw expert ids here (the kernel rewrites them to
            # slot ids in place), so snapshot the routing histogram before that happens.
            ids = expert_ids.reshape(-1).long()
            self.decode_freq[layer_id].scatter_add_(0, ids, torch.ones_like(ids))
        self._pending_src_layer = layer_id
        self._pending_whole_layer = False
        audit = self.audit
        if audit is not None and update_state and layer_id == self.evict_clock_layer():
            audit.begin_step()
        lowpri = self.prefetch_on
        width = expert_ids.numel()
        if self.cache_policy_id == 0 and not lowpri:
            ensure_experts(self, layer_id, expert_ids)
        else:
            # prefetch on: LRU runs the vendored kernel (flashlib's cannot key low-priority slots)
            ensure_experts_scored(
                self,
                layer_id,
                expert_ids,
                bump_tok=update_state and layer_id == self.evict_clock_layer(),
                update_state=update_state,
                router_logits=router_logits,
                pin_since=pin_since,
                lowpri=lowpri,
                pf_count=self.prefetch.count_args(layer_id) if lowpri and update_state else None,
            )
        if audit is not None:
            from freetoken.moe import slot_audit as sa

            kind = sa.DEMAND_INSTALL if update_state else sa.PREFILL_INSTALL
            self._pending_kind = sa.DEMAND_COPY if update_state else sa.PREFILL_COPY
            self._pending_width = width
            audit.record_plan(self, kind, layer_id, self.evict_slots, self.src_indices, self.num_indices, meta=True, width=width)

    def prefetch_ensure(
        self, layer_id: int, query: torch.Tensor, dst_slots: torch.Tensor, src_rows: torch.Tensor,
        num: torch.Tensor, stats_row: torch.Tensor, ready: torch.Tensor, budget: int | None = None,
    ) -> None:
        """Install the first ``budget`` of ``query``'s non-resident experts of ``layer_id`` as
        low-priority slots and plan their copy into ``dst_slots``/``src_rows``/``num`` (never the
        demand plan, which copy(L-1) still reads). Runs on the compute stream like every other slot
        map writer."""
        from freetoken.moe.offload_kernels import prefetch_ensure_experts

        prefetch_ensure_experts(self, layer_id, query, dst_slots, src_rows, num, stats_row, ready, budget=budget)
        if self.audit is not None:
            from freetoken.moe.slot_audit import PREFETCH_INSTALL

            self.audit.record_plan(self, PREFETCH_INSTALL, layer_id, dst_slots, src_rows, num, meta=True)

    def ensure_experts_hybrid(
        self, layer_id: int, expert_ids: torch.Tensor, weights: torch.Tensor, clamp: bool = False
    ) -> HybridRoutes:
        """Capped-fetch LRU for the hybrid backend.

        Like :meth:`ensure_experts` but assigns slots to (and schedules copies for) at
        most ``hybrid_max_fetch`` -- or ``~hybrid_fetch_fraction * misses`` when the
        fraction is set -- of this step's missing experts; the overflow misses are
        left non-resident and ``expert_ids`` is rewritten to their cache slot (hit or
        freshly fetched) or ``-1`` (overflow -> compute on the CPU). ``num_indices`` holds
        the capped fetch count (for ``copy_missing``); ``num_missing_full`` the pre-cap
        miss count (for stats). All device-side / fixed-shape, so it is CUDA-graph safe.
        Returns the routes split between GPU and CPU (:class:`freetoken.moe.offload_kernels.HybridRoutes`)."""
        from freetoken.moe.offload_kernels import ensure_experts_hybrid

        if self.collect_decode_freq:
            ids = expert_ids.reshape(-1).long()
            self.decode_freq[layer_id].scatter_add_(0, ids, torch.ones_like(ids))
        self._pending_src_layer = layer_id
        self._pending_whole_layer = False
        fraction = self.hybrid_fetch_fraction[layer_id] if self.hybrid_fetch_fraction is not None else 0.0
        return ensure_experts_hybrid(self, layer_id, expert_ids, weights, self.hybrid_max_fetch, fraction, clamp)

    def materialize_layer(self, layer_id: int) -> None:
        from freetoken.moe.offload_kernels import materialize_layer

        self._pending_src_layer = layer_id
        self._pending_whole_layer = True
        audit = self.audit
        if audit is not None:
            from freetoken.moe import slot_audit as sa

            # the kernel also frees this layer's slots above E; record them while they still name it
            audit.record_clear(self, sa.MATERIALIZE_CLEAR, layer_id, 0, self.cache_size, layer_above_e=True)
        materialize_layer(self, layer_id)
        if self.evict_slot_owner is not None:
            self.evict_slot_owner[: self.num_experts].fill_(-1)  # it installs the layer into these slots
        if audit is not None:
            self._pending_kind = sa.MATERIALIZE_COPY
            self._pending_width = self.num_experts
            audit.record_plan(self, sa.MATERIALIZE_INSTALL, layer_id, self.evict_slots, self.src_indices, self.num_indices,
                              meta=True, width=self.num_experts)

    def reset(self) -> None:
        from freetoken.moe.offload_kernels import reset_cache

        if self.audit is not None:
            from freetoken.moe.slot_audit import RESET

            self.audit.record_clear(self, RESET, -1, 0, self.cache_size)
        reset_cache(self)
        # Per-expert recency is not cache_size-shaped, so reset_cache leaves it alone; wipe
        # it here so a new sequence starts with cold hybrid fetch priorities.
        self.expert_recency.fill_(-1)
        self.reset_evict_state()
        if self.prefetch is not None:
            # the graph runner resets after each capture: drop the capture-time counts too
            self.prefetch.reset_counters()

    def reset_stats(self) -> None:
        self.prefill_hit_rows = 0
        self.prefill_total_rows = 0
        self.lru_stats.zero_()
        self.stat_missing.zero_()
        self.stat_active.zero_()
        self.stat_calls.zero_()
        self.stat_fetched.zero_()
        self.stat_missing_layer.zero_()
        self.stat_active_layer.zero_()
        self.stat_fetched_layer.zero_()
        self.stat_steps_layer.zero_()

    def record_decode_stats(self, layer_id: int) -> None:
        """No-op: ``ensure_experts`` accumulates into ``lru_stats`` inside its own launch.

        Kept so the hybrid and non-hybrid call sites stay symmetric. The previous version
        was eight torch ops per layer per step, all captured into the decode graph.
        """

    def decode_miss_stats(self) -> dict:
        if self.decode_target == "hybrid":
            active = int(self.stat_active.item())
            missing = int(self.stat_missing.item())
            calls = int(self.stat_calls.item())
        else:
            active, missing, calls = (int(x) for x in self.lru_stats.sum(0))
        fetched = int(self.stat_fetched.item())
        return {
            "layer_calls": calls,
            "active_per_layer": (active / calls) if calls else 0.0,
            "missing_per_layer": (missing / calls) if calls else 0.0,
            "miss_rate": (missing / active) if active else 0.0,
            # hybrid: how the misses split between PCIe fetch (GPU) and CPU compute.
            "fetched_per_layer": (fetched / calls) if calls else 0.0,
            "cpu_per_layer": ((missing - fetched) / calls) if calls else 0.0,
            "fetch_rate": (fetched / missing) if missing else 0.0,
            # prefill hit-D2D split: expert rows served from the cache (D2D) vs all
            # rows prefetched into the double buffer since the last reset.
            "prefill_hit_rows": self.prefill_hit_rows,
            "prefill_rows": self.prefill_total_rows,
        }

    def decode_miss_stats_per_layer(self) -> dict:
        """Per-MoE-layer realized decode stats for one (reset_stats-delimited) window.

        Requires ``collect_stats`` and the call sites passing ``layer_id``. Returns python
        lists indexed by MoE-layer id: missing/active experts per step and the realized
        miss_rate (missing/active) -- i.e. how cacheable each layer's routing actually was
        under the running LRU. Reads device tensors once (no per-step host sync)."""
        if self.decode_target == "hybrid":
            steps = self.stat_steps_layer.tolist()
            missing = self.stat_missing_layer.tolist()
            active = self.stat_active_layer.tolist()
        else:
            cols = self.lru_stats.t().tolist()
            active, missing, steps = cols[Stat.ACTIVE], cols[Stat.MISS], cols[Stat.CALLS]
        fetched = self.stat_fetched_layer.tolist()
        per_layer = []
        for L in range(self.num_layers):
            s, m, a, f = steps[L], missing[L], active[L], fetched[L]
            per_layer.append({
                "layer": L,
                "pool": self.pool_of_layer[L],
                "pool_slots": self.pool_slots(L),
                "steps": s,
                "active_per_step": (a / s) if s else 0.0,
                "missing_per_step": (m / s) if s else 0.0,
                "miss_rate": (m / a) if a else 0.0,
                "fetched_per_step": (f / s) if s else 0.0,
            })
        return {"per_layer": per_layer}

    def decode_routing_stats(self) -> dict:
        """Per-layer decode routing concentration, for cache-skew analysis.

        Uses the histogram from ``collect_decode_freq``. ``static_topk_hit_at_slots`` is the
        hit rate of the best *fixed* expert set: each layer's ``cache_size/num_layers`` (rounded)
        most frequent experts over the whole histogram. It measures routing skew only. It is
        not an upper bound on a dynamic cache: LRU exploits temporal locality and can beat it.
        """
        freq = self.decode_freq.float()
        total = freq.sum(dim=1)
        valid = total > 0
        if int(valid.sum()) == 0:
            return {}
        slots_per_layer = self.cache_size / self.num_layers
        C = max(1, int(round(slots_per_layer)))
        sorted_f, _ = torch.sort(freq, dim=1, descending=True)
        static_hit = (sorted_f[:, :C].sum(dim=1)[valid] / total[valid]).mean().item()
        ws = (freq > 0).sum(dim=1).float()
        cdf = torch.cumsum(sorted_f, dim=1) / total.clamp(min=1).unsqueeze(1)
        cover90 = ((cdf < 0.9).sum(dim=1).float() + 1)[valid]
        p = freq / total.clamp(min=1).unsqueeze(1)
        ent = -(p * p.clamp(min=1e-12).log()).sum(dim=1)[valid]
        norm_ent = (ent / torch.log(torch.tensor(float(self.num_experts)))).mean().item()
        return {
            "slots_per_layer": slots_per_layer,
            "working_set_mean": ws[valid].mean().item(),
            "working_set_max": int(ws[valid].max().item()),
            "experts_for_90pct": cover90.mean().item(),
            "static_topk_hit_at_slots": static_hit,
            "norm_entropy": norm_ent,
        }

    def copy_missing(self, layer_id: int | None = None) -> None:
        """Copy the misses the last ensure_experts/materialize_layer staged, from ``layer_id``'s
        host banks (default: the layer that staged them)."""
        assert self.banks, "set_bank_sources must register the banks first"
        if layer_id is None:
            layer_id = self._pending_src_layer
        assert layer_id is not None, "no staged misses (ensure_experts/materialize_layer first)"
        if layer_id in self._unpinned_layers:
            if not self._pending_whole_layer:
                raise RuntimeError(
                    f"layer {layer_id} is unpinned: its only copy is the whole-layer "
                    f"pageable materialize (position == expert id); ensure_experts's "
                    f"LRU slot remap cannot be honored without a device alias"
                )
            # the only copy a non-pinned layer ever needs is the non-overlap prefill materialize, which schedules the whole layer into staging buffer 0 with position == expert id -- a plain synchronous pageable H2D copy
            # never CUDA-graph captured: prefill is not captured, and decode never reaches this branch (it routes to the CPU executor)
            for (per_layer, _), view in zip(self.banks, self.stage_views(layer_id, 0)):
                view.copy_(per_layer[layer_id])
            if self.audit is not None:
                from freetoken.moe.slot_audit import PAGEABLE_COPY

                self.audit.record_range(self, PAGEABLE_COPY, layer_id, 0, self.num_experts, meta=False)
            return
        # materialize stages rows of staging buffer 0; ensure stages slots local to the layer's pool
        self.copy_rows(layer_id, self.evict_slots, self.src_indices, self.num_indices, slim=_SLIM_COPY,
                       kind=self._pending_kind, width=self._pending_width, whole=self._pending_whole_layer)

    def copy_rows(
        self, layer_id: int, dst_slots: torch.Tensor, src_rows: torch.Tensor, num: torch.Tensor, *, slim: bool,
        kind: int = 0, width: int | None = None, whole: bool = False,
    ) -> None:
        """Copy host rows ``src_rows[:num]`` of ``layer_id``'s banks into slots ``dst_slots[:num]``
        on the current stream (the demand plan, or a prefetch plan on the prefetch copy stream);
        ``kind`` labels the copy for the slot audit and ``width`` bounds ``num`` for it. ``dst_slots``
        are local to the layer's pool, or rows of staging buffer 0 when ``whole`` (materialize)."""
        self._copy_rows(layer_id, dst_slots, src_rows, num, slim=slim, whole=whole)
        if self.audit is not None:
            from freetoken.moe.slot_audit import OTHER_COPY

            self.audit.record_plan(self, kind or OTHER_COPY, layer_id, dst_slots, src_rows, num, meta=False, width=width)

    def _copy_rows(
        self, layer_id: int, dst_slots: torch.Tensor, src_rows: torch.Tensor, num: torch.Tensor, *, slim: bool,
        whole: bool = False,
    ) -> None:
        if self._copy_fused_ok:
            from freetoken.kernel.fast_index_copy import fast_index_copy_multi_jit, fast_index_copy_multi_slim_jit

            # One launch copies the missing rows for every bank (instead of one launch per
            # bank). The slot/row plan is shared across banks; src_rows holds layer-local
            # expert rows, resolved against this layer's source pointers (layer_id is a
            # static int per captured graph node).
            dst_ptrs = self._copy_dst_ptrs if whole else self._copy_dst_ptrs_pool[self.pool_of_layer[layer_id]]
            feat_bytes = self._copy_feat_bytes[self.layer_layout[layer_id]]
            if slim:
                fast_index_copy_multi_slim_jit(dst_ptrs, self._copy_src_ptrs[layer_id], feat_bytes, dst_slots, src_rows, num)
            else:
                fast_index_copy_multi_jit(
                    dst_ptrs, self._copy_src_ptrs[layer_id], feat_bytes, dst_slots, src_rows, num,
                    blocks_per_bank=self._copy_blocks_per_bank,
                )
            return

        from freetoken.kernel import fast_index_copy_jit

        views = self.stage_views(layer_id, 0) if whole else self.pool_bank_views(layer_id)
        for (per_layer, _), cache in zip(self.banks, views):
            if cache.numel() == 0:
                continue
            fast_index_copy_jit(cache, dst_slots, per_layer[layer_id], src_rows, num)

    @contextmanager
    def decode_side(self, layer_id: int) -> Iterator[None]:
        """Run the body on ``decode_copy_stream``, ordered after the work already on the current
        stream; ``join_decode_side`` must follow before its results are read.

        The copy stays on the current stream right after its ensure, so it never waits on a
        cross-stream edge; the side work (slack-rich) takes that latency and any SM contention.
        Under capture the event pair becomes a fork and a join edge of the graph."""
        forked, done = self._decode_copy_events[layer_id]
        forked.record(torch.cuda.current_stream(self.device))
        self.decode_copy_stream.wait_event(forked)
        with torch.cuda.stream(self.decode_copy_stream):
            yield
            done.record(self.decode_copy_stream)

    def join_decode_side(self, layer_id: int) -> None:
        torch.cuda.current_stream(self.device).wait_event(self._decode_copy_events[layer_id][1])


def iter_offload_moe_layers(model) -> Iterator:
    from freetoken.layers import OffloadMoELayer, iter_moe_layers

    return (layer for layer in iter_moe_layers(model) if isinstance(layer, OffloadMoELayer))


def attach_offload_moe_cache(model, cache: OffloadMoeCache) -> list:
    layers = list(iter_offload_moe_layers(model))
    for layer in layers:
        layer.offload_cache = cache
    return layers
