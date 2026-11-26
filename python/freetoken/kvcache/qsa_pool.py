"""QSA compressed-block sparse KV pool: paged GQA K/V + compressed index keys + pending ring.

Qwen3.8-Flash-Next scores whole ``index_ratio``-token groups instead of single tokens, so
its indexer slab holds ONE compressed key row per group, addressed by ``slot //
index_ratio``. Because ``page_size % index_ratio == 0``, a group's tokens always live in one
page at consecutive slots, which makes that division well-defined: the compressed rows are a
1/ratio shadow of the K/V pages and follow page sharing and eviction for free -- no
allocator, no free, no clear (SGLang qsa_kv_pool / vLLM compressed-region precedent).

Two tiers ride alongside the shadow slab and are NOT per-token:
- ``pending_ring``: the last ``ring_capacity`` pre-RoPE index keys of each running request (sized by ``ring_capacity_for``), indexed by ``Req.table_idx``. A group that straddles two forwards (chunked prefill, and
  every decode step) reads its already-consumed members from here. Never cleared: a new
  tenant of a table_idx starts at a group boundary (cached_len is 0 or a page multiple), so
  its first closing group takes every member from its own forward.
- scratch rows at ``cmp_scratch_base``: one row per request slot, the write target for rows
  whose group does not close in this forward, so the compress kernel scatters unconditionally
  with no negative index and no cross-row conflict (DSV4 precedent).

The slab is amortized into the per-token KV price (``unit_bytes``); the ring and scratch are
fixed and priced through ``kv_cost``'s ``fixed_cache_size``.
"""

from __future__ import annotations

import ctypes
import math
import mmap
import os
from collections.abc import Sequence

import torch

from .format import NONE, KVFormat
from .mha_pool import MHAKVCache

# The index tiers are always 2-byte (compute dtype); spec_kv_bytes_per_token budgets the same.
_INDEX_DTYPE_BYTES = 2
# t/h/w int32 rope position kept per KV slot on mrope models
_ROPE_POS_BYTES = 3 * 4
_BLK = 4096
_PIN_GROUP_BYTES = 4 << 30


def _align_up(n: int, align: int) -> int:
    return ((n + align - 1) // align) * align


def _coalesce_page_runs(src: list[int], dst: list[int]) -> list[tuple[int, int, int]]:
    """Merge consecutive ``(src[i], dst[i])`` pairs into ``(src_start, dst_start, length)`` runs."""
    if not src:
        return []
    runs: list[tuple[int, int, int]] = []
    s0, d0, n = int(src[0]), int(dst[0]), 1
    for i in range(1, len(src)):
        si, di = int(src[i]), int(dst[i])
        if si == s0 + n and di == d0 + n:
            n += 1
        else:
            runs.append((s0, d0, n))
            s0, d0, n = si, di, 1
    runs.append((s0, d0, n))
    return runs


class _QSAHostArena:
    """Bank-major host arena backing ``num_host_pages`` whole QSA KV pages in one ``mmap``."""

    def __init__(
        self,
        num_host_pages: int,
        dev_banks: list[torch.Tensor],
        *,
        device: torch.device,
        pin_memory: bool,
        engine_stream: torch.cuda.Stream | None = None,
    ) -> None:
        self.num_host_pages = num_host_pages
        self.device = device
        self.engine_stream = engine_stream
        self.copy_stream = (
            torch.cuda.Stream(device=device) if device.type == "cuda" else None
        )
        self.pending_d2h = False
        self._pinned_spans: list[tuple[int, int]] = []

        self.feat_elems = [int(b.shape[1]) for b in dev_banks]
        self.dtypes = [b.dtype for b in dev_banks]
        self.feat_bytes = [
            n * dt.itemsize for n, dt in zip(self.feat_elems, self.dtypes, strict=True)
        ]
        self.bytes_per_page = sum(self.feat_bytes)

        offsets: list[int] = []
        spans: list[int] = []
        cursor = 0
        for fb in self.feat_bytes:
            cursor = _align_up(cursor, _BLK)
            offsets.append(cursor)
            span = num_host_pages * fb
            spans.append(span)
            cursor += span
        asize = max(_BLK, _align_up(cursor, _BLK))
        self._buf = mmap.mmap(-1, asize)
        madv_hp = getattr(mmap, "MADV_HUGEPAGE", 14)
        try:
            self._buf.madvise(madv_hp)
        except OSError:
            pass
        self.addr = ctypes.addressof(ctypes.c_char.from_buffer(self._buf))
        # Prefault + zero before cudaHostRegister (pin-after-fill avoids double page-fault pass).
        torch.frombuffer(self._buf, dtype=torch.uint8, count=asize).zero_()

        mv = memoryview(self._buf)
        self.host_banks: list[torch.Tensor] = [
            torch.frombuffer(
                mv[off : off + span], dtype=dt, count=num_host_pages * elems
            ).view(num_host_pages, elems)
            for off, span, dt, elems in zip(
                offsets, spans, self.dtypes, self.feat_elems, strict=True
            )
        ]

        skip_pin = os.environ.get("FREETOKEN_SKIP_BANK_PIN", "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        self.pinned = False
        if pin_memory and device.type == "cuda" and not skip_pin:
            self._pin_bank_groups(offsets, spans, asize)
            self.pinned = True

        self._batch_memcpy_ok: bool | None = None
        self.multi_h2d_ok = False
        self.dev_bank_ptrs: torch.Tensor | None = None
        self.host_bank_ptrs: torch.Tensor | None = None
        self.bank_feat_bytes: torch.Tensor | None = None
        self.h2d_blocks = 8
        self.refresh_device_banks(dev_banks)

    def _pin_bank_groups(
        self, offsets: list[int], spans: list[int], asize: int
    ) -> None:
        """Register whole-bank groups (<= 4 GiB when possible) so no bank crosses a register boundary."""
        from freetoken.kernel.pinned import host_register
        from freetoken.moe.host_banks import PinFailed, charge_pin, refund_pin

        groups: list[tuple[int, int]] = []
        g_start = 0
        for off, span in zip(offsets, spans, strict=True):
            bank_end = _align_up(off + span, _BLK)
            if bank_end - g_start > _PIN_GROUP_BYTES and off > g_start:
                g_end = _align_up(off, _BLK)
                groups.append((g_start, g_end - g_start))
                g_start = g_end
        if asize > g_start:
            groups.append((g_start, asize - g_start))

        try:
            for g_off, g_len in groups:
                charge_pin(g_len)
                try:
                    host_register(self.addr + g_off, g_len)
                except RuntimeError as exc:
                    refund_pin(g_len)
                    raise PinFailed(
                        f"cudaHostRegister failed for {g_len / 2**30:.1f} GiB KV host arena"
                    ) from exc
                self._pinned_spans.append((self.addr + g_off, g_len))
        except BaseException:
            self.close_pins()
            raise

    def close_pins(self) -> None:
        if not self._pinned_spans:
            return
        if self.device.type == "cuda":
            if self.copy_stream is not None:
                self.copy_stream.synchronize()
            if self.engine_stream is not None:
                self.engine_stream.synchronize()
            else:
                torch.cuda.synchronize(self.device)
            self.pending_d2h = False
        from freetoken.kernel.pinned import host_unregister
        from freetoken.moe.host_banks import refund_pin

        for ptr, span in reversed(self._pinned_spans):
            host_unregister(ptr)
            refund_pin(span)
        self._pinned_spans.clear()
        self.pinned = False

    def __del__(self) -> None:
        try:
            self.close_pins()
        except Exception:
            pass

    def refresh_device_banks(self, dev_banks: list[torch.Tensor]) -> None:
        self.dev_banks = dev_banks
        self.multi_h2d_ok = False
        if self.device.type != "cuda" or not self.pinned:
            return
        from freetoken.kernel.fast_index_copy import default_blocks_per_bank
        from freetoken.kernel.pinned import device_ptr

        dev_ptrs = [b.data_ptr() for b in dev_banks]
        host_ptrs = [device_ptr(h) for h in self.host_banks]
        if any(fb % 16 != 0 for fb in self.feat_bytes):
            return
        if any(p % 16 != 0 for p in dev_ptrs) or any(p % 16 != 0 for p in host_ptrs):
            return
        self.dev_bank_ptrs = torch.tensor(
            dev_ptrs, dtype=torch.int64, device=self.device
        )
        self.host_bank_ptrs = torch.tensor(
            host_ptrs, dtype=torch.int64, device=self.device
        )
        self.bank_feat_bytes = torch.tensor(
            self.feat_bytes, dtype=torch.int64, device=self.device
        )
        self.h2d_blocks = default_blocks_per_bank(self.device)
        self.multi_h2d_ok = True

    def to_host(self, src_pages: list[int], dst_host_pages: list[int]) -> None:
        if not src_pages:
            return
        runs = _coalesce_page_runs(src_pages, dst_host_pages)
        if self.copy_stream is None or not self.pinned:
            for dev_b, host_b in zip(self.dev_banks, self.host_banks, strict=True):
                for s0, d0, n in runs:
                    host_b[d0 : d0 + n].copy_(dev_b[s0 : s0 + n].to("cpu"))
            return

        cur = torch.cuda.current_stream(self.device)
        self.copy_stream.wait_stream(cur)
        if self.engine_stream is not None and self.engine_stream != cur:
            self.copy_stream.wait_stream(self.engine_stream)

        if self._batch_memcpy_ok is not False:
            try:
                from freetoken.kernel.batch_memcpy import batch_memcpy_jit

                # CUDA 13.0 cudaMemcpyBatchAsync blocks the host thread when one batch mixes
                # >= 256 KiB and < 253 KiB entries on cudaHostRegister memory (offload_cache.py:23).
                big_dst, big_src, big_sz = [], [], []
                small_dst, small_src, small_sz = [], [], []
                for dev_b, host_b, fb in zip(
                    self.dev_banks, self.host_banks, self.feat_bytes, strict=True
                ):
                    d_base = host_b.data_ptr()
                    s_base = dev_b.data_ptr()
                    for s0, d0, n in runs:
                        sz = n * fb
                        if sz >= 256 * 1024:
                            big_dst.append(d_base + d0 * fb)
                            big_src.append(s_base + s0 * fb)
                            big_sz.append(sz)
                        else:
                            small_dst.append(d_base + d0 * fb)
                            small_src.append(s_base + s0 * fb)
                            small_sz.append(sz)
                for dst_list, src_list, sz_list in (
                    (big_dst, big_src, big_sz),
                    (small_dst, small_src, small_sz),
                ):
                    if dst_list:
                        batch_memcpy_jit(
                            torch.tensor(dst_list, dtype=torch.int64),
                            torch.tensor(src_list, dtype=torch.int64),
                            torch.tensor(sz_list, dtype=torch.int64),
                            self.copy_stream.cuda_stream,
                        )
                self._batch_memcpy_ok = True
                self.pending_d2h = True
                return
            except Exception:
                self._batch_memcpy_ok = False

        with torch.cuda.stream(self.copy_stream):
            for dev_b, host_b in zip(self.dev_banks, self.host_banks, strict=True):
                for s0, d0, n in runs:
                    host_b[d0 : d0 + n].copy_(dev_b[s0 : s0 + n], non_blocking=True)
        self.pending_d2h = True

    def from_host(self, src_host_pages: list[int], dst_pages: list[int]) -> None:
        if not src_host_pages:
            return
        if self.multi_h2d_ok:
            from freetoken.kernel.fast_index_copy import fast_index_copy_multi_jit
            from freetoken.kernel.sm_upload import upload_step_tensor

            dst_idx = upload_step_tensor(
                torch.tensor(dst_pages, dtype=torch.int32), self.device
            )
            src_idx = upload_step_tensor(
                torch.tensor(src_host_pages, dtype=torch.int32), self.device
            )
            fast_index_copy_multi_jit(
                self.dev_bank_ptrs,
                self.host_bank_ptrs,
                self.bank_feat_bytes,
                dst_idx,
                src_idx,
                blocks_per_bank=self.h2d_blocks,
            )
            return

        runs = _coalesce_page_runs(src_host_pages, dst_pages)
        nb = self.copy_stream is not None and self.pinned
        for dev_b, host_b in zip(self.dev_banks, self.host_banks, strict=True):
            for s0, d0, n in runs:
                dev_b[d0 : d0 + n].copy_(host_b[s0 : s0 + n], non_blocking=nb)

    def fence(self) -> None:
        if self.pending_d2h and self.copy_stream is not None:
            torch.cuda.current_stream(self.device).wait_stream(self.copy_stream)
            self.pending_d2h = False


class QSAKVCache(MHAKVCache):
    """MHA paged pool + the compressed index-key slab + the per-request pending ring.

    ``cmp_k_cache(slot)`` is row-flat ``[num_pages * page_size // index_ratio + num_req_slots,
    index_head_dim]``: row ``r < cmp_scratch_base`` holds the compressed key of the token group
    whose K/V slots are ``[r * index_ratio, (r + 1) * index_ratio)``, and the rows from
    ``cmp_scratch_base`` on are the per-request-slot scratch sinks. ``slot`` is the sparse
    layer's order in the attention backend, same convention as BSAKVCache/DSAKVCache.
    """

    @classmethod
    def ring_capacity_for(
        cls, index_ratio: int, num_speculative_tokens: int = 0
    ) -> int:
        """Ring depth: one row per pending position, keyed ``position % capacity``; spec decode widens by the draft depth (vLLM sizing)."""
        return index_ratio * math.ceil(
            (index_ratio + num_speculative_tokens) / index_ratio
        )

    def __init__(
        self,
        num_kv_heads: int,
        num_layers: int,
        head_dim: int,
        num_pages: int,
        page_size: int,
        dtype: torch.dtype,
        device: torch.device,
        index_head_dim: int,
        num_index_layers: int,
        index_ratio: int,
        num_req_slots: int,
        ring_capacity: int | None = None,
        layer_ids: Sequence[int] | None = None,
        mrope: bool = False,
        kv_format: KVFormat = NONE,
        kv_buffer: torch.Tensor | None = None,
    ) -> None:
        if index_ratio < 1 or page_size % index_ratio != 0:
            # slot // index_ratio only names one group when a group never straddles a page.
            raise ValueError(
                f"QSA needs page_size ({page_size}) divisible by index_ratio ({index_ratio})"
            )
        if ring_capacity is None:
            ring_capacity = self.ring_capacity_for(index_ratio)
        if ring_capacity < index_ratio:
            # A closing group reads up to index_ratio - 1 past members plus this forward's.
            raise ValueError(
                f"QSA needs ring_capacity ({ring_capacity}) >= index_ratio ({index_ratio})"
            )
        # Index keys ride the compute dtype (the model's index_k is engine-dtype). The KV cost
        # model budgets 2 bytes per token per index layer for the slab
        # (base.spec_kv_bytes_per_token); keep the two in lockstep.
        assert dtype.itemsize == _INDEX_DTYPE_BYTES, (
            f"QSA index slab budgets 2 bytes/token (spec_kv_bytes_per_token); got {dtype}"
        )
        self._index_head_dim = index_head_dim
        self._num_index_layers = num_index_layers
        self._index_ratio = index_ratio
        self._num_req_slots = num_req_slots
        self._ring_capacity = ring_capacity
        self._index_dtype = dtype
        self._page_size = page_size
        self._mrope = mrope
        super().__init__(
            num_kv_heads=num_kv_heads,
            num_layers=num_layers,
            head_dim=head_dim,
            num_pages=num_pages,
            page_size=page_size,
            dtype=dtype,
            device=device,
            layer_ids=layer_ids,
            kv_format=kv_format,
            kv_buffer=kv_buffer,
        )
        if kv_buffer is None and not self.kv_format.quantized:
            self._zero_kv_slabs()
        self._alloc_index_tiers(num_pages)

    def _zero_kv_slabs(self) -> None:
        # Defense-in-depth: the attend kernels pos-mask every K/V load (the real fix for
        # torch.empty's recycled NaN/Inf bit patterns), but a zeroed slab keeps any future
        # unmasked read finite instead of model-poisoning. One memset per (re)allocation.
        self._kv_buffer.zero_()

    def _alloc_index_tiers(self, num_pages: int) -> None:
        # ZERO-initialized: the score kernel reads whole rows of blocks unmasked and relies on
        # never-written tail rows dotting to a finite 0. Written rows are never cleared again,
        # so the kernel must clamp visible blocks to kvlen // index_ratio.
        self._cmp_scratch_base = num_pages * self._page_size // self._index_ratio
        self._cmp_k_buffer = torch.zeros(
            self._num_index_layers,
            self._cmp_scratch_base + self._num_req_slots,
            self._index_head_dim,
            dtype=self._index_dtype,
            device=self._device,
        )
        self._pending_ring = torch.zeros(
            self._num_req_slots,
            self._num_index_layers,
            self._ring_capacity,
            self._index_head_dim,
            dtype=self._index_dtype,
            device=self._device,
        )
        # 3-axis rope position of every stored token: a compressed group ropes at its first token, which under mrope is not derivable from the logical position
        self._rope_positions = (
            torch.zeros(
                num_pages * self._page_size, 3, dtype=torch.int32, device=self._device
            )
            if self._mrope
            else None
        )

    def rebuild(self, num_pages: int) -> None:
        # Free the index tiers BEFORE the K/V realloc (super().rebuild frees + syncs +
        # empty_cache), then re-derive them at the new page count. If the index alloc itself
        # fails (OOM), null the K/V slab too and re-raise: a pool with a grown K/V slab and no
        # index slab would mis-serve silently. Rebuild is idle-only, so zeroing the ring here
        # cannot drop a live request's pending members.
        self._cmp_k_buffer = None
        self._pending_ring = None
        self._rope_positions = None
        super().rebuild(num_pages)
        if not self.kv_format.quantized:
            self._zero_kv_slabs()
        # The host arena keeps its size (a page's shape does not depend on the page count), but
        # its contents belonged to the old pool: every host page is free again.
        if self.num_host_pages:
            self.reclaim_all_host_pages()
        try:
            self._alloc_index_tiers(num_pages)
        except Exception:
            self._kv_buffer = None
            self._k_buffer = None
            self._v_buffer = None
            self._scale_buffer = None
            raise
        if getattr(self, "_host_arena", None) is not None:
            self._host_arena.refresh_device_banks(self._bank_views())

    @classmethod
    def kv_cost(cls, config) -> tuple[int, int, int, int]:
        from freetoken.attention import AttnType

        from .base import spec_kv_bytes_per_token

        num_req_slots = config.max_running_req + 1
        per_token = 0
        fixed = 0
        for spec in config.model_config.kv_cache_group_specs():
            if spec.is_swa:
                continue
            per_token += spec_kv_bytes_per_token(
                spec, config, getattr(config, "kv_format", NONE)
            )
            if spec.attn_type is AttnType.QSA:
                # One index-key row = all index layers at one position.
                row = spec.index_head_dim * spec.num_index_layers * _INDEX_DTYPE_BYTES
                ring = cls.ring_capacity_for(
                    spec.index_ratio, getattr(config, "decode_rows_per_req", 1) - 1
                )
                fixed += num_req_slots * row * (ring + 1)
                if config.model_config.model_is_mrope:
                    per_token += _ROPE_POS_BYTES
        return per_token * config.page_size, fixed, config.page_size, 0

    def unit_bytes(self) -> tuple[int, int]:
        # Only the shadow slab scales with pages, and only its non-scratch rows; the ring and
        # the scratch rows are the fixed term kv_cost reports separately.
        kv, swa = super().unit_bytes()
        tokens = int(self._kv_buffer.shape[2]) * int(self._kv_buffer.shape[3])
        slab = (
            self._num_index_layers
            * self._cmp_scratch_base
            * self._index_head_dim
            * self._index_dtype.itemsize
        )
        return kv + slab // tokens + (_ROPE_POS_BYTES if self._mrope else 0), swa

    def cmp_k_cache(self, slot: int) -> torch.Tensor:
        """Compressed index keys of one sparse layer: ``[rows, index_head_dim]``."""
        return self._cmp_k_buffer[slot]

    def pending_ring(self, slot: int) -> torch.Tensor:
        """One sparse layer's pending ring: ``[num_req_slots, ring_capacity, index_head_dim]``."""
        return self._pending_ring[:, slot]

    @property
    def rope_positions(self) -> torch.Tensor:
        """``[num_tokens, 3]`` int32 t/h/w rope position per KV slot (written by the QSA backend)."""
        assert self._rope_positions is not None, (
            "rope positions are only kept on mrope models"
        )
        return self._rope_positions

    @property
    def cmp_scratch_base(self) -> int:
        """First scratch row of ``cmp_k_cache``; row ``cmp_scratch_base + table_idx`` sinks a
        forward whose group does not close."""
        return self._cmp_scratch_base

    @property
    def index_ratio(self) -> int:
        return self._index_ratio

    @property
    def index_head_dim(self) -> int:
        return self._index_head_dim

    @property
    def ring_capacity(self) -> int:
        return self._ring_capacity

    @property
    def num_req_slots(self) -> int:
        return self._num_req_slots

    # --- P2c cold tier: pinned bank-major host arena (KV_HOST_TIER_DESIGN.md 3.1) ---
    # Same interface as the GDN host bank (linear_state_pool: alloc_host / free_host / to_host /
    # from_host / fence_host_copies). One bank-major mmap arena backs every per-page tensor of
    # QSAKVCache (K/V planes, optional FP8 scales, compressed index-key slabs, and mrope positions).
    # On CUDA, D2H coalesces runs on a side copy stream via batch_memcpy_jit (with fence() ordering
    # reuse of freed device pages), and H2D gathers all 61 banks in a single fast_index_copy_multi_jit
    # launch on the engine stream.

    def _bank_views(self) -> list[torch.Tensor]:
        """Contiguous ``[num_pages, feat_elems]`` views of every per-page device bank."""
        kv_kinds, layers, pages, _ps, _heads, _elem = self._kv_buffer.shape
        banks: list[torch.Tensor] = [
            self._kv_buffer[kv_idx, layer_idx].view(pages, -1)
            for kv_idx in range(kv_kinds)
            for layer_idx in range(layers)
        ]
        if self._scale_buffer is not None:
            for kv_idx in range(kv_kinds):
                for layer_idx in range(layers):
                    banks.append(self._scale_buffer[kv_idx, layer_idx].view(pages, -1))
        for slot in range(self._num_index_layers):
            banks.append(
                self._cmp_k_buffer[slot, : self._cmp_scratch_base].view(pages, -1)
            )
        if self._rope_positions is not None:
            banks.append(self._rope_positions.view(pages, -1))
        return banks

    def close(self) -> None:
        old_arena = getattr(self, "_host_arena", None)
        if old_arena is not None:
            old_arena.close_pins()
            self._host_arena = None
        self._num_host_pages = 0
        self._free_host_pages = []
        self._host_pages = {}

    def attach_host_pages(
        self,
        num_host_pages: int,
        *,
        pin_memory: bool = True,
        engine_stream: torch.cuda.Stream | None = None,
    ) -> None:
        old_arena = getattr(self, "_host_arena", None)
        if old_arena is not None:
            old_arena.close_pins()
        n = max(0, int(num_host_pages))
        self._num_host_pages = n
        self._free_host_pages: list[int] = list(range(n))
        self._host_pages: dict[str, torch.Tensor] = {}
        self.host_copies = None
        if not n:
            self._host_arena = None
            return
        self._host_arena = _QSAHostArena(
            n,
            self._bank_views(),
            device=self._device,
            pin_memory=pin_memory,
            engine_stream=engine_stream,
        )
        if self._rope_positions is not None:
            self._host_pages["rope"] = self._host_arena.host_banks[-1].view(
                n, self._page_size, 3
            )

    @property
    def num_host_pages(self) -> int:
        return getattr(self, "_num_host_pages", 0)

    @property
    def num_free_host_pages(self) -> int:
        return len(getattr(self, "_free_host_pages", ()))

    def host_bytes_per_page(self) -> int:
        arena = getattr(self, "_host_arena", None)
        return 0 if arena is None else arena.bytes_per_page

    def alloc_host_pages(self, n: int = 1) -> list[int]:
        if n > self.num_free_host_pages:
            raise RuntimeError(
                f"KV host tier exhausted: need {n}, have {self.num_free_host_pages}"
            )
        return [self._free_host_pages.pop() for _ in range(n)]

    def free_host_pages(self, host_pages: Sequence[int]) -> None:
        self._free_host_pages.extend(int(h) for h in host_pages)

    def reclaim_all_host_pages(self) -> None:
        self._free_host_pages = list(range(self.num_host_pages))

    @staticmethod
    def _check_pairs(a: Sequence[int], b: Sequence[int]) -> tuple[list[int], list[int]]:
        if len(a) != len(b):
            raise ValueError(f"page lists differ in length: {len(a)} and {len(b)}")
        return [int(x) for x in a], [int(y) for y in b]

    def pages_to_host(self, pages: Sequence[int], host_pages: Sequence[int]) -> None:
        """Copy device pages ``pages[i]`` into host pages ``host_pages[i]``."""
        src, dst = self._check_pairs(pages, host_pages)
        if not src:
            return
        arena = self._host_arena
        assert arena is not None, "no KV host arena attached"
        if self.host_copies is not None:
            self.host_copies.to_host(
                self._arena_pairs(arena.host_banks, dst, arena.dev_banks, src)
            )
            return
        arena.to_host(src, dst)

    def pages_from_host(self, host_pages: Sequence[int], pages: Sequence[int]) -> None:
        """Copy host pages ``host_pages[i]`` into device pages ``pages[i]``."""
        src, dst = self._check_pairs(host_pages, pages)
        if not src:
            return
        arena = self._host_arena
        assert arena is not None, "no KV host arena attached"
        if self.host_copies is not None:
            self.host_copies.from_host(
                self._arena_pairs(arena.dev_banks, dst, arena.host_banks, src)
            )
            return
        arena.from_host(src, dst)

    @staticmethod
    def _arena_pairs(
        dst_banks: list[torch.Tensor],
        dst: list[int],
        src_banks: list[torch.Tensor],
        src: list[int],
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        # views, not gathers: a test copier that copies later must read the page as it is then
        return [
            (dst_b[d], src_b[s])
            for dst_b, src_b in zip(dst_banks, src_banks, strict=True)
            for d, s in zip(dst, src, strict=True)
        ]

    def fence_host_copies(self) -> None:
        """Order a freed page's next write after its host copy."""
        if self.host_copies is not None:
            self.host_copies.fence()
            return
        arena = getattr(self, "_host_arena", None)
        if arena is not None:
            arena.fence()


__all__ = ["QSAKVCache"]
