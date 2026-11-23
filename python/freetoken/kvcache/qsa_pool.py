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

import math
from typing import Sequence

import torch

from .format import NONE, KVFormat
from .mha_pool import MHAKVCache

# The index tiers are always 2-byte (compute dtype); spec_kv_bytes_per_token budgets the same.
_INDEX_DTYPE_BYTES = 2
# t/h/w int32 rope position kept per KV slot on mrope models
_ROPE_POS_BYTES = 3 * 4


class QSAKVCache(MHAKVCache):
    """MHA paged pool + the compressed index-key slab + the per-request pending ring.

    ``cmp_k_cache(slot)`` is row-flat ``[num_pages * page_size // index_ratio + num_req_slots,
    index_head_dim]``: row ``r < cmp_scratch_base`` holds the compressed key of the token group
    whose K/V slots are ``[r * index_ratio, (r + 1) * index_ratio)``, and the rows from
    ``cmp_scratch_base`` on are the per-request-slot scratch sinks. ``slot`` is the sparse
    layer's order in the attention backend, same convention as BSAKVCache/DSAKVCache.
    """

    @classmethod
    def ring_capacity_for(cls, index_ratio: int, num_speculative_tokens: int = 0) -> int:
        """Ring depth: one row per pending position, keyed ``position % capacity``; spec decode widens by the draft depth (vLLM sizing)."""
        return index_ratio * math.ceil((index_ratio + num_speculative_tokens) / index_ratio)

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
        )
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
            torch.zeros(num_pages * self._page_size, 3, dtype=torch.int32, device=self._device)
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

    @classmethod
    def kv_cost(cls, config) -> tuple[int, int, int, int]:
        from .base import spec_kv_bytes_per_token
        from freetoken.attention import AttnType

        num_req_slots = config.max_running_req + 1
        per_token = 0
        fixed = 0
        for spec in config.model_config.kv_cache_group_specs():
            if spec.is_swa:
                continue
            per_token += spec_kv_bytes_per_token(spec, config, getattr(config, "kv_format", NONE))
            if spec.attn_type is AttnType.QSA:
                # One index-key row = all index layers at one position.
                row = spec.index_head_dim * spec.num_index_layers * _INDEX_DTYPE_BYTES
                ring = cls.ring_capacity_for(spec.index_ratio, getattr(config, "decode_rows_per_req", 1) - 1)
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
        assert self._rope_positions is not None, "rope positions are only kept on mrope models"
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

    # --- P2c cold tier: host copies of whole pages (KV_HOST_TIER_DESIGN.md 3.1) ---
    # Same interface as the GDN host bank (linear_state_pool: alloc_host / free_host / to_host /
    # from_host / fence_host_copies). A host page holds every per-page tensor of one device page:
    # the K/V codes, their scales, the page's slab rows (slot // index_ratio) and, on mrope models,
    # its rope rows. The pending ring and the scratch rows are per request and never travel.
    # v1 copies synchronously; the async side stream and the pinned arena come with the CUDA path.

    def attach_host_pages(self, num_host_pages: int, *, pin_memory: bool = False) -> None:
        n = max(0, int(num_host_pages))
        self._num_host_pages = n
        self._free_host_pages: list[int] = list(range(n))
        self._host_pages: dict[str, torch.Tensor] = {}
        # A copier with the GDN host bank's interface (to_host / from_host / fence over (dst, src)
        # views); None copies at once. TODO(T12): the pinned arena, a side stream and batched copies.
        self.host_copies = None
        if not n:
            return
        for name, t in self._page_tensors().items():
            shape = (n,) + tuple(t.shape[1:])
            self._host_pages[name] = torch.zeros(shape, dtype=t.dtype, device="cpu", pin_memory=pin_memory)

    def _page_tensors(self) -> dict[str, torch.Tensor]:
        """Device tensors with the page id as dim 0 (views; no copy)."""
        _, layers, pages, ps, heads, _elem = self._kv_buffer.shape
        out = {"kv": self._kv_buffer.movedim(2, 0)}
        if self._scale_buffer is not None:
            sd = self._scale_buffer.shape[-1]
            out["scale"] = self._scale_buffer.view(2, layers, pages, ps, heads, sd).movedim(2, 0)
        rows = ps // self._index_ratio
        out["slab"] = self._cmp_k_buffer[:, : self._cmp_scratch_base].view(
            self._num_index_layers, pages, rows, self._index_head_dim
        ).movedim(1, 0)
        if self._rope_positions is not None:
            out["rope"] = self._rope_positions.view(pages, ps, 3)
        return out

    @property
    def num_host_pages(self) -> int:
        return getattr(self, "_num_host_pages", 0)

    @property
    def num_free_host_pages(self) -> int:
        return len(getattr(self, "_free_host_pages", ()))

    def host_bytes_per_page(self) -> int:
        host = getattr(self, "_host_pages", {})
        return sum(t[0].numel() * t.element_size() for t in host.values())

    def alloc_host_pages(self, n: int = 1) -> list[int]:
        if n > self.num_free_host_pages:
            raise RuntimeError(f"KV host tier exhausted: need {n}, have {self.num_free_host_pages}")
        return [self._free_host_pages.pop() for _ in range(n)]

    def free_host_pages(self, host_pages: Sequence[int]) -> None:
        self._free_host_pages.extend(int(h) for h in host_pages)

    def reclaim_all_host_pages(self) -> None:
        self._free_host_pages = list(range(self.num_host_pages))

    @staticmethod
    def _check_pairs(a: Sequence[int], b: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor]:
        if len(a) != len(b):
            raise ValueError(f"page lists differ in length: {len(a)} and {len(b)}")
        return torch.as_tensor(list(a), dtype=torch.long), torch.as_tensor(list(b), dtype=torch.long)

    def pages_to_host(self, pages: Sequence[int], host_pages: Sequence[int]) -> None:
        """Copy device pages ``pages[i]`` into host pages ``host_pages[i]``."""
        src, dst = self._check_pairs(pages, host_pages)
        if self.host_copies is not None:
            self.host_copies.to_host(self._page_pairs(self._host_pages, dst, self._page_tensors(), src))
            return
        for name, dev in self._page_tensors().items():
            self._host_pages[name][dst] = dev[src.to(dev.device)].to("cpu")

    def pages_from_host(self, host_pages: Sequence[int], pages: Sequence[int]) -> None:
        """Copy host pages ``host_pages[i]`` into device pages ``pages[i]``."""
        src, dst = self._check_pairs(host_pages, pages)
        if self.host_copies is not None:
            self.host_copies.from_host(self._page_pairs(self._page_tensors(), dst, self._host_pages, src))
            return
        for name, dev in self._page_tensors().items():
            dev[dst.to(dev.device)] = self._host_pages[name][src].to(dev.device)

    @staticmethod
    def _page_pairs(dst_tensors, dst, src_tensors, src) -> list[tuple[torch.Tensor, torch.Tensor]]:
        # views, not gathers: a copier that copies later must read the page as it is then
        return [(dst_tensors[name][d], src_tensors[name][s])
                for name in src_tensors for d, s in zip(dst.tolist(), src.tolist())]

    def fence_host_copies(self) -> None:
        """Order a freed page's next write after its host copy (the copier's fence; at once: no-op)."""
        if self.host_copies is not None:
            self.host_copies.fence()


__all__ = ["QSAKVCache"]
