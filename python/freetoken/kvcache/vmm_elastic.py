"""Dynamic KV-to-expert VMM page lending (ElasticKvMoePool).

Instead of statically partitioning GPU HBM between a large peak KV pool (e.g. 1.97M tokens
on Qwen3.8-Flash-Next) and a truncated MoE expert slot cache (65% residency), ``ElasticKvMoePool``
reserves virtual address (VA) ranges for both the full peak KV capacity and an expanded MoE
slot cache (up to 100% expert residency), backed by a fixed pool of 2 MiB CUDA Driver VMM
physical allocations (``cuMemCreate`` / ``cuMemMap`` / ``cuMemUnmap``):

1. At startup, KV blocks above ``--kv-elastic-floor-tokens`` (default 131,072 tokens) lend
   their 2 MiB physical slabs to ``OffloadMoeCache`` tail slots, lifting steady-state MoE
   residency from ~65% to >90% with zero extra physical VRAM.
2. Under KV pressure, ``CacheManager`` ALWAYS demotes unlocked prefixes to the ``P2c`` host
   RAM tier first (``evict_full``). Idle cached prefixes never evict resident MoE experts.
3. Only when active, locked running requests exceed currently mapped KV pages does
   ``ElasticKvMoePool.borrow_kv_pages`` retire tail MoE slots (marking ``usage = RETIRED_SLOT_USAGE``
   with zero CUDA-graph recapture), unmap their 2 MiB slabs, and map fresh KV page blocks.
4. When active KV usage drops back below a hysteresis band, ``return_free_kv_blocks`` unmaps
   free elastic KV blocks and restores MoE tail slots automatically.
"""

from __future__ import annotations

import ctypes
import math
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import torch

RETIRED_SLOT_USAGE = 1 << 62
_ELASTIC_MOE_MIN_ROW_BYTES = 16384

_CU_MEM_ALLOCATION_TYPE_PINNED = 1
_CU_MEM_LOCATION_TYPE_DEVICE = 1
_CU_MEM_ALLOC_GRANULARITY_RECOMMENDED = 1
_CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 3

_TORCH_TO_TYPESTR: dict[torch.dtype, str] = {
    torch.uint8: "|u1",
    torch.int8: "|i1",
    torch.float8_e4m3fn: "|u1",
    torch.int16: "<i2",
    torch.float16: "<f2",
    torch.bfloat16: "<u2",
    torch.int32: "<i4",
    torch.float32: "<f4",
    torch.int64: "<i8",
}


def _div_ceil(a: int, b: int) -> int:
    return (a + b - 1) // b


def _align_up(n: int, align: int) -> int:
    return _div_ceil(n, align) * align


def kv_moe_elastic_enabled(flag: bool = False) -> bool:
    if flag:
        return True
    raw = os.environ.get("FREETOKEN_KV_MOE_ELASTIC", "").strip().lower()
    return raw in ("1", "true", "yes", "on")


class VmmBackend(Protocol):
    granularity: int

    def reserve_va(self, size_bytes: int, align_bytes: int = 0) -> int: ...

    def free_va(self, va_ptr: int, size_bytes: int) -> None: ...

    def create_slab(self) -> int: ...

    def release_slab(self, handle: int) -> None: ...

    def map_slab(self, va_ptr: int, handle: int) -> None: ...

    def unmap_slab(self, va_ptr: int) -> None: ...

    def zero_range(self, va_ptr: int, size_bytes: int) -> None: ...

    def wrap_tensor(
        self, va_ptr: int, shape: tuple[int, ...], dtype: torch.dtype
    ) -> torch.Tensor: ...

    def synchronize(self) -> None: ...


class _CUmemLocation(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class _CUmemAllocationProp(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("requestedHandleTypes", ctypes.c_int),
        ("location", _CUmemLocation),
        ("win32HandleMetaData", ctypes.c_void_p),
        ("allocFlags", ctypes.c_ubyte * 64),
    ]


class _CUmemAccessDesc(ctypes.Structure):
    _fields_ = [
        ("location", _CUmemLocation),
        ("flags", ctypes.c_int),
    ]


class _CudaArrayView:
    def __init__(self, ptr: int, shape: tuple[int, ...], typestr: str) -> None:
        self.__cuda_array_interface__ = {
            "shape": shape,
            "typestr": typestr,
            "data": (int(ptr), False),
            "version": 3,
        }


class CudaDriverVmmBackend:
    """CUDA Driver API VMM backend over ``libcuda.so.1``."""

    def __init__(self, device: torch.device) -> None:
        if device.type != "cuda":
            raise ValueError(f"CudaDriverVmmBackend requires CUDA device, got {device}")
        self.device = device
        self.device_index = 0 if device.index is None else int(device.index)
        torch.cuda.init()
        lib = ctypes.CDLL("libcuda.so.1")
        lib.cuMemGetAllocationGranularity.argtypes = [
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(_CUmemAllocationProp),
            ctypes.c_int,
        ]
        lib.cuMemGetAllocationGranularity.restype = ctypes.c_int
        lib.cuGetErrorString.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_char_p),
        ]
        lib.cuGetErrorString.restype = ctypes.c_int
        lib.cuMemAddressReserve.argtypes = [
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_uint64,
            ctypes.c_uint64,
        ]
        lib.cuMemAddressReserve.restype = ctypes.c_int
        lib.cuMemAddressFree.argtypes = [ctypes.c_uint64, ctypes.c_size_t]
        lib.cuMemAddressFree.restype = ctypes.c_int
        lib.cuMemCreate.argtypes = [
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_size_t,
            ctypes.POINTER(_CUmemAllocationProp),
            ctypes.c_uint64,
        ]
        lib.cuMemCreate.restype = ctypes.c_int
        lib.cuMemRelease.argtypes = [ctypes.c_uint64]
        lib.cuMemRelease.restype = ctypes.c_int
        lib.cuMemMap.argtypes = [
            ctypes.c_uint64,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_uint64,
            ctypes.c_uint64,
        ]
        lib.cuMemMap.restype = ctypes.c_int
        lib.cuMemSetAccess.argtypes = [
            ctypes.c_uint64,
            ctypes.c_size_t,
            ctypes.POINTER(_CUmemAccessDesc),
            ctypes.c_size_t,
        ]
        lib.cuMemSetAccess.restype = ctypes.c_int
        lib.cuMemUnmap.argtypes = [ctypes.c_uint64, ctypes.c_size_t]
        lib.cuMemUnmap.restype = ctypes.c_int
        self._lib = lib

        self._prop = _CUmemAllocationProp()
        self._prop.type = _CU_MEM_ALLOCATION_TYPE_PINNED
        self._prop.requestedHandleTypes = 0
        self._prop.location.type = _CU_MEM_LOCATION_TYPE_DEVICE
        self._prop.location.id = self.device_index

        self._access = _CUmemAccessDesc()
        self._access.location.type = _CU_MEM_LOCATION_TYPE_DEVICE
        self._access.location.id = self.device_index
        self._access.flags = _CU_MEM_ACCESS_FLAGS_PROT_READWRITE

        gran = ctypes.c_size_t(0)
        self._check(
            self._lib.cuMemGetAllocationGranularity(
                ctypes.byref(gran),
                ctypes.byref(self._prop),
                ctypes.c_int(_CU_MEM_ALLOC_GRANULARITY_RECOMMENDED),
            ),
            "cuMemGetAllocationGranularity",
        )
        self.granularity = int(gran.value)

    def _check(self, status: int, fn_name: str) -> None:
        if status != 0:
            err_str = ctypes.c_char_p()
            self._lib.cuGetErrorString(ctypes.c_int(status), ctypes.byref(err_str))
            detail = (
                err_str.value.decode("utf-8", errors="replace")
                if err_str.value
                else str(status)
            )
            raise RuntimeError(f"{fn_name} failed with CUDA error {status}: {detail}")

    def reserve_va(self, size_bytes: int, align_bytes: int = 0) -> int:
        ptr = ctypes.c_uint64(0)
        align = align_bytes or self.granularity
        self._check(
            self._lib.cuMemAddressReserve(
                ctypes.byref(ptr),
                ctypes.c_size_t(size_bytes),
                ctypes.c_size_t(align),
                ctypes.c_uint64(0),
                ctypes.c_uint64(0),
            ),
            "cuMemAddressReserve",
        )
        return int(ptr.value)

    def free_va(self, va_ptr: int, size_bytes: int) -> None:
        self._check(
            self._lib.cuMemAddressFree(
                ctypes.c_uint64(va_ptr), ctypes.c_size_t(size_bytes)
            ),
            "cuMemAddressFree",
        )

    def create_slab(self) -> int:
        handle = ctypes.c_uint64(0)
        self._check(
            self._lib.cuMemCreate(
                ctypes.byref(handle),
                ctypes.c_size_t(self.granularity),
                ctypes.byref(self._prop),
                ctypes.c_uint64(0),
            ),
            "cuMemCreate",
        )
        return int(handle.value)

    def release_slab(self, handle: int) -> None:
        self._check(self._lib.cuMemRelease(ctypes.c_uint64(handle)), "cuMemRelease")

    def map_slab(self, va_ptr: int, handle: int) -> None:
        self._check(
            self._lib.cuMemMap(
                ctypes.c_uint64(va_ptr),
                ctypes.c_size_t(self.granularity),
                ctypes.c_size_t(0),
                ctypes.c_uint64(handle),
                ctypes.c_uint64(0),
            ),
            "cuMemMap",
        )
        self._check(
            self._lib.cuMemSetAccess(
                ctypes.c_uint64(va_ptr),
                ctypes.c_size_t(self.granularity),
                ctypes.byref(self._access),
                ctypes.c_size_t(1),
            ),
            "cuMemSetAccess",
        )

    def unmap_slab(self, va_ptr: int) -> None:
        self._check(
            self._lib.cuMemUnmap(
                ctypes.c_uint64(va_ptr), ctypes.c_size_t(self.granularity)
            ),
            "cuMemUnmap",
        )

    def zero_range(self, va_ptr: int, size_bytes: int) -> None:
        if size_bytes <= 0:
            return
        view = torch.as_tensor(
            _CudaArrayView(va_ptr, (size_bytes,), "|u1"), device=self.device
        )
        view.zero_()

    def wrap_tensor(
        self, va_ptr: int, shape: tuple[int, ...], dtype: torch.dtype
    ) -> torch.Tensor:
        if dtype == torch.bfloat16 or dtype == torch.float8_e4m3fn:
            raw_typestr = _TORCH_TO_TYPESTR[dtype]
            raw = torch.as_tensor(
                _CudaArrayView(va_ptr, shape, raw_typestr), device=self.device
            )
            return raw.view(dtype)
        typestr = _TORCH_TO_TYPESTR[dtype]
        return torch.as_tensor(
            _CudaArrayView(va_ptr, shape, typestr), device=self.device
        )

    def synchronize(self) -> None:
        torch.cuda.synchronize(self.device)


class MockCpuVmmBackend:
    """Deterministic CPU VMM simulator enforcing exact granule alignment and unmapped poison."""

    def __init__(self, granularity: int = 2 << 20) -> None:
        if granularity <= 0 or granularity % 16 != 0:
            raise ValueError(f"invalid mock VMM granularity {granularity}")
        self.granularity = granularity
        self._next_va = 0x1000_0000_0000
        self._next_handle = 1
        self._reservations: dict[int, tuple[int, torch.Tensor]] = {}
        self._slabs: dict[int, bool] = {}  # handle -> is_currently_mapped
        self._mapped: dict[int, int] = {}  # granule_va -> handle
        self.sync_calls = 0

    @property
    def num_created_slabs(self) -> int:
        return len(self._slabs)

    @property
    def num_mapped_slabs(self) -> int:
        return len(self._mapped)

    def _find_reservation(
        self, va_ptr: int, size_bytes: int
    ) -> tuple[int, torch.Tensor]:
        for base, (total, buf) in self._reservations.items():
            if base <= va_ptr and va_ptr + size_bytes <= base + total:
                return base, buf
        raise KeyError(
            f"VA range [{va_ptr:#x}, +{size_bytes}) outside active reservations"
        )

    def reserve_va(self, size_bytes: int, align_bytes: int = 0) -> int:
        if size_bytes <= 0 or size_bytes % self.granularity != 0:
            raise ValueError(f"VA size {size_bytes} not aligned to {self.granularity}")
        align = max(align_bytes, self.granularity)
        va = _align_up(self._next_va, align)
        self._next_va = va + size_bytes + self.granularity
        # Poison unmapped VA bytes with 0xDD so any illegal read on CPU is immediately visible.
        buf = torch.full((size_bytes,), 0xDD, dtype=torch.uint8)
        self._reservations[va] = (size_bytes, buf)
        return va

    def free_va(self, va_ptr: int, size_bytes: int) -> None:
        total, _ = self._reservations.get(va_ptr, (0, None))
        if total != size_bytes:
            raise ValueError(
                f"free_va mismatch at {va_ptr:#x}: expected {total}, got {size_bytes}"
            )
        for off in range(0, size_bytes, self.granularity):
            if (va_ptr + off) in self._mapped:
                raise RuntimeError(f"free_va on still-mapped granule {va_ptr + off:#x}")
        del self._reservations[va_ptr]

    def create_slab(self) -> int:
        h = self._next_handle
        self._next_handle += 1
        self._slabs[h] = False
        return h

    def release_slab(self, handle: int) -> None:
        if handle not in self._slabs:
            raise KeyError(f"unknown slab handle {handle}")
        if self._slabs[handle]:
            raise RuntimeError(f"cannot release mapped slab {handle}")
        del self._slabs[handle]

    def map_slab(self, va_ptr: int, handle: int) -> None:
        if va_ptr % self.granularity != 0:
            raise ValueError(f"unaligned map_slab va_ptr {va_ptr:#x}")
        if va_ptr in self._mapped:
            raise RuntimeError(f"granule {va_ptr:#x} already mapped")
        if self._slabs.get(handle) is not False:
            raise RuntimeError(f"slab {handle} invalid or already mapped")
        base, buf = self._find_reservation(va_ptr, self.granularity)
        off = va_ptr - base
        buf[off : off + self.granularity].zero_()
        self._mapped[va_ptr] = handle
        self._slabs[handle] = True

    def unmap_slab(self, va_ptr: int) -> None:
        handle = self._mapped.pop(va_ptr, None)
        if handle is None:
            raise RuntimeError(f"unmap_slab on unmapped granule {va_ptr:#x}")
        self._slabs[handle] = False
        base, buf = self._find_reservation(va_ptr, self.granularity)
        off = va_ptr - base
        buf[off : off + self.granularity].fill_(0xDD)

    def zero_range(self, va_ptr: int, size_bytes: int) -> None:
        if size_bytes <= 0:
            return
        for off in range(0, size_bytes, self.granularity):
            if (va_ptr + off) not in self._mapped:
                raise RuntimeError(
                    f"zero_range touched unmapped granule {va_ptr + off:#x}"
                )
        base, buf = self._find_reservation(va_ptr, size_bytes)
        off = va_ptr - base
        buf[off : off + size_bytes].zero_()

    def wrap_tensor(
        self, va_ptr: int, shape: tuple[int, ...], dtype: torch.dtype
    ) -> torch.Tensor:
        nbytes = math.prod(shape) * dtype.itemsize
        base, buf = self._find_reservation(va_ptr, nbytes)
        off = va_ptr - base
        return buf[off : off + nbytes].view(dtype).view(shape)

    def synchronize(self) -> None:
        self.sync_calls += 1


@dataclass(frozen=True)
class ElasticGeometry:
    granularity: int
    num_kv_planes: int
    plane_page_bytes: int
    usable_kv_pages: int
    va_kv_pages: int
    kv_quantum_pages: int
    kv_slabs_per_plane_block: int
    kv_slabs_per_block: int
    floor_kv_blocks: int
    elastic_end_kv_block: int
    elastic_moe_banks: tuple[str, ...]
    moe_quantum_slots: int
    moe_slabs_per_bank_block: dict[str, int]
    moe_slabs_per_block: int
    floor_moe_blocks: int
    va_moe_blocks: int
    va_moe_slots: int

    @property
    def num_elastic_kv_blocks(self) -> int:
        return max(0, self.elastic_end_kv_block - self.floor_kv_blocks)

    @property
    def max_extra_moe_blocks(self) -> int:
        return max(0, self.va_moe_blocks - self.floor_moe_blocks)


def compute_elastic_geometry(
    *,
    granularity: int,
    num_kv_planes: int,
    plane_page_bytes: int,
    usable_kv_pages: int,
    floor_kv_pages: int,
    moe_row_bytes: dict[str, int],
    base_moe_slots: int,
    max_moe_slots: int,
    num_experts: int,
    prefill_overlap: bool,
    strict_budget: bool = False,
) -> ElasticGeometry:
    kv_quantum_pages = math.lcm(plane_page_bytes, granularity) // plane_page_bytes
    kv_plane_quantum_bytes = kv_quantum_pages * plane_page_bytes
    kv_slabs_per_plane_block = kv_plane_quantum_bytes // granularity
    kv_slabs_per_block = num_kv_planes * kv_slabs_per_plane_block

    elastic_moe_banks = tuple(
        name for name, rb in moe_row_bytes.items() if rb >= _ELASTIC_MOE_MIN_ROW_BYTES
    )
    if not elastic_moe_banks:
        elastic_moe_banks = tuple(name for name, rb in moe_row_bytes.items() if rb > 0)
    moe_quantum_slots = math.lcm(
        *(
            granularity // math.gcd(granularity, moe_row_bytes[n])
            for n in elastic_moe_banks
        )
    )
    moe_slabs_per_bank_block = {
        n: (moe_quantum_slots * moe_row_bytes[n]) // granularity
        for n in elastic_moe_banks
    }
    moe_slabs_per_block = sum(moe_slabs_per_bank_block.values())

    hard_min_moe_slots = 2 * num_experts if prefill_overlap else num_experts
    min_moe_slots = max(base_moe_slots, hard_min_moe_slots)
    max_cap_blocks = max_moe_slots // moe_quantum_slots
    hard_min_moe_blocks = min(
        max_cap_blocks, _div_ceil(hard_min_moe_slots, moe_quantum_slots)
    )
    floor_moe_blocks = min(max_cap_blocks, _div_ceil(min_moe_slots, moe_quantum_slots))

    if strict_budget:
        budget_bytes = (
            usable_kv_pages + 1
        ) * num_kv_planes * plane_page_bytes + base_moe_slots * sum(
            moe_row_bytes[n] for n in elastic_moe_banks
        )
        budget_slabs = budget_bytes // granularity
        total_kv_blocks = (
            _align_up(usable_kv_pages + 1, kv_quantum_pages) // kv_quantum_pages
        )
        min_kv_blocks = max(2, _div_ceil(max(1, floor_kv_pages) + 1, kv_quantum_pages))
        # Drop an unaligned dummy-tail KV block spillover before trimming base MoE blocks.
        if (
            total_kv_blocks * kv_slabs_per_block
            + floor_moe_blocks * moe_slabs_per_block
            > budget_slabs
            and (usable_kv_pages + 1) % kv_quantum_pages != 0
            and total_kv_blocks > min_kv_blocks
        ):
            total_kv_blocks -= 1
            usable_kv_pages = total_kv_blocks * kv_quantum_pages - 1
        while (
            floor_moe_blocks > hard_min_moe_blocks
            and total_kv_blocks * kv_slabs_per_block
            + floor_moe_blocks * moe_slabs_per_block
            > budget_slabs
        ):
            floor_moe_blocks -= 1
        while (
            total_kv_blocks > min_kv_blocks
            and total_kv_blocks * kv_slabs_per_block
            + floor_moe_blocks * moe_slabs_per_block
            > budget_slabs
        ):
            total_kv_blocks -= 1
            usable_kv_pages = min(
                usable_kv_pages, total_kv_blocks * kv_quantum_pages - 1
            )

    va_kv_pages = _align_up(usable_kv_pages + 1, kv_quantum_pages)
    elastic_end_kv_block = usable_kv_pages // kv_quantum_pages
    floor_kv_blocks = min(
        elastic_end_kv_block,
        max(1, _div_ceil(max(1, floor_kv_pages), kv_quantum_pages)),
    )
    num_elastic_kv_blocks = max(0, elastic_end_kv_block - floor_kv_blocks)

    lent_slabs = num_elastic_kv_blocks * kv_slabs_per_block
    extra_moe_blocks = min(
        max(0, max_cap_blocks - floor_moe_blocks),
        lent_slabs // moe_slabs_per_block if moe_slabs_per_block > 0 else 0,
    )
    va_moe_blocks = floor_moe_blocks + extra_moe_blocks
    effective_min_moe = (
        max(hard_min_moe_slots, floor_moe_blocks * moe_quantum_slots)
        if strict_budget
        else min_moe_slots
    )
    va_moe_slots = max(effective_min_moe, va_moe_blocks * moe_quantum_slots)
    va_moe_slots = min(max_moe_slots, va_moe_slots)
    if va_moe_blocks * moe_quantum_slots >= hard_min_moe_slots:
        va_moe_slots = min(va_moe_slots, va_moe_blocks * moe_quantum_slots)
    else:
        va_moe_blocks = _div_ceil(va_moe_slots, moe_quantum_slots)

    return ElasticGeometry(
        granularity=granularity,
        num_kv_planes=num_kv_planes,
        plane_page_bytes=plane_page_bytes,
        usable_kv_pages=usable_kv_pages,
        va_kv_pages=va_kv_pages,
        kv_quantum_pages=kv_quantum_pages,
        kv_slabs_per_plane_block=kv_slabs_per_plane_block,
        kv_slabs_per_block=kv_slabs_per_block,
        floor_kv_blocks=floor_kv_blocks,
        elastic_end_kv_block=elastic_end_kv_block,
        elastic_moe_banks=elastic_moe_banks,
        moe_quantum_slots=moe_quantum_slots,
        moe_slabs_per_bank_block=moe_slabs_per_bank_block,
        moe_slabs_per_block=moe_slabs_per_block,
        floor_moe_blocks=floor_moe_blocks,
        va_moe_blocks=va_moe_blocks,
        va_moe_slots=va_moe_slots,
    )


class ElasticKvMoePool:
    """Shared physical VMM slab pool dynamically backing QSA KV pages and MoE expert slots."""

    def __init__(
        self,
        geometry: ElasticGeometry,
        *,
        device: torch.device,
        backend: VmmBackend | None = None,
        hysteresis_blocks: int = 2,
    ) -> None:
        self.geo = geometry
        self.device = device
        self.backend: VmmBackend = backend or CudaDriverVmmBackend(device)
        self.kv_quantum_pages = geometry.kv_quantum_pages
        self.hysteresis_pages = max(1, hysteresis_blocks) * geometry.kv_quantum_pages
        self.moe_cache = None

        g = geometry.granularity
        total_kv_blocks = geometry.va_kv_pages // geometry.kv_quantum_pages
        self._total_kv_blocks = total_kv_blocks
        self._kv_plane_bytes = total_kv_blocks * geometry.kv_slabs_per_plane_block * g
        self._kv_total_va_bytes = 0
        self._kv_va = 0
        self._moe_vas: dict[str, tuple[int, int]] = {}
        self._all_slabs: list[int] = []
        self._free_slabs: list[int] = []
        self._va_to_handle: dict[int, int] = {}
        self._kv_block_mapped: list[bool] = [False] * total_kv_blocks
        self._moe_block_mapped: list[bool] = [False] * geometry.va_moe_blocks
        self._active_moe_blocks = 0

        try:
            kv_va_bytes = geometry.num_kv_planes * self._kv_plane_bytes
            self._kv_va = self.backend.reserve_va(kv_va_bytes, g)
            self._kv_total_va_bytes = kv_va_bytes

            for bank_name in geometry.elastic_moe_banks:
                bank_bytes = (
                    geometry.va_moe_blocks
                    * geometry.moe_slabs_per_bank_block[bank_name]
                    * g
                )
                va = self.backend.reserve_va(bank_bytes, g)
                self._moe_vas[bank_name] = (va, bank_bytes)

            fixed_kv_blocks = geometry.floor_kv_blocks + (
                total_kv_blocks - geometry.elastic_end_kv_block
            )
            total_slabs = (
                (fixed_kv_blocks + geometry.num_elastic_kv_blocks)
                * geometry.kv_slabs_per_block
                + geometry.floor_moe_blocks * geometry.moe_slabs_per_block
            )
            for _ in range(total_slabs):
                h = self.backend.create_slab()
                self._all_slabs.append(h)
                self._free_slabs.append(h)

            # Map permanent KV blocks: [0 : floor_kv_blocks) and [elastic_end_kv_block : total_kv_blocks).
            for b in range(geometry.floor_kv_blocks):
                self._map_kv_block(b)
            for b in range(geometry.elastic_end_kv_block, total_kv_blocks):
                self._map_kv_block(b)

            # Map permanent + extra lent MoE blocks: [0 : va_moe_blocks).
            for mb in range(geometry.va_moe_blocks):
                if len(self._free_slabs) < geometry.moe_slabs_per_block:
                    break
                self._map_moe_block(mb)
                self._active_moe_blocks = mb + 1
        except BaseException:
            self.close()
            raise

    def _kv_block_va(self, plane: int, block: int) -> int:
        block_bytes = self.geo.kv_slabs_per_plane_block * self.geo.granularity
        return self._kv_va + plane * self._kv_plane_bytes + block * block_bytes

    def _map_kv_block(self, block: int) -> None:
        assert not self._kv_block_mapped[block]
        g = self.geo.granularity
        slabs_per_plane = self.geo.kv_slabs_per_plane_block
        block_bytes = slabs_per_plane * g
        for p in range(self.geo.num_kv_planes):
            base_va = self._kv_block_va(p, block)
            for s in range(slabs_per_plane):
                va = base_va + s * g
                h = self._free_slabs.pop()
                self.backend.map_slab(va, h)
                self._va_to_handle[va] = h
            self.backend.zero_range(base_va, block_bytes)
        self._kv_block_mapped[block] = True

    def _unmap_kv_block(self, block: int) -> None:
        assert self._kv_block_mapped[block]
        g = self.geo.granularity
        slabs_per_plane = self.geo.kv_slabs_per_plane_block
        for p in range(self.geo.num_kv_planes):
            base_va = self._kv_block_va(p, block)
            for s in range(slabs_per_plane):
                va = base_va + s * g
                # Recover slab handle via tracking dict before unmapping.
                handle = self._va_to_handle.pop(va)
                self.backend.unmap_slab(va)
                self._free_slabs.append(handle)
        self._kv_block_mapped[block] = False

    def _map_moe_block(self, mblock: int) -> None:
        assert not self._moe_block_mapped[mblock]
        g = self.geo.granularity
        for bank_name in self.geo.elastic_moe_banks:
            va_base, _ = self._moe_vas[bank_name]
            slabs = self.geo.moe_slabs_per_bank_block[bank_name]
            b_bytes = slabs * g
            start_va = va_base + mblock * b_bytes
            for s in range(slabs):
                va = start_va + s * g
                h = self._free_slabs.pop()
                self.backend.map_slab(va, h)
                self._va_to_handle[va] = h
            self.backend.zero_range(start_va, b_bytes)
        self._moe_block_mapped[mblock] = True

    def _unmap_moe_block(self, mblock: int) -> None:
        assert self._moe_block_mapped[mblock]
        g = self.geo.granularity
        for bank_name in self.geo.elastic_moe_banks:
            va_base, _ = self._moe_vas[bank_name]
            slabs = self.geo.moe_slabs_per_bank_block[bank_name]
            b_bytes = slabs * g
            start_va = va_base + mblock * b_bytes
            for s in range(slabs):
                va = start_va + s * g
                h = self._va_to_handle.pop(va)
                self.backend.unmap_slab(va)
                self._free_slabs.append(h)
        self._moe_block_mapped[mblock] = False

    @property
    def unmapped_kv_blocks(self) -> int:
        return sum(
            not self._kv_block_mapped[b]
            for b in range(self.geo.floor_kv_blocks, self.geo.elastic_end_kv_block)
        )

    @property
    def borrowed_kv_blocks(self) -> int:
        return self.geo.num_elastic_kv_blocks - self.unmapped_kv_blocks

    @property
    def unmapped_kv_pages(self) -> int:
        return self.unmapped_kv_blocks * self.geo.kv_quantum_pages

    @property
    def mapped_usable_kv_pages(self) -> int:
        return self.geo.usable_kv_pages - self.unmapped_kv_pages

    @property
    def active_moe_slots(self) -> int:
        if self._active_moe_blocks == self.geo.va_moe_blocks:
            return self.geo.va_moe_slots
        return min(
            self.geo.va_moe_slots, self._active_moe_blocks * self.geo.moe_quantum_slots
        )

    def bind_moe_cache(self, moe_cache) -> None:
        self.moe_cache = moe_cache
        moe_cache.set_active_slots(self.active_moe_slots)

    def wrap_kv_buffer(
        self,
        num_storage_layers: int,
        page_size: int,
        local_kv_heads: int,
        elem_dim: int,
        elem_dtype: torch.dtype,
    ) -> torch.Tensor:
        shape = (
            2,
            num_storage_layers,
            self.geo.va_kv_pages,
            page_size,
            local_kv_heads,
            elem_dim,
        )
        return self.backend.wrap_tensor(self._kv_va, shape, elem_dtype)

    def make_moe_bank_allocator(
        self,
    ) -> Callable[[str, tuple[int, ...], torch.dtype, torch.device], torch.Tensor]:
        def _alloc(
            name: str, shape: tuple[int, ...], dtype: torch.dtype, device: torch.device
        ) -> torch.Tensor:
            from freetoken.moe.legacy_format import canonical_role

            role = canonical_role(name)
            match = (
                name
                if name in self._moe_vas
                else (role if role in self._moe_vas else None)
            )
            if match is not None:
                va_base, _ = self._moe_vas[match]
                return self.backend.wrap_tensor(va_base, shape, dtype)
            return torch.zeros(shape, dtype=dtype, device=device)

        return _alloc

    def initial_free_kv_slots(
        self, usable_kv_pages: int, page_size: int, device: torch.device
    ) -> torch.Tensor:
        pages: list[int] = []
        qp = self.geo.kv_quantum_pages
        for b in range(self._total_kv_blocks):
            if not self._kv_block_mapped[b]:
                continue
            lo = b * qp
            hi = min(usable_kv_pages, (b + 1) * qp)
            if hi > lo:
                pages.extend(range(lo, hi))
        return torch.tensor(pages, dtype=torch.int32, device=device) * page_size

    def borrow_kv_pages(
        self, needed_pages: int, *, device: torch.device, page_size: int
    ) -> torch.Tensor:
        if needed_pages <= 0 or self.unmapped_kv_blocks == 0:
            return torch.empty(0, dtype=torch.int32, device=device)
        blocks_wanted = min(
            self.unmapped_kv_blocks, _div_ceil(needed_pages, self.geo.kv_quantum_pages)
        )
        slabs_needed = blocks_wanted * self.geo.kv_slabs_per_block

        # Retire MoE tail blocks if free slab pool is short.
        if len(self._free_slabs) < slabs_needed:
            deficit = slabs_needed - len(self._free_slabs)
            retire_blocks = _div_ceil(deficit, self.geo.moe_slabs_per_block)
            new_active_blocks = max(
                self.geo.floor_moe_blocks, self._active_moe_blocks - retire_blocks
            )
            if new_active_blocks < self._active_moe_blocks:
                new_slots = new_active_blocks * self.geo.moe_quantum_slots
                # Drain any in-flight forward MoE kernels before retiring slots, then drain
                # the retirement kernel before unmapping physical slabs.
                self.backend.synchronize()
                if self.moe_cache is not None:
                    self.moe_cache.set_active_slots(new_slots)
                    self.backend.synchronize()
                for mb in range(self._active_moe_blocks - 1, new_active_blocks - 1, -1):
                    self._unmap_moe_block(mb)
                self._active_moe_blocks = new_active_blocks

        qp = self.geo.kv_quantum_pages
        borrowed_pages: list[int] = []
        mapped_now = 0
        for b in range(self.geo.floor_kv_blocks, self.geo.elastic_end_kv_block):
            if mapped_now >= blocks_wanted:
                break
            if self._kv_block_mapped[b]:
                continue
            if len(self._free_slabs) < self.geo.kv_slabs_per_block:
                break
            self._map_kv_block(b)
            borrowed_pages.extend(range(b * qp, (b + 1) * qp))
            mapped_now += 1
        return (
            torch.tensor(borrowed_pages, dtype=torch.int32, device=device) * page_size
        )

    def return_free_kv_blocks(
        self, free_slots: torch.Tensor, *, page_size: int
    ) -> torch.Tensor:
        if self.borrowed_kv_blocks == 0:
            return free_slots
        qp = self.geo.kv_quantum_pages
        if len(free_slots) < qp + self.hysteresis_pages:
            return free_slots

        free_pages = set((free_slots // page_size).tolist())
        blocks_to_unmap: list[int] = []
        for b in range(
            self.geo.elastic_end_kv_block - 1, self.geo.floor_kv_blocks - 1, -1
        ):
            if not self._kv_block_mapped[b]:
                continue
            if len(free_pages) - qp < self.hysteresis_pages:
                break
            span = range(b * qp, (b + 1) * qp)
            if all(p in free_pages for p in span):
                for p in span:
                    free_pages.remove(p)
                blocks_to_unmap.append(b)

        if not blocks_to_unmap:
            return free_slots

        self.backend.synchronize()
        for b in blocks_to_unmap:
            self._unmap_kv_block(b)

        # Restore as many MoE tail blocks as free slabs now fund.
        grew_moe = False
        while (
            self._active_moe_blocks < self.geo.va_moe_blocks
            and len(self._free_slabs) >= self.geo.moe_slabs_per_block
        ):
            self._map_moe_block(self._active_moe_blocks)
            self._active_moe_blocks += 1
            grew_moe = True
        if grew_moe and self.moe_cache is not None:
            self.moe_cache.set_active_slots(self.active_moe_slots)

        ordered = sorted(free_pages)
        return (
            torch.tensor(ordered, dtype=torch.int32, device=free_slots.device)
            * page_size
        )

    def close(self) -> None:
        self.backend.synchronize()
        for mb in range(self.geo.va_moe_blocks - 1, -1, -1):
            if self._moe_block_mapped[mb]:
                self._unmap_moe_block(mb)
        for b in range(self._total_kv_blocks - 1, -1, -1):
            if self._kv_block_mapped[b]:
                self._unmap_kv_block(b)
        while self._free_slabs:
            self.backend.release_slab(self._free_slabs.pop())
        self._all_slabs.clear()
        for va, sz in self._moe_vas.values():
            self.backend.free_va(va, sz)
        self._moe_vas.clear()
        if self._kv_total_va_bytes > 0:
            self.backend.free_va(self._kv_va, self._kv_total_va_bytes)
            self._kv_total_va_bytes = 0
