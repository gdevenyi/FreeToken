from __future__ import annotations

import struct

import pytest
import torch
from freetoken.distributed.impl import DistributedImpl, PcieSysmemDistributedImpl
from freetoken.kernel.pcie_allreduce import (
    DMA_THRESHOLD_BYTES,
    SIGNAL_BANK_FLAGS_BYTE_OFFSET,
    SIGNAL_CAP_BYTE_OFFSET,
    SIGNAL_STRIDE_BYTES,
    arena_layout,
    init_arena_signals,
    pcie_allreduce_enabled,
    resolve_pcie_ar_cap_bytes,
    simulate_tp2_arena_all_reduce,
)


def test_pcie_allreduce_env_and_cap_resolution(monkeypatch):
    monkeypatch.delenv("FREETOKEN_PCIE_ALLREDUCE", raising=False)
    monkeypatch.delenv("FREETOKEN_PCIE_AR_MAX_BYTES", raising=False)
    assert not pcie_allreduce_enabled()
    assert resolve_pcie_ar_cap_bytes() == 8 * 1024 * 1024

    monkeypatch.setenv("FREETOKEN_PCIE_ALLREDUCE", "1")
    monkeypatch.setenv("FREETOKEN_PCIE_AR_MAX_BYTES", "131073")
    assert pcie_allreduce_enabled()
    # Rounded up to 16-byte alignment
    assert resolve_pcie_ar_cap_bytes() == 131088


def test_arena_layout_and_signal_abi():
    cap = 262144  # 256 KiB
    total, sig0, sig1, st0, st1 = arena_layout(cap)
    assert sig0 == 0
    assert sig1 == SIGNAL_STRIDE_BYTES
    assert st0 == 2 * SIGNAL_STRIDE_BYTES
    assert st1 == st0 + 3 * cap
    assert total == st1 + 3 * cap

    arena = bytearray(total)
    init_arena_signals(arena, cap)
    assert struct.unpack_from("<I", arena, sig0 + SIGNAL_CAP_BYTE_OFFSET)[0] == cap
    assert struct.unpack_from("<I", arena, sig1 + SIGNAL_CAP_BYTE_OFFSET)[0] == cap

    with pytest.raises(ValueError, match="16-byte aligned"):
        arena_layout(65538)


@pytest.mark.parametrize(
    "dtype", [torch.bfloat16, torch.float16, torch.float32, torch.int32]
)
def test_two_tier_pingpong_and_dma_simulation(dtype: torch.dtype):
    torch.manual_seed(19)
    cap = 262144
    total, sig0, sig1, _, _ = arena_layout(cap)
    arena = bytearray(total)
    init_arena_signals(arena, cap)

    # 1. Decode shape (< 64 KiB, e.g. 1 token x 2560 = 5120B bf16): uses 1-barrier slot 2 (odd=True on call 1)
    if dtype == torch.int32:
        x0 = torch.randint(-1000, 1000, (1, 2560), dtype=dtype)
        x1 = torch.randint(-1000, 1000, (1, 2560), dtype=dtype)
    else:
        x0 = torch.randn((1, 2560), dtype=dtype)
        x1 = torch.randn((1, 2560), dtype=dtype)
    o0, o1, meta1 = simulate_tp2_arena_all_reduce(arena, cap, x0, x1)
    assert meta1["dma_tx"] == 0
    assert meta1["slot"] == 2
    assert meta1["epoch0"] == 1
    assert torch.equal(o0, o1)  # Bit-exact across Rank 0 and Rank 1!

    # 2. Second < 64 KiB shape with wider grid (8 tokens x 2560 = 40960B): flips to even bank slot 1
    if dtype == torch.int32:
        y0 = torch.randint(-1000, 1000, (5, 2560), dtype=dtype)
        y1 = torch.randint(-1000, 1000, (5, 2560), dtype=dtype)
    else:
        y0 = torch.randn((5, 2560), dtype=dtype)
        y1 = torch.randn((5, 2560), dtype=dtype)
    oy0, oy1, meta2 = simulate_tp2_arena_all_reduce(arena, cap, y0, y1)
    assert meta2["dma_tx"] == 0
    assert meta2["slot"] == 1
    assert meta2["epoch0"] == 2
    assert torch.equal(oy0, oy1)

    # All 144 CTA bank_flags stay synchronized across varying gridDim.x
    flags0 = struct.unpack_from("<144I", arena, sig0 + SIGNAL_BANK_FLAGS_BYTE_OFFSET)
    flags1 = struct.unpack_from("<144I", arena, sig1 + SIGNAL_BANK_FLAGS_BYTE_OFFSET)
    assert len(set(flags0)) == 1 and flags0[0] == 2
    assert len(set(flags1)) == 1 and flags1[0] == 2

    # 3. Medium prefill shape (>= 64 KiB, e.g. 20 tokens x 2560 >= 102400B): uses DMA slot 0 + 2 barriers,
    #    without advancing the 1-barrier ping-pong bank parity!
    if dtype == torch.int32:
        z0 = torch.randint(-1000, 1000, (20, 2560), dtype=dtype)
        z1 = torch.randint(-1000, 1000, (20, 2560), dtype=dtype)
    else:
        z0 = torch.randn((20, 2560), dtype=dtype)
        z1 = torch.randn((20, 2560), dtype=dtype)
    assert z0.nbytes >= DMA_THRESHOLD_BYTES
    oz0, oz1, meta3 = simulate_tp2_arena_all_reduce(arena, cap, z0, z1)
    assert meta3["dma_tx"] == 1
    assert meta3["slot"] == 0
    assert meta3["epoch0"] == 4  # +2 barriers
    assert torch.equal(oz0, oz1)
    flags_after_dma = struct.unpack_from(
        "<144I", arena, sig0 + SIGNAL_BANK_FLAGS_BYTE_OFFSET
    )
    assert flags_after_dma[0] == 2  # Unchanged by DMA TX!


class _MockPcieComm:
    def __init__(self):
        self.calls = 0

    def all_reduce(self, x: torch.Tensor) -> None:
        self.calls += 1
        x.mul_(2)


class _MockFallback(DistributedImpl):
    def __init__(self):
        self.ar_calls = 0
        self.ag_calls = 0

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        self.ar_calls += 1
        return x * 3

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        self.ag_calls += 1
        return torch.cat([x, x], dim=0)


def test_pcie_sysmem_distributed_impl_dispatch_and_fallback(monkeypatch):
    pcie = _MockPcieComm()
    fb = _MockFallback()
    impl = PcieSysmemDistributedImpl(pcie_comm=pcie, fallback=fb, max_bytes=1024)

    # Empty tensor -> no-op
    empty = torch.empty((0, 16), dtype=torch.bfloat16)
    assert impl.all_reduce(empty) is empty
    assert pcie.calls == 0 and fb.ar_calls == 0

    # CPU tensor -> delegates to fallback
    cpu_t = torch.ones((4, 4), dtype=torch.bfloat16)
    out_cpu = impl.all_reduce(cpu_t)
    assert fb.ar_calls == 1 and pcie.calls == 0
    assert torch.all(out_cpu == 3)

    # all_gather -> always delegates to fallback
    ag = impl.all_gather(cpu_t)
    assert fb.ag_calls == 1
    assert ag.shape == (8, 4)

    # Simulate CUDA tensor eligibility via monkeypatched Tensor.is_cuda property if no GPU locally
    if not torch.cuda.is_available():
        monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda _self: True))
        small = torch.ones((8, 16), dtype=torch.bfloat16)  # 256B <= 1024B
        impl.all_reduce(small)
        assert pcie.calls == 1 and fb.ar_calls == 1
        assert torch.all(small == 2)

        # Oversized tensor (> max_bytes=1024B) -> fallback
        huge = torch.ones((32, 32), dtype=torch.bfloat16)  # 2048B > 1024B
        impl.all_reduce(huge)
        assert pcie.calls == 1 and fb.ar_calls == 2

        # Non-contiguous tensor -> fallback
        non_contig = torch.ones((16, 16), dtype=torch.bfloat16)[:, :8]
        assert not non_contig.is_contiguous()
        impl.all_reduce(non_contig)
        assert pcie.calls == 1 and fb.ar_calls == 3

        # Unsupported dtype (int64) -> fallback
        i64 = torch.ones((4, 4), dtype=torch.int64)
        impl.all_reduce(i64)
        assert pcie.calls == 1 and fb.ar_calls == 4
