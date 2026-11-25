"""Two-tier pinned-sysmem PCIe TP=2 AllReduce communicator (opt-in via ``FREETOKEN_PCIE_ALLREDUCE=1``).

Allocates a single shared ``MAP_SHARED`` pinned host arena across two local TP ranks:
  - ``[0, 64 KiB)``: Rank 0 ``Signal`` struct (128-byte aligned counters + epoch parity)
  - ``[64 KiB, 128 KiB)``: Rank 1 ``Signal`` struct
  - ``[128 KiB, 128 KiB + 3 * cap_bytes)``: Rank 0 staging (`[0, cap)` DMA TX, `[cap, 3*cap)` ping-pong)
  - ``[128 KiB + 3 * cap_bytes, 128 KiB + 6 * cap_bytes)``: Rank 1 staging
"""

from __future__ import annotations

import functools
import mmap
import os
import struct
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from freetoken.utils import init_logger

from .utils import load_aot

if TYPE_CHECKING:
    from tvm_ffi import Module

logger = init_logger(__name__)

_ENV_TRUE = {"1", "true", "yes", "on"}

SIGNAL_ABI_BYTES = 19072
SIGNAL_STRIDE_BYTES = 65536
# Byte offset of Signal::end[128][0] inside Signal (start[144][16] = 9216B, end[128][0] = +8192B).
SIGNAL_CAP_BYTE_OFFSET = 17408
SIGNAL_BANK_FLAGS_BYTE_OFFSET = 9216 + 129 * 16 * 4
SIGNAL_EPOCH_FLAGS_BYTE_OFFSET = 18432
DMA_THRESHOLD_BYTES = 65536
DEFAULT_CAP_BYTES = (
    8 * 1024 * 1024
)  # 8 MiB per epoch slot -> 48.125 MiB total shared arena
TP2_PCIE_BLOCK_CAP = 16


def pcie_allreduce_enabled() -> bool:
    """True when ``FREETOKEN_PCIE_ALLREDUCE=1`` opts into the pinned-sysmem TP=2 AllReduce."""
    return os.getenv("FREETOKEN_PCIE_ALLREDUCE", "0").strip().lower() in _ENV_TRUE


def resolve_pcie_ar_cap_bytes() -> int:
    """Per-slot staging capacity in bytes (16-byte aligned, >= 64 KiB)."""
    raw = os.getenv("FREETOKEN_PCIE_AR_MAX_BYTES", "").strip()
    cap = int(raw) if raw else DEFAULT_CAP_BYTES
    cap = max(DMA_THRESHOLD_BYTES, (cap + 15) & ~15)
    if cap >= (1 << 32):
        raise ValueError(f"FREETOKEN_PCIE_AR_MAX_BYTES={cap} must fit in uint32")
    return cap


def arena_layout(cap_bytes: int) -> tuple[int, int, int, int, int]:
    """Return ``(total_bytes, sig0_off, sig1_off, stage0_off, stage1_off)`` for ``cap_bytes``."""
    if cap_bytes < DMA_THRESHOLD_BYTES or (cap_bytes & 15) != 0:
        raise ValueError(
            f"cap_bytes ({cap_bytes}) must be >= {DMA_THRESHOLD_BYTES} and 16-byte aligned"
        )
    sig0_off = 0
    sig1_off = SIGNAL_STRIDE_BYTES
    stage0_off = 2 * SIGNAL_STRIDE_BYTES
    rank_stage = 3 * cap_bytes
    stage1_off = stage0_off + rank_stage
    total_bytes = stage1_off + rank_stage
    return total_bytes, sig0_off, sig1_off, stage0_off, stage1_off


def init_arena_signals(buf: memoryview | bytearray, cap_bytes: int) -> None:
    """Zero Signal regions and stamp ``cap_bytes`` into ``sig0->end[128][0]`` and ``sig1->end[128][0]``."""
    _, sig0_off, sig1_off, stage0_off, _ = arena_layout(cap_bytes)
    buf[:stage0_off] = b"\x00" * stage0_off
    packed_cap = struct.pack("<I", cap_bytes)
    for base in (sig0_off, sig1_off):
        off = base + SIGNAL_CAP_BYTE_OFFSET
        buf[off : off + 4] = packed_cap


@functools.cache
def _load_pcie_ar_module() -> Module:
    return load_aot("pcie_allreduce", cuda_files=["pcie_allreduce.cu"])


@functools.cache
def _get_pcie_ar_wrapper_cls():
    import tvm_ffi

    @tvm_ffi.register_object("freetoken.PcieAllReduceWrapper")
    class PcieAllReduceImpl(tvm_ffi.Object):
        def __init__(self, *args):
            self.__ffi_init__(*args)

    return PcieAllReduceImpl


@dataclass
class PcieSysmemCommunicator:
    """Owns the pinned ``HostBank`` arena and the TVM-FFI ``PcieAllReduceWrapper``."""

    wrapper: Any
    bank: Any
    max_bytes: int
    rank: int
    block_cap: int

    def all_reduce(self, x: torch.Tensor) -> None:
        self.wrapper.all_reduce(x)


def _pick_shm_dir(required_bytes: int) -> str | None:
    candidates = []
    if env_dir := os.getenv("FREETOKEN_SHM_DIR", "").strip():
        candidates.append(env_dir)
    candidates.extend(["/dev/shm", "/var/tmp"])
    for d in candidates:
        try:
            if not os.path.isdir(d):
                continue
            st = os.statvfs(d)
            if st.f_bavail * st.f_frsize >= required_bytes + (16 << 20):
                return d
        except OSError:
            continue
    return None


def init_pcie_allreduce(
    *,
    tp_rank: int,
    tp_size: int,
    tp_cpu_group: torch.distributed.ProcessGroup,
    cap_bytes: int | None = None,
) -> PcieSysmemCommunicator | None:
    """Initialize shared pinned-sysmem TP=2 AllReduce across local ranks, or return ``None`` on fallback."""
    import torch.distributed as dist

    from freetoken.kernel.pinned import device_ptr
    from freetoken.moe.host_banks import HostBank, device_numa_cpus, thread_affinity

    if tp_size != 2:
        logger.warning_rank0(
            f"FREETOKEN_PCIE_ALLREDUCE=1 requires tp_size == 2 (got {tp_size}); keeping PyNCCL"
        )
        return None

    cap = resolve_pcie_ar_cap_bytes() if cap_bytes is None else int(cap_bytes)
    total_bytes, sig0_off, sig1_off, stage0_off, stage1_off = arena_layout(cap)

    nodename = os.uname().nodename if hasattr(os, "uname") else "localhost"
    dev_idx = torch.cuda.current_device()
    sm_count = int(torch.cuda.get_device_properties(dev_idx).multi_processor_count)

    peers: list[tuple[int, str, int] | None] = [None] * tp_size
    dist.all_gather_object(peers, (tp_rank, nodename, sm_count), group=tp_cpu_group)
    assert all(p is not None for p in peers)
    hosts = {p[1] for p in peers if p is not None}
    if len(hosts) != 1:
        logger.warning_rank0(
            f"FREETOKEN_PCIE_ALLREDUCE=1 requires single-host TP=2 (got {sorted(hosts)}); keeping PyNCCL"
        )
        return None
    min_sms = min(p[2] for p in peers if p is not None)
    block_cap = max(1, min(TP2_PCIE_BLOCK_CAP, min_sms))

    module = _load_pcie_ar_module()
    assert int(module.pcie_signal_bytes()) == SIGNAL_ABI_BYTES
    module.pcie_warmup_kernels()
    cls = _get_pcie_ar_wrapper_cls()

    offer_path: str | None = None
    shared_buf: mmap.mmap | None = None
    if tp_rank == 0:
        shm_dir = _pick_shm_dir(total_bytes)
        if shm_dir is not None:
            path = os.path.join(
                shm_dir, f"freetoken-pcie-ar-{os.getpid()}-{uuid.uuid4().hex[:8]}"
            )
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
            try:
                os.ftruncate(fd, total_bytes)
                shared_buf = mmap.mmap(
                    fd, total_bytes, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE
                )
            except BaseException:
                try:
                    os.unlink(path)
                except OSError:
                    pass
                raise
            finally:
                os.close(fd)
            with thread_affinity(device_numa_cpus(dev_idx)):
                mv = memoryview(shared_buf)
                # Touch all pages on GPU's PCIe NUMA node before pinning.
                mv[:] = b"\x00" * total_bytes
                init_arena_signals(mv, cap)
                del mv
            offer_path = path

    offers: list[str | None] = [None] * tp_size
    dist.all_gather_object(offers, offer_path, group=tp_cpu_group)
    shm_path = offers[0]
    if shm_path is None:
        logger.warning_rank0(
            "FREETOKEN_PCIE_ALLREDUCE=1 skipped: no shared-memory directory had enough free space"
        )
        return None

    attach_err: str | None = None
    if tp_rank != 0:
        try:
            fd = os.open(shm_path, os.O_RDWR)
            try:
                shared_buf = mmap.mmap(
                    fd, total_bytes, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE
                )
            finally:
                os.close(fd)
        except BaseException as exc:  # noqa: BLE001
            attach_err = repr(exc)

    try:
        errs: list[str | None] = [None] * tp_size
        dist.all_gather_object(errs, attach_err, group=tp_cpu_group)
    finally:
        if tp_rank == 0:
            try:
                os.unlink(shm_path)
            except OSError:
                pass

    if any(errs):
        if shared_buf is not None:
            shared_buf.close()
        raise RuntimeError(f"PCIe AllReduce shared arena attach failed: {errs}")

    assert shared_buf is not None
    bank = HostBank((total_bytes,), torch.uint8, _shared_buf=shared_buf)
    bank.pin()
    base_dev = int(device_ptr(bank.tensor))
    dist.barrier(group=tp_cpu_group)

    wrapper = cls(
        tp_rank,
        base_dev + sig0_off,
        base_dev + sig1_off,
        base_dev + stage0_off,
        base_dev + stage1_off,
        cap,
        block_cap,
    )
    # Paired hardware self-test across Tier 1 (< 64 KiB) and Tier 2 (>= 64 KiB).
    test_err: str | None = None
    try:
        dev = torch.device("cuda", dev_idx)
        for n_elems in (128, min(cap // 2, 40960)):
            if n_elems <= 0:
                continue
            val = 1.25 if tp_rank == 0 else 2.5
            probe = torch.full((n_elems,), val, dtype=torch.bfloat16, device=dev)
            wrapper.all_reduce(probe)
            torch.cuda.synchronize(dev)
            expected = torch.full((n_elems,), 3.75, dtype=torch.bfloat16, device=dev)
            if not torch.equal(probe, expected):
                test_err = f"self-test mismatch at n_elems={n_elems}"
                break
    except BaseException as exc:  # noqa: BLE001
        test_err = repr(exc)

    probe_errs: list[str | None] = [None] * tp_size
    dist.all_gather_object(probe_errs, test_err, group=tp_cpu_group)
    dist.barrier(group=tp_cpu_group)
    if any(probe_errs):
        logger.warning_rank0(
            f"FREETOKEN_PCIE_ALLREDUCE=1 self-test failed ({probe_errs}); falling back to PyNCCL"
        )
        bank.unpin()
        shared_buf.close()
        return None

    logger.info_rank0(
        f"Enabled pinned-sysmem PCIe TP=2 AllReduce: arena={total_bytes / 2**20:.2f} MiB, "
        f"cap={cap / 2**20:.2f} MiB, blocks<={block_cap} (self-test PASS)"
    )
    return PcieSysmemCommunicator(
        wrapper=wrapper,
        bank=bank,
        max_bytes=cap,
        rank=tp_rank,
        block_cap=block_cap,
    )


def simulate_tp2_arena_all_reduce(
    arena: bytearray,
    cap_bytes: int,
    x0: torch.Tensor,
    x1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    """CPU reference simulation of ``pcie_allreduce.cu`` over a shared ``arena`` buffer."""
    assert x0.shape == x1.shape and x0.dtype == x1.dtype
    assert x0.is_contiguous() and x1.is_contiguous()
    _, sig0_off, sig1_off, stage0_off, stage1_off = arena_layout(cap_bytes)
    nbytes = x0.numel() * x0.element_size()
    assert nbytes <= cap_bytes
    dma_tx = nbytes >= DMA_THRESHOLD_BYTES

    raw_blocks = max(1, min(TP2_PCIE_BLOCK_CAP, (nbytes + 8191) // 8192))

    # Verify cap_bytes stamped in both Signal::end[128][0]
    for sig_off in (sig0_off, sig1_off):
        stamped = struct.unpack_from("<I", arena, sig_off + SIGNAL_CAP_BYTE_OFFSET)[0]
        assert stamped == cap_bytes

    slot_index = 0
    if not dma_tx:
        for sig_off in (sig0_off, sig1_off):
            bp = (
                struct.unpack_from(
                    "<I", arena, sig_off + SIGNAL_BANK_FLAGS_BYTE_OFFSET
                )[0]
                + 1
            )
            for b in range(144):
                struct.pack_into(
                    "<I", arena, sig_off + SIGNAL_BANK_FLAGS_BYTE_OFFSET + 4 * b, bp
                )
            slot_index = 2 if (bp & 1) else 1

    stage0_slot = stage0_off + slot_index * cap_bytes
    stage1_slot = stage1_off + slot_index * cap_bytes

    b0 = bytes(x0.view(torch.uint8).reshape(-1).numpy())
    b1 = bytes(x1.view(torch.uint8).reshape(-1).numpy())
    arena[stage0_slot : stage0_slot + nbytes] = b0
    arena[stage1_slot : stage1_slot + nbytes] = b1

    # Advance per-CTA barrier epochs (1 barrier for < 64 KiB, 2 barriers for >= 64 KiB)
    barriers = 2 if dma_tx else 1
    for _ in range(barriers):
        for rank, (self_off, peer_off) in enumerate(
            ((sig0_off, sig1_off), (sig1_off, sig0_off))
        ):
            for b in range(raw_blocks):
                ep = (
                    struct.unpack_from(
                        "<I", arena, self_off + SIGNAL_EPOCH_FLAGS_BYTE_OFFSET + 4 * b
                    )[0]
                    + 1
                )
                # Peer start[b][rank]
                struct.pack_into("<I", arena, peer_off + (b * 16 + rank) * 4, ep)
                struct.pack_into(
                    "<I", arena, self_off + SIGNAL_EPOCH_FLAGS_BYTE_OFFSET + 4 * b, ep
                )

    peer_for_0 = torch.frombuffer(
        memoryview(arena)[stage1_slot : stage1_slot + nbytes], dtype=x0.dtype
    ).reshape(x0.shape)
    peer_for_1 = torch.frombuffer(
        memoryview(arena)[stage0_slot : stage0_slot + nbytes], dtype=x1.dtype
    ).reshape(x1.shape)

    if x0.dtype in (torch.bfloat16, torch.float16):
        out0 = (x0.float() + peer_for_0.float()).to(x0.dtype)
        out1 = (peer_for_1.float() + x1.float()).to(x1.dtype)
    else:
        out0 = x0 + peer_for_0
        out1 = peer_for_1 + x1

    ep0 = struct.unpack_from("<I", arena, sig0_off + SIGNAL_EPOCH_FLAGS_BYTE_OFFSET)[0]
    return (
        out0.clone(),
        out1.clone(),
        {
            "dma_tx": int(dma_tx),
            "slot": slot_index,
            "blocks": raw_blocks,
            "epoch0": ep0,
        },
    )


__all__ = [
    "DEFAULT_CAP_BYTES",
    "DMA_THRESHOLD_BYTES",
    "SIGNAL_ABI_BYTES",
    "SIGNAL_CAP_BYTE_OFFSET",
    "SIGNAL_STRIDE_BYTES",
    "PcieSysmemCommunicator",
    "arena_layout",
    "init_arena_signals",
    "init_pcie_allreduce",
    "pcie_allreduce_enabled",
    "resolve_pcie_ar_cap_bytes",
    "simulate_tp2_arena_all_reduce",
]
