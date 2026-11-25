"""SM-driven step-input upload ring (P3 copy-engine isolation).

Small per-step scheduler tensors (positions, mrope positions, input/write token mappings,
page-table scatter indices) normally call ``host.to(device, non_blocking=True)``, which
allocates from PyTorch's pinned caching allocator and enqueues ``cudaMemcpyAsync`` onto the
GPU's hardware DMA copy engine FIFO. Under active MoE PCIe streaming or P2c host-tier D2H
demotions, those tiny metadata copies queue behind multi-megabyte DMA bursts.

``SmUploadRing`` stages small contiguous tensors (<= 4 MiB) into a pre-pinned 32 MiB host
FIFO ring (8 segments of 4 MiB) and copies them on SMs via a 1D Triton kernel on the
caller's stream, bypassing the hardware DMA copy engine completely. One lightweight CUDA
event per 4 MiB segment fences ring wrap-around without per-tensor event overhead.
"""

from __future__ import annotations

import os
from collections.abc import Callable

import torch

_DEFAULT_RING_BYTES = 32 << 20
_DEFAULT_NUM_SEGMENTS = 8
_RING_ALIGN = 16


def _align_up(n: int, align: int = _RING_ALIGN) -> int:
    return ((n + align - 1) // align) * align


def sm_upload_enabled() -> bool:
    raw = os.environ.get("FREETOKEN_SM_UPLOAD", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


try:
    import triton
    import triton.language as tl

    @triton.jit
    def _sm_copy_u32_kernel(
        dst_ptr_i64,
        src_ptr_i64,
        n_words,
        BLOCK: tl.constexpr,
    ):
        dst_ptr = dst_ptr_i64.to(tl.pointer_type(tl.uint32))
        src_ptr = src_ptr_i64.to(tl.pointer_type(tl.uint32))
        pid = tl.program_id(0).to(tl.int64)
        offs = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
        mask = offs < n_words
        vals = tl.load(src_ptr + offs, mask=mask, other=0)
        tl.store(dst_ptr + offs, vals, mask=mask)

    _TRITON_OK = True
except Exception:  # pragma: no cover - CPU-only env without triton
    _TRITON_OK = False


def _default_cuda_copy(
    dst: torch.Tensor, host_dev_ptr: int, byte_offset: int, nbytes: int
) -> None:
    n_words = nbytes // 4
    block = 1024
    grid = ((n_words + block - 1) // block,)
    _sm_copy_u32_kernel[grid](
        dst.data_ptr(),
        host_dev_ptr + byte_offset,
        n_words,
        BLOCK=block,
    )


class SmUploadRing:
    """Segmented pinned FIFO ring with per-segment completion fencing."""

    def __init__(
        self,
        device: torch.device,
        *,
        ring_bytes: int = _DEFAULT_RING_BYTES,
        num_segments: int = _DEFAULT_NUM_SEGMENTS,
        alloc_fn: Callable[[int], tuple[torch.Tensor, int]] | None = None,
        copy_fn: Callable[[torch.Tensor, int, int, int], None] | None = None,
        event_factory: Callable[[], object] | None = None,
    ) -> None:
        if num_segments < 2 or ring_bytes % (num_segments * _RING_ALIGN) != 0:
            raise ValueError(
                f"invalid ring geometry: ring_bytes={ring_bytes}, num_segments={num_segments}"
            )
        self.device = device
        self.ring_bytes = ring_bytes
        self.num_segments = num_segments
        self.segment_bytes = ring_bytes // num_segments

        if alloc_fn is None:
            from freetoken.kernel.pinned import alloc_pinned_tensor, device_ptr

            buf = alloc_pinned_tensor(ring_bytes, dtype=torch.uint8)
            dev_ptr = device_ptr(buf)
        else:
            buf, dev_ptr = alloc_fn(ring_bytes)
        self.host_buf = buf
        self.host_dev_ptr = dev_ptr
        self._copy_fn = copy_fn or _default_cuda_copy
        self._event_factory = event_factory or (
            lambda: torch.cuda.Event(enable_timing=False)
        )

        self.seg = 0
        self.seg_offset = 0
        self.seg_dirty = False
        self.seg_events: list[object | None] = [None] * num_segments

    def _advance_segment(self) -> None:
        if self.seg_dirty:
            ev = self.seg_events[self.seg]
            if ev is None:
                ev = self._event_factory()
                self.seg_events[self.seg] = ev
            ev.record()
            self.seg_dirty = False
        next_seg = (self.seg + 1) % self.num_segments
        next_ev = self.seg_events[next_seg]
        if next_ev is not None:
            next_ev.synchronize()
        self.seg = next_seg
        self.seg_offset = 0

    def can_stage(self, src: torch.Tensor) -> bool:
        if not src.is_contiguous():
            return False
        nbytes = src.numel() * src.element_size()
        return 0 < nbytes <= self.segment_bytes and (nbytes % 4 == 0)

    def upload(self, src: torch.Tensor) -> torch.Tensor:
        nbytes = src.numel() * src.element_size()
        aligned = _align_up(nbytes, _RING_ALIGN)
        if self.seg_offset + aligned > self.segment_bytes:
            self._advance_segment()
        byte_offset = self.seg * self.segment_bytes + self.seg_offset
        self.seg_offset += aligned
        self.seg_dirty = True

        src_u8 = src.view(torch.uint8).reshape(-1)
        self.host_buf[byte_offset : byte_offset + nbytes].copy_(src_u8)
        dst = torch.empty(src.shape, dtype=src.dtype, device=self.device)
        self._copy_fn(dst, self.host_dev_ptr, byte_offset, nbytes)
        return dst


_RINGS: dict[int, SmUploadRing | bool] = {}


def _get_ring(device: torch.device) -> SmUploadRing | None:
    idx = 0 if device.index is None else int(device.index)
    ring = _RINGS.get(idx)
    if ring is False:
        return None
    if ring is not None:
        return ring  # type: ignore[return-value]
    if not _TRITON_OK:
        _RINGS[idx] = False
        return None
    try:
        created = SmUploadRing(device)
    except Exception:
        _RINGS[idx] = False
        return None
    _RINGS[idx] = created
    return created


def upload_step_tensor(host_tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Upload a small scheduler step-input tensor to ``device``, preferring the SM copy ring."""
    if device.type != "cuda":
        return host_tensor if host_tensor.device == device else host_tensor.to(device)
    nbytes = host_tensor.numel() * host_tensor.element_size()
    if nbytes == 0:
        return torch.empty(host_tensor.shape, dtype=host_tensor.dtype, device=device)
    if sm_upload_enabled() and not torch.cuda.is_current_stream_capturing():
        ring = _get_ring(device)
        if ring is not None and ring.can_stage(host_tensor):
            return ring.upload(host_tensor)
    staged = host_tensor if host_tensor.is_pinned() else host_tensor.pin_memory()
    return staged.to(device, non_blocking=True)


__all__ = ["SmUploadRing", "sm_upload_enabled", "upload_step_tensor"]
