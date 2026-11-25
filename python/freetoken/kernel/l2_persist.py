"""CUDA persisting L2 cache window helper for activation ping-pong pools (sm_80 / sm_89+).

On GPUs with large L2 caches (e.g. NVIDIA RTX 6000 Ada sm_89 with 96 MiB L2 and ~64 MiB max
persisting capacity), streaming 60-100+ MiB of MoE expert weights per layer can evict the
small [T, 4, 2560] Hyper-Connection residual and activation buffers captured inside the
CUDA graph private memory pool (~48 MiB total). Pinning a bounded persisting L2 window over
the primary CUDA graph segment keeps step activations L2-resident with zero numerical change.
"""

from __future__ import annotations

import ctypes
import sys

import torch

from freetoken.utils import init_logger

logger = init_logger(__name__)

# CUDA runtime constants (driver/runtime ABI stable across CUDA 11/12/13).
_CUDA_LIMIT_PERSISTING_L2_CACHE_SIZE = 0x06
_CUDA_DEV_ATTR_MAX_PERSISTING_L2_CACHE_SIZE = 108
_CUDA_DEV_ATTR_MAX_ACCESS_POLICY_WINDOW_SIZE = 109
_CUDA_STREAM_ATTRIBUTE_ACCESS_POLICY_WINDOW = 1
_CUDA_ACCESS_PROPERTY_NORMAL = 0
_CUDA_ACCESS_PROPERTY_PERSISTING = 2
_MIN_PERSISTING_L2_CAPACITY_BYTES = 16 * 1024 * 1024


class _CudaAccessPolicyWindow(ctypes.Structure):
    _fields_ = [
        ("base_ptr", ctypes.c_void_p),
        ("num_bytes", ctypes.c_size_t),
        ("hitRatio", ctypes.c_float),
        ("hitProp", ctypes.c_int),
        ("missProp", ctypes.c_int),
    ]


class _CudaStreamAttrValue(ctypes.Union):
    _fields_ = [
        ("accessPolicyWindow", _CudaAccessPolicyWindow),
        ("_pad", ctypes.c_byte * 64),
    ]


def _load_cudart():
    if sys.platform != "linux" or torch.version.hip is not None:
        return None
    for name in ("libcudart.so", "libcudart.so.13", "libcudart.so.12"):
        try:
            rt = ctypes.CDLL(name)
        except OSError:
            continue
        rt.cudaGetDevice.argtypes = [ctypes.POINTER(ctypes.c_int)]
        rt.cudaGetDevice.restype = ctypes.c_int
        rt.cudaDeviceGetAttribute.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_int,
        ]
        rt.cudaDeviceGetAttribute.restype = ctypes.c_int
        rt.cudaDeviceSetLimit.argtypes = [ctypes.c_int, ctypes.c_size_t]
        rt.cudaDeviceSetLimit.restype = ctypes.c_int
        rt.cudaStreamSetAttribute.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.POINTER(_CudaStreamAttrValue),
        ]
        rt.cudaStreamSetAttribute.restype = ctypes.c_int
        rt.cudaGetLastError.argtypes = []
        rt.cudaGetLastError.restype = ctypes.c_int
        return rt
    return None


def find_graph_pool_window(
    pool_id: tuple[int, int] | None, device_index: int
) -> tuple[int, int] | None:
    """Return ``(base_ptr, size_bytes)`` of the largest CUDA-graph private pool segment on ``device_index``."""
    if pool_id is None:
        return None
    try:
        snap = torch.cuda.memory_snapshot()
    except (RuntimeError, AttributeError):
        return None
    best: tuple[int, int] | None = None
    for seg in snap:
        if seg.get("device") != device_index:
            continue
        if seg.get("segment_pool_id") != pool_id:
            continue
        addr = int(seg.get("address", 0))
        size = int(seg.get("total_size", 0))
        if addr > 0 and size > 0 and (best is None or size > best[1]):
            best = (addr, size)
    return best


def set_stream_persisting_l2_window(
    stream: torch.cuda.Stream,
    base_ptr: int,
    num_bytes: int,
    *,
    max_window_mb: int = 32,
    cudart=None,
) -> int:
    """Attach a persisting L2 access policy window on ``stream`` covering ``[base_ptr, +window_bytes)``.

    Returns the configured ``window_bytes`` (``0`` if unsupported, disabled, or below the 16 MiB
    architecture gate so small-L2 GPUs keep 100% streaming L2 capacity).
    """
    if max_window_mb <= 0 or base_ptr <= 0 or num_bytes <= 0:
        return 0
    rt = cudart if cudart is not None else _load_cudart()
    if rt is None:
        return 0

    dev = ctypes.c_int(0)
    if rt.cudaGetDevice(ctypes.byref(dev)) != 0:
        rt.cudaGetLastError()
        return 0

    max_persisting = ctypes.c_int(0)
    if (
        rt.cudaDeviceGetAttribute(
            ctypes.byref(max_persisting),
            _CUDA_DEV_ATTR_MAX_PERSISTING_L2_CACHE_SIZE,
            int(dev.value),
        )
        != 0
    ):
        rt.cudaGetLastError()
        return 0
    if int(max_persisting.value) < _MIN_PERSISTING_L2_CAPACITY_BYTES:
        return 0

    max_policy_win = ctypes.c_int(0)
    if (
        rt.cudaDeviceGetAttribute(
            ctypes.byref(max_policy_win),
            _CUDA_DEV_ATTR_MAX_ACCESS_POLICY_WINDOW_SIZE,
            int(dev.value),
        )
        != 0
    ):
        rt.cudaGetLastError()
        return 0

    limit_bytes = min(int(max_window_mb) * 1024 * 1024, int(max_persisting.value))
    if int(max_policy_win.value) > 0:
        limit_bytes = min(limit_bytes, int(max_policy_win.value))
    if limit_bytes <= 0:
        return 0

    if (
        rt.cudaDeviceSetLimit(
            _CUDA_LIMIT_PERSISTING_L2_CACHE_SIZE,
            limit_bytes,
        )
        != 0
    ):
        rt.cudaGetLastError()
        return 0

    window_bytes = min(int(num_bytes), limit_bytes)
    attr = _CudaStreamAttrValue()
    attr.accessPolicyWindow.base_ptr = int(base_ptr)
    attr.accessPolicyWindow.num_bytes = int(window_bytes)
    attr.accessPolicyWindow.hitRatio = 1.0
    attr.accessPolicyWindow.hitProp = _CUDA_ACCESS_PROPERTY_PERSISTING
    attr.accessPolicyWindow.missProp = _CUDA_ACCESS_PROPERTY_NORMAL
    if (
        rt.cudaStreamSetAttribute(
            int(stream.cuda_stream),
            _CUDA_STREAM_ATTRIBUTE_ACCESS_POLICY_WINDOW,
            ctypes.byref(attr),
        )
        != 0
    ):
        rt.cudaGetLastError()
        return 0
    return window_bytes


def clear_stream_persisting_l2_window(
    stream: torch.cuda.Stream | None, *, cudart=None
) -> None:
    """Reset any persisting L2 access policy window on ``stream`` back to normal."""
    if stream is None:
        return
    rt = cudart if cudart is not None else _load_cudart()
    if rt is None:
        return
    attr = _CudaStreamAttrValue()
    attr.accessPolicyWindow.base_ptr = 0
    attr.accessPolicyWindow.num_bytes = 0
    attr.accessPolicyWindow.hitRatio = 0.0
    attr.accessPolicyWindow.hitProp = _CUDA_ACCESS_PROPERTY_NORMAL
    attr.accessPolicyWindow.missProp = _CUDA_ACCESS_PROPERTY_NORMAL
    rt.cudaStreamSetAttribute(
        int(stream.cuda_stream),
        _CUDA_STREAM_ATTRIBUTE_ACCESS_POLICY_WINDOW,
        ctypes.byref(attr),
    )
    rt.cudaGetLastError()
