"""The autotune L2-flush buffer is sized to the device instead of triton's flat 256 MB."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

MB = 1 << 20


@pytest.mark.parametrize("l2_bytes, flush_bytes", [
    (2816 * 1024, 11 * MB),  # GTX 1080 Ti: 4x its 2.75 MB L2
    (0, 8 * MB),  # no L2 size reported: the floor
    (128 * MB, 256 * MB),  # a large L2: triton's own 256 MB is the cap
])
def test_flush_buffer_is_four_times_the_l2_within_bounds(monkeypatch, l2_bytes, flush_bytes):
    from triton.backends.nvidia.driver import CudaDriver

    from freetoken.kernel.triton.autotune_cache import bound_autotune_flush_buffer

    monkeypatch.setattr(CudaDriver, "get_empty_cache_for_benchmark", CudaDriver.get_empty_cache_for_benchmark)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda index: SimpleNamespace(L2_cache_size=l2_bytes))
    bound_autotune_flush_buffer(0)
    bound_autotune_flush_buffer(0)  # the engine may call it again; the patch must not stack
    monkeypatch.setattr(torch, "empty", lambda n, dtype, device: SimpleNamespace(nbytes=n * 4, dtype=dtype, device=device))
    buf = CudaDriver.get_empty_cache_for_benchmark(None)
    assert (buf.nbytes, buf.dtype, buf.device) == (flush_bytes, torch.int, "cuda")
