"""MXFP8 (block-32 e8m0) W8A16 linear: kernels vs the dequant reference, plus the
Gemma (1+w) norm and swigluoai activation the MiniMax-M3 modules ride on."""

from __future__ import annotations

import logging

import pytest
import torch
import triton
from triton.runtime.errors import OutOfResources

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

DEV = "cuda"


def _make_mxfp8(N: int, K: int, seed: int = 0, device: str = DEV):
    torch.manual_seed(seed)
    wf = torch.randn(N, K, device=device) * 0.05
    codes = torch.randint(110, 132, (N, K // 32), device=device, dtype=torch.uint8)
    descale = torch.exp2(codes.float() - 127.0)
    w8 = (
        (wf.view(N, -1, 32) / descale.unsqueeze(-1)).clamp(-448, 448).view(N, K)
    ).to(torch.float8_e4m3fn)
    return w8, codes


# 1 = m1 kernel; 2..256 = dot GEMV across its M_TILE buckets {16,32,64,128,256}
# (17/33/129 exercise row-padding masks); 257/300 = dequant+cuBLAS past
# _GEMV_MAX_M (257 pins the boundary).
@cuda
@pytest.mark.parametrize("M", [1, 2, 8, 16, 17, 33, 64, 129, 256, 257, 300])
# (511, 6144): N not a BLOCK_N multiple (n-mask tail); (640, 6112): K a multiple
# of the 32-wide scale block but not of BLOCK_K=128 (k-mask + OOB scale codes).
@pytest.mark.parametrize("N,K", [(512, 6144), (9216, 6144), (640, 6144), (511, 6144), (640, 6112)])
def test_mxfp8_linear_matches_dequant_reference(M: int, N: int, K: int):
    from freetoken.kernel.triton.mxfp8_linear import mxfp8_dequant, mxfp8_linear

    w8, codes = _make_mxfp8(N, K, seed=M)
    ref_w = mxfp8_dequant(w8, codes, torch.float32)
    x = torch.randn(M, K, device=DEV, dtype=torch.bfloat16)
    y = mxfp8_linear(x, w8, codes)
    y_ref = (x.float() @ ref_w.t()).to(torch.bfloat16)
    rel = (y.float() - y_ref.float()).abs().max() / y_ref.float().abs().max().clamp(min=1e-6)
    assert rel.item() < 2e-2, rel.item()


@cuda
def test_mxfp8_linear_past_the_gemv_bounds_its_dequant_transient(monkeypatch):
    """A wide weight (lm_head-like) past the GEMV must not materialize whole in bf16."""
    import freetoken.kernel.triton.mxfp8_linear as mod

    N, K, M = 8192, 2560, 300
    w8, codes = _make_mxfp8(N, K)
    x = torch.randn(M, K, device=DEV, dtype=torch.bfloat16)
    y_full = mod.mxfp8_linear(x, w8, codes)
    monkeypatch.setattr(mod, "_DEQUANT_CHUNK_BYTES", 2 << 20)  # 409 rows of K=2560 bf16
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    y = mod.mxfp8_linear(x, w8, codes)
    peak = torch.cuda.max_memory_allocated() - base
    whole_weight = N * K * 2
    assert peak < whole_weight // 4, (peak, whole_weight)
    ref = (x.float() @ mod.mxfp8_dequant(w8, codes, torch.float32).t()).to(torch.bfloat16)
    rel = (y.float() - ref.float()).abs().max() / ref.float().abs().max()
    assert rel.item() < 2e-2
    assert torch.allclose(y.float(), y_full.float(), rtol=1e-2, atol=1e-2)


def _gemv_with_smem_for(max_m_tile: int):
    """A stand-in for ``_gemv`` whose dot kernel runs out of shared memory past ``max_m_tile``; NaN marks its output."""

    def gemv(a, weight, scale_codes, out_dtype):
        m_tile = max(16, triton.next_power_of_2(a.shape[0]))
        if a.shape[0] > 1 and m_tile > max_m_tile:
            raise OutOfResources(m_tile * 512, max_m_tile * 512, "shared memory")
        return torch.full((a.shape[0], weight.shape[0]), float("nan"), dtype=out_dtype)

    return gemv


def test_gemv_cap_drops_with_a_warning_when_an_m_tile_overflows_shared_memory(monkeypatch, caplog):
    import freetoken.kernel.triton.mxfp8_linear as mod

    monkeypatch.setattr(mod, "_gemv_cap", mod._GEMV_MAX_M)
    monkeypatch.setattr(mod, "_small_m_gemv_ok", lambda: True)
    monkeypatch.setattr(mod, "e4m3_kernel_view", lambda w: w)
    monkeypatch.setattr(mod, "_gemv", _gemv_with_smem_for(64))
    w8, codes = _make_mxfp8(64, 128, device="cpu")
    x = torch.randn(100, 128, dtype=torch.bfloat16)
    with caplog.at_level(logging.WARNING, logger=mod.__name__):
        y = mod.mxfp8_linear(x, w8, codes)
    assert mod._gemv_cap == 64
    assert "M_TILE 128" in caplog.text and "M > 64" in caplog.text
    ref = x.float() @ mod.mxfp8_dequant(w8, codes, torch.float32).t()
    assert ((y.float() - ref).abs().max() / ref.abs().max()).item() < 2e-2  # the dequant fallback served it
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=mod.__name__):
        assert torch.isnan(mod.mxfp8_linear(x[:64], w8, codes)).all()  # still the GEMV up to the cap
        assert not torch.isnan(mod.mxfp8_linear(x, w8, codes)).any()  # past it, no retry
    assert not caplog.records


@cuda
def test_gemma_plus_one_norm_matches_flashinfer_semantics():
    """Triton fallback vs the (1+w) definition; per-head 3D strided in-place."""
    from freetoken.kernel.triton.norm import gemma_fused_add_rmsnorm, gemma_rmsnorm

    torch.manual_seed(0)
    x = torch.randn(64, 6144, device=DEV, dtype=torch.bfloat16)
    w = torch.randn(6144, device=DEV, dtype=torch.bfloat16) * 0.1

    def ref(v):
        vf = v.float()
        inv = torch.rsqrt(vf.pow(2).mean(-1, keepdim=True) + 1e-6)
        return (vf * inv * (1.0 + w.float())).to(torch.bfloat16)

    out = gemma_rmsnorm(x, w, 1e-6)
    assert (out.float() - ref(x).float()).abs().max().item() < 2e-2

    a, r = x.clone(), x.clone()
    gemma_fused_add_rmsnorm(a, r, w, 1e-6)
    assert torch.equal(r, (x.float() + x.float()).to(torch.bfloat16))
    assert (a.float() - ref(r).float()).abs().max().item() < 2e-2

    # per-head strided in-place (the fused-qkv slice pattern)
    q = torch.randn(16, 9216, device=DEV, dtype=torch.bfloat16)
    wq = torch.randn(128, device=DEV, dtype=torch.bfloat16) * 0.1
    qh = q[:, :8192].view(16, 64, 128)
    ref3 = torch.empty_like(qh)
    for h in range(64):
        vf = qh[:, h].float()
        inv = torch.rsqrt(vf.pow(2).mean(-1, keepdim=True) + 1e-6)
        ref3[:, h] = (vf * inv * (1.0 + wq.float())).to(torch.bfloat16)
    gemma_rmsnorm(qh, wq, 1e-6, out=qh)
    assert (qh.float() - ref3.float()).abs().max().item() < 2e-2


@cuda
def test_swigluoai_and_mul_uninterleaved():
    from freetoken.layers import swigluoai_and_mul

    torch.manual_seed(1)
    d = 3072
    x = torch.randn(500, 2 * d, device=DEV, dtype=torch.bfloat16) * 3
    gate, up = x[:, :d].float(), x[:, d:].float()
    alpha, limit = 1.702, 7.0
    g = gate.clamp(max=limit)
    ref = g * torch.sigmoid(g * alpha) * (up.clamp(-limit, limit) + 1.0)
    out = swigluoai_and_mul(x, alpha=alpha, limit=limit)
    assert (out.float() - ref).abs().max().item() < 0.15
