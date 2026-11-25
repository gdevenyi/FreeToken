"""Unit tests for C4b (int8 g32 W8A16 dense), hc8 Hyper-Connections, and FP32 router logits."""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn.functional as F
from freetoken.layers.int8_g32 import (
    Int8G32Linear,
    Int8G32Replicated,
    dequantize_int8_g32,
    quantize_int8_g32,
)
from freetoken.models.config import (
    fp32_router_enabled,
    hc_int8_enabled,
    int8_dense_enabled,
)
from freetoken.models.qwen4_exp.weight import (
    _quantize_per_channel,
    _quantize_per_tensor,
    dense_emitter,
)


def _rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(
        ((a.float() - b.float()).norm() / b.float().norm().clamp(min=1e-12)).item()
    )


def test_int8_g32_cuts_weight_error_over_3x_vs_fp8_e4m3():
    """Verify C4b math on real Qwen3.8 TP=2 dense shapes (K=2560, N=1024):
    FP8 E4M3 sits at ~2.6% relative error (both per-tensor and per-channel), whereas
    symmetric int8 g32 sits at ~0.6% (>3.5x lower error)."""
    torch.manual_seed(42)
    w = (torch.randn(1024, 2560) * 0.02).to(torch.bfloat16)

    w_t, s_t = _quantize_per_tensor(w)
    err_fp8_t = _rel_err(w_t.float() * s_t, w)

    w_c, s_c = _quantize_per_channel(w)
    err_fp8_c = _rel_err(w_c.float() * s_c[:, None], w)

    qw, sc = quantize_int8_g32(w)
    err_int8_g32 = _rel_err(dequantize_int8_g32(qw, sc, out_dtype=torch.float32), w)

    assert 0.020 < err_fp8_t < 0.032
    assert 0.020 < err_fp8_c < 0.032
    assert err_int8_g32 < 0.0085
    assert err_int8_g32 < err_fp8_c / 3.2


def test_int8_g32_linear_and_hc8_emitter_and_fp32_router(monkeypatch):
    monkeypatch.setenv("FREETOKEN_INT8_DENSE", "1")
    monkeypatch.setenv("FREETOKEN_HC_INT8", "1")
    monkeypatch.setenv("FREETOKEN_FP32_ROUTER", "1")
    assert int8_dense_enabled() is True
    assert hc_int8_enabled() is True
    assert fp32_router_enabled() is True

    g = SimpleNamespace(
        num_key_heads=4, key_head_dim=32, num_value_heads=8, value_head_dim=32
    )
    cfg = SimpleNamespace(
        attn_quant="int8_g32",
        quant=None,
        tie_word_embeddings=False,
        linear_attention_group=lambda: g,
    )
    emit, quant_active, _ = dense_emitter(cfg, rank=0, world=2)
    assert quant_active is True

    # 1. GDN in_proj splits into int8 g32 in_proj_qkvz + bf16 in_proj_ba
    # world=2 -> nk=2 (2*2*32=128), nv=4 (2*4*32=256) -> qkvz=384, ba=8 -> local=392, global=784
    in_proj_global = torch.randn(784, 128, dtype=torch.bfloat16)
    emitted = dict(emit("model.layers.0.linear_attn.in_proj.weight", in_proj_global))
    assert emitted["model.layers.0.linear_attn.in_proj_qkvz.weight"].dtype == torch.int8
    assert emitted["model.layers.0.linear_attn.in_proj_qkvz.weight"].shape == (384, 128)
    assert emitted["model.layers.0.linear_attn.in_proj_qkvz.weight_scale"].shape == (
        384,
        4,
    )
    assert (
        emitted["model.layers.0.linear_attn.in_proj_ba.weight"].dtype == torch.bfloat16
    )
    assert emitted["model.layers.0.linear_attn.in_proj_ba.weight"].shape == (8, 128)

    # 2. hc8 quantizes input_mix_weight_down_block_inject (N=336, K=10240) and input_mix_weight_up
    hc_down = torch.randn(336, 256, dtype=torch.bfloat16) * 0.05
    hc_out = dict(
        emit(
            "model.layers.0.attn_hc.input_mix_weight_down_block_inject.weight", hc_down
        )
    )
    assert (
        hc_out["model.layers.0.attn_hc.input_mix_weight_down_block_inject.weight"].dtype
        == torch.int8
    )
    assert hc_out[
        "model.layers.0.attn_hc.input_mix_weight_down_block_inject.weight_scale"
    ].shape == (336, 8)

    op = Int8G32Replicated(256, 336, output_sizes=[320, 16])
    op.weight.copy_(
        hc_out["model.layers.0.attn_hc.input_mix_weight_down_block_inject.weight"]
    )
    op.weight_scale.copy_(
        hc_out["model.layers.0.attn_hc.input_mix_weight_down_block_inject.weight_scale"]
    )
    x = torch.randn(5, 256, dtype=torch.bfloat16)
    y = op.forward(x)
    y_ref = F.linear(x, hc_down)
    assert y.shape == (5, 336)
    assert _rel_err(y, y_ref) < 0.015

    # Zero-row stability check
    w_zero = torch.zeros(64, 64, dtype=torch.bfloat16)
    qw0, sc0 = quantize_int8_g32(w_zero)
    lin0 = Int8G32Linear(64, 64)
    lin0.weight.copy_(qw0)
    lin0.weight_scale.copy_(sc0)
    assert torch.all(lin0.forward(torch.randn(3, 64, dtype=torch.bfloat16)) == 0)
