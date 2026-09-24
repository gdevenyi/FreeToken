"""Opt-in post-load FP8 for the qwen4_exp shared expert and hyper-connection projections."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch

from freetoken.distributed import set_tp_info, try_get_tp_info
from freetoken.layers.base import BaseOP
from freetoken.layers.fp8_dynamic import Fp8DynamicLinear


class _Root(BaseOP):
    def __init__(self, **ops):
        self.__dict__.update(ops)


def _config():
    args = SimpleNamespace(hc_count=4, hidden_size=64, hc_lowrank=32, ple_state_width=4 * 64)
    return SimpleNamespace(qwen4_args=args, rms_norm_eps=1e-6, quant=None)


def _linears(root):
    return {
        "se.gate_up_proj": root.se.gate_up_proj, "se.down_proj": root.se.down_proj,
        "hc.down": root.hc.input_mix_weight_down_block_inject, "hc.up": root.hc.input_mix_weight_up,
        "mixer.down": root.mixer.input_mix_weight_down, "mixer.up": root.mixer.input_mix_weight_up,
    }


def _root(device="cpu"):
    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    from freetoken.models.qwen3_5_moe.moe import _SharedExpert
    from freetoken.models.qwen4_exp.hc import GatedResidual

    root = _Root(
        se=_SharedExpert(_config(), 64, 32, prefix="se"),
        hc=GatedResidual(_config(), prefix="hc"),
        mixer=GatedResidual(_config(), use_combine=False, prefix="mixer"),
    )
    g = torch.Generator().manual_seed(0)
    for lin in _linears(root).values():
        lin.weight = (torch.randn(lin.weight.shape, generator=g) * 0.05).to(torch.bfloat16).to(device)
    root.hc.hc_norm.weight = torch.zeros_like(root.hc.hc_norm.weight, dtype=torch.bfloat16).to(device)
    root.mixer.hc_norm.weight = torch.zeros_like(root.mixer.hc_norm.weight, dtype=torch.bfloat16).to(device)
    return root


def test_flags_off_leaves_everything_bf16(monkeypatch):
    from freetoken.models.qwen4_exp.fp8_post_load import quantize_after_load

    monkeypatch.delenv("FREETOKEN_FP8_SHARED_EXPERT", raising=False)
    monkeypatch.delenv("FREETOKEN_FP8_HC", raising=False)
    root = _root()
    assert quantize_after_load(root) == 0
    assert all(lin.weight.dtype == torch.bfloat16 for lin in _linears(root).values())


@pytest.mark.parametrize("shared, hc, expect", [("1", "0", 2), ("0", "1", 4), ("1", "1", 6)])
def test_swaps_exactly_the_opted_in_projections(monkeypatch, shared, hc, expect):
    from freetoken.models.qwen4_exp.fp8_post_load import quantize_after_load

    monkeypatch.setenv("FREETOKEN_FP8_SHARED_EXPERT", shared)
    monkeypatch.setenv("FREETOKEN_FP8_HC", hc)
    root = _root()
    before = {k: v.weight.float().clone() for k, v in _linears(root).items()}
    assert quantize_after_load(root) == expect
    for name, lin in _linears(root).items():
        want_fp8 = (name.startswith("se.") and shared == "1") or (not name.startswith("se.") and hc == "1")
        assert isinstance(lin, Fp8DynamicLinear) == want_fp8, name
        if want_fp8:
            deq = lin.weight.float() * lin.weight_scale
            rel = (deq - before[name]).norm() / before[name].norm()
            assert rel < 0.05, (name, float(rel))
    assert root.hc.input_mix_weight_down_block_inject.weight.shape[0] % 16 == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA (_scaled_mm)")
def test_fp8_forward_tracks_the_bf16_forward(monkeypatch):
    from freetoken.models.qwen4_exp.fp8_post_load import quantize_after_load

    monkeypatch.setenv("FREETOKEN_FP8_SHARED_EXPERT", "1")
    monkeypatch.setenv("FREETOKEN_FP8_HC", "1")
    ref = _root("cuda")
    fp8 = copy.deepcopy(ref)
    quantize_after_load(fp8)
    x = torch.randn(3, 64, dtype=torch.bfloat16, device="cuda")
    R = torch.randn(3, 256, dtype=torch.bfloat16, device="cuda")
    torch.testing.assert_close(fp8.se.forward(x).float(), ref.se.forward(x).float(), rtol=0.1, atol=0.02)
    for a, b in zip(fp8.hc.mix(R), ref.hc.mix(R)):
        torch.testing.assert_close(a.float(), b.float(), rtol=0.1, atol=0.05)
