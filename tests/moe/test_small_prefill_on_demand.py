"""Short prefills load only their routed experts (FREETOKEN_MOE_SMALL_PREFILL_TOKENS) instead of
streaming whole layers; above the threshold, or with it off, the whole-layer path is unchanged."""

import pytest
import torch

import freetoken.layers.moe as moe_mod
from freetoken.distributed import set_tp_info, try_get_tp_info


def _layer_and_cache(device, num_experts=8, top_k=2, hidden=16, inter=32):
    from freetoken.layers.moe import OffloadMoELayer
    from freetoken.layers.quantization import NoQuantConfig
    from freetoken.moe.offload_cache import OffloadMoeCache

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    layer = OffloadMoELayer(0, num_experts, top_k, hidden, inter, quant_config=NoQuantConfig(),
                            prefix="model.layers.0.mlp.experts")
    cache = OffloadMoeCache(num_layers=1, num_experts=num_experts, cache_size=num_experts, device=device)
    g = torch.Generator().manual_seed(0)
    cache.set_bank_sources({
        "gate_up": [torch.randn(num_experts, 2 * inter, hidden, generator=g, dtype=torch.bfloat16) * 0.1],
        "down": [torch.randn(num_experts, hidden, inter, generator=g, dtype=torch.bfloat16) * 0.1],
    })
    layer.offload_cache = cache
    return layer, cache


@pytest.mark.parametrize("tokens, threshold, on_demand", [
    (3, 8, True), (8, 8, True), (9, 8, False), (3, 0, False),
])
def test_small_prefill_dispatch(monkeypatch, tokens, threshold, on_demand):
    layer, cache = _layer_and_cache(torch.device("cpu"))
    monkeypatch.setattr(moe_mod, "_SMALL_PREFILL_TOKENS", threshold)
    calls = []
    monkeypatch.setattr(cache, "ensure_experts", lambda layer_id, ids: calls.append("ensure"))
    monkeypatch.setattr(cache, "materialize_layer", lambda layer_id: calls.append("materialize"))
    monkeypatch.setattr(cache, "copy_missing", lambda: calls.append("copy"))
    monkeypatch.setattr(cache, "begin_prefill", lambda: calls.append("begin_prefill"), raising=False)
    got = {}

    def fake_gemm(cache_, hs, w, ids, *, views, n, alphas, is_prefill):
        got.update(n=n, is_prefill=is_prefill)
        return hs

    monkeypatch.setattr(layer, "_expert_gemm", fake_gemm)
    hs = torch.randn(tokens, 16)
    layer._prefill_routed(hs, torch.full((tokens, 2), 0.5), torch.zeros(tokens, 2, dtype=torch.int32))
    if on_demand:
        assert calls == ["ensure", "copy"] and got == {"n": None, "is_prefill": False}
    else:
        assert "ensure" not in calls and "begin_prefill" not in calls[1:] and got["n"] == layer.num_experts


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_small_prefill_matches_the_whole_layer_path(monkeypatch):
    from freetoken.core import get_global_ctx  # noqa: F401  (layer reads the ctx only in forward())

    dev = torch.device("cuda")
    hs = torch.randn(5, 16, dtype=torch.bfloat16, device=dev) * 0.5
    ids = torch.tensor([[0, 3], [1, 3], [7, 2], [3, 4], [5, 0]], dtype=torch.int32, device=dev)
    w = torch.tensor([[0.6, 0.4], [0.5, 0.5], [0.9, 0.1], [0.3, 0.7], [0.2, 0.8]], device=dev)
    outs = {}
    for threshold in (0, 8):
        monkeypatch.setattr(moe_mod, "_SMALL_PREFILL_TOKENS", threshold)
        layer, cache = _layer_and_cache(dev)
        outs[threshold] = layer._prefill_routed(hs.clone(), w.clone(), ids.clone()).float()
    torch.testing.assert_close(outs[8], outs[0], rtol=2e-2, atol=2e-2)
