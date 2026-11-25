"""Unit tests for Qwen3.8 MoE fast paths:
1. Inactive-slot early exit (supports_inactive_slots=True on TritonNvfp4MoEKernel).
2. Combined Recency-Frequency (CRF) + ClassLRU step-lease protection (--moe-cache-policy crf).
3. Wide-route small-prefill grouped Tensor Core dispatch (FREETOKEN_MOE_GROUPED_ROUTES).
"""

from __future__ import annotations

import json

import freetoken.layers.moe as moe_mod
import torch
from freetoken.distributed import set_tp_info, try_get_tp_info
from freetoken.layers.moe import OffloadMoELayer
from freetoken.layers.quantization import NoQuantConfig
from freetoken.layers.quantization.moe.nvfp4 import (
    TritonNvfp4MoEKernel,
    W4a4Nvfp4MoEKernel,
)
from freetoken.moe.offload_cache import RETIRED_SLOT_USAGE, OffloadMoeCache
from freetoken.moe.offload_kernels import crf_hybrid_step_reference
from freetoken.server.args import parse_args

DEV = torch.device("cpu")


def test_nvfp4_triton_kernels_declare_inactive_slots_and_hybrid_skips_clamp(
    monkeypatch,
):
    assert TritonNvfp4MoEKernel.supports_inactive_slots is True
    assert W4a4Nvfp4MoEKernel.supports_inactive_slots is True

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    layer = OffloadMoELayer(
        0,
        num_experts=8,
        top_k=2,
        hidden_size=16,
        intermediate_size=32,
        quant_config=NoQuantConfig(),
        prefix="model.layers.0.mlp.experts",
        decode_target="hybrid",
    )
    monkeypatch.setattr(
        type(layer.quant_method.kernel), "supports_inactive_slots", True, raising=False
    )
    assert layer.skips_inactive_slots is True

    seen_clamp: list[bool] = []
    cache = OffloadMoeCache(
        num_layers=1,
        num_experts=8,
        cache_size=8,
        device=DEV,
        decode_target="hybrid",
    )
    sources = {
        "gate_up": [torch.zeros(8, 64, 16, dtype=torch.bfloat16)],
        "down": [torch.zeros(8, 16, 32, dtype=torch.bfloat16)],
    }
    cache.set_bank_sources(sources)

    def fake_ensure_hybrid(lid, ids, w, clamp=False):
        seen_clamp.append(clamp)
        return crf_hybrid_step_reference(cache, lid, ids, w, max_fetch=1, clamp=clamp)

    class DummyExec:
        def decode_submit(self, lid, hs, w, cpu_ids):
            # Verify GPU-handled slots became -1 in cpu_ids and CPU-handled slots stayed >= 0
            return cpu_ids.clone()

        def decode_sync(self, pending, add_to=None):
            return add_to if add_to is not None else torch.zeros_like(hs)

    monkeypatch.setattr(cache, "ensure_experts_hybrid", fake_ensure_hybrid)
    monkeypatch.setattr(cache, "copy_missing", lambda: None)
    monkeypatch.setattr(
        layer, "_expert_gemm", lambda *a, **k: torch.zeros(2, 16, dtype=torch.bfloat16)
    )
    cache.set_cpu_executor(DummyExec())
    layer.offload_cache = cache

    hs = torch.randn(2, 16, dtype=torch.bfloat16)
    ids = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
    w = torch.full((2, 2), 0.5, dtype=torch.float32)
    layer._decode_hybrid(cache, hs, w, ids)
    assert seen_clamp == [False]
    # Only 1 expert fetched onto GPU; remaining 3 routes MUST carry -1 in ids (not clamped to 0)!
    assert int((ids == -1).sum().item()) == 3


def test_crf_policy_prefers_recurring_warm_misses_and_protects_hits_and_retired_slots(
    tmp_path,
):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "llama",
                "architectures": ["LlamaForCausalLM"],
                "torch_dtype": "bfloat16",
            }
        ),
        encoding="utf-8",
    )
    cfg, _ = parse_args(["--model", str(tmp_path), "--moe-cache-policy", "crf"])
    assert cfg.moe_cache_policy == "crf"

    cache = OffloadMoeCache(
        num_layers=3,
        num_experts=6,
        cache_size=16,
        device=DEV,
        cache_policy="crf",
        decode_target="hybrid",
    )
    # Retire tail slots [6..15] as ElasticKvMoePool does, leaving only 6 active GPU slots [0..5]!
    cache.set_active_slots(6)
    assert cache.active_cache_size == 6

    # Warm up layer 0 experts 0..5 across 12 steps so they hold high CRF and fill active slots [0..5].
    warm_ids = torch.tensor([[0, 1, 2, 3, 4, 5]], dtype=torch.int32)
    warm_w = torch.full((1, 6), 1.0 / 6.0)
    for _ in range(12):
        crf_hybrid_step_reference(cache, 0, warm_ids.clone(), warm_w, max_fetch=6)
    assert all(int(cache.slot_for_id[0, e].item()) in range(6) for e in range(6))

    # On layer 1 (sharing the same unified 6-slot active pool!), build recurring history on expert 1
    # (on CPU overflow with max_fetch=0 for 6 steps), then touch expert 2 ONCE right before a capped fetch:
    for _ in range(6):
        crf_hybrid_step_reference(
            cache,
            1,
            torch.tensor([[1]], dtype=torch.int32),
            torch.ones((1, 1)),
            max_fetch=0,
        )
    crf_hybrid_step_reference(
        cache,
        1,
        torch.tensor([[2]], dtype=torch.int32),
        torch.ones((1, 1)),
        max_fetch=0,
    )
    assert cache.expert_recency[1, 2] > cache.expert_recency[1, 1]
    assert cache.expert_crf[1, 1] > cache.expert_crf[1, 2]

    # Put one hit on layer 1 first (expert 0 into one slot), then test [0, 1, 2, 3] with max_fetch=1:
    crf_hybrid_step_reference(
        cache,
        1,
        torch.tensor([[0]], dtype=torch.int32),
        torch.ones((1, 1)),
        max_fetch=1,
    )
    slot0 = int(cache.slot_for_id[1, 0].item())
    cache.usage[slot0] = 1
    routes = crf_hybrid_step_reference(
        cache,
        1,
        torch.tensor([[0, 1, 2, 3]], dtype=torch.int32),
        torch.full((1, 4), 0.25),
        max_fetch=1,
    )
    # 1. Expert 0 stayed resident in slot0 (current-step hit never evicted!).
    assert int(cache.slot_for_id[1, 0].item()) == slot0
    # 2. Recurring expert 1 beat more-recent 1-hit expert 2 AND cold expert 3!
    assert int(cache.src_indices[0].item()) == 1
    assert int(cache.slot_for_id[1, 1].item()) in range(6)
    assert int(cache.slot_for_id[1, 2].item()) == -1
    assert torch.equal(
        routes.cpu_ids, torch.tensor([[-1, -1, 2, 3]], dtype=torch.int32)
    )
    # 3. Retired VMM tail slots [6..15] were NEVER touched!
    assert torch.all(cache.usage[6:] == RETIRED_SLOT_USAGE)
    assert torch.all(cache.id_of_slot[6:] == -1)

    # Reset clears expert_crf while preserving retired tail slots.
    cache.reset()
    assert float(cache.expert_crf.abs().max().item()) == 0.0
    assert torch.all(cache.usage[6:] == RETIRED_SLOT_USAGE)


def test_wide_small_prefill_compacts_unique_slots_and_dispatches_grouped_gemm(
    monkeypatch,
):
    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    num_experts, top_k, hidden, inter, cache_size = 16, 4, 8, 16, 32
    layer = OffloadMoELayer(
        0,
        num_experts,
        top_k,
        hidden,
        inter,
        quant_config=NoQuantConfig(),
        prefix="model.layers.0.mlp.experts",
    )
    cache = OffloadMoeCache(
        num_layers=1, num_experts=num_experts, cache_size=cache_size, device=DEV
    )
    sources = {
        "gate_up": [
            torch.randn(num_experts, 2 * inter, hidden, dtype=torch.bfloat16) * 0.1
        ],
        "down": [torch.randn(num_experts, hidden, inter, dtype=torch.bfloat16) * 0.1],
    }
    cache.set_bank_sources(sources)
    # Populate cache slots at shifted positions (slot = expert + 10) to verify compaction indexing!
    for e in range(num_experts):
        s = e + 10
        cache.slot_for_id[0, e] = s
        cache.id_of_slot[s] = e
        cache.bank_caches["gate_up"][s].copy_(sources["gate_up"][0][e])
        cache.bank_caches["down"][s].copy_(sources["down"][0][e])
    layer.offload_cache = cache

    monkeypatch.setattr(moe_mod, "_SMALL_PREFILL_TOKENS", 128)
    monkeypatch.setattr(moe_mod, "_SMALL_PREFILL_GROUPED_ROUTES", 16)
    monkeypatch.setattr(
        cache,
        "ensure_experts",
        lambda lid, q: q.copy_(cache.slot_for_id[lid, q.long()].to(torch.int32)),
    )
    monkeypatch.setattr(cache, "copy_missing", lambda: None)

    captured: dict[str, object] = {}

    def ref_gemm(cache_, hs, w, ids, *, views, n, alphas, is_prefill):
        captured["n"] = n
        captured["is_prefill"] = is_prefill
        captured["view_rows"] = views[0].shape[0]
        # Compute exact reference SwiGLU MoE output from passed views and ids!
        out = torch.zeros_like(hs, dtype=torch.float32)
        gu_bank, dn_bank = views[0].float(), views[1].float()
        for tok in range(hs.shape[0]):
            x = hs[tok].float()
            for k in range(ids.shape[1]):
                idx = int(ids[tok, k].item())
                gu = gu_bank[idx] @ x
                g_part, u_part = gu[:inter], gu[inter:]
                act = torch.nn.functional.silu(g_part) * u_part
                out[tok] += float(w[tok, k].item()) * (dn_bank[idx] @ act)
        return out.to(hs.dtype)

    monkeypatch.setattr(layer, "_expert_gemm", ref_gemm)

    # 1. Narrow batch: 2 tokens * 4 = 8 routes (< 16 threshold) -> decode GEMV path (is_prefill=False, n=None)
    hs_narrow = torch.randn(2, hidden, dtype=torch.bfloat16)
    ids_narrow = torch.tensor([[1, 3, 5, 7], [1, 2, 3, 4]], dtype=torch.int32)
    w_narrow = torch.full((2, top_k), 0.25)
    out_narrow = layer._prefill_routed(hs_narrow, w_narrow, ids_narrow.clone())
    assert captured == {"n": None, "is_prefill": False, "view_rows": cache_size}

    # 2. Wide batch: 6 tokens * 4 = 24 routes (>= 16 threshold), touching only 5 distinct experts ->
    #    compacts views to [5, ...] and dispatches grouped Tensor Core path (is_prefill=True, n=5),
    #    and matches decode path bit-for-bit on the same inputs!
    hs_wide = torch.randn(6, hidden, dtype=torch.bfloat16)
    ids_wide = torch.tensor(
        [
            [1, 3, 5, 7],
            [3, 5, 7, 9],
            [1, 5, 7, 9],
            [1, 3, 7, 9],
            [1, 3, 5, 9],
            [1, 3, 5, 7],
        ],
        dtype=torch.int32,
    )
    w_wide = torch.full((6, top_k), 0.25)
    out_grouped = layer._prefill_routed(hs_wide, w_wide, ids_wide.clone())
    assert captured == {"n": 5, "is_prefill": True, "view_rows": 5}

    # Compare against uncompacted path (threshold disabled):
    monkeypatch.setattr(moe_mod, "_SMALL_PREFILL_GROUPED_ROUTES", 0)
    out_uncompacted = layer._prefill_routed(hs_wide, w_wide, ids_wide.clone())
    assert captured["is_prefill"] is False
    torch.testing.assert_close(out_grouped, out_uncompacted, rtol=0, atol=0)
    assert out_narrow.shape == (2, hidden)
