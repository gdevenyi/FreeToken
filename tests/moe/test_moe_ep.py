from __future__ import annotations

import json
from dataclasses import dataclass, replace

import pytest
import safetensors.torch
import torch
from freetoken.distributed import DistributedInfo
from freetoken.distributed import info as dist_info
from freetoken.layers.moe import MoELayer, OffloadMoELayer
from freetoken.layers.quantization.moe.base import MoEConfig
from freetoken.models.config import (
    FullAttentionGroupConfig,
    ModelConfig,
    RotaryConfig,
    moe_ep_enabled,
    moe_ep_slice,
)
from freetoken.models.nvfp4_banks import Nvfp4ExpertSourceSpec, iter_nvfp4_expert_pieces
from freetoken.moe.expert_pieces import stacked_expert_pieces
from freetoken.moe.hot_rankings import HotExpertRankings
from freetoken.moe.offload_cache import OffloadMoeCache
from freetoken.moe.offload_kernels import crf_hybrid_step_reference
from freetoken.server.args import parse_args


def _set_tp(rank: int, size: int) -> None:
    dist_info._TP_INFO = DistributedInfo(rank, size)


@pytest.fixture(autouse=True)
def _restore_tp_info(monkeypatch):
    monkeypatch.delenv("FREETOKEN_MOE_EP", raising=False)
    monkeypatch.setattr(dist_info, "_TP_INFO", DistributedInfo(0, 1))
    yield


@dataclass(frozen=True)
class _DummyQwen4Args:
    hc_count: int = 3
    hc_lowrank: int = 32
    ple_layer_ids: tuple[int, ...] = (0, 1)


def _dummy_model_config(
    *, num_experts: int = 8, inter: int = 128, moe_ep: bool = False
) -> ModelConfig:
    rot = RotaryConfig(
        head_dim=64, rotary_dim=64, max_position=4096, base=10000.0, scaling=None
    )
    return ModelConfig(
        num_layers=2,
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim=64,
        hidden_size=64,
        vocab_size=1024,
        intermediate_size=256,
        rms_norm_eps=1e-6,
        rotary_config=rot,
        hidden_act="silu",
        tie_word_embeddings=False,
        num_experts=num_experts,
        num_experts_per_tok=4,
        moe_intermediate_size=inter,
        norm_topk_prob=True,
        model_type="qwen4_exp",
        architectures=["Qwen4ExpForConditionalGeneration"],
        moe_enabled=True,
        moe_ep=moe_ep,
        attention_groups=(
            FullAttentionGroupConfig(
                name="qsa",
                layer_ids=(0, 1),
                num_kv_heads=2,
                head_dim=64,
                rotary_config=rot,
                index_head_dim=64,
                num_index_layers=2,
                index_ratio=4,
            ),
        ),
        qwen4_args=_DummyQwen4Args(),
    )


def test_moe_ep_slice_and_env(monkeypatch):
    assert not moe_ep_enabled()
    monkeypatch.setenv("FREETOKEN_MOE_EP", "1")
    assert moe_ep_enabled()

    assert moe_ep_slice(512, moe_ep=False, tp_info=DistributedInfo(0, 2)) == (1, 0, 512)
    assert moe_ep_slice(512, moe_ep=True, tp_info=DistributedInfo(0, 1)) == (1, 0, 512)
    assert moe_ep_slice(512, moe_ep=True, tp_info=DistributedInfo(0, 2)) == (2, 0, 256)
    assert moe_ep_slice(512, moe_ep=True, tp_info=DistributedInfo(1, 2)) == (
        2,
        256,
        256,
    )

    with pytest.raises(ValueError, match="divisible"):
        moe_ep_slice(513, moe_ep=True, tp_info=DistributedInfo(0, 2))


def test_model_config_and_moe_config_ep_geometry():
    cfg = _dummy_model_config(num_experts=512, inter=640, moe_ep=True)
    _set_tp(0, 2)
    assert cfg.num_experts == 512
    assert cfg.local_num_experts == 256
    assert cfg.local_moe_intermediate_size == 640

    layer0 = OffloadMoELayer(
        layer_id=0,
        num_experts=512,
        top_k=10,
        hidden_size=2560,
        intermediate_size=640,
        moe_ep=True,
    )
    assert layer0.global_num_experts == 512
    assert layer0.ep_size == 2
    assert layer0.ep_offset == 0
    assert layer0.num_experts == 256
    mcfg0 = MoEConfig.from_layer(layer0, None)
    assert mcfg0.ep_size == 2
    assert mcfg0.num_experts == 256
    assert mcfg0.local_intermediate == 640

    _set_tp(1, 2)
    assert cfg.local_num_experts == 256
    layer1 = OffloadMoELayer(
        layer_id=0,
        num_experts=512,
        top_k=10,
        hidden_size=2560,
        intermediate_size=640,
        moe_ep=True,
    )
    assert layer1.ep_offset == 256
    assert layer1.num_experts == 256


def test_remap_ep_routes_exact_partition_sum():
    """Verify Rank 0 ([0..4), full I) + Rank 1 ([4..8), full I) matches single-rank [0..8) exactly."""
    torch.manual_seed(7)
    E, H, I, M, K = 8, 32, 64, 6, 4
    gate = torch.randn(E, I, H, dtype=torch.float32) * 0.1
    up = torch.randn(E, I, H, dtype=torch.float32) * 0.1
    down = torch.randn(E, H, I, dtype=torch.float32) * 0.1
    x = torch.randn(M, H, dtype=torch.float32)
    logits = torch.randn(M, E, dtype=torch.float32)
    probs = torch.softmax(logits, dim=-1)
    topk_w, topk_ids = torch.topk(probs, K, dim=-1)
    topk_w = (topk_w / topk_w.sum(dim=-1, keepdim=True)).to(torch.float32)
    topk_ids = topk_ids.to(torch.int32)

    def eval_experts(x_in, g_w, u_w, d_w, w_sel, id_sel):
        out = torch.zeros_like(x_in)
        for m in range(x_in.shape[0]):
            for k in range(id_sel.shape[1]):
                eid = int(id_sel[m, k].item())
                if eid < 0:
                    continue
                gv = x_in[m] @ g_w[eid].T
                uv = x_in[m] @ u_w[eid].T
                act = torch.nn.functional.silu(gv) * uv
                out[m] += w_sel[m, k] * (act @ d_w[eid].T)
        return out

    ref = eval_experts(x, gate, up, down, topk_w, topk_ids)

    _set_tp(0, 2)
    l0 = MoELayer(E, K, H, I, allocate_experts=False, moe_ep=True)
    w0, ids0 = l0._remap_ep_routes(topk_w.clone(), topk_ids.clone())
    part0 = eval_experts(x, gate[:4], up[:4], down[:4], w0, ids0)

    _set_tp(1, 2)
    l1 = MoELayer(E, K, H, I, allocate_experts=False, moe_ep=True)
    w1, ids1 = l1._remap_ep_routes(topk_w.clone(), topk_ids.clone())
    part1 = eval_experts(x, gate[4:], up[4:], down[4:], w1, ids1)

    # Every route is valid on exactly one rank and inactive (-1, weight 0) on the other
    assert torch.equal(
        (ids0 >= 0) ^ (ids1 >= 0), torch.ones_like(ids0, dtype=torch.bool)
    )
    assert torch.all(w0[ids0 < 0] == 0.0)
    assert torch.all(w1[ids1 < 0] == 0.0)
    torch.testing.assert_close(part0 + part1, ref, rtol=1e-6, atol=1e-6)


def test_crf_hybrid_reference_handles_inactive_ep_routes():
    cache = OffloadMoeCache(
        num_layers=1,
        num_experts=8,
        cache_size=8,
        device=torch.device("cpu"),
        cache_policy="crf",
        decode_target="hybrid",
        ep_size=2,
        ep_offset=8,
    )
    cache.set_bank_sources(
        {
            "gate_up": [torch.zeros((8, 16, 8), dtype=torch.bfloat16)],
            "down": [torch.zeros((8, 8, 8), dtype=torch.bfloat16)],
        }
    )
    # Mix local experts (0, 3, 5) with remote-rank inactive routes (-1)
    ids = torch.tensor([[0, -1, 3, -1], [-1, 5, -1, 0]], dtype=torch.int32)
    w = torch.tensor([[0.4, 0.0, 0.6, 0.0], [0.0, 0.7, 0.0, 0.3]], dtype=torch.float32)
    routes = crf_hybrid_step_reference(cache, 0, ids, w, max_fetch=2, clamp=False)

    # Inactive (-1) routes must neither go to GPU (slot -1, w 0) nor go to CPU (cpu_id -1)
    assert int(routes.gpu_slots[0, 1].item()) == -1
    assert int(routes.cpu_ids[0, 1].item()) == -1
    assert float(routes.gpu_w[0, 1].item()) == 0.0
    assert int(routes.gpu_slots[1, 0].item()) == -1
    assert int(routes.cpu_ids[1, 0].item()) == -1
    # Out of 3 distinct local misses (0, 3, 5), max_fetch=2 admits 2 to GPU and sends 1 to CPU
    assert int(cache.num_missing_full.item()) == 3
    assert int(cache.num_indices.item()) == 2
    cpu_valid = routes.cpu_ids[routes.cpu_ids >= 0].tolist()
    assert len(cpu_valid) == 1
    assert cpu_valid[0] in (0, 3, 5)


def test_stacked_and_nvfp4_expert_pieces_ep_slicing(tmp_path):
    import re

    cfg = _dummy_model_config(num_experts=4, inter=32, moe_ep=True)
    gu = torch.arange(4 * 64 * 64, dtype=torch.bfloat16).reshape(4, 64, 64)
    dn = torch.arange(4 * 64 * 32, dtype=torch.bfloat16).reshape(4, 64, 32)
    raw = [
        ("model.layers.0.mlp.experts.gate_up_proj", gu),
        ("model.layers.0.mlp.experts.down_proj", dn),
        ("model.layers.1.mlp.experts.gate_up_proj", gu + 1),
        ("model.layers.1.mlp.experts.down_proj", dn + 1),
    ]

    _set_tp(0, 2)
    p0 = list(stacked_expert_pieces(raw, cfg))
    assert len(p0) == 2
    assert p0[0][:3] == (0, 0, 2)
    assert torch.equal(p0[0][3]["gate_up"], gu[:2])
    assert torch.equal(p0[0][3]["down"], dn[:2])

    _set_tp(1, 2)
    p1 = list(stacked_expert_pieces(raw, cfg))
    assert p1[0][:3] == (0, 0, 2)
    assert torch.equal(p1[0][3]["gate_up"], gu[2:4])
    assert torch.equal(p1[0][3]["down"], dn[2:4])

    # Also test iter_nvfp4_expert_pieces under EP=2 (unsharded I=32, disjoint experts [0,2) and [2,4))
    shard_tensors: dict[str, torch.Tensor] = {}
    for layer in range(2):
        for exp in range(4):
            val = float(layer * 10 + exp + 1)
            for proj, rows, cols in (
                ("gate_proj", 32, 64),
                ("up_proj", 32, 64),
                ("down_proj", 64, 32),
            ):
                base = f"model.language_model.layers.{layer}.mlp.experts.{exp}.{proj}"
                shard_tensors[f"{base}.weight"] = torch.full(
                    (rows, cols // 2), exp + 1, dtype=torch.uint8
                )
                shard_tensors[f"{base}.weight_scale"] = torch.ones(
                    (rows, cols // 16), dtype=torch.float8_e4m3fn
                )
                shard_tensors[f"{base}.weight_scale_2"] = torch.tensor(
                    val, dtype=torch.float32
                )

    shard_file = tmp_path / "model-00001-of-00001.safetensors"
    safetensors.torch.save_file(shard_tensors, str(shard_file))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {k: shard_file.name for k in shard_tensors}}),
        encoding="utf-8",
    )
    spec = Nvfp4ExpertSourceSpec(
        key_pattern=re.compile(
            r"^model\.language_model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
            r"(?P<proj>gate_proj|up_proj|down_proj)\.(?P<kind>weight|weight_scale|weight_scale_2)$"
        ),
        proj_to_role={"gate_proj": "gate", "up_proj": "up", "down_proj": "down"},
        layer_to_bank=lambda layer, _c: layer,
        desc="test nvfp4 ep",
    )

    _set_tp(1, 2)
    nv_pieces = list(
        iter_nvfp4_expert_pieces(
            str(tmp_path), cfg, spec, drop_page_cache=lambda _p: None
        )
    )
    # 2 layers * 2 local experts = 4 pieces, remapped to local expert indices {0, 1}, full I=32 (down cols//2 == 16, not 8!)
    assert len(nv_pieces) == 4
    by_le = {(l, e0): piece for l, e0, e1, piece in nv_pieces}
    assert sorted(by_le) == [(0, 0), (0, 1), (1, 0), (1, 1)]
    # Local expert 0 on rank 1 is global expert 2 (filled with byte 3), with full down width 32//2 = 16!
    assert by_le[(0, 0)]["down"].shape == (1, 64, 16)
    assert int(by_le[(0, 0)]["down"][0, 0, 0].item()) == 3
    assert int(by_le[(0, 1)]["down"][0, 0, 0].item()) == 4


def test_hot_rankings_ep_offset_and_mtp_draft_disables_ep():
    from freetoken.models.qwen4_exp.mtp import draft_config

    rankings = HotExpertRankings.from_dict(
        {
            "layers": {
                "0": {
                    "ranked_expert_ids": [5, 1, 7, 2, 6, 0, 4, 3],
                    "num_experts": 8,
                    "hit_rate_by_cache_size": {
                        "top_2": 0.5,
                        "top_4": 0.8,
                        "top_6": 0.95,
                    },
                },
                "1": {
                    "ranked_expert_ids": [0, 4, 1, 5, 2, 6, 3, 7],
                    "num_experts": 8,
                    "hit_rate_by_cache_size": {
                        "top_2": 0.3,
                        "top_4": 0.6,
                        "top_6": 0.85,
                    },
                },
            }
        }
    )
    assert rankings is not None
    # Rank 0 (ep_offset=0, num_experts=4): global [1, 2, 0, 3] -> local [1, 2, 0, 3]
    assert rankings.select_for_layer(0, 4, 3, ep_offset=0) == [1, 2, 0]
    # Rank 1 (ep_offset=4, num_experts=4): global [5, 7, 6, 4] -> local [1, 3, 2, 0]
    assert rankings.select_for_layer(0, 4, 3, ep_offset=4) == [1, 3, 2]
    caps = rankings.plan_layer_capacities(
        [0, 1], total_slots=4, num_experts=4, step=1, ep_size=2
    )
    assert sum(caps.values()) == 4

    # MTP draft_config always resets moe_ep=False because MTP resident experts shard along I
    cfg = _dummy_model_config(num_experts=8, inter=64, moe_ep=True)
    dcfg = draft_config(replace(cfg))
    assert dcfg.moe_ep is False


def test_parse_args_moe_ep_flag(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "qwen4_exp", "torch_dtype": "bfloat16"}),
        encoding="utf-8",
    )
    args, _ = parse_args(["--model", str(tmp_path), "--tp-size", "2", "--moe-ep"])
    assert args.moe_ep is True
