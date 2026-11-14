"""TP=2 support for Qwen3.8-Flash-Next MTP speculative decoding, draft weight sharding, keep_rows, and cross-rank lockstep."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

import freetoken.distributed.info as dist_info
from freetoken.distributed import DistributedInfo
from freetoken.kvcache.format import NONE
from freetoken.layers import ParallelLMHead, rotary, set_rope_device
from freetoken.layers.fp8_dynamic import E4M3_MAX, FP8, Fp8ParallelLMHead
from freetoken.layers.quantization import NameMap, QuantConfig, set_quant_config
from freetoken.models.loader import shard_tensor
from freetoken.models.qwen4_exp.config import parse_config
from freetoken.models.qwen4_exp.mtp import Qwen4ExpMTPDraft, draft_config
from freetoken.models.register import get_model_spec
from freetoken.spec.mtp.proposer import MTPProposer, with_resident_moe
from freetoken.spec.proposer import SpecProposal
from freetoken.spec.runner import SpecRunner
from freetoken.utils import cached_load_hf_config, torch_dtype

from .common import NVIDIA_NVFP4, QWEN_FP8, hf_config


def _hf_dict(quant_cfg: dict | None = None, **overrides) -> dict:
    defaults = dict(
        num_layers=4,
        head_dim=64,
        num_q=4,
        num_kv=2,
        index_head_dim=64,
        index_heads=2,
        budget=16,
        hidden=128,
        max_position=4096,
        rope_theta=10000.0,
        moe_intermediate_size=128,
        shared_expert_intermediate_size=128,
    )
    defaults.update(overrides)
    cfg = hf_config(**defaults)
    text = vars(cfg.text_config).copy()
    out = {
        "model_type": "qwen4_exp",
        "architectures": ["Qwen4ExpForConditionalGeneration"],
        "text_config": text,
    }
    if quant_cfg is not None:
        out["quantization_config"] = quant_cfg
    return out


def _quantize_fp8_block(w: torch.Tensor, block: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    rows, cols = w.shape
    assert rows % block == 0 and cols % block == 0, (rows, cols, block)
    blocks = w.float().view(rows // block, block, cols // block, block)
    amax = blocks.abs().amax(dim=(1, 3)).clamp_min(1e-6)
    scale = amax / E4M3_MAX
    s_full = scale.repeat_interleave(block, dim=0).repeat_interleave(block, dim=1)
    q = (w.float() / s_full).clamp(-E4M3_MAX, E4M3_MAX).to(FP8)
    return q, scale.to(torch.float32)


def _write_mtp_checkpoint(
    folder: Path,
    hf_dict: dict,
    *,
    expert_mode: str = "stacked_bf16",
    dense_fp8_block: bool = False,
) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "config.json").write_text(json.dumps(hf_dict), encoding="utf-8")

    text = hf_dict["text_config"]
    H = text["hidden_size"]
    hq = text["num_attention_heads"]
    hkv = text["num_key_value_heads"]
    hd = text["head_dim"]
    idx_h = text["indexer_n_heads"]
    idx_hd = text["indexer_head_dim"]
    E = text["num_experts"]
    I = text["moe_intermediate_size"]
    SI = text["shared_expert_intermediate_size"]
    hc = text["hc_count"]
    lr = text["hc_lowrank"]
    W = hc * H

    g = torch.Generator().manual_seed(1234)

    def bf16(*shape):
        return (torch.randn(*shape, generator=g, dtype=torch.float32) * 0.05).to(torch.bfloat16)

    tensors: dict[str, torch.Tensor] = {}

    def put_linear(name: str, rows: int, cols: int, *, allow_fp8_block: bool = True):
        w = bf16(rows, cols)
        if dense_fp8_block and allow_fp8_block:
            q, s = _quantize_fp8_block(w, 128)
            tensors[f"{name}.weight"] = q
            tensors[f"{name}.weight_scale_inv"] = s
        else:
            tensors[f"{name}.weight"] = w

    sa = "mtp.layers.0.self_attn"
    mlp = "mtp.layers.0.mlp"
    tensors["mtp.pre_fc_norm_embedding.weight"] = bf16(H)
    tensors["mtp.pre_fc_norm_hidden.weight"] = bf16(W)
    put_linear("mtp.fc_embedding", H, H)
    put_linear("mtp.fc_hidden", H, H)

    tensors[f"{sa}.q_norm.weight"] = bf16(hd)
    tensors[f"{sa}.k_norm.weight"] = bf16(hd)
    tensors[f"{sa}.indexer.q_layernorm.weight"] = bf16(idx_hd)
    tensors[f"{sa}.indexer.k_layernorm.weight"] = bf16(idx_hd)
    put_linear(f"{sa}.q_proj", hq * 2 * hd, H)
    put_linear(f"{sa}.k_proj", hkv * hd, H)
    put_linear(f"{sa}.v_proj", hkv * hd, H)
    put_linear(f"{sa}.o_proj", H, hq * hd)
    put_linear(f"{sa}.indexer.index_qk_proj", (idx_h + 1) * idx_hd, H)

    tensors[f"{mlp}.gate.weight"] = bf16(E, H)
    tensors[f"{mlp}.shared_expert_gate.weight"] = bf16(1, H)
    put_linear(f"{mlp}.shared_expert.gate_proj", SI, H)
    put_linear(f"{mlp}.shared_expert.up_proj", SI, H)
    put_linear(f"{mlp}.shared_expert.down_proj", H, SI)

    for hc_name in ("mtp.layers.0.attn_hyper_connection", "mtp.layers.0.mlp_hyper_connection"):
        tensors[f"{hc_name}.hc_norm.weight"] = bf16(W)
        put_linear(f"{hc_name}.input_mix_weight_down", lr, W, allow_fp8_block=False)
        put_linear(f"{hc_name}.block_inject_weight", hc, W, allow_fp8_block=False)
        put_linear(f"{hc_name}.input_mix_weight_up", W, lr, allow_fp8_block=False)

    tensors["mtp.hyper_connection_mixer.hc_norm.weight"] = bf16(W)
    put_linear("mtp.hyper_connection_mixer.input_mix_weight_down", lr, W, allow_fp8_block=False)
    put_linear("mtp.hyper_connection_mixer.input_mix_weight_up", W, lr, allow_fp8_block=False)

    if expert_mode == "stacked_bf16":
        tensors[f"{mlp}.experts.gate_up_proj"] = bf16(E, 2 * I, H)
        tensors[f"{mlp}.experts.down_proj"] = bf16(E, H, I)
    elif expert_mode == "per_expert_fp8":
        for e in range(E):
            for proj, r, c in (("gate_proj", I, H), ("up_proj", I, H), ("down_proj", H, I)):
                q, s = _quantize_fp8_block(bf16(r, c), 128)
                tensors[f"{mlp}.experts.{e}.{proj}.weight"] = q
                tensors[f"{mlp}.experts.{e}.{proj}.weight_scale_inv"] = s
    else:
        raise ValueError(expert_mode)

    save_file(tensors, str(folder / "model.safetensors"))


@pytest.fixture
def cpu_rope():
    prev = rotary._ROPE_DEVICE
    set_rope_device(torch.device("cpu"))
    try:
        yield
    finally:
        rotary.get_rope.cache_clear()
        set_rope_device(prev)


def _dummy_target():
    return SimpleNamespace(model=SimpleNamespace(embed_tokens=object()))


def test_mtp_adjust_config_and_budget_tp_gating(tmp_path):
    ckpt = tmp_path / "qwen4_mtp"
    hf = _hf_dict()
    _write_mtp_checkpoint(ckpt, hf)

    hf_obj = cached_load_hf_config(str(ckpt))
    mc = parse_config(hf_obj)
    cfg = SimpleNamespace(
        tp_info=DistributedInfo(0, 2),
        hf_config=hf_obj,
        model_spec=get_model_spec("Qwen4ExpForConditionalGeneration"),
        model_config=mc,
        use_dummy_weight=False,
        model_path=str(ckpt),
        speculative_draft_model_path=None,
        speculative_token_map=None,
        speculative_num_steps=3,
        max_running_req=8,
        dtype=torch.bfloat16,
        page_size=64,
        kv_format=NONE,
    )
    # Qwen4ExpMTPDraft declares supports_tp=True -> TP=2 accepted and budget computed cleanly
    MTPProposer.adjust_config(cfg, lambda k, v: setattr(cfg, k, v))
    fixed, per_tok = MTPProposer.budget(cfg)
    assert fixed > 0 and per_tok > 0

    # GGUF Strata MTP pack must still reject TP > 1 at config validation time
    gguf_pack = tmp_path / "draft.gguf"
    gguf_pack.write_bytes(b"GGUF")
    cfg_gguf = replace_ns(cfg, speculative_draft_model_path=str(gguf_pack))
    with pytest.raises(ValueError, match="Strata GGUF MTP packs require --tp 1"):
        MTPProposer.adjust_config(cfg_gguf, lambda k, v: None)

    # Families without supports_tp (e.g. Qwen3_5MoeForConditionalGeneration) still reject TP > 1
    cfg_q35 = replace_ns(
        cfg,
        hf_config=SimpleNamespace(architectures=["Qwen3_5MoeForConditionalGeneration"]),
        model_spec=get_model_spec("Qwen3_5MoeForConditionalGeneration"),
    )
    with pytest.raises(ValueError, match="speculative decoding needs --tp 1"):
        MTPProposer.adjust_config(cfg_q35, lambda k, v: None)


def replace_ns(ns: SimpleNamespace, **updates) -> SimpleNamespace:
    d = vars(ns).copy()
    d.update(updates)
    return SimpleNamespace(**d)


@pytest.mark.parametrize("fp8_dense", ["0", "1"])
@pytest.mark.parametrize("tp_size", [1, 2])
def test_nvidia_nvfp4_mtp_loads_across_tp_and_fp8_dense(tmp_path, monkeypatch, cpu_rope, fp8_dense, tp_size):
    """NVIDIA_NVFP4 ships bf16 mtp.* dense linears + 128x128 FP8_PB_WO mtp experts (whose local_intermediate=64 at TP=2 is not 128-divisible)."""
    monkeypatch.setenv("FREETOKEN_FP8_DENSE", fp8_dense)
    ckpt = tmp_path / f"nvfp4_tp{tp_size}_fp8{fp8_dense}"
    hf = _hf_dict(quant_cfg=NVIDIA_NVFP4)
    _write_mtp_checkpoint(ckpt, hf, expert_mode="per_expert_fp8", dense_fp8_block=False)

    hf_obj = cached_load_hf_config(str(ckpt))
    packed = get_model_spec("Qwen4ExpForConditionalGeneration").packed_modules_mapping
    quant = QuantConfig.from_hf(hf_obj, name_map=NameMap(packed=packed))
    set_quant_config(quant)
    mc = with_resident_moe(replace(parse_config(hf_obj), quant=quant))

    if tp_size == 1 and not torch.cuda.is_available():
        # At TP=1, FP8_PB_WO experts select TritonFp8BlockMoEKernel; verify TP=2 CPU load and TP=1 iter_mtp_weights
        monkeypatch.setattr(dist_info, "_TP_INFO", DistributedInfo(0, 1))
        from freetoken.models.qwen4_exp.weight import iter_mtp_weights

        with torch.device("meta"), torch_dtype(torch.bfloat16):
            meta_draft = Qwen4ExpMTPDraft(mc, _dummy_target())
        emitted = dict(iter_mtp_weights(str(ckpt), torch.device("cpu")))
        assert set(emitted) == set(meta_draft.state_dict())
        return

    for rank in range(tp_size):
        monkeypatch.setattr(dist_info, "_TP_INFO", DistributedInfo(rank, tp_size))
        draft = Qwen4ExpMTPDraft.load(mc, _dummy_target(), str(ckpt), torch.device("cpu"), torch.bfloat16)
        moe = draft.layers.op_list[0].mlp.experts
        if tp_size > 1:
            # 128 // 2 = 64 local_intermediate -> dequantized to BF16 FusedMoEKernel per rank
            assert moe._expert_banks["gate_up"].shape == (8, 2 * (128 // tp_size), 128)
            assert moe._expert_banks["down"].shape == (8, 128, 128 // tp_size)
        attn = draft.layers.op_list[0].self_attn
        if fp8_dense == "1":
            assert attn.qkv_proj.weight.dtype is FP8
            assert attn.o_proj.weight.dtype is FP8


def test_qwen_fp8_mtp_dequantizes_and_shards_at_tp2(tmp_path, monkeypatch, cpu_rope):
    """QWEN_FP8 ships 128x128 block-fp8 for both mtp.* dense projections and mtp routed experts."""
    monkeypatch.setenv("FREETOKEN_FP8_DENSE", "0")
    qwen_fp8 = dict(QWEN_FP8)
    qwen_fp8["modules_to_not_convert"] = [
        *QWEN_FP8["modules_to_not_convert"],
        "*hyper_connection*",
    ]
    ckpt = tmp_path / "qwen_fp8_tp2"
    hf = _hf_dict(quant_cfg=qwen_fp8, hidden=128, head_dim=64, index_head_dim=64, index_heads=3)
    _write_mtp_checkpoint(ckpt, hf, expert_mode="per_expert_fp8", dense_fp8_block=True)

    hf_obj = cached_load_hf_config(str(ckpt))
    packed = get_model_spec("Qwen4ExpForConditionalGeneration").packed_modules_mapping
    quant = QuantConfig.from_hf(hf_obj, name_map=NameMap(packed=packed))
    set_quant_config(quant)
    mc = with_resident_moe(replace(parse_config(hf_obj), quant=quant))

    shards = []
    for rank in range(2):
        monkeypatch.setattr(dist_info, "_TP_INFO", DistributedInfo(rank, 2))
        draft = Qwen4ExpMTPDraft.load(mc, _dummy_target(), str(ckpt), torch.device("cpu"), torch.bfloat16)
        shards.append(draft)

    # Replicated projections match bit-for-bit across ranks; sharded ones split cleanly in half
    assert torch.equal(shards[0].fc_embedding.weight, shards[1].fc_embedding.weight)
    assert shards[0].layers.op_list[0].self_attn.o_proj.weight.shape == (128, 128)
    assert shards[0].layers.op_list[0].mlp.shared_expert.down_proj.weight.shape == (128, 64)
    assert shards[0].layers.op_list[0].mlp.experts._expert_banks["gate_up"].shape == (8, 128, 128)


def test_tp2_mtp_mlp_and_head_input_numerical_equivalence(tmp_path, monkeypatch, cpu_rope):
    """TP=2 sharded MTP MLP + HC mixer + fc_embedding/fc_hidden produces the same output as TP=1."""
    monkeypatch.setenv("FREETOKEN_FP8_DENSE", "0")
    ckpt = tmp_path / "bf16_mtp"
    hf = _hf_dict()
    _write_mtp_checkpoint(ckpt, hf, expert_mode="stacked_bf16", dense_fp8_block=False)

    hf_obj = cached_load_hf_config(str(ckpt))
    set_quant_config(None)
    mc = with_resident_moe(parse_config(hf_obj))

    monkeypatch.setattr(dist_info, "_TP_INFO", DistributedInfo(0, 1))
    draft_tp1 = Qwen4ExpMTPDraft.load(mc, _dummy_target(), str(ckpt), torch.device("cpu"), torch.bfloat16)

    ranks = []
    for rank in range(2):
        monkeypatch.setattr(dist_info, "_TP_INFO", DistributedInfo(rank, 2))
        ranks.append(Qwen4ExpMTPDraft.load(mc, _dummy_target(), str(ckpt), torch.device("cpu"), torch.bfloat16))

    from freetoken.models.qwen4_exp.hc import grouped_plus_one_rms_norm

    # Compare fc_embedding + fc_hidden + shared_expert + routed expert weights + head_input
    torch.manual_seed(7)
    x = torch.randn(3, 128, dtype=torch.bfloat16)
    prev = torch.randn(3, 4 * 128, dtype=torch.bfloat16)

    def pre_layer(d):
        en = grouped_plus_one_rms_norm(x, d.pre_fc_norm_embedding.weight, d.pre_fc_norm_embedding.eps, 1)
        hn = grouped_plus_one_rms_norm(prev, d.pre_fc_norm_hidden.weight, d.pre_fc_norm_hidden.eps, 1)
        e = d.fc_embedding.forward(en)
        r = d.fc_hidden.forward(hn.view(3 * 4, 128)).view(3, 4, 128) + e.unsqueeze(1)
        return r.view(3, -1)

    r1 = pre_layer(draft_tp1)
    assert torch.equal(r1, pre_layer(ranks[0]))
    assert torch.equal(r1, pre_layer(ranks[1]))

    # Shared expert TP=2 sum matches TP=1 shared expert
    se_tp1 = draft_tp1.layers.op_list[0].mlp.shared_expert
    se_r0 = ranks[0].layers.op_list[0].mlp.shared_expert
    se_r1 = ranks[1].layers.op_list[0].mlp.shared_expert

    def run_shared(se, inp):
        gu = inp.float() @ se.gate_up_proj.weight.float().T
        g_part, u_part = gu.chunk(2, dim=-1)
        act = torch.nn.functional.silu(g_part) * u_part
        return act @ se.down_proj.weight.float().T

    torch.testing.assert_close(
        run_shared(se_r0, x) + run_shared(se_r1, x),
        run_shared(se_tp1, x),
        rtol=1e-2,
        atol=1e-2,
    )

    # Routed expert banks: concatenating TP=2 shards reproduces TP=1 expert banks bit-for-bit
    v_tp1 = draft_tp1.layers.op_list[0].mlp.experts._expert_banks
    v0 = ranks[0].layers.op_list[0].mlp.experts._expert_banks
    v1 = ranks[1].layers.op_list[0].mlp.experts._expert_banks
    g0, u0 = v0["gate_up"].chunk(2, dim=1)
    g1, u1 = v1["gate_up"].chunk(2, dim=1)
    assert torch.equal(torch.cat([g0, g1], dim=1), v_tp1["gate_up"][:, :128, :])
    assert torch.equal(torch.cat([u0, u1], dim=1), v_tp1["gate_up"][:, 128:, :])
    assert torch.equal(torch.cat([v0["down"], v1["down"]], dim=2), v_tp1["down"])

    # head_input matches identically across ranks
    assert torch.equal(draft_tp1.head_input(r1), ranks[0].head_input(r1))


def test_parallel_lm_head_keep_rows_tp2_even_and_odd_vocab(monkeypatch):
    """ParallelLMHead.keep_rows at TP=2 builds a replicated (tp_size=1) draft head matching full logits[:, ids]."""
    torch.manual_seed(99)
    vocab, hidden = 301, 32  # intentionally odd vocab size
    full_weight = torch.randn(vocab, hidden, dtype=torch.bfloat16)
    ids = torch.tensor([0, 5, 150, 151, 300, 42, 299], dtype=torch.int64)

    heads = []
    for rank in range(2):
        monkeypatch.setattr(dist_info, "_TP_INFO", DistributedInfo(rank, 2))
        h = ParallelLMHead(vocab, hidden, prefix="lm_head")
        shard = shard_tensor("lm_head.weight", full_weight, rank=rank, world_size=2, num_kv_heads=None)
        h.weight = shard
        heads.append(h)

    # Simulate synchronous two-rank all_reduce across the two shards
    recorded = []

    class _SyncComm:
        def __init__(self, idx):
            self.idx = idx

        def all_reduce(self, x):
            recorded.append(x.clone())
            return x

        def all_gather(self, x):
            raise AssertionError("replicated kept_rows head must never call all_gather")

    # Run pass 1 to collect per-rank local buffers, then pass 2 returning their sum
    for idx, h in enumerate(heads):
        h._comm = _SyncComm(idx)
        h.keep_rows(ids, None)
    summed = recorded[0] + recorded[1]
    for h in heads:
        h._comm = SimpleNamespace(
            all_reduce=lambda _x, s=summed: s.clone(),
            all_gather=lambda _x: (_ for _ in ()).throw(AssertionError("unexpected all_gather")),
        )
        h.keep_rows(ids, None)

    x = torch.randn(4, hidden, dtype=torch.bfloat16)
    ref_full = x @ full_weight.T
    for h in heads:
        kept_head, padded_ids = h.kept_rows
        assert kept_head.tp_size == 1
        assert padded_ids.numel() == 128
        got = kept_head.logits(x)
        torch.testing.assert_close(got, ref_full[:, padded_ids], rtol=0, atol=0)


def test_fp8_parallel_lm_head_keep_rows_tp1_and_tp2(monkeypatch):
    """Fp8ParallelLMHead.keep_rows handles distinct per-rank weight_scales at TP=2 and produces a replicated FP8 head."""
    torch.manual_seed(11)
    vocab, hidden = 260, 32
    # Give the second half a 10x larger dynamic range so rank 0 and rank 1 have very different weight_scales
    w0 = torch.randn(130, hidden, dtype=torch.float32) * 0.2
    w1 = torch.randn(130, hidden, dtype=torch.float32) * 2.0
    full_w = torch.cat([w0, w1], dim=0).to(torch.bfloat16)
    ids = torch.tensor([3, 129, 130, 200, 259], dtype=torch.int64)

    def quant_per_tensor_cpu(w: torch.Tensor):
        wf = w.float()
        s = (wf.abs().amax() / E4M3_MAX).clamp_min(1e-12)
        return (wf / s).clamp(-E4M3_MAX, E4M3_MAX).to(FP8), s.reshape(())

    # TP=1 check
    monkeypatch.setattr(dist_info, "_TP_INFO", DistributedInfo(0, 1))
    h1 = Fp8ParallelLMHead(vocab, hidden)
    h1.weight, h1.weight_scale = quant_per_tensor_cpu(full_w)
    h1.keep_rows(ids, None)
    kept1, padded1 = h1.kept_rows
    assert kept1.tp_size == 1
    assert kept1.weight.shape == (128, hidden)
    assert kept1.weight.dtype is FP8

    # TP=2 check with distinct shard scales
    heads = []
    partials = []
    for rank, shard_bf16 in enumerate((full_w[:130], full_w[130:])):
        monkeypatch.setattr(dist_info, "_TP_INFO", DistributedInfo(rank, 2))
        h = Fp8ParallelLMHead(vocab, hidden)
        h.weight, h.weight_scale = quant_per_tensor_cpu(shard_bf16)
        h._comm = SimpleNamespace(all_reduce=lambda x, p=partials: (p.append(x.clone()), x)[1])
        h.keep_rows(ids, None)
        heads.append(h)

    assert not torch.allclose(heads[0].weight_scale, heads[1].weight_scale)
    summed_bf16 = partials[0] + partials[1]
    for h in heads:
        h._comm = SimpleNamespace(all_reduce=lambda _x, s=summed_bf16: s.clone())
        h.keep_rows(ids, None)

    for h in heads:
        kept, padded = h.kept_rows
        assert kept.tp_size == 1
        deq = (kept.weight.float() * kept.weight_scale)
        expected = full_w[padded].float()
        torch.testing.assert_close(deq, expected, rtol=0.1, atol=0.15)


def test_cross_rank_lockstep_in_draft_token_and_verify_step(monkeypatch):
    """Rank 0's draft tokens and verify [out, emitted] decisions override rank 1 via all_reduce."""
    # 1. MTPProposer._draft_token lockstep
    def make_proposer(rank: int, local_logits: torch.Tensor, comm):
        p = object.__new__(MTPProposer)
        p.draft = SimpleNamespace(head_input=lambda h: h)
        p.lm_head = SimpleNamespace(logits=lambda h: local_logits)
        p.hot_ids = torch.tensor([10, 20, 30, 40], dtype=torch.int64)
        p._tp_info = DistributedInfo(rank, 2)
        p._comm = comm
        return p

    # Rank 0 argmax -> index 1 (token 20), index 3 (token 40); Rank 1 argmax -> index 2 (token 30), index 0 (token 10)
    logits_r0 = torch.tensor([[0.1, 1.0, 0.99, 0.0], [0.0, 0.2, 0.99, 1.0]])
    logits_r1 = torch.tensor([[0.1, 0.99, 1.0, 0.0], [1.0, 0.2, 0.99, 0.0]])
    sent = []
    rec_comm = SimpleNamespace(all_reduce=lambda t: (sent.append(t.clone()), t)[1])
    make_proposer(0, logits_r0, rec_comm)._draft_token(torch.zeros(2, 4))
    make_proposer(1, logits_r1, rec_comm)._draft_token(torch.zeros(2, 4))
    reduced = sent[0] + sent[1]
    assert reduced.tolist() == [20, 40]
    sync_comm = SimpleNamespace(all_reduce=lambda _t: reduced.clone())
    assert make_proposer(0, logits_r0, sync_comm)._draft_token(torch.zeros(2, 4)).tolist() == [20, 40]
    assert make_proposer(1, logits_r1, sync_comm)._draft_token(torch.zeros(2, 4)).tolist() == [20, 40]

    # 2. SpecRunner.verify_step lockstep (even when stochastic reject differs across ranks)
    import freetoken.spec.runner as runner_mod

    monkeypatch.setattr(runner_mod.torch.cuda, "Event", lambda: SimpleNamespace(record=lambda _s: None))

    decisions = [
        (torch.tensor([[101, 102, 103]], dtype=torch.int64), torch.tensor([2], dtype=torch.int64)),  # rank 0: accept 2
        (torch.tensor([[101, 999, 0]], dtype=torch.int64), torch.tensor([0], dtype=torch.int64)),    # rank 1: reject first!
    ]
    post_seen = []
    sent_verify = []

    def run_verify(rank: int, comm):
        monkeypatch.setattr(runner_mod, "reject", lambda *_a, r=rank: decisions[r])
        sr = object.__new__(SpecRunner)
        sr.k = 2
        sr._tp_info = DistributedInfo(rank, 2)
        sr._comm = comm
        sr._host = torch.zeros(2, 4, dtype=torch.int64)
        sr._proposal = SpecProposal(torch.tensor([[101, 102]], dtype=torch.int64))
        sr.feature_sink = lambda _b: None
        sr.proposer = SimpleNamespace(post_verify=lambda _b, out, em: post_seen.append((rank, out.tolist(), em.tolist())))
        sr.engine = SimpleNamespace(
            forward_logits=lambda _b: torch.zeros(3, 16),
            sampler=object(),
            stream=None,
        )
        batch = SimpleNamespace(size=1, input_ids=torch.tensor([50, 101, 102], dtype=torch.int64), features=None)
        return sr.verify_step(batch, None)

    rec_v = SimpleNamespace(all_reduce=lambda t: (sent_verify.append(t.clone()), t)[1])
    run_verify(0, rec_v)
    run_verify(1, rec_v)
    v_reduced = sent_verify[0] + sent_verify[1]
    post_seen.clear()

    sync_v = SimpleNamespace(all_reduce=lambda _t: v_reduced.clone())
    out0 = run_verify(0, sync_v)
    out1 = run_verify(1, sync_v)

    assert out0.next_tokens_cpu.tolist() == [[101, 102, 103]]
    assert out1.next_tokens_cpu.tolist() == [[101, 102, 103]]
    assert out0.emitted_cpu.tolist() == [2]
    assert out1.emitted_cpu.tolist() == [2]
    assert post_seen == [(0, [[101, 102, 103]], [2]), (1, [[101, 102, 103]], [2])]


def test_sidecar_names_map_to_the_nextn_block():
    # llama.cpp stores fc_embedding | fc_hidden as one eh_proj and the draft's final mixer as hc_head_*
    from freetoken.models.qwen4_exp.gguf import sidecar_to_gguf

    tensors = {"blk.48.nextn.eh_proj.weight": 8, "blk.48.attn_q.weight": 8}
    assert sidecar_to_gguf("mtp.fc_embedding", tensors) == ("blk.48.nextn.eh_proj",)
    assert sidecar_to_gguf("mtp.fc_hidden", tensors) == ("blk.48.nextn.eh_proj",)
    assert sidecar_to_gguf("mtp.hyper_connection_mixer.input_mix_weight_down", tensors) == ("blk.48.nextn.hc_head_down",)
    assert sidecar_to_gguf("mtp.hyper_connection_mixer.input_mix_weight_up", tensors) == ("blk.48.nextn.hc_head_up",)
    assert sidecar_to_gguf("mtp.layers.0.self_attn.q_proj", tensors) == ("blk.48.attn_q",)
    assert sidecar_to_gguf("mtp.layers.0.mlp.experts.0.down_proj", tensors) == ("blk.48.ffn_down_exps",)
