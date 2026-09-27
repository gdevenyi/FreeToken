"""Skeleton tests: the frozen qwen4_exp interfaces and their torch references.

Everything runs on a scaled-down copy of the real geometry (4 layers, full attention on layer 3,
PLE on layer 1, hc_count 4, 3-gram hash) so the shapes and the layer split are the shipping ones.
The hyper-connection and PLE references transcribed here are HF ``modeling_qwen4_exp.py``
(``Qwen4ExpTextGatedResidual``:941, ``Qwen4ExpTextNGramEmbedding``:1018, ``Qwen4ExpTextPLELayer``
:1117), so the torch implementations are checked against the math, not against themselves.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from freetoken.layers import BaseOP, LinearReplicated
from freetoken.models.config import ModelConfig
from freetoken.models.qwen4_exp.config import parse_config
from freetoken.models.qwen4_exp.hc import GatedResidual
from freetoken.models.qwen4_exp.ple import GpuResidentTable, PLELayer, PLEMetadata

from .common import EOS, hash_constants, requires_cuda, toy_hf_config


def _config(num_layers: int = 4) -> ModelConfig:
    return parse_config(toy_hf_config(num_layers))


def _fill(op, gen: torch.Generator, scale: float = 0.05) -> None:
    """Random floats / zeroed ints for every state-dict tensor of an op tree."""
    for tensor in op.state_dict().values():
        if tensor.is_floating_point():
            tensor.normal_(0.0, scale, generator=gen)
        else:
            tensor.zero_()


def _group_norm(x, weight, eps, groups):
    xf = x.float().reshape(*x.shape[:-1], groups, -1)
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (xf.flatten(-2) * (1.0 + weight.float())).type_as(x)


# --------------------------------------------------------------------------------------
# hyper-connections
# --------------------------------------------------------------------------------------


def _hf_gated_residual(hc, R, w_norm, w_down, w_up, w_inject, hidden, eps):
    xn = _group_norm(R, w_norm, eps, hc)
    mix = F.silu(F.linear(xn, w_down) / hc)
    mix = torch.sigmoid(F.linear(mix, w_up)).unflatten(-1, (hc, hidden))
    mixed = (mix * xn.unflatten(-1, (hc, hidden))).mean(-2)
    if w_inject is None:
        return mixed, None
    return mixed, 2 * torch.sigmoid(F.linear(xn, w_inject) / hc)


@pytest.mark.parametrize("tokens", [1, 7])
def test_hc_mix_and_combine_match_hf(tokens: int):
    torch.manual_seed(0)
    config = _config()
    args = config.qwen4_args
    hc = GatedResidual(config)
    _fill(hc, torch.Generator().manual_seed(1))

    R = torch.randn(tokens, args.ple_state_width)
    y = torch.randn(tokens, args.hidden_size)
    x, s = hc.mix(R)
    got = hc.combine(R, y, s)

    merged = hc.input_mix_weight_down_block_inject.weight
    ref_x, ref_inject = _hf_gated_residual(
        args.hc_count,
        R,
        hc.hc_norm.weight,
        merged[: args.hc_lowrank],
        hc.input_mix_weight_up.weight,
        merged[args.hc_lowrank : args.hc_lowrank + args.hc_count],
        args.hidden_size,
        config.rms_norm_eps,
    )
    ref = R.unflatten(-1, (args.hc_count, args.hidden_size))
    ref = (ref + y.unsqueeze(-2) * ref_inject.unsqueeze(-1)).flatten(-2)

    assert torch.allclose(x, ref_x, rtol=1e-5, atol=1e-6)
    assert torch.allclose(got, ref, rtol=1e-5, atol=1e-6)


def test_hc_merged_gemm_layout_and_top_level_mixer():
    config = _config()
    args = config.qwen4_args
    hc = GatedResidual(config)
    # 320 + 4 lowrank/inject rows padded to a multiple of 16 in the real config
    assert hc.pad_size == (-(args.hc_lowrank + args.hc_count)) % 16
    merged = hc.input_mix_weight_down_block_inject.weight
    assert merged.shape == (
        args.hc_lowrank + args.hc_count + hc.pad_size,
        args.ple_state_width,
    )
    assert set(hc.state_dict()) == {
        "hc_norm.weight",
        "input_mix_weight_down_block_inject.weight",
        "input_mix_weight_up.weight",
    }

    mixer = GatedResidual(config, use_combine=False)
    _fill(mixer, torch.Generator().manual_seed(2))
    assert set(mixer.state_dict()) == {
        "hc_norm.weight",
        "input_mix_weight_down.weight",
        "input_mix_weight_up.weight",
    }
    R = torch.randn(5, args.ple_state_width)
    x, s = mixer.mix(R)
    assert s is None
    ref_x, ref_inject = _hf_gated_residual(
        args.hc_count,
        R,
        mixer.hc_norm.weight,
        mixer.input_mix_weight_down.weight,
        mixer.input_mix_weight_up.weight,
        None,
        args.hidden_size,
        config.rms_norm_eps,
    )
    assert ref_inject is None
    assert torch.allclose(x, ref_x, rtol=1e-5, atol=1e-6)


# --------------------------------------------------------------------------------------
# PLE
# --------------------------------------------------------------------------------------


def _hf_shift_right(tokens, shift, eos):
    if shift == 0:
        return tokens
    batch, seq_len = tokens.shape
    positions = torch.arange(seq_len)
    eos_positions = torch.where(tokens == eos, positions, torch.full_like(positions, -1))
    previous = torch.cummax(eos_positions, dim=1).values
    previous = torch.cat([eos_positions.new_full((batch, 1), -1), previous[:, :-1]], dim=1)
    in_segment = positions.unsqueeze(0) - (previous + 1)
    source = positions - shift
    shifted = tokens.gather(1, source.clamp_min(0).unsqueeze(0).expand(batch, -1))
    valid = (in_segment >= shift) & (source.unsqueeze(0) >= 0)
    return torch.where(valid, shifted, tokens.new_full((), eos))


def _hf_ngram_ids(tokens, context, args, multipliers, sizes, offsets):
    """HF Qwen4ExpTextNGramEmbedding id computation over a dense [B, L] batch."""
    history = torch.cat([context, tokens], dim=-1)
    shifted = [_hf_shift_right(history, s, args.ngram_boundary_token_id) for s in range(args.ngram_size)]
    blocks = []
    for ngram in range(2, args.ngram_size + 1):
        start = (ngram - 2) * args.heads_per_ngram
        end = start + args.heads_per_ngram
        mixed = shifted[0] * multipliers[0]
        for position in range(1, ngram):
            mixed = torch.bitwise_xor(mixed, shifted[position] * multipliers[position])
        ids = torch.remainder(mixed.unsqueeze(-1), sizes[start:end].view(1, 1, -1))
        blocks.append(ids + offsets[start:end].view(1, 1, -1))
    return torch.cat(blocks, dim=-1)[:, -tokens.shape[1] :]


def _ragged(sequences, contexts, args, device="cpu"):
    """Ragged PLEMetadata (prefill) for a list of per-request token lists."""
    lens = [len(s) for s in sequences]
    cu = torch.tensor([0, *lens], dtype=torch.int64).cumsum(0)
    return PLEMetadata(
        input_ids=torch.tensor([t for s in sequences for t in s], dtype=torch.int64, device=device),
        cu_seqlens=cu.to(device),
        seq_lens=tuple(lens),
        ngram_context=torch.tensor(contexts, dtype=torch.int64, device=device),
        state_slots=torch.arange(len(sequences), dtype=torch.int64, device=device),
        fresh_slots=None,
        is_decode=False,
    )


def _make_ple(config, seed: int = 3, rows: int = 4096):
    args = config.qwen4_args
    gen = torch.Generator().manual_seed(seed)
    layer = PLELayer(config, args.ple_layer_ids[0])
    _fill(layer, gen)
    multipliers, sizes, offsets = hash_constants(args)
    layer.ple_embedding.layer_multipliers.copy_(multipliers)
    layer.ple_embedding.ngram_heads_vocab_sizes.copy_(sizes)
    layer.ple_embedding.ngram_heads_offsets.copy_(offsets)
    table = torch.randn(rows, args.ngram_head_dim, generator=gen) * 0.05
    layer.ple_embedding.attach_table(GpuResidentTable(table, dtype=torch.float32))
    return layer, (multipliers, sizes, offsets)


def test_ple_hash_matches_hf():
    """Ragged hash ids vs the HF dense reference, including eos inside and at the start of a request."""
    config = _config()
    args = config.qwen4_args
    layer, (multipliers, sizes, offsets) = _make_ple(config)

    sequences = [[3, 4, EOS, 5, 6], [EOS, 11, 12], [9]]
    contexts = [[EOS, EOS], [21, 22], [EOS, 31]]
    meta = _ragged(sequences, contexts, args)
    got = layer.ple_embedding.row_ids(meta)

    offset = 0
    for tokens, context in zip(sequences, contexts):
        ref = _hf_ngram_ids(
            torch.tensor([tokens]),
            torch.tensor([context]),
            args,
            multipliers,
            sizes,
            offsets,
        )[0]
        assert torch.equal(got[offset : offset + len(tokens)], ref)
        offset += len(tokens)
    # sequences[0][3] sits right after the boundary token, so its window is cut to eos padding
    after_eos = layer.ple_embedding.row_ids(_ragged([[5]], [[EOS, EOS]], args))[0]
    with_history = layer.ple_embedding.row_ids(_ragged([[5]], [[3, 4]], args))[0]
    assert torch.equal(got[3], after_eos)
    assert not torch.equal(got[3], with_history)


def test_ple_forward_matches_hf():
    """Full PLE forward (fp32) against the HF gate/norm chain and an explicit conv tap sum."""
    torch.manual_seed(4)
    config = _config()
    args = config.qwen4_args
    layer, (multipliers, sizes, offsets) = _make_ple(config)
    hidden, hc = args.hidden_size, args.hc_count

    sequences = [[3, 4, EOS, 5, 6, 8], [2, EOS, 11, 12, 13, 14]]
    contexts = [[EOS, EOS], [21, 22]]
    meta = _ragged(sequences, contexts, args)
    total = sum(len(s) for s in sequences)
    R = torch.randn(total, args.ple_state_width)
    states = torch.randn(len(sequences), args.ple_state_width, args.ple_conv_state_len) * 0.1
    got = layer.forward(R, batch=None, meta=meta, conv_states=states.clone())

    offset = 0
    for i, (tokens, context) in enumerate(zip(sequences, contexts)):
        ids = _hf_ngram_ids(
            torch.tensor([tokens]), torch.tensor([context]), args, multipliers, sizes, offsets
        )[0]
        embed = layer.ple_embedding.table.weight[ids.reshape(-1)].view(len(tokens), -1)
        key = _group_norm(
            F.linear(embed, layer.key_proj.weight), layer.norm_key.weight, config.rms_norm_eps, hc
        ).unflatten(-1, (hc, hidden))
        value = F.linear(embed, layer.value_proj.weight)
        rows = R[offset : offset + len(tokens)]
        query = _group_norm(
            rows, layer.norm_query.weight, config.rms_norm_eps, hc
        ).unflatten(-1, (hc, hidden))
        gate = (key * query).sum(-1, keepdim=True) / math.sqrt(hidden)
        gate = torch.sigmoid(gate.sign() * gate.abs().clamp_min(1e-6).sqrt())
        gated = (gate * value.unsqueeze(-2)).flatten(-2)
        normed = _group_norm(gated, layer.norm_conv.weight, config.rms_norm_eps, hc)
        history = torch.cat([states[i], normed.transpose(0, 1)], dim=-1)
        taps = sum(
            layer.conv1d.weight[:, 0, k].unsqueeze(0)
            * history.transpose(0, 1)[k * args.ple_conv_dilation :][: len(tokens)]
            for k in range(args.ple_conv_kernel_size)
        )
        ref = gated + F.silu(taps)
        assert torch.allclose(got[offset : offset + len(tokens)], ref, rtol=1e-4, atol=1e-5)
        offset += len(tokens)


def _fresh_ctx(**fields):
    import freetoken.core as core
    from freetoken.core import Context, set_global_ctx

    core._GLOBAL_CTX = None  # test-only: each scenario builds its own ctx
    ctx = Context(page_size=64)
    for name, value in fields.items():
        setattr(ctx, name, value)
    set_global_ctx(ctx)
    return ctx


def _plus_one_rmsnorm(x, weight, eps):
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (xf * (1.0 + weight.float())).type_as(x)


def _hf_rope(x, positions, rotary_dim, base):
    """HF apply_rotary_pos_emb on [T, H, D], rotating only the first rotary_dim dims."""
    inv = 1.0 / (
        base ** (torch.arange(0, rotary_dim, 2, device=x.device, dtype=torch.float32) / rotary_dim)
    )
    freqs = positions.float().unsqueeze(-1) * inv
    cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1).unsqueeze(1)
    sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1).unsqueeze(1)
    rot = x[..., :rotary_dim].float()
    half = rotary_dim // 2
    rotated = torch.cat([-rot[..., half:], rot[..., :half]], dim=-1)
    out = (rot * cos + rotated * sin).type_as(x)
    return torch.cat([out, x[..., rotary_dim:]], dim=-1)


def _hf_attention(x, attn, config, positions):
    """HF Qwen4ExpTextAttention with a dense causal mask (QSA selects every block at this length)."""
    num_q, num_kv, dim = attn.num_q, attn.num_kv, attn.head_dim
    qkv = F.linear(x, attn.qkv_proj.weight)
    qg, k, v = qkv.split(attn._qkv_split, dim=-1)
    qg = qg.view(-1, num_q, dim * 2)
    q, gate = qg[..., :dim], qg[..., dim:].reshape(-1, num_q * dim)
    q = _hf_rope(_plus_one_rmsnorm(q, attn.q_norm.weight, config.rms_norm_eps), positions,
                 config.rotary_config.rotary_dim, config.rotary_config.base)
    k = _hf_rope(
        _plus_one_rmsnorm(k.view(-1, num_kv, dim), attn.k_norm.weight, config.rms_norm_eps),
        positions, config.rotary_config.rotary_dim, config.rotary_config.base,
    )
    v = v.view(-1, num_kv, dim)
    rep = num_q // num_kv
    scores = torch.einsum("qhd,khd->hqk", q.float(), k.repeat_interleave(rep, 1).float())
    scores = scores * dim**-0.5
    mask = torch.arange(x.shape[0], device=x.device) > positions.unsqueeze(-1)
    out = torch.einsum(
        "hqk,khd->qhd", scores.masked_fill(mask, float("-inf")).softmax(-1),
        v.repeat_interleave(rep, 1).float(),
    ).to(x.dtype)
    return F.linear(out.reshape(-1, num_q * dim) * torch.sigmoid(gate), attn.o_proj.weight)


@requires_cuda
def test_qsa_layer_matches_hf_dense():
    """The QSA layer under the dense oracle backend equals HF attention, and freezes what the indexer hands the backend."""
    from freetoken.models.qwen4_exp.attention import Qwen4ExpAttention, TorchDenseQSAReference
    from freetoken.utils.torch_utils import torch_dtype

    torch.manual_seed(6)
    config = _config()
    device, dtype = torch.device("cuda"), torch.bfloat16
    with torch.device(device), torch_dtype(dtype):
        attn = Qwen4ExpAttention(config, layer_id=3)
    _fill(attn, torch.Generator(device=device).manual_seed(7))

    seq_len = 24
    x = (torch.randn(seq_len, config.hidden_size, device=device, dtype=dtype) * 0.5)
    positions = torch.arange(seq_len, device=device, dtype=torch.int64)
    req = SimpleNamespace(extend_len=seq_len, cached_len=0, table_idx=1)
    batch = SimpleNamespace(
        padded_reqs=[req], reqs=[req], positions=positions, get_attn_positions=lambda: positions
    )

    backend = TorchDenseQSAReference(config, num_slots=4, max_len=64, device=device, dtype=dtype)
    _fresh_ctx(attn_backend=backend)
    ref = _hf_attention(x, attn, config, positions)
    got = attn.forward(x, batch)
    assert torch.allclose(got.float(), ref.float(), rtol=2e-2, atol=2e-2)

    index = attn.indexer.forward(x)
    args = config.qwen4_args
    raw = F.linear(x, attn.indexer.index_qk_proj.weight)
    assert index.q.shape == (seq_len, args.index_n_heads, args.index_head_dim)
    assert index.k.shape == (seq_len, args.index_head_dim)
    assert torch.equal(index.q.reshape(seq_len, -1), raw[:, : args.index_n_heads * args.index_head_dim])
    assert torch.equal(index.k, raw[:, args.index_n_heads * args.index_head_dim :])
    assert index.q_norm_weight.data_ptr() == attn.indexer.q_layernorm.weight.data_ptr()


class _StubLinearMixer(BaseOP):
    """Stands in for the GDN layer; same [T, hidden] -> [T, hidden] shape."""

    def __init__(self, config, layer_id, prefix=""):
        self.out_proj = LinearReplicated(config.hidden_size, config.hidden_size, has_bias=False)

    def forward(self, x):
        return self.out_proj.forward(x)


@requires_cuda
def test_shared_expert_gate_fusion_matches_eager():
    """Qwen4ExpMoE only swaps qwen3_5's gemv+sigmoid+mul+add gate chain for two triton kernels."""
    from freetoken.models.qwen3_5_moe.moe import Qwen3_5MoE
    from freetoken.models.qwen4_exp.moe import Qwen4ExpMoE
    from freetoken.utils.torch_utils import torch_dtype

    config = _config()
    device, dtype = torch.device("cuda"), torch.bfloat16
    with torch.device(device), torch_dtype(dtype):
        moe = Qwen4ExpMoE(config, 0)
    _fill(moe, torch.Generator(device=device).manual_seed(21), scale=0.2)
    _fresh_ctx(_batch=SimpleNamespace(is_prefill=True))

    x = torch.randn(6, config.hidden_size, device=device, dtype=dtype) * 0.5
    fused = moe.forward(x.clone())
    eager = Qwen3_5MoE.forward(moe, x.clone())

    routed = moe.experts.forward(hidden_states=x.clone(), router_logits=moe.gate.forward(x))
    gate = torch.sigmoid(x.float() @ moe.shared_expert_gate.weight.float().view(-1))
    ref = routed.float() + gate.unsqueeze(1) * moe.shared_expert.forward(x).float()

    assert fused.shape == x.shape and fused.dtype == dtype
    torch.testing.assert_close(fused, eager, rtol=2e-2, atol=2e-2)
    # The fused gate stays in fp32 where the eager chain rounds the scalar to bf16.
    assert (fused.float() - ref).abs().max() <= (eager.float() - ref).abs().max()


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("num_tokens,hidden", [(1, 2560), (7, 640)])
def test_shared_gate_kernels_match_torch(num_tokens, hidden, dtype):
    """Shipping hidden size for the two shared-gate kernels, against the torch chain they replace."""
    from freetoken.kernel.triton.moe_shared_gate import shared_gate_mul_add, shared_gate_sigmoid

    gen = torch.Generator(device="cuda").manual_seed(hidden + num_tokens)
    kw = {"generator": gen, "device": "cuda", "dtype": dtype}
    x = torch.randn(num_tokens, hidden, **kw)
    weight = torch.randn(1, hidden, **kw) * 0.05
    shared = torch.randn(num_tokens, hidden, **kw)
    routed = torch.randn(num_tokens, hidden, **kw)

    fused = shared_gate_mul_add(routed, shared, shared_gate_sigmoid(x, weight.view(-1)))
    eager = routed + shared * torch.sigmoid(F.linear(x, weight))
    ref = routed.float() + shared.float() * torch.sigmoid(x.float() @ weight.float().view(-1))[:, None]

    assert fused.dtype == dtype and fused.shape == routed.shape
    torch.testing.assert_close(fused, eager, rtol=2e-2, atol=2e-2)
    assert (fused.float() - ref).abs().max() <= (eager.float() - ref).abs().max() + 1e-6


def _offload_experts(kind: str, layer_id: int, num_experts: int, top_k: int, hidden: int, inter: int):
    from freetoken.layers.moe import OffloadMoELayer
    from freetoken.layers.quantization import NoQuantConfig, QuantBackend, QuantConfig, set_quant_backend

    if kind == "bf16":
        quant = NoQuantConfig()
    else:
        set_quant_backend(QuantBackend.parse("moe.nvfp4=triton"))  # the sm_120 decode kernel
        quant = QuantConfig.from_hf({"quantization_config": {"quant_method": "modelopt", "quant_algo": "NVFP4", "ignore": ["lm_head"]}})
    return OffloadMoELayer(
        layer_id, num_experts, top_k, hidden, inter, quant_config=quant,
        prefix=f"model.layers.{layer_id}.mlp.experts",
    )


def _copy_overlap_run(kind, banks, overlap, bs, graph, inputs, monkeypatch):
    """Two Qwen4ExpMoE layers sharing one offload cache as small as a layer, so every step evicts
    and copies: warm up eagerly (and capture), reset the cache, then decode ``inputs``."""
    import freetoken.kernel.fast_index_copy as fic
    from flashlib.kernels.slot_cache import Stat
    from freetoken.models.qwen4_exp.moe import Qwen4ExpMoE
    from freetoken.moe.offload_cache import OffloadMoeCache
    from freetoken.utils.torch_utils import torch_dtype

    num_experts, top_k, hidden, inter = 8, 2, 256, 128
    config = parse_config(toy_hf_config(
        hidden_size=hidden, num_experts=num_experts, num_experts_per_tok=top_k,
        moe_intermediate_size=inter, shared_expert_intermediate_size=inter,
    ))
    device = torch.device("cuda")
    with torch.device(device), torch_dtype(torch.bfloat16):
        moes = [Qwen4ExpMoE(config, layer_id) for layer_id in range(2)]
    gen = torch.Generator(device=device).manual_seed(41)
    for moe in moes:
        _fill(moe, gen, scale=0.2)
    experts = [_offload_experts(kind, i, num_experts, top_k, hidden, inter) for i in range(2)]
    cache = OffloadMoeCache(
        num_layers=2, num_experts=num_experts, cache_size=num_experts, device=device,
        quant_format=banks.quant_format, layout=banks.layout,
        max_slots=experts[0].quant_method.slot_limit(), decode_copy_overlap=overlap,
    )
    cache.set_bank_sources(banks.sources)
    cache.set_alphas(banks.gate_up_alpha, banks.down_alpha)
    cache.collect_stats = True
    for moe, ex in zip(moes, experts):
        ex.offload_cache = cache
        moe.experts = ex
    cache.reset()

    import freetoken.models.qwen4_exp.moe as qmoe

    copy_streams, gate_streams = [], []
    real_gate = qmoe.shared_gate_sigmoid
    for name in ("fast_index_copy_multi_jit", "fast_index_copy_multi_slim_jit"):  # FREETOKEN_MOE_SLIM_COPY
        real_copy = getattr(fic, name)
        monkeypatch.setattr(
            fic, name,
            lambda *a, _real=real_copy, **k: (copy_streams.append(torch.cuda.current_stream().cuda_stream), _real(*a, **k))[1],
        )
    monkeypatch.setattr(
        qmoe, "shared_gate_sigmoid",
        lambda *a, **k: (gate_streams.append(torch.cuda.current_stream().cuda_stream), real_gate(*a, **k))[1],
    )

    def step(x0, x1):
        return moes[0].forward(x0.clone()), moes[1].forward(x1.clone())

    _fresh_ctx(_batch=SimpleNamespace(is_prefill=False))
    static = [torch.randn(bs, hidden, device=device, dtype=torch.bfloat16) for _ in range(2)]
    step(*static)  # the eager warm-up the graph runner also does
    if graph:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            outs = step(*static)
    cache.reset()
    cache.reset_stats()
    results = []
    for x0, x1 in inputs:
        if graph:
            static[0].copy_(x0)
            static[1].copy_(x1)
            g.replay()
        else:
            outs = step(x0, x1)
        results.append(tuple(o.clone() for o in outs))
    torch.cuda.synchronize()
    state = [t.clone() for t in (cache.slot_for_id, cache.id_of_slot, cache.usage, *cache.bank_views())]
    misses = int(cache.lru_stats[:, Stat.MISS].sum())
    side = getattr(cache.decode_copy_stream, "cuda_stream", None)
    return results, state, misses, [s == side for s in copy_streams], [s == side for s in gate_streams]


@requires_cuda
@pytest.mark.parametrize("kind", ["bf16", "nvfp4"])
@pytest.mark.parametrize("graph", [False, True], ids=["eager", "graph"])
@pytest.mark.parametrize("bs", [1, 2])
def test_moe_copy_overlap_is_bitwise_identical(kind, graph, bs, monkeypatch):
    """FREETOKEN_MOE_COPY_OVERLAP moves the shared expert onto a side stream beside the decode miss
    copy; the MoE outputs, the slot map and the slot contents must not change by a single bit."""
    from freetoken.moe.expert_banks import build_expert_banks

    torch.manual_seed(7)
    method = _offload_experts(kind, 0, 8, 2, 256, 128).quant_method
    banks = build_expert_banks(method, 2, None, device=torch.device("cuda"), dummy=True)
    gen = torch.Generator(device="cuda").manual_seed(bs)
    inputs = [
        tuple(torch.randn(bs, 256, device="cuda", dtype=torch.bfloat16, generator=gen) * 0.5 for _ in range(2))
        for _ in range(6)
    ]
    off = _copy_overlap_run(kind, banks, False, bs, graph, inputs, monkeypatch)
    on = _copy_overlap_run(kind, banks, True, bs, graph, inputs, monkeypatch)

    assert off[2] > 0 and on[2] == off[2], "the sequence must miss, and copy, in both runs"
    assert off[3] and not any(off[3]) and not any(off[4]), "flag off: everything stays on the compute stream"
    assert on[3] and not any(on[3]), "flag on: the miss copy stays right behind its ensure"
    assert on[4] and all(on[4]), "flag on: the shared expert runs on the cache's side stream"
    for got, want in zip(on[0], off[0]):
        for a, b in zip(got, want):
            assert torch.isfinite(b.float()).all()
            assert torch.equal(a, b)
    for a, b in zip(on[1], off[1]):
        assert torch.equal(a, b)


# FREETOKEN_MOE_PREFETCH stack: 4 layers (full attention on layer 3), 16 experts, top-2, and one
# shared cache as small as a layer so every step evicts
_PF_LAYERS, _PF_EXPERTS, _PF_TOPK, _PF_HIDDEN, _PF_INTER = 4, 16, 2, 256, 128
# ~50-100 us between layers: the predictor must finish before the next ensure even at toy size
_PF_SLEEP_CYCLES = 200_000


def _prefetch_banks(kind):
    from freetoken.moe.expert_banks import build_expert_banks

    torch.manual_seed(7)
    method = _offload_experts(kind, 0, _PF_EXPERTS, _PF_TOPK, _PF_HIDDEN, _PF_INTER).quant_method
    return build_expert_banks(method, _PF_LAYERS, None, device=torch.device("cuda"), dummy=True)


def _prefetch_stack(kind, banks, mode, monkeypatch, *, k=6, budget=0, wire=True, overlap=False, policy="rule",
                    cache_size=_PF_EXPERTS, **cache_kw):
    from freetoken.env import ENV
    from freetoken.models.qwen4_exp.moe import Qwen4ExpMoE, wire_router_lookahead
    from freetoken.moe.offload_cache import OffloadMoeCache
    from freetoken.utils.torch_utils import torch_dtype

    monkeypatch.setattr(ENV.MOE_PREFETCH_K, "value", k)
    monkeypatch.setattr(ENV.MOE_PREFETCH_BUDGET, "value", budget)
    config = parse_config(toy_hf_config(
        _PF_LAYERS, hidden_size=_PF_HIDDEN, num_experts=_PF_EXPERTS, num_experts_per_tok=_PF_TOPK,
        moe_intermediate_size=_PF_INTER, shared_expert_intermediate_size=_PF_INTER,
    ))
    device = torch.device("cuda")
    with torch.device(device), torch_dtype(torch.bfloat16):
        moes = [Qwen4ExpMoE(config, layer_id) for layer_id in range(_PF_LAYERS)]
    gen = torch.Generator(device=device).manual_seed(43)
    for moe in moes:
        _fill(moe, gen, scale=0.2)
    experts = [_offload_experts(kind, i, _PF_EXPERTS, _PF_TOPK, _PF_HIDDEN, _PF_INTER) for i in range(_PF_LAYERS)]
    cache = OffloadMoeCache(
        num_layers=_PF_LAYERS, num_experts=_PF_EXPERTS, cache_size=cache_size, device=device,
        cache_policy=policy, quant_format=banks.quant_format, layout=banks.layout,
        max_slots=experts[0].quant_method.slot_limit(), prefetch_mode=mode, decode_copy_overlap=overlap,
        **cache_kw,
    )
    cache.set_bank_sources(banks.sources)
    cache.set_alphas(banks.gate_up_alpha, banks.down_alpha)
    cache.collect_stats = True
    for moe, ex in zip(moes, experts):
        ex.offload_cache = cache
        moe.experts = ex
    if wire:
        wire_router_lookahead(moes, config)
    cache.reset()
    return moes, cache


def _prefetch_run(moes, cache, bs, graph, inputs, before_decode=None, sleep=_PF_SLEEP_CYCLES, before_step=None):
    """Warm up eagerly (and capture), reset the cache, then decode ``inputs`` (per step, one
    ``[bs, hidden]`` input per layer); ``sleep`` cycles between layers, ``before_step(i)`` before step i."""
    from flashlib.kernels.slot_cache import Stat

    _fresh_ctx(_batch=SimpleNamespace(is_prefill=False))

    def step(xs):
        outs = []
        for layer, (moe, x) in enumerate(zip(moes, xs)):
            if layer and sleep:
                torch.cuda._sleep(sleep)
            outs.append(moe.forward(x.clone()))
        return outs

    static = [torch.randn(bs, _PF_HIDDEN, device="cuda", dtype=torch.bfloat16) for _ in moes]
    step(static)  # the eager warm-up the graph runner also does
    if graph:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            outs = step(static)
    cache.reset()
    cache.reset_stats()
    if before_decode is not None:
        before_decode()
    results = []
    for i, xs in enumerate(inputs):
        if before_step is not None:
            before_step(i)
        if graph:
            for s, x in zip(static, xs):
                s.copy_(x)
            g.replay()
        else:
            outs = step(xs)
        results.append([o.clone() for o in outs])
    torch.cuda.synchronize()
    state = [t.clone() for t in (cache.slot_for_id, cache.id_of_slot, cache.usage, *cache.bank_views())]
    counters = None if cache.prefetch is None else cache.prefetch.counters.cpu()
    return results, state, counters, cache.lru_stats[:, Stat.MISS].cpu()


def _prefetch_inputs(bs, steps=8, chain=False, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed * 10 + bs)

    def x():
        return torch.randn(bs, _PF_HIDDEN, device="cuda", dtype=torch.bfloat16, generator=gen) * 0.5

    if chain:  # every layer sees the same input, so the lookahead equals the next router
        return [[x()] * _PF_LAYERS for _ in range(steps)]
    return [[x() for _ in range(_PF_LAYERS)] for _ in range(steps)]


@requires_cuda
@pytest.mark.parametrize("kind", ["bf16", "nvfp4"])
@pytest.mark.parametrize("graph", [False, True], ids=["eager", "graph"])
@pytest.mark.parametrize("bs", [1, 2])
@pytest.mark.parametrize("overlap,policy", [(False, "rule"), (True, "rule"), (True, "lru")], ids=["serial", "overlap", "overlap-lru"])
def test_moe_prefetch_measure_is_bitwise_identical(kind, graph, bs, overlap, policy, monkeypatch):
    """FREETOKEN_MOE_PREFETCH=measure predicts and counts beside the decode (with and without
    FREETOKEN_MOE_COPY_OVERLAP's side stream); the MoE outputs, the slot map, the usage clock and
    the slot contents must not change by a single bit."""
    banks = _prefetch_banks(kind)
    inputs = _prefetch_inputs(bs)
    off = _prefetch_run(*_prefetch_stack(kind, banks, "off", monkeypatch, overlap=overlap, policy=policy), bs, graph, inputs)
    on = _prefetch_run(*_prefetch_stack(kind, banks, "measure", monkeypatch, overlap=overlap, policy=policy), bs, graph, inputs)

    assert off[2] is None and int(off[3].sum()) > 0 and torch.equal(on[3], off[3])
    for got, want in zip(on[0], off[0]):
        for a, b in zip(got, want):
            assert torch.isfinite(b.float()).all()
            assert torch.equal(a, b)
    for a, b in zip(on[1], off[1]):
        assert torch.equal(a, b)
    from freetoken.moe.prefetch import CALLS, ISSUED, ROWS

    counters = on[2]
    assert counters[0].tolist() == [0] * counters.shape[1], "layer 0 has no predecessor"
    assert counters[1:, CALLS].tolist() == [len(inputs)] * (_PF_LAYERS - 1)
    assert counters[1:, ROWS].tolist() == [len(inputs) * bs] * (_PF_LAYERS - 1)
    assert int(counters[1:, ISSUED].sum()) > 0


def _reference_counters(selects, ensures, budget_override, k):
    """CPU counters from the spied predictor logits and each target layer's routing and residency."""
    from freetoken.moe.prefetch import NUM_COLS, default_budget
    from tests.moe.ref_prefetch import ref_count, ref_select

    config = parse_config(toy_hf_config(_PF_LAYERS))
    counters = torch.zeros((_PF_LAYERS, NUM_COLS), dtype=torch.int64)
    for layer in range(1, _PF_LAYERS):
        assert len(selects[layer]) == len(ensures[layer]) > 0
        want_budget = budget_override or default_budget(config.is_linear_layer(layer))
        for (logits, budget), (raw, resident) in zip(selects[layer], ensures[layer]):
            assert budget == want_budget
            sel, res = ref_select(logits.tolist(), resident.tolist(), k, budget)
            routed = raw.view(-1).tolist()
            misses = len({e for e in routed if resident[e] < 0})
            counters[layer] += torch.tensor(ref_count(sel, res, routed, misses, logits.shape[0]))
    return counters


@requires_cuda
@pytest.mark.parametrize("kind", ["bf16", "nvfp4"])
@pytest.mark.parametrize("bs", [1, 2])
@pytest.mark.parametrize("budget", [0, 5])
def test_moe_prefetch_counters_match_reference(kind, bs, budget, monkeypatch):
    """Eager counters equal a CPU reference built from the predictor's own logits and each layer's
    routing and pre-ensure residency; the captured graph counts the same. The predictor runs on
    its dedicated stream, the count on the compute stream."""
    import freetoken.moe.prefetch as pf_mod

    banks = _prefetch_banks(kind)
    inputs = _prefetch_inputs(bs, seed=1)
    k = 6
    moes, cache = _prefetch_stack(kind, banks, "measure", monkeypatch, k=k, budget=budget)
    selects = {layer: [] for layer in range(_PF_LAYERS)}
    ensures = {layer: [] for layer in range(_PF_LAYERS)}
    streams = {"select": set(), "count": set()}
    real_select, real_count, real_ensure = pf_mod.lookahead_select, pf_mod.prefetch_count, cache.ensure_experts

    def spy_select(logits, resident, sel, res, *, k, budget):
        layer = (sel.data_ptr() - cache.prefetch.sel.data_ptr()) // cache.prefetch.sel[0].nbytes
        selects[layer].append((logits.float().cpu(), budget))
        streams["select"].add(torch.cuda.current_stream().cuda_stream)
        return real_select(logits, resident, sel, res, k=k, budget=budget)

    def spy_count(*args, **kwargs):
        streams["count"].add(torch.cuda.current_stream().cuda_stream)
        return real_count(*args, **kwargs)

    def spy_ensure(layer_id, expert_ids, **kwargs):
        torch.cuda.synchronize()  # the prediction forked after the previous ensure has landed
        ensures[layer_id].append((expert_ids.cpu(), cache.slot_for_id[layer_id].cpu()))
        return real_ensure(layer_id, expert_ids, **kwargs)

    def install():
        monkeypatch.setattr(pf_mod, "lookahead_select", spy_select)
        monkeypatch.setattr(pf_mod, "prefetch_count", spy_count)
        monkeypatch.setattr(cache, "ensure_experts", spy_ensure)

    eager = _prefetch_run(moes, cache, bs, False, inputs, before_decode=install)
    want = _reference_counters(selects, ensures, budget, k)
    assert torch.equal(eager[2], want), (eager[2], want)
    assert int(want[:, 1].sum()) > 0, "the sequence must have useful predictions"
    assert streams["select"] == {cache.prefetch.stream.cuda_stream}
    assert streams["count"] == {torch.cuda.current_stream().cuda_stream}

    monkeypatch.undo()
    moes, cache = _prefetch_stack(kind, banks, "measure", monkeypatch, k=k, budget=budget)
    graph = _prefetch_run(moes, cache, bs, True, inputs)
    assert torch.equal(graph[2], eager[2])
    for got, want_out in zip(graph[0], eager[0]):
        for a, b in zip(got, want_out):
            assert torch.equal(a, b)


@requires_cuda
@pytest.mark.parametrize("graph", [False, True], ids=["eager", "graph"])
@pytest.mark.parametrize("bs", [1, 2])
def test_moe_prefetch_perfect_lookahead_counts_every_miss(graph, bs, monkeypatch):
    """Known answer: when layer L-1's router input equals layer L's, the lookahead is layer L's own
    router, so with the budget covering every candidate each demand miss is a useful prediction."""
    from freetoken.moe.prefetch import ISSUED, MISSES, RESIDENT_HITS, USEFUL

    banks = _prefetch_banks("nvfp4")
    moes, cache = _prefetch_stack("nvfp4", banks, "measure", monkeypatch, k=4, budget=8)
    _, _, counters, lru_misses = _prefetch_run(moes, cache, bs, graph, _prefetch_inputs(bs, chain=True, seed=2))
    assert int(counters[1:, MISSES].sum()) > 0
    assert counters[1:, USEFUL].tolist() == counters[1:, MISSES].tolist() == lru_misses[1:].tolist()
    assert (counters[1:, ISSUED] >= counters[1:, USEFUL]).all()
    assert int(counters[1:, RESIDENT_HITS].sum()) > 0, "routed hits among the top candidates"


@requires_cuda
def test_moe_prefetch_off_launches_nothing_new(monkeypatch):
    """With the flag off the wired lookahead is inert: no prefetcher, and the decode launches
    exactly the kernels of an unwired stack (measure adds some, so the check can see them)."""
    from torch.profiler import ProfilerActivity, profile

    banks = _prefetch_banks("bf16")
    inputs = _prefetch_inputs(1, steps=2)

    def kernels(mode, wire):
        moes, cache = _prefetch_stack("bf16", banks, mode, monkeypatch, wire=wire)
        _prefetch_run(moes, cache, 1, False, inputs[:1])  # warm up and compile outside the trace
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            _prefetch_run(moes, cache, 1, False, inputs)
        events = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
        return cache, [e.name for e in sorted(events, key=lambda e: e.time_range.start)]

    cache, wired_off = kernels("off", True)
    assert cache.prefetch is None
    _, unwired_off = kernels("off", False)
    _, measured = kernels("measure", True)
    assert wired_off == unwired_off
    assert len(measured) > len(wired_off)


@requires_cuda
def test_moe_prefetch_survives_a_rebuild(monkeypatch):
    """A runtime rebuild resets the counters and keeps the predictor stream; a recaptured graph
    counts like an eager run on the rebuilt cache."""
    banks = _prefetch_banks("bf16")
    inputs = _prefetch_inputs(1, seed=3)
    moes, cache = _prefetch_stack("bf16", banks, "measure", monkeypatch)
    before = _prefetch_run(moes, cache, 1, True, inputs)[2]
    stream = cache.prefetch.stream
    assert int(before.sum()) > 0
    cache.rebuild(_PF_EXPERTS + 4)
    assert int(cache.prefetch.counters.abs().sum()) == 0 and cache.prefetch.stream is stream
    graph = _prefetch_run(moes, cache, 1, True, inputs)
    eager = _prefetch_run(moes, cache, 1, False, inputs)
    assert int(graph[2].sum()) > 0 and torch.equal(graph[2], eager[2])
    for a, b in zip(graph[1], eager[1]):
        assert torch.equal(a, b)


@requires_cuda
@pytest.mark.parametrize("kind", ["bf16", "nvfp4"])
@pytest.mark.parametrize("bs", [1, 2])
def test_moe_prefetch_predicts_layer_l_from_layer_l_minus_1_input(kind, bs, monkeypatch):
    """With a distinct input per layer, the prediction for layer L reads exactly layer L-1's router
    input through layer L's router, and the select sees those logits and layer L's residency."""
    import freetoken.moe.prefetch as pf_mod

    banks = _prefetch_banks(kind)
    inputs = _prefetch_inputs(bs, steps=3, seed=4)
    moes, cache = _prefetch_stack(kind, banks, "measure", monkeypatch)
    forks, selects = [], []
    real_fork, real_select = cache.prefetch.fork, pf_mod.lookahead_select

    def spy_fork(target, x, gate, resident, budget):
        forks.append((target, x.clone(), gate, resident.data_ptr()))
        return real_fork(target, x, gate, resident, budget)

    def spy_select(logits, resident, sel, res, *, k, budget):
        selects.append(logits.clone())
        return real_select(logits, resident, sel, res, k=k, budget=budget)

    def install():
        monkeypatch.setattr(cache.prefetch, "fork", spy_fork)
        monkeypatch.setattr(pf_mod, "lookahead_select", spy_select)

    _prefetch_run(moes, cache, bs, False, inputs, before_decode=install)
    assert [f[0] for f in forks] == list(range(1, _PF_LAYERS)) * len(inputs) and len(selects) == len(forks)
    for call, ((target, x, gate, resident), logits) in enumerate(zip(forks, selects)):
        assert torch.equal(x, inputs[call // (_PF_LAYERS - 1)][target - 1])
        assert gate is moes[target].gate and resident == cache.slot_for_id[target].data_ptr()
        assert torch.equal(logits, F.linear(x, moes[target].gate.weight))


# FREETOKEN_MOE_PREFETCH=on: the same stack, with copies. No sleep between layers, so the prefetch
# streams race the compute stream at toy size and only the joins keep the GEMMs on landed bytes.
_PF_ON_STEPS = 200


def _assert_slots_hold_their_experts(cache):
    """Lossless: every held slot's bytes equal the host rows of the expert its id names."""
    held = (cache.id_of_slot >= 0).nonzero().view(-1).cpu()
    ids = cache.id_of_slot[held.cuda()].long().cpu()
    assert held.numel() > 0
    for (per_layer, slots), view in zip(cache.banks, cache.bank_views()):
        host = torch.stack(per_layer).flatten(0, 1)
        want = host[ids].contiguous().view(torch.uint8)
        got = view[held.cuda()].cpu().contiguous().view(torch.uint8)
        assert torch.equal(got, want)


def _routed_ids(moes, inputs):
    """[step][layer] -> the experts each layer routes (the router alone decides; no cache input)."""
    from freetoken.moe.fused import fused_topk

    out = []
    for xs in inputs:
        row = []
        for moe, x in zip(moes, xs):
            _, ids = fused_topk(hidden_states=x, gating_output=moe.gate.forward(x), topk=_PF_TOPK, renormalize=True)
            row.append(ids.view(-1).tolist())
        out.append(row)
    return out


def _garbage(routed, width, kind, seed):
    """Per step a [layers, width] candidate override: 'random' ids (and -1), 'adversarial' ones
    that mix layer L's routed ids with ids it does not route, duplicates and padding, or 'flood':
    every expert of layer L, so a prefetch wants more slots than the unpinned ones."""
    gen = torch.Generator().manual_seed(seed)
    steps = []
    for row in routed:
        sel = torch.full((_PF_LAYERS, width), -1, dtype=torch.int32)
        for layer in range(1, _PF_LAYERS):
            if kind == "random":
                ids = torch.randint(-1, _PF_EXPERTS, (width,), generator=gen)
            elif kind == "flood":
                ids = torch.randperm(_PF_EXPERTS, generator=gen)[:width]
            else:
                hit = list(dict.fromkeys(row[layer]))
                miss = [e for e in torch.randperm(_PF_EXPERTS, generator=gen).tolist() if e not in hit][:3]
                ids = torch.tensor((hit[:3] + miss + hit[:1] + [-1] + miss[:1]) * 2)[:width]
                ids = ids[torch.randperm(ids.numel(), generator=gen)]
            sel[layer, : ids.numel()] = ids.to(torch.int32)
        steps.append(sel)
    return steps


def _on_stack(kind, banks, monkeypatch, *, policy, overlap, k=6, budget=0):
    return _prefetch_stack(kind, banks, "on", monkeypatch, k=k, budget=budget, overlap=overlap, policy=policy)


def _with_override(cache, steps, **delays):
    """before_step hook that loads step i's candidate override into the prefetcher's static buffer."""
    pf = cache.prefetch
    pf.sel_override = torch.full_like(pf.sel, -1)
    for name, cycles in delays.items():
        setattr(pf, name, cycles)

    def before_step(i):
        pf.sel_override.copy_(steps[i][:, : pf.sel.shape[1]])

    return before_step


def _assert_same_outputs(got, want):
    for a_step, b_step in zip(got, want):
        for a, b in zip(a_step, b_step):
            assert torch.isfinite(b.float()).all()
            assert torch.equal(a, b)


@requires_cuda
@pytest.mark.parametrize("kind", ["bf16", "nvfp4"])
@pytest.mark.parametrize("graph", [False, True], ids=["eager", "graph"])
@pytest.mark.parametrize("bs", [1, 2])
@pytest.mark.parametrize("policy", ["rule", "lru"])
@pytest.mark.parametrize("overlap", [False, True], ids=["serial", "overlap"])
def test_moe_prefetch_on_is_bitwise_identical(kind, graph, bs, policy, overlap, monkeypatch):
    """FREETOKEN_MOE_PREFETCH=on copies predicted experts into low-priority slots beside the decode:
    over 200 steps of a cache that evicts every step the MoE outputs are those of prefetch off to
    the bit, every held slot holds its expert's bytes, and the prefetches were used."""
    from freetoken.moe.prefetch import CALLS, COPIED, ISSUED, USEFUL

    banks = _prefetch_banks(kind)
    inputs = _prefetch_inputs(bs, steps=_PF_ON_STEPS, seed=5)
    moes, cache = _prefetch_stack(kind, banks, "off", monkeypatch, overlap=overlap, policy=policy)
    off = _prefetch_run(moes, cache, bs, graph, inputs, sleep=0)
    _assert_slots_hold_their_experts(cache)
    moes, cache = _on_stack(kind, banks, monkeypatch, policy=policy, overlap=overlap)
    on = _prefetch_run(moes, cache, bs, graph, inputs, sleep=0)
    _assert_same_outputs(on[0], off[0])
    _assert_slots_hold_their_experts(cache)
    stats = cache.prefetch.stats.cpu()
    assert stats[0].tolist() == [0] * stats.shape[1], "layer 0 is never prefetched"
    assert stats[1:, CALLS].tolist() == [_PF_ON_STEPS] * (_PF_LAYERS - 1)
    assert int(stats[:, ISSUED].sum()) > 0 and int(stats[:, USEFUL].sum()) > 0 and int(stats[:, COPIED].sum()) > 0
    assert int(on[3].sum()) < int(off[3].sum()), "useful prefetches remove demand misses"


@requires_cuda
@pytest.mark.parametrize("graph", [False, True], ids=["eager", "graph"])
@pytest.mark.parametrize("bs", [1, 2])
@pytest.mark.parametrize("policy", ["rule", "lru"])
@pytest.mark.parametrize("garbage", ["random", "adversarial"])
def test_moe_prefetch_on_survives_a_garbage_predictor(graph, bs, policy, garbage, monkeypatch):
    """Whatever the candidates (random ids and padding, or the next layer's routed ids mixed with ids
    it will not route, duplicates and -1), prefetch on changes no output bit and no slot's bytes."""
    from freetoken.moe.prefetch import ISSUED, USEFUL

    banks = _prefetch_banks("nvfp4")
    inputs = _prefetch_inputs(bs, steps=_PF_ON_STEPS, seed=6)
    moes, cache = _prefetch_stack("nvfp4", banks, "off", monkeypatch, overlap=True, policy=policy)
    off = _prefetch_run(moes, cache, bs, graph, inputs, sleep=0)
    moes, cache = _on_stack("nvfp4", banks, monkeypatch, policy=policy, overlap=True, budget=8)
    steps = _garbage(_routed_ids(moes, inputs), 12, garbage, seed=bs)
    on = _prefetch_run(moes, cache, bs, graph, inputs, sleep=0, before_step=_with_override(cache, steps))
    _assert_same_outputs(on[0], off[0])
    _assert_slots_hold_their_experts(cache)
    stats = cache.prefetch.stats.cpu()
    assert int(stats[:, ISSUED].sum()) > 0 and int(stats[:, USEFUL].sum()) > 0
    assert int(stats[:, USEFUL].sum()) < int(stats[:, ISSUED].sum()), "some prefetches must be wrong"


# ~0.5 ms at the RTX 5080's clock: far longer than a toy layer, so every delayed stage is late
_PF_DELAY_CYCLES = 1_000_000


@requires_cuda
@pytest.mark.parametrize("graph", [False, True], ids=["eager", "graph"])
@pytest.mark.parametrize("bs", [1, 2])
@pytest.mark.parametrize("policy", ["rule", "lru"])
@pytest.mark.parametrize("delay", ["copy", "predict", "both"])
def test_moe_prefetch_on_joins_hold_when_the_prefetch_streams_run_late(graph, bs, policy, delay, monkeypatch):
    """Race test: a sleep on the prefetch copy stream (every copy lands after its layer's ensure) or on
    the predictor stream (prefetch_ensure runs long after the fork) must not change a bit. The
    candidates include each layer's routed ids, so GEMMs do read prefetched slots. The joins also
    make the slot maps timing-free: the delayed run ends with the undelayed run's maps and counts."""
    from freetoken.moe.prefetch import COPIED, LATE, NUM_STAT_COLS, USEFUL

    banks = _prefetch_banks("nvfp4")
    inputs = _prefetch_inputs(bs, steps=_PF_ON_STEPS, seed=7)
    moes, cache = _prefetch_stack("nvfp4", banks, "off", monkeypatch, overlap=True, policy=policy)
    off = _prefetch_run(moes, cache, bs, graph, inputs, sleep=0)
    runs = []
    for cycles in (0, _PF_DELAY_CYCLES):
        moes, cache = _on_stack("nvfp4", banks, monkeypatch, policy=policy, overlap=True, budget=8)
        steps = _garbage(_routed_ids(moes, inputs), 12, "adversarial", seed=bs + 10)
        delays = {"delay_copy_cycles": cycles * (delay in ("copy", "both")),
                  "delay_predict_cycles": cycles * (delay in ("predict", "both"))}
        on = _prefetch_run(moes, cache, bs, graph, inputs, sleep=0, before_step=_with_override(cache, steps, **delays))
        _assert_same_outputs(on[0], off[0])
        _assert_slots_hold_their_experts(cache)
        runs.append((on[1][:3], cache.prefetch.stats.cpu()))
    (maps, stats), (delayed_maps, delayed_stats) = runs
    assert int(stats[:, USEFUL].sum()) > 0
    for a, b in zip(maps, delayed_maps):
        assert torch.equal(a, b)
    cols = [c for c in range(NUM_STAT_COLS) if c != LATE]
    assert torch.equal(stats[:, cols], delayed_stats[:, cols])
    if delay == "copy":
        # the late counter sees what the join waited for: the copies were still sleeping at ensure(L)
        # (all of them under replay; in eager a loaded host can take longer than the sleep per layer)
        late, copied = int(delayed_stats[:, LATE].sum()), int(delayed_stats[:, COPIED].sum())
        assert copied > 0 and (late == copied if graph else late > 0)


@requires_cuda
def test_moe_prefetch_race_test_catches_a_missing_copy_join(monkeypatch):
    """Negative control for the race test: without the compute stream's wait on the prefetch copy,
    GEMMs read prefetched slots before their bytes land and the outputs change."""
    from freetoken.moe.prefetch import ExpertPrefetcher

    banks = _prefetch_banks("nvfp4")
    inputs = _prefetch_inputs(1, steps=40, seed=7)
    moes, cache = _prefetch_stack("nvfp4", banks, "off", monkeypatch, overlap=True)
    off = _prefetch_run(moes, cache, 1, False, inputs, sleep=0)
    moes, cache = _on_stack("nvfp4", banks, monkeypatch, policy="rule", overlap=True, budget=8)

    def no_wait(self, layer_id):
        self._inflight[layer_id] = None

    monkeypatch.setattr(ExpertPrefetcher, "join_copy", no_wait)
    steps = _garbage(_routed_ids(moes, inputs), 12, "adversarial", seed=11)
    on = _prefetch_run(moes, cache, 1, False, inputs, sleep=0,
                       before_step=_with_override(cache, steps, delay_copy_cycles=_PF_DELAY_CYCLES))
    assert any(not torch.equal(a, b) for got, want in zip(on[0], off[0]) for a, b in zip(got, want))


def _replay_on_reference(policy, calls, rows):
    """Replay the spied prefetch_ensure / ensure sequence on the CPU reference; per-layer stats."""
    from freetoken.moe.prefetch import CALLS, COPIED, ISSUED, MISSES, ROWS, USEFUL
    from freetoken.moe.scored_ensure import POLICY_IDS
    from tests.moe.ref_scored_cache import RefScoredCache

    ref = RefScoredCache(_PF_LAYERS, _PF_EXPERTS, _PF_EXPERTS, POLICY_IDS[policy])
    stats = torch.zeros((_PF_LAYERS, 8), dtype=torch.int64)
    planned = {}
    for call in calls:
        if call[0] == "prefetch":
            _, layer, sel = call
            src, _ = ref.prefetch(layer, sel.tolist())
            stats[layer, ISSUED] += src.size
            planned[layer] = src.size
        else:
            _, layer, ids, logits = call
            _, src, _ = ref.ensure(layer, ids.view(-1).numpy(), bump_tok=layer == 0, lowpri=True,
                                   logits=None if logits is None else logits.float().numpy(), thr=0.25)
            if layer in planned:
                stats[layer, [USEFUL, CALLS, ROWS, MISSES, COPIED]] += torch.tensor(
                    [ref.useful, 1, rows, src.size, int(planned.pop(layer) > 0)])
    return ref, stats


@requires_cuda
@pytest.mark.parametrize("bs", [1, 2])
@pytest.mark.parametrize("policy", ["rule", "lru"])
def test_moe_prefetch_on_counters_match_reference(bs, policy, monkeypatch):
    """issued / useful / calls / rows / misses / copied equal a CPU replay of the spied candidates
    and routings, the slot maps end identical, late is 0 when each ensure waits for the copies
    (known answer), and a captured graph counts the same as eager. prefetch_ensure runs on the
    predictor stream, the slim prefetch copy on the copy stream, the demand copy on the compute one."""
    import freetoken.kernel.fast_index_copy as fic
    from freetoken.moe.prefetch import LATE, NUM_STAT_COLS, RESIDENT_HITS

    banks = _prefetch_banks("nvfp4")
    inputs = _prefetch_inputs(bs, steps=60, seed=8)
    moes, cache = _on_stack("nvfp4", banks, monkeypatch, policy=policy, overlap=True)
    calls, streams = [], {"pf_ensure": set(), "slim": set(), "demand": set()}
    real_pf, real_ensure = cache.prefetch_ensure, cache.ensure_experts
    real_slim, real_multi = fic.fast_index_copy_multi_slim_jit, fic.fast_index_copy_multi_jit

    def spy_pf(layer, query, *args):
        streams["pf_ensure"].add(torch.cuda.current_stream().cuda_stream)
        calls.append(("prefetch", layer, query.clone()))  # enqueued on the predictor stream after the select
        return real_pf(layer, query, *args)

    def spy_ensure(layer, ids, **kw):
        torch.cuda.synchronize()  # the copies forked before this ensure have landed: late must be 0
        logits = kw.get("router_logits")
        calls.append(("ensure", layer, ids.clone(), None if logits is None or policy != "rule" else logits.clone()))
        return real_ensure(layer, ids, **kw)

    def spy_slim(*args):
        streams["slim"].add(torch.cuda.current_stream().cuda_stream)
        return real_slim(*args)

    def spy_multi(*args):
        streams["demand"].add(torch.cuda.current_stream().cuda_stream)
        return real_multi(*args)

    def install():
        monkeypatch.setattr(cache, "prefetch_ensure", spy_pf)
        monkeypatch.setattr(cache, "ensure_experts", spy_ensure)
        monkeypatch.setattr(fic, "fast_index_copy_multi_slim_jit", spy_slim)
        monkeypatch.setattr(fic, "fast_index_copy_multi_jit", spy_multi)

    eager = _prefetch_run(moes, cache, bs, False, inputs, before_decode=install, sleep=0)
    torch.cuda.synchronize()
    calls = [c if c[0] == "prefetch" else (c[0], c[1], c[2].cpu(), None if c[3] is None else c[3].cpu()) for c in calls]
    calls = [(c[0], c[1], c[2].cpu()) if c[0] == "prefetch" else c for c in calls]
    ref, want = _replay_on_reference(policy, calls, bs)
    got = cache.prefetch.stats.cpu()
    assert got.shape == (_PF_LAYERS, NUM_STAT_COLS)
    assert int(got[:, LATE].sum()) == 0 and int(got[:, RESIDENT_HITS].sum()) == 0
    assert torch.equal(got, want), (got, want)
    assert int(want[:, 1].sum()) > 0, "the sequence must have useful prefetches"
    assert torch.equal(cache.slot_for_id.view(-1).cpu().long(), torch.from_numpy(ref.slot_of_id))
    assert torch.equal(cache.id_of_slot.cpu().long(), torch.from_numpy(ref.id_of_slot))
    assert torch.equal(cache.usage.cpu(), torch.from_numpy(ref.usage))
    assert streams["pf_ensure"] == {cache.prefetch.stream.cuda_stream}
    assert streams["slim"] == {cache.prefetch.copy_stream.cuda_stream}
    assert streams["demand"] == {torch.cuda.current_stream().cuda_stream}

    monkeypatch.undo()
    moes, cache = _on_stack("nvfp4", banks, monkeypatch, policy=policy, overlap=True)
    graph = _prefetch_run(moes, cache, bs, True, inputs, sleep=_PF_SLEEP_CYCLES)
    _assert_same_outputs(graph[0], eager[0])
    cols = [c for c in range(NUM_STAT_COLS) if c != LATE]
    assert torch.equal(cache.prefetch.stats.cpu()[:, cols], got[:, cols])


@requires_cuda
def test_moe_prefetch_on_rebuild_and_reset_clear_the_prefetch_state(monkeypatch):
    """A rebuild keeps both prefetch streams, the events and plan buffers, and drops the counters,
    the in-flight state and every low-priority slot; a recaptured graph then matches eager and off.
    A reset (as after a capture) also drops the counters and low-priority slots."""
    banks = _prefetch_banks("bf16")
    inputs = _prefetch_inputs(1, steps=40, seed=9)
    moes, cache = _on_stack("bf16", banks, monkeypatch, policy="rule", overlap=True)
    _prefetch_run(moes, cache, 1, True, inputs, sleep=0)
    pf = cache.prefetch
    streams, plan = (pf.stream, pf.copy_stream), (pf.pf_slots, pf.pf_src, pf.pf_num, pf.pf_ready)
    assert int(pf.stats.sum()) > 0 and bool(((cache.id_of_slot >= 0) & (cache.usage == 0)).any())
    pf._inflight[2] = torch.zeros(1)
    cache.rebuild(_PF_EXPERTS + 4)
    assert (pf.stream, pf.copy_stream) == streams and (pf.pf_slots, pf.pf_src, pf.pf_num, pf.pf_ready) == plan
    assert int(pf.stats.abs().sum()) == 0 and int(pf.totals.abs().sum()) == 0 and pf._inflight == [None] * _PF_LAYERS
    assert not bool(((cache.id_of_slot >= 0) & (cache.usage == 0)).any())
    graph = _prefetch_run(moes, cache, 1, True, inputs, sleep=0)
    eager = _prefetch_run(moes, cache, 1, False, inputs, sleep=0)
    _assert_same_outputs(graph[0], eager[0])
    _assert_slots_hold_their_experts(cache)
    assert int(pf.stats.sum()) > 0
    cache.reset()
    assert int(pf.stats.abs().sum()) == 0 and not bool(((cache.id_of_slot >= 0) & (cache.usage == 0)).any())
    moes_off, cache_off = _prefetch_stack("bf16", banks, "off", monkeypatch, overlap=True)
    cache_off.rebuild(_PF_EXPERTS + 4)
    off = _prefetch_run(moes_off, cache_off, 1, False, inputs, sleep=0)
    _assert_same_outputs(eager[0], off[0])


@requires_cuda
def test_moe_prefetch_on_skips_batches_past_the_select_tile(monkeypatch):
    """An eager decode wider than MAX_ROWS forks nothing: no join, no count, the same outputs."""
    from freetoken.moe.prefetch import MAX_ROWS

    banks = _prefetch_banks("bf16")
    bs = MAX_ROWS + 1
    inputs = _prefetch_inputs(bs, steps=6, seed=10)
    off = _prefetch_run(*_prefetch_stack("bf16", banks, "off", monkeypatch), bs, False, inputs, sleep=0)
    moes, cache = _on_stack("bf16", banks, monkeypatch, policy="rule", overlap=True)
    on = _prefetch_run(moes, cache, bs, False, inputs, sleep=0)
    _assert_same_outputs(on[0], off[0])
    assert int(cache.prefetch.stats.abs().sum()) == 0 and cache.prefetch._inflight == [None] * _PF_LAYERS


def _stall_hooks(monkeypatch):
    """Sleeps (cycles, read at enqueue time) on the compute stream right before each decode GEMM, i.e.
    after issue_copy(L+1), and on the stream that runs the shared expert gate (the side stream)."""
    import freetoken.models.qwen4_exp.moe as q4moe
    from freetoken.layers.moe import OffloadMoELayer

    stall = {"gemm": 0, "shared": 0}
    real_gemm, real_gate = OffloadMoELayer._expert_gemm, q4moe.shared_gate_sigmoid

    def gemm(self, cache, *args, is_prefill, **kw):
        if stall["gemm"] and not is_prefill:
            torch.cuda._sleep(stall["gemm"])
        return real_gemm(self, cache, *args, is_prefill=is_prefill, **kw)

    def gate(*args, **kw):
        if stall["shared"]:
            torch.cuda._sleep(stall["shared"])
        return real_gate(*args, **kw)

    monkeypatch.setattr(OffloadMoELayer, "_expert_gemm", gemm)
    monkeypatch.setattr(q4moe, "shared_gate_sigmoid", gate)
    return stall


@requires_cuda
@pytest.mark.parametrize("graph", [False, True], ids=["eager", "graph"])
@pytest.mark.parametrize("bs", [1, 2])
@pytest.mark.parametrize("policy", ["rule", "lru"])
@pytest.mark.parametrize("where", ["gemm", "shared"])
@pytest.mark.parametrize("cands", ["adversarial", "flood"])
def test_moe_prefetch_on_holds_when_the_compute_side_stalls(graph, bs, policy, where, cands, monkeypatch):
    """Race test from the other side: a sleep on the compute stream between issue_copy(L+1) and
    GEMM(L) lets copy(L+1) land before GEMM(L) reads, so a prefetch into any of layer L's routed
    slots changes outputs deterministically; a sleep on the shared expert's side stream must not
    matter either. 'flood' asks for all 16 experts a layer, more than the unpinned slots, so even
    LRU's recency order would reach the pinned ones. Bit-identical to off, and the same final maps
    as the unstalled run."""
    k, width = (8, 16) if cands == "flood" else (6, 12)
    banks = _prefetch_banks("nvfp4")
    inputs = _prefetch_inputs(bs, steps=80, seed=21)
    moes, cache = _prefetch_stack("nvfp4", banks, "off", monkeypatch, overlap=True, policy=policy)
    off = _prefetch_run(moes, cache, bs, graph, inputs, sleep=0)
    stall = _stall_hooks(monkeypatch)
    runs = []
    for cycles in (0, _PF_DELAY_CYCLES):
        stall[where] = cycles
        moes, cache = _on_stack("nvfp4", banks, monkeypatch, policy=policy, overlap=True, k=k, budget=width)
        steps = _garbage(_routed_ids(moes, inputs), width, cands, seed=bs + 20)
        on = _prefetch_run(moes, cache, bs, graph, inputs, sleep=0, before_step=_with_override(cache, steps))
        _assert_same_outputs(on[0], off[0])
        _assert_slots_hold_their_experts(cache)
        runs.append(on[1][:3])
    for a, b in zip(*runs):
        assert torch.equal(a, b)


def _assert_held_slots_hold_their_experts(cache):
    """_assert_slots_hold_their_experts, but a prefill may leave no slot held."""
    if bool((cache.id_of_slot >= 0).any()):
        _assert_slots_hold_their_experts(cache)


def _mixed_run(moes, cache, bs, graph, schedule, before_step=None):
    """Warm up (and capture) the decode like _prefetch_run, reset, then run ``schedule``: ("decode", xs)
    is one decode step (a replay under ``graph``), ("prefill", xs) an eager prefill of xs's tokens."""
    ctx = _fresh_ctx(_batch=SimpleNamespace(is_prefill=False))

    def step(xs):
        return [moe.forward(x.clone()) for moe, x in zip(moes, xs)]

    static = [torch.randn(bs, _PF_HIDDEN, device="cuda", dtype=torch.bfloat16) for _ in moes]
    step(static)
    if graph:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            graph_outs = step(static)
    cache.reset()
    results = []
    for i, (phase, xs) in enumerate(schedule):
        if before_step is not None:
            before_step(i)
        ctx._batch.is_prefill = phase == "prefill"
        if phase == "decode" and graph:
            for s, x in zip(static, xs):
                s.copy_(x)
            g.replay()
            outs = graph_outs
        else:
            outs = step(xs)
        results.append([o.clone() for o in outs])
        if phase == "prefill":
            torch.cuda.synchronize()
            _assert_held_slots_hold_their_experts(cache)
    torch.cuda.synchronize()
    _assert_held_slots_hold_their_experts(cache)
    return results


_PREFILL_PATHS = {
    # (cache_size, prefill tokens, cache kwargs, FREETOKEN_MOE_SMALL_PREFILL_TOKENS)
    "materialize": (_PF_EXPERTS + 4, 24, {}, 0),
    "overlap": (2 * _PF_EXPERTS + 4, 24, {"prefill_overlap": True}, 0),
    "overlap_d2d": (2 * _PF_EXPERTS + 4, 24, {"prefill_overlap": True, "prefill_hit_d2d": True}, 0),
    "small": (_PF_EXPERTS, 3, {}, 8),
}


@requires_cuda
@pytest.mark.parametrize("graph", [False, True], ids=["eager", "graph"])
@pytest.mark.parametrize("bs", [1, 2])
@pytest.mark.parametrize("policy", ["rule", "lru"])
@pytest.mark.parametrize("path", list(_PREFILL_PATHS))
def test_moe_prefetch_on_survives_prefills_between_decodes(graph, bs, policy, path, monkeypatch):
    """Prefills between decode steps rewrite slots behind the prefetch's back: the whole-layer
    materialize installs slots [0, E), the overlap double buffers invalidate and stream [0, 2E)
    (with the hit-D2D gather reading resident slots), and small prefills ensure their experts
    under the low-priority keys. Decode, prefill, decode, prefill, decode: every output of prefetch
    on equals off to the bit, and after each prefill every held slot holds its expert."""
    import freetoken.layers.moe as layers_moe

    cache_size, tokens, cache_kw, small = _PREFILL_PATHS[path]
    monkeypatch.setattr(layers_moe, "_SMALL_PREFILL_TOKENS", small)
    banks = _prefetch_banks("nvfp4")
    decode = _prefetch_inputs(bs, steps=18, seed=31)
    gen = torch.Generator(device="cuda").manual_seed(32)
    prefills = [[torch.randn(tokens, _PF_HIDDEN, device="cuda", dtype=torch.bfloat16, generator=gen) * 0.5
                 for _ in range(_PF_LAYERS)] for _ in range(2)]
    schedule = ([("decode", xs) for xs in decode[:6]] + [("prefill", prefills[0])]
                + [("decode", xs) for xs in decode[6:12]] + [("prefill", prefills[1])]
                + [("decode", xs) for xs in decode[12:]])
    kw = dict(overlap=True, policy=policy, cache_size=cache_size, **cache_kw)
    moes, cache = _prefetch_stack("nvfp4", banks, "off", monkeypatch, **kw)
    off = _mixed_run(moes, cache, bs, graph, schedule)
    moes, cache = _prefetch_stack("nvfp4", banks, "on", monkeypatch, budget=8, **kw)
    steps = _garbage(_routed_ids(moes, [xs for _, xs in schedule]), 12, "adversarial", seed=bs + 30)
    on = _mixed_run(moes, cache, bs, graph, schedule, before_step=_with_override(cache, steps))
    _assert_same_outputs(on, off)
    if path == "overlap_d2d":
        assert cache.prefill_hit_rows > 0, "the hit-D2D gather must have read resident slots"
    assert int(cache.prefetch.stats[:, 0].sum()) > 0


@requires_cuda
@pytest.mark.parametrize("kind", ["bf16", "nvfp4"])
@pytest.mark.parametrize("graph", [False, True], ids=["eager", "graph"])
@pytest.mark.parametrize("bs", [1, 2])
def test_moe_prefetch_on_per_bank_copy_fallback_is_bitwise_identical(kind, graph, bs, monkeypatch):
    """Without the fused copy plan (FREETOKEN_FUSED_COPY=0 or misaligned banks) the prefetch copy
    takes copy_rows' per-bank launches on the copy stream; late copies still change no bit."""
    import freetoken.moe.offload_cache as offload_cache

    monkeypatch.setattr(offload_cache, "_FUSED_COPY", False)
    banks = _prefetch_banks(kind)
    inputs = _prefetch_inputs(bs, steps=60, seed=5)
    moes, cache = _prefetch_stack(kind, banks, "off", monkeypatch, overlap=True)
    off = _prefetch_run(moes, cache, bs, graph, inputs, sleep=0)
    moes, cache = _on_stack(kind, banks, monkeypatch, policy="rule", overlap=True, budget=8)
    assert not cache._copy_fused_ok
    steps = _garbage(_routed_ids(moes, inputs), 12, "adversarial", seed=bs)
    on = _prefetch_run(moes, cache, bs, graph, inputs, sleep=0,
                       before_step=_with_override(cache, steps, delay_copy_cycles=_PF_DELAY_CYCLES // 2))
    _assert_same_outputs(on[0], off[0])
    _assert_slots_hold_their_experts(cache)
    assert int(cache.prefetch.stats[:, 0].sum()) > 0


def _multi_graph_run(moes, cache, schedule, before_step=None):
    """The graph runner's sequence: per graph size (largest first) an eager warm-up, a capture into
    one shared pool and a reset; then ``schedule`` of (bs, xs, replay) steps, replays and eager
    steps interleaved."""
    _fresh_ctx(_batch=SimpleNamespace(is_prefill=False))

    def step(xs):
        return [moe.forward(x.clone()) for moe, x in zip(moes, xs)]

    graphs, pool = {}, None
    for bs in (2, 1):
        static = [torch.randn(bs, _PF_HIDDEN, device="cuda", dtype=torch.bfloat16) for _ in moes]
        step(static)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=pool):
            outs = step(static)
        cache.reset()
        pool = g.pool() if pool is None else pool
        graphs[bs] = (g, static, outs)
    results = []
    for i, (bs, xs, replay) in enumerate(schedule):
        if before_step is not None:
            before_step(i)
        if replay:
            g, static, outs = graphs[bs]
            for s, x in zip(static, xs):
                s.copy_(x)
            g.replay()
        else:
            outs = step(xs)
        results.append([o.clone() for o in outs])
    torch.cuda.synchronize()
    return results


@requires_cuda
@pytest.mark.parametrize("policy", ["rule", "lru"])
def test_moe_prefetch_on_interleaves_graph_sizes_and_eager_steps(policy, monkeypatch):
    """bs 1 and bs 2 graphs captured into one pool (as the graph runner does), replayed in an
    arbitrary order with eager steps between them, including an eager bs 9 step that forks nothing:
    every output equals off's to the bit and every held slot holds its expert."""
    from freetoken.moe.prefetch import MAX_ROWS

    banks = _prefetch_banks("nvfp4")
    gen = torch.Generator().manual_seed(41)
    order = [(int(b), bool(r)) for b, r in zip(torch.randint(0, 4, (60,), generator=gen), torch.randint(0, 2, (60,), generator=gen))]
    sizes = {0: 1, 1: 2, 2: 3, 3: MAX_ROWS + 1}
    xs_by_bs = {bs: iter(_prefetch_inputs(bs, steps=60, seed=42 + bs)) for bs in sizes.values()}
    schedule = []
    for b, r in order:
        bs = sizes[b]
        schedule.append((bs, next(xs_by_bs[bs]), r and bs <= 2))
    moes, cache = _prefetch_stack("nvfp4", banks, "off", monkeypatch, overlap=True, policy=policy)
    off = _multi_graph_run(moes, cache, schedule)
    moes, cache = _on_stack("nvfp4", banks, monkeypatch, policy=policy, overlap=True, budget=8)
    steps = _garbage(_routed_ids(moes, [xs for _, xs, _ in schedule]), 12, "adversarial", seed=43)
    on = _multi_graph_run(moes, cache, schedule, before_step=_with_override(cache, steps))
    _assert_same_outputs(on, off)
    _assert_slots_hold_their_experts(cache)
    assert cache.prefetch._inflight == [None] * _PF_LAYERS
    assert int(cache.prefetch.stats[:, 0].sum()) > 0 and int(cache.prefetch.stats[:, 1].sum()) > 0


@requires_cuda
@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("rows", ["bs1", "bs2", "bs2_same"])
@pytest.mark.parametrize("policy", ["rule", "lru"])
def test_moe_prefetch_on_fuzzes_per_layer_stream_delays(seed, rows, policy, monkeypatch):
    """Eager fuzz: before every layer a random choice of sleeps on the predictor stream, the prefetch
    copy stream, the compute stream before the GEMM and the shared expert's side stream, with up to
    12 candidates a layer (most of the 16 slots re-installed every call) and, for bs2_same, two
    identical rows. Outputs equal off's, held slots hold their experts, and the final maps and
    stats (all but LATE) equal those of the same run without delays."""
    from freetoken.moe.prefetch import LATE, NUM_STAT_COLS

    bs = 1 if rows == "bs1" else 2
    inputs = _prefetch_inputs(bs, steps=60, seed=50 + seed)
    if rows == "bs2_same":
        inputs = [[x[:1].expand(2, -1).contiguous() for x in xs] for xs in inputs]
    banks = _prefetch_banks("nvfp4")
    moes, cache = _prefetch_stack("nvfp4", banks, "off", monkeypatch, overlap=True, policy=policy)
    off = _prefetch_run(moes, cache, bs, False, inputs, sleep=0)
    stall = _stall_hooks(monkeypatch)
    gen = torch.Generator().manual_seed(seed)
    choices = torch.randint(0, 16, (len(inputs), _PF_LAYERS), generator=gen).tolist()
    runs = []
    for delayed in (False, True):
        moes, cache = _on_stack("nvfp4", banks, monkeypatch, policy=policy, overlap=True, budget=12)
        steps = _garbage(_routed_ids(moes, inputs), 12, "adversarial", seed=60 + seed)
        pf = cache.prefetch
        load = _with_override(cache, steps)
        at = {"step": -1}

        def before_step(i, load=load):
            at["step"] = i
            load(i)

        for layer, moe in enumerate(moes):
            real = moe.forward

            def forward(x, layer=layer, real=real, pf=pf):
                bits = choices[at["step"]][layer] if delayed and at["step"] >= 0 else 0
                # layer L's fork runs layer L+1's prediction and copy: these delays hit them
                pf.delay_predict_cycles = _PF_DELAY_CYCLES // 2 if bits & 1 else 0
                pf.delay_copy_cycles = _PF_DELAY_CYCLES // 2 if bits & 2 else 0
                stall["gemm"] = _PF_DELAY_CYCLES // 2 if bits & 4 else 0
                stall["shared"] = _PF_DELAY_CYCLES // 2 if bits & 8 else 0
                return real(x)

            monkeypatch.setattr(moe, "forward", forward)
        on = _prefetch_run(moes, cache, bs, False, inputs, sleep=0, before_step=before_step)
        _assert_same_outputs(on[0], off[0])
        _assert_slots_hold_their_experts(cache)
        runs.append((on[1][:3], pf.stats.cpu()))
    (maps, stats), (delayed_maps, delayed_stats) = runs
    for a, b in zip(maps, delayed_maps):
        assert torch.equal(a, b)
    cols = [c for c in range(NUM_STAT_COLS) if c != LATE]
    assert torch.equal(stats[:, cols], delayed_stats[:, cols])


@requires_cuda
def test_decoder_stack_prefill_and_decode(monkeypatch):
    """Ragged bs=3 prefill then a bs=3 decode step through the whole model with dummy weights."""
    from freetoken.kvcache.linear_state_pool import LinearStatePool
    from freetoken.models.qwen4_exp import model as model_module
    from freetoken.models.qwen4_exp.attention import TorchDenseQSAReference
    from freetoken.models.qwen4_exp.ple import GpuResidentTable
    from freetoken.utils.torch_utils import torch_dtype

    torch.manual_seed(8)
    config = _config()
    args = config.qwen4_args
    device, dtype = torch.device("cuda"), torch.bfloat16
    monkeypatch.setattr(model_module, "build_linear_mixer", _StubLinearMixer)

    with torch.device(device), torch_dtype(dtype):
        model = model_module.Qwen4ExpForCausalLM(config)
    gen = torch.Generator(device=device).manual_seed(9)
    _fill(model, gen)
    multipliers, sizes, offsets = hash_constants(args)
    table = torch.randn(4096, args.ngram_head_dim, generator=gen, device=device, dtype=dtype) * 0.05
    for ple in model.model.ple_layers:
        ple.ple_embedding.layer_multipliers.copy_(multipliers)
        ple.ple_embedding.ngram_heads_vocab_sizes.copy_(sizes)
        ple.ple_embedding.ngram_heads_offsets.copy_(offsets)
        ple.ple_embedding.attach_table(GpuResidentTable(table, dtype=dtype))

    num_slots, max_len = 4, 64
    pool = LinearStatePool(
        config.linear_attention_group(), num_slots, dtype, device,
        slot_states=config.slot_states,
    )
    prompts = [[3, 4, EOS, 5, 6, 8], [2, EOS, 11, 12], [9, 10, 11, 12, 13]]
    ctx = _fresh_ctx(
        attn_backend=TorchDenseQSAReference(config, num_slots, max_len, device, dtype),
        linear_state_pool=pool,
    )
    reqs = [
        SimpleNamespace(
            extend_len=len(p), cached_len=0, table_idx=i + 1, linear_slot_idx=None,
            input_ids=torch.tensor(p, dtype=torch.int64),
        )
        for i, p in enumerate(prompts)
    ]
    flat = [t for p in prompts for t in p]
    last = torch.tensor(
        [sum(len(p) for p in prompts[: i + 1]) - 1 for i in range(len(prompts))], device=device
    )
    positions = torch.cat([torch.arange(len(p)) for p in prompts]).to(device)
    batch = SimpleNamespace(
        padded_reqs=reqs, reqs=reqs, size=len(reqs), is_prefill=True, is_decode=False,
        input_ids=torch.tensor(flat, dtype=torch.int64, device=device),
        positions=positions, get_attn_positions=lambda: positions, mm_embeds=None,
        attn_metadata=SimpleNamespace(get_last_indices=lambda bs: last[:bs]),
    )
    with ctx.forward_batch(batch):
        logits = model.forward()
    assert logits.shape == (len(prompts), config.vocab_size)
    assert torch.isfinite(logits.float()).all()

    for r, p in zip(reqs, prompts):
        r.cached_len = len(p)
        r.extend_len = 1
        r.input_ids = torch.cat([r.input_ids, torch.tensor([14], dtype=torch.int64)])
    decode_positions = torch.tensor([len(p) for p in prompts], dtype=torch.int64, device=device)
    decode = SimpleNamespace(
        padded_reqs=reqs, reqs=reqs, size=len(reqs), is_prefill=False, is_decode=True,
        input_ids=torch.tensor([14] * len(reqs), dtype=torch.int64, device=device),
        positions=decode_positions, get_attn_positions=lambda: decode_positions, mm_embeds=None,
        attn_metadata=None,
    )
    with ctx.forward_batch(decode):
        decode_logits = model.forward()
    assert decode_logits.shape == (len(prompts), config.vocab_size)
    assert torch.isfinite(decode_logits.float()).all()
