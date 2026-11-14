"""TP sharding of the qwen4_exp dense weights (pure tensor math, no TP runtime needed)."""

from types import SimpleNamespace

import torch
from freetoken.models.qwen4_exp.weight import _shard, _shard_rows


def _cfg():
    g = SimpleNamespace(
        num_key_heads=4, key_head_dim=8, num_value_heads=12, value_head_dim=8
    )
    return SimpleNamespace(
        num_qo_heads=6, num_kv_heads=2, head_dim=8, linear_attention_group=lambda: g
    )


def _gather(name, t, cfg, world, dim=0):
    return torch.cat([_shard(name, t, cfg, r, world) for r in range(world)], dim=dim)


def test_shard_rows_splits_each_part_by_head():
    t = torch.arange(4 * 4 + 2 * 4).reshape(
        -1, 1
    )  # part A: 4 heads x 4 rows, part B: 2 heads x 4
    s = [_shard_rows(t, [(4, 4), (2, 4)], r, 4) for r in range(4)]
    assert all(x.shape[0] == 8 for x in s)
    assert s[0][:4].flatten().tolist() == [0, 1, 2, 3]
    assert s[3][:4].flatten().tolist() == [12, 13, 14, 15]
    # 2 kv heads over 4 ranks replicate: rank * heads // world -> heads 0, 0, 1, 1
    assert s[0][4:].equal(s[1][4:]) and s[2][4:].flatten().tolist() == [20, 21, 22, 23]


def test_shard_round_trips_the_fused_projections():
    cfg, world = _cfg(), 2
    qkv = torch.randn(6 * 16 + 2 * 8 + 2 * 8, 5)
    assert (
        _gather("model.layers.0.self_attn.qkv_proj.weight", qkv, cfg, world).shape
        == qkv.shape
    )
    q0 = _shard("model.layers.0.self_attn.qkv_proj.weight", qkv, cfg, 0, world)
    assert q0.shape[0] == 3 * 16 + 8 + 8
    torch.testing.assert_close(q0[:48], qkv[:48])  # q heads 0-2 (16 rows each: q|gate)
    torch.testing.assert_close(q0[48:56], qkv[96:104])  # kv head 0 of k
    torch.testing.assert_close(q0[56:64], qkv[112:120])  # kv head 0 of v
    kd, vd, nv = 4 * 8, 12 * 8, 12
    in_proj = torch.randn(kd + kd + vd + vd + nv + nv, 5)
    s = _shard("model.layers.1.linear_attn.in_proj.weight", in_proj, cfg, 1, world)
    assert s.shape[0] == (kd + kd + vd + vd + nv + nv) // 2
    torch.testing.assert_close(s[:16], in_proj[16:32])  # q: k heads 2,3
    torch.testing.assert_close(s[-6:], in_proj[-6:])  # a: v heads 6-11
    conv = torch.randn(kd + kd + vd, 1, 4)
    assert _shard(
        "model.layers.1.linear_attn.conv1d.weight", conv, cfg, 0, world
    ).shape == ((kd + kd + vd) // 2, 1, 4)
    a_log = torch.randn(nv)
    torch.testing.assert_close(
        _shard("model.layers.1.linear_attn.A_log", a_log, cfg, 1, world), a_log[6:]
    )


def test_shard_row_parallel_and_vocab_and_replicated():
    cfg, world = _cfg(), 2
    for name in (
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.1.linear_attn.out_proj.weight",
        "model.layers.0.mlp.shared_expert.down_proj.weight",
    ):
        t = torch.randn(3, 8)
        torch.testing.assert_close(_gather(name, t, cfg, world, dim=1), t)
    gate_up = torch.randn(2 * 6, 3)
    s1 = _shard(
        "model.layers.0.mlp.shared_expert.gate_up_proj.weight", gate_up, cfg, 1, world
    )
    torch.testing.assert_close(s1, torch.cat([gate_up[3:6], gate_up[9:12]]))
    emb = torch.randn(10, 3)
    torch.testing.assert_close(
        _gather("model.embed_tokens.weight", emb, cfg, world), emb
    )
    torch.testing.assert_close(_gather("lm_head.weight", emb, cfg, world), emb)
    for name in (
        "model.layers.0.mlp.gate.weight",
        "model.layers.0.self_attn.indexer.index_qk_proj.weight",
        "model.layers.0.attn_hyper_connection.input_mix_weight_down_block_inject.weight",
        "model.layers.1.ple.value_proj.weight",
        "model.layers.0.mlp.shared_expert_gate.weight",
    ):
        t = torch.randn(4, 6)
        assert _shard(name, t, cfg, 1, world).equal(t)
    assert _shard("model.layers.0.self_attn.qkv_proj.weight", gate_up, cfg, 0, 1).equal(
        gate_up
    )


def _vision_cfg(heads=4, hidden=16):
    return SimpleNamespace(
        vision_config=SimpleNamespace(num_heads=heads, hidden_size=hidden)
    )


def test_vision_block_partial_sums_equal_the_whole_tower_block():
    # Each rank runs what LinearOProj / LinearRowParallel run: a GEMM WITH its own bias, then an
    # all-reduce (a sum over ranks). The per-rank outputs of the sharded tensors must add up to
    # the unsharded block, so a row-parallel bias may reach the sum only once.
    torch.manual_seed(0)
    cfg, world, heads, hidden, inter, S = _vision_cfg(), 2, 4, 16, 24, 5
    hd = hidden // heads
    p = "visual.blocks.0."
    w = {
        p + "attn.qkv.weight": torch.randn(3 * hidden, hidden),
        p + "attn.qkv.bias": torch.randn(3 * hidden),
        p + "attn.proj.weight": torch.randn(hidden, hidden),
        p + "attn.proj.bias": torch.randn(hidden),
        p + "mlp.linear_fc1.weight": torch.randn(inter, hidden),
        p + "mlp.linear_fc1.bias": torch.randn(inter),
        p + "mlp.linear_fc2.weight": torch.randn(hidden, inter),
        p + "mlp.linear_fc2.bias": torch.randn(hidden),
    }
    x = torch.randn(S, hidden)

    def attn(t, n):
        qkv = x @ t[p + "attn.qkv.weight"].T + t[p + "attn.qkv.bias"]
        q, k, v = qkv.view(S, 3, n, hd).unbind(1)
        o = torch.softmax(torch.einsum("shd,thd->hst", q, k) / hd**0.5, -1)
        o = torch.einsum("hst,thd->shd", o, v).reshape(S, n * hd)
        return o @ t[p + "attn.proj.weight"].T + t[p + "attn.proj.bias"]

    def mlp(t):
        h = torch.nn.functional.gelu(
            x @ t[p + "mlp.linear_fc1.weight"].T + t[p + "mlp.linear_fc1.bias"],
            approximate="tanh",
        )
        return h @ t[p + "mlp.linear_fc2.weight"].T + t[p + "mlp.linear_fc2.bias"]

    sh = [{k: _shard(k, v, cfg, r, world) for k, v in w.items()} for r in range(world)]
    for r in range(world):
        assert sh[r][p + "attn.qkv.weight"].shape == (3 * hidden // world, hidden)
    tp_attn = sum(attn(s, heads // world) for s in sh)
    tp_mlp = sum(mlp(s) for s in sh)
    torch.testing.assert_close(tp_attn, attn(w, heads), rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(tp_mlp, mlp(w), rtol=1e-4, atol=1e-4)


def test_vision_merger_and_replicated_tower_tensors():
    cfg, world = _vision_cfg(), 2
    fc1, fc2 = torch.randn(64, 64), torch.randn(8, 64)
    torch.testing.assert_close(
        _gather("visual.merger.linear_fc1.weight", fc1, cfg, world), fc1
    )
    torch.testing.assert_close(
        _gather("visual.merger.linear_fc2.weight", fc2, cfg, world, dim=1), fc2
    )
    for name in (
        "visual.patch_embed.proj.weight",
        "visual.pos_embed.weight",
        "visual.blocks.0.norm1.weight",
        "visual.merger.norm.bias",
    ):
        t = torch.randn(4, 6)
        assert _shard(name, t, cfg, 1, world).equal(t)


def _q8_0_rows(rows: int, k: int, seed: int):
    """Packed Q8_0 rows [rows, k // 32 * 34]: an f16 scale (some negative) then 32 int8 codes per block."""
    import numpy as np

    rng = np.random.default_rng(seed)
    nb = rows * k // 32
    d = rng.uniform(-0.02, 0.02, nb).astype(np.float16).view(np.uint8).reshape(nb, 2)
    q = rng.integers(-127, 128, (nb, 32), dtype=np.int8).view(np.uint8)
    return np.concatenate([d, q], axis=1).reshape(rows, k // 32 * 34)


def test_gguf_expert_slice_keeps_whole_blocks_and_partial_sums():
    # TP slices a packed down row by bytes at block boundaries; an element-wise cut would decode garbage
    import numpy as np
    from freetoken.models.gguf.experts import _shard_intermediate
    from freetoken.third_party.gguf import quants
    from freetoken.third_party.gguf.constants import GGMLQuantizationType as QT

    E, inter, hidden, world = 2, 64, 32, 2
    gate, up = (torch.from_numpy(_q8_0_rows(E * inter, hidden, s)).view(E, inter, -1) for s in (0, 1))
    down = torch.from_numpy(_q8_0_rows(E * hidden, inter, 2)).view(E, hidden, -1)
    dq = lambda t, k: quants.dequantize(t.reshape(-1, t.shape[-1]).numpy(), QT.Q8_0).reshape(*t.shape[:-1], k)  # noqa: E731
    full = [dq(gate, hidden), dq(up, hidden), dq(down, inter)]
    x = np.random.default_rng(3).standard_normal(hidden).astype(np.float32)

    def mlp(g, u, d):
        a = g @ x
        return d @ (a / (1 + np.exp(-a)) * (u @ x))

    total = 0
    for rank in range(world):
        (g, u), d = _shard_intermediate([gate, up], down, int(QT.Q8_0), inter, rank, world)
        lo, n = rank * inter // world, inter // world
        assert np.array_equal(dq(g, hidden), full[0][:, lo : lo + n])
        assert np.array_equal(dq(u, hidden), full[1][:, lo : lo + n])
        assert np.array_equal(dq(d, n), full[2][:, :, lo : lo + n])
        total = total + mlp(dq(g, hidden)[0], dq(u, hidden)[0], dq(d, n)[0])
    np.testing.assert_allclose(total, mlp(full[0][0], full[1][0], full[2][0]), rtol=1e-5, atol=1e-6)


def test_gguf_expert_slice_refuses_a_k_quant_down_that_splits_a_block():
    # a 256-value K-quant down row at I=640 cannot split into two 320-value halves
    import pytest
    from freetoken.models.gguf.experts import _shard_intermediate
    from freetoken.third_party.gguf.constants import GGMLQuantizationType as QT

    down = torch.zeros(1, 4, 640 // 256 * 144 + 144, dtype=torch.uint8)
    with pytest.raises(NotImplementedError, match="cannot split"):
        _shard_intermediate([torch.zeros(1, 640, 1, dtype=torch.uint8)] * 2, down, int(QT.Q4_K), 640, 0, 2)



def test_gguf_float_dense_is_its_own_switch(monkeypatch):
    # the FP8 switches alone must leave a GGUF's own ggml types in charge (the checkpoint's scheme wins at TP=1)
    from freetoken.models.qwen4_exp.gguf import gguf_float_dense, gguf_float_modules

    monkeypatch.delenv("FREETOKEN_GGUF_FLOAT_DENSE", raising=False)
    monkeypatch.setenv("FREETOKEN_FP8_DENSE", "1")
    monkeypatch.setenv("FREETOKEN_FP8_LMHEAD", "1")
    assert not gguf_float_dense() and gguf_float_modules() == ()
    monkeypatch.setenv("FREETOKEN_GGUF_FLOAT_DENSE", "1")
    assert gguf_float_dense() and gguf_float_modules()


def test_gguf_float_globs_cover_every_dense_checkpoint_name_but_no_expert():
    from freetoken.layers.quantization.names import name_set
    from freetoken.models.qwen4_exp.gguf import _FLOAT_DENSE_MODULES, _LAYER_LEAVES, _TOP_LEAVES

    hit = name_set(_FLOAT_DENSE_MODULES)
    dense = [f"model.language_model.layers.3.{leaf}" for leaf in _LAYER_LEAVES if not leaf.startswith("mlp.experts.")]
    dense += list(_TOP_LEAVES)
    assert [n for n in dense if not hit(n)] == []
    assert not any(hit(f"model.language_model.layers.3.mlp.experts.0.{p}_proj") for p in ("gate", "up", "down"))


def test_packed_gguf_refuses_tp_without_float_dense(monkeypatch):
    import freetoken.distributed as dist
    import pytest
    from freetoken.models.qwen4_exp import gguf as g

    monkeypatch.delenv("FREETOKEN_GGUF_FLOAT_DENSE", raising=False)
    monkeypatch.setattr(g, "parse_gguf_config", lambda shim: SimpleNamespace())
    monkeypatch.setattr("freetoken.utils.cached_load_hf_config", lambda path: None)
    monkeypatch.setattr(dist, "try_get_tp_info", lambda: SimpleNamespace(rank=0, size=2))
    with pytest.raises(ValueError, match="FREETOKEN_GGUF_FLOAT_DENSE=1"):
        next(g.iter_gguf_weights("x.gguf", torch.device("cpu"), include_moe_experts=False, include_non_moe=True))


def test_float_reader_applies_an_nvfp4_weight_scale_per_row(monkeypatch):
    # the packed path scales an NVFP4 tensor's rows by its weight_scale_2; the float path must not drop it
    from freetoken.layers.quantization.ggml_types import NVFP4
    from freetoken.models.qwen4_exp import gguf as g

    w = torch.ones(3, 32)
    monkeypatch.setattr(g, "dequantized_tensor", lambda info: w)
    for scale, expect in ((torch.tensor([0.5]), w * 0.5), (torch.tensor([1.0, 2.0, 3.0]), w * torch.tensor([[1.0], [2.0], [3.0]]))):
        monkeypatch.setattr(g, "weight_scale", lambda info, s=scale: s)
        info = SimpleNamespace(ggml_type=NVFP4, name="t")
        got = g._DenseReader({}, SimpleNamespace(hidden_size=32), float_dense=True).mat(info)
        assert torch.allclose(got.float(), expect)
