"""Per-output-channel FP8 weight scales for the load-time FP8 dense layers and lm_head.

``FREETOKEN_FP8_DENSE_SCALE=channel`` / ``FREETOKEN_FP8_LMHEAD_SCALE=channel`` give one fp32 weight scale per
output row (per rank, after the TP shard) in place of one per tensor. The default (``tensor``) must stay
bit-identical to the per-tensor path. CPU tests cover the env, the load side, the TP shards, the ops and
the channel GEMM branch with its two Triton steps replaced by torch; the rest needs a GPU.

E4M3 is a float format: its rounding error is relative (3 mantissa bits, about 2.6 % RMS) and does not
depend on the scale until a value goes subnormal (below amax / 2**15). A finer scale thus helps only
rows whose range is far below the tensor's. The tests assert "not worse" on normal weights and
"lower" only on rows that per-tensor scaling pushes into the subnormal range.
"""

from __future__ import annotations

from types import SimpleNamespace

import freetoken.distributed.info as dist_info
import pytest
import torch
from freetoken.distributed import DistributedInfo
from freetoken.models.qwen4_exp.weight import _fp8_dense, _quantize_per_tensor, _shard

from .common import requires_cuda

E4M3_MAX = 448.0
FP8 = torch.float8_e4m3fn


def _old_per_tensor(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The per-tensor reader formula at dcb708a1, copied: the default mode must reproduce it bit for bit."""
    w = w.float()
    scale = (w.abs().amax() / E4M3_MAX).clamp_min(1e-12)
    return (w / scale).clamp_(-E4M3_MAX, E4M3_MAX).to(FP8), scale.reshape(())


def _deq(w8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return w8.float() * (scale[:, None] if scale.ndim == 1 else scale)


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a - b).norm() / b.norm()).item()


def _row_rel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a - b).norm(dim=1) / b.norm(dim=1)


def _cfg(**extra):
    g = SimpleNamespace(
        num_key_heads=4, key_head_dim=8, num_value_heads=12, value_head_dim=8
    )
    return SimpleNamespace(linear_attention_group=lambda: g, **extra)


# ---------------------------------------------------------------------------------------------- env


@pytest.mark.parametrize(
    "var, fn",
    [
        ("FREETOKEN_FP8_DENSE_SCALE", "fp8_dense_scale"),
        ("FREETOKEN_FP8_LMHEAD_SCALE", "fp8_lmhead_scale"),
    ],
)
def test_scale_mode_env_parsing(monkeypatch, var, fn):
    import freetoken.models.config as mc

    resolve = getattr(mc, fn)
    monkeypatch.delenv(var, raising=False)
    assert resolve() == "tensor"
    for raw, want in (
        ("tensor", "tensor"),
        ("channel", "channel"),
        (" Channel ", "channel"),
        ("", "tensor"),
    ):
        monkeypatch.setenv(var, raw)
        assert resolve() == want, raw
    for bad in ("row", "group", "block128", "1"):
        monkeypatch.setenv(var, bad)
        with pytest.raises(ValueError, match=var):
            resolve()


def test_dense_and_lmhead_scale_modes_are_independent(monkeypatch):
    from freetoken.models.config import fp8_dense_scale, fp8_lmhead_scale

    monkeypatch.setenv("FREETOKEN_FP8_DENSE_SCALE", "channel")
    monkeypatch.delenv("FREETOKEN_FP8_LMHEAD_SCALE", raising=False)
    assert (fp8_dense_scale(), fp8_lmhead_scale()) == ("channel", "tensor")
    monkeypatch.delenv("FREETOKEN_FP8_DENSE_SCALE", raising=False)
    monkeypatch.setenv("FREETOKEN_FP8_LMHEAD_SCALE", "channel")
    assert (fp8_dense_scale(), fp8_lmhead_scale()) == ("tensor", "channel")


# ---------------------------------------------------------------------------------------- load side


def test_quantize_per_channel_round_trips_within_e4m3():
    from freetoken.models.qwen4_exp.weight import _quantize_per_channel

    w = torch.randn(64, 32) * torch.logspace(-3, 0, 64)[:, None]
    w[5] = 0  # an all-zero row must give a finite scale and zero codes
    w8, scale = _quantize_per_channel(w)
    assert w8.dtype is FP8 and w8.shape == w.shape
    assert scale.dtype == torch.float32 and scale.shape == (64,)
    assert torch.isfinite(scale).all() and (w8[5].float() == 0).all()
    nz = torch.arange(64) != 5
    assert torch.allclose(scale[nz] * E4M3_MAX, w[nz].abs().amax(dim=1))
    # every row uses the top of the e4m3 range, so each row is in the normal range and rounds to 2^-4
    tol = w.abs() * 2**-4 + scale[:, None] * 2**-9 + 1e-12
    assert ((_deq(w8, scale) - w).abs() <= tol).all()


def test_channel_error_is_not_worse_than_tensor_on_normal_weights():
    """Gaussian weights at a real shape (in_proj_qkvz rank shard, K = 2560): about 2.6 % for both."""
    from freetoken.models.qwen4_exp.weight import _quantize_per_channel

    torch.manual_seed(0)
    w = (torch.randn(1024, 2560) * 0.02).to(torch.bfloat16).float()
    t_err = _rel(_deq(*_quantize_per_tensor(w)), w)
    c_err = _rel(_deq(*_quantize_per_channel(w)), w)
    assert 0.02 < t_err < 0.03 and 0.02 < c_err < 0.03
    assert c_err <= t_err * 1.01
    # the worst row gains a little: its own range starts at the e4m3 top
    assert (
        _row_rel(_deq(*_quantize_per_channel(w)), w).max()
        <= _row_rel(_deq(*_quantize_per_tensor(w)), w).max()
    )


def test_channel_error_is_lower_on_rows_that_per_tensor_makes_subnormal():
    """One loud row in 97 at 1e4x: per-tensor scaling pushes the quiet rows into the e4m3 subnormals."""
    from freetoken.models.qwen4_exp.weight import _quantize_per_channel

    torch.manual_seed(0)
    w = torch.randn(970, 2560) * 0.02
    loud = torch.arange(970) % 97 == 0
    w[loud] *= 1e4
    w = w.to(torch.bfloat16).float()
    t_rows = _row_rel(_deq(*_quantize_per_tensor(w)), w)[~loud]
    c_rows = _row_rel(_deq(*_quantize_per_channel(w)), w)[~loud]
    assert c_rows.mean() < 0.03
    assert c_rows.mean() < t_rows.mean() / 2  # measured: 2.6 % against 6.4 %


def test_default_mode_is_bit_identical_to_the_per_tensor_reader():
    cfg, world = _cfg(), 2
    torch.manual_seed(1)
    t = torch.randn(128 + 12, 16, dtype=torch.bfloat16)
    out = dict(_fp8_dense("model.layers.3.linear_attn.in_proj.weight", t, cfg, world))
    w8, s = _old_per_tensor(t[:128])
    assert torch.equal(
        out["model.layers.3.linear_attn.in_proj_qkvz.weight"].view(torch.uint8),
        w8.view(torch.uint8),
    )
    assert torch.equal(out["model.layers.3.linear_attn.in_proj_qkvz.weight_scale"], s)
    t = torch.randn(32, 16, dtype=torch.bfloat16)
    for name, kw in (
        ("x.self_attn.qkv_proj.weight", {}),
        ("x.self_attn.o_proj.weight", {}),
        ("x.linear_attn.out_proj.weight", {}),
        ("lm_head.weight", {"dense": False, "lm_head": True}),
    ):
        out = dict(_fp8_dense(name, t, cfg, 1, **kw))
        w8, s = _old_per_tensor(t)
        assert torch.equal(out[name].view(torch.uint8), w8.view(torch.uint8)), name
        sname = name[: -len("weight")] + "weight_scale"
        assert out[sname].shape == () and torch.equal(out[sname], s), name


def test_channel_mode_emits_one_scale_per_output_row():
    from freetoken.models.qwen4_exp.weight import _quantize_per_channel

    cfg = _cfg()
    t = torch.randn(
        128 + 12, 16, dtype=torch.bfloat16
    )  # world 2: qkvz = 128 local rows, ba = 12
    out = dict(
        _fp8_dense("m.linear_attn.in_proj.weight", t, cfg, 2, dense_channel=True)
    )
    w8, s = _quantize_per_channel(t[:128])
    assert out["m.linear_attn.in_proj_qkvz.weight_scale"].shape == (128,)
    assert torch.equal(
        out["m.linear_attn.in_proj_qkvz.weight"].view(torch.uint8), w8.view(torch.uint8)
    )
    assert torch.equal(out["m.linear_attn.in_proj_qkvz.weight_scale"], s)
    assert torch.equal(out["m.linear_attn.in_proj_ba.weight"], t[128:])
    t = torch.randn(48, 32, dtype=torch.bfloat16)
    for name in (
        "x.self_attn.qkv_proj.weight",
        "x.self_attn.o_proj.weight",
        "x.linear_attn.out_proj.weight",
    ):
        out = dict(_fp8_dense(name, t, cfg, 1, dense_channel=True))
        assert out[name[: -len("weight")] + "weight_scale"].shape == (48,), name
    # each flag drives only its own tensors
    out = dict(
        _fp8_dense(
            "lm_head.weight", t, cfg, 1, dense=False, lm_head=True, dense_channel=True
        )
    )
    assert out["lm_head.weight_scale"].shape == ()
    out = dict(
        _fp8_dense(
            "lm_head.weight", t, cfg, 1, dense=False, lm_head=True, lmhead_channel=True
        )
    )
    assert out["lm_head.weight_scale"].shape == (48,)
    out = dict(_fp8_dense("x.self_attn.o_proj.weight", t, cfg, 1, lmhead_channel=True))
    assert out["x.self_attn.o_proj.weight_scale"].shape == ()


# ------------------------------------------------------------------------------------------ TP shards


def _attn_cfg():
    # 4 q heads of 2 * 8 rows (q | gate), 2 kv heads of 8: qkv rows = 64 + 16 + 16 = 96, K = 32
    return _cfg(num_qo_heads=4, num_kv_heads=2, head_dim=8)


@pytest.mark.parametrize(
    "name, rows",
    [
        ("model.layers.1.self_attn.qkv_proj.weight", 96),
        ("model.layers.0.linear_attn.in_proj.weight", 4 * 8 * 2 + 12 * 8 * 2 + 2 * 12),
        ("lm_head.weight", 64),
    ],
)
def test_column_parallel_shards_are_rows_of_the_full_channel_result(name, rows):
    """A per-row scale never mixes rows, so (shard, then quantize) == (quantize, then shard), bit for bit,
    and the rank outputs, put back in the full row order, are the full output."""
    from freetoken.models.qwen4_exp.weight import _quantize_per_channel

    cfg, world = _attn_cfg(), 2
    lm = name == "lm_head.weight"
    kw = (
        {"dense": False, "lm_head": True, "lmhead_channel": True}
        if lm
        else {"dense_channel": True}
    )
    torch.manual_seed(2)
    full = (torch.randn(rows, 32) * torch.logspace(-2, 0, rows)[:, None]).to(
        torch.bfloat16
    )
    full8, full_s = _quantize_per_channel(full)
    x = torch.randn(3, 32)
    y_full = x @ _deq(full8, full_s).T
    idx = torch.arange(rows, dtype=torch.float32)[:, None]
    for rank in range(world):
        out = dict(
            _fp8_dense(name, _shard(name, full, cfg, rank, world), cfg, world, **kw)
        )
        my_rows = _shard(name, idx, cfg, rank, world)[:, 0].long()
        if name.endswith("in_proj.weight"):
            base = name[: -len("in_proj.weight")]
            w8, s = (
                out[base + "in_proj_qkvz.weight"],
                out[base + "in_proj_qkvz.weight_scale"],
            )
            my_rows = my_rows[: w8.shape[0]]  # the b|a rows stay bf16
        else:
            w8, s = out[name], out[name[: -len("weight")] + "weight_scale"]
        assert s.shape == (my_rows.numel(),)
        assert torch.equal(w8.view(torch.uint8), full8[my_rows].view(torch.uint8))
        assert torch.equal(s, full_s[my_rows])
        torch.testing.assert_close(
            x @ _deq(w8, s).T, y_full[:, my_rows], rtol=1e-5, atol=1e-6
        )


@pytest.mark.parametrize(
    "name",
    [
        "model.layers.1.self_attn.o_proj.weight",
        "model.layers.0.linear_attn.out_proj.weight",
    ],
)
def test_row_parallel_ranks_scale_their_own_k_shard_and_sum_to_the_full_output(name):
    """o_proj / out_proj: every rank holds all N rows over K / world inputs; each rank's per-row scale comes
    from its own K shard (so the ranks differ), and the all-reduce sum of the rank outputs is the full output."""
    cfg, world = _attn_cfg(), 2
    torch.manual_seed(3)
    full = torch.randn(48, 64)
    full[:, 32:] *= (
        10  # rank 1's K half is louder: its scales must differ from rank 0's
    )
    full = full.to(torch.bfloat16)
    x = torch.randn(5, 64)
    total = torch.zeros(5, 48)
    scales = []
    for rank in range(world):
        out = dict(
            _fp8_dense(
                name,
                _shard(name, full, cfg, rank, world),
                cfg,
                world,
                dense_channel=True,
            )
        )
        w8, s = out[name], out[name[: -len("weight")] + "weight_scale"]
        assert w8.shape == (48, 32) and s.shape == (48,)
        scales.append(s)
        total += x[:, rank * 32 : (rank + 1) * 32] @ _deq(w8, s).T
    assert (scales[1] > 5 * scales[0]).all()
    ref = x @ full.float().T
    assert _rel(total, ref) < 0.05  # e4m3 weight rounding, about 2.6 % RMS


# ----------------------------------------------------------------------------------------------- ops


@pytest.fixture
def tp1(monkeypatch):
    monkeypatch.setattr(dist_info, "_TP_INFO", DistributedInfo(0, 1))


def test_ops_declare_a_per_row_scale_in_channel_mode_and_a_scalar_by_default(tp1):
    from freetoken.layers.fp8_dynamic import (
        Fp8DynamicColMerged,
        Fp8DynamicRowParallel,
        Fp8ParallelLMHead,
    )

    for channel in (False, True):
        col = Fp8DynamicColMerged(32, [64, 16, 16], channel=channel)
        row = Fp8DynamicRowParallel(64, 32, channel=channel)
        head = Fp8ParallelLMHead(80, 32, channel=channel)
        for op, rows in ((col, 96), (row, 32), (head, 80)):
            sd = op.state_dict()
            assert (
                sd["weight"].dtype is FP8 and sd["weight_scale"].dtype == torch.float32
            )
            assert sd["weight_scale"].shape == ((rows,) if channel else ())
    assert (
        Fp8DynamicColMerged(32, [64]).weight_scale.shape == ()
    )  # the default is per tensor


@pytest.mark.parametrize("channel", [False, True])
def test_ops_and_lm_head_load_the_reader_output(tp1, channel):
    """The reader's tensors go through each op's own load_state_dict (ParallelLMHead has its own), with
    the dtype cast the engine applies (to the buffer dtype), and land unchanged."""
    from freetoken.layers.fp8_dynamic import (
        Fp8DynamicColMerged,
        Fp8DynamicRowParallel,
        Fp8ParallelLMHead,
    )

    torch.manual_seed(12)
    ops = {
        "x.self_attn.qkv_proj.weight": Fp8DynamicColMerged(
            32, [64, 16, 16], channel=channel
        ),
        "x.self_attn.o_proj.weight": Fp8DynamicRowParallel(64, 32, channel=channel),
        "lm_head.weight": Fp8ParallelLMHead(80, 32, channel=channel),
    }
    for name, op in ops.items():
        rows, cols = op.weight.shape
        t = (torch.randn(rows, cols) * 0.02).to(torch.bfloat16)
        lm = name == "lm_head.weight"
        kw = {
            "dense": not lm,
            "lm_head": lm,
            "dense_channel": channel,
            "lmhead_channel": channel,
        }
        out = dict(_fp8_dense(name, t, _cfg(), 1, **kw))
        want = op.state_dict()
        sd = {k: out[name[: -len("weight")] + k] for k in ("weight", "weight_scale")}
        sd = {k: v.to(want[k].dtype) for k, v in sd.items()}
        op.load_state_dict(dict(sd))
        assert op.weight.dtype is FP8 and op.weight_scale.shape == (
            (rows,) if channel else ()
        )
        assert torch.equal(op.weight.view(torch.uint8), sd["weight"].view(torch.uint8))
        assert torch.equal(op.weight_scale, sd["weight_scale"])


def _torch_quant_per_row(x: torch.Tensor):
    xf = x.float()
    s = (xf.abs().amax(dim=1, keepdim=True) / E4M3_MAX).clamp_min(1e-12)
    return (xf / s).clamp(-E4M3_MAX, E4M3_MAX).to(FP8), s


def _torch_scale_rows_cols_(
    y: torch.Tensor, row_scale: torch.Tensor, col_scale: torch.Tensor
) -> torch.Tensor:
    y.copy_(y.float() * row_scale * col_scale[None, :])
    return y


@pytest.mark.parametrize("rows", [1, 4, 37])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32])
def test_channel_linear_branch_on_cpu_matches_the_dequantized_reference(
    monkeypatch, rows, out_dtype
):
    """The channel branch of fp8_dynamic_linear with its two Triton steps (activation quant, scale apply)
    replaced by torch: _scaled_mm with unit scales, then the row and column scales, for 1 and many rows."""
    import freetoken.layers.fp8_dynamic as m
    from freetoken.models.qwen4_exp.weight import _quantize_per_channel

    monkeypatch.setattr(m, "quant_per_row", _torch_quant_per_row)
    monkeypatch.setattr(
        m,
        "quant_per_tensor",
        lambda x: pytest.fail("channel mode must use the per-row quant"),
    )
    monkeypatch.setattr(m, "scale_rows_cols_", _torch_scale_rows_cols_)
    torch.manual_seed(4)
    w = (torch.randn(64, 128) * torch.logspace(-2, 0, 64)[:, None]).to(torch.bfloat16)
    w8, s = _quantize_per_channel(w)
    x = torch.randn(2, rows, 128, dtype=torch.bfloat16)
    got = m.fp8_dynamic_linear(x, w8, s, out_dtype=out_dtype)
    assert got.shape == (2, rows, 64) and got.dtype == out_dtype
    x8, xs = _torch_quant_per_row(x.reshape(-1, 128))
    exact = ((x8.float() * xs) @ _deq(w8, s).T).reshape(2, rows, 64)
    torch.testing.assert_close(
        got.float(), exact, rtol=2**-7, atol=1e-6 * exact.abs().max().item()
    )
    ref = x.float() @ w.float().T  # end to end: e4m3 on both operands, about 2.6 % each
    assert _rel(got.float(), ref) < 0.08


def test_per_tensor_scale_never_reaches_the_channel_branch(monkeypatch):
    import freetoken.layers.fp8_dynamic as m

    monkeypatch.setattr(
        m,
        "scale_rows_cols_",
        lambda *a: pytest.fail("per-tensor mode must keep its own path"),
    )
    monkeypatch.setattr(m, "quant_per_row", _torch_quant_per_row)
    w8, s = _quantize_per_tensor(torch.randn(32, 64) * 0.02)
    x = torch.randn(3, 64, dtype=torch.bfloat16)
    m.fp8_dynamic_linear(x, w8, s)


_INTERPRET_SCRIPT = """
import torch
from freetoken.layers.fp8_dynamic import scale_rows_cols_
torch.manual_seed(5)
for m, n in ((1, 4096), (7, 3000), (5, 1000)):
    y = torch.randn(m, n) * 1e5
    rs, cs = torch.rand(m, 1) * 1e-2, torch.rand(n) * 1e-4
    ref = y * rs * cs[None, :]
    out = scale_rows_cols_(y, rs, cs)
    assert out.data_ptr() == y.data_ptr() and torch.equal(out, ref), (m, n)
print("ok")
"""


def test_scale_rows_cols_kernel_under_the_triton_interpreter():
    """The scale-apply kernel's indexing and math on the CPU (TRITON_INTERPRET=1, read at import, so a
    subprocess). fp32 output only: the interpreter truncates fp32 -> bf16 (the GPU rounds to nearest even),
    so the bf16 case is the GPU test below."""
    import os
    import subprocess
    import sys

    import freetoken

    env = dict(os.environ, TRITON_INTERPRET="1")
    env["PYTHONPATH"] = os.pathsep.join(
        filter(
            None,
            [
                os.path.dirname(os.path.dirname(freetoken.__file__)),
                env.get("PYTHONPATH"),
            ],
        )
    )
    res = subprocess.run(
        [sys.executable, "-c", _INTERPRET_SCRIPT],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert res.returncode == 0 and res.stdout.strip().endswith("ok"), res.stderr[-2000:]


def _quantize_rows_cpu(w: torch.Tensor):
    from freetoken.models.qwen4_exp.weight import _quantize_per_channel

    return _quantize_per_channel(w)


def test_fp8_lm_head_keep_rows_channel_tp1_and_tp2(monkeypatch):
    """keep_rows (MTP token map) keeps the e4m3 codes and the per-row scales as they are: no requant."""
    from freetoken.layers.fp8_dynamic import Fp8ParallelLMHead

    torch.manual_seed(11)
    vocab, hidden = 260, 32
    full = (torch.randn(vocab, hidden) * torch.logspace(-2, 0, vocab)[:, None]).to(
        torch.bfloat16
    )
    full8, full_s = _quantize_rows_cpu(full)
    ids = torch.tensor([3, 129, 130, 200, 259])

    monkeypatch.setattr(dist_info, "_TP_INFO", DistributedInfo(0, 1))
    h1 = Fp8ParallelLMHead(vocab, hidden, channel=True)
    h1.weight, h1.weight_scale = full8, full_s
    h1.keep_rows(ids, None)
    kept1, padded1 = h1.kept_rows
    assert kept1.weight_scale.shape == (128,)
    assert torch.equal(kept1.weight.view(torch.uint8), full8[padded1].view(torch.uint8))
    assert torch.equal(kept1.weight_scale, full_s[padded1])

    heads, partials = [], []
    for rank in range(2):
        monkeypatch.setattr(dist_info, "_TP_INFO", DistributedInfo(rank, 2))
        h = Fp8ParallelLMHead(vocab, hidden, channel=True)
        lo = rank * 130
        h.weight, h.weight_scale = (
            full8[lo : lo + 130].clone(),
            full_s[lo : lo + 130].clone(),
        )
        h._comm = SimpleNamespace(
            all_reduce=lambda x, p=partials: (p.append(x.clone()), x)[1]
        )
        h.keep_rows(ids, None)
        heads.append(h)
    # two all-reduces per rank (codes, scales), each rank contributing its own rows and zeros elsewhere
    n = len(partials) // 2
    summed = [partials[i] + partials[n + i] for i in range(n)]
    for h in heads:
        it = iter(summed)
        h._comm = SimpleNamespace(all_reduce=lambda _x, it=it: next(it).clone())
        h.keep_rows(ids, None)
        kept, padded = h.kept_rows
        assert (
            kept.tp_size == 1
            and kept.weight.dtype is FP8
            and kept.weight_scale.shape == (128,)
        )
        assert torch.equal(
            kept.weight.view(torch.uint8), full8[padded].view(torch.uint8)
        )
        assert torch.equal(kept.weight_scale, full_s[padded])


# ------------------------------------------------------------------------ builder + reader, strict load


@pytest.mark.parametrize(
    "dense_scale, lmhead_scale",
    [
        ("tensor", "tensor"),
        ("channel", "tensor"),
        ("channel", "channel"),
        ("tensor", "channel"),
    ],
)
def test_reader_and_builder_agree_on_the_scale_shape(
    tmp_path, monkeypatch, dense_scale, lmhead_scale
):
    """The model built under the env and the reader under the same env give the same keys, shapes and dtypes
    (a mismatch fails the strict load)."""
    from freetoken.distributed import set_tp_info, try_get_tp_info

    from .common import meta_state_dict
    from .test_weight import _load, _raw_checkpoint, _write_checkpoint

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    monkeypatch.setenv("FREETOKEN_FP8_DENSE", "1")
    monkeypatch.setenv("FREETOKEN_FP8_LMHEAD", "1")
    monkeypatch.setenv("FREETOKEN_FP8_DENSE_SCALE", dense_scale)
    monkeypatch.setenv("FREETOKEN_FP8_LMHEAD_SCALE", lmhead_scale)
    torch.manual_seed(0)
    folder, _raw = _write_checkpoint(tmp_path, _raw_checkpoint(), None)
    loaded, state = _load(folder, vision=False), meta_state_dict(folder)
    assert set(loaded) == set(state)
    modules = (
        "model.layers.1.self_attn.qkv_proj",
        "model.layers.1.self_attn.o_proj",
        "model.layers.0.linear_attn.in_proj_qkvz",
        "model.layers.0.linear_attn.out_proj",
    )
    for module in modules:
        w, s = loaded[module + ".weight"], loaded[module + ".weight_scale"]
        assert (
            w.dtype is state[module + ".weight"].dtype is FP8
            and s.dtype == torch.float32
        )
        assert (
            w.shape == state[module + ".weight"].shape
            and s.shape == state[module + ".weight_scale"].shape
        ), module
        assert s.shape == ((w.shape[0],) if dense_scale == "channel" else ()), module
    # the toy checkpoint ships 11 vocab rows against a 512-row config, so compare each side with its own rows
    for sd in (loaded, state):
        w, s = sd["lm_head.weight"], sd["lm_head.weight_scale"]
        assert w.dtype is FP8 and s.shape == (
            (w.shape[0],) if lmhead_scale == "channel" else ()
        )


@pytest.mark.parametrize("tp_size", [1, 2])
def test_mtp_draft_loads_channel_scales_across_tp(tmp_path, monkeypatch, tp_size):
    """The MTP draft shares the decoder-layer builders and the reader's _fp8_dense: channel scales load
    strictly at TP=1 (keys and shapes) and TP=2 (one scale per local row on each rank)."""
    from dataclasses import replace

    from freetoken.layers import rotary, set_rope_device
    from freetoken.layers.quantization import NameMap, QuantConfig, set_quant_config
    from freetoken.models.qwen4_exp.config import parse_config
    from freetoken.models.qwen4_exp.mtp import Qwen4ExpMTPDraft
    from freetoken.models.register import get_model_spec
    from freetoken.spec.mtp.proposer import with_resident_moe
    from freetoken.utils import cached_load_hf_config, torch_dtype

    from .common import NVIDIA_NVFP4
    from .test_mtp_tp import _dummy_target, _hf_dict, _write_mtp_checkpoint

    monkeypatch.setenv("FREETOKEN_FP8_DENSE", "1")
    monkeypatch.setenv("FREETOKEN_FP8_DENSE_SCALE", "channel")
    prev = rotary._ROPE_DEVICE
    set_rope_device(torch.device("cpu"))
    try:
        ckpt = tmp_path / f"mtp_channel_tp{tp_size}"
        _write_mtp_checkpoint(
            ckpt,
            _hf_dict(quant_cfg=NVIDIA_NVFP4),
            expert_mode="per_expert_fp8",
            dense_fp8_block=False,
        )
        hf_obj = cached_load_hf_config(str(ckpt))
        packed = get_model_spec(
            "Qwen4ExpForConditionalGeneration"
        ).packed_modules_mapping
        quant = QuantConfig.from_hf(hf_obj, name_map=NameMap(packed=packed))
        set_quant_config(quant)
        mc = with_resident_moe(replace(parse_config(hf_obj), quant=quant))
        if tp_size == 1:
            # TP=1 picks a CUDA-only expert kernel at load: compare the reader's output with the meta draft
            from freetoken.models.qwen4_exp.weight import iter_mtp_weights

            monkeypatch.setattr(dist_info, "_TP_INFO", DistributedInfo(0, 1))
            with torch.device("meta"), torch_dtype(torch.bfloat16):
                state = Qwen4ExpMTPDraft(mc, _dummy_target()).state_dict()
            emitted = dict(iter_mtp_weights(str(ckpt), torch.device("cpu")))
            assert set(emitted) == set(state)
            scales = [k for k in state if k.endswith("weight_scale")]
            assert scales
            for key in scales:
                assert (
                    emitted[key].shape
                    == state[key].shape
                    == (state[key[: -len("_scale")]].shape[0],)
                ), key
            return
        for rank in range(tp_size):
            monkeypatch.setattr(dist_info, "_TP_INFO", DistributedInfo(rank, tp_size))
            draft = Qwen4ExpMTPDraft.load(
                mc, _dummy_target(), str(ckpt), torch.device("cpu"), torch.bfloat16
            )
            attn = draft.layers.op_list[0].self_attn
            for op in (attn.qkv_proj, attn.o_proj):
                assert op.weight.dtype is FP8 and op.weight_scale.shape == (
                    op.weight.shape[0],
                )
    finally:
        rotary.get_rope.cache_clear()
        set_rope_device(prev)


# --------------------------------------------------------------------------------------------- GPU


def _cuda_weight(rows: int, k: int, seed: int = 0):
    from freetoken.models.qwen4_exp.weight import _quantize_per_channel

    torch.manual_seed(seed)
    w = (torch.randn(rows, k) * 0.02 * torch.logspace(0, -2, rows)[:, None]).to(
        torch.bfloat16
    )
    w8, s = _quantize_per_channel(w)
    return w.cuda(), w8.cuda(), s.cuda()


@requires_cuda
@pytest.mark.parametrize(
    "rows", [1, 4, 16, 300], ids=["decode-1", "verify-4", "decode-16", "prefill"]
)
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32])
def test_cuda_channel_linear_matches_the_dequantized_weight(rows, out_dtype):
    from freetoken.layers.fp8_dynamic import fp8_dynamic_linear, quant_per_row

    _w, w8, s = _cuda_weight(256, 128)
    x = torch.randn(rows, 128, device="cuda", dtype=torch.bfloat16)
    got = fp8_dynamic_linear(x, w8, s, out_dtype=out_dtype)
    assert got.shape == (rows, 256) and got.dtype == out_dtype
    x8, xs = quant_per_row(x)
    exact = (x8.float() * xs) @ (w8.float() * s[:, None]).T
    # bf16: the unit-scale GEMM output and the scaled result are each rounded once (2^-8 relative);
    # sm_89/sm_90 FP8 tensor cores accumulate 4 K-steps in ~13-bit mantissa (~2^-12) before fp32 promotion.
    rtol = 2**-7 if out_dtype is torch.bfloat16 else 5e-4
    torch.testing.assert_close(
        got.float(), exact, rtol=rtol, atol=5e-4 * exact.abs().max().item()
    )


@requires_cuda
def test_cuda_scale_rows_cols_matches_torch():
    from freetoken.layers.fp8_dynamic import scale_rows_cols_

    torch.manual_seed(5)
    for m, n, dtype in (
        (1, 4096, torch.bfloat16),
        (7, 6144, torch.bfloat16),
        (300, 1000, torch.float32),
    ):
        y = (torch.randn(m, n, device="cuda") * 1e5).to(dtype)
        rs = torch.rand(m, 1, device="cuda") * 1e-2
        cs = torch.rand(n, device="cuda") * 1e-4
        ref = (y.float() * rs * cs[None, :]).to(dtype)
        out = scale_rows_cols_(y, rs, cs)
        assert out.data_ptr() == y.data_ptr()  # in place
        assert torch.equal(y, ref)


@requires_cuda
def test_cuda_channel_row_alone_equals_the_row_in_a_verify_batch():
    """Channel mode runs one code path for 1 and for many rows, so a decode step at batch 1 and a verify row
    differ only by the GEMM's accumulation order."""
    from freetoken.layers.fp8_dynamic import fp8_dynamic_linear

    _w, w8, s = _cuda_weight(4096, 512, seed=6)
    x = torch.randn(4, 512, device="cuda", dtype=torch.bfloat16)
    alone = fp8_dynamic_linear(x[:1], w8, s, out_dtype=torch.float32)
    batch = fp8_dynamic_linear(x, w8, s, out_dtype=torch.float32)[:1]
    torch.testing.assert_close(
        alone, batch, rtol=1e-5, atol=1e-5 * alone.abs().max().item()
    )


@requires_cuda
def test_cuda_channel_row_does_not_depend_on_the_other_rows():
    from freetoken.layers.fp8_dynamic import fp8_dynamic_linear

    _w, w8, s = _cuda_weight(256, 128, seed=7)
    mine = torch.randn(61, 128, device="cuda", dtype=torch.bfloat16)
    other = torch.randn(53, 128, device="cuda", dtype=torch.bfloat16)
    quiet = fp8_dynamic_linear(torch.cat([mine, other]), w8, s)[:61]
    loud = fp8_dynamic_linear(torch.cat([mine, other * 8]), w8, s)[:61]
    assert torch.equal(quiet, loud)


@requires_cuda
@pytest.mark.parametrize("rows", [1, 4])
def test_cuda_channel_linear_captures_in_a_cuda_graph(rows):
    from freetoken.layers.fp8_dynamic import fp8_dynamic_linear

    _w, w8, s = _cuda_weight(512, 256, seed=8)
    x = torch.randn(rows, 256, device="cuda", dtype=torch.bfloat16)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fp8_dynamic_linear(x, w8, s)  # warm up (Triton compile) outside the capture
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fp8_dynamic_linear(x, w8, s)
    x.copy_(torch.randn_like(x))
    eager = fp8_dynamic_linear(x, w8, s)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, eager)


@requires_cuda
def test_cuda_channel_lm_head_gives_fp32_logits(monkeypatch):
    from freetoken.layers.fp8_dynamic import Fp8ParallelLMHead

    monkeypatch.setattr(dist_info, "_TP_INFO", DistributedInfo(0, 1))
    w, w8, s = _cuda_weight(1024, 256, seed=9)
    head = Fp8ParallelLMHead(1024, 256, channel=True)
    head.weight, head.weight_scale = w8, s
    x = torch.randn(4, 256, device="cuda", dtype=torch.bfloat16)
    y = head._local_logits(x)
    assert y.dtype == torch.float32 and y.shape == (4, 1024)
    assert _rel(y, x.float() @ w.float().T) < 0.08
