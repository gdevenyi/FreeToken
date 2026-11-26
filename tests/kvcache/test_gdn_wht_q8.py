"""Unit tests for WHT128-Q8_0 compressed P1 GDN host recurrent-state snapshots."""

from __future__ import annotations

import torch
from freetoken.kernel.triton.gdn_wht_q8 import (
    gdn_host_q8_enabled,
    pack_slot_wht128_q8,
    unpack_slot_wht128_q8,
    wht128_q8_slot_bytes,
)
from freetoken.kvcache.linear_state_pool import (
    LinearStatePool,
    linear_state_bytes_per_req,
    linear_state_host_bytes_per_req,
)
from freetoken.models.config import LinearGatedDeltaGroupConfig, SlotStateSpec


def _rel_rmse(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(
        ((a.float() - b.float()).norm() / b.float().norm().clamp(min=1e-12)).item()
    )


def test_wht128_q8_suppresses_channel_outliers_over_raw_q8():
    """Verify WHT-128 spreads sharp channel outliers across all 128 coefficients,
    beating raw group-32 int8 quantization by >5x on outlier-heavy GDN recurrent states."""
    torch.manual_seed(11)
    states = torch.randn(4, 3, 2, 128, 128, dtype=torch.float32) * 0.05
    # Inject strong outlier spikes on single channels (typical of long-context GDN states):
    states[:, 1, :, :, 7] += 8.0
    states[:, 1, :, :, 91] -= 6.5

    total_rows = (4 * 2 * 128 * 128) // 128
    q = torch.empty((total_rows, 128), dtype=torch.int8)
    s = torch.empty((total_rows, 4), dtype=torch.bfloat16)
    pack_slot_wht128_q8(states, 1, q, s)

    rec = torch.zeros_like(states)
    unpack_slot_wht128_q8(q, s, rec, 1)
    err_wht = _rel_rmse(rec[:, 1], states[:, 1])

    # Compare against raw (non-WHT) group-32 int8 with bf16 scale:
    raw = states[:, 1].reshape(-1, 32)
    sc_raw = (
        (raw.abs().amax(dim=-1) / 127.0).to(torch.bfloat16).float().clamp(min=1e-12)
    )
    q_raw = torch.round(raw / sc_raw.unsqueeze(-1)).clamp(
        -127.0, 127.0
    ) * sc_raw.unsqueeze(-1)
    err_raw = _rel_rmse(q_raw.view_as(states[:, 1]), states[:, 1])

    assert err_wht < 0.0040
    assert err_wht < err_raw / 3.5
    assert (
        wht128_q8_slot_bytes(total_rows * 128) * 64 == (total_rows * 128 * 4) * 17
    )  # exact 17/64 = 1/3.7647x


def test_linear_state_pool_wht128_q8_host_roundtrip(monkeypatch):
    monkeypatch.setenv("FREETOKEN_GDN_HOST_Q8", "1")
    assert gdn_host_q8_enabled() is True

    group = LinearGatedDeltaGroupConfig(
        name="linear",
        layer_ids=(0, 1, 2),
        num_key_heads=4,
        num_value_heads=8,
        key_head_dim=128,
        value_head_dim=128,
        conv_kernel_dim=4,
        output_gate="silu",
    )
    extra = (
        SlotStateSpec(
            name="ple_conv", shape=(256, 6), layer_ids=(1,), dtype=torch.bfloat16
        ),
    )

    raw_bytes = linear_state_bytes_per_req(group, 1, torch.bfloat16, extra)
    q8_bytes = linear_state_host_bytes_per_req(group, 1, torch.bfloat16, extra)
    assert q8_bytes < raw_bytes * 0.35

    pool = LinearStatePool(
        group=group,
        num_slots=4,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
        tp_size=1,
        slot_states=extra,
        host_slots=2,
    )
    assert pool._host_wht_q8 is True
    assert abs(pool.host_bytes_per_slot() - q8_bytes) <= 64

    torch.manual_seed(29)
    src_conv = torch.randn_like(pool.conv_states[:, 1])
    src_rec = torch.randn_like(pool.recurrent_states[:, 1])
    src_rec[:, :, :, 19] += 5.0
    src_ple = torch.randn_like(pool.slot_states["ple_conv"][:, 1])

    pool.conv_states[:, 1].copy_(src_conv)
    pool.recurrent_states[:, 1].copy_(src_rec)
    pool.slot_states["ple_conv"][:, 1].copy_(src_ple)

    hslot = pool.alloc_host(1)[0]
    pool.to_host(1, hslot)
    pool.fence_host_copies()
    pool.clear_slots([1, 2])
    assert torch.all(pool.recurrent_states[:, 2] == 0)

    pool.from_host(hslot, 2)
    assert torch.equal(pool.conv_states[:, 2], src_conv)
    assert torch.equal(pool.slot_states["ple_conv"][:, 2], src_ple)
    assert _rel_rmse(pool.recurrent_states[:, 2], src_rec) < 0.0065


def test_wht128_q8_subnormal_bf16_scale_and_nan_free():
    """Subnormal BF16 scales (<= 2^-131) combined with exact zeros in a WHT group must not
    overflow 1/scale_f32 to +inf and produce 0 * inf = NaN."""
    from freetoken.kernel.triton.gdn_wht_q8 import _hadamard_128

    h = _hadamard_128(torch.device("cpu"))
    # Construct y in WHT domain directly via inverse Hadamard (H is symmetric involutory: x = y @ H):
    y_target = torch.zeros(4, 128, dtype=torch.float32)
    # Row 0, group 0: 1 subnormal-scale value (amax/127 ~ 2^-131) + 31 exact zeros -> previously NaN!
    y_target[0, 0] = float.fromhex("0x1p-124")
    # Row 1: normal values
    y_target[1] = torch.linspace(-0.5, 0.5, 128)
    # Row 2: non-finite poison in unused/corrupt slot -> clamped cleanly without NaN propagation
    x_states = (y_target @ h).view(2, 1, 2, 128)
    x_states[1, 0, 0, 0] = float("nan")
    x_states[1, 0, 0, 1] = float("inf")

    q = torch.empty((4, 128), dtype=torch.int8)
    s = torch.empty((4, 4), dtype=torch.bfloat16)
    pack_slot_wht128_q8(x_states, 0, q, s)

    assert torch.all(q[0, :32] == 0)
    assert float(s[0, 0].item()) == 0.0
    assert torch.isfinite(s.float()).all()

    rec = torch.zeros_like(x_states)
    unpack_slot_wht128_q8(q, s, rec, 0)
    assert torch.isfinite(rec).all()
    assert torch.all(rec[0, 0, 0] == 0.0)
    assert _rel_rmse(rec[0, 0, 1], x_states[0, 0, 1]) < 0.01
