"""Fused WHT128-Q8_0 pack/unpack kernels for P1 GDN host recurrent-state snapshots.

Applies a normalized self-inverse 128-point Walsh-Hadamard transform (`H_128 / sqrt(128)`)
along the inner 128-element state axis to disperse channel outliers, then quantizes each
32-element block into symmetric `int8` (`[-127, 127]`) with one `bfloat16` scale (`34` bytes
per `32` FP32 elements = `3.765x` compression of `recurrent_states`, `< 0.25%` relative RMSE).
"""

from __future__ import annotations

import functools
import math
import os

import torch
import triton
import triton.language as tl

WHT_DIM = 128
Q8_GROUP = 32
GROUPS_PER_WHT = WHT_DIM // Q8_GROUP  # 4
_ENV_TRUE = ("1", "true", "yes", "on")


def gdn_host_q8_enabled() -> bool:
    """Whether P1 GDN host snapshots compress `recurrent_states` with `WHT128-Q8_0` (`FREETOKEN_GDN_HOST_Q8=1`)."""
    return os.getenv("FREETOKEN_GDN_HOST_Q8", "0").strip().lower() in _ENV_TRUE


def can_wht128_q8(shape: tuple[int, ...], dtype: torch.dtype) -> bool:
    """True when `recurrent_states` is float32 with inner dimension divisible by 128."""
    return dtype == torch.float32 and len(shape) >= 2 and shape[-1] % WHT_DIM == 0


def wht128_q8_slot_bytes(num_elems_per_slot: int) -> int:
    """Compressed bytes for `num_elems_per_slot` float32 values (`1` int8 + `2/32` bf16 scale bytes/elem)."""
    assert num_elems_per_slot % WHT_DIM == 0, num_elems_per_slot
    rows = num_elems_per_slot // WHT_DIM
    return rows * WHT_DIM + rows * GROUPS_PER_WHT * 2


@functools.lru_cache(maxsize=8)
def _hadamard_128(device: torch.device) -> torch.Tensor:
    """Return normalized symmetric `[128, 128]` float32 Walsh-Hadamard matrix on `device`."""
    h = torch.tensor([[1.0, 1.0], [1.0, -1.0]], dtype=torch.float32, device=device)
    out = h
    for _ in range(6):  # 2^7 = 128
        out = torch.kron(out, h)
    return (out * (1.0 / math.sqrt(float(WHT_DIM)))).contiguous()


@triton.jit
def _wht128_q8_pack_slot_kernel(
    states_ptr,
    h_ptr,
    q_ptr,
    s_ptr,
    slot_idx,
    num_layers,
    rows_per_layer,
    stride_layer,
    stride_slot,
    BLOCK_M: tl.constexpr,
    IS_HIP: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    rows_per_layer_i64 = rows_per_layer.to(tl.int64)
    total_rows = num_layers.to(tl.int64) * rows_per_layer_i64
    mask_m = offs_m < total_rows
    layer = offs_m // rows_per_layer_i64
    row_in_layer = offs_m - layer * rows_per_layer_i64
    stride_layer_i64 = stride_layer.to(tl.int64)
    stride_slot_i64 = stride_slot.to(tl.int64)
    slot_idx_i64 = slot_idx.to(tl.int64)
    offs_k = tl.arange(0, 128)

    y = tl.zeros((BLOCK_M, 128), dtype=tl.float32)
    for k0 in tl.static_range(0, 128, 32):
        offs_ki = k0 + tl.arange(0, 32)
        row_ptrs = (
            states_ptr
            + layer[:, None] * stride_layer_i64
            + slot_idx_i64 * stride_slot_i64
            + row_in_layer[:, None] * 128
            + offs_ki[None, :]
        )
        xk = tl.load(row_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)
        hk = tl.load(h_ptr + offs_ki[:, None] * 128 + offs_k[None, :]).to(tl.float32)
        y += tl.dot(xk, hk, allow_tf32=False)

    yg = tl.reshape(y, (BLOCK_M, 4, 32))
    yg = tl.where(yg == yg, tl.minimum(tl.maximum(yg, -3.4e38), 3.4e38), 0.0)  # noqa: PLR0124
    amax = tl.max(tl.abs(yg), axis=2)
    raw_bf16 = (amax * (1.0 / 127.0)).to(tl.bfloat16)
    scale_f32 = raw_bf16.to(tl.float32)
    # Flush subnormal BF16 scales (< 2^-126): 1/scale_f32 overflows FP32 at <= 2^-128 (0 * inf = NaN).
    valid_scale = scale_f32 >= 1.1754944e-38
    scale_bf16 = tl.where(valid_scale, raw_bf16, 0.0).to(tl.bfloat16)
    safe_scale = tl.where(valid_scale, scale_f32, 1.0)
    scaled = tl.where(valid_scale[:, :, None], yg / safe_scale[:, :, None], 0.0)
    scaled = tl.minimum(tl.maximum(scaled, -127.0), 127.0)
    if IS_HIP:
        q_f32 = tl.where(
            scaled >= 0.0, tl.floor(scaled + 0.5), -tl.floor(-scaled + 0.5)
        )
    else:
        q_f32 = tl.extra.cuda.libdevice.rint(scaled)
    q_i8 = tl.reshape(q_f32.to(tl.int8), (BLOCK_M, 128))

    tl.store(
        q_ptr + offs_m[:, None] * 128 + offs_k[None, :],
        q_i8,
        mask=mask_m[:, None],
    )
    offs_g = tl.arange(0, 4)
    tl.store(
        s_ptr + offs_m[:, None] * 4 + offs_g[None, :],
        scale_bf16,
        mask=mask_m[:, None],
    )


@triton.jit
def _wht128_q8_unpack_slot_kernel(
    q_ptr,
    s_ptr,
    h_ptr,
    states_ptr,
    slot_idx,
    num_layers,
    rows_per_layer,
    stride_layer,
    stride_slot,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    rows_per_layer_i64 = rows_per_layer.to(tl.int64)
    total_rows = num_layers.to(tl.int64) * rows_per_layer_i64
    mask_m = offs_m < total_rows
    layer = offs_m // rows_per_layer_i64
    row_in_layer = offs_m - layer * rows_per_layer_i64
    stride_layer_i64 = stride_layer.to(tl.int64)
    stride_slot_i64 = stride_slot.to(tl.int64)
    slot_idx_i64 = slot_idx.to(tl.int64)
    offs_k = tl.arange(0, 128)
    offs_g = tl.arange(0, 4)

    q_i8 = tl.load(
        q_ptr + offs_m[:, None] * 128 + offs_k[None, :],
        mask=mask_m[:, None],
        other=0,
    )
    sc = tl.load(
        s_ptr + offs_m[:, None] * 4 + offs_g[None, :],
        mask=mask_m[:, None],
        other=0.0,
    ).to(tl.float32)

    yg = tl.reshape(q_i8.to(tl.float32), (BLOCK_M, 4, 32)) * sc[:, :, None]
    x = tl.zeros((BLOCK_M, 128), dtype=tl.float32)
    for k0 in tl.static_range(0, 128, 32):
        offs_ki = k0 + tl.arange(0, 32)
        g_idx = k0 // 32
        # Extract group g_idx ([BLOCK_M, 32]) via masked sum over the 4 groups
        g_mask = (tl.arange(0, 4) == g_idx)[None, :, None]
        yk = tl.sum(tl.where(g_mask, yg, 0.0), axis=1)
        hk = tl.load(h_ptr + offs_ki[:, None] * 128 + offs_k[None, :]).to(tl.float32)
        x += tl.dot(yk, hk, allow_tf32=False)

    row_ptrs = (
        states_ptr
        + layer[:, None] * stride_layer_i64
        + slot_idx_i64 * stride_slot_i64
        + row_in_layer[:, None] * 128
        + offs_k[None, :]
    )
    tl.store(
        row_ptrs,
        x.to(states_ptr.dtype.element_ty),
        mask=mask_m[:, None],
    )


def pack_slot_wht128_q8(
    recurrent_states: torch.Tensor,
    slot_idx: int,
    dst_q: torch.Tensor,
    dst_s: torch.Tensor,
) -> None:
    """Pack `recurrent_states[:, slot_idx]` (`[L, S, ...]`) into contiguous `dst_q [R, 128] int8` + `dst_s [R, 4] bf16`."""
    num_layers = recurrent_states.shape[0]
    elems_per_layer = recurrent_states[0, 0].numel()
    assert elems_per_layer % WHT_DIM == 0, elems_per_layer
    rows_per_layer = elems_per_layer // WHT_DIM
    total_rows = num_layers * rows_per_layer
    if total_rows == 0:
        return

    h = _hadamard_128(recurrent_states.device)
    q_2d = dst_q.view(total_rows, WHT_DIM)
    s_2d = dst_s.view(total_rows, GROUPS_PER_WHT)

    if recurrent_states.is_cuda and recurrent_states[0, 0].is_contiguous():
        block_m = 32
        grid = (triton.cdiv(total_rows, block_m),)
        _wht128_q8_pack_slot_kernel[grid](
            recurrent_states,
            h,
            q_2d,
            s_2d,
            int(slot_idx),
            num_layers,
            rows_per_layer,
            recurrent_states.stride(0),
            recurrent_states.stride(1),
            BLOCK_M=block_m,
            IS_HIP=torch.version.hip is not None,
            num_warps=4,
            num_stages=1,
        )
        return

    slot_data = (
        recurrent_states[:, slot_idx].contiguous().view(total_rows, WHT_DIM).float()
    )
    y = torch.nan_to_num(
        (slot_data @ h).view(total_rows, GROUPS_PER_WHT, Q8_GROUP),
        nan=0.0,
        posinf=3.4e38,
        neginf=-3.4e38,
    )
    amax = y.abs().amax(dim=-1)
    raw_bf16 = (amax * (1.0 / 127.0)).to(torch.bfloat16)
    sc_f32 = raw_bf16.float()
    valid = sc_f32 >= 1.1754944e-38
    sc_bf16 = torch.where(valid, raw_bf16, torch.zeros_like(raw_bf16))
    safe_sc = torch.where(valid, sc_f32, torch.ones_like(sc_f32))
    scaled = torch.where(
        valid.unsqueeze(-1), y / safe_sc.unsqueeze(-1), torch.zeros_like(y)
    )
    q = torch.round(scaled).clamp_(-127.0, 127.0).to(torch.int8)
    q_2d.copy_(q.view(total_rows, WHT_DIM))
    s_2d.copy_(sc_bf16)


def unpack_slot_wht128_q8(
    src_q: torch.Tensor,
    src_s: torch.Tensor,
    recurrent_states: torch.Tensor,
    slot_idx: int,
) -> None:
    """Unpack contiguous `src_q [R, 128] int8` + `src_s [R, 4] bf16` directly into `recurrent_states[:, slot_idx]`."""
    num_layers = recurrent_states.shape[0]
    elems_per_layer = recurrent_states[0, 0].numel()
    assert elems_per_layer % WHT_DIM == 0, elems_per_layer
    rows_per_layer = elems_per_layer // WHT_DIM
    total_rows = num_layers * rows_per_layer
    if total_rows == 0:
        return

    h = _hadamard_128(recurrent_states.device)
    q_2d = src_q.view(total_rows, WHT_DIM)
    s_2d = src_s.view(total_rows, GROUPS_PER_WHT)

    if recurrent_states.is_cuda and recurrent_states[0, 0].is_contiguous():
        block_m = 32
        grid = (triton.cdiv(total_rows, block_m),)
        _wht128_q8_unpack_slot_kernel[grid](
            q_2d,
            s_2d,
            h,
            recurrent_states,
            int(slot_idx),
            num_layers,
            rows_per_layer,
            recurrent_states.stride(0),
            recurrent_states.stride(1),
            BLOCK_M=block_m,
            num_warps=4,
            num_stages=1,
        )
        return

    y = (
        q_2d.float().view(total_rows, GROUPS_PER_WHT, Q8_GROUP)
        * s_2d.float().unsqueeze(-1)
    ).view(total_rows, WHT_DIM)
    x = (y @ h).to(recurrent_states.dtype).view(num_layers, *recurrent_states.shape[2:])
    recurrent_states[:, slot_idx].copy_(x)


__all__ = [
    "GROUPS_PER_WHT",
    "Q8_GROUP",
    "WHT_DIM",
    "can_wht128_q8",
    "gdn_host_q8_enabled",
    "pack_slot_wht128_q8",
    "unpack_slot_wht128_q8",
    "wht128_q8_slot_bytes",
]
