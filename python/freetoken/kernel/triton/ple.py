# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# Adapted from SGLang (python/sglang/srt/models/qwen4_exp.py)
"""UVA row gather for the Qwen3.8-Flash-Next PLE n-gram table.

The table (320,001,536 rows x 160, FP8-e4m3 + one scalar scale = 47.7 GiB) stays in pinned
host memory and the GPU dereferences it in place over PCIe -- at its host VA on Linux/UVA, at
the mapped device address on WDDM (``kernel/pinned.device_ptr``). One program per requested
row: read the row, widen to fp32, apply the per-tensor scale, store bf16.

Ids outside the table store zeros.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from freetoken.kernel.triton.e4m3_compat import e4m3_native_cx, e4m3_u8_to_f32

# Latency-bound over PCIe, so keep the block small and let many of them be in flight.
_NUM_WARPS = 1


@triton.jit
def _ple_gather_kernel(
    table_ptr,
    ids_ptr,
    out_ptr,
    scale,
    num_rows,
    EMB_DIM: tl.constexpr,
    IS_FP8: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    idx = tl.load(ids_ptr + row).to(tl.int64)
    in_range = (idx >= 0) & (idx < num_rows)
    idx = tl.where(in_range, idx, 0)
    offsets = tl.arange(0, BLOCK_D)
    mask = offsets < EMB_DIM
    # the table is a host allocation: rebuild the typed pointer from the raw address
    if IS_FP8:
        if e4m3_native_cx():
            base = table_ptr.to(tl.int64).to(tl.pointer_type(tl.float8e4nv))
            values = tl.load(base + idx * EMB_DIM + offsets, mask=mask, other=0.0).to(
                tl.float32
            )
        else:
            # pre-sm_89 has no fp8e4nv type: load raw bytes and decode in software
            base = table_ptr.to(tl.int64).to(tl.pointer_type(tl.uint8))
            values = e4m3_u8_to_f32(
                tl.load(base + idx * EMB_DIM + offsets, mask=mask, other=0)
            )
    else:
        base = table_ptr.to(tl.int64).to(tl.pointer_type(tl.bfloat16))
        values = tl.load(base + idx * EMB_DIM + offsets, mask=mask, other=0.0).to(
            tl.float32
        )
    values = tl.where(in_range, values * scale, 0.0)
    tl.store(
        out_ptr + row * EMB_DIM + offsets,
        values.to(out_ptr.dtype.element_ty),
        mask=mask,
    )


def ple_gather_rows(
    table_ptr: int,
    num_rows: int,
    embed_dim: int,
    row_ids: torch.Tensor,
    out: torch.Tensor,
    scale: float = 1.0,
    is_fp8: bool = True,
) -> torch.Tensor:
    """Gather ``row_ids`` from the host-resident table at ``table_ptr`` into ``out``.

    ``row_ids`` is a flat device int tensor; ``out`` is ``[row_ids.numel(), embed_dim]``
    bf16 on the same device. ``table_ptr`` is the address the GPU must dereference
    (``kernel/pinned.device_ptr``), not necessarily the host ``data_ptr``.
    """
    n = row_ids.numel()
    assert out.shape == (n, embed_dim) and out.is_contiguous(), out.shape
    if n:
        _ple_gather_kernel[(n,)](
            table_ptr,
            row_ids,
            out,
            float(scale),
            num_rows,
            EMB_DIM=embed_dim,
            IS_FP8=is_fp8,
            BLOCK_D=triton.next_power_of_2(embed_dim),
            num_warps=_NUM_WARPS,
        )
    return out


@triton.jit
def _ple_gather_lut4_kernel(
    table_ptr,
    lut_ptr,
    ids_ptr,
    out_ptr,
    num_rows,
    EMB_DIM: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    PACKED_BYTES: tl.constexpr,
    ROW_BYTES: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    idx = tl.load(ids_ptr + row).to(tl.int64)
    in_range = (idx >= 0) & (idx < num_rows)
    idx = tl.where(in_range, idx, 0)
    offsets = tl.arange(0, BLOCK_D)
    mask = offsets < EMB_DIM

    row_base = table_ptr.to(tl.int64).to(tl.pointer_type(tl.uint8)) + idx * ROW_BYTES
    byte_idx = offsets >> 1
    shift = ((offsets & 1) * 4).to(tl.uint8)
    packed = tl.load(row_base + byte_idx, mask=mask, other=0)
    nibble = ((packed >> shift) & 0xF).to(tl.int32)
    code = tl.load(lut_ptr + nibble, mask=mask, other=0.0).to(tl.float32)

    grp = offsets // GROUP_SIZE
    scale_ptr = (row_base + PACKED_BYTES).to(tl.pointer_type(tl.float16))
    grp_scale = tl.load(scale_ptr + grp, mask=mask, other=0.0).to(tl.float32)
    values = tl.where(in_range, code * grp_scale, 0.0)
    tl.store(
        out_ptr + row * EMB_DIM + offsets,
        values.to(out_ptr.dtype.element_ty),
        mask=mask,
    )


def ple_gather_lut4_cpu(
    packed_table: torch.Tensor,
    weight_lut: torch.Tensor,
    embed_dim: int,
    group_size: int,
    row_ids: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    """Reference CPU gather for ``lut4-gptq`` packed ``[num_rows, packed_bytes + scale_bytes]`` rows."""
    flat_ids = row_ids.reshape(-1).to(torch.int64)
    n = flat_ids.numel()
    assert out.shape == (n, embed_dim), (out.shape, n, embed_dim)
    if n == 0:
        return out
    num_rows = packed_table.shape[0]
    packed_bytes = embed_dim // 2
    num_groups = embed_dim // group_size
    in_range = (flat_ids >= 0) & (flat_ids < num_rows)
    safe_ids = torch.where(in_range, flat_ids, torch.zeros_like(flat_ids))
    rows = packed_table.index_select(0, safe_ids.cpu())
    nib_bytes = rows[:, :packed_bytes]
    lo = (nib_bytes & 0x0F).to(torch.long)
    hi = ((nib_bytes >> 4) & 0x0F).to(torch.long)
    nibbles = torch.stack((lo, hi), dim=-1).reshape(n, embed_dim)
    lut_cpu = weight_lut.detach().to(device="cpu", dtype=torch.float32)
    codes = lut_cpu[nibbles]
    scales = (
        rows[:, packed_bytes : packed_bytes + num_groups * 2]
        .contiguous()
        .view(torch.float16)
        .to(torch.float32)
    )
    vals = (codes.view(n, num_groups, group_size) * scales.unsqueeze(-1)).reshape(
        n, embed_dim
    )
    vals = torch.where(in_range.cpu().unsqueeze(-1), vals, torch.zeros_like(vals))
    out.copy_(vals.to(dtype=out.dtype, device=out.device))
    return out


def ple_gather_lut4_rows(
    table_ptr: int,
    weight_lut: torch.Tensor,
    num_rows: int,
    embed_dim: int,
    group_size: int,
    row_ids: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    """Gather ``lut4-gptq`` rows (`[80B nibbles | 10B fp16 scales]`) over UVA into ``out``."""
    n = row_ids.numel()
    assert embed_dim % 4 == 0 and embed_dim % group_size == 0, (embed_dim, group_size)
    assert out.shape == (n, embed_dim) and out.is_contiguous(), out.shape
    packed_bytes = embed_dim // 2
    row_bytes = packed_bytes + (embed_dim // group_size) * 2
    if n:
        _ple_gather_lut4_kernel[(n,)](
            table_ptr,
            weight_lut,
            row_ids,
            out,
            num_rows,
            EMB_DIM=embed_dim,
            GROUP_SIZE=group_size,
            PACKED_BYTES=packed_bytes,
            ROW_BYTES=row_bytes,
            BLOCK_D=triton.next_power_of_2(embed_dim),
            num_warps=_NUM_WARPS,
        )
    return out


__all__ = ["ple_gather_lut4_cpu", "ple_gather_lut4_rows", "ple_gather_rows"]
