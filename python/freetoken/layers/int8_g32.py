"""Load-time symmetric int8 group-32 W8A16 linear layers (``C4b`` dense & ``hc8`` Hyper-Connections).

Quantizes BF16 weights at load time (after TP sharding) into symmetric ``int8`` codes ``[-127, 127]``
with one ``fp16`` scale per group of 32 input channels (`8.5` bits/weight, `0.56-0.84%` relative weight
error vs `2.65%` for FP8 E4M3).

Execution paths (selected automatically per device/runtime, zero host syncs, CUDA-graph safe):
1. ``vllm`` Marlin ``uint8b128`` (`num_bits=8, group_size=32`) when available on CUDA and shapes align
   (automatically zero-padding ``N`` up to a multiple of 64 when ``pad_n`` is enabled, e.g. ``hc.down``
   `336 -> 384`).
2. Fused Triton W8A16 ``int8 g32`` GEMM on CUDA when Marlin is unavailable or dimensions are unaligned.
3. Exact PyTorch dequantize + ``F.linear`` reference on CPU.
"""

from __future__ import annotations

import functools

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from freetoken.distributed import DistributedCommunicator, get_tp_info
from freetoken.utils import div_even

from .base import BaseOP
from .embedding import ParallelLMHead

GROUP_SIZE = 32


def quantize_int8_g32(
    w: torch.Tensor, group_size: int = GROUP_SIZE
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``[N, K]`` float/bf16 weight into ``(qweight int8 [N, K], scale fp16 [N, K // group_size])``."""
    assert w.ndim == 2 and w.shape[1] % group_size == 0, (w.shape, group_size)
    n, k = w.shape
    wf = w.float().view(n, k // group_size, group_size)
    amax = wf.abs().amax(dim=-1)
    scale_f32 = (amax / 127.0).clamp(min=1e-8, max=65504.0)
    scale_f16 = scale_f32.to(torch.float16)
    inv = 1.0 / scale_f16.float().clamp(min=1e-8)
    q = torch.round(wf * inv.unsqueeze(-1)).clamp_(-127.0, 127.0).to(torch.int8)
    # Exact zero groups stay zero with finite minimal scale.
    zero_mask = amax == 0
    if bool(zero_mask.any()):
        q = torch.where(zero_mask.unsqueeze(-1), torch.zeros_like(q), q)
    return q.view(n, k).contiguous(), scale_f16.contiguous()


def dequantize_int8_g32(
    qweight: torch.Tensor, scale: torch.Tensor, out_dtype: torch.dtype = torch.bfloat16
) -> torch.Tensor:
    """Reference ``(qweight * scale)`` reconstruction to ``out_dtype``."""
    n, k = qweight.shape
    num_groups = scale.shape[1]
    group_size = k // num_groups
    wf = qweight.float().view(n, num_groups, group_size) * scale.float().unsqueeze(-1)
    return wf.view(n, k).to(out_dtype)


@triton.jit
def _int8_g32_w8a16_gemm_kernel(
    x_ptr,
    qw_ptr,
    sc_ptr,
    y_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wk,
    stride_sn,
    stride_sg,
    stride_ym,
    stride_yn,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    NUM_GROUPS_PER_STEP: tl.constexpr = BLOCK_K // GROUP_SIZE

    for k0 in range(0, K, BLOCK_K):
        for g_sub in tl.static_range(0, NUM_GROUPS_PER_STEP):
            gk = k0 + g_sub * GROUP_SIZE
            offs_k = gk + tl.arange(0, GROUP_SIZE)
            mask_k = offs_k < K
            a = tl.load(
                x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
                mask=mask_m[:, None] & mask_k[None, :],
                other=0.0,
            )
            qw = tl.load(
                qw_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk,
                mask=mask_k[:, None] & mask_n[None, :],
                other=0,
            )
            g_idx = gk // GROUP_SIZE
            sc = tl.load(
                sc_ptr + offs_n * stride_sn + g_idx * stride_sg,
                mask=mask_n,
                other=0.0,
            ).to(tl.float32)
            w_deq = (qw.to(tl.float32) * sc[None, :]).to(a.dtype)
            acc += tl.dot(a, w_deq, allow_tf32=False)

    out = acc.to(y_ptr.dtype.element_ty)
    tl.store(
        y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn,
        out,
        mask=mask_m[:, None] & mask_n[None, :],
    )


def int8_g32_triton_linear(
    x2: torch.Tensor,
    qweight: torch.Tensor,
    scale: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """CUDA-graph-safe Triton W8A16 GEMM for ``qweight [N, K] int8`` and ``scale [N, K//32] fp16``."""
    m, k = x2.shape
    n = qweight.shape[0]
    y = torch.empty((m, n), dtype=out_dtype, device=x2.device)
    if m == 0 or n == 0:
        return y
    if m == 1:
        # M=1 decode/verify skinny path: vectorized group dot avoids tensor-core M-padding overhead
        block_m, block_n, block_k, warps = 16, 64, 64, 4
    elif m <= 16:
        block_m, block_n, block_k, warps = 16, 64, 64, 4
    else:
        block_m, block_n, block_k, warps = 64, 64, 64, 4
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    _int8_g32_w8a16_gemm_kernel[grid](
        x2,
        qweight,
        scale,
        y,
        m,
        n,
        k,
        x2.stride(0),
        x2.stride(1),
        qweight.stride(0),
        qweight.stride(1),
        scale.stride(0),
        scale.stride(1),
        y.stride(0),
        y.stride(1),
        GROUP_SIZE=GROUP_SIZE,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=warps,
    )
    return y


@functools.cache
def _shared_marlin_workspace(device_index: int) -> torch.Tensor:
    from vllm.model_executor.layers.quantization.utils import marlin_utils

    max_blocks = int(getattr(marlin_utils, "MARLIN_MAX_BLOCKS_PER_SM", 4))
    dev = torch.device("cuda", device_index)
    return marlin_utils.marlin_make_workspace_new(dev, max_blocks)


def _try_pack_marlin_u8b128(
    qweight: torch.Tensor, scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int] | None:
    """Attempt to repack ``int8 [N, K]`` + ``fp16 [N, K//32]`` into vLLM Marlin ``uint8b128`` layout.

    Pads ``N`` up to a multiple of 64 if needed (`hc.down` `336 -> 384`). Returns
    ``(marlin_qweight, marlin_scales, workspace, padded_n)`` or ``None`` if Marlin is unavailable.
    """
    if not qweight.is_cuda:
        return None
    n, k = qweight.shape
    if k % 64 != 0:
        return None
    try:
        from vllm import _custom_ops as ops
        from vllm.model_executor.layers.quantization.utils.marlin_utils import (
            marlin_permute_scales,
        )
        from vllm.scalar_type import scalar_types
    except Exception:  # noqa: BLE001
        return None

    if not hasattr(scalar_types, "uint8b128"):
        return None

    padded_n = ((n + 63) // 64) * 64
    if padded_n != n:
        pad_rows = padded_n - n
        qw_use = F.pad(qweight, (0, 0, 0, pad_rows), value=0)
        sc_use = F.pad(scale, (0, 0, 0, pad_rows), value=0.0)
    else:
        qw_use = qweight
        sc_use = scale

    # Bias signed int8 [-127, 127] by +128 -> uint8b128 [1, 255] (zero -> 128).
    u8 = (qw_use.to(torch.int16) + 128).to(torch.uint8)
    # Pack 4 uint8 rows along K into int32 [K // 4, padded_n] for gptq_marlin_repack(num_bits=8).
    u8_kn = u8.t().contiguous().view(k // 4, 4, padded_n)
    packed_i32 = (
        u8_kn[:, 0, :].to(torch.int32)
        | (u8_kn[:, 1, :].to(torch.int32) << 8)
        | (u8_kn[:, 2, :].to(torch.int32) << 16)
        | (u8_kn[:, 3, :].to(torch.int32) << 24)
    ).contiguous()

    try:
        marlin_qw = ops.gptq_marlin_repack(
            b_q_weight=packed_i32,
            perm=torch.empty(0, dtype=torch.int, device=qweight.device),
            size_k=k,
            size_n=padded_n,
            num_bits=8,
        )
        marlin_sc = marlin_permute_scales(
            sc_use.to(qweight.dtype if qweight.is_floating_point() else torch.bfloat16)
            .t()
            .contiguous(),
            size_k=k,
            size_n=padded_n,
            group_size=GROUP_SIZE,
        )
        dev_idx = (
            torch.cuda.current_device()
            if qweight.device.index is None
            else int(qweight.device.index)
        )
        workspace = _shared_marlin_workspace(dev_idx)
    except Exception:  # noqa: BLE001
        return None
    return marlin_qw, marlin_sc, workspace, padded_n


def int8_g32_linear(
    x: torch.Tensor,
    qweight: torch.Tensor,
    weight_scale: torch.Tensor,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Evaluate ``x @ dequant(qweight, weight_scale)^T`` on CPU or CUDA."""
    out_dtype = x.dtype if out_dtype is None else out_dtype
    *lead, k = x.shape
    x2 = x.reshape(-1, k).contiguous()
    if not x2.is_cuda:
        w = dequantize_int8_g32(qweight, weight_scale, out_dtype=x2.dtype)
        return F.linear(x2, w).to(out_dtype).reshape(*lead, qweight.shape[0])
    return int8_g32_triton_linear(x2, qweight, weight_scale, out_dtype).reshape(
        *lead, qweight.shape[0]
    )


class Int8G32Linear(BaseOP):
    """Symmetric ``int8 g32`` W8A16 linear layer (Marlin ``uint8b128`` fast path + Triton/CPU fallback)."""

    def __init__(
        self,
        local_isize: int,
        local_osize: int,
        *,
        all_reduce: bool = False,
        group_size: int = GROUP_SIZE,
    ) -> None:
        assert local_isize % group_size == 0, (local_isize, group_size)
        self.local_input_size = local_isize
        self.local_output_size = local_osize
        self.group_size = group_size
        self.weight = torch.empty((local_osize, local_isize), dtype=torch.int8)
        self.weight_scale = torch.empty(
            (local_osize, local_isize // group_size), dtype=torch.float16
        )
        self._marlin_pack: (
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int] | None
        ) = None
        self._marlin_checked = False
        self._comm = (
            DistributedCommunicator() if all_reduce and get_tp_info().size > 1 else None
        )

    def finalize_quant(self) -> None:
        self.prepare_marlin()

    def prepare_marlin(self) -> bool:
        """Optionally repack ``(weight, weight_scale)`` into Marlin ``uint8b128`` layout on GPU."""
        self._marlin_checked = True
        packed = _try_pack_marlin_u8b128(self.weight, self.weight_scale)
        if packed is None:
            return False
        mqw, msc, ws, padded_n = packed
        dev = self.weight.device
        empty_i32 = torch.empty(0, dtype=torch.int, device=dev)
        self._marlin_pack = (mqw, msc, ws, empty_i32, padded_n)
        self.weight = torch.empty(0, dtype=torch.int8, device=dev)
        self.weight_scale = torch.empty(0, dtype=torch.float16, device=dev)
        return True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (
            not self._marlin_checked
            and x.is_cuda
            and not torch.cuda.is_current_stream_capturing()
        ):
            self.prepare_marlin()
        if self._marlin_pack is not None:
            from vllm.model_executor.layers.quantization.utils.marlin_utils import (
                apply_gptq_marlin_linear,
            )
            from vllm.scalar_type import scalar_types

            mqw, msc, ws, empty_i32, padded_n = self._marlin_pack
            *lead, k = x.shape
            x2 = x.reshape(-1, k)
            y = apply_gptq_marlin_linear(
                input=x2,
                weight=mqw,
                weight_scale=msc,
                weight_zp=empty_i32,
                g_idx=empty_i32,
                g_idx_sort_indices=empty_i32,
                workspace=ws,
                w_type=scalar_types.uint8b128,
                output_size_per_partition=padded_n,
                input_size_per_partition=self.local_input_size,
                is_k_full=True,
            )
            if padded_n != self.local_output_size:
                y = y[:, : self.local_output_size]
            y = y.reshape(*lead, self.local_output_size)
        else:
            y = int8_g32_linear(x, self.weight, self.weight_scale)
        if self._comm is not None:
            y = self._comm.all_reduce(y)
        return y


class Int8G32ColMerged(Int8G32Linear):
    """Column-parallel merged ``int8 g32`` projection."""

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        local_output_sizes: list[int] | None = None,
    ) -> None:
        tp = get_tp_info()
        if local_output_sizes is None:
            local_output_sizes = [div_even(size, tp.size) for size in output_sizes]
        self.output_sizes = list(output_sizes)
        self.local_output_sizes = list(local_output_sizes)
        super().__init__(input_size, sum(local_output_sizes))


class Int8G32RowParallel(Int8G32Linear):
    """Row-parallel ``int8 g32`` projection with post-GEMM TP ``all_reduce``."""

    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__(
            div_even(input_size, get_tp_info().size), output_size, all_reduce=True
        )


class Int8G32Replicated(Int8G32Linear):
    """Replicated ``int8 g32`` projection (e.g. ``hc8`` Hyper-Connection ``down`` / ``up`` mixers)."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        output_sizes: list[int] | None = None,
    ) -> None:
        self.output_sizes = (
            list(output_sizes) if output_sizes is not None else [output_size]
        )
        super().__init__(input_size, output_size, all_reduce=False)


class Int8G32ParallelLMHead(ParallelLMHead):
    """Vocabulary-parallel ``lm_head`` backed by ``int8 g32`` W8A16 weights."""

    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__(num_embeddings, embedding_dim, tie_word_embeddings=False)
        local_vocab = self.weight.shape[0]
        self.weight = torch.empty((local_vocab, embedding_dim), dtype=torch.int8)
        self.weight_scale = torch.empty(
            (local_vocab, embedding_dim // GROUP_SIZE), dtype=torch.float16
        )

    def _project(self, x: torch.Tensor) -> torch.Tensor:
        return int8_g32_linear(
            x, self.weight, self.weight_scale, out_dtype=torch.float32
        )


__all__ = [
    "GROUP_SIZE",
    "Int8G32ColMerged",
    "Int8G32Linear",
    "Int8G32ParallelLMHead",
    "Int8G32Replicated",
    "Int8G32RowParallel",
    "dequantize_int8_g32",
    "int8_g32_linear",
    "quantize_int8_g32",
]
