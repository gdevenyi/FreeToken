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
import inspect

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from freetoken.distributed import DistributedCommunicator, get_tp_info
from freetoken.models.config import verify_invariant_enabled
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
    pid_m = tl.program_id(0).to(tl.int64)
    pid_n = tl.program_id(1).to(tl.int64)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
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
                mask=mask_n & (gk < K),
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


@triton.jit
def _int8_g32_gemv_kernel(
    x_ptr,
    qw_ptr,
    sc_ptr,
    y_ptr,
    N,
    K,
    stride_wn,
    stride_wk,
    stride_sn,
    stride_sg,
    GROUP_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0).to(tl.int64)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
    mask_n = offs_n < N
    w_row = qw_ptr + offs_n[:, None] * stride_wn
    s_row = sc_ptr + offs_n[:, None] * stride_sn
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    NUM_GROUPS_PER_STEP: tl.constexpr = BLOCK_K // GROUP_SIZE

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        xk = tl.load(x_ptr + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        qw = tl.load(
            w_row + offs_k[None, :] * stride_wk,
            mask=mask_n[:, None] & mask_k[None, :],
            other=0,
        ).to(tl.float32)
        offs_g = (k0 // GROUP_SIZE) + tl.arange(0, NUM_GROUPS_PER_STEP)
        mask_g = (offs_g * GROUP_SIZE) < K
        sc = tl.load(
            s_row + offs_g[None, :] * stride_sg,
            mask=mask_n[:, None] & mask_g[None, :],
            other=0.0,
        ).to(tl.float32)
        qw_g = tl.reshape(qw, (BLOCK_N, NUM_GROUPS_PER_STEP, GROUP_SIZE))
        xk_g = tl.reshape(xk, (1, NUM_GROUPS_PER_STEP, GROUP_SIZE))
        grp_dot = tl.sum(qw_g * xk_g, axis=2)
        acc += tl.sum(grp_dot * sc, axis=1)

    tl.store(y_ptr + offs_n, acc.to(y_ptr.dtype.element_ty), mask=mask_n)


def int8_g32_triton_linear(
    x2: torch.Tensor,
    qweight: torch.Tensor,
    scale: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """CUDA-graph-safe Triton W8A16 GEMM/GEMV for ``qweight [N, K] int8`` and ``scale [N, K//32] fp16``."""
    m, k = x2.shape
    n = qweight.shape[0]
    y = torch.empty((m, n), dtype=out_dtype, device=x2.device)
    if m == 0 or n == 0:
        return y
    inv = verify_invariant_enabled()
    if m == 1 and not inv:
        block_n = 4
        block_k = min(max(GROUP_SIZE, triton.next_power_of_2(k)), 1024)
        _int8_g32_gemv_kernel[(triton.cdiv(n, block_n),)](
            x2,
            qweight,
            scale,
            y,
            n,
            k,
            qweight.stride(0),
            qweight.stride(1),
            scale.stride(0),
            scale.stride(1),
            GROUP_SIZE=GROUP_SIZE,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            num_warps=4,
        )
        return y
    if m <= 16 or (inv and m <= 64):
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


@functools.cache
def _marlin_accepted_params(fn) -> frozenset[str]:
    return frozenset(inspect.signature(fn).parameters)


def _call_gptq_marlin_repack(
    repack_fn,
    *,
    b_q_weight: torch.Tensor,
    perm: torch.Tensor,
    size_k: int,
    size_n: int,
    num_bits: int,
) -> torch.Tensor:
    accepted = _marlin_accepted_params(repack_fn)
    kwargs = {
        "b_q_weight": b_q_weight,
        "perm": perm,
        "size_k": size_k,
        "size_n": size_n,
        "num_bits": num_bits,
    }
    return repack_fn(**{k: v for k, v in kwargs.items() if k in accepted})


def _call_gptq_marlin_linear(
    apply_fn,
    *,
    input_t: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    empty_i32: torch.Tensor,
    workspace: torch.Tensor,
    w_type,
    output_size_per_partition: int,
    input_size_per_partition: int,
) -> torch.Tensor:
    accepted = _marlin_accepted_params(apply_fn)
    kwargs = {
        "input": input_t,
        "weight": weight,
        "weight_scale": weight_scale,
        "weight_zp": empty_i32,
        "g_idx": empty_i32,
        "g_idx_sort_indices": empty_i32,
        "workspace": workspace,
        "output_size_per_partition": output_size_per_partition,
        "input_size_per_partition": input_size_per_partition,
        "is_k_full": True,
    }
    if "w_type" in accepted:
        kwargs["w_type"] = w_type
    elif "wtype" in accepted:
        kwargs["wtype"] = w_type
    return apply_fn(**{k: v for k, v in kwargs.items() if k in accepted})


def _try_pack_marlin_u8b128(
    qweight: torch.Tensor,
    scale: torch.Tensor,
    act_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int] | None:
    """Attempt to repack ``int8 [N, K]`` + ``fp16 [N, K//32]`` into vLLM Marlin ``uint8b128`` layout.

    Pads ``N`` up to a multiple of 64 if needed (`hc.down` `336 -> 384`). Returns
    ``(marlin_qweight, marlin_scales, workspace, padded_n)`` or ``None`` if Marlin is unavailable.
    """
    if not qweight.is_cuda or qweight.ndim != 2 or qweight.numel() == 0:
        return None
    n, k = qweight.shape
    if k % 64 != 0:
        return None
    scale_dtype = (
        act_dtype if act_dtype in (torch.float16, torch.bfloat16) else torch.bfloat16
    )
    try:
        from vllm import _custom_ops as ops
        from vllm.model_executor.layers.quantization.utils import marlin_utils
        from vllm.model_executor.layers.quantization.utils.marlin_utils import (
            apply_gptq_marlin_linear,
            marlin_permute_scales,
        )
        from vllm.scalar_type import scalar_types
    except Exception:  # noqa: BLE001
        return None

    if not hasattr(scalar_types, "uint8b128"):
        return None

    padded_n = ((n + 63) // 64) * 64
    if k % 128 != 0 and padded_n % 256 != 0:
        padded_n = ((n + 255) // 256) * 256

    dev_idx = (
        torch.cuda.current_device()
        if qweight.device.index is None
        else int(qweight.device.index)
    )
    try:
        if hasattr(marlin_utils, "check_marlin_supported"):
            cap = torch.cuda.get_device_capability(dev_idx)
            cap_int = cap[0] * 10 + cap[1]
            if not marlin_utils.check_marlin_supported(
                quant_type=scalar_types.uint8b128,
                group_size=GROUP_SIZE,
                has_zp=False,
                device_capability=cap_int,
            ):
                return None
    except Exception:  # noqa: BLE001
        pass

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
        with torch.cuda.device(qweight.device):
            empty_i32 = torch.empty(0, dtype=torch.int, device=qweight.device)
            marlin_qw = _call_gptq_marlin_repack(
                ops.gptq_marlin_repack,
                b_q_weight=packed_i32,
                perm=empty_i32,
                size_k=k,
                size_n=padded_n,
                num_bits=8,
            )
            marlin_sc = marlin_permute_scales(
                sc_use.to(scale_dtype).t().contiguous(),
                size_k=k,
                size_n=padded_n,
                group_size=GROUP_SIZE,
            )
            workspace = _shared_marlin_workspace(dev_idx)
            probe_x = torch.zeros((1, k), dtype=scale_dtype, device=qweight.device)
            _ = _call_gptq_marlin_linear(
                apply_gptq_marlin_linear,
                input_t=probe_x,
                weight=marlin_qw,
                weight_scale=marlin_sc,
                empty_i32=empty_i32,
                workspace=workspace,
                w_type=scalar_types.uint8b128,
                output_size_per_partition=padded_n,
                input_size_per_partition=k,
            )
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
        self._act_dtype = torch.get_default_dtype()
        self._comm = (
            DistributedCommunicator() if all_reduce and get_tp_info().size > 1 else None
        )

    def input_format(self, _x: torch.Tensor) -> str | None:
        return None

    def finalize_quant(self) -> None:
        self.prepare_marlin()

    def prepare_marlin(self, act_dtype: torch.dtype | None = None) -> bool:
        """Optionally repack ``(weight, weight_scale)`` into Marlin ``uint8b128`` layout on GPU."""
        if verify_invariant_enabled():
            return False
        if self._marlin_pack is not None:
            return True
        if self._marlin_checked or self.weight.ndim != 2 or self.weight.numel() == 0:
            return False
        self._marlin_checked = True
        packed = _try_pack_marlin_u8b128(
            self.weight, self.weight_scale, act_dtype or self._act_dtype
        )
        if packed is None:
            return False
        mqw, msc, ws, padded_n = packed
        dev = self.weight.device
        empty_i32 = torch.empty(0, dtype=torch.int, device=dev)
        self._marlin_pack = (mqw, msc, ws, empty_i32, padded_n)
        self.weight = torch.empty(0, dtype=torch.int8, device=dev)
        self.weight_scale = torch.empty(0, dtype=torch.float16, device=dev)
        return True

    def forward(self, x: torch.Tensor, _xq: torch.Tensor | None = None) -> torch.Tensor:
        if (
            not self._marlin_checked
            and x.is_cuda
            and not torch.cuda.is_current_stream_capturing()
        ):
            self.prepare_marlin(act_dtype=x.dtype)
        if self._marlin_pack is not None:
            from vllm.model_executor.layers.quantization.utils.marlin_utils import (
                apply_gptq_marlin_linear,
            )
            from vllm.scalar_type import scalar_types

            mqw, msc, ws, empty_i32, padded_n = self._marlin_pack
            *lead, k = x.shape
            x2 = x.reshape(-1, k).contiguous()
            if x2.shape[0] == 0:
                y = torch.empty(
                    (*lead, self.local_output_size), dtype=x.dtype, device=x.device
                )
            else:
                if x2.dtype != msc.dtype:
                    x2 = x2.to(msc.dtype)
                y_m = _call_gptq_marlin_linear(
                    apply_gptq_marlin_linear,
                    input_t=x2,
                    weight=mqw,
                    weight_scale=msc,
                    empty_i32=empty_i32,
                    workspace=ws,
                    w_type=scalar_types.uint8b128,
                    output_size_per_partition=padded_n,
                    input_size_per_partition=self.local_input_size,
                )
                if padded_n != self.local_output_size:
                    y_m = y_m[:, : self.local_output_size]
                if y_m.dtype != x.dtype:
                    y_m = y_m.to(x.dtype)
                y = y_m.reshape(*lead, self.local_output_size)
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

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        tp_size: int | None = None,
        tp_rank: int | None = None,
    ) -> None:
        super().__init__(
            num_embeddings,
            embedding_dim,
            tie_word_embeddings=False,
            tp_size=tp_size,
            tp_rank=tp_rank,
        )
        local_vocab = self.num_embeddings_tp
        self.weight = torch.empty((local_vocab, embedding_dim), dtype=torch.int8)
        self.weight_scale = torch.empty(
            (local_vocab, embedding_dim // GROUP_SIZE), dtype=torch.float16
        )

    def _local_logits(self, x: torch.Tensor) -> torch.Tensor:
        y = int8_g32_linear(x, self.weight, self.weight_scale, out_dtype=torch.float32)
        return y if self.bias is None else y + self.bias

    def keep_rows(self, ids: torch.Tensor, quant_config=None) -> None:
        from freetoken.utils import div_ceil

        ids = ids[torch.arange(div_ceil(ids.numel(), 128) * 128) % ids.numel()]
        rows = ids.numel()
        with torch.device("meta"):
            head = Int8G32ParallelLMHead(rows, self.embedding_dim, tp_size=1, tp_rank=0)
        device = self.weight.device
        ids = ids.to(device=device, dtype=torch.long)
        if self.tp_size == 1:
            head.weight = self.weight.index_select(0, ids)
            head.weight_scale = self.weight_scale.index_select(0, ids)
        else:
            start, count = self.vocab_range
            in_shard = (ids >= start) & (ids < start + count)
            local_ids = ids[in_shard] - start
            codes = torch.zeros(
                rows, self.embedding_dim, dtype=torch.int32, device=device
            )
            scales = torch.zeros(
                rows,
                self.embedding_dim // GROUP_SIZE,
                dtype=torch.float32,
                device=device,
            )
            if local_ids.numel():
                codes[in_shard] = self.weight.index_select(0, local_ids).to(torch.int32)
                scales[in_shard] = self.weight_scale.index_select(0, local_ids).to(
                    torch.float32
                )
            codes = self._comm.all_reduce(codes)
            scales = self._comm.all_reduce(scales)
            head.weight = codes.to(torch.int8)
            head.weight_scale = scales.to(torch.float16)
        self.kept_rows = head, ids


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
