"""MXFP8 (block-32 e8m0-scaled FP8) W8A16 dense linear, shared across models.

MiniMax-M3's modelopt checkpoint quantizes every dense projection (q/k/v/o, dense
MLPs, shared experts, indexer) to MXFP8: fp8-e4m3 ``weight [N, K]`` plus uint8 e8m0
exponent codes ``weight_scale_inv [N, K//32]``, dequant multiplier ``2**(code - 127)``.

Decode is weight-bandwidth bound, so small batches read the fp8 weight directly in a
split-K GEMV (half the traffic of a bf16-resident weight). The block-32 scale is
loaded once per tile and broadcast in registers; a per-element scale gather + exp2
costs ~3x the whole GEMV. Past the GEMV cap the forward dequantizes to bf16
(``fp8 * 2**k`` is exact in bf16) and runs cuBLAS -- a fused inline-dequant GEMM
never came close to cuBLAS on these shapes, so the transient is the fast option.

The GEMV accumulates in fp32; the cuBLAS path is bf16 with fp32 accumulate, same as
any other bf16 projection.
"""

from __future__ import annotations

import logging

import torch
import triton
import triton.language as tl
from triton.runtime.errors import OutOfResources

from freetoken.kernel.triton.e4m3_compat import e4m3_kernel_view, e4m3_native_cx, e4m3_u8_to_f32

logger = logging.getLogger(__name__)

FP8 = torch.float8_e4m3fn
MXFP8_BLOCK = 32
_TL_DTYPE = {torch.bfloat16: tl.bfloat16, torch.float16: tl.float16, torch.float32: tl.float32}


def mxfp8_dequant(weight: torch.Tensor, scale_codes: torch.Tensor,
                  dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """Dequant: ``w[n, k] * 2**(codes[n, k//32] - 127)`` -> ``dtype``.

    Serves the large-M forward (dequant + cuBLAS), the load-time bf16 ablation
    path and the kernels' numeric reference in tests. For 16-bit dtypes the
    whole computation runs in that dtype -- the pow2 descale is lossless there,
    and fp32 transients would double the peak footprint (which lands in the
    graph pool when captured). fp32 keeps fp32 compute: it is the reference.
    """
    assert weight.shape[-1] % MXFP8_BLOCK == 0
    assert scale_codes.shape[-1] == weight.shape[-1] // MXFP8_BLOCK
    compute = torch.float32 if dtype == torch.float32 else dtype
    descale = torch.exp2(scale_codes.to(torch.float32) - 127.0).to(compute)
    w = weight.to(compute).view(*weight.shape[:-1], -1, MXFP8_BLOCK)
    return (w * descale.unsqueeze(-1)).view(weight.shape).to(dtype)


# GEMV cap. Must cover the CUDA-graph ladder max (256): a captured dequant+cuBLAS
# call would bake its full-weight transient into the shared graph pool (~2 GiB
# across the resident shapes), and the GEMV is also simply faster everywhere in
# graph range (M=128: ~230us vs ~930us on RTX PRO 6000). M_TILE buckets {16..256};
# 512 exceeds sm_120's shared-memory budget. Past the cap only prefill lands here,
# where cuBLAS amortizes the transient.
_GEMV_MAX_M = 256
# Lowered in place when a device cannot allocate an M_TILE's shared memory (Pascal's 48 KB
# fits 64): OutOfResources is raised at launch setup, before any GPU work, so a retry is safe.
_gemv_cap = _GEMV_MAX_M
# Weight rows dequantized per cuBLAS call past the GEMV: bounds the 16-bit transient, which for
# a whole lm_head would be ~1.2 GiB (x2 inside mxfp8_dequant), without splitting dense projections.
_DEQUANT_CHUNK_BYTES = 64 << 20


def _small_m_gemv_ok() -> bool:
    # Below sm_70 tl.dot has no tensor cores and the M>1 GEMV runs 5-30x behind dequant+cuBLAS.
    from freetoken.kernel.backend import device_capability

    return device_capability() >= (7, 0)


@triton.jit
def _mxfp8_gemv_m1_splitk_kernel(
    a_ptr, w_ptr, s_ptr, part_ptr, N, K, n_kb, kb_per,
    stride_ak, stride_wn, stride_wk, stride_sn, stride_sk, stride_pk, stride_pn,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """M == 1 specialization: a plain fp32 multiply-reduce. The dot tile's 16-row
    padding costs ~30% at M=1 on RTX PRO 6000 / 5090, hence this kernel -- but on
    H100 the dot kernel is ~26% faster even at M=1. If sm_90 becomes a serving
    target, re-benchmark and dispatch by arch (or drop this kernel)."""
    KB32: tl.constexpr = BLOCK_K // 32
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    off_kb32 = tl.arange(0, KB32)
    kb_start = pid_k * kb_per
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for i in range(kb_per):
        kb = kb_start + i
        if kb < n_kb:
            offs_k = kb * BLOCK_K + tl.arange(0, BLOCK_K)
            k_mask = offs_k < K
            a = tl.load(a_ptr + offs_k * stride_ak, mask=k_mask, other=0.0).to(tl.float32)
            if e4m3_native_cx():
                w = tl.load(
                    w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk,
                    mask=n_mask[:, None] & k_mask[None, :], other=0.0,
                ).to(tl.float32)
            else:
                w = e4m3_u8_to_f32(tl.load(
                    w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk,
                    mask=n_mask[:, None] & k_mask[None, :], other=0,
                ))
            codes = tl.load(
                s_ptr + offs_n[:, None] * stride_sn
                + (kb * KB32 + off_kb32[None, :]) * stride_sk,
                mask=n_mask[:, None] & ((kb * KB32 + off_kb32[None, :]) * 32 < K),
                other=127,
            ).to(tl.float32)  # [BLOCK_N, KB32]
            scale = tl.exp2(codes - 127.0)
            prod = tl.reshape(w * a[None, :], (BLOCK_N, KB32, 32))
            acc += tl.sum(tl.sum(prod, axis=2) * scale, axis=1)
    tl.store(part_ptr + pid_k * stride_pk + offs_n * stride_pn, acc, mask=n_mask)


@triton.jit
def _mxfp8_gemv_splitk_kernel(
    a_ptr, w_ptr, s_ptr, part_ptr, M, N, K, n_kb, kb_per,
    stride_am, stride_ak, stride_wn, stride_wk, stride_sn, stride_sk,
    stride_pk, stride_pm, stride_pn,
    M_TILE: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Each (pid_n, pid_k) accumulates ``kb_per`` BLOCK_K chunks of ``a[:M] @ w^T``
    for a BLOCK_N slice of outputs. BLOCK_K is a multiple of 32, so each chunk
    covers ``BLOCK_K // 32`` whole scale blocks: one ``[BLOCK_N, KB32]`` code load
    + exp2 per tile, broadcast over the 32-wide inner axis via a 3D register view,
    folded into the fp8 weight BEFORE the dot (pow2 scaling is lossless in bf16)."""
    KB32: tl.constexpr = BLOCK_K // 32
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_m = tl.arange(0, M_TILE)
    m_mask = offs_m < M
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    off_kb32 = tl.arange(0, KB32)
    kb_start = pid_k * kb_per
    acc = tl.zeros((M_TILE, BLOCK_N), dtype=tl.float32)
    for i in range(kb_per):
        kb = kb_start + i
        if kb < n_kb:
            offs_k = kb * BLOCK_K + tl.arange(0, BLOCK_K)
            k_mask = offs_k < K
            a = tl.load(
                a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
                mask=m_mask[:, None] & k_mask[None, :], other=0.0,
            )  # [M_TILE, BLOCK_K] bf16
            if e4m3_native_cx():
                w = tl.load(
                    w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk,
                    mask=n_mask[:, None] & k_mask[None, :], other=0.0,
                ).to(tl.float32)
            else:
                w = e4m3_u8_to_f32(tl.load(
                    w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk,
                    mask=n_mask[:, None] & k_mask[None, :], other=0,
                ))
            codes = tl.load(
                s_ptr + offs_n[:, None] * stride_sn
                + (kb * KB32 + off_kb32[None, :]) * stride_sk,
                mask=n_mask[:, None] & ((kb * KB32 + off_kb32[None, :]) * 32 < K),
                other=127,
            ).to(tl.float32)  # [BLOCK_N, KB32]
            scale = tl.exp2(codes - 127.0)
            w3 = tl.reshape(w, (BLOCK_N, KB32, 32)) * scale[:, :, None]
            w_scaled = tl.reshape(w3, (BLOCK_N, BLOCK_K)).to(a.dtype)
            acc += tl.dot(a, tl.trans(w_scaled), out_dtype=tl.float32)
    part_base = part_ptr + pid_k * stride_pk
    tl.store(
        part_base + offs_m[:, None] * stride_pm + offs_n[None, :] * stride_pn,
        acc,
        mask=m_mask[:, None] & n_mask[None, :],
    )


@triton.jit
def _splitk_reduce_kernel(
    part_ptr, out_ptr, N, SPLIT_K: tl.constexpr,
    stride_pk, stride_pm, stride_pn, stride_om, stride_on,
    BLOCK: tl.constexpr, OUT: tl.constexpr,
):
    pid_m = tl.program_id(1)
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for k in tl.static_range(SPLIT_K):
        acc += tl.load(
            part_ptr + k * stride_pk + pid_m * stride_pm + offs * stride_pn,
            mask=mask, other=0.0,
        )
    tl.store(out_ptr + pid_m * stride_om + offs * stride_on, acc.to(OUT), mask=mask)


def _gemv(a: torch.Tensor, weight: torch.Tensor, scale_codes: torch.Tensor,
          out_dtype: torch.dtype) -> torch.Tensor:
    """Small-M split-K GEMV. ``a`` [M, K] bf16 (M <= _GEMV_MAX_M); ``weight``
    [N, K] fp8; ``scale_codes`` [N, K//32] uint8."""
    M, K = a.shape
    N = weight.shape[0]
    BLOCK_K, BLOCK_N, m1_warps, programs = 128, 16, 1, 1536
    if M == 1 and not _small_m_gemv_ok():
        # Pascal: wide K tiles and more warps reach ~330 GB/s of fp8 (1.4-2.2x bf16 cuBLAS)
        # where 128x16 on one warp stalls at ~100-200 GB/s.
        BLOCK_K, BLOCK_N = 512, (8 if N <= 4096 else 16)
        m1_warps, programs = (4 if BLOCK_N == 8 else 8), 448
    n_kb = triton.cdiv(K, BLOCK_K)
    n_tiles = triton.cdiv(N, BLOCK_N)
    split_k = max(1, min(programs // n_tiles, n_kb))
    split_k = 1 << (split_k.bit_length() - 1)  # pow2 -> stable reduction order
    kb_per = triton.cdiv(n_kb, split_k)
    part = torch.empty((split_k, M, N), dtype=torch.float32, device=a.device)
    if M == 1:
        _mxfp8_gemv_m1_splitk_kernel[(n_tiles, split_k)](
            a, weight, scale_codes, part, N, K, n_kb, kb_per,
            a.stride(1), weight.stride(0), weight.stride(1),
            scale_codes.stride(0), scale_codes.stride(1),
            part.stride(0), part.stride(2),
            BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, num_warps=m1_warps,
        )
    else:
        # M_TILE buckets to the next pow2 >= 16 (tl.dot minimum); the whole batch
        # rides one weight pass regardless of bucket, so the only cost of a bigger
        # tile is row padding, amortized with proportionally more warps.
        m_tile = max(16, triton.next_power_of_2(M))
        _mxfp8_gemv_splitk_kernel[(n_tiles, split_k)](
            a, weight, scale_codes, part, M, N, K, n_kb, kb_per,
            a.stride(0), a.stride(1), weight.stride(0), weight.stride(1),
            scale_codes.stride(0), scale_codes.stride(1),
            part.stride(0), part.stride(1), part.stride(2),
            M_TILE=m_tile, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=m_tile // 16,
        )
    out = torch.empty((M, N), dtype=out_dtype, device=a.device)
    _splitk_reduce_kernel[(triton.cdiv(N, 256), M)](
        part, out, N, split_k,
        part.stride(0), part.stride(1), part.stride(2), out.stride(0), out.stride(1),
        BLOCK=256, OUT=_TL_DTYPE[out_dtype if out_dtype in _TL_DTYPE else torch.bfloat16],
        num_warps=2,
    )
    return out


def mxfp8_linear(
    x: torch.Tensor, weight: torch.Tensor, scale_codes: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """``y = x @ dequant(weight, scale_codes)^T``. Small batches
    (M <= 256) -> split-K GEMV (one pass over the fp8 weight); larger M -> bf16
    dequant + cuBLAS (see the module docstring). ``weight`` [N, K] fp8-e4m3;
    ``scale_codes`` [N, K//32] uint8 e8m0 exponent codes (dequant multiplier
    ``2**(code - 127)``)."""
    *lead, K = x.shape
    N = weight.shape[0]
    M = x.numel() // K
    assert K % MXFP8_BLOCK == 0 and scale_codes.shape == (N, K // MXFP8_BLOCK)
    global _gemv_cap
    out = None
    while M <= _gemv_cap and (M == 1 or _small_m_gemv_ok()):
        try:
            out = _gemv(x.reshape(M, K), e4m3_kernel_view(weight), scale_codes, x.dtype).reshape(*lead, N)
            break
        except OutOfResources:
            if M == 1:
                raise
            m_tile = max(16, triton.next_power_of_2(M))
            _gemv_cap = m_tile // 2 if m_tile > 16 else 1  # strictly lower: M=1 has its own kernel
            logger.warning(
                "mxfp8 GEMV: M_TILE %d does not fit the shared memory of %s; M > %d falls back to dequant + "
                "cuBLAS, whose transient a CUDA-graph capture keeps in its pool", m_tile, x.device, _gemv_cap,
            )
    if out is None:
        # Per-call bf16 transient (pow2 descale is lossless in bf16) + cuBLAS.
        x2 = x.reshape(-1, K)
        rows = max(1, _DEQUANT_CHUNK_BYTES // (K * x.element_size()))
        if rows >= N:
            out = torch.nn.functional.linear(x2, mxfp8_dequant(weight, scale_codes, dtype=x.dtype))
        else:
            out = torch.empty((x2.shape[0], N), dtype=x.dtype, device=x.device)
            for n0 in range(0, N, rows):
                w = mxfp8_dequant(weight[n0:n0 + rows], scale_codes[n0:n0 + rows], dtype=x.dtype)
                out[:, n0:n0 + rows] = torch.nn.functional.linear(x2, w)
                del w
        out = out.reshape(*lead, N)
    if bias is not None:
        out = out + bias.to(out.dtype)
    return out


__all__ = ["FP8", "MXFP8_BLOCK", "mxfp8_linear", "mxfp8_dequant"]
