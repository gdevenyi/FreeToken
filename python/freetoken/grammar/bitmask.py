"""Packed int32 vocabulary bitmask utilities and CUDA/ROCm Triton + CPU appliers."""

from __future__ import annotations

import torch


def bitmask_word_count(vocab_size: int) -> int:
    """Number of int32 words needed to store one bit per token in ``[0, vocab_size)``."""
    return (int(vocab_size) + 31) // 32


def allocate_host_bitmask(
    num_rows: int, vocab_size: int, *, pin_memory: bool = False
) -> torch.Tensor:
    """Allocate a ``[num_rows, ceil(vocab_size / 32)]`` int32 host bitmask initialized to all-ones."""
    words = bitmask_word_count(vocab_size)
    buf = torch.empty((num_rows, words), dtype=torch.int32, pin_memory=pin_memory)
    buf.fill_(-1)
    return buf


def bitmask_allows_token(bitmask: torch.Tensor, row_idx: int, token_id: int) -> bool:
    """Return True if ``token_id`` is allowed in ``bitmask[row_idx]`` (CPU helper)."""
    word_idx = token_id >> 5
    if word_idx < 0 or word_idx >= bitmask.shape[1]:
        return False
    word = int(bitmask[row_idx, word_idx].item())
    return bool((word >> (token_id & 31)) & 1)


_TRITON_KERNEL = None


def _get_triton_bitmask_kernel():
    global _TRITON_KERNEL
    if _TRITON_KERNEL is not None:
        return _TRITON_KERNEL
    import triton
    import triton.language as tl

    @triton.jit
    def _apply_bitmask_kernel(
        logits_ptr,
        bitmask_ptr,
        rows_ptr,
        logits_stride,
        bitmask_stride,
        vocab_size,
        HAS_ROWS: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        g_idx = tl.program_id(0)
        blk = tl.program_id(1)
        if HAS_ROWS:
            row_idx = tl.load(rows_ptr + g_idx).to(tl.int64)
        else:
            row_idx = g_idx.to(tl.int64)
        offs = blk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < vocab_size
        word_idx = offs >> 5
        bit_pos = offs & 31
        bm_base = g_idx.to(tl.int64) * bitmask_stride
        words = tl.load(bitmask_ptr + bm_base + word_idx, mask=mask, other=-1)
        allowed = ((words >> bit_pos) & 1) != 0
        disallowed = mask & (~allowed)
        row_base = row_idx * logits_stride
        tl.store(logits_ptr + row_base + offs, float("-inf"), mask=disallowed)

    _TRITON_KERNEL = _apply_bitmask_kernel
    return _TRITON_KERNEL


def apply_token_bitmask_inplace(
    logits: torch.Tensor,
    bitmask: torch.Tensor,
    rows: torch.Tensor | None = None,
    vocab_size: int | None = None,
) -> None:
    """Mask disallowed tokens (bit == 0) to ``-inf`` in-place on ``logits``.

    Works identically on CUDA, ROCm (via Triton), and CPU (via vectorized PyTorch bit
    unpacking). ``bitmask`` has shape ``[g, ceil(V / 32)]`` and dtype ``torch.int32``;
    ``rows`` (optional ``[g]`` int64) maps each bitmask row to a row in ``logits``.
    """
    if bitmask.numel() == 0 or bitmask.shape[0] == 0:
        return
    v = int(vocab_size if vocab_size is not None else logits.shape[-1])
    g = int(bitmask.shape[0])
    if logits.is_cuda:
        import triton

        if bitmask.device != logits.device:
            bitmask = bitmask.to(logits.device, non_blocking=True)
        if rows is not None and rows.device != logits.device:
            rows = rows.to(logits.device, non_blocking=True)
        kernel = _get_triton_bitmask_kernel()
        block = 1024
        grid = (g, triton.cdiv(v, block))
        kernel[grid](
            logits,
            bitmask,
            rows if rows is not None else bitmask,
            logits.stride(0),
            bitmask.stride(0),
            v,
            HAS_ROWS=rows is not None,
            BLOCK_SIZE=block,
        )
        return

    # CPU / non-CUDA vectorized fallback.
    bm = bitmask.to(device=logits.device, dtype=torch.int32)
    shifts = torch.arange(32, dtype=torch.int32, device=logits.device)
    bits = ((bm.unsqueeze(-1) >> shifts) & 1).ne(0).reshape(g, -1)[:, :v]
    if rows is None:
        logits[:g, :v].masked_fill_(~bits, float("-inf"))
    else:
        row_idx = rows.to(device=logits.device, dtype=torch.long)
        selected = logits.index_select(0, row_idx)
        selected[:, :v].masked_fill_(~bits, float("-inf"))
        logits.index_copy_(0, row_idx, selected)
