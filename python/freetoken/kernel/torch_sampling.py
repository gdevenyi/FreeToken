"""Torch sampling fallback for GPUs whose atomics triton cannot lower.

``freetoken.kernel.triton.sampling`` finds its top-p / top-k thresholds with histogram
and reduction passes that accumulate through ``tl.atomic_add`` / ``tl.atomic_max``.
Triton lowers every atomic to a scoped, ordered PTX encoding (``atom.global.gpu.<sem>``)
whose scope and memory order both arrived with sm_70, so on Pascal ptxas rejects those
kernels outright and any request with ``top_k`` or ``top_p`` set -- the default for most
models -- kills the scheduler.

The threshold search is what needs atomics; sorting does not. These wrappers reproduce
the same selection with plain torch ops, sharing the triton ``softmax`` (which has no
atomics and compiles everywhere). Sorting the whole vocabulary is slower than the
bracketed histogram search the triton path uses, but it is correct, it is CUDA-graph
capturable, and on a card in this class sampling is not the bottleneck.

Selected by :func:`freetoken.engine.sample.sampling_module`; nothing else should import it.
"""

from __future__ import annotations

import torch

from freetoken.kernel.triton.sampling import softmax as _triton_softmax


def softmax(
    logits: torch.Tensor,
    temperature: torch.Tensor | float | None = None,
    enable_pdl: bool = False,
) -> torch.Tensor:
    if logits.device.type != "cuda":
        scaled = logits.float()
        if temperature is not None:
            t = _per_row(temperature, scaled.shape[0], scaled.device, torch.float32)
            scaled = scaled / t.clamp_min(1e-6)
        return torch.softmax(scaled, dim=-1)
    return _triton_softmax(logits, temperature, enable_pdl=enable_pdl)


__all__ = [
    "sampling_from_probs",
    "softmax",
    "top_k_renorm_probs",
    "top_k_sampling_from_probs",
    "top_k_top_p_sampling_from_probs",
    "top_p_renorm_probs",
    "top_p_sampling_from_probs",
]


def _per_row(
    value, rows: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """A scalar or per-row tensor as a ``(rows, 1)`` column, for broadcasting."""
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype).reshape(rows, 1)
    return torch.full((rows, 1), value, device=device, dtype=dtype)


def _renormalize(kept: torch.Tensor) -> torch.Tensor:
    total = kept.sum(-1, keepdim=True)
    # A row whose whole mass was filtered out (all-zero probs) would divide by zero;
    # leave it uniform rather than emitting nan, matching the kernel path's behaviour
    # of always returning a drawable distribution.
    return torch.where(
        total > 0,
        kept / total.clamp_min(torch.finfo(kept.dtype).tiny),
        torch.full_like(kept, 1.0 / kept.shape[-1]),
    )


def top_k_renorm_probs(probs: torch.Tensor, top_k) -> torch.Tensor:
    """Zero everything below the k-th largest probability per row, then renormalize.

    Values tied with the k-th are kept, so a row can retain more than ``k`` entries --
    the same behaviour as the threshold-based kernel path, which cannot separate ties
    either. Rows with ``k >= vocab`` are returned unchanged so solo and mixed batches
    produce bit-identical distributions.
    """
    probs = probs.float()
    rows, vocab = probs.shape
    k = _per_row(top_k, rows, probs.device, torch.long).clamp_(1, vocab)
    kth = probs.sort(dim=-1, descending=True).values.gather(1, k - 1)
    active = k < vocab
    kept = probs.masked_fill(active & (probs < kth), 0.0)
    return torch.where(active, _renormalize(kept), probs)


def top_p_renorm_probs(probs: torch.Tensor, top_p) -> torch.Tensor:
    """Keep the shortest descending prefix whose mass reaches ``top_p``, renormalize.

    Rows with ``top_p >= 1.0`` are returned unchanged so solo and mixed batches produce
    bit-identical distributions.
    """
    probs = probs.float()
    rows, _ = probs.shape
    p = _per_row(top_p, rows, probs.device, torch.float32)
    ordered, order = probs.sort(dim=-1, descending=True)
    # Exclusive cumulative mass: an entry is dropped only once everything *before* it
    # already reached p, which keeps the element that crosses the threshold.
    drop = (ordered.cumsum(-1) - ordered) >= p
    drop[:, 0] = False
    active = p < 1.0
    drop = drop & active
    kept = torch.zeros_like(probs).scatter_(1, order, ordered.masked_fill(drop, 0.0))
    return torch.where(active, _renormalize(kept), probs)


def _draw(
    probs: torch.Tensor,
    seed=None,
    offset=None,
    uniforms: torch.Tensor | None = None,
) -> torch.Tensor:
    """Inverse-CDF draw, one token per row.

    ``searchsorted`` over the cumulative distribution rather than ``torch.multinomial``:
    it stays capturable in a CUDA graph and never syncs to the host. When ``uniforms``
    (``[rows]`` float32 in ``[0, 1)``) is supplied, each row draws from its own uniform
    independently of other rows in the batch.
    """
    rows, vocab = probs.shape
    if uniforms is not None:
        u = (
            uniforms.to(device=probs.device, dtype=torch.float32)
            .reshape(rows, 1)
            .clamp(0.0, 1.0 - 2.0**-24)
        )
    elif seed is not None and not torch.cuda.is_current_stream_capturing():
        generator = torch.Generator(device=probs.device)
        s = int(seed if not isinstance(seed, torch.Tensor) else seed.view(-1)[0])
        o = (
            0
            if offset is None
            else int(
                offset if not isinstance(offset, torch.Tensor) else offset.view(-1)[0]
            )
        )
        generator.manual_seed((s * 0x9E3779B97F4A7C15 + o) & 0x7FFFFFFFFFFFFFFF)
        u = torch.rand(
            rows, 1, device=probs.device, dtype=torch.float32, generator=generator
        )
    else:
        u = torch.rand(rows, 1, device=probs.device, dtype=torch.float32)
    cdf = probs.cumsum(-1)
    idx = torch.searchsorted(
        cdf.contiguous(), (u * cdf[:, -1:]).contiguous(), right=True
    )
    # Guard against fp32 rounding past the last positive-mass token (e.g. masked tails).
    ar = torch.arange(vocab, device=probs.device).unsqueeze(0)
    last_nz = torch.where(probs > 0, ar, -1).amax(dim=-1, keepdim=True).clamp_min(0)
    idx = torch.minimum(idx, last_nz)
    return idx.squeeze(-1).to(torch.int32)


def _finish(out: torch.Tensor, indices, return_valid: bool):
    out = out.to(indices.dtype) if indices is not None else out
    return (out, torch.ones_like(out, dtype=torch.bool)) if return_valid else out


def _source(probs: torch.Tensor, indices) -> torch.Tensor:
    probs = probs.float()
    return probs if indices is None else probs[indices].contiguous()


def sampling_from_probs(
    probs,
    indices=None,
    deterministic=True,
    generator=None,
    check_nan=False,
    seed=None,
    offset=None,
    return_valid=False,
    uniforms=None,
):
    src = _source(probs, indices)
    return _finish(_draw(src, seed, offset, uniforms=uniforms), indices, return_valid)


def top_k_sampling_from_probs(
    probs,
    top_k,
    indices=None,
    deterministic=True,
    generator=None,
    check_nan=False,
    seed=None,
    offset=None,
    return_valid=False,
    uniforms=None,
):
    src = _source(probs, indices)
    return _finish(
        _draw(top_k_renorm_probs(src, top_k), seed, offset, uniforms=uniforms),
        indices,
        return_valid,
    )


def top_p_sampling_from_probs(
    probs,
    top_p,
    indices=None,
    deterministic=True,
    generator=None,
    check_nan=False,
    seed=None,
    offset=None,
    return_valid=False,
    uniforms=None,
):
    src = _source(probs, indices)
    return _finish(
        _draw(top_p_renorm_probs(src, top_p), seed, offset, uniforms=uniforms),
        indices,
        return_valid,
    )


def top_k_top_p_sampling_from_probs(
    probs,
    top_k,
    top_p,
    indices=None,
    filter_apply_order="top_k_first",
    deterministic=True,
    generator=None,
    check_nan=False,
    seed=None,
    offset=None,
    return_valid=False,
    uniforms=None,
):
    src = _source(probs, indices)
    renormed = top_p_renorm_probs(top_k_renorm_probs(src, top_k), top_p)
    return _finish(
        _draw(renormed, seed, offset, uniforms=uniforms), indices, return_valid
    )
