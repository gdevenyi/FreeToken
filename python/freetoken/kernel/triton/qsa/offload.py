"""Triton kernels for the QSA KV host offload (``kvcache/kv_host_offload.py``).

The page table, radix tree and compressed-index slab all run in LOGICAL page space; the GPU
KV buffer is an LRU cache of physical slots over the pinned host mirror. These kernels are
the fixed-shape, CUDA-graph-safe glue between the two spaces:

* :func:`write_pages` -- logical page id of every written token slot (``out_loc // page_size``).
* :func:`compact_selected_pages` -- per query row, the distinct logical pages the sparse
  selection touches (``block_table[req, indices//page_size]``), first-element padded.
* :func:`translate_table` -- logical block table -> physical slot table (post-ensure).
* :func:`mirror_store` -- write-through of this forward's K/V rows into the pinned host
  mirror (UVA stores), so a physical slot is always drop-clean.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from freetoken.kernel.backend import device_capability


@triton.jit
def _write_pages_kernel(out_loc, write_pages, page_size: tl.constexpr):
    i = tl.program_id(0)
    tl.store(write_pages + i, tl.load(out_loc + i) // page_size)


def write_pages(out_loc: torch.Tensor, out: torch.Tensor, page_size: int) -> None:
    """``out[i] = out_loc[i] // page_size`` for i < out.numel() (out_loc may be wider)."""
    n = out.numel()
    if n:
        _write_pages_kernel[(n,)](out_loc, out, page_size)


@triton.jit
def _mark_sel_cols_kernel(indices, marks, sel_width, width, page_size: tl.constexpr, BLOCK: tl.constexpr):
    # Plain stores of the same value: concurrent writers to one column need no atomics.
    r = tl.program_id(0)
    j = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    tok = tl.load(indices + r * sel_width + j, mask=j < sel_width, other=-1)
    col = tl.where(tok >= 0, tok // page_size, 0)
    tl.store(marks + r.to(tl.int64) * width + col, tl.full([BLOCK], 1, tl.int8), mask=tok >= 0)


@triton.jit
def _compact_sel_kernel(
    marks,          # [rows, W] int8: 1 on every page column the row selected; cleared here
    token_to_req,   # [rows] int32
    block_table,    # [reqs, W] int32: logical page ids
    out,            # [rows, MAXP] int32: distinct logical pages per row, column order
    trunc,          # [1] int32: pages dropped past MAXP (0 = none dropped)
    counts,         # [rows] int32: distinct pages stored per row (<= MAXP)
    width,          # W: block_table row stride
    maxp: tl.constexpr,
    maxv: tl.constexpr,      # next_pow2(maxp)
    dummy_page,              # logical id used when a row selects nothing
    HAS_ATOMICS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    r = tl.program_id(0)
    req = tl.load(token_to_req + r).to(tl.int64)
    row_marks = marks + r.to(tl.int64) * width
    lanes = tl.arange(0, BLOCK)
    count = tl.zeros((), tl.int32)
    first = tl.full((), dummy_page, tl.int32)
    for c0 in range(0, width, BLOCK):
        cols = c0 + lanes
        m = tl.load(row_marks + cols, mask=cols < width, other=0).to(tl.int32)
        tl.store(row_marks + cols, tl.zeros([BLOCK], tl.int8), mask=(cols < width) & (m != 0))
        pos = count + tl.cumsum(m, 0) - m
        take = (m != 0) & (pos < maxp)
        pg = tl.load(block_table + req * width + cols, mask=take, other=0)
        tl.store(out + r * maxp + pos, pg, mask=take)
        first = tl.where((count == 0) & (tl.sum(m, 0) > 0), tl.max(tl.where(take & (pos == 0), pg, -1), 0), first)
        count += tl.sum(m, 0)
    stored = tl.minimum(count, maxp)
    pad = tl.arange(0, maxv)
    tl.store(out + r * maxp + pad, tl.full([maxv], 0, tl.int32) + first, mask=(pad >= stored) & (pad < maxp))
    tl.store(counts + r, stored)
    ndrop = count - stored
    if HAS_ATOMICS:
        tl.atomic_max(trunc, ndrop)
    else:
        # Triton lowers every atomic to sm_70+ PTX. Racing plain stores still leave the
        # counter non-zero whenever any row dropped, which is all its readers test.
        tl.store(trunc, ndrop, mask=ndrop > 0)


_COMPACT_BLOCK = 1024


def compact_selected_pages(
    indices: torch.Tensor,
    token_to_req: torch.Tensor,
    block_table: torch.Tensor,
    out: torch.Tensor,
    page_size: int,
    dummy_page: int,
    trunc: torch.Tensor | None = None,
    counts: torch.Tensor | None = None,
    marks: torch.Tensor | None = None,
) -> None:
    """Per query row, the distinct logical pages its selected tokens live on, in page-column order.

    ``out`` is ``[rows, maxp]`` int32; rows with fewer hits are padded with the row's first
    page (duplicates collapse in lru_ensure), or ``dummy_page`` when the row selected nothing.
    ``trunc`` (optional [1] int32, device) accumulates how many pages were dropped per row
    because they exceeded ``maxp`` -- 0 means no truncation happened. ``counts`` (optional
    [rows] int32, device) receives each row's stored distinct-page count. ``marks`` is a
    zeroed int8 scratch of at least ``[rows, block_table.shape[1]]`` that is left zeroed; pass
    one allocated outside CUDA-graph capture.
    """
    rows, sel = indices.shape
    maxp = out.shape[1]
    width = block_table.shape[1]
    if trunc is None:
        trunc = getattr(compact_selected_pages, "_dummy", None)
        if trunc is None or trunc.device != out.device:
            trunc = compact_selected_pages._dummy = torch.zeros(1, dtype=torch.int32, device=out.device)
    if counts is None:
        c = getattr(compact_selected_pages, "_dummy_counts", None)
        if c is None or c.numel() < rows or c.device != out.device:
            c = compact_selected_pages._dummy_counts = torch.empty(
                max(rows, 4096), dtype=torch.int32, device=out.device
            )
        counts = c
    if marks is None:
        marks = compact_marks(rows, width, out.device)
    assert marks.dtype == torch.int8 and marks.numel() >= rows * width
    if rows:
        _mark_sel_cols_kernel[(rows, triton.cdiv(sel, _COMPACT_BLOCK))](
            indices, marks, sel, width, page_size, BLOCK=_COMPACT_BLOCK,
        )
        _compact_sel_kernel[(rows,)](
            marks, token_to_req, block_table, out, trunc, counts,
            width, maxp, triton.next_power_of_2(maxp), dummy_page,
            HAS_ATOMICS=device_capability() >= (7, 0), BLOCK=_COMPACT_BLOCK, num_warps=4,
        )


def compact_marks(rows: int, width: int, device: torch.device) -> torch.Tensor:
    """A zeroed mark scratch for :func:`compact_selected_pages`, grown and cached per device."""
    cache = compact_marks.__dict__.setdefault("_bufs", {})
    buf = cache.get(device)
    if buf is None or buf.numel() < rows * width:
        buf = cache[device] = torch.zeros(max(rows * width, 1 << 16), dtype=torch.int8, device=device)
    return buf


@triton.jit
def _translate_kernel(table, phys_of, out, total, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < total
    pg = tl.load(table + i, mask=mask, other=0)
    ph = tl.load(phys_of + pg, mask=mask, other=0)
    # A -1 entry means the page was not ensured (only possible for a truncated selection,
    # reported via the trunc counter): point it at slot 0 rather than indexing garbage.
    tl.store(out + i, tl.where(ph < 0, 0, ph), mask=mask)


def translate_table(table: torch.Tensor, phys_of: torch.Tensor, out: torch.Tensor) -> None:
    """``out = phys_of[table]`` elementwise over ``out.numel()`` entries (all int32)."""
    n = out.numel()
    if n:
        _translate_kernel[(triton.cdiv(n, 1024),)](table, phys_of, out, n, 1024)


@triton.jit
def _translate_slots_kernel(out_loc, phys_of, out, page_size: tl.constexpr):
    i = tl.program_id(0)
    slot = tl.load(out_loc + i)
    tl.store(out + i, tl.load(phys_of + slot // page_size) * page_size + slot % page_size)


def translate_slots(out_loc: torch.Tensor, phys_of: torch.Tensor, out: torch.Tensor, page_size: int) -> None:
    """Logical token slot -> physical token slot (page translate, offset preserved)."""
    n = out.numel()
    if n:
        _translate_slots_kernel[(n,)](out_loc, phys_of, out, page_size)


@triton.jit
def _mirror_store_kernel(
    k_bytes,      # [T, ROW] uint8 view of this layer's K rows
    v_bytes,      # [T, ROW] uint8 view of this layer's V rows
    out_loc,      # [T] int32 logical token slots
    host_ptrs,    # [2*L] int64: per dense layer the K/V host bank base addresses
    layer,
    row_bytes: tl.constexpr,
    block: tl.constexpr,
):
    t = tl.program_id(0)
    slot = tl.load(out_loc + t).to(tl.int64)
    offs = tl.arange(0, block)
    mask = offs < row_bytes
    kb = tl.load(host_ptrs + 2 * layer).to(tl.pointer_type(tl.uint8))
    vb = tl.load(host_ptrs + 2 * layer + 1).to(tl.pointer_type(tl.uint8))
    tl.store(kb + slot * row_bytes + offs, tl.load(k_bytes + t * row_bytes + offs, mask=mask), mask=mask)
    tl.store(vb + slot * row_bytes + offs, tl.load(v_bytes + t * row_bytes + offs, mask=mask), mask=mask)


def mirror_store(
    k: torch.Tensor,
    v: torch.Tensor,
    out_loc: torch.Tensor,
    host_ptrs: torch.Tensor,
    dense_layer: int,
) -> None:
    """Write-through: this layer's K/V rows of every token in ``out_loc`` into the host mirror."""
    k = k.reshape(k.shape[0], -1)
    v = v.reshape(v.shape[0], -1)
    if not k.is_contiguous():
        k = k.contiguous()
    if not v.is_contiguous():
        v = v.contiguous()
    assert k.shape == v.shape and k.dtype == v.dtype
    t = k.shape[0]
    row_bytes = k.shape[1] * k.element_size()
    if t:
        _mirror_store_kernel[(t,)](
            k.view(torch.uint8), v.view(torch.uint8), out_loc, host_ptrs, dense_layer,
            row_bytes, triton.next_power_of_2(row_bytes),
        )


@triton.jit
def _mirror_store_quant_kernel(
    src_ptrs,     # [4L] int64 GPU: por camada, bases pool (kcode, kscale, vcode, vscale)
    host_ptrs,    # [4L] int64 GPU: por camada, bases host UVA (kcode, kscale, vcode, vscale)
    out_phys,     # [T] int32: slots físicos (pool) de cada token
    out_log,      # [T] int32: slots lógicos (espelho) de cada token
    layer,
    code_bytes: tl.constexpr, scale_bytes: tl.constexpr,
    BLOCK_C: tl.constexpr, BLOCK_S: tl.constexpr,
):
    """fp8 write-through: lê códigos+scales do pool (pós-quantização) e espelha no host."""
    t = tl.program_id(0)
    phys = tl.load(out_phys + t).to(tl.int64)
    logi = tl.load(out_log + t).to(tl.int64)
    oc = tl.arange(0, BLOCK_C)
    mc = oc < code_bytes
    os_ = tl.arange(0, BLOCK_S)
    ms = os_ < scale_bytes
    kc_h = tl.load(host_ptrs + 4 * layer).to(tl.pointer_type(tl.uint8))
    ks_h = tl.load(host_ptrs + 4 * layer + 1).to(tl.pointer_type(tl.uint8))
    vc_h = tl.load(host_ptrs + 4 * layer + 2).to(tl.pointer_type(tl.uint8))
    vs_h = tl.load(host_ptrs + 4 * layer + 3).to(tl.pointer_type(tl.uint8))
    kc_s = tl.load(src_ptrs + 4 * layer).to(tl.pointer_type(tl.uint8))
    ks_s = tl.load(src_ptrs + 4 * layer + 1).to(tl.pointer_type(tl.uint8))
    vc_s = tl.load(src_ptrs + 4 * layer + 2).to(tl.pointer_type(tl.uint8))
    vs_s = tl.load(src_ptrs + 4 * layer + 3).to(tl.pointer_type(tl.uint8))
    tl.store(kc_h + logi * code_bytes + oc, tl.load(kc_s + phys * code_bytes + oc, mask=mc), mask=mc)
    tl.store(vc_h + logi * code_bytes + oc, tl.load(vc_s + phys * code_bytes + oc, mask=mc), mask=mc)
    tl.store(ks_h + logi * scale_bytes + os_, tl.load(ks_s + phys * scale_bytes + os_, mask=ms), mask=ms)
    tl.store(vs_h + logi * scale_bytes + os_, tl.load(vs_s + phys * scale_bytes + os_, mask=ms), mask=ms)


def mirror_store_quant(
    kc_pool: torch.Tensor,   # [P*page, H*D] e4m3 (view)
    ks_pool: torch.Tensor,   # [P*page, H] fp32
    vc_pool: torch.Tensor,
    vs_pool: torch.Tensor,
    out_phys: torch.Tensor,  # [T] int32 slots físicos
    out_log: torch.Tensor,   # [T] int32 slots lógicos
    src_ptrs: torch.Tensor,  # [4L] int64 GPU
    host_ptrs: torch.Tensor, # [4L] int64 GPU
    dense_layer: int,
) -> None:
    """Espelha códigos+scales de uma camada fp8 pro host (graph-safe: UVA stores)."""
    t = out_phys.numel()
    if not t:
        return
    code_bytes = kc_pool.shape[1] * kc_pool.element_size()
    scale_bytes = ks_pool.shape[1] * ks_pool.element_size()
    _mirror_store_quant_kernel[(t,)](
        src_ptrs, host_ptrs, out_phys, out_log, dense_layer,
        code_bytes, scale_bytes,
        triton.next_power_of_2(code_bytes), triton.next_power_of_2(scale_bytes),
    )


__all__ = ["write_pages", "compact_selected_pages", "compact_marks", "translate_table", "translate_slots", "mirror_store", "mirror_store_quant"]
