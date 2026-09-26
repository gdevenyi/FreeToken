"""The fused multi-bank ``copy_missing`` path must move exactly the same bytes as the
legacy per-bank ``fast_index_copy_jit`` loop, for every miss count (including the
zero-copy case), across banks of differing per-row sizes. The opt-in slim kernel
(``FREETOKEN_MOE_SLIM_COPY``) must move exactly the same bytes as the fused one, also when
replayed from a CUDA graph with a device-side plan.
"""

from __future__ import annotations

import pytest
import torch

from freetoken.moe import offload_cache
from freetoken.moe.offload_cache import _BANK_SCHEMAS, OffloadMoeCache

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

# mxfp4_triton 6-bank schema with mixed 16B-aligned per-row sizes (bytes), >=256 so the
# legacy per-bank kernel's vectorized template is valid. Sizes not covered by a model in
# kernel/aot_models.py must be listed in its TEST_FEATURE_SIZES so the per-bank kernels
# stay prebuilt under FREETOKEN_DISABLE_JIT=1.
FEATS = [8192, 512, 256, 4096, 512, 256]
# qwen4_exp (Qwen3.8-Flash-Next) triton-NVFP4 rows in _BANK_SCHEMAS["nvfp4"] order: gate_up
# packed/scale/global, down packed/scale/global -- 2,772,480 B per expert.
NVFP4_FEATS = [1638400, 204800, 2560, 819200, 102400, 5120]
# 16 * odd, several rows shorter than one grid pass, so segments end mid-round and the
# stride phase wraps across banks and rows.
ODD_FEATS = [16 * 1031, 16 * 7, 16 * 250, 16, 16 * 4099, 16 * 33]


def _build_cache(num_layers, num_experts, cache_size, quant_format="mxfp4_triton", feats=FEATS, pinned=False):
    dev = torch.device("cuda")
    cache = OffloadMoeCache(
        num_layers=num_layers, num_experts=num_experts, cache_size=cache_size,
        device=dev, cache_policy="lru", prefill_overlap=False, quant_format=quant_format,
    )
    schema = _BANK_SCHEMAS[quant_format]
    # Views into one flat tensor: only per-layer addressing matters here, not
    # independent allocations.
    sources = {}
    for name, feat in zip(schema, feats):
        if pinned:
            flat = torch.randint(0, 256, (num_layers * num_experts, feat), dtype=torch.uint8).pin_memory()
        else:
            flat = torch.randint(0, 256, (num_layers * num_experts, feat), dtype=torch.uint8, device=dev)
        sources[name] = list(flat.split(num_experts))
    cache.set_bank_sources(sources)  # also builds the fused-copy descriptor
    return cache


@CUDA
@pytest.mark.slow
@pytest.mark.parametrize("num_indices", [0, 1, 4, 8])
def test_fused_copy_matches_per_bank(num_indices):
    num_layers, num_experts, cache_size = 8, 8, 32
    layer_id = 3  # exercise a non-zero per-layer source selection, not just layer 0
    cache = _build_cache(num_layers, num_experts, cache_size)
    assert cache._copy_fused_ok, "fused copy should activate for 16B-aligned banks"
    # copy_missing resolves the per-layer source through this (normally set by
    # ensure_experts/materialize_layer); poked directly here since this test drives
    # evict_slots/src_indices/num_indices by hand.
    cache._pending_src_layer = layer_id

    cache.num_indices.fill_(num_indices)
    if num_indices:
        dev = torch.device("cuda")
        cache.evict_slots[:num_indices] = torch.arange(num_indices, dtype=torch.int32, device=dev) % cache_size
        # src_indices are layer-local expert rows (0..num_experts) under the new contract.
        cache.src_indices[:num_indices] = torch.arange(num_indices, dtype=torch.int32, device=dev) % num_experts

    # reference: legacy per-bank loop
    for _, c in cache.banks:
        c.zero_()
    cache._copy_fused_ok = False
    cache.copy_missing()
    torch.cuda.synchronize()
    ref = [c.clone() for _, c in cache.banks]

    # fused multi-bank launch
    for _, c in cache.banks:
        c.zero_()
    cache._copy_fused_ok = True
    cache.copy_missing()
    torch.cuda.synchronize()

    for b, (r, (_, c)) in enumerate(zip(ref, cache.banks)):
        assert torch.equal(r, c), f"bank {b} (feat={FEATS[b]}) fused != per-bank at num_indices={num_indices}"


def _host_banks(feats, n_src, seed):
    from freetoken.kernel.pinned import device_ptr

    gen = torch.Generator().manual_seed(seed)
    src = [torch.randint(0, 256, (n_src, f), dtype=torch.uint8, generator=gen).pin_memory() for f in feats]
    dev = torch.device("cuda")
    src_ptrs = torch.tensor([device_ptr(t) for t in src], dtype=torch.int64, device=dev)
    feat_bytes = torch.tensor(feats, dtype=torch.int64, device=dev)
    return src, src_ptrs, feat_bytes


def _slot_banks(feats, n_slots):
    dst = [torch.full((n_slots, f), 0xAB, dtype=torch.uint8, device="cuda") for f in feats]
    return dst, torch.tensor([t.data_ptr() for t in dst], dtype=torch.int64, device="cuda")


def _expected(src, feats, n_slots, di, si, num):
    # untouched slots keep the 0xAB fill; plan rows land at their dst slots
    out = []
    for b, f in enumerate(feats):
        e = torch.full((n_slots, f), 0xAB, dtype=torch.uint8)
        e[di[:num].long().cpu()] = src[b][si[:num].long().cpu()]
        out.append(e.cuda())
    return out


def _random_plan(gen, n_src, n_slots, plan, dtype):
    di = torch.randperm(n_slots, generator=gen)[:plan].to(dtype).cuda()  # distinct: no write races
    si = torch.randint(0, n_src, (plan,), generator=gen).to(dtype).cuda()
    return di, si


def _check_slim_matches_multi(feats, idx_dtype, **grid):
    from freetoken.kernel.fast_index_copy import fast_index_copy_multi_jit, fast_index_copy_multi_slim_jit

    n_src, n_slots, plan = 24, 16, 16
    src, src_ptrs, feat_bytes = _host_banks(feats, n_src, seed=0)
    gen = torch.Generator().manual_seed(1)
    for num in [0, 1, 3, 10, plan, None]:  # None: no num_indices, the whole index length
        di, si = _random_plan(gen, n_src, n_slots, plan, idx_dtype)
        count = None if num is None else torch.tensor([num], dtype=torch.int64, device="cuda")
        ref, ref_ptrs = _slot_banks(feats, n_slots)
        fast_index_copy_multi_jit(ref_ptrs, src_ptrs, feat_bytes, di, si, count)
        out, out_ptrs = _slot_banks(feats, n_slots)
        fast_index_copy_multi_slim_jit(out_ptrs, src_ptrs, feat_bytes, di, si, count, **grid)
        torch.cuda.synchronize()
        exp = _expected(src, feats, n_slots, di, si, plan if num is None else num)
        for b, f in enumerate(feats):
            assert torch.equal(ref[b], exp[b]), f"multi reference wrong: bank {b} (feat={f}) num={num}"
            assert torch.equal(out[b], ref[b]), f"slim != multi: bank {b} (feat={f}) num={num} grid={grid}"


@CUDA
@pytest.mark.parametrize("idx_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("feats", [NVFP4_FEATS, ODD_FEATS], ids=["nvfp4", "odd"])
def test_slim_copy_matches_multi(feats, idx_dtype):
    _check_slim_matches_multi(feats, idx_dtype)


@CUDA
@pytest.mark.slow
@pytest.mark.parametrize("grid", [(3, 128, 2), (16, 256, 2), (5, 64, 8)])
@pytest.mark.parametrize("feats", [NVFP4_FEATS, ODD_FEATS], ids=["nvfp4", "odd"])
def test_slim_copy_matches_multi_other_grids(feats, grid):
    blocks, threads, unroll = grid
    _check_slim_matches_multi(feats, torch.int32, num_blocks=blocks, num_threads=threads, unroll=unroll)


@CUDA
def test_slim_copy_graph_replay_follows_device_plan():
    from freetoken.kernel.fast_index_copy import fast_index_copy_multi_slim_jit

    feats, n_src, n_slots, plan = NVFP4_FEATS, 24, 16, 16
    src, src_ptrs, feat_bytes = _host_banks(feats, n_src, seed=2)
    dst, dst_ptrs = _slot_banks(feats, n_slots)
    di = torch.zeros(plan, dtype=torch.int32, device="cuda")
    si = torch.zeros(plan, dtype=torch.int32, device="cuda")
    count = torch.zeros(1, dtype=torch.int64, device="cuda")
    fast_index_copy_multi_slim_jit(dst_ptrs, src_ptrs, feat_bytes, di, si, count)  # JIT outside capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fast_index_copy_multi_slim_jit(dst_ptrs, src_ptrs, feat_bytes, di, si, count)

    gen = torch.Generator().manual_seed(3)
    for num in [4, 0, plan, 1, 10]:
        new_di, new_si = _random_plan(gen, n_src, n_slots, plan, torch.int32)
        di.copy_(new_di)
        si.copy_(new_si)
        count.fill_(num)
        for d in dst:
            d.fill_(0xAB)
        graph.replay()
        torch.cuda.synchronize()
        exp = _expected(src, feats, n_slots, di, si, num)
        for b, f in enumerate(feats):
            assert torch.equal(dst[b], exp[b]), f"graph replay: bank {b} (feat={f}) num={num}"


def test_slim_copy_flag_defaults_off(monkeypatch):
    import importlib.util

    monkeypatch.delenv("FREETOKEN_MOE_SLIM_COPY", raising=False)
    spec = importlib.util.find_spec("freetoken.moe.offload_cache")
    fresh = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fresh)  # a private copy: the imported module and its classes stay untouched
    assert fresh._SLIM_COPY is False
    monkeypatch.setenv("FREETOKEN_MOE_SLIM_COPY", "1")
    spec.loader.exec_module(fresh)
    assert fresh._SLIM_COPY is True


@CUDA
@pytest.mark.parametrize("slim", [False, True], ids=["multi", "slim"])
def test_copy_missing_slim_switch(monkeypatch, slim):
    from freetoken.kernel import fast_index_copy as fic

    calls = []
    for name in ("fast_index_copy_multi_jit", "fast_index_copy_multi_slim_jit"):
        real = getattr(fic, name)
        monkeypatch.setattr(fic, name, lambda *a, _n=name, _r=real, **k: (calls.append(_n), _r(*a, **k)))
    monkeypatch.setattr(offload_cache, "_SLIM_COPY", slim)

    num_layers, num_experts, cache_size, layer_id, num = 2, 8, 16, 1, 5
    cache = _build_cache(num_layers, num_experts, cache_size, "nvfp4", NVFP4_FEATS, pinned=True)
    assert cache._copy_fused_ok
    cache._pending_src_layer = layer_id
    dev = torch.device("cuda")
    cache.num_indices.fill_(num)
    cache.evict_slots[:num] = torch.tensor([9, 2, 15, 0, 7], dtype=torch.int32, device=dev)
    cache.src_indices[:num] = torch.tensor([3, 3, 0, 7, 5], dtype=torch.int32, device=dev)
    for _, c in cache.banks:
        c.fill_(0xAB)
    cache.copy_missing()
    torch.cuda.synchronize()

    assert calls == ["fast_index_copy_multi_slim_jit" if slim else "fast_index_copy_multi_jit"]
    slots, rows = cache.evict_slots[:num].long().cpu(), cache.src_indices[:num].long().cpu()
    for b, (per_layer, c) in enumerate(cache.banks):
        exp = torch.full(c.shape, 0xAB, dtype=torch.uint8)
        exp[slots] = per_layer[layer_id][rows]
        assert torch.equal(c.cpu(), exp), f"bank {b} (feat={NVFP4_FEATS[b]})"
