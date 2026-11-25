"""QSAKVCache tiers (paged K/V, compressed index slab, pending ring, scratch).

Pins the three things the QSA kernels and the startup budget both depend on: the compressed
slab is a 1/index_ratio shadow of the K/V pages with the scratch rows behind it, the ring and
scratch are fixed (concurrency-sized) and priced apart from the per-token slider, and the K/V
slabs cover the sparse layers only. The PLE conv history rides the GDN slots, so it must
follow every slot operation and show up in the state-pool byte account.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from freetoken.attention import AttnType
from freetoken.kvcache.base import spec_kv_bytes_per_token
from freetoken.kvcache.qsa_pool import QSAKVCache
from freetoken.models.config import KVCacheGroupSpec

DEV = torch.device("cpu")

# Qwen3.8-Flash-Next: 48 layers, every 4th is QSA; 2 kv heads x 256, indexer 128 wide, ratio 4.
FULL_LAYER_IDS = tuple(range(3, 48, 4))
REAL_KV_BYTES = 2 * 256 * 2 * 2 * 12
REAL_INDEX_BYTES = 128 * 12 * 2 // 4


@pytest.fixture(autouse=True)
def _tp(monkeypatch):
    from freetoken.distributed.info import DistributedInfo

    monkeypatch.setattr(
        "freetoken.kvcache.mha_pool.get_tp_info",
        lambda: DistributedInfo(rank=0, size=1),
    )


def _pool(num_pages=4, page_size=64, index_ratio=4, num_req_slots=4, ring_capacity=None):
    return QSAKVCache(
        num_kv_heads=2,
        num_layers=8,
        head_dim=64,
        num_pages=num_pages,
        page_size=page_size,
        dtype=torch.bfloat16,
        device=DEV,
        index_head_dim=32,
        num_index_layers=4,
        index_ratio=index_ratio,
        num_req_slots=num_req_slots,
        ring_capacity=ring_capacity,
        layer_ids=(1, 3, 5, 7),
    )


def _spec(*, index_ratio=4, attn_type=AttnType.QSA, num_kv_heads=2, head_dim=64,
          index_head_dim=32, num_index_layers=4, layer_ids=(1, 3, 5, 7)):
    return KVCacheGroupSpec(
        name="full",
        layer_ids=layer_ids,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        sliding_window=None,
        index_head_dim=index_head_dim,
        num_index_layers=num_index_layers,
        index_ratio=index_ratio,
        attn_type=attn_type,
    )


def _config(spec, *, page_size=64, max_running_req=3):
    mc = SimpleNamespace(num_layers=8, has_swa_attention=False, has_linear_attention=True, model_is_mrope=False)
    mc.kv_cache_group_specs = lambda: (spec,)
    return SimpleNamespace(
        model_config=mc,
        page_size=page_size,
        dtype=torch.bfloat16,
        tp_info=SimpleNamespace(size=1),
        max_running_req=max_running_req,
    )


# --------------------------------------------------------------------- slab / ring geometry


def test_slab_ring_and_scratch_shapes():
    pool = _pool(num_pages=4)
    # 4 pages x 64 tokens / ratio 4 = 64 shadow rows, then one scratch row per request slot
    assert pool.cmp_scratch_base == 64
    assert pool.cmp_k_cache(0).shape == (64 + 4, 32)
    assert pool.cmp_k_cache(3).shape == (64 + 4, 32)
    assert pool.pending_ring(0).shape == (4, QSAKVCache.ring_capacity_for(4), 32)
    assert pool.cmp_k_cache(0).abs().sum().item() == 0.0
    assert pool.k_cache(1).shape == (4, 64, 2, 64)


def test_kv_slabs_cover_sparse_layers_only():
    # Copying the BSA branch (no layer_ids) would back all 8 model layers instead of 4.
    pool = _pool()
    assert pool._kv_buffer.shape[1] == 4
    pool.k_cache(7)
    with pytest.raises(KeyError):
        pool.k_cache(0)


def test_ring_capacity_and_ratio_are_parameters():
    pool = _pool(num_pages=8, index_ratio=2, num_req_slots=3, ring_capacity=6)
    assert pool.index_ratio == 2 and pool.ring_capacity == 6
    assert pool.cmp_scratch_base == 8 * 64 // 2
    assert pool.pending_ring(0).shape == (3, 6, 32)


def test_ring_capacity_formula_and_floor():
    assert QSAKVCache.ring_capacity_for(4) == 4
    assert QSAKVCache.ring_capacity_for(8) == 8
    assert QSAKVCache.ring_capacity_for(4, num_speculative_tokens=3) == 8
    with pytest.raises(ValueError, match="ring_capacity"):
        _pool(ring_capacity=2, index_ratio=4)


def test_group_must_not_straddle_a_page():
    with pytest.raises(ValueError, match="divisible"):
        _pool(page_size=6, index_ratio=4)


def test_index_slab_needs_a_two_byte_dtype():
    with pytest.raises(AssertionError, match="2 bytes"):
        QSAKVCache(
            num_kv_heads=2, num_layers=8, head_dim=64, num_pages=4, page_size=64,
            dtype=torch.float32, device=DEV, index_head_dim=32, num_index_layers=4,
            index_ratio=4, num_req_slots=4, layer_ids=(1, 3, 5, 7),
        )


def test_shadow_row_is_shared_by_a_whole_group():
    pool = _pool()
    cmp = pool.cmp_k_cache(2)
    row = torch.randn(32, dtype=torch.bfloat16)
    for slot in (64, 65, 66, 67):
        assert slot // pool.index_ratio == 16
    cmp[16] = row
    assert torch.equal(pool.cmp_k_cache(2)[16], row)
    # the other sparse layers keep their own rows
    assert pool.cmp_k_cache(1)[16].abs().sum().item() == 0.0


def test_rebuild_resizes_every_tier_and_keeps_identity():
    pool = _pool(num_pages=4)
    ident = id(pool)
    pool.rebuild(16)
    assert id(pool) == ident
    assert pool.k_cache(1).shape == (16, 64, 2, 64)
    assert pool.cmp_scratch_base == 16 * 64 // 4
    assert pool.cmp_k_cache(0).shape == (16 * 64 // 4 + 4, 32)
    assert pool.pending_ring(3).shape == (4, pool.ring_capacity, 32)
    assert pool._kv_buffer.shape[1] == 4  # sparse-layer slabs survive the resize
    pool.k_cache(7)


# ------------------------------------------------------------------------------ budgeting


def test_spec_bytes_per_token_divides_the_index_slab():
    spec = _spec(num_kv_heads=2, head_dim=256, index_head_dim=128, num_index_layers=12,
                 layer_ids=FULL_LAYER_IDS)
    config = _config(spec)
    assert spec_kv_bytes_per_token(spec, config) == REAL_KV_BYTES + REAL_INDEX_BYTES
    assert spec_kv_bytes_per_token(spec, config) == 24576 + 768

    # BSA/DSA keep one index row per token (ratio 1)
    bsa = _spec(num_kv_heads=2, head_dim=256, index_head_dim=128, num_index_layers=12,
                layer_ids=FULL_LAYER_IDS, index_ratio=1, attn_type=AttnType.BSA)
    assert spec_kv_bytes_per_token(bsa, config) == REAL_KV_BYTES + 128 * 12 * 2


def test_kv_cost_prices_ring_and_scratch_as_fixed():
    spec = _spec()
    config = _config(spec, max_running_req=3)
    per_page, fixed, page_tokens, min_reserve = QSAKVCache.kv_cost(config)
    assert per_page == spec_kv_bytes_per_token(spec, config) * 64
    assert page_tokens == 64 and min_reserve == 0
    row = 32 * 4 * 2
    assert fixed == 4 * row * (QSAKVCache.ring_capacity_for(4) + 1)


def test_unit_bytes_matches_the_cost_model():
    spec = _spec()
    config = _config(spec)
    pool = _pool()
    kv_bytes, swa_bytes = pool.unit_bytes()
    assert swa_bytes == 0
    # the scratch rows and the ring must NOT inflate the per-token slider
    assert kv_bytes == spec_kv_bytes_per_token(spec, config)
    assert kv_bytes * 64 == QSAKVCache.kv_cost(config)[0]


def test_resolve_pool_class_and_factory():
    from freetoken.kvcache import create_kvcache_pool, resolve_pool_class

    spec = _spec()
    mc = SimpleNamespace(
        model_is_mrope=False,
        num_layers=8, has_swa_attention=False, has_linear_attention=True,
        num_kv_heads=2, head_dim=64, dsv4_args=None,
    )
    mc.kv_cache_group_specs = lambda: (spec,)
    assert resolve_pool_class(mc) is QSAKVCache

    pool = create_kvcache_pool(
        mc, num_pages=4, page_size=64, dtype=torch.bfloat16, device=DEV, num_req_slots=4
    )
    assert isinstance(pool, QSAKVCache)
    assert pool._kv_buffer.shape[1] == 4  # not the model's 8 layers
    assert pool.cmp_k_cache(0).shape == (4 * 64 // 4 + 4, 32)

    with pytest.raises(ValueError, match="num_req_slots"):
        create_kvcache_pool(mc, num_pages=4, page_size=64, dtype=torch.bfloat16, device=DEV)


# --- P2c cold tier: host copies of whole KV pages (KV_HOST_TIER_DESIGN.md 3.1; P2C_GAP_ANALYSIS.md T1, T2) ---
# The pool keeps a host arena of whole pages, with the same interface as the GDN host bank
# (linear_state_pool: alloc_host / free_host / to_host / from_host / fence_host_copies).
# A page travels with every per-page tensor: the K/V codes, their scales, the page's slab rows
# (slot // index_ratio) and, on mrope models, its rope rows. The pending ring and the scratch
# rows are per request, not per page, so they never travel.


def _tier_pool(num_pages=4, page_size=16, *, fmt=None, mrope=True):
    from freetoken.kvcache.format import FP8_PER_TOKEN_HEAD

    return QSAKVCache(
        num_kv_heads=2, num_layers=8, head_dim=64, num_pages=num_pages, page_size=page_size,
        dtype=torch.bfloat16, device=DEV, index_head_dim=32, num_index_layers=4, index_ratio=4,
        num_req_slots=3, layer_ids=(1, 3, 5, 7), mrope=mrope,
        kv_format=FP8_PER_TOKEN_HEAD if fmt is None else fmt,
    )


def _page_views(pool, page):
    """Every per-page tensor of one page (what a host copy must carry), as views into the pool."""
    ps, r = pool._page_size, pool.index_ratio
    views = [pool._kv_buffer[:, :, page]]
    if pool._scale_buffer is not None:
        views.append(pool._scale_buffer[:, :, page * ps:(page + 1) * ps])
    views.append(pool._cmp_k_buffer[:, page * ps // r:(page + 1) * ps // r])
    if pool._rope_positions is not None:
        views.append(pool._rope_positions[page * ps:(page + 1) * ps])
    return views


def _fill_random(pool, seed=0):
    g = torch.Generator().manual_seed(seed)
    for t in (pool._kv_buffer, pool._scale_buffer, pool._cmp_k_buffer, pool._pending_ring, pool._rope_positions):
        if t is None:
            continue
        if t.element_size() == 1 or not t.is_floating_point():
            raw = t.view(torch.uint8) if t.element_size() == 1 else t
            raw.copy_(torch.randint(0, 120 if t.element_size() == 1 else 1000, raw.shape, generator=g, dtype=raw.dtype))
        else:
            t.copy_(torch.randn(t.shape, generator=g).to(t.dtype))


def _snapshot(pool):
    return {name: (t.clone() if t is not None else None) for name, t in (
        ("kv", pool._kv_buffer), ("scale", pool._scale_buffer), ("slab", pool._cmp_k_buffer),
        ("ring", pool._pending_ring), ("rope", pool._rope_positions))}


@pytest.mark.parametrize("mrope", [True, False])
def test_host_page_round_trip_carries_every_per_page_tensor(mrope):
    """T1: a page copied to the host and restored into ANOTHER page id holds every per-page tensor
    of the original; other pages, the scratch rows and the pending ring do not change."""
    pool = _tier_pool(mrope=mrope)
    pool.attach_host_pages(3)
    assert (pool.num_host_pages, pool.num_free_host_pages) == (3, 3)
    assert pool.host_bytes_per_page() == pool.unit_bytes()[0] * pool._page_size
    _fill_random(pool)
    original = [v.clone() for v in _page_views(pool, 1)]
    before = _snapshot(pool)

    (h,) = pool.alloc_host_pages(1)
    pool.pages_to_host([1], [h])
    pool.fence_host_copies()
    for v in _page_views(pool, 1):
        v.zero_()
    pool.pages_from_host([h], [2])

    for got, want in zip(_page_views(pool, 2), original, strict=True):
        assert torch.equal(got, want)
    for page in (0, 3):  # untouched pages
        for got, want in zip(_page_views(pool, page), [v for v in _page_views_of(before, pool, page)], strict=True):
            assert torch.equal(got, want)
    scratch = pool.cmp_scratch_base
    assert torch.equal(pool._cmp_k_buffer[:, scratch:], before["slab"][:, scratch:])
    assert torch.equal(pool._pending_ring, before["ring"])

    pool.free_host_pages([h])
    assert pool.num_free_host_pages == 3


def _page_views_of(snap, pool, page):
    """_page_views over a _snapshot dict instead of the live pool."""
    ps, r = pool._page_size, pool.index_ratio
    views = [snap["kv"][:, :, page]]
    if snap["scale"] is not None:
        views.append(snap["scale"][:, :, page * ps:(page + 1) * ps])
    views.append(snap["slab"][:, page * ps // r:(page + 1) * ps // r])
    if snap["rope"] is not None:
        views.append(snap["rope"][page * ps:(page + 1) * ps])
    return views


def test_host_pages_copy_in_batches_with_any_mapping():
    """T1: one call moves several pages; host page i receives device page i of the list, in list order."""
    pool = _tier_pool(num_pages=6)
    pool.attach_host_pages(4)
    _fill_random(pool, seed=1)
    want = {p: [v.clone() for v in _page_views(pool, p)] for p in (0, 4, 5)}
    hosts = pool.alloc_host_pages(3)
    pool.pages_to_host([5, 0, 4], hosts)
    pool.fence_host_copies()
    pool.pages_from_host(hosts, [1, 2, 3])  # 5 -> 1, 0 -> 2, 4 -> 3
    for src, dst in ((5, 1), (0, 2), (4, 3)):
        for got, exp in zip(_page_views(pool, dst), want[src], strict=True):
            assert torch.equal(got, exp)


def test_host_page_allocation_limits():
    pool = _tier_pool()
    pool.attach_host_pages(2)
    pool.alloc_host_pages(2)
    with pytest.raises(RuntimeError):
        pool.alloc_host_pages(1)
    pool.reclaim_all_host_pages()
    assert pool.num_free_host_pages == 2
    with pytest.raises(ValueError):
        pool.pages_to_host([0, 1], [0])  # one host page for two device pages


def test_no_host_pages_by_default():
    pool = _tier_pool()
    assert (pool.num_host_pages, pool.num_free_host_pages, pool.host_bytes_per_page()) == (0, 0, 0)


def test_host_pages_follow_a_rebuild():
    """T2: after rebuild the host arena keeps its size, every host page is free again (its contents
    belonged to the old pool), and a round trip works on the new device tensors."""
    pool = _tier_pool(num_pages=4)
    pool.attach_host_pages(2)
    per_page = pool.host_bytes_per_page()
    pool.alloc_host_pages(2)
    pool.rebuild(6)
    assert (pool.num_host_pages, pool.num_free_host_pages, pool.host_bytes_per_page()) == (2, 2, per_page)
    _fill_random(pool, seed=2)
    want = [v.clone() for v in _page_views(pool, 5)]
    (h,) = pool.alloc_host_pages(1)
    pool.pages_to_host([5], [h])
    pool.fence_host_copies()
    pool.pages_from_host([h], [0])
    for got, exp in zip(_page_views(pool, 0), want, strict=True):
        assert torch.equal(got, exp)


def test_coalesce_page_runs_merges_consecutive_pairs():
    from freetoken.kvcache.qsa_pool import _coalesce_page_runs

    assert _coalesce_page_runs([], []) == []
    assert _coalesce_page_runs([4, 5, 6, 10, 11, 2], [0, 1, 2, 8, 9, 7]) == [
        (4, 0, 3),
        (10, 8, 2),
        (2, 7, 1),
    ]


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason="T12 needs a GPU: pinned bank-major arena, batch_memcpy_jit D2H, fast_index_copy_multi_jit H2D",
            ),
        ),
    ],
)
def test_bank_major_host_arena_round_trip_and_rebuild(device):
    """T12: bank-major mmap arena (61 banks on Qwen3.8 geometry), coalesced D2H + multi-bank H2D,
    stream fence before reusing freed device pages, and descriptor refresh across rebuild."""
    from freetoken.kvcache.format import FP8_PER_TOKEN_HEAD

    dev = torch.device(device)
    pool = QSAKVCache(
        num_kv_heads=1,
        num_layers=48,
        head_dim=256,
        num_pages=6,
        page_size=64,
        dtype=torch.bfloat16,
        device=dev,
        index_head_dim=128,
        num_index_layers=12,
        index_ratio=4,
        num_req_slots=4,
        layer_ids=FULL_LAYER_IDS,
        mrope=True,
        kv_format=FP8_PER_TOKEN_HEAD,
    )
    assert len(pool._bank_views()) == 61
    pool.attach_host_pages(4)
    assert pool.host_bytes_per_page() == 449_280
    if device == "cuda":
        assert pool._host_arena.pinned and pool._host_arena.multi_h2d_ok
    _fill_random(pool, seed=7)
    want = {p: [v.clone() for v in _page_views(pool, p)] for p in (1, 2, 4)}
    hosts = pool.alloc_host_pages(3)
    pool.pages_to_host([1, 2, 4], hosts)
    pool.fence_host_copies()
    for p in (1, 2, 4):
        for v in _page_views(pool, p):
            v.zero_()
    pool.pages_from_host(hosts, [3, 5, 0])
    for src, dst in ((1, 3), (2, 5), (4, 0)):
        for got, exp in zip(_page_views(pool, dst), want[src], strict=True):
            assert torch.equal(got, exp)
    pool.rebuild(8)
    _fill_random(pool, seed=9)
    want7 = [v.clone() for v in _page_views(pool, 7)]
    (h0,) = pool.alloc_host_pages(1)
    pool.pages_to_host([7], [h0])
    pool.fence_host_copies()
    pool.pages_from_host([h0], [2])
    for got, exp in zip(_page_views(pool, 2), want7, strict=True):
        assert torch.equal(got, exp)


def test_hybrid_radix_eviction_is_lockstep_under_skewed_wall_clock(monkeypatch):
    """TP=2 ranks with skewed/coarse wall clocks must evict identical mamba and KV victims."""
    import time as time_mod
    from freetoken.kvcache.hybrid_radix_cache import HybridRadixCache

    ps = 64
    rank0 = HybridRadixCache(DEV, ps)
    rank1 = HybridRadixCache(DEV, ps)

    # Freeze time.monotonic_ns for rank0 while jittering it backwards/forwards for rank1.
    monkeypatch.setattr(time_mod, "monotonic_ns", lambda: 100)
    for idx in range(6):
        keys = torch.arange(idx * 1000, idx * 1000 + 2 * ps, dtype=torch.int32)
        vals = torch.arange(idx * 2 * ps, (idx + 1) * 2 * ps, dtype=torch.int32)
        rank0.insert(keys, vals, idx + 1)
    rank0.match_prefix(torch.arange(2000, 2000 + 2 * ps, dtype=torch.int32))
    rank0.match_prefix(torch.arange(0, 2 * ps, dtype=torch.int32))

    ticks = iter([999999, 500, 500, 12, 88888, 12, 77, 1])
    monkeypatch.setattr(time_mod, "monotonic_ns", lambda: next(ticks, 42))
    for idx in range(6):
        keys = torch.arange(idx * 1000, idx * 1000 + 2 * ps, dtype=torch.int32)
        vals = torch.arange(idx * 2 * ps, (idx + 1) * 2 * ps, dtype=torch.int32)
        rank1.insert(keys, vals, idx + 1)
    rank1.match_prefix(torch.arange(2000, 2000 + 2 * ps, dtype=torch.int32))
    rank1.match_prefix(torch.arange(0, 2 * ps, dtype=torch.int32))

    er0 = rank0.evict_mamba(3)
    er1 = rank1.evict_mamba(3)
    assert er0.mamba_slots == er1.mamba_slots
    assert torch.equal(er0.kv_indices, er1.kv_indices)

    ef0 = rank0.evict_full(2 * ps)
    ef1 = rank1.evict_full(2 * ps)
    assert ef0.mamba_slots == ef1.mamba_slots
    assert torch.equal(ef0.kv_indices, ef1.kv_indices)
