"""The host KV tiers (upstream #499 / #525) mirror codes plus the fp8 pool's per-token scales;
an nvfp4 pool keeps block scales they do not copy, so both must refuse it up front. The
prefix tier likewise refuses pools whose layout or index slab it does not copy."""

from types import SimpleNamespace

import pytest
import torch

from freetoken.kvcache.dsa_pool import KpoolDSAKVCache, MLAKVCache
from freetoken.kvcache.host_prefix_tier import HostPrefixTier
from freetoken.kvcache.kv_host_offload import KVHostOffloader


def _kv_pool(block_scaled: bool):
    return SimpleNamespace(
        _kv_buffer=torch.zeros(2, 1, 3, 4, 1, 8, dtype=torch.uint8),  # [2, L, P+1, page, H, D]
        _scale_buffer=None,
        _block_scale_buffer=torch.zeros(1) if block_scaled else None,
    )


def _state_pool():
    return SimpleNamespace(
        conv_states=torch.zeros(2, 2, 4, 3),
        recurrent_states=torch.zeros(2, 2, 2, 4),
        slot_states={},
    )


def test_kv_host_offloader_refuses_an_nvfp4_pool():
    with pytest.raises(ValueError, match="--kv-host-pages .* nvfp4"):
        KVHostOffloader(_kv_pool(block_scaled=True), num_logical_pages=4, device=torch.device("cpu"))


def test_prefix_tier_refuses_kv_pages_on_an_nvfp4_pool():
    with pytest.raises(ValueError, match="FT_PREFIX_HOST .* nvfp4"):
        HostPrefixTier(_state_pool(), kv_pool=_kv_pool(block_scaled=True), gdn_slots_host=2,
                       kv_budget_bytes=1 << 16, device="cpu")


def test_prefix_tier_gdn_snapshots_do_not_touch_the_kv_pool():
    # the snapshot-only tier (no KV budget) never reads the pool, nvfp4 or not
    tier = HostPrefixTier(_state_pool(), kv_pool=_kv_pool(block_scaled=True), gdn_slots_host=2,
                          kv_budget_bytes=0, device="cpu")
    assert tier.num_slots == 2 and not tier.has_kv


def test_prefix_tier_refuses_kv_pages_on_a_qsa_pool():
    # the KV-page store mirrors codes and scales, not the QSA indexer's compressed-key slab
    pool = SimpleNamespace(**vars(_kv_pool(block_scaled=False)), _cmp_k_buffer=torch.zeros(1))
    with pytest.raises(ValueError, match="FT_PREFIX_HOST .* index slab"):
        HostPrefixTier(_state_pool(), kv_pool=pool, gdn_slots_host=2, kv_budget_bytes=1 << 16, device="cpu")


@pytest.mark.parametrize("kv_quant", ["none", "fp8"])
def test_prefix_tier_refuses_kv_pages_on_an_mla_pool(kv_quant):
    # one latent per token (and 2-D fp8 scales), not the [2, L, P, page, H, D] pages it copies
    pool = MLAKVCache(16, 2, 6, 4, torch.bfloat16, torch.device("cpu"), kv_quant=kv_quant)
    with pytest.raises(ValueError, match="FT_PREFIX_HOST .* layout"):
        HostPrefixTier(_state_pool(), kv_pool=pool, gdn_slots_host=2, kv_budget_bytes=1 << 16, device="cpu")


@pytest.mark.parametrize("kv_quant", ["none", "fp8"])
def test_prefix_tier_refuses_kv_pages_on_a_kpool_dsa_pool(kv_quant):
    # GLM-5.3-Flash: a hybrid linear model whose DSA indexer slab is keyed by the same pages
    pool = KpoolDSAKVCache(latent_dim=16, num_layers=2, num_pages=6, page_size=4,
                           dtype=torch.bfloat16, device=torch.device("cpu"), index_head_dim=8,
                           num_index_layers=2, num_req_slots=2, index_ratio=4, kv_quant=kv_quant)
    with pytest.raises(ValueError, match="FT_PREFIX_HOST .* index slab"):
        HostPrefixTier(_state_pool(), kv_pool=pool, gdn_slots_host=2, kv_budget_bytes=1 << 16, device="cpu")


def test_prefix_tier_refuses_kv_pages_on_an_index_slab_pool():
    # BSA keeps the paged K/V layout plus an index slab it does not mirror
    pool = SimpleNamespace(**vars(_kv_pool(block_scaled=False)), _index_k_buffer=torch.zeros(1))
    with pytest.raises(ValueError, match="FT_PREFIX_HOST .* index slab"):
        HostPrefixTier(_state_pool(), kv_pool=pool, gdn_slots_host=2, kv_budget_bytes=1 << 16, device="cpu")


@pytest.mark.parametrize("kv_quant", ["none", "fp8"])
def test_prefix_tier_round_trips_kv_pages_of_an_mha_pool(kv_quant, monkeypatch):
    from freetoken.distributed.info import DistributedInfo
    from freetoken.kvcache.mha_pool import MHAKVCache

    monkeypatch.setattr("freetoken.kvcache.mha_pool.get_tp_info", lambda: DistributedInfo(rank=0, size=1))
    pool = MHAKVCache(num_kv_heads=2, num_layers=2, head_dim=16, num_pages=6, page_size=4,
                      dtype=torch.bfloat16, device=torch.device("cpu"), kv_quant=kv_quant)
    pool._kv_buffer.view(torch.uint8)[:, :, 1] = 7
    tier = HostPrefixTier(_state_pool(), kv_pool=pool, gdn_slots_host=2, kv_budget_bytes=1 << 16, device="cpu")
    tier.read_kv(tier.write_kv(torch.arange(4, 8, dtype=torch.int32)), torch.tensor([3]))
    assert torch.equal(pool._kv_buffer[:, :, 3], pool._kv_buffer[:, :, 1])
