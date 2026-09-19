"""The host KV tiers (upstream #499 / #525) mirror codes plus the fp8 pool's per-token scales;
an nvfp4 pool keeps block scales they do not copy, so both must refuse it up front."""

from types import SimpleNamespace

import pytest
import torch

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
