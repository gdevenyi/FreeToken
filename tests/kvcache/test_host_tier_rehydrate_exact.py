"""The GDN snapshot tier (upstream #525) may only revive a tombstone from a snapshot of
exactly that node's prefix: a shorter snapshot would resume the recurrence with the tokens
in between silently missing."""

from types import SimpleNamespace

import torch

from freetoken.kvcache.host_prefix_tier import HostPrefixTier
from freetoken.scheduler.cache import CacheManager


def _manager(key_len: int = 128):
    state = SimpleNamespace(conv_states=torch.zeros(1, 4, 1, 1),
                            recurrent_states=torch.zeros(1, 4, 1, 1), slot_states={})
    tier = HostPrefixTier(state, gdn_slots_host=4, device="cpu")
    pool = SimpleNamespace(num_free_slots=4, alloc=lambda n: [3])
    manager = SimpleNamespace(
        host_tier=tier,
        prefix_cache=SimpleNamespace(_collect_key=lambda node: torch.arange(key_len)),
        page_size=64,
        linear_state_pool=pool,
        ensure_mamba_slots=lambda n: None,
    )
    return manager, tier, state


def _snapshot(tier, state, key, value):
    state.conv_states[:, 1] = value
    tier.put_prefix(key, None, tier.write_gdn(1))


def test_a_shorter_host_snapshot_neither_revives_nor_refreshes_a_deeper_node():
    manager, tier, state = _manager(key_len=128)
    _snapshot(tier, state, torch.arange(64), 1.0)
    _snapshot(tier, state, torch.arange(1000, 1064), 2.0)
    assert CacheManager._rehydrate_snapshot(manager, node=object()) is None
    # the rejected entry keeps its LRU age
    assert [len(e.key_tokens) for e in tier._index.values()] == [64, 64]
    assert next(iter(tier._index.values())).key_tokens[0] == 0
    assert torch.all(state.conv_states[:, 3] == 0)


def test_an_exact_host_snapshot_revives_the_node():
    manager, tier, state = _manager(key_len=128)
    _snapshot(tier, state, torch.arange(128), 5.0)
    assert CacheManager._rehydrate_snapshot(manager, node=object()) == 3
    assert torch.all(state.conv_states[:, 3] == 5.0)
