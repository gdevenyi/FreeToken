"""The GDN snapshot tier (upstream #525) may only revive a tombstone from a snapshot of
exactly that node's prefix: the host index lookup is longest-prefix, and a shorter
snapshot would resume the recurrence with the tokens in between silently missing."""

from types import SimpleNamespace

import torch

from freetoken.kvcache.host_prefix_tier import HostMatch
from freetoken.scheduler.cache import CacheManager


def _manager(host_len: int, key_len: int = 128):
    reads = []
    tier = SimpleNamespace(
        lookup=lambda key, page_size: HostMatch(host_len, None, 7),
        read_gdn=lambda host_slot, gpu_slot: reads.append((host_slot, gpu_slot)),
    )
    pool = SimpleNamespace(num_free_slots=4, alloc=lambda n: [3])
    manager = SimpleNamespace(
        host_tier=tier,
        prefix_cache=SimpleNamespace(_collect_key=lambda node: torch.arange(key_len)),
        page_size=64,
        linear_state_pool=pool,
        ensure_mamba_slots=lambda n: None,
    )
    return manager, reads


def test_a_shorter_host_snapshot_does_not_revive_a_deeper_node():
    manager, reads = _manager(host_len=64)
    assert CacheManager._rehydrate_snapshot(manager, node=object()) is None
    assert reads == []


def test_an_exact_host_snapshot_revives_the_node():
    manager, reads = _manager(host_len=128)
    assert CacheManager._rehydrate_snapshot(manager, node=object()) == 3
    assert reads == [(7, 3)]
