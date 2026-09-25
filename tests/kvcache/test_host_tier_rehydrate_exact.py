"""The GDN snapshot tier (upstream #525) may only revive a tombstone from a snapshot of
exactly that node's prefix: a shorter snapshot would resume the recurrence with the tokens
in between silently missing. Making room for the revival evicts other prefixes into the
tier, which may recycle the entry's own host slots, so it must not read stale ones."""

from types import SimpleNamespace

import torch

import freetoken.core as core
from freetoken.kvcache.host_prefix_tier import HostPrefixTier
from freetoken.kvcache.linear_state_pool import LinearStatePool
from freetoken.models.config import LinearGatedDeltaGroupConfig
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


def _cache_manager(monkeypatch, num_pages, page_size, state_slots, **env):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    g = LinearGatedDeltaGroupConfig(
        name="linear", layer_ids=(0,), num_key_heads=2, num_value_heads=4,
        key_head_dim=16, value_head_dim=16, conv_kernel_dim=4, output_gate="silu",
    )
    pool = LinearStatePool(group=g, num_slots=state_slots, dtype=torch.bfloat16,
                           device=torch.device("cpu"), tp_size=1)
    page_table = torch.zeros(2, 64, dtype=torch.int32)
    cm = CacheManager(num_pages, page_size, page_table, "hybrid_radix", linear_state_pool=pool)
    return cm, pool


def _put(cm, pool, ids, state, stamp, kv_pool=None, kv_value=None):
    kv = cm._page_to_token(cm._allocate(len(ids) // cm.page_size))
    if kv_pool is not None:
        kv_pool._kv_buffer[:, :, kv[:: cm.page_size] // cm.page_size] = kv_value
    slot = pool.alloc(1)[0]
    pool.conv_states[:, slot] = state
    cm.prefix_cache.insert(ids, kv, slot)
    cm.prefix_cache.match_prefix(ids).node.timestamp = stamp


def _match(cm, ids):
    tail = torch.tensor([7, 9], dtype=torch.int32)
    return cm.match_req(SimpleNamespace(input_ids=torch.cat([ids, tail]), input_len=len(ids) + 2))


def test_making_room_to_revive_does_not_restore_the_victim_into_the_host_slot(monkeypatch):
    # one host slot: evicting B for the revived slot drops A's entry and writes B there
    cm, pool = _cache_manager(monkeypatch, 8, 64, 3, FT_GDN_HOST_TIER="1", FT_GDN_HOST_SLOTS="1")
    a = torch.arange(128, dtype=torch.int32)
    b = torch.arange(1000, 1128, dtype=torch.int32)
    _put(cm, pool, a, 1.0, stamp=1)
    _put(cm, pool, b, 2.0, stamp=2)
    cm.ensure_mamba_slots(1)  # A -> host tier, A's node becomes a tombstone
    pool.conv_states[:, pool.alloc(1)[0]] = 9.0  # someone else takes the freed slot

    mr = _match(cm, a)
    restored = None if mr.mamba_value is None else pool.conv_states[:, mr.mamba_value].unique().tolist()
    assert restored in (None, [1.0]), (mr.cuda_handle.cached_len, restored)
    assert cm.host_tier.lookup_exact(b) is not None


def test_making_room_to_rehydrate_a_host_prefix_does_not_restore_the_victim(monkeypatch):
    # a 4-page KV store: writing the 4-page victim V drops the 2-page hit H and reuses its slots
    page, n_pages = 4, 6
    kv_pool = SimpleNamespace(_kv_buffer=torch.zeros(2, 1, n_pages, page, 1, 2),
                              _scale_buffer=None, _block_scale_buffer=None)
    monkeypatch.setattr(core, "_GLOBAL_CTX", SimpleNamespace(kv_cache=kv_pool, kv_offloader=None))
    page_bytes = kv_pool._kv_buffer[:, :, 0].numel() * kv_pool._kv_buffer.element_size()
    cm, pool = _cache_manager(monkeypatch, n_pages, page, 4, FT_PREFIX_HOST="1",
                              FT_PREFIX_HOST_GB=repr(4 * page_bytes / (1 << 30)),
                              FT_GDN_HOST_SLOTS="4")
    h = torch.arange(2 * page, dtype=torch.int32)
    v = torch.arange(100, 100 + 4 * page, dtype=torch.int32)
    _put(cm, pool, h, 1.0, stamp=1, kv_pool=kv_pool, kv_value=1.0)
    er = cm.prefix_cache.evict_full(len(h))  # H -> host tier (KV + snapshot), out of the tree
    pool.free(er.mamba_slots)
    cm._free(er.kv_indices)
    _put(cm, pool, v, 2.0, stamp=2, kv_pool=kv_pool, kv_value=2.0)
    cm._allocate(len(cm.free_slots))  # a running request holds the rest of the GPU pages

    mr = _match(cm, h)
    handle = mr.cuda_handle
    if handle.cached_len:
        pages = handle.kv_indices[::page] // page
        assert pool.conv_states[:, mr.mamba_value].unique().tolist() == [1.0]
        assert kv_pool._kv_buffer[:, :, pages].unique().tolist() == [1.0]
    assert cm.host_tier.lookup_exact(v) is not None
