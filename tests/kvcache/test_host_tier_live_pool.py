"""The GDN snapshot tier (upstream #525) must copy against the pool's CURRENT tensors, which
LinearStatePool.rebuild replaces, and must order its copies after the engine stream."""

import contextlib
from types import SimpleNamespace

import torch

import freetoken.kvcache.host_prefix_tier as tier_module
from freetoken.kvcache.host_prefix_tier import HostPrefixTier


def _pool(slots: int):
    return SimpleNamespace(
        conv_states=torch.zeros(2, slots, 4, 3),
        recurrent_states=torch.zeros(2, slots, 2, 4),
        slot_states={},
    )


def test_read_after_a_pool_rebuild_lands_in_the_new_tensors():
    pool = _pool(3)
    tier = HostPrefixTier(pool, gdn_slots_host=2, device="cpu")
    pool.conv_states[:, 1] = 5.0
    pool.recurrent_states[:, 1] = 7.0
    host = tier.write_gdn(1)
    # a runtime rebuild swaps in fresh (larger) state tensors
    pool.conv_states = torch.zeros(2, 6, 4, 3)
    pool.recurrent_states = torch.zeros(2, 6, 2, 4)
    tier.read_gdn(host, 5)
    assert torch.all(pool.conv_states[:, 5] == 5.0)
    assert torch.all(pool.recurrent_states[:, 5] == 7.0)


def test_copies_wait_for_the_engine_stream(monkeypatch):
    events = []
    engine = object()

    class FakeStream:
        def wait_stream(self, other):
            events.append(("wait", other))

        def synchronize(self):
            events.append(("sync",))

    monkeypatch.setattr(tier_module.torch.cuda, "current_stream", lambda device=None: engine)
    monkeypatch.setattr(tier_module.torch.cuda, "stream", lambda s: contextlib.nullcontext())
    pool = _pool(3)
    tier = HostPrefixTier(pool, gdn_slots_host=2, device="cpu")
    tier.copy_stream = FakeStream()
    host = tier.write_gdn(1)
    tier.read_gdn(host, 2)
    assert events == [("wait", engine), ("sync",), ("wait", engine), ("sync",)]
