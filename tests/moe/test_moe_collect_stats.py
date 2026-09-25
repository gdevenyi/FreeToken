"""--moe-collect-stats: the flag, and the report it produces.

The counters themselves are accumulated device-side inside ``ensure_experts`` and were
already covered; what was missing until this flag existed was any way to turn them on from
the command line or read them back. These tests cover that wiring -- the flag reaching
``ServerArgs``, and the emit formatting the numbers and resetting the window afterwards.
"""

import contextlib
import io
from types import SimpleNamespace

import pytest
import torch

from freetoken.engine.engine import MOE_STATS_INTERVAL, Engine
from freetoken.moe.offload_cache import OffloadMoeCache
from freetoken.server.args import ServerArgs, parse_args


def test_flag_is_registered_and_defaults_off():
    """``--help`` short-circuits before the model is resolved, so this needs no checkpoint."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.suppress(SystemExit):
        parse_args(["--help"])
    assert "--moe-collect-stats" in buf.getvalue()
    # Off unless asked for: the counters ride in the decode CUDA graph and cost throughput.
    assert ServerArgs.moe_collect_stats is False


class _StubCache:
    """Just enough cache to exercise the emit: the four readers plus the window reset."""

    def __init__(self, decode_target="gpu", layer_calls=512):
        self.collect_stats = True
        self.decode_target = decode_target
        self._layer_calls = layer_calls
        self.reset_calls = 0

    def decode_miss_stats(self):
        return {
            "layer_calls": self._layer_calls,
            "active_per_layer": 8.0,
            "missing_per_layer": 2.0,
            "miss_rate": 0.25,
            "fetched_per_layer": 1.5,
            "cpu_per_layer": 0.5,
            "fetch_rate": 0.75,
            "prefill_hit_rows": 0,
            "prefill_rows": 0,
        }

    def decode_miss_stats_per_layer(self):
        return {
            "per_layer": [
                {"layer": 0, "steps": 4, "miss_rate": 0.5},
                {"layer": 1, "steps": 4, "miss_rate": 0.1},
                # steps == 0 means the layer never ran in this window; it must not be
                # ranked as a 0.0-miss-rate "best" layer.
                {"layer": 2, "steps": 0, "miss_rate": 0.0},
            ]
        }

    def decode_routing_stats(self):
        return {
            "slots_per_layer": 56.7,
            "working_set_mean": 173.1,
            "working_set_max": 243,
            "experts_for_90pct": 92.3,
            "static_topk_hit_at_slots": 0.764,
            "norm_entropy": 0.813,
        }

    def reset_stats(self):
        self.reset_calls += 1


def _emit(cache, caplog):
    engine = SimpleNamespace(moe_offload_cache=cache, _emit_moe_stats=None)
    with caplog.at_level("INFO"):
        Engine._emit_moe_stats(engine)
    return "\n".join(r.getMessage() for r in caplog.records)


def test_emit_reports_and_resets_the_window(caplog):
    cache = _StubCache()
    out = _emit(cache, caplog)
    assert "miss_rate=0.250" in out
    # Labelled as the fixed-set figure it is; "oracle" implied an upper bound it is not.
    assert "static_topk_hit=0.764" in out
    assert "oracle" not in out
    assert "(realized 0.750)" in out
    # Ranked worst-first, and the layer that never ran is left out entirely.
    assert "L0=0.500, L1=0.100" in out
    assert "L2" not in out
    assert cache.reset_calls == 1


def test_hybrid_split_only_reported_for_hybrid(caplog):
    assert "fetch_rate" not in _emit(_StubCache(decode_target="gpu"), caplog)
    caplog.clear()
    assert "fetch_rate=0.750" in _emit(_StubCache(decode_target="hybrid"), caplog)


def test_idle_window_emits_nothing_and_keeps_counters(caplog):
    """No decode ran, so there is nothing to report -- and nothing to reset either."""
    cache = _StubCache(layer_calls=0)
    assert _emit(cache, caplog) == ""
    assert cache.reset_calls == 0


def test_interval_is_a_sane_window():
    assert MOE_STATS_INTERVAL >= 1


def test_static_topk_hit_is_not_an_upper_bound_on_lru():
    """One slot, four experts each routed in a run of four tokens: A A A A B B B B ...

    The best fixed expert catches 4 of 16 tokens; an LRU holding one slot misses only on
    each run's first token. The figure is routing skew, not a ceiling on a dynamic cache.
    """
    trace = [e for e in range(4) for _ in range(4)]
    cached, lru_hits = None, 0
    for e in trace:
        lru_hits += e == cached
        cached = e
    freq = torch.bincount(torch.tensor(trace), minlength=4).unsqueeze(0)
    cache = SimpleNamespace(decode_freq=freq, cache_size=1, num_layers=1, num_experts=4)
    stats = OffloadMoeCache.decode_routing_stats(cache)
    assert stats["static_topk_hit_at_slots"] == 0.25
    assert "oracle_hit_at_slots" not in stats
    assert lru_hits / len(trace) == 0.75


def _reference_lru_ensure(query, slot_of_id, id_of_slot, lru_usage, lru_step, out_indices,
                          src_indices, dst_indices, num_copy, stats=None, id_base=0):
    """Python stand-in for flashlib's lru_ensure: same maps, plan and in-place rewrite."""
    assert stats is None, "below sm_70 the stats atomic must stay out of the kernel"
    lru_step += 1
    step = int(lru_step)
    ids = [int(q) + id_base for q in query.tolist()]
    missing = []
    for g in ids:
        s = int(slot_of_id[g])
        if s >= 0:
            lru_usage[s] = step
        elif g not in missing:
            missing.append(g)
    # misses take victims in ascending id order, victims in ascending (usage, slot) order
    for i, g in enumerate(sorted(missing)):
        free = [c for c in range(id_of_slot.numel()) if int(lru_usage[c]) != step]
        victim = min(free, key=lambda c: (int(lru_usage[c]), c))
        old = int(id_of_slot[victim])
        if old >= 0:
            slot_of_id[old] = -1
        id_of_slot[victim], slot_of_id[g], lru_usage[victim] = g, victim, step
        src_indices[i], dst_indices[i] = g - id_base, victim
    num_copy[0] = len(missing)
    out_indices.copy_(torch.tensor([int(slot_of_id[g]) for g in ids], dtype=out_indices.dtype))


def _stats_trace(num_experts):
    g = torch.Generator().manual_seed(0)
    trace = []
    for step in range(12):
        for layer in (0, 1):
            ids = torch.randint(0, num_experts, (6,), generator=g, dtype=torch.int32)
            ids[1] = ids[0]  # duplicates collapse to one active expert and one copy
            trace.append((layer, ids))
    return trace


def test_pre_sm70_stats_match_the_kernel_counters(monkeypatch):
    """On Pascal is_sm70_supported() is False, so ensure_experts keeps the stats itself.
    ACTIVE must be counted from the raw ids, before the in-place slot rewrite."""
    from freetoken.moe import offload_kernels

    monkeypatch.setattr(offload_kernels, "is_sm70_supported", lambda: False, raising=False)
    monkeypatch.setattr(offload_kernels, "lru_ensure", _reference_lru_ensure)
    E = 8
    cache = OffloadMoeCache(num_layers=2, num_experts=E, cache_size=12,
                            device=torch.device("cpu"), quant_format="bf16")
    cache.collect_stats = True
    expected = torch.zeros_like(cache.lru_stats)
    for layer, ids in _stats_trace(E):
        raw = ids.tolist()
        cache.ensure_experts(layer, ids)
        assert ids.tolist() != raw or len(set(raw)) == 1  # rewritten to slot ids in place
        expected[layer] += torch.tensor([len(set(raw)), int(cache.num_indices[0]), 1])
    assert torch.equal(cache.lru_stats, expected)
    assert cache.decode_miss_stats()["layer_calls"] == 24


def test_distinct_count_matches_unique():
    from freetoken.moe.offload_kernels import _distinct_count

    g = torch.Generator().manual_seed(1)
    for n in (0, 1, 2, 7, 40):
        ids = torch.randint(0, 6, (n,), generator=g, dtype=torch.int32)
        assert int(_distinct_count(ids)) == torch.unique(ids).numel()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_pre_sm70_stats_path_on_the_gpu(monkeypatch):
    """The torch stats path against the in-kernel counters; on a pre-sm_70 GPU (where the
    kernel counters cannot compile) against the Python reference instead."""
    from freetoken.moe import offload_kernels

    E, dev = 8, torch.device("cuda")
    kernel_ok = offload_kernels.is_sm70_supported()

    def run(torch_stats: bool, ensure=None):
        monkeypatch.setattr(offload_kernels, "is_sm70_supported", lambda: not torch_stats)
        if ensure is not None:
            monkeypatch.setattr(offload_kernels, "lru_ensure", ensure)
        cache = OffloadMoeCache(num_layers=2, num_experts=E, cache_size=12, device=dev,
                                quant_format="bf16")
        cache.collect_stats = True
        for layer, ids in _stats_trace(E):
            cache.ensure_experts(layer, ids.to(dev))
        torch.cuda.synchronize()
        return cache.lru_stats.cpu()

    got = run(torch_stats=True)
    if kernel_ok:
        assert torch.equal(got, run(torch_stats=False))
    else:
        cpu = OffloadMoeCache(num_layers=2, num_experts=E, cache_size=12,
                              device=torch.device("cpu"), quant_format="bf16")
        cpu.collect_stats = True
        monkeypatch.setattr(offload_kernels, "lru_ensure", _reference_lru_ensure)
        for layer, ids in _stats_trace(E):
            cpu.ensure_experts(layer, ids.clone())
        assert torch.equal(got, cpu.lru_stats)
