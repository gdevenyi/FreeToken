"""FREETOKEN_MOE_PREFETCH stage A: the router-lookahead select / count kernels against a Python
reference, the mode switch, and the predictor stream's lifetime across resets and rebuilds.
The qwen4_exp stack tests (bit-identical outputs, counters over a real decode) live in
tests/models/qwen4_exp/test_skeleton.py."""

from __future__ import annotations

import pytest
import torch

from .ref_prefetch import ref_count, ref_select

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _padded(ids, width):
    return ids + [-1] * (width - len(ids))


def _select(logits, resident, k, budget, width=16):
    from freetoken.moe.prefetch import lookahead_select

    sel = torch.full((width,), -7, dtype=torch.int32, device="cuda")
    res = torch.full((width,), -7, dtype=torch.int32, device="cuda")
    lookahead_select(
        torch.tensor(logits, dtype=torch.bfloat16, device="cuda"),
        torch.tensor(resident, dtype=torch.int32, device="cuda"),
        sel, res, k=k, budget=budget,
    )
    return sel.tolist(), res.tolist()


def test_mode_switch():
    from freetoken.moe.prefetch import default_budget, resolve_mode

    assert resolve_mode("off") == "off" and resolve_mode(" Measure ") == "measure" and resolve_mode("ON") == "on"
    with pytest.raises(ValueError):
        resolve_mode("lookahead")
    assert (default_budget(True), default_budget(False)) == (3, 4)


@requires_cuda
def test_cache_builds_the_prefetcher_only_when_asked():
    from freetoken.moe.offload_cache import OffloadMoeCache

    def cache(**kw):
        return OffloadMoeCache(num_layers=2, num_experts=4, cache_size=4, device=torch.device("cuda"), **kw)

    assert cache(prefetch_mode="off").prefetch is None
    assert cache(prefetch_mode="measure").prefetch.copy_stream is None
    on = cache(prefetch_mode="on")
    assert on.prefetch.mode == "on" and on.prefetch.cache is on and on.prefetch_on
    for mode in ("measure", "on"):
        with pytest.raises(ValueError):
            cache(prefetch_mode=mode, decode_target="hybrid")


@requires_cuda
def test_select_known_answers():
    # order by logit: 1, 4, 6, 2, 3, 7, 0, 5; experts 4 and 2 are resident
    logits = [[0.1, 0.9, 0.5, 0.3, 0.8, 0.0, 0.7, 0.2]]
    resident = [-1, -1, 7, -1, 3, -1, -1, -1]
    assert _select(logits, resident, 5, 2) == (_padded([1, 6], 16), _padded([4], 16))
    assert _select(logits, resident, 5, 4) == (_padded([1, 6, 3], 16), _padded([4, 2], 16))
    # the top-k cap: 3 and 7 are out of a top-4 list
    assert _select(logits, resident, 4, 4) == (_padded([1, 6], 16), _padded([4, 2], 16))
    # all-equal logits rank by id
    assert _select([[0.5] * 8], [-1] * 8, 8, 3) == (_padded([0, 1, 2], 16), _padded([], 16))
    # bs 2: row 0 ranks 1, 4, 6 and row 1 ranks 4, 0, 5 -> merged 1, 4, 0, 6, 5; expert 0 resident
    rows = [[0.0, 0.9, 0.1, 0.1, 0.8, 0.1, 0.7, 0.1], [0.8, 0.1, 0.1, 0.1, 0.9, 0.7, 0.1, 0.1]]
    resident = [2, -1, -1, -1, -1, -1, -1, -1]
    assert _select(rows, resident, 3, 3) == (_padded([1, 4, 6], 16), _padded([0], 16))
    assert _select(rows, resident, 3, 16) == (_padded([1, 4, 6, 5], 16), _padded([0], 16))


@requires_cuda
@pytest.mark.parametrize("rows", [1, 2])
@pytest.mark.parametrize("k,budget", [(3, 3), (16, 4), (16, 3), (8, 16)])
def test_select_matches_reference(rows, k, budget):
    gen = torch.Generator().manual_seed(rows * 100 + k + budget)
    for trial in range(20):
        # coarse values make bf16 ties common, so the tie rule is exercised
        logits = (torch.randint(-24, 24, (rows, 512), generator=gen).float() / 8).tolist()
        resident = torch.where(torch.rand(512, generator=gen) < 0.4, 1, -1).tolist()
        want = ref_select(logits, resident, k, budget)
        width = 2 * k
        got = _select(logits, resident, k, budget, width=width)
        assert got == (_padded(want[0], width), _padded(want[1], width)), trial


@requires_cuda
def test_count_known_answers():
    from freetoken.moe.prefetch import prefetch_count

    num_experts, layer = 8, 2
    base = layer * num_experts
    # slots 0, 1, 3 hold this layer's experts 3, 1, 6; slot 4 another layer's expert
    id_of_slot = torch.tensor([base + 3, base + 1, -1, base + 6, 5, -1], dtype=torch.int32, device="cuda")
    slots = torch.tensor([0, 1, 3], dtype=torch.int32, device="cuda")
    sel = torch.tensor(_padded([3, 5], 6), dtype=torch.int32, device="cuda")
    res = torch.tensor(_padded([1, 4], 6), dtype=torch.int32, device="cuda")
    misses = torch.tensor([2], dtype=torch.int64, device="cuda")
    counters = torch.zeros(6, dtype=torch.int64, device="cuda")
    for _ in range(2):
        prefetch_count(slots, id_of_slot, misses, sel, res, counters, id_base=base, rows=1)
    want = ref_count([3, 5], [1, 4], [3, 1, 6], 2, 1)
    assert counters.tolist() == [2 * v for v in want] == [4, 2, 2, 2, 2, 4]
    # bs 2 with a duplicate routed expert counts it once
    counters.zero_()
    slots2 = torch.tensor([0, 1, 0, 3], dtype=torch.int32, device="cuda")
    prefetch_count(slots2, id_of_slot, misses, sel, res, counters, id_base=base, rows=2)
    assert counters.tolist() == ref_count([3, 5], [1, 4], [3, 1, 3, 6], 2, 2)


@requires_cuda
def test_prefetch_stream_never_aliases_a_pooled_stream():
    from freetoken.moe.offload_cache import OffloadMoeCache

    cache = OffloadMoeCache(
        num_layers=2, num_experts=4, cache_size=4, device=torch.device("cuda"),
        decode_copy_overlap=True, prefetch_mode="measure",
    )
    pred = cache.prefetch.stream.cuda_stream
    pooled = {torch.cuda.Stream().cuda_stream for _ in range(64)}
    assert pred not in pooled and pred != torch.cuda.current_stream().cuda_stream
    assert pred != cache.decode_copy_stream.cuda_stream


@requires_cuda
def test_prefetch_copy_stream_is_a_third_dedicated_stream():
    from freetoken.moe.offload_cache import OffloadMoeCache

    cache = OffloadMoeCache(
        num_layers=2, num_experts=4, cache_size=4, device=torch.device("cuda"),
        decode_copy_overlap=True, prefetch_mode="on",
    )
    pf = cache.prefetch
    copy = pf.copy_stream.cuda_stream
    pooled = {torch.cuda.Stream().cuda_stream for _ in range(64)}
    assert copy not in pooled and pf.stream.cuda_stream not in pooled
    assert len({copy, pf.stream.cuda_stream, cache.decode_copy_stream.cuda_stream, torch.cuda.current_stream().cuda_stream}) == 4


@requires_cuda
def test_rebuild_and_reset_clear_the_counters():
    from freetoken.moe.offload_cache import OffloadMoeCache

    cache = OffloadMoeCache(
        num_layers=2, num_experts=4, cache_size=4, device=torch.device("cuda"), prefetch_mode="measure"
    )
    cache.set_bank_sources({
        "gate_up": [torch.zeros(4, 32, 8, dtype=torch.bfloat16).pin_memory() for _ in range(2)],
        "down": [torch.zeros(4, 8, 16, dtype=torch.bfloat16).pin_memory() for _ in range(2)],
    })
    pf = cache.prefetch
    stream, sel = pf.stream, pf.sel
    pf.counters.fill_(3)
    cache.reset()
    assert int(pf.counters.abs().sum()) == 0
    pf.counters.fill_(5)
    pf.take_window()
    assert int(pf.totals.sum()) > 0
    pf.counters.fill_(2)
    pf._inflight[1] = torch.zeros(1)
    cache.rebuild(6)
    assert cache.prefetch is pf and pf.stream is stream and pf.sel is sel
    assert int(pf.counters.abs().sum()) == 0 and int(pf.totals.abs().sum()) == 0
    assert pf._inflight == [None, None]


def _emit_prefetch(window, monkeypatch, *, debug=False, mode="measure"):
    from types import SimpleNamespace

    from freetoken.engine import engine as engine_mod
    from freetoken.env import ENV

    taken, lines = [], []
    prefetch = SimpleNamespace(mode=mode, take_window=lambda: taken.append(1) or window)
    engine = SimpleNamespace(moe_offload_cache=SimpleNamespace(prefetch=prefetch))
    monkeypatch.setattr(ENV.MOE_PREFETCH_DEBUG, "value", debug)
    # the freetoken loggers do not propagate, so caplog only sees them in some import orders
    monkeypatch.setattr(engine_mod.logger, "info_rank0", lambda msg, *a, **k: lines.append(msg))
    engine_mod.Engine._emit_prefetch_stats(engine)
    return "\n".join(lines), taken


def test_engine_reports_one_prefetch_window(monkeypatch):
    from freetoken.engine.engine import MOE_STATS_INTERVAL

    window = torch.tensor([[0] * 6, [30, 12, 4, 10, 10, 40], [40, 20, 6, 10, 10, 30]])
    out, taken = _emit_prefetch(window, monkeypatch)
    assert taken == [1]
    assert f"MoE prefetch measure ({MOE_STATS_INTERVAL} decode steps): issued/layer=3.50, useful/layer=1.60" in out
    assert "per layer" not in out
    out, _ = _emit_prefetch(window, monkeypatch, debug=True)
    assert "L1=1.20/3.00/4.00, L2=2.00/4.00/3.00" in out and "L0" not in out
    assert _emit_prefetch(torch.zeros((3, 6), dtype=torch.int64), monkeypatch)[0] == ""


def test_engine_reports_the_on_mode_window(monkeypatch):
    from freetoken.engine.engine import MOE_STATS_INTERVAL

    # issued, useful, resident_hits, calls, rows, misses, late, copied
    window = torch.tensor([[0] * 8, [30, 12, 0, 10, 10, 18, 2, 10], [40, 20, 0, 10, 10, 20, 0, 6]])
    out, _ = _emit_prefetch(window, monkeypatch, mode="on")
    assert f"MoE prefetch on ({MOE_STATS_INTERVAL} decode steps): issued/layer=3.50, useful/layer=1.60" in out
    assert "misses/layer=1.90, coverage=0.457, late=0.125 of 16 copies" in out and "resident_hits" not in out


def test_summary_rates():
    from freetoken.moe.prefetch import format_per_layer, format_summary, summarize

    # layer 0 never predicts; layers 1 and 2 over 10 bs-1 steps
    counters = torch.tensor([[0] * 6, [30, 12, 4, 10, 10, 40], [40, 20, 6, 10, 10, 30]])
    stats = summarize(counters)
    assert stats["layer_calls"] == 20 and stats["tokens"] == 10
    assert stats["issued_per_layer"] == 3.5 and stats["useful_per_layer"] == 1.6
    assert stats["precision"] == 32 / 70 and stats["coverage"] == 32 / 70
    assert stats["useful_per_token"] == 3.2 and stats["resident_hits_per_layer"] == 0.5
    assert "useful/token=3.2" in format_summary(stats)
    assert format_per_layer(counters) == "L1=1.20/3.00/4.00, L2=2.00/4.00/3.00"
