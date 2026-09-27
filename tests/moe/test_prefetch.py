"""FREETOKEN_MOE_PREFETCH stage A: the router-lookahead select / count kernels against a Python
reference, the mode switch, and the predictor stream's lifetime across resets and rebuilds.
The qwen4_exp stack tests (bit-identical outputs, counters over a real decode) live in
tests/models/qwen4_exp/test_skeleton.py."""

from __future__ import annotations

import weakref

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
    assert on.prefetch.mode == "on" and on.prefetch.cache.prefetch is on.prefetch and on.prefetch_on
    # no reference cycle: dropping the cache frees its tensors at once
    ref = weakref.ref(on)
    del on
    assert ref() is None
    for mode in ("measure", "on"):
        with pytest.raises(ValueError):
            cache(prefetch_mode=mode, decode_target="hybrid")


@requires_cuda
def test_the_prefetch_env_flag_skips_caches_that_do_not_decode_on_the_gpu(monkeypatch):
    # a process-wide flag must not break hybrid or CPU-decoded caches, as FREETOKEN_MOE_COPY_OVERLAP does not
    from freetoken.env import ENV
    from freetoken.moe.offload_cache import OffloadMoeCache

    monkeypatch.setattr(ENV.MOE_PREFETCH, "value", "on")
    for target in ("hybrid", "cpu"):
        c = OffloadMoeCache(num_layers=2, num_experts=4, cache_size=4, device=torch.device("cuda"), decode_target=target)
        assert c.prefetch is None and not c.prefetch_on
    assert OffloadMoeCache(num_layers=2, num_experts=4, cache_size=4, device=torch.device("cuda")).prefetch_on


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
    # its CTAs go ahead of pending compute ones, so a late copy does not also wait for SMs
    from cuda.bindings import runtime as cudart

    _, _, greatest = cudart.cudaDeviceGetStreamPriorityRange()
    assert cudart.cudaStreamGetPriority(copy)[1] == greatest < 0
    assert cudart.cudaStreamGetPriority(pf.stream.cuda_stream)[1] == 0


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


# FREETOKEN_MOE_PREFETCH_VERIFY (moe/verify.py): the mode switch, where the verifier is built, and
# the engine's report; the checks themselves run on the qwen4_exp stacks in test_skeleton.py


def test_verify_mode_switch():
    from freetoken.moe.verify import resolve_mode

    assert [resolve_mode(m) for m in ("0", "", "off", "1", " True ", "full", "META")] == [
        "off", "off", "off", "full", "full", "full", "meta"]
    with pytest.raises(ValueError):
        resolve_mode("bytes")


@requires_cuda
def test_cache_builds_the_verifier_only_for_gpu_decode(monkeypatch):
    from freetoken.env import ENV
    from freetoken.moe.offload_cache import OffloadMoeCache

    def cache(**kw):
        return OffloadMoeCache(num_layers=2, num_experts=4, cache_size=4, device=torch.device("cuda"), **kw)

    monkeypatch.setattr(ENV.MOE_PREFETCH_VERIFY, "value", "0")
    assert cache().verify is None and cache(verify_mode="off").verify is None
    full = cache(verify_mode="full", prefetch_mode="on")
    assert full.verify.mode == "full" and cache(verify_mode="meta").verify.mode == "meta"
    # the verifier holds no reference to its cache
    ref = weakref.ref(full)
    del full
    assert ref() is None
    with pytest.raises(ValueError):
        cache(verify_mode="full", decode_target="hybrid")
    monkeypatch.setattr(ENV.MOE_PREFETCH_VERIFY, "value", "1")
    for target in ("hybrid", "cpu"):
        assert cache(decode_target=target).verify is None
    assert cache().verify.mode == "full"


@requires_cuda
def test_engine_reports_the_verify_window_and_session(monkeypatch):
    from types import SimpleNamespace

    from freetoken.engine import engine as engine_mod
    from freetoken.engine.engine import MOE_STATS_INTERVAL
    from freetoken.moe.offload_cache import OffloadMoeCache
    from freetoken.moe.verify import BYTES_PRE, CHECKS, NUM_FIELDS, PRE_BAD

    cache = OffloadMoeCache(num_layers=3, num_experts=4, cache_size=4, device=torch.device("cuda"), verify_mode="full")
    cache.set_bank_sources({
        "gate_up": [torch.randn(4, 32, 8, dtype=torch.bfloat16).pin_memory() for _ in range(3)],
        "down": [torch.randn(4, 8, 16, dtype=torch.bfloat16).pin_memory() for _ in range(3)],
    })
    v = cache.verify
    lines = []
    for level in ("info_rank0", "warning_rank0"):
        monkeypatch.setattr(engine_mod.logger, level, lambda msg, *a, level=level, **k: lines.append((level, msg)))
    engine = SimpleNamespace(moe_offload_cache=cache)

    v.counters[:, CHECKS] = 256
    engine_mod.Engine._emit_verify_stats(engine)
    assert lines == [("info_rank0", f"MoE verify full ({MOE_STATS_INTERVAL} decode steps): 768 layer calls checked, "
                                    "meta_bad=0, pre_bad=0, post_bad=0, post_meta_bad=0; session bad=0")]
    lines.clear()
    v.counters[:, CHECKS] = 256
    v.counters[1, PRE_BAD] = 1
    # call 7, layer 1, row 3 of a top-2 call, expert 2, slot 1, bank down, before the GEMM; slot 1 holds
    # layer 1's expert 2, usage 9 at lru step 9, demand-copied; bytes 64.. differ in 4 words
    record = [7, 1, 3, 2, 1, 1, BYTES_PRE, 6, 1, 9, 9, 1, 64, 4]
    assert len(record) == NUM_FIELDS
    v.ring[0] = torch.tensor(record)
    v.cursor.fill_(1)
    v._top_k = 2
    engine_mod.Engine._emit_verify_stats(engine)
    assert [level for level, _ in lines] == ["warning_rank0", "warning_rank0"]
    assert "pre_bad=1" in lines[0][1] and "(meta/pre/post/post_meta by layer: L1=0/1/0/0); session bad=1" in lines[0][1]
    assert lines[1][1] == (
        "MoE verify record: bytes_pre call=7 layer=1 row=3 (token 1, rank 1) expert=2 slot=1 bank=down "
        "first_bad_byte=64 bad_words=4 | slot now holds L1/e2, slot_for_id=1, usage=9, lru_step=9, slot in: demand copy")
    lines.clear()
    engine_mod.Engine._emit_verify_stats(engine)  # nothing checked and no new record: silent
    assert lines == []
    session = v.report_session()
    assert session[0] == ("warning", "MoE verify full (session): 1536 layer calls checked, meta_bad=0, pre_bad=1, "
                                     "post_bad=0, post_meta_bad=0 (meta/pre/post/post_meta by layer: L1=0/1/0/0)")
    assert session[1:] == [("warning", "MoE verify record: " + v.format_record(record))]


# FREETOKEN_MOE_SLOT_AUDIT (moe/slot_audit.py): the switch, where the auditor is built, and the
# engine's report; the recorders and the audit run on the qwen4_exp stacks in test_skeleton.py


def test_slot_audit_interval_switch(monkeypatch):
    from freetoken.env import ENV
    from freetoken.moe.slot_audit import resolve_interval

    monkeypatch.setattr(ENV.MOE_SLOT_AUDIT, "value", 64)
    assert [resolve_interval(v) for v in (None, 0, 1, 8)] == [64, 0, 1, 8]
    with pytest.raises(ValueError):
        resolve_interval(-1)


@requires_cuda
def test_cache_builds_the_auditor_only_for_gpu_decode(monkeypatch):
    from freetoken.env import ENV
    from freetoken.moe.offload_cache import OffloadMoeCache

    def cache(**kw):
        return OffloadMoeCache(num_layers=2, num_experts=4, cache_size=4, device=torch.device("cuda"), **kw)

    monkeypatch.setattr(ENV.MOE_SLOT_AUDIT, "value", 0)
    assert cache().audit is None and cache(slot_audit=0).audit is None
    on = cache(slot_audit=8, prefetch_mode="on")
    assert on.audit.interval == 8 and on.slot_audit == 8
    ref = weakref.ref(on)  # the auditor holds no reference to its cache
    del on
    assert ref() is None
    with pytest.raises(ValueError):
        cache(slot_audit=8, decode_target="hybrid")
    monkeypatch.setattr(ENV.MOE_SLOT_AUDIT, "value", 16)
    for target in ("hybrid", "cpu"):
        assert cache(decode_target=target).audit is None
    assert cache().audit.interval == 16
    monkeypatch.setenv("FREETOKEN_SKIP_FAST_INDEX_COPY", "1")
    assert cache().audit is None, "copies that move nothing cannot be audited"


@requires_cuda
def test_engine_reports_the_audit_window_and_session(monkeypatch):
    from types import SimpleNamespace

    from freetoken.engine import engine as engine_mod
    from freetoken.engine.engine import MOE_STATS_INTERVAL
    from freetoken.moe.offload_cache import OffloadMoeCache

    cache = OffloadMoeCache(num_layers=3, num_experts=4, cache_size=6, device=torch.device("cuda"), slot_audit=2)
    cache.set_bank_sources({
        "gate_up": [torch.randn(4, 32, 8, dtype=torch.bfloat16).pin_memory() for _ in range(3)],
        "down": [torch.randn(4, 8, 16, dtype=torch.bfloat16).pin_memory() for _ in range(3)],
    })
    cache.reset()
    lines = []
    for level in ("info_rank0", "warning_rank0"):
        monkeypatch.setattr(engine_mod.logger, level, lambda msg, *a, level=level, **k: lines.append((level, msg)))
    engine = SimpleNamespace(moe_offload_cache=cache)
    engine_mod.Engine._emit_audit_stats(engine)
    assert lines == [], "no audit ran: nothing to say"

    # one decode step of layer 1 routing experts 2 and 0: two demand installs and their copy
    ids = torch.tensor([[2, 0]], dtype=torch.int32, device="cuda")
    cache.ensure_experts(1, ids)
    cache.copy_missing()
    audit = cache.audit
    assert not audit.end_decode_step(cache) and audit.end_decode_step(cache), "every 2 decode steps"
    engine_mod.Engine._emit_audit_stats(engine)
    assert lines == [("info_rank0", f"MoE slot audit ({MOE_STATS_INTERVAL} decode steps, every 2): 1 audits, 2 held slots "
                                    "an audit (2 byte-checked), bad slots=0 (bytes=0, map=0, stale slot_for_id=0, owner out "
                                    "of range=0), new=0, repeat=0, events=4; session: 1 audits, 0 bad (slot, owner) pairs")]
    lines.clear()
    slot = int(ids[0, 0])
    cache.bank_views()[1][slot].view(torch.uint8).view(-1)[8:12].bitwise_not_()
    audit.scan(cache)
    engine_mod.Engine._emit_audit_stats(engine)
    assert [level for level, _ in lines] == ["warning_rank0", "warning_rank0"]
    assert "1 audits, 2 held slots an audit (2 byte-checked), bad slots=1 (bytes=1," in lines[0][1]
    assert lines[1][1].startswith(
        f"MoE slot audit record: audit=1 step=0 lru=1 slot={slot} holds L1/e2 (slot_for_id={slot}, usage=1) | bytes differ in "
        "down (first at down+8, 1 bad words) | diagnosis: demand install and demand copy of L1/e2 (#"), lines[1][1]
    assert "then the bytes changed with no recorded write" in lines[1][1]
    # the kernel ranks misses by id: e0 installs first (#0), e2 second (#1); the copy records after both
    assert lines[1][1].endswith("history (1 map / 1 byte events in all): #1 demand install L1/e2 @step 0/lru 1 (then held "
                                "L1/e2, usage 1); #3 demand copy L1/e2 @step 0/lru 1 (slot held L1/e2, usage 1)"), lines[1][1]
    lines.clear()
    engine_mod.Engine._emit_audit_stats(engine)
    assert lines == []
    session = audit.report_session()
    assert session[0][0] == "warning" and session[0][1].startswith("MoE slot audit (session, every 2 decode steps): 2 audits")
    assert session[1:] == [("warning", "MoE slot audit record: " + audit.format_record(audit.ring[0].tolist()))]
