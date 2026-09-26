"""--moe-cache-policy kd / kdfb / rule: the vendored scored-victim ensure kernel.

The kernel must reproduce flashlib's LRU bit for bit under POLICY=lru, and match the CPU
reference (ref_scored_cache.py, a port of the eviction replay) victim for victim under the
scored policies. Set FREETOKEN_MOE_ROUTING_TRACE to a compact routing trace directory
(experts.npy, logit_idx.npy, logit_val.npy) to also replay real decode rows.
"""
import os
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest
import torch

L, E, S, TOP_K = 48, 512, 1650, 10
NEAR_MISS_THR = 0.25
_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
_TRACE = os.environ.get("FREETOKEN_MOE_ROUTING_TRACE", "")


def _trace():
    if not _TRACE or not (Path(_TRACE) / "experts.npy").exists():
        pytest.skip("FREETOKEN_MOE_ROUTING_TRACE is not set to a routing trace directory")
    load = lambda name: np.load(Path(_TRACE) / name, mmap_mode="r")  # noqa: E731
    return load("experts.npy"), load("logit_idx.npy"), load("logit_val.npy")


def _trace_logits(li, lv):
    """Router logits rebuilt from the trace's recorded top-32 (the rest far below any margin)."""
    li, lv = np.asarray(li).astype(np.int64), np.asarray(lv).astype(np.float32)
    logits = np.full(li.shape[:-1] + (E,), -1e4, np.float32)
    np.put_along_axis(logits, li, lv, axis=-1)
    return logits


def _zipf_rows(n, rows_per_step=1, seed=0, num_layers=L, num_experts=E, top_k=TOP_K, skew=0.9):
    """Skewed decode routing ``[n, L, rows_per_step, top_k]``: distinct ids within a row, duplicates across rows."""
    rng = np.random.default_rng(seed)
    p = 1.0 / np.arange(1, num_experts + 1) ** skew
    perm = [rng.permutation(num_experts) for _ in range(num_layers)]
    out = np.empty((n, num_layers, rows_per_step, top_k), np.int64)
    for r in range(n):
        for layer in range(num_layers):
            for b in range(rows_per_step):
                pick = rng.choice(num_experts, top_k, replace=False, p=p / p.sum())
                out[r, layer, b] = perm[layer][pick]
    return out


def _random_logits(rows, seed=1, num_experts=E):
    """bf16-exact router logits shaped like ``rows`` with the routed ids on top (~10% near misses at 0.25)."""
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=rows.shape[:-1] + (num_experts,)).astype(np.float32)
    np.put_along_axis(logits, rows, np.take_along_axis(logits, rows, axis=-1) + 3.0, axis=-1)
    return torch.from_numpy(logits).bfloat16().float().numpy()


def _cache(policy, cache_size=S, num_layers=L, num_experts=E, device="cuda", **kw):
    from freetoken.moe.offload_cache import OffloadMoeCache

    return OffloadMoeCache(
        num_layers=num_layers, num_experts=num_experts, cache_size=cache_size,
        device=torch.device(device), cache_policy=policy, **kw,
    )


def _ref(policy, cache_size=S, num_layers=L, num_experts=E):
    from freetoken.moe.scored_ensure import POLICY_IDS

    from .ref_scored_cache import RefScoredCache

    return RefScoredCache(num_layers, num_experts, cache_size, POLICY_IDS[policy])


def _run_gpu(cache, rows, logits=None, dtype=torch.bfloat16):
    """Decode ``rows`` ``[N, L, B, K]`` through ``cache.ensure_experts``; per-row miss counts."""
    dev = cache.device
    ids = torch.from_numpy(np.ascontiguousarray(rows, dtype=np.int32)).to(dev)
    lg = None if logits is None else torch.from_numpy(np.ascontiguousarray(logits)).to(dev, dtype)
    cache.collect_stats = True
    cache.lru_stats.zero_()
    marks = torch.zeros(rows.shape[0], dtype=torch.int64, device=dev)
    for r in range(rows.shape[0]):
        for layer in range(rows.shape[1]):
            cache.ensure_experts(layer, ids[r, layer], router_logits=None if lg is None else lg[r, layer])
        marks[r] = cache.lru_stats[:, 1].sum()
    return np.diff(marks.cpu().numpy(), prepend=0)


def _run_ref(ref, rows, logits=None):
    from .ref_scored_cache import replay

    return replay(ref, rows, logits, NEAR_MISS_THR)


def _assert_same_tables(cache, ref):
    got = lambda t: t.detach().cpu().numpy().astype(np.int64)  # noqa: E731
    np.testing.assert_array_equal(got(cache.slot_for_id.view(-1)), ref.slot_of_id)
    np.testing.assert_array_equal(got(cache.id_of_slot), ref.id_of_slot)
    np.testing.assert_array_equal(got(cache.usage), ref.usage)
    assert int(cache.step) == ref.step
    if ref.policy:
        assert int(cache.evict_tok) == ref.tok
        np.testing.assert_array_equal(got(cache.evict_last_tok), ref.last_tok)
    if ref.policy >= 2:
        np.testing.assert_array_equal(got(cache.evict_lc), ref.lc)
        np.testing.assert_array_equal(got(cache.evict_ct), ref.ct)
    if ref.policy:
        _assert_mirrors_hold(cache)


def _assert_mirrors_hold(cache):
    """Every slot whose owner tag matches its id mirrors that id's state."""
    got = lambda t: t.detach().cpu().numpy().astype(np.int64)  # noqa: E731
    ids, owner = got(cache.id_of_slot), got(cache.evict_slot_owner)
    tagged = (ids >= 0) & (owner == ids)
    np.testing.assert_array_equal(got(cache.evict_slot_last_tok)[tagged], got(cache.evict_last_tok)[ids[tagged]])
    if cache.evict_lc is not None:
        np.testing.assert_array_equal(got(cache.evict_slot_lc)[tagged], got(cache.evict_lc)[ids[tagged]])
        np.testing.assert_array_equal(got(cache.evict_slot_ct)[tagged], got(cache.evict_ct)[ids[tagged]])
    return tagged


def _check_against_ref(policy, rows, logits=None, dtype=torch.bfloat16):
    cache, ref = _cache(policy), _ref(policy)
    assert policy != "rule" or cache.near_miss_thr == NEAR_MISS_THR
    got = _run_gpu(cache, rows, logits, dtype)
    want = _run_ref(ref, rows, logits)
    np.testing.assert_array_equal(got, want)
    _assert_same_tables(cache, ref)
    return got


# (a) POLICY=lru is flashlib's kernel -------------------------------------------------------


@_cuda
@pytest.mark.parametrize("rows_per_step", [1, 2])
def test_vendored_lru_is_bit_identical_to_flashlib(rows_per_step):
    from flashlib.kernels.slot_cache import N_STATS, lru_ensure

    from freetoken.moe.scored_ensure import scored_ensure

    dev = torch.device("cuda")
    k = rows_per_step * TOP_K
    # skewed enough that some calls hit every id
    stream = _zipf_rows(63, rows_per_step, seed=rows_per_step, skew=1.6).reshape(-1, k)[:3000]
    layers = np.arange(stream.shape[0]) % L

    def state():
        return dict(
            slot_of_id=torch.full((L * E,), -1, dtype=torch.int32, device=dev),
            id_of_slot=torch.full((S,), -1, dtype=torch.int32, device=dev),
            usage=torch.zeros(S, dtype=torch.int64, device=dev),
            step=torch.zeros((), dtype=torch.int64, device=dev),
            src=torch.zeros(S, dtype=torch.int32, device=dev),
            dst=torch.zeros(S, dtype=torch.int32, device=dev),
            num=torch.zeros((), dtype=torch.int64, device=dev),
            stats=torch.zeros(N_STATS, dtype=torch.int64, device=dev),
        )

    a, b = state(), state()
    queries = torch.from_numpy(stream.astype(np.int32)).to(dev)
    qa, qb = queries.clone(), queries.clone()
    plans = ([], [])
    for i in range(stream.shape[0]):
        base = int(layers[i]) * E
        lru_ensure(qa[i], a["slot_of_id"], a["id_of_slot"], a["usage"], a["step"], qa[i],
                   a["src"], a["dst"], a["num"], stats=a["stats"], id_base=base)
        scored_ensure(qb[i], b["slot_of_id"], b["id_of_slot"], b["usage"], b["step"], qb[i],
                      b["src"], b["dst"], b["num"], policy=0, num_layers=L, num_experts=E,
                      stats=b["stats"], id_base=base)
        for st, plan in zip((a, b), plans):
            plan.append(torch.cat([st["num"].view(1), st["src"][:k].long(), st["dst"][:k].long()]))
    assert torch.equal(qa, qb), "out slots differ"
    pa, pb = (torch.stack(plan) for plan in plans)
    n = pa[:, 0]
    assert int(n.sum()) > 1000 and int((n == 0).sum()) > 0  # the stream exercises misses and all-hit calls
    for i in range(pa.shape[0]):
        m = int(n[i])
        assert torch.equal(pa[i, : 1 + m], pb[i, : 1 + m]) and torch.equal(pa[i, 1 + k : 1 + k + m], pb[i, 1 + k : 1 + k + m])
    for key in ("slot_of_id", "id_of_slot", "usage", "step", "stats"):
        assert torch.equal(a[key], b[key]), key


def test_lru_policy_stays_on_flashlib(monkeypatch):
    import freetoken.moe.offload_kernels as ok

    cache = _cache("lru", device="cpu")
    calls = []
    monkeypatch.setattr(ok, "lru_ensure", lambda *a, **kw: calls.append("flashlib"))
    monkeypatch.setattr(ok, "ensure_experts_scored", lambda *a, **kw: calls.append("scored"))
    cache.ensure_experts(3, torch.zeros(TOP_K, dtype=torch.int32))
    cache.ensure_experts(3, torch.zeros(TOP_K, dtype=torch.int32), update_state=False)
    assert calls == ["flashlib", "flashlib"]
    assert cache.evict_tok is None and cache.evict_last_tok is None and cache.near_miss_thr is None


# (b) scored policies == CPU reference ------------------------------------------------------


@_cuda
@pytest.mark.parametrize("policy", ["lru", "kd", "kdfb", "rule"])
def test_scored_kernel_matches_cpu_reference_on_synthetic_routing(policy):
    rows = _zipf_rows(40, seed=3)
    got = _check_against_ref(policy, rows, _random_logits(rows) if policy == "rule" else None)
    assert got.sum() > 0


@_cuda
@pytest.mark.slow
@pytest.mark.parametrize("policy", ["kd", "kdfb", "rule"])
def test_scored_kernel_matches_cpu_reference_on_trace_rows(policy):
    experts, li, lv = _trace()
    rows = np.asarray(experts[:2000]).astype(np.int64)[:, :, None, :]
    logits = _trace_logits(li[:2000], lv[:2000])[:, :, None, :] if policy == "rule" else None
    got = _check_against_ref(policy, rows, logits, torch.float32)
    lru = _run_ref(_ref("lru"), rows)
    assert got.sum() < lru.sum()  # the point of the policy


# (c) victims: never the call's own ids, empty slots first ----------------------------------


@_cuda
@pytest.mark.parametrize("policy", ["kd", "kdfb", "rule"])
@pytest.mark.parametrize("rows_per_step", [1, 2])
def test_victims_skip_the_call_and_take_empty_slots_first(policy, rows_per_step):
    from freetoken.moe import scored_ensure as se

    num_layers, num_experts, cache_size = 4, 64, 100
    cache = _cache(policy, cache_size, num_layers, num_experts)
    ref = _ref(policy, cache_size, num_layers, num_experts)
    warm = _zipf_rows(6, rows_per_step, seed=5, num_layers=num_layers, num_experts=num_experts)
    _run_gpu(cache, warm)
    _run_ref(ref, warm)
    assert int((cache.id_of_slot >= 0).sum()) == cache_size

    layer = 2
    held = cache.id_of_slot.cpu().numpy()
    resident = sorted(int(i) - layer * num_experts for i in held if layer * num_experts <= i < (layer + 1) * num_experts)
    absent = sorted(set(range(num_experts)) - set(resident))
    k = rows_per_step * TOP_K
    hits = resident[: k // 2]
    ids = np.array((hits + absent)[:k], np.int64)
    if rows_per_step == 2:
        ids[-1] = ids[0]  # a duplicate hit
    # make this call's hits the stalest ids in the cache, so only the pin keeps them
    hit_keys = [layer * num_experts + e for e in hits]
    for key in hit_keys:
        cache.evict_last_tok[key] = se.LAST_TOK_NEVER
        ref.last_tok[key] = se.LAST_TOK_NEVER
        if cache.evict_lc is not None:
            cache.evict_lc[key] = se.LC_NEVER
            ref.lc[key] = se.LC_NEVER
    hit_slots = {int(cache.slot_for_id[layer, e]) for e in hits}
    # invalidate three slots (as the prefill overlap does): they must be the first victims
    empty = sorted([s for s in (7, 40, 91, 13, 55, 77) if s not in hit_slots][:3])
    for slot in empty:
        for t, v in ((cache.id_of_slot, held[slot]), (ref.id_of_slot, held[slot])):
            t[slot] = -1
        cache.slot_for_id.view(-1)[int(held[slot])] = -1
        ref.slot_of_id[int(held[slot])] = -1
        cache.usage[slot] = 0
        ref.usage[slot] = 0

    q = torch.tensor(ids, dtype=torch.int32, device="cuda")
    lg = _random_logits(ids.reshape(rows_per_step, TOP_K), seed=9, num_experts=num_experts)
    cache.ensure_experts(layer, q, router_logits=torch.from_numpy(lg).cuda())
    out, src, dst = ref.ensure(layer, ids, logits=lg, thr=NEAR_MISS_THR)

    n = int(cache.num_indices)
    got_dst = cache.evict_slots[:n].tolist()
    assert n == len(set(ids.tolist()) - set(hits)) == src.size
    assert got_dst[: len(empty)] == empty
    assert not hit_slots & set(got_dst)
    assert q.tolist() == out.tolist()
    assert all(int(cache.slot_for_id[layer, e]) >= 0 for e in ids.tolist())
    assert got_dst == dst.tolist()
    _assert_same_tables(cache, ref)


# (d) the decode-step clock -----------------------------------------------------------------


@_cuda
@pytest.mark.parametrize("rows_per_step", [1, 2])
def test_token_clock_moves_once_per_decode_step(rows_per_step):
    cache = _cache("kdfb")
    rows = _zipf_rows(5, rows_per_step, seed=7)
    ids = torch.from_numpy(rows.astype(np.int32)).cuda()
    for r in range(rows.shape[0]):
        for layer in range(L):
            cache.ensure_experts(layer, ids[r, layer])
            assert int(cache.evict_tok) == r + 1
    # a CPU-decoded first layer moves the clock to the first GPU layer
    cache.reset()
    cache.cpu_layer_ids = frozenset({0, 1})
    ids = torch.from_numpy(rows.astype(np.int32)).cuda()
    for layer in range(2, L):
        cache.ensure_experts(layer, ids[0, layer])
    assert int(cache.evict_tok) == 1


@_cuda
def test_small_prefill_ensures_leave_the_state_alone(monkeypatch):
    import freetoken.layers.moe as moe_mod
    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.layers.moe import OffloadMoELayer
    from freetoken.layers.quantization import NoQuantConfig

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    dev = torch.device("cuda")
    experts, top_k, hidden, inter = 64, 4, 16, 32
    layer = OffloadMoELayer(0, experts, top_k, hidden, inter, quant_config=NoQuantConfig(),
                            prefix="model.layers.0.mlp.experts")
    cache = _cache("kdfb", 96, 1, experts)
    g = torch.Generator().manual_seed(0)
    cache.set_bank_sources({
        "gate_up": [torch.randn(experts, 2 * inter, hidden, generator=g, dtype=torch.bfloat16) * 0.1],
        "down": [torch.randn(experts, hidden, inter, generator=g, dtype=torch.bfloat16) * 0.1],
    })
    layer.offload_cache = cache
    decode = torch.tensor([[1, 2, 3, 4]], dtype=torch.int32, device=dev)
    cache.ensure_experts(0, decode)
    before = [t.clone() for t in (cache.evict_tok, cache.evict_last_tok, cache.evict_lc, cache.evict_ct)]

    monkeypatch.setattr(moe_mod, "_SMALL_PREFILL_TOKENS", 64)
    tokens = 40
    ids = torch.randint(0, experts // 2, (tokens, top_k), dtype=torch.int32, device=dev)
    w = torch.full((tokens, top_k), 1.0 / top_k, device=dev)
    layer._prefill_routed(torch.randn(tokens, hidden, dtype=torch.bfloat16, device=dev), w, ids.clone())
    assert bool((cache.slot_for_id[0, ids.long().unique()] >= 0).all())  # it did make them resident
    assert int((cache.slot_for_id[0] >= 0).sum()) < experts  # through the on-demand path, not the whole layer
    after = (cache.evict_tok, cache.evict_last_tok, cache.evict_lc, cache.evict_ct)
    for b, a in zip(before, after):
        assert torch.equal(b, a)


@_cuda
def test_dsv4_on_demand_prefill_leaves_the_state_alone():
    from types import SimpleNamespace

    from freetoken.models.deepseek_v4.moe import DSV4OffloadMoELayer

    experts, top_k = 64, 4
    cache = _cache("kdfb", 96, 1, experts)
    cache.set_bank_sources({"gate_up": [torch.zeros(experts, 2, 4)], "down": [torch.zeros(experts, 4, 2)]})
    cache.ensure_experts(0, torch.tensor([1, 2, 3, 4], dtype=torch.int32, device="cuda"))
    before = [t.clone() for t in (cache.evict_tok, cache.evict_last_tok, cache.evict_lc, cache.evict_ct)]
    # few enough routes (3 x 4 < 64) for the slot path rather than whole-layer streaming
    layer = SimpleNamespace(offload_cache=cache, num_experts=experts, top_k=top_k, layer_id=0,
                            _expert_gemm=lambda *a, **kw: None)
    ids = torch.tensor([[5, 6, 7, 8], [9, 10, 11, 12], [5, 13, 14, 15]], dtype=torch.int32, device="cuda")
    DSV4OffloadMoELayer._prefill_routed(layer, torch.zeros(3, 8, device="cuda"), torch.full((3, top_k), 0.25), ids)
    assert bool((cache.slot_for_id[0, 5:16] >= 0).all())
    for b, a in zip(before, (cache.evict_tok, cache.evict_last_tok, cache.evict_lc, cache.evict_ct)):
        assert torch.equal(b, a)


# (e) reset() and rebuild() -----------------------------------------------------------------


@_cuda
@pytest.mark.parametrize("policy", ["kd", "kdfb", "rule"])
def test_reset_and_rebuild_clear_the_state(policy):
    from freetoken.moe import scored_ensure as se

    cache = _cache(policy, 1480)
    cache.set_bank_sources({"gate_up": [torch.zeros(E, 2, 4)] * L, "down": [torch.zeros(E, 4, 2)] * L})
    rows = _zipf_rows(3, seed=11)
    fresh = _ref(policy)

    def assert_cold():
        assert int(cache.evict_tok) == 0
        assert bool((cache.evict_last_tok == se.LAST_TOK_NEVER).all())
        assert bool((cache.evict_slot_owner == -1).all()) and cache.evict_slot_owner.numel() == cache.cache_size
        if cache.evict_lc is not None:
            assert bool((cache.evict_lc == se.LC_NEVER).all()) and bool((cache.evict_ct == 0).all())

    for clear in (cache.reset, lambda: cache.rebuild(S)):
        _run_gpu(cache, rows)
        assert int(cache.evict_tok) == rows.shape[0]
        clear()
        assert_cold()
    assert cache.cache_size == S
    # a rebuilt cache replays exactly like a fresh one
    np.testing.assert_array_equal(_run_gpu(cache, rows), _run_ref(fresh, rows))
    _assert_same_tables(cache, fresh)


# (f) CUDA graph replay == eager ------------------------------------------------------------


@_cuda
@pytest.mark.parametrize("policy", ["kdfb", "rule"])
@pytest.mark.parametrize("rows_per_step", [1, 2])
def test_graph_replay_matches_eager(policy, rows_per_step):
    steps = 50
    rows = _zipf_rows(steps + 1, rows_per_step, seed=13)
    ids = torch.from_numpy(rows.astype(np.int32)).cuda()
    logits = torch.from_numpy(_random_logits(rows, seed=14)).cuda().bfloat16()

    eager = _cache(policy)
    for r in range(1, steps + 1):
        for layer in range(L):
            eager.ensure_experts(layer, ids[r, layer].clone(), router_logits=logits[r, layer])

    graphed = _cache(policy)
    static_ids = ids[0].clone()
    static_logits = logits[0].clone()

    def step():
        for layer in range(L):
            graphed.ensure_experts(layer, static_ids[layer], router_logits=static_logits[layer])

    step()  # warm-up compiles the kernel variants outside capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    graphed.reset()  # as graph.py does after capture
    for r in range(1, steps + 1):
        static_ids.copy_(ids[r])
        static_logits.copy_(logits[r])
        graph.replay()
    torch.cuda.synchronize()

    assert int(graphed.evict_tok) == steps
    for name in ("slot_for_id", "id_of_slot", "usage", "step", "evict_tok", "evict_last_tok", "evict_lc", "evict_ct",
                 "evict_slot_owner", "evict_slot_last_tok", "evict_slot_lc", "evict_slot_ct"):
        assert torch.equal(getattr(eager, name), getattr(graphed, name)), name


# (g) bs=2: two rows per call ---------------------------------------------------------------


@_cuda
@pytest.mark.parametrize("policy", ["kd", "kdfb", "rule"])
def test_two_rows_per_call_match_cpu_reference(policy):
    rows = _zipf_rows(30, 2, seed=17)
    assert any(len(set(rows[r, layer].ravel())) < 2 * TOP_K for r in range(30) for layer in range(L))
    _check_against_ref(policy, rows, _random_logits(rows, seed=18) if policy == "rule" else None)


@_cuda
@pytest.mark.slow
@pytest.mark.parametrize("policy", ["kdfb", "rule"])
def test_two_trace_streams_per_call_match_cpu_reference(policy):
    experts, li, lv = _trace()
    a, b = slice(0, 600), slice(45000, 45600)  # two different requests decoding together
    rows = np.stack([np.asarray(experts[a]), np.asarray(experts[b])], axis=2).astype(np.int64)
    logits = None
    if policy == "rule":
        logits = np.stack([_trace_logits(li[a], lv[a]), _trace_logits(li[b], lv[b])], axis=2)
    _check_against_ref(policy, rows, logits, torch.float32)


# the slot mirrors: foreign installs and state-free ensures fall back to the per-id state -----


@_cuda
@pytest.mark.parametrize("policy", ["kd", "kdfb", "rule"])
def test_materialized_prefill_between_decodes_matches_cpu_reference(policy):
    cache, ref = _cache(policy), _ref(policy)
    rows = _zipf_rows(24, seed=21)
    logits = _random_logits(rows, seed=22) if policy == "rule" else None
    # layer 5 comes back to slots it lost to layer 9: its tags there would otherwise look current
    for part, layer in ((slice(0, 6), 5), (slice(6, 12), 9), (slice(12, 18), 5), (slice(18, 24), None)):
        lg = None if logits is None else logits[part]
        np.testing.assert_array_equal(_run_gpu(cache, rows[part], lg), _run_ref(ref, rows[part], lg))
        _assert_same_tables(cache, ref)
        if layer is not None:
            # the whole-layer prefill path re-seats this layer's experts at slots [0, E)
            cache.materialize_layer(layer)
            ref.materialize(layer)
            assert not bool((cache.evict_slot_owner[:E] >= 0).any())
            _assert_same_tables(cache, ref)


@_cuda
@pytest.mark.parametrize("policy", ["kd", "kdfb"])
def test_materialize_untags_the_slots_it_fills(policy):
    # expert 3 of layer 0 leaves slot 3, is used elsewhere, then materialize puts it back at slot 3
    cache = _cache(policy, 24, 2, 8)
    dev = torch.device("cuda")
    ids = lambda *e: torch.tensor(e, dtype=torch.int32, device=dev)  # noqa: E731
    cache.materialize_layer(0)
    cache.ensure_experts(0, ids(3, 4))  # a hit at slot 3 tags it with expert 3
    assert int(cache.evict_slot_owner[3]) == 3
    cache.materialize_layer(1)  # slot 3 now holds layer 1's expert 3
    cache.ensure_experts(1, ids(0, 1))
    cache.ensure_experts(0, ids(3, 5))  # expert 3 comes back elsewhere, with new state
    assert int(cache.slot_for_id[0, 3]) != 3
    cache.materialize_layer(0)
    assert int(cache.slot_for_id[0, 3]) == 3
    _assert_mirrors_hold(cache)


@_cuda
@pytest.mark.parametrize("policy", ["kd", "kdfb", "rule"])
@pytest.mark.parametrize("chunks", [1, 2])
def test_state_free_ensures_between_decodes_match_cpu_reference(policy, chunks):
    cache, ref = _cache(policy), _ref(policy)
    rows = _zipf_rows(12, seed=23)
    extra = _zipf_rows(4, 3, seed=24, skew=0.3)
    logits = _random_logits(rows, seed=25) if policy == "rule" else None
    for r in range(rows.shape[0]):
        lg = None if logits is None else logits[r : r + 1]
        np.testing.assert_array_equal(_run_gpu(cache, rows[r : r + 1], lg), _run_ref(ref, rows[r : r + 1], lg))
        if r % 3 == 2:
            for layer in range(0, L, 7):  # a small prefill's chunked ensures
                ids = np.unique(extra[r // 3, layer])
                # two chunks pin each other's slots as _ensure_unique does; one chunk keeps the plain pin
                pin = cache.step.clone() if chunks > 1 else None
                ref_pin = ref.step if chunks > 1 else None
                for part in np.array_split(ids, chunks):
                    q = torch.from_numpy(part.astype(np.int32)).cuda()
                    cache.ensure_experts(layer, q, update_state=False, pin_since=pin)
                    out, _, dst = ref.ensure(layer, part, update_state=False, pin_since=ref_pin)
                    assert q.tolist() == out.tolist()
                    assert bool((cache.evict_slot_owner[torch.from_numpy(dst).cuda()] == -1).all())
                assert bool((cache.slot_for_id[layer, torch.from_numpy(ids).cuda()] >= 0).all())
        _assert_same_tables(cache, ref)
    assert not _assert_mirrors_hold(cache).all()  # some slots still read the per-id state


# rule: the near misses come from the router logits ----------------------------------------


@_cuda
@pytest.mark.parametrize("rows_per_step", [1, 2])
def test_rule_refreshes_the_router_near_misses(rows_per_step):
    from freetoken.moe import scored_ensure as se

    cache = _cache("rule")
    layer = 5
    g = torch.Generator(device="cuda").manual_seed(rows_per_step)
    logits = (torch.randn(rows_per_step, E, device="cuda", generator=g) * 1.5).bfloat16()
    ids = torch.topk(logits.float(), TOP_K, dim=-1).indices.to(torch.int32)
    cache.ensure_experts(0, torch.arange(TOP_K, dtype=torch.int32, device="cuda"))  # step 1 starts
    cache.ensure_experts(layer, ids.clone(), router_logits=logits)
    kth = logits.float().gather(1, ids.long()).min(dim=1, keepdim=True).values
    near = (logits.float() >= kth - NEAR_MISS_THR).any(0)
    routed = torch.zeros(E, dtype=torch.bool, device="cuda")
    routed[ids.long().view(-1)] = True
    last = cache.evict_last_tok[layer * E : (layer + 1) * E]
    assert bool((last[routed] == 1).all())
    assert bool((last[near & ~routed] == 0).all())
    assert bool((last[~near & ~routed] == se.LAST_TOK_NEVER).all())
    assert 0 < int((near & ~routed).sum()) < E - TOP_K * rows_per_step


@pytest.mark.parametrize("policy", ["lru", "kdfb", "rule"])
def test_decode_forward_hands_the_router_logits_to_ensure(monkeypatch, policy):
    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.layers.moe import OffloadMoELayer
    from freetoken.layers.quantization import NoQuantConfig

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    layer = OffloadMoELayer(0, 8, 2, 8, 16, quant_config=NoQuantConfig(), prefix="model.layers.0.mlp.experts")
    layer.offload_cache = cache = _cache(policy, 12, 1, 8, device="cpu")
    seen = {}
    monkeypatch.setattr(
        "freetoken.layers.moe.fused_topk",
        lambda **kw: (torch.full((2, 2), 0.5), torch.zeros((2, 2), dtype=torch.int32)),
    )
    monkeypatch.setattr(cache, "ensure_experts", lambda lid, ids, **kw: seen.update(kw))
    monkeypatch.setattr(cache, "copy_missing", lambda: None)
    monkeypatch.setattr(cache, "bank_views", lambda: ())
    monkeypatch.setattr(layer, "_expert_gemm", lambda *a, **kw: None)
    router_logits = torch.randn(2, 8)
    layer.decode_forward(torch.randn(2, 8), router_logits)
    assert seen["router_logits"] is router_logits
    assert cache.near_miss_thr == (NEAR_MISS_THR if policy == "rule" else None)


def test_policy_flag_and_gpu_only():
    import contextlib
    import io

    from freetoken.server.args import parse_args

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.suppress(SystemExit):
        parse_args(["--help"])
    assert "{lru,kd,kdfb,rule}" in buf.getvalue()
    for policy in ("kd", "kdfb", "rule"):
        _cache(policy, 12, 1, 8, device="cpu")
        with pytest.raises(ValueError, match="GPU decode"):
            _cache(policy, 12, 1, 8, device="cpu", decode_target="hybrid")


@_cuda
@pytest.mark.parametrize("policy", ["kd", "kdfb", "rule"])
def test_an_all_pinned_call_never_stores_past_the_slot_arrays(policy):
    # 5 slots pad to an 8-lane block; with every slot pinned the sentinel key must not decode to lane 7.
    # The cache enforces cache_size >= num_experts, so the kernel gets 5-slot views of a 16-slot cache.
    from freetoken.moe.offload_kernels import ensure_experts_scored

    s, pad = 5, 3
    cache = _cache(policy, cache_size=16, num_layers=1, num_experts=16)
    guarded = {}
    for name in ("id_of_slot", "usage", "evict_slot_owner", "evict_slot_last_tok", "evict_slot_lc", "evict_slot_ct"):
        t = getattr(cache, name)
        if t is None:  # kd keeps no lc/ct mirrors
            continue
        buf = torch.full((s + pad,), 7, dtype=t.dtype, device=t.device)
        buf[:s] = t[:s]
        setattr(cache, name, buf[:s])
        guarded[name] = buf
    dev = cache.id_of_slot.device
    ensure_experts_scored(cache, 0, torch.arange(s, dtype=torch.int32, device=dev), bump_tok=True, update_state=True)
    # 6 distinct ids against 5 slots breaks the caller contract; the kernel must still stay in bounds
    q = torch.arange(s + 1, dtype=torch.int32, device=dev)
    ensure_experts_scored(cache, 0, q, bump_tok=True, update_state=True)
    torch.cuda.synchronize()
    for name, buf in guarded.items():
        assert torch.all(buf[s:] == 7), f"{name} written past slot {s - 1}"
    assert int(cache.num_indices) == 1 and 0 <= int(cache.evict_slots[0]) < s


# (h) FREETOKEN_MOE_PREFETCH=on: prefetch installs and low-priority keys ----------------------

_PF_L, _PF_E, _PF_S, _PF_K, _PF_W = 4, 64, 100, 6, 16


def _pf_cache(policy, cache_size=_PF_S, num_layers=_PF_L, num_experts=_PF_E):
    return _cache(policy, cache_size, num_layers, num_experts, prefetch_mode="on")


def _gpu_prefetch(cache, layer, sel, stats_row, width=_PF_W):
    """prefetch_ensure on the current stream; returns (src_ids, dst_slots), the plan it wrote."""
    from freetoken.moe.offload_kernels import prefetch_ensure_experts

    dev = cache.id_of_slot.device
    query = torch.tensor(list(sel) + [-1] * (width - len(sel)), dtype=torch.int32, device=dev)
    dst = torch.full((width,), -7, dtype=torch.int32, device=dev)
    src = torch.full((width,), -7, dtype=torch.int32, device=dev)
    num = torch.full((1,), 99, dtype=torch.int64, device=dev)
    ready = torch.ones(1, dtype=torch.int32, device=dev)
    prefetch_ensure_experts(cache, layer, query, dst, src, num, stats_row, ready)
    n = int(num)
    assert int(ready) == 0, "prefetch_ensure clears the copy's ready flag"
    assert dst[n:].tolist() == [-7] * (width - n)
    return src[:n].cpu().numpy().astype(np.int64), dst[:n].cpu().numpy().astype(np.int64), num


def _lowpri_slots(cache):
    return set(((cache.id_of_slot >= 0) & (cache.usage == 0)).nonzero().view(-1).tolist())


@_cuda
@pytest.mark.parametrize("policy", ["lru", "kd", "kdfb", "rule"])
@pytest.mark.parametrize("rows_per_step", [1, 2])
def test_prefetch_mode_matches_cpu_reference(policy, rows_per_step):
    """prefetch_ensure and the low-priority demand ensure against the CPU reference victim for victim,
    with adversarial candidates (next routed ids, resident ids, duplicates, padding, junk) and a
    materialized prefill in between; the per-layer prefetch stats match the reference's counts."""
    from freetoken.moe.offload_kernels import ensure_experts_scored
    from freetoken.moe.prefetch import CALLS, COPIED, ISSUED, LATE, MISSES, RESIDENT_HITS, ROWS, USEFUL

    rng = np.random.default_rng(31 + rows_per_step)
    rows = _zipf_rows(40, rows_per_step, seed=33, num_layers=_PF_L, num_experts=_PF_E, top_k=_PF_K)
    logits = _random_logits(rows, seed=34, num_experts=_PF_E) if policy == "rule" else None
    cache, ref = _pf_cache(policy), _ref(policy, _PF_S, _PF_L, _PF_E)
    dev = cache.id_of_slot.device
    stats = torch.zeros((_PF_L, 8), dtype=torch.int64, device=dev)
    want = np.zeros((_PF_L, 8), np.int64)
    ready = torch.zeros(1, dtype=torch.int32, device=dev)
    for r in range(rows.shape[0]):
        if r == 20:
            cache.materialize_layer(2)
            ref.materialize(2)
            assert not {s for s in _lowpri_slots(cache) if s < _PF_E}, "materialize installs at usage step"
        for layer in range(_PF_L):
            ids = rows[r, layer]
            plan = None
            if layer and rng.random() < 0.9:
                routed = list(dict.fromkeys(ids.ravel().tolist()))
                held = [e for e in range(_PF_E) if ref.slot_of_id[layer * _PF_E + e] >= 0]
                sel = rng.choice(routed, size=min(3, len(routed)), replace=False).tolist()
                sel += rng.choice(_PF_E, size=4).tolist() + held[:2] + sel[:1] + [-1]
                rng.shuffle(sel)
                pinned = set(np.flatnonzero((ref.usage == ref.step) & (ref.id_of_slot >= 0)).tolist())
                src, dst, num = _gpu_prefetch(cache, layer, sel, stats[layer])
                want_src, want_dst = ref.prefetch(layer, sel)
                np.testing.assert_array_equal(src, want_src)
                np.testing.assert_array_equal(dst, want_dst)
                assert not pinned & set(dst.tolist()), "a prefetch evicted a slot of the last demand call"
                assert set(dst.tolist()) <= _lowpri_slots(cache)
                want[layer, ISSUED] += len(want_src)
                _assert_same_tables(cache, ref)
                plan = num
                ready.fill_(int(rng.random() < 0.5))  # a copy that has (1) or has not (0) landed
            q = torch.from_numpy(np.ascontiguousarray(ids, dtype=np.int32)).to(dev).view(-1)
            lg = None if logits is None else torch.from_numpy(np.ascontiguousarray(logits[r, layer])).to(dev, torch.bfloat16)
            lowpri_before = _lowpri_slots(cache)
            ensure_experts_scored(
                cache, layer, q, bump_tok=layer == 0, update_state=True, router_logits=lg, lowpri=True,
                pf_count=None if plan is None else (stats[layer], ready, plan, rows_per_step),
            )
            out, src, _ = ref.ensure(layer, ids, bump_tok=layer == 0, logits=None if lg is None else lg.float().cpu().numpy(),
                                     thr=NEAR_MISS_THR, lowpri=True)
            np.testing.assert_array_equal(q.cpu().numpy(), out)
            _assert_same_tables(cache, ref)
            # a demand hit makes a prefetched slot an ordinary resident, tagged with its owner's state
            converted = lowpri_before & set(out.tolist())
            assert all(int(cache.usage[s]) == int(cache.step) for s in converted)
            if policy != "lru":
                assert all(int(cache.evict_slot_owner[s]) == int(cache.id_of_slot[s]) for s in converted)
            if plan is not None:
                n = int(plan)
                want[layer, [USEFUL, CALLS, ROWS, MISSES]] += [ref.useful, 1, rows_per_step, src.size]
                want[layer, LATE] += int(n > 0 and int(ready) == 0)
                want[layer, COPIED] += int(n > 0)
    got = stats.cpu().numpy()
    np.testing.assert_array_equal(got, want)
    assert got[:, USEFUL].sum() > 0 and got[:, LATE].sum() > 0 and got[:, RESIDENT_HITS].sum() == 0
    assert got[:, COPIED].sum() > got[:, LATE].sum()


@_cuda
@pytest.mark.parametrize("policy", ["lru", "rule"])
@pytest.mark.parametrize("rows_per_step", [1, 2, 8])
def test_prefetch_mode_matches_cpu_reference_at_production_geometry(policy, rows_per_step):
    """The same victim-for-victim check at the deployment's shape: 48 layers x 512 experts, 1480
    slots (a 2048-wide, 8-warp victim scan), top-10, up to 8 rows (every graph size that forks)
    and 32-wide prefetch queries, with small-prefill ensures (pinned since their first chunk) and a
    materialized layer in between. No prefetch ever evicts a slot of the last demand call."""
    from freetoken.moe.offload_kernels import ensure_experts_scored

    num_layers, num_experts, size, width = L, E, 1480, 32
    rng = np.random.default_rng(71 + rows_per_step)
    rows = _zipf_rows(4, rows_per_step, seed=72, num_layers=num_layers, num_experts=num_experts, top_k=TOP_K)
    logits = _random_logits(rows, seed=73, num_experts=num_experts) if policy == "rule" else None
    cache, ref = _pf_cache(policy, size, num_layers, num_experts), _ref(policy, size, num_layers, num_experts)
    dev = cache.id_of_slot.device
    stats = torch.zeros((num_layers, 8), dtype=torch.int64, device=dev)
    ready = torch.zeros(1, dtype=torch.int32, device=dev)
    installs = 0
    for r in range(rows.shape[0]):
        if r == 2:
            cache.materialize_layer(5)
            ref.materialize(5)
        for layer in range(num_layers):
            ids = rows[r, layer]
            plan = None
            if layer:
                routed = list(dict.fromkeys(ids.ravel().tolist()))
                held = [e for e in range(num_experts) if ref.slot_of_id[layer * num_experts + e] >= 0]
                sel = rng.choice(routed, size=min(6, len(routed)), replace=False).tolist()
                sel += rng.choice(num_experts, size=16).tolist() + held[:4] + sel[:2] + [-1, -1]
                sel = sel[:width]
                rng.shuffle(sel)
                pinned = set(np.flatnonzero((ref.usage == ref.step) & (ref.id_of_slot >= 0)).tolist())
                src, dst, plan = _gpu_prefetch(cache, layer, sel, stats[layer], width=width)
                want_src, want_dst = ref.prefetch(layer, sel)
                np.testing.assert_array_equal(src, want_src)
                np.testing.assert_array_equal(dst, want_dst)
                assert not pinned & set(dst.tolist()), "a prefetch evicted a slot of the last demand call"
                installs += len(dst)
                _assert_same_tables(cache, ref)
            q = torch.from_numpy(np.ascontiguousarray(ids, dtype=np.int32)).to(dev).view(-1)
            lg = None if logits is None else torch.from_numpy(np.ascontiguousarray(logits[r, layer])).to(dev, torch.bfloat16)
            ensure_experts_scored(
                cache, layer, q, bump_tok=layer == 0, update_state=True, router_logits=lg, lowpri=True,
                pf_count=None if plan is None else (stats[layer], ready, plan, rows_per_step),
            )
            out, _, _ = ref.ensure(layer, ids, bump_tok=layer == 0, logits=None if lg is None else lg.float().cpu().numpy(),
                                   thr=NEAR_MISS_THR, lowpri=True)
            np.testing.assert_array_equal(q.cpu().numpy(), out)
            _assert_same_tables(cache, ref)
        # a short prefill of one layer between decode steps: chunked, pinned since its first chunk
        layer = int(rng.integers(num_layers))
        uniq = np.unique(rng.choice(num_experts, size=60, replace=False))
        pin = cache.step.clone() if policy != "lru" else None
        ref_pin = ref.step if policy != "lru" else None
        for start in range(0, uniq.size, 32):
            part = torch.from_numpy(uniq[start : start + 32].astype(np.int32)).to(dev)
            cache.ensure_experts(layer, part, update_state=False, pin_since=pin)
            out, _, _ = ref.ensure(layer, uniq[start : start + 32], update_state=False, pin_since=ref_pin, lowpri=True)
            np.testing.assert_array_equal(part.cpu().numpy(), out)
            _assert_same_tables(cache, ref)
    assert installs > 0 and int((ref.id_of_slot >= 0).sum()) == size, "the cache must be full and evicting"


def _plant(cache, ref, slot, flat_id, usage):
    for t_id, t_slot in ((cache.id_of_slot, cache.slot_for_id.view(-1)), (ref.id_of_slot, ref.slot_of_id)):
        t_id[slot] = flat_id
        t_slot[flat_id] = slot
    cache.usage[slot] = usage
    ref.usage[slot] = usage


@_cuda
@pytest.mark.parametrize("policy", ["lru", "kd", "kdfb", "rule"])
def test_lowpri_band_sits_between_empty_slots_and_every_resident(policy):
    """An unconsumed prefetch is evicted after every empty slot and before any resident, even the
    coldest resident the score caps allow, whatever the slot order."""
    from freetoken.moe import scored_ensure as se
    from freetoken.moe.offload_kernels import ensure_experts_scored

    num_layers, num_experts, size = 2, 16, 16
    cache, ref = _pf_cache(policy, size, num_layers, num_experts), _ref(policy, size, num_layers, num_experts)
    dev = cache.id_of_slot.device
    cache.step.fill_(5)
    ref.step = 5
    # slots 0-12: layer-0 residents at the oldest usage 1 with never-seen state (the lowest possible
    # score); slot 13 empty; slots 14, 15 unconsumed prefetches
    for slot in range(13):
        _plant(cache, ref, slot, slot, 1)
    _plant(cache, ref, 14, 13, 0)
    _plant(cache, ref, 15, 14, 0)
    if policy != "lru":
        cache.evict_slot_owner.fill_(-1)
    victims = []
    for e in (15, 16 + 0, 16 + 1, 16 + 2):  # four misses, one per call
        layer, expert = divmod(e, num_experts)
        q = torch.tensor([expert], dtype=torch.int32, device=dev)
        ensure_experts_scored(cache, layer, q, bump_tok=False, update_state=True, lowpri=True)
        ref.ensure(layer, [expert], lowpri=True)
        victims.append(int(cache.evict_slots[0]))
        _assert_same_tables(cache, ref)
    assert victims[:3] == [13, 14, 15] and victims[3] < 13
    # the band's keys are below 2 << SLOT_BITS; the caps' worst resident score keeps score + 2^31 >= 2
    beta_step, w_q4, _, _ = se.score_params(se.MAX_BETA, se.MAX_W, 1, 1)
    worst = -((se.K_MAX << se.Q) + 1 * beta_step) + ((w_q4 * se.LC_FLOOR) >> 4)
    assert worst + se.SCORE_BIAS >= 2


@_cuda
@pytest.mark.parametrize("policy", ["lru", "rule"])
def test_prefetch_installs_only_into_evictable_slots(policy):
    """With fewer evictable slots than candidates a prefetch installs the first candidates only, and
    with every slot pinned by the last demand call it installs nothing (no all-pinned fallback)."""
    num_experts = 16
    cache, ref = _pf_cache(policy, num_experts, 2, num_experts), _ref(policy, num_experts, 2, num_experts)
    stats = torch.zeros(8, dtype=torch.int64, device="cuda")
    ids = np.arange(12)
    cache.ensure_experts(0, torch.from_numpy(ids.astype(np.int32)).cuda())
    ref.ensure(0, ids, bump_tok=True, lowpri=True)
    pinned = cache.id_of_slot.clone()
    src, dst, _ = _gpu_prefetch(cache, 1, [9, 3, 7, 1, 5, 11, 2], stats)
    assert src.tolist() == [9, 3, 7, 1] and sorted(dst.tolist()) == [12, 13, 14, 15]
    np.testing.assert_array_equal(np.stack(ref.prefetch(1, [9, 3, 7, 1, 5, 11, 2])), np.stack([src, dst]))
    assert torch.equal(cache.id_of_slot[:12], pinned[:12]) and int(stats[0]) == 4
    all16 = np.arange(16)
    cache.ensure_experts(0, torch.from_numpy(all16.astype(np.int32)).cuda())
    before = [t.clone() for t in (cache.id_of_slot, cache.slot_for_id, cache.usage)]
    src, dst, num = _gpu_prefetch(cache, 1, [0, 1, 2], stats)
    assert int(num) == 0 and src.size == 0 and int(stats[0]) == 4
    for a, b in zip(before, (cache.id_of_slot, cache.slot_for_id, cache.usage)):
        assert torch.equal(a, b)


@_cuda
@pytest.mark.parametrize("rows_per_step", [1, 2])
def test_lowpri_lru_without_prefetches_is_flashlib(rows_per_step):
    """Prefetch on routes LRU through the vendored kernel with the low-priority key; while no slot is
    a prefetch it evicts exactly what flashlib's lru_ensure does."""
    num_layers, num_experts, size = 8, 64, 200
    on = _pf_cache("lru", size, num_layers, num_experts)
    off = _cache("lru", size, num_layers, num_experts)
    rows = _zipf_rows(30, rows_per_step, seed=41, num_layers=num_layers, num_experts=num_experts)
    ids = torch.from_numpy(rows.astype(np.int32)).cuda()
    for r in range(rows.shape[0]):
        for layer in range(num_layers):
            a, b = ids[r, layer].clone(), ids[r, layer].clone()
            on.ensure_experts(layer, a)
            off.ensure_experts(layer, b)
            assert torch.equal(a, b)
            assert torch.equal(on.evict_slots[: int(on.num_indices)], off.evict_slots[: int(off.num_indices)])
    for name in ("slot_for_id", "id_of_slot", "usage", "step"):
        assert torch.equal(getattr(on, name), getattr(off, name)), name


def test_lru_policy_leaves_flashlib_only_when_prefetch_is_on(monkeypatch):
    import freetoken.moe.offload_kernels as ok

    calls = []
    monkeypatch.setattr(ok, "lru_ensure", lambda *a, **kw: calls.append("flashlib"))
    monkeypatch.setattr(ok, "ensure_experts_scored", lambda *a, **kw: calls.append(("scored", kw["lowpri"])))
    for mode in ("off", "measure", "on"):
        cache = _cache("lru", 12, 2, 8, device="cpu", prefetch_mode=mode)
        cache.prefetch = SimpleNamespace(mode=mode, count_args=lambda layer: None) if mode != "off" else None
        cache.ensure_experts(1, torch.zeros(2, dtype=torch.int32))
    assert calls == ["flashlib", "flashlib", ("scored", True)]
