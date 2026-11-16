"""P2b integration: CacheManager hybrid path (match_req -> cache_req donate -> prefix hit).
CPU, real LinearStatePool + page_table, hand-built Reqs. Exercises the two-currency wiring
without the full scheduler/engine."""
from __future__ import annotations

from types import SimpleNamespace

import torch
import pytest

from freetoken.core import Req, SamplingParams
from freetoken.kvcache.linear_state_pool import LinearStatePool
from freetoken.models.config import LinearGatedDeltaGroupConfig
from freetoken.scheduler.cache import CacheManager


def _pool(num_slots=16):
    g = LinearGatedDeltaGroupConfig(
        name="linear", layer_ids=(0,), num_key_heads=2, num_value_heads=4,
        key_head_dim=16, value_head_dim=16, conv_kernel_dim=4, output_gate="silu",
    )
    return LinearStatePool(group=g, num_slots=num_slots, dtype=torch.bfloat16,
                           device=torch.device("cpu"), tp_size=1)


def _pend(ids):
    # int32 to match production Req.input_ids dtype (fast_compare_key needs consistent dtype)
    t = torch.tensor(ids, dtype=torch.int32)
    return SimpleNamespace(input_ids=t, input_len=len(ids))


def test_hybrid_cache_manager_donate_then_hit():
    pool = _pool()
    page_table = torch.zeros(4, 64, dtype=torch.int32)
    cm = CacheManager(64, 1, page_table, "hybrid_radix", linear_state_pool=pool)
    assert cm.is_hybrid

    # cold match on an empty tree
    mr = cm.match_req(_pend([1, 2, 3, 4, 5]))
    assert mr.cuda_handle.cached_len == 0 and mr.mamba_value is None

    # admit req A: allocate live + ping-pong, stage KV pages, mark a ×N snapshot at boundary 4
    live, pp = pool.alloc(1)[0], tuple(pool.alloc(2))
    page_table[0, :4] = torch.tensor([100, 101, 102, 103], dtype=torch.int32)
    reqA = Req(input_ids=torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32), table_idx=0,
               cached_len=4, output_len=1, uid=0, sampling_params=SamplingParams(),
               cache_handle=mr.cuda_handle)
    reqA.linear_slot_idx, reqA.mamba_ping_pong = live, pp
    reqA.mamba_next_track_idx = 1            # flipped from 0 in build_fla_metadata; frozen = pp[0]
    reqA.mamba_last_track_seqlen = 4
    cm.lock(mr.cuda_handle)

    free_before = pool.num_free_slots
    cm.cache_req(reqA, finished=False)       # donate pp[0] at boundary 4; replace it in the pair
    # pp[0] donated to the tree; a fresh replacement was alloc'd -> net free-slot count unchanged
    assert pool.num_free_slots == free_before - 1  # one replacement alloc'd (donated slot now tree-owned)
    assert reqA.mamba_ping_pong[0] != pp[0]        # slot 0 replaced; pp[0] now lives in the tree

    # req B shares the [1,2,3,4] prefix -> HIT: returns the donated snapshot + reused KV
    mrB = cm.match_req(_pend([1, 2, 3, 4, 9]))
    assert mrB.cuda_handle.cached_len == 4
    assert mrB.mamba_value == pp[0]
    assert mrB.cuda_handle.get_matched_indices().tolist() == [100, 101, 102, 103]


def test_hybrid_finish_donates_live_slot():
    pool = _pool()
    page_table = torch.zeros(4, 64, dtype=torch.int32)
    cm = CacheManager(64, 1, page_table, "hybrid_radix", linear_state_pool=pool)

    mr = cm.match_req(_pend([7, 8, 9, 10]))
    live, pp = pool.alloc(1)[0], tuple(pool.alloc(2))
    page_table[1, :3] = torch.tensor([200, 201, 202], dtype=torch.int32)
    req = Req(input_ids=torch.tensor([7, 8, 9, 10], dtype=torch.int32), table_idx=1,
              cached_len=3, output_len=1, uid=1, sampling_params=SamplingParams(),
              cache_handle=mr.cuda_handle)
    req.linear_slot_idx, req.mamba_ping_pong = live, pp
    cm.lock(mr.cuda_handle)

    cm.cache_req(req, finished=True)         # donate the live slot directly (final state)
    # ping-pong pair freed; live slot kept (now owned by the tree)
    mr2 = cm.match_req(_pend([7, 8, 9, 10]))
    assert mr2.cuda_handle.cached_len == 3 and mr2.mamba_value == live


def test_free_req_slots_idempotent():
    """C2: a finish/abort double-free of the same request must NOT push its GDN slots twice."""
    pool = _pool()
    pt = torch.zeros(4, 64, dtype=torch.int32)
    cm = CacheManager(64, 1, pt, "hybrid_radix", linear_state_pool=pool)
    live, pp = pool.alloc(1)[0], tuple(pool.alloc(2))
    req = Req(input_ids=torch.tensor([1, 2, 3], dtype=torch.int32), table_idx=0, cached_len=2,
              output_len=1, uid=0, sampling_params=SamplingParams(), cache_handle=None)
    req.linear_slot_idx, req.mamba_ping_pong = live, pp
    base = pool.num_free_slots
    cm._free_req_slots(req)
    assert pool.num_free_slots == base + 3        # live + 2 ping-pong returned once
    cm._free_req_slots(req)                        # second free (abort/finish race)
    assert pool.num_free_slots == base + 3         # idempotent: nothing pushed twice


def test_rebuild_reclaims_donated_gdn_slots():
    """C5: a runtime cache rebuild must return the discarded tree's GDN snapshot slots (idle)."""
    pool = _pool(num_slots=16)
    pt = torch.zeros(4, 64, dtype=torch.int32)
    cm = CacheManager(64, 1, pt, "hybrid_radix", linear_state_pool=pool)
    mr = cm.match_req(_pend([7, 8, 9, 10]))
    live, pp = pool.alloc(1)[0], tuple(pool.alloc(2))
    pt[1, :3] = torch.tensor([200, 201, 202], dtype=torch.int32)
    req = Req(input_ids=torch.tensor([7, 8, 9, 10], dtype=torch.int32), table_idx=1, cached_len=3,
              output_len=1, uid=1, sampling_params=SamplingParams(), cache_handle=mr.cuda_handle)
    req.linear_slot_idx, req.mamba_ping_pong = live, pp
    cm.lock(mr.cuda_handle)
    cm.cache_req(req, finished=True)              # donates `live` to the tree, frees ping-pong
    assert pool.num_free_slots < pool.num_slots - 1   # a slot is now tree-owned
    cm.rebuild(64, pt)                            # idle rebuild discards the tree
    assert pool.num_free_slots == pool.num_slots - 1  # all GDN slots reclaimed (no leak)


def test_prefill_chunk_ends_on_a_page_boundary():
    """A hybrid chunk must end page-aligned: the snapshot commit skips any other boundary."""
    from freetoken.scheduler.prefill import ChunkedReq, PrefillAdder
    from freetoken.scheduler.table import TableManager
    from freetoken.scheduler.utils import PendingReq

    pool = _pool()
    pt = torch.zeros(4, 512, dtype=torch.int32)
    cm = CacheManager(64, 64, pt, "hybrid_radix", linear_state_pool=pool)
    assert cm.prefill_chunk_align == 64
    tm = TableManager(max_running_reqs=4, page_table=pt)
    pending = PendingReq(0, torch.arange(300, dtype=torch.int32), SamplingParams(max_tokens=1))

    adder = PrefillAdder(token_budget=100, reserved_size=0, cache_manager=cm, table_manager=tm)
    req = adder.try_add_one(pending)
    assert isinstance(req, ChunkedReq) and req.extend_len == 64

    # a budget below one page keeps the unaligned chunk rather than stalling the request
    adder = PrefillAdder(token_budget=40, reserved_size=0, cache_manager=cm, table_manager=tm)
    assert adder.try_add_one(pending).extend_len == 40


def test_naive_cache_does_not_align_prefill_chunks():
    """The alignment hook is hybrid-only; every other cache keeps the raw budget chunk."""
    from freetoken.scheduler.prefill import PrefillAdder
    from freetoken.scheduler.table import TableManager
    from freetoken.scheduler.utils import PendingReq

    pt = torch.zeros(4, 512, dtype=torch.int32)
    cm = CacheManager(64, 64, pt, "radix")
    assert cm.prefill_chunk_align == 1
    tm = TableManager(max_running_reqs=4, page_table=pt)
    adder = PrefillAdder(token_budget=100, reserved_size=0, cache_manager=cm, table_manager=tm)
    pending = PendingReq(0, torch.arange(300, dtype=torch.int32), SamplingParams(max_tokens=1))
    assert adder.try_add_one(pending).extend_len == 100


@pytest.mark.parametrize("policy", ["chunk", "structural"])
@pytest.mark.parametrize("full_chunks", [2, 3])
@pytest.mark.parametrize("tail", [1, 59, 64, 65])
@pytest.mark.parametrize("finish_early", [False, True])
def test_chunked_prefill_retains_resumable_snapshot(monkeypatch, full_chunks, tail, finish_early, policy):
    """A short final extend must retain the prior state; a longer one must replace it. A short
    prompt has no structural point; "structural" only keeps the final chunk long enough to
    track, so a short tail resumes at the deepest x64 boundary, not one page before it."""
    import freetoken.core as core
    from freetoken.attention.linear import build_fla_metadata
    from freetoken.scheduler.decode import DecodeManager
    from freetoken.scheduler.prefill import ChunkedReq, PrefillManager
    from freetoken.scheduler.scheduler import Scheduler
    from freetoken.scheduler.table import TableManager
    from freetoken.scheduler.utils import PendingReq

    pool = _pool()
    pt = torch.zeros(2, 1024, dtype=torch.int32)
    cm = CacheManager(32, 64, pt, "hybrid_radix", linear_state_pool=pool, snapshot_policy=policy)
    tm = TableManager(max_running_reqs=1, page_table=pt)
    pm = PrefillManager(cm, tm, DecodeManager(page_size=64))
    monkeypatch.setattr(core, "_GLOBAL_CTX", core.Context(page_size=64, linear_state_pool=pool))
    prompt = torch.arange(full_chunks * 128 + tail, dtype=torch.int32)
    pm.pending_list = [PendingReq(0, prompt, SamplingParams(max_tokens=8))]
    last_batch = None
    states = {}
    final = None

    while pm.runnable or last_batch is not None:
        # Match overlap ordering: prepare the continuation before draining the prior chunk.
        batch = pm.schedule_next_batch(128)
        if batch is not None:
            batch.padded_reqs = batch.reqs
            cm.allocate_paged(batch.reqs)
            req = batch.reqs[0]
            meta = build_fla_metadata(batch, torch.device("cpu"), pool)
            if meta.track_dst is not None:
                slot = meta.track_dst.item()
                value = len(states) + 1
                states[req.mamba_last_track_seqlen] = value
                pool.recurrent_states[:, slot].fill_(value)
                pool.conv_states[:, slot].fill_(value)
            pool.recurrent_states[:, req.linear_slot_idx].fill_(-9)
            pool.conv_states[:, req.linear_slot_idx].fill_(-9)
            req.complete_one()
        if last_batch is not None:
            previous = last_batch.reqs[0]
            if isinstance(previous, ChunkedReq):
                cm.commit_chunk(previous)  # as Scheduler._process_last_data
            else:
                final = previous
                if not finish_early:
                    cm.cache_req(final, finished=False)
        last_batch = batch

    # Before the finish path can donate an aligned live state, require the frozen checkpoint.
    short = full_chunks * 128 if policy == "structural" else full_chunks * 128 - 64
    expected = short if tail <= 64 else full_chunks * 128 + 64
    if not finish_early:
        match = cm.match_req(_pend(prompt.tolist() + [999]))
        assert match.cuda_handle.cached_len == expected
        assert torch.all(pool.recurrent_states[:, match.mamba_value] == states[expected])
        assert torch.all(pool.conv_states[:, match.mamba_value] == states[expected])

    # Finish/abort may occur before the prefill commit; use the real idempotent cleanup path.
    stub = SimpleNamespace(cache_manager=cm, table_manager=tm, spec=None)
    Scheduler._free_req_resources(stub, final)
    Scheduler._free_req_resources(stub, final)
    match = cm.match_req(_pend(prompt.tolist() + [999]))
    expected_finish = len(prompt) if tail == 64 else expected
    assert match.cuda_handle.cached_len == expected_finish
    value = -9 if tail == 64 else states[expected]
    assert torch.all(pool.recurrent_states[:, match.mamba_value] == value)
    assert torch.all(pool.conv_states[:, match.mamba_value] == value)
    assert tm.available_size == 1
    cm.check_integrity()
    assert pool.num_free_slots + cm.prefix_cache.mamba_evictable_size == pool.num_slots - 1
    cm.ensure_mamba_slots(pool.num_slots - 1)
    cm.check_integrity()
    assert pool.num_free_slots == pool.num_slots - 1
    assert len(cm.free_slots) == cm.num_pages


@pytest.mark.parametrize("radix_lookahead", [False, True])   # MTP keys position i by (t_i, t_i+1)
@pytest.mark.parametrize("overlap", [True, False])            # no spec / DFlash and MTP (normal_loop)
@pytest.mark.parametrize("policy", ["chunk", "structural"])
@pytest.mark.parametrize("prompt_len", [213_000, 524_288])    # a long agent prompt; YaRN's whole context
def test_snapshot_placement_on_a_cold_long_prefill(monkeypatch, prompt_len, policy, overlap, radix_lookahead):
    """Design P1a (3.4). "structural" tracks and donates only at the first turn boundary (pinned),
    each 32,768 tokens and the prompt end, keeps at most 2 snapshots on its path and takes at most 2
    idle snapshots of other sessions; "chunk" (today) donates once. Every track lies at most 64
    tokens before its chunk end, so the DFlash window copied with it holds its own rows, and the
    finished request resumes at the same length under both policies."""
    from freetoken import core
    from freetoken.attention.linear import build_fla_metadata
    from freetoken.scheduler.decode import DecodeManager
    from freetoken.scheduler.prefill import ChunkedReq, PrefillManager
    from freetoken.scheduler.scheduler import Scheduler
    from freetoken.scheduler.table import TableManager
    from freetoken.scheduler.utils import PendingReq

    ps, budget, turn, others = 64, 8192, 7, 12
    pool = _pool(num_slots=others + 4)                # the others' snapshots + the 3 working slots
    pages = prompt_len // ps + others + 8
    pt = torch.zeros(2, pages * ps, dtype=torch.int32)
    cm = CacheManager(pages, ps, pt, "hybrid_radix", linear_state_pool=pool,
                      radix_lookahead=radix_lookahead, snapshot_policy=policy)
    cm.turn_end_id = turn
    monkeypatch.setattr(core, "_GLOBAL_CTX", core.Context(page_size=ps, linear_state_pool=pool))
    sessions = [torch.arange(60_000 + 200 * i, 60_000 + 200 * i + 2 * ps, dtype=torch.int32) for i in range(others)]
    for ids in sessions:                              # older idle sessions, one snapshot each
        cm.prefix_cache.insert(cm._keys(ids, ps), cm._page_to_token(cm._allocate(1)), pool.alloc(1)[0])
    prompt = torch.arange(prompt_len, dtype=torch.int32) % 50_000 + 100
    prompt[4_999] = turn                              # the system block ends at 5,000

    def resume(ids, n):
        return cm.match_req(_pend(ids[:n].tolist() + [12345])).cuda_handle.cached_len

    pc = cm.prefix_cache
    tm = TableManager(max_running_reqs=1, page_table=pt)
    pm = PrefillManager(cm, tm, DecodeManager(page_size=ps))
    pm.pending_list = [PendingReq(0, prompt, SamplingParams(max_tokens=8))]
    tracked, tails, tree_max, final, pending = [], [], 0, None, None
    while pm.runnable or pending is not None:
        batch = pm.schedule_next_batch(budget)
        if batch is not None:
            batch.padded_reqs = batch.reqs
            cm.allocate_paged(batch.reqs)
            (req,) = batch.reqs
            if build_fla_metadata(batch, torch.device("cpu"), pool).track_dst is not None:
                tracked.append((req.mamba_last_track_seqlen, req.device_len))
            req.complete_one()
        drained, pending = ([pending], batch) if overlap else ([batch], None)
        for done in filter(None, drained):            # as Scheduler._process_last_data
            (r,) = done.reqs
            if isinstance(r, ChunkedReq):
                cm.commit_chunk(r)
            else:
                cm.cache_req(r, finished=False)
                final = r
            tails.append(resume(prompt, r.cached_len))
            tree_max = max(tree_max, pc.mamba_evictable_size + pc.mamba_protected)
    Scheduler._free_req_resources(SimpleNamespace(cache_manager=cm, table_manager=tm, spec=None), final)

    alive = sum(resume(ids, len(ids)) == ps for ids in sessions)
    # the prompt end's track. "chunk" (today): 213,000 ends on an 8-token chunk, so the chunk before
    # tracks 64 tokens earlier; "structural" leaves the final chunk long enough to track the deepest
    end = {213_000: 212_928 if policy == "chunk" else 212_992, 524_288: 524_224}[prompt_len]
    assert all(0 < e - b <= ps for b, e in tracked)
    # the final state rides on an aligned end; pair keys need the token after it
    live = prompt_len % ps == 0 and not radix_lookahead
    assert resume(prompt, prompt_len) == (prompt_len if live else end)
    if policy == "chunk":
        assert set(tails) - {0} == {end} and others - alive == 1
        assert resume(prompt, 5_000) == 0
    else:
        points = [4_992, *(k * 32_768 - ps for k in range(1, -(-prompt_len // 32_768))), end]
        assert [b for b, _ in tracked] == points       # no track anywhere else
        assert sorted(set(tails)) == points            # each point was donated and became the tail
        assert tree_max - alive <= 2                   # path snapshots after every commit
        assert others - alive == 2
        assert resume(prompt, 5_000) == 4_992          # the pinned root survived the cap
    cm.check_integrity()
    assert pc.full_protected == 0 and pc.mamba_protected == 0
    assert pool.num_free_slots + pc.mamba_evictable_size == pool.num_slots - 1


@pytest.mark.parametrize("radix_lookahead", [False, True])
@pytest.mark.parametrize("overlap", [True, False])
@pytest.mark.parametrize("policy", ["chunk", "structural"])
# next turn: "anchor" rewrites the tool call after its opener, "prompt_end" drops the reasoning
# (both with --enable-special-token-ckpt), "resend" sends the generation back with no anchor
# (production at c323e469: no anchor, and an unaligned finish frees the generation's KV)
@pytest.mark.parametrize("diverge", ["anchor", "prompt_end", "resend"])
def test_snapshots_per_path_over_agent_turns(monkeypatch, policy, overlap, radix_lookahead, diverge):
    """Each turn adds its prompt end and, at finish, its tool-call anchor. "chunk" keeps them all;
    the per-path cap thins the old turn at the next turn's first donation, never at a finish, so
    the next turn still resumes at the anchor or the prompt end. A fork stays: when the reasoning is
    dropped, the old prompt end branches into the old generation and the new prompt."""
    from freetoken import core
    from freetoken.attention.linear import build_fla_metadata
    from freetoken.scheduler.decode import DecodeManager
    from freetoken.scheduler.prefill import ChunkedReq, PrefillManager
    from freetoken.scheduler.scheduler import Scheduler
    from freetoken.scheduler.table import TableManager
    from freetoken.scheduler.utils import PendingReq

    ps, turns, turn = 64, 5, 7
    pool = _pool(num_slots=2 * turns + 6)
    pt = torch.zeros(2, 8192, dtype=torch.int32)
    cm = CacheManager(128, ps, pt, "hybrid_radix", linear_state_pool=pool,
                      radix_lookahead=radix_lookahead, snapshot_policy=policy)
    cm.turn_end_id = turn
    monkeypatch.setattr(core, "_GLOBAL_CTX", core.Context(page_size=ps, linear_state_pool=pool))
    tm = TableManager(max_running_reqs=1, page_table=pt)
    pm = PrefillManager(cm, tm, DecodeManager(page_size=ps))
    stub = SimpleNamespace(cache_manager=cm, table_manager=tm, spec=None)
    prompt = torch.arange(100, 800, dtype=torch.int32)
    prompt[319] = turn                                 # the system block ends at 320
    resumed, anchors, ends = [], [], []
    for k in range(turns):
        pm.pending_list = [PendingReq(k, prompt, SamplingParams(max_tokens=256, ignore_eos=True))]
        final, pending = None, None
        while pm.runnable or pending is not None:
            batch = pm.schedule_next_batch(256)
            if batch is not None:
                batch.padded_reqs = batch.reqs
                cm.allocate_paged(batch.reqs)
                if build_fla_metadata(batch, torch.device("cpu"), pool).track_dst is not None:
                    last_track = batch.reqs[0].mamba_last_track_seqlen
                if batch.prompt_admissions:
                    resumed.append(batch.prompt_admissions[0][2])
                batch.reqs[0].complete_one()
            drained, pending = ([pending], batch) if overlap else ([batch], None)
            for done in filter(None, drained):
                (r,) = done.reqs
                if isinstance(r, ChunkedReq):
                    cm.commit_chunk(r)
                else:
                    cm.cache_req(r, finished=False)
                    final = r
        # decode; the tool-call opener closes a page, so the anchor snapshot is page-aligned
        anchor = (len(prompt) // ps + 2) * ps
        final.toolcall_anchor_len = None if diverge == "resend" else anchor
        while final.cached_len < anchor + 40:
            final.append_host(torch.tensor([20_000 + 1_000 * k + final.cached_len % 1_000], dtype=torch.int32))
            cm.allocate_paged([final])
            cm.snapshot_toolcall_anchor([final])
            final.complete_one()
        anchors.append(anchor)
        ends.append(last_track)
        keep = {"anchor": anchor + 8, "prompt_end": len(prompt), "resend": final.input_ids.numel()}[diverge]
        conversation = final.input_ids[:keep].clone()
        Scheduler._free_req_resources(stub, final)
        # the client rewrites the echoed tool call after its first tokens, adds the tool result and a new message
        prompt = torch.cat([conversation, torch.arange(40_000 + 300 * k, 40_000 + 300 * k + 300, dtype=torch.int32)])

    assert resumed == [0, *(anchors if diverge == "anchor" else ends)[:-1]]
    pc = cm.prefix_cache
    snapshots = pc.mamba_evictable_size + pc.mamba_protected
    if diverge == "resend":    # one prompt end per turn; "structural": the pinned root and the last
        assert snapshots == (turns if policy == "chunk" else 2)
    elif diverge == "anchor":  # a prompt end and an anchor per turn; the root and the last two
        assert snapshots == (2 * turns if policy == "chunk" else 3)
    else:                      # every old prompt end is a fork; each old generation keeps its anchor leaf
        assert snapshots == 2 * turns + (policy == "structural")
    cm.check_integrity()
    assert pool.num_free_slots + pc.mamba_evictable_size == pool.num_slots - 1


@pytest.mark.parametrize("radix_lookahead", [False, True])
@pytest.mark.parametrize("overlap", [True, False])
@pytest.mark.parametrize("policy", ["chunk", "structural"])
def test_fork_snapshot_and_pinned_roots(monkeypatch, policy, overlap, radix_lookahead):
    """A prompt that leaves a cached path donates a snapshot at the branch point, which serves a
    third session and which the cap keeps as a fork. At most 4 first-turn roots stay pinned (the
    least recently used loses its pin), and evict_mamba takes them only after every other one."""
    from freetoken import core
    from freetoken.attention.linear import build_fla_metadata
    from freetoken.scheduler.decode import DecodeManager
    from freetoken.scheduler.prefill import ChunkedReq, PrefillManager
    from freetoken.scheduler.scheduler import Scheduler
    from freetoken.scheduler.table import TableManager
    from freetoken.scheduler.utils import PendingReq

    ps, turn = 64, 7
    pool = _pool(num_slots=24)
    pt = torch.zeros(2, 2048, dtype=torch.int32)
    cm = CacheManager(256, ps, pt, "hybrid_radix", linear_state_pool=pool,
                      radix_lookahead=radix_lookahead, snapshot_policy=policy)
    cm.turn_end_id = turn
    monkeypatch.setattr(core, "_GLOBAL_CTX", core.Context(page_size=ps, linear_state_pool=pool))
    tm = TableManager(max_running_reqs=1, page_table=pt)
    pm = PrefillManager(cm, tm, DecodeManager(page_size=ps))
    stub = SimpleNamespace(cache_manager=cm, table_manager=tm, spec=None)
    a = torch.arange(1_000, 2_000, dtype=torch.int32)
    a[199] = turn                                      # a's system block ends at 200
    b = torch.cat([a[:600], torch.arange(5_000, 5_400, dtype=torch.int32)])
    c = torch.cat([a[:600], torch.arange(6_000, 6_400, dtype=torch.int32)])
    systems = []                                       # five more sessions, each with its own system block
    for i in range(5):
        d = torch.arange(10_000 + 1_000 * i, 10_000 + 1_000 * i + 300, dtype=torch.int32)
        d[99] = turn
        systems.append(d)
    resumed = []
    for uid, prompt in enumerate([a, b, c, *systems]):
        pm.pending_list = [PendingReq(uid, prompt, SamplingParams(max_tokens=4))]
        final, pending = None, None
        while pm.runnable or pending is not None:
            batch = pm.schedule_next_batch(8192)
            if batch is not None:
                batch.padded_reqs = batch.reqs
                cm.allocate_paged(batch.reqs)
                build_fla_metadata(batch, torch.device("cpu"), pool)
                if batch.prompt_admissions:
                    resumed.append(batch.prompt_admissions[0][2])
                batch.reqs[0].complete_one()
            drained, pending = ([pending], batch) if overlap else ([batch], None)
            for done in filter(None, drained):
                (r,) = done.reqs
                if isinstance(r, ChunkedReq):
                    cm.commit_chunk(r)
                else:
                    cm.cache_req(r, finished=False)
                    final = r
        Scheduler._free_req_resources(stub, final)

    pc = cm.prefix_cache
    # b resumes at a's root (structural) and leaves a's path at 576; c resumes at that branch point
    assert resumed[:3] == ([0, 192, 576] if policy == "structural" else [0, 0, 0])
    if policy == "structural":
        assert len(pc.pinned) == 4 and pc.counts["unpin"] == 2   # a's root and the first system's lost the pin
        # no probe before the eviction: a match stamps its path, and LRU alone would then spare the roots
        unpinned = pc.mamba_evictable_size - 4
        cm.ensure_mamba_slots(pool.num_free_slots + unpinned)
        assert pc.counts["victim_pinned"] == 0 and pc.mamba_evictable_size == 4
        assert [cm.match_req(_pend(d[:100].tolist())).cuda_handle.cached_len for d in systems] == [0] + [64] * 4
        cm.ensure_mamba_slots(pool.num_free_slots + 1)
        assert pc.counts["victim_pinned"] == 1
    cm.check_integrity()
    assert pool.num_free_slots + pc.mamba_evictable_size == pool.num_slots - 1


def test_pool_sizing_covers_4mr_floor():
    """C6: pool must reserve the 4-slot-per-request non-evictable floor even at a tiny ratio."""
    from types import SimpleNamespace
    from freetoken.kvcache.linear_state_pool import _linear_pool_num_slots
    for mr in (1, 8, 64):
        c = SimpleNamespace(max_running_req=mr, cache_type="hybrid_radix",
                            linear_state_cache_ratio=0.1)
        assert _linear_pool_num_slots(c) >= 4 * mr + 1, (mr, _linear_pool_num_slots(c))


def _finish_read(policy: str, sp: SamplingParams, *, warm_prefix: bool = False, num_pages: int = 256,
                 isolate: bool = False, radix_lookahead: bool = False):
    """A 130-token prompt finishing on its prefill forward, with a frozen x64 snapshot at 128
    (the deepest boundary strictly inside it) and its live state at 130. ``warm_prefix`` first
    donates a 64-token prefix so the read resumes from a locked prefix-cache hit."""
    pool = _pool()
    pt = torch.zeros(4, 256, dtype=torch.int32)
    cm = CacheManager(num_pages, 1, pt, "hybrid_radix", linear_state_pool=pool,
                      radix_lookahead=radix_lookahead,
                      logprob_read_snapshot_donate=policy, isolate_logprob_reads=isolate)
    prompt = list(range(1, 131))
    if warm_prefix:
        warm = Req(input_ids=torch.tensor(prompt[:64] + [999], dtype=torch.int32), table_idx=1,
                   cached_len=64, output_len=1, uid=9, sampling_params=SamplingParams(),
                   cache_handle=cm.match_req(_pend(prompt[:64])).cuda_handle)
        pt[1, :64] = cm._allocate(64)
        warm.linear_slot_idx, warm.mamba_ping_pong = pool.alloc(1)[0], tuple(pool.alloc(2))
        cm.lock(warm.cache_handle)
        cm.cache_req(warm, finished=True)
    mr = cm.match_req(_pend(prompt))
    hit = mr.cuda_handle.cached_len
    pt[0, :hit] = mr.cuda_handle.get_matched_indices()
    pt[0, hit:130] = cm._allocate(130 - hit)
    req = Req(input_ids=torch.tensor(prompt + [7], dtype=torch.int32), table_idx=0, cached_len=130,
              output_len=1, uid=0, sampling_params=sp, cache_handle=mr.cuda_handle)
    req.linear_slot_idx, req.mamba_ping_pong = pool.alloc(1)[0], tuple(pool.alloc(2))
    req.mamba_next_track_idx, req.mamba_last_track_seqlen = 1, 128
    cm.lock(mr.cuda_handle)
    cm.cache_req(req, finished=True)
    _assert_no_leak(cm, num_pages)
    return cm, prompt


def _assert_no_leak(cm, num_pages=256):
    pc, pool = cm.prefix_cache, cm.linear_state_pool
    assert len(cm.free_slots) + pc.full_evictable + pc.full_protected == num_pages  # every page is free or tree-owned
    assert pc.full_protected == 0 and pc.mamba_protected == 0  # nothing left locked
    assert pool.num_free_slots + pc.mamba_evictable_size == pool.num_slots - 1  # no GDN slot leak
    cm.check_integrity()


def _resume_len(cm, prompt, n):
    return cm.match_req(_pend(prompt[:n] + [12345])).cuda_handle.cached_len


READ = SamplingParams(max_tokens=1, logprobs=True, top_logprobs=4, allowed_token_ids=[32, 33])


@pytest.mark.parametrize("warm_prefix", [False, True])
@pytest.mark.parametrize(
    ("policy", "snapshots", "resume_at_128", "resume_at_130"),
    [("all", 2, 128, 130), ("boundary", 1, 128, 128), ("none", 0, 0, 0)],
)
def test_logprob_read_snapshot_donate_policy(policy, snapshots, resume_at_128, resume_at_130, warm_prefix):
    # A burst of AnyJev reads must not evict chat users' GDN snapshots: "boundary" and "none"
    # donate less, and a node no snapshot can resume is never inserted.
    cm, prompt = _finish_read(policy, READ, warm_prefix=warm_prefix)
    warm = 1 if warm_prefix else 0  # the warm-up request's own 64-token snapshot
    assert cm.prefix_cache.mamba_evictable_size == snapshots + warm
    assert _resume_len(cm, prompt, 128) == max(resume_at_128, 64 * warm)
    assert _resume_len(cm, prompt, 130) == max(resume_at_130, 64 * warm)


@pytest.mark.parametrize("policy", ["boundary", "none"])
@pytest.mark.parametrize(
    "sp",
    [
        SamplingParams(max_tokens=1),  # no logprobs
        SamplingParams(max_tokens=2, logprobs=True, top_logprobs=4),  # not a single-token read
    ],
)
def test_snapshot_donate_policy_leaves_other_requests_alone(policy, sp):
    cm, prompt = _finish_read(policy, sp)
    assert cm.prefix_cache.mamba_evictable_size == 2
    assert _resume_len(cm, prompt, 130) == 130


@pytest.mark.parametrize(
    ("policy", "isolate", "snapshots"),
    [("all", False, 2), ("boundary", False, 1), ("none", False, 0), ("all", True, 0)],
)
def test_read_donate_keeps_pages_and_slots_whole_with_radix_lookahead(policy, isolate, snapshots):
    # production runs MTP, whose radix keys are (token, next token) pairs
    cm, _ = _finish_read(policy, READ, isolate=isolate, radix_lookahead=True)
    assert cm.prefix_cache.mamba_evictable_size == snapshots


@pytest.mark.parametrize("isolate", [False, True])
def test_isolated_logprob_read_ignores_and_keeps_out_of_the_prefix_cache(isolate):
    """An identical earlier read donated under "all"; an isolated read still prefills cold and stores nothing."""
    from freetoken.scheduler.decode import DecodeManager
    from freetoken.scheduler.prefill import PrefillManager
    from freetoken.scheduler.table import TableManager
    from freetoken.scheduler.utils import PendingReq

    cm, prompt = _finish_read("all", READ, num_pages=512)  # room for both reads: no eviction
    cm.isolate_logprob_reads = isolate
    tree = (cm.prefix_cache.full_evictable, cm.prefix_cache.mamba_evictable_size)
    pm = PrefillManager(cm, TableManager(max_running_reqs=2, page_table=cm.page_table),
                        DecodeManager(page_size=1), isolate_logprob_reads=isolate)
    pm.pending_list = [PendingReq(1, torch.tensor(prompt, dtype=torch.int32), READ)]
    batch = pm.schedule_next_batch(4096)
    (req,) = batch.reqs
    assert req.cache_handle.cached_len == batch.log_cached_tokens == (0 if isolate else 128)

    cm.allocate_paged(batch.reqs)
    req.complete_one()
    if isolate:  # a cold 130-token extend tracks the x64 boundary at 128 (build_fla_metadata)
        req.mamba_next_track_idx, req.mamba_last_track_seqlen = 1, 128
    cm.cache_req(req, finished=True)
    if isolate:
        assert (cm.prefix_cache.full_evictable, cm.prefix_cache.mamba_evictable_size) == tree
    _assert_no_leak(cm, num_pages=512)


@pytest.mark.parametrize(("policy", "isolate"), [("none", False), ("all", True), ("boundary", True)])
def test_non_donating_read_chunk_commit_leaves_the_tree_alone(policy, isolate):
    """The prefill-commit branch donated the x64 snapshot whatever the read policy said."""
    num_pages, pool = 256, _pool()
    pt = torch.zeros(4, 256, dtype=torch.int32)
    cm = CacheManager(num_pages, 1, pt, "hybrid_radix", linear_state_pool=pool,
                      logprob_read_snapshot_donate=policy, isolate_logprob_reads=isolate)
    prompt = list(range(1, 131))
    mr = cm.match_req(_pend(prompt))
    pt[0, :130] = cm._allocate(130)
    row = pt[0, :130].clone()
    req = Req(input_ids=torch.tensor(prompt + [7], dtype=torch.int32), table_idx=0, cached_len=130,
              output_len=1, uid=0, sampling_params=READ, cache_handle=mr.cuda_handle)
    req.linear_slot_idx, req.mamba_ping_pong = pool.alloc(1)[0], tuple(pool.alloc(2))
    req.mamba_next_track_idx, req.mamba_last_track_seqlen = 1, 128
    cm.lock(mr.cuda_handle)

    cm.cache_req(req, finished=False)
    assert cm.prefix_cache.full_evictable + cm.prefix_cache.full_protected == 0
    assert cm.prefix_cache.mamba_evictable_size + cm.prefix_cache.mamba_protected == 0
    assert torch.equal(pt[0, :130], row)  # the request still owns its pages

    cm.cache_req(req, finished=True)
    assert len(cm.free_slots) == num_pages
    _assert_no_leak(cm)


def test_kvstats_classifies_admissions_and_snapshot_victims():
    """P0: each admission's walk before and after the snapshot cut, an evicted continuation of
    the prompt, and what evict_mamba takes, counted on the admission's own walk."""
    from freetoken.scheduler.decode import DecodeManager
    from freetoken.scheduler.prefill import PrefillManager
    from freetoken.scheduler.table import TableManager
    from freetoken.scheduler.utils import PendingReq

    pool = _pool(num_slots=32)
    pt = torch.zeros(5, 1024, dtype=torch.int32)
    cm = CacheManager(64, 64, pt, "hybrid_radix", linear_state_pool=pool)
    tree = cm.prefix_cache
    x = torch.arange(1, 257, dtype=torch.int32)        # snapshots at 128 and 256
    y = torch.arange(1001, 1257, dtype=torch.int32)    # the same shape
    for ids in (x, y):
        kv = cm._page_to_token(cm._allocate(4))
        tree.insert(ids[:128], kv[:128], pool.alloc(1)[0])
        tree.insert(ids, kv, pool.alloc(1)[0])
    y_leaf = tree.match_prefix(y).node
    tree.inc_lock(y_leaf)
    x_leaf = tree.match_prefix(x).node
    tree.inc_lock(x_leaf)
    pool.free(cm.prefix_cache.evict_mamba(2).mamba_slots)   # both interior snapshots: tombstones
    tree.dec_lock(x_leaf)
    ev = tree.evict_mamba(1)                                # x's leaf, and its tombstone parent with it
    pool.free(ev.mamba_slots)
    cm._free(ev.kv_indices)
    tree.dec_lock(y_leaf)
    # the leaf victim took its KV and the tombstone above it: 256 tokens
    assert (tree.counts["victim_interior"], tree.counts["victim_leaf"], tree.counts["victim_kv_tokens"]) == (2, 1, 256)

    def pend(uid, ids):
        return PendingReq(uid, torch.cat([ids, torch.tensor([7], dtype=torch.int32)]),
                          SamplingParams(max_tokens=4))

    pm = PrefillManager(cm, TableManager(max_running_reqs=4, page_table=pt), DecodeManager(page_size=64))
    pm.pending_list = [
        pend(1, torch.cat([x[:200], torch.full((60,), 5, dtype=torch.int32)])),  # x was evicted
        pend(2, torch.cat([y[:192], torch.full((70,), 6, dtype=torch.int32)])),  # KV to 192, no snapshot
        pend(3, y),                                                               # full hit at 256
        pend(4, torch.arange(5001, 5300, dtype=torch.int32)),                     # never seen
    ]
    batch = pm.schedule_next_batch(4096)
    assert [r.cache_handle.cached_len for r in batch.reqs] == [0, 0, 256, 0]

    s = cm.stats
    assert (s.admits, s.walk_tokens, s.kept_tokens) == (4, 192 + 256, 256)
    assert (s.cut_admits, s.cut_tokens) == (1, 192)
    assert (s.evicted_admits, s.evicted_tokens) == (1, 128)   # the cascaded parent [0, 128)
    assert s.cold_admits == 2
    line = cm.kv_stats_line()
    assert line.startswith("KVSTATS ") and "victim_leaf=1" in line
    assert cm.stats.admits == 0 and not tree.counts           # the line resets the window


@pytest.mark.parametrize("cause", ["kv", "gdn", "table"])
def test_kvstats_counts_admission_refusals_by_cause(cause):
    """A queued request is refused again on every pass: count the events and the distinct
    requests per cause, so a KV-reservation wait (design 7.2) is not read as a slot wait."""
    from freetoken.scheduler.decode import DecodeManager
    from freetoken.scheduler.prefill import PrefillManager
    from freetoken.scheduler.table import TableManager
    from freetoken.scheduler.utils import PendingReq

    pool = _pool(num_slots=16)
    pt = torch.zeros(3, 256, dtype=torch.int32)
    cm = CacheManager(8 if cause == "kv" else 64, 1, pt, "hybrid_radix", linear_state_pool=pool)
    tm = TableManager(max_running_reqs=2, page_table=pt)
    if cause == "table":
        tm.allocate(), tm.allocate()
    if cause == "gdn":
        pool.alloc(pool.num_free_slots - 2)
    pm = PrefillManager(cm, tm, DecodeManager(page_size=1))
    pm.pending_list = [PendingReq(5, torch.arange(1, 9, dtype=torch.int32), SamplingParams(max_tokens=4))]
    for _ in range(2):
        assert pm.schedule_next_batch(4096) is None
    assert cm.stats.refusals == {cause: (2, {5})}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
