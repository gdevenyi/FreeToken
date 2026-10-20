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


@pytest.mark.parametrize("full_chunks", [2, 3])
@pytest.mark.parametrize("tail", [1, 59, 64, 65])
@pytest.mark.parametrize("finish_early", [False, True])
def test_chunked_prefill_retains_resumable_snapshot(monkeypatch, full_chunks, tail, finish_early):
    """A short final extend must retain the prior state; a longer one must replace it."""
    import freetoken.core as core
    from freetoken.attention.linear import build_fla_metadata
    from freetoken.scheduler.decode import DecodeManager
    from freetoken.scheduler.prefill import ChunkedReq, PrefillManager
    from freetoken.scheduler.scheduler import Scheduler
    from freetoken.scheduler.table import TableManager
    from freetoken.scheduler.utils import PendingReq

    pool = _pool()
    pt = torch.zeros(2, 1024, dtype=torch.int32)
    cm = CacheManager(32, 64, pt, "hybrid_radix", linear_state_pool=pool)
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
            meta = build_fla_metadata(batch, torch.device("cpu"))
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
            if not isinstance(previous, ChunkedReq):
                final = previous
                if not finish_early:
                    cm.cache_req(final, finished=False)
        last_batch = batch

    # Before the finish path can donate an aligned live state, require the frozen checkpoint.
    expected = full_chunks * 128 - 64 if tail <= 64 else full_chunks * 128 + 64
    if not finish_early:
        match = cm.match_req(_pend(prompt.tolist() + [999]))
        assert match.cuda_handle.cached_len == expected
        assert torch.all(pool.recurrent_states[:, match.mamba_value] == states[expected])
        assert torch.all(pool.conv_states[:, match.mamba_value] == states[expected])

    # Finish/abort may occur before the prefill commit; use the real idempotent cleanup path.
    stub = SimpleNamespace(cache_manager=cm, table_manager=tm, _prefill_start={})
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


def test_pool_sizing_covers_4mr_floor():
    """C6: pool must reserve the 4-slot-per-request non-evictable floor even at a tiny ratio."""
    from types import SimpleNamespace
    from freetoken.kvcache.linear_state_pool import _linear_pool_num_slots
    for mr in (1, 8, 64):
        c = SimpleNamespace(max_running_req=mr, cache_type="hybrid_radix",
                            linear_state_cache_ratio=0.1)
        assert _linear_pool_num_slots(c) >= 4 * mr + 1, (mr, _linear_pool_num_slots(c))


def test_host_snapshot_tier_lowers_the_gpu_snapshot_floor(monkeypatch):
    """With FT_GDN_HOST_TIER evicted snapshots live in RAM, so one GPU snapshot slot is enough."""
    from types import SimpleNamespace
    from freetoken.kvcache.linear_state_pool import _linear_pool_num_slots
    c = SimpleNamespace(max_running_req=1, cache_type="hybrid_radix", linear_state_cache_ratio=0.0)
    monkeypatch.delenv("FT_GDN_HOST_TIER", raising=False)
    assert _linear_pool_num_slots(c) == 4 + 4 + 1
    monkeypatch.setenv("FT_GDN_HOST_TIER", "1")
    assert _linear_pool_num_slots(c) == 4 + 1 + 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))


def _finish_read(policy: str, sp: SamplingParams, *, warm_prefix: bool = False, num_pages: int = 256):
    """A 130-token prompt finishing on its prefill forward, with a frozen x64 snapshot at 128
    (the deepest boundary strictly inside it) and its live state at 130. ``warm_prefix`` first
    donates a 64-token prefix so the read resumes from a locked prefix-cache hit."""
    pool = _pool()
    pt = torch.zeros(4, 256, dtype=torch.int32)
    cm = CacheManager(num_pages, 1, pt, "hybrid_radix", linear_state_pool=pool,
                      logprob_read_snapshot_donate=policy)
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

    pc = cm.prefix_cache
    tree_pages = pc.full_evictable + pc.full_protected
    assert len(cm.free_slots) + tree_pages == num_pages  # every page is free or tree-owned
    assert pc.full_protected == 0 and pc.mamba_protected == 0  # nothing left locked
    assert pool.num_free_slots + pc.mamba_evictable_size == pool.num_slots - 1  # no GDN slot leak
    cm.check_integrity()
    return cm, prompt


def _resume_len(cm, prompt, n):
    return cm.match_req(_pend(prompt[:n] + [12345])).cuda_handle.cached_len


READ = SamplingParams(max_tokens=1, logprobs=True, top_logprobs=4, allowed_token_ids=[32, 33])


@pytest.mark.parametrize("warm_prefix", [False, True])
@pytest.mark.parametrize(
    ("policy", "snapshots", "resume_at_128", "resume_at_130"),
    [("all", 2, 128, 130), ("boundary", 1, 128, 128), ("none", 0, 0, 0)],
)
def test_logprob_read_snapshot_donate_policy(policy, snapshots, resume_at_128, resume_at_130, warm_prefix):
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


def test_snapshot_donate_policy_is_validated():
    with pytest.raises(ValueError, match="snapshot donate"):
        CacheManager(8, 1, torch.zeros(1, 8, dtype=torch.int32), "hybrid_radix",
                     linear_state_pool=_pool(), logprob_read_snapshot_donate="some")


def test_snapshot_donate_policy_is_a_server_arg():
    from unittest.mock import patch

    from freetoken.server.args import parse_args

    config = SimpleNamespace(to_dict=lambda: {"architectures": ["LlamaForCausalLM"], "torch_dtype": "bfloat16"})
    with patch("freetoken.utils.cached_load_hf_config", lambda _path: config):
        assert parse_args(["--model", "/models/anon"])[0].logprob_read_snapshot_donate == "all"
        args, _ = parse_args(["--model", "/models/anon", "--logprob-read-snapshot-donate", "none"])
    assert args.logprob_read_snapshot_donate == "none"


@pytest.mark.parametrize(
    ("sp", "read"),
    [
        (READ, True),
        (SamplingParams(max_tokens=1, logprobs=True), True),
        (SamplingParams(max_tokens=1), False),
        (SamplingParams(max_tokens=2, logprobs=True, top_logprobs=4), False),
    ],
)
def test_is_logprob_read(sp, read):
    assert sp.is_logprob_read is read


def test_logprob_read_isolate_is_a_server_arg():
    from unittest.mock import patch

    from freetoken.server.args import parse_args

    config = SimpleNamespace(to_dict=lambda: {"architectures": ["LlamaForCausalLM"], "torch_dtype": "bfloat16"})
    with patch("freetoken.utils.cached_load_hf_config", lambda _path: config):
        assert parse_args(["--model", "/models/anon"])[0].logprob_read_isolate is False
        args, _ = parse_args(["--model", "/models/anon", "--logprob-read-isolate"])
    assert args.logprob_read_isolate is True


def _assert_no_leak(cm, num_pages=256):
    pc, pool = cm.prefix_cache, cm.linear_state_pool
    assert len(cm.free_slots) + pc.full_evictable + pc.full_protected == num_pages
    assert pc.full_protected == 0 and pc.mamba_protected == 0
    assert pool.num_free_slots + pc.mamba_evictable_size == pool.num_slots - 1
    cm.check_integrity()


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
    if isolate:  # nor may the host tier rehydrate a prefix for it
        cm.host_tier = SimpleNamespace(has_kv=True, lookup=lambda *_: pytest.fail("host tier lookup"))
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
