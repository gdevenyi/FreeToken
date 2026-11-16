"""The scheduler side of a speculative verify step: what the drain commits, what it gives back, and in which order.

The real unbound Scheduler methods run against CPU-built hybrid managers; the engine is a recorder, so the target state commit is checked by its arguments and its order against the finish frees.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from freetoken.core import Batch, Req, SamplingParams
from freetoken.kvcache.linear_state_pool import LinearStatePool
from freetoken.models.config import LinearGatedDeltaGroupConfig
from freetoken.scheduler.cache import CacheManager
from freetoken.scheduler.decode import DecodeManager
from freetoken.scheduler.prefill import PrefillManager
from freetoken.scheduler.scheduler import Scheduler
from freetoken.scheduler.table import TableManager

K = 3
PROMPT = 12


def _setup():
    group = LinearGatedDeltaGroupConfig(
        name="linear", layer_ids=(0,), num_key_heads=2, num_value_heads=4,
        key_head_dim=16, value_head_dim=16, conv_kernel_dim=4, output_gate="silu",
    )
    pool = LinearStatePool(group=group, num_slots=16, dtype=torch.bfloat16, device=torch.device("cpu"), tp_size=1)
    page_table = torch.zeros(4, 64, dtype=torch.int32)
    cm = CacheManager(128, 1, page_table, "hybrid_radix", linear_state_pool=pool)
    tm = TableManager(max_running_reqs=4, page_table=page_table)
    dm = DecodeManager(page_size=1, token_headroom=K)
    pm = PrefillManager(cm, tm, dm, token_headroom=K)

    events = []
    engine = SimpleNamespace(
        commit_verify=lambda reqs, counts, out: events.append(("commit", [r.uid for r in reqs], list(counts))),
    )

    stub = SimpleNamespace(
        cache_manager=cm, table_manager=tm, decode_manager=dm, prefill_manager=pm, spec=None, engine=engine,
        config=SimpleNamespace(decode_rows_per_req=K + 1), device=torch.device("cpu"),
        token_pool=torch.zeros(4, 64, dtype=torch.int32), eos_token_ids=set(), toolcall_anchor_id=None,
        _match_stop_str=lambda _req: None,
    )

    def free(req):
        events.append(("free", req.uid))
        Scheduler._free_req_resources(stub, req)

    stub._free_req_resources = free
    stub._commit_tokens = lambda req, tokens: Scheduler._commit_tokens(stub, req, tokens)
    return pool, cm, tm, dm, stub, events


def _decoding_req(pool, cm, tm, uid: int, max_tokens: int, timings_per_token: bool = False) -> Req:
    """A request whose prefill has drained: KV for the prompt, its first output token pending at PROMPT."""
    prompt = torch.arange(1, PROMPT + 1, dtype=torch.int32) + 100 * uid
    match = cm.match_req(SimpleNamespace(input_ids=prompt, input_len=PROMPT))
    req = Req(input_ids=prompt, table_idx=tm.allocate(), cached_len=0, output_len=max_tokens, uid=uid,
              sampling_params=SamplingParams(max_tokens=max_tokens, ignore_eos=True,
                                             timings_per_token=timings_per_token),
              cache_handle=match.cuda_handle)
    req.linear_slot_idx = pool.alloc(1)[0]
    req.mamba_ping_pong = tuple(pool.alloc(2))
    req.mamba_next_track_idx = 1
    cm.lock(match.cuda_handle)
    cm.allocate_paged([req])
    req.complete_one()
    req.append_host(torch.tensor([7], dtype=torch.int32))
    return req


def test_drain_commits_the_accepted_tokens_and_returns_the_rejected_slots():
    pool, cm, tm, dm, stub, events = _setup()
    # one accepted draft; the output limit cut inside the step; every draft accepted
    reqs = [_decoding_req(pool, cm, tm, 1, 16), _decoding_req(pool, cm, tm, 2, 3), _decoding_req(pool, cm, tm, 3, 16)]
    dm.filter_reqs(reqs)
    # the verify launch: k provisional slots past each pending token
    for req in reqs:
        req.widen_pending(K)
    cm.allocate_paged(reqs)
    free_before = cm.available_size
    out = torch.tensor([[11, 12, 13, 14], [21, 22, 23, 24], [31, 32, 33, 34]])
    output = SimpleNamespace(next_tokens_gpu=out, next_tokens_cpu=out, emitted_cpu=torch.tensor([1, 3, 3]))
    batch = Batch(reqs=reqs, phase="verify", rows_per_req=K + 1)
    reply, finished = [], set()
    tables = [r.table_idx for r in reqs]

    committed = Scheduler._drain_verify(stub, batch, output, reply, finished)

    assert committed == 2 + 2 + 4
    assert [m.next_tokens for m in reply] == [[11, 12], [21, 22], [31, 32, 33, 34]]
    assert [m.finished for m in reply] == [False, True, False]
    # the target state is committed once, before the finished request gives its GDN slot back
    assert events == [("commit", [1, 2, 3], [2, 2, 4]), ("free", 2)]
    assert finished == {reqs[1]}
    a, c = reqs[0], reqs[2]
    assert (a.cached_len, a.device_len) == (PROMPT + 2, PROMPT + 3)
    assert (c.cached_len, c.device_len) == (PROMPT + 4, PROMPT + 5)
    # the last committed token is pending at cached_len for the next step
    assert int(stub.token_pool[tables[0], PROMPT + 2]) == 12
    assert int(stub.token_pool[tables[2], PROMPT + 4]) == 34
    # slots from cached_len on go back (allocate_paged allocates from cached_len again next step); b gave everything back
    assert cm.available_size - free_before == (K + 1 - 2) + (K + 1 - 4) + (PROMPT + K + 1)
    assert dm.running_reqs == {a, c}
    # KVSTATS: three verify steps of context PROMPT (< 32k), 8 committed tokens
    assert cm.stats.accept[0] == [3, 8]
    # no slot was freed twice: with every request gone the pool adds up
    for req in (a, c):
        stub._free_req_resources(req)
    cm.check_integrity()


def test_drain_counts_drafts_and_attaches_timings():
    """The verify drain feeds ReqStats: k drafts per step, accepted = min(n, emitted)
    (a stop landing on a draft makes every committed token an accepted draft), and the
    terminal reply carries the finished request's spans."""
    pool, cm, tm, dm, stub, events = _setup()
    reqs = [_decoding_req(pool, cm, tm, 1, 16), _decoding_req(pool, cm, tm, 2, 3)]
    dm.filter_reqs(reqs)
    for req in reqs:
        req.widen_pending(K)
    cm.allocate_paged(reqs)
    # spans the prefill drain would have stamped: enqueued -> first chunk -> first token
    for req in reqs:
        req.stats.enqueued_at = 1.0
        req.stats.first_chunk_at = 2.0
        req.stats.first_token_at = 3.0
        req.stats.prompt_n = PROMPT
    out = torch.tensor([[11, 12, 13, 14], [21, 22, 23, 24]])
    output = SimpleNamespace(next_tokens_gpu=out, next_tokens_cpu=out, emitted_cpu=torch.tensor([3, 3]))
    batch = Batch(reqs=reqs, phase="verify", rows_per_req=K + 1)
    reply, finished = [], set()

    Scheduler._drain_verify(stub, batch, output, reply, finished)

    # req 1: every draft accepted, n = emitted + 1 -> 3 accepted drafts
    assert reqs[0].stats.draft_n == K and reqs[0].stats.draft_n_accepted == 3
    # req 2: all 3 drafts accepted but the length budget cuts the step to n = 2, and
    # both committed tokens ARE accepted drafts (the target's own never landed): 2
    assert reqs[1].stats.draft_n == K and reqs[1].stats.draft_n_accepted == 2
    # only the terminal reply carries timings
    assert reply[0].timings is None
    t = reply[1].timings
    assert t["queue_ms"] == 1000.0  # enqueued -> first chunk
    assert t["prompt_ms"] == 1000.0  # first chunk -> first token
    assert t["prompt_n"] == PROMPT
    assert t["predicted_n"] == 3  # first token + the two committed here
    assert t["predicted_ms"] > 0.0  # first -> last committed token
    assert t["draft_n"] == K and t["draft_n_accepted"] == 2


def test_drain_stamps_intermediate_timings_when_requested():
    """A request whose sampling_params ask for per-token timings get the ReqStats
    snapshot on every reply, not only the terminal one; the non-asking table above
    keeps the old None-on-intermediate rule."""
    pool, cm, tm, dm, stub, events = _setup()
    reqs = [_decoding_req(pool, cm, tm, 1, 16, timings_per_token=True),
            _decoding_req(pool, cm, tm, 2, 3, timings_per_token=True)]
    dm.filter_reqs(reqs)
    for req in reqs:
        req.widen_pending(K)
    cm.allocate_paged(reqs)
    for req in reqs:
        req.stats.enqueued_at = 1.0
        req.stats.first_chunk_at = 2.0
        req.stats.first_token_at = 3.0
        req.stats.prompt_n = PROMPT
    out = torch.tensor([[11, 12, 13, 14], [21, 22, 23, 24]])
    output = SimpleNamespace(next_tokens_gpu=out, next_tokens_cpu=out, emitted_cpu=torch.tensor([3, 3]))
    batch = Batch(reqs=reqs, phase="verify", rows_per_req=K + 1)
    reply, finished = [], set()

    Scheduler._drain_verify(stub, batch, output, reply, finished)

    # req 1's step is not terminal, yet its reply already carries the growing snapshot
    assert not reply[0].finished and reply[0].timings is not None
    assert reply[0].timings["predicted_n"] == 5  # first token + the four committed here
    assert reply[0].timings["prompt_n"] == PROMPT
    assert reply[0].timings["draft_n"] == K and reply[0].timings["draft_n_accepted"] == 3
    # the terminal reply of the same drain keeps the unchanged terminal snapshot
    assert reply[1].finished and reply[1].timings["predicted_n"] == 3


def test_drain_of_a_no_draft_step_commits_one_token_and_counts_no_draft():
    # past --speculative-max-batch-size a step verifies only the pending token: rows_per_req 1, nothing widened.
    # The drain must take the width from the batch, not from decode_rows_per_req (which still says k + 1).
    pool, cm, tm, dm, stub, events = _setup()
    reqs = [_decoding_req(pool, cm, tm, 1, 16), _decoding_req(pool, cm, tm, 2, 16)]
    dm.filter_reqs(reqs)
    cm.allocate_paged(reqs)
    free_before = cm.available_size
    out = torch.tensor([[11], [21]])
    output = SimpleNamespace(next_tokens_gpu=out, next_tokens_cpu=out, emitted_cpu=torch.tensor([0, 0]))
    batch = Batch(reqs=reqs, phase="verify", rows_per_req=1)
    reply, finished = [], set()
    tables = [r.table_idx for r in reqs]

    committed = Scheduler._drain_verify(stub, batch, output, reply, finished)

    assert committed == 2
    assert [m.next_tokens for m in reply] == [[11], [21]]
    assert events == [("commit", [1, 2], [1, 1])]
    for req, table, tok in zip(reqs, tables, (11, 21)):
        assert (req.cached_len, req.device_len) == (PROMPT + 1, PROMPT + 2)
        assert req.stats.draft_n == 0 and req.stats.draft_n_accepted == 0
        assert int(stub.token_pool[table, PROMPT + 1]) == tok
    # nothing was widened, so nothing comes back
    assert cm.available_size == free_before
    for req in reqs:
        stub._free_req_resources(req)
    cm.check_integrity()


def test_a_padded_one_row_batch_gets_one_row_per_padding_request(monkeypatch):
    # past --speculative-max-batch-size a batch of 3 replays the bs=4 one-row graph: _make_positions gives every padded
    # request its own extend_len rows, so the padding must be the engine's one-token twin, not the k+1-token dummy
    from freetoken.scheduler.scheduler import _make_input_tuple, _make_positions

    real_empty = torch.empty  # the helpers pin their host buffers, which needs a CUDA driver
    monkeypatch.setattr(torch, "empty", lambda *a, pin_memory=False, **kw: real_empty(*a, **kw))
    pool, cm, tm, dm, stub, events = _setup()
    reqs = [_decoding_req(pool, cm, tm, uid, 16) for uid in (1, 2, 3)]

    def pad(n):
        return Req(input_ids=torch.zeros(n, dtype=torch.int32), table_idx=3, cached_len=0, output_len=1, uid=-1,
                   sampling_params=None, cache_handle=None)

    batch = Batch(reqs=reqs, phase="verify", rows_per_req=1)
    batch.padded_reqs = reqs + [pad(1)]
    batch.positions = _make_positions(batch, torch.device("cpu"))
    rows, pos = _make_input_tuple(batch, torch.device("cpu"))
    assert batch.positions.numel() == rows.numel() == batch.padded_size * batch.rows_per_req == 4
    assert batch.positions.tolist() == [PROMPT, PROMPT, PROMPT, 0]
    batch.padded_reqs = reqs + [pad(K + 1)]  # the decode / verify dummy overflows a one-row batch
    assert _make_positions(batch, torch.device("cpu")).numel() == 3 + K + 1
    for req in reqs:
        stub._free_req_resources(req)


def test_a_one_token_read_takes_its_logprobs_from_the_prefill_sample(monkeypatch):
    """Under MTP a logprob read (max_tokens=1) finishes on its prefill forward: the engine reads
    the logprobs of the prefill sample and the drain hands them on. verify_step computes none,
    so a read that reached it, or a spec prefill that dropped them, would answer without them."""
    from freetoken.engine import engine as engine_mod
    from freetoken.engine.engine import Engine
    from freetoken.engine.sample import Sampler
    from freetoken.scheduler.scheduler import ForwardInput

    # the CPU build has no CUDA stream or event; the drain only waits on the event
    monkeypatch.setattr(engine_mod.torch.cuda, "current_stream", lambda: "stream")
    monkeypatch.setattr(engine_mod.torch.cuda, "Event", lambda: SimpleNamespace(record=lambda _s: None, synchronize=lambda: None))
    pool, cm, tm, _dm, stub, events = _setup()
    vocab, labels = 64, [10, 11, 12]
    logits = torch.randn(2, vocab, generator=torch.Generator().manual_seed(4))
    spec = SimpleNamespace(
        feature_sink=lambda _batch: None,
        on_prefill=lambda _batch, tokens: events.append(("draft", tokens.tolist())),
        verify_step=lambda *_: pytest.fail("a prefill reached verify_step"),
        release=lambda req: events.append(("release", req.uid)),
    )
    engine = SimpleNamespace(stream="stream", spec=spec, _comm=None, sampler=Sampler(torch.device("cpu"), vocab),
                             forward_logits=lambda _batch: logits)
    engine.forward_batch = lambda batch, args: Engine.forward_batch(engine, batch, args)
    sent = []
    stub.__dict__.update(
        engine=engine, spec=spec, finished_reqs=set(), send_result=sent.extend,
        status_reporter=SimpleNamespace(report_batch=lambda *_, **__: None),
        _kv_usage_pages=cm.page_usage, _mamba_slot_usage=lambda: None, _swa_token_usage=lambda: None,
        _gpu_mem_bytes=lambda: 0, prefill_manager=SimpleNamespace(pending_list=[]),
        config=SimpleNamespace(decode_rows_per_req=K + 1, page_size=1),
    )

    def prefill_req(uid, sp):
        prompt = torch.arange(1, PROMPT + 1, dtype=torch.int32) + 100 * uid
        match = cm.match_req(SimpleNamespace(input_ids=prompt, input_len=PROMPT))
        req = Req(input_ids=prompt, table_idx=tm.allocate(), cached_len=0, output_len=sp.max_tokens, uid=uid,
                  sampling_params=sp, cache_handle=match.cuda_handle)
        req.linear_slot_idx, req.mamba_ping_pong = pool.alloc(1)[0], tuple(pool.alloc(2))
        req.mamba_next_track_idx = 1
        cm.lock(match.cuda_handle)
        cm.allocate_paged([req])
        return req

    chat = prefill_req(1, SamplingParams(max_tokens=4))
    read = prefill_req(2, SamplingParams(max_tokens=1, logprobs=True, top_logprobs=4, allowed_token_ids=labels))
    batch = Batch(reqs=[chat, read], phase="prefill")  # the read in row 1: the drain must not read row 0
    tables = torch.tensor([chat.table_idx, read.table_idx])
    forward_input = ForwardInput(
        batch=batch,
        sample_args=engine.sampler.prepare(batch),
        input_tuple=(tables.repeat_interleave(PROMPT), torch.arange(PROMPT).repeat(2)),
        write_tuple=(tables, torch.tensor([PROMPT, PROMPT])),
    )

    Scheduler._process_last_data(stub, (forward_input, Scheduler._forward(stub, forward_input)))

    by_uid = {m.uid: m for m in sent}
    expected = torch.log_softmax(logits[1, labels], dim=-1)
    token = labels[int(expected.argmax())]
    assert by_uid[2].next_tokens == [token] and by_uid[2].finished
    assert by_uid[2].chosen_logprob == pytest.approx(float(expected.max()))
    # three labels asked four alternatives: the -inf tail past the set is no candidate
    order = expected.argsort(descending=True).tolist()
    assert by_uid[2].top_ids == [labels[k] for k in order]
    assert by_uid[2].top_logprobs == pytest.approx(expected[order].tolist())
    assert by_uid[1].chosen_logprob is None and not by_uid[1].finished
    assert ("draft", [int(logits[0].argmax()), token]) in events  # the draft still saw the prefill sample
