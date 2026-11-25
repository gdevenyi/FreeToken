"""Unit tests for P3 copy-engine isolation: host-KV restore token cap and SM step-input upload ring."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from freetoken.core import Batch, Req, SamplingParams
from freetoken.kernel.sm_upload import (
    SmUploadRing,
    sm_upload_enabled,
    upload_step_tensor,
)
from freetoken.kvcache.linear_state_pool import LinearStatePool
from freetoken.kvcache.qsa_pool import QSAKVCache
from freetoken.models.config import LinearGatedDeltaGroupConfig
from freetoken.scheduler.cache import CacheManager, _write_page_table
from freetoken.scheduler.decode import DecodeManager
from freetoken.scheduler.prefill import PrefillManager
from freetoken.scheduler.scheduler import (
    Scheduler,
    _make_input_tuple,
    _make_mrope_positions,
    _make_positions,
    _make_write_tuple,
)
from freetoken.scheduler.table import TableManager
from freetoken.scheduler.utils import PendingReq
from freetoken.server.args import parse_args

DEV = torch.device("cpu")


def _setup_host_restore_env(
    monkeypatch,
    *,
    pages: int = 32,
    kv_host_pages: int = 16,
    max_reqs: int = 4,
    cap: int = 256,
):
    from freetoken import core
    from freetoken.distributed import DistributedInfo
    from freetoken.distributed import info as dist_info

    monkeypatch.setattr(dist_info, "_TP_INFO", DistributedInfo(0, 1))
    ps = 64
    grp = LinearGatedDeltaGroupConfig(
        name="linear",
        layer_ids=(0,),
        num_key_heads=2,
        num_value_heads=4,
        key_head_dim=16,
        value_head_dim=16,
        conv_kernel_dim=4,
        output_gate="silu",
    )
    pool = LinearStatePool(
        group=grp,
        num_slots=24,
        dtype=torch.bfloat16,
        device=DEV,
        tp_size=1,
    )
    kv = QSAKVCache(
        num_kv_heads=1,
        num_layers=2,
        head_dim=8,
        num_pages=pages + 1,
        page_size=ps,
        dtype=torch.bfloat16,
        device=DEV,
        index_head_dim=8,
        num_index_layers=1,
        index_ratio=4,
        num_req_slots=max_reqs + 1,
        layer_ids=(1,),
        mrope=True,
    )
    kv.attach_host_pages(kv_host_pages, pin_memory=False)
    pt = torch.zeros(max_reqs + 1, pages * ps, dtype=torch.int32)
    cm = CacheManager(pages, ps, pt, "hybrid_radix", linear_state_pool=pool, kv_host=kv)
    monkeypatch.setattr(
        core, "_GLOBAL_CTX", core.Context(page_size=ps, linear_state_pool=pool)
    )
    tm = TableManager(max_running_reqs=max_reqs, page_table=pt)
    dm = DecodeManager(page_size=ps)
    pm = PrefillManager(cm, tm, dm, kv_host_restore_tokens=cap)
    stub = SimpleNamespace(cache_manager=cm, table_manager=tm, spec=None)
    return pool, kv, cm, tm, dm, pm, stub


def _seed_demoted_session(
    pool, kv, cm, pm, stub, uid: int, base_tok: int, length: int = 192
) -> torch.Tensor:
    prompt = torch.arange(base_tok, base_tok + length, dtype=torch.int32)
    pm.pending_list = [PendingReq(uid, prompt, SamplingParams(max_tokens=4))]
    batch = pm.schedule_next_batch(8192)
    assert batch is not None
    (req,) = batch.reqs
    cm.allocate_paged(batch.reqs)
    req.mamba_last_track_seqlen = (length // 64) * 64
    req.mamba_next_track_idx = 1
    req.complete_one()
    cm.cache_req(req, finished=False)
    Scheduler._free_req_resources(stub, req)
    return prompt


def test_kv_host_restore_tokens_cli_parsing(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "llama",
                "architectures": ["LlamaForCausalLM"],
                "torch_dtype": "bfloat16",
            }
        ),
        encoding="utf-8",
    )
    model_dir = str(tmp_path)
    cfg, _ = parse_args(["--model", model_dir])
    assert cfg.kv_host_restore_tokens == 131072

    cfg2, _ = parse_args(["--model", model_dir, "--kv-host-restore-tokens", "0"])
    assert cfg2.kv_host_restore_tokens == 0

    with pytest.raises(SystemExit):
        parse_args(["--model", model_dir, "--kv-host-restore-tokens", "-1"])


def test_restore_cap_bounds_concurrent_restores_under_active_decode_without_starving_large_one(
    monkeypatch,
):
    """When decodes are running, host-KV restores batch up to kv_host_restore_tokens, defer overflow,
    and still admit a single large restore exceeding the cap when restored_tokens == 0."""
    pool, kv, cm, tm, dm, pm, stub = _setup_host_restore_env(monkeypatch, cap=256)

    # Seed 3 distinct 128-token (2-page) prefixes + 1 large 384-token (6-page) prefix, then demote all to host.
    s0 = _seed_demoted_session(
        pool, kv, cm, pm, stub, 10, 10_000, length=140
    )  # 128 cached
    s1 = _seed_demoted_session(
        pool, kv, cm, pm, stub, 11, 20_000, length=140
    )  # 128 cached
    s2 = _seed_demoted_session(
        pool, kv, cm, pm, stub, 12, 30_000, length=140
    )  # 128 cached
    big = _seed_demoted_session(
        pool, kv, cm, pm, stub, 13, 40_000, length=400
    )  # 384 cached > cap=256

    evicted = cm._evict(128 * 3 + 384)
    cm._free(evicted)
    assert cm.prefix_cache.full_evictable == 0
    assert cm.prefix_cache.host_kv_tokens == 128 * 3 + 384

    # Simulate an active decode request so restore_token_budget=256 activates.
    dec_req = Req(
        input_ids=torch.arange(8, dtype=torch.int32),
        table_idx=0,
        cached_len=7,
        output_len=4,
        uid=999,
        sampling_params=SamplingParams(max_tokens=4),
        cache_handle=None,
    )
    dm.running_reqs.add(dec_req)
    assert dm.runnable

    # Enqueue 3 returning 128-token sessions: s0 + s1 = 256 fits; s2 (would reach 384 > 256) defers!
    r0 = torch.cat([s0[:128], torch.arange(90_000, 90_020, dtype=torch.int32)])
    r1 = torch.cat([s1[:128], torch.arange(91_000, 91_020, dtype=torch.int32)])
    r2 = torch.cat([s2[:128], torch.arange(92_000, 92_020, dtype=torch.int32)])
    pm.pending_list = [
        PendingReq(20, r0, SamplingParams(max_tokens=4)),
        PendingReq(21, r1, SamplingParams(max_tokens=4)),
        PendingReq(22, r2, SamplingParams(max_tokens=4)),
    ]
    batch1 = pm.schedule_next_batch(8192)
    assert batch1 is not None
    assert [r.uid for r in batch1.reqs] == [20, 21]
    assert [r.uid for r in pm.pending_list] == [22]
    assert cm.prefix_cache.counts["kv_restore_defer"] == 1

    # Complete s0, s1 restores and free them.
    cm.allocate_paged(batch1.reqs)
    for req in batch1.reqs:
        kv.pages_from_host(req.kv_restore.host_pages, req.kv_restore.dev_pages)
        req.kv_restore = None
        req.complete_one()
        Scheduler._free_req_resources(stub, req)

    # Next pass: enqueue oversized big (384 > 256) FIRST, followed by s2 (128):
    # big MUST NOT starve (admitted alone), and s2 defers behind it!
    rbig = torch.cat([big[:384], torch.arange(93_000, 93_020, dtype=torch.int32)])
    pm.pending_list = [
        PendingReq(23, rbig, SamplingParams(max_tokens=4)),
        PendingReq(22, r2, SamplingParams(max_tokens=4)),
    ]
    batch2 = pm.schedule_next_batch(8192)
    assert batch2 is not None
    assert [r.uid for r in batch2.reqs] == [23]
    assert [r.uid for r in pm.pending_list] == [22]

    # Once decode finishes (dm.runnable == False), remaining restores run without cap.
    dm.running_reqs.clear()
    cm.allocate_paged(batch2.reqs)
    for req in batch2.reqs:
        kv.pages_from_host(req.kv_restore.host_pages, req.kv_restore.dev_pages)
        req.kv_restore = None
        req.complete_one()
        Scheduler._free_req_resources(stub, req)

    batch3 = pm.schedule_next_batch(8192)
    assert batch3 is not None and [r.uid for r in batch3.reqs] == [22]
    cm.allocate_paged(batch3.reqs)
    for req in batch3.reqs:
        kv.pages_from_host(req.kv_restore.host_pages, req.kv_restore.dev_pages)
        req.kv_restore = None
        req.complete_one()
        Scheduler._free_req_resources(stub, req)
    cm.check_integrity()


def test_sm_upload_ring_alignment_wraparound_and_event_fencing(monkeypatch):
    events_created = []

    class FakeEvent:
        def __init__(self):
            self.records = 0
            self.syncs = 0
            events_created.append(self)

        def record(self):
            self.records += 1

        def synchronize(self):
            self.syncs += 1

    backing = torch.zeros(256, dtype=torch.uint8)
    offsets_seen: list[tuple[int, int]] = []

    def fake_alloc(nbytes: int):
        assert nbytes == 256
        return backing, 0x10000

    def fake_copy(dst: torch.Tensor, host_dev_ptr: int, byte_offset: int, nbytes: int):
        assert host_dev_ptr == 0x10000
        assert byte_offset % 16 == 0
        offsets_seen.append((byte_offset, nbytes))
        dst.view(torch.uint8).reshape(-1).copy_(
            backing[byte_offset : byte_offset + nbytes]
        )

    # 256-byte ring with 4 segments of 64 bytes each.
    ring = SmUploadRing(
        DEV,
        ring_bytes=256,
        num_segments=4,
        alloc_fn=fake_alloc,
        copy_fn=fake_copy,
        event_factory=FakeEvent,
    )
    assert ring.segment_bytes == 64

    # 12 bytes (3 int32s) -> rounds to 16B; 24 bytes (3 int64s) -> 32B; next 20 bytes -> 32B overflows 64B -> seg 1!
    t1 = torch.tensor([11, 22, 33], dtype=torch.int32)
    t2 = torch.tensor([101, 202, 303], dtype=torch.int64)
    t3 = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32)
    assert torch.equal(ring.upload(t1), t1)
    assert torch.equal(ring.upload(t2), t2)
    assert torch.equal(ring.upload(t3), t3)
    assert offsets_seen[:3] == [(0, 12), (16, 24), (64, 20)]

    # Advance through remaining segments to wrap around to segment 0 and verify synchronize() fires!
    big_seg = torch.arange(16, dtype=torch.int32)  # 64 bytes = full segment
    for _ in range(4):
        got = ring.upload(big_seg)
        assert torch.equal(got, big_seg)
    assert len(events_created) == 4
    assert events_created[0].records >= 1 and events_created[0].syncs >= 1

    # Oversized (> 64B) or odd-byte tensors are rejected by can_stage().
    assert not ring.can_stage(torch.zeros(17, dtype=torch.int32))
    assert not ring.can_stage(torch.zeros(3, dtype=torch.uint8))

    monkeypatch.setenv("FREETOKEN_SM_UPLOAD", "0")
    assert not sm_upload_enabled()
    monkeypatch.setenv("FREETOKEN_SM_UPLOAD", "1")
    assert sm_upload_enabled()


def test_step_input_helpers_round_trip_on_cpu():
    pt = torch.full((3, 16), -1, dtype=torch.int32)
    allocated = torch.arange(100, 108, dtype=torch.int32)
    _write_page_table(pt, allocated, [(0, 0, 1), (2, 1, 2)], page_size=4)
    assert torch.equal(
        pt[0, 0:4], torch.tensor([100, 101, 102, 103], dtype=torch.int32)
    )
    assert torch.equal(
        pt[2, 4:8], torch.tensor([104, 105, 106, 107], dtype=torch.int32)
    )

    r0 = Req(
        input_ids=torch.arange(10, dtype=torch.int32),
        table_idx=1,
        cached_len=4,
        output_len=4,
        uid=1,
        sampling_params=SamplingParams(),
        cache_handle=None,
    )
    batch = Batch(reqs=[r0], phase="prefill")
    batch.padded_reqs = [r0]
    batch.positions = _make_positions(batch, DEV)
    assert torch.equal(batch.positions, torch.arange(4, 10, dtype=torch.int32))
    mrope = _make_mrope_positions(batch, DEV)
    assert mrope.shape == (3, 6) and torch.equal(mrope[0], batch.positions)
    inp_map, inp_pos = _make_input_tuple(batch, DEV)
    assert torch.equal(inp_map, torch.full((6,), 1, dtype=torch.int64))
    assert torch.equal(inp_pos, torch.arange(4, 10, dtype=torch.int64))
    w_map, w_pos = _make_write_tuple(batch, DEV)
    assert torch.equal(w_map, torch.tensor([1], dtype=torch.int64))
    assert torch.equal(w_pos, torch.tensor([10], dtype=torch.int64))
    assert torch.equal(
        upload_step_tensor(torch.empty(0, dtype=torch.int32), DEV),
        torch.empty(0, dtype=torch.int32),
    )
