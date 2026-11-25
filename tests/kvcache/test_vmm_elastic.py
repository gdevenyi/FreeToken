"""Unit and integration tests for dynamic KV-to-expert VMM page lending (ElasticKvMoePool)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import torch
from freetoken.core import SamplingParams
from freetoken.distributed import DistributedInfo
from freetoken.distributed import info as dist_info
from freetoken.kvcache.format import FP8_PER_TOKEN_HEAD
from freetoken.kvcache.linear_state_pool import LinearStatePool
from freetoken.kvcache.qsa_pool import QSAKVCache
from freetoken.kvcache.vmm_elastic import (
    RETIRED_SLOT_USAGE,
    ElasticKvMoePool,
    MockCpuVmmBackend,
    compute_elastic_geometry,
    kv_moe_elastic_enabled,
)
from freetoken.models.config import LinearGatedDeltaGroupConfig
from freetoken.moe.offload_cache import OffloadMoeCache
from freetoken.scheduler.cache import CacheManager
from freetoken.scheduler.decode import DecodeManager
from freetoken.scheduler.prefill import PrefillManager
from freetoken.scheduler.scheduler import Scheduler
from freetoken.scheduler.table import TableManager
from freetoken.scheduler.utils import PendingReq
from freetoken.server.args import parse_args

DEV = torch.device("cpu")


def test_qwen38_tp2_2mib_vmm_geometry_exact_quanta():
    """Verify exact 2 MiB VMM quantum alignment on chatdnp Qwen3.8-Flash-Next TP=2 geometry:
    - 128 KV pages (8,192 tokens) = 24 x 2 MiB slabs (48 MiB)
    - 64 MoE NVFP4 slots = 81 x 2 MiB slabs (162 MiB)
    - Startup lends idle KV slabs above 128K floor to lift MoE slots from 15,959 to >22,000 (>91%)."""
    gran = 2 << 20  # 2 MiB CUDA VMM granule
    moe_row_bytes = {
        "gate_up_packed": 1536 * 1024,  # 1,572,864 B
        "gate_up_scale": 1536 * 128,  # 196,608 B
        "gate_up_global": 1536 * 4,  # 6,144 B (< 16 KiB -> non-elastic resident)
        "down_packed": 2048 * 384,  # 786,432 B
        "down_scale": 2048 * 48,  # 98,304 B
        "down_global": 2048 * 4,  # 8,192 B (< 16 KiB -> non-elastic resident)
    }
    geo = compute_elastic_geometry(
        granularity=gran,
        num_kv_planes=24,  # 2 * 12 full-attn layers
        plane_page_bytes=64 * 1 * 256,  # 16,384 B
        usable_kv_pages=30_720,  # 1,966,080 tokens
        floor_kv_pages=2_048,  # 131,072 tokens
        moe_row_bytes=moe_row_bytes,
        base_moe_slots=15_959,
        max_moe_slots=48 * 512,  # 24,576 total experts
        num_experts=512,
        prefill_overlap=True,
    )
    assert geo.kv_quantum_pages == 128
    assert geo.kv_slabs_per_plane_block == 1
    assert geo.kv_slabs_per_block == 24
    assert geo.moe_quantum_slots == 64
    assert geo.moe_slabs_per_block == 81
    assert geo.va_moe_slots == 20_224
    assert geo.va_moe_slots > 15_959 + 4_200


def test_qwen38_tp2_real_2560x320_strict_budget_conservation():
    """Regression test for live chatdnp Qwen3.8-Flash-Next TP=2 (2560x320 NVFP4, 30,320 KV pages):
    Without strict_budget, rounding va_kv_pages (30,321 -> 30,336) and floor_moe_blocks (15,959 -> 16,384)
    upward allocated 16,488 x 2 MiB slabs (+566 MiB above baseline KV + MoE bytes), shrinking post-graph
    free VRAM to 1.72 GiB and triggering a 160 MiB OOM during 8,192-token prefill chunks.
    With strict_budget=True, physical slabs drop from 16,488 to 15,789 (-1,398 MiB, 832 MiB below baseline)
    while still boosting active MoE residency to 22,528 / 24,576 (91.67%)."""
    gran = 2 << 20
    num_kv_planes = 24
    plane_page_bytes = 64 * 1 * 256  # 16,384 B -> kv_quantum_pages = 128
    usable_kv_pages = 30_320  # 1,940,480 tokens
    base_moe_slots = 15_959
    moe_row_bytes = {
        "gate_up_packed": 640 * 1280,  # 819,200 B
        "gate_up_scale": 640 * 160,  # 102,400 B
        "gate_up_global": 2 * 4,  # 8 B (< 16 KiB -> non-elastic)
        "down_packed": 2560 * 160,  # 409,600 B
        "down_scale": 2560 * 20,  # 51,200 B -> moe_quantum_slots = 1,024
        "down_global": 4,  # 4 B (< 16 KiB -> non-elastic)
    }
    unbounded = compute_elastic_geometry(
        granularity=gran,
        num_kv_planes=num_kv_planes,
        plane_page_bytes=plane_page_bytes,
        usable_kv_pages=usable_kv_pages,
        floor_kv_pages=2_048,
        moe_row_bytes=moe_row_bytes,
        base_moe_slots=base_moe_slots,
        max_moe_slots=48 * 512,
        num_experts=512,
        prefill_overlap=True,
        strict_budget=False,
    )
    assert unbounded.va_moe_slots == 23_552

    geo = compute_elastic_geometry(
        granularity=gran,
        num_kv_planes=num_kv_planes,
        plane_page_bytes=plane_page_bytes,
        usable_kv_pages=usable_kv_pages,
        floor_kv_pages=2_048,
        moe_row_bytes=moe_row_bytes,
        base_moe_slots=base_moe_slots,
        max_moe_slots=48 * 512,
        num_experts=512,
        prefill_overlap=True,
        strict_budget=True,
    )
    elastic_row_sum = sum(moe_row_bytes[n] for n in geo.elastic_moe_banks)
    budget_bytes = (
        usable_kv_pages + 1
    ) * num_kv_planes * plane_page_bytes + base_moe_slots * elastic_row_sum
    total_kv_blocks = geo.va_kv_pages // geo.kv_quantum_pages
    total_slabs = (
        total_kv_blocks * geo.kv_slabs_per_block
        + geo.floor_moe_blocks * geo.moe_slabs_per_block
    )
    assert total_slabs == 15_789
    assert total_slabs * gran <= budget_bytes
    assert geo.usable_kv_pages == 30_207  # 1,933,248 tokens (-0.37% spillover trim)
    assert geo.va_moe_slots == 22_528  # 91.67% of all 24,576 experts resident!


def _build_elastic_test_stack(monkeypatch, *, kv_host_pages: int = 16):
    """Build a compact CPU VMM-backed QSAKVCache + OffloadMoeCache + CacheManager stack."""
    from freetoken import core

    monkeypatch.setattr(dist_info, "_TP_INFO", DistributedInfo(0, 1))
    gran = 4096
    ps = 64
    usable_pages = 32  # 2,048 tokens across 8 blocks of 4 pages
    floor_pages = (
        8  # 2 blocks (512 tokens) permanently mapped; 6 blocks (24 pages) elastic
    )
    num_layers_moe = 4
    num_experts = 16
    max_moe_slots = num_layers_moe * num_experts  # 64 slots

    # Pick row shapes so quantum math is clean at gran=4096:
    # KV: head_dim=16 fp8 -> plane_page_bytes = 64 * 1 * 16 = 1024 B -> kv_quantum_pages = 4!
    # MoE: gate_up [32, 512] uint8 = 16384 B (4 slabs/slot), down [32, 512] uint8 = 16384 B (4 slabs/slot),
    #      global [32] fp32 = 128 B (non-elastic).
    moe_row_bytes = {
        "gate_up_packed": 16_384,
        "gate_up_scale": 128,
        "gate_up_global": 128,
        "down_packed": 16_384,
        "down_scale": 128,
        "down_global": 128,
    }
    geo = compute_elastic_geometry(
        granularity=gran,
        num_kv_planes=2,  # 2 * 1 storage layer
        plane_page_bytes=1024,
        usable_kv_pages=usable_pages,
        floor_kv_pages=floor_pages,
        moe_row_bytes=moe_row_bytes,
        base_moe_slots=32,
        max_moe_slots=max_moe_slots,
        num_experts=num_experts,
        prefill_overlap=True,
    )
    assert geo.kv_quantum_pages == 4
    assert geo.moe_quantum_slots == 1

    backend = MockCpuVmmBackend(granularity=gran)
    elastic = ElasticKvMoePool(
        geometry=geo, device=DEV, backend=backend, hysteresis_blocks=1
    )
    initial_slabs = backend.num_created_slabs

    moe = OffloadMoeCache(
        num_layers=num_layers_moe,
        num_experts=num_experts,
        cache_size=geo.va_moe_slots,
        device=DEV,
        prefill_overlap=True,
        quant_format="nvfp4",
        decode_target="hybrid",
        bank_allocator=elastic.make_moe_bank_allocator(),
    )
    sources = {
        "gate_up_packed": [
            torch.zeros((num_experts, 32, 512), dtype=torch.uint8)
            for _ in range(num_layers_moe)
        ],
        "gate_up_scale": [
            torch.zeros((num_experts, 32, 4), dtype=torch.float8_e4m3fn)
            for _ in range(num_layers_moe)
        ],
        "gate_up_global": [
            torch.zeros((num_experts, 32), dtype=torch.float32)
            for _ in range(num_layers_moe)
        ],
        "down_packed": [
            torch.zeros((num_experts, 32, 512), dtype=torch.uint8)
            for _ in range(num_layers_moe)
        ],
        "down_scale": [
            torch.zeros((num_experts, 32, 4), dtype=torch.float8_e4m3fn)
            for _ in range(num_layers_moe)
        ],
        "down_global": [
            torch.zeros((num_experts, 32), dtype=torch.float32)
            for _ in range(num_layers_moe)
        ],
    }
    moe.set_bank_sources(sources)
    elastic.bind_moe_cache(moe)

    kv_buf = elastic.wrap_kv_buffer(
        num_storage_layers=1,
        page_size=ps,
        local_kv_heads=1,
        elem_dim=16,
        elem_dtype=torch.uint8,
    )
    kv = QSAKVCache(
        num_kv_heads=1,
        num_layers=2,
        head_dim=16,
        num_pages=geo.va_kv_pages,
        page_size=ps,
        dtype=torch.bfloat16,
        device=DEV,
        index_head_dim=8,
        num_index_layers=1,
        index_ratio=4,
        num_req_slots=9,
        layer_ids=(1,),
        mrope=True,
        kv_format=FP8_PER_TOKEN_HEAD,
        kv_buffer=kv_buf,
    )
    if kv_host_pages:
        kv.attach_host_pages(kv_host_pages, pin_memory=False)

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
    lpool = LinearStatePool(
        group=grp, num_slots=32, dtype=torch.bfloat16, device=DEV, tp_size=1
    )
    pt = torch.zeros(9, usable_pages * ps, dtype=torch.int32)
    cm = CacheManager(
        usable_pages,
        ps,
        pt,
        "hybrid_radix",
        linear_state_pool=lpool,
        kv_host=kv if kv_host_pages else None,
        elastic_pool=elastic,
    )
    monkeypatch.setattr(
        core, "_GLOBAL_CTX", core.Context(page_size=ps, linear_state_pool=lpool)
    )
    tm = TableManager(max_running_reqs=8, page_table=pt)
    dm = DecodeManager(page_size=ps)
    pm = PrefillManager(cm, tm, dm)
    stub = SimpleNamespace(cache_manager=cm, table_manager=tm, spec=None)
    return backend, initial_slabs, elastic, moe, kv, lpool, cm, tm, dm, pm, stub


def test_p2c_demotes_first_before_borrowing_from_moe_then_expands_and_reclaims(
    monkeypatch,
):
    """Proves all 4 policy invariants:
    1. Startup boosts active MoE slots above base_moe_slots=32 using lent KV slabs.
    2. Idle prefixes in floor KV demote to P2c host RAM FIRST without retiring a single MoE slot!
    3. Concurrent locked active requests exceeding floor KV borrow slabs from MoE tail slots
       (marking retired slots with RETIRED_SLOT_USAGE, constant physical VMM slabs).
    4. Finishing active requests reclaims elastic KV blocks (demoting excess idle cache to P2c)
       and restores MoE tail slots automatically, preserving exact CacheManager.check_integrity()."""
    backend, initial_slabs, elastic, moe, kv, _lpool, cm, _tm, _dm, pm, stub = (
        _build_elastic_test_stack(monkeypatch, kv_host_pages=16)
    )
    cm.check_integrity()
    startup_moe = moe.active_cache_size
    assert startup_moe > 32
    assert (
        len(cm.free_slots) == 8
    )  # floor_pages = 8 mapped initially; 24 pages lent to MoE
    assert cm.available_size == 32 * 64  # full 32 usable pages advertisable!

    # Step A: Run two sequential 6-page (384-token) requests that finish and sit IDLE in radix cache.
    # Total cached = 12 pages > 8 mapped floor pages, so second request forces eviction of first!
    # Because P2c host tier has 16 pages, first prefix MUST demote to P2c WITHOUT borrowing from MoE!
    for uid in (1, 2):
        prompt = torch.arange(uid * 10_000, uid * 10_000 + 390, dtype=torch.int32)
        pm.pending_list = [PendingReq(uid, prompt, SamplingParams(max_tokens=2))]
        batch = pm.schedule_next_batch(8192)
        assert batch is not None
        (req,) = batch.reqs
        cm.allocate_paged(batch.reqs)
        req.mamba_last_track_seqlen = 384
        req.mamba_next_track_idx = 1
        req.complete_one()
        cm.cache_req(req, finished=False)
        Scheduler._free_req_resources(stub, req)

    assert cm.prefix_cache.counts["kv_demote"] >= 1
    assert elastic.borrowed_kv_blocks == 0
    assert moe.active_cache_size == startup_moe
    assert backend.num_created_slabs == initial_slabs
    cm.check_integrity()

    # Step B: Launch 2 CONCURRENT large active requests (14 pages = 896 tokens each -> 28 active pages!).
    # Floor has only 8 pages, so CacheManager demotes remaining idle prefix to P2c AND borrows 5 KV blocks (20 pages)!
    active_reqs = []
    for uid in (3, 4):
        prompt = torch.arange(uid * 10_000, uid * 10_000 + 896, dtype=torch.int32)
        pm.pending_list.append(PendingReq(uid, prompt, SamplingParams(max_tokens=2)))
    batch_big = pm.schedule_next_batch(8192)
    assert batch_big is not None and len(batch_big.reqs) == 2
    cm.allocate_paged(batch_big.reqs)
    active_reqs.extend(batch_big.reqs)

    assert elastic.borrowed_kv_blocks == 5  # 8 floor + 20 borrowed = 28 mapped pages!
    assert moe.active_cache_size < startup_moe
    # Every retired MoE tail slot must carry RETIRED_SLOT_USAGE and id_of_slot == -1.
    assert torch.all(moe.usage[moe.active_cache_size :] == RETIRED_SLOT_USAGE)
    assert torch.all(moe.id_of_slot[moe.active_cache_size :] == -1)
    # Zero extra physical slabs allocated!
    assert backend.num_created_slabs == initial_slabs
    # Dummy page (usable_pages = 32) in tail block stays mapped and writable!
    kv._kv_buffer[:, :, 32].fill_(7)
    assert int(kv._kv_buffer[0, 0, 32, 0, 0, 0].item()) == 7

    # Step C: Finish both big requests inside lazy_free_region -> excess idle KV demotes to P2c / drops,
    # borrowed KV blocks unmap, and MoE active_cache_size climbs back up!
    with cm.lazy_free_region():
        for req in active_reqs:
            req.mamba_last_track_seqlen = 896
            req.mamba_next_track_idx = 1
            req.complete_one()
            cm.cache_req(req, finished=False)
            Scheduler._free_req_resources(stub, req)

    cm.check_integrity()
    assert elastic.borrowed_kv_blocks <= 1  # hysteresis keeps at most 1 block
    assert moe.active_cache_size > 32
    assert backend.num_created_slabs == initial_slabs

    # Step D: Flush cache -> returns remaining borrowed blocks and verifies clean close().
    cm.flush()
    cm.check_integrity()
    elastic.close()
    assert backend.num_created_slabs == 0
    assert backend.num_mapped_slabs == 0


def test_tp2_lockstep_elastic_borrowing_and_cli_flags(monkeypatch, tmp_path):
    """Two simulated TP ranks running the same workload borrow and return identical KV/MoE quanta."""
    s0 = _build_elastic_test_stack(monkeypatch, kv_host_pages=8)
    s1 = _build_elastic_test_stack(monkeypatch, kv_host_pages=8)
    _, _, e0, m0, _, _, cm0, _, _, pm0, stub0 = s0
    _, _, e1, m1, _, _, cm1, _, _, pm1, stub1 = s1

    for step_idx, tok_len in enumerate((256, 768, 768, 128)):
        p = torch.arange(
            50_000 + step_idx * 2000,
            50_000 + step_idx * 2000 + tok_len,
            dtype=torch.int32,
        )
        pm0.pending_list.append(
            PendingReq(step_idx, p.clone(), SamplingParams(max_tokens=2))
        )
        pm1.pending_list.append(
            PendingReq(step_idx, p.clone(), SamplingParams(max_tokens=2))
        )
    b0 = pm0.schedule_next_batch(8192)
    b1 = pm1.schedule_next_batch(8192)
    cm0.allocate_paged(b0.reqs)
    cm1.allocate_paged(b1.reqs)
    assert e0.borrowed_kv_blocks == e1.borrowed_kv_blocks > 0
    assert m0.active_cache_size == m1.active_cache_size
    assert torch.equal(cm0.free_slots, cm1.free_slots)

    with cm0.lazy_free_region(), cm1.lazy_free_region():
        for r0, r1 in zip(b0.reqs, b1.reqs, strict=True):
            r0.complete_one()
            r1.complete_one()
            Scheduler._free_req_resources(stub0, r0)
            Scheduler._free_req_resources(stub1, r1)
    assert e0.borrowed_kv_blocks == e1.borrowed_kv_blocks
    assert m0.active_cache_size == m1.active_cache_size
    cm0.check_integrity()
    cm1.check_integrity()
    e0.close()
    e1.close()

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
    cfg, _ = parse_args(
        [
            "--model",
            str(tmp_path),
            "--kv-moe-elastic",
            "--kv-elastic-floor-tokens",
            "262144",
        ]
    )
    assert cfg.kv_moe_elastic is True
    assert cfg.kv_elastic_floor_tokens == 262144
    monkeypatch.setenv("FREETOKEN_KV_MOE_ELASTIC", "1")
    assert kv_moe_elastic_enabled(False) is True
