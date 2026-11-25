"""Unit tests for persisting L2 cache windows and MoE static hot-cache pre-seeding."""

from __future__ import annotations

import ctypes
import json
from types import SimpleNamespace

import torch
from freetoken.kernel.l2_persist import (
    clear_stream_persisting_l2_window,
    find_graph_pool_window,
    set_stream_persisting_l2_window,
)
from freetoken.moe.hot_rankings import HotExpertRankings
from freetoken.moe.offload_cache import OffloadMoeCache
from freetoken.moe.pool_policy import load_profile


class _FakeCudaRt:
    def __init__(self, *, max_persisting: int, max_window: int = 64 << 20):
        self.max_persisting = max_persisting
        self.max_window = max_window
        self.limit_set: int | None = None
        self.window_calls: list[tuple[int, int, float, int, int]] = []

    def cudaGetDevice(self, ptr) -> int:
        ctypes.cast(ptr, ctypes.POINTER(ctypes.c_int))[0] = 0
        return 0

    def cudaDeviceGetAttribute(self, ptr, attr, _dev) -> int:
        val = self.max_persisting if int(attr.value) == 108 else self.max_window
        ctypes.cast(ptr, ctypes.POINTER(ctypes.c_int))[0] = val
        return 0

    def cudaDeviceSetLimit(self, _limit, value) -> int:
        self.limit_set = int(value.value)
        return 0

    def cudaStreamSetAttribute(self, _stream, _attr, value_ptr) -> int:
        win = value_ptr._obj.accessPolicyWindow
        self.window_calls.append(
            (
                int(win.base_ptr or 0),
                int(win.num_bytes),
                float(win.hitRatio),
                int(win.hitProp),
                int(win.missProp),
            )
        )
        return 0

    def cudaGetLastError(self) -> int:
        return 0


def test_persisting_l2_window_pins_sm89_and_skips_small_l2(monkeypatch):
    fake_stream = SimpleNamespace(cuda_stream=0x12345678)

    # RTX 6000 Ada sm_89 (~64 MiB persisting L2 capacity, 47.7 MiB graph pool, 32 MiB cap):
    rt_ada = _FakeCudaRt(max_persisting=64 << 20, max_window=64 << 20)
    pinned = set_stream_persisting_l2_window(
        fake_stream, 0x7F00_0000_0000, 48 << 20, max_window_mb=32, cudart=rt_ada
    )
    assert pinned == 32 << 20
    assert rt_ada.limit_set == 32 << 20
    assert rt_ada.window_calls == [(0x7F00_0000_0000, 32 << 20, 1.0, 2, 0)]

    clear_stream_persisting_l2_window(fake_stream, cudart=rt_ada)
    assert rt_ada.window_calls[-1] == (0, 0, 0.0, 0, 0)

    # Small L2 GPU (< 16 MiB persisting capacity, e.g. sm_86 RTX 3090 Ti 6 MiB L2) -> skipped!
    rt_small = _FakeCudaRt(max_persisting=4 << 20, max_window=4 << 20)
    assert (
        set_stream_persisting_l2_window(
            fake_stream, 0x7F00_0000_0000, 48 << 20, max_window_mb=32, cudart=rt_small
        )
        == 0
    )
    assert rt_small.limit_set is None

    monkeypatch.setattr(
        torch.cuda,
        "memory_snapshot",
        lambda: [
            {
                "device": 0,
                "segment_pool_id": (1, 9),
                "address": 0x1000,
                "total_size": 4096,
            },
            {
                "device": 0,
                "segment_pool_id": (1, 9),
                "address": 0x2000,
                "total_size": 50_000_000,
            },
            {
                "device": 1,
                "segment_pool_id": (1, 9),
                "address": 0x3000,
                "total_size": 99_000_000,
            },
        ],
    )
    assert find_graph_pool_window((1, 9), 0) == (0x2000, 50_000_000)


def test_hot_rankings_formats_entropy_split_and_cache_seeding(tmp_path, monkeypatch):
    # 1. warp34 expert_routing_profile.json with steep layer 0 curve vs flat layer 1 curve
    profile_path = tmp_path / "runtime" / "expert_routing_profile.json"
    profile_path.parent.mkdir(parents=True)
    profile_path.write_text(
        json.dumps(
            {
                "layers": {
                    "0": {
                        "num_experts": 16,
                        "ranked_expert_ids": [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4],
                        "hit_rate_by_cache_size": {
                            "top_4": 0.20,
                            "top_8": 0.90,
                            "top_12": 0.99,
                        },
                    },
                    "1": {
                        "num_experts": 16,
                        "ranked_expert_ids": [3, 2, 1, 0, 7, 6, 5, 4],
                        "hit_rate_by_cache_size": {
                            "top_4": 0.65,
                            "top_8": 0.70,
                            "top_12": 0.75,
                        },
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    rankings = HotExpertRankings.discover(tmp_path)
    assert rankings is not None
    caps = rankings.plan_layer_capacities([0, 1], 16, 16, step=2)
    assert sum(caps.values()) == 16
    assert caps[0] > caps[1]  # Layer 0 has much higher marginal hit gain from 6 -> 10!

    # Pool policy load_profile also accepts warp34 / albucino rankings JSON!
    pseudo = load_profile(profile_path, 2, 16)
    assert pseudo.shape == (2, 16)
    assert int(pseudo[0].argmax().item()) == 15
    assert int(pseudo[1].argmax().item()) == 3

    # 2. Seed OffloadMoeCache (with prefill_overlap=True reserving [0 : 32) staging slots)
    cache = OffloadMoeCache(
        num_layers=2,
        num_experts=16,
        cache_size=48,  # 32 staging + 16 unreserved slots
        device=torch.device("cpu"),
        cache_policy="crf",
        prefill_overlap=True,
        quant_format="bf16",
        decode_target="hybrid",
        hot_rankings=rankings,
    )
    # Give each (layer, expert) a unique signature in its weights so we can verify exact copies!
    gate_up = [
        torch.arange(16, dtype=torch.bfloat16)
        .view(16, 1, 1)
        .expand(16, 4, 8)
        .contiguous()
        + (l * 100)
        for l in range(2)
    ]
    down = [
        torch.arange(16, dtype=torch.bfloat16)
        .view(16, 1, 1)
        .expand(16, 8, 2)
        .contiguous()
        + (l * 100)
        for l in range(2)
    ]
    cache.set_bank_sources({"gate_up": gate_up, "down": down})
    seeded = cache.seed_hot_experts()
    assert seeded == 16
    # Staging area [0 : 32) stays untouched (-1); slots [32 : 48) hold the top-ranked experts!
    assert torch.all(cache.id_of_slot[:32] == -1)
    assert torch.all(cache.id_of_slot[32:48] >= 0)
    s_l0_top = int(cache.slot_for_id[0, 15].item())
    assert 32 <= s_l0_top < 48
    assert float(cache.bank_caches["gate_up"][s_l0_top, 0, 0].item()) == 15.0
    assert (
        float(cache.expert_crf[0, 15].item())
        > float(cache.expert_crf[0, 14].item())
        > 0.0
    )

    # Rebuild preserves warm seeding automatically!
    cache.rebuild(44)
    assert torch.all(cache.id_of_slot[:32] == -1)
    assert int((cache.id_of_slot[32:44] >= 0).sum().item()) == 12

    # 3. Plain static_hot_cache_rankings.json format via FREETOKEN_MOE_HOT_RANKINGS env var
    static_path = tmp_path / "static_hot_cache_rankings.json"
    static_path.write_text(
        json.dumps({"0": [5, 4, 3], "1": [9, 8, 7]}), encoding="utf-8"
    )
    monkeypatch.setenv("FREETOKEN_MOE_HOT_RANKINGS", str(static_path))
    r2 = HotExpertRankings.discover(None)
    assert r2 is not None
    assert r2.select_for_layer(0, 16, 4) == [5, 4, 3, 0]
