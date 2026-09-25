"""Hybrid decode's bandwidth-matched fetch split.

Covers the two halves of --moe-hybrid-max-fetch auto: the profile reader that turns
`ft bench bw` kernel bandwidths into a fetch fraction, and the ensure kernel's
per-step integer split (GPU kernel vs CPU reference mirror, and the balance rule).
"""

import json
import os

import pytest
import torch

from freetoken.moe.bench_profile import default_profile_path, load_backend_recommendation, load_hybrid_fetch_fraction
from freetoken.moe.offload_cache import OffloadMoeCache

Q = 1 << 16


def _balanced_fetch(num_missing: int, frac_q16: int) -> int:
    """Reference split: F ~ frac * misses, rounded to whichever integer neighbor
    minimizes the slower overlapped side (fetch ~ F*(1-frac), CPU ~ (M-F)*frac)."""
    lo = (num_missing * frac_q16) >> 16
    cost = lambda f: max(f * (Q - frac_q16), (num_missing - f) * frac_q16)  # noqa: E731
    return min(num_missing, lo if cost(lo) <= cost(lo + 1) else lo + 1)


def test_balanced_fetch_tracks_fraction():
    # The split follows fetched : cpu = pcie : (cpu - pcie) up to integer rounding, and
    # never over/under-shoots by more than one expert.
    for frac in (0.1, 0.415, 0.454, 0.7, 1.0):
        q = round(frac * Q)
        for m in range(0, 65):
            f = _balanced_fetch(m, q)
            assert 0 <= f <= m
            assert abs(f - frac * m) <= 1.0
    # ceil would over-fetch here (the regression this rule fixed): 41.5% of 3 misses is
    # 1.24 -> fetching 2 makes the PCIe side ~1.6x slower than balance; keep it at 1.
    assert _balanced_fetch(3, round(0.415 * Q)) == 1
    assert _balanced_fetch(4, round(0.415 * Q)) == 2


def test_load_hybrid_fetch_fraction(tmp_path):
    prof = {
        "gpu": {"name": "FAKE GPU"},
        "dtype_kernels": {
            "bf16": {"cpu_moe_gbs": 100.0, "pcie_gather_gbs": 40.0},
            # overlapped (contended) pair wins over the standalone numbers when present
            "nvfp4_x": {"cpu_moe_gbs": 100.0, "pcie_gather_gbs": 40.0,
                        "cpu_moe_overlap_gbs": 90.0, "pcie_gather_overlap_gbs": 30.0},
        },
        "workloads": {
            "m": {"kernels": {"ds_fp4": {"cpu_moe_gbs": 80.0, "pcie_gather_gbs": 50.0}}}
        },
    }
    path = tmp_path / "benchbw.json"
    path.write_text(json.dumps(prof))
    # standalone fallback: full-contention assumption -> pcie / cpu
    assert load_hybrid_fetch_fraction("bf16", path=str(path)) == pytest.approx(0.4)
    # overlapped pair preferred: pcie_ov / (pcie_ov + cpu_ov)
    assert load_hybrid_fetch_fraction("nvfp4_x", path=str(path)) == pytest.approx(0.25)
    # per-model fallback when there is no per-dtype entry for the format
    assert load_hybrid_fetch_fraction("ds_fp4", path=str(path)) == pytest.approx(0.625)
    assert load_hybrid_fetch_fraction("nvfp4", path=str(path)) is None
    # a profile from different hardware is ignored
    assert load_hybrid_fetch_fraction("bf16", gpu_name="OTHER", path=str(path)) is None


def test_backend_pick_follows_the_served_expert_geometry(tmp_path):
    """Numbers from an RTX 4070 SUPER: the nvfp4 dtype bench (3072x1536, 7.97 MB experts) says
    hybrid at 89.7 GB/s CPU vs 25.2 PCIe, but Qwen3.6-35B-A3B's own geometry (2048x512,
    1.78 MB) benches at 32.2 GB/s -> offload, and hybrid decodes 3x slower than offload there."""
    small = {"hidden": 2048, "inter": 512, "experts": 256, "top_k": 8}
    prof = {
        "gpu": {"name": "FAKE GPU"},
        "dtypes": {"nvfp4": "hybrid"},
        "dtype_kernels": {"nvfp4": {
            "expert_bytes": 7974912, "cpu_moe_gbs": 89.7, "pcie_gather_gbs": 25.2,
            "cpu_moe_overlap_gbs": 74.1, "pcie_gather_overlap_gbs": 25.2,
        }},
        "workloads": {"qwen3.6-moe": {
            "model": dict(small),
            "kernels": {"nvfp4": {
                "expert_bytes": 1775616, "recommended": "offload", "cpu_moe_gbs": 32.2,
                "pcie_gather_gbs": 25.9, "cpu_moe_overlap_gbs": 28.2, "pcie_gather_overlap_gbs": 25.4,
            }},
        }},
    }
    path = tmp_path / "benchbw.json"
    path.write_text(json.dumps(prof))
    p = str(path)

    # a bench of exactly this geometry beats the dtype verdict, for the pick and the split
    assert load_backend_recommendation("nvfp4", path=p, expert_bytes=1775616, geometry=small) == "offload"
    assert load_hybrid_fetch_fraction("nvfp4", path=p, expert_bytes=1775616, geometry=small) == pytest.approx(25.4 / (25.4 + 28.2))
    # callers that pass no geometry keep the format-only join
    assert load_backend_recommendation("nvfp4", path=p) == "hybrid"

    # no bench of this geometry: a verdict from 4.5x larger experts is not applied ...
    del prof["workloads"]
    path.write_text(json.dumps(prof))
    assert load_backend_recommendation("nvfp4", path=p, expert_bytes=1775616, geometry=small) is None
    assert load_hybrid_fetch_fraction("nvfp4", path=p, expert_bytes=1775616, geometry=small) is None
    # ... while comparable experts (within 2x) inherit it as before
    assert load_backend_recommendation("nvfp4", path=p, expert_bytes=6_000_000, geometry={"hidden": 3072, "inter": 1024, "experts": 128, "top_k": 8}) == "hybrid"
    assert load_hybrid_fetch_fraction("nvfp4", path=p, expert_bytes=6_000_000) == pytest.approx(25.2 / (25.2 + 74.1))


def test_profile_lookup_prefers_the_gpu_uuid_file(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.delenv("FREETOKEN_BENCHBW_PATH", raising=False)
    uuid = "GPU-2f3a9b1c-0000-1111-2222-333344445555"

    def write(path, name, verdict):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({"gpu": {"name": name}, "dtypes": {"bf16": verdict}}, f)

    # legacy single file only: used when the name matches, ignored otherwise
    write(default_profile_path(), "FAKE GPU", "hybrid")
    assert load_backend_recommendation("bf16", gpu_name="FAKE GPU", gpu_uuid=uuid) == "hybrid"
    assert load_backend_recommendation("bf16", gpu_name="OTHER", gpu_uuid=uuid) is None
    # this card's own file wins over the legacy one
    write(default_profile_path(uuid), "FAKE GPU", "offload")
    assert load_backend_recommendation("bf16", gpu_name="FAKE GPU", gpu_uuid=uuid) == "offload"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_hybrid_fraction_gpu_matches_cpu_reference():
    torch.manual_seed(0)
    num_experts, cache_size, top_k, frac = 32, 40, 8, 0.415

    def make():
        return OffloadMoeCache(
            num_layers=2, num_experts=num_experts, cache_size=cache_size,
            device=torch.device("cuda"), quant_format="bf16", decode_target="hybrid",
            hybrid_max_fetch=num_experts, hybrid_fetch_fraction=frac,
        )

    gpu, ref = make(), make()
    frac_q16 = round(frac * Q)
    for step in range(64):
        ids = torch.randperm(num_experts)[:top_k].to(torch.int32)
        g, c = ids.clone().cuda(), ids.clone()  # a CPU ids tensor drives the reference path
        gpu.ensure_experts_hybrid(0, g)
        ref.ensure_experts_hybrid(0, c)
        missing = int(gpu.num_missing_full.item())
        fetched = int(gpu.num_indices.item())
        assert missing == int(ref.num_missing_full.item())
        assert fetched == int(ref.num_indices.item()) == _balanced_fetch(missing, frac_q16)
        # slot rewrites (hit/fetched -> slot, overflow -> -1) and LRU state stay identical
        assert torch.equal(g.cpu(), c)
        assert torch.equal(gpu.slot_for_id.cpu(), ref.slot_for_id.cpu())
        assert torch.equal(gpu.id_of_slot.cpu(), ref.id_of_slot.cpu())
        assert (g >= 0).sum().item() == len(set(ids.tolist())) - (missing - fetched)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_hybrid_fixed_cap_unchanged():
    # fraction 0 (no profile / explicit --moe-hybrid-max-fetch) keeps the fixed cap.
    cache = OffloadMoeCache(
        num_layers=1, num_experts=32, cache_size=40, device=torch.device("cuda"),
        quant_format="bf16", decode_target="hybrid", hybrid_max_fetch=1,
    )
    ids = torch.arange(8, dtype=torch.int32).cuda()
    cache.ensure_experts_hybrid(0, ids)
    assert int(cache.num_missing_full.item()) == 8
    assert int(cache.num_indices.item()) == 1


def test_auto_fetch_keeps_a_split_benched_on_other_sized_experts(tmp_path, monkeypatch):
    """Qwen3.8-Flash-Next (2560x640, 2.64 MB experts) with only the nvfp4 dtype bench (7.97 MB):
    the backend verdict is rightly withheld, but explicit hybrid must not drop to a fetch cap
    of 1 -- the mis-sized split is still far closer than fetching one miss per layer."""
    from types import SimpleNamespace

    from freetoken.engine import engine as engine_mod

    prof = {
        "gpu": {"name": "FAKE GPU"},
        "dtypes": {"nvfp4": "hybrid"},
        "dtype_kernels": {"nvfp4": {
            "expert_bytes": 7974912, "cpu_moe_gbs": 89.7, "pcie_gather_gbs": 25.2,
            "cpu_moe_overlap_gbs": 74.1, "pcie_gather_overlap_gbs": 25.2,
        }},
    }
    path = tmp_path / "benchbw.json"
    path.write_text(json.dumps(prof))
    monkeypatch.setenv("FREETOKEN_BENCHBW_PATH", str(path))
    monkeypatch.setattr(engine_mod, "_profile_gpu", lambda index=None: ("FAKE GPU", None))

    model_config = SimpleNamespace(hidden_size=2560, moe_intermediate_size=640,
                                   num_experts=512, num_experts_per_tok=10)
    config = SimpleNamespace(moe_hybrid_max_fetch=-1, model_config=model_config)
    cache = SimpleNamespace(quant_format="nvfp4", num_experts=512,
                            hybrid_max_fetch=None, hybrid_fetch_fraction=0.0)
    fake_engine = SimpleNamespace(device=torch.device("cpu"))

    assert engine_mod._model_expert_bytes("nvfp4", model_config) == 2772480
    assert load_backend_recommendation(
        "nvfp4", path=str(path), expert_bytes=2772480,
        geometry=engine_mod._model_geometry(model_config)) is None

    engine_mod.Engine._resolve_hybrid_fetch(fake_engine, config, cache)
    assert cache.hybrid_fetch_fraction == pytest.approx(25.2 / (25.2 + 74.1))
    assert cache.hybrid_max_fetch == 512


def test_auto_fetch_logs_one_consistent_message_per_outcome(tmp_path, monkeypatch):
    """The other-size fallback reads the profile once: one warning that says the split is
    approximate (not a withheld backend "verdict" next to it), and a profile from another GPU
    is reported once."""
    from types import SimpleNamespace

    from freetoken.engine import engine as engine_mod
    from freetoken.moe import bench_profile

    path = tmp_path / "benchbw.json"
    path.write_text(json.dumps({
        "gpu": {"name": "FAKE GPU"},
        "dtype_kernels": {"nvfp4": {
            "expert_bytes": 7974912, "cpu_moe_overlap_gbs": 74.1, "pcie_gather_overlap_gbs": 25.2,
        }},
    }))
    monkeypatch.setenv("FREETOKEN_BENCHBW_PATH", str(path))
    monkeypatch.setattr(bench_profile, "_warned", set())
    warnings = []
    monkeypatch.setattr(bench_profile.logger, "warning", warnings.append)
    monkeypatch.setattr(engine_mod.logger, "warning_rank0", warnings.append)
    model_config = SimpleNamespace(hidden_size=2560, moe_intermediate_size=640,
                                   num_experts=512, num_experts_per_tok=10)
    config = SimpleNamespace(moe_hybrid_max_fetch=-1, model_config=model_config)

    def resolve(gpu):
        monkeypatch.setattr(engine_mod, "_profile_gpu", lambda index=None: (gpu, None))
        cache = SimpleNamespace(quant_format="nvfp4", num_experts=512,
                                hybrid_max_fetch=None, hybrid_fetch_fraction=0.0)
        warnings.clear()
        engine_mod.Engine._resolve_hybrid_fetch(SimpleNamespace(device=torch.device("cpu")), config, cache)
        return cache

    cache = resolve("FAKE GPU")
    assert cache.hybrid_fetch_fraction == pytest.approx(25.2 / (25.2 + 74.1))
    assert len(warnings) == 1 and "approximate" in warnings[0] and "verdict" not in warnings[0]

    cache = resolve("OTHER GPU")
    assert cache.hybrid_max_fetch == 1
    assert sum("not this GPU" in w for w in warnings) == 1 and len(warnings) == 2
