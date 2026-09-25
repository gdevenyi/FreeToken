"""Hybrid decode's bandwidth-matched fetch split.

Covers the profile reader of --moe-hybrid-max-fetch auto, which turns `ft bench bw`
kernel bandwidths into a fetch fraction.
"""

import json
import os

import pytest
import torch

from freetoken.moe.bench_profile import PROFILE_VERSION, default_profile_path, load_backend_recommendation, load_hybrid_fetch_fraction


def test_load_hybrid_fetch_fraction(tmp_path):
    prof = {
        "version": PROFILE_VERSION,
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
    # so is one written before the current set of CPU formats
    path.write_text(json.dumps({**prof, "version": PROFILE_VERSION - 1}))
    assert load_hybrid_fetch_fraction("bf16", path=str(path)) is None


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
            json.dump({"version": PROFILE_VERSION, "gpu": {"name": name}, "dtypes": {"bf16": verdict}}, f)

    # legacy single file only: used when the name matches, ignored otherwise
    write(default_profile_path(), "FAKE GPU", "hybrid")
    assert load_backend_recommendation("bf16", gpu_name="FAKE GPU", gpu_uuid=uuid) == "hybrid"
    assert load_backend_recommendation("bf16", gpu_name="OTHER", gpu_uuid=uuid) is None
    # this card's own file wins over the legacy one
    write(default_profile_path(uuid), "FAKE GPU", "offload")
    assert load_backend_recommendation("bf16", gpu_name="FAKE GPU", gpu_uuid=uuid) == "offload"


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
