"""`ft bench bw` must not present a coin flip as a fact.

The hybrid-vs-offload verdict is a bandwidth ratio against a threshold. Some formats
measure the same to within a percent; others swing far enough between runs to land on
both sides of it -- mxfp4 was observed anywhere from 19 to 71 GB/s on one machine,
flipping its recommendation run to run while each individual run reported a single
confident number. `verdict` bounds the ratio by the extremes actually observed and
withholds the call when that interval straddles the threshold.
"""

from __future__ import annotations

import torch

from freetoken.moe.benchbw import verdict


def test_clearly_above_threshold_is_hybrid():
    pick, confident, (lo, hi) = verdict([96.0, 97.0, 95.5], [25.0, 25.1, 25.0], 2.0)
    assert (pick, confident) == ("hybrid", True)
    assert lo > 2.0 and hi > lo


def test_clearly_below_threshold_is_offload():
    pick, confident, _ = verdict([40.0, 41.0, 39.5], [25.0, 25.1, 25.0], 2.0)
    assert (pick, confident) == ("offload", True)


def test_straddling_the_threshold_is_withheld():
    """The real mxfp4 case: the median says hybrid, the spread says nobody knows."""
    pick, confident, (lo, hi) = verdict([42.2, 50.2, 60.0], [25.2, 25.0, 25.1], 2.0)
    assert lo < 2.0 < hi, (lo, hi)
    assert confident is False
    assert pick == "offload", "an undecided measurement must fall back to the safe backend"


def test_a_single_run_still_decides():
    """One rep has no spread to speak of, so it behaves exactly as before."""
    assert verdict([96.0], [25.0], 2.0)[:2] == ("hybrid", True)
    assert verdict([40.0], [25.0], 2.0)[:2] == ("offload", True)


def test_exactly_at_the_threshold_is_not_hybrid():
    """`recommend` is a strict >, so the boundary resolves to offload, not a coin flip."""
    pick, confident, _ = verdict([50.0], [25.0], 2.0)
    assert (pick, confident) == ("offload", True)


class _FakeHostBank:
    built = 0

    def __init__(self, shape, dtype):
        type(self).built += 1
        self.tensor = torch.empty(shape, dtype=dtype)

    def pin(self):
        pass


def test_production_banks_are_reused_across_measurements(monkeypatch):
    """Registered pages are never released, so --reps must not pin a fresh set per run.

    Two banks of one gather set can share a shape; they are live together and must still
    be distinct banks.
    """
    from freetoken.moe import benchbw, host_banks

    monkeypatch.setattr(host_banks, "HostBank", _FakeHostBank)
    monkeypatch.setattr(_FakeHostBank, "built", 0)
    monkeypatch.setattr(benchbw, "_PRODUCTION_ALLOC", True)
    monkeypatch.setattr(benchbw, "_PRODUCTION_BANKS", {}, raising=False)
    monkeypatch.setattr(benchbw, "_LIVE_BENCH_BANKS", [])

    def gather_set():
        return {name: benchbw._alloc_bank(4, 64, dtype=torch.float16, role=("gather", "nvfp4", name))
                for name in ("gate_up_global", "down_global")}

    first = gather_set()
    assert _FakeHostBank.built == len(first)
    assert first["gate_up_global"] is not first["down_global"]

    second = gather_set()
    assert _FakeHostBank.built == len(first)
    assert all(second[k] is first[k] for k in first)


def test_step_cost_fit_splits_fixed_and_per_expert():
    from freetoken.moe.benchbw import _step_cost_fit

    fit = _step_cost_fit(0.27, 1.60, 8)  # 0.08 fixed + 0.19/expert
    assert fit["fit_ok"] is True
    assert abs(fit["fixed_ms"] - 0.08) < 1e-9 and abs(fit["per_expert_ms"] - 0.19) < 1e-9


def test_step_cost_fit_flags_a_negative_intercept():
    """t(top_k) > top_k * t(1) is noise, not a negative fixed cost."""
    from freetoken.moe.benchbw import _step_cost_fit

    fit = _step_cost_fit(0.10, 1.20, 8)
    assert fit["fixed_ms"] < 0
    assert fit["fit_ok"] is False
