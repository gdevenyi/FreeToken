"""CPU MoE pool spin-wait (``FREETOKEN_CPU_MOE_SPIN``), CPU only.

Workers either catch the next task while spinning, or park on the condvar and get woken;
both hand-offs must run every task exactly once on that task's inputs. The pool drops the
spin when its pinned threads leave no CPU for the launch thread.
"""

from __future__ import annotations

import os
import time
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as Fn

L, E, H, I, TOP_K, BS = 2, 8, 256, 128, 2, 3
SPIN_WINDOW_S = 0.05  # kSpinWindow in csrc/cpu_moe/cpu_moe_ext.cpp


@pytest.fixture
def cache():
    gen = torch.Generator().manual_seed(0)
    gate_up = (torch.randn(L * E, 2 * I, H, generator=gen) * 0.1).to(torch.bfloat16)
    down = (torch.randn(L * E, H, I, generator=gen) * 0.1).to(torch.bfloat16)
    return SimpleNamespace(
        quant_format="bf16",
        bank_sources={"gate_up": list(gate_up.split(E)), "down": list(down.split(E))},
        num_layers=L,
        num_experts=E,
    )


@pytest.fixture
def make_executor():
    from freetoken.moe.cpu_executor import CpuMoeExecutor

    # the executor clamps torch's intra-op pool for its pinned workers; keep that local
    threads = torch.get_num_threads()

    def make(cache, num_threads=2):
        return CpuMoeExecutor(
            cache, top_k=TOP_K, activation="silu", apply_router_weight_on_input=False,
            num_threads=num_threads, max_tokens=BS, device=torch.device("cpu"),
        )

    yield make
    torch.set_num_threads(threads)


def _reference(cache, layer, x, ids, w):
    gate_up = cache.bank_sources["gate_up"][layer].float()
    down = cache.bank_sources["down"][layer].float()
    out = torch.zeros(x.shape[0], H)
    for t in range(x.shape[0]):
        for k in range(TOP_K):
            e = int(ids[t, k])
            h = gate_up[e] @ x[t].float()
            out[t] += w[t, k] * (down[e] @ (Fn.silu(h[:I]) * h[I:]))
    return out


def _run_steps(ex, cache, steps, gap_s):
    x = torch.empty(BS, H, dtype=torch.bfloat16)
    ids = torch.empty(BS, TOP_K, dtype=torch.int32)
    w = torch.empty(BS, TOP_K, dtype=torch.float32)
    y = torch.empty(BS, H, dtype=torch.bfloat16)
    tasks = [
        ex._ext.create_task(layer, BS, x.data_ptr(), ids.data_ptr(), w.data_ptr(), y.data_ptr())
        for layer in range(L)
    ]
    gen = torch.Generator().manual_seed(1)
    outputs = []
    for step in range(steps):
        layer = step % L
        x.copy_(torch.randn(BS, H, generator=gen))
        ids.copy_(torch.stack([torch.randperm(E, generator=gen)[:TOP_K] for _ in range(BS)]))
        w.copy_(torch.rand(BS, TOP_K, generator=gen))
        if gap_s:
            time.sleep(gap_s)
        ex._ext.run_task(tasks[layer])
        ref = _reference(cache, layer, x, ids, w)
        rel = (y.float() - ref).abs().max() / ref.abs().max()
        assert rel < 2e-2, f"step {step}: rel err {rel.item()}"
        outputs.append(y.clone())
    return outputs


def test_spinning_and_parked_workers_run_every_task(cache, make_executor):
    """Spin on, spin off, and a gap between tasks longer than the spin window (the workers
    leave the spin and park, so the condvar wakes them) give identical outputs."""
    runs = {}
    for name, spin, gap_s in [("spin", True, 0), ("off", False, 0), ("park", True, 2 * SPIN_WINDOW_S)]:
        ex = make_executor(cache)
        ex._ext.set_spin_wait(spin)
        assert ex._ext.get_spin_wait() is spin
        runs[name] = _run_steps(ex, cache, steps=40 if gap_s == 0 else 8, gap_s=gap_s)
        del ex
    for name in ("off", "park"):
        for step, (got, want) in enumerate(zip(runs[name], runs["spin"])):
            assert torch.equal(got, want), f"{name} differs from spin at step {step}"


def test_spin_env_opts_out(cache, make_executor, monkeypatch):
    from freetoken.moe import cpu_executor

    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(8)))
    monkeypatch.setattr(cpu_executor, "_SPIN_WAIT", True)
    assert make_executor(cache)._ext.get_spin_wait() is True
    monkeypatch.setattr(cpu_executor, "_SPIN_WAIT", False)
    ex = make_executor(cache)
    assert ex.spin_wait is False and ex._ext.get_spin_wait() is False


@pytest.mark.parametrize("cpus, spin", [(3, False), (8, True)])
def test_spin_needs_a_spare_cpu_for_the_launch_thread(cache, make_executor, monkeypatch, cpus, spin):
    """Two pinned workers plus the main and callback threads need four CPUs."""
    from freetoken.moe import cpu_executor

    monkeypatch.setattr(cpu_executor, "_SPIN_WAIT", True)
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(cpus)))
    assert make_executor(cache, num_threads=2)._ext.get_spin_wait() is spin


@pytest.mark.parametrize("controls", [(), ("set_worker_spin_ms", "worker_spin_ms")])
def test_a_pre_spin_extension_still_serves(cache, make_executor, monkeypatch, controls):
    """A prebuilt _cpu_moe .so from before the spin-wait has no set_spin_wait (or only the
    older worker-spin control). The executor must build and run on it rather than die after
    the whole model load."""
    from freetoken.kernel import _cpu_moe

    real = _cpu_moe.CpuMoeExecutor
    calls = []

    class PreSpinExecutor:
        def __init__(self, **kwargs):
            self._real = real(**kwargs)
            self._real.set_spin_wait(False)

        def __getattr__(self, name):
            if name in ("set_spin_wait", "get_spin_wait"):
                raise AttributeError(name)
            return getattr(self._real, name)

    for name in controls:
        setattr(PreSpinExecutor, name, lambda self, *a, _n=name: calls.append((_n, *a)) or 0)
    monkeypatch.setattr(_cpu_moe, "CpuMoeExecutor", PreSpinExecutor)
    ex = make_executor(cache)
    if controls:
        assert calls == [("set_worker_spin_ms", 50 if ex.spin_wait else 0)]
    else:
        assert ex.spin_wait is False
    _run_steps(ex, cache, steps=4, gap_s=0)
