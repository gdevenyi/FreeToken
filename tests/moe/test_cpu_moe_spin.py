"""CPU MoE worker hot-spin (``FREETOKEN_CPU_MOE_SPIN_MS``), CPU only.

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
DEFAULT_SPIN_MS = 50


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
    """Spin on, spin off, and a spin window shorter than the gap between tasks (the
    workers leave the spin and park, so the condvar wakes them) give identical outputs."""
    runs = {}
    for name, spin_ms, gap_s in [("spin", DEFAULT_SPIN_MS, 0), ("off", 0, 0), ("park", 1, 0.02)]:
        ex = make_executor(cache)
        ex._ext.set_worker_spin_ms(spin_ms)
        assert ex._ext.worker_spin_ms() == spin_ms
        runs[name] = _run_steps(ex, cache, steps=40 if gap_s == 0 else 8, gap_s=gap_s)
        del ex
    for name in ("off", "park"):
        for step, (got, want) in enumerate(zip(runs[name], runs["spin"])):
            assert torch.equal(got, want), f"{name} differs from spin at step {step}"


def test_spin_ms_env_overrides_the_default(cache, make_executor, monkeypatch):
    # an exported spin A/B setting or a small runner would move the default under test
    monkeypatch.delenv("FREETOKEN_CPU_MOE_SPIN_MS", raising=False)
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(8)))
    assert make_executor(cache)._ext.worker_spin_ms() == DEFAULT_SPIN_MS
    monkeypatch.setenv("FREETOKEN_CPU_MOE_SPIN_MS", "7")
    assert make_executor(cache)._ext.worker_spin_ms() == 7
    monkeypatch.setenv("FREETOKEN_CPU_MOE_SPIN_MS", "0")
    assert make_executor(cache)._ext.worker_spin_ms() == 0


@pytest.mark.parametrize("cpus, spin_ms", [(3, 0), (8, DEFAULT_SPIN_MS)])
def test_spin_needs_a_spare_cpu_for_the_launch_thread(cache, make_executor, monkeypatch, cpus, spin_ms):
    """Two pinned workers plus the launch thread and one more need four CPUs."""
    monkeypatch.delenv("FREETOKEN_CPU_MOE_SPIN_MS", raising=False)
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(cpus)))
    assert make_executor(cache, num_threads=2)._ext.worker_spin_ms() == spin_ms


def test_a_pre_spin_extension_still_serves_with_parked_workers(cache, make_executor, monkeypatch):
    """A prebuilt _cpu_moe .so from before the spin has no spin controls. Its workers always
    park (the spin 0 mode), so the executor must build and run on it rather than die after the
    whole model load."""
    from freetoken.kernel import _cpu_moe

    real = _cpu_moe.CpuMoeExecutor

    class PreSpinExecutor:
        def __init__(self, **kwargs):
            self._real = real(**kwargs)
            self._real.set_worker_spin_ms(0)

        def __getattr__(self, name):
            if name in ("worker_spin_ms", "set_worker_spin_ms"):
                raise AttributeError(name)
            return getattr(self._real, name)

    monkeypatch.setattr(_cpu_moe, "CpuMoeExecutor", PreSpinExecutor)
    ex = make_executor(cache)
    assert not ex._has_spin
    _run_steps(ex, cache, steps=4, gap_s=0)
