"""The PLE disk backend's wait-sync fill must not depend on the graph launch returning.

The captured decode graph WAITs on ``_flag``; ``cuGraphLaunch`` may block until the GPU drains
(a large graph fills the launch queue), and the GPU is parked on that WAIT. So the fill that sets
the flag has to run on its own thread, started before the launch (fork issue #65).
These tests stand in for the GPU with host fakes: the "launch" blocks until the flag is set.
"""

from __future__ import annotations

import queue
import threading
import time
from types import SimpleNamespace

import pytest
import torch

import freetoken.models.qwen4_exp.ple_disk as ple_disk
from freetoken.models.qwen4_exp.ple_disk import DiskRowTable

EOS = 9


class _Store:
    """Stands in for the C++ PleStore: records calls and the thread that made them."""

    def __init__(self, flag: torch.Tensor):
        self.flag = flag
        self.calls: list[tuple[str, list[int] | None, str]] = []

    def stage(self, tokens_addr, n, staging_addr):
        tokens = ctypes_read(tokens_addr, n + 2)
        self.calls.append(("stage", tokens, threading.current_thread().name))

    def flush(self, signal_addr):
        self.calls.append(("flush", None, threading.current_thread().name))
        if signal_addr:
            self.flag[0] = 1


def ctypes_read(addr: int, n: int) -> list[int]:
    import ctypes

    return list((ctypes.c_int64 * n).from_address(addr))


class _Event:
    def record(self, stream=None):
        pass

    def synchronize(self):
        pass


def _table(monkeypatch) -> DiskRowTable:
    # host-only: no CUDA stream behind the (faked) readback event
    monkeypatch.setattr(ple_disk.torch.cuda, "current_stream", lambda *a, **k: None)
    monkeypatch.setattr(ple_disk.torch.cuda, "Event", _Event)
    t = DiskRowTable.__new__(DiskRowTable)
    t._device = None
    t._wait_sync = True
    t._flag = torch.zeros(1, dtype=torch.int64)
    t._store = _Store(t._flag)
    t._token_readback = torch.zeros(8, dtype=torch.int32)
    t._graph_pinned = torch.zeros(64, dtype=torch.uint8)
    t._eager_pinned = torch.zeros(64, dtype=torch.uint8)
    t._token_bytes = 1
    t.eos_token_id = EOS
    t.image_token_id = None
    return t


def _close(t: DiskRowTable) -> None:
    close = getattr(t, "close", None)
    if close is not None:
        close()


def _decode_batch(prev_tokens: list[int], new_token: int):
    req = SimpleNamespace(input_ids=torch.tensor(prev_tokens + [new_token]), device_len=len(prev_tokens) + 1)
    return SimpleNamespace(is_decode=True, reqs=[req], padded_size=1,
                           input_ids=torch.tensor([new_token], dtype=torch.int32))


def _blocking_launch(flag: torch.Tensor, timeout: float = 5.0) -> bool:
    """A cuGraphLaunch that cannot return until the GPU passes the graph's WAIT(flag)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if int(flag[0]) == 1:
            flag[0] = 0  # the graph's WAIT-then-RESET
            return True
        time.sleep(0.001)
    return False


def test_graph_fill_does_not_wait_for_the_launch_to_return(monkeypatch):
    t = _table(monkeypatch)
    with t.forward_host_ctx(_decode_batch([1, 2, 3], 7), use_graph=True):
        assert _blocking_launch(t._flag), "deadlock: the fill only runs after the launch returns"
    _close(t)


def test_fills_run_in_step_order_on_one_thread_with_contexts_taken_before_the_launch(monkeypatch):
    t = _table(monkeypatch)
    for prev, new in (([1, 2, 3], 7), ([1, 2, 3, 7], 8), ([1, 2, 3, 7, 8], 4)):
        batch = _decode_batch(prev, new)
        with t.forward_host_ctx(batch, use_graph=True):
            # the engine mutates the request as soon as the launch returns (overlap scheduling)
            batch.reqs[0].input_ids = torch.tensor([0, 0, 0, 0, 0, 0])
            batch.reqs[0].device_len += 1
            assert _blocking_launch(t._flag)
    _close(t)
    staged = [c for c in t._store.calls if c[0] == "stage"]
    assert [c[1] for c in staged] == [[2, 3, 7], [3, 7, 8], [7, 8, 4]]
    assert len({c[2] for c in t._store.calls}) == 1  # the store stays single-threaded


def test_eager_fill_completes_before_the_launch(monkeypatch):
    t = _table(monkeypatch)
    req = SimpleNamespace(input_ids=torch.tensor([1, 2, 3, 4]), cached_len=1, device_len=4)
    batch = SimpleNamespace(is_decode=False, padded_reqs=[req])
    with t.forward_host_ctx(batch, use_graph=False):
        assert [c[0] for c in t._store.calls] == ["stage", "flush"]
    _close(t)


def test_a_failed_launch_leaves_no_stale_signal(monkeypatch):
    t = _table(monkeypatch)
    with pytest.raises(RuntimeError, match="launch failed"):
        with t.forward_host_ctx(_decode_batch([1, 2, 3], 7), use_graph=True):
            raise RuntimeError("launch failed")
    # no WAIT consumed it, so a leftover 1 would let the next step read stale rows
    assert int(t._flag[0]) == 0
    _close(t)


class _Gpu:
    """One in-order stream: each launched graph parks on WAIT(flag), then RESETs it."""

    def __init__(self, flag: torch.Tensor):
        self.flag = flag
        self.enqueued = self.completed = 0
        self.stalled = False
        self.cv = threading.Condition()
        self._launches: queue.SimpleQueue = queue.SimpleQueue()
        threading.Thread(target=self._run, daemon=True).start()

    def launch(self) -> None:
        self.enqueued += 1
        self._launches.put(True)

    def close(self) -> None:
        self._launches.put(False)

    def _run(self) -> None:
        while self._launches.get():
            self.stalled |= not _blocking_launch(self.flag)
            with self.cv:
                self.completed += 1
                self.cv.notify_all()

    def wait_completed(self, n: int, timeout: float = 5.0) -> bool:
        with self.cv:
            return self.cv.wait_for(lambda: self.completed >= n, timeout)


class _StreamEvent:
    """cudaEvent semantics: synchronize() waits for the work enqueued before the LAST record."""

    def __init__(self, gpu: _Gpu):
        self.gpu, self.target = gpu, 0

    def record(self, stream=None):
        self.target = self.gpu.enqueued

    def synchronize(self):
        if not self.gpu.wait_completed(self.target, timeout=1.0):
            raise TimeoutError("the readback waits on a graph parked on this fill's own WAIT")


def _two_steps_behind_a_slow_filler(t: DiskRowTable, gpu: _Gpu) -> None:
    """Steps N and N+1 both launch before the filler thread reaches fill N."""
    gate = threading.Event()
    t._filler().submit(gate.wait)
    for prev, new in (([1, 2, 3], 7), ([1, 2, 3, 7], 8)):
        with t.forward_host_ctx(_decode_batch(prev, new), use_graph=True):
            gpu.launch()
    gate.set()
    assert gpu.wait_completed(2), "decode deadlocked"
    t._filler().submit(lambda: None).result()  # every submitted fill has finished
    gpu.close()


def test_each_fill_waits_on_its_own_steps_readback(monkeypatch):
    """A readback event shared across steps is re-recorded by step N+1 behind graph N, so
    fill N would wait on the graph that is parked on fill N's own WAIT."""
    t = _table(monkeypatch)
    gpu = _Gpu(t._flag)
    monkeypatch.setattr(ple_disk.torch.cuda, "Event", lambda: _StreamEvent(gpu))
    _two_steps_behind_a_slow_filler(t, gpu)
    t._raise_failed_fill()
    _close(t)
    assert not gpu.stalled
    # the fake D2H readback is not stream-ordered, so compare only the contexts
    assert [c[1][:2] for c in t._store.calls if c[0] == "stage"] == [[2, 3], [3, 7]]


def test_a_failed_fill_is_raised_after_the_next_step_was_submitted(monkeypatch):
    """Fill N fails (its graph runs on stale rows) after step N+1 already submitted: the error
    must still reach the engine thread at the next launch."""
    t = _table(monkeypatch)
    gpu = _Gpu(t._flag)
    monkeypatch.setattr(ple_disk.torch.cuda, "Event", lambda: _StreamEvent(gpu))
    flush = t._store.flush

    def fail_step_n(signal_addr):
        if t._store.calls[-1][1][:2] == [2, 3]:  # the run staged just before is step N's
            raise OSError("PLE row read failed")
        flush(signal_addr)

    t._store.flush = fail_step_n
    _two_steps_behind_a_slow_filler(t, gpu)
    req = SimpleNamespace(input_ids=torch.tensor([1, 2, 3, 4]), cached_len=1, device_len=4)
    with pytest.raises(OSError, match="PLE row read failed"):
        t.host_fill_batch(SimpleNamespace(is_decode=False, padded_reqs=[req]), use_graph=False)
    _close(t)


def test_the_readback_event_is_never_freed_on_the_filler_thread(monkeypatch):
    """cuEventDestroy on the filler blocks on the driver lock a blocked cuGraphLaunch holds (host
    nodes in the graph make the launch wait for the GPU, which is parked on the next fill's WAIT)."""
    import gc

    freed_on: list[str] = []

    class _TrackedEvent(_Event):
        def __del__(self):
            freed_on.append(threading.current_thread().name)

    t = _table(monkeypatch)
    monkeypatch.setattr(ple_disk.torch.cuda, "Event", _TrackedEvent)
    for prev, new in (([1, 2, 3], 7), ([1, 2, 3, 7], 8)):
        with t.forward_host_ctx(_decode_batch(prev, new), use_graph=True):
            assert _blocking_launch(t._flag)
    time.sleep(0.05)  # let the filler drop its work items
    gc.collect()
    t._raise_failed_fill()
    gc.collect()
    assert "ple-filler" not in freed_on, freed_on
    assert len(t._readback_pool()) >= 1  # finished fills hand their events back for reuse
    _close(t)
