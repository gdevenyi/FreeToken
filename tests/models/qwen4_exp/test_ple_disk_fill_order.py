"""The PLE disk backend's wait-sync fill must not depend on the graph launch returning.

The captured decode graph WAITs on ``_flag``; ``cuGraphLaunch`` may block until the GPU drains
(a large graph fills the launch queue), and the GPU is parked on that WAIT. So the fill that sets
the flag has to run on its own thread, started before the launch (fork issue #65).
These tests stand in for the GPU with host fakes: the "launch" blocks until the flag is set.
"""

from __future__ import annotations

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
    t = DiskRowTable.__new__(DiskRowTable)
    t._device = None
    t._wait_sync = True
    t._flag = torch.zeros(1, dtype=torch.int64)
    t._store = _Store(t._flag)
    t._token_readback = torch.zeros(8, dtype=torch.int32)
    t._readback_event = _Event()
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
