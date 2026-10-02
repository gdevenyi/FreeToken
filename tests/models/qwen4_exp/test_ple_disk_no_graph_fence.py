"""The wait-sync graph fill must never wait on an event recorded after its own launch.

Upstream #588 fences ``_graph_pinned`` with ``_graph_consumed``, recorded once the launch returns and
synchronized by the next graph fill. Upstream runs the fill after the launch returns; here (fork
#65/#67) the fill runs on the filler thread BEFORE its own graph can pass its WAIT, so that fence
makes fill k wait on graph k, which waits on fill k: decode hangs on the second overlapped step.
The previous graph's read of ``_graph_pinned`` is already ordered before the next fill by the
per-fill readback event. This test drives the overlapped decode loop against an asynchronous,
in-order fake stream (launch returns at once; events record a stream position) and fails if any
step stalls, including when the fence is reintroduced.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import torch

import freetoken.models.qwen4_exp.ple_disk as ple_disk
from freetoken.models.qwen4_exp.ple_disk import DiskRowTable

EOS = 9


class _Stream:
    """One in-order stream: a graph item WAITs on the flag, then RESETs it; launch is async."""

    def __init__(self, flag: torch.Tensor):
        self.flag = flag
        self.cv = threading.Condition()
        self.queue: list[str] = []
        self.enqueued = self.done = 0
        self.stop = False
        threading.Thread(target=self._run, daemon=True).start()

    def enqueue(self, kind: str) -> None:
        with self.cv:
            self.queue.append(kind)
            self.enqueued += 1
            self.cv.notify_all()

    def _run(self) -> None:
        while True:
            with self.cv:
                self.cv.wait_for(lambda: self.queue or self.stop)
                if self.stop:
                    return
                kind = self.queue.pop(0)
            if kind == "graph":
                deadline = time.monotonic() + 10.0
                while int(self.flag[0]) != 1 and time.monotonic() < deadline:
                    time.sleep(0.0005)
                self.flag[0] = 0
            with self.cv:
                self.done += 1
                self.cv.notify_all()

    def wait_done(self, n: int, timeout: float) -> bool:
        with self.cv:
            return self.cv.wait_for(lambda: self.done >= n, timeout)

    def close(self) -> None:
        with self.cv:
            self.stop = True
            self.cv.notify_all()


class _Event:
    """cudaEvent semantics: synchronize() waits for the work enqueued before the LAST record()."""

    def __init__(self, stream: _Stream):
        self.stream, self.pos = stream, 0

    def record(self, _stream=None) -> None:
        self.pos = self.stream.enqueued

    def synchronize(self) -> None:
        if not self.stream.wait_done(self.pos, 2.0):
            raise TimeoutError(f"a fill waited on stream position {self.pos}, which never completed")


class _Store:
    def __init__(self, flag: torch.Tensor):
        self.flag = flag

    def stage(self, tokens_addr, n, staging_addr):
        pass

    def flush(self, signal_addr):
        if signal_addr:
            self.flag[0] = 1


def test_overlapped_graph_fills_never_wait_on_their_own_launch(monkeypatch):
    flag = torch.zeros(1, dtype=torch.int64)
    stream = _Stream(flag)
    monkeypatch.setattr(ple_disk.torch.cuda, "current_stream", lambda *a, **k: None)
    monkeypatch.setattr(ple_disk.torch.cuda, "Event", lambda *a, **k: _Event(stream))
    t = DiskRowTable.__new__(DiskRowTable)
    t._device = None
    t._wait_sync = True
    t._flag = flag
    t._store = _Store(flag)
    t._token_readback = torch.zeros(8, dtype=torch.int32)
    t._graph_pinned = torch.zeros(64, dtype=torch.uint8)
    t._eager_pinned = [torch.zeros(64, dtype=torch.uint8), torch.zeros(64, dtype=torch.uint8)]
    t._eager_read = [_Event(stream), _Event(stream)]
    t._eager_slot = 0
    # if upstream's post-launch fence is ever reintroduced, model it with real stream semantics
    # so this test fails by hanging the way production would, not on a missing attribute
    t._graph_consumed = _Event(stream)
    t._token_bytes = 1
    t.eos_token_id = EOS
    t.image_token_id = None

    tokens = [1, 2, 3]
    steps = 12
    try:
        for step in range(steps):
            new = 10 + step
            req = SimpleNamespace(input_ids=torch.tensor(tokens + [new]), device_len=len(tokens) + 1)
            batch = SimpleNamespace(is_decode=True, reqs=[req], padded_size=1,
                                    input_ids=torch.tensor([new], dtype=torch.int32))
            with t.forward_host_ctx(batch, use_graph=True):
                stream.enqueue("graph")  # cuGraphLaunch returns before the GPU passes the WAIT
            tokens.append(new)
            # the overlap loop keeps the engine at most one launch ahead of the GPU
            assert stream.wait_done(step, 4.0), f"decode hung: graph {step - 1} never completed"
            t._raise_failed_fill()
        assert stream.wait_done(steps, 4.0), "decode hung on the last step"
        t._raise_failed_fill()
    finally:
        t.close()
        stream.close()
