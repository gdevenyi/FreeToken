"""Regression test for #301: the TP rendezvous store must not fall back to a wildcard
bind. torch's tcp:// rendezvous handler ignores the host it's given when binding the
master's listening socket and always listens on every interface, so
init_method="tcp://127.0.0.1:PORT" looks loopback-only but isn't -- see
Engine._make_distributed_store. This drives the real store-construction path across two
spawned local ranks (CPU/gloo) and checks the actual bind address, not just the URL.
"""

from __future__ import annotations

import gc
import multiprocessing as mp
import os
from datetime import timedelta
from types import SimpleNamespace

import torch
import torch.distributed as dist

from freetoken.distributed import DistributedInfo
from freetoken.engine.engine import Engine

WORLD_SIZE = 2
TIMEOUT = timedelta(seconds=15)


def _config(rank: int, port: int) -> SimpleNamespace:
    return SimpleNamespace(
        tp_info=DistributedInfo(rank, WORLD_SIZE),
        distributed_timeout=TIMEOUT.total_seconds(),
        distributed_port=port,
    )


def _run_rank(rank: int, port: int, result_q: mp.Queue) -> None:
    engine = Engine.__new__(Engine)  # only _make_distributed_store is under test
    store = engine._make_distributed_store(_config(rank, port))
    dist.init_process_group(
        backend="gloo", rank=rank, world_size=WORLD_SIZE, timeout=TIMEOUT, store=store
    )
    total = torch.tensor([float(rank)])
    dist.all_reduce(total)
    bind_host = engine._distributed_listen_addr[0] if rank == 0 else None
    result_q.put((rank, total.item(), bind_host))
    dist.destroy_process_group()


def test_two_ranks_rendezvous_with_loopback_only_store():
    ctx = mp.get_context("spawn")
    port = 29511
    result_q = ctx.Queue()
    procs = [ctx.Process(target=_run_rank, args=(rank, port, result_q)) for rank in range(WORLD_SIZE)]
    for p in procs:
        p.start()

    results = {}
    for _ in procs:
        rank, total, bind_host = result_q.get(timeout=30)
        results[rank] = (total, bind_host)
    for p in procs:
        p.join(timeout=30)
        assert p.exitcode == 0

    # both ranks completed the same all_reduce over the store -- rendezvous worked
    assert results[0][0] == results[1][0] == 1.0  # sum(0, 1)
    # the master's listening socket was pre-bound to loopback, not the wildcard
    # address the C10d TCPStore server binds to on its own
    assert results[0][1] == "127.0.0.1"


def _teardown_rank(port: int, result_q: mp.Queue) -> None:
    listen_fds = []
    real_store = dist.TCPStore

    def recording_store(*args, **kwargs):
        listen_fds.append(kwargs.get("master_listen_fd"))
        return real_store(*args, **kwargs)

    dist.TCPStore = recording_store
    engine = Engine.__new__(Engine)
    config = SimpleNamespace(
        tp_info=DistributedInfo(0, 1), distributed_timeout=TIMEOUT.total_seconds(),
        distributed_port=port,
    )
    store = engine._make_distributed_store(config)
    dist.init_process_group(backend="gloo", rank=0, world_size=1, timeout=TIMEOUT, store=store)
    dist.destroy_process_group()
    del store
    gc.collect()
    # the listening fd number is reused by an unrelated file, as a later open() would
    fd = listen_fds[0]
    r, w = os.pipe()
    os.dup2(r, fd)
    del engine
    gc.collect()
    try:
        os.fstat(fd)
        result_q.put("open")
    except OSError as exc:
        result_q.put(f"closed: {exc}")


def test_the_engine_does_not_close_the_listen_fd_the_store_owns():
    # TCPStore takes ownership of master_listen_fd and closes it itself; an engine that also
    # kept the Python socket closed whatever file reused that number when it was freed
    ctx = mp.get_context("spawn")
    result_q = ctx.Queue()
    proc = ctx.Process(target=_teardown_rank, args=(29513, result_q))
    proc.start()
    outcome = result_q.get(timeout=30)
    proc.join(timeout=30)
    assert proc.exitcode == 0
    assert outcome == "open"
