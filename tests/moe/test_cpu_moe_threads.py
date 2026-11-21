"""CPU MoE worker placement under tensor parallelism (no GPU needed).

Every TP rank builds its own pinned, spin-waiting CPU MoE pool on the same host. If two
ranks pin their workers to the same cores, nothing fails: decode just gets slower, because
the spin barrier degrades badly when oversubscribed. These tests guard that silent case."""

from __future__ import annotations

import pytest

from freetoken.moe import cpu_executor as cx

# a two-socket host like chatdnp: 32 physical cores (0-15 socket 0, 16-31 socket 1), SMT siblings 32-63
PHYSICAL = list(range(32))
LOGICAL = list(range(64))


@pytest.fixture
def host(monkeypatch):
    monkeypatch.setattr(cx.os, "sched_getaffinity", lambda _pid: set(LOGICAL))
    monkeypatch.setattr(cx, "physical_core_cpus", lambda: list(PHYSICAL))


def test_auto_pool_without_tp_takes_every_physical_core(host):
    assert cx.resolve_threads_and_affinity(0) == (32, PHYSICAL)


@pytest.mark.parametrize("ranks", [2, 4])
def test_auto_pools_of_tp_ranks_never_share_a_core(host, ranks):
    pools = [cx.resolve_threads_and_affinity(0, r, ranks)[1] for r in range(ranks)]
    flat = [c for pool in pools for c in pool]
    assert len(flat) == len(set(flat)), "two TP ranks pin workers to the same core"
    assert sorted(flat) == PHYSICAL  # together they still use the whole machine


def test_tp_rank_blocks_are_contiguous_so_a_rank_stays_on_one_socket(host):
    rank0 = cx.resolve_threads_and_affinity(0, 0, 2)[1]
    rank1 = cx.resolve_threads_and_affinity(0, 1, 2)[1]
    assert rank0 == list(range(0, 16)) and rank1 == list(range(16, 32))


def test_explicit_counts_of_tp_ranks_use_their_own_siblings_before_sharing(host):
    # 20 workers per rank: 16 physical cores, then 4 SMT siblings, all from the rank's own share
    pools = [cx.resolve_threads_and_affinity(20, r, 2)[1] for r in range(2)]
    assert all(len(p) == 20 and len(set(p)) == 20 for p in pools)
    assert not set(pools[0]) & set(pools[1])


def test_more_ranks_than_cores_still_gives_every_rank_workers(monkeypatch):
    monkeypatch.setattr(cx.os, "sched_getaffinity", lambda _pid: {0})
    monkeypatch.setattr(cx, "physical_core_cpus", lambda: [0])
    assert cx.resolve_threads_and_affinity(0, 1, 2) == (1, [0])
