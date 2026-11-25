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
    assert rank0 == list(range(16)) and rank1 == list(range(16, 32))


def test_explicit_counts_of_tp_ranks_use_their_own_siblings_before_sharing(host):
    # 20 workers per rank: 16 physical cores, then 4 SMT siblings, all from the rank's own share
    pools = [cx.resolve_threads_and_affinity(20, r, 2)[1] for r in range(2)]
    assert all(len(p) == 20 and len(set(p)) == 20 for p in pools)
    assert not set(pools[0]) & set(pools[1])


def test_more_ranks_than_cores_still_gives_every_rank_workers(monkeypatch):
    monkeypatch.setattr(cx.os, "sched_getaffinity", lambda _pid: {0})
    monkeypatch.setattr(cx, "physical_core_cpus", lambda: [0])
    assert cx.resolve_threads_and_affinity(0, 1, 2) == (1, [0])


def test_rank_affinity_cpus_gives_each_tp_rank_its_full_socket_including_siblings(host):
    assert cx.rank_affinity_cpus(0, 1) is None
    assert cx.rank_affinity_cpus(0, 2) == set(range(16)) | set(range(32, 48))
    assert cx.rank_affinity_cpus(1, 2) == set(range(16, 32)) | set(range(48, 64))


def test_device_numa_cpus_parses_pci_sysfs_and_intersects_affinity(tmp_path, monkeypatch):
    from freetoken.moe import host_banks as hb

    pci = tmp_path / "bus" / "pci" / "devices" / "0000:2a:00.0"
    node0 = tmp_path / "devices" / "system" / "node" / "node0"
    pci.mkdir(parents=True)
    node0.mkdir(parents=True)
    (pci / "numa_node").write_text("0\n")
    (node0 / "cpulist").write_text("0-15,32-47\n")

    monkeypatch.setattr(hb, "_device_pci_bdf", lambda _dev: "0000:2a:00.0")
    monkeypatch.setattr(hb.os, "sched_getaffinity", lambda _pid: set(LOGICAL))
    assert hb.device_numa_cpus(0, sysfs=str(tmp_path)) == set(range(16)) | set(range(32, 48))


def test_rank_host_affinity_binds_and_restores_socket_cpus(host, monkeypatch):
    from freetoken.moe import host_banks as hb

    current = set(LOGICAL)
    seen: list[set[int]] = []

    def get_aff(_pid):
        return set(current)

    def set_aff(_pid, cpus):
        nonlocal current
        current = set(cpus)
        seen.append(set(cpus))

    monkeypatch.setattr(cx, "_tp_rank_and_size", lambda: (1, 2))
    monkeypatch.setattr(hb.os, "sched_getaffinity", get_aff)
    monkeypatch.setattr(hb.os, "sched_setaffinity", set_aff)

    with hb.rank_host_affinity():
        assert current == set(range(16, 32)) | set(range(48, 64))
    assert current == set(LOGICAL)
    assert seen == [set(range(16, 32)) | set(range(48, 64)), set(LOGICAL)]


def test_locked_fetch_copy_threads_stay_in_their_ranks_block(host, monkeypatch):
    # no SMT sibling free: the fallback must take this rank's free cores, never another rank's
    from types import SimpleNamespace

    from freetoken.moe import locked_fetch as lf

    monkeypatch.setattr(cx, "_tp_rank_and_size", lambda: (1, 2))
    monkeypatch.setattr(lf, "_smt_siblings", lambda cores: [])
    ex = SimpleNamespace(core_ids=list(range(16, 28)), _coord_core=28)
    assert lf._copy_cpus(ex, 4) == [29, 30, 31]
