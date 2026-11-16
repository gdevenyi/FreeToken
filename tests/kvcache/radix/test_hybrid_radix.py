"""HybridRadixCache -- the GDN-snapshot currency.

What is genuinely different from the SWA sibling, and therefore what this module pins:

  * a snapshot is ONE opaque external slot id attached at a node's END boundary (point-attached,
    not a window), so a *prefix* of a snapshot-bearing node is not reusable at all;
  * ``match_prefix`` truncates the reusable prefix to the DEEPEST live snapshot;
  * ``split_at`` does not copy the snapshot -- the root-side half comes out empty;
  * ``insert`` dedups an existing snapshot, and refills a node whose snapshot was evicted;
  * ``evict_mamba`` counts SNAPSHOTS, tombstones internal nodes in place (keeping their KV), and
    frees KV eagerly when it takes a leaf, cascading through the tombstone leaves it exposes.

Page size: apart from the constructor's ``CHUNK_SIZE % page_size == 0`` assertion, the class is
page_size-agnostic -- ``align_down`` in ``_walk`` / ``insert`` is the only page-size-sensitive
code.  The scenarios therefore run at one page size (4, big enough that page keying differs from
token keying); the truncation test sweeps the page-size matrix.

Expectations come from the page-keyed reference model in ``model.py``; ``Session.do_*`` compares
every public result against it and ``Session.check()`` runs the invariant battery.
"""
from __future__ import annotations

import time
from typing import Sequence, Tuple

import pytest
import torch

from .driver import CacheSpec, Session

PAGE = 4
SPEC = CacheSpec("hybrid", PAGE)


def page_ids(page_size: int, *labels: int) -> Tuple[int, ...]:
    """Concatenate whole pages.  Pages deliberately share leading tokens (the lead repeats every
    three labels), which is exactly the shape a token-keyed expectation gets wrong once
    page_size > 1: the tree's reuse unit is a whole page."""
    out = []
    for lab in labels:
        if page_size == 1:
            out.append(lab)
        else:
            out.extend([(lab % 3) + 1] + [7] * (page_size - 2) + [lab])
    return tuple(out)


def ids(*labels: int) -> Tuple[int, ...]:
    return page_ids(PAGE, *labels)


def donated(s: Session) -> int:
    """The GDN slot the harness donated on the most recent insert."""
    assert s.second is not None
    return s.second.handed[-1]


def events(s: Session):
    """Reference-model branch counters (``Session.events`` is documented but not implemented)."""
    return s.model.events


def full_size(s: Session) -> int:
    return s.model.counters()["full_evictable"]


def live_snapshots(s: Session) -> int:
    c = s.model.counters()
    return c["mamba_evictable"] + c["mamba_protected"]


def host_snapshots(s: Session) -> int:
    c = s.model.counters()
    return c["mamba_host_evictable"] + c["mamba_host_protected"]


@pytest.fixture
def hyb() -> Session:
    """A hybrid cache at page_size=4 with its reference model and slot ledgers."""
    return Session(SPEC)


def two_node_tree(s: Session) -> Tuple[Sequence[int], int, int]:
    """X=[page1,page2] with snapshot mx, its child Y=[page3,page4] with snapshot my.

    Built through the realistic lifecycle, so Y's insert reuses X's own slots and X ends up
    strictly older than Y (the deterministic clock makes that ordering reproducible)."""
    s.do_insert(ids(1, 2))
    mx = donated(s)
    s.do_request(ids(1, 2, 3, 4), prompt_pages=2)
    my = donated(s)
    s.check()
    return ids(1, 2, 3, 4), mx, my


# --------------------------------------------------------------------------- constructor
@pytest.mark.parametrize("page_size, ok", [(1, True), (4, True), (64, True),
                                           (3, False), (48, False), (128, False)])
def test_page_size_must_divide_chunk_size(page_size, ok):
    """Snapshots land on x CHUNK_SIZE boundaries, so a page must not straddle one."""
    from freetoken.kernel.fla.chunk import CHUNK_SIZE
    from freetoken.kvcache.hybrid_radix_cache import HybridRadixCache

    assert (CHUNK_SIZE % page_size == 0) is ok, "the parameter table assumed CHUNK_SIZE == 64"
    if not ok:
        with pytest.raises(AssertionError,
                           match=rf"CHUNK_SIZE\({CHUNK_SIZE}\) % page_size\({page_size}\)"):
            HybridRadixCache(torch.device("cpu"), page_size=page_size)
        return
    c = HybridRadixCache(torch.device("cpu"), page_size=page_size)
    assert c.page_size == page_size
    assert (c.full_evictable_size, c.mamba_evictable_size) == (0, 0)
    cold = c.match_prefix(torch.tensor([1] * page_size, dtype=torch.int64))
    assert (cold.cached_len, cold.mamba_value) == (0, None)


# --------------------------------------------------------------------------- insert / match
def test_insert_attaches_snapshot_and_match_restores_it(hyb):
    slots = hyb.kv.take(2 * PAGE)
    got, exp = hyb.do_insert(ids(1, 2), slots=slots)
    assert (got.matched_len, got.second_exists) == (0, False)
    assert exp.adopted == slots and exp.dups == []
    hyb.check()

    m, _ = hyb.do_match(ids(1, 2))
    assert m.cached_len == 2 * PAGE
    assert m.indices == slots
    assert m.second == donated(hyb)
    assert events(hyb)["insert.snapshot_attach"] == 1
    assert live_snapshots(hyb) == 1


def test_deepest_snapshot_wins_and_unmatched_suffix_is_dropped(hyb):
    chain, mx, my = two_node_tree(hyb)
    assert mx != my

    m, _ = hyb.do_match(chain)
    assert (m.cached_len, m.second) == (4 * PAGE, my)          # the deeper snapshot supersedes
    # a query running past the end of the tree still resumes from the deepest snapshot
    m, _ = hyb.do_match(chain + ids(5))
    assert (m.cached_len, m.second) == (4 * PAGE, my)
    assert events(hyb)["match.snapshot_truncation"] == 0        # nothing was truncated: 4P is the end
    hyb.check()


def test_split_leaves_the_snapshot_on_the_suffix_half(hyb):
    """``split_at`` does not copy ``mamba_value``, so the root-side half of a split node has no
    snapshot and a match that ends there falls back to the nearest ancestor that does."""
    chain, mx, my = two_node_tree(hyb)

    m, _ = hyb.do_match(ids(1, 2, 3, 5))       # diverges inside Y -> Y splits into [3] + [4]
    assert events(hyb)["node.split"] == 1
    assert events(hyb)["match.snapshot_truncation"] == 1
    assert (m.cached_len, m.second) == (2 * PAGE, mx)           # truncated back to X's snapshot
    hyb.check()

    m, _ = hyb.do_match(chain)                 # the suffix half kept my, so the deep hit survives
    assert (m.cached_len, m.second) == (4 * PAGE, my)
    assert live_snapshots(hyb) == 2
    assert full_size(hyb) == 4 * PAGE                          # a split moves no KV


def test_prefix_of_a_snapshot_node_is_not_reusable(hyb):
    """The snapshot is point-attached at the node's end boundary: matching an interior page
    boundary of that node yields nothing at all, not a shortened hit."""
    slots = hyb.kv.take(4 * PAGE)
    hyb.do_insert(ids(1, 2, 3, 4), slots=slots)
    ms = donated(hyb)
    hyb.check()

    m, _ = hyb.do_match(ids(1, 2))             # splits the node; the [1,2] half owns no snapshot
    assert (m.cached_len, m.indices, m.second) == (0, [], None)
    assert events(hyb)["node.split"] == 1
    assert events(hyb)["match.no_snapshot"] == 1

    m, _ = hyb.do_match(ids(1, 2, 3, 4))       # ... while the full prefix is still a hit
    assert (m.cached_len, m.indices, m.second) == (4 * PAGE, slots, ms)
    assert full_size(hyb) == 4 * PAGE
    hyb.check()


def test_insert_dedups_and_the_caller_frees_the_donated_slot(hyb):
    hyb.do_insert(ids(1, 2))
    mx = donated(hyb)

    slots = hyb.kv.take(2 * PAGE)
    got, exp = hyb.do_insert(ids(1, 2), slots=slots)            # same boundary, second donation
    assert (got.matched_len, got.second_exists) == (2 * PAGE, True)
    assert exp.dups == slots and exp.adopted == []               # caller keeps every duplicate page
    assert events(hyb)["insert.snapshot_dedup"] == 1
    assert live_snapshots(hyb) == 1                              # still exactly one snapshot
    hyb.check()

    m, _ = hyb.do_match(ids(1, 2))
    assert m.second == mx                                        # the original snapshot is kept
    assert donated(hyb) in hyb.second.free                       # the loser was handed back


@pytest.mark.parametrize("P", [1, 4, 64], ids=["p1", "p4", "p64"])
def test_insert_stores_whole_pages_only(P):
    """``align_down`` governs both ends: fewer tokens than a page reaches the root (which cannot
    hold a snapshot, hence ``exists=True``), and a ragged tail is left with the caller."""
    s = Session(CacheSpec("hybrid", P))
    stub = page_ids(P, 1)[: P - 1]                               # () at P == 1
    got, _ = s.do_insert(stub)
    assert (got.matched_len, got.second_exists) == (0, True)
    assert full_size(s) == 0 and live_snapshots(s) == 0
    s.check()

    slots = s.kv.take(2 * P + (P - 1))
    got, exp = s.do_insert(page_ids(P, 1, 2) + page_ids(P, 3)[: P - 1], slots=slots)
    assert (got.matched_len, got.second_exists) == (0, False)
    assert exp.adopted == slots[: 2 * P]                         # the tail was never adopted
    s.check()

    m, _ = s.do_match(page_ids(P, 1, 2))
    assert (m.cached_len, m.indices) == (2 * P, slots[: 2 * P])
    assert full_size(s) == 2 * P


# --------------------------------------------------------------------------- evict_mamba
def test_evict_mamba_tombstones_an_internal_node_and_keeps_its_kv(hyb):
    chain, mx, my = two_node_tree(hyb)

    hyb.do_evict_second(1)                     # X is the LRU snapshot and is internal
    assert events(hyb)["evict_mamba.tombstone_in_place"] == 1
    assert mx in hyb.second.free and hyb.kv.free == set()        # no KV was reclaimed
    assert full_size(hyb) == 4 * PAGE and live_snapshots(hyb) == 1
    hyb.check()

    m, _ = hyb.do_match(ids(1, 2))             # the tombstoned boundary is no longer resumable
    assert (m.cached_len, m.second) == (0, None)
    m, _ = hyb.do_match(chain)                 # the descendant snapshot is untouched
    assert (m.cached_len, m.second) == (4 * PAGE, my)


def test_insert_refills_a_tombstoned_node(hyb):
    two_node_tree(hyb)
    hyb.do_evict_second(1)                     # tombstone X (KV kept, snapshot gone)
    hyb.check()

    slots = hyb.kv.take(2 * PAGE)
    got, exp = hyb.do_insert(ids(1, 2), slots=slots)
    assert (got.matched_len, got.second_exists) == (2 * PAGE, False)   # attaches, does not dedup
    assert exp.dups == slots                                          # KV stays canonical
    assert events(hyb)["insert.snapshot_attach"] == 3
    hyb.check()

    m, _ = hyb.do_match(ids(1, 2))
    assert (m.cached_len, m.second) == (2 * PAGE, donated(hyb))
    assert live_snapshots(hyb) == 2


def test_evict_mamba_on_a_leaf_frees_kv_and_cascades_through_tombstones(hyb):
    chain, _mx, _my = two_node_tree(hyb)
    hyb.do_evict_second(1)                     # X -> KV-only tombstone
    hyb.do_evict_second(1)                     # Y is now the only snapshot node, and a leaf
    assert events(hyb)["evict_mamba.leaf_free"] == 1
    assert events(hyb)["evict_mamba.cascade"] == 1                    # X reclaimed in the same call
    hyb.check()

    assert full_size(hyb) == 0 and live_snapshots(hyb) == 0
    assert hyb.kv.in_use() == set() and hyb.second.in_use() == set()
    m, _ = hyb.do_match(chain)
    assert (m.cached_len, m.second) == (0, None)


def test_evict_mamba_counts_snapshots_not_tokens(hyb):
    """A one-page node holds PAGE tokens but one snapshot; ``evict_mamba(2)`` must take two
    snapshots, which a token-counting loop would never do."""
    hyb.do_insert(ids(1))
    hyb.do_request(ids(1, 2), prompt_pages=1)
    hyb.do_request(ids(1, 2, 3), prompt_pages=2)
    hyb.check()
    assert live_snapshots(hyb) == 3 and full_size(hyb) == 3 * PAGE

    hyb.do_evict_second(2)
    assert len(hyb.second.free) == 2                                  # exactly two snapshots
    assert events(hyb)["evict_mamba.tombstone_in_place"] == 2         # both were internal
    assert hyb.kv.free == set() and full_size(hyb) == 3 * PAGE        # no KV touched
    assert live_snapshots(hyb) == 1
    hyb.check()


def test_snapshot_slots_are_conserved_across_eviction_waves(hyb):
    """Every donated slot comes back exactly once (the ledger raises on a double free), and
    draining the snapshot currency drains the KV with it via the leaf cascade."""
    n = 8
    hyb.do_insert(ids(1))
    for k in range(2, n + 1):
        hyb.do_request(ids(*range(1, k + 1)), prompt_pages=k - 1)
    hyb.check()
    assert live_snapshots(hyb) == n and full_size(hyb) == n * PAGE
    handed = list(hyb.second.handed)

    for guard in range(10):
        if live_snapshots(hyb) == 0:
            break
        hyb.do_evict_second(3)
        hyb.check()
    else:
        pytest.fail(f"evict_mamba did not converge: {live_snapshots(hyb)} snapshot(s) left")

    assert sorted(hyb.second.free) == sorted(handed)
    assert hyb.kv.in_use() == set() and full_size(hyb) == 0
    assert hyb.do_match(ids(*range(1, n + 1)))[0].cached_len == 0


# --------------------------------------------------------------------------- evict_full / locks
def test_evict_full_takes_the_leaf_snapshot_and_leaves_the_ancestor_usable(hyb):
    chain, mx, my = two_node_tree(hyb)

    hyb.do_evict_full(2 * PAGE)                # the only unlocked leaf is Y
    assert my in hyb.second.free and mx not in hyb.second.free
    assert events(hyb)["evict_full.cascade"] == 0   # X still owns a snapshot, so it is not reclaimed
    assert full_size(hyb) == 2 * PAGE and live_snapshots(hyb) == 1
    hyb.check()

    m, _ = hyb.do_match(chain)                 # the request re-hits at the surviving boundary
    assert (m.cached_len, m.second) == (2 * PAGE, mx)


def test_evict_full_cascades_through_an_exposed_tombstone_leaf(hyb):
    """A KV-only tombstone leaf is reclaimed eagerly in the same ``evict_full`` that exposes it,
    so ``evict_full(2*PAGE)`` returns twice that many tokens."""
    two_node_tree(hyb)
    hyb.do_evict_second(1)                     # X -> tombstone
    hyb.do_evict_full(2 * PAGE)                # evicting Y exposes X and reclaims it too
    assert events(hyb)["evict_full.cascade"] == 1
    hyb.check()

    assert full_size(hyb) == 0 and live_snapshots(hyb) == 0
    assert hyb.kv.in_use() == set() and hyb.second.in_use() == set()


def test_lock_pins_the_snapshot_and_the_whole_kv_path(hyb):
    """``inc_lock`` takes a mamba ref on the matched node and a full ref up to the root, so
    neither currency can evict it -- but locking a descendant does NOT protect an ancestor's
    snapshot (``full_ref >= mamba_ref``, not the other way round)."""
    chain, mx, _my = two_node_tree(hyb)
    held = hyb.do_lock(chain)
    assert held is not None
    hyb.check()

    hyb.do_evict_full(4 * PAGE)                # Y is locked, X is internal -> nothing is evictable
    assert hyb.kv.free == set() and full_size(hyb) == 0        # all 4 pages are protected now

    hyb.do_evict_second(1)                     # X's snapshot is unprotected -> tombstoned in place
    assert hyb.second.free == {mx}
    hyb.do_evict_second(1)                     # Y's snapshot is pinned by the lock
    assert hyb.second.free == {mx}
    hyb.check()

    hyb.do_unlock(held)
    assert full_size(hyb) == 4 * PAGE
    hyb.do_evict_full(2 * PAGE)                # Y is evictable again; X cascades behind it
    assert events(hyb)["evict_full.cascade"] == 1
    assert hyb.kv.in_use() == set() and hyb.second.in_use() == set()
    hyb.check()


# --------------------------------------------------------------------------- host bank (--mamba-host-slots)
def test_evict_mamba_demotes_to_the_host_bank_and_a_match_resumes_there():
    """With a bank, ``evict_mamba`` copies the snapshot to the host instead of freeing it: the node
    keeps its KV, even a leaf, and a match returns the same pages and the bank slot."""
    s = Session(SPEC, host_slots=2)
    chain, mx, my = two_node_tree(s)
    pages = s.do_match(chain)[0].indices

    s.do_evict_second(1)                       # X (internal, LRU)
    s.do_evict_second(1)                       # Y (a leaf): without a bank its KV would go too
    assert events(s)["evict_mamba.demote"] == 2 and events(s)["evict_mamba.leaf_free"] == 0
    assert {mx, my} <= s.second.free and s.kv.free == set()
    assert (live_snapshots(s), host_snapshots(s), full_size(s)) == (0, 2, 4 * PAGE)
    s.check()

    m, _ = s.do_match(chain)
    assert (m.cached_len, m.indices, m.second) == (4 * PAGE, pages, None) and m.host is not None
    m2, _ = s.do_match(ids(1, 2))
    assert m2.cached_len == 2 * PAGE and m2.host not in (None, m.host)
    s.check()


def test_a_full_bank_drops_its_lru_copy_and_a_bare_leaf_goes_with_its_kv():
    """The bank holds one copy. Demoting B drops A's older copy; A is a leaf with no snapshot left,
    so its KV goes too (an unlocked leaf keeps a snapshot on some tier)."""
    s = Session(SPEC, host_slots=1)
    a = s.kv.take(PAGE)
    s.do_insert(ids(1), slots=a)
    s.do_insert(ids(2))
    s.do_evict_second(1)                       # A -> the bank
    s.do_evict_second(1)                       # B -> the bank, which drops A
    assert events(s)["evict_mamba.demote"] == 2 and events(s)["evict_host.leaf_free"] == 1
    assert set(a) <= s.kv.free and host_snapshots(s) == 1 and full_size(s) == PAGE
    s.check()
    assert s.do_match(ids(1))[0].cached_len == 0
    assert s.do_match(ids(2))[0].host is not None


def test_a_full_bank_tombstones_an_internal_copy():
    s = Session(SPEC, host_slots=1)
    chain, _mx, _my = two_node_tree(s)
    s.do_evict_second(1)                       # X -> the bank
    s.do_evict_second(1)                       # Y -> the bank; X's copy goes, X keeps its KV for Y
    assert events(s)["evict_host.tombstone_in_place"] == 1 and s.kv.free == set()
    s.check()
    assert s.do_match(ids(1, 2))[0].cached_len == 0
    assert s.do_match(chain)[0].cached_len == 4 * PAGE


def test_nothing_evictable_in_the_bank_falls_back_to_a_free():
    """Every bank copy is a restore source (locked): the next victim is freed as without a bank."""
    s = Session(SPEC, host_slots=1)
    s.do_insert(ids(1))
    s.do_evict_second(1)
    held = s.do_lock(ids(1))
    s.do_insert(ids(2))
    s.do_evict_second(1)
    assert events(s)["evict_mamba.leaf_free"] == 1 and host_snapshots(s) == 1
    s.check()
    s.do_unlock(held)
    s.check()


def test_evict_full_frees_the_host_copy_with_the_kv():
    """A host copy describes its node's KV: when the KV goes, the bank slot goes with it."""
    s = Session(SPEC, host_slots=2)
    chain, _mx, _my = two_node_tree(s)
    s.do_evict_second(2)                       # both to the bank
    s.do_evict_full(2 * PAGE)                  # Y's KV and its host copy
    assert len(s.host.free) == 1 and host_snapshots(s) == 1
    s.check()
    assert s.do_match(chain)[0].cached_len == 2 * PAGE
    s.do_evict_full(2 * PAGE)
    assert len(s.host.free) == 2 and s.kv.in_use() == set()
    s.check()


def test_device_donation_displaces_an_unlocked_host_copy_only():
    """One tier per snapshot: a device donation replaces an unlocked host copy (the caller frees its
    bank slot), but a locked one is a running restore's source, so the donation is a duplicate."""
    s = Session(SPEC, host_slots=2)
    s.do_insert(ids(1, 2))
    s.do_evict_second(1)
    held = s.do_lock(ids(1, 2))                # a request restores from the host copy
    got, _ = s.do_insert(ids(1, 2))
    assert got.second_exists and got.host_freed == []
    s.check()
    s.do_unlock(held)
    got, _ = s.do_insert(ids(1, 2))
    assert not got.second_exists and len(got.host_freed) == 1
    assert events(s)["insert.host_displaced"] == 1
    m, _ = s.do_match(ids(1, 2))
    assert (m.second, m.host) == (donated(s), None)
    s.check()


def test_host_lru_runs_on_the_logical_clock(monkeypatch):
    """The host LRU must not read the wall clock (TP ranks agree on the operation order, not on the
    time): with a clock that ties everything, the least recently matched copy still goes first."""
    monkeypatch.setattr(time, "monotonic_ns", lambda: 7)
    s = Session(SPEC, host_slots=2)
    s.do_insert(ids(1))                        # A
    s.do_insert(ids(2))                        # B
    s.do_evict_second(2)                       # both to the bank
    s.do_insert(ids(3))                        # C
    s.do_match(ids(1))                         # A is used again
    s.do_evict_second(1)                       # C to the full bank: B goes, not A
    s.check()
    assert s.do_match(ids(2))[0].cached_len == 0
    assert s.do_match(ids(1))[0].host is not None


def test_evict_host_drops_copies_lru_first():
    s = Session(SPEC, host_slots=3)
    chain, _mx, _my = two_node_tree(s)
    s.do_insert(ids(5))
    s.do_evict_second(3)
    s.do_match(chain)                          # X and Y are used again; the session at 5 is the LRU
    s.do_evict_host(2)                         # the leaf at 5 (with its KV), then X (tombstone)
    assert events(s)["evict_host.leaf_free"] == 1 and events(s)["evict_host.tombstone_in_place"] == 1
    s.check()
    assert s.do_match(chain)[0].cached_len == 4 * PAGE
