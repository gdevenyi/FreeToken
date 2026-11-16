"""Hybrid (full-attn KV + GDN linear-state) radix cache.

A SEPARATE class from ``RadixPrefixCache`` (Option C) that REUSES the shared ``RadixTreeNode``
and walk/split logic, so the production KV radix is untouched (zero risk to non-hybrid models).
It adds a second "currency": an optional GDN state snapshot (``node.mamba_value`` = a
LinearStatePool slot id) attached at chunk/page-aligned boundary nodes, with its own LRU
eviction. Mirrors sglang ``MambaRadixCache`` (donate-not-copy, dual eviction, internal-node
tombstone, ``full_ref >= mamba_ref``) on FreeToken's tree.

Currency seam: the secondary value + its eviction is the slot a future SWA component plugs
into. This class is pool-agnostic -- it stores/returns slot ids and KV page indices; the
caller (CacheManager / scheduler) does the actual LinearStatePool / KV-pool free.
"""
from __future__ import annotations

import heapq
import time
from collections import Counter, OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import List, NamedTuple, Optional, Tuple

import torch

from freetoken.utils import align_down

from .base import BaseCacheHandle
from .radix_cache import RadixTreeNode, _get_key_fn


@dataclass(frozen=True)
class HybridCacheHandle(BaseCacheHandle):
    """Lock handle for a matched hybrid prefix: the matched node (lock target) + the reusable
    KV page indices. ``cached_len`` is already truncated to the deepest live-snapshot boundary.
    Plugs into PrefillAdder (reads ``.cached_len`` / ``.get_matched_indices()``) like the plain
    RadixCacheHandle; the restore slot rides on ``MatchResult.mamba_value``."""

    node: RadixTreeNode
    kv_indices: torch.Tensor
    walk_len: int = 0             # matched KV length before the snapshot cut (statistics only)

    def get_matched_indices(self) -> torch.Tensor:
        return self.kv_indices


class HybridMatch(NamedTuple):
    kv_indices: torch.Tensor      # reused KV page indices for [0:cached_len)
    cached_len: int               # truncated to the deepest LIVE-snapshot boundary
    mamba_value: Optional[int]    # GDN snapshot slot to restore from (None = cold start)
    node: RadixTreeNode           # the matched node (lock target)
    walk_len: int = 0             # matched KV length before the snapshot cut
    mamba_host: int | None = None  # host-bank slot to restore from, when there is no device one


class EvictResult(NamedTuple):
    kv_indices: torch.Tensor      # KV page indices to free
    mamba_slots: List[int]        # GDN state slots to free
    host_slots: Sequence[int] = ()                 # host-bank slots to free
    # (device slot, host slot): copy each to the host before its device slot (in mamba_slots) is reused
    demoted: Sequence[tuple[int, int]] = ()


# --mamba-host-slots: returns a free host-bank slot, or None when the bank has none
HostAlloc = Callable[[], int | None]


# Evicted-node record for the KVSTATS miss causes: keyed by the node's start depth and its
# first GHOST_KEYS keys, so an admission can tell "this prompt's continuation was evicted"
# from "never cached". Nodes shorter than GHOST_KEYS are not recognised.
GHOST_KEYS = 64
GHOST_CAP = 4096
# --gdn-snapshot-policy structural: pinned first-turn snapshots, the LRU one loses its pin
MAX_PINNED_ROOTS = 4
# --mamba-host-slots: a full bank first drops a host copy that a deeper one supersedes within the same
# window of this many tokens (the structural stride; design 3.4: a leaf and one ancestor per window)
HOST_THIN_WINDOW = 32768


class HybridRadixCache:
    def __init__(self, device: torch.device, page_size: int, ghosts: bool = False) -> None:
        from freetoken.kernel.fla.chunk import CHUNK_SIZE
        # Snapshots land on ×CHUNK_SIZE boundaries; require them to be page-aligned so the KV
        # node boundary and the GDN-state boundary coincide (page_size in {1,2,4,8,16,32,64}).
        assert CHUNK_SIZE % page_size == 0, (
            f"hybrid_radix needs CHUNK_SIZE({CHUNK_SIZE}) % page_size({page_size}) == 0"
        )
        self.device = device
        self.page_size = page_size
        self.key_fn = _get_key_fn(page_size)
        self.empty = torch.empty(0, dtype=torch.int32, device=device)
        self.root = RadixTreeNode(self.key_fn)
        self.root.set_key_value(self.empty, self.empty)
        self.root.ref_count = 1  # root is always protected
        self.full_evictable = 0
        self.full_protected = 0
        self.mamba_evictable = 0     # number of live, unlocked snapshots
        self.mamba_protected = 0
        # host-tier snapshots (--mamba-host-slots), counted apart from the device slots
        self.mamba_host_evictable = 0
        self.mamba_host_protected = 0
        self.clock = 0               # logical clock of the host LRU: one tick per walk
        # Statistics only (KVSTATS); no decision reads them.
        self.counts: Counter = Counter()
        self.ghosts: OrderedDict | None = OrderedDict() if ghosts else None
        # uuid -> node of the pinned snapshots, least recently used first (structural policy only)
        self.pinned: OrderedDict = OrderedDict()
        # uuids of prompt-end snapshots (structural policy only): the cap keeps the previous one
        self.prompt_ends: set = set()

    # ---------------------------------------------------------------- match / insert
    def match_prefix(self, input_ids: torch.Tensor) -> HybridMatch:
        """Match the token prefix, then truncate the reusable length to the deepest node on
        the path that still owns a snapshot, on the device or in the host bank (a continuation
        can only resume the GDN recurrence from a checkpointed boundary)."""
        node, _ = self._walk(input_ids)
        # walk up to the deepest node whose END boundary has a snapshot
        cur, end_len = node, self._path_len(node)
        walk_len = end_len
        while not cur.is_root():
            if cur.mamba_value is not None or cur.mamba_host is not None:
                self._touch_pins(cur)
                return HybridMatch(self._collect_kv(cur), end_len, cur.mamba_value, cur, walk_len,
                                   cur.mamba_host)
            end_len -= cur.length
            cur = cur.parent
        return HybridMatch(self.empty, 0, None, self.root, walk_len)

    def insert(self, input_ids: torch.Tensor, kv_indices: torch.Tensor,
               mamba_value: int, host_freed: list[int] | None = None) -> Tuple[int, bool]:
        """Insert the committed KV prefix and DONATE ``mamba_value`` at the (page-aligned) end
        boundary node. Returns (matched_prefix_len, mamba_exist). If the boundary node already
        owns a live snapshot, returns mamba_exist=True and does not attach (caller frees the
        donated slot -- dedup). A snapshot is on one tier: the donation replaces an unlocked host
        copy (its bank slot goes to ``host_freed``) and is a duplicate of a locked one."""
        insert_len = align_down(len(input_ids), self.page_size)
        input_ids, kv_indices = input_ids[:insert_len], kv_indices[:insert_len]
        node, prefix_len = self._walk(input_ids)
        if prefix_len != insert_len:
            new_node = RadixTreeNode(self.key_fn)
            # A view would keep the request's full output-budget token buffer alive.
            new_node.set_key_value(
                input_ids[prefix_len:].clone(), kv_indices[prefix_len:].clone()
            )
            new_node.set_parent(node)
            self.full_evictable += new_node.length
            new_node.host_tick = self.clock
            node = new_node
        if node.is_root():
            return prefix_len, True   # root can't hold a snapshot; report exist so caller frees it
        if node.mamba_value is not None:
            return prefix_len, True                 # dedup: caller frees its donated slot
        if node.mamba_host is not None:
            if node.mamba_host_ref_count > 0 or host_freed is None:
                return prefix_len, True             # a restore reads the host copy: keep it
            self._detach_host(node, host_freed)
        node.mamba_value = mamba_value              # fills a fresh node or a tombstone
        if node.mamba_ref_count == 0:
            self.mamba_evictable += 1
        return prefix_len, False

    # ---------------------------------------------------------------- locking (dual)
    def inc_lock(self, node: RadixTreeNode) -> None:
        """Protect a matched node's snapshot (mamba ref on the node) and its KV path
        (full ref node..root). Enforces full_ref >= mamba_ref: using a snapshot at N pins the
        whole root..N KV chain."""
        if node.mamba_value is not None:
            if node.mamba_ref_count == 0:
                self.mamba_evictable -= 1
                self.mamba_protected += 1
            node.mamba_ref_count += 1
        if node.mamba_host is not None:
            if node.mamba_host_ref_count == 0:
                self.mamba_host_evictable -= 1
                self.mamba_host_protected += 1
            node.mamba_host_ref_count += 1
        cur = node
        while not cur.is_root():
            if cur.ref_count == 0:
                self.full_evictable -= cur.length
                self.full_protected += cur.length
            cur.ref_count += 1
            cur = cur.parent

    def dec_lock(self, node: RadixTreeNode) -> None:
        if node.mamba_value is not None and node.mamba_ref_count > 0:
            node.mamba_ref_count -= 1
            if node.mamba_ref_count == 0:
                self.mamba_evictable += 1
                self.mamba_protected -= 1
        if node.mamba_host is not None and node.mamba_host_ref_count > 0:
            node.mamba_host_ref_count -= 1
            if node.mamba_host_ref_count == 0:
                self.mamba_host_evictable += 1
                self.mamba_host_protected -= 1
        cur = node
        while not cur.is_root():
            cur.ref_count -= 1
            assert cur.ref_count >= 0
            if cur.ref_count == 0:
                self.full_evictable += cur.length
                self.full_protected -= cur.length
            cur = cur.parent

    # ---------------------------------------------------------------- eviction (dual)
    def evict_full(self, num_tokens: int) -> EvictResult:
        """Evict KV tokens by LRU over UNLOCKED LEAF nodes (an internal node's KV is a prefix
        dependency for all descendants). Frees each evicted node's snapshot too, on either tier."""
        leaves = [n for n in self._leaves() if n.ref_count == 0]
        heapq.heapify(leaves)
        kv, mamba, host, freed = [], [], [], 0
        while freed < num_tokens and leaves:
            node = heapq.heappop(leaves)
            if node.ref_count != 0 or not node.is_leaf() or node.is_root():
                continue
            freed += node.length
            kv.append(node.value)
            self.full_evictable -= node.length
            if node.mamba_value is not None:
                self.counts["kv_snapshot"] += 1
            self._free_node_mamba(node, mamba)
            self._detach_host(node, host)
            self._record_ghost(node)
            parent, casc = self._cascade_tombstone_leaves(self._unlink(node), kv)
            freed += casc
            if parent.is_leaf() and parent.ref_count == 0 and not parent.is_root():
                heapq.heappush(leaves, parent)
        self.counts["kv_tokens"] += freed
        return EvictResult(torch.cat(kv) if kv else self.empty, mamba, host)

    def evict_mamba(self, num: int, host_alloc: HostAlloc | None = None) -> EvictResult:
        """Evict GDN snapshots by LRU over UNLOCKED snapshot-bearing nodes -- internal nodes
        too. Internal node -> TOMBSTONE (free the slot, keep KV + children). Leaf node -> free
        both KV and slot and unlink, then cascade-delete any KV-only tombstone leaves it exposes
        upward (so a leaf always carries a live snapshot -- mirrors sglang). With ``host_alloc``
        (--mamba-host-slots) a victim is DEMOTED instead: the node, leaf or not, keeps its KV and
        moves its snapshot to a bank slot, and the caller copies it before reusing the device slot."""
        # pinned roots go last (structural policy), so a small idle cache cannot stall admission
        cands = [(n.uuid in self.pinned, n) for n in self._snapshot_nodes() if n.mamba_ref_count == 0]
        heapq.heapify(cands)
        kv, mamba, demoted, freed = [], [], [], 0
        while freed < num and cands:
            pinned, node = heapq.heappop(cands)
            if node.mamba_value is None or node.mamba_ref_count != 0 or node.is_root():
                continue
            if pinned:
                self.counts["victim_pinned"] += 1
            if self._demote(node, host_alloc, mamba, demoted, kv):
                freed += 1
            elif node.is_leaf() and node.ref_count == 0:
                kv.append(node.value)
                self.full_evictable -= node.length
                self._free_node_mamba(node, mamba)
                freed += 1
                self._record_ghost(node)
                _, casc = self._cascade_tombstone_leaves(self._unlink(node), kv)
                self.counts["victim_leaf"] += 1
                self.counts["victim_kv_tokens"] += node.length + casc
            else:
                self.counts["victim_fork" if len(node.children) > 1 else "victim_interior"] += 1
                self._free_node_mamba(node, mamba)  # tombstone internal (or locked-KV) node
                freed += 1
        return EvictResult(torch.cat(kv) if kv else self.empty, mamba, (), demoted)

    def evict_host(self, num: int) -> EvictResult:
        """Drop up to ``num`` unlocked host copies, least recently used first (see _drop_host)."""
        kv, host = [], []
        while len(host) < num and (slot := self._drop_host(kv)) is not None:
            host.append(slot)
        return EvictResult(torch.cat(kv) if kv else self.empty, [], host)

    def _demote(self, node: RadixTreeNode, host_alloc: HostAlloc | None, mamba: list[int],
                demoted: list[tuple[int, int]], kv: list[torch.Tensor]) -> bool:
        """Move an unlocked device snapshot to a bank slot: a free one, else the slot of the least
        recently used unlocked host copy. False (no bank, or no slot) leaves the node alone."""
        if host_alloc is None:
            return False
        slot = host_alloc()
        if slot is None:
            slot = self._drop_host(kv)
        if slot is None:
            return False
        demoted.append((node.mamba_value, slot))
        self.counts["demote"] += 1
        self._free_node_mamba(node, mamba)
        node.mamba_host = slot
        self.mamba_host_evictable += 1
        return True

    def _drop_host(self, kv: list[torch.Tensor]) -> int | None:
        """Drop an unlocked host copy and return its bank slot (None: nothing to drop). Superseded
        copies go first (see _superseded), then the least recently used one. At a tie an internal
        copy goes first: dropping a leaf's copy costs its KV, since an unlocked leaf must keep a
        snapshot on some tier."""
        cands = [n for n in self._host_nodes() if n.mamba_host_ref_count == 0]
        if not cands:
            return None
        node = min(cands, key=lambda n: (not self._superseded(n), n.host_tick, n.is_leaf(), n.uuid))
        slot: list[int] = []
        self._detach_host(node, slot)
        self.counts["host_drop"] += 1
        if node.is_leaf() and node.ref_count == 0 and node.mamba_value is None:
            kv.append(node.value)
            self.full_evictable -= node.length
            self._record_ghost(node)
            _, casc = self._cascade_tombstone_leaves(self._unlink(node), kv)
            self.counts["host_drop_kv_tokens"] += node.length + casc
        return slot[0]

    @property
    def full_evictable_size(self) -> int:
        return self.full_evictable

    @property
    def mamba_evictable_size(self) -> int:
        return self.mamba_evictable

    @property
    def size_info(self):
        """KV-page currency, for code that reads a BasePrefixCache size_info (metrics/usage).
        The GDN-snapshot currency is reported via mamba_evictable_size."""
        from .base import SizeInfo
        return SizeInfo(evictable_size=self.full_evictable, protected_size=self.full_protected)

    def check_integrity(self) -> None:
        # Structural: every snapshot-bearing node holds a real slot id; ref counts non-negative.
        # (KV/page conservation is checked by CacheManager.check_integrity.)
        for n in self._snapshot_nodes():
            assert n.mamba_value is not None and n.mamba_ref_count >= 0 and n.ref_count >= 0
        host_nodes = self._host_nodes()
        for n in host_nodes:
            assert n.mamba_value is None, "a snapshot is on one tier"
            assert n.mamba_host_ref_count >= 0 and n.ref_count >= n.mamba_host_ref_count
        assert self.mamba_host_evictable + self.mamba_host_protected == len(host_nodes), (
            f"host bank: evictable({self.mamba_host_evictable}) + "
            f"protected({self.mamba_host_protected}) != nodes({len(host_nodes)})")

    # ---------------------------------------------------------------- helpers
    def _free_node_mamba(self, node: RadixTreeNode, out: List[int]) -> None:
        if node.mamba_value is not None:
            out.append(node.mamba_value)
            node.mamba_value = None
            self.pinned.pop(node.uuid, None)
            self.prompt_ends.discard(node.uuid)
            if node.mamba_ref_count == 0:
                self.mamba_evictable -= 1

    def _superseded(self, node: RadixTreeNode) -> bool:
        """An internal host copy is superseded when a deeper internal host copy follows it on its
        chain (no fork between) within the same HOST_THIN_WINDOW: an edit between the two resumes
        at most a window early. So a chain keeps its leaf and its deepest copy per window (the
        previous prompt end), and an active session's old turns do not push other sessions' resume
        points out of the bank. A leaf or a fork is never superseded."""
        if len(node.children) != 1:
            return False
        depth = self._path_len(node)
        end = (depth // HOST_THIN_WINDOW + 1) * HOST_THIN_WINDOW
        cur = node
        while len(cur.children) == 1:
            cur = next(iter(cur.children.values()))
            depth += cur.length
            if depth >= end:
                return False
            if cur.mamba_host is not None and not cur.is_leaf():
                return True
        return False

    def _detach_host(self, node: RadixTreeNode, out: list[int]) -> None:
        if node.mamba_host is None:
            return
        assert node.mamba_host_ref_count == 0, "dropping a host copy that a restore reads"
        out.append(node.mamba_host)
        node.mamba_host = None
        self.mamba_host_evictable -= 1

    # ---------------------------------------------------------------- structural policy (P1a)
    def pin(self, node: RadixTreeNode) -> None:
        """Pin a node's snapshot (a first turn boundary): the per-path cap keeps it and evict_mamba
        takes it last. At most MAX_PINNED_ROOTS; the least recently used one loses its pin."""
        if node.mamba_value is None or node.is_root():
            return
        self.pinned[node.uuid] = node
        self.pinned.move_to_end(node.uuid)
        while len(self.pinned) > MAX_PINNED_ROOTS:
            self.pinned.popitem(last=False)
            self.counts["unpin"] += 1

    def mark_prompt_end(self, node: RadixTreeNode) -> None:
        if node.mamba_value is not None:
            self.prompt_ends.add(node.uuid)

    def cap_path(self, node: RadixTreeNode, cap: int, host_alloc: HostAlloc | None = None) -> EvictResult:
        """Free the shallowest snapshots on the path root..node while it holds more than ``cap``.
        Keeps the deepest one (the tail), pinned roots, forks (several children) and locked ones,
        and on top of the cap the deepest prompt end above the tail (the previous turn's, where an
        edited, regenerated or sibling turn resumes). A freed snapshot is interior, so its node
        keeps its KV as a tombstone. The cap counts device snapshots only; with ``host_alloc`` a
        thinned snapshot is demoted to the bank (as in evict_mamba) instead of freed."""
        path, cur = [], node
        while not cur.is_root():
            if cur.mamba_value is not None:
                path.append(cur)
            cur = cur.parent
        previous = next((n for n in path[1:] if n.uuid in self.prompt_ends), None)
        excess, freed, kv, demoted = len(path) - (previous is not None) - cap, [], [], []
        for cur in reversed(path[1:]):                 # shallowest first, never the tail
            if excess <= 0:
                break
            if cur is previous or cur.uuid in self.pinned or len(cur.children) > 1 or cur.mamba_ref_count > 0:
                continue
            if not self._demote(cur, host_alloc, freed, demoted, kv):
                self._free_node_mamba(cur, freed)
            self.counts["cap"] += 1
            excess -= 1
        return EvictResult(torch.cat(kv) if kv else self.empty, freed, (), demoted)

    def _touch_pins(self, node: RadixTreeNode) -> None:
        if not self.pinned:
            return
        while not node.is_root():
            if node.uuid in self.pinned:
                self.pinned.move_to_end(node.uuid)
            node = node.parent

    def _unlink(self, node: RadixTreeNode) -> RadixTreeNode:
        parent = node.parent
        del parent.children[self.key_fn(node._key)]
        return parent

    def _cascade_tombstone_leaves(self, parent: RadixTreeNode, kv_out: List[torch.Tensor]):
        """After a leaf is unlinked, eagerly reclaim the KV-only tombstone leaves it exposes
        upward (mamba_value None, no children, unlocked): free their KV and unlink, walking up.
        Keeps the 'a leaf always carries a live snapshot' invariant (sglang
        _iteratively_delete_tombstone_leaf). Returns (highest surviving ancestor, freed_tokens)."""
        freed = 0
        while (parent.mamba_value is None and parent.mamba_host is None and parent.is_leaf()
               and parent.ref_count == 0 and not parent.is_root()):
            kv_out.append(parent.value)
            self.full_evictable -= parent.length
            freed += parent.length
            self._record_ghost(parent)
            parent = self._unlink(parent)
        return parent, freed

    def _record_ghost(self, node: RadixTreeNode) -> None:
        """Remember an evicted node (call before unlinking it), for the KVSTATS miss causes."""
        if self.ghosts is None:
            return
        key = (self._path_len(node) - node.length, _ghost_hash(node._key[:GHOST_KEYS]))
        self.ghosts[key] = node.length
        self.ghosts.move_to_end(key)
        if len(self.ghosts) > GHOST_CAP:
            self.ghosts.popitem(last=False)

    def evicted_len(self, start: int, keys: torch.Tensor) -> int:
        """Length of an evicted node that started at ``start`` with these keys, else 0."""
        if self.ghosts is None or len(keys) == 0:
            return 0
        return self.ghosts.get((start, _ghost_hash(keys[:GHOST_KEYS])), 0)

    def _path_len(self, node: RadixTreeNode) -> int:
        n, total = node, 0
        while not n.is_root():
            total += n.length
            n = n.parent
        return total

    def _collect_kv(self, node: RadixTreeNode) -> torch.Tensor:
        vals: List[torch.Tensor] = []
        n = node
        while not n.is_root():
            vals.append(n.value)
            n = n.parent
        vals.reverse()
        return torch.cat(vals) if vals else self.empty

    def _leaves(self) -> List[RadixTreeNode]:
        out, stack = [], [self.root]
        while stack:
            n = stack.pop()
            if n.is_leaf():
                if not n.is_root():
                    out.append(n)
            else:
                stack.extend(n.children.values())
        return out

    def _host_nodes(self) -> list[RadixTreeNode]:
        out, stack = [], [self.root]
        while stack:
            n = stack.pop()
            if n.mamba_host is not None and not n.is_root():
                out.append(n)
            stack.extend(n.children.values())
        return out

    def _snapshot_nodes(self) -> List[RadixTreeNode]:
        out, stack = [], [self.root]
        while stack:
            n = stack.pop()
            if n.mamba_value is not None and not n.is_root():
                out.append(n)
            stack.extend(n.children.values())
        return out

    def _walk(self, input_ids: torch.Tensor) -> Tuple[RadixTreeNode, int]:
        prefix_len, total = 0, len(input_ids)
        node = self.root
        tic = time.monotonic_ns()
        self.clock += 1
        while prefix_len < total:
            child = node.children.get(self.key_fn(input_ids[prefix_len:]))
            if child is None:
                return node, prefix_len
            node = child
            match_len = align_down(node.get_match_len(input_ids[prefix_len:]), self.page_size)
            prefix_len += match_len
            if match_len != node.length:
                node = node.split_at(match_len)
                node.timestamp = tic
                node.host_tick = self.clock
                return node, prefix_len
            node.timestamp = tic
            node.host_tick = self.clock
        return node, prefix_len


def _ghost_hash(keys: torch.Tensor) -> int:
    return hash(keys.cpu().numpy().tobytes())


__all__ = ["HybridRadixCache", "HybridMatch", "EvictResult", "HybridCacheHandle", "HostAlloc"]
