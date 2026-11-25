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
from collections import Counter, OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import List, NamedTuple, Optional, Protocol, Tuple

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
    # --kv-host-tier-gb: host slots of the matched span past the device pages, [len(kv_indices), cached_len)
    host_kv: torch.Tensor | None = None

    def get_matched_indices(self) -> torch.Tensor:
        return self.kv_indices


class HybridMatch(NamedTuple):
    kv_indices: torch.Tensor      # reused KV page indices for [0:cached_len)
    cached_len: int               # truncated to the deepest LIVE-snapshot boundary
    mamba_value: Optional[int]    # GDN snapshot slot to restore from (None = cold start)
    node: RadixTreeNode           # the matched node (lock target)
    walk_len: int = 0             # matched KV length before the snapshot cut
    mamba_host: int | None = None  # host-bank slot to restore from, when there is no device one
    # --kv-host-tier-gb: host slots of the matched span whose KV is on the host only, after kv_indices
    host_kv: torch.Tensor | None = None


class EvictResult(NamedTuple):
    kv_indices: torch.Tensor      # KV page indices to free
    mamba_slots: List[int]        # GDN state slots to free
    host_slots: Sequence[int] = ()                 # host-bank slots to free
    # (device slot, host slot): copy each to the host before its device slot (in mamba_slots) is reused
    demoted: Sequence[tuple[int, int]] = ()
    # --kv-host-tier-gb: (device slots, host slots) of each demoted node; copy them before the device
    # pages (also in kv_indices) are reused
    kv_demoted: Sequence[tuple[torch.Tensor, torch.Tensor]] = ()


# --mamba-host-slots: returns a free host-bank slot, or None when the bank has none
HostAlloc = Callable[[], int | None]


class KVHostPages(Protocol):
    """--kv-host-tier-gb: the host pages the tree puts demoted KV in. The tree owns every page it
    takes until it frees it; nothing else allocates from it."""

    capacity: int
    num_free: int

    def alloc(self, n: int) -> list[int]: ...

    def free(self, pages: Sequence[int]) -> None: ...


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
        self._tic = 0
        self.root = RadixTreeNode(self.key_fn, self._next_tic())
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
        # --kv-host-tier-gb: evict_full demotes KV to these host pages (None: no KV host tier, as before)
        self.kv_host: KVHostPages | None = None
        self.host_kv_tokens = 0      # tokens whose KV has a host copy (host-only or on both tiers)
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
                dev, host = self._collect_kv(cur)
                return HybridMatch(dev, end_len, cur.mamba_value, cur, walk_len, cur.mamba_host, host)
            end_len -= cur.length
            cur = cur.parent
        return HybridMatch(self.empty, 0, None, self.root, walk_len)

    def insert(self, input_ids: torch.Tensor, kv_indices: torch.Tensor,
               mamba_value: int, host_freed: list[int] | None = None) -> Tuple[int, bool]:
        """Insert the committed KV prefix and DONATE ``mamba_value`` at the (page-aligned) end
        boundary node. Returns (matched_prefix_len, mamba_exist). If the boundary node already
        owns a live snapshot, returns mamba_exist=True and does not attach (caller frees the
        donated slot -- dedup). A snapshot is on one tier: the donation replaces an unlocked host
        copy (its bank slot goes to ``host_freed``) and is a duplicate of a locked one. With a KV host
        tier the returned length ends at the first host-only node on the path: those nodes adopt the
        request's pages, so only the request pages before it are duplicates."""
        insert_len = align_down(len(input_ids), self.page_size)
        input_ids, kv_indices = input_ids[:insert_len], kv_indices[:insert_len]
        node, walk_len = self._walk(input_ids)
        # --kv-host-tier-gb: the request recomputed host-only nodes on the path; they adopt its pages,
        # so the returned length (the span whose request pages are duplicates) stops before them
        prefix_len = self._adopt_request_pages(node, walk_len, kv_indices)
        if walk_len != insert_len:
            new_node = RadixTreeNode(self.key_fn, self._next_tic())
            # A view would keep the request's full output-budget token buffer alive.
            new_node.set_key_value(input_ids[walk_len:].clone(), kv_indices[walk_len:].clone())
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

    def _adopt_request_pages(self, node: RadixTreeNode, prefix_len: int, kv_indices: torch.Tensor) -> int:
        """Give the host-only nodes of the path root..node the request's pages for their span (the
        request recomputed them). An unlocked host copy is dropped; a locked one is a restore's source
        and stays. Returns where the adopted span starts (``prefix_len`` when nothing was adopted)."""
        start, cur = prefix_len, node
        while not cur.is_root() and not cur.on_device:
            start -= cur.length
            self._adopt(cur, kv_indices[start : start + cur.length].clone())
            if cur.host_ref_count == 0:
                self._free_host_kv(cur)
            self.counts["kv_host_adopt"] += 1
            cur = cur.parent
        return start

    def host_only_path(self, node: RadixTreeNode) -> list[RadixTreeNode]:
        """The host-only nodes of the path root..node, root side first (a restore's targets)."""
        out = []
        while not node.is_root() and not node.on_device:
            out.append(node)
            node = node.parent
        return out[::-1]

    def adopt(self, node: RadixTreeNode, value: torch.Tensor) -> None:
        """Admission restore: a host-only node takes fresh device pages; its host copy stays (a
        later demotion then needs no copy)."""
        assert not node.on_device and node.on_host and len(value) == node.length
        self._adopt(node, value)

    def unadopt(self, node: RadixTreeNode) -> torch.Tensor:
        """Undo ``adopt``: the node is host-only again; returns the device pages to free."""
        value = node.value
        if node.ref_count == 0:
            self.full_evictable -= node.length
        else:
            self.full_protected -= node.length
        node.drop_device_value()
        return value

    def _adopt(self, node: RadixTreeNode, value: torch.Tensor) -> None:
        node.set_key_value(node._key, value)
        if node.ref_count == 0:
            self.full_evictable += node.length
        else:
            self.full_protected += node.length

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
            if cur.ref_count == 0 and cur.on_device:
                self.full_evictable -= cur.length
                self.full_protected += cur.length
            cur.ref_count += 1
            if cur.on_host:   # --kv-host-tier-gb: a restore may read it; the host LRU must keep it
                cur.host_ref_count += 1
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
            if cur.ref_count == 0 and cur.on_device:
                self.full_evictable += cur.length
                self.full_protected -= cur.length
            if cur.on_host:
                cur.host_ref_count -= 1
                assert cur.host_ref_count >= 0, "host copy unlocked more often than locked"
            cur = cur.parent

    # ---------------------------------------------------------------- eviction (dual)
    def evict_full(self, num_tokens: int, demote: bool = True) -> EvictResult:
        """Evict KV tokens by LRU over UNLOCKED LEAF nodes (an internal node's KV is a prefix
        dependency for all descendants). Frees each evicted node's snapshot too, on either tier.
        With a KV host tier (and ``demote``) the device pages are demoted instead (_evict_full_to_host)."""
        if self.kv_host is not None and demote:
            return self._evict_full_to_host(num_tokens)
        leaves = [n for n in self._leaves() if n.ref_count == 0]
        heapq.heapify(leaves)
        kv, mamba, host, freed = [], [], [], 0
        while freed < num_tokens and leaves:
            node = heapq.heappop(leaves)
            if node.ref_count != 0 or not node.is_leaf() or node.is_root():
                continue
            freed += self._free_node_kv(node, kv)
            if node.mamba_value is not None:
                self.counts["kv_snapshot"] += 1
            if node.mamba_host is not None:
                self.counts["kv_host"] += 1
            self._free_node_mamba(node, mamba)
            self._detach_host(node, host)
            self._record_ghost(node)
            parent, casc = self._cascade_tombstone_leaves(self._unlink(node), kv)
            freed += casc
            if parent.is_leaf() and parent.ref_count == 0 and not parent.is_root():
                heapq.heappush(leaves, parent)
        self.counts["kv_tokens"] += freed
        return EvictResult(torch.cat(kv) if kv else self.empty, mamba, host)

    def _evict_full_to_host(self, num_tokens: int) -> EvictResult:
        """--kv-host-tier-gb: free device KV by demoting nodes to the host tier, least recently used
        first on the logical clock. A candidate is an unlocked device node with no device child (the
        device set stays closed toward the root), interior or leaf. A node with a host copy just drops
        its device pages; another one gets host pages (the host LRU makes room) and the caller copies
        it before the pages are reused. A node keeps its snapshots. A node the host tier cannot take
        goes as without the tier, with the host-only nodes below it, so every unlocked device token
        stays evictable (``available_size`` counts on it)."""
        kv, mamba, host, kv_demoted, freed = [], [], [], [], 0
        heap = [(self._lru_key(n), n.uuid, n) for n in self._device_nodes() if self._demotable(n)]
        heapq.heapify(heap)

        def expose(parent: RadixTreeNode) -> None:
            if self._demotable(parent):
                heapq.heappush(heap, (self._lru_key(parent), parent.uuid, parent))

        while freed < num_tokens and heap:
            node = heapq.heappop(heap)[2]
            if not self._attached(node) or not self._demotable(node):
                continue
            if node.on_host:
                self.counts["kv_demote_both"] += 1
            else:
                pages, casc = self._host_pages_for(node.length // self.page_size, kv, mamba, host, expose)
                freed += casc
                if not self._attached(node) or not self._demotable(node):
                    continue                    # a host drop's cascade took it
                if pages is None:
                    self._drop_host_subtree(node, mamba, host)
                    self.counts["kv_tokens_dropped"] += node.length
                    self._free_node_mamba(node, mamba)
                    self._detach_host(node, host)
                    freed += self._free_node_kv(node, kv)
                    self._record_ghost(node)
                    parent, c = self._cascade_tombstone_leaves(self._unlink(node), kv)
                    freed += c
                    expose(parent)
                    continue
                node.set_host_value(self._page_to_token(pages))
                self.host_kv_tokens += node.length
                kv_demoted.append((node.value, node.host_value))
                self.counts["kv_demote"] += 1
            freed += node.length
            kv.append(node.value)
            self.full_evictable -= node.length
            node.drop_device_value()
            expose(node.parent)
        self.counts["kv_tokens"] += freed
        return EvictResult(torch.cat(kv) if kv else self.empty, mamba, host, (), kv_demoted)

    def evict_host_kv(self, num_pages: int | None = None) -> EvictResult:
        """Drop unlocked host copies (least recently used first) until ``num_pages`` host pages were
        freed, or every one (None: a flush)."""
        kv, mamba, host = [], [], []
        before = self.kv_host.num_free
        while num_pages is None or self.kv_host.num_free - before < num_pages:
            if self._drop_host_kv(kv, mamba, host) is None:
                break
        return EvictResult(torch.cat(kv) if kv else self.empty, mamba, host)

    def _host_pages_for(self, n: int, kv: list, mamba: list, host: list, expose) -> tuple[list[int] | None, int]:
        """``n`` host pages, after dropping least recently used host copies to make room. Returns
        (pages or None, device tokens a drop's cascade freed)."""
        kh, freed = self.kv_host, 0
        if n > kh.capacity:
            return None, 0
        while kh.num_free < n:
            got = self._drop_host_kv(kv, mamba, host, expose)
            if got is None:
                return None, freed
            freed += got
        return kh.alloc(n), freed

    def _drop_host_kv(self, kv: list, mamba: list, host: list, expose=None) -> int | None:
        """Drop the least recently used unlocked host copy: of a node on both tiers (it keeps its
        device pages), or of a host-only leaf, which goes with its snapshots (cascading upward).
        Returns the device tokens the cascade freed, or None when nothing can be dropped."""
        cands = [n for n in self._host_kv_nodes()
                 if n.host_ref_count == 0 and (n.on_device or (n.is_leaf() and n.ref_count == 0))]
        if not cands:
            return None
        node = min(cands, key=lambda n: (self._host_lru_key(n), n.uuid))
        if node.on_device:
            self._free_host_kv(node)
            self.counts["kv_host_drop_copy"] += 1
            return 0
        self.counts["kv_host_drop"] += 1
        self._free_node_mamba(node, mamba)
        self._detach_host(node, host)
        self._free_node_kv(node, kv)
        self._record_ghost(node)
        parent, casc = self._cascade_tombstone_leaves(self._unlink(node), kv)
        if expose is not None:
            expose(parent)
        return casc

    def _drop_host_subtree(self, node: RadixTreeNode, mamba: list, host: list) -> None:
        """Remove every node below ``node`` (all host-only and unlocked, since ``node`` is an unlocked
        device node with no device child), freeing their host pages and snapshots."""
        stack = list(node.children.values())
        while stack:
            n = stack.pop()
            stack.extend(n.children.values())
            assert not n.on_device and n.ref_count == 0
            self._free_node_mamba(n, mamba)
            self._detach_host(n, host)
            self._free_host_kv(n)
            self.counts["kv_host_drop"] += 1
        node.children.clear()

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
                self._free_node_kv(node, kv)
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
            self._free_node_kv(node, kv)
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
        # --kv-host-tier-gb: every node holds its KV on some tier, the device set is closed toward the
        # root, and a host lock comes with a full lock (inc_lock takes both)
        host_kv = 0
        stack = list(self.root.children.values())
        while stack:
            n = stack.pop()
            stack.extend(n.children.values())
            assert n.on_device or n.on_host, "a node with its KV on no tier"
            assert not n.on_device or n.parent.is_root() or n.parent.on_device, "a device node below a host-only node"
            assert 0 <= n.host_ref_count <= n.ref_count and (n.on_host or n.host_ref_count == 0)
            host_kv += n.length if n.on_host else 0
        assert host_kv == self.host_kv_tokens, f"host KV tokens: counted {self.host_kv_tokens}, nodes hold {host_kv}"

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
        _iteratively_delete_tombstone_leaf). Returns (highest surviving ancestor, freed device tokens)."""
        freed = 0
        while (parent.mamba_value is None and parent.mamba_host is None and parent.is_leaf()
               and parent.ref_count == 0 and not parent.is_root()):
            freed += self._free_node_kv(parent, kv_out)
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

    def _collect_kv(self, node: RadixTreeNode) -> tuple[torch.Tensor, torch.Tensor | None]:
        """(device slots of the path root..node, host slots of its host-only tail or None)."""
        vals: list[torch.Tensor] = []
        host: list[torch.Tensor] = []
        n = node
        while not n.is_root():
            (vals if n.on_device else host).append(n.value if n.on_device else n.host_value)
            n = n.parent
        vals.reverse()
        host.reverse()
        return (torch.cat(vals) if vals else self.empty), (torch.cat(host) if host else None)

    def _free_node_kv(self, node: RadixTreeNode, kv: list[torch.Tensor]) -> int:
        """Release the KV of an unlocked node that leaves the tree: its device pages go to ``kv`` (the
        caller frees them), its host pages back to the host tier. Returns the device tokens freed."""
        self._free_host_kv(node)
        if not node.on_device:
            return 0
        kv.append(node.value)
        self.full_evictable -= node.length
        return node.length

    def _free_host_kv(self, node: RadixTreeNode) -> None:
        if node.on_host:
            assert node.host_ref_count == 0, "dropping a host copy that a restore reads"
            self.kv_host.free(self._token_to_page(node.host_value))
            self.host_kv_tokens -= node.length
            node.set_host_value(None)

    def _page_to_token(self, pages: Sequence[int]) -> torch.Tensor:
        p = torch.as_tensor(list(pages), dtype=torch.int32, device=self.device)
        offsets = torch.arange(self.page_size, dtype=torch.int32, device=self.device)
        return (p.unsqueeze(1) * self.page_size + offsets).flatten()

    def _token_to_page(self, slots: torch.Tensor) -> list[int]:
        return (slots[:: self.page_size] // self.page_size).tolist()

    def _demotable(self, node: RadixTreeNode) -> bool:
        return (not node.is_root() and node.on_device and node.ref_count == 0
                and not any(c.on_device for c in node.children.values()))

    def _lru_key(self, node: RadixTreeNode) -> tuple[int, int]:
        """KV host tier LRU order: the logical clock, then the deeper node first. Nodes with one
        clock value were stamped by one walk, so they lie on one path and differ in depth."""
        return node.host_tick, -self._path_len(node)

    def _host_lru_key(self, node: RadixTreeNode) -> tuple[int, bool, int]:
        """The host LRU's order: the clock, then a copy of a device node first (it loses nothing)."""
        return node.host_tick, not node.on_device, -self._path_len(node)

    def _attached(self, node: RadixTreeNode) -> bool:
        while not node.is_root():
            if node.parent.children.get(self.key_fn(node._key)) is not node:
                return False
            node = node.parent
        return True

    def _device_nodes(self) -> list[RadixTreeNode]:
        out, stack = [], [self.root]
        while stack:
            n = stack.pop()
            for c in n.children.values():
                if c.on_device:
                    out.append(c)
                    stack.append(c)
        return out

    def _host_kv_nodes(self) -> list[RadixTreeNode]:
        out, stack = [], [self.root]
        while stack:
            n = stack.pop()
            if n.on_host and not n.is_root():
                out.append(n)
            stack.extend(n.children.values())
        return out

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

    def _next_tic(self) -> int:
        self._tic += 1
        return self._tic

    def _walk(self, input_ids: torch.Tensor) -> Tuple[RadixTreeNode, int]:
        prefix_len, total = 0, len(input_ids)
        node = self.root
        tic = self._next_tic()
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


__all__ = ["HybridRadixCache", "HybridMatch", "EvictResult", "HybridCacheHandle", "HostAlloc", "KVHostPages"]
