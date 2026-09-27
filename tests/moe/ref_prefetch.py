"""CPU reference for the router-lookahead select and count (freetoken.moe.prefetch)."""
from __future__ import annotations


def ref_select(logits, resident, k: int, budget: int):
    """Each row's top-k (ties to the lowest id), merged rank by rank, duplicates dropped; the first
    ``budget`` non-resident ids, and the resident ids among the first ``budget`` distinct ones."""
    rows = [sorted(range(len(row)), key=lambda e: (-row[e], e))[:k] for row in logits]
    merged = []
    for r in range(k):
        for row in rows:
            if row[r] not in merged:
                merged.append(row[r])
    sel = [e for e in merged if resident[e] < 0][:budget]
    res = [e for e in merged[:budget] if resident[e] >= 0]
    return sel, res


def ref_count(sel, res, routed, misses: int, rows: int):
    """``[issued, useful, resident_hits, calls, rows, misses]`` for one layer call."""
    routed = set(routed)
    return [len(sel), sum(e in routed for e in sel), sum(e in routed for e in res), 1, rows, misses]
