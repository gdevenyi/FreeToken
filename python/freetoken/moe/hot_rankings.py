"""Static hot-expert rankings and entropy-weighted per-layer MoE cache pre-seeding.

Supports three ranking/profile JSON formats:
1. ``expert_routing_profile.json`` (warp34):
   ``{"layers": {"0": {"ranked_expert_ids": [...], "hit_rate_by_cache_size": {"top_64": 0.52, ...}}}}``
2. ``static_hot_cache_rankings.json`` (albucino / warp34):
   ``{"0": [e0, e1, ...], "1": [e0, e1, ...]}``
3. FreeToken ``expert_counts`` profile:
   ``{"expert_counts": [[c_0_0, ...], ...]}``
"""

from __future__ import annotations

import itertools
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

_CANDIDATE_REL_PATHS = (
    "runtime/expert_routing_profile.json",
    "expert_routing_profile.json",
    "static_hot_cache_rankings.json",
    "runtime/static_hot_cache_rankings.json",
)


@dataclass(frozen=True)
class HotExpertRankings:
    by_layer: dict[int, tuple[int, ...]]
    hit_curves: dict[int, tuple[tuple[int, float], ...]] = field(default_factory=dict)
    source_path: str = ""

    @classmethod
    def from_dict(cls, data: Any, *, source_path: str = "") -> HotExpertRankings | None:
        if not isinstance(data, dict):
            return None

        # Format 3: FreeToken {"expert_counts": [[...], ...]}
        if isinstance(data.get("expert_counts"), list):
            counts = data["expert_counts"]
            by_layer: dict[int, tuple[int, ...]] = {}
            hit_curves: dict[int, tuple[tuple[int, float], ...]] = {}
            for layer_id, row in enumerate(counts):
                if not isinstance(row, list) or not row:
                    continue
                t = torch.tensor(row, dtype=torch.float64)
                order = torch.argsort(t, descending=True, stable=True).tolist()
                by_layer[layer_id] = tuple(int(i) for i in order)
                tot = float(t.sum().item())
                if tot > 0:
                    sorted_vals = t[order]
                    cdf = torch.cumsum(sorted_vals, dim=0) / tot
                    pts = [(0, 0.0)] + [
                        (k, min(1.0, max(0.0, float(cdf[k - 1].item()))))
                        for k in (16, 32, 64, 128, 192, 256, 320, 384, 448, len(row))
                        if 0 < k <= len(row)
                    ]
                    dedup = dict(pts)
                    if len(dedup) >= 3:
                        hit_curves[layer_id] = tuple(sorted(dedup.items()))
            if by_layer:
                return cls(
                    by_layer=by_layer, hit_curves=hit_curves, source_path=source_path
                )

        # Format 1: warp34 {"layers": {"0": {"ranked_expert_ids": [...], "hit_rate_by_cache_size": ...}}}
        layers_obj = data.get("layers")
        if isinstance(layers_obj, dict):
            by_layer = {}
            hit_curves = {}
            for k, v in layers_obj.items():
                try:
                    layer_id = int(k)
                except (TypeError, ValueError):
                    continue
                if not isinstance(v, dict):
                    continue
                arr = v.get("ranked_expert_ids")
                if isinstance(arr, list):
                    ids = tuple(
                        int(x)
                        for x in arr
                        if isinstance(x, int) and not isinstance(x, bool) and x >= 0
                    )
                    if ids:
                        by_layer[layer_id] = ids
                hr = v.get("hit_rate_by_cache_size")
                if isinstance(hr, dict):
                    pts_map: dict[int, float] = {0: 0.0}
                    for hk, hv in hr.items():
                        if isinstance(hk, str) and hk.startswith("top_"):
                            try:
                                sz = int(hk[4:])
                                rate = float(hv)
                            except (TypeError, ValueError):
                                continue
                            if sz > 0 and math.isfinite(rate):
                                pts_map[sz] = min(1.0, max(0.0, rate))
                    num_exp = v.get("num_experts")
                    if not isinstance(num_exp, int) or num_exp <= 0:
                        num_exp = len(by_layer.get(layer_id, ())) or 512
                    pts_map[int(num_exp)] = 1.0
                    if len(pts_map) >= 3:
                        hit_curves[layer_id] = tuple(sorted(pts_map.items()))
            if by_layer:
                return cls(
                    by_layer=by_layer, hit_curves=hit_curves, source_path=source_path
                )

        # Format 2: {"0": [e0, e1, ...], "1": [...]}
        by_layer = {}
        for k, v in data.items():
            try:
                layer_id = int(k)
            except (TypeError, ValueError):
                continue
            if isinstance(v, list):
                ids = tuple(
                    int(x)
                    for x in v
                    if isinstance(x, int) and not isinstance(x, bool) and x >= 0
                )
                if ids:
                    by_layer[layer_id] = ids
        if by_layer:
            return cls(by_layer=by_layer, hit_curves={}, source_path=source_path)
        return None

    @classmethod
    def from_path(cls, path: str | Path) -> HotExpertRankings | None:
        p = Path(path)
        if p.is_dir():
            for rel in _CANDIDATE_REL_PATHS:
                cand = p / rel
                if cand.is_file():
                    loaded = cls.from_path(cand)
                    if loaded is not None:
                        return loaded
            return None
        if not p.is_file():
            return None
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return cls.from_dict(raw, source_path=str(p))

    @classmethod
    def discover(cls, model_path: str | Path | None) -> HotExpertRankings | None:
        env_path = os.environ.get("FREETOKEN_MOE_HOT_RANKINGS", "").strip()
        if env_path:
            if env_path.lower() in ("0", "off", "none", "false"):
                return None
            return cls.from_path(env_path)
        if not model_path:
            return None
        return cls.from_path(model_path)

    def interp_hit_rate(self, layer_id: int, k: int) -> float | None:
        pts = self.hit_curves.get(layer_id)
        if not pts:
            return None
        if k <= 0:
            return 0.0
        for (k0, r0), (k1, r1) in itertools.pairwise(pts):
            if k <= k1:
                if k1 == k0:
                    return r1
                t = (k - k0) / (k1 - k0)
                return r0 + t * (r1 - r0)
        return pts[-1][1]

    def plan_layer_capacities(
        self,
        layers: list[int] | tuple[int, ...],
        total_slots: int,
        num_experts: int,
        *,
        step: int = 8,
    ) -> dict[int, int]:
        """Allocate ``total_slots`` across ``layers`` (each ``<= num_experts``) by greedy marginal
        hit-rate gain when ``hit_curves`` cover all ``layers``, else uniform round-robin."""
        n = len(layers)
        if n == 0 or total_slots <= 0:
            return {l: 0 for l in layers}
        total_slots = min(total_slots, n * num_experts)
        base = total_slots // n
        rem = total_slots % n
        uniform = {l: base + (1 if idx < rem else 0) for idx, l in enumerate(layers)}
        if base >= num_experts or not all(l in self.hit_curves for l in layers):
            return uniform

        step = max(1, step)
        floor = min(base, max(0, (base * 3 // 4) // step * step))
        ceil = min(num_experts, max(base, ((base * 5 // 4) + step - 1) // step * step))
        caps = {l: floor for l in layers}
        used = floor * n
        while used + step <= total_slots:
            best_l = None
            best_gain = -1.0
            for l in layers:
                if caps[l] + step > ceil:
                    continue
                r_now = self.interp_hit_rate(l, caps[l]) or 0.0
                r_nxt = self.interp_hit_rate(l, caps[l] + step) or 0.0
                gain = r_nxt - r_now
                if gain > best_gain:
                    best_gain = gain
                    best_l = l
            if best_l is None:
                break
            caps[best_l] += step
            used += step
        for l in layers:
            if used >= total_slots:
                break
            add = min(total_slots - used, num_experts - caps[l])
            caps[l] += add
            used += add
        return caps if sum(caps.values()) == total_slots else uniform

    def select_for_layer(
        self, layer_id: int, num_experts: int, k_hot: int
    ) -> list[int]:
        if k_hot <= 0:
            return []
        k_hot = min(k_hot, num_experts)
        ranked = self.by_layer.get(layer_id, ())
        chosen: list[int] = []
        seen: set[int] = set()
        for eid in ranked:
            if 0 <= eid < num_experts and eid not in seen:
                chosen.append(eid)
                seen.add(eid)
                if len(chosen) == k_hot:
                    return chosen
        for eid in range(num_experts):
            if eid not in seen:
                chosen.append(eid)
                seen.add(eid)
                if len(chosen) == k_hot:
                    return chosen
        return chosen

    def to_pseudo_counts(self, num_layers: int, num_experts: int) -> torch.Tensor:
        """Convert rankings / hit curves into ``[num_layers, num_experts]`` weights for ``split_slots``."""
        out = torch.zeros((num_layers, num_experts), dtype=torch.float64)
        for l in range(num_layers):
            order = self.select_for_layer(l, num_experts, num_experts)
            prev_w: float | None = None
            for rank, eid in enumerate(order):
                r0 = self.interp_hit_rate(l, rank)
                r1 = self.interp_hit_rate(l, rank + 1)
                if r0 is not None and r1 is not None and r1 >= r0:
                    w = max(1e-9, r1 - r0)
                else:
                    # Zipf-like rank prior when only ordinal rankings are given.
                    w = 1.0 / (rank + 1.0)
                if prev_w is not None and w >= prev_w:
                    w = prev_w * (1.0 - 1e-6)
                prev_w = w
                out[l, eid] = w
            s = float(out[l].sum().item())
            if s > 0:
                out[l] /= s
        return out
