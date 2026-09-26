"""CPU reference for the scored slot-cache ensure (freetoken.moe.scored_ensure).

A port of the adversarial-synthesis replay (global slot cache with lru_ensure semantics: the
call's ids pinned, empty slots first, ties to the lowest slot, missing ids ascending paired with
ascending victims) onto the kernel's Q16 integer arithmetic, so victims match the GPU exactly.
"""
from __future__ import annotations

import numpy as np

from freetoken.moe import scored_ensure as se

_KEY_MAX = np.iinfo(np.int64).max


class RefScoredCache:
    def __init__(self, num_layers, num_experts, cache_size, policy,
                 beta=se.BETA, w=se.W, halflife=se.HALFLIFE):
        self.L, self.E, self.S, self.policy = num_layers, num_experts, cache_size, policy
        self.beta_step, self.w_q4, self.decay_mul, self.dt_max = se.score_params(beta, w, halflife, num_layers)
        block_c = 1 << max(cache_size - 1, 1).bit_length()
        self.slot_bits = max(block_c.bit_length() - 1, 1)
        self.g = se.softplus2_table().numpy().astype(np.int64)
        self.reset()

    def reset(self):
        n = self.L * self.E
        self.slot_of_id = np.full(n, -1, np.int64)
        self.id_of_slot = np.full(self.S, -1, np.int64)
        self.usage = np.zeros(self.S, np.int64)
        self.step = 0
        self.tok = 0
        self.last_tok = np.full(n, se.LAST_TOK_NEVER, np.int64)
        self.lc = np.full(n, se.LC_NEVER, np.int64)
        self.ct = np.zeros(n, np.int64)

    def _decayed(self, ids, now):
        dt = np.minimum(np.maximum(now - self.ct[ids], 0), self.dt_max)
        return self.lc[ids] - ((dt * self.decay_mul) >> se.DECAY_SHIFT)

    def _softplus2(self, x):
        shift = se.Q - se.G_STEP_BITS
        u = np.abs(x)
        i = np.minimum(u >> shift, se.G_LAST)
        frac = u & ((1 << shift) - 1)
        g0, g1 = self.g[i], self.g[i + 1]
        g = np.where(i < se.G_LAST, g0 + (((g1 - g0) * frac) >> shift), 0)
        return np.maximum(x, 0) + g

    def _keys(self, layer, tok, now):
        c = np.arange(self.S, dtype=np.int64)
        oid = self.id_of_slot
        held = oid >= 0
        if self.policy == 0:
            key = (self.usage << self.slot_bits) | c
        else:
            ids = np.where(held, oid, 0)
            lk = ids // self.E
            ahead = lk > layer
            d = np.where(ahead, lk - layer, self.L - layer + lk)
            k = np.where(ahead, tok - 1, tok) - self.last_tok[ids]
            k = np.minimum(np.maximum(k, 0), se.K_MAX)
            score = -((k << se.Q) + d * self.beta_step)
            if self.policy >= 2:
                lc = self.lc[ids]
                lcv = self._decayed(ids, now)
                lcv = np.where(lc == se.LC_NEVER, se.LC_FLOOR, np.maximum(lcv, se.LC_FLOOR))
                score = score + ((self.w_q4 * lcv) >> 4)
            assert np.abs(score).max() < 2**31  # the kernel computes it in int32
            key = np.where(held, ((score + se.SCORE_BIAS) << self.slot_bits) | c, c)
        return np.where(self.usage != self.step, key, _KEY_MAX)

    def ensure(self, layer, ids, *, bump_tok=False, update_state=True, logits=None, thr=0.0, near_miss=None):
        """One ensure call; returns ``(out_slots, src_ids, dst_slots)`` like the kernel's plan.

        Policy 3 refreshes the ids whose ``logits`` (``[rows, E]``) are within ``thr`` of their
        row's lowest routed logit, in fp32 as the kernel does, or the ids of a given ``near_miss`` mask.
        """
        self.step += 1
        base = layer * self.E
        tok = now = 0
        if self.policy != 0:
            if bump_tok:
                self.tok += 1
            tok = self.tok
            now = tok * self.L + layer
            if self.policy == 3 and update_state and logits is not None:
                lg = np.asarray(logits, dtype=np.float32).reshape(-1, self.E)
                routed = np.asarray(ids, dtype=np.int64).reshape(lg.shape[0], -1)
                kth = lg[np.arange(lg.shape[0])[:, None], routed].min(axis=1)
                near_miss = lg >= (kth - np.float32(thr))[:, None]
            if self.policy == 3 and update_state and near_miss is not None:
                near = base + np.flatnonzero(np.asarray(near_miss).reshape(-1, self.E).any(0))
                self.last_tok[near] = np.maximum(self.last_tok[near], tok - 1)
        q = np.asarray(ids, dtype=np.int64).reshape(-1) + base
        slots = self.slot_of_id[q]
        self.usage[slots[slots >= 0]] = self.step
        uniq = np.unique(q)
        missing = uniq[self.slot_of_id[uniq] < 0]  # ascending, as the kernel ranks them
        src = missing - base
        dst = np.empty(missing.size, np.int64)
        if missing.size:
            keys = self._keys(layer, tok, now)
            part = np.argpartition(keys, missing.size - 1)[: missing.size]
            victims = part[np.argsort(keys[part])]
            for i, (e, v) in enumerate(zip(missing, victims)):
                old = self.id_of_slot[v]
                if old >= 0:
                    self.slot_of_id[old] = -1
                self.id_of_slot[v] = e
                self.slot_of_id[e] = v
                self.usage[v] = self.step
                dst[i] = v
        out = self.slot_of_id[q]
        if self.policy != 0 and update_state:
            self.last_tok[uniq] = tok
            if self.policy >= 2:
                lc = self.lc[uniq]
                x = self._decayed(uniq, now)
                self.lc[uniq] = np.where(lc == se.LC_NEVER, 0, self._softplus2(x))
                self.ct[uniq] = now
        return out, src, dst


    def materialize(self, layer):
        """The prefill materialize kernel: the whole layer into slots [0, E), position == expert id."""
        base = layer * self.E
        same = (self.id_of_slot >= base) & (self.id_of_slot < base + self.E)
        self.id_of_slot[same] = -1
        self.usage[same] = 0
        old = self.id_of_slot[: self.E].copy()
        self.slot_of_id[old[old >= 0]] = -1
        self.step += 1
        self.id_of_slot[: self.E] = base + np.arange(self.E)
        self.slot_of_id[base : base + self.E] = np.arange(self.E)
        self.usage[: self.E] = self.step


def replay(cache: RefScoredCache, rows, logits=None, thr=0.0, near_miss=None):
    """Decode ``rows`` ``[N, L, K]`` (one step per row, all layers); per-row miss counts.

    ``rows`` may be ``[N, L, B, K]`` for a batch of B rows per step, one K*B call per layer.
    Policy 3 takes the matching ``[N, L, B, E]`` router ``logits`` (or a ``near_miss`` mask).
    """
    rows = np.asarray(rows)
    misses = np.zeros(rows.shape[0], np.int64)
    for r in range(rows.shape[0]):
        for layer in range(rows.shape[1]):
            lg = None if logits is None else logits[r, layer]
            nm = None if near_miss is None else near_miss[r, layer]
            _, src, _ = cache.ensure(layer, rows[r, layer], bump_tok=layer == 0, logits=lg, thr=thr, near_miss=nm)
            misses[r] += src.size
    return misses
