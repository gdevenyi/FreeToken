"""Decode ensure cost per call: flashlib lru_ensure vs the scored --moe-cache-policy kernels.

One ensure call per MoE layer per decode step, on a full cache (S slots over L x E experts).
Each timed call routes K ids of which exactly m miss: the hits are hot, recently used experts
and the victims are cold filler, under every policy. A CUDA graph of 2 decode steps (2 * L
calls; 1 step when that many fresh ids would not fit) is replayed from a restored snapshot,
so the number is the graph-node cost the decode graph pays. The miss count is verified
through the kernels' stats before timing.

Run: CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python python benchmarks/bench_scored_ensure.py
"""

from __future__ import annotations

import argparse
import statistics

import numpy as np
import torch
from flashlib.kernels.slot_cache import N_STATS, Stat, lru_ensure

from freetoken.moe import scored_ensure as se

L, E, TOP_K = 48, 512, 10
VARIANTS = ("flashlib-lru", "lru", "kd", "kdfb", "rule")


def build(variant: str, slots: int, k: int, m: int, dev: torch.device):
    """Full-cache state plus 1-2 steps of L queries of k ids, the first k - m hot, the rest fresh misses."""
    policy = 0 if variant.endswith("lru") else se.POLICY_IDS[variant]
    hot = k - m
    calls = 2 * L if hot * L + 2 * L * m <= slots else L
    step0, tok0 = 1_000_000, 1000
    slot_of_id = np.full(L * E, -1, np.int32)
    id_of_slot = np.full(slots, -1, np.int32)
    usage = np.zeros(slots, np.int64)
    last_tok = np.full(L * E, se.LAST_TOK_NEVER, np.int64)
    lc = np.full(L * E, se.LC_NEVER, np.int32)
    ct = np.zeros(L * E, np.int64)
    ids = [layer * E + e for layer in range(L) for e in range(hot)]
    assert len(ids) + calls * m <= slots, "fresh ids would evict hot ones"
    filler = (layer * E + e for e in range(k + m, E) for layer in range(L))
    ids += [next(filler) for _ in range(slots - len(ids))]
    for s, key in enumerate(ids):
        id_of_slot[s], slot_of_id[key] = key, s
        is_hot = key % E < hot
        usage[s] = step0 - 1 if is_hot else 1 + s
        last_tok[key] = tok0 if is_hot else tok0 - 500 - s % 7
        if is_hot:
            lc[key], ct[key] = 5 << se.Q, tok0 * L + key // E
    queries = np.empty((calls, k), np.int32)
    for i in range(calls):
        visit = i // L
        queries[i] = np.r_[np.arange(hot), hot + visit * m + np.arange(m)]
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(calls, k // TOP_K, E)).astype(np.float32)
    rows = queries.reshape(calls, k // TOP_K, -1).astype(np.int64)
    np.put_along_axis(logits, rows, np.take_along_axis(logits, rows, -1) + 3.0, -1)
    held = id_of_slot.astype(np.int64)
    t = lambda a, dtype=None: torch.from_numpy(a).to(dev, dtype)  # noqa: E731
    state = dict(
        owner=t(id_of_slot.copy()), slot_last=t(last_tok[held]), slot_lc=t(lc[held]), slot_ct=t(ct[held]),
        slot_of_id=t(slot_of_id), id_of_slot=t(id_of_slot), usage=t(usage),
        step=torch.tensor(step0, dtype=torch.int64, device=dev),
        tok=torch.tensor(tok0, dtype=torch.int64, device=dev),
        last_tok=t(last_tok), lc=t(lc), ct=t(ct), queries=t(queries),
        src=torch.zeros(slots, dtype=torch.int32, device=dev),
        dst=torch.zeros(slots, dtype=torch.int32, device=dev),
        num=torch.zeros((), dtype=torch.int64, device=dev),
        stats=torch.zeros(L, N_STATS, dtype=torch.int64, device=dev),
    )
    return policy, state, t(logits, torch.bfloat16), se.softplus2_table(dev)


def step_fn(variant, policy, st, logits, g_table, stats: bool):
    def run():
        for i in range(st["queries"].shape[0]):
            layer = i % L
            q = st["queries"][i]
            s = st["stats"][layer] if stats else None
            if variant == "flashlib-lru":
                lru_ensure(q, st["slot_of_id"], st["id_of_slot"], st["usage"], st["step"], q,
                           st["src"], st["dst"], st["num"], stats=s, id_base=layer * E)
                continue
            se.scored_ensure(
                q, st["slot_of_id"], st["id_of_slot"], st["usage"], st["step"], q,
                st["src"], st["dst"], st["num"], policy=policy, num_layers=L, num_experts=E,
                tok=st["tok"], last_tok=st["last_tok"], lc=st["lc"], ct=st["ct"], g_table=g_table,
                slot_owner=st["owner"], slot_last_tok=st["slot_last"], slot_lc=st["slot_lc"], slot_ct=st["slot_ct"],
                router_logits=logits[i] if policy == 3 else None, near_miss_thr=0.25,
                stats=s, id_base=layer * E, bump_tok=layer == 0,
            )
    return run


def bench(variant, slots, k, m, dev, reps, stats):
    policy, st, logits, g_table = build(variant, slots, k, m, dev)
    snap = {name: v.clone() for name, v in st.items()}

    def restore():
        for name, v in snap.items():
            st[name].copy_(v)

    # verify the miss count (a stats build of the same calls), then time the requested build
    restore()
    step_fn(variant, policy, st, logits, g_table, True)()
    got = int(st["stats"][:, Stat.MISS].sum())
    calls = st["queries"].shape[0]
    assert got == calls * m, f"{variant} K={k} m={m}: {got} misses, expected {calls * m}"
    restore()
    run = step_fn(variant, policy, st, logits, g_table, stats)
    run()  # compile outside capture
    restore()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    times = []
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for rep in range(reps + 3):
        restore()
        torch.cuda.synchronize()
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        if rep >= 3:
            times.append(start.elapsed_time(end) * 1000.0 / calls)
    return statistics.median(times)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slots", type=int, default=1650)
    ap.add_argument("--k", default="10,20", help="routed ids per call (10 = bs 1, 20 = bs 2)")
    ap.add_argument("--misses", default="0-10", help="list or range; capped at K per call")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--stats", action="store_true", help="time with --moe-collect-stats accumulation on")
    args = ap.parse_args()
    dev = torch.device("cuda")
    variants = args.variants.split(",")
    misses = []
    for part in args.misses.split(","):
        lo, _, hi = part.partition("-")
        misses += range(int(lo), int(hi or lo) + 1)
    print(f"# {torch.cuda.get_device_name(dev)}, S={args.slots}, us per ensure call "
          f"(median of {args.reps} graph replays), stats={'on' if args.stats else 'off'}")
    print(f"{'K':>3} {'m':>3} " + " ".join(f"{v:>13}" for v in variants) + "  max delta vs flashlib-lru")
    for k in (int(x) for x in args.k.split(",")):
        for m in (x for x in misses if x <= k):
            us = {v: bench(v, args.slots, k, m, dev, args.reps, args.stats) for v in variants}
            base = us.get("flashlib-lru")
            delta = "" if base is None else f"  {max(us[v] - base for v in variants):+.2f}"
            print(f"{k:>3} {m:>3} " + " ".join(f"{us[v]:>13.2f}" for v in variants) + delta, flush=True)


if __name__ == "__main__":
    main()
