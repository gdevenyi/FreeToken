"""FREETOKEN_MOE_PREFETCH cost and gain: the router-lookahead predictor alone, its effect on the
decode compute stream in measure mode, and (--on) whole decode steps with prefetch off vs on.

1. Predictor alone, per layer: the lookahead GEMV ([bs, 2560] x [2560, 512] bf16, a distinct router
   weight per layer as in the model), the top-K + select kernel, and both; one stream, CUDA graph.
2. The count kernel alone (it runs on the compute stream in measure mode).
3. A real-geometry decode stack (Qwen3.8-Flash-Next: hidden 2560, 512 experts, top-10, moe and
   shared intermediate 640, NVFP4 triton banks) of N Qwen4ExpMoE blocks: router GEMV, shared
   expert (on the FREETOKEN_MOE_COPY_OVERLAP side stream by default), top-k, rule ensure, miss
   copy of exactly m experts from pinned host, grouped GEMM. The routing is forced (10 - m hot
   experts + m cold ones per row) and the cache restored before every replay, so both arms copy
   the same bytes. One CUDA graph per step, prefetch off vs measure. No attention runs, so the
   percentages overstate the model's.
4. --on: the same stack at the model's depth (48 layers), with a bf16 GEMV "filler" before every
   block standing in for attention and the rest of the layer (--filler-us, ~225 us between one
   demand copy and the next in production). Prefetch on installs a fixed candidate list per layer:
   3 before GDN / 4 before attention layers, of which --useful are the layer's cold experts and the
   rest never-routed ids (production: ~3.24 issued, ~1.75 useful, ~4.2 misses per layer at bs 1).
   Reports ms/step, demand misses/step and the late fraction; --delay-us adds a sleep ahead of every
   prefetch copy to price the compute stream's wait on a late copy.
5. --steady --trace DIR: steady-state decode on a recorded routing trace (no restore, so wrong
   prefetches and evictions carry over), --steady-layers layers of it with --steady-slots slots,
   a GEMV filler of --filler-gdn-us / --filler-qsa-us before GDN / QSA layers, and a lookahead
   that is each layer's traced top-32 logits plus N(0, --sigma) noise. Every arm (off, measure,
   on[:gdn/qsa budgets | :empty]) keeps its own cache state and all arms replay step by step in
   turn, so drift hits them alike. Reports ms/step (mean, p50, p90), demand misses per layer,
   issued/useful/precision, non-resident demands per layer (misses + prefetch hits: above off's
   misses is pollution) and the late fraction before GDN and QSA layers.

Run: CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python python benchmarks/bench_moe_prefetch.py
     ... benchmarks/bench_moe_prefetch.py --on --bs 1,2 --misses 4
     ... benchmarks/bench_moe_prefetch.py --steady --trace DIR --steady-slots 740 --sigma 0.55
"""

from __future__ import annotations

import argparse
import gc
import statistics

import torch
import torch.nn.functional as F

H, E, TOP_K, INTER = 2560, 512, 10, 640


def time_graph(run, reps: int, before=None) -> float:
    """Median ms of one replay of ``run`` captured as a CUDA graph (after an eager warm-up)."""
    if before:
        before()
    run()
    torch.cuda.synchronize()
    if before:
        before()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    times = []
    for rep in range(reps + 3):
        if before:
            before()
        torch.cuda.synchronize()
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        if rep >= 3:
            times.append(start.elapsed_time(end))
    return statistics.median(times)


def bench_predictor(bs: int, layers: int, k: int, budget: int, reps: int) -> dict:
    from freetoken.moe.prefetch import lookahead_select, prefetch_count

    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(bs)
    weights = [torch.randn(E, H, device=dev, dtype=torch.bfloat16, generator=gen) * 0.02 for _ in range(layers)]
    x = torch.randn(bs, H, device=dev, dtype=torch.bfloat16, generator=gen)
    resident = torch.where(torch.rand(E, device=dev, generator=gen) < 0.4, 1, -1).to(torch.int32)
    sel = torch.full((layers, 2 * k), -1, dtype=torch.int32, device=dev)
    res = sel.clone()
    logits = [F.linear(x, w) for w in weights]

    def gemv():
        for w in weights:
            F.linear(x, w)

    def select():
        for layer in range(layers):
            lookahead_select(logits[layer], resident, sel[layer], res[layer], k=k, budget=budget)

    def both():
        for layer, w in enumerate(weights):
            lookahead_select(F.linear(x, w), resident, sel[layer], res[layer], k=k, budget=budget)

    slots = torch.randint(0, 1024, (bs * TOP_K,), device=dev, dtype=torch.int32, generator=gen)
    id_of_slot = torch.randint(0, E, (1024,), device=dev, dtype=torch.int32, generator=gen)
    misses = torch.ones(1, dtype=torch.int64, device=dev)
    counters = torch.zeros((layers, 6), dtype=torch.int64, device=dev)

    def count():
        for layer in range(layers):
            prefetch_count(slots, id_of_slot, misses, sel[layer], res[layer], counters[layer], id_base=0, rows=bs)

    per = 1000.0 / layers
    return {name: time_graph(fn, reps) * per for name, fn in
            (("gemv", gemv), ("select", select), ("predictor", both), ("count", count))}


class ForcedGate:
    """A layer's real router GEMV whose logits are then replaced, so every replay routes (and
    misses) the same experts; the next layer's lookahead runs through it too."""

    def __init__(self, gate, logits: torch.Tensor):
        self.gate, self.logits = gate, logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.gate.forward(x)
        return self.logits


class DecodeStack:
    """N Qwen4ExpMoE blocks (router, shared expert, nvfp4 offload experts) over one shared cache,
    wired like qwen4_exp for the lookahead."""

    def __init__(self, layers: int, slots: int, bs: int, misses: int, mode: str, overlap: bool, filler=None,
                 routing=None, budgets=(3, 4)):
        import freetoken.core as core
        from types import SimpleNamespace

        from freetoken.core import Context, set_global_ctx
        from freetoken.distributed import set_tp_info, try_get_tp_info
        from freetoken.layers.moe import OffloadMoELayer
        from freetoken.layers.quantization import QuantBackend, QuantConfig, set_quant_backend
        from freetoken.models.qwen4_exp.moe import Qwen4ExpMoE
        from freetoken.moe.expert_banks import build_expert_banks
        from freetoken.moe.offload_cache import OffloadMoeCache
        from freetoken.utils.torch_utils import torch_dtype

        if try_get_tp_info() is None:
            set_tp_info(rank=0, size=1)
        set_quant_backend(QuantBackend.parse("moe.nvfp4=triton"))
        quant = QuantConfig.from_hf({"quantization_config": {"quant_method": "modelopt", "quant_algo": "NVFP4", "ignore": ["lm_head"]}})
        dev = torch.device("cuda")
        config = SimpleNamespace(
            hidden_size=H, num_experts=E, num_experts_per_tok=TOP_K, moe_intermediate_size=INTER,
            shared_expert_intermediate_size=INTER, norm_topk_prob=True, quant=None,
            moe_strategy="offload", decode_target="gpu",
        )
        with torch.device(dev), torch_dtype(torch.bfloat16):
            self.moes = [Qwen4ExpMoE(config, i, prefix=f"model.layers.{i}.mlp") for i in range(layers)]
        gen = torch.Generator(device=dev).manual_seed(0)
        for moe in self.moes:
            for t in moe.state_dict().values():
                if t.is_floating_point():
                    t.normal_(0.0, 0.02, generator=gen)
        experts = [
            OffloadMoELayer(i, E, TOP_K, H, INTER, quant_config=quant, prefix=f"model.layers.{i}.mlp.experts")
            for i in range(layers)
        ]
        method = experts[0].quant_method
        if not hasattr(DecodeStack, "_banks"):
            DecodeStack._banks = build_expert_banks(method, 1, None, device=dev, dummy=True)  # one layer, shared
        banks = DecodeStack._banks
        self.cache = cache = OffloadMoeCache(
            num_layers=layers, num_experts=E, cache_size=slots, device=dev, cache_policy="rule",
            quant_format=banks.quant_format, layout=banks.layout, max_slots=method.slot_limit(),
            decode_copy_overlap=overlap, prefetch_mode=mode,
        )
        cache.set_bank_sources({role: per_layer * layers for role, per_layer in banks.sources.items()})
        cache.set_alphas(banks.gate_up_alpha, banks.down_alpha)
        core._GLOBAL_CTX = None
        ctx = Context(page_size=1)
        set_global_ctx(ctx)
        ctx._batch = SimpleNamespace(is_prefill=False)
        self.x = [torch.randn(bs, H, device=dev, dtype=torch.bfloat16, generator=gen) * 0.5 for _ in range(layers)]
        self.filler = filler
        if routing is not None:
            # --steady: every layer routes this step's row of route_buf and layer L's lookahead (from
            # layer L-1) returns look_buf[L]; both run the real router GEMV first
            route_buf, look_buf = routing
            gates = [moe.gate for moe in self.moes]
            for i, (moe, ex) in enumerate(zip(self.moes, experts)):
                ex.offload_cache = cache
                moe.experts = ex
                moe.gate = ForcedGate(gates[i], route_buf[i])
            for i in range(1, layers):
                experts[i - 1].set_lookahead(ForcedGate(gates[i], look_buf[i]), i, budgets[(i + 1) % 4 == 0])
            self.bs = bs
            cache.reset()
            torch.cuda.synchronize()
            return
        # forced routing: row r routes hot experts [r*hot, (r+1)*hot) and m fresh ones past all rows' hot ones
        hot = TOP_K - misses
        for moe, ex in zip(self.moes, experts):
            ex.offload_cache = cache
            moe.experts = ex
            lg = torch.zeros(bs, E, device=dev, dtype=torch.bfloat16)
            for r in range(bs):
                lg[r, list(range(r * hot, (r + 1) * hot))] = 10.0
                lg[r, list(range(bs * hot + r * misses, bs * hot + (r + 1) * misses))] = 10.0
            moe.gate = ForcedGate(moe.gate, lg)
        for i in range(1, layers):
            experts[i - 1].set_lookahead(self.moes[i].gate, i, budgets[(i + 1) % 4 == 0])
        self.bs, self.hot, self.misses = bs, hot, misses
        # warm the hot experts in, then drop the fresh ones: every replay then misses exactly those
        cache.reset()
        self.step()
        cache.slot_for_id[:, bs * hot:].fill_(-1)
        fresh = (cache.id_of_slot >= 0) & (cache.id_of_slot % E >= bs * hot)
        cache.id_of_slot[fresh] = -1
        cache.usage[fresh] = 0
        torch.cuda.synchronize()
        self.state = [cache.slot_for_id, cache.id_of_slot, cache.usage, cache.step]
        for name in ("evict_tok", "evict_last_tok", "evict_lc", "evict_ct", "evict_slot_owner",
                     "evict_slot_last_tok", "evict_slot_lc", "evict_slot_ct"):
            if getattr(cache, name) is not None:
                self.state.append(getattr(cache, name))
        self.snap = [t.clone() for t in self.state]

    def restore(self):
        for dst, src in zip(self.state, self.snap):
            dst.copy_(src)

    def step(self):
        for i, (moe, x) in enumerate(zip(self.moes, self.x)):
            filler = self.filler[i] if isinstance(self.filler, list) else self.filler
            if filler is not None:
                F.linear(x, filler)
            moe.forward(x.clone())

    def cold(self, layer: int) -> list[int]:
        """The experts ``layer`` misses every replay, merged rank by rank across rows like the select."""
        per_row = [list(range(self.bs * self.hot + r * self.misses, self.bs * self.hot + (r + 1) * self.misses))
                   for r in range(self.bs)]
        return [row[i] for i in range(self.misses) for row in per_row]


def bench_stack(layers, slots, bs, misses, reps, overlap) -> dict:
    from flashlib.kernels.slot_cache import Stat

    out = {}
    for mode in ("off", "measure"):
        stack = DecodeStack(layers, slots, bs, misses, mode, overlap)
        stack.cache.collect_stats = True
        stack.restore()
        stack.cache.lru_stats.zero_()
        stack.step()
        got = stack.cache.lru_stats[:, Stat.MISS].tolist()
        assert got == [bs * misses] * layers, f"expected {bs * misses} misses per layer, got {got}"
        stack.cache.collect_stats = False
        out[mode] = time_graph(stack.step, reps, before=stack.restore) * 1000.0 / layers
        if stack.cache.prefetch is not None:
            stack.cache.prefetch.counters.zero_()
            stack.restore()
            stack.step()
            c = stack.cache.prefetch.counters.sum(0).tolist()
            out["issued/layer"] = c[0] / max(c[3], 1)
        del stack
        torch.cuda.empty_cache()
    return out


def budget_of(layer: int) -> int:
    return 3 if (layer + 1) % 4 else 4


def bench_on(layers, slots, bs, misses, useful, reps, filler_us, delays_us) -> list[dict]:
    """ms per decode step, prefetch off vs on (and on with every prefetch copy delayed)."""
    from flashlib.kernels.slot_cache import Stat

    from freetoken.moe.prefetch import COPIED, ISSUED, LATE, USEFUL

    dev = torch.device("cuda")
    x = torch.randn(1, H, device=dev, dtype=torch.bfloat16)
    probe = torch.randn(32768, H, device=dev, dtype=torch.bfloat16)
    per_row_us = time_graph(lambda: F.linear(x, probe), reps) * 1000.0 / 32768
    del probe
    filler = torch.randn(max(int(filler_us / per_row_us), 1), H, device=dev, dtype=torch.bfloat16) * 0.01
    cycles_per_us = 1_000_000 / (time_graph(lambda: torch.cuda._sleep(1_000_000), 5) * 1000.0)
    out = []
    for arm in ["off", "on", *[f"on+{d}us" for d in delays_us if d]]:
        stack = DecodeStack(layers, slots, bs, misses, "off" if arm == "off" else "on", True, filler=filler)
        cache = stack.cache
        want = [bs * misses] * layers
        if arm != "off":
            pf = cache.prefetch
            pf.sel_override = torch.full_like(pf.sel, -1)
            for layer in range(1, layers):
                u = min(useful[layer % len(useful)], bs * misses)
                wrong = list(range(E - 1, E - 1 - (budget_of(layer) - u), -1))
                ids = stack.cold(layer)[:u] + wrong
                pf.sel_override[layer, : len(ids)] = torch.tensor(ids, dtype=torch.int32)
                want[layer] -= u
            if arm.startswith("on+"):
                pf.delay_copy_cycles = int(float(arm[3:-2]) * cycles_per_us)
        cache.collect_stats = True
        stack.restore()
        cache.lru_stats.zero_()
        stack.step()
        got = cache.lru_stats[:, Stat.MISS].tolist()
        assert got == want, f"{arm}: expected demand misses {want}, got {got}"
        cache.collect_stats = False
        stack.restore()
        stack.step()
        torch.cuda.synchronize()
        stack.restore()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            stack.step()
        if cache.prefetch is not None:
            cache.prefetch.stats.zero_()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        times = []
        for rep in range(reps + 3):
            stack.restore()
            torch.cuda.synchronize()
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            if rep >= 3:
                times.append(start.elapsed_time(end))
        row = {"arm": arm, "ms": statistics.median(times), "misses": sum(want)}
        if cache.prefetch is not None:
            st = cache.prefetch.stats.sum(0).tolist()
            calls = max(reps + 3, 1) * (layers - 1)
            row.update(issued=st[ISSUED] / calls, useful=st[USEFUL] / calls, late=st[LATE] / max(st[COPIED], 1))
        out.append(row)
        del graph, stack, cache
        gc.collect()  # the module graph holds cycles: free this arm's slot banks before the next one
        torch.cuda.empty_cache()
    return out


def load_trace(path: str, layers: int, offset: int, bs: int, steps: int, skip: int, sigma: float, seed: int):
    """Per-step routing and lookahead logits ``[steps, layers, bs, E]`` from a routing trace.

    ``path`` holds experts.npy ``[N, 48, 10]`` (routed ids), logit_idx.npy / logit_val.npy
    ``[N, 48, 32]`` (each layer's own top-32 router logits) and req_index.npy ``[N]``. Row r of the
    batch follows the r-th longest request from its ``skip``-th token on. The routing logits pick
    exactly the traced top-10; the lookahead of layer L is layer L's traced top-32 logits plus
    N(0, sigma) noise (everything else far below), a stand-in for W_gate[L] . x_{L-1}."""
    import os
    from collections import Counter

    import numpy as np

    ex = np.load(os.path.join(path, "experts.npy"), mmap_mode="r")
    li = np.load(os.path.join(path, "logit_idx.npy"), mmap_mode="r")
    lv = np.load(os.path.join(path, "logit_val.npy"), mmap_mode="r")
    req = np.load(os.path.join(path, "req_index.npy"))
    reqs = [r for r, n in Counter(req.tolist()).most_common(bs)]
    rng = np.random.default_rng(seed)
    route = np.full((steps, layers, bs, E), -10.0, np.float32)
    look = np.full((steps, layers, bs, E), -30.0, np.float32)
    lsl = slice(offset, offset + layers)
    for r, rid in enumerate(reqs):
        rows = np.flatnonzero(req == rid)[skip: skip + steps]
        assert len(rows) == steps, f"request {rid} has {len(rows) + skip} tokens, fewer than skip + steps"
        routed = np.asarray(ex[rows, lsl]).astype(np.int64)  # [steps, layers, 10]
        np.put_along_axis(route[:, :, r], routed, 10.0 - 0.1 * np.arange(TOP_K), -1)
        idx = np.asarray(li[rows, lsl]).astype(np.int64)
        val = np.asarray(lv[rows, lsl]).astype(np.float32) + rng.normal(0.0, sigma, idx.shape).astype(np.float32)
        np.put_along_axis(look[:, :, r], idx, val, -1)
    dev = torch.device("cuda")
    return (torch.from_numpy(route).to(dev, torch.bfloat16), torch.from_numpy(look).to(dev, torch.bfloat16), reqs)


def bench_steady(args, bs: int, arms: list[str]) -> list[dict]:
    """ms per decode step in steady state on traced routing: no restore, the cache evolves.

    One stack, one graph per arm, each arm with its own copy of the cache's slot maps, eviction
    state and counters: every step replays all arms back to back (rotating their order), so
    clock or link drift hits every arm alike. Slot bytes are shared (timing does not read them)."""
    from flashlib.kernels.slot_cache import Stat

    from freetoken.moe.prefetch import COPIED, ISSUED, LATE, MISSES, USEFUL

    dev = torch.device("cuda")
    steps = args.warm + args.timed
    route_all, look_all, reqs = load_trace(args.trace, args.steady_layers, args.layer_offset, bs, steps,
                                           args.skip, args.sigma, args.seed)
    layers = args.steady_layers
    x = torch.randn(1, H, device=dev, dtype=torch.bfloat16)
    probe = torch.randn(32768, H, device=dev, dtype=torch.bfloat16)
    per_row_us = time_graph(lambda: F.linear(x, probe), args.reps) * 1000.0 / 32768
    del probe
    fill = {us: torch.randn(max(int(us / per_row_us), 1), H, device=dev, dtype=torch.bfloat16) * 0.01
            for us in (args.filler_gdn_us, args.filler_qsa_us)}
    # the filler of layer i stands in for its attention: QSA layers are 3, 7, ... (i % 4 == 3)
    fillers = [fill[args.filler_qsa_us if i % 4 == 3 else args.filler_gdn_us] for i in range(layers)]
    qsa = torch.tensor([i % 4 == 3 for i in range(layers)])
    route_buf = torch.empty_like(route_all[0])
    look_buf = torch.empty_like(look_all[0])
    stack = DecodeStack(layers, args.steady_slots, bs, 0, "on", True, filler=fillers, routing=(route_buf, look_buf))
    cache, pf = stack.cache, stack.cache.prefetch
    cache.collect_stats = True
    experts = [moe.experts for moe in stack.moes]
    state = [cache.slot_for_id, cache.id_of_slot, cache.usage, cache.step, cache.lru_stats, pf.stats]
    for name in ("evict_tok", "evict_last_tok", "evict_lc", "evict_ct", "evict_slot_owner",
                 "evict_slot_last_tok", "evict_slot_lc", "evict_slot_ct"):
        if getattr(cache, name) is not None:
            state.append(getattr(cache, name))
    init = [t.clone() for t in state]

    def load(snap):
        for dst, src in zip(state, snap):
            dst.copy_(src)

    def save(snap):
        for dst, src in zip(snap, state):
            dst.copy_(src)

    route_buf.copy_(route_all[0])
    look_buf.copy_(look_all[0])
    graphs, snaps, keep = {}, {}, []
    for arm in dict.fromkeys(arms):
        mode, _, spec = arm.partition(":")
        budgets = tuple(int(b) for b in spec.split("/")) if spec and spec != "empty" else (3, 4)
        for i in range(1, layers):
            experts[i - 1].set_lookahead(experts[i - 1]._lookahead[0], i, budgets[(i + 1) % 4 == 0])
        # every mode switch is read at enqueue time, so each arm's graph bakes its own path in
        pf.mode = "on" if mode == "off" else mode
        pf.sel_override = torch.full_like(pf.sel, -1) if spec == "empty" else None
        keep.append(pf.sel_override)
        cache.prefetch = None if mode == "off" else pf
        load(init)
        stack.step()  # compile outside capture
        torch.cuda.synchronize()
        load(init)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            stack.step()
        graphs[arm], snaps[arm] = graph, [t.clone() for t in init]
    cache.prefetch, pf.mode, pf.sel_override = pf, "on", None
    names = list(graphs)
    events = {arm: [] for arm in names}
    for t in range(steps):
        if t == args.warm:
            for arm in names:
                snaps[arm][4].zero_()
                snaps[arm][5].zero_()
        route_buf.copy_(route_all[t])
        look_buf.copy_(look_all[t])
        for j in range(len(names)):
            arm = names[(t + j) % len(names)]
            load(snaps[arm])
            ev = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            ev[0].record()
            graphs[arm].replay()
            ev[1].record()
            save(snaps[arm])
            if t >= args.warm:
                events[arm].append(ev)
        if t % 50 == 49:
            torch.cuda.synchronize()  # bound the event backlog
    torch.cuda.synchronize()
    out = []
    n = args.timed
    for arm in names:
        mode = arm.partition(":")[0]
        times = sorted(a.elapsed_time(b) for a, b in events[arm])
        miss = snaps[arm][4][:, Stat.MISS].double().cpu()
        row = {"arm": arm, "mean": statistics.fmean(times), "p50": times[len(times) // 2],
               "p90": times[int(len(times) * 0.9)], "miss": miss.sum().item() / (n * layers),
               "miss1": miss[1:].sum().item() / (n * (layers - 1))}
        if mode != "off":
            st = snaps[arm][5].double().cpu()
            calls = n * (layers - 1)
            row.update(issued=st[:, ISSUED].sum().item() / calls, useful=st[:, USEFUL].sum().item() / calls)
            row["precision"] = row["useful"] / row["issued"] if row["issued"] else 0.0
            if mode == "measure":
                row["coverage"] = st[:, USEFUL].sum().item() / max(st[:, MISSES].sum().item(), 1)
            if mode == "on":
                # every routed id that was not a normal resident: residual misses plus prefetch hits
                row["nonres1"] = row["miss1"] + row["useful"]
                for name, mask in (("late_gdn", ~qsa), ("late_qsa", qsa)):
                    m = mask.clone()
                    m[0] = False
                    row[name] = st[m, LATE].sum().item() / max(st[m, COPIED].sum().item(), 1)
                row["late"] = st[:, LATE].sum().item() / max(st[:, COPIED].sum().item(), 1)
        out.append(row)
    print(f"# peak allocated {torch.cuda.max_memory_allocated(dev) / 2**30:.2f} GiB, requests {reqs}", flush=True)
    del graphs, stack, cache, pf, events
    gc.collect()
    torch.cuda.empty_cache()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bs", default="1,2")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--budget", type=int, default=4)
    ap.add_argument("--layers", type=int, default=8, help="MoE layers in the decode stack")
    ap.add_argument("--slots", type=int, default=512, help="expert slots (2.77 MB each)")
    ap.add_argument("--misses", default="0,4", help="demand misses per layer per row")
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--overlap", type=int, default=1, help="FREETOKEN_MOE_COPY_OVERLAP in the stack (production: 1)")
    ap.add_argument("--on", action="store_true", help="time whole decode steps with prefetch off vs on instead")
    ap.add_argument("--on-layers", type=int, default=48)
    ap.add_argument("--on-slots", type=int, default=0, help="0 = every layer's routed experts + 64 (the warm-up must not evict)")
    ap.add_argument("--filler-us", type=float, default=160.0, help="per-layer compute standing in for attention")
    ap.add_argument("--useful", default="2,1,2,2", help="useful prefetches per layer, cycled over layers")
    ap.add_argument("--delay-us", default="0,150,300", help="sleep ahead of every prefetch copy (0 = none)")
    ap.add_argument("--steady", action="store_true", help="steady-state decode on a routing trace instead")
    ap.add_argument("--trace", default=None, help="--steady: dir with experts/logit_idx/logit_val/req_index .npy")
    ap.add_argument("--steady-layers", type=int, default=24, help="--steady: layers (VRAM: slots * 2.77 MB)")
    ap.add_argument("--layer-offset", type=int, default=0, help="--steady: first traced layer")
    ap.add_argument("--steady-slots", type=int, default=825, help="--steady: slots (1650 * 24 / 48 by default)")
    ap.add_argument("--warm", type=int, default=300, help="--steady: untimed steps before timing")
    ap.add_argument("--timed", type=int, default=500)
    ap.add_argument("--skip", type=int, default=200, help="--steady: tokens skipped at each request's start")
    ap.add_argument("--sigma", type=float, default=0.3, help="--steady: lookahead noise on the traced logits")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--filler-gdn-us", type=float, default=171.0, help="--steady: compute before a GDN layer's MoE")
    ap.add_argument("--filler-qsa-us", type=float, default=212.0, help="--steady: compute before a QSA layer's MoE")
    ap.add_argument("--arms", default="off,measure,on,on:empty,on:2/3,on:3/3,on:4/4",
                    help="--steady: mode[:gdn/qsa budgets | :empty (on, nothing selected)]")
    args = ap.parse_args()
    from freetoken.env import ENV

    ENV.MOE_PREFETCH_K.value, ENV.MOE_PREFETCH_BUDGET.value = args.k, 0 if args.steady else args.budget
    dev = torch.device("cuda")
    if args.steady:
        print(f"# {torch.cuda.get_device_name(dev)}, steady state on {args.trace}: {args.steady_layers} layers from "
              f"{args.layer_offset}, {args.steady_slots} slots, rule, copy overlap on, filler {args.filler_gdn_us:.0f}/"
              f"{args.filler_qsa_us:.0f} us (GDN/QSA), sigma {args.sigma}, {args.warm} warm + {args.timed} timed steps")
        print(f"{'bs':>3} {'arm':>10} {'mean ms':>8} {'vs off':>7} {'p50':>7} {'p90':>7} {'miss/L':>7} {'miss/L1+':>8} "
              f"{'iss/L':>6} {'use/L':>6} {'prec':>6} {'nonres/L1+':>10} {'late':>6} {'lateGDN':>7} {'lateQSA':>7}")
        for bs in (int(b) for b in args.bs.split(",")):
            rows = bench_steady(args, bs, args.arms.split(","))
            base = rows[0]["mean"]
            for r in rows:
                g = r.get
                print(f"{bs:>3} {r['arm']:>10} {r['mean']:>8.3f} {r['mean'] / base - 1:>+7.1%} {r['p50']:>7.3f} "
                      f"{r['p90']:>7.3f} {r['miss']:>7.3f} {r['miss1']:>8.3f} {g('issued', 0):>6.2f} {g('useful', 0):>6.2f} "
                      f"{g('precision', 0):>6.3f} {g('nonres1', 0):>10.3f} {g('late', 0):>6.3f} {g('late_gdn', 0):>7.3f} "
                      f"{g('late_qsa', 0):>7.3f}", flush=True)
        return
    if args.on:
        useful = [int(u) for u in args.useful.split(",")]
        delays = [float(d) for d in args.delay_us.split(",")]
        print(f"# {torch.cuda.get_device_name(dev)}, {args.on_layers} Qwen4ExpMoE nvfp4 layers, "
              f"{args.on_slots or 'layers * bs * top-k + 64'} slots, "
              f"rule, copy overlap on, filler {args.filler_us:.0f} us/layer, useful {useful}, median of {args.reps} replays")
        print(f"{'bs':>3} {'m':>3} {'arm':>10} {'ms/step':>9} {'vs off':>8} {'misses/step':>12} "
              f"{'issued/layer':>13} {'useful/layer':>13} {'late':>6}")
        for bs in (int(b) for b in args.bs.split(",")):
            for m in (int(x) for x in args.misses.split(",")):
                slots = args.on_slots or args.on_layers * bs * TOP_K + 64
                rows = bench_on(args.on_layers, slots, bs, m, useful, args.reps, args.filler_us, delays)
                base = rows[0]["ms"]
                for r in rows:
                    extra = (f"{r['issued']:>13.2f} {r['useful']:>13.2f} {r['late']:>6.3f}" if "late" in r
                             else f"{'':>13} {'':>13} {'':>6}")
                    print(f"{bs:>3} {m:>3} {r['arm']:>10} {r['ms']:>9.3f} {r['ms'] / base - 1:>+8.1%} "
                          f"{r['misses']:>12d} {extra}", flush=True)
        print(f"# peak allocated {torch.cuda.max_memory_allocated(dev) / 2**30:.2f} GiB")
        return
    print(f"# {torch.cuda.get_device_name(dev)}, K={args.k}, budget={args.budget}, median of {args.reps} graph replays")
    print("# predictor alone, us per layer (47 layers, one stream)")
    print(f"{'bs':>3} {'gemv':>8} {'select':>8} {'predictor':>10} {'count':>8}")
    for bs in (int(b) for b in args.bs.split(",")):
        r = bench_predictor(bs, 47, args.k, args.budget, args.reps)
        print(f"{bs:>3} {r['gemv']:>8.2f} {r['select']:>8.2f} {r['predictor']:>10.2f} {r['count']:>8.2f}", flush=True)
    print(f"# decode stack ({args.layers} Qwen4ExpMoE nvfp4 layers, {args.slots} slots, rule, overlap={args.overlap}), "
          "us per layer, prefetch off vs measure")
    print(f"{'bs':>3} {'m':>3} {'off':>9} {'measure':>9} {'delta':>8} {'issued/layer':>13}")
    for bs in (int(b) for b in args.bs.split(",")):
        for m in (int(x) for x in args.misses.split(",")):
            r = bench_stack(args.layers, args.slots, bs, m, args.reps, bool(args.overlap))
            print(f"{bs:>3} {m:>3} {r['off']:>9.2f} {r['measure']:>9.2f} {r['measure'] - r['off']:>+8.2f} "
                  f"{r.get('issued/layer', 0.0):>13.2f}", flush=True)


if __name__ == "__main__":
    main()
