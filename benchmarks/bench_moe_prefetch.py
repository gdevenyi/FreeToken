"""FREETOKEN_MOE_PREFETCH=measure cost: the router-lookahead predictor alone, and its effect on the
decode compute stream when it runs beside it.

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

Run: CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python python benchmarks/bench_moe_prefetch.py
"""

from __future__ import annotations

import argparse
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

    def __init__(self, layers: int, slots: int, bs: int, misses: int, mode: str, overlap: bool):
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
            experts[i - 1].set_lookahead(self.moes[i].gate, i, 3 if (i + 1) % 4 else 4)
        core._GLOBAL_CTX = None
        ctx = Context(page_size=1)
        set_global_ctx(ctx)
        ctx._batch = SimpleNamespace(is_prefill=False)
        self.x = [torch.randn(bs, H, device=dev, dtype=torch.bfloat16, generator=gen) * 0.5 for _ in range(layers)]
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
        for moe, x in zip(self.moes, self.x):
            moe.forward(x.clone())


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
    args = ap.parse_args()
    from freetoken.env import ENV

    ENV.MOE_PREFETCH_K.value, ENV.MOE_PREFETCH_BUDGET.value = args.k, args.budget
    dev = torch.device("cuda")
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
