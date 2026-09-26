"""Decode MoE copy/compute overlap bench (FREETOKEN_MOE_COPY_OVERLAP).

Qwen4ExpMoE layers at the Qwen3.8-Flash-Next expert geometry (hidden 2560, moe and shared
intermediate 640, top-10, NVFP4 triton banks: 2,772,480 B per expert) over one offload cache,
with fewer experts per layer than the model so host and GPU memory stay small. Each arm
(overlap off / on) captures one decode step through every layer as a CUDA graph, then:
  1. bitwise check of the two arms over the same fresh inputs,
  2. replay time per step with CUDA events, arms interleaved ABAB, fresh inputs every round,
  3. a torch.profiler (CUPTI) trace of graph replays: per layer, the copy kernel's start slip
     after the ensure, the shared expert's span (side stream) and whether it ends inside the copy, the join
     gap, and the copy's duration against the off arm.
No attention or hyper-connection work runs, so the percentages overstate the model's.

Run: CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python python benchmarks/bench_moe_copy_overlap.py
"""

from __future__ import annotations

import argparse
import statistics
from types import SimpleNamespace

import torch

H, I, K = 2560, 640, 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--experts", type=int, default=64, help="experts per layer")
    parser.add_argument("--slots", type=int, default=160, help="cache slots (160 of 4 x 64 gives ~4.4 misses/layer)")
    parser.add_argument("--bs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=40, help="decode steps per timing round")
    parser.add_argument("--rounds", type=int, default=6)
    return parser.parse_args()


def nvfp4_quant():
    from freetoken.layers.quantization import QuantBackend, QuantConfig, set_quant_backend

    set_quant_backend(QuantBackend.parse("moe.nvfp4=triton"))
    return QuantConfig.from_hf(
        {"quantization_config": {"quant_method": "modelopt", "quant_algo": "NVFP4", "ignore": ["lm_head"]}}
    )


def build_arm(args, overlap: bool, banks) -> SimpleNamespace:
    import freetoken.core as core
    from freetoken.core import Context, set_global_ctx
    from freetoken.layers.moe import OffloadMoELayer
    from freetoken.models.qwen4_exp.moe import Qwen4ExpMoE
    from freetoken.moe.offload_cache import OffloadMoeCache
    from freetoken.utils.torch_utils import torch_dtype

    E, L = args.experts, args.layers
    config = SimpleNamespace(
        hidden_size=H, num_experts=E, num_experts_per_tok=K, moe_intermediate_size=I,
        shared_expert_intermediate_size=I, norm_topk_prob=True, quant=None,
        moe_strategy="offload", decode_target="gpu",
    )
    device = torch.device("cuda")
    with torch.device(device), torch_dtype(torch.bfloat16):
        moes = [Qwen4ExpMoE(config, l, prefix=f"model.layers.{l}.mlp") for l in range(L)]
    gen = torch.Generator(device=device).manual_seed(11)
    for moe in moes:
        for t in moe.state_dict().values():
            if t.is_floating_point():
                t.normal_(0.0, 0.02, generator=gen)
    quant = nvfp4_quant()
    experts = [
        OffloadMoELayer(l, E, K, H, I, quant_config=quant, prefix=f"model.layers.{l}.mlp.experts")
        for l in range(L)
    ]
    cache = OffloadMoeCache(
        num_layers=L, num_experts=E, cache_size=args.slots, device=device,
        quant_format=banks.quant_format, layout=banks.layout,
        max_slots=experts[0].quant_method.slot_limit(), decode_copy_overlap=overlap,
    )
    cache.set_bank_sources(banks.sources)
    cache.set_alphas(banks.gate_up_alpha, banks.down_alpha)
    cache.collect_stats = True
    for moe, ex in zip(moes, experts):
        ex.offload_cache = cache
        moe.experts = ex
    cache.reset()
    core._GLOBAL_CTX = None
    ctx = Context(page_size=1)
    set_global_ctx(ctx)
    ctx._batch = SimpleNamespace(is_prefill=False)

    static = torch.randn(args.bs, H, device=device, dtype=torch.bfloat16)
    out = torch.empty(L, args.bs, H, device=device, dtype=torch.bfloat16)

    def step():
        for l, moe in enumerate(moes):
            out[l].copy_(moe.forward(static.clone()))

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        step()  # the eager warm-up the graph runner also does
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    cache.reset()
    cache.reset_stats()
    # the graph holds raw pointers only: the modules must outlive it or replays read freed weights
    return SimpleNamespace(graph=graph, static=static, out=out, cache=cache, moes=moes)


def fresh_inputs(args, n: int, seed: int) -> list[torch.Tensor]:
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return [torch.randn(args.bs, H, device="cuda", dtype=torch.bfloat16, generator=gen) for _ in range(n)]


def replay(arm, inputs, keep: bool = False) -> list[torch.Tensor]:
    outs = []
    for x in inputs:
        arm.static.copy_(x)
        arm.graph.replay()
        if keep:
            outs.append(arm.out.clone())
    return outs


def time_arm(arm, inputs) -> float:
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    replay(arm, inputs)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / len(inputs)


def trace(arm, inputs) -> list[tuple[float, float, str]]:
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        replay(arm, inputs)
        torch.cuda.synchronize()
    return sorted(
        (e.time_range.start, e.time_range.end, e.name) for e in prof.events() if e.device_type.name == "CUDA"
    )


def layer_calls(ev, overlap: bool) -> list[dict]:
    """Per MoE layer call, located by kernel name (graph replay spreads branches over streams)."""
    rows = []
    for i, (_, ens_end, name) in enumerate(ev):
        if "_lru_ensure_kernel" not in name:
            continue
        copy = next((k for k in ev[i + 1:] if "fast_index_copy" in k[2]), None)
        gemm = next((k for k in ev[i + 1:] if "_decode_nvfp4" in k[2]), None)
        if copy is None or gemm is None:
            continue
        # the shared expert chain starts at the gate kernel before the ensure; with the overlap it runs on
        # the side stream beside the routed path, without it serially ahead of that path's topk
        j = max(j for j in range(i) if "_gate_sigmoid_kernel" in ev[j][2])
        routed_path = ("router", "topk", "ensure", "fast_index_copy")
        chain = [k for k in (ev[j:] if overlap else ev[j:i]) if k[1] <= gemm[0]
                 and not any(m in k[2] for m in routed_path)]
        rows.append({
            "copy_us": copy[1] - copy[0],
            "slip_us": copy[0] - ens_end,
            "join_gap_us": gemm[0] - copy[1],
            "shared_span_us": max(k[1] for k in chain) - chain[0][0],
            "shared_ends_in_copy": max(k[1] for k in chain) <= copy[1],
        })
    return rows


def pct(xs, q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def main() -> None:
    args = parse_args()
    from freetoken.distributed import set_tp_info
    from freetoken.layers.moe import OffloadMoELayer
    from freetoken.moe.expert_banks import build_expert_banks

    set_tp_info(rank=0, size=1)
    torch.manual_seed(0)
    method = OffloadMoELayer(0, args.experts, K, H, I, quant_config=nvfp4_quant(), prefix="model.layers.0.mlp.experts").quant_method
    banks = build_expert_banks(method, args.layers, None, device=torch.device("cuda"), dummy=True)
    expert_bytes = sum(layers[0][0].numel() * layers[0].element_size() for layers in banks.sources.values())
    print(f"expert {expert_bytes} B; {args.layers} layers x {args.experts} experts, {args.slots} slots, bs {args.bs}")
    arms = {False: build_arm(args, False, banks), True: build_arm(args, True, banks)}

    seed = 100
    inputs = fresh_inputs(args, args.steps, seed)
    outs = {k: replay(arm, inputs, keep=True) for k, arm in arms.items()}
    same = all(torch.equal(a, b) for a, b in zip(outs[True], outs[False]))
    active, missed, calls = arms[False].cache.lru_stats.sum(0).tolist()
    print(f"bitwise identical on/off over {args.steps} steps: {same}; misses per layer call {missed / calls:.2f}")

    times = {False: [], True: []}
    for r in range(args.rounds):
        seed += 1
        inputs = fresh_inputs(args, args.steps, seed)
        for k in (False, True) if r % 2 == 0 else (True, False):
            times[k].append(time_arm(arms[k], inputs))
    for k in (False, True):
        t = times[k]
        print(f"overlap {'on ' if k else 'off'}: {statistics.median(t):.4f} ms/step median (min {min(t):.4f}, max {max(t):.4f}, n={len(t)})")
    per_layer = [1000 * (a - b) / args.layers for a, b in zip(times[False], times[True])]
    print(f"saving per layer, paired rounds (us): {[round(x, 1) for x in per_layer]}, median {statistics.median(per_layer):.1f}")

    inputs = fresh_inputs(args, 12, seed + 1)
    rows = {k: layer_calls(trace(arms[k], inputs), k) for k in (False, True)}
    for k in (False, True):
        r = [x for x in rows[k] if x["copy_us"] > 5.0]  # layer calls with at least one miss
        if not r:
            continue
        slips = [x["slip_us"] for x in r]
        print(
            f"overlap {'on ' if k else 'off'}: {len(r)} layer calls with misses; copy {statistics.median(x['copy_us'] for x in r):.1f} us; "
            f"copy start - ensure end median {statistics.median(slips):.2f} us (p90 {pct(slips, 0.9):.2f}, <=5 us in {100 * sum(s <= 5 for s in slips) / len(slips):.0f}%); "
            f"GEMM start - copy end median {statistics.median(x['join_gap_us'] for x in r):.2f} us; "
            f"shared expert span median {statistics.median(x['shared_span_us'] for x in r):.1f} us"
            + (f", ends inside the copy in {100 * sum(x['shared_ends_in_copy'] for x in r) / len(r):.0f}%" if k else "")
        )
    pairs = [(a["copy_us"], b["copy_us"]) for a, b in zip(rows[False], rows[True]) if a["copy_us"] > 5.0]
    if pairs:
        ratio = [b / a for a, b in pairs]
        print(f"copy duration on/off: median {statistics.median(ratio):.3f} (p10 {pct(ratio, 0.1):.3f}, p90 {pct(ratio, 0.9):.3f})")


if __name__ == "__main__":
    main()
