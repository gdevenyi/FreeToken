"""Expert-copy kernel sweep and co-run stretch on real-size NVFP4 expert banks.

Two measurements, no checkpoint needed:

* sweep: GB/s of one fused expert copy (all 6 banks, 1..10 experts per plan) for each
  ``fast_index_copy_multi`` (num_threads, blocks_per_bank) config, and for each
  ``fast_index_copy_multi_slim`` (blocks, threads, unroll) config when that kernel exists;
* co-run: the stretch ``sigma = t_corun / t_alone - 1`` of decode compute kernels (bf16 and
  fp8 GEMV [1,H]x[H,N], a small bf16 matmul, the NVFP4 routed decode GEMV) while a copy loop
  runs on another stream, and the copy's GB/s while each compute loop runs.

The host banks are registered anonymous mmaps (``HostBank``, like the serving banks) with
the qwen4_exp triton-NVFP4 row sizes. Source rows are not reused within a loop, the GEMV
weights are cycled over copies that together exceed L2, and every loop is a CUDA graph.
Alone and co-run trials are interleaved; each cell reports min and median over trials.

Copy loops run in two modes. "hot": copy nodes back to back. "cold": a one-element spacer
kernel after every copy, as the ensure kernel precedes every decode copy. On the RTX 5080, in
graph replay, a copy node that follows another kernel's node runs ~8 us longer than one that
follows a copy node, for every variant (CUPTI puts it in the copy's own duration; eager
launches do not show it; cause not identified). Cold GB/s subtracts the spacer's alone time.

Run: CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python python benchmarks/bench_expert_copy_corun.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import tempfile
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F

from freetoken.gpu_select import assign_gpu, bind_assigned_gpu, single_gpu_arg
from freetoken.kernel import fast_index_copy as fic
from freetoken.kernel.pinned import device_ptr
from freetoken.moe.host_banks import HostBank
from freetoken.moe.offload_cache import _BANK_BYTES_PER_EXPERT

L2_BYTES = 64 << 20
SLEEP_CYCLES = 2_000_000  # ~1 ms head start so the host enqueues every graph before the GPU starts
LMAX = 16  # plan buffer length; the kernels read the live count from num_indices


def nvfp4_bank_bytes(h: int, i: int) -> list[int]:
    # triton NVFP4 layout() order: gate_up, gate_up_scale, gate_up_global, down, down_scale, down_global
    return [2 * i * (h // 2), 2 * i * (h // 16), 2 * i * 2, h * (i // 2), h * (i // 16), h * 2]


@dataclass(frozen=True)
class CopyVariant:
    name: str
    launch: Callable[..., None]


def multi_variant(threads: int, blocks_per_bank: int) -> CopyVariant:
    def launch(dst, src, feat, di, si, num):
        fic.fast_index_copy_multi_jit(
            dst, src, feat, di, si, num, num_threads=threads, blocks_per_bank=blocks_per_bank
        )

    return CopyVariant(f"multi {threads}x{blocks_per_bank}", launch)


def slim_variant(blocks: int, threads: int, unroll: int) -> CopyVariant:
    def launch(dst, src, feat, di, si, num):
        fic.fast_index_copy_multi_slim_jit(
            dst, src, feat, di, si, num, num_blocks=blocks, num_threads=threads, unroll=unroll
        )

    return CopyVariant(f"slim {blocks}x{threads}u{unroll}", launch)


class CopyRig:
    """Registered host banks, GPU slot caches and the fused-copy descriptor tensors."""

    def __init__(self, feats: list[int], n_src: int, n_slots: int, device: torch.device, seed: int):
        self.feats = feats
        self.expert_bytes = sum(feats)
        self.n_src = n_src
        self.n_slots = n_slots
        self.device = device
        gen = torch.Generator().manual_seed(seed)
        self.host = []
        for feat in feats:
            bank = HostBank((n_src, feat), torch.uint8)
            bank.tensor.copy_(torch.randint(0, 256, (n_src, feat), dtype=torch.uint8, generator=gen))
            bank.pin()
            self.host.append(bank)
        self.slots = [torch.empty((n_slots, feat), dtype=torch.uint8, device=device) for feat in feats]
        self.dst_ptrs = torch.tensor([c.data_ptr() for c in self.slots], dtype=torch.int64, device=device)
        self.src_ptrs = torch.tensor([device_ptr(b.tensor) for b in self.host], dtype=torch.int64, device=device)
        self.feat_bytes = torch.tensor(feats, dtype=torch.int64, device=device)
        self.spacer_buf = torch.zeros(1, dtype=torch.int32, device=device)
        self.gen = gen
        self._cursor = 0
        self._perm = torch.randperm(n_src, generator=gen)

    def plans(self, n_launch: int, experts: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """[n_launch, LMAX] dst/src plans: source rows walk a permutation of the host rows
        (no reuse until all are used), destination slots are distinct within a plan."""
        src = torch.zeros((n_launch, LMAX), dtype=torch.int32)
        dst = torch.zeros((n_launch, LMAX), dtype=torch.int32)
        for r in range(n_launch):
            for j in range(experts):
                if self._cursor == self.n_src:
                    self._perm = torch.randperm(self.n_src, generator=self.gen)
                    self._cursor = 0
                src[r, j] = self._perm[self._cursor]
                self._cursor += 1
            dst[r, :experts] = torch.randperm(self.n_slots, generator=self.gen)[:experts]
        num = torch.full((1,), experts, dtype=torch.int64, device=self.device)
        return dst.to(self.device), src.to(self.device), num

    def copy_loop(self, variant: CopyVariant, n_launch: int, experts: int, cold: bool) -> Callable[[], None]:
        dst, src, num = self.plans(n_launch, experts)

        def run():
            for r in range(n_launch):
                variant.launch(self.dst_ptrs, self.src_ptrs, self.feat_bytes, dst[r], src[r], num)
                if cold:
                    self.spacer_buf.add_(1)

        return run

    def spacer_us(self, stream: torch.cuda.Stream, trials: int) -> float:
        n = 200

        def run():
            for _ in range(n):
                self.spacer_buf.add_(1)

        g = capture(run, stream)
        return statistics.median(time_alone(g, n, stream) for _ in range(trials))


@dataclass
class ComputeOp:
    name: str
    make: Callable[[int], Callable[[], None]]  # n_launch -> loop body
    single: Callable[[], None]


def _weight_copies(nbytes: int) -> int:
    return max(2, math.ceil(2 * L2_BYTES / nbytes))


def gemv_bf16(h: int, n: int, device) -> ComputeOp:
    x = torch.randn(1, h, dtype=torch.bfloat16, device=device)
    ws = [torch.randn(n, h, dtype=torch.bfloat16, device=device) for _ in range(_weight_copies(n * h * 2))]

    def make(n_launch):
        def run():
            for r in range(n_launch):
                F.linear(x, ws[r % len(ws)])

        return run

    return ComputeOp(f"gemv_bf16 1x{h}x{n}", make, lambda: F.linear(x, ws[0]))


def gemv_fp8(h: int, n: int, device) -> ComputeOp:
    fp8 = torch.float8_e4m3fn
    x = torch.randn(1, h, device=device).to(fp8)
    ws = [torch.randn(n, h, device=device).to(fp8) for _ in range(_weight_copies(n * h))]
    one = torch.ones((), dtype=torch.float32, device=device)

    def mm(w):
        torch._scaled_mm(x, w.t(), scale_a=one, scale_b=one, out_dtype=torch.bfloat16)

    def make(n_launch):
        def run():
            for r in range(n_launch):
                mm(ws[r % len(ws)])

        return run

    return ComputeOp(f"gemv_fp8 1x{h}x{n}", make, lambda: mm(ws[0]))


def matmul_bf16(m: int, h: int, n: int, device) -> ComputeOp:
    a = torch.randn(m, h, dtype=torch.bfloat16, device=device)
    ws = [torch.randn(n, h, dtype=torch.bfloat16, device=device) for _ in range(_weight_copies(n * h * 2))]

    def make(n_launch):
        def run():
            for r in range(n_launch):
                F.linear(a, ws[r % len(ws)])

        return run

    return ComputeOp(f"mm_bf16 {m}x{h}x{n}", make, lambda: F.linear(a, ws[0]))


def nvfp4_routed(h: int, i: int, top_k: int, n_slots: int, device) -> ComputeOp:
    """The decode routed-expert GEMV pair (gate_up -> act -> down) over its own slot banks."""
    from freetoken.moe.fused_nvfp4 import fused_experts_decode_nvfp4_marlin

    fp8 = torch.float8_e4m3fn
    gu = torch.randint(0, 256, (n_slots, 2 * i, h // 2), dtype=torch.uint8, device=device)
    gus = torch.full((n_slots, 2 * i, h // 16), 1.0, device=device).to(fp8)
    gug = torch.full((n_slots, 2 * i), 1.0, dtype=torch.float16, device=device)
    dn = torch.randint(0, 256, (n_slots, h, i // 2), dtype=torch.uint8, device=device)
    dns = torch.full((n_slots, h, i // 16), 1.0, device=device).to(fp8)
    dng = torch.full((n_slots, h), 1.0, dtype=torch.float16, device=device)
    x = torch.randn(1, h, dtype=torch.bfloat16, device=device) * 0.01
    w = torch.full((1, top_k), 1.0 / top_k, dtype=torch.float32, device=device)
    groups = max(1, n_slots // top_k)
    ids = [
        (torch.arange(top_k, dtype=torch.int32, device=device) + g * top_k).view(1, top_k) % n_slots
        for g in range(groups)
    ]

    def call(t):
        fused_experts_decode_nvfp4_marlin(x, gu, gus, gug, dn, dns, dng, w, t, "silu", False)

    def make(n_launch):
        def run():
            for r in range(n_launch):
                call(ids[r % groups])

        return run

    return ComputeOp(f"nvfp4_routed k={top_k}", make, lambda: call(ids[0]))


def capture(fn: Callable[[], None], stream: torch.cuda.Stream) -> torch.cuda.CUDAGraph:
    with torch.cuda.stream(stream):
        fn()  # warm-up (JIT / Triton compile, cuBLAS heuristics) outside capture
    stream.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=stream):
        fn()
    torch.cuda.synchronize()
    g.keepalive = fn  # the closure owns the plan tensors the graph reads
    return g


def _event() -> torch.cuda.Event:
    return torch.cuda.Event(enable_timing=True)


def time_alone(g: torch.cuda.CUDAGraph, n: int, stream: torch.cuda.Stream) -> float:
    """us per launch of a graph holding ``n`` launches, replayed alone."""
    start, end = _event(), _event()
    with torch.cuda.stream(stream):
        torch.cuda._sleep(SLEEP_CYCLES)
        start.record(stream)
        g.replay()
        end.record(stream)
    end.synchronize()
    return start.elapsed_time(end) * 1e3 / n


def time_corun(
    g_fg: torch.cuda.CUDAGraph, n_fg: int, s_fg: torch.cuda.Stream,
    g_bg: torch.cuda.CUDAGraph, s_bg: torch.cuda.Stream,
) -> tuple[float, bool]:
    """us per launch of the foreground graph while the background graph runs on another
    stream, and whether the background outlasted the foreground (a valid co-run)."""
    ev0, bg_end, fg_start, fg_end = _event(), _event(), _event(), _event()
    s0 = torch.cuda.current_stream()
    torch.cuda._sleep(SLEEP_CYCLES)
    ev0.record(s0)
    s_bg.wait_event(ev0)
    s_fg.wait_event(ev0)
    with torch.cuda.stream(s_bg):
        g_bg.replay()
        bg_end.record(s_bg)
    with torch.cuda.stream(s_fg):
        torch.cuda._sleep(SLEEP_CYCLES // 20)  # let the background reach steady state
        fg_start.record(s_fg)
        g_fg.replay()
        fg_end.record(s_fg)
    torch.cuda.synchronize()
    covered = ev0.elapsed_time(bg_end) >= ev0.elapsed_time(fg_end)
    return fg_start.elapsed_time(fg_end) * 1e3 / n_fg, covered


def kernel_info(fn: Callable[[], None]) -> list[dict]:
    """(name, grid, block, regs/thread, smem) of every kernel one call launches."""
    from torch.profiler import ProfilerActivity, profile

    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "trace.json")
        prof.export_chrome_trace(path)
        with open(path) as f:
            events = json.load(f)["traceEvents"]
    out = []
    for e in events:
        if e.get("cat") != "kernel":
            continue
        a = e.get("args", {})
        out.append({
            "name": e["name"][:80],
            "grid": a.get("grid"),
            "block": a.get("block"),
            "regs": a.get("registers per thread"),
            "smem": a.get("shared memory"),
        })
    return out


def _fmt_info(info: list[dict]) -> str:
    return "; ".join(
        f"{k['name'][:48]} grid={k['grid']} block={k['block']} regs={k['regs']} smem={k['smem']}" for k in info
    )


def _stats(xs: list[float]) -> tuple[float, float]:
    return min(xs), statistics.median(xs)


def parse_pairs(text: str, n: int) -> list[tuple[int, ...]]:
    out = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        parts = tuple(int(p) for p in item.lower().replace("u", "x").split("x"))
        if len(parts) != n:
            raise argparse.ArgumentTypeError(f"{item!r}: expected {n} 'x'-separated ints")
        out.append(parts)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gpu", type=single_gpu_arg, default=None,
                   help="GPU UUID or nvidia-smi index (default: the first visible GPU)")
    p.add_argument("--hidden", type=int, default=2560)
    p.add_argument("--inter", type=int, default=640)
    p.add_argument("--src-experts", type=int, default=256, help="host rows per bank (pinned)")
    p.add_argument("--slots", type=int, default=64, help="GPU slot-cache rows per bank")
    p.add_argument("--multi", default="1024x8,1024x4,512x8,512x4,256x16,256x8,128x32",
                   help="fast_index_copy_multi configs, THREADSxBLOCKS_PER_BANK")
    p.add_argument("--slim", default="8x256x4,16x256x2,12x256x4,16x256x4,8x256x2,4x256x4",
                   help="fast_index_copy_multi_slim configs, BLOCKSxTHREADSxUNROLL "
                        "(skipped if the kernel is absent)")
    p.add_argument("--experts", default="1,2,3,4,5,6,7,8,9,10", help="plan sizes for the sweep")
    p.add_argument("--corun-experts", default="4", help="plan sizes for the co-run copy loop")
    p.add_argument("--link", choices=["hot", "cold", "both"], default="both", help="sweep link mode(s)")
    p.add_argument("--corun-link", choices=["hot", "cold"], default="cold", help="co-run copy loop link mode")
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--copy-ms", type=float, default=6.0, help="target length of one timed copy loop")
    p.add_argument("--compute-ms", type=float, default=3.0, help="target length of one timed compute loop")
    p.add_argument("--cover", type=float, default=3.0,
                   help="background loop length as a multiple of the foreground's alone time")
    p.add_argument("--skip-sweep", action="store_true")
    p.add_argument("--skip-corun", action="store_true")
    p.add_argument("--ops", default="all",
                   help="comma list of co-run ops by prefix: gemv_bf16,gemv_fp8,mm_bf16,nvfp4_routed")
    p.add_argument("--json", default=None, help="write all results to this JSON path")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def build_variants(args) -> list[CopyVariant]:
    variants = [multi_variant(t, b) for t, b in parse_pairs(args.multi, 2)]
    if hasattr(fic, "fast_index_copy_multi_slim_jit"):
        variants += [slim_variant(g, t, u) for g, t, u in parse_pairs(args.slim, 3)]
    else:
        print("fast_index_copy_multi_slim_jit not available: slim configs skipped", flush=True)
    return variants


def run_sweep(rig: CopyRig, variants, experts_list, args, stream, spacer_us, results) -> None:
    modes = ["hot", "cold"] if args.link == "both" else [args.link]
    for mode in modes:
        cold = mode == "cold"
        sub = spacer_us if cold else 0.0
        print(f"\n== sweep ({mode} link{f', spacer {sub:.2f} us subtracted' if cold else ''}): copy alone, "
              f"expert={rig.expert_bytes} B, trials={args.trials}; cells are GB/s at min-time / "
              f"median-time (us/copy median)", flush=True)
        print("variant".ljust(18) + "".join(f"{e:>20d}" for e in experts_list), flush=True)
        for v in variants:
            row = v.name.ljust(18)
            for e in experts_list:
                n_launch = max(8, math.ceil(args.copy_ms * 1e3 / (e * 55.0)))
                g = capture(rig.copy_loop(v, n_launch, e, cold), stream)
                ts = [time_alone(g, n_launch, stream) - sub for _ in range(args.trials)]
                t_min, t_med = _stats(ts)
                nbytes = e * rig.expert_bytes
                results["sweep"].append({
                    "link": mode, "variant": v.name, "experts": e, "us_min": t_min, "us_median": t_med,
                    "gbps_best": nbytes / (t_min * 1e3), "gbps_median": nbytes / (t_med * 1e3), "us_all": ts,
                })
                row += f"{nbytes / (t_min * 1e3):>7.1f}/{nbytes / (t_med * 1e3):5.1f} ({t_med:5.0f})"
                del g
            print(row, flush=True)


def run_corun(rig: CopyRig, variants, ops, corun_experts, args, s_fg, s_bg, spacer_us, results) -> None:
    cold = args.corun_link == "cold"
    sub = spacer_us if cold else 0.0
    print(f"\n== co-run ({args.corun_link} link copy loop): compute on one stream, copy loop on another; "
          f"trials={args.trials} (interleaved alone/co-run); sigma = t_corun / t_alone - 1", flush=True)
    for op in ops:
        n_fg_probe = 20
        g_probe = capture(op.make(n_fg_probe), s_fg)
        t_probe = statistics.median(time_alone(g_probe, n_fg_probe, s_fg) for _ in range(3))
        del g_probe
        n_op = max(20, math.ceil(args.compute_ms * 1e3 / t_probe))
        g_op = capture(op.make(n_op), s_fg)
        print(f"\n-- {op.name}: ~{t_probe:.1f} us/launch alone, loop of {n_op}", flush=True)
        results["kernels"][op.name] = kernel_info(op.single)
        print(f"   kernels: {_fmt_info(results['kernels'][op.name])}", flush=True)
        print("   " + "copy variant".ljust(18) + "E".rjust(3)
              + "  op_alone_us(min/med)  op_corun_us(min/med)  sigma(min/med)"
              + "  copy_GBps alone(med) -> under op(med)  valid  (! = alone median >10% over the probe:"
              + " other GPU load, trust the min column)", flush=True)
        for v in variants:
            for e in corun_experts:
                copy_launch_us = e * 55.0
                n_copy_long = max(4, math.ceil(args.cover * n_op * t_probe / copy_launch_us))
                g_copy_long = capture(rig.copy_loop(v, n_copy_long, e, cold), s_bg)
                n_copy = max(8, math.ceil(args.copy_ms * 1e3 / copy_launch_us))
                g_copy = capture(rig.copy_loop(v, n_copy, e, cold), s_bg)
                n_op_long = max(n_op, math.ceil(args.cover * n_copy * copy_launch_us / t_probe))
                g_op_long = capture(op.make(n_op_long), s_fg) if n_op_long > n_op else g_op
                op_alone, op_co, cp_alone, cp_co = [], [], [], []
                valid_op = valid_cp = 0
                for _ in range(args.trials):
                    op_alone.append(time_alone(g_op, n_op, s_fg))
                    t, ok = time_corun(g_op, n_op, s_fg, g_copy_long, s_bg)
                    op_co.append(t)
                    valid_op += ok
                    cp_alone.append(time_alone(g_copy, n_copy, s_bg) - sub)
                    t, ok = time_corun(g_copy, n_copy, s_bg, g_op_long, s_fg)
                    cp_co.append(t - sub)
                    valid_cp += ok
                a_min, a_med = _stats(op_alone)
                c_min, c_med = _stats(op_co)
                sig_min, sig_med = c_min / a_min - 1, c_med / a_med - 1
                nbytes = e * rig.expert_bytes
                gb_alone = nbytes / (statistics.median(cp_alone) * 1e3)
                gb_co = nbytes / (statistics.median(cp_co) * 1e3)
                suspect = a_med > 1.1 * t_probe
                results["corun"].append({
                    "link": args.corun_link, "op": op.name, "variant": v.name, "experts": e, "suspect": suspect,
                    "op_alone_us": op_alone, "op_corun_us": op_co,
                    "sigma_min": sig_min, "sigma_median": sig_med,
                    "copy_alone_us": cp_alone, "copy_corun_us": cp_co,
                    "copy_gbps_alone_median": gb_alone, "copy_gbps_corun_median": gb_co,
                    "valid_op_trials": valid_op, "valid_copy_trials": valid_cp,
                })
                print(f"   {v.name.ljust(18)}{e:>3d}  {a_min:8.2f}/{a_med:8.2f}    {c_min:8.2f}/{c_med:8.2f}"
                      f"    {sig_min:+6.3f}/{sig_med:+6.3f}   {gb_alone:6.1f} -> {gb_co:6.1f}"
                      f"          {valid_op}/{valid_cp}{'  !' if suspect else ''}", flush=True)
                del g_copy_long, g_copy, g_op_long
        del g_op


def main() -> None:
    args = parse_args()
    assert torch.cuda.is_available(), "CUDA is required"
    try:
        assign_gpu(args.gpu)
        device = bind_assigned_gpu()
    except (ValueError, RuntimeError) as e:
        raise SystemExit(f"error: {e}") from e

    feats = nvfp4_bank_bytes(args.hidden, args.inter)
    assert sum(feats) == _BANK_BYTES_PER_EXPERT["nvfp4"](args.hidden, args.inter)
    if (args.hidden, args.inter) == (2560, 640):
        assert sum(feats) == 2_772_480
    print("gpu", torch.cuda.get_device_name(device), "| banks", feats, "| expert", sum(feats), "B", flush=True)
    rig = CopyRig(feats, args.src_experts, args.slots, device, args.seed)
    variants = build_variants(args)
    results = {"feats": feats, "sweep": [], "corun": [], "kernels": {}}

    s_fg = torch.cuda.Stream(device=device)
    s_bg = torch.cuda.Stream(device=device)
    spacer_us = rig.spacer_us(s_bg, args.trials)
    results["spacer_us"] = spacer_us
    print(f"spacer kernel alone: {spacer_us:.2f} us", flush=True)
    for v in variants:
        dst, src, num = rig.plans(1, 4)
        info = kernel_info(lambda: v.launch(rig.dst_ptrs, rig.src_ptrs, rig.feat_bytes, dst[0], src[0], num))
        results["kernels"][v.name] = info
        print(f"{v.name.ljust(18)} {_fmt_info(info)}", flush=True)

    if not args.skip_sweep:
        experts_list = [int(x) for x in args.experts.split(",") if x]
        run_sweep(rig, variants, experts_list, args, s_bg, spacer_us, results)

    if not args.skip_corun:
        h, i = args.hidden, args.inter
        all_ops = [
            gemv_bf16(h, 5120, device), gemv_bf16(h, 12288, device),
            gemv_fp8(h, 5120, device), gemv_fp8(h, 12288, device),
            matmul_bf16(256, h, h, device),
            nvfp4_routed(h, i, 10, args.slots, device),
        ]
        wanted = [w.strip() for w in args.ops.split(",") if w.strip()]
        ops = all_ops if wanted == ["all"] else [o for o in all_ops if any(o.name.startswith(w) for w in wanted)]
        corun_experts = [int(x) for x in args.corun_experts.split(",") if x]
        run_corun(rig, variants, ops, corun_experts, args, s_fg, s_bg, spacer_us, results)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=1)
        print(f"\nwrote {args.json}", flush=True)


if __name__ == "__main__":
    main()
