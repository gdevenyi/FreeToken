"""Speed of the load-time FP8 dense GEMMs: per-tensor against per-channel weight scales.

Run on one GPU (sm_89+), no server needed:

    python scripts/fp8_dense_bench.py            # the qwen4_exp TP=2 per-rank shapes
    python scripts/fp8_dense_bench.py --rows 1 4 32

For each shape and row count it times, in a CUDA graph (as the decode step runs):
  tensor   fp8_dynamic_linear with a scalar weight scale (FREETOKEN_FP8_DENSE_SCALE=tensor, production)
  channel  fp8_dynamic_linear with one scale per output row (FREETOKEN_FP8_DENSE_SCALE=channel)
  rowwise  torch._scaled_mm with scale_a [M, 1] and scale_b [1, N] (the scales in the GEMM epilogue);
           not used by FreeToken, printed to show whether this torch build has the mode on this GPU
and prints microseconds per call, the channel/tensor ratio, and the sum over one decode step
(12 QSA layers x (qkv_proj + o_proj) + 36 GDN layers x (in_proj_qkvz + out_proj) + lm_head).
"""

from __future__ import annotations

import argparse
from functools import partial

import torch
from freetoken.layers.fp8_dynamic import FP8, fp8_dynamic_linear

# (name, N local, K local, uses per step, output dtype) for Qwen3.8-Flash-Next at TP=2, per rank
SHAPES = (
    ("qsa.qkv_proj", 6656, 2560, 12, torch.bfloat16),
    ("qsa.o_proj", 2560, 3072, 12, torch.bfloat16),
    ("gdn.in_proj_qkvz", 8192, 2560, 36, torch.bfloat16),
    ("gdn.out_proj", 2560, 3072, 36, torch.bfloat16),
    ("lm_head", 124160, 2560, 1, torch.float32),
)


def _time_graph(fn, iters: int) -> float:
    """Microseconds per call of ``fn`` replayed from one CUDA graph of ``iters`` calls."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(iters):
            fn()
    graph.replay()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(5):
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        best = min(best, start.elapsed_time(end) * 1000 / iters)
    return best


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--rows", type=int, nargs="+", default=[1, 4, 8, 16, 32, 64, 2048, 8192]
    )
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()
    assert torch.cuda.is_available(), "needs a GPU"
    cap = torch.cuda.get_device_capability()
    print(
        f"{torch.cuda.get_device_name()} sm_{cap[0]}{cap[1]}, torch {torch.__version__}"
    )
    torch.manual_seed(0)
    step = {}
    print(
        f"{'shape':18s} {'rows':>5s} {'tensor us':>10s} {'channel us':>11s} {'ratio':>6s} {'rowwise us':>11s}"
    )
    for name, n, k, uses, out_dtype in SHAPES:
        w = (torch.randn(n, k, device="cuda") * 0.02).to(torch.bfloat16)
        w8 = (w.float() / (w.float().abs().amax() / 448)).to(FP8)
        s_tensor = (w.float().abs().amax() / 448).reshape(())
        s_channel = (w.float().abs().amax(dim=1) / 448).contiguous()
        for m in args.rows:
            iters = max(2, args.iters if m <= 64 else args.iters // 10)
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            run = partial(fp8_dynamic_linear, x, w8, out_dtype=out_dtype)
            t_us = _time_graph(partial(run, s_tensor), iters)
            c_us = _time_graph(partial(run, s_channel), iters)
            try:
                x8 = x.float().clamp(-448, 448).to(FP8)
                rowwise_mm = partial(
                    torch._scaled_mm,
                    x8,
                    w8.t(),
                    scale_a=torch.ones(m, 1, device="cuda"),
                    scale_b=s_channel.reshape(1, n),
                    out_dtype=torch.bfloat16,
                )
                r_us = _time_graph(rowwise_mm, iters)
                rowwise = f"{r_us:11.1f}"
            except (RuntimeError, NotImplementedError) as err:
                rowwise = f"  n/a ({str(err).splitlines()[0][:60]})"
            print(
                f"{name:18s} {m:5d} {t_us:10.1f} {c_us:11.1f} {c_us / t_us:6.2f} {rowwise}"
            )
            step.setdefault(m, [0.0, 0.0])
            step[m][0] += uses * t_us
            step[m][1] += uses * c_us
    print(
        "\none decode step per rank (97 GEMMs; the activation quant and the scale pass included):"
    )
    for m, (t_us, c_us) in step.items():
        print(
            f"  rows {m:5d}: tensor {t_us / 1000:7.3f} ms  channel {c_us / 1000:7.3f} ms  ratio {c_us / t_us:5.3f}"
        )


if __name__ == "__main__":
    main()
