# benchmarks

Run from the repo root with `PYTHONPATH=python:.`, pinned to one GPU
(`CUDA_VISIBLE_DEVICES=0`). Each script's `--help` / docstring has the details.

**`bench_decode_moe.py`** — bs=1 decode tok/s of a served MoE model. Spawns `ft serve`
per backend and times token arrivals over streamed `/v1/chat/completions`, so numbers
include the full serving path. AIME-25 prompt, checkpoint-recommended sampling.

```bash
python benchmarks/bench_decode_moe.py --model /path/to/model --backend offload,cpu,hybrid
```

**`bench_load_weight_generic.py`** — expert-bank load time: serial vs parallel O_DIRECT
vs pre-repacked FTW, each mode in its own subprocess. Linux-only; stages the FTW under
`/var/tmp` (`--ftw-dir` overrides; roughly checkpoint-sized).

```bash
python benchmarks/bench_load_weight_generic.py --model /path/to/model
```

**`bench_offload_cache_copy.py`** — synthetic (no checkpoint): per-layer decode expert
copy cost (`ensure_experts` + `copy_missing`), swept over bank layout x cache slots x
batch size x miss rate.

```bash
python benchmarks/bench_offload_cache_copy.py
```

**`bench_scored_ensure.py`** — synthetic (no checkpoint): us per decode `ensure` call at a
fixed miss count, flashlib `lru_ensure` vs the `--moe-cache-policy` kernels (lru, kd, kdfb,
rule), from a CUDA graph of two decode steps on a full cache. `--stats` times it with
`--moe-collect-stats` accumulation on.

```bash
python benchmarks/bench_scored_ensure.py --slots 1650 --k 10,20
```

**`bench_moe_copy_overlap.py`** — synthetic (no checkpoint): `FREETOKEN_MOE_COPY_OVERLAP`
off vs on for Qwen4ExpMoE decode layers at the Qwen3.8-Flash-Next expert geometry, as CUDA
graphs. Checks the outputs are bitwise identical, times replays ABAB, and profiles the copy's
start after the ensure and how much of the side-stream shared expert runs inside it.

```bash
python benchmarks/bench_moe_copy_overlap.py --bs 1 --slots 160
```

**`bench_moe_prefetch.py`** — synthetic (no checkpoint): `FREETOKEN_MOE_PREFETCH=measure`
cost. The router-lookahead predictor alone per layer (GEMV `[bs, 2560] x [2560, 512]`, top-K
select, and the count kernel), then prefetch off vs measure on a real-geometry Qwen4ExpMoE decode
stack (512 experts, rule eviction, copy overlap on) with forced routing that misses exactly m
experts per layer, as CUDA graphs. `--on` times whole 48-layer decode steps with prefetch off vs
on instead: a GEMV per layer stands in for attention (`--filler-us`), each layer prefetches 3 or
4 candidates of which `--useful` are its misses, and `--delay-us` holds every prefetch copy back
to price a late one. It reports ms/step, demand misses/step and the late fraction. `--steady
--trace DIR` runs steady-state decode on a recorded routing trace instead (experts.npy,
logit_idx.npy, logit_val.npy, req_index.npy): no cache restore, so wrong prefetches and evictions
carry over, and a noisy copy of each layer's traced logits (`--sigma`) stands in for the
lookahead. Arms such as `off,measure,on,on:4/4` (budgets before GDN / QSA layers) replay step by
step in turn, each on its own copy of the cache state; it adds pollution (non-resident demands
per layer against off's misses) and the late fraction per layer kind. Slots and sigma were
calibrated to production's measure line (misses/layer ~4.3, precision ~0.56) at 24 layers.

```bash
python benchmarks/bench_moe_prefetch.py --bs 1,2 --misses 0,4
python benchmarks/bench_moe_prefetch.py --on --bs 1,2 --misses 4 --delay-us 0,50,150,300
python benchmarks/bench_moe_prefetch.py --steady --trace DIR --bs 1,2 --steady-slots 740 --sigma 0.55 \
    --arms off,measure,on,on:empty,on:3/3,on:4/4,on:4/5
```

**`bench_expert_copy_corun.py`** — synthetic (no checkpoint): the fused expert copy kernel
alone and beside compute, on registered host banks with the qwen4_exp NVFP4 row sizes. Sweeps
`fast_index_copy_multi` (threads x blocks/bank) and `fast_index_copy_multi_slim`
(blocks x threads x unroll) over 1..10-expert plans (GB/s, "hot" back-to-back and "cold"
spacer-separated copies), then reports the stretch sigma of bf16/fp8 GEMVs, a small matmul and
the NVFP4 routed decode GEMV while a copy loop runs on another stream. `--json` keeps the raw
per-trial times.

```bash
python benchmarks/bench_expert_copy_corun.py --trials 5 --json copy_corun.json
```

For host RAM vs PCIe bandwidth and the offload/hybrid backend pick, use `ft bench bw`
instead — it writes the JSON profile the engine reads.

**`bench_kv_quant.py`** compares BF16, FP8 and NVFP4 KV storage bytes, one-step
scatter latency and paged decode latency on synthetic inputs. No checkpoint is
required. Keep the GPU idle and use identical arguments for A/B comparisons;
this does not measure model quality or end-to-end serving throughput.

```bash
PYTHONPATH=python:. uv run python benchmarks/bench_kv_quant.py --lengths 1024,8192,32768
```
