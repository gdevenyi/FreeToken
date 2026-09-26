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
