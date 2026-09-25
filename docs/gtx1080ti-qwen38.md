# Qwen3.8-Flash-Next on one GTX 1080 Ti

Branch `1080deploy` serves [nvidia/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4)
to a single agentic user at the native 262,144-token context from one 11 GB Pascal card.
Run it with [`scripts/freetoken-qwen38-1080.sh`](../scripts/freetoken-qwen38-1080.sh).

## The box

| | |
|---|---|
| GPU | GTX 1080 Ti, sm_61, 11 GB, PCIe 3.0 x16 (12.4 GB/s measured). No tensor cores, no bf16/fp8/fp4 math, 48 KB shared memory |
| CPU | 2x Broadwell Xeon, 10 cores each (AVX2, no VNNI), two NUMA nodes, GPU on node 0 |
| RAM | 125 GB. `/tmp` is a RAM disk: keep scratch and `TMPDIR` on disk |
| Disk | NVMe for the model and the PLE table |
| Driver | 580.x with `options nvidia NVreg_EnableStreamMemOPs=1` (`/etc/modprobe.d/nvidia-streammemops.conf`, also in the initramfs) |

## Checkpoint

`nvidia/Qwen3.8-Flash-Next-NVFP4`: the best quality evidence among the NVFP4 builds (it passed a needle test that the
RadixArk build failed). Its dense layers are bf16 and need about 9 GB of VRAM, so the engine requantizes them to MXFP8
at load (`--dense-quant`, `--lm-head-quant`, `--hc-quant`; blocks of 32 with e8m0 scales, finer than the 128x128 FP8
builds on the Hub) and keeps the token embedding in pinned host RAM (`--embed-weights host`).

Measured cost on 10,485 positions of fixed real text, teacher-forced, against a bf16 reference:
dense MXFP8 +0.0053 +- 0.0038 nats/token, hyper-connections MXFP8 -0.0019 (noise), lm_head MXFP8 +0.0011;
about +0.006 nats/token (0.5% perplexity) in total, within noise.

Not used: the official FP8 and bf16-PLE builds (larger than RAM or NVMe), int4 GPTQ/AWQ and MXFP4 experts
(no loader or CPU kernel), local-inference-lab's NVFP4 PLE (no loader), REAP-pruned builds (no evals).

## Serving configuration

| Setting | Why |
|---|---|
| `--moe-strategy hybrid --moe-hybrid-max-fetch 1` | Decode computes expert-cache misses on 19 CPU workers (AVX2 int8 NVFP4 kernel, rows placed per NUMA node) and fetches at most one per layer over PCIe |
| `--memory-ratio 0.86` | ~1130 GPU expert slots: enough (>= 1024) for prefill overlap; the engine reserves the host KV tier and a 512 MiB margin itself |
| `--kv-cache-dtype fp8 --kv-reserve-tokens 32768 --kv-host-pages 3584` | 32K tokens of KV on the GPU plus a pinned host mirror: 262,208 logical tokens |
| `--max-prefill-length 2048` | Prefill chunk; the whole-layer expert stream (~6 s) is amortized over it |
| `FT_GDN_HOST_TIER=1 FT_GDN_HOST_SLOTS=32` | GDN state snapshots evicted to host RAM, so an agent's long conversation is not re-prefilled after side requests |
| `FREETOKEN_MOE_SMALL_PREFILL_TOKENS=256` | Turns of up to 256 new tokens load only their routed experts instead of streaming all 48 layers |
| `--max-running-requests 1 --cuda-graph-max-bs 1` | Single user |
| `--max-output-tokens 65536` | The default thinking effort runs past the 32K default on agentic tasks |
| `numactl --interleave=all` | Non-expert allocations spread over both sockets' memory controllers |

## Measured performance

Decode, steady state, 1K context, temperature 0 (in-process A/B with the production flags unless noted):

| Change | Decode tok/s |
|---|---|
| Before the CPU MoE and memops work (fetch 3) | 9.0 (older timing that also counted a ~6 s re-prefill, so not directly comparable) |
| AVX2 int8 NVFP4 kernel, NUMA placement, worker hot-spin, fetch 1 | 18.15 (host-func handshake) |
| + 32-bit stream-memops flag handshake | 19.08 |
| Server, with the 262K host KV tier, before the selection-compaction fix | 14.0 |
| Server, with the host KV tier, after it | 17.2 (in-process) |

Short agent turns over a cached 4K prefix (time to the first token): 40 new tokens 1.9 s, 128 about 3.2 s
(5.9 s each when every prefill streams all experts).

Prefill (server): 10K prompt 173 tok/s, 32K prompt 150 tok/s; 8K prompt 200 tok/s in-process after the
GDN tile change (143 before).

Long context through the server (recall = three codes planted at 5%, 50% and 95% of the prompt):

| Prompt tokens | Time to first token | Prefill tok/s | Decode tok/s (streamed) | Recall |
|---|---|---|---|---|
| 1,026 | 8.1 s | 126 | 17.4 | - |
| 27,337 | 2.9 min | 158 | 16.6 | 3/3 |
| 84,232 | 18.2 min | 77 | 11.0 | 3/3 |
| 247,231 | 2 h 20 min | 29 | 6.8 | 3/3 |

Prefill and decode slow down with depth: the QSA indexer scores every earlier token, and past 32K tokens
the selected KV pages come from the host mirror. At ~250K a few selections span more pages than the GPU
pool can hold for one row (the server logs "selected pages dropped"); recall stayed 3/3.
A cold 262K prompt is therefore a multi-hour job; agent sessions grow their context incrementally through
the prefix cache, so each turn only prefills its new tokens.

Tool-call round trips (call, tool result, answer) pass on `/v1/chat/completions`, `/v1/messages` and
`/v1/responses`, with and without thinking.

## Pascal changes on this branch

- FlashInfer and sgl_kernel treated as unavailable below sm_75 / sm_80.
- Triton: no atomics below sm_70 (QSA offload counter, --moe-collect-stats), no `.evict_first`, 48 KB tiles:
  QSA sparse attend retry ladder and 16-wide tiles (4.4x), GDN chunk tiles (12-13x per kernel, +38% prefill),
  NVFP4 MoE prefill and decode tiles, arithmetic e2m1 decode, MXFP8 GEMV sized for Pascal.
- MXFP8 dequant fallback in bounded row chunks (the whole lm_head was a 2.4 GiB transient).
- 32-bit stream memops when the device rejects the 64-bit ones (CPU MoE handshake, PLE row store).
- PLE fill events recycled on the engine thread (a deadlock with the host-func handshake).
- QSA selection compaction by column mark and scan (1.28 ms -> 0.17 ms per layer).

## Operations

```bash
scripts/freetoken-qwen38-1080.sh start     # background, returns when ready (log ~/.cache/freetoken/qwen38-1080.log)
scripts/freetoken-qwen38-1080.sh status
scripts/freetoken-qwen38-1080.sh stop      # add --force to escalate
scripts/freetoken-qwen38-1080.sh test      # one request
EXTRA_ARGS="--enable-metrics-report" scripts/freetoken-qwen38-1080.sh start
```

As a systemd user service: `cp scripts/freetoken-qwen38-1080.service ~/.config/systemd/user/ && systemctl --user
enable --now freetoken-qwen38-1080`; run `sudo loginctl enable-linger $USER` so it survives logout.

After changing the C++ sources or pulling: `CUDA_HOME=/opt/cuda NVCC_CCBIN=/usr/bin/g++-14 .venv/bin/python setup.py
build_ext --inplace` (the CPU MoE executor degrades to the host-func handshake on a stale build).

Clients: OpenAI chat/Responses and Anthropic Messages on port 8080, model `qwen3.8-flash-next`, tool parser
`qwen3_coder` and reasoning parser `qwen3` auto-detected. Send reasoning back on later turns (preserve_thinking)
so the prefix cache keeps hitting.

Known test failures on this card (other models' kernels that need more than 48 KB of shared memory or sm_70+):
MiniMax-M3 sparse attention, GLM DSA, FP8 block-scale MoE, NVFP4 sparse-MLA KV; plus
`test_scheme_for_agrees_with_the_stored_tensors[sentence-transformers/all-MiniLM-L6-v2]`, which scans a local HF cache.
Full suite: 2771 passed, 50 failed (all in that list), 234 skipped.
