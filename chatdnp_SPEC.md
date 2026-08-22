# chatdnp — System Spec and FreeToken Baseline

Target machine for FreeToken inference-server optimization.
Audit date: 2026-08-21. Keep this file current when hardware or baselines change.

## 1. Access

| Item | Value |
|---|---|
| SSH | `ssh -J douglas localadmin@chatdnp` |
| Hostname | `dnpgpu01` |
| Repo | `/home/localadmin/FreeToken` (editable uv venv at `.venv`) |
| FreeToken | 0.1.2, commit `0ab982f` |
| Bench outputs | `~/ftbench/` on the remote host |

## 2. CPU

| Item | Value |
|---|---|
| Model | 2 x Intel Xeon Gold 6526Y (Emerald Rapids) |
| Cores | 16 per socket, 32 physical, 64 logical (SMT on) |
| Clock | 800 MHz min, 3900 MHz max, `schedutil` governor |
| L2 | 2 MiB per core (64 MiB total) |
| L3 | 37.5 MiB per socket (75 MiB total) |
| NUMA | 2 nodes. node0 = CPU 0-15,32-47. node1 = CPU 16-31,48-63 |
| NUMA distance | local 10, remote 21 |

ISA of interest: `avx512f`, `avx512bf16`, `avx512_vnni`, `avx512_fp16`,
`avx_vnni`, `amx_bf16`, `amx_tile`, `amx_int8`.

**AMX is available.** The CPU MoE kernel does not use it. See finding F3.

## 3. Memory

| Item | Value |
|---|---|
| Total | 503 GiB usable (8 x 64 GB DDR5 RDIMM, 2Rx4, ECC) |
| Rated / configured | 5600 MT/s rated, **5200 MT/s configured** |
| Population | **4 of 8 channels per socket** (A1, C1, E1, G1 on each) |
| Per node | node0 257 GB, node1 258 GB |
| THP | `madvise` |
| Explicit hugepages | none reserved (`HugePages_Total: 0`) |
| Swap | 8 GiB, unused |

Theoretical DRAM read bandwidth with the current population:

- Per socket: 4 ch x 5200 MT/s x 8 B = **166 GB/s**
- System: **333 GB/s**

**Half the memory channels are empty.** Filling B1/D1/F1/H1 on both sockets
doubles the theoretical ceiling to 665 GB/s. Hybrid MoE decode is DRAM-read
bound, so this is the single largest hardware win available. See finding F1.

## 4. GPU

| Item | Value |
|---|---|
| Count | 2 x NVIDIA RTX 6000 Ada Generation (AD102GL) |
| VRAM | 49140 MiB each (48 GiB, ECC off) |
| Compute capability | 8.9 (sm_89) |
| Power cap | 300 W each |
| PCIe | Gen4 x16 (`LnkCap: 16GT/s, Width x16`) |
| Bus IDs | GPU0 `0000:2a:00.0`, GPU1 `0000:3d:00.0` |
| NUMA affinity | **both on node0** (CPU affinity 0-15,32-47) |
| Interconnect | `NODE` (PCIe through host bridges). **No NVLink.** |
| Driver | 595.71.05, CUDA 13.2 |

`nvidia-smi` reports `pcie.link.gen.current = 1` and `lspci` reports
`LnkSta: Speed 2.5GT/s (downgraded)` at idle. This is ASPM link power saving,
not a fault. Measured 25.1 GB/s H2D under load confirms the link trains up to
Gen4 (80% of the 31.5 GB/s Gen4 x16 theoretical maximum).

## 5. Storage

| Mount | Device | Size | Free | Notes |
|---|---|---|---|---|
| `/` | LVM on 2 x Samsung MZ7L3480 SATA SSD | 437 G | 216 G | OS, home |
| `/scratch` | Samsung MZQL23T8HCLS NVMe | 3.4 T | 2.6 T | **put model weights here** |

## 6. Software

| Item | Value |
|---|---|
| OS | Ubuntu 24.04.4 LTS, kernel 7.1.8-zabbly+ |
| Python | 3.12.3 |
| PyTorch | 2.11.0+cu130 (CUDA 13.0) |
| numactl | available |

## 7. Baseline: `ft bench bw`

Default per-dtype tuning bench, no CPU binding, 32 threads on 32 physical
cores across both sockets. Raw JSON: `~/ftbench/baseline-dtype.json`.

```
host dnpgpu01   gpu cuda:0 (RTX 6000 Ada)   cpu 32c/32t
ceilings: CPU STREAM read 265.1  |  PCIe linear H2D 25.1  D2H 26.4  GB/s

format   expert      CPU-MoE   PCIe-gather  CPU/PCIe  backend
bf16     9.00 MB    71.9 GB/s   25.1 GB/s     2.87x   hybrid
nvfp4    7.61 MB    61.2 GB/s   25.1 GB/s     2.44x   hybrid
fp8      3.00 MB          n/a   25.1 GB/s        --   offload  (no CPU fp8 path)
mxfp4   12.62 MB    44.6 GB/s   25.1 GB/s     1.78x   offload
ds_fp4  12.75 MB    65.6 GB/s   25.1 GB/s     2.61x   hybrid
```

Overlapped (CPU MoE and PCIe gather run together, `load_hybrid_fetch_fraction`):

| format | CPU-MoE | PCIe | fetch fraction |
|---|---|---|---|
| bf16 | 66.8 | 15.4 | 18.8% |
| nvfp4 | 46.0 | 24.1 | 34.4% |
| mxfp4 | 43.5 | 25.0 | 36.5% |
| ds_fp4 | 53.0 | 19.6 | 27.0% |

Note: bf16 loses 10 GB/s of PCIe under overlap (25.1 -> 15.4). The other
formats keep almost full PCIe. The bf16 CPU kernel starves the DMA engine of
host DRAM.

## 8. Thread, SMT and NUMA sweep

All numbers below are from a **clean idle machine**: page cache dropped, both NUMA
nodes at ~253 GB free, load average 0.47, download finished. Median of 3 runs.
See 12.7 for why that matters — a skewed page cache halves these figures.

`numactl` binds CPUs and memory to the named node. Thread counts come from
`--cpu-threads`. `physical_core_cpus()` fills physical cores first, then SMT
siblings, so 32 threads inside a 16-core node means SMT2.

### 8.1 Results (GB/s, median of 3)

| Configuration | STREAM | bf16 | nvfp4 | mxfp4 | ds_fp4 |
|---|---|---|---|---|---|
| 32T, both sockets (**default**) | 264.8 | 67.1 | 58.8 | 47.8 | 65.2 |
| 32T, `numactl -N0 -m0` (SMT2) | 135.9 | **123.8** | **93.6** | **53.9** | 80.6 |
| 32T, `numactl -N1 -m1` (SMT2) | 135.3 | 116.7 | 91.7 | 51.8 | **89.4** |
| 16T, `numactl -N1 -m1` | 137.9 | 113.4 | 82.2 | 49.9 | 89.5 |

Gain of one-socket SMT2 over the default:

| Format | Default | Best bound | Gain |
|---|---|---|---|
| bf16 | 67.1 | 123.8 (N0) | **+84%** |
| nvfp4 | 58.8 | 93.6 (N0) | **+59%** |
| ds_fp4 | 65.2 | 89.4 (N1) | **+37%** |
| mxfp4 | 47.8 | 53.9 (N0) | +13% |

`ds_fp4` is the format DeepSeek-V4-Flash uses (section 12), so +37% is the number
that matters for the model on this box.

### 8.2 These are microbenchmark numbers and they do not predict serving

**Read F2 before acting on anything in 8.1.** Every figure above comes from
`ft bench bw`, which times one kernel in isolation over a ~2 GiB synthetic bank.
End-to-end measurement on a real 143 GiB model reversed the sign: node binding is
6.7% slower under `--moe-backend hybrid` and 28% slower under `--moe-backend cpu`.

The table is retained only as a record of how far a microbenchmark can mislead —
it predicted +42.5% where reality delivered -6.7%.

### 8.3 What to actually run

No binding. The default every-core pool is correct on this machine:

```bash
ft serve --moe-backend hybrid --model <dir> ...
```

Do **not** wrap it in `numactl --cpunodebind/--membind`, and do not set
`--moe-cpu-threads`: the auto sizing (one thread per physical core, all sockets)
measured fastest end to end. `~/run.sh` on chatdnp reflects this.

The one placement thing that *does* matter is page-cache skew — see 12.7. That is
about keeping both nodes able to allocate, not about confining anything.

## 9. Findings and optimization candidates

Ranked by measured value. Every item must stay optional or auto-detected so
other machines are unaffected.

### F1 — Half the DRAM channels are empty (hardware, largest single win)

4 of 8 channels populated per socket caps the box at 333 GB/s instead of 665.
Fill B1/D1/F1/H1 on both sockets. Hybrid decode is DRAM-read bound, so this
scales the CPU-MoE number almost linearly. No code.

### F2 — NUMA confinement is WRONG here. The existing every-core default is right.

Two rounds of end-to-end measurement killed this. Recorded in full because the
microbenchmark was emphatically, repeatably wrong.

`ft bench bw` said confining the CPU MoE pool and its banks to the GPU's node was
worth +42.5% (bf16) and +29.9% (ds_fp4). Serving said otherwise.
DeepSeek-V4-Flash-0731, alternating configs, two cycles each, caches dropped:

| backend | confined | today (`=off`) | verdict |
|---|---|---|---|
| `hybrid` | 17.93 / 18.07 | **19.21 / 19.39** | 6.7% slower |
| `cpu` | 8.01 / 7.91 | **11.21 / 11.02** | **28% slower** |

The `cpu` result is the decisive one. It was meant to be the *favourable* case —
no PCIe gather competing for DRAM — and confinement lost four times harder.

**The principle:** expert decode is a read-once stream. Every expert byte of every
layer comes from DRAM and is never reused. For that shape, what matters is
**aggregate memory-controller bandwidth**, not locality. Both sockets give ~333
GB/s across two controllers; one node gives ~166 GB/s. A UPI-remote read is
cheaper than surrendering half the bandwidth.

Why the microbenchmark inverted it: its synthetic banks are ~2 GiB and one layer
deep, so confined + `mbind`-ed they sit wholly in the local node and are re-read
hot. Production streams 143 GiB.

So `cpu_moe_ext.cpp:1094`'s "NUMA: a single node is assumed" is **not a latent bug**.
Spreading the pool across every core is correct. <https://github.com/FlashML-org/FreeToken/pull/18>
is closed.

Untested alternative, and the opposite of what was tried: bank pages currently land
wherever the 8 unpinned loader threads run, so the node split is a lottery.
`MPOL_INTERLEAVE` would force an even 50/50 and load both controllers evenly. That
is a candidate, not a recommendation — no end-to-end numbers.

**Method rule this establishes: `ft bench bw` is not evidence for a serving change.
It times one kernel in isolation over a small synthetic bank. Confirm with tokens/s
on a real model, alternating configs across at least two cycles, before claiming
anything.**

### F3 — No CPU path for `fp8_block` (code, unlocks a checkpoint tier)

`_WFMT_IDS` has no `fp8_block`, so every block-fp8 checkpoint is offload-only
and capped at 25 GB/s PCIe. The format is the simplest of all of them:
fp8-e4m3 weights plus a per-128x128 block scale.
The executor already carries `float e4m3_lut[256]`.

e4m3 to bf16 is pure bit manipulation (`sign<<15 | (exp-7+127)<<7 | mant<<4`,
plus a masked fixup for the denormal exponent), which feeds straight into
`_mm512_dpbf16_ps` on this CPU. Expected to land in the bf16 kernel's
bandwidth class at half the bytes. Where the scale is `ue8m0` (a power of two)
it folds into the bf16 exponent as an integer add, with no multiply at all.

Adding a format touches three tables (`_BANK_SCHEMAS`, `_PROVIDERS`,
`_expert_gemm`) plus a `WFmt` id and a dot function with scalar, AVX2 and
AVX-512 variants. Purely additive; nothing else changes.

**This does not block DeepSeek-V4-Flash.** See section 12 — that checkpoint's
*experts* are fp4, which the CPU path already serves. Priority for F3 rests on
other checkpoints (Qwen3.6-FP8, Qwen3.5-FP8, DeepSeek-V3-style block-fp8), not
on the model staged here.

### F4 — mxfp4 is compute-bound, but the VNNI fix is layout-blocked

**Correction.** An earlier draft of this document claimed the nvfp4 W4A8 VNNI path
"ports across almost directly" to mxfp4 because both use e2m1 codes. That is wrong,
and the reason is the bank layout, not the codes.

| format | bank | contiguous dim |
|---|---|---|
| nvfp4 | row-major `[rows, K]` | **K** |
| mxfp4 | `blocks_t [K//2, N]` (transposed split-K) | **N** (output columns) |

`VPDPBUSD` reduces four adjacent int8 pairs into one int32. For nvfp4 those four
are adjacent K, which is exactly the dot product. For mxfp4 the adjacent bytes are
four different *output columns*, so the same instruction would sum across columns
and return garbage. `_mm512_dpbf16_ps` has the same problem for the same reason.

mxfp4's kernel is an outer-product accumulation instead: broadcast `x[2k]` and
`x[2k+1]`, FMA into N column accumulators. Its cost is real —
`_mm512_cvtepu8_epi32` consumes only 16 of 64 byte lanes per load, then two
`_mm512_permutexvar_ps` and two FMAs per 16 bytes. That is why mxfp4 is the only
compute-bound format (59.8 GB/s against bf16's 96.2, on *more* bytes per expert).

Making VNNI usable needs the bank transposed to K-contiguous, and `blocks_t` is
shared with the Triton GPU decode kernel and the transposed grouped prefill path.
That is a much larger and riskier change than a kernel port. Candidates that stay
inside the current layout, none yet measured:

- Decode 64 bytes per iteration with `_mm512_shuffle_epi8` (as the nvfp4 W4A8 path
  already does) instead of 16 bytes with `_mm512_cvtepu8_epi32`.
- `_mm512_permutexvar_epi16` against a bf16 LUT to cut the widening chain.

Profile before building either. Not a quick win.

### F5 — Huge pages for the host banks: MEASURED, NO EFFECT. Do not do this.

The theory was good and the theory was wrong. Recorded so nobody re-derives it.

`HostBank` mmaps get zero huge pages today, and for two reasons, not one:
`mmap.mmap(-1, n)` defaults to **MAP_SHARED**, so the banks are shmem-backed
(`numa_maps` shows `file=/dev/zero (deleted)`), and `shmem_enabled` is `[never]`
by default — so `MADV_HUGEPAGE` on them can never do anything. Switching to
`MAP_PRIVATE|MAP_ANONYMOUS` plus `MADV_HUGEPAGE` does work, and the huge pages
survive `cudaHostRegister` (verified: 524288 of 524288 kB still huge after pinning).

It buys nothing. Measured against banks allocated the production way
(mmap -> fill -> register), median of 3:

| bank mapping | bf16 | ds_fp4 | AnonHugePages |
|---|---|---|---|
| MAP_SHARED (today) | 106.5 | 87.0 | 0 |
| MAP_PRIVATE | 107.1 | 86.3 | 0 |
| MAP_PRIVATE + MADV_HUGEPAGE | 106.7 | 84.8 | 3456 / 8352 MiB |

All within run-to-run noise; ds_fp4 is marginally *worse*. The GEMV streams each
expert's rows contiguously, so the hardware prefetcher covers the walks and the
random part — which expert — is amortized over 12.75 MB of sequential reads. The
working set already exceeds the STLB at 4 KiB *and* at 2 MiB, so nothing changes.

Untested and separate: 130 GiB of 4 KiB pages means ~34M faults at **load time**.
Huge pages could cut model start-up latency even though they do nothing for
decode. Needs a real model load to measure (blocked here by the memlock cap).

### F6 — One thread policy for two kernel regimes

Unbound, bf16 peaks near 16 threads while mxfp4 keeps climbing to 32 — mxfp4 is
compute bound, the rest are DRAM bound. Binding to one node mostly dissolves it
(one config now wins every format), but the general fix is for `ft bench bw` to record
the winning (threads, node) pair per dtype into `benchbw.json` alongside the
backend pick it already stores, and for the executor to read it. Absent or
stale entry falls back to today's behaviour, so other machines are unaffected.

### F7 — Both GPUs sit on node0, and TP=2 doubles the PCIe budget

`--tensor-parallel-size 2` is supported. The two GPUs hang off separate root
ports on socket 0, so TP=2 gives two independent Gen4 x16 links: about 50 GB/s
aggregate host-to-device instead of 25. It also doubles resident VRAM to 96 GiB,
which cuts the expert miss rate before any streaming happens.

`ft bench bw` measures one device and does not model either effect, so its
hybrid-vs-offload recommendation is pessimistic for a TP=2 deployment. Worth a
`--tp` aware mode.

### F8 — AMX is present and unused

`amx_bf16` and `amx_int8` are available. The in-code comment defers AMX
because decode is an M=1 GEMV. It only pays off together with the expert dedup
for batch size > 1 that the same comment defers, or with speculative decoding
where the draft gives a real M. Measure before building.

### F9 — mxfp4 backend pick is not reproducible

The CPU/PCIe ratio is 1.78x against a 2.0x threshold and the measurement swings
between 19 and 71 GB/s across configurations and repeats. The recommendation
flips between `hybrid` and `offload` on repeat runs of the same command. F4
should move it clear of the boundary. Until then, pin it explicitly.

### F10 — Other quantization formats

Hybrid decode is DRAM-read bound, so bytes per weight sets throughput almost
directly. Tiers, and what is missing:

| Tier | Bits/weight | CPU path today |
|---|---|---|
| bf16 | 16 | yes |
| fp8_block, W8A8 int8 | ~8 | **no** (F3) |
| nvfp4, mxfp4, ds_fp4, q4_0 | 4.25-4.5 | yes |
| AWQ/GPTQ int4 | ~4.25 | no |
| GGUF K-quants (q4_K, q5_K, q6_K) | 4.5-6.6 | no |
| i-quants (IQ2, IQ3), q2_K, q3_K | 2-3.4 | no |

- **fp8_block is the only gap that buys speed.** See F3. It is also the tier
  this machine should prefer: 503 GiB of RAM holds a 290B model in fp8 (~290
  GB) with room to spare, so there is no need to drop to 4-bit for capacity
  here. Better accuracy *and* 4x the PCIe bandwidth.
- **AWQ/GPTQ int4 and GGUF K-quants buy model coverage, not speed.** They sit
  in tiers already covered by nvfp4 and q4_0. Add them if a wanted checkpoint
  ships only in that format.
- **Sub-4-bit is the wrong direction on this box.** 2-3 bit formats halve the
  bytes again, but their decode chains (superblocks, nested scales, LUT
  gathers) are exactly what makes mxfp4 compute-bound today. This machine has
  bandwidth headroom relative to consumer hardware, not compute headroom.
  Accuracy loss on MoE experts is also worst where routing is sparse. Do not
  chase this before F4 proves the decode chain can keep up.
- **MXFP6/NVFP6** would be the interesting middle tier, but few checkpoints
  ship it.

## 10. Measurement protocol

- Run `ft bench bw` at least 3 times and take the median. Single runs of
  mxfp4 are not trustworthy: observed 19 to 71 GB/s across this audit.
- Drop the page cache first and check both nodes have free memory (12.7).
- Record `free -h` and `uptime` before each run.
- `nvidia-smi` shows `pcie.link.gen.current = 1` at idle. Ignore it.
- Bind explicitly when comparing: `numactl --cpunodebind=N --membind=N`.
  Unbound runs have about twice the spread of bound runs.
- `FREETOKEN_CPU_MOE_PF_BLOCKS` overrides the W4A8 prefetch distance. The
  default (4 KiB, capped at 2 rows) was tuned on Emerald Rapids, which is this
  CPU, but the comment records +20% from an explicit value elsewhere. Sweep it.
- `FREETOKEN_CPU_MOE_ISA` forces the ISA tier. `--isa all` sweeps it.

### Provenance

Section 7's baseline was taken while a 156 GiB `hf download` was running.
Section 8 was re-measured afterwards on a clean idle machine with the page cache
dropped and both nodes balanced, and **section 8 supersedes section 7 wherever
they disagree**. The cached runtime profile at
`~/.cache/freetoken/benchbw.json` was last written by the clean unbound run.

## 11. Commands that need root

```bash
sudo dmidecode -t memory                                 # DIMM population and speed
sudo lspci -vv -s 2a:00.0 | grep -E 'LnkCap:|LnkSta:'    # real PCIe link caps
```

## 12. Serving DeepSeek-V4-Flash on this box

### 12.1 The checkpoint

Downloaded and complete at:

```
/scratch/localadmin/hf-cache/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/7872f01b1d1fe23eabc4c98b48bffcef5a386062
```

156 GiB, 48 safetensors shards, 0 incomplete blobs. `HF_HOME=/scratch/localadmin/hf-cache` (set in `~/.bashrc`).

**Read `inference/config.json`, not the HF `config.json`.** `models.md` says the
DeepSeek-V4 subdir is authoritative, and the two disagree on what matters:

| Source | Says |
|---|---|
| HF `config.json` `quantization_config` | `quant_method: fp8`, `weight_block_size: [128,128]` |
| `inference/config.json` (**authoritative**) | `dtype: fp8`, `scale_fmt: ue8m0`, **`expert_dtype: fp4`** |

The `fp8` applies to the dense and attention weights. The **experts are fp4**,
which the CPU path already serves as `ds_fp4`. 156 GiB on disk confirms fp4
experts; fp8 experts would be roughly 277 GB.

So this model is **not** blocked by the missing `fp8_block` CPU path (F3).
Baseline measures `ds_fp4` at 65.6 GB/s CPU-MoE against 25.1 GB/s PCIe, ratio
2.61x, recommendation `hybrid`.

Geometry: `dim` 4096, `moe_inter_dim` 2048, `n_layers` 43, `n_routed_experts`
256, `n_shared_experts` 1, `n_activated_experts` 6. Matches the bench's `dsv4`
workload exactly.

### 12.2 Downloading

`hf download <repo>` is the correct procedure. It fetches the whole repo,
including the `inference/` and `encoding/` subdirs that DeepSeek-V4 needs.

Do **not** narrow it with `--include "*.safetensors"`: `load_args` looks for
`inference/config.json` (then `model_args.json`) and hard-fails without it.

### 12.3 `ft serve --model <hf-repo-id>` (fixed in PR #17)

```
FileNotFoundError: No DeepSeek-V4 ModelArgs JSON found under
deepseek-ai/DeepSeek-V4-Flash-0731 (looked for inference/config.json)
```

Not a download problem. `server/args.py` only calls `snapshot_download` on the
**modelscope** branch; for the default `--model-source huggingface` the repo id
is passed through as a literal filesystem path. And `download_hf_weight()`
(`utils/hf.py`) uses `allow_patterns=["*.safetensors"]`, so even where it does
run it would never fetch `inference/config.json`.

Fixed by <https://github.com/FlashML-org/FreeToken/pull/17> (branch
`fix/hf-repo-id-model-path`), which resolves the repo id to a local snapshot for
the `huggingface` source too. Until that lands upstream, **pass a local
directory**.

```bash
ft serve --model /scratch/localadmin/hf-cache/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/7872f01b1d1fe23eabc4c98b48bffcef5a386062
```

Verified: `cached_load_hf_config` + `parse_config` succeed on that path. Full
weight load is untested.

### 12.4 Backend selection

`--moe-backend auto` upgrades `offload` to `hybrid` only when a cached profile
at the **default** path recommends it:

```
/home/localadmin/.cache/freetoken/benchbw.json
```

Present and complete, with `ds_fp4: hybrid`. Any `ft bench bw -o <other path>`
run does **not** refresh it — this audit wrote most of its runs elsewhere, so
re-run `ft bench bw` with no `-o` after any tuning change, or the runtime keeps
using a stale split.

If in doubt, pass `--moe-backend hybrid` explicitly.

### 12.5 Order of operations

1. Serve unbound first and confirm it works end to end.
2. Only then try the section 8.3 binding. The banks are ~130-156 GiB pinned
   against node1's 258 GiB: it fits, but strict `--membind` converts tight into
   OOM rather than spillover.
3. A bound deployment needs its **own** default-path `ft bench bw` run under
   the same binding, so the cached fetch split matches what the runtime does.

### 12.6 FTW conversion (optional)

`ft checkpoint --model <dir> --out <ftw_dir>` pre-converts to the fast-load
format; `ft serve --model <ftw_dir>` auto-detects it. Safe for DeepSeek-V4:
`convert.py:61` documents the DSV4 case specifically and `_copy_metadata`
copies every non-weight file preserving relative paths, so `inference/` and
`encoding/` survive. Put the output on `/scratch` (2.6 TiB free), not `/`
(216 GiB free).

### 12.7 Page cache skews NUMA free memory and halves DRAM bandwidth

After copying the 156 GiB checkpoint between `/scratch` directories, an **idle**
machine measured STREAM at 137-147 GB/s — half the 265-278 GB/s measured
earlier under light load.

Cause:

```
free:        36 GiB free, 457 GiB buff/cache
node 0 free:  1353 MB      <-- starved
node 1 free: 35948 MB
```

The file copy filled node0 with page cache. New allocations can no longer
first-touch there, so the STREAM buffers all land on node1. Half the workers
then read remote and the result collapses to the single-node figure.

This is not a bench artifact. It hits the real thing harder: the expert banks
first-touch during model load, so loading in this state puts all ~130-156 GiB
of pinned banks on node1 while **both GPUs sit on node0** — every PCIe gather
then crosses UPI, permanently, for the life of the process.

**Before loading a model after any large file copy, drop the page cache.**
Needs root:

```bash
sync && echo 3 | sudo tee /proc/sys/vm/drop_caches
```

Then confirm both nodes have free memory before starting:

```bash
numactl --hardware | grep free
```

This also invalidates any `ft bench bw` run made while the nodes are skewed.
The profile at `~/.cache/freetoken/benchbw.json` was last written at
2026-08-22 00:25, **during** the skew, so its ceilings are low. Re-run it after
dropping caches.
