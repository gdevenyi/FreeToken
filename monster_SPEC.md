# monster — machine specification

Audited 2026-08-21, software matrix verified 2026-08-22, by direct measurement on the host. Every number below is either read
from the system or measured with the microbenchmarks kept in
`~/scratch/agents/claude/monster-spec/` on the host (`stream.c`, `bw.cu`, `gemm3.cu`, `gemm4.cu`, `vram.cu`, `t1_torch.py`, `t3_verify.py`,
`t4_perf.py`, `t6_int8.py`, plus the `tritontest` venv).

## 1. Access and layout

| Item | Value |
|---|---|
| SSH | `ssh monster` (user `gdevenyi`, uid 1000, in `wheel`, passwordless sudo) |
| LAN | `192.168.1.221` (eno1), Tailscale `100.96.159.9` |
| FreeToken checkout | `/home/gdevenyi/FreeToken` (**not** `/home/localadmin/FreeToken` — that path does not exist) |
| Venv | `/home/gdevenyi/FreeToken/.venv`, Python 3.11.15, currently **empty** (no torch, no pip) |
| `uv` | `/home/gdevenyi/.local/bin/uv` 0.11.24 — not on the non-interactive SSH `PATH` |
| Scratch | `~/scratch/agents/claude/monster-spec` |

Non-interactive SSH gets `PATH=/usr/bin:/bin:/usr/sbin:/sbin` only. Call `uv`, `nvcc`
by absolute path or export `PATH` first.

## 2. Chassis and board

| Item | Value |
|---|---|
| Board | Supermicro X10DAL-i rev 1.02 (Intel C612) |
| BIOS | AMI 3.2, 2019-11-26 |
| Uptime at audit | 22 days |

PCIe slots (from SMBIOS):

| Slot | Electrical | Occupant |
|---|---|---|
| CPU1 SLOT1 (x16 mech) | Gen3 x8 | LSI SAS2308 HBA (the 8 spinning disks) |
| CPU1 SLOT5 | Gen3 x16 | **GTX 1080 Ti** (`0000:02:00.0`, NUMA node 0) |
| CPU2 SLOT2 (x8 mech) | Gen3 x4 | occupied (`0000:82:00.0`) |
| CPU1 SLOT3 | Gen3 x16 | **free** — a second GPU fits here |
| PCH SLOT6 (x8 mech) | Gen2 x4 | free |

## 3. CPU

| Item | Value |
|---|---|
| Model string | `Genuine Intel(R) CPU 0000 @ 2.20GHz` — an **engineering sample** |
| Identity | family 6, model 79, stepping 1 = Broadwell-EP; 10C/2.2 GHz/25 MB L3 matches Xeon E5-2630 v4 |
| Sockets | 2 |
| Cores / threads | 20 physical / 40 logical (10C/20T per socket, HT on) |
| Clocks | 1.2 GHz min, 2.2 GHz base, 3.1 GHz max |
| Cache | L1d 32 KB ×20, L1i 32 KB ×20, L2 256 KB ×20, L3 25 MB ×2 |
| Governor | `powersave` (intel_pstate HWP); `power-profiles-daemon` running |
| Microcode | 0xb000038 |

### ISA — this is the important part

Present: `avx`, `avx2`, `fma`, `f16c`, `aes`, `bmi1/2`, `rtm`, `hle`.

**Absent: every AVX-512 flag, `avx512bf16`, `avx_vnni`, `amx_*`.**

Consequence for FreeToken: `freetoken.kernel._cpu_moe` gates its bf16 GEMV microkernels
on `avx512bf16`/`avx512f` via `__builtin_cpu_supports`. On this host every dispatch lands
in the **scalar fallback**. The `--moe-backend cpu` path will build and run but will be
far off its designed throughput. AVX2+FMA microkernels would be the optimization to add.

### NUMA

```
node 0: cpus 0-9,20-29   64 GB   <- GTX 1080 Ti lives here (local_cpulist 0-9,20-29)
node 1: cpus 10-19,30-39 64 GB
distance 0->1: 21 (vs 10 local)
```

Pin the engine to node 0 (`numactl --cpunodebind=0 --membind=0`) so pinned host buffers
and the GPU share a socket. Measured cross-socket penalty is 21% on memory bandwidth.

## 4. Memory

| Item | Value |
|---|---|
| Installed | 128 GB = 4 × 32 GB DDR4 RDIMM, SK Hynix, dual-rank, ECC (multi-bit) |
| Rated / actual | 2400 MT/s parts running at **2133 MT/s** |
| Population | `P1-DIMMA1`, `P1-DIMMB1`, `P2-DIMME1`, `P2-DIMMF1` — **2 of 4 channels per socket** |
| Free at audit | ~120 GB available |
| Swap | 126 GB zram (`zram0`) |
| THP | `always` |

**Half the memory channels are empty.** Adding 4 more DIMMs (C1/D1, G1/H1) would roughly
double memory bandwidth — the single biggest hardware upgrade for a CPU-offload MoE runtime.

Measured STREAM triad (`gcc -O3 -march=native -fopenmp`):

| Configuration | GB/s |
|---|---|
| 40 threads, both sockets | **43.8** |
| node 0 only, 20 threads, local memory | **22.1** |
| node 0 CPUs, node 1 memory (cross-socket) | **17.4** |

Theoretical per socket with 2 channels @2133 = 34.1 GB/s; 22.1 GB/s achieved is the
usual ~65% triad efficiency.

## 5. GPU

| Item | Value |
|---|---|
| Model | NVIDIA GeForce GTX 1080 Ti (GP102, Pascal) |
| **Compute capability** | **6.1 (sm_61)** |
| SMs / CUDA cores | 28 / 3584 |
| VRAM | 11264 MiB GDDR5X, 352-bit, ECC **off** (not supported) |
| Clocks | 1607 MHz base boost, 1923 MHz max; memory 5505 MHz |
| Power | 250 W default limit, 300 W max settable |
| PCIe | Gen3 x16 (link drops to Gen1 at idle for power saving) |
| Driver | 580.173.02 (**last branch that supports Pascal**), reports "CUDA Version: 13.0" |
| VBIOS | 86.02.39.00.54 |
| UUID | `GPU-ffce7021-6689-bbe2-1aa6-aca83735d427` |
| Persistence mode | Enabled during this audit (`nvidia-smi -pm 1`); not persistent across reboot |

### Measured throughput

Persistence mode on; GPU settled at ~1800 MHz SM / 5508 MHz mem / ~100 W / 70 C.

| Benchmark | Measured | Theoretical |
|---|---|---|
| cuBLAS SGEMM fp32, n=2048 / 4096 / 8192 | **8.5 / 10.0 / 9.9 TFLOPS** | 11.3 |
| cuBLAS GEMM fp16 in / fp32 accumulate | **7.7 TFLOPS** | - |
| cuBLAS GEMM bf16 in / fp32 accumulate | **4.8 TFLOPS** | - |
| cuBLAS GEMM int8 in / int32 accumulate (dp4a) | **42.6 TOPS** | 45.9 |
| VRAM triad | **359 GB/s** | 484 at 5508 MHz |
| PCIe H2D, pinned, 1 GiB | **12.44 GB/s** | 15.75 |
| PCIe D2H, pinned, 1 GiB | **13.20 GB/s** | 15.75 |
| PCIe H2D, pageable | 12.04 GB/s | - |

Measure with a warm Triton cache and no concurrent JIT: a background compile halves every
number on this 2.2 GHz CPU.

### What Pascal can and cannot do

- **No tensor cores.** No bf16 anywhere, no fp8, no TF32, no MMA instructions.
- Native fp16 arithmetic is 1:64 rate — useless for compute. But **fp16 storage with fp32
  compute costs almost nothing** (8.81 vs 9.98 TFLOPS) and halves VRAM and PCIe traffic.
  That is the right weight format for this card.
- **`dp4a` int8 is the fast path: 4.3× fp32.** Any W8A8 quantized GEMM is the single
  largest available win on this GPU.
- No async copy (`cp.async`), no `griddepcontrol`/PDL (sm_90+), no cluster/CTA groups.
- 96 KB shared memory per SM, 48 KB per block without opt-in.

### The PCIe wall

12.4 GB/s H2D is the hard ceiling on MoE expert streaming. Per decode step, whatever
weight bytes cross PCIe cost `bytes / 12.4e9` seconds and cannot overlap away beyond the
duplex limit. Compare: VRAM is 29× faster. Keeping hot experts resident in the 11 GB VRAM
budget matters far more here than on a card with a fatter link.

## 6. Storage

| Device | Size | Type | Role |
|---|---|---|---|
| `nvme0n1` | 466 GB | Samsung 960 EVO 500GB (PCIe 3.0 x4) | `/boot` (4 G vfat) + `/` (462 G xfs, 42% used, 268 G free) |
| `sdb`–`sdi` | 8 × 1.8 TB | Seagate ST32000444SS 7.2k SAS | btrfs **RAID10** at `/storage`, 7.3 TB usable, 1.2 TB used |
| `sda` | 30 GB | Transcend TS32GMTS400 SSD | unmounted |

Measured raw sequential read (`hdparm -t --direct`):

- NVMe: **1887 MB/s**
- Single SAS disk: **144 MB/s** (the RAID10 set should aggregate several hundred MB/s)

Model storage today: `~/.cache/huggingface` = 14 GB on the NVMe; `/storage/models` is
empty; `/storage/old` holds 450 GB of unrelated data.

Put weights on the NVMe. Loading a 30 GB checkpoint costs ~16 s from NVMe versus ~3.5 min
from a single spindle.

## 7. Network

| Item | Value |
|---|---|
| `eno1` | Intel I210, **1 Gb/s**, up, `192.168.1.221/24` + public IPv6 |
| `eno2` | Intel I210, down |
| `tailscale0` | `100.96.159.9` |

1 GbE = ~110 MB/s. Downloading a 30 GB checkpoint takes ~5 minutes at line rate.

## 8. OS and toolchain

| Item | Value |
|---|---|
| Distro | CachyOS (Arch rolling) |
| Kernel | 7.1.5-1-cachyos |
| System Python | 3.14.7 |
| CUDA toolkit | **12.9** (`/opt/cuda`, `nvcc` V12.9.86) — not on the default `PATH` |
| cuDNN | 9.10.2.21 (cuda12.9) |
| NCCL | 2.30.7 (cuda12.9) |
| Default compiler | **gcc 16.2.1** |
| Also installed | **gcc-14** (`/usr/bin/gcc-14`) |
| cmake / ninja | 4.4.2 / 1.13.2 |

### Two toolchain notes

1. **nvcc's host-compiler guard is stripped in Arch's package, not satisfied.**
   `/opt/cuda/targets/x86_64-linux/include/crt/host_config.h` line 141 reads
   `#if __GNUC__ > 14` with an **empty body** — upstream has `#error unsupported GNU
   version` there. So gcc 16.2 compiles unguarded rather than supported: a trivial
   `nvcc -arch=sm_61` build with the default host compiler succeeds and runs. Do not rely
   on that. Pin the supported host compiler explicitly, `gcc-14` / `g++-14` are installed:
   `CUDAHOSTCXX=/usr/bin/g++-14`, or `-ccbin g++-14` for a direct nvcc call.
2. **nvcc warns on sm_61**: "Support for offline compilation for architectures prior to
   sm_75 will be removed in a future release." CUDA 12.9 is therefore among the **last**
   toolkits that can target this GPU. Suppress with `-Wno-deprecated-gpu-targets`.

## 9. What already runs here

- `llama-server` from `/opt/buun-llama-cpp`, up 4 days, listening on `127.0.0.1:8080`,
  launched from `~/models` with `--models-preset models.ini`
  (`ct=vbr`, `fit-target=128`, `spec-type=draft-mtp`, `--threads 10 --threads-batch 20`).
- Nothing else competes for the GPU: 5 MiB VRAM used, 0% utilisation, load average 0.00.

Note `--threads 10 --threads-batch 20`: the existing setup already treats one socket as
the working set.

## 10. Why FreeToken does not install here

The repo pins, in `pyproject.toml`:

```
requires = ["setuptools>=77", "torch>=2.11,<2.12", "wheel"]
torch = { index = "pytorch-cu130" }
sglang-kernel = { index = "sglang-cu130" }
```

**CUDA 13.0 removed Maxwell, Pascal and Volta.** PyTorch 2.11's cu130 wheel is built for
`7.5;8.0;8.6;9.0;12.0` — sm_61 is absent, so the wheel cannot run this GPU at all.

From `pytorch/pytorch@v2.11.0:.ci/manywheel/build_cuda.sh`:

| Wheel | `TORCH_CUDA_ARCH_LIST` | Runs on sm_61? |
|---|---|---|
| cu126 | `5.0;6.0;7.0;7.5;8.0;8.6;9.0` | **yes**, via sm_60 cubin (CUDA guarantees binary compatibility upward across minor revisions) |
| cu128 | `7.5;8.0;8.6;9.0;10.0;12.0` | no |
| cu129 | `7.5;8.0;8.6;9.0;10.0;12.0+PTX` | no |
| cu130 | `7.5;8.0;8.6;9.0;10.0;12.0` | no |

`torch 2.11.0+cu126` **does exist** on `download.pytorch.org/whl/cu126`, so the version
range in `pyproject.toml` can be satisfied without loosening it — only the index needs to
change. cu126 also keeps `torch.version.cuda` major = 12, which matches the host's
nvcc 12.9 and so satisfies `_toolchain.check_nvcc_matches_torch()`.

### Known blockers beyond torch itself

| Component | Status on sm_61 |
|---|---|
| `sglang-kernel==0.4.5` (extra `sgl`) | AOT cu130 wheel, sm_75+ — **unavailable**. Optional extra; core falls back to Triton. |
| `flashinfer-python[cu13]` (extra `fi`) | cu13 cubins — **unavailable**. Also optional. |
| **`triton==3.6.0`** | **Works** on sm_61 for elementwise, reductions and fp32/fp16/bf16 `tl.dot`, despite the README claiming CC 8.0+. Fails for int8 `tl.dot` and for any tile over 48 KB shared. See the verified matrix below. |
| `freetoken.kernel._pinned_tensor` | Plain C++ + cudart, no arch gate — should build. |
| `freetoken.kernel._cpu_moe` | Builds; bf16 microkernels take the scalar fallback (no AVX-512, see §3). |
| fp8 paths (`e4m3_compat`, `fp8_pertensor_linear`) | Gated on sm_89+; correctly excluded. |
| TF32 (`fla/utils.py: is_tf32_supported`) | Gated on sm_80+; correctly excluded. |
| Marlin W4A16 / nvfp4 MoE | Gated sm_80-99 and sm_120+; excluded. |

### Verified on the hardware, 2026-08-22

A scratch venv at `~/scratch/agents/claude/monster-spec/tritontest` (Python 3.11,
`torch==2.11.0+cu126`, `triton==3.6.0`) was built and exercised. Results:

```
torch 2.11.0+cu126   cuda 12.6   triton 3.6.0
torch.cuda.is_available()  -> True
torch.cuda.get_arch_list() -> ['sm_50','sm_60','sm_70','sm_75','sm_80','sm_86','sm_90']
device                     -> NVIDIA GeForce GTX 1080 Ti  capability (6, 1)
```

**torch 2.11.0+cu126 runs correctly on this GPU.** The sm_60 cubin executes on sm_61 as
CUDA's minor-revision binary compatibility guarantees. fp32, fp16 and bf16 matmuls all
return correct results against a float64 CPU reference.

**Triton 3.6 also works on sm_61**, despite its README claiming CC 8.0+ :

| Triton construct | Result on sm_61 |
|---|---|
| elementwise (`tl.load`/`tl.store`, masks) | **PASS** |
| reduction (`tl.max`, `tl.sum`, softmax) | **PASS**, matches `torch.softmax` |
| `tl.dot` fp32 | **PASS**, rel err 8.6e-07 |
| `tl.dot` fp16 | **PASS**, rel err 8.3e-07 |
| `tl.dot` bf16 | **PASS**, rel err 5.3e-07 |
| `tl.dot` **int8 -> int32** | **FAIL** — `PassManager::run failed`, MLIR pipeline aborts at `tritongpu-accelerate-matmul` for `target=cuda:61`. No integer MMA below sm_75. |
| tile 128x256x64, `num_stages=3` | **FAIL** — `OutOfResources: shared memory Required: 65536, Hardware limit: 49152` |
| tiles 128x128x32 s3, 64x64x128 s4, 64x128x32 s3 | **PASS** |

Two hard constraints follow.

1. **48 KB shared memory per block, no opt-in.** Pascal's 96 KB/SM cannot be raised past
   the 48 KB per-block default; `cudaFuncAttributeMaxDynamicSharedMemorySize` is sm_70+.
   Every FreeToken tile config tuned for sm_80/sm_90 (routinely 100-228 KB) will raise
   `OutOfResources` here. Tile tables need an sm_61 row, not a code change.
2. **The int8 fast path is unreachable from Triton.** `torch._int_mm` *does* work and is
   bit-exact (verified against an int32 CPU reference), so the 42.6 TOPS dp4a path is
   reachable through cuBLAS only. Any W8A8 work on this box must call `torch._int_mm`, not
   a Triton kernel.

### GEMM throughput, cuBLAS vs Triton, n=4096

| dtype | cuBLAS (TFLOPS) | best Triton `tl.dot` (TFLOPS) | best tile |
|---|---|---|---|
| fp32 | 9.15 | **7.50** (82% of cuBLAS) | 64x64x32, w4 s3 |
| fp16 | 7.68 | **2.16** (28%) | 64x128x32, w8 s3 |
| bf16 | 4.79 | **2.14** (45%) | 64x128x32, w8 s3 |
| int8 | 42.6 TOPS | does not compile | - |

**fp32 is the fast dtype on this GPU, and bf16 is the slowest.** With no tensor cores every
`tl.dot` lowers to fp32 FMA anyway, so 16-bit inputs buy nothing in compute and Triton's
16-bit codegen loses a further 3.5x against its own fp32 path. Since FreeToken routes
almost every GEMM through Triton and modern checkpoints are bf16, the naive path lands at
~2.1 TFLOPS — roughly a fifth of what the card can do.

The ordering to exploit, best to worst: int8 via `torch._int_mm` (42.6) > fp32 cuBLAS
(9.2) > fp32 Triton (7.5) > fp16 cuBLAS (7.7) > bf16 cuBLAS (4.8) > fp16/bf16 Triton (2.1).

### FreeToken installs and imports, 2026-08-22

```
uv pip install --no-build-isolation --index-strategy unsafe-best-match \
    --extra-index-url https://download.pytorch.org/whl/cu126 -e .
```

with `torch==2.11.0+cu126` pre-installed and `CUDA_HOME=/opt/cuda`. Result: **freetoken
0.1.2 built and installed**. Both C++ extensions compiled, and every core module imports
on sm_61:

| module | |
|---|---|
| `freetoken.kernel._pinned_tensor` | OK |
| `freetoken.kernel._cpu_moe` | OK |
| `flashlib` 0.3.0 / `tvm_ffi` (the `slot_cache` LRU) | OK — imports |
| `freetoken.engine.engine`, `.kernel.triton`, `.moe.offload_cache`, `.layers.moe`, `.attention.fa` | OK |
| `sgl_kernel`, `flashinfer` | absent, as expected (optional extras, no sm_61 wheel) |

`uv pip install` does **not** apply the `[tool.uv.sources]` cu130 pin, so no pyproject
edit is needed to install here — only an explicit torch pin and extra index.

## 10a. Test suite on sm_61

`pytest tests/ -m "not slow" --timeout=180` -> **1270 passed, 66 failed, 27 skipped**
in 6m33s. Every failure traces to one of five causes, and none is a packaging problem.

| n | root cause | where |
|---|---|---|
| 33 | **Triton tile exceeds 48 KB shared memory** | `test_minimax_m3_sparse`, `test_mxfp8_linear`, `test_triton_attention`, `test_glm_dsa`, `test_gpt_oss` |
| 13 | **ptxas: `.acq_rel` requires sm_70** | `test_cpu_moe_q4_0`, `test_fused_moe`, `test_dsfp4_grouped_prefill` |
| 4 | **ptxas: `tanh` requires sm_75** | GELU-tanh activation paths |
| 5 | **JIT nvcc: `__grid_constant__` requires compute_70** | `test_offload`, `test_gpt_oss`, `test_bsa_pool`, `test_pinned_tensor` |
| 2 | flashinfer not installed | `test_cache_budget` — environment, not hardware |
| 2 | `cudaHostGetDevicePointer` on pageable memory returns `invalid argument` | `test_pinned_tensor`, `test_triton_attention` |

### The four real defects, located

1. **`tanh.approx.f32` inline PTX** — `python/freetoken/kernel/triton/activation.py:50`.
   `_fast_tanh` emits the instruction directly via `tl.inline_asm_elementwise`; it is
   sm_75+. Only `ACT == 2` (GELU_TANH, line 99) uses it. The same file already gates PDL
   on sm_90+ at line 43, so the guard pattern is established — this wants a computed-tanh
   fallback under sm_75.

2. **`.acq_rel` atomics** — Triton's `tl.atomic_add` defaults to `sem="acq_rel"`, which
   lowers to `atom.acq_rel.gpu`, sm_70+. Hit from `kernel/triton/moe_align.py:104`,
   `kernel/triton/nvfp4_linear.py:305`, and `kernel/triton/sampling.py`. Note
   `nvfp4_linear.py:251` documents that it *relies* on acq_rel to publish partial stores,
   so that site needs a real fence on Pascal, not a silent downgrade to `relaxed`.

3. **`__grid_constant__`** — `kernel/csrc/jit/fast_index_copy.cuh:488`,
   `kernel/csrc/jit/index.cu:34` and `:65`, `kernel/csrc/jit/store.cu:28`. compute_70+.
   Verified in isolation: forcing `-ccbin g++-14` leaves this as the *only* error, so the
   host compiler is not implicated. `__nanosleep` in the same header
   (`fast_index_copy.cuh:79`) is **already** correctly guarded with
   `#if __CUDA_ARCH__ >= 700` — the precedent for the fix is in the same file.

4. **48 KB shared-memory ceiling** — the largest bucket, and not a code defect: tile
   tables tuned for sm_80/sm_90 simply do not fit. Needs an sm_61 row, not new logic.

None of these blocks the others: they are four independent, small, well-localised changes.

## 11. Standing recommendations for this host

- Build CUDA with `-ccbin g++-14 -arch=sm_61 -Wno-deprecated-gpu-targets`. gcc 16 is the
  default and is not blocked, but it is outside CUDA 12.9's supported host range.
- Pin the serving process to NUMA node 0: `numactl --cpunodebind=0 --membind=0`.
- Prefer int8 through `torch._int_mm` (42.6 TOPS) over everything else. Failing that,
  fp32 compute: it is the fastest dtype here, in cuBLAS and in Triton alike.
- Never leave bf16 as the compute dtype. Upcast bf16 checkpoints once at load; a bf16
  Triton `tl.dot` runs at 2.1 TFLOPS against fp32's 7.5.
- Cap Triton tiles at 48 KB shared memory per block. Pascal cannot opt in past it.
- Keep weights on the NVMe (`/` or `~/.cache/huggingface`), never on `/storage`.
- Budget PCIe traffic against 12.4 GB/s, not against VRAM's 359 GB/s.
- `nvidia-smi -pm 1` and `nvidia-smi -pl 300` are available if benchmark variance from
  clock ramping becomes a problem.
- Cheap hardware wins, if ever wanted: fill the 4 empty DIMM slots (≈2× memory
  bandwidth); one free Gen3 x16 slot remains for a second GPU.
