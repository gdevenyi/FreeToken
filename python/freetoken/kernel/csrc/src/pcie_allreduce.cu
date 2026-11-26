// Two-tier Pinned-Sysmem TP=2 AllReduce for PCIe PHB/SYS topologies (e.g. 2x RTX 6000 Ada sm_89).
//
// Tier 1 (< 64 KiB, all decode & small-verify shapes):
//   1-barrier in-kernel SM write-back TX (st.global.wb.v4.u32) into double-buffered
//   mapped host staging ([cap_bytes, 2*cap_bytes) and [2*cap_bytes, 3*cap_bytes)),
//   peer-only system-scope release/acquire barrier (st.release.sys.global.u32 /
//   ld.acquire.sys.global.u32), and vectorized FP32/INT32 reduction.
// Tier 2 (64 KiB .. cap_bytes, medium prefill/verify shapes):
//   Copy-Engine async D2H DMA TX (cudaMemcpyAsync into [0, cap_bytes)) + 2-barrier SM RX.
//
// Fully CUDA-Graph capture-safe: monotonic epoch and bank-parity state live in the
// shared mapped Signal structures, with zero host synchronization or dynamic queries.

#include <freetoken/tensor.h>
#include <freetoken/utils.cuh>
#include <freetoken/utils.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>

#include <cstddef>
#include <cstdint>

namespace {

constexpr int kMaxBlocks = 144;
constexpr int kTp2PcieBlockCap = 16;
constexpr int kRanks = 16;
constexpr size_t kDmaThresholdBytes = 65536;
constexpr size_t kEvenStageEpochSlot = 1;
constexpr size_t kOddStageEpochSlot = 2;
constexpr int kThreadsPerBlock = 512;
constexpr size_t kBytesPerBlock = static_cast<size_t>(kThreadsPerBlock) * 16ULL;

struct Signal {
  alignas(128) uint32_t start[kMaxBlocks][kRanks];
  alignas(128) uint32_t end[kMaxBlocks][kRanks];
  alignas(128) uint32_t flag[kMaxBlocks];
};

static_assert(sizeof(Signal) == 19072, "Signal ABI size mismatch");
static_assert(offsetof(Signal, end) + 128 * kRanks * sizeof(uint32_t) == 17408,
              "Signal::end[128][0] offset mismatch");

__device__ __forceinline__ void st_global_wb_v4(uint4 *addr, uint4 val) {
  asm volatile("st.global.wb.v4.u32 [%0], {%1, %2, %3, %4};"
               :
               : "l"(addr), "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w)
               : "memory");
}

__device__ __forceinline__ uint4 ld_global_cv_v4(const uint4 *addr) {
  uint4 val;
  asm volatile("ld.global.cv.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(val.x), "=r"(val.y), "=r"(val.z), "=r"(val.w)
               : "l"(addr)
               : "memory");
  return val;
}

template <typename T>
__device__ __forceinline__ T ld_peer_elem(const T *addr) {
  if constexpr (sizeof(T) == 2) {
    uint16_t raw;
    asm volatile("ld.global.cv.u16 %0, [%1];"
                 : "=h"(raw)
                 : "l"(addr)
                 : "memory");
    return *reinterpret_cast<const T *>(&raw);
  } else {
    uint32_t raw;
    asm volatile("ld.global.cv.u32 %0, [%1];"
                 : "=r"(raw)
                 : "l"(addr)
                 : "memory");
    return *reinterpret_cast<const T *>(&raw);
  }
}

template <typename T>
__device__ __forceinline__ void st_stage_elem(T *addr, T val) {
  if constexpr (sizeof(T) == 2) {
    uint16_t raw = *reinterpret_cast<const uint16_t *>(&val);
    asm volatile("st.global.wb.u16 [%0], %1;"
                 :
                 : "l"(addr), "h"(raw)
                 : "memory");
  } else {
    uint32_t raw = *reinterpret_cast<const uint32_t *>(&val);
    asm volatile("st.global.wb.u32 [%0], %1;"
                 :
                 : "l"(addr), "r"(raw)
                 : "memory");
  }
}

__device__ __forceinline__ void st_flag_release(uint32_t *addr, uint32_t val) {
  asm volatile("st.release.sys.global.u32 [%1], %0;"
               :
               : "r"(val), "l"(addr)
               : "memory");
}

__device__ __forceinline__ uint32_t ld_flag_acquire(uint32_t *addr) {
  uint32_t val;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];"
               : "=r"(val)
               : "l"(addr)
               : "memory");
  return val;
}

template <typename T>
__device__ __forceinline__ T *stage_epoch_ptr(T *base, const Signal *sig,
                                              bool odd, size_t fallback_elems) {
  const size_t cap_bytes = static_cast<size_t>(sig->end[128][0]) & ~15ULL;
  const size_t min_bytes = fallback_elems * sizeof(T);
  if (cap_bytes >= min_bytes && cap_bytes > 0) {
    const size_t off_bytes =
        cap_bytes * (odd ? kOddStageEpochSlot : kEvenStageEpochSlot);
    return reinterpret_cast<T *>(reinterpret_cast<char *>(base) + off_bytes);
  }
  if (!odd) {
    return base;
  }
  const size_t aligned_min_bytes = (min_bytes + 15ULL) & ~15ULL;
  return reinterpret_cast<T *>(reinterpret_cast<char *>(base) +
                               aligned_min_bytes);
}

__device__ __forceinline__ bool next_comm_bank_odd(Signal *self_sg,
                                                   bool advance_bank) {
  if (!advance_bank) {
    return false;
  }
  uint32_t *bank_flags = &self_sg->end[129][0];
  const uint32_t bp = bank_flags[blockIdx.x] + 1u;
  __syncthreads();
  if (threadIdx.x == 0) {
    bank_flags[blockIdx.x] = bp;
  }
  if (blockIdx.x == 0) {
    for (int b = static_cast<int>(gridDim.x) + static_cast<int>(threadIdx.x);
         b < kMaxBlocks; b += static_cast<int>(blockDim.x)) {
      bank_flags[b] = bp;
    }
  }
  return (bp & 1u) != 0u;
}

__device__ __forceinline__ void barrier_epoch_release(Signal *sig0,
                                                      Signal *sig1,
                                                      Signal *self_sg,
                                                      int rank) {
  __syncthreads();
  const uint32_t epoch = self_sg->flag[blockIdx.x] + 1u;
  if (threadIdx.x == 0) {
    // Write only to peer's Signal struct and poll our own slot written by peer
    // to avoid dirtying the same 128B host cache line from both root complexes.
    Signal *peer_sg = (rank == 0) ? sig1 : sig0;
    uint32_t *peer_counter = &peer_sg->start[blockIdx.x][rank];
    uint32_t *self_counter = &self_sg->start[blockIdx.x][1 - rank];
    st_flag_release(peer_counter, epoch);
    uint32_t spins = 0;
    while (static_cast<int32_t>(ld_flag_acquire(self_counter) - epoch) < 0) {
      if (++spins > 32u) {
        __nanosleep(64);
      }
    }
    self_sg->flag[blockIdx.x] = epoch;
  }
  __syncthreads();
}

template <typename T> struct ElemTraits;

template <> struct ElemTraits<__nv_bfloat16> {
  using Acc = float;
  static constexpr int kVec = 8;
  __device__ __forceinline__ static float to_acc(__nv_bfloat16 v) {
    return __bfloat162float(v);
  }
  __device__ __forceinline__ static __nv_bfloat16 from_acc(float v) {
    return __float2bfloat16(v);
  }
};

template <> struct ElemTraits<__half> {
  using Acc = float;
  static constexpr int kVec = 8;
  __device__ __forceinline__ static float to_acc(__half v) {
    return __half2float(v);
  }
  __device__ __forceinline__ static __half from_acc(float v) {
    return __float2half(v);
  }
};

template <> struct ElemTraits<float> {
  using Acc = float;
  static constexpr int kVec = 4;
  __device__ __forceinline__ static float to_acc(float v) { return v; }
  __device__ __forceinline__ static float from_acc(float v) { return v; }
};

template <> struct ElemTraits<int32_t> {
  using Acc = int32_t;
  static constexpr int kVec = 4;
  __device__ __forceinline__ static int32_t to_acc(int32_t v) { return v; }
  __device__ __forceinline__ static int32_t from_acc(int32_t v) { return v; }
};

template <typename T>
__global__ void __launch_bounds__(kThreadsPerBlock, 1)
    tp2_pcie_all_reduce_kernel(const T *local_in, T *out, T *s0, T *s1,
                               Signal *__restrict__ sig0,
                               Signal *__restrict__ sig1, int rank,
                               size_t num_elems) {
  using Traits = ElemTraits<T>;
  using Acc = typename Traits::Acc;
  constexpr int kVec = Traits::kVec;

  Signal *self_sg = (rank == 0) ? sig0 : sig1;
  const bool dma_tx = (num_elems * sizeof(T)) >= kDmaThresholdBytes;
  const bool odd = next_comm_bank_odd(self_sg, !dma_tx);

  T *s0_buf = dma_tx ? s0 : stage_epoch_ptr(s0, self_sg, odd, num_elems);
  T *s1_buf = dma_tx ? s1 : stage_epoch_ptr(s1, self_sg, odd, num_elems);
  T *my_s = (rank == 0) ? s0_buf : s1_buf;
  const T *peer_s = (rank == 0) ? s1_buf : s0_buf;

  const bool vec_ok =
      ((num_elems & static_cast<size_t>(kVec - 1)) == 0) &&
      (((reinterpret_cast<uintptr_t>(local_in) |
         reinterpret_cast<uintptr_t>(out) | reinterpret_cast<uintptr_t>(my_s) |
         reinterpret_cast<uintptr_t>(peer_s)) &
        15u) == 0);

  const size_t tid =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;

  if (!dma_tx) {
    if (vec_ok) {
      const size_t num_vec = num_elems / static_cast<size_t>(kVec);
      const uint4 *local_v = reinterpret_cast<const uint4 *>(local_in);
      uint4 *my_v = reinterpret_cast<uint4 *>(my_s);
      for (size_t i = tid; i < num_vec; i += stride) {
        st_global_wb_v4(my_v + i, local_v[i]);
      }
    } else {
      for (size_t idx = tid; idx < num_elems; idx += stride) {
        st_stage_elem(my_s + idx, local_in[idx]);
      }
    }
  }
  __threadfence_system();

  barrier_epoch_release(sig0, sig1, self_sg, rank);

  if (vec_ok) {
    const size_t num_vec = num_elems / static_cast<size_t>(kVec);
    const uint4 *local_v = reinterpret_cast<const uint4 *>(local_in);
    const uint4 *peer_v = reinterpret_cast<const uint4 *>(peer_s);
    uint4 *out_v = reinterpret_cast<uint4 *>(out);
    for (size_t i = tid; i < num_vec; i += stride) {
      const uint4 lv = local_v[i];
      const uint4 pv = ld_global_cv_v4(peer_v + i);
      const T *lt = reinterpret_cast<const T *>(&lv);
      const T *pt = reinterpret_cast<const T *>(&pv);
      uint4 ov;
      T *ot = reinterpret_cast<T *>(&ov);
#pragma unroll
      for (int d = 0; d < kVec; ++d) {
        const Acc a_loc = Traits::to_acc(lt[d]);
        const Acc a_peer = Traits::to_acc(pt[d]);
        ot[d] = Traits::from_acc((rank == 0) ? (a_loc + a_peer)
                                             : (a_peer + a_loc));
      }
      out_v[i] = ov;
    }
  } else {
    for (size_t idx = tid; idx < num_elems; idx += stride) {
      const Acc a_loc = Traits::to_acc(local_in[idx]);
      const Acc a_peer = Traits::to_acc(ld_peer_elem(peer_s + idx));
      out[idx] = Traits::from_acc((rank == 0) ? (a_loc + a_peer)
                                              : (a_peer + a_loc));
    }
  }

  if (dma_tx) {
    barrier_epoch_release(sig0, sig1, self_sg, rank);
  }
}

struct PcieAllReduceWrapper : public tvm::ffi::Object {
public:
  PcieAllReduceWrapper(int rank, int64_t sig0_ptr, int64_t sig1_ptr,
                       int64_t stage0_ptr, int64_t stage1_ptr,
                       int64_t cap_bytes, int block_cap)
      : m_rank(rank),
        m_sig0(reinterpret_cast<Signal *>(static_cast<uintptr_t>(sig0_ptr))),
        m_sig1(reinterpret_cast<Signal *>(static_cast<uintptr_t>(sig1_ptr))),
        m_stage0(reinterpret_cast<void *>(static_cast<uintptr_t>(stage0_ptr))),
        m_stage1(reinterpret_cast<void *>(static_cast<uintptr_t>(stage1_ptr))),
        m_cap_bytes(static_cast<size_t>(cap_bytes)),
        m_block_cap(block_cap < 1
                        ? 1
                        : (block_cap > kTp2PcieBlockCap ? kTp2PcieBlockCap
                                                        : block_cap)) {
    host::RuntimeCheck(rank == 0 || rank == 1,
                       "PcieAllReduceWrapper requires TP=2 rank in {0, 1}");
    host::RuntimeCheck(m_sig0 != nullptr && m_sig1 != nullptr &&
                           m_stage0 != nullptr && m_stage1 != nullptr,
                       "PcieAllReduceWrapper pointers must be non-null");
    host::RuntimeCheck(m_cap_bytes >= kDmaThresholdBytes &&
                           (m_cap_bytes & 15ULL) == 0,
                       "PcieAllReduceWrapper cap_bytes must be >= 64 KiB and "
                       "16-byte aligned");
  }

  auto all_reduce(tvm::ffi::TensorView t) const -> void {
    using namespace host;
    RuntimeCheck(t.device().device_type == kDLCUDA,
                 "Tensor must be on CUDA device");
    RuntimeCheck(t.is_contiguous(), "Tensor must be contiguous");
    const auto num_elems = static_cast<size_t>(t.shape().Product());
    if (num_elems == 0) {
      return;
    }
    const auto dtype = t.dtype();
    RuntimeCheck(dtype.lanes == 1, "Vector lanes != 1 unsupported");
    const size_t elem_bytes = static_cast<size_t>(dtype.bits / 8);
    const size_t size_bytes = num_elems * elem_bytes;
    RuntimeCheck(size_bytes <= m_cap_bytes,
                 "Tensor exceeds PcieAllReduceWrapper staging capacity");

    void *data_ptr = t.data_ptr();
    const auto stream = LaunchKernel::resolve_device(t.device());

    if (size_bytes >= kDmaThresholdBytes) {
      void *my_stage = (m_rank == 0) ? m_stage0 : m_stage1;
      CUDA_CHECK(::cudaMemcpyAsync(my_stage, data_ptr, size_bytes,
                                   ::cudaMemcpyDefault, stream));
    }

    size_t raw_blocks = (size_bytes + kBytesPerBlock - 1ULL) / kBytesPerBlock;
    if (raw_blocks < 1ULL) {
      raw_blocks = 1ULL;
    }
    if (raw_blocks > static_cast<size_t>(m_block_cap)) {
      raw_blocks = static_cast<size_t>(m_block_cap);
    }
    const dim3 grid(static_cast<unsigned int>(raw_blocks));
    const dim3 block(static_cast<unsigned int>(kThreadsPerBlock));

    if (dtype.code == DLDataTypeCode::kDLBfloat && dtype.bits == 16) {
      tp2_pcie_all_reduce_kernel<__nv_bfloat16><<<grid, block, 0, stream>>>(
          static_cast<const __nv_bfloat16 *>(data_ptr),
          static_cast<__nv_bfloat16 *>(data_ptr),
          static_cast<__nv_bfloat16 *>(m_stage0),
          static_cast<__nv_bfloat16 *>(m_stage1), m_sig0, m_sig1, m_rank,
          num_elems);
    } else if (dtype.code == DLDataTypeCode::kDLFloat && dtype.bits == 16) {
      tp2_pcie_all_reduce_kernel<__half><<<grid, block, 0, stream>>>(
          static_cast<const __half *>(data_ptr),
          static_cast<__half *>(data_ptr), static_cast<__half *>(m_stage0),
          static_cast<__half *>(m_stage1), m_sig0, m_sig1, m_rank, num_elems);
    } else if (dtype.code == DLDataTypeCode::kDLFloat && dtype.bits == 32) {
      tp2_pcie_all_reduce_kernel<float><<<grid, block, 0, stream>>>(
          static_cast<const float *>(data_ptr), static_cast<float *>(data_ptr),
          static_cast<float *>(m_stage0), static_cast<float *>(m_stage1),
          m_sig0, m_sig1, m_rank, num_elems);
    } else if (dtype.code == DLDataTypeCode::kDLInt && dtype.bits == 32) {
      tp2_pcie_all_reduce_kernel<int32_t><<<grid, block, 0, stream>>>(
          static_cast<const int32_t *>(data_ptr),
          static_cast<int32_t *>(data_ptr), static_cast<int32_t *>(m_stage0),
          static_cast<int32_t *>(m_stage1), m_sig0, m_sig1, m_rank, num_elems);
    } else {
      RuntimeCheck(false, "Unsupported dtype for PcieAllReduceWrapper");
    }
    CUDA_CHECK(::cudaGetLastError());
  }

  auto max_bytes() const -> int64_t {
    return static_cast<int64_t>(m_cap_bytes);
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("freetoken.PcieAllReduceWrapper",
                                    PcieAllReduceWrapper, tvm::ffi::Object);

private:
  int m_rank;
  Signal *m_sig0;
  Signal *m_sig1;
  void *m_stage0;
  void *m_stage1;
  size_t m_cap_bytes;
  int m_block_cap;
};

auto pcie_signal_bytes() -> int64_t {
  return static_cast<int64_t>(sizeof(Signal));
}

auto pcie_warmup_kernels() -> void {
  cudaFuncAttributes attr{};
  host::CUDA_CHECK(::cudaFuncGetAttributes(
      &attr, tp2_pcie_all_reduce_kernel<__nv_bfloat16>));
  host::CUDA_CHECK(
      ::cudaFuncGetAttributes(&attr, tp2_pcie_all_reduce_kernel<__half>));
  host::CUDA_CHECK(
      ::cudaFuncGetAttributes(&attr, tp2_pcie_all_reduce_kernel<float>));
  host::CUDA_CHECK(
      ::cudaFuncGetAttributes(&attr, tp2_pcie_all_reduce_kernel<int32_t>));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<PcieAllReduceWrapper>()
      .def(refl::init<int, int64_t, int64_t, int64_t, int64_t, int64_t, int>(),
           "__init__")
      .def("all_reduce", &PcieAllReduceWrapper::all_reduce)
      .def("max_bytes", &PcieAllReduceWrapper::max_bytes);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(pcie_signal_bytes, &pcie_signal_bytes);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(pcie_warmup_kernels, &pcie_warmup_kernels);

} // namespace
