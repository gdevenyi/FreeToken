#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <cuda_runtime_api.h>

#include "gpu_sync.h"

namespace freetoken::cpu_moe {
namespace {

// =====================================================================================
// CUDA stream memory operations (driver API, resolved via dlopen -- no link-time or
// toolchain dependence). The GPU side of the flag handshake: submit = WRITE_VALUE
// (done=0 then ready=task word), sync = WAIT_VALUE(done >= 1). The wait is
// executed by the GPU front-end (no SM-resident kernel), so GPU "utilization" stays
// truthful during CPU compute windows -- a resident spin kernel pinned it at 99%,
// which laptop CPU/GPU dynamic power schedulers answered by clamping the CPU's max
// frequency (GEMV workers -1.5x: the reported edge regression). Availability is
// probed functionally at startup (memops_probe); anything unsupported (some driver models,
// vGPU, old drivers) falls back to the cudaLaunchHostFunc path.
#if defined(_WIN32)
static void* cumemop_dlopen() { return (void*)::LoadLibraryA("nvcuda.dll"); }
static void* cumemop_dlsym(void* h, const char* n) {
  return (void*)::GetProcAddress((HMODULE)h, n);
}
#else
static void* cumemop_dlopen() {
  void* h = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_LOCAL);
  if (h == nullptr) h = dlopen("libcuda.so", RTLD_LAZY | RTLD_LOCAL);
  return h;
}
static void* cumemop_dlsym(void* h, const char* n) { return dlsym(h, n); }
#endif

using cuMemOp64_fn = int (*)(void* stream, unsigned long long addr, unsigned long long value,
                             unsigned int flags);
using cuMemOp32_fn = int (*)(void* stream, unsigned long long addr, unsigned int value,
                             unsigned int flags);
static cuMemOp64_fn g_cu_write64 = nullptr;
static cuMemOp64_fn g_cu_wait64 = nullptr;
static cuMemOp32_fn g_cu_write32 = nullptr;
static cuMemOp32_fn g_cu_wait32 = nullptr;
// Width the probe settled on. Pre-Volta devices (CAN_USE_64_BIT_STREAM_MEM_OPS = 0) take only
// the 32-bit ops, which act on the low word of each little-endian int64 flag.
static int g_memop_bits = 64;
static constexpr unsigned int kCuWaitValueGeq = 0x0;   // CU_STREAM_WAIT_VALUE_GEQ
static constexpr unsigned int kCuWriteDefault = 0x0;   // CU_STREAM_WRITE_VALUE_DEFAULT

static bool cumemop_resolve() {
  static bool resolved = [] {
    void* h = cumemop_dlopen();
    if (h == nullptr) return false;
    // 11.7+ made the v2 entry points the default; older drivers export only the v1
    // names with the same signature.
    auto sym = [h](const char* v2, const char* v1) {
      void* f = cumemop_dlsym(h, v2);
      return f != nullptr ? f : cumemop_dlsym(h, v1);
    };
    g_cu_write64 = reinterpret_cast<cuMemOp64_fn>(sym("cuStreamWriteValue64_v2", "cuStreamWriteValue64"));
    g_cu_wait64 = reinterpret_cast<cuMemOp64_fn>(sym("cuStreamWaitValue64_v2", "cuStreamWaitValue64"));
    g_cu_write32 = reinterpret_cast<cuMemOp32_fn>(sym("cuStreamWriteValue32_v2", "cuStreamWriteValue32"));
    g_cu_wait32 = reinterpret_cast<cuMemOp32_fn>(sym("cuStreamWaitValue32_v2", "cuStreamWaitValue32"));
    return (g_cu_write64 != nullptr && g_cu_wait64 != nullptr) ||
           (g_cu_write32 != nullptr && g_cu_wait32 != nullptr);
  }();
  return resolved;
}

static bool has_memops(int bits) {
  return bits == 64 ? g_cu_write64 != nullptr && g_cu_wait64 != nullptr
                    : g_cu_write32 != nullptr && g_cu_wait32 != nullptr;
}

// value fits 32 bits whenever bits == 32 (the callers see to it)
static int cumemop_write(void* s, unsigned long long addr, uint64_t value, int bits) {
  if (bits == 32) return g_cu_write32(s, addr, static_cast<unsigned int>(value), kCuWriteDefault);
  return g_cu_write64(s, addr, value, kCuWriteDefault);
}

static int cumemop_wait_geq(void* s, unsigned long long addr, uint64_t value, int bits) {
  if (bits == 32) return g_cu_wait32(s, addr, static_cast<unsigned int>(value), kCuWaitValueGeq);
  return g_cu_wait64(s, addr, value, kCuWaitValueGeq);
}

// GPU side of the flag handshake (see the block comment above): enqueued on the
// caller's (possibly capturing) stream; the WAIT immediate is the constant 1,
// replay-safe under CUDA graphs.
// The startup probe validates EAGER memops; a driver could still reject them at graph
// capture time. Those enqueue errors would otherwise be swallowed here and surface only
// as a later EndCapture failure -- log the first CUresult so triage is one step.
static void cumemop_check(int rc, const char* what) {
  static std::atomic<bool> warned{false};
  if (rc != 0 && !warned.exchange(true)) {
    std::fprintf(stderr,
                 "[freetoken/cpu_moe] %s failed with CUresult=%d (first occurrence; "
                 "subsequent errors are not repeated). If this happened during CUDA "
                 "graph capture, the driver lacks capture support for stream memops -- "
                 "set FREETOKEN_CPU_MOE_FLAG_SYNC=0.\n",
                 what, rc);
  }
}

}  // namespace

// ready and done live on separate cache lines of one mapped-pinned block; with UVA the GPU memops and the coordinator use the same addresses.
struct GpuSync::Impl {
  void* block = nullptr;
  void* device_block = nullptr;
  volatile int64_t* ready = nullptr;
  volatile int64_t* done = nullptr;
  // the width the last probe chose, latched so a later probe cannot change how this instance encodes ready
  const int bits = g_memop_bits;
  // 32-bit memops cannot write a whole task word, so ready carries its index + 1 here.
  // Words are registered at submit (capture) time and never removed; a graph replays its index.
  static constexpr size_t kMaxWords = 1 << 16;
  std::mutex words_mtx;
  std::unordered_map<uint64_t, uint32_t> index_of;
  std::unique_ptr<std::atomic<uint64_t>[]> words{new std::atomic<uint64_t>[kMaxWords]()};

  uint64_t doorbell(uint64_t task) {
    if (bits == 64) return task;
    std::lock_guard<std::mutex> lk(words_mtx);
    const auto it = index_of.find(task);
    if (it != index_of.end()) return it->second + 1;
    const size_t i = index_of.size();
    if (i == kMaxWords) throw std::runtime_error("CPU MoE flag sync: too many distinct tasks for 32-bit memops");
    words[i].store(task, std::memory_order_release);
    index_of.emplace(task, static_cast<uint32_t>(i));
    return i + 1;
  }

  uint64_t task_of(uint64_t bell) const {
    if (bits == 64) return bell;
    return bell >= 1 && bell <= kMaxWords ? words[bell - 1].load(std::memory_order_acquire) : 0;
  }
};

GpuSync::GpuSync() : impl_(std::make_unique<Impl>()) {
  if (!cumemop_resolve()) throw std::runtime_error("CPU MoE flag sync: CUDA stream memops are unavailable");
  const cudaError_t rc = cudaHostAlloc(&impl_->block, 128, cudaHostAllocPortable | cudaHostAllocMapped);
  if (rc != cudaSuccess)
    throw std::runtime_error(std::string("CPU MoE flag allocation failed: ") + cudaGetErrorString(rc));
  std::memset(impl_->block, 0, 128);
  impl_->ready = static_cast<volatile int64_t*>(impl_->block);
  impl_->done = impl_->ready + 8;
  const cudaError_t dev_rc = cudaHostGetDevicePointer(&impl_->device_block, impl_->block, 0);
  if (dev_rc != cudaSuccess) {
    cudaFreeHost(impl_->block);
    throw std::runtime_error(std::string("CPU MoE flag device address: ") + cudaGetErrorString(dev_rc));
  }
}

GpuSync::~GpuSync() { cudaFreeHost(impl_->block); }

// Functional probe on a scratch pinned int64: enqueue WRITE(7) + WAIT(>=7) + sync, 64-bit
// ops first, then 32-bit. Returns true only if the whole memop path works on THIS
// stream/device/driver; a rejected enqueue (CUDA_ERROR_NOT_SUPPORTED) leaves the stream usable.
bool GpuSync::probe(uintptr_t stream) {
  if (!cumemop_resolve()) return false;
  void* scratch = nullptr;
  if (cudaHostAlloc(&scratch, sizeof(int64_t), cudaHostAllocPortable | cudaHostAllocMapped) != cudaSuccess)
    return false;
  auto* s = reinterpret_cast<void*>(stream);
  const auto addr = reinterpret_cast<unsigned long long>(scratch);
  bool ok = false;
  for (int bits : {64, 32}) {
    if (!has_memops(bits)) continue;
    *static_cast<volatile int64_t*>(scratch) = 0;
    ok = cumemop_write(s, addr, 7, bits) == 0 && cumemop_wait_geq(s, addr, 7, bits) == 0 &&
         cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream)) == cudaSuccess &&
         *static_cast<volatile int64_t*>(scratch) == 7;
    // a failed enqueue can leave the WRITE pending on the scratch word
    cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream));
    if (ok) {
      g_memop_bits = bits;
      break;
    }
  }
  cudaFreeHost(scratch);
  return ok;
}

int GpuSync::memop_bits() { return g_memop_bits; }

bool GpuSync::host_func_capture_safe() { return true; }

void GpuSync::gpu_submit(uintptr_t stream, uint64_t task) {
  auto* s = reinterpret_cast<void*>(stream);
  // Order matters and is preserved by the front end: reset done BEFORE raising ready,
  // so the coordinator's completion write for THIS step can never be wiped.
  const uint64_t bell = impl_->doorbell(task);
  cumemop_check(cumemop_write(s, reinterpret_cast<unsigned long long>(impl_->done), 0, impl_->bits), "cuStreamWriteValue(done)");
  cumemop_check(cumemop_write(s, reinterpret_cast<unsigned long long>(impl_->ready), bell, impl_->bits),
                "cuStreamWriteValue(ready)");
}

void GpuSync::gpu_wait(uintptr_t stream) {
  cumemop_check(cumemop_wait_geq(reinterpret_cast<void*>(stream), reinterpret_cast<unsigned long long>(impl_->done), 1,
                                 impl_->bits),
                "cuStreamWaitValue(done)");
}

uint64_t GpuSync::cpu_poll() {
  // Acquire: everything the GPU made visible before writing ready -- the D2H input copies -- is visible to the worker pool after this read.
  const int64_t task = flag_load_acquire(impl_->ready);
  if (task == 0) return 0;
  flag_store_release(impl_->ready, 0);  // consume this step's doorbell
  return impl_->task_of(static_cast<uint64_t>(task));
}

uint64_t GpuSync::doorbell(uint64_t task) { return impl_->doorbell(task); }

void GpuSync::cpu_done() { flag_store_release(impl_->done, 1); }

bool GpuSync::pending() const { return flag_load_acquire(impl_->ready) != 0 && flag_load_acquire(impl_->done) == 0; }

uintptr_t GpuSync::ready_word() const { return reinterpret_cast<uintptr_t>(impl_->device_block); }

uintptr_t GpuSync::done_word() const { return reinterpret_cast<uintptr_t>(impl_->device_block) + 8 * sizeof(int64_t); }

}  // namespace freetoken::cpu_moe
