---
title: "Allocators & caching"
summary: >-
  Explains how device memory is allocated behind get_allocator() and handed to
  StorageView. The Allocator interface (allocator.h) is deliberately pointer-
  based and typeless (allocate returns a raw void*, free, and an optional
  clear_cache), which is exactly what unified memory exploits. The runtime
  get_allocator(Device) dispatches over per-device template specializations
  except Device::METAL, which early-returns before DEVICE_DISPATCH to avoid
  instantiating primitives<Device::METAL>. CPU uses a 64-byte-aligned
  posix_memalign allocator (or an MKL variant); CUDA picks between cub
  CachingDeviceAllocator (bin growth 4/min 3/max 12, 200MB cap, one per
  thread) and cudaMallocAsync via the CT2_CUDA_ALLOCATOR env var. StorageView
  owns a buffer iff it allocated it (owns_data ⇔ _allocator != nullptr), the
  allocator never tracks lifetimes, and ReplicaPool::clear_cache walks
  workers. The Metal allocator returns the contents pointer of a Shared
  MTLBuffer with an address-ordered side table mapping pointers back to
  buffers, no ARC and no clear_cache override.
semantic_id: "yx4w9olvk23Tk6VdwtkrHQcQpWuzyasCShNwGpC1Ys4LKX0GZ-9kGo90ek1V764LgA8suu5bzY5-Nae_Tuxv4Q"
---

How device memory is allocated, cached, and handed to `StorageView`.

**Sources (all citations from real lines):**

- `include/ctranslate2/allocator.h` (the abstraction)
- `src/allocator.cc` (the runtime registry)
- `src/cpu/allocator.cc` (CPU aligned / MKL allocators)
- `src/cuda/allocator.cc` (cub caching + async CUDA allocators)

The resize/reserve contract on the StorageView side is owned by `storageview.md`; this file covers what's behind `get_allocator()`.

---

## 1. The `Allocator` interface

`allocator.h:7-22` — three virtuals and two convenience overloads:

```cpp
class Allocator {
  virtual void* allocate(size_t size, int device_index) = 0;
  virtual void free(void* ptr, int device_index) = 0;
  virtual void clear_cache() {};          // default no-op
  void* allocate(size_t size) { return allocate(size, -1); }   // -1 = current device
  void free(void* ptr)        { free(ptr, -1); }
};
```

The contract is **pointer-based and typeless**: `allocate` returns a raw `void*` that `StorageView` stores in `_data`. There is no handle/object type — any backend whose memory can be addressed through a plain pointer fits (this is exactly what unified memory exploits, see the Metal bridge below).

## 2. The registry: `get_allocator`

Two forms (`allocator.h:24-26`): a compile-time `get_allocator<Device D>()` template (each backend file defines its specialization) and a runtime `get_allocator(Device)`.

The runtime form (`src/allocator.cc:7-20`) is a thin `DEVICE_DISPATCH` over the template — with one exception: **`Device::METAL` early-returns before the dispatch macro** (`allocator.cc:8-14`), because routing Metal through `DEVICE_DISPATCH` would require `primitives<Device::METAL>` to exist at every dispatch site (see `dispatch-and-op-implementation.md` §3).

Allocator instances are function-local statics — created once, never destroyed, shared by all StorageViews on that device.

## 3. CPU: aligned allocator (or MKL)

`src/cpu/allocator.cc:78-87` — `get_allocator<Device::CPU>()` returns one of two singletons, both with **64-byte alignment** (`constexpr size_t alignment = 64`, matching the StorageView header's "allocation is aligned by default to 64 bytes"):

- `cpu::AlignedAllocator` (`cpu/allocator.cc:16-46`): `posix_memalign` (or `_aligned_malloc` on Windows). No caching; `clear_cache()` is the base no-op.
- `cpu::MklAllocator` (`cpu/allocator.cc:49-73`, only under `CT2_WITH_MKL`): `mkl_malloc`/`mkl_free`; `clear_cache()` → `mkl_free_buffers()`.

## 4. CUDA: two allocators, selected at runtime

`src/cuda/allocator.cc` defines two implementations; `resolve_cuda_allocator()` (`cuda/allocator.cc:144-169`) picks one via the **`CT2_CUDA_ALLOCATOR`** env var (`"cub_caching"` or `"cuda_malloc_async"`), defaulting to `cuda_malloc_async` when every GPU reports `cudaDevAttrMemoryPoolsSupported` (`support_cuda_malloc_async`, `cuda/allocator.cc:125-137`), else `cub_caching`. The choice is logged once (`spdlog::info("Using CUDA allocator: …")`, line 165).

### `CubCachingAllocator` (`cuda/allocator.cc:34-77`)

Wraps `cub::CachingDeviceAllocator`. Default config (`cuda/allocator.cc:37-40`):

| param              | default                    |
| ------------------ | -------------------------- |
| `bin_growth`       | 4                          |
| `min_bin`          | 3                          |
| `max_bin`          | 12                         |
| `max_cached_bytes` | 200 MB (`200 * (1 << 20)`) |

Overridable via the **`CT2_CUDA_CACHING_ALLOCATOR_CONFIG`** env var, format `bin_growth,min_bin,max_bin,max_cached_bytes` — exactly 4 comma-separated values or it throws (`cuda/allocator.cc:42-53`). `allocate` is stream-aware: `DeviceAllocate(device_index, &ptr, size, cuda::get_cuda_stream())` (line 63). `clear_cache()` → `FreeAllCached()` (lines 71-73). One instance **per thread** (`static thread_local`, "Use 1 allocator per thread for performance", `cuda/allocator.cc:177-181`).

### `CudaAsyncAllocator` (`cuda/allocator.cc:79-123`)

`cudaMallocAsync`/`cudaFreeAsync` on the current CUDA stream (the driver's own memory pool does the caching). Requires CUDA ≥ 11.2 (`CT2_USE_ASYNC_ALLOC`, line 26); honors an explicit `device_index` by `cudaSetDevice`-ing around the call and restoring (lines 83-95). A single shared instance (`cuda/allocator.cc:183-184`). Because frees are stream-ordered, idle replica workers call `synchronize_stream` so the pool can actually release memory (`replica_pool.h:349-353`).

(HIP builds reuse this file with `hipcub`/`hipMallocAsync` aliases, `cuda/allocator.cc:10-22`; async alloc is disabled on Windows.)

## 5. Who owns the pointer — the StorageView contract

- `StorageView::reserve` lazily binds `_allocator = &get_allocator(_device)` and calls `allocate(size * item_size(), _device_index)` (`storage_view.cc:161-171`); `release()` frees through the same allocator at the same `_device_index`.
- A StorageView **owns** its buffer iff it allocated it: `owns_data()` ⇔ `_allocator != nullptr`; views over external buffers never touch an allocator (`storage_view.cc:382-395`).
- The allocator never tracks lifetimes — it's malloc/free semantics; `StorageView` is the owner, the caching layer just makes churn cheap.
- `ReplicaPool::clear_cache()` walks the workers and calls `allocator->clear_cache()` on each (`replica_pool.h:141-148`); each worker registered its device allocator in `ReplicaWorker::initialize()` (`replica_pool.h:346`).

Some primitives also use the allocator directly for scratch (e.g. the MKL batched-GEMM pointer array, `src/cpu/primitives.cc:1112-1130`, via the file-static `allocator` ref at `primitives.cc:36`).

### Relevance to the Metal backend

- `get_allocator(Device::METAL)` is an **early-return before `DEVICE_DISPATCH`** (`src/allocator.cc:8-14`) — Metal StorageViews get real `MTLBuffer`s, not CPU memory, without instantiating `primitives<Device::METAL>` anywhere.
- The Metal allocator (`src/metal/allocator.mm`, `MetalAllocator`) returns the `contents` pointer of a `MTLResourceStorageModeShared` buffer — CPU-addressable under unified memory, so it satisfies the same pointer-based contract; an address-ordered side table maps a pointer (or an offset into an allocation) back to its `MTLBuffer` for kernel dispatch.
- It is Objective-C++ **without ARC** (manual `[buffer release]` in `free`), and there is no `clear_cache()` override (the base no-op applies).
- Buffer storage modes and CPU↔GPU coherence details: `apple-silicon` skill, `storage-and-synchronization.md`.
