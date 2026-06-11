# Devices & device management

The `Device` enum and the small set of functions that resolve, set, and synchronize devices. Short file — `src/devices.cc` is the whole story.

**Sources (all citations from real lines):**

- `include/ctranslate2/devices.h`
- `src/devices.cc`

---

## 1. The `Device` enum and string conversion

`devices.h:13-17`:

```cpp
enum class Device { CPU, CUDA, METAL };
```

- `str_to_device` (`devices.cc:18-42`) accepts `"cpu"`, `"cuda"`, `"metal"` (and uppercase), throwing if the binary wasn't compiled with that backend. `"auto"` resolves at runtime: CUDA if built with CUDA and `cuda::has_gpu()`, else Metal if built with Metal and `metal::has_gpu()`, else CPU (`devices.cc:33-40`).
- `device_to_str(Device)` → `"cpu"`/`"cuda"`/`"metal"` (`devices.cc:44-54`); the 2-arg overload formats `"<device>:<index>"` (`devices.cc:56-58`) — the form used in load logs ("Loaded model … on device cuda:1", `src/models/model.cc:871-874`).
- `get_device_count` (`devices.cc:60-78`): `cuda::get_gpu_count()` / `metal::get_gpu_count()` per backend, and always **1 for CPU**.

## 2. Device index: get/set per backend

The per-backend implementations are template specializations `get_device_index<D>()` / `set_device_index<D>(int)` (`devices.cc:80-122`):

- **CPU**: index is always 0; setting anything else throws (`devices.cc:85-94`).
- **CUDA**: `cudaGetDevice`/`cudaSetDevice` (`devices.cc:96-108`) — the index is _thread-global CUDA state_, which is why it must be set per worker thread.
- **METAL**: a single default device, index pinned to 0 (`devices.cc:110-122`).

The runtime wrappers `get_device_index(Device)` / `set_device_index(Device, int)` (`devices.cc:124-144`) are `DEVICE_DISPATCH` switches with the usual **Metal early-return before the macro** (`devices.cc:126-131` and `137-142`) so the dispatch never instantiates a Metal case (see `dispatch-and-op-implementation.md` §3).

### `ScopedDeviceSetter`

Header-only RAII (`devices.h:33-54`): records `get_device_index(device)`, sets the new index if different, restores the previous one in the destructor. Used anywhere code must run "on device N" temporarily — e.g. `synchronize_device` below, and model loading.

## 3. How `device:index` threads through to replicas

A model is loaded once per entry in `device_indices` and then **copied** (`copy_to(device, device_index)`) rather than re-read for subsequent indices; each device's model is then shared by `num_replicas_per_device` replicas (`src/models/model.cc:861-893`). Each replica lives in its own `ReplicaWorker`, which captures `model->device()` / `model->device_index()` at construction and, in `initialize()` (run **on the worker thread**), calls `set_device_index(_device, _device_index)` before anything else (`include/ctranslate2/replica_pool.h:307-314`, `339-347`). So the CUDA thread-global current device is established once per worker thread, and every op that thread runs inherits it. See `parallelism-and-thread-config.md` for the pool itself.

## 4. Synchronization semantics

Two functions, both declared at `devices.h:28-29`, with per-backend meaning:

- **`synchronize_device(device, index)`** (`devices.cc:146-160`) — full-device barrier. CUDA: `cudaDeviceSynchronize()` under a `ScopedDeviceSetter` for that index. Metal: `metal::synchronize()` (index ignored — single device). CPU / unbuilt backends: no-op.
- **`synchronize_stream(device)`** (`devices.cc:162-173`) — current-stream barrier. CUDA: `cudaStreamSynchronize(cuda::get_cuda_stream())`. Metal: also `metal::synchronize()` (Metal has no separate stream granularity here). CPU: no-op.

Callers that matter: model loading synchronizes after moving weights to the device; `ReplicaWorker::idle()` calls `synchronize_stream(_device)` when the work queue runs dry "so that the CudaAsyncAllocator can release some memory" (`replica_pool.h:349-353`); `StorageView::copy_from(..., synchronous=true)` ends with a stream sync (`storage_view.cc:408-429`).

`destroy_context(device)` (`devices.cc:175-183`) is the worker-teardown hook — on CUDA it frees the curand states; called from `ReplicaWorker::finalize()` (`replica_pool.h:355-359`).

## 5. Tensor-parallel ranks (one paragraph)

`devices.h:56-81` also hosts `ScopedMPISetter` and the globals `my_rank` / `local_rank` / `n_ranks` (defined `devices.cc:189-191`). Under `CT2_WITH_TENSOR_PARALLEL` it initializes MPI, derives `local_rank` by hashing hostnames, and lazily creates one NCCL communicator per thread (`devices.cc:193-256`). Without the flag everything is a no-op and the ranks stay 0/0/1. Unrelated to single-device work; just know it lives in this file.

### Relevance to the Metal backend

- Metal is a **single-device backend**: index is always 0 and `set_device_index<Device::METAL>` throws on anything else (`devices.cc:110-122`) — `device_indices=[0]` is the only valid configuration.
- `str_to_device("auto")` prefers Metal over CPU when the binary is built with `WITH_METAL` and a GPU is present (`devices.cc:36-37`).
- Both `synchronize_device` and `synchronize_stream` collapse to `metal::synchronize()` — the wait-for-all-committed-command-buffers point; the cheaper per-op coherence point is `metal::flush()` in `METAL_DEVICE_CASE` (see `apple-silicon` skill, `storage-and-synchronization.md`).
- The runtime index getters/setters early-return for Metal before `DEVICE_DISPATCH` (`devices.cc:126-131`, `137-142`) for the same no-`primitives<Device::METAL>` reason as the allocator.
