---
title: "Parallelism & thread configuration"
summary: >-
  Explains CPU-side threading: the two levels inter_threads (replica copies
  processing different batches via CT2's std::thread ThreadPool) versus
  intra_threads (compute threads inside one op via parallel_for), which
  multiply. It covers the two intra-op runtimes selected by OPENMP_RUNTIME
  (Intel OpenMP versus the vendored BS_thread_pool when _OPENMP is undefined,
  which is what Apple Silicon Metal builds use with OPENMP_RUNTIME=NONE), the
  GRAIN_SIZE=32768 default and nesting rules, and the thread-count plumbing
  from Python args down through set_num_threads (default min(4,hw) favoring
  replica parallelism) enforced per-worker in ReplicaWorker::initialize. The
  replica ThreadPool mechanics cover backpressure at max_queued_batches,
  worker lifecycle, the idle() synchronize_stream hook (metal::synchronize on
  Metal), and Linux-only core pinning.
---

CPU-side threading: the two parallel runtimes, `parallel_for`, and how `inter_threads`/`intra_threads` flow from the public API down to the loops. Practical focus: which knob does what, where it's enforced.

**Sources (all citations from real lines):**

- `src/cpu/parallel.h` / `parallel.cc` (`parallel_for`, the non-OpenMP pool)
- `src/utils.cc` (`set_num_threads`)
- `include/ctranslate2/replica_pool.h` (replica workers, where both knobs land)
- `src/thread_pool.cc` (the replica job queue/workers)
- `CMakeLists.txt` (`OPENMP_RUNTIME`)
- `python/cpp/replica_pool.h` (the Python `inter_threads`/`intra_threads` surface)

---

## 1. Two distinct levels of parallelism

- **inter_threads** — how many _replicas_ (model copies) process **different batches in parallel**. Implemented by CT2's own `ThreadPool` of `ReplicaWorker`s (always `std::thread`-based, regardless of OpenMP).
- **intra_threads** — how many compute threads **one op** uses inside one replica (the `parallel_for` below; "Number of OpenMP threads per worker" in the Python docstrings, e.g. `python/cpp/generator.cc:160`).

Total CPU threads ≈ `inter_threads × intra_threads`. On GPU, intra-op threading is the device's business; inter still controls concurrent batches.

## 2. The two intra-op runtimes (`OPENMP_RUNTIME`)

`parallel.h:5-9` picks at compile time: `_OPENMP` defined → `omp.h`; otherwise → **`BS_thread_pool.hpp`** (the vendored single-header pool at `third_party/BS_thread_pool.hpp`). The CMake knob is `OPENMP_RUNTIME` (default `INTEL`, `CMakeLists.txt:57`):

- `INTEL` / `COMP` — compile with OpenMP flags (`CMakeLists.txt:313-321`) and link `iomp5` or the compiler's runtime (`CMakeLists.txt:323-352`); MKL gets the matching threading layer (`CMakeLists.txt:395-413`).
- `NONE` — no OpenMP flags, so `_OPENMP` is undefined and the BS pool path compiles in. **This machine's Metal builds use `-DOPENMP_RUNTIME=NONE`** (the Intel OpenMP default doesn't exist on Apple Silicon — see CLAUDE.md build recipe).

In the `NONE` build the per-thread state is `cpu::set_num_threads`/`get_num_threads` over a `thread_local size_t num_threads = 1` plus a `static thread_local BS::thread_pool` (`parallel.cc:8-21`). Gotcha: the pool is constructed **on first use in that thread, sized from `num_threads` at that moment** — so `set_num_threads` must run before the first `parallel_for` on a worker thread (it does: `ReplicaWorker::initialize`, §4).

## 3. `cpu::parallel_for` — grain size and nesting

`parallel.h:38-86`, signature `parallel_for(begin, end, grain_size, f)` where `f(begin, end)` gets a contiguous chunk. `GRAIN_SIZE = 32768` (`parallel.h:22`) is the default "don't parallelize below this many elements" constant; kernels divide it by per-element work (`parallel_unary_transform` passes `GRAIN_SIZE / work_size`, `parallel.h:97-106`) or hardcode smaller grains for heavy math (`gelu` uses 512, `src/cpu/primitives.cc:305`).

- **OpenMP path** (`parallel.h:49-68`): runs serially if `omp_get_max_threads() == 1`, **if already inside a parallel region** (`omp_in_parallel()` — no nesting), or if `size <= grain_size`. Else `#pragma omp parallel` with the thread count capped at `ceil_divide(size, grain_size)` and the range split into equal chunks.
- **BS-pool path** (`parallel.h:70-84`): `num_blocks = min(get_num_threads(), ceil_divide(size, grain_size))`; serial if 1, else `thread_pool.detach_blocks(begin, end, f, num_blocks)` + `wait()`. Nesting is implicitly safe-ish because an inner `parallel_for` on a pool thread sees that thread's `thread_local num_threads` (1 unless someone set it), so it runs serially.

## 4. Thread-count plumbing, top to bottom

1. **Python/C++ API**: `inter_threads` and `intra_threads` constructor args (e.g. `python/cpp/generator.cc:143-144`). In the binding, `intra_threads` → `_pool_config.num_threads_per_replica` and `inter_threads` → `model_loader.num_replicas_per_device` (`python/cpp/replica_pool.h:45-64`).
2. **`ReplicaPoolConfig.num_threads_per_replica = 0`** is the default (`replica_pool.h:13-17`), meaning "auto".
3. **`ctranslate2::set_num_threads(n)`** (`src/utils.cc:85-98`) is the enforcement point: `n == 0` → read **`OMP_NUM_THREADS`** env, default `min(4, hardware_concurrency)` (`utils.cc:77-83`); then `omp_set_num_threads` (OpenMP) or `cpu::set_num_threads` (BS pool), plus the Ruy context max (`utils.cc:95-97`). Note the deliberately small default of **4** — CT2 favors replica parallelism over per-op threads.
4. It is called twice: once on the loading thread before model load ("The same number of computation threads should be used for loading and running model", `replica_pool.h:236-241`), and once **per worker thread** in `ReplicaWorker::initialize()` (`replica_pool.h:342-343`) — necessary because both `omp_set_num_threads` and the BS-pool count are per-thread state.

## 5. The replica `ThreadPool` (`inter_threads` mechanics)

`ReplicaPool::initialize_pool` builds one `ReplicaWorker` per loaded model (devices × replicas-per-device, `replica_pool.h:243-259`) and wraps them in `ThreadPool` — so `num_replicas() == _thread_pool->num_threads()`. Key behaviors:

- **Backpressure**: the job queue blocks `put` at `max_queued_batches` (default `4 × workers`, `replica_pool.h:251-255`; `JobQueue::put` waits on `_can_put_job`, `thread_pool.cc:37-44`).
- **Worker lifecycle** (`thread_pool.cc:108-124`): each worker thread runs `initialize()` (set device index, set thread count, register allocator — `replica_pool.h:339-347`), then loops `job_queue.get(...)` → `job->run()`, then `finalize()` (drop replica, `destroy_context`). A `thread_local local_worker` pointer lets a running job find its own replica (`ThreadPool::get_local_worker`, `thread_pool.cc:178-182`; used by `ReplicaPool::get_thread_replica`, `replica_pool.h:231-234`).
- **Idle hook**: when the queue is empty the worker calls `idle()` → `synchronize_stream(_device)` before blocking (`thread_pool.cc:113`, `replica_pool.h:349-353`).
- **Core pinning**: `ReplicaPoolConfig.cpu_core_offset` pins worker i to core `offset + i` — Linux-only and **only in non-OpenMP builds** (`set_thread_affinity` throws otherwise, `thread_pool.cc:79-95`).

Rule of thumb: throughput on CPU → raise `inter_threads` (more batches in flight, memory permitting); latency of a single batch → raise `intra_threads`; the two multiply, so oversubscription is easy.

### Relevance to the Metal backend

- This machine builds with `OPENMP_RUNTIME=NONE`, so every CPU-reference op running on Metal-resident data (the `D = CPU` binding) is parallelized by the **BS thread pool path** of §3, sized per §4.
- `ReplicaWorker::idle()`'s `synchronize_stream` maps to `metal::synchronize()` on Metal (`src/devices.cc:162-173`) — a full GPU drain whenever a worker goes idle.
- Metal is single-device (index 0), so `inter_threads > 1` means multiple replicas sharing the one GPU; the per-op dispatch/overlap consequences are in the `apple-silicon` skill (`dispatch-overlap-and-perf-model.md`).
- `cpu_core_offset` _is_ usable on Metal-capable builds in principle, but the affinity call is Linux-only — it throws on macOS (`thread_pool.cc:80-84`).
