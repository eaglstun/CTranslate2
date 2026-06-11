# Replica pools & the async API

The top of the engine stack: how `Translator`/`Generator`/`Encoder`/`Whisper` are all the
same `ReplicaPool<Replica>` template, how work becomes `Job`s on a `ThreadPool`, and the
future-based async contract. This file owns the POOL structure and lifecycle; thread
_counts_ are `parallelism-and-thread-config.md`, the `rebatch_input` sort and promise
ordering details are `batching-and-length-sorting.md`.

**Sources (all citations from real lines):**

- `include/ctranslate2/replica_pool.h` — the whole pool is **header-only** (there is NO
  `src/replica_pool.cc`; the template lives entirely in the header)
- `include/ctranslate2/thread_pool.h` / `src/thread_pool.cc` (Job, JobQueue, Worker)
- `src/models/model.cc` (`ModelLoader::load`), `include/ctranslate2/models/model.h`
- `src/translator.cc`, `src/generator.cc`, `src/encoder.cc`, `src/models/whisper.cc`

## 1. One template, five products

`ReplicaPool<Replica>` (`replica_pool.h:23-24`) is parameterized by a replica type that
must provide `static create_from_model(const Model&)` (called in
`ReplicaWorker::set_model`, `replica_pool.h:322-324`). The instantiations:

- `Translator : ReplicaPool<models::SequenceToSequenceReplica>` (`include/ctranslate2/translator.h:26`)
- `Generator : ReplicaPool<models::SequenceGeneratorReplica>` (`include/ctranslate2/generator.h:10`)
- `Encoder : ReplicaPool<models::SequenceEncoderReplica>` (`include/ctranslate2/encoder.h:9`)
- `Whisper : ReplicaPool<WhisperReplica>` (`include/ctranslate2/models/whisper.h:158`); same
  pattern for `Wav2Vec2`/`Wav2Vec2Bert` (`include/ctranslate2/models/wav2vec2.h:61`,
  `include/ctranslate2/models/wav2vec2bert.h:62`)

Each subclass is a thin veneer: e.g. `Translator::translate_batch_async` is just
`post_examples<TranslationResult>(load_examples(...), ..., [options](replica, batch){ return run_translation(...); })`
(`src/translator.cc:15-28`). Whisper skips `post_examples` (its inputs are feature
tensors, not token streams) and calls `post`/`post_batch` directly (`src/models/whisper.cc:661,672`).

## 2. ModelLoader and the replica count

`models::ModelLoader` (`models/model.h:211-227`) is a plain struct: `model_reader`,
`device`, `device_indices = {0}`, `num_replicas_per_device = 1`, `compute_type`,
`use_flash_attention`, `tensor_parallel`. `ModelLoader::load()`
(`src/models/model.cc:824-896`) produces `device_indices.size() × num_replicas_per_device`
entries (`:861`), and the sharing rule is explicit:

- The model is **loaded once** for the first device index, then `copy_to` for each further
  device (`model.cc:866-870`).
- All `num_replicas_per_device` replicas on one device index get **the same
  `shared_ptr<const Model>`** (`model.cc:891-892`) — replicas on a device **share const
  weight storage**; only cross-device replicas copy weights. (Header comment:
  "Replicas on the same device ID will reference the same model instance",
  `models/model.h:216-217`.)

`ReplicaPool::initialize_pool(model_loader, config)` calls `set_num_threads` _before_
loading ("same number of computation threads should be used for loading and running",
`replica_pool.h:236-241`), then builds one `ReplicaWorker<Replica>` per loaded model and a
`ThreadPool` with queue cap `max_queued_batches` (default `4 × workers`,
`replica_pool.h:251-255`). So `num_replicas() == _thread_pool->num_threads()` (`:110-112`).

## 3. Job / JobQueue / Worker mechanics

- `Job` (`thread_pool.h:16-26`) — `run()` + an active-job counter decremented in the
  destructor (`thread_pool.cc:7-15`).
- `JobQueue` (`thread_pool.h:29-54`) — mutex + two condvars; `put` blocks when full
  (backpressure, `thread_pool.cc:37-44`); `get` takes a `before_wait` hook that the worker
  uses as its `idle()` call (`thread_pool.cc:46-64`, hook wired at `:113-116`); `close()`
  drains to nullptr, ending workers.
- `Worker::run` (`thread_pool.cc:109-124`): sets the `thread_local local_worker` pointer
  (`:97,110`), `initialize()`, loop `get→run`, `finalize()`. `ReplicaWorker::initialize`
  sets device index + thread count + allocator (`replica_pool.h:339-347`); `idle()` is
  `synchronize_stream(_device)` (`:349-353`); `finalize()` drops the replica and calls
  `destroy_context` (`:355-359`).
- A running job finds _its own_ replica via `ThreadPool::get_local_worker()`
  (`thread_pool.cc:180-184`) → `ReplicaPool::get_thread_replica()` (`replica_pool.h:231-234`).

## 4. The async contract: futures, exceptions, ordering

Everything funnels into `BatchJob` (`replica_pool.h:268-298`): it owns N promises + one
function; `run()` calls the function once, and on exception **sets the same
`exception_ptr` on every promise** (`:277-292`) — exceptions propagate to the caller
through `future.get()`, never crash the worker. Entry points:

- `post<Result>(func)` → single future (`replica_pool.h:60-71`).
- `post_batch<Result>(func, num_results)` → N futures from one job (`:76-87`).
- `post_examples` → promises **created in original input order**, then matched to sorted
  batches by `batch.example_index` (`:158-189`) — see batching-and-length-sorting.md.
- `consume_batches` — the streaming file path: reads `read_batch_size` ahead (default
  `max_batch_size * 16`, `:210-211`), pops finished futures non-blocking between reads (`:198-225`).

The synchronous variants (`translate_batch`, etc.) just `future.get()` in a loop
(`src/translator.cc:53-60`). **Token streaming is NOT a separate API** — it's the
`callback` field on the options structs (`std::function<bool(GenerationStepResult)>`,
`include/ctranslate2/translation.h:85`, `generation.h:77`; greedy-search only; return
`true` to stop early). `restore_batch_ids_in_callback` (`generation.h:119-133`) remaps the
callback's `batch_id` back through `example_index`. (There is no `long_callback` symbol.)

## 5. Pool lifecycle beyond construction

- `detach_models()` / `set_models()` (`replica_pool.h:114-137`) — pull the shared_ptrs out
  of all workers / push replacements in; **not thread-safe**; basis of the Python
  `unload_model(to_cpu=True)` / `load_model` round-trip (`python/cpp/replica_pool.h:110-156`).
- `clear_cache()` (`:139-148`) — calls `allocator->clear_cache()` per worker (CUDA caching
  allocator trim).
- Destruction: `~ThreadPool` closes the queue and joins all workers (`thread_pool.cc:148-152`);
  each worker's `finalize()` releases its replica on its own thread.

### Relevance to the Metal backend

- Metal is single-device, so `device_indices = {0}` and **all** replicas share one const
  `Model` — int8 weights are resident in `MTLBuffer`s once, regardless of `inter_threads`
  (the Qwen RSS numbers in `METAL_BENCHMARKS.md` are per-process, not per-replica).
- `ReplicaWorker::idle()` → `synchronize_stream` is a full `metal::synchronize()` drain
  whenever a worker finds the queue empty (see parallelism-and-thread-config.md's bridge).
- `ReplicaWorker::initialize`'s `get_allocator(_device)` is where each worker binds the
  Metal allocator — the `src/allocator.cc` early-return for `Device::METAL` (CLAUDE.md)
  makes this return real `MTLBuffer` allocation, not the CPU allocator.
- The generation `callback` runs on the worker thread mid-decode; on Metal that interleaves
  Python-side work with the GPU command stream — keep callbacks cheap or decode stalls.
