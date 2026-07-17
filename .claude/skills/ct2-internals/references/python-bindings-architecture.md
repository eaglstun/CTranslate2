---
title: "Python bindings architecture"
summary: >-
  Describes how python/cpp/*.cc wraps the C++ engine into the ctranslate2
  package: the PYBIND11_MODULE(_ext) layout with free functions
  (contains_model, get_supported_compute_types via mayiuse_* queries) and per-
  translation-unit register_* calls, and the ReplicaPoolHelper<T> base shared
  by every *Wrapper that maps inter_threads/intra_threads/compute_type ctor
  args and guards load/unload with a shared_mutex. Details the three GIL-
  release points (gil_scoped_release call_guard on blocking methods,
  AsyncResult<T>::result() releasing only around future.get, and load/unload
  in ctor/dtor) plus the streaming callback re-acquiring the GIL. Covers async
  results via maybe_wait_on_futures and declare_async_wrapper, and the
  StorageView bridge exposing __array_interface__/__cuda_array_interface__
  (array-interface protocols, not DLPack) for zero-copy views, noting the
  Python Device enum has only cpu and cuda so Metal is addressed by
  device="metal" string. Explains the build-linkage reality that _ext links
  the prebuilt installed libctranslate2 (so a C++ change needs both a library
  rebuild and a wheel rebuild, with rpath/CTRANSLATE2_ROOT and the downstream
  install_name_tool rig), which bites hardest for Metal kernel changes hidden
  in the dylib.
semantic_id: "yxqx0o2qs-22gy1tRnUrvKcho-vTxatBChVmG5Ttx3aMLXgXocc9nye4-lk9rg8hJikk-sf5jY56tYUvxjx9sw"
---

How `python/cpp/*.cc` wraps the C++ engine into the `ctranslate2` package: module layout,
the GIL discipline, the async-result wrappers, the StorageView array-interface bridge, and
the build linkage reality (the extension links the _installed_ `libctranslate2`).

**Sources (all citations from real lines):**

- `python/cpp/module.cc` (the `_ext` module), `python/cpp/utils.h` (`AsyncResult`, resolvers)
- `python/cpp/replica_pool.h` (`ReplicaPoolHelper` — shared base of all model wrappers)
- `python/cpp/translator.cc`, `generator.cc`, `encoder.cc`, `whisper.cc`, `storage_view.cc`, `logging.cc`
- `python/setup.py` (linkage), `tests/downstream/README.md` (the install-into-consumers rig)

## 1. Module layout

`PYBIND11_MODULE(_ext, m)` (`module.cc:48-92`) registers a handful of free functions
(`contains_model`, `get_cuda_device_count`, `get_supported_compute_types` — built from the
`mayiuse_*` capability queries, `module.cc:13-45` — and `set_random_seed`), then calls one
`register_*` per translation unit: logging, storage_view, translation/scoring/generation
results, translator, generator, encoder, whisper, wav2vec2(+bert), mpi (`module.cc:79-91`).
The pure-Python package re-exports `_ext` and adds the converters/specs on top.

Every model class follows the same shape: a `*Wrapper` class deriving from
`ReplicaPoolHelper<T>` (e.g. `TranslatorWrapper : ReplicaPoolHelper<Translator>`,
`python/cpp/translator.cc:24`). `ReplicaPoolHelper` (`python/cpp/replica_pool.h:38-194`) owns the
ctor argument mapping — `inter_threads` → `model_loader.num_replicas_per_device`,
`intra_threads` → `_pool_config.num_threads_per_replica`, `compute_type` (string OR
per-device dict) via `ComputeTypeResolver` (`:57-66`, resolvers in `python/cpp/utils.h:28-60`) — plus
the `unload_model`/`load_model` CPU-cache dance (`:110-156`) guarded by a `shared_mutex`
(readers = inference calls, writer = load/unload, `:168-171`). The `files` ctor arg builds
a `ModelMemoryReader` so a model can be loaded from in-memory bytes (`:11-36`).

## 2. The GIL story

Three distinct release points, all real:

1. **Blocking method calls** use `py::call_guard<py::gil_scoped_release>()` on the `def`
   (e.g. `translate_batch` at `python/cpp/translator.cc:368`, same pattern in
   `python/cpp/generator.cc:215`, `python/cpp/encoder.cc:129`, `python/cpp/whisper.cc:229`,
   `python/cpp/storage_view.cc:209`) — the GIL is dropped for the
   whole C++ call, so worker threads and other Python threads run freely.
2. **`AsyncResult<T>::result()`** re-acquires lazily: it releases the GIL only around
   `_future.get()` (`python/cpp/utils.h:70-85`), catching the exception into an `exception_ptr` and
   rethrowing once the GIL is back.
3. **Model load/unload**: `ReplicaPoolHelper`'s constructor and destructor both take
   `gil_scoped_release` (`python/cpp/replica_pool.h:55,73`) — model loading is long.

Inverse direction: the streaming `callback` option is a `std::function` wrapping a Python
callable (`python/cpp/translator.cc:152,181`); pybind11's functional caster re-acquires the GIL when
the decode loop invokes it from a worker thread.

## 3. Async results

`asynchronous=True` doesn't change the C++ call — `maybe_wait_on_futures`
(`python/cpp/utils.h:108-120`) either drains the `std::future`s (sync) or moves each into an
`AsyncResult<T>` (`python/cpp/utils.h:62-97`) exposing `.result()` (blocking, rethrows) and `.done()`
(non-blocking poll, `:87-90`). `declare_async_wrapper` (`python/cpp/utils.h:122-134`) stamps out the
Python classes (`AsyncTranslationResult` etc., declared in each `register_*`).

## 4. StorageView exposure

`python/cpp/storage_view.cc` exposes `DataType`, `Device`, and `StorageView` via the
**array interface protocols, not DLPack**:

- `StorageView.from_array(obj)` reads `__array_interface__` or
  `__cuda_array_interface__` (`python/cpp/storage_view.cc:58-85`) — zero-copy view over the caller's
  memory (`view.view((void*)ptr, shape)`, `:82-84`; `py::keep_alive` ties lifetimes,
  `:144`). Non-contiguous and read-only arrays are rejected (`:70-71,79-80`).
- `__array_interface__` / `__cuda_array_interface__` properties (`:175-187`) export the
  raw pointer + typestr so numpy/torch wrap it without copy; each throws when the data is
  on the other device class.
- Methods `to(dtype)` and `to_device(device)` (`:201-237`) copy-convert and synchronize.
- **The Python `Device` enum only has `cpu` and `cuda`** (`python/cpp/storage_view.cc:109-112`) —
  `Device::METAL` is not exposed; Python code addresses Metal models via the constructor
  `device="metal"` string (`str_to_device`), not via this enum.

## 5. Build linkage reality

The extension is **not** self-contained: `setup.py` compiles `python/cpp/*.cc` and links
`libraries=["ctranslate2"]` (`setup.py:58-66`) against a prebuilt install. `CTRANSLATE2_ROOT`
is honored by `_maybe_add_library_root("CTRANSLATE2")` (`setup.py:31-42`), adding
`$ROOT/include` and `$ROOT/lib{,64}`; otherwise it relies on the baked rpath
(`-Wl,-rpath,/usr/local/lib` on macOS, `:50`). Consequences (the CLAUDE.md rule):

- A C++ change is invisible from Python until the C++ library is rebuilt/installed **and**
  the wheel is rebuilt against it.
- At import time the loader must find `libctranslate2` at the rpath — a wheel built
  against a non-standard prefix needs that prefix on the rpath or `DYLD_LIBRARY_PATH`.
- This project's downstream rig automates exactly that: build → `cmake --install` to a
  pinned prefix → rebuild wheel with `CTRANSLATE2_ROOT=<prefix>` → `uv pip` force-reinstall
  into each consumer venv **plus `install_name_tool -add_rpath`** to point the freshly
  built `_ext` at the prefix (`tests/downstream/README.md:16-21`).

### Relevance to the Metal backend

- The Python surface needs **zero Metal-specific bindings**: `device="metal"` flows as a
  string through `str_to_device` in `ReplicaPoolHelper` (`python/cpp/replica_pool.h:57`),
  and `get_supported_compute_types("metal")` works through the same `mayiuse_*` queries.
- `StorageView.__array_interface__` works on Metal-resident tensors _because of unified
  memory_ — the buffer pointer is CPU-addressable — but the property itself would route to
  the CUDA branch check (`view.device() == Device::CUDA` is false → CPU path), so numpy
  reads Metal buffers directly; ensure GPU work is flushed first (the
  `metal::flush()` coherence rule in dispatch-and-op-implementation.md §3).
- The wheel-rebuild rule bites hardest here: Metal kernel changes live in
  `libctranslate2.dylib`, so a stale consumer venv silently runs old kernels — the
  downstream rig's force-reinstall exists because of this.
