# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

CTranslate2 is a C++ and Python library for efficient inference with Transformer models. It implements a custom runtime with optimizations like weight quantization, layer fusion, batch reordering, and runtime CPU dispatch, targeting both CPU and GPU. The C++ library is the engine; the Python package wraps it and adds the model converters.

> **Note from CONTRIBUTING.md:** This is a low-level, performance-critical codebase. A misplaced pointer or an inefficient allocation can cost hours to debug, and verifying changes for correctness/performance is more work than writing them. Maintainers expect deep understanding of any change and explicit disclosure of AI assistance. Stay within the area you actually understand.

## Build

The C++ library builds first, then the Python wrapper links against it. Submodules must be present (`git submodule update --init --recursive`).

### C++ library

```bash
mkdir build && cd build
cmake ..          # add -DOPTION=VALUE flags as needed
make -j$(nproc)
sudo make install && sudo ldconfig
```

Requires a C++17 compiler and CMake ≥ 3.15. Defaults to the Intel MKL CPU backend (`WITH_MKL=ON`), which must be installed separately.

**On Apple Silicon** the MKL default fails (x86-only) and so does the default Intel OpenMP runtime. Build with:

```bash
cmake .. -DWITH_MKL=OFF -DWITH_ACCELERATE=ON -DOPENMP_RUNTIME=NONE -DBUILD_TESTS=ON
# add -DWITH_METAL=ON for the Metal GPU backend
```

Key CMake options (see `docs/installation.md` for the full table):

| Option                                                                      | Default  | Purpose                                                                            |
| --------------------------------------------------------------------------- | -------- | ---------------------------------------------------------------------------------- |
| `WITH_CUDA` / `WITH_CUDNN`                                                  | OFF      | CUDA / cuDNN GPU backend                                                           |
| `WITH_HIP`                                                                  | OFF      | AMD ROCm GPU backend                                                               |
| `WITH_METAL`                                                                | OFF      | Apple Metal GPU backend (Apple Silicon only) — see `METAL_BACKEND.md`              |
| `WITH_MKL` / `WITH_DNNL` / `WITH_OPENBLAS` / `WITH_RUY` / `WITH_ACCELERATE` | MKL only | CPU compute backends                                                               |
| `WITH_TENSOR_PARALLEL`                                                      | OFF      | NCCL + MPI for multi-GPU tensor parallelism                                        |
| `WITH_FLASH_ATTN`                                                           | OFF      | Flash Attention 2                                                                  |
| `ENABLE_CPU_DISPATCH`                                                       | ON       | Compile CPU kernels for multiple ISAs (AVX/AVX2/AVX512/NEON) and select at runtime |
| `CUDA_DYNAMIC_LOADING`                                                      | OFF      | `dlopen` CUDA libs at runtime instead of linking                                   |
| `BUILD_TESTS`                                                               | OFF      | Build the C++ test binary                                                          |
| `BUILD_CLI`                                                                 | ON       | Build the command-line clients                                                     |
| `ENABLE_PROFILING`                                                          | OFF      | Integrated profiler (off in production)                                            |

### Python wrapper

After the C++ library is installed (or point `CTRANSLATE2_ROOT` at its install prefix):

```bash
cd python
pip install -r install_requirements.txt
python setup.py bdist_wheel
pip install dist/*.whl
```

`python/cpp/*.cc` are the pybind11 bindings; the extension links against the prebuilt `ctranslate2` shared library. A C++ change is not visible from Python until the C++ library is rebuilt/reinstalled **and** the wheel is rebuilt.

## Tests

**C++** — configure with `-DBUILD_TESTS=ON`, then run the Google Test binary with the path to test data:

```bash
./tests/ctranslate2_test ../tests/data
```

**Python** — uses pytest:

```bash
cd python
pip install -r tests/requirements.txt
pytest tests/                          # all
pytest tests/test_translator.py -k name   # single test
```

## Lint (Python only)

```bash
cd python
black . && isort . && flake8 .
```

## Performance checks

The CLI (`cli/translator.cc`, installed as `ct2-translator`) carries two flags maintainers use to gate changes:

- `--log_throughput` — target tokens/second on stderr (the metric to compare runs; higher is better).
- `--log_profiling` — per-function execution profile (requires `ENABLE_PROFILING=ON`).

Metal micro-benchmarks live in `tests/metal_test.cc` as `DISABLED_Benchmark*` cases; run with `--gtest_also_run_disabled_tests --gtest_filter='*Benchmark*'`. Results and analysis are in `METAL_BENCHMARKS.md`.

## Architecture

### Abstraction layers (lowest → highest)

This layering is the mental model for the whole engine — changes usually belong to exactly one level:

- **kernels** (`src/cpu/kernels*`, `src/cuda/`) — low-level compute (e.g. CUDA Softmax).
- **primitives** (`src/cpu/primitives.cc`, `src/cuda/primitives.cu`) — basic vector/matrix functions over raw arrays.
- **ops** (`src/ops/`, `include/ctranslate2/ops/`) — neural-net operations (Gemm, Softmax, LayerNorm…), interface loosely follows ONNX.
- **layers** (`src/layers/`) — stateful layers (attention, transformer blocks, decoder).
- **models** (`src/models/`) — collections of layers + weights (`transformer.cc`, `whisper.cc`, `language_model.cc`, `sequence_to_sequence.cc`, `wav2vec2*`).
- **replicas / replica pool** (`src/translator.cc`, `src/generator.cc`, `src/encoder.cc`, `src/thread_pool.cc`) — runnable model instances and the thread pool that runs batches in parallel/async.

### `StorageView` — the core data structure

`src/storage_view.cc` / `include/ctranslate2/storage_view.h`. A row-major tensor-like buffer wrapper _without_ math semantics — it views data in a shape and provides resize/reshape/copy. Its dtype (e.g. `float`) and location (e.g. GPU #1) are resolved **at runtime**. Performance depends on avoiding allocations: resizing smaller does not reallocate, and caching allocators reuse buffers. Treat allocation churn as a bug.

### Op implementation pattern

Each op is split across files so the header stays flag-free (the library is meant to be embedded):

```text
include/ctranslate2/ops/my_op.h   # interface — NO compilation flags here
src/ops/my_op.cc                  # input checks + dispatch on device & dtype
src/ops/my_op_cpu.cc              # CPU implementation
src/ops/my_op_gpu.cu              # CUDA implementation
```

Dispatch machinery lives in `src/dispatch.h`, `src/device_dispatch.h`, `src/type_dispatch.h`. Runtime CPU ISA selection is in `src/cpu/cpu_isa.*` and `src/cpu/cpu_info.*` (vectorized paths in `src/cpu/vec_avx.h`, `vec_avx512.h`, `vec_neon.h`). When `ENABLE_CPU_DISPATCH=ON` one binary holds several ISA variants chosen at runtime.

### Apple Metal backend (`Device::METAL`)

A GPU backend for Apple Silicon, in `src/metal/` (Objective-C++ `.mm` + MSL kernels in `kernels/kernels_msl.h`). **Read `METAL_BACKEND.md` before touching it** — it has the full design, file map, milestone status, and contributor gotchas; `METAL_BENCHMARKS.md` has perf numbers. Unlike AMD HIP (which reuses the CUDA `.cu` path), Metal is a genuinely separate backend. The load-bearing design choices a change must respect:

- **It does NOT add `Device::METAL` as a real `DEVICE_CASE`.** Doing so would force `primitives<Device::METAL>`/`compute<Device::METAL,T>` to be instantiated at ~50 dispatch sites and break the link. Instead `device_dispatch.h` binds the METAL dispatch case to `constexpr Device D = Device::CPU` — the CPU kernels run correctly on Metal-resident data because Apple Silicon has **unified memory** (a shared-storage `MTLBuffer`'s `contents` pointer is CPU-addressable and satisfies the existing pointer-based `Allocator`/`StorageView` contract).
- **GPU kernels are added by targeted routing**, not by flipping a switch: an op checks `x.device() == Device::METAL` at the `operator()` level and calls a `metal::` entry point (returning before the generic dispatch), else falls through to the CPU reference. Parity is verified by the existing op suite running on `Device::METAL` (`tests/ops_test.cc` etc. are parameterized over it; Metal-specific tests are in `tests/metal_test.cc`).
- **Allocation/device-index must NOT follow the CPU binding** — `src/allocator.cc` and `src/devices.cc` early-return for `Device::METAL` so Metal StorageViews get real `MTLBuffer`s, not CPU memory.
- The forward pass (GEMM via MPS, softmax, norms, rotary, gather, bias+activation, mul) runs on the GPU in fp32 and fp16; a full transformer runs end-to-end matching CPU output. Sampling/conv/etc. run via the CPU reference. The current perf bottleneck is **per-op command-buffer sync** (one `commit`+`waitUntilCompleted` per op) — see the benchmarks doc.

### Python side: converters and specs

The Python package is more than bindings — it owns model import:

- `python/ctranslate2/converters/` — convert external checkpoints (`transformers.py`, `fairseq.py`, `marian.py`, `opennmt_py.py`, `opennmt_tf.py`, `opus_mt.py`, `openai_gpt2.py`, `eole_ct2.py`) into the CTranslate2 model format. Exposed as `ct2-*-converter` console scripts.
- `python/ctranslate2/specs/` — model _specifications_: the declarative layer/weight layout (`transformer_spec.py`, `attention_spec.py`, `common_spec.py`, `whisper_spec.py`, etc.) that converters populate and the C++ model loader reads.

**Adding support for a new model architecture** almost always means: add/extend a spec in `specs/`, teach the relevant converter in `converters/` to map weights into it, and wire the model type into the C++ loader (`src/models/model_factory.cc` and the relevant `src/models/*.cc` / `src/layers/*.cc`).

## Conventions & gotchas

- Op/layer headers in `include/` must not use compilation flags (`#ifdef WITH_CUDA`, etc.) — keep those in `.cc`/`.cu` files so the project stays usable as a library.
- Backward compatibility is a guarantee (see `docs/versioning.md`); the model format and public API are stable surfaces — don't break them casually.
- Releases: note in `CHANGELOG.md`, bump `python/ctranslate2/version.py`, tag `vX.Y.Z`. The C++ source stays CUDA 11-compatible even though Python wheels ship CUDA 12.x.
- Search for `TODO` comments and _help wanted_ issues for scoped contribution tasks (per CONTRIBUTING.md).
