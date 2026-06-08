# Apple Metal Backend — Progress & Roadmap

Status as of 2026-06-08. This document tracks the in-progress Apple Metal GPU backend
(`Device::METAL`, built with `-DWITH_METAL=ON`) for CTranslate2.

## TL;DR

A new GPU backend for Apple Silicon. Five milestones are complete and verified on real
hardware: tensors live on the GPU, the entire op/layer/storage test suite runs on
`Device::METAL`, matmul runs on Metal Performance Shaders, a **full transformer
translates end-to-end on the GPU** with output identical to the CPU, and GEMM + softmax
also run in **float16**. GEMM, softmax, and elementwise-add execute as real GPU kernels
today; every other operation runs correctly (float32) via a CPU-reference path over
unified memory, ready to be graduated to GPU kernels one at
a time behind a regression net.

## Why this is tractable

Two facts shape the whole design:

1. **Apple Silicon has unified memory.** A Metal buffer allocated with
   `MTLResourceStorageModeShared` exposes a CPU-addressable `contents` pointer. That
   pointer satisfies CTranslate2's existing pointer-based `Allocator`/`StorageView`
   contract unchanged, host↔device copies become plain `memcpy`, and CPU code can operate
   directly on Metal-resident data.

2. **Metal is not source-compatible with CUDA.** Unlike AMD ROCm (which rides the CUDA
   `.cu` files via HIP), Metal needs its own Objective-C++ (`.mm`) host code and Metal
   Shading Language (`.metal`/MSL) kernels, plus Metal Performance Shaders (MPS) for GEMM.
   So it is a genuinely new backend, not a re-skin.

## Build

Requires macOS on Apple Silicon (arm64). Initialize submodules first
(`git submodule update --init --recursive`).

```bash
mkdir build && cd build
cmake .. -DWITH_METAL=ON -DWITH_MKL=OFF -DWITH_ACCELERATE=ON \
         -DOPENMP_RUNTIME=NONE -DBUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
make -j$(sysctl -n hw.ncpu)
./tests/ctranslate2_test ../tests/data --gtest_filter='METAL/*:MetalTest.*'
```

Non-obvious flags (defaults fail on Apple Silicon):

- `-DWITH_MKL=OFF -DWITH_ACCELERATE=ON` — Intel MKL is x86-only; Accelerate is the
  Apple-Silicon CPU BLAS (and the CPU-reference backend for Metal).
- `-DOPENMP_RUNTIME=NONE` — the default `INTEL` runtime expects libiomp5, which is absent.

`WITH_METAL` is mutually exclusive with `WITH_CUDA`/`WITH_HIP` and requires arm64.

## Design

### Memory: a shared-buffer allocator with an offset-aware side table

`src/metal/allocator.mm` specializes `get_allocator<Device::METAL>()`. Each allocation is
an `MTLBuffer` (shared storage); the allocator returns `[buffer contents]` and records the
allocation in an address-ordered `std::map<uintptr_t, {buffer, size}>`. `buffer_and_offset(ptr)`
does a range lookup so that pointers offset into an allocation (StorageView sub-views,
strided-batch GEMM matrices) resolve to the owning buffer plus a byte offset. Manual
retain/release (not ARC).

### Dispatch: a CPU-reference binding, graduated by targeted routing

`src/device_dispatch.h` binds the `Device::METAL` dispatch case to
`constexpr Device D = Device::CPU`. Because Metal memory is CPU-addressable, the existing
CPU kernels run correctly on Metal-resident data — so the whole engine works on
`Device::METAL` immediately, with **no** per-op `compute<Device::METAL,T>` instantiations
and **no** `primitives<Device::METAL>` specialization.

Individual hot paths are then "graduated" to real GPU kernels by **targeted routing**: an
op checks the real device (`a.device() == Device::METAL`, which is correct even though the
dispatch bound `D=CPU`) and calls a `metal::` entry point before falling through to the
CPU path. Example (`src/ops/matmul.cc`):

```cpp
#ifdef CT2_WITH_METAL
if constexpr (std::is_same<T, float>::value) {
  if (a.device() == Device::METAL) { metal::gemm(...); return; }
}
#endif
```

**Important invariant:** allocation and device-index resolution must NOT follow the
CPU binding — `get_allocator(Device)` (`src/allocator.cc`) and `get_device_index`/
`set_device_index` (`src/devices.cc`) early-return for `Device::METAL` to the real Metal
allocator/index. If they fell through to the CPU-bound dispatch, Metal StorageViews would
get plain CPU memory and GPU kernels could not find their buffers.

### Kernels & device context

`src/metal/device.mm` holds a process-wide singleton: `MTLDevice`, one
`MTLCommandQueue`, and an `MTLLibrary` compiled at runtime via `newLibraryWithSource`
from MSL embedded as a C++ raw string in `src/metal/kernels/kernels_msl.h`. (Runtime
compilation avoids `.metallib` path-resolution issues with the bare shared library; an
offline `.metallib` is a future optimization.) Compute pipeline states are cached per
kernel name. Each `metal::` op currently commits one command buffer and waits
synchronously.

## File map

| File                                 | Role                                                                                        |
| ------------------------------------ | ------------------------------------------------------------------------------------------- |
| `src/metal/utils.h`                  | Pure-C++ surface (`has_gpu`, `get_gpu_count`, `synchronize`) — safe to include from `.cc`   |
| `src/metal/device.h`                 | Objective-C++ internals (device/queue/pipeline accessors, `buffer_and_offset`) — `.mm` only |
| `src/metal/device.mm`                | Device/queue/library singleton, pipeline cache, lifecycle                                   |
| `src/metal/allocator.mm`             | Shared-buffer allocator + address-ordered side table                                        |
| `src/metal/primitives.h`             | Pure-C++ declarations of `metal::` compute entry points                                     |
| `src/metal/primitives.mm`            | `metal::add`, `metal::softmax` (kernel dispatch)                                            |
| `src/metal/gemm.mm`                  | `metal::gemm` / `gemm_batch_strided` via MPSMatrixMultiplication                            |
| `src/metal/kernels/kernels_msl.h`    | MSL kernel source (add, softmax) as a raw string                                            |
| `src/device_dispatch.h`              | `METAL_DEVICE_CASE` (CPU-reference binding)                                                 |
| `src/allocator.cc`, `src/devices.cc` | METAL early-returns for allocator/device-index/synchronize                                  |
| `tests/metal_test.cc`                | Metal-specific tests incl. end-to-end translation                                           |

CMake wiring lives in the `if(WITH_METAL)` block in `CMakeLists.txt` (Objective-C++ via
`enable_language(OBJCXX)`, links `Metal`/`MetalPerformanceShaders`/`Foundation`).

## Milestones

### ✅ M1 — Tracer bullet

Enum + string/lifecycle wiring, shared-buffer allocator, device/queue/library singleton,
one hand-written `add` kernel. Proves a `StorageView` lives on the GPU and survives a
kernel round-trip. Verified by `tests/metal_test.cc`.

### ✅ M2 — Dispatch on-ramp (CPU reference)

The `METAL→CPU` dispatch binding. The entire existing op / layer / storage-view test
suite is parameterized over `Device::METAL` and passes (executing the CPU reference on
unified memory). This is the regression harness that guards every later GPU kernel.

### ✅ M3 — GEMM on the GPU (MPS)

`metal::gemm` + `gemm_batch_strided` (float32) via `MPSMatrixMultiplication`, row-major
(no cuBLAS-style operand swap). Routed from `MatMul` and `Gemm` ops. Required the
offset-aware allocator for strided-batch and sub-view operands. All layer tests — whose
attention and FFN matmuls now run on MPS — pass.

### ✅ M4 — End-to-end model + first reduction kernel

- **End-to-end:** the default transliteration transformer translates on `Device::METAL`
  with output identical to CPU (`MetalTest.EndToEndTranslation`). Full encoder-decoder,
  attention, FFN, layer norm, and beam search on the GPU-backed device.
- **GPU softmax:** `ct2_softmax_float` (threadgroup tree reduction), matching CPU
  semantics including `lengths` masking and log-softmax. Routed from the `SoftMax` op.

### ✅ M5 — fp16 foundation

fp16 for the ops already on the GPU. `mayiuse_float16(Device::METAL)` → true so the
`FLOAT16` compute type resolves on Metal (`AUTO` left CPU-like, so fp16 is explicit
opt-in). float16 overloads of `metal::gemm`/`gemm_batch_strided` (`MPSDataTypeFloat16`)
and a `ct2_softmax_half` kernel. Op routing moved to the `operator()` level (the generic
dispatch throws on fp16 in a non-CUDA build), covering fp32 + fp16 in one path. Verified
by `Float16GemmMatchesFloat32` / `Float16SoftMaxMatchesFloat32` (parity vs fp32, tol
2e-2). **A full fp16 model is not yet possible** — it needs the remaining decoder ops
graduated to half kernels (the CPU reference is float32-only).

### Verification snapshot

- Full METAL suite: **79 passed, 2 skipped, 0 failed** (skips = Conv1D dilation and
  grouped-quantized, which the CPU reference does not implement).
- Full suite (all devices): **266/270** — the lone failure is a pre-existing
  `CPU/...Conv1DGroupNoBiasQuantized` "No INT8 GEMM backend for CPU" artifact of the
  MKL-less build, unrelated to Metal.
- CPU-only build: no new warnings introduced.

## What runs where today

| Operation                                                                                               | Metal execution                                                                                               |
| ------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| GEMM / MatMul (float32 and float16)                                                                     | **GPU** — MPSMatrixMultiplication                                                                             |
| SoftMax / LogSoftMax (float32 and float16)                                                              | **GPU** — custom kernel                                                                                       |
| Elementwise add (float32)                                                                               | **GPU** — custom kernel                                                                                       |
| Everything else (norms, gather, rotary, bias/activation, sampling, concat/split, conv, quantization, …) | CPU reference over unified memory (correct, float32 only)                                                     |
| fp16 for ungraduated ops                                                                                | Not yet — CPU reference is float32-only, so a full fp16 model needs those ops graduated to half kernels first |
| bf16 compute                                                                                            | Not yet                                                                                                       |

## What's left

### Near term — graduate more ops to GPU kernels

Each follows the established pattern: write an MSL kernel, add a `metal::` entry point,
add `if (device == Device::METAL)` routing in the op, verify parity against the CPU
reference via the existing suite.

- `RMSNorm`, `LayerNorm` (threadgroup reductions, like softmax)
- `Gather` (embedding lookup)
- `Rotary` (RoPE)
- `BiasAdd` + activations (GELU, etc.)
- Sampling: `TopK`, `TopPMask`, `Multinomial`
- Remaining elementwise/`mul` variants used in residuals and gating

### fp16 — foundation done, full-model fp16 remaining

Where Apple Silicon actually gets fast. The foundation shipped in M5: `mayiuse_float16`
is true for Metal, and GEMM + softmax run in fp16. **A full fp16 model is still blocked**
because the CPU-reference binding is float32-only — every op a model touches needs a half
kernel. So full fp16 ≈ the "graduate more ops" list above, done with `half` kernels.
Remaining fp16 work:

- fp16 `half` kernels for the rest of the decoder path (norms, gather, rotary, bias/act).
- Optionally a `Device::METAL`-aware `DEVICE_AND_FLOAT_DISPATCH` in `src/dispatch.h` (it
  hardcodes `Device::CUDA` for fp16/bf16) if any fp16 op is routed through the generic
  dispatch rather than at `operator()` level.
- `get_preferred_size_multiple` `Device::METAL` branch in `src/types.cc` (padding hint;
  currently returns 1).
- Consider enabling fp16 in the `AUTO` compute-type path once the full path supports it.

### Performance work (after correctness)

- Batch op encoding into fewer command buffers / reduce per-op synchronize.
- Offline `.metallib` compilation (faster startup than `newLibraryWithSource`), located
  at runtime via `dladdr()` with a source-compile fallback.
- Avoid the first-GEMM MPS pipeline warmup cost on the hot path.

### Deferred / out of scope for now

- Flash-attention (Metal equivalent of the CUDA flash-attn path)
- `Conv1D` (CUDA path uses cuDNN; needed for Whisper / Wav2Vec2 encoders)
- AWQ int4 GEMM
- bf16 (only on newer Apple GPUs)
- NCCL / tensor parallelism (multi-GPU)

## Gotchas for contributors

- **Don't add `Device::METAL` as a real `DEVICE_CASE`** in `device_dispatch.h` — it would
  force `primitives<Device::METAL>`/`compute<Device::METAL,T>` to be instantiated at ~50
  sites and break the link. Graduate ops via targeted routing instead.
- **Allocator/device-index early-returns are load-bearing** (see the invariant above).
- **All referenced Metal buffers must be bound** before dispatch — e.g. `metal::softmax`
  binds the input buffer as a dummy at the `lengths` index when there is no mask.
- **MPS is row-major** like `StorageView`; do not copy the cuBLAS column-major a/b swap.
- The `.mm` files use **manual retain/release (not ARC)**.
