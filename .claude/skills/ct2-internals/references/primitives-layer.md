---
title: "The primitives<Device> layer"
summary: >-
  The primitives<Device> layer — CT2's BLAS-like substrate below ops and above
  ISA kernels, a single templated struct of static methods operating on raw
  typed pointers plus element counts (no StorageView, shapes, or dtype
  dispatch). It surveys the families (fill/copy/convert, reductions,
  elementwise arithmetic with batch/depth/block broadcast, decoding helpers
  like penalize_previous_tokens, transpose, math/activations, and
  gemm/gemm_batch_strided/quantized-GEMM support), notes what is NOT here
  (softmax, norms, topk, quantize live as ops), and the
  cross_device_primitives::copy. It explains the specialization-per-device-
  with-explicit-instantiation pattern (the reason adding a Device to
  DEVICE_DISPATCH is expensive), the CPU two-level split into parallel_for +
  CPU_ISA_DISPATCH + kernels.cc, and CUDA's Thrust+cuBLAS implementation.
  Crucially there is no primitives<Device::METAL>: METAL_DEVICE_CASE binds
  D=CPU so generic dispatch runs CPU primitives on Metal-resident pointers,
  while GPU compute enters via separate metal:: entry points.
---

The "basic vector/matrix functions over raw arrays" level of the engine — below ops, above ISA kernels. CLAUDE.md names it; this documents it.

**Sources (all citations from real lines):**

- `include/ctranslate2/primitives.h` (the full interface — one struct)
- `src/cpu/primitives.cc` (CPU specializations + instantiations)
- `src/cuda/primitives.cu` (CUDA specializations + instantiations)

---

## 1. What lives at this level

`primitives.h:10-11` declares a single templated struct of static methods:

```cpp
template <Device D = Device::CPU>
struct primitives { ... };
```

Everything takes **raw typed pointers + element counts** — no `StorageView`, no shapes, no dtype dispatch. Surveying the header, the families are:

- **Fill / copy / convert** — `at`, `fill`, `strided_fill`, `indexed_fill`, `copy`, `convert` (dtype cast) (`primitives.h:13-26`).
- **Reductions** — `sum`, `mean`, `max_element`, `max`, `amax`, plus `logsumexp` (`primitives.h:28-42`, `168-169`).
- **Elementwise arithmetic** — `add`, `sub`, `mul`, `min`, `max` in scalar-and-array / array-and-array forms, plus `add_batch_broadcast`, `add_depth_broadcast`, `add_block_broadcast`, `mul_batch_broadcast` (`primitives.h:44-142`).
- **Decoding helpers** — `penalize_previous_tokens`, `prepare_length_mask` (`primitives.h:144-159`).
- **Layout** — `transpose_2d/3d/4d` (`primitives.h:161-166`).
- **Math & activations** — `exp`, `log`, `cos`, `sin`, `tanh`; `relu`, `gelu`, `gelu_tanh`, `gelu_sigmoid`, `sigmoid`, `swish` (`primitives.h:171-194`).
- **GEMM and quantized-GEMM support** — `compute_u8_compensation`, `gemm_pack_b`, `gemm`, `gemm_batch_strided` (`primitives.h:196-232`). The `gemm` signature carries `In`/`Out` types (int8→int32 etc.), packing flags, and the MKL `a_shift_compensation` pointer — the integer alpha/beta contract is documented in `gemm-op-and-dtype-dispatch.md`.

Notably **not** here: softmax, layer/rms norm, topk, quantize/dequantize. Those are ops with their own `compute<D,T>` kernels (`src/ops/*_cpu.cc` / `*_gpu.cu`); primitives is only the BLAS-like substrate (the header's own comment: "Low-level (BLAS-like) primitives", `primitives.h:1`).

### `cross_device_primitives`

`primitives.h:235-239` — a second tiny struct, `cross_device_primitives<D1, D2>::copy`, the only cross-device operation. Implemented for CPU↔CUDA as `cudaMemcpyAsync` H2D/D2H on the current stream (`src/cuda/primitives.cu:744-754`); used by `StorageView::copy_from` for device moves and by `primitives<CUDA>::at` to read one element back (`primitives.cu:40-46`).

## 2. The specialization-per-device pattern

There is no generic definition — each backend file provides `template<> template<typename T>` member definitions for its device and then **explicitly instantiates** them so the linker finds every `<D, T>` combination an op might dispatch to:

- **CPU** (`src/cpu/primitives.cc`): definitions throughout; instantiation macros at the bottom — `DECLARE_IMPL` × `DECLARE_ALL_TYPES` (`primitives.cc:1160-1218`) for every dtype, and `DECLARE_IMPL_NO_FLOAT` for int8/int16/int32/fp16/bf16 overloads whose float version is hand-specialized (`primitives.cc:1220-1240`). `convert` pairings are instantiated individually (`primitives.cc:77-82`: fp32↔fp16↔bf16).
- **CUDA** (`src/cuda/primitives.cu`): same shape — `DECLARE_ALL_TYPES(DECLARE_IMPL)` at `primitives.cu:830`, plus `DECLARE_FLOAT_IMPL` for float/fp16/bf16 activations and math (`primitives.cu:833-849`).

This explicit-instantiation discipline is _why_ adding a new `Device` enum case to `DEVICE_DISPATCH` is expensive: every dispatch site would demand a `primitives<NewDevice>` instantiation (see the Metal bridge).

## 3. CPU: the two-level split (primitives → kernels)

On CPU, a primitive is **orchestration**, and the ISA-vectorized inner loop lives in `src/cpu/kernels.cc`. The pattern, e.g. `gelu` (`src/cpu/primitives.cc:303-309`):

```cpp
void primitives<Device::CPU>::gelu(const float* x, float* y, dim_t size) {
  cpu::parallel_for(0, size, /*grain_size=*/512,
                    [x, y](dim_t begin, dim_t end) {
                      CPU_ISA_DISPATCH((cpu::gelu<ISA>(x + begin, y + begin, end - begin)));
                    });
}
```

Three layers in five lines: thread parallelism (`cpu::parallel_for`, see `parallelism-and-thread-config.md`), runtime ISA selection (`CPU_ISA_DISPATCH`, see `cpu-isa-dispatch-and-kernels.md`), and the vectorized kernel template `cpu::gelu<ISA>` declared in `src/cpu/kernels.h`. Simple memory-bound primitives skip kernels and use the STL directly (`fill`/`copy` = `std::fill`/`std::copy`, `primitives.cc:44-69`); reductions go straight to `cpu::reduce_sum<ISA>` etc. (`primitives.cc:86-112`).

**GEMM** is the exception — it delegates to an external backend, not to kernels.cc. The backend per compute type is resolved once into file-statics (`primitives.cc:543-545`: `sgemm_backend`, `gemm_s8_backend`, `gemm_s16_backend` via `cpu::get_gemm_backend`), then each `gemm` overload switches over it: float32 → MKL/DNNL/Accelerate/OpenBLAS/Ruy (`primitives.cc:644-792`), int8→int32 → MKL `cblas_gemm_s8u8s32` (with the uint8-shift compensation dance, `primitives.cc:903-1010`), DNNL, or Ruy. `gemm_batch_strided` uses MKL's batched API when available, else loops `gemm` inside a `parallel_for` (`primitives.cc:1077-1157`).

## 4. CUDA: Thrust + cuBLAS

`primitives<Device::CUDA>` members are implemented with Thrust iterators for fill/reductions (`THRUST_CALL(thrust::fill, ...)`, permutation iterators for `strided_fill`/`indexed_fill`, `primitives.cu:48-70`), `cudaMemcpyAsync` for `copy` (`primitives.cu:72-77`), `cuda::unary_transform` functors for math/activations (`primitives.cu:714-742`), and cuBLAS for GEMM — `cublasGemmEx` with `CUBLAS_COMPUTE_32I` for int8 (`primitives.cu:571-597`), `cublasGemmStridedBatchedEx` for fp16/bf16 batches (`primitives.cu:622-683`). Everything is enqueued on `cuda::get_cuda_stream()`; nothing here synchronizes. HIP builds reuse this file through `#define` aliases (`primitives.cu:3-26`).

## 5. How ops call primitives

An op's `compute<D, T>` body calls `primitives<D>::…` with the `D` bound by the dispatch macros — e.g. `Gemm` → `primitives<D>::gemm` (`src/ops/gemm.cc:172`), `BiasAdd` → `primitives<D>::add_batch_broadcast` (`src/ops/bias_add_cpu.cc:12`), `MatMul` → `primitives<D>::gemm_batch_strided` (`src/ops/matmul.cc:118`). `StorageView` itself uses them for `fill`/`convert`/`copy` (`storage_view.cc:100`, `399`, `422`). Rule of thumb when placing new code: **per-element/array math over raw pointers → primitives; anything needing shapes, masks, or op semantics → an op; a new CPU inner loop that should vectorize per-ISA → kernels.cc.**

### Relevance to the Metal backend

- There is **no `primitives<Device::METAL>`** — and that's load-bearing. `METAL_DEVICE_CASE` binds `constexpr Device D = Device::CPU` (`src/device_dispatch.h:34-47`), so generic dispatch runs `primitives<Device::CPU>` directly on Metal-resident pointers (valid under unified memory).
- Metal GPU compute enters through separate `metal::` entry points (`src/metal/primitives.h/.mm`, GEMM in `src/metal/gemm.mm`) reached by targeted routing at the op level, _not_ through this struct.
- The explicit-instantiation cost in §2 is the concrete reason Metal was kept off `DEVICE_DISPATCH` (~50 sites would each demand the specialization; the comment in `src/allocator.cc:9-11` records this).
- On this machine the CPU fallback path runs the NEON kernels under the two-level split in §3 — see `apple-silicon` skill for which ops have graduated to real GPU kernels.
