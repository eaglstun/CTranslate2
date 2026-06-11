# The CUDA Backend's Structure

The CUDA backend as the reference implementation for any new GPU backend — what lives
where, the handle/stream model, the cuBLAS GEMM wrapping, and how its op files plug into
the dispatch macros. This is THE template the int8-Metal work mirrored.

**Sources (read these, all citations below are from real lines):**

- `src/cuda/utils.h` / `src/cuda/utils.cc` — handles, streams, capability queries
- `src/cuda/primitives.cu` — `primitives<Device::CUDA>` incl. the cuBLAS GEMM wrappers
- `src/cuda/allocator.cc`, `src/cuda/helpers.h`
- `src/device_dispatch.h`, `src/ops/gemm.cc` (dispatch + the Metal mirror sites)

---

## 1. File layout — shared infra in `src/cuda/`, per-op kernels in `src/ops/*_gpu.cu`

A common misconception: the per-op CUDA kernels do **not** live in `src/cuda/`. That
directory holds only the **shared backend infrastructure**; each op's kernels sit next to
the op in `src/ops/<op>_gpu.cu` (e.g. `softmax_gpu.cu`, `quantize_gpu.cu`), per the 4-file
op pattern (`dispatch-and-op-implementation.md` §1).

`src/cuda/` contents:

- `primitives.cu` — the `primitives<Device::CUDA>` specializations (fill/copy/reductions
  via Thrust, transpose kernels, and all the GEMM wrappers, §3).
- `utils.{h,cc}` — error-check macros (`CUDA_CHECK`/`CUBLAS_CHECK`/`CUDNN_CHECK`,
  `utils.h:68-90`), per-thread stream/handle caching (§2), GPU capability queries
  (`gpu_supports_int8` = compute capability ≥ 6.1, `src/cuda/utils.cc:225-228`;
  `gpu_has_int8_tensor_cores` ≥ 7.2, `:230-233`; `gpu_has_fp16_tensor_cores` ≥ 7.0,
  `:235-238` — these feed `mayiuse_*` in compute-type resolution).
- `allocator.cc` — the two `Allocator`s behind `get_allocator<Device::CUDA>()`
  (`src/cuda/allocator.cc:173-185`): cub `CachingDeviceAllocator` (bin growth 4 / min 3 / max 12,
  200MB cap, `CT2_CUDA_CACHING_ALLOCATOR_CONFIG` env, `:36-58`; one **per thread**,
  `:178-180`) vs `cudaMallocAsync` (`:79-123`), selected by `CT2_CUDA_ALLOCATOR`
  (`:144-169`). Details in `allocators-and-caching.md`.
- `helpers.h` — kernel-side conveniences: `index_t` (32-bit for perf, `helpers.h:39`),
  `max_threads = 1024` / `max_blocks` (`:41-42`), host↔device type mapping
  (`DeviceType<float16_t> = __half`, `:44-57`), and Thrust-backed
  `unary_transform`/`binary_transform`/`permute` (`:75-107`).
- `random.{h,cu}` — curand state; `cublas_stub.cc`/`nccl_stub.cc`/`mpi_stub.cc` — `dlopen`
  stubs for `CUDA_DYNAMIC_LOADING`/tensor-parallel builds.

AMD HIP is **not** a separate backend: `utils.h:5-29` remaps the cuda/cublas symbols to
hip/hipblas and the same `.cu` sources compile for ROCm (contrast: Metal is a genuinely
separate backend; see CLAUDE.md).

## 2. Per-thread streams and handles

The model is **one stream + one cuBLAS/cuDNN handle per host thread**, created lazily and
cached `thread_local` (comment at `utils.cc:119-120`):

- `get_cuda_stream()` (`utils.cc:122-125`) — `static thread_local CudaStream`. The main
  thread gets the **default stream** (it loads replicas on multiple devices); every other
  thread gets its own `cudaStreamCreate`d stream bound to its current device
  (`utils.cc:70-98`).
- `get_cublas_handle()` (`utils.cc:127-130`) — `static thread_local CublasHandle`; the
  ctor creates the handle and binds it to the thread's stream (`cublasSetStream`,
  `utils.cc:100-117`). `get_cudnn_handle()` is identical (`:133-155`).
- Destruction re-enters the creating device via `ScopedDeviceSetter`
  (`include/ctranslate2/devices.h:33-54` — RAII save/`set_device_index`/restore), because
  CUDA's current-device is thread-global state.
- `get_device_properties` keeps a `thread_local` per-device cache (`utils.cc:187-203`).

Since each replica worker is a thread (`parallelism-and-thread-config.md`), this gives
each replica an independent stream with zero locking.

## 3. The cuBLAS GEMM wrappers — `cublasGemmEx` dtype combos

`primitives<Device::CUDA>::gemm` has one explicit specialization per dtype combo in
`src/cuda/primitives.cu`. All swap A/B because cuBLAS is column-major (comment `:496`):

| In → Out         | Call           | dtype/compute enums                                                                                                                                                 | Cite                    |
| ---------------- | -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------- |
| f32 → f32        | `cublasSgemm`  | —                                                                                                                                                                   | `primitives.cu:487-506` |
| f16 → f16        | `cublasGemmEx` | `CUDA_R_16F`, compute `CUBLAS_COMPUTE_16F` — or `32F` with float alpha/beta when `use_true_fp16_gemm()` is off (`CT2_CUDA_TRUE_FP16_GEMM`, `src/cuda/utils.cc:260`) | `primitives.cu:510-544` |
| bf16 → bf16      | `cublasGemmEx` | `CUDA_R_16BF`, compute `CUBLAS_COMPUTE_32F`                                                                                                                         | `primitives.cu:548-569` |
| **int8 → int32** | `cublasGemmEx` | operands `CUDA_R_8I`, output `CUDA_R_32I`, compute `CUBLAS_COMPUTE_32I`; alpha/beta **truncated to int32** (`:582-583`)                                             | `primitives.cu:573-597` |

The int8 wrapper's `a_shift_compensation` parameter is **unnamed and ignored**
(`primitives.cu:581`) — cuBLAS is natively signed s8s8s32, so the MKL u8-shift story
(`gemm-op-and-dtype-dispatch.md` §5) simply doesn't exist here. Scaling/bias/activation
happen **after** the GEMM in `Dequantize`'s fused epilogue kernel
(`src/ops/dequantize_gpu.cu:40-52`).

## 4. Thrust / CUB usage points

- `THRUST_CALL(FUN, ...)` (`src/cuda/utils.h:136-141`) runs any Thrust algorithm on the calling
  thread's stream with `par_nosync`. Used for fill/copy/reduce/max_element in
  `primitives.cu` (`:51-133`), the `unary/binary_transform` helpers (`helpers.h:75-107`),
  and gather/scatter-style ops (`src/ops/gather_gpu.cu:49`, `tile_gpu.cu:50-60`,
  `concat_split_slide_gpu.cu:91`).
- CUB appears two ways: `cub::BlockReduce` inside hand-written kernels
  (`src/ops/layer_norm_gpu.cu:180`, `rms_norm_gpu.cu:26`, `mean_gpu.cu:26`,
  `topk_gpu.cu:187`), and `cub::CachingDeviceAllocator` in `src/cuda/allocator.cc:55-58`.
- Headers come from the vendored `third_party/thrust` (NVIDIA CCCL) submodule.

## 5. How `_gpu.cu` files meet the dispatch macros

There is no `CUDA_CASE`/`GPU_CASE` macro — the real names are `DEVICE_CASE` and
`DEVICE_DISPATCH` (`src/device_dispatch.h:17-22`, `:49-63`). When `CT2_WITH_CUDA` is
defined, `DEVICE_DISPATCH` includes `DEVICE_CASE(Device::CUDA, ...)`
(`device_dispatch.h:57-62`), binding `constexpr Device D = Device::CUDA` so the op's
`.cc` instantiates `compute<Device::CUDA, T>` — whose definition + explicit instantiation
live in that op's `_gpu.cu`. `DEVICE_AND_FLOAT_DISPATCH` additionally forces fp16/bf16 to
`D = Device::CUDA` (`src/dispatch.h:25-41`). So CUDA participates as a **real device case**
selected by the generic switch — unlike Metal's operator()-level targeted routing
(`dispatch-and-op-implementation.md` §3).

---

### Relevance to the Metal backend

"Mirror the CUDA path" was the int8-Metal design rule. The three load-bearing properties
mirrored, with cites on both sides:

- **Signed int8 operands, no shift compensation** — CUDA: natively s8s8s32, compensation
  param ignored (`src/cuda/primitives.cu:577-581`). Metal: `metal::gemm_s8` takes signed
  `int8_t*` and the route requires `!a_shift_compensation` (`src/ops/gemm.cc:113`); the
  u8-shift is left as the CPU-GEMM-backend quirk it is.
- **int32 accumulation with integer alpha/beta** — CUDA: `CUBLAS_COMPUTE_32I`, alpha/beta
  truncated to int (`primitives.cu:582-595`). Metal: `c.data<int32_t>()` with
  `static_cast<int32_t>(alpha)` and a beta==0 + integral-alpha guard
  (`src/ops/gemm.cc:92-94`, `:113-114`).
- **Dequantize-after as a separate fused epilogue** — CUDA: scales+bias+activation in the
  `dequantize_gemm_output` kernel (`src/ops/dequantize_gpu.cu:40-52`). Metal:
  `metal::dequantize_gemm_output_s8` (`src/metal/primitives.h:117-121`), same contract.
- The structural difference: CUDA is a real `DEVICE_CASE` with `primitives<Device::CUDA>`
  instantiated everywhere; Metal deliberately is not (it routes per-op and falls back to
  the CPU reference). Per-op kernel specifics: the `apple-silicon` skill.
