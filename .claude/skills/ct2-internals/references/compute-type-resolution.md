---
title: "Compute-type resolution"
summary: >-
  Traces how a requested ComputeType string (auto, int8, int8_float16,
  float16, etc.) becomes the effective per-weight dtypes a loaded model uses
  via resolve_compute_type in src/types.cc. Every model tracks three types —
  saved (inferred from serialized weights), requested, and effective — and
  resolution begins with the four mayiuse_bfloat16/float16/int16/int8 device-
  capability queries (on CPU backed by cpu::has_gemm_backend walking
  MKL/DNNL/Accelerate/OpenBLAS/Ruy in priority order). It tabulates the per-
  requested-type fallback chains, notably that plain int8 expands per the
  saved model's float dtype and throws rather than silently un-quantizing,
  that DEFAULT recurses on the saved type so any model loads anywhere, and
  that compute_type_to_data_type produces the (weight_dtype, float_dtype) pair
  converting quantizable weight variables. For Metal specifically the branch
  makes mayiuse_int8/float16 return true while AUTO deliberately pins to
  FLOAT32 so int8 stays an explicit opt-in; also covers the Python
  get_supported_compute_types surface.
semantic_id: "yho41ovuu22ygaVdyl1LMac4p2uTwqMpDhd2G9S15tQ6B80U7a-NWC0xas09y6cSBiwk88_zjd5e_Q2TDPxu4Q"
---

How a requested compute type (`"auto"`, `"int8"`, `"int8_float16"`, `"float16"`, …) becomes
the effective per-weight dtypes a loaded model actually uses.

**Sources (all citations from real lines):**

- `include/ctranslate2/types.h` (the `ComputeType` / `DataType` enums)
- `src/types.cc` (`resolve_compute_type`, the `mayiuse_*` capability queries, conversions)
- `include/ctranslate2/models/model.h` / `src/models/model.cc` (the three per-model types)
- `src/cpu/backend.cc` (`has_gemm_backend` — what backs the CPU capability answer)
- `python/cpp/module.cc`, `python/cpp/utils.h` (the Python surface)

---

## 1. Three compute types per model

Every loaded model tracks three (`model.h:71-81`, set in `Model::set_compute_type`,
`model.cc:176-188`):

- **saved** — inferred from the dtypes of the serialized weights (`infer_compute_type`,
  `model.cc:370-385`, via `data_type_to_compute_type`, `types.cc:339-360`).
- **requested** — what the user asked for (`ComputeType::DEFAULT` if nothing).
- **effective** — `resolve_compute_type(requested, saved, device, device_index)`
  (`model.cc:182-185`). This is the one that drives weight conversion.

The `ComputeType` enum (`types.h:28-39`) has two pseudo-values besides the concrete types:
`DEFAULT` ("keep what the model was saved with, fall back if unsupported") and `AUTO`
("pick the fastest supported type for this device"). String names map through
`str_to_compute_type` / `compute_type_to_str` (`types.cc:44-66`, `68-92`).

## 2. Device capability queries (`mayiuse_*`)

`resolve_compute_type` starts by asking four questions (`types.cc:176-179`):

- `mayiuse_bfloat16` (`types.cc:94-108`) — CUDA only, compute capability ≥ 8 (or
  `CT2_CUDA_ALLOW_BF16`).
- `mayiuse_float16` (`types.cc:110-132`) — CUDA fp16 tensor cores (or
  `CT2_CUDA_ALLOW_FP16`); a `Device::METAL` case returns true under `CT2_WITH_METAL`
  (`types.cc:121-128`).
- `mayiuse_int16` (`types.cc:134-141`) — CPU only, via `cpu::has_gemm_backend(INT16)`.
- `mayiuse_int8` (`types.cc:143-164`) — CUDA: `cuda::gpu_supports_int8`; CPU:
  `cpu::has_gemm_backend(INT8)`; Metal: true under `CT2_WITH_METAL` (`types.cc:154-160`,
  added by the int8-on-Metal branch).

On CPU the real answer is "is there a GEMM backend compiled in that can do this type":
`cpu::has_gemm_backend` (`backend.cc:92-94`) checks `get_gemm_backend`, which walks the
compiled backends in priority order (MKL → DNNL → Accelerate → OpenBLAS → Ruy,
`backend.cc:50-90`). So e.g. an Accelerate-only build has **no** int8 CPU GEMM unless Ruy
is also enabled.

## 3. The resolution logic (`resolve_compute_type`, `types.cc:171-316`)

A switch on the requested type; each case either returns a supported type, throws
`unsupported_compute_type` (`types.cc:166-169`) when `enable_fallback` is false, or walks
a fallback chain:

| requested              | resolution                                                                                                                                                                                                                                                   |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `FLOAT32`              | always itself (`types.cc:183-185`)                                                                                                                                                                                                                           |
| `FLOAT16` / `BFLOAT16` | itself if supported; else throw, or `FLOAT32` under fallback (`types.cc:187-201`)                                                                                                                                                                            |
| `INT16`                | itself; else `INT8_FLOAT32` (CPU) / `FLOAT16` (CUDA) / `FLOAT32` (`types.cc:203-213`)                                                                                                                                                                        |
| `INT8`                 | expands to `INT8_FLOAT32/16/BF16` based on the **saved model's** float dtype (`types.cc:215-229`), re-resolves that with fallback, then **throws if the resolved weight type is not int8** (`types.cc:237-241`) — plain `"int8"` never silently un-quantizes |
| `INT8_FLOAT32`         | itself; else `INT16` (CPU) / `FLOAT16` (CUDA) / `FLOAT32` (`types.cc:244-254`)                                                                                                                                                                               |
| `INT8_FLOAT16`         | itself; else `INT8_FLOAT32` → `FLOAT16` → `INT16` → `FLOAT32` (`types.cc:256-268`)                                                                                                                                                                           |
| `INT8_BFLOAT16`        | itself; else `INT8_FLOAT32` → `BFLOAT16` → `FLOAT32` (`types.cc:270-280`)                                                                                                                                                                                    |
| `AUTO`                 | per device: CUDA picks `INT8_FLOAT16` → `INT8_FLOAT32` → `FLOAT16`; others pick `INT8_FLOAT32` → `INT16`; else `FLOAT32` (`types.cc:282-303`). A `Device::METAL` early-return pins AUTO to `FLOAT32` (`types.cc:283-288`)                                    |
| `DEFAULT`              | recurse with the **saved** type and `enable_fallback=true` (`types.cc:305-313`) — any model loads anywhere, silently upgraded/downgraded as needed                                                                                                           |

When `DEFAULT` resolution had to change the type, `ModelLoader::load` logs a warning
("the model weights have been automatically converted…", `model.cc:881-889`). The chosen
type is also logged at every load ("Selected compute type", `model.cc:878-879`).

## 4. From compute type to per-weight dtypes

`compute_type_to_data_type` (`types.cc:318-337`) maps the effective type to a
**(weight_dtype, float_dtype)** pair — e.g. `INT8_FLOAT16 → (INT8, FLOAT16)`. In
`Model::set_compute_type` (`model.cc:190-231`):

- quantizable variables (names ending in `"weight"`, `is_quantizable`,
  `model.cc:287-289`) are converted to `weight_dtype`;
- other convertible float variables (biases, norms — not scalars, not `_scale`,
  `is_convertible`, `model.cc:299-301`) are cast to `float_dtype`.

The actual quantize/dequantize/cast mechanics are in
`weight-loading-and-conversion.md` (`ensure_dtype`), including the conv-weight guard that
keeps conv weights at `float_dtype` on devices without quantized convolution.

`get_preferred_size_multiple` (`types.cc:366-384`) is the other consumer of the effective
type: it returns the vocab-padding multiple (8/16 on CUDA tensor cores, else 1).

## 5. The Python surface

- `ctranslate2.Generator(path, device=..., compute_type=...)` (same for `Translator`,
  `Encoder`, `Whisper`): the `compute_type` argument is a string **or** a per-device dict,
  resolved by `ComputeTypeResolver` (`python/cpp/utils.h:28-49`) into
  `str_to_compute_type`, then stored on `ModelLoader.compute_type`
  (`python/cpp/replica_pool.h:59`; loader fields at `model.h:220-226`).
- `ctranslate2.get_supported_compute_types(device, device_index)` is built directly from
  the four `mayiuse_*` queries (`python/cpp/module.cc:13-45`): `float32` always, plus
  `int8`/`int8_float32` when int8 is supported, plus `int8_float16`/`int8_bfloat16` when
  the half types are also supported.
- The resolved choice is observable as `generator.compute_type`, which returns
  `effective_compute_type()` as a string (`python/cpp/replica_pool.h:85-87`).

### Relevance to the Metal backend

- This branch made `mayiuse_int8` and `mayiuse_float16` return true for `Device::METAL`
  (`types.cc:154-160`, `121-128`), so `get_supported_compute_types("metal")` now reports
  the int8 family and explicit `compute_type="int8*"` requests resolve instead of falling
  back. (The in-code comment at `types.cc:156-158` describing a "cast shim" predates the
  Phase 2 native int8 GEMM/GEMV kernels.)
- `AUTO` on Metal deliberately resolves to `FLOAT32` (`types.cc:283-288`): int8 must be an
  explicit opt-in there, so pre-existing "auto" users see no behavior change.
- Resolving an int8 type makes load-time quantization apply to **all** `"weight"`
  variables — including conv weights, which Metal cannot run quantized; the guard that
  exempts them is in `weight-loading-and-conversion.md`.
- Backend-side detail (which int8 kernels exist, the ALU ceiling, perf) is the
  `apple-silicon` skill.
