---
title: "Metal 4 tensors & Metal Performance Primitives — is there a supported int8 matmul?"
summary: >-
  Answers whether Metal 4 offers a supported int8 matmul: yes — MSL §7.2.1
  Table 7.3 lists char×char→int as a base-Metal-4 matmul2d combination, CT2's
  exact int8×int8→int32 contract. It covers the host-side MTLTensor (zero-copy
  MTLBuffer wrapping), MTLTensorDataType (int8/int4/fp8/bfloat16 storage), the
  MTL4MachineLearningCommandEncoder network surface, and the shader-side MSL
  tensor/cooperative_tensor types plus mpp::tensor_ops::matmul2d with dynamic
  K and SIMD-group execution scopes. The Task-6 measurement (M4 Max,
  2026-06-11) resolved the open questions: char×char→int matmul2d ties MPS
  fp16 GEMM (2048³ 1.51ms, 4.8× over the hand-tiled kernel), is int32-bit-
  exact, compiles via newLibraryWithSource, but requires 2 cooperating SIMD-
  groups (not Apple's 4) and exact-type dispatch. It is the documented
  successor to the hand-tiled ct2_gemm_s8.
semantic_id: "ylo0wo2Oq33Dw-l9R8QJPAc4NSuTyCIhLlf0nZTV5t4KHX0GxaarXa10bh9sj-4aDC0kw8zzjYZu9QWhzLx_pQ"
---

Sources: Apple DocC JSON (fetched 2026-06-11):
<https://developer.apple.com/documentation/metal/mtltensor>,
<https://developer.apple.com/documentation/metal/mtltensordatatype>,
<https://developer.apple.com/documentation/metal/mtl4machinelearningcommandencoder>;
MSL spec v4.1 (2026-06-04) §2.22 (Tensor Types) and §7 (Metal Performance Primitives),
<https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf>.
Everything below is what those sources state; gaps are flagged. The backend uses none of
this today.

**THE answer this file exists for: YES — MSL §7.2.1 Table 7.3 lists `char × char → int`
as a supported `matmul2d` combination at base Metal 4** (macOS 26 per the DocC platform
stamps; this repo's dev box runs macOS 26.4.1). Details and the honest caveats below.

## MTLTensor — the host-side resource [Apple doc, macOS 26.0+]

`protocol MTLTensor : MTLResource` — "a multi-dimensional array that you can use with
machine learning workloads." Properties: `dataType`, `dimensions`, `strides`, `usage`,
`gpuResourceID`, and crucially `buffer`/`bufferOffset` — a tensor **can share storage
with an existing `MTLBuffer`** (nil/zero when it doesn't), i.e. a zero-copy wrap of
already-resident weights. `getBytes`/`replace` copy slices CPU-side. Created via
`MTLTensorDescriptor` (separate page, same macOS 26 stamp).

`MTLTensorDataType` enumerates: **int8**, uint8, int4, uint4, int2, uint2, int16/uint16,
int32/uint32, float16, **bfloat16**, float32, plus fp8 (e4m3, e5m2, ue8m0) and fp4
(e2m1). So int8 _storage_ is first-class.

## MTL4 machine-learning encoder [Apple doc, macOS 26.0+]

Metal 4 command buffers add `MTL4MachineLearningCommandEncoder`: `setPipelineState(_:)`,
`setArgumentTable(_:)`, `dispatchNetwork(intermediatesHeap:)` — it dispatches a whole
pre-built ML _network_ as a pass, with a heap for intermediates. This is the
graph/network surface (closer to "run a CoreML-style net inside Metal" — how the network
pipeline state is built is on pages not fetched here); it is NOT the per-op matmul
surface CT2 would want. That one is shader-side, below.

## MSL §2.22: `tensor<>` in shaders [spec, Metal 4+]

- `tensor<ElementType, Extents, DescriptorType, Tags…>` — a **non-owning view**; element
  value types: half, bfloat, float, **char, uchar**, short, ushort, int, uint (Metal 4.1
  adds packed "format types" — int4b/uint4b/int2b/uint2b, fp8/fp4 — with 128-byte
  alignment and block-multiple extent rules, Table 2.24).
- **Shader-allocated tensors (§2.22.2.7): a kernel can construct a `tensor_inline` view
  directly from a raw `device`/`threadgroup`/`constant` pointer + extents (+ strides)** —
  no host-side `MTLTensor` object needed. This is the load-bearing fact for retrofitting:
  existing `MTLBuffer`-bound kernel args can be wrapped in-shader.
- `cooperative_tensor<>` (§2.22.3): thread-local storage pre-partitioned across
  participating threads — the accumulator form for matmul.

## MSL §7: Metal Performance Primitives [spec, Metal 4+]

"All OS: Metal 4 and later." Header `<MetalPerformancePrimitives/MetalPerformancePrimitives.h>`,
namespace `mpp::tensor_ops`, "tuned for Apple silicon GPUs"; for supported GPU families
the spec defers to the Metal Feature Set Tables (not stated inline — **flag: hardware
floor unverified here**, only the OS floor is documented).

- Execution scopes (§7.1): `execution_thread`, `execution_simdgroup`,
  `execution_simdgroups<N>` — a matmul executed cooperatively by N SIMD-groups. All
  threads in scope must call `run` (uniform-control-flow rule); barrier before reading
  device/threadgroup results.
- `matmul2d_descriptor(M, N, K, transpose_left, transpose_right, relaxed_precision,
mode::multiply|multiply_accumulate)`; `K = dynamic_length_v<int>` is the default —
  **dynamic K is in the API**. Instantiate `matmul2d<Desc, Scope>`, call
  `run(A, B, C)` on tensors (C may be a `cooperative_tensor`). `convolution2d` likewise
  (§7.2.2).

### Table 7.3 — the dtype combinations that matter to CT2 [spec, verbatim subset]

| A      | B               | C            | Since              |
| ------ | --------------- | ------------ | ------------------ |
| char   | char            | **int**      | **Metal 4** (base) |
| half   | half            | half/float   | Metal 4            |
| char   | half            | half/float   | Metal 4            |
| bfloat | bfloat          | bfloat/float | Metal 4 + OS 26.1  |
| char   | int4b_format    | int          | Metal 4 + OS 26.4  |
| half   | fp8/fp4 formats | half/float   | Metal 4.1          |

(Full table also covers uchar at OS 26.4 and int2b at Metal 4.1.) `char×char→int` —
int8×int8 with an int32 destination, CT2's exact contract — is base-Metal-4. The int4
weight path (`char/half/bfloat × int4b_format`) is OS 26.4+.

## What the docs did NOT say — ANSWERED BY MEASUREMENT 2026-06-11 (M4 Max)

The flagged unknowns below were all resolved by the Task-6 experiment
(`experiments/mpp_matmul2d_proto.mm` / `mpp_matmul2d_tune.mm`; integrated as
`ct2_mpp_gemm_s8_nt` in `src/metal/kernels/kernels_mpp_msl.h`; numbers in
`METAL_BENCHMARKS.md`):

- **Performance: YES, dramatically.** char×char→int `matmul2d` ties MPS fp16 GEMM
  (2048³: 1.51 ms ≈ fp16's 1.49; ~11.4 T-eff-FLOPS) — 4.8× over the hand-tiled ALU
  kernel. The win REQUIRES tuning: **2 cooperating SIMD-groups** (Apple's 4-SG header
  example is 2–5× slower on every shape), 16×64 tiles, and the interior/edge
  static-extent `slice<Extents...>` split.
- **Accumulator: int32-bit-exact**, verified vs a host triple loop at k=2048 over the
  full int8 range plus saturated inputs. `mode::multiply` overwrites C (no read).
- **Runtime compile: works** via `newLibraryWithSource` with
  `options.languageVersion = MTLLanguageVersion4_0` and
  `#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>` — no Metal-4
  pipeline/encoder machinery needed; classic compute encoder + `setBuffer` suffice
  (inline tensors wrap the raw pointers in-shader).
- **Gotchas:** MPP's dispatch matches element types EXACTLY — `int8_t`/`int32_t`,
  non-const (`char` or `const int8_t` → "Unsupported type" static_assert). The
  stdlib spells the header comment's `static_slice` as `slice<Extents...>`.
  `relaxed_precision` remains float-only.
- **Hardware floor.** DocC stamps say OS 26.0; which `MTLGPUFamily` supports MPP is
  delegated to the Feature Set Tables (not fetched — PDF, not DocC).
- Whether `matmul2d` is callable from a _classic_ (pre-MTL4) compute pipeline compiled at
  runtime via `newLibraryWithSource` (the backend's path, `pipeline-and-library-compilation.md`)
  vs requiring Metal-4 compilation — not established by the fetched pages.

### Relevance to the CT2 Metal backend

- This is the documented successor question to `ct2_gemm_s8`
  (`src/metal/kernels/kernels_msl.h`): the hand-tiled kernel exists because MPSMatrix is
  float-only and `simdgroup_matrix` has no int8 (`int8-gemm-kernel-design.md`). Two
  candidates now exist: `MPSNDArrayQuantizedMatrixMultiplication` (macOS 15+,
  `mpsndarray.md`) and MSL-4 `mpp::tensor_ops::matmul2d` char×char→int (macOS 26+).
- The cheapest experiment is shader-side: §2.22.2.7 inline tensors wrap the _existing_
  buffer arguments of `ct2_gemm_s8`'s entry point, so a `matmul2d` variant could slot in
  behind the same routing in `src/metal/primitives.mm` (`metal::gemm_s8`) with the
  hand-tiled kernel as fallback — gated on language version, parity-suite first
  (`tests/metal_test.cc`, keep `Int8GemmDeepAccumulatorMatchesHostReference` green),
  then the Phase-2 benchmark shapes (`METAL_BENCHMARKS.md`).
- The target regime is the **ALU-bound large-m tiled case** (int8 7.28 ms vs fp16
  1.48 ms at 2048³) — the GEMV decode win is bandwidth-bound and already beats fp16; it
  has nothing to gain unless MPP also lifts bandwidth efficiency.
- OS floor: macOS 26 — fine for the M4 Max dev box (26.4.1), but an availability gate +
  fallback is mandatory for the library (`mtlgpufamily-and-feature-availability.md`).
