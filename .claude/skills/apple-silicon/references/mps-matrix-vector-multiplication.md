---
title: "MPSMatrixVectorMultiplication (GEMV)"
summary: >-
  Reference for Apple's MPSMatrixVectorMultiplication (GEMV, y = alpha*op(A)*x
  + beta*y), sourced from DocC JSON plus the MPSMatrix.framework SDK header.
  Covers the initializers (full
  initWithDevice:transpose:rows:columns:alpha:beta: and the convenience no-
  transpose alpha=1 beta=0 form), the
  encodeToCommandBuffer:inputMatrix:inputVector:resultVector: call with its
  in/out resultVector contract, the MPSVector/MPSVectorDescriptor wrappers
  including strided batch-of-vectors descriptors, and the
  shape/transpose/scalar-fixed-at-init, operands-at-encode property that makes
  per-shape caching valid. Emphasizes that no integer matrix-vector multiply
  is documented anywhere (unlike MPSMatrixSoftMax/FindTopK which mandate
  float), so float32/float16 is the proven set and int8 lives in the custom
  ct2_gemv_s8 kernel. Frames MPSMatrixVectorMultiplication as the untried MPS-
  native alternative for the fp16 m=1 decode path currently riding the
  degenerate MPSMatrixMultiplication, and sketches a fair A/B against the
  current matrix kernel and the int8 GEMV floor.
semantic_id: "yhwQ9o8eu32CwetVTvEJvSd2q2fTi6olCldivNi95t5qH70OIa-tVy98Xp1xr6waDE0su-77045-NYzhDrBvpA"
---

Source: https://developer.apple.com/documentation/metalperformanceshaders/mpsmatrixvectormultiplication
(fetched via DocC JSON, 2026-06-11). The DocC pages for the MPSMatrix family have been
stripped to bare declarations; the discussion/constraint prose below comes from the
macOS SDK header `MPSMatrix.framework/Headers/MPSMatrixMultiplication.h` (CommandLineTools
SDK, same machine), which is the HeaderDoc source the old web docs rendered.

`MPSMatrixVectorMultiplication` computes a matrix-vector product on the GPU:

```
y = alpha * op(A) * x + beta * y
```

- `A` is an `MPSMatrix`, `x` and `y` are `MPSVector`s; `op(A)` is `A` or `Aᵀ`.
- alpha/beta are scalars "of the same data type as values of y".
- `y` is **in/out**: the encode call's `resultVector` is "the addend vector which will
  also be overwritten by the result" — with beta != 0 the previous contents are read.
- Available macOS 10.13+. Subclass of `MPSMatrixBinaryKernel`.

## Initializers

```objc
// Full: op(A) transpose, sizes, scalars
- initWithDevice:transpose:rows:columns:alpha:beta:
// Convenience: no transpose, alpha = 1.0, beta = 0.0
- initWithDevice:rows:columns:
```

| Parameter   | Meaning                                                               |
| ----------- | --------------------------------------------------------------------- |
| `transpose` | if YES, `op(A) == Aᵀ`                                                 |
| `rows`      | rows of **op(A)** = number of elements in `y`                         |
| `columns`   | columns of **op(A)** = number of elements in `x`                      |
| `alpha`     | product scale, passed as `double`, converted to the compute precision |
| `beta`      | scale on the initial `y` values, same double-to-precision conversion  |

Like the GEMM kernel, shape/transpose/scalars are fixed at **init**; operands arrive at
**encode** — so one object per shape is reusable and cacheable.

## Encoding

```objc
- encodeToCommandBuffer:inputMatrix:inputVector:resultVector:
```

Size contract (header): the matrix must hold `rows x columns` elements from
`primarySourceMatrixOrigin`; `x` must hold `columns` elements; `y` must hold `rows`
elements. Vector origins beyond `.x` must be zero.

## MPSVector / MPSVectorDescriptor

`MPSVector` wraps an `MTLBuffer` (`init(buffer:descriptor:)`, `init(buffer:offset:descriptor:)`
— offset variant for sub-views, like the GEMM path's `buffer_and_offset`).

```objc
+ vectorDescriptorWithLength:dataType:                       // one contiguous vector
+ vectorDescriptorWithLength:vectors:vectorBytes:dataType:   // array of vectors, strided
+ vectorBytesForLength:dataType:                             // recommended stride helper
```

The strided form describes `vectors` vectors of `length` elements spaced `vectorBytes`
apart — i.e. a batch of rows in one buffer.

## Data types — what the docs actually say

- `MPSVectorDescriptor.dataType` takes any `MPSDataType`, and that enum includes integer
  types (`int8`, `int16`, `uInt32`, …) — but that only describes **storage**.
- Neither the DocC pages nor the SDK header state a supported-dtype table for the GEMV
  **kernel** itself. Notably, `MPSMatrixSoftMax`/`MPSMatrixFindTopK` headers explicitly
  say "must be MPSDataTypeFloat32 or MPSDataTypeFloat16", while the GEMM/GEMV headers
  say nothing.
- **No integer matrix-vector multiply is documented anywhere on these pages.** Treat
  float32/float16 (the combination this repo has proven for `MPSMatrixMultiplication`)
  as the supported set; int8 via this kernel is unverified-and-undocumented. The
  documented quantized path in MPS lives in MPSNDArray, not MPSMatrix — see
  `mpsndarray.md`.

### Relevance to the CT2 Metal backend

- Today, decode-shaped fp32/fp16 GEMMs (m=1) ride the **matrix** kernel: `src/metal/gemm.mm`'s
  `gemm_impl` routes every shape through `MPSMatrixMultiplication` with no small-m branch
  (verified 2026-06-11). Only the **int8** path has a dedicated GEMV
  (`ct2_gemv_s8`, routed at `src/metal/primitives.mm` for Dense m≤8 — see
  `int8-gemv-simdgroup-decode.md`); that kernel exists precisely because MPS documents no
  int8 matmul in the MPSMatrix family.
- `MPSMatrixVectorMultiplication` is the **untried** MPS-native alternative for the fp16
  decode path (per-step lm_head and per-layer projections at m=1). It may have a leaner
  internal dispatch than a degenerate 1×n×k GEMM — or not; nothing in the docs says.
- A fair A/B (per `benchmarking-and-profiling.md`, measure before believing):
  - same shapes as the real decode profile (Qwen2.5-0.5B m=1 projections + lm_head),
    fp16, async per-op commit unchanged;
  - cache the kernel object per `{transpose, rows, columns, alpha, beta}` exactly like
    `cached_gemm()` — operands-at-encode makes that valid;
  - mind the in/out contract: CT2's `beta == 0` GEMMs are fine, but any `beta != 0` call
    reads `y`'s prior contents, same as the GEMM path;
  - compare against both the current MPS matrix-kernel time and the int8 GEMV floor
    (`dispatch-overlap-and-perf-model.md` for the per-op API floor that bounds any win).
- Plumbing cost is small: `gemm.mm` already resolves `(MTLBuffer, offset)` via
  `buffer_and_offset`; an m==1 branch building one `MPSMatrix` + two `MPSVector`s is
  symmetric with `encode_gemm`.
