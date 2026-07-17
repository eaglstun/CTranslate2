---
title: "MPSNDArray — the modern n-D MPS API (and its quantized matmul)"
summary: >-
  Documents MPSNDArray, the modern up-to-16-dimensional successor to
  MPSMatrix, and critically its quantized matmul: MPSNDArrayDescriptor (the
  shape: slowest-first vs dimensionSizes: fastest-first factory gotcha),
  buffer aliasing via initWithBuffer:offset:descriptor: (macOS 15+ only, no
  zero-copy before that), MPSNDArrayMatrixMultiplication with native batch
  broadcasting, and MPSNDArrayQuantizedMatrixMultiplication with its
  Affine/LUT quantization schemes and int8/int4/int2 MPSDataType storage (all
  macOS 15+). The load-bearing conclusion is that MPS does ship a native
  quantized matmul since macOS 15 — so the premise MPS is float-only holds
  only for the MPSMatrix family — but the backend uses none of it (GEMM rides
  MPSMatrixMultiplication, int8 the hand-tiled ct2_gemm_s8), and it was
  deliberately not benchmarked because Metal-4 MPP matmul2d already ties MPS
  fp16 while staying int32-exact, plus MPS documents nothing about its
  accumulator exactness.
---

Sources: https://developer.apple.com/documentation/metalperformanceshaders/mpsndarray,
https://developer.apple.com/documentation/metalperformanceshaders/mpsndarraymatrixmultiplication,
https://developer.apple.com/documentation/metalperformanceshaders/mpsndarrayquantizedmatrixmultiplication
(fetched via DocC JSON, 2026-06-11; availability verified there). DocC strips the
discussion text, so the contract prose below is from the SDK headers
`MPSCore.framework/Headers/MPSNDArray.h` and `MPSNDArray.framework/Headers/`
(`MPSNDArrayMatrixMultiplication.h`, `MPSNDArrayQuantization.h`,
`MPSNDArrayQuantizedMatrixMultiplication.h`).

MPSNDArray (macOS 10.15+) is the n-dimensional successor to `MPSMatrix` for new MPS
work: up to **16 dimensions**, kernels in the `MPSNDArrayMultiaryKernel` family.

## MPSNDArrayDescriptor

```objc
+ descriptorWithDataType:shape:                      // NSArray<NSNumber*>, SLOWEST→fastest (numpy order)
+ descriptorWithDataType:dimensionCount:dimensionSizes:  // C array, FASTEST→slowest — reversed!
```

- `dataType` is any `MPSDataType`; product of all dimension lengths must be < 2^31.
- Mutators: `reshape…`, `transposeDimension:withDimension:`,
  `sliceDimension:withSubrange:` — views/permutes the description before kernel use.
- `preferPackedRows` (**macOS 15.0+**): pack rows instead of padded row strides.
- Watch the two factory orders — `shape:` is row-major-style slowest-first;
  `dimensionSizes:` is fastest-first. Easy to transpose a network by accident.

## MPSNDArray

- `initWithDevice:descriptor:` — MPS-owned storage (lazily allocated backing store).
- **`initWithBuffer:offset:descriptor:` — macOS 15.0+ only**: aliases an existing
  `MTLBuffer` at a byte offset (the analogue of `MPSMatrix initWithBuffer:` that the
  whole CT2 GEMM path depends on); `userBuffer` returns it. Before macOS 15 there is
  NO zero-copy from-buffer construction — only `importData`/`exportData` copies.
- `arrayViewWithShape:strides:` (macOS 15+) — strided reinterpretation; strides must be
  monotonic per the header (no arbitrary gather).
- Kernels encode via `MPSNDArrayMultiaryKernel`
  `encodeToCommandBuffer:sourceArrays:destinationArray:` (and a result-allocating
  variant).

## MPSNDArrayMatrixMultiplication (macOS 10.15+)

For each 2-D matrix in the two most-major dimensions of a (up to) 4-D array:

```
D = alpha * A * B + beta * C
```

- Batch semantics: dims 3/4 are batch dims; **an input whose 3rd/4th dim is 1 is
  broadcast** to the other operand's batch — native batched GEMM, unlike the per-matrix
  encode loop CT2 runs today.
- `alpha`/`beta` are mutable `double` properties (not init-frozen as in `MPSMatrixMultiplication`).
- No supported-dtype table is documented on the page or in the header.

## Quantization — the load-bearing answer (all macOS 15.0+ / iOS 18.0+)

Documented, verified on DocC 2026-06-11:

- `MPSDataType` itself enumerates `int8`, `uInt8`, `int4`, `uInt4`, `int2`, … — int8
  **storage** in an MPSNDArray is a documented type.
- `MPSNDArrayQuantizationScheme`: `None`, `Affine`
  (dequant `y = scale*(input − zeroPoint) + minValue`), `LUT` (`y = lut[input]`).
- `MPSNDArrayQuantizationDescriptor` (base; default dtype `MPSDataTypeUint8`) →
  `MPSNDArrayAffineQuantizationDescriptor` (`initWithDataType:hasZeroPoint:hasMinValue:`,
  plus `implicitZeroPoint` — Int4-only stored-as-unsigned trick) and
  `MPSNDArrayLUTQuantizationDescriptor` (scalar or per-axis vector LUT).
- **`MPSNDArrayQuantizedMatrixMultiplication`** (subclass of the matmul above):
  `initWithDevice:leftQuantizationDescriptor:rightQuantizationDescriptor:` — either side
  may be quantized or float. Encode inputs are a compacted array
  `[LHS, RHS, <LHS quant inputs>, <RHS quant inputs>]`, affine order
  `(quantized, scale, zeroPoint, minValue)`; scale/zeroPoint/minValue are themselves
  MPSNDArrays "with same transposes as quantized input" (so per-axis scales are shaped
  inputs, but the docs don't pin down which broadcast granularities are legal).
- Standalone dequant kernels: `MPSNDArrayLUTDequantize`, `MPSNDArrayVectorLUTDequantize`,
  `MPSNDArrayAffineInt4Dequantize`.
- What the docs do NOT say: performance, whether int8×int8 or only int8×float is
  hardware-fast, accumulator type/exactness, or supported scale shapes. None of that is
  on the fetched pages — measure, don't assume.

### Relevance to the CT2 Metal backend

- The backend uses none of this today: `grep -rn MPSNDArray src/` is empty (2026-06-11);
  GEMM rides `MPSMatrixMultiplication` (`src/metal/gemm.mm`), int8 rides the hand-tiled
  `ct2_gemm_s8`/`ct2_gemv_s8` (`src/metal/kernels/kernels_msl.h`, routed in
  `src/metal/primitives.mm`).
- **So MPS does, since macOS 15, ship a native quantized matmul** — the premise
  "MPS is float-only" (recorded in `int8-gemm-kernel-design.md` and the
  `mps-matrix-multiplication.md` card) is true for the MPSMatrix family but NOT for
  MPSNDArray on current OSes. That makes `MPSNDArrayQuantizedMatrixMultiplication` a
  legitimate benchmark candidate against `ct2_gemm_s8`'s ALU-bound large-m regime.
- Mapping CT2's scheme: per-row symmetric int8 (scales, no zero point, no minValue) →
  affine descriptor `hasZeroPoint:NO hasMinValue:NO` with a scale NDArray; whether MPS
  accepts CT2's per-output-row scale shape is undocumented — first thing the experiment
  must establish. Exactness differs too: CT2's contract is exact int32 accumulation
  (`int8-gemm-kernel-design.md`); MPS documents nothing about its accumulator, so parity
  must be re-validated, not assumed.
- **2026-06-11 verdict: deliberately not benchmarked.** Task 6 measured the other
  candidate first — Metal-4 MPP `matmul2d` (`metal4-tensors-and-mpp.md`) — and it ties
  MPS fp16 GEMM while staying int32-bit-exact, leaving no headroom for this float-output
  API to win and nothing it could add but pipeline restructuring plus an unresolved
  exactness question. Revisit only if the MPP path must be abandoned or a macOS 15–25
  deployment target ever matters (MPSNDArray quantized matmul is macOS 15+, MPP is 26+).
- Constraints: `initWithBuffer:` (zero-copy over CT2's MTLBuffers) and the quantized
  kernel are both macOS 15+ → needs an availability gate plus the existing kernel as
  fallback. Per the project's measure-first rule, the experiment is: same shapes as the
  Phase-2 benchmark tables (METAL_BENCHMARKS.md), one cached kernel object per shape,
  parity suite on `Device::METAL` before any timing claim.
