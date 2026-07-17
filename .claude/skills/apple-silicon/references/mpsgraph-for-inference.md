---
title: "MPSGraph for inference (one card) — and its quantization surface"
summary: >-
  MPSGraph, the symbolic whole-graph framework above MPS, and its quantization
  op surface, evaluated as a candidate for CT2 inference. It covers building
  graphs with placeholder and per-op tensor methods (matrixMultiplication,
  softMax, topK with no documented k limit, scaledDotProductAttention, and erf
  which MSL lacks), executing against real MTLBuffers via MPSGraphTensorData,
  and the production compile path to a cached MPSGraphExecutable that can
  encode(to:) an MPSCommandBuffer. The verified quantization facts:
  quantize/dequantize (macOS 13.1+, i8/u8, per-axis scaleTensor and LUT
  variants exist) but there is NO quantized/integer matmul op, only
  dequantize-then-float-matmul. CT2 uses no MPSGraph anywhere and is
  structurally mismatched because CT2 itself is the imperative graph executor;
  the only realistic experiment is wrapping one cached weight-stationary
  quantized matmul.
semantic_id: "yx0w5o8vu-2S8-1Nx80LtCcwp2uTzq8lCtf7i9T_wvoiCXwGZK-tXY-4Wplvr5QvjG8ku93xzY5ONQ3zxLRvpQ"
---

Source: https://developer.apple.com/documentation/metalperformanceshadersgraph/mpsgraph
(fetched via DocC JSON, 2026-06-11; op pages fetched individually — unlike the MPSMatrix
family, the MPSGraph DocC pages still carry full parameter/discussion prose).

`MPSGraph` (macOS 11+) is "the optimized representation of a compute graph of operations
and tensors" — a symbolic, whole-graph framework layered above MPS.

## Build

- `placeholder(shape:dataType:name:)` → `MPSGraphTensor` (nil shape = unranked).
- Every op is a method on the graph returning new tensors. The inventory is huge and
  inference-relevant: `matrixMultiplication(primary:secondary:name:)` (broadcasting,
  macOS 11+), `softMax(with:axis:name:)`, `topK(_:k:name:)` (no documented k limit,
  unlike `MPSMatrixFindTopK`'s 16), `scaledDotProductAttention(query:key:value:mask:scale:)`,
  `convolution2D(...)`, and — notably absent from MSL — `erf(with:name:)`.

## Execute against MTLBuffers

- `MPSGraphTensorData` bridges real storage: `init(MTLBuffer:shape:dataType:)`
  (+ `rowBytes:` variant), plus `init(MPSMatrix:)`, `init(MPSVector:)`,
  `init(MPSNDArray:)`. "A reference will be taken to your data and used just in time
  when the graph is run."
- One-shot: `run(feeds:targetTensors:targetOperations:)` with a
  placeholder→tensor-data dictionary (synchronous; `runAsync...` variants exist).
- Production path: `compile(with:feeds:targetTensors:targetOperations:compilationDescriptor:)`
  → **`MPSGraphExecutable`** (macOS 12+), "a compiled graph for specific feeds/targets";
  it can `specialize(...)`, `run`/`runAsync(on:inputs:results:executionDescriptor:)`, and
  critically **`encode(to:inputs:results:executionDescriptor:)`** onto an
  `MPSCommandBuffer` — i.e. it can be slotted into an existing command-buffer stream
  instead of owning execution.

## Quantization surface — exactly what exists (verified per-op)

- `quantize(_:scale:zeroPoint:dataType:name:)` — **macOS 13.1+**:
  `result = (tensor / scale) + zeroPoint`, "convert the float to an **i8 or u8**
  tensor"; `dataType` is the integer result type.
- `dequantize(_:scale:zeroPoint:dataType:name:)` — macOS 13.1+: inverse,
  `result = scale(tensor − zeroPoint)`, "convert the i8 or u8 to a float tensor".
- Tensor-valued / per-axis variants (from the op inventory):
  `quantize(_:scaleTensor:zeroPoint:dataType:axis:name:)`,
  `quantize(_:scaleTensor:zeroPointTensor:dataType:axis:name:)`, and matching
  `dequantize` forms (incl. `scaleTensor`-only) — per-axis scales/zero-points as tensors.
- LUT dequantization: `dequantize(_:LUTTensor:name:)` and `dequantize(_:LUTTensor:axis:name:)`.
- **There is NO quantized/integer matmul op.** `matrixMultiplication(primary:secondary:name:)`
  is the only matmul on the class page; no `quantizedMatrixMultiplication` exists in the
  MPSGraph reference map (checked 2026-06-11). The expressible pattern is
  `dequantize → matrixMultiplication`; whether the compiler fuses that into an
  int8-reading kernel is **not documented** — do not claim it either way without
  measuring. (A true documented quantized-matmul kernel exists one level down in
  MPSNDArray, macOS 15+ — see `mpsndarray.md`.)

### Relevance to the CT2 Metal backend

- CT2 does not use MPSGraph anywhere (`grep -rn MPSGraph src/` is empty, 2026-06-11) —
  and structurally it's a mismatch: **CT2 IS the graph executor.** Ops are imperative
  C++ calls over `StorageView`s (`src/ops/`), each `metal::` op encoding into its own
  async command buffer (`src/metal/device.mm`); MPSGraph wants to own the graph,
  compilation, and scheduling. Adopting it per-op means paying graph-build +
  compile per call unless the compiled executable is cached.
- The realistic experiment is therefore narrow: **wrap one weight-stationary quantized
  matmul** — `placeholder(activations) → [dequantize(weights_i8, scaleTensor, axis)] →
matrixMultiplication` — compile once per `{shape, dtype}` into a cached
  `MPSGraphExecutable`, and `encode(to:...)` it into the backend's command-buffer flow
  (an `MPSCommandBuffer` wraps an `MTLCommandBuffer`). Compare against `ct2_gemm_s8` /
  fp16 MPS GEMM at the Phase-2 benchmark shapes.
- Two honest caveats before anyone starts: (1) the win condition is unclear — MPSGraph's
  int8 story is dequantize-then-float-matmul unless its compiler proves otherwise, which
  is exactly the Phase-1 shim this project already built and discarded
  (`int8-gemm-kernel-design.md`'s graveyard); (2) executable caching must respect the
  per-op CPU/GPU overlap model (`dispatch-overlap-and-perf-model.md`) — a synchronous
  `run` call would reintroduce the per-op wait the backend removed. Measure-first rule
  applies (`benchmarking-and-profiling.md`); record results in the docs immediately.
- Side note: the op inventory (native `topK`, masked `scaledDotProductAttention`, `erf`)
  makes MPSGraph a useful _reference oracle_ for future kernels even if it never ships
  in the hot path.
