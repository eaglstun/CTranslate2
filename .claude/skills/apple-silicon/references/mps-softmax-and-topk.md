---
title: "MPSMatrixSoftMax / MPSMatrixLogSoftMax / MPSMatrixFindTopK"
summary: >-
  MPSMatrixSoftMax, MPSMatrixLogSoftMax, and MPSMatrixFindTopK: row-wise
  softmax/log-softmax over an MPSMatrix (fp32 or fp16, matching input/result,
  no masking or per-row lengths, always reduces the full sourceColumns width)
  and per-row top-k with the hard constraint numberOfTopKValues <= 16, UInt32
  index matrix, and indexOffset. The key CT2 finding is that the backend
  deliberately does NOT use MPS softmax because its custom
  ct2_softmax_float/ct2_softmax_half kernels take a lengths buffer for masked
  attention softmax that MPS cannot express, whereas TopK/sampling (currently
  CPU-reference) is the real graduation opportunity, gated by the k<=16 limit:
  greedy and beam sizes fit but k=40-style top-k sampling does not. Lays out
  the four-step graduation path including the UInt32-to-int32 index shim.
---

Sources: https://developer.apple.com/documentation/metalperformanceshaders/mpsmatrixsoftmax,
https://developer.apple.com/documentation/metalperformanceshaders/mpsmatrixlogsoftmax,
https://developer.apple.com/documentation/metalperformanceshaders/mpsmatrixfindtopk
(fetched via DocC JSON, 2026-06-11). DocC has stripped these classes to bare
declarations; constraint prose below is from the SDK headers
`MPSMatrix.framework/Headers/MPSMatrixSoftMax.h` / `MPSMatrixFindTopK.h`.

## MPSMatrixSoftMax (+ MPSMatrixLogSoftMax)

Row-wise softmax over an `MPSMatrix` (macOS 10.13+):

```
B_ij = exp(A_ij) / Sum_k exp(A_ik)     // sum runs over column indices
```

`MPSMatrixLogSoftMax` is the same kernel shape computing `ln` of that.

```objc
- initWithDevice:                                  // no shape params at init
- encodeToCommandBuffer:inputMatrix:resultMatrix:
```

- `sourceRows` / `sourceColumns` properties (default `NSUIntegerMax`) clip to the source
  at encode time; set them before encoding for a sub-rectangle.
- Batch: `batchStart` / `batchSize` (inherited matrix-kernel batching).
- **Dtypes (header, encode discussion): input and result "must match and be either
  MPSDataTypeFloat32 or MPSDataTypeFloat16."** No masking, no per-row lengths — the
  reduction always runs over the full `sourceColumns` width.

## MPSMatrixFindTopK

Per-row top-k values + indices (macOS 10.13.4+):

```objc
- initWithDevice:numberOfTopKValues:
- encodeToCommandBuffer:inputMatrix:resultIndexMatrix:resultValueMatrix:
```

- Operates "independently on the rows and matrices in batch of the source matrix";
  results are `sourceRows x numberOfTopKValues` matrices of values and indices.
- **`numberOfTopKValues` must be ≤ 16; requesting more is undefined behavior** (header,
  stated on both the property and the initializer).
- `resultIndexMatrix` **must have dataType `MPSDataTypeUInt32`**; `inputMatrix` and
  `resultValueMatrix` must match and be **Float32 or Float16** (header, encode
  discussion).
- `indexOffset` (default 0) is added to every written index — for recovering
  original-matrix column indices when encoding from a `sourceMatrixOrigin.y` offset.
- `sourceRows` / `sourceColumns` clip like the softmax kernel; `batchStart`/`batchSize`
  batch it.

### Relevance to the CT2 Metal backend

- **Softmax: CT2 deliberately does NOT use MPS here.** `ct2_softmax_float` /
  `ct2_softmax_half` in `src/metal/kernels/kernels_msl.h` are custom kernels, and the
  reason is visible in their signature: a `lengths [[buffer(2)]]` input gives each row
  its own valid width (`size = lengths[row]`), the masked tail `[size, depth)` is zeroed,
  and the reduction runs only over `[0, size)` — that's CT2's masked attention softmax
  (`SoftMax(x, lengths)`), which `MPSMatrixSoftMax` cannot express (it always reduces the
  full row). The same kernel also folds in the log-softmax variant (`is_log`) and keeps
  bit-parity with the CPU reduction order — see `math-functions-and-numeric-parity.md`
  for why that parity is fragile under fast math. Don't "simplify" softmax onto MPS; the
  masking variant is the common case in attention.
- **TopK / sampling is the actual opportunity.** METAL_BACKEND.md lists sampling
  (`TopK`, `TopPMask`, `Multinomial`) as CPU-reference today and names them first under
  "Near term — graduate more ops"; `src/ops/topk.cc` runs a comparison-based CPU kernel
  (with a Metal fp16 carve-out that still executes on CPU). Meanwhile the logits are
  already GPU-resident right after the lm_head GEMM — every decode step round-trips them
  through unified memory for a CPU top-k.
- Fit check against the k ≤ 16 limit:
  - greedy (k=1) and typical beam sizes (≤ 16): **fits**;
  - top-k _sampling_ with k > 16 (k=40 style): **does not fit** — would need chunked
    multi-pass or stay on CPU. Route by `k`.
- What graduation would take (per `op-graduation-playbook.md`):
  1. `metal::` entry point wrapping a per-`{k}` cached `MPSMatrixFindTopK` (operands at
     encode time, same caching argument as `cached_gemm()` in `src/metal/gemm.mm`);
  2. wrap the logits `StorageView` via `buffer_and_offset` as an `MPSMatrix`
     (batch×vocab, fp16 or fp32 — both documented);
  3. an index dtype shim: MPS writes `UInt32`, CT2's TopK contract is `int32` indices
     (`compute<..., int32_t>` in `src/ops/topk.cc`) — bit-identical for vocab-sized
     values, but the result matrix dtype must be declared UInt32;
  4. routing in `TopK::operator()` before generic dispatch, parity via the existing
     suite on `Device::METAL`.
- Measure before claiming a win: decode is per-op-API-floor bound
  (`dispatch-overlap-and-perf-model.md`); the gain here is removing a CPU sync point,
  not FLOPs.
