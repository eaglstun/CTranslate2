# MPSMatrixMultiplication

`MPSMatrixMultiplication` is a Metal Performance Shaders kernel that computes a general matrix multiply (GEMM) on the GPU:

```
C = alpha * op(A) * op(B) + beta * C
```

- `A`, `B`, `C` are `MPSMatrix` objects; `alpha`/`beta` are scalars of the same data type as `C`.
- `A` = left input, `B` = right input, `C` = result (written in place).
- `op(X)` is `X` or `Xᵀ` — each of `A` and `B` may be independently transposed.
- Inherits from `MPSKernel`.

## Initializers

```objc
// Basic: alpha = 1.0, beta = 0.0
- init(device:resultRows:resultColumns:interiorColumns:)

// Full control
- init(device:transposeLeft:transposeRight:
        resultRows:resultColumns:interiorColumns:alpha:beta:)
```

| Parameter         | Meaning                                                                |
| ----------------- | ---------------------------------------------------------------------- |
| `device`          | `MTLDevice`                                                            |
| `transposeLeft`   | `Bool` — transpose `A`                                                 |
| `transposeRight`  | `Bool` — transpose `B`                                                 |
| `resultRows`      | rows of `C` (M)                                                        |
| `resultColumns`   | columns of `C` (N)                                                     |
| `interiorColumns` | shared inner dim (K): columns of `A`, or rows of `B` if `B` transposed |
| `alpha`           | scalar multiplier for the product (default `1.0`)                      |
| `beta`            | scalar multiplier for existing `C` (default `0.0`)                     |

## Encoding

```objc
- encode(commandBuffer:leftMatrix:rightMatrix:resultMatrix:)
```

- `commandBuffer` — `MTLCommandBuffer` to encode into.
- `leftMatrix` (A), `rightMatrix` (B), `resultMatrix` (C — output).
- Operands are supplied **at encode time, not at init**. One MM object of a given shape/config can therefore be reused to encode many multiplications.

## Properties

- `leftMatrixOrigin`, `rightMatrixOrigin`, `resultMatrixOrigin` — origin offsets into each matrix.
- `batchSize`, `batchStart` — batched operation controls.

## MPSMatrix / MPSMatrixDescriptor

An `MPSMatrix` wraps an `MTLBuffer` plus an `MPSMatrixDescriptor`:

| Field      | Meaning                                                       |
| ---------- | ------------------------------------------------------------- |
| `rows`     | number of rows                                                |
| `columns`  | number of columns                                             |
| `rowBytes` | bytes per row (stride)                                        |
| `dataType` | element type, e.g. `MPSDataTypeFloat32`, `MPSDataTypeFloat16` |

- Data layout is **row-major**.
- `rowBytes = leadingDimension * sizeof(element)` and must be `>= columns * sizeof(element)`.
- Apple recommends rounding `rowBytes` up via `MPSMatrixDescriptor.rowBytes(forColumns:dataType:)` for alignment, but the exact `ld * sizeof(element)` value is valid.

### Relevance to the CT2 Metal backend

- `src/metal/gemm.mm` implements `metal::gemm` and `metal::gemm_batch_strided` (fp32 + fp16) on top of `MPSMatrixMultiplication`.
- MPS is row-major exactly like CTranslate2's `StorageView`, so operands map directly: `rowBytes = ld * sizeof(element)`. There is NO cuBLAS-style a/b operand swap (that swap exists only because cuBLAS is column-major).
- Because MM takes operands only at encode time, the backend caches one `MPSMatrixMultiplication` object per distinct `{m, n, k, transposeA, transposeB, alpha, beta, dtype}` key (thread-local `std::map` in `cached_gemm()`). A decoder repeats only a few GEMM shapes per layer/step, so the cache hit rate is high — this was a ~35% end-to-end win. Caching `MPSMatrixDescriptor`s on top was net-zero and reverted.
- fp16 GEMM uses `MPSDataTypeFloat16` and `float16_t` operands; the encode helper is parameterized by element size + `MPSDataType`.
- Strided batched GEMM: matrix `i` lives at `base_offset + i * stride * sizeof(element)`; the Metal allocator's address-ordered `std::map` side table + `buffer_and_offset(ptr)` resolves sub-view / batch-element pointers to `(MTLBuffer, offset)`.
- INT8/INT16 GEMM are NOT on MPS — they stay on the CPU reference path.
- First MPS GEMM call pays a one-time ~493ms pipeline warmup, then is fast.
