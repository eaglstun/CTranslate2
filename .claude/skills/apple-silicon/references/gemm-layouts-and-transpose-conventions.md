---
title: "GEMM layouts & transpose conventions — the row/column-major Rosetta stone"
summary: >-
  The row-major-versus-column-major Rosetta stone for GEMM across CT2's
  backends, resolving the recurring transpose confusion. Everything in CT2 is
  row-major, so for C[m,n]=op(A)op(B) a leading dimension lda/ldb/ldc means
  the stored row length, and src/ops/gemm.cc derives m/n/k and the ld values
  once; Dense passes trans_a=false, trans_b=true with weights stored
  [output,input]=[n,k]. MPS on Metal maps operands directly (stored dims into
  MPSMatrixDescriptor, transpose flags into the cached
  MPSMatrixMultiplication, lda into rowBytes) with an explicit do-NOT-
  replicate-the-cuBLAS-swap warning, whereas cuBLAS is column-major and the
  CUDA wrapper swaps A and B. The int8 MSL kernel ct2_gemm_s8 resolves all
  four transpose combinations at tile-load time while ct2_gemv_s8 and the
  dequant epilogue are Dense-layout-only, enforced by host routing.
---

Sources: repo code read 2026-06-11 (`src/ops/gemm.cc`, `src/layers/common.cc`,
`src/metal/gemm.mm`, `src/cuda/primitives.cu`, `src/metal/kernels/kernels_msl.h`) plus
this skill's `mps-matrix-multiplication.md` and `int8-gemm-kernel-design.md`. No new
Apple-doc claims here — this card is the synthesis that stops the recurring
row-vs-column-major confusion (called out as a sharp edge in SKILL.md's intro).

## Ground truth: everything in CT2 is row-major

A `StorageView` is a row-major buffer. For `C[m,n] = op(A)·op(B)`:

- `A` is stored `[m,k]` normally, `[k,m]` when `trans_a`;
- `B` is stored `[k,n]` normally, `[n,k]` when `trans_b`;
- `C` is always stored `[m,n]`.

The Gemm op computes the dims and leading dimensions in `src/ops/gemm.cc` (same logic
duplicated in the Metal route, `metal_gemm` in the same file):

```cpp
const dim_t k = a.dim(trans_a ? -2 : -1);
const dim_t n = b.dim(trans_b ? -2 : -1);
const dim_t m = a.size() / k;        // collapse leading dims
const dim_t lda = trans_a ? m : k;   // = width of the STORED matrix
const dim_t ldb = trans_b ? k : n;
const dim_t ldc = n;
```

**In CT2's row-major world, a leading dimension is the stored row length** (elements per
stored row, i.e. the stride between consecutive rows) — `lda` is the stored A's column
count, whichever way A is stored. Contrast: in column-major BLAS, `lda` is the stride
between consecutive _columns_ (the stored row count). Same name, transposed meaning.

## What Dense actually passes: `trans_a=false, trans_b=true`

`src/layers/common.cc`, the Dense constructor: `_gemm_op(alpha=1, beta=0,
/*trans_a=*/false, /*trans_b=*/true, …)`. Weights are stored `[output_size, input_size]`
= `[n, k]` (verified: `Dense::output_size()` returns `_weight.dim(0)`). So the everyday
CT2 GEMM is `[m,k]·[n,k]ᵀ`: activations un-transposed, weight transposed. Any kernel that
only handles this layout (the int8 GEMV below) must check the flags, not assume them.

## MPS (Metal): row-major too — operands map DIRECTLY

`MPSMatrixDescriptor` takes `rows`/`columns`/`rowBytes` and is row-major
(`mps-matrix-multiplication.md`). The header comment of `src/metal/gemm.mm` is the
contract: "a stored A of shape (rows, cols) with leading dimension lda becomes an
MPSMatrix with rowBytes = lda \* element_size … do NOT replicate the cuBLAS swap."
`encode_gemm` builds descriptors from the **stored (pre-transpose)** dims:

```objc
const NSUInteger a_rows = transpose_a ? k : m;   // stored shape
const NSUInteger a_cols = transpose_a ? m : k;
// descriptor(a_rows, a_cols, lda, …); transposeLeft/Right passed to
// MPSMatrixMultiplication at init (cached by shape key — see mps ref)
```

So on Metal: stored dims → descriptor, transpose flags → the MPS kernel object, lda →
`rowBytes`. Nothing is swapped.

## cuBLAS (CUDA): column-major — the wrapper swaps A and B

`src/cuda/primitives.cu`, `primitives<Device::CUDA>::gemm`: "cuBLAS assumes column-major
storage, so swap a and b accordingly":

```cpp
cublasSgemm(handle,
            transpose_b ? CUBLAS_OP_T : CUBLAS_OP_N,   // B first!
            transpose_a ? CUBLAS_OP_T : CUBLAS_OP_N,
            n, m, k,  &alpha,  b, ldb,  a, lda,  &beta,  c, ldc);
```

The identity behind it: a row-major `[m,n]` matrix reinterpreted as column-major is its
transpose, so `C = A·B` (row-major) ≡ `Cᵀ = Bᵀ·Aᵀ` (column-major) — compute B·A with
swapped dims `n, m` and the column-major result _is_ the row-major C, with the row-major
`ldb/lda/ldc` values passing through unchanged. **This trick belongs to the CUDA file
only**; copying it into a Metal or MSL path double-transposes.

## The int8 MSL kernels: transposes resolved at tile-load time

`ct2_gemm_s8` (`src/metal/kernels/kernels_msl.h`) takes `trans_a`/`trans_b` as kernel
arguments and resolves them in the global→threadgroup-tile load, so the inner MAC loop is
layout-free:

```msl
v = (trans_a != 0u) ? a[(ulong)gk * lda + gi] : a[(ulong)gi * lda + gk];
v = (trans_b != 0u) ? b[(ulong)gj * ldb + gk] : b[(ulong)gk * ldb + gj];
```

(`lda`/`ldb` carry the same stored-row-length meaning as everywhere else in CT2.) The
decode GEMV `ct2_gemv_s8` is the opposite design point: **Dense layout only**
(`!trans_a && trans_b`, per its header comment), with the host route checking the flags
before dispatch (`int8-gemv-simdgroup-decode.md`). The dequant epilogue likewise assumes
the Dense layout for its per-row/per-column scale orientation (comment above
`ct2_dequant_gemm_out_*`).

## One-line summary per backend

| Backend        | Storage view | Transpose handled by           | `ld*` means                |
| -------------- | ------------ | ------------------------------ | -------------------------- |
| CT2 core / CPU | row-major    | flags into the BLAS call       | stored row length          |
| Metal MPS      | row-major    | `transposeLeft/Right` at init  | `rowBytes` = ld·elsize     |
| Metal int8 MSL | row-major    | index swap at tile load        | stored row length          |
| CUDA cuBLAS    | column-major | swap A↔B + reversed flag order | row-major values pass thru |

### Relevance to the CT2 Metal backend

- `src/metal/gemm.mm` — the direct row-major mapping and the "do NOT replicate the
  cuBLAS swap" warning; descriptors take stored dims, the cached
  `MPSMatrixMultiplication` takes the flags.
- `src/ops/gemm.cc` — `metal_gemm` / `compute` own the m/n/k/ld derivation; any new GEMM
  route must reuse it, not re-derive.
- `src/metal/kernels/kernels_msl.h` — `ct2_gemm_s8` handles all four flag combinations at
  tile load; `ct2_gemv_s8` + `ct2_dequant_gemm_out_*` are Dense-layout-only and the host
  routing enforces it.
- `src/layers/common.cc` — Dense passes `trans_a=false, trans_b=true` with `[out,in]`
  weights; that is the layout 99% of inference GEMMs arrive in.
