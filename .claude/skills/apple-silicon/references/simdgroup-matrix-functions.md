---
title: "SIMD-group matrix functions (WMMA-style 8×8 matmul)"
summary: >-
  Covers MSL SIMD-group matrix functions (the WMMA-style tensor-core
  primitive) from the Metal Shading Language Specification sections 2.4 and
  6.8: the <metal_simdgroup_matrix> header, cooperative-per-SIMD-group
  execution under uniform control flow, simdgroup_load/simdgroup_store with
  elements_per_row stride and matrix_origin, and simdgroup_multiply_accumulate
  (d = a*b + c). The ground-truth fact driving the int8 design decision is
  that the supported types are exhaustively floating-point 8x8 only —
  simdgroup_half8x8, simdgroup_bfloat8x8 (Metal 3.1+), simdgroup_float8x8 —
  with NO integer variant, so int8xint8-to-int32 cannot be expressed as WMMA
  on Apple GPUs, which is why ct2_gemm_s8 is hand-tiled through threadgroup
  memory. Notes the single shared accumulator type (no mixed-precision
  fp16-in/fp32-out form) and that these primitives remain candidates if an
  fp16/fp32 GEMM ever moves off MPS or a fused-attention kernel is attempted.
---

Source (Apple): Metal Shading Language Specification, §2.4 (types) and §6.8 (functions) (v4.1,
2026-06-04). PDF: <https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf>
(The MSL standard-library functions are only in the spec PDF — there is no DocC HTML/JSON
page for them, so this was extracted from the PDF, not the usual DocC-JSON endpoint.)

Header `<metal_simdgroup_matrix>`. Operations on SIMD-group matrices are executed
**cooperatively by the threads in one SIMD-group** — the tensor-core-style "WMMA" primitive.
All operations must run under **uniform control flow within the SIMD-group** or behavior is
undefined. The mapping of matrix elements to threads is **unspecified** (you cannot poke at
individual elements; data moves only through `simdgroup_load`/`simdgroup_store`).

## Supported types (§2.4) — THE ground truth for the int8 decision

Spec, verbatim: "Metal supports the following SIMD-group matrix type names, where `T` is
`half`, `bfloat` (in Metal 3.1 and later) or `float` and `Cols` and `Rows` are 8:"

- `simdgroup_half8x8`
- `simdgroup_bfloat8x8` (Metal 3.1 and later)
- `simdgroup_float8x8`

That list is exhaustive: **floating-point element types only, 8×8 only. There is NO integer
`simdgroup_matrix` — no int8, no int32, nothing** — so an int8×int8→int32 matmul cannot be
expressed with these primitives on Apple GPUs. (Verified the hard way by this project; see
relevance below.)

Availability: "All OS: Metal 2.3 and later support SIMD-group matrix types" (§2.4). Apple's
separate _Metal Feature Set Tables_ map this to the Apple7 GPU family and later — i.e. every
Apple Silicon Mac (M1 = Apple7). `bfloat` element type additionally needs Metal 3.1.

Note the spec's own steer (§6.8, Metal 4): "Instead of using simdgroup matrix multiplication,
consider using Tensors (section 2.22) and Metal Performance Primitives (section 7)" — the
newer, more general matmul library surface. Worth evaluating alongside raw simdgroup_matrix
if a custom GEMM is ever written.

## Creating, loading, storing (Table 6.9)

```metal
simdgroup_matrix<T,Cols,Rows>(T dval)            // diagonal matrix with the given value
simdgroup_matrix<T,Cols,Rows>
  make_filled_simdgroup_matrix(T value)          // matrix filled with the given value

void simdgroup_load(thread simdgroup_matrix<T,Cols,Rows>& d,
                    const threadgroup T *src,    // or: const device T *src
                    ulong elements_per_row = Cols,
                    ulong2 matrix_origin = 0,
                    bool transpose_matrix = false)

void simdgroup_store(thread simdgroup_matrix<T,Cols,Rows> a,
                     threadgroup T *dst,         // or: device T *dst
                     ulong elements_per_row = Cols,
                     ulong2 matrix_origin = 0,
                     bool transpose_matrix = false)
```

`elements_per_row` is the row stride (in elements) of the source/destination memory layout —
this is how an 8×8 tile is addressed inside a larger matrix; `matrix_origin` offsets into it.
Load/store work on both `threadgroup` and `device` memory.

## Matrix operations (Table 6.10)

```metal
void simdgroup_multiply_accumulate(thread simdgroup_matrix<T,Cols,Rows>& d,
                                   thread simdgroup_matrix<T,K,Rows>&    a,
                                   thread simdgroup_matrix<T,Cols,K>&    b,
                                   thread simdgroup_matrix<T,Cols,Rows>& c)  // d = a*b + c
void simdgroup_multiply(thread simdgroup_matrix<T,Cols,Rows>& d,
                        thread simdgroup_matrix<T,K,Rows>&    a,
                        thread simdgroup_matrix<T,Cols,K>&    b)             // d = a*b
// operator* is also listed: a * b
```

Accumulator and operands share one element type `T` — there is no mixed-precision form (no
fp16-in/fp32-accumulate signature in the MSL surface).

## Spec example (verbatim, §6.8.2)

```metal
kernel void float_matmad(device float *pMatA, device float *pMatB,
                         device float *pMatC, device float *pMatR)
{
    simdgroup_float8x8 sgMatA;
    simdgroup_float8x8 sgMatB;
    simdgroup_float8x8 sgMatC;
    simdgroup_float8x8 sgMatR;

    simdgroup_load(sgMatA, pMatA);
    simdgroup_load(sgMatB, pMatB);
    simdgroup_load(sgMatC, pMatC);
    simdgroup_multiply_accumulate(sgMatR, sgMatA, sgMatB, sgMatC);
    simdgroup_store(sgMatR, pMatR);
}
```

(The spec text has a typo — a missing comma after `pMatB` — preserved-but-fixed here.)

---

### Relevance to the CT2 Metal backend

- **This is why the int8 GEMM is hand-tiled.** `ct2_gemm_s8` in
  `src/metal/kernels/kernels_msl.h` carries the decision in its header comment: "MPS has no
  integer GEMM and simdgroup_matrix has no int8 element type (MSL spec 2.4: half/bfloat/float
  only), so this is hand-tiled." The §2.4 type list above is the spec ground truth for that
  claim — there is no WMMA-style path for int8×int8→int32 on Apple GPUs, so the backend stages
  `char` tiles through threadgroup memory and multiply-accumulates in `int32` ALU ops
  (4-per-`int4`). Same story for `ct2_gemv_s8`. See `METAL_BACKEND.md` (M12) for the measured
  consequence: int8 wins the memory-bound GEMV regime, not the ALU-bound tiled regime.
- **Round-tripping int8 through `simdgroup_half8x8` is not a loophole** — converting operands
  to half loses the exact-int32 accumulation the int8 path is built on (half has an 11-bit
  significand; products up to 127×127×k overflow it immediately).
- simdgroup_matrix **remains the candidate** if a future **fp16/fp32 GEMM ever moves off MPS**
  or a **fused-attention kernel** is attempted (the long-standing TODO in `SKILL.md` §6.8) —
  8×8 fp16 tiles with `simdgroup_multiply_accumulate` are the building block, and Metal 4's
  Tensors + Metal Performance Primitives (§2.22/§7) are the newer alternative to evaluate
  first. Note the accumulator-type limitation above when budgeting fp16 accuracy; CT2's
  parity culture (see `math-functions-and-numeric-parity.md`) would push toward
  `simdgroup_float8x8` accumulation or MPP.
- Uniform-control-flow rule: any tile loop using these must keep whole SIMD-groups converged —
  the same discipline `ct2_gemv_s8` already follows for its `simd_sum` (its early-`return` is
  uniform per SIMD-group; see `simd-group-functions.md`).
