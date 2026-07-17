---
title: "int8 GEMM kernel design — hand-tiled int8×int8→int32 (`ct2_gemm_s8`)"
summary: >-
  Details ct2_gemm_s8, the hand-tiled int8xint8->int32 GEMM in kernels_msl.h,
  needed because Apple GPUs have no integer matrix units
  (MPSMatrixMultiplication is float-only, simdgroup_matrix has no int8 type).
  A 16x16 threadgroup computes a 64x64 C-tile with k in 32-deep depth-major
  chunks staged through threadgroup memory, each thread holding a 4x4 int4
  micro-tile; transposes resolve at tile-load time, edges zero-pad on load and
  guard only on store, and the inner loop reinterprets tiles as char4 widened
  to int4 for 4 MACs per op. int32 accumulation is bit-exact at any real depth
  (products bounded by 127*127<2^14), proven by
  Int8GemmSaturatedAccumulatorExact past fp32's 2^24 ceiling. The gemm.cc
  routing guard requires beta==0, no u8-shift compensation, and integral
  alpha. The tiled kernel is ALU-bound ~2.4 T-MAC/s (structurally ~3-5x slower
  than MPS fp16) and is now the fallback since the Metal-4 MPP matmul2d path
  ties MPS fp16, but int8 still wins peak RSS (-42%).
semantic_id: "yhkQ5o-em33Sw-1dSvkrnYMwr2uTmqIpCheyudS3xN4CP-kHR66rVi80bhVnj4w6zC0k2s37IZ5u9c3TjOztpA"
---

Source: `src/metal/kernels/kernels_msl.h` (kernel `ct2_gemm_s8`, lines ~744–829),
`src/metal/primitives.mm` (`metal::gemm_s8`, ~586–673), `src/ops/gemm.cc` (INT8 routing,
~107–120 and `metal_gemm_int8` ~78–95). Commits `eaa0f187` (Phase 1 shim),
`ab8a617a` (Phase 2 native GEMM), `68239d13` (int4-vectorized inner loop).
Measurements: `METAL_BENCHMARKS.md` "int8: native int8×int8→int32 GEMM", M4 Max
(40-core GPU), Release, 2026-06-11.

## Why hand-tiled — there is no native int8 matmul primitive on Apple GPUs

Two facts force a hand-written kernel (both stated in the kernel's header comment):

1. **MPS has no integer GEMM** — `MPSMatrixMultiplication` is float-only
   (see `mps-matrix-multiplication.md`).
2. **`simdgroup_matrix` has no int8 element type** — MSL spec 2.4 allows
   half/bfloat/float only (see `simdgroup-matrix-functions.md`). Apple GPUs have **no
   integer matrix units** at all; an int32 MAC costs two ALU ops.

So a bit-exact int8 path cannot ride any accelerated primitive. The design goal was
**exactness + int8-residency** (weights stay int8 on the GPU, no per-call widening),
not beating MPS fp16 on raw throughput — which it structurally cannot (below).

## Tile structure

One **16×16 threadgroup computes a 64×64 tile of C**; k is consumed in **32-deep
chunks** staged through threadgroup memory; each thread accumulates a **4×4 register
micro-tile** held as four `int4`s. Constants (mirrored on the host as
`kGemmS8TileM/N` in `primitives.mm` — they must match):

```c
constant uint CT2_GEMM_S8_BM = 64;   // C-tile rows
constant uint CT2_GEMM_S8_BN = 64;   // C-tile cols
constant uint CT2_GEMM_S8_BK = 32;   // k chunk depth
threadgroup char As[CT2_GEMM_S8_BK][CT2_GEMM_S8_BM];  // As[kk][i]: depth-major
threadgroup char Bs[CT2_GEMM_S8_BK][CT2_GEMM_S8_BN];  // Bs[kk][j]: depth-major
```

Key choices:

- **Transposes are resolved once at tile-load time.** All 256 threads cooperatively
  fill `As`/`Bs` (linear thread id strides the tile), and the per-element load picks
  `a[gk*lda + gi]` vs `a[gi*lda + gk]` based on `trans_a` (same for B). The compute
  loop never sees the flags — all four layouts hit the same inner loop.
  `Int8GemmAllTransposeCombinations` (tests/metal_test.cc:520) covers all four against
  a host triple loop with edge-size dims.
- **Both staging tiles are depth-major** (`[kk][i]` / `[kk][j]`), so in the inner loop
  both operand reads are contiguous along the micro-tile axis regardless of layout.
- **Edges zero-pad on load, guard only on store.** Out-of-range loads stage `0` (a
  no-op in the dot product), so the rounded-up grid
  (`⌈n/64⌉ × ⌈m/64⌉` threadgroups of 16×16, `dispatchThreadgroups`) needs bounds
  checks only when writing C.

## The char4/int4-vectorized inner loop (commit 68239d13)

The depth-major tiles make both reads reinterpretable as `char4` (`tid.{x,y}*4` is
4-aligned; tile rows are 64 bytes). Widen once to `int4`, then 4 MACs per vector op:

```c
for (uint kk = 0; kk < CT2_GEMM_S8_BK; ++kk) {
  const int4 av = int4(*(const threadgroup char4*)(&As[kk][tid.y * 4u]));
  const int4 bv = int4(*(const threadgroup char4*)(&Bs[kk][tid.x * 4u]));
  acc[0] += av.x * bv;   // acc[r] is int4: micro-tile row r, 4 cols
  acc[1] += av.y * bv;
  acc[2] += av.z * bv;
  acc[3] += av.w * bv;
}
```

**Why int32 accumulation is exact:** every product is bounded by 127·127 = 16129 < 2¹⁴,
so even an all-saturated same-sign reduction stays inside int32 (±2³¹) up to
k ≈ 2³¹/127² ≈ 133 000 ≈ 2¹⁷ — beyond any real model depth. The repo's claim is
"bit-exact int32 **at any depth**" and the suite proves the regime fp32 could not
represent: `Int8GemmSaturatedAccumulatorExact` (metal_test.cc:506) drives the
accumulator to 2048·127·127 = 33 032 192 > 2²⁴ (fp32's integer-exact ceiling), which
the retired Phase-1 shim could not produce.

## The integer alpha/beta contract — verify against `src/ops/gemm.cc`

The kernel computes `C = alpha · op(A)·op(B)` with **int32 alpha applied at the C
store** (`c[...] = alpha * acc[r][s]`) and **no beta term at all** — there is no beta
parameter in `ct2_gemm_s8` or `metal::gemm_s8`. The routing guard in
`Gemm::operator()` (src/ops/gemm.cc:113) enforces exactly:

```cpp
if (a.device() == Device::METAL && !a_shift_compensation && _beta == 0
    && _alpha == static_cast<float>(static_cast<int32_t>(_alpha))) {
  metal_gemm_int8(_alpha, _trans_a, _trans_b, a, b, c);
```

i.e. **no u8-shift compensation, beta == 0, and an integral alpha** (a float alpha
cannot be applied exactly to an int32 accumulator). Anything else falls through to the
generic dispatch (CPU reference over unified memory) rather than silently dropping
terms. Scaling by the quantization scales does NOT happen here — it lives in the
dequantize epilogue (`ct2_dequant_gemm_out_*`, see `quantize-dequantize-kernels.md`),
which is the only form CT2's quantized Dense uses (alpha is 1 in practice).

Small-m calls don't reach this kernel: `metal::gemm_s8` routes Dense-layout m ≤ 8 to
the SIMD-group GEMV first (see `int8-gemv-simdgroup-decode.md`).

## Phase-1 history — the dequant shim and why it had to die

Phase 1 (commit `eaa0f187`) shipped a **shim**: widen int8 operands to fp32, ride the
cached MPS float GEMM, cast back to int32. Correct below 2²⁴ (fp16 was rejected
outright — real LLM accumulators overflow its integer-exact range), and it proved the
plumbing (92/100 teacher-forced agreement vs fp16, same as Phase 2). But it
**defeated the entire point of int8**: weights widened per call, so there was no
resident-memory win and the extra cast passes made it ~2.5× slower than fp16 e2e.
Phase 2 deleted the shim casts (`s8_to_float` / `float_to_s32`); stale comments
referencing the shim survive in `src/types.cc` (~156, ~284) — the _behavior_ there
(`mayiuse_int8(METAL)` true, `ComputeType::AUTO` pinned to FLOAT32 on Metal) is
current, the prose is not.

## Measured (M4 Max, 2026-06-11, `DISABLED_BenchmarkGemmInt8`, 30 iters/cell)

| m, n, k (kernel)         | Metal int8 | Metal fp16 (MPS) | Metal fp32 (MPS) |
| ------------------------ | ---------- | ---------------- | ---------------- |
| 2048, 2048, 2048 (tiled) | 7.28 ms    | **1.48 ms**      | 1.72 ms          |
| 256, 4864, 896 (tiled)   | 1.15 ms    | **0.36 ms**      | 0.38 ms          |

**The tiled kernel is ALU-bound at ~2.4 T-MAC/s vs a ~3.7 ceiling** (int MAC = 2 ALU
ops, no integer matrix units), while MPS fp16 rides dedicated FMA pipelines to
~5.9 T-FMA/s. Large-m int8 is therefore structurally ~3–5× slower than MPS fp16
_for a hand-written ALU kernel_. **SUPERSEDED 2026-06-11 for m>8 on macOS 26+:**
the Metal-4 MPP `matmul2d` path (`metal4-tensors-and-mpp.md`, `ct2_mpp_gemm_s8_nt`
in `kernels_mpp_msl.h`) is int32-bit-exact and **ties MPS fp16** (2048³: 1.51 vs
1.49 ms, ~11.4 T-eff-FLOPS) — the tiled kernel is now the fallback (pre-26 OSes,
non-NT layouts, integral alpha ≠ 1). The `simdgroup_float8x8` staging trick
(int8→float tiles exact, per-1024-chunk accumulation < 2²⁴) is therefore moot —
kept here only in case the MPP path ever has to be abandoned.
E2e the package wins where it matters: **peak RSS 1453 vs 2494 MB (−42%)**, e2e
prefill within ~17% of fp16 on Qwen2.5-0.5B post-MPP (was ~1.26×; shim was ~2.5×).

### Relevance to the CT2 Metal backend

- Wired: `ops::Gemm::operator()` INT8 branch → `metal::gemm_s8` (`primitives.mm`) →
  `ct2_gemm_s8` / `ct2_gemv_s8` (`kernels_msl.h`). Exercised by every quantized Dense
  on Metal (`device="metal"`, `compute_type="int8"` / `int8_float16`).
- Regressions to watch: host/kernel tile-size constants must stay in sync
  (`kGemmS8TileM/N` ↔ `CT2_GEMM_S8_BM/BN`, 16×16 group); the routing guard's
  exactness contract (beta==0, integral alpha) must not be loosened; the
  saturated-accumulator and all-transpose tests are the tripwires.
- Future work touching this: any `simdgroup_float8x8` staging experiment for large-m;
  int8 KV-cache or fused-attention work would route here too.
