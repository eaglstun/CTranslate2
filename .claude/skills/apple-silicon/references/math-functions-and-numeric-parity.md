---
title: "MSL math functions & numeric parity (for norm / reduction kernels)"
summary: >-
  MSL math functions and the fast-versus-precise numeric-parity trap for norm
  and reduction kernels (RMSNorm, LayerNorm, softmax), sourced from the Metal
  Shading Language Specification's §6.6 and the §8.4 ULP/relative-error
  tables. It catalogs the <metal_math> function set (notably no erf/erfc,
  which forces an Abramowitz-Stegun approximation for exact GELU) and explains
  that fast math is the default: under it sqrt is implemented as x*rsqrt(x)
  and division is only 2.5-ulp accurate, so bit-parity with a CPU is not
  guaranteed. Two rules the CT2 kernels bake in: use 1.0f/sqrt(...) not rsqrt
  (double-rounding matches the CPU), and beware FMA contraction fusing a*b+c.
  The parity lever is setting MTLMathModeSafe or metal::precise:: in
  ensure_library, since src/metal/device.mm compiles the kernel library with
  default fast math.
semantic_id: "yhiwxoyqqm3Tw_39TvArvaVoJy-Ty6MhDkXzGpyf0vQuHekDxa8rXyc4elw_i58Kim0Em8z7wY5-tO7Dhrx_pg"
---

Source: Metal Shading Language Specification (vendored at
`../sources/Metal-Shading-Language-Specification.pdf`), §6.6 Math Functions,
§6.10.1 Threadgroup synchronization, §8.4 ULPs and Relative Error (Tables 8.1/8.2),
§1.6.3 math compiler options. Not on any DocC page — MSL stdlib lives only in the PDF.

The reference for anyone writing or debugging a reduction kernel (RMSNorm, LayerNorm,
softmax) where the result must match the CPU reference. The reduction _mechanics_
(SIMD/threadgroup reductions, `simd_sum`) are in `simd-group-functions.md`; this file
is about the _numerics_.

## Available math functions (§6.6, `<metal_math>`)

`T` is a scalar/vector `half` or `float`. The full set includes:
`sqrt`, `rsqrt` (inverse sqrt), `divide(x,y)`, `fma(a,b,c)`, `exp`/`exp2`/`exp10`,
`log`/`log2`/`log10`, `pow`/`powr`, `fabs`, `fmax`/`fmin` (+ `fmax3`/`fmin3`/`fmedian3`),
`floor`/`ceil`/`round`/`rint`/`trunc`, `copysign`, `fmod`, `fract`, `frexp`/`ldexp`,
trig + hyperbolic (`sin`/`cos`/`tan`/`sincos`/`*pi`/`*h`/inverse), `nextafter`.

**Notably absent: `erf` / `erfc`.** There is no error function in MSL at any language
version (this is why exact-GELU needs an Abramowitz-Stegun approximation — see the
`op-graduation-playbook.md` landmines section).

## The fast-vs-precise reality (this is the parity trap)

For single-precision, most functions in §6.6 have **fast** and **precise** variants.
Selection:

- Compiler-wide: `-fmetal-math-fp32-functions=<fast|precise>` and
  `-fmetal-math-mode=<fast|relaxed|safe>`. **Both default to `fast`.** Equivalently,
  fast math is on unless you pass `-fno-fast-math`.
- Per-call namespaces: `metal::fast::sin(x)`, `metal::precise::cos(x)`. A bare `sin(x)`
  uses whichever the compile mode selected.
- Host-side, when compiling at runtime: `MTLCompileOptions.mathMode` (older:
  `fastMathEnabled`). A freshly `init`'d `MTLCompileOptions` = **fast math**.

### Accuracy of the ops a norm kernel actually uses

| op                  | precise (Table 8.1) | **fast / default** (Table 8.2)                            |
| ------------------- | ------------------- | --------------------------------------------------------- |
| `x+y`, `x-y`, `x*y` | correctly rounded   | correctly rounded                                         |
| `1.0/x`             | correctly rounded   | ≤ 1 ulp                                                   |
| `x/y`               | correctly rounded   | ≤ 2.5 ulp                                                 |
| `sqrt(x)`           | correctly rounded   | **implemented as `x * rsqrt(x)`** (not correctly rounded) |
| `rsqrt(x)`          | correctly rounded   | ≤ 2 ulp                                                   |
| `fma`               | correctly rounded   | correctly rounded                                         |
| `exp`, `exp2`       | ≤ 4 ulp             | `≤ 3 + floor(fabs(2x))` ulp                               |
| `log`, `log2`       | ≤ 4 ulp             | abs err ≤ 2⁻²¹ on [0.5,2], else ≤ 3 ulp                   |

Two consequences that bite reduction kernels:

1. **Under default fast math, `sqrt` is `x*rsqrt(x)`** — so even "the safe-looking call"
   isn't correctly rounded. You cannot get true bit-parity with a CPU that uses a
   correctly-rounded `std::sqrt` unless you compile the kernel **precise/safe**.
2. **`1.0/sqrt(x)` ≠ `rsqrt(x)` even in precise mode.** `1.0/sqrt(x)` double-rounds
   (round the sqrt, then round the reciprocal) — which matches how the CPU computes
   `1.0 / std::sqrt(x)`. `rsqrt(x)` is a single correctly-rounded `1/√x`, a _different_
   value. So `1.0f/sqrt(...)` is the parity-preserving spelling; `rsqrt` is the
   fast-but-divergent one. (This is the rule already baked into CT2's norm kernels.)

### FMA contraction

Fast/relaxed math lets the compiler fuse `a*b + c` into a single `fma` (one rounding
instead of two). Great for speed, but it changes results bit-for-bit vs a CPU that does
the multiply and add separately. Relevant to LayerNorm's `sum + sum_of_squares`
accumulation and any Welford-style or two-pass mean/variance. If a parity test drifts by
a ULP or two after a kernel rewrite, suspect contraction before you suspect a real bug.

## Barriers for the reduction (§6.10.1)

A tree/threadgroup reduction must fence threadgroup memory between phases:

```metal
threadgroup_barrier(mem_flags::mem_threadgroup);
```

`mem_flags` values (Table 6.13): `mem_none` (execution-only barrier, no fence),
`mem_device` (orders device-memory ops), `mem_threadgroup` (orders threadgroup-memory
ops — the one reductions need), `mem_texture`, `mem_threadgroup_imageblock`,
`mem_object_data`. `simdgroup_barrier(...)` is the SIMD-group-scoped equivalent (rarely
needed — lanes in a SIMD-group advance in lockstep; see `simd-group-functions.md`).

---

### Relevance to the CT2 Metal backend

- **`src/metal/device.mm:77` compiles the kernel library with a bare
  `[[MTLCompileOptions alloc] init]` → DEFAULT FAST MATH.** So today every CT2 kernel
  (`ct2_rms_norm`, `ct2_layer_norm`, `ct2_softmax`, …) runs fast-math: `sqrt` = `x*rsqrt(x)`,
  division ≤1–2.5 ulp, FMA contraction allowed. True bit-for-bit CPU parity is therefore
  **not** guaranteed by construction — the real gate is the op-suite tolerance, not exact
  equality (the memory's "bit-for-bit" note is about choosing `1/sqrt` over `rsqrt`, which
  reduces divergence; it is not literal correctly-rounded equality under fast math).
- The norm kernels already use `1.0f/sqrt(...)`, **not** `rsqrt` — keep it that way; it's
  the double-rounding-matches-CPU rule above, and switching to `rsqrt` for "speed" will
  shift results and can break parity.
- **Lever if a norm/softmax parity test fails or you need tighter agreement:** set
  `options.mathMode = MTLMathModeSafe` (or `options.fastMathEnabled = NO` on older SDKs)
  in `ensure_library()`, or wrap the divergent ops in `metal::precise::`. Costs some GPU
  throughput; buys correctly-rounded `sqrt`/`div` and no FMA contraction. Measure before
  committing — this would slow _every_ kernel, not just norms.
- fp16 norms compute reductions in `float` then cast (per the existing kernels); the ULP
  tables above are single-precision, so the float-accumulate path is what governs parity,
  with a final `half` rounding on store (hence the looser ~2e-2 fp16 tolerance).
