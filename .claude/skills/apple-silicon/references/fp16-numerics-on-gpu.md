# fp16 numerics on the GPU (the half-precision survival card)

Sources: MSL spec §2.1 (scalar types: `half` "must conform to the IEEE 754 binary16
storage format"; `bfloat` Metal 3.1+), §2.24 (implicit conversions + conversion
rounding rules), v4.1 2026-06-04
(<https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf>);
IEEE-754 binary16 arithmetic facts (labeled **[standard]**); this repo's kernels and
history for the bug story (**[measured here]**). Fast-math behavior is owned by
`math-functions-and-numeric-parity.md`; memory layout by `msl-data-types-and-alignment.md`
— neither is duplicated here.

## The numbers that bite [spec §2.1 + standard]

`half` = 1 sign + 5 exponent + 10 stored mantissa bits (11-bit significand):

- **max finite: 65504** — anything beyond rounds to ±Inf.
- min normal: 2⁻¹⁴ ≈ **6.10e-5**; subnormals reach 2⁻²⁴ ≈ 5.96e-8.
- ~3 decimal digits of precision; integers are exact only up to 2¹¹ = 2048.

The overflow class: any intermediate that can exceed 65504 — a sum over a long row, an
`exp()` of an argument > ln(65504) ≈ 11.09, a product of large activations — silently
becomes Inf, and the first Inf−Inf or Inf/Inf becomes **NaN**, which then propagates
through every downstream op until the output collapses.

## THE case study: Gemma2 `<pad>`-collapse (tanh overflow → NaN)

**[measured here — fixed 2026-06-09]** Gemma2 on Metal emitted `<pad>`-collapsed garbage.
Cause, now documented at the fix site (`src/metal/kernels/kernels_msl.h`, `ct2_tanh_safe`,
~lines 509–526): Metal's `tanh(x)` computes `(exp(2x)−1)/(exp(2x)+1)`; for large |x|,
`exp(2x)` overflows to Inf and Inf/Inf = **NaN**, whereas `tanh` mathematically saturates
to ±1 (and CPU `std::tanh` does saturate). Gemma2's huge deep-layer activations make the
GELU-tanh cubic argument big enough to trip this _even in float_ (float `exp(2x)`
overflows at x ≳ 44); in half the same class trips at x ≳ 5.5. The fix:

```metal
inline float ct2_tanh_safe(float x) {
  return tanh(clamp(x, -15.0f, 15.0f));   // tanh(±15) already rounds to ±1
}
```

used by both GELU-tanh and the Tanh activation in `ct2_apply_activation`. Clamp semantics
are in `common-functions.md`; the debugging story (per-layer NaN tripwires, CPU-ref
bisection, why mid-pipeline reads of MPS output are unreliable) is project history —
trust layer-boundary reads.

Moral: saturating functions are only saturating if their _implementation_ is; bound the
argument, not the result.

## The rule the backend lives by: store half, compute float

**[measured here — verifiable in `src/metal/kernels/kernels_msl.h`]** Every `_half`
kernel widens to `float` on load, does ALL arithmetic — especially reductions — in float,
and rounds to half exactly once, at the store:

- `ct2_softmax_half`: `local_max`/`local_sum` are `float`; `exp((float)x[j] - x_max)`;
  store `(half)(...)` (~lines 121–177).
- `ct2_rms_norm_half` / `ct2_add_rms_norm_half`: sum of squares accumulates `float`;
  `inv_rms = 1.0f / sqrt(...)`; final `(T)(...)` cast (~lines 194–270).
- `ct2_layer_norm_half`: float `mean`/`variance` from float partial sums (~lines 316–335).
- Even elementwise ops follow it: `ct2_add_half` computes `(T)((float)a[gid] + bv)`;
  activations run `(float)x` through `ct2_apply_activation` and cast back; quantize takes
  `fabs((float)x[j])` for the amax reduction.

Why: a float has 24 significand bits — k half values summed in float stay exact in the
ways that matter, while summing _in half_ loses to both rounding (RMS over a 4096-wide
row) and the 65504 ceiling. fp16's looser test tolerance (~2e-2) covers only the final
store rounding, not accumulated drift — keep it that way when adding kernels.

## Literals and implicit promotions [spec §2.1, §2.24]

- Suffixes: `0.5f` (float), `0.5h` (half), `0.5bf` (bfloat, Metal 3.1+). Write the suffix
  you mean: `half h; h * 1.0f` promotes the math to float (usually what you want here);
  `h * 1.0h` keeps it in half — the overflow/precision class above.
- §2.24: implicit scalar conversions are value conversions; **bfloat is asymmetric** —
  bfloat→float implicitly converts, but bfloat↛half and float/half↛bfloat (explicit cast
  required). Vector→vector implicit conversions are a compile error.
- Backend convention: explicit `(float)` casts at loads + `f`-suffixed literals, so the
  precision of every expression is visible in the source. Follow it.

## bfloat: the no-overflow alternative — NOT used here [spec §2.1]

`bfloat` (Metal 3.1+) is truncated float32: 8 exponent bits → range ≈ float (no 65504
cliff; the Gemma2 class of bug can't happen), but only 8-bit significand (7 stored) →
_worse_ precision than half. The backend does not use it: storage is half, accumulation
is float, which gets half's precision _and_ float's range where it counts. If a future
model overflow can't be clamped locally, bfloat storage is the documented escape hatch
(dtype/alignment in `msl-data-types-and-alignment.md`).

## Rounding at the half store [spec §2.24]

- float→half conversion rounds **ties-to-even**; half→float is lossless; denormals
  produced on the way down "may not be flushed to zero."
- **Fast math does not change conversion accuracy** (spec, verbatim) — the store rounding
  is dependable even though the arithmetic before it is fast-math
  (`math-functions-and-numeric-parity.md`).
- float→int rounds toward zero and NaN→0 — owned by
  `conversion-and-packing-functions.md` (it's why quantize calls `rint` first).

### Relevance to the CT2 Metal backend

- `ct2_tanh_safe` (`src/metal/kernels/kernels_msl.h` ~509) guards every GELU-tanh/Tanh
  on the GPU; `DecodeParityLLM` (fp32+fp16) in `tests/metal_test.cc` is the regression
  net that caught its absence. Don't "simplify" the clamp away.
- The store-half/compute-float pattern is load-bearing in every `_half` kernel listed
  above; a new fp16 kernel that accumulates in half will pass small unit shapes and fail
  on real 896–4096-wide rows.
- fp16 ops that still run the CPU reference get upcast to fp32 first (the Whisper
  bringup fix — Conv1D et al.); software-emulated half on the CPU was the 27× `Add`
  regression (`dispatch-overlap-and-perf-model.md`).
- Parity tolerances: fp32 tight, fp16 ~2e-2 in the op suite (`tests/ops_test.cc`,
  `tests/metal_test.cc`) — the half-store rounding budget, nothing more.
