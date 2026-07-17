---
title: "Relational functions (isnan, isinf, select, all/any, signbit)"
summary: >-
  Reference for MSL's <metal_relational> functions from Metal Shading Language
  Specification section 6.5: the classification predicates isnan, isinf,
  isfinite, isnormal, isordered/isunordered, signbit, the componentwise
  vector-collapse all/any and not, and the branchless select(a, b, c) whose
  true condition picks the second argument b. The critical caveat is that fast
  math (-fmetal-math-mode=fast, this backend's default) enables No-NaNs/No-
  INFs, letting the compiler legally fold an in-kernel isnan(x) to false, so
  reliable NaN checks require compiling that source with mathMode relaxed/safe
  or a math_mode(safe) pragma. Ties to the CT2 backend by noting the Gemma2
  pad-collapse bug was found with host-side NaN tripwires (where std::isnan is
  reliable) not in-kernel, landing on Metal tanh overflow fixed by clamp in
  ct2_tanh_safe. Distinguishes componentwise all/any from the cross-lane
  simd_all/simd_any votes, and notes no relational function is currently
  called in kernels_msl.h, making this a debugging-session card.
---

Source (Apple): Metal Shading Language Specification, §6.5 (v4.1, 2026-06-04); fast-math
semantics from §1.6.3.
PDF: <https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf>
(The MSL standard-library functions are only in the spec PDF — there is no DocC HTML/JSON
page for them, so this was extracted from the PDF, not the usual DocC-JSON endpoint.)

Header `<metal_relational>`. `T` = scalar/vector floating-point **including bfloat**;
`Ti` = scalar/vector integer or boolean; `Tb` = scalar/vector boolean. The vector forms
return a boolean **vector** (componentwise), hence `all`/`any` to collapse it.

## Table 6.3

| Function                   | Returns                                                                                                                                                     |
| -------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `bool all(Tb x)`           | true only if **all** components of x are true                                                                                                               |
| `bool any(Tb x)`           | true only if **any** component of x is true                                                                                                                 |
| `Tb isfinite(T x)`         | test for finite value                                                                                                                                       |
| `Tb isinf(T x)`            | test for ±infinity                                                                                                                                          |
| `Tb isnan(T x)`            | test for NaN                                                                                                                                                |
| `Tb isnormal(T x)`         | test for a normal value                                                                                                                                     |
| `Tb isordered(T x, T y)`   | `(x == x) && (y == y)`                                                                                                                                      |
| `Tb isunordered(T x, T y)` | true if x or y is NaN                                                                                                                                       |
| `Tb not(Tb x)`             | componentwise logical complement                                                                                                                            |
| `T select(T a, T b, Tb c)` | vector: `result[i] = c[i] ? b[i] : a[i]`; scalar: `c ? b : a`. Note the order — **true picks `b`**, the _second_ value. Also `Ti select(Ti a, Ti b, Tb c)`. |
| `Tb signbit(T x)`          | true if the sign bit is set (distinguishes -0.0, negative NaN)                                                                                              |

## The fast-math caveat on isnan/isinf

The spec does not exempt the classification functions from fast math. §1.6.3:
`-fmetal-math-mode=fast` (the **default**, and what this backend builds with — see
`math-functions-and-numeric-parity.md`) enables "No NaNs" / "No INFs": the compiler may
_assume arguments and results are never NaN/INF_ and optimize accordingly — "the use of
fast math asserts that the shader will never generate INF or NaN", and a program that does
is undefined. So an in-kernel `isnan(x)` can legally be folded to `false`. To make NaN
checks reliable: compile that source with `mathMode = relaxed` or `safe` (relaxed keeps
most speed but "honors INFs and NaNs"), or use `#pragma METAL fp math_mode(safe)` around
the check. (§1.6.3 also notes Apple GPU Family 4+ _math functions_ honor INF/NaN even
under `fp32-functions=fast` — that covers `sin`/`exp` etc. propagating NaN, not the
optimizer's right to delete your `isnan`.)

---

### Relevance to the CT2 Metal backend

- **NaN tripwires found the Gemma2 bug — but on the host, not in-kernel.** The
  `<pad>`-collapse was bisected with per-layer NaN checks on layer-boundary outputs read
  back CPU-side (where `std::isnan` is reliable), landing on Metal `tanh` overflowing in
  the GELU-tanh path; the fix is the `clamp(x, -15.0f, 15.0f)` in `ct2_tanh_safe`
  (`src/metal/kernels/kernels_msl.h` — see `common-functions.md`). If you ever put a
  tripwire _inside_ a kernel, the fast-math caveat above applies: default-built kernels
  may optimize the `isnan` away, so guard such debug code with `math_mode(safe)`.
- **`select()` is the branchless guard** for divergence-sensitive spots (a per-lane
  ternary as a function, no flow control). The backend currently uses plain ternaries
  (`ct2_quantize_s8_impl`'s `amax != 0.0f ? … : 1.0f`, `ct2_rotary_*`'s interleave pick) —
  semantically equivalent for scalars; `select` earns its keep on vector types where a
  ternary on a `bool4` doesn't compose. Remember the argument order (true → `b`).
- `all`/`any` here are the **componentwise** vector collapses — distinct from the
  cross-lane `simd_all`/`simd_any` votes in `simd-group-functions.md`.
- No relational function is currently called in `kernels_msl.h` (verified by grep) — this
  card exists for debugging sessions, which is exactly when you'll reach for it.
