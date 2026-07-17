---
title: "Common functions (clamp, mix, saturate, sign, step, smoothstep)"
summary: >-
  Documents the MSL metal_common standard-library functions from spec section
  6.3: clamp, mix, saturate, sign, step, and smoothstep, all float/half-only,
  with each function's exact return semantics and undefined-behavior cases
  (clamp undefined when minval greater than maxval, sign returns 0.0 for NaN,
  smoothstep's Hermite interpolation). It flags that clamp and saturate have
  fast versus precise variants differing only in NaN handling, and that the
  backend's default fast-math build makes clamp(NaN,...) undefined. The CT2
  relevance ties this to the Gemma2 NaN fix in kernels_msl.h: ct2_tanh_safe
  clamps the tanh argument to [-15,15] before the call (clamping the argument,
  not the NaN result), and notes ct2_quantize_s8_impl deliberately does not
  clamp to plus/minus 127.
semantic_id: "yl6w1ozrq-3WgWj9wuEJPZMgJ2uTzqshClfkGpzt5twGP1kGba8tVg80Wh19zz9Kim0su8X5hYZOva-vhvhthA"
---

Source (Apple): Metal Shading Language Specification, §6.3 (v4.1, 2026-06-04).
PDF: <https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf>
(The MSL standard-library functions are only in the spec PDF — there is no DocC HTML/JSON
page for them, so this was extracted from the PDF, not the usual DocC-JSON endpoint.)

Header `<metal_common>`. **`T` is scalar or vector `half` or `float` only** — the whole
table is floating-point. (Integer `clamp`/`abs`/`min`/`max`/`absdiff` live in §6.4
`<metal_integer>`; float `fmin`/`fmax`/`fabs` and the `min`/`max`/`min3`/`max3` family are
§6.6 `<metal_math>`.)

## Table 6.1

| Function                              | Returns                                                                                                                                    |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `T clamp(T x, T minval, T maxval)`    | `fmin(fmax(x, minval), maxval)`. **Undefined if `minval > maxval`.**                                                                       |
| `T mix(T x, T y, T a)`                | linear blend `x + (y - x) * a`. **Undefined if `a` outside [0, 1].**                                                                       |
| `T saturate(T x)`                     | clamp to [0.0, 1.0]. Float-only (like everything here).                                                                                    |
| `T sign(T x)`                         | `1.0` if x > 0, `-0.0`/`+0.0` for ∓0, `-1.0` if x < 0, **`0.0` if x is NaN**.                                                              |
| `T step(T edge, T x)`                 | `0.0` if `x < edge`, else `1.0`.                                                                                                           |
| `T smoothstep(T edge0, T edge1, T x)` | Hermite 0→1 over (edge0, edge1): `t = clamp((x-edge0)/(edge1-edge0), 0, 1); t*t*(3-2*t)`. Undefined if `edge0 >= edge1` or any arg is NaN. |

## Precise vs fast

For single precision, **`clamp` and `saturate` have fast and precise variants** (the only
two in this section). Difference is NaN handling only: fast = NaN behavior undefined,
precise = IEEE 754 NaN rules. Selected by the fast-math compile mode (default **fast** —
see `math-functions-and-numeric-parity.md`) or explicitly via `metal::fast::` /
`metal::precise::`. So under the backend's default build, `clamp(NaN, lo, hi)` is
undefined — clamp bounds an overflow _before_ it happens; it does not launder a NaN.

---

### Relevance to the CT2 Metal backend

All in `src/metal/kernels/kernels_msl.h`:

- **The Gemma2 NaN fix is a `clamp`**: `ct2_tanh_safe(x)` = `tanh(clamp(x, -15.0f,
15.0f))` — Metal's `tanh` computes `(exp(2x)-1)/(exp(2x)+1)`, which overflows to
  Inf/Inf = NaN for large |x|; tanh(±15) is already ±1.0 in float32, so the clamp is exact
  in the saturated region. Used by the GELU-tanh and Tanh cases of
  `ct2_apply_activation`. Note the order: clamp the _argument_, because clamping the NaN
  _result_ would hit the fast-variant undefined-NaN behavior above.
- **`ct2_quantize_s8_impl` does NOT clamp** the quantized value to ±127: the scale is
  `127 / amax(row)`, so `|x * scale| <= 127` by construction and the result is cast
  straight to `char` (after `rint` or truncation). Don't "add the missing clamp"
  reflexively — the CPU reference it must match does the same.
- The float `max`/`fabs` calls in the reductions (`local_max = max(local_max, x[j])` in
  `ct2_softmax_*`, `max(local_amax, fabs(...))` in quantize) are §6.6 math functions, not
  this table; `ct2_apply_activation`'s ReLU is `max(v, 0.0f)` likewise.
- `saturate`/`mix`/`step`/`smoothstep` are currently unused in the backend; they are
  float/half-only, so none of them applies to the int8 path.
