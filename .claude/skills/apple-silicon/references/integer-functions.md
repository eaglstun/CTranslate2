---
title: "Integer functions"
summary: >-
  The MSL <metal_integer> standard-library functions from spec section 6.4
  Table 6.2: saturating arithmetic (addsat, subsat, madsat), averaging (hadd,
  rhadd), high-half products (mulhi, madhi), 24-bit mul24/mad24,
  min3/max3/median3, and bit manipulation (clz, ctz, popcount, reverse_bits,
  rotate, extract_bits, insert_bits). Its distinctive angle is an honest
  inventory of what an int8 GEMM kernel actually needs, concluding almost none
  of the table: plain int/int4 multiply-accumulate suffices for
  int8xint8->int32 because a char product is at most 16129 and int32
  accumulates ~2^17 terms before overflow. Explains why mul24/mad24,
  saturating helpers, clamp, and mulhi/madhi are irrelevant to CT2's float-
  side dequant and bit-exact contract, tying back to ct2_gemm_s8 and
  ct2_gemv_s8 in kernels_msl.h.
semantic_id: "yhowxo4vq32zw-39z9UJvYcYJ2uTyqMlDgVmOPSX5t4-HWkGR60vVy8xHh19rxwanIkkk837hY5-9e2hBrg9oA"
---

Source (Apple): Metal Shading Language Specification, §6.4, Table 6.2 (v4.1, 2026-06-04).
PDF: <https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf>
(The MSL standard-library functions are only in the spec PDF — there is no DocC HTML/JSON
page for them, so this was extracted from the PDF, not the usual DocC-JSON endpoint.)

Header `<metal_integer>`. `T` = any scalar or **vector** integer type (so `char4`/`int4` work
componentwise); `Tu` = the corresponding unsigned type; `T32` = 32-bit `int`/`uint` only.

## Arithmetic

```metal
T   abs(T x)                       // |x|
Tu  absdiff(T x, T y)              // |x-y| without modulo overflow
T   addsat(T x, T y)               // x + y, saturated
T   subsat(T x, T y)               // x - y, saturated
T   madsat(T a, T b, T c)          // a*b + c, saturated
T   hadd(T x, T y)                 // (x + y) >> 1, intermediate sum doesn't overflow
T   rhadd(T x, T y)                // (x + y + 1) >> 1, ditto
T   mulhi(T x, T y)                // high half of the full product x*y
T   madhi(T a, T b, T c)           // mulhi(a, b) + c
T32 mul24(T32 x, T32 y)            // 24-bit multiply, 32-bit result (Metal 2.1+)
T32 mad24(T32 x, T32 y, T32 z)     // mul24(x, y) + z (Metal 2.1+)
```

`mul24`/`mad24` are only defined for operands in [-2^23, 2^23-1] (signed) / [0, 2^24-1]
(unsigned); outside that, the result is implementation-defined.

## Min / max / clamp

```metal
T clamp(T x, T minval, T maxval)   // min(max(x, minval), maxval); UB if minval > maxval
T min(T x, T y)        T max(T x, T y)
T min3(T x, T y, T z)  T max3(T x, T y, T z)  T median3(T x, T y, T z)   // Metal 2.1+
```

## Bit manipulation

```metal
T clz(T x)                          // leading zeros (x==0 -> bit width)
T ctz(T x)                          // trailing zeros (x==0 -> bit width)
T popcount(T x)                     // nonzero bits
T reverse_bits(T x)                 // bit reversal (Metal 2.1+)
T rotate(T v, T i)                  // rotate left, componentwise
T extract_bits(T x, uint offset, uint bits)            // Metal 1.2+; sign-extends for signed T
T insert_bits(T base, T insert, uint offset, uint bits) // Metal 1.2+
// Metal 4.1+: interleave / deinterleave — (de)interleave even/odd bits between an n-bit
// pair and a 2n-bit value; pairs (uchar<->ushort), (ushort<->uint), (uint<->ulong).
```

## What an int8 kernel actually needs (honest inventory)

- **Plain `*` and `+=` on `int`/`int4` suffice for int8×int8→int32.** A `char`×`char` product
  is at most 127·127 = 16129 (min −128·127), so int32 accumulates ~2^17 such terms before any
  overflow risk — transformer depths (k ≤ a few thousand) are nowhere close. No saturating or
  widening helper is needed, and none is used.
- **`mul24`/`mad24` are irrelevant here**: sign-extended int8 operands trivially fit 24 bits,
  but Apple GPU int32 multiply doesn't need the hint and the spec gives no Apple-specific
  speed promise — measure before bothering (the project's perf culture; see
  `dispatch-overlap-and-perf-model.md`).
- **`addsat`/`subsat`/`madsat` would matter only for sub-int32 accumulation** (int16
  accumulators to double SIMD width) — a wrong-results-vs-speed tradeoff CT2's bit-exact int8
  contract rules out.
- **`clamp` is the saturation tool for quantization** if the scheme ever needs it (the current
  symmetric scale makes it unnecessary; see `conversion-and-packing-functions.md`).
- **`mulhi`/`madhi`** are the fixed-point requantization primitives (integer-only dequant à la
  ONNX Runtime / gemmlowp). CT2 dequantizes in float, so unused — but they're the spec surface
  if an integer-only epilogue is ever explored.

---

### Relevance to the CT2 Metal backend

- `ct2_gemm_s8` and `ct2_gemv_s8` (`src/metal/kernels/kernels_msl.h`) deliberately use **none
  of Table 6.2**: the inner loops are `acc += av.x * bv` style `int4` MACs — exact by the range
  argument above, which is what makes the GEMM "bit-exact by construction" (kernel header
  comment) and lets `tests/metal_test.cc` compare int32 outputs with zero tolerance.
- The quantize/dequantize kernels also stay off this table — their work is float-side
  (`fabs`, `max`, `precise::divide`, `rint`); the integer story there is the cast, covered in
  `conversion-and-packing-functions.md`.
- If a future kernel packs four int8 MACs into a software dot product over packed `uint`
  lanes, `extract_bits` (sign-extending) is the spec-blessed unpacking primitive — but the
  existing `char4`-pointer + `int4()` widening already compiles to the same vector loads, so
  there is no current gap to fill.
