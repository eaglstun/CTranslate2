---
title: "MSL data types, sizes & alignment"
summary: >-
  The MSL scalar, vector, packed, and matrix type tables with their sizes and
  alignments (from spec §2.1-2.5): the unsupported types (double, long long),
  the bfloat 3.1+ and half formats, and the trap that 3-component vectors pad
  to 4 (float3 is 16/16, char3 is 4/4) while packed_ variants stay byte-tight.
  It covers host/device struct-sharing gotchas — a C++ float[3] is not an MSL
  float3, simd::float3 is 16 bytes and matches. CT2 relevance: char4/int4 are
  the int8 GEMM/GEMV workhorses (4-byte aligned tile loads), half kernels
  accumulate in float because half's 11-bit significand can't survive a
  256-wide sum, simdgroup_matrix has no int8 element type (why int8 GEMM is
  hand-tiled not WMMA), and scalar params are 4-byte types passed by setBytes.
semantic_id: "yhww4o-Ou33Twe1dz8kLvYc4p0uT2rGhDlVmGPyX5twcHfkPS66tXS99Kh0_664IBE0s28z7AY5-_I-VzvQvpA"
---

Source (Apple): Metal Shading Language Specification, §2.1–2.5 (v4.1, 2026-06-04).
PDF: <https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf>
(The MSL type tables are only in the spec PDF — there is no DocC HTML/JSON page for them,
so this was extracted from the PDF, not the usual DocC-JSON endpoint.)

## Scalar types (§2.1, Tables 2.1–2.2)

**Not supported**: `double`, `long long`, `unsigned long long`, `long double`.

| Type (aliases)                             | Size | Align | Notes                                                       |
| ------------------------------------------ | ---- | ----- | ----------------------------------------------------------- |
| `bool`                                     | 1    | 1     |                                                             |
| `char` / `int8_t`, `uchar` / `uint8_t`     | 1    | 1     | two's complement                                            |
| `short` / `int16_t`, `ushort` / `uint16_t` | 2    | 2     |                                                             |
| `int` / `int32_t`, `uint` / `uint32_t`     | 4    | 4     |                                                             |
| `long` / `int64_t`, `ulong` / `uint64_t`   | 8    | 8     | Metal 2.2+                                                  |
| `size_t` / `ptrdiff_t`                     | 8    | 8     | 64-bit                                                      |
| `half`                                     | 2    | 2     | IEEE 754 binary16                                           |
| `bfloat`                                   | 2    | 2     | **Metal 3.1+**; truncated float (8-bit, 7 stored, mantissa) |
| `float`                                    | 4    | 4     | IEEE 754 single                                             |

Literal suffixes: `f/F` (float), `h/H` (half), `bf/BF` (bfloat), `u/U`, `l/L`.

## Vector types (§2.2, Table 2.3)

`charn ucharn shortn ushortn intn uintn longn ulongn halfn bfloatn floatn booln` for
n = 2, 3, 4 (also `vec<T,n>`). **Alignment = the full vector size, and a 3-component
vector is padded to the 4-component size** — the trap row:

| Type               | Size  | Align |     | Type            | Size      | Align |
| ------------------ | ----- | ----- | --- | --------------- | --------- | ----- |
| `char2`/`uchar2`   | 2     | 2     |     | `half2`         | 4         | 4     |
| `char3`/`uchar3`   | **4** | 4     |     | `half3`         | **8**     | 8     |
| `char4`/`uchar4`   | 4     | 4     |     | `half4`         | 8         | 8     |
| `short2`/`ushort2` | 4     | 4     |     | `float2`        | 8         | 8     |
| `short3`/`ushort3` | **8** | 8     |     | `float3`        | **16**    | 16    |
| `short4`/`ushort4` | 8     | 8     |     | `float4`        | 16        | 16    |
| `int2`/`uint2`     | 8     | 8     |     | `int3`/`uint3`  | **16**    | 16    |
| `int4`/`uint4`     | 16    | 16    |     | `bfloat2/3/4`   | 4/**8**/8 | 4/8/8 |
| `long2`            | 16    | 16    |     | `long3`/`long4` | **32**/32 | 32    |

`sizeof(float3) == sizeof(float4) == 16` (§2.2.1, stated explicitly). Component access:
`v[i]`, `.xyzw`/`.rgba` swizzles (no mixing `.xg`; no pointer-to-swizzle).

## packed\_ vector types (§2.2.3, Table 2.4) — byte-tight layouts

`packed_<scalar>n` (no `packed_bool`; `packed_long` Metal 2.3+, `packed_bfloat` 3.1+):
size = n × scalar size, **alignment = the scalar's alignment**. So `packed_float3` is
12 bytes / align 4 (vs `float3` 16/16); `packed_char4` is 4 bytes / align 1. Storage
format: loads/stores convert to/from aligned vectors; assignment + arithmetic operators
work; index access always, swizzles Metal 2.1+. Use when a buffer shared with the host
must have a C-style tight layout.

## Matrix types (§2.3, brief)

`halfnxm` / `floatnxm` / `matrix<T,c,r>` (n,m ∈ {2,3,4}); column-major construction; a
`floatnxm` is n `floatm` column vectors, so alignment follows the _column vector_ (e.g.
`float3x3`: 48 bytes, align 16). Not used by this backend (GEMM is MPS or hand-tiled).
SIMD-group matrices (§2.4): `simdgroup_half8x8`, `simdgroup_float8x8`,
`simdgroup_bfloat8x8` (3.1+) — element types are **half/bfloat/float only; no int8**.

## Host/device struct sharing gotchas (§2.5)

The compiler aligns per the tables and **assumes any kernel pointer argument is aligned as
its type requires**. When a C++ host struct is read as an MSL struct:

- A C++ `float[3]` array or a 12-byte xyz struct is NOT an MSL `float3` (16/16) — use
  `packed_float3` on the MSL side or pad to 16 on the host. Same for every 3-vector.
- `simd::float3` from `<simd/simd.h>` on the host IS 16 bytes — matches `float3`.
- `alignas` works in MSL for explicit control.
- `bool` is 1 byte — matches C++ `bool`, but don't mix with Obj-C `BOOL` assumptions.

---

### Relevance to the CT2 Metal backend

All in `src/metal/kernels/kernels_msl.h`:

- **`char4`/`int4` are the int8 workhorses**: `ct2_gemm_s8` reads tiles as
  `(const threadgroup char4*)` (4-byte loads, align 4 — guaranteed because `tid.{x,y}*4`
  is 4-aligned and tile rows are 64 bytes), widens with the `int4(char4)` constructor, and
  MACs in `int4`. `ct2_gemv_s8` strides `device const char4*` views of the operand rows —
  legal only because the host routes shapes where `k % 4 == 0` and the row pointers are
  4-aligned (char4 align = 4).
- **half kernels accumulate in float**: every `_half` kernel (`ct2_softmax_half`,
  `ct2_rms_norm_half`, …) keeps `threadgroup float scratch[…]` and casts `(float)x[j]` —
  half's 11-bit significand cannot survive a 256-wide sum. Pattern: half for
  storage/bandwidth, float for arithmetic.
- **bfloat exists in MSL (3.1+) but the backend has no bf16 path** — and note
  `simdgroup_matrix` has no int8 element type either (the comment above `ct2_gemm_s8`
  cites spec §2.4 for exactly that; it's why the int8 GEMM is hand-tiled, not WMMA).
- **Scalar params are 4-byte types on both sides**: `constant uint&`/`constant int&` args
  match the host's `uint32_t`/`int32_t` locals passed via `setBytes` (`primitives.mm`) —
  no struct, so no padding risk. If a param struct is ever introduced, the float3/vec3
  padding trap above is the thing to check first.
- Byte-generic kernels (`ct2_strided_copy_bytes`, `ct2_gather_bytes`) use `uchar` (1/1) so
  any dtype can ride them.
