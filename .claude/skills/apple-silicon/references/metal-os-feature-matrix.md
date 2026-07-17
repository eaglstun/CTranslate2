---
title: "Metal feature ↔ version ↔ OS floor — the matrix that stops version-guessing"
summary: >-
  A lookup matrix mapping Metal features to their MSL/API requirement and
  macOS OS floor to stop version-guessing: MTLBlitCommandEncoder (10.11),
  function constants, SIMD-group reductions, simdgroup_matrix (MSL 2.3,
  Apple7/M1), atomic_float (MSL 3), MTLIOCommandQueue and metal3 query (13.0),
  bfloat (MSL 3.1), mathMode and MPSNDArrayQuantizedMatrixMultiplication
  (15.0), and MTLTensor/MPP matmul2d/metal4 (26.0). It notes the MSL-version
  column is authoritative and hardware gates are separate from OS gates.
  Crucially it records that this repo is version-silent: CMakeLists WITH_METAL
  requires only APPLE+arm64 with no deployment pin, src/metal/device.mm has no
  availability checks, and the newest language dependency actually used is
  Metal-2-era SIMD-group reductions, so the implicit M1/macOS-11 floor is
  untested (only run on M4 Max).
semantic_id: "yhw4zo2Ou13Tw-1VxuULvYc8u2uTn6shDldqvJT9wMQqHXkGwa4rUg10fh2v7587jC8U28z7lYZOtSzzRjh_pA"
---

Sources: each row cites either an existing reference in this skill (which carries the
original Apple citation) or a DocC platform stamp fetched 2026-06-11
(`MTLFunctionConstantValues`, `MTLGPUFamily.metal3`, `MTLGPUFamily.metal4`,
`MTLBinaryArchive`, `MTLIOCommandQueue`, `MTLBlitCommandEncoder` pages). Repo facts from
`CMakeLists.txt` and `src/metal/` read the same day.

## The matrix

| Feature                                               | Needs (MSL / API)         | OS floor (macOS)                      | Documented in                                          |
| ----------------------------------------------------- | ------------------------- | ------------------------------------- | ------------------------------------------------------ |
| `MTLBlitCommandEncoder`                               | Metal 1                   | 10.11                                 | DocC stamp; `blit-command-encoder.md`                  |
| Function constants (`MTLFunctionConstantValues`)      | MSL 1.2-era API           | 10.12                                 | DocC stamp; `pipeline-and-library-compilation.md`      |
| SIMD-group reductions (`simd_sum`, shuffles)          | macOS Metal 2 / 2.1+      | 10.13-era                             | `simd-group-functions.md`                              |
| `simdgroup_matrix` (8×8 WMMA-style)                   | MSL 2.3, **Apple7+ (M1)** | 11-era                                | `simdgroup-matrix-functions.md`                        |
| `simd_shuffle_and_fill_up/down`                       | Metal 2.4                 | 12-era                                | `simd-group-functions.md`                              |
| `MTLBinaryArchive`                                    | API                       | 11.0                                  | DocC stamp; `binary-archives-and-pso-caching.md`       |
| `atomic_float` (device `fetch_add/sub` only)          | MSL 3 ("Metal 3+")        | 13-era                                | `atomic-functions.md`                                  |
| `MTLGPUFamily.metal3` query                           | API                       | 13.0                                  | DocC stamp; `mtlgpufamily-and-feature-availability.md` |
| `MTLIOCommandQueue` (fast resource loading)           | API ("Metal 3")           | 13.0 (compressed file handles 14.0)   | DocC stamps; `mtlio-command-queue.md`                  |
| `bfloat` scalar/vector type                           | **MSL 3.1**               | 14-era                                | `msl-data-types-and-alignment.md`                      |
| `MTLCompileOptions.mathMode` (fast/relaxed/safe)      | API                       | **15.0** (replaces `fastMathEnabled`) | `pipeline-and-library-compilation.md`                  |
| `MPSNDArrayQuantizedMatrixMultiplication` (int8/int4) | MPS API                   | **15.0** (iOS 18)                     | `mpsndarray.md`                                        |
| `MTLGPUFamily.metal4` query                           | API                       | 26.0                                  | DocC stamp                                             |
| `MTLTensor` / MPP `matmul2d` (incl. char×char→int)    | Metal 4 / MSL §7          | **26.0**                              | `metal4-tensors-and-mpp.md`                            |

"-era" floors are inferred from the Metal version, not from a fetched platform stamp —
treat them as approximate; the MSL-version column is the authoritative gate (the spec
states MSL versions, not macOS versions). Hardware gates are separate from OS gates:
`simdgroup_matrix` needs Apple7+, which every Apple Silicon Mac satisfies (M1 = apple7,
M2 = apple8, M3/M4 = apple9 — `mtlgpufamily-and-feature-availability.md`).

## Deployment reality: what THIS repo requires today

Verified by grep on 2026-06-11:

- **Build gate:** `CMakeLists.txt` (`if (WITH_METAL)`) requires only APPLE + arm64 and
  links Metal/MPS/Foundation. No deployment-target pin, no Metal-version compile flag.
- **Runtime gate:** none. `src/metal/device.mm` checks only that a default
  `MTLDevice` exists; there is no `supportsFamily`, `@available`, or
  `__builtin_available` anywhere in `src/metal/`.
- **Features actually used:** MPSMatrix-family GEMM, runtime `newLibraryWithSource` with
  a default-init `MTLCompileOptions`, Shared buffers, and MSL kernels whose newest
  language dependency is SIMD-group reductions (`simd_sum` / `simd_shuffle_down` —
  Metal 2-era). Nothing from the macOS 13+/15+/26 rows is load-bearing.
- **Implicit floor (inference, not a measurement):** any Apple Silicon Mac ≈ M1 /
  macOS 11. But the backend has only ever run on one machine — the M4 Max (apple9,
  macOS 26.x per `mtlgpufamily-and-feature-availability.md`) — so the floor is untested.

## How to use this card

- Before reaching for a row's feature, open the cited reference — it carries the sharp
  edges (e.g. `atomic_float` is add/sub-only; `simdgroup_matrix` has no int8 element
  type; `mathMode` defaults to _relaxed_ on Apple silicon).
- If a change starts using anything at macOS 13.0 or later, that is the moment the
  backend's "no availability checks" stance breaks: add the `supportsFamily` /
  `@available` assert listed in `mtlgpufamily-and-feature-availability.md` rather than
  letting older systems fail at first dispatch.
- macOS-26-only surfaces (`MTLTensor`, MPP) are evaluation targets, not portable code
  paths — anything shipped on them needs a fallback to the current kernels.

### Relevance to the CT2 Metal backend

- `CMakeLists.txt` (the `WITH_METAL` block) and `src/metal/device.mm` are the only
  places version policy could live today — both are version-silent, which is exactly why
  this index exists: the floor is implicit and easy to break accidentally.
- The int8 path's future options ladder reads straight off the matrix:
  `MPSNDArrayQuantizedMatrixMultiplication` (macOS 15) and MPP `matmul2d` (macOS 26) vs
  the version-free hand-tiled `ct2_gemm_s8` in `src/metal/kernels/kernels_msl.h`.
- When a benchmark or hardening pass pins down a real floor (e.g. CI on an M1), record
  it here and in `mtlgpufamily-and-feature-availability.md`.
