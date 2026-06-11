# MTLGPUFamily and feature availability checks

Sources: https://developer.apple.com/documentation/metal/mtlgpufamily,
https://developer.apple.com/documentation/metal/mtldevice/supportsfamily(_:),
https://developer.apple.com/documentation/metal/mtldevice/maxbufferlength
(fetched via DocC JSON, 2026-06-11). Per-family feature matrices live in Apple's
Metal-Feature-Set-Tables.pdf (linked from the DocC pages):
https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf

`MTLGPUFamily` "represents the functionality for families of GPUs"; query support with
`MTLDevice.supportsFamily(_:) -> Bool`. Families are cumulative capability tiers — a device
that supports `apple9` also supports `apple8` and below.

## The families (Apple Silicon view)

Verified directly from the DocC case abstracts:

| Case              | Corresponds to (DocC wording)                                                                                                                       |
| ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| `apple9`          | **Apple A17, M3, and M4 GPUs**                                                                                                                      |
| `apple8`          | Apple A15, A16, and **M2** GPUs                                                                                                                     |
| `apple7`          | Apple A14 and **M1** GPUs                                                                                                                           |
| `apple6`          | A13                                                                                                                                                 |
| `apple5`          | A12                                                                                                                                                 |
| `apple4`          | A11                                                                                                                                                 |
| `apple1`–`apple3` | A7–A10 (pre-Apple-Silicon-Mac era)                                                                                                                  |
| `apple10`         | exists as a case; DocC abstract is empty — chip mapping unverified (presumably A18/M5-class; treat as "commonly reported" until Apple documents it) |

So: **every Apple Silicon Mac is at least `apple7`** (M1). Cross-vendor tiers: `common1`–
`common3` (lowest common denominator across all Metal GPUs), `mac2` (Mac-class features),
and the version tiers `metal3` / `metal4` ("Represents the Metal 3 features" — i.e. checks
the Metal 3 / Metal 4 feature umbrella rather than a hardware generation).

## Capability checks a compute backend cares about

- **SIMD-group reductions** (`simd_sum`, `simd_shuffle_down`, …) and **`simdgroup_matrix`**:
  available on Apple-family GPUs from `apple7` (MSL spec: simdgroup-matrix is Metal 2.3+ on
  Apple7+ — see this skill's `simdgroup-matrix-functions.md` and
  `simd-group-functions.md`). Gate: `supportsFamily(.apple7)`.
- **bfloat**: an MSL 3.1 language feature (see `msl-data-types-and-alignment.md`), not a
  per-family DocC query — gate on the Metal version umbrella (`supportsFamily(.metal3)`)
  plus OS supporting MSL 3.1. There is no dedicated `supportsBfloat` API; flagging this as
  spec-derived, not DocC-verified.
- **Buffer capacity**: `MTLDevice.maxBufferLength` — "the largest amount of memory a GPU
  device can allocate to a buffer", documented minimum 256 MB. On unified-memory Macs it is
  a large fraction of RAM, but it is a queryable limit, not infinity — a single fp16 weight
  blob or KV cache must fit under it.
- Per-family numeric limits (argument-table sizes, threadgroup memory, etc.) are in the
  Feature Set Tables PDF, not in DocC symbol pages.

`supportsFamily(_:)` requires macOS 10.15+ — always true for this backend (Apple Silicon
implies macOS 11+).

### Relevance to the CT2 Metal backend

- **The backend currently checks nothing.** `MetalContext()` in `src/metal/device.mm`
  calls `MTLCreateSystemDefaultDevice()` and `newCommandQueue` and assumes the rest; there
  is no `supportsFamily` call anywhere in `src/metal/` (grep — zero hits). Development has
  been on a single machine (M4 Max = `apple9`), so every assumption silently holds.
- What the kernels actually assume, i.e. what a production hardening pass in the
  `MetalContext()` constructor (`src/metal/device.mm`) would assert at init:
  1. `supportsFamily(MTLGPUFamilyApple7)` — the int8 GEMV's `simd_sum`/lane-strided
     reduction (`ct2_gemv_s8` in `src/metal/kernels/kernels_msl.h`) and the SIMD-group
     reduction patterns need Apple7+; this also covers fp16 arithmetic and nonuniform
     threadgroup dispatch. Fail loudly (the backend already throws on missing device/queue
     — same pattern).
  2. `maxBufferLength` vs. the largest single weight/KV allocation — `MetalAllocator`
     (`src/metal/allocator.mm`) currently just throws if `newBufferWithLength:` returns
     nil; a proactive check gives a better error than "failed to allocate buffer of size N".
  3. If a bfloat path ever lands: gate on `supportsFamily(MTLGPUFamilyMetal3)` + OS check,
     since it's a language-version feature.
- Until a second machine (M1/M2, or iOS-class) actually runs this backend, these are
  documented-but-untriggered: every Apple Silicon Mac is ≥ `apple7`, so the realistic
  failure mode is a future kernel quietly using an `apple8`/`apple9`-only feature — that's
  the moment to add the gate, with the Feature Set Tables PDF as the authority.
