---
name: apple-silicon
description: >-
  Apple Silicon GPU / Metal reference for the CTranslate2 Metal backend. Use when
  writing or debugging Metal code in src/metal/ — MSL compute kernels, threadgroup
  & grid sizing, MPSMatrixMultiplication (GEMM), command-buffer dispatch, CPU↔GPU
  synchronization, resource storage modes, and unified memory. Pull the matching
  reference file before reasoning about Metal API specifics; do not answer from memory.
---

# Apple Silicon (Metal) reference

Condensed, source-cited notes from Apple's developer documentation, each tied to how
the CTranslate2 Metal backend (`src/metal/`, `-DWITH_METAL=ON`) actually uses it. See
`METAL_BACKEND.md` at the repo root for the backend's design and milestone status; the
files here are the upstream-API ground truth behind it.

**When working on Metal, read the relevant reference first — these APIs have sharp
edges (no `erf` in MSL, every bound buffer must exist, row-major vs column-major) that
are easy to get wrong from memory.**

## References

- **[references/op-graduation-playbook.md](references/op-graduation-playbook.md)**
  — The CT2 procedure for moving an op onto a Metal GPU kernel: targeted routing (not a
  dispatch switch), the fp16 real-kernel-vs-bypass decision and the `metal::synchronize()`
  flush nuance the bypass needs, MSL landmines (no `erf`, lazy compile, `1/sqrt` for
  bit-parity, every-buffer-must-be-bound, host scalars by value), and the parity-via-
  existing-suite verification strategy. _Read first when adding/routing any kernel; the
  three API references below are the ground truth it builds on._

- **[references/compute-kernels-and-dispatch.md](references/compute-kernels-and-dispatch.md)**
  — Running a compute kernel: `MTLDevice` → library/function → `MTLComputePipelineState`
  → command queue/buffer → `MTLComputeCommandEncoder` → commit. Writing MSL `kernel`
  functions, address-space keywords, `[[thread_position_in_grid]]`. Threadgroup & grid
  sizing: `threadExecutionWidth`, `maxTotalThreadsPerThreadgroup`, `dispatchThreads`
  (non-uniform, no bounds check) vs `dispatchThreadgroups` (manual, needs a guard).
  _Read when adding/editing an MSL kernel or its dispatch in `primitives.mm`._

- **[references/simd-group-functions.md](references/simd-group-functions.md)**
  — SIMD-group / quad-group functions (`simd_sum`, `simd_max`, `simd_shuffle_down`,
  `simd_broadcast`, ballot/vote, prefix scans) for cross-lane data sharing without
  threadgroup memory or barriers, plus the canonical two-level threadgroup reduction. The
  reference for optimizing CT2's row-reduction kernels (softmax/rms*norm/layer_norm) off
  their current 256-thread tree reductions. \_Read when writing or optimizing a reduction
  kernel in `kernels_msl.h`.*

- **[references/math-functions-and-numeric-parity.md](references/math-functions-and-numeric-parity.md)**
  — MSL math functions (`sqrt`/`rsqrt`/`divide`/`fma`/`exp`/`log`; no `erf`) and the
  fast-vs-precise numeric reality: the library compiles with DEFAULT FAST MATH, under
  which `sqrt` is `x*rsqrt(x)` and FMA contracts — so CPU parity rides on tolerance, the
  `1.0f/sqrt` (not `rsqrt`) spelling, and float-accumulated reductions. Includes the ULP
  tables and the `mathMode = Safe` lever. _Read when writing/debugging a norm, softmax,
  or any reduction kernel whose output must match the CPU reference._

> Norm **placement** (pre / post / pre-post sandwich / parallel-residual) is CPU
> orchestration, not Metal — it moved to the **`ct2-internals`** skill
> (`norm-placement-in-transformers.md`). A norms task spans both skills: structure there,
> kernel numerics (this skill's `math-functions-and-numeric-parity.md`) here.

- **[references/mps-matrix-multiplication.md](references/mps-matrix-multiplication.md)**
  — `MPSMatrixMultiplication` for GPU GEMM: `C = α·op(A)·op(B) + β·C`, initializer
  params, `encode(commandBuffer:…)`, `MPSMatrix`/`MPSMatrixDescriptor` (row-major,
  `rowBytes`), batched origins. Why operands-at-encode-time enables shape-keyed object
  caching. _Read when touching `src/metal/gemm.mm` or GEMM routing._

- **[references/storage-and-synchronization.md](references/storage-and-synchronization.md)**
  — Unified memory and storage modes (Shared / Private / Memoryless); why the whole
  backend rides on Shared + the CPU-addressable `contents` pointer. CPU↔GPU
  synchronization: async commit, completion handlers, semaphores, and how that maps to
  CT2's global `flush()` / `synchronize()` model and the global-vs-thread-local
  command-buffer lesson. _Read when touching allocation, `device.mm` command-buffer
  lifecycle, or debugging stale/garbage GPU reads. Covers the **mechanics**; the
  **performance reasoning** is the next reference._

- **[references/dispatch-overlap-and-perf-model.md](references/dispatch-overlap-and-perf-model.md)**
  — **The canonical home for the backend's perf conclusions.** The per-op GPU-API floor and
  the CPU/GPU **overlap principle** (per-op commit lets the GPU run op N while the CPU
  encodes N+1), the prefill-wins / decode-loses regime split, and the WINS (async batching
  ~20%, MPS-object shape-cache ~35%, the fp16 `Add` fix 27×). Critically the **graveyard**:
  command-buffer reuse — tried, parity-passed, measured −23% on bs8 prefill, REVERTED,
  here's why — so nobody re-digs it. _Read before chasing a perf change or when tempted to
  "batch the command buffers."_

- **[references/benchmarking-and-profiling.md](references/benchmarking-and-profiling.md)**
  — The methodology that produced every number above: the `DISABLED_Benchmark*` harness +
  gtest flags, `CT2_LLM_MODEL` / `CT2_LLM_PROFILE` / `ENABLE_PROFILING`, and the
  **probe-isolation trick** (commit many, flush once) that separates per-op encode cost
  from GPU execution — how the 0.042→0.031 ms floor was found. "Profile, don't guess," with
  the worked `Add`-regression tale. _Read before measuring a change or claiming a speedup._

### int8 path (project-proven, 2026-06)

- **[references/int8-gemm-kernel-design.md](references/int8-gemm-kernel-design.md)**
  — The hand-tiled int8×int8→int32 GEMM (`ct2_gemm_s8`): why hand-tiled is the only native
  path (MPS is float-only, `simdgroup_matrix` has no int8), the 64×64 tile / 32-deep k chunks
  / 4×4 `int4` micro-tile, transposes resolved at tile-load, the beta==0 + integral-alpha
  exactness contract, the Phase-1 shim graveyard, and the ALU-bound ~3–5×-slower-than-fp16
  reality vs the −42% RSS win. _Read before touching `gemm_s8` or expecting a tiling tweak to
  beat MPS fp16 at large m._

- **[references/int8-gemv-simdgroup-decode.md](references/int8-gemv-simdgroup-decode.md)**
  — The small-m SIMD-group GEMV (`ct2_gemv_s8`): one SIMD-group per output element,
  lane-strided `char4` k-loop + `simd_sum`, the exact routing condition (Dense layout, m≤8,
  4-byte-aligned k/lds/offsets), the 8-SIMD-groups-per-threadgroup host coupling, and the
  bandwidth-bound arithmetic of why int8 **beats** fp16 MPS at decode (lm*head 0.49 vs
  0.84 ms). \_Read when touching decode-path GEMMs or anything that could silently break the
  alignment preconditions.*

- **[references/quantize-dequantize-kernels.md](references/quantize-dequantize-kernels.md)**
  — The three kernels around the int8 GEMM: `ct2_quantize_s8_*` (one 256-thread threadgroup
  per row, amax tree reduce, `precise::divide` + `rint` for bit-parity with the CPU),
  `ct2_dequantize_s8_*` (reciprocal-then-multiply, spelled like the CPU kernel), and the
  `ct2_dequant_gemm_out_*` epilogue (int32 → /(a*scale·b_scale) + bias + all 7 activations,
  dummy-bias buffer binding). \_Read when touching quantization parity, scales, or fusing the
  Dense epilogue.*

### MSL spec — language & stdlib

- **[references/simdgroup-matrix-functions.md](references/simdgroup-matrix-functions.md)**
  — SIMD-group 8×8 matrix (WMMA-style) matmul: `simdgroup_load/store`,
  `simdgroup_multiply[_accumulate]`, and THE type table (§2.4): **half/bfloat/float only —
  no integer element types**, the spec ground truth for why `ct2_gemm_s8` is hand-tiled.
  Metal 2.3+/Apple7+; Metal 4 steers toward Tensors + MPP instead. _Read if a future
  fp16 GEMM moves off MPS or fused attention is attempted (closes the old §6.8 TODO)._

- **[references/conversion-and-packing-functions.md](references/conversion-and-packing-functions.md)**
  — Conversions & reinterpretation: float→int casts round **toward zero** with **no
  saturation** (why `rint` precedes the `(char)` cast in `ct2_quantize_s8`), `as_type<T>`
  bit reinterpretation vs the pointer-cast `char4` loads the int8 GEMM actually uses,
  NaN→0, §6.15 norm-pack functions (wrong tool for raw int8), Metal 4.1 packed-numeric
  templates (the future int4 path). _Read when writing any quantize/dequantize or packed-
  load code._

- **[references/integer-functions.md](references/integer-functions.md)**
  — §6.4 integer builtins lookup table (`abs/absdiff`, `addsat/subsat/madsat`,
  `mulhi/madhi`, `mul24/mad24`, `clz/ctz/popcount`, `extract_bits`…) with an honest
  inventory: the int8 GEMM/GEMV need **none of them** — plain int32 `*`/`+=` is exact for
  char×char at transformer depths; `clamp` and `mulhi` are the ones to reach for if the
  quantization scheme ever changes. _Read when an integer kernel tempts you toward a
  builtin._

- **[references/atomic-functions.md](references/atomic-functions.md)**
  — §6.16 atomics: `atomic_int/uint/bool/ulong/float` only (no char/short/half — int8
  partials must widen), **relaxed is the only memory order for atomic ops** pre-Metal-4.1,
  fetch*add/max/min table, `atomic_thread_fence` + thread scopes, and why the current
  kernels are deliberately atomics-free (each threadgroup owns its output tile) — int32
  split-k via `fetch_add` would stay bit-exact, float wouldn't. \_Read before adding any
  cross-threadgroup accumulation.*

- **[references/threadgroup-and-simdgroup-synchronization.md](references/threadgroup-and-simdgroup-synchronization.md)**
  — `threadgroup_barrier`/`simdgroup_barrier` and the `mem_flags` variants (what each
  orders; `mem_none` = execution-only), the all-threads-must-reach-it divergence rule
  (per-iteration in loops), and when SIMD-group functions need no barrier at all. Maps to
  the int8 GEMM's load→barrier→MAC→barrier loop and the uniform-early-exit guards.
  _Read when adding a barrier, a row guard, or any threadgroup-memory phase._

- **[references/msl-address-spaces.md](references/msl-address-spaces.md)**
  — device / constant / threadgroup / thread: access rules, the no-address-space-cast rule,
  program-scope-must-be-constant (+ core-constant-expression init), and why kernel args are
  `device T*` for arrays vs `constant T&` for setBytes scalars. Covers the int8 GEMM's
  threadgroup tiles and the in-space `char4` reinterpretation trick.
  _Read when declaring kernel signatures or staging tiles in `kernels_msl.h`._

- **[references/msl-data-types-and-alignment.md](references/msl-data-types-and-alignment.md)**
  — Scalar/vector size & alignment tables from §2 (no `double`; `bfloat` = Metal 3.1+),
  the vec3-pads-to-vec4 trap (`sizeof(float3)` = 16), `packed_` types for byte-tight
  host-shared layouts, and why `simdgroup_matrix` has no int8 element type. Maps to the
  int8 kernels' `char4`/`int4` widening and the half-storage/float-accumulate pattern.
  _Read before sharing a struct with the host or reinterpreting buffer element types._

- **[references/common-functions.md](references/common-functions.md)**
  — §6.3 Table 6.1 (`clamp`/`mix`/`saturate`/`sign`/`step`/`smoothstep`): float/half ONLY
  (integer clamp lives in §6.4), and fast-vs-precise exists just for clamp/saturate (NaN
  handling). Home of the Gemma2 fix's `clamp(x,-15,15)` in `ct2_tanh_safe` — and the note
  that quantize deliberately does NOT clamp to ±127. _Lookup card; read when a kernel
  leans on clamp/saturate semantics near NaNs._

- **[references/relational-and-select-functions.md](references/relational-and-select-functions.md)**
  — §6.5 Table 6.3 (`isnan`/`isinf`/`isfinite`, `select` — true picks the SECOND arg,
  `all`/`any`, `signbit`) plus the fast-math caveat: under the default build the compiler
  may assume no-NaN and fold `isnan` to false, so in-kernel tripwires need
  `math_mode(safe)`; the Gemma2 NaN hunt worked because the checks ran host-side.
  _Lookup card; read when writing NaN tripwires or branchless vector guards._

## Conventions for this skill

- Each reference cites its Apple source URL at the top and ends with a
  `### Relevance to the CT2 Metal backend` section connecting the API to specific files.
- Keep SKILL.md lean: one-line pointers only. Detail lives in `references/`.
- **Code citations here drift.** Run `bash scripts/audit-citations.sh` (`-q` for
  problems-only) to flag missing files / out-of-range lines / ambiguous basenames; it
  prints each cited line so you can eyeball content drift it can't auto-detect. Most refs
  here cite by symbol/filename (drift-proof) — but a "verified on DATE" stamp dies the
  moment the file changes, so re-run rather than trust it.
- To add a topic from Apple's **DocC docs** (Metal framework API — `MTLDevice`,
  `MPSMatrixMultiplication`, etc.): fetch via the DocC JSON endpoint
  (`https://developer.apple.com/tutorials/data/documentation/<path>.json` — the human
  doc pages are JS SPAs that return only a title to scrapers), write a new
  `references/<topic>.md` with the source URL + a CT2-relevance section, and add a
  pointer above.
- To add a topic from the **MSL standard library** (kernel-side functions / language —
  these are NOT on any DocC page): extract from the vendored spec PDF at
  `sources/Metal-Shading-Language-Specification.pdf` (WebFetch rejects it — >10MB; needs
  `poppler`). Recipe:
  ```bash
  pdftotext -layout sources/Metal-Shading-Language-Specification.pdf /tmp/msl.txt
  grep -nE "^\s*6\.[0-9]" /tmp/msl.txt   # find the section's TOC entry + page
  sed -n 'START,ENDp' /tmp/msl.txt        # pull the section's line range
  ```

## TODO: more MSL references worth mining from the spec PDF

Carve these into `references/*.md` when a task needs them (CT2-relevance, in priority order).
Don't pre-build speculatively — pull on demand, same discipline as the rest of the backend.

- ~~**§6.8 SIMD-Group Matrix Functions**~~ — DONE: see `references/simdgroup-matrix-functions.md`
  (and the proven no-int8 fact that decided the int8 GEMM design).
- ~~**§6.16 Atomic Functions**~~ — DONE: see `references/atomic-functions.md`.
- ~~**§6.6 Math Functions**~~ — DONE: see `references/math-functions-and-numeric-parity.md`
  (math builtins, no-`erf`, fast-vs-precise ULP tables, the fast-math parity trap).
- ~~**§6.10.1 / §4.4.1** Threadgroup & SIMD-group **synchronization**~~ — DONE: see
  `references/threadgroup-and-simdgroup-synchronization.md`.
- ~~**§6.3 Common Functions**~~ — DONE: see `references/common-functions.md` (with §6.5
  relational/select in `references/relational-and-select-functions.md`).
- NOT worth mining for CT2: textures (§6.13), imageblocks (§6.14), graphics/fragment
  (§6.11), geometric (§6.9) — no render passes in this backend.
