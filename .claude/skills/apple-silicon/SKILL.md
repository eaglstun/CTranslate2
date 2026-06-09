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
  lifecycle, or debugging stale/garbage GPU reads._

## Conventions for this skill

- Each reference cites its Apple source URL at the top and ends with a
  `### Relevance to the CT2 Metal backend` section connecting the API to specific files.
- Keep SKILL.md lean: one-line pointers only. Detail lives in `references/`.
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

- **§6.8 SIMD-Group Matrix Functions** (`simdgroup_matrix`, load/store/`multiply`) — the
  tensor-core-style WMMA matmul primitives. The reference to write if GEMM ever moves off
  MPS or a fused-attention kernel is attempted. Highest future value.
- **§6.16 Atomic Functions** (+ §6.16.1 memory order, §6.16.3 fences) — needed for any kernel
  that accumulates across threadgroups (e.g. a reduction writing partials, as in the
  spec's own reduce example).
- **§6.6 Math Functions** — the authoritative list of what MSL provides (and the home of the
  **no-`erf`** gotcha); worth a reference enumerating available vs missing math builtins.
- **§6.10.1 / §4.4.1** Threadgroup & SIMD-group **synchronization** (barriers, `mem_flags`,
  the SIMD-group model) — partially covered in storage-and-synchronization.md; promote to
  its own reference if barrier semantics get hairy.
- **§6.3 Common Functions** (`clamp`, `mix`, `saturate`, `sign`…) — low priority, mostly
  obvious, but cheap to add if a kernel leans on them.
- NOT worth mining for CT2: textures (§6.13), imageblocks (§6.14), graphics/fragment
  (§6.11), geometric (§6.9) — no render passes in this backend.
