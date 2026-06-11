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

### Metal API surface

- **[references/mtlbuffer-api.md](references/mtlbuffer-api.md)**
  — The MTLBuffer/MTLDevice allocation lookup card: the three `makeBuffer` variants
  (length = zero-filled — the only one the allocator uses; bytes = copy; bytesNoCopy =
  page-aligned single-VM-region wrap), `.contents` nil-only-for-private,
  `setPurgeableState`, and `label` as the free debugging win the backend doesn't use.
  _Read when touching `allocator.mm` or wondering what alignment Metal guarantees (none —
  the int8 GEMV checks offsets explicitly)._

- **[references/resource-storage-modes-and-options.md](references/resource-storage-modes-and-options.md)**
  — The `MTLResourceOptions` bitmask lookup card: storage modes (+ the Managed one-liner),
  `defaultCache` vs `writeCombined` (write-combined = bug here, CPU-ref ops READ buffers),
  tracked vs untracked hazard modes, and the three `makeBuffer` variants incl. `bytesNoCopy`
  (page-aligned, whole pages — NOT used; weights go allocate-then-memcpy).
  _Read when creating a buffer with anything but the one combination `allocator.mm` uses._

- **[references/mtlheap.md](references/mtlheap.md)**
  — `MTLHeap`/`MTLHeapDescriptor`: one pool, suballocated buffers (automatic vs placement),
  sizing via `heapBufferSizeAndAlign`/`maxAvailableSize`, `makeAliasable()` ping-pong reuse,
  and the trap: heaps are **untracked by default** → fence discipline. NOT used today — the
  backend allocates individual Shared buffers (`allocator.mm`); this is the evaluated answer
  if allocator churn ever shows on a profile. _Read before reaching for a buffer pool._

- **[references/mtlevent-and-mtlfence.md](references/mtlevent-and-mtlfence.md)**
  — The three explicit sync primitives: `MTLFence` (between passes in a queue, untracked
  resources), `MTLEvent` (cross-command-buffer GPU↔GPU, monotonic signal/wait values),
  `MTLSharedEvent` (CPU↔GPU: `signaledValue`, listeners). The backend's single-queue +
  tracked-buffers model needs NONE of them; concrete triggers: a second queue, an untracked
  heap, or finer-than-flush() CPU waits. _Read before adding a queue or untracked resource._

- **[references/mtlgpufamily-and-feature-availability.md](references/mtlgpufamily-and-feature-availability.md)**
  — `MTLGPUFamily` + `supportsFamily(_:)`: M1=apple7, M2=apple8, M3/M4=apple9 (from the
  DocC case abstracts), `metal3`/`metal4` umbrellas, `maxBufferLength`. The backend checks
  NOTHING at init (`device.mm` assumes the M4 Max); lists the asserts a hardening pass
  would add. _Read before using a per-family feature or porting off the dev box._

- **[references/argument-buffers.md](references/argument-buffers.md)**
  — Argument buffers: resource handles in a buffer (tier1 needs `MTLArgumentEncoder`; tier2
  = write `gpuAddress` directly, the Metal 3 bindless path) + the `useResource`/`useHeap`
  residency rule. The backend binds ≤6 buffers per dispatch — no problem to solve; trigger is
  a decode-dispatch-batching redesign (see dispatch-overlap-and-perf-model.md's encode floor).
  _Read only if batching tiny decode ops via ICBs ever gets attempted._

- **[references/pipeline-and-library-compilation.md](references/pipeline-and-library-compilation.md)**
  — Runtime MSL → library → pipeline: `newLibraryWithSource` (no include path),
  `MTLCompileOptions.mathMode` (fast/relaxed/safe — **relaxed is the Apple-silicon
  default**, numerics in math-functions ref), function constants as cheap
  dead-code-eliminated variants vs `#if`, PSO properties, the ~493 ms first-MPS-GEMM
  warmup, and the measured-dead `.metallib` receipt. _Read before touching `device.mm`
  compilation or proposing precompiled shaders._

- **[references/gpu-counters-and-timestamps.md](references/gpu-counters-and-timestamps.md)**
  — GPU-side timing in three tiers: `gpuStartTime`/`gpuEndTime` (free whole-buffer timing,
  valid only after completion — the zero-effort upgrade to the CPU-side `time_ms()`
  harness), `sampleTimestamps()` GPU↔CPU clock correlation, and counter sample buffers
  (`MTLCommonCounterSet.timestamp`, `resolveCounterRange`) as the per-kernel scalpel.
  _Read before adding GPU-side measurement to the benchmark harness._

- **[references/gpu-capture-and-shader-validation.md](references/gpu-capture-and-shader-validation.md)**
  — The misplaced-pointer toolkit: programmatic `.gputrace` capture (`MTL_CAPTURE_ENABLED=1`
  - `MTLCaptureManager`), Shader Validation (`MTL_SHADER_VALIDATION=1` — OOB
    device/threadgroup detection, perf cost) and `MTL_DEBUG_LAYER`, with a recipe for
    `ctranslate2_test`. Honest note: it would NOT have caught the Gemma2 tanh-NaN —
    validation is for memory bugs, tripwires for numeric ones. _Read when a kernel
    scribbles, hangs, or reads garbage._

- **[references/memory-footprint-and-residency.md](references/memory-footprint-and-residency.md)**
  — Measuring/bounding GPU memory on unified memory: `recommendedMaxWorkingSetSize` (the
  model-fits preflight), `currentAllocatedSize`, `allocatedSize` vs `length`, purgeable
  state as the cache lever. Carries the int8 headline (Qwen RSS 1453 vs 2494 MB, −42%)
  and the Whisper wired-vs-heap caveat. _Read when chasing footprint or before claiming a
  memory win._

### MPS beyond GEMM

- **[references/mps-matrix-vector-multiplication.md](references/mps-matrix-vector-multiplication.md)**
  — `MPSMatrixVectorMultiplication`: `y = α·op(A)·x + β·y`, init/encode split (cacheable
  like `cached_gemm`), `MPSVector`/`MPSVectorDescriptor` incl. strided batches, and the
  dtype truth: no integer GEMV documented anywhere — the **untried** MPS-native option for
  fp16 decode m=1 GEMMs (which today ride the matrix kernel). _Read before A/B-ing the
  decode GEMV path._

- **[references/mps-softmax-and-topk.md](references/mps-softmax-and-topk.md)**
  — `MPSMatrixSoftMax`/`LogSoftMax` (row-wise, fp32/fp16 only, NO masking — why CT2's
  lengths-masked custom kernel won) and `MPSMatrixFindTopK`: **k ≤ 16 or UB**, UInt32
  index matrix, batching. The GPU-sampling candidate — logits are already GPU-resident
  post-lm*head while TopK runs CPU-side. \_Read when graduating sampling ops.*

- **[references/mpsndarray.md](references/mpsndarray.md)**
  — `MPSNDArray`/`MPSNDArrayMatrixMultiplication` (modern n-D API, native batch broadcast)
  and THE load-bearing find: **macOS 15+ ships `MPSNDArrayQuantizedMatrixMultiplication`**
  — affine/LUT quantization descriptors, int8/int4 dtypes, `initWithBuffer:` zero-copy —
  so "MPS is float-only" is now MPSMatrix-family-only. Benchmark candidate vs
  `ct2_gemm_s8`. _Read before any int8-GEMM rework._

- **[references/mpsgraph-for-inference.md](references/mpsgraph-for-inference.md)**
  — MPSGraph in one card: placeholders→ops→`MPSGraphTensorData(MTLBuffer:…)`, compile to a
  cached `MPSGraphExecutable` that can `encode(to:)` an existing command buffer.
  Quantization surface verified: `quantize`/`dequantize` (macOS 13.1+, i8/u8, per-axis
  scaleTensor, LUT) but **no quantized matmul op** — dequant→float-matmul is the Phase-1
  shim by another name. _Read before wrapping any op in MPSGraph._

- **[references/mps-convolution-options.md](references/mps-convolution-options.md)**
  — Graduating `Conv1D` off the CPU (today: CPU-ref + fp16→fp32 upcast; conv weights stay
  float on Metal): MPSCNNConvolution (MPSImage repacking — poor fit) vs MPSGraph
  `convolution2D` (buffer-native, H=1 trick) vs a custom MSL kernel (recommended first
  prototype). Whisper's 2-conv encode stem is the consumer. _Read when scoping the conv
  graduation._

### Objective-C++ runtime, debugging & profiling

- **[references/autoreleasepool-in-long-loops.md](references/autoreleasepool-in-long-loops.md)**
  — THE memory lesson: autoreleased Metal/MPS temporaries (+0 command buffers, encoders,
  MPS descriptors) never drain on C++ worker threads with no run loop — the Whisper fp16
  730s run climbed to 9.07 GB wired and SIGKILLed; fix = the per-op thread-local pool in
  `new_command_buffer()`/`commit_command_buffer()` (commit `868d12d3`, RSS → 2.06 GB flat).
  Includes the diagnostic signature (RSS climbs, heap flat). _Read before adding any Metal
  code path that bypasses the new→commit pair._

- **[references/objcpp-interop-for-mm-files.md](references/objcpp-interop-for-mm-files.md)**
  — Objective-C++ survival card: this backend is **MRC, not ARC** (no `-fobjc-arc` in
  CMake), the split-header discipline (`utils.h`/`primitives.h` C++-safe vs `.mm`-only
  `device.h` — no `#ifdef __OBJC__` anywhere), the three ownership patterns the repo uses
  to hold ObjC objects from C++, bridge-cast semantics, nil-messaging-returns-zero, and
  the NSError\*\* convention. _Read before editing any `.mm` file._

- **[references/occupancy-and-threadgroup-memory.md](references/occupancy-and-threadgroup-memory.md)**
  — The occupancy levers: PSO `maxTotalThreadsPerThreadgroup` (register-pressure-dependent
  — the first check when a 256-thread kernel underperforms), `threadExecutionWidth`, static
  vs encoder-set threadgroup memory, and the 32 KB budget (measured on the M4 Max via
  `maxThreadgroupMemoryLength`). With the arithmetic for the real kernels: `ct2_gemm_s8`
  uses 4096 B, the 256-thread reductions 1–2 KB, the GEMV zero. _Read when sizing a tile
  or chasing low occupancy._

- **[references/instruments-gpu-profiling.md](references/instruments-gpu-profiling.md)**
  — The GPU-side complement to `benchmarking-and-profiling.md`: Metal System Trace
  (CPU-encode vs GPU-execute lanes — would _show_ the overlap the perf model rides on),
  the GPU Counters instrument (the limiter view that would settle "is `ct2_gemm_s8`
  ALU-bound?" by measurement), os*signpost interval labeling from C++, and headless
  `xctrace record` recipes. \_A recipe card — read before the first profiling session.*

- **[references/command-buffer-errors-and-hangs.md](references/command-buffer-errors-and-hangs.md)**
  — Failure diagnosis: the status lifecycle (notEnqueued→…→completed|error), the real
  `MTLCommandBufferError` codes (timeout, pageFault, …), `errorOptions =
.encoderExecutionStatus` for per-encoder blame — and the repo punchline: `flush()`
  checks NOTHING today, so a GPU fault reads back as silent garbage; carries the 5-line
  status check and the 4-step garbage-output triage order. _Read when output is wrong and
  you don't yet know which kind of wrong._

### Hardware & future surfaces

- **[references/apple-gpu-architecture-for-compute.md](references/apple-gpu-architecture-for-compute.md)**
  — The calibrate-your-mental-model card, every claim provenance-labeled: TBDR is
  render-lore (compute sees ALU + unified memory), 32-wide SIMD, M4 Max 546 GB/s marketing
  vs ~280 GB/s measured sustained, NO int8 matrix hardware (vs CUDA dp4a/tensor cores),
  and fp16≈1.55× fp32 GEMM measured (hardware ratio unpublished). _Read before reasoning
  about "what the GPU can do."_

- **[references/fp16-numerics-on-gpu.md](references/fp16-numerics-on-gpu.md)**
  — Half-precision survival card: 65504/6.1e-5 limits, THE Gemma2 tanh-overflow→NaN→`<pad>`
  case and its `ct2_tanh_safe` clamp, the store-half/compute-float rule (verified across
  every `_half` kernel), literal suffixes + the bfloat promotion asymmetry, and
  ties-to-even store rounding. _Read before writing any fp16 kernel._

- **[references/indirect-command-buffers.md](references/indirect-command-buffers.md)**
  — ICBs for compute: descriptor (`.concurrentDispatch`, inherit\*), the
  MTLIndirectComputeCommand surface (NO `setBytes` — scalars must move to buffers),
  `executeCommandsInBuffer`, plus the simpler `dispatchThreadgroups(indirectBuffer:)`
  primitive. Honest verdict: the API-level "encode once, replay per token" answer that
  likely loses like command-buffer reuse did (overlap destruction) — measure first.
  _Read before proposing decode-loop dispatch batching._

- **[references/metal4-tensors-and-mpp.md](references/metal4-tensors-and-mpp.md)**
  — The Metal 4 (macOS 26) ML surface and THE find: MSL §7 MPP `matmul2d` supports
  **char×char→int at base Metal 4** — a documented int8 matmul path that could challenge
  the hand-tiled `ct2_gemm_s8`'s ALU-bound large-m regime. MTLTensor (int8 dtype,
  buffer-wrapping), shader-allocated `tensor_inline` views over raw pointers, the MTL4 ML
  encoder, and the flags: performance, accumulator exactness, GPU-family floor all
  unstated. _Read alongside mpsndarray.md before any int8-GEMM rework._

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
