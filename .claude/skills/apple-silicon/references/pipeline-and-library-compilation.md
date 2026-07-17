---
title: "Runtime MSL compilation & pipeline creation"
summary: >-
  Runtime MSL compilation and compute-pipeline creation:
  makeLibrary(source:options:) compiling an MSL source string (which can only
  import the Metal default library), MTLCompileOptions with mathMode
  (.fast/.relaxed default on Apple silicon/.safe) replacing fastMathEnabled,
  languageVersion and preprocessorMacros, and function constants
  ([[function_constant]] + MTLFunctionConstantValues) versus preprocessor
  macros for building pipeline variants with back-end dead-code elimination.
  It covers makeComputePipelineState cost (front-load it), threadExecutionWidt
  h/maxTotalThreadsPerThreadgroup/staticThreadgroupMemoryLength, the ~493 ms
  first-MPS-GEMM warmup, and why precompiled .metallib was measured dead
  (macOS caches compiled shaders by source hash, so runtime compile is ~0.5 ms
  after the first ~123 ms cold run). CT2 compiles via device.mm
  ensure_library() with default options and caches pipelines in a name-keyed
  map; function constants are the clean lever for int8 GEMM tile-size
  variants.
---

Sources (Apple Developer Documentation, fetched via DocC JSON, 2026-06-11):

- <https://developer.apple.com/documentation/metal/mtldevice/makelibrary(source:options:)> (+ async `completionHandler:` variant)
- <https://developer.apple.com/documentation/metal/mtlcompileoptions> (+ `mathmode`, `mtlmathmode` cases, `languageversion`, `preprocessormacros`)
- <https://developer.apple.com/documentation/metal/mtllibrary/makefunction(name:)> and `makefunction(name:constantvalues:)`, `mtlfunctionconstantvalues`
- <https://developer.apple.com/documentation/metal/mtlcomputepipelinedescriptor> (+ `mtlcomputepipelinestate` and its properties)
- <https://developer.apple.com/documentation/metal/using-function-specialization-to-build-pipeline-variants>

## Source → library

`makeLibrary(source:options:)` (ObjC `newLibraryWithSource:options:error:`) compiles an
MSL source **string** synchronously and throws/returns-nil with an `NSError` on failure.
An async variant takes a completion handler. Constraint from the doc: "because there's no
search path to find other functions, the source may only import the Metal default
library" — runtime-compiled source can't `#include` your own headers.

**`MTLCompileOptions`** (all optional):

- **`mathMode: MTLMathMode`** (macOS 15+; replaces the old `fastMathEnabled` bool):
  - `.fast` — "aggressive, potentially lossy assumptions about floating-point math";
    doc: default for **Intel and AMD** devices.
  - `.relaxed` — same aggressive/lossy latitude "while honoring Inf/NaN"; doc: default
    for **Apple silicon** devices.
  - `.safe` — disables unsafe FP optimizations; no result-affecting transformations.
  - Setting `fastMathEnabled = true` maps to `mathMode = .fast` (+ fast libraries);
    `false` maps to `.safe`. The _numeric consequences_ for CT2 kernels (contracted FMA,
    `sqrt` as `x*rsqrt(x)`, folded `isnan`) are owned by
    `math-functions-and-numeric-parity.md` — this file is just the knob.
- **`languageVersion: MTLLanguageVersion`** — "by default, Metal uses the most recent
  language version."
- **`preprocessorMacros: [String : NSObject]?`** — values must be `NSString`/`NSNumber`;
  default `nil`. The runtime equivalent of `-D` flags.

## Function constants vs preprocessor macros

`makeFunction(name:)` returns a proxy for one `kernel` function (`nil` if absent) — not
executable code. If the function uses `[[function_constant(index)]]` constants, the
unspecialized proxy **cannot build a pipeline** — it's only good for querying
`functionConstants`. To specialize: fill an `MTLFunctionConstantValues`
(`setConstantValue(ptr, type:, index:)`; values matched by index then name, extras
silently ignored) and call `makeFunction(name:constantValues:)` (throws on invalid
values).

What specialization buys (per Apple's function-specialization sample): runtime `if` on a
uniform flag costs registers/occupancy; `#if` preprocessor variants force a full
source recompile per variant; **function constants let the front-end compile once and
the back-end dead-code-eliminate per variant at pipeline-creation time** — N cheap
specializations of one compiled library instead of N source compiles or one branchy
kernel.

## Pipeline state

`makeComputePipelineState(function:)` (throws) finishes compiling **for this specific
GPU**. Doc guidance: "you typically make pipeline states at a noncritical time … graphics
drivers may need time to evaluate and build each pipeline state" — i.e. the lazy-compile
cost is real and Apple tells you to front-load it. The descriptor form
(`makeComputePipelineState(descriptor:options:reflection:)` with
`MTLComputePipelineDescriptor`) adds `maxTotalThreadsPerThreadgroup` as an input hint:
0 (default) = let Metal calculate; if you also use the MSL
`[[max_total_threads_per_threadgroup]]` attribute the two values must match or you can
get a runtime error; Metal may silently lower the limit, costing performance.

`MTLComputePipelineState` read-only properties: `threadExecutionWidth` (SIMD width —
size threadgroups in multiples), `maxTotalThreadsPerThreadgroup` (fixed per PSO, varies
_between_ PSOs with register/memory pressure), `staticThreadgroupMemoryLength`
(statically allocated `threadgroup` bytes — a register-pressure/occupancy tell).

**First-dispatch cost in this project:** the first MPS GEMM pays a one-time **~493 ms**
pipeline warmup, then is fast (see `references/mps-matrix-multiplication.md`; the
benchmark harness warms explicitly — `tests/metal_test.cc` `time_ms()` comments "warmup
(MPS pipeline compilation, allocator priming)").

## Precompiled `.metallib` — measured-dead, don't re-dig

Offline `.metallib` / binary archives exist as the official way to skip runtime
compilation. **This backend measured it dead (2026-06-09, per project log):** the
header comment in `src/metal/kernels/kernels_msl.h` records the receipt —
`newLibraryWithSource` costs **~123 ms only on a truly cold system shader cache** (clean
install / driver update); macOS caches compiled shaders **by source hash at the system
level**, so every subsequent process pays **~0.5 ms**. Precompiling would save ~123 ms
once per machine in exchange for `xcrun metal` build plumbing plus the NSBundle
path-resolution problem (CT2 ships a bare `.dylib`, not a framework bundle). Bad trade;
the inline source stays. (Note: `METAL_BENCHMARKS.md` line ~124 still lists `.metallib`
as "Unmeasured" — the kernels_msl.h header is the newer, measured word.)

---

### Relevance to the CT2 Metal backend

- **How kernels are compiled today:** `src/metal/device.mm` `ensure_library()` —
  `newLibraryWithSource:` over the raw string from
  `get_kernels_source()` (`src/metal/kernels/kernels_msl.h`), with a **default-init
  `MTLCompileOptions`** — so on Apple silicon the library compiles under the default
  (fast/relaxed) math mode; that is why kernel numerics ride on the spellings documented
  in `math-functions-and-numeric-parity.md`. Compilation is lazy (first `get_pipeline()`
  call), so MPS-only paths work even if a kernel fails to compile.
- **Why pipelines are cached in a map:** `MetalContext::pipeline()` (`device.mm`) keeps
  `std::unordered_map<std::string, id<MTLComputePipelineState>>` keyed by kernel name,
  built once via `newFunctionWithName:` + `newComputePipelineStateWithFunction:error:` —
  exactly the "expensive; create once, reuse" contract above. The simple
  function-form (no descriptor) is used; no `maxTotalThreadsPerThreadgroup` hints.
- **Function constants are the clean int8 tile-variant lever:** the int8 GEMM tile sizes
  are baked into the MSL source (`CT2_GEMM_S8_BM/BN` in `kernels_msl.h`) and must match
  host constants `kGemmS8TileM/N` in `src/metal/primitives.mm`. If tile-size variants
  are ever needed (e.g. a skinny-k or small-n tile), `[[function_constant]]` +
  `makeFunction(name:constantValues:)` generates per-shape pipelines from one source
  with dead-code elimination — extend the pipeline-cache key with the constant values.
