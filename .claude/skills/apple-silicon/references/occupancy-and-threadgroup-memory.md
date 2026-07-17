---
title: "Compute occupancy & threadgroup memory (the API-side levers)"
summary: >-
  Covers the API-side levers bounding concurrent occupancy on Apple GPUs: per-
  PSO properties maxTotalThreadsPerThreadgroup (kernel-dependent, first thing
  to check on low occupancy or dispatch asserts), threadExecutionWidth (32),
  and staticThreadgroupMemoryLength, plus the descriptor hint
  maxTotalThreadsPerThreadgroup / max_total_threads_per_threadgroup attribute
  that the backend does not use. It gives the threadgroup-memory budget (32 KB
  measured on M4 Max via maxThreadgroupMemoryLength) and the occupancy
  arithmetic. A table tallies per-group threadgroup bytes for the real
  kernels: ct2_gemm_s8 uses 4096 B (12.5% of budget, register-limited), the
  softmax/norm/quantize reductions use ~1-2 KB with 256-thread groups, and
  ct2_gemv_s8 uses zero threadgroup memory by design. Ends with a low-
  occupancy triage checklist.
---

What bounds how many threads/threadgroups actually run concurrently, and the properties
to check when a kernel "mysteriously" underperforms. Companion to
`compute-kernels-and-dispatch.md` (which covers grid/threadgroup _sizing_ mechanics);
this file is about the _limits_.

Sources (Apple DocC JSON, fetched 2026-06-11):
`MTLComputePipelineState.maxTotalThreadsPerThreadgroup`, `.threadExecutionWidth`,
`.staticThreadgroupMemoryLength`,
`MTLComputeCommandEncoder.setThreadgroupMemoryLength(_:index:)`,
`MTLDevice.maxThreadgroupMemoryLength`,
`MTLComputePipelineDescriptor.maxTotalThreadsPerThreadgroup` — all under
<https://developer.apple.com/tutorials/data/documentation/metal/...json>. Device limit
measured locally on the dev box (see below).

## The pipeline properties (per-PSO, computed at pipeline creation)

- **`maxTotalThreadsPerThreadgroup`** — "the maximum number of threads in a threadgroup
  that you can dispatch to the pipeline." Computed when the PSO is created; "this value
  never changes, but may be different for different pipeline objects" — i.e. it is a
  function of the _kernel_ (register pressure, threadgroup memory), not just the device.
  A heavy kernel can come back with, say, 512 instead of 1024. **This is the first thing
  to check when a kernel runs at low occupancy or asserts on dispatch:** if
  `pso.maxTotalThreadsPerThreadgroup` < the threadgroup size the host hard-codes (256
  for every reduction kernel here), the dispatch is invalid.
- **`threadExecutionWidth`** — "the number of threads that the GPU executes
  simultaneously" (the SIMD-group width; 32 on Apple GPUs in practice — DocC does not
  state the number, the kernels read it at runtime). Apple: make the threadgroup size a
  multiple of it.
- **`staticThreadgroupMemoryLength`** — "the length, in bytes, of statically allocated
  threadgroup memory": the sum of the `threadgroup` arrays declared _inside_ the kernel.

## Raising the ceiling: the descriptor hint

`MTLComputePipelineDescriptor.maxTotalThreadsPerThreadgroup` lets the _host_ declare the
intended upper bound before compilation (default: Metal computes it from "the device's
capabilities and the compute shader's memory usage"). DocC warns: Metal may return an
error if the value exceeds available resources, **or may lower the thread limit when
creating the PSO, "which can reduce runtime performance"** — the compiler trades
registers for occupancy to honor it. The MSL-side equivalent is the
`[[max_total_threads_per_threadgroup(N)]]` kernel attribute. This backend uses neither:
pipelines come from `newComputePipelineStateWithFunction:` with default options
(`MetalContext::pipeline()`, `src/metal/device.mm`), so the limits are whatever the
compiler derives.

## Threadgroup memory: two ways to get it, one budget

1. **Declared in-kernel** (`threadgroup float scratch[256];` in the kernel body, or a
   `[[threadgroup(n)]]` argument) — static; shows up in
   `staticThreadgroupMemoryLength`. **Every kernel in this backend uses the in-body
   form** — no `[[threadgroup(n)]]` arguments exist in `kernels_msl.h`.
2. **Sized at encode time** — `setThreadgroupMemoryLength(_:index:)` on the encoder
   configures a block for a `[[threadgroup(n)]]` kernel argument ("before using any
   threadgroup memory, call this method to configure the threadgroup memory argument
   table"). Unused here (grep `setThreadgroupMemoryLength` in `src/metal/` — nothing),
   and only needed if a tile size ever becomes a runtime choice.

**The budget:** `MTLDevice.maxThreadgroupMemoryLength` — "the maximum threadgroup
memory available to a compute kernel, in bytes." DocC doesn't publish per-family
numbers; measured on this project's dev box (Apple M4 Max, via a one-line Swift query,
2026-06-11): **32768 bytes (32 KB)**. Treat it as "commonly 32 KB on Apple GPUs —
verify per device via `maxThreadgroupMemoryLength`".

## What actually bounds concurrent threadgroups per core

A GPU core runs multiple threadgroups concurrently only while _each_ fits its slice of
shared resources. Two divisible resources dominate (the occupancy arithmetic is standard
GPU reasoning; Apple does not publish per-core register-file/threadgroup-slot numbers —
unverifiable from DocC, use Instruments' occupancy view to observe it instead):

- **threadgroup memory**: concurrent groups ≤ `maxThreadgroupMemoryLength` / (per-group
  usage). At 32 KB total, a 4 KB kernel allows up to 8 resident groups; a 16 KB kernel
  allows 2.
- **registers**: high per-thread register use lowers `maxTotalThreadsPerThreadgroup`
  and the number of co-resident groups. Visible only via the PSO property and profiling.

## The real kernels, with the arithmetic (`src/metal/kernels/kernels_msl.h`)

| Kernel                             | threadgroup arrays                                           | bytes/group               | of 32 KB                                                                                             |
| ---------------------------------- | ------------------------------------------------------------ | ------------------------- | ---------------------------------------------------------------------------------------------------- |
| `ct2_gemm_s8`                      | `char As[32][64]` + `char Bs[32][64]` (BK=32, BM=BN=64)      | 2048+2048 = **4096 B**    | 12.5% — memory allows ~8 groups/core; registers (4×4 `int4 acc` micro-tile) are the likelier limiter |
| softmax (`CT2_SOFTMAX_TG`=256)     | `float scratch[256]`                                         | **1024 B**                | 3%                                                                                                   |
| rms/layer norm (`CT2_NORM_TG`=256) | `s_sum[256]` + `s_sq[256]` floats (layer_norm; rms uses one) | **2048 B** (1024 for rms) | 6%                                                                                                   |
| quantize (`CT2_QUANT_TG`=256)      | `float scratch[256]`                                         | **1024 B**                | 3%                                                                                                   |
| `ct2_gemv_s8`                      | none — `simd_sum` cross-lane, zero threadgroup memory        | **0 B**                   | the design point: nothing but registers limits residency                                             |

So threadgroup memory is nowhere near the budget in this backend; there is roughly 28 KB
of headroom in `ct2_gemm_s8` before the 32 KB wall — relevant if a bigger-BK or
double-buffered tile variant is ever attempted.

## Low-occupancy triage checklist

1. `pso.maxTotalThreadsPerThreadgroup` ≥ the host's hard-coded group size? (256 for the
   reductions/`ct2_gemm_s8`'s 16×16; `threadExecutionWidth * 8` = 256 for the GEMV.)
   The elementwise kernels in `primitives.mm` already dispatch with
   `pso.maxTotalThreadsPerThreadgroup` directly — they self-adapt.
2. `pso.staticThreadgroupMemoryLength` what you computed? (Catches an accidental tile
   constant bump.)
3. Group size a multiple of `threadExecutionWidth`? (256 = 8×32 ✓ everywhere here.)
4. Still slow → measure, don't reason: Instruments' GPU counters show the occupancy and
   limiter directly (`instruments-gpu-profiling.md`).

### Relevance to the CT2 Metal backend

- Host-side sizing lives in `src/metal/primitives.mm`: fixed 256-thread groups for the
  row-reduction kernels (`kSoftmaxThreadgroup`/`kNormThreadgroup`/`kQuantizeThreadgroup` — kept in sync with
  the `CT2_*_TG` constants by comment only, no compile-time check), `MTLSizeMake(16,16,1)` for `ct2_gemm_s8`,
  `threadExecutionWidth * 8` for `ct2_gemv_s8`, and
  `pso.maxTotalThreadsPerThreadgroup` for the elementwise `dispatchThreads` paths.
- Kernel-side declarations and tile constants: `src/metal/kernels/kernels_msl.h`
  (`CT2_GEMM_S8_BM/BN/BK`, the `As`/`Bs` tiles, the 256-float scratch arrays).
- Pipeline creation (where the descriptor hint would go if ever needed):
  `MetalContext::pipeline()` in `src/metal/device.mm`; see
  `pipeline-and-library-compilation.md`.
- Tiling design rationale for the int8 GEMM/GEMV: `int8-gemm-kernel-design.md` and
  `int8-gemv-simdgroup-decode.md`.
