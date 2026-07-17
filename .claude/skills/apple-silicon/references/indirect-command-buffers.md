---
title: "Indirect command buffers (ICBs) for compute — and the honest decode-loop verdict"
summary: >-
  Indirect command buffers (MTLIndirectCommandBuffer) hold reusable compute
  commands encoded once on CPU or GPU and replayed via
  executeCommandsInBuffer, configured through
  MTLIndirectCommandBufferDescriptor (concurrentDispatch types,
  inheritPipelineState/Buffers, maxKernelBufferBindCount). The doc documents
  the MTLIndirectComputeCommand surface and its constraints — no setBytes
  (scalars must live in buffers), no textures, barriers only where you
  setBarrier — plus the simpler dispatchThreadgroups(indirectBuffer:) for
  dynamic grids. Its central argument is a CT2 verdict: ICBs are the tempting
  answer to the per-op encode floor but likely lose like command-buffer reuse
  did (−6% bs8 decode, −23% prefill from killed CPU/GPU overlap), shapes
  change every decode step, and MPS GEMMs can't participate; the honest first
  step is indirect dispatch plus op fusion, not ICBs.
semantic_id: "yh0xxglvqm3Uk219xukpnYMgrXvTy6uZDpdiEpT35taqOXkG56ctXw08esXv7w47pAcgsu3x5Y5-9W__TrQrpA"
---

Sources (Apple DocC JSON, fetched 2026-06-11):
<https://developer.apple.com/documentation/metal/mtlindirectcommandbuffer>,
<https://developer.apple.com/documentation/metal/mtlindirectcommandbufferdescriptor>,
<https://developer.apple.com/documentation/metal/mtlindirectcomputecommand>,
<https://developer.apple.com/documentation/metal/mtlindirectcommandtype>,
<https://developer.apple.com/documentation/metal/mtlcomputecommandencoder/executecommandsinbuffer(_:range:)>,
<https://developer.apple.com/documentation/metal/mtlcomputecommandencoder/dispatchthreadgroups(indirectbuffer:indirectbufferoffset:threadsperthreadgroup:)>.
CT2 perf framing: `dispatch-overlap-and-perf-model.md` (the encode floor + the
command-buffer-reuse graveyard). NOT used by the backend today — this file exists so the
next person evaluates the idea with eyes open.

## What an ICB is [Apple doc]

`MTLIndirectCommandBuffer` (macOS 10.14+; compute commands macOS 11+): "a command buffer
containing reusable commands, encoded either on the CPU or GPU." You "encode commands
once and reuse them," from multiple CPU threads or from a GPU kernel. It is an
`MTLResource` (allocated like a buffer, has `gpuResourceID`), created by
`MTLDevice.makeIndirectCommandBuffer(descriptor:maxCommandCount:options:)`, with `size`
commands, `indirectComputeCommandAt(_:)` accessors, and `reset(_:)` to clear a range.

## Configuring: MTLIndirectCommandBufferDescriptor [Apple doc]

- `commandTypes` — for compute: `.concurrentDispatch` (grid aligned to threadgroup
  boundaries) and `.concurrentDispatchThreads` (arbitrarily sized grid).
- `inheritPipelineState` / `inheritBuffers` — whether commands take the PSO / buffer
  bindings from the parent encoder at execute time instead of storing their own.
- `maxKernelBufferBindCount` — max buffers settable per compute command (the per-command
  binding budget you must declare up front).

## Per-command surface: MTLIndirectComputeCommand [Apple doc]

Reset a command before re-encoding it. The documented compute surface is exactly:
`setComputePipelineState(_:)`, `setKernelBuffer(_:offset:at:)`,
`setThreadgroupMemoryLength(_:index:)`, `setImageblockWidth(_:height:)`,
`setStageInRegion(_:)`, `setBarrier()`/`clearBarrier()` (order against prior commands in
the ICB), then `concurrentDispatchThreadgroups(_:threadsPerThreadgroup:)` or
`concurrentDispatchThreads(...)`.

**What is NOT in that surface** (the constraints, read off the documented API): no
`setBytes` — every scalar argument must live in a real `MTLBuffer`; no texture/sampler
setters on compute commands; commands within one ICB serialize only where you `setBarrier()`
(the command types are _concurrent_ dispatches). The same surface exists GPU-side in MSL
(`command_buffer` argument type) for GPU-driven encoding. [unverified beyond the topic
lists: residency rules for buffers referenced inside an ICB — the argument-buffer
`useResource` discipline (`argument-buffers.md`) is the pattern to check before building.]

## Executing [Apple doc]

On a normal `MTLComputeCommandEncoder`:
`executeCommandsInBuffer(_ icb, range:)` (macOS 11+) — "encodes an instruction to run
commands from an indirect buffer"; a GPU-driven-range variant
`executeCommands(in:indirectBuffer:offset:)` also exists. So the replay itself still
costs one encoder call per step — what you save is re-encoding N dispatches.

## The simpler primitive first: indirect dispatch [Apple doc]

`dispatchThreadgroups(indirectBuffer:indirectBufferOffset:threadsPerThreadgroup:)`
(macOS 10.11+ — much older than ICBs): grid dimensions come from a `MTLDispatchThreadgroupsIndirectArguments`
struct in a buffer; "the GPU fetches parameters from the indirect buffer just before the
thread grid starts," letting a kernel size the next dispatch "based on GPU feedback,
without latency from data transfer between the CPU and the GPU." This handles _dynamic
shapes_ (a grid that depends on the growing KV length) without any ICB.

## The CT2 framing: tempting, probably a loss — measure first

The per-op encode floor is real (~0.03–0.04 ms/op, `dispatch-overlap-and-perf-model.md`),
and ICBs are the API-level answer to "encode the decode step once, replay per token."
Before building it, weigh **[measured here / project history]**:

1. **The shape changes every step.** KV-length-dependent ops (attention GEMMs, softmax,
   the cache concat) get new grids and new operand sizes each token; activation buffers
   come from a caching allocator whose addresses can change. A replayable ICB needs every
   varying quantity indirected through buffers (indirect dispatch for grids, no
   `setBytes` so scalars move to buffers) — a real redesign, not a swap.
2. **The graveyard says batching loses.** Command-buffer reuse — one commit per decode
   step — passed parity and measured **−6% bs8 decode, −23% bs8 prefill** because it
   destroyed CPU/GPU overlap; per-op commit keeps the GPU running op N while the CPU
   encodes N+1. An ICB replayed as one block per step has the same one-big-batch
   character; the saved encode time must beat the lost overlap, which is exactly the
   trade that already lost once. Read the coffin in `dispatch-overlap-and-perf-model.md`
   before re-digging.
3. **MPS doesn't participate.** The GEMMs are MPS-encoded (`mps-matrix-multiplication.md`);
   only the custom-kernel ops (norms, rotary, elementwise, int8 GEMM/GEMV) could enter an
   ICB, so the replayable fraction of a decode step is partial anyway.

Expectation, stated plainly: **likely loses like buffer-reuse did.** The plausible first
step is not ICBs but `dispatchThreadgroups(indirectBuffer:)` for KV-length-dependent
kernels, paired with op _fusion_ (the one decode lever that measured a win) — fewer,
bigger ops attack the floor without giving up overlap. If anyone does try ICBs: bench
exactly as `benchmarking-and-profiling.md` prescribes, on Qwen2.5 decode AND prefill,
and keep the revert path warm.

### Relevance to the CT2 Metal backend

- Nothing in `src/metal/` uses ICBs or indirect dispatch today; all dispatches are
  CPU-encoded per op in `src/metal/primitives.mm` and committed per op via
  `src/metal/device.mm` (`g_last_committed`).
- Scalars are currently passed by `setBytes` (the op-graduation playbook's "host scalars
  by value") — the no-`setBytes` ICB constraint would force those into buffers; count
  that cost in any proposal.
- The ops with KV-length-dependent grids each step are the attention path's
  (`ct2-internals` skill, `attention-and-kv-cache.md`); those are where indirect dispatch
  would plug in.
