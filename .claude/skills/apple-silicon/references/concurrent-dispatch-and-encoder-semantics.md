# Compute-encoder dispatch semantics — serial vs concurrent, and the backend's one-op encoders

Sources: https://developer.apple.com/documentation/metal/mtlcommandbuffer/makecomputecommandencoder(dispatchtype:),
https://developer.apple.com/documentation/metal/mtldispatchtype/serial,
https://developer.apple.com/documentation/metal/mtldispatchtype/concurrent,
https://developer.apple.com/documentation/metal/mtlcomputecommandencoder/memorybarrier(scope:),
https://developer.apple.com/documentation/metal/mtlcomputecommandencoder/memorybarrier(resources:),
https://developer.apple.com/documentation/metal/mtlbarrierscope,
https://developer.apple.com/documentation/metal/resource-synchronization
(fetched via DocC JSON, 2026-06-11). Repo facts read from source the same day.

## Serial is the default, and what it guarantees

From the _Resource synchronization_ article: "By default, an `MTLComputeCommandEncoder`
encodes a compute pass that runs its commands serially." `MTLDispatchType.serial` "sets a
command encoder to dispatch encoded commands serially during your pass" — dispatch N's
memory effects are visible to dispatch N+1 within the same encoder, with no barriers
needed from you. The plain `[commandBuffer computeCommandEncoder]` factory yields a
serial encoder.

## Concurrent dispatch — the opt-in and its obligations

Created via `makeComputeCommandEncoder(dispatchType: .concurrent)` or by setting
`dispatchType` on an `MTLComputePassDescriptor`. `.concurrent` lets the GPU overlap the
dispatches within the pass, and the contract flips: "If you encode multiple commands that
access a single resource, you're responsible for synchronizing the memory operations to
that resource." The in-pass tools (encoder methods, between dispatches):

- `memoryBarrier(scope:)` — orders writes→reads for whole resource _types_;
  `MTLBarrierScope` options are `.buffers`, `.textures`, `.renderTargets` (a compute
  backend uses `.buffers`).
- `memoryBarrier(resources:)` — same, for an explicit resource list (tighter than a
  whole-scope barrier).

Reads racing reads are fine ("multiple commands can load segments of a buffer at the same
time, even if those segments overlap"); any write racing anything needs a barrier.

## Between encoders: automatic hazard tracking

Also from the article: the framework automatically synchronizes access conflicts for
commands submitted to an `MTLCommandQueue`, **for resources that are
`hazardTrackingMode = .tracked` and directly bound to an encoder**. Device-created
resources default to tracked; heap suballocations default to untracked (the
`mtlheap.md` trap). So encoder boundaries on tracked buffers are ordering points Metal
inserts for you — within a concurrent pass, you insert them yourself.

## What the backend actually does (verified)

Every kernel host in `src/metal/primitives.mm` follows one pattern: per op →
`new_command_buffer()` → `[command_buffer computeCommandEncoder]` (**default = serial**,
~17 sites) → set PSO/buffers/bytes → exactly **one** dispatch → `endEncoding` →
`commit_command_buffer(...)`. One op = one command buffer = one encoder = one dispatch.
GEMMs don't even create a compute encoder — `MPSMatrixMultiplication encode...` manages
its own encoding inside `src/metal/gemm.mm`. Cross-op correctness therefore rides
entirely on automatic hazard tracking between command buffers on the single queue, with
zero explicit barriers/fences anywhere in `src/metal/` (consistent with
`mtlevent-and-mtlfence.md`).

Consequence: **`.concurrent` has nothing to act on today** — with one dispatch per
encoder, intra-pass concurrency is vacuous. The dispatch type only becomes a real lever
after a structural change: batching several ops into one encoder.

## The lever, honestly priced

A fused multi-op encoder (several elementwise kernels of a decode step in one
`.concurrent` pass, barriers only at true dependencies) would cut per-op command-buffer
overhead — but that is the same family of idea as command-buffer reuse, which
parity-passed and then **lost 23% on prefill by destroying CPU/GPU overlap**
(`dispatch-overlap-and-perf-model.md`, the graveyard). Per-op commit is what lets the GPU
run op N while the CPU encodes N+1; a long fused pass re-serializes that. Independent
same-stage ops that could genuinely run concurrently are also rare in an autoregressive
decode step (the big trio — q/k/v — are MPS GEMMs, which can't share a compute encoder
anyway). Measure first, against the encode-floor numbers in
`benchmarking-and-profiling.md`.

### Relevance to the CT2 Metal backend

- `src/metal/primitives.mm` — the one-op/one-serial-encoder pattern every host function
  follows; any fused-encoder experiment changes this file plus the
  `new_command_buffer()`/`commit_command_buffer()` lifecycle in `src/metal/device.mm`.
- `src/metal/gemm.mm` — MPS encodes internally; MPS ops can interleave with custom
  kernels only at command-buffer granularity, capping how much a fused pass can contain.
- The safety story to preserve: tracked Shared buffers + single queue + serial encoders
  = zero manual synchronization (`storage-and-synchronization.md`). Going `.concurrent`
  anywhere imports the manual-barrier obligation listed above — budget for it in review.
