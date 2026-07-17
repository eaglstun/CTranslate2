---
title: "GPU counters & timestamps (GPU-side timing)"
summary: >-
  Covers three tiers of GPU-side timing on Metal. Tier 1 is free whole-buffer
  timing via MTLCommandBuffer.gpuStartTime/gpuEndTime (readable only after
  completion, command-buffer granularity). Tier 2 is
  MTLDevice.sampleTimestamps() returning paired CPU/GPU timestamps to
  correlate the two clocks by linear interpolation. Tier 3 is counter sample
  buffers (MTLCounterSampleBufferDescriptor, MTLCommonCounterSet.timestamp,
  sampleCounters, resolveCounterRange) for per-kernel sampling at
  stage/dispatch boundaries with runtime support queries. For the CT2 Metal
  backend it argues gpuStartTime/gpuEndTime is the zero-effort upgrade over
  today's CPU-side time_ms() wall-clock method, directly separating encode
  from execute since the backend commits one buffer per op and already tracks
  g_last_committed.
---

Sources (Apple Developer Documentation, fetched via DocC JSON, 2026-06-11):

- <https://developer.apple.com/documentation/metal/mtlcommandbuffer/gpustarttime> (+ `gpuendtime`)
- <https://developer.apple.com/documentation/metal/mtlcountersamplebuffer> (+ descriptor, `resolvecounterrange(_:)`, `mtlcommoncounterset`, `mtlcounterset`, `mtldevice/countersets`, `supportscountersampling(_:)`, `mtlcountersamplingpoint`)
- <https://developer.apple.com/documentation/metal/mtlcomputecommandencoder/samplecounters(samplebuffer:sampleindex:barrier:)>
- <https://developer.apple.com/documentation/metal/converting-gpu-timestamps-into-cpu-time> (+ `mtldevice/sampletimestamps()`)

Three tiers, cheapest first.

## Tier 1: `gpuStartTime` / `gpuEndTime` — free whole-buffer timing

`MTLCommandBuffer.gpuStartTime` / `.gpuEndTime` (`CFTimeInterval`, seconds): "the host
time … when the GPU starts/finishes command buffer execution." Subtract for GPU
execution time of that buffer; both are "relative to system mach time."

- Both **remain 0.0 until the GPU finishes** the buffer — read only after
  `waitUntilCompleted` returns or inside an `addCompletedHandler:` block.
- No setup, no descriptors, no support query — it's on every command buffer.
- Granularity is the **command buffer**, not the kernel. In a backend that commits one
  buffer per op, that _is_ per-op GPU time.

## Tier 2: `sampleTimestamps()` — correlating GPU and CPU clocks

`MTLDevice.sampleTimestamps()` → `(cpu: MTLTimestamp, gpu: MTLTimestamp)` — "captures
and returns a CPU timestamp and a GPU timestamp from the same moment in time" (ObjC:
`sampleTimestamps:gpuTimestamp:`). GPU and CPU clocks differ, so converting raw GPU
counter timestamps to wall time requires a baseline: sample **twice** (e.g. at
command-buffer creation and in its completion handler), then linearly interpolate:
`(gpu_ts − gpu_start) / gpu_span × cpu_span + cpu_start`. CPU timestamps are in
nanoseconds. Doc warning: "call sampleTimestamps sparingly because doing so may trap to
the kernel … which can affect your app's runtime performance."

## Tier 3: counter sample buffers — the per-kernel scalpel

- **Discover support:** `MTLDevice.counterSets` (`[any MTLCounterSet]?`) lists supported
  sets; `MTLCommonCounterSet.timestamp` is the common name for the set containing the
  timestamp counter ("some GPUs may only support some of the counters within a set").
- **Where you may sample:** `device.supportsCounterSampling(_:)` per
  `MTLCounterSamplingPoint` case — `atStageBoundary`, `atDispatchBoundary`,
  `atDrawBoundary`, `atBlitBoundary`, `atTileDispatchBoundary`. **Honest availability
  note:** which boundaries a GPU supports varies; query each point before designing
  instrumentation (Apple GPUs and immediate-mode GPUs differ on stage- vs
  dispatch-boundary sampling — the fetched docs don't enumerate per-GPU support, so
  don't assume; check at runtime).
- **Create:** `MTLCounterSampleBufferDescriptor` → `device.makeCounterSampleBuffer
(descriptor:)` (throws; "may produce an error if the GPU driver has exhausted its
  underlying resources for counter sample buffers").
- **Sample:** `MTLComputeCommandEncoder.sampleCounters(sampleBuffer:sampleIndex:
barrier:)` encodes a sampling command. `barrier: true` ensures previously encoded
  commands complete before sampling (accurate, slower); `false` lets sampling run
  concurrently. DocC note: to use a sample buffer it must also be part of the
  `sampleBufferAttachments` on the compute pass descriptor.
- **Read back:** `MTLCounterSampleBuffer.resolveCounterRange(_:)` → `Data` of standard
  Metal structs (resolvable ranges require a shared-storage sample buffer); error slots
  are filled with `MTLCounterErrorValue`. Convert the resolved GPU timestamps to CPU
  time via the Tier-2 baseline.

---

### Relevance to the CT2 Metal backend

- **Today's methodology is CPU-side**: `benchmarking-and-profiling.md` — `time_ms()` in
  `tests/metal_test.cc` wraps host wall-clock around `synchronize_device()`, and the
  probe-isolation trick (commit many, flush once) _infers_ the encode-vs-execute split
  from two CPU numbers. It never observes GPU execution directly.
- **`gpuStartTime`/`gpuEndTime` is the zero-effort upgrade**: the backend already holds
  the last-committed buffer (`g_last_committed` in `src/metal/device.mm`) and
  `flush()` does `waitUntilCompleted` — reading `gpuEndTime − gpuStartTime` right after
  that wait gives true GPU execution time per committed op, directly separating encode
  (CPU) from execute (GPU) instead of inferring it. Also exposes scheduling gaps:
  CPU-wall minus GPU-execution minus encode = time the buffer sat queued — the per-op
  API floor made visible.
- **Counter sample buffers** matter only if one command buffer ever carries multiple
  dispatches (e.g. if command-buffer batching is revisited, or for the multi-kernel
  Dense epilogue chain): they time _inside_ a buffer, where Tier 1 can't see. For the
  current one-op-per-buffer design, Tier 1 already gives per-kernel resolution.
- Pair with the probe-isolation trick (`benchmarking-and-profiling.md`): the probe
  isolates the _encode floor_; `gpuStart/EndTime` isolates the _execution_; together
  they fully decompose an op's cost without guessing.
