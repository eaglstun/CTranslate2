---
title: "Memory footprint & residency on unified-memory Macs"
summary: >-
  Explains memory footprint and residency on unified-memory Macs through the
  MTLDevice query surface — recommendedMaxWorkingSetSize (the model-fits
  preflight number), currentAllocatedSize (live Metal footprint),
  MTLResource.allocatedSize vs MTLBuffer.length, hasUnifiedMemory, and
  maxBufferLength (256 MB floor) — and the setPurgeableState cache lever
  (nonVolatile/volatile/empty, where volatile resources do not count toward
  the limit but must be relocked and checked). Because unified memory makes
  GPU bytes into process RSS, it records two project measurements:
  int8-resident Qwen2.5-0.5B weights cut peak RSS from 2494 to 1453 MB (−42%),
  and a Whisper wired-memory balloon (9.07 to 2.06 GB) from autoreleased
  command buffers where heap RSS stayed flat. Notes the allocator does not
  cache (allocator.mm makes a fresh newBufferWithLength per request) and
  suggests preflight against recommendedMaxWorkingSetSize plus ru_maxrss as
  the measurement default.
semantic_id: "yxwx5ho_l13zwa1dSs1JNEMYNSNRyKqgClXjm5zVwkoKC_0W7a4hDo91_s1troc6DC8k-t7b25ZupQbbzvxvhQ"
---

Sources (Apple Developer Documentation, fetched via DocC JSON, 2026-06-11):

- <https://developer.apple.com/documentation/metal/mtldevice/recommendedmaxworkingsetsize> (+ `currentallocatedsize`, `hasunifiedmemory`, `maxbufferlength`)
- <https://developer.apple.com/documentation/metal/mtlresource/allocatedsize> (+ `setpurgeablestate(_:)`, `mtlpurgeablestate`)
- <https://developer.apple.com/documentation/metal/reducing-the-memory-footprint-of-metal-apps>

## The query surface

- **`MTLDevice.recommendedMaxWorkingSetSize`** (`UInt64`) — "an approximation of how
  much memory, in bytes, this GPU device can allocate without affecting its runtime
  performance. You can help the GPU maintain its performance by keeping the total memory
  footprint of its resources and heaps less than this threshold value." The
  model-fits-or-not preflight number.
- **`MTLDevice.currentAllocatedSize`** (`Int`) — "the total amount of memory, in bytes,
  the GPU device is using for **all** of its resources" — i.e. the live Metal footprint
  of the process, queryable at any point.
- **`MTLResource.allocatedSize`** (`Int`) — the resource's actual size in bytes (the
  physical allocation, vs `MTLBuffer.length` which is the logical size you requested).
- **`MTLDevice.hasUnifiedMemory`** (`Bool`) — "whether the GPU shares all of its memory
  with the CPU"; true = typically an integrated GPU (always true on Apple Silicon —
  the premise of this whole backend, see `storage-and-synchronization.md`).
- **`MTLDevice.maxBufferLength`** — largest single buffer (doc floor: 256 MB).

## Purgeable state — the cache lever

`setPurgeableState(_:)` (returns the **prior** state; `.keepCurrent` queries):

- `.nonVolatile` — data must not be discarded.
- `.volatile` — "can be discarded … the implementation can reclaim the underlying
  storage at any time without informing the app." The doc's stated purpose: "may enable
  an app to keep larger caches of idle memory … without the risk of preventing the
  allocation of more important memory." Apple's footprint article adds: volatile/empty
  resources **don't count toward the memory limit**.
- `.empty` — contents no longer needed.
- Rule: **lock (`.nonVolatile`) before accessing again** and check the previous state —
  if the OS purged it, the data is gone and must be regenerated.

## Unified memory means GPU bytes are process bytes

On Apple Silicon a shared-mode `MTLBuffer` is plain process-addressable memory, so model
weights held in `MTLBuffer`s appear in the process footprint (RSS / "Memory" in Activity
Monitor) like any malloc. Two project-measured consequences:

- **The int8 headline:** Qwen2.5-0.5B with int8-resident weights, **peak RSS 1453 MB vs
  2494 MB fp16 — −42%** (`METAL_BENCHMARKS.md` int8 e2e table; `METAL_BACKEND.md` M12
  milestone, measured 2026-06-11, M4 Max, 3 runs each, warm). Phase 1's per-call
  widening shim had **no** RSS win — residency only moved when the weights stayed int8
  on the GPU.
- **The Whisper caveat — heap RSS is not the whole story:** autoreleased command-buffer
  objects accumulate as **wired** memory that ballooned the process to a SIGKILL while
  _heap_ RSS stayed flat (`METAL_BACKEND.md` "Autorelease pool (load-bearing)";
  `WHISPER_METAL_BRINGUP.md`: peak RSS 9.07 GB climbing → 2.06 GB flat after the fix).
  When a Metal process grows, check wired/footprint, not just heap.

**Measurement recipe, honestly:** the docs record the −42% numbers but **not the
measurement tool** — it is unrecorded whether peak RSS came from `ps`, Activity Monitor,
`/usr/bin/time -l` (`maximum resident set size`), or `ru_maxrss`. The repo's one
precedent is `tools/tune_inter_intra.py`, which reads `os.wait4(pid, 0)[2].ru_maxrss` —
the same kernel counter `/usr/bin/time -l` reports, and the sane default for reproducing
the number. For _GPU-attributed_ footprint specifically, `device.currentAllocatedSize`
before/after model load is the in-process probe the backend doesn't use yet.

---

### Relevance to the CT2 Metal backend

- **The allocator does not cache** (`src/metal/allocator.mm`): `allocate()` creates a
  fresh `newBufferWithLength:` per request and `free()` releases it — unlike CT2's CUDA
  caching allocator. CLAUDE.md treats allocation churn as a bug, so if a buffer cache is
  ever added here, `setPurgeableState(.volatile)` on idle cached buffers is the
  documented way to keep the cache _and_ let the OS reclaim it under pressure — with the
  must-relock-and-check rule above, since a purged weight buffer would otherwise read as
  silent zeros.
- **Preflight:** model load (`src/models/model.cc` → Metal allocator) currently just
  throws on a failed allocation. Comparing the converted model's weight bytes against
  `recommendedMaxWorkingSetSize` (and `maxBufferLength` for any single tensor) before
  loading would turn an OOM-kill into a clean "model doesn't fit" error.
- **Footprint accounting:** `device.currentAllocatedSize` distinguishes "Metal buffers"
  from "everything else in RSS" — exactly the split the Whisper wired-memory hunt needed
  and did by hand. Cheap to log at model-load and per-N-steps when chasing a leak.
- The int8 work is the proof that residency, not speed, was the quantization win on this
  GPU (`int8-gemm-kernel-design.md`: ALU-bound kernels, −42% RSS) — footprint numbers
  are first-class results here; per the standing rule, new measurements go into
  `METAL_BENCHMARKS.md` with date/machine/run-count.
