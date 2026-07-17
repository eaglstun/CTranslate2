---
title: "MTLHeap — suballocating resources from one memory pool"
summary: >-
  MTLHeap, a memory pool for suballocating buffers/textures from one driver-
  level allocation, all sharing the heap's storage and cache mode. Covers
  MTLHeapDescriptor, the automatic/placement/sparse heap types, makeBuffer
  with and without caller-chosen offset, sizing via heapBufferSizeAndAlign and
  maxAvailableSize, and explicit memory reuse through makeAliasable (automatic
  heaps only, one-way, UB to touch afterward, requires MTLFence/MTLEvent
  ordering). Stresses that heaps are untracked by default (hazardTrackingMode
  .default equals untracked), unlike individual resources. For CT2 this is
  unused: MetalAllocator::allocate makes one Shared MTLBuffer per call with no
  pooling, and adopting a heap would force fence discipline across
  primitives.mm/gemm.mm; presented as an evaluated future option triggered
  only if profiling shows allocation churn in the decode loop's transient
  activations.
---

Sources: https://developer.apple.com/documentation/metal/mtlheap,
https://developer.apple.com/documentation/metal/mtlheapdescriptor,
https://developer.apple.com/documentation/metal/mtlheaptype,
https://developer.apple.com/documentation/metal/mtldevice/heapbuffersizeandalign(length:options:),
https://developer.apple.com/documentation/metal/mtlresource/makealiasable(),
https://developer.apple.com/documentation/metal/mtlpurgeablestate
(fetched via DocC JSON, 2026-06-11).

An `MTLHeap` is "a memory pool from which you can suballocate resources." One driver-level
allocation up front; buffers and textures are then carved out of it without further
system-memory allocations. All resources suballocated from the same heap **share the heap's
storage mode and CPU cache mode**.

## Creating a heap

Configure an `MTLHeapDescriptor` and pass it to `MTLDevice.makeHeap(descriptor:)`. A heap
inherits the descriptor at creation, so one descriptor can be reused for several heaps.
Descriptor properties: `size` (total bytes), `type`, `storageMode`, `cpuCacheMode`,
`hazardTrackingMode`, `resourceOptions` (the combined bitmask), `sparsePageSize`.

### Heap types (`MTLHeapType`)

| Type         | Meaning                                                           |
| ------------ | ----------------------------------------------------------------- |
| `.automatic` | The heap places new resource allocations itself.                  |
| `.placement` | The app controls placement: `makeBuffer(length:options:offset:)`. |
| `.sparse`    | Sparse texture tiles (render-side; not relevant here).            |

## Suballocating buffers

- `heap.makeBuffer(length:options:)` — heap picks the offset (automatic heaps).
- `heap.makeBuffer(length:options:offset:)` — caller picks the byte offset (placement heaps,
  i.e. you run your own allocator arithmetic).

The `options` must be compatible with the heap (same storage/cache mode).

## Sizing and alignment queries

- `MTLDevice.heapBufferSizeAndAlign(length:options:)` → `MTLSizeAndAlign`: the size and
  alignment a buffer **will actually occupy inside a heap**. Apple's stated use: "estimate
  an appropriate size for a new heap before you create it." Sum these (rounded up to each
  alignment), don't sum raw lengths.
- `heap.maxAvailableSize(alignment:)` — largest single resource currently allocatable.
- `heap.size` / `heap.usedSize` / `heap.currentAllocatedSize` — capacity vs. occupancy.

## Aliasing — explicit memory reuse

A suballocated resource is **non-aliased by default** (its memory is reserved). Calling
`resource.makeAliasable()` releases the backing range for reuse by _future_ heap
allocations; `isAliasable()` queries the flag.

Sharp edges (all from the `makeAliasable()` page):

- Only valid for resources on **automatic** heaps. On placement heaps you alias by simply
  allocating at overlapping offsets yourself.
- Aliasing is one-way: an aliased resource "can't be un-aliased or moved", and reading or
  writing through it afterward is **undefined behavior** / possible memory corruption.
- The documented pattern is stage-based ping-pong: allocate stage N's scratch, run the GPU
  stage, `makeAliasable()` when it completes, allocate stage N+1 from the reclaimed space —
  and "use an MTLEvent or MTLFence instance to protect access" so two aliases of the same
  bytes are never touched concurrently.

## Hazard tracking on heaps

A new `MTLHeapDescriptor`'s `hazardTrackingMode` is `.default`, **which for heaps is
equivalent to `.untracked`** — "heaps don't track resources by default" (stated on the
`useHeap(_:)` page). So suballocated resources get NO automatic barriers between passes
unless you explicitly create the heap (or individual resources) with `.tracked`; untracked
suballocations must be ordered manually with `MTLFence`/`MTLEvent`
(see `mtlevent-and-mtlfence.md`). Tracked heaps trade that bookkeeping back for runtime
overhead.

## Purgeability

`heap.setPurgeableState(_:)` (`MTLPurgeableState`: `.keepCurrent` to query, `.nonVolatile`,
`.volatile` — system may discard, `.empty`). Purgeability is set on the **heap**, not on
its suballocated resources — they "can only reflect the heap's purgeability state."

### Relevance to the CT2 Metal backend

- **Not used today.** The backend allocates one individual shared `MTLBuffer` per
  `allocate()` call — `MetalAllocator::allocate` in `src/metal/allocator.mm` calls
  `newBufferWithLength:options:` with `MTLResourceStorageModeShared` and tracks the
  `contents` pointer in an address-ordered side table (which already handles sub-views via
  range lookup, so a heap would slot under it cleanly). There is no pooling or caching at
  the Metal layer; churn is kept down by `StorageView`'s resize-smaller-doesn't-reallocate
  behavior, not by the allocator.
- **The trigger:** if profiling ever shows `newBufferWithLength:` allocation cost or
  fragmentation in the decode loop's transient activations, a heap is the API answer —
  one upfront allocation, cheap suballocation, and `makeAliasable()` maps naturally onto
  the decode loop's short-lived ping-pong scratch buffers.
- **The cost that must be priced in:** heaps are untracked by default, and the whole
  backend currently relies on automatic hazard tracking + single-queue FIFO order with
  zero fences (grep `MTLFence` in `src/metal/` — no hits). An untracked heap means adding
  fence discipline to every kernel in `src/metal/primitives.mm`/`gemm.mm`; a `.tracked`
  heap avoids that but gives back some of the win.
- This is an **evaluated future option, not a recommendation** — the backend's perf
  culture is measure-first (see `dispatch-overlap-and-perf-model.md`'s graveyard), and no
  measurement has yet shown allocator churn on the profile.
