---
title: "MTLResourceOptions — the buffer-creation bitmask, in detail"
summary: >-
  The MTLResourceOptions buffer-creation bitmask reference: one storage mode
  (Shared, Private, Memoryless textures-only, Managed legacy-x86) plus one CPU
  cache mode (defaultCache vs writeCombined) plus one hazard tracking mode
  (.default, tracked, untracked), and the three makeBuffer variants (zero-
  cleared, bytes-copy, and bytesNoCopy zero-copy with its page-aligned/whole-
  pages/single-VM-region/deallocator constraints). Documents that CT2 uses
  exactly one combination everywhere via MetalAllocator::allocate in
  src/metal/allocator.mm: StorageModeShared plus default cache plus default
  (tracked) hazard mode, which is what lets the whole backend run with zero
  fences. Explains why bytesNoCopy is unused, why writeCombined would be a bug
  (CPU-reference fallback ops read buffer contents), and why
  Private/Memoryless are unusable by design.
semantic_id: "yx0w1wjPq23TE_1Nw81pPIMgJWuTwqsjClfyGZStxt6ADXkH9e8tXk94Pg1Xqo8ajC8Emu37xY5-_a3XTv5-pQ"
---

Sources: https://developer.apple.com/documentation/metal/mtlresourceoptions,
https://developer.apple.com/documentation/metal/mtlstoragemode,
https://developer.apple.com/documentation/metal/mtlcpucachemode,
https://developer.apple.com/documentation/metal/mtlhazardtrackingmode,
https://developer.apple.com/documentation/metal/setting-resource-storage-modes,
https://developer.apple.com/documentation/metal/mtldevice/makebuffer(bytesnocopy:length:options:deallocator:),
https://developer.apple.com/documentation/metal/mtldevice/maxbufferlength
(fetched via DocC JSON, 2026-06-11).

This is the options-bitmask lookup card for buffer creation. Unified-memory background and
the why-Shared design live in `storage-and-synchronization.md` — not repeated here.

`MTLResourceOptions` is an option set combining **one storage mode + one CPU cache mode +
one hazard tracking mode** ("you can combine multiple resource options but you can set
only one storage mode").

## Storage mode bits

| Option                  | Meaning (DocC abstract)                                                                                                   |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `storageModeShared`     | CPU and GPU share access; system memory. **Default on Apple silicon.**                                                    |
| `storageModePrivate`    | GPU only.                                                                                                                 |
| `storageModeMemoryless` | GPU tile memory, exists only during a render pass; **textures only**.                                                     |
| `storageModeManaged`    | CPU and GPU may keep separate copies, explicit sync — legacy x86 Macs (Intel/discrete-GPU); not a thing on Apple silicon. |

## CPU cache mode bits

| Option                      | Meaning                                                                         |
| --------------------------- | ------------------------------------------------------------------------------- |
| (default) `defaultCache`    | "Guarantees that read and write operations are executed in the expected order." |
| `cpuCacheModeWriteCombined` | "Optimized for resources that the CPU **writes into, but never reads**."        |

Write-combined helps streaming CPU→GPU upload paths (writes bypass the cache hierarchy);
it actively hurts any path where the CPU later _reads_ the buffer — uncached reads are very
slow. Rule of thumb: only for write-once-read-never-from-CPU staging data.

## Hazard tracking bits

| Option                        | Meaning                                                                                                                                                                                          |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| (default) `.default`          | Metal picks per resource type: tracked for individual resources, **untracked for heaps**.                                                                                                        |
| `hazardTrackingModeTracked`   | Metal "applies safeguards at runtime": delays writes until prior reads finish, holds subsequent commands until writes complete.                                                                  |
| `hazardTrackingModeUntracked` | Metal "ignores memory hazards": no automatic barriers between encoders/passes, max scheduler freedom — and the cost is manual `MTLFence`/`MTLEvent` discipline (see `mtlevent-and-mtlfence.md`). |

Tracking only applies to work submitted to an `MTLCommandQueue`; the Metal 4
`MTL4CommandQueue` never tracks, even for tracked resources.

## The makeBuffer variants (`MTLDevice`)

| Method                                                | Behavior                                                   |
| ----------------------------------------------------- | ---------------------------------------------------------- |
| `makeBuffer(length:options:)`                         | New allocation, **cleared to zero**.                       |
| `makeBuffer(bytes:length:options:)`                   | New allocation, initialized by **copying** the given data. |
| `makeBuffer(bytesNoCopy:length:options:deallocator:)` | **Wraps an existing allocation** — zero copy.              |

`bytesNoCopy` fine print (all from the DocC parameter docs):

- the pointer must be **page-aligned**;
- the length must result in a **page-aligned region** (i.e. whole pages);
- the memory must lie within a **single VM region**;
- the optional `deallocator` block is invoked when the buffer is deallocated so you can
  release the underlying memory (pass `nil` to opt out).

All variants return `nil` on failure — check it. `MTLDevice.maxBufferLength` gives the
largest allocatable buffer (documented ≥ 256 MB; tens of GB on Apple silicon — query, don't
assume).

### Relevance to the CT2 Metal backend

- The backend uses exactly **one** combination, everywhere:
  `MTLResourceStorageModeShared` + default cache + default (tracked) hazard mode, via
  `newBufferWithLength:options:` in `MetalAllocator::allocate`
  (`src/metal/allocator.mm`). Tracked-by-default is what lets the whole backend run with
  zero fences (see `mtlevent-and-mtlfence.md`).
- **`bytesNoCopy` is NOT used** (grep `bytesNoCopy` in `src/metal/` — no hits). Host data
  reaches Metal buffers by allocate-then-copy: the METAL dispatch case binds to the CPU
  primitives (`src/device_dispatch.h`), so copies into a Shared buffer's `contents` are
  plain host `memcpy`s. The zero-copy wrap would only pay off for something like mapping
  model weights straight from a page-aligned `mmap` of the model file — and the
  whole-pages / single-VM-region / lifetime constraints make that a real project, not a
  flag flip.
- **`cpuCacheModeWriteCombined` would be a bug here**, not an optimization: every op that
  falls through to the CPU reference (and `metal::flush()`-then-read paths in the tests)
  READS buffer contents from the CPU. Don't reach for it.
- `Private`/`Memoryless` remain unusable by design — CPU-reference fallback ops need
  host-addressable memory, and there are no render passes (`storage-and-synchronization.md`).
