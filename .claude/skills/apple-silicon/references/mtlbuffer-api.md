---
title: "MTLBuffer & the MTLDevice allocation surface (lookup card)"
summary: >-
  MTLBuffer and the MTLDevice allocation surface: the three creation variants
  (makeBuffer(length:options:) which zero-fills, makeBuffer(bytes:...) which
  copies, and makeBuffer(bytesNoCopy:...) which wraps existing page-aligned
  single-VM-region memory), plus the instance surface — contents() (nil only
  for private storage), length vs allocatedSize, gpuAddress (Metal 3),
  didModifyRange (managed-only), setPurgeableState, and label. The CT2 backend
  uses exactly one variant,
  newBufferWithLength:options:MTLResourceStorageModeShared in
  src/metal/allocator.mm, whose contents pointer feeds StorageView (the
  unified-memory trick), with an address-ordered side table mapping any
  interior pointer back to buffer+offset. Notes bytesNoCopy as the unused
  mmap-weights opportunity, the explicit %4 offset alignment checks gating
  int8 GEMV, and that no buffer sets label so GPU captures show anonymous
  buffers.
---

Sources (Apple Developer Documentation, fetched via DocC JSON, 2026-06-11):

- <https://developer.apple.com/documentation/metal/mtlbuffer> (+ `contents()`, `length`, `gpuaddress`, `didmodifyrange(_:)`)
- <https://developer.apple.com/documentation/metal/mtldevice/makebuffer(length:options:)> (+ `bytes:` and `bytesNoCopy:` variants, `maxbufferlength`)
- <https://developer.apple.com/documentation/metal/mtlresource/setpurgeablestate(_:)> and `mtlresource/label`

`MTLBuffer` is "a resource that stores data in a format defined by your app" — Metal
knows only its **size**, never its contents/layout. A buffer is usable **only with the
`MTLDevice` that created it**. You never implement the protocol; you create instances
via three `MTLDevice` factory methods.

## The three creation variants

| Method (Swift name / ObjC selector)                                                  | What it does                                                                                                                                 |
| ------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `makeBuffer(length:options:)` / `newBufferWithLength:options:`                       | New allocation, **cleared with zero values**. Returns `nil` on failure.                                                                      |
| `makeBuffer(bytes:length:options:)` / `newBufferWithBytes:…`                         | New allocation, initialized by **copying** `length` bytes from your pointer.                                                                 |
| `makeBuffer(bytesNoCopy:length:options:deallocator:)` / `newBufferWithBytesNoCopy:…` | **Wraps an existing allocation** — no new storage. Framework calls `deallocator(ptr, len)` when it frees the buffer (pass `nil` to opt out). |

**The `bytesNoCopy` alignment rule (exact, from the doc):** the pointer must be
**page-aligned**, the `length` must be a size that "results in a page-aligned region of
memory", and the existing allocation "needs to exist within a single virtual memory (VM)
region." So: `mmap`/`vm_allocate`/`posix_memalign(getpagesize(), …)` memory qualifies;
an arbitrary `malloc` pointer does not.

`options:` is `MTLResourceOptions` — storage mode (Shared/Private/Managed/Memoryless)
plus hazard-tracking mode; see `storage-and-synchronization.md` for why this backend is
all-Shared. `MTLDevice.maxBufferLength` caps a single buffer (doc guarantees ≥ 256 MB;
on this machine it is far larger — query, don't assume).

## Instance surface

- **`contents()` / `.contents`** → `UnsafeMutableRawPointer`. "A pointer to the shared
  copy of the buffer data, or **NULL for buffers allocated with a private resource
  storage mode**" — i.e. the doc's only nil case is `.private` ("private resources
  aren't CPU-accessible"). For the Shared buffers this backend allocates, it is a valid
  host pointer. (The doc does not literally say "never nil for shared on Apple Silicon";
  it says nil happens for private.)
- **`length`** — the **logical** size in bytes (what you asked for; the OS may round the
  physical allocation up — that rounded number is `allocatedSize`, see
  `memory-footprint-and-residency.md`).
- **`gpuAddress`** — `MTLGPUAddress`, **Metal 3** (macOS 13+ / iOS 16+). The buffer's
  raw GPU virtual address, for bindless/argument-buffer-style access. The DocC page
  carries a declaration only (no discussion); unused here.
- **`didModifyRange(_:)`** — **managed-mode only** (macOS / Mac Catalyst API). After CPU
  writes to a `.managed` buffer you must call it or GPU reads are _undefined behavior_.
  Legacy note for this backend: Shared buffers on unified memory never need it.
- **`setPurgeableState(_:)`** (from `MTLResource`) — returns the **prior** state;
  `.keepCurrent` queries without changing. `.volatile` marks contents discardable at any
  time without informing the app — the documented pattern for "larger caches of idle
  memory" that don't count against you. Lock (`.nonVolatile`) before touching the data
  again. Details in `memory-footprint-and-residency.md`.
- **`label`** (from `MTLResource`) — `String?`, settable. "Object and command labels are
  useful identifiers at runtime or when profiling and debugging your app using any Metal
  tool" — i.e. this is what names a buffer in Xcode GPU captures and Instruments.

---

### Relevance to the CT2 Metal backend

- The allocator (`src/metal/allocator.mm`, `MetalAllocator::allocate`) uses exactly
  **one** variant: `newBufferWithLength:options:MTLResourceStorageModeShared`. So every
  CT2 Metal allocation is zero-filled by Metal, and `[buffer contents]` (line 36) is the
  pointer handed to `StorageView` — the unified-memory trick the whole backend rides on.
  The address-ordered side table maps that pointer (or any pointer _inside_ the
  allocation) back to `{buffer, offset}` via `buffer_and_offset()` for kernel binding.
- `bytesNoCopy` is the unused variant to remember: it could wrap CT2's mmap'd model
  weights into an `MTLBuffer` with zero copy — but only if the region is page-aligned
  and a single VM region, per the rule above.
- **Alignment story for the int8 GEMV:** the GEMV preconditions in
  `src/metal/primitives.mm` (`gemm_s8`: `k % 4 == 0 && lda % 4 == 0 && ldb % 4 == 0 &&
a_buffer.offset % 4 == 0 && b_buffer.offset % 4 == 0`) are checked **explicitly
  against the side-table offset** — the backend deliberately does not assume anything
  about `MTLBuffer` base alignment (the fetched docs state no base-alignment guarantee
  for `newBufferWithLength:`), so sub-view offsets are what decide GEMV vs the slower
  tiled fallback.
- **Free debugging win, currently unused:** nothing in `src/metal/` sets `label` —
  every buffer shows up anonymous in a GPU capture (`gpu-capture-and-shader-validation.md`).
  A one-line `buffer.label = @"ct2 <size>"` in `allocator.mm` (or per-weight names at
  load) would make capture/Instruments memory views readable at zero runtime cost.
