---
title: "Objective-C++ interop survival card (editing the `.mm` files)"
summary: >-
  A survival card for editing the src/metal/*.mm Objective-C++ files, whose
  single most important local fact is that the backend is compiled WITHOUT ARC
  (manual retain/release / MRC — no -fobjc-arc anywhere, OBJCXX standard 17 in
  CMakeLists.txt). It explains the header-split discipline (pure-C++ utils.h
  and primitives.h safe from any .cc, .mm-only device.h importing
  Metal/Metal.h) that keeps ObjC types out of C++ translation units by passing
  raw float pointers mapped back via buffer_and_offset(), and the three MRC
  ownership patterns used: ObjC id members in a C++ class (MetalContext), a
  raw-object side table (allocator.mm), and a thread-local +1 cache (gemm.mm).
  Adds the MRC ownership cheat-sheet (alloc/new/copy give +1 owned vs
  autoreleased +0), bridge-cast reference, and two silent-failure rules —
  messaging nil is a no-op returning zero, and NSError** out-params signal
  failure via the return value not the error object.
semantic_id: "ihoz5o8urm2X07t9Q3krPYE4J-ORyqsxDlNwnpCdxv4aHVgH57YrW6_4Xk0uq4wYbisk-sz5jY5-ve2Tjr7tpg"
---

How this repo actually mixes Objective-C and C++ in `src/metal/`, plus the language
rules a contributor needs before touching a `.mm` file. **The single most important
local fact: this backend is compiled WITHOUT ARC** (manual retain/release — MRC), so
half the internet's advice ("ARC handles it") does not apply here.

Sources: clang ARC specification,
<https://clang.llvm.org/docs/AutomaticReferenceCounting.html> (bridge casts, ObjC++
ownership-qualified members); repo files `src/metal/device.mm`, `device.h`, `utils.h`,
`primitives.h`, `gemm.mm`, `allocator.mm`, and the `WITH_METAL` block in
`CMakeLists.txt` (~line 455).

## How the build wires it

`CMakeLists.txt` enables `OBJCXX`, sets `CMAKE_OBJCXX_STANDARD 17`, and marks the four
`src/metal/*.mm` files `LANGUAGE OBJCXX`. **No `-fobjc-arc` is set anywhere** — all four
TUs are MRC. `device.mm` says so at the top:

> This translation unit is compiled as Objective-C++ WITHOUT ARC (manual
> retain/release). Objects created via Create/New rules are owned (+1); singleton-held
> objects live for the process lifetime and are intentionally never released.

## Header discipline: split headers, not `#ifdef __OBJC__`

This repo does **not** use the `#ifdef __OBJC__` dual-personality header trick. It
splits the surface instead:

- `src/metal/utils.h` — pure C++ (`has_gpu()`, `flush()`, `synchronize()`). Safe from
  any `.cc` file; this is what `src/devices.cc`, ops, etc. include.
- `src/metal/primitives.h` — pure C++ kernel entry points (`metal::add`,
  `metal::gemm_s8`, …) taking raw pointers + `dim_t`. Also `.cc`-safe; this is the
  routing target ops call.
- `src/metal/device.h` — `#import <Metal/Metal.h>`, declares `id<MTLDevice>`-returning
  functions and `BufferRange { id<MTLBuffer>; NSUInteger offset; }`. Its own comment:
  "may ONLY be included from .mm translation units."

So ObjC types never leak into a C++ TU; the boundary is raw `const float*` pointers,
mapped back to `MTLBuffer`+offset inside `.mm` code via `buffer_and_offset()`
(`device.h`). Keep new APIs on the same pattern: C++ signature in `primitives.h`,
ObjC implementation in a `.mm`.

## Holding ObjC objects from C++ — the repo's three patterns (all MRC)

1. **ObjC members in a C++ class** — `MetalContext` (`device.mm`) holds
   `id<MTLDevice> _device`, `id<MTLCommandQueue> _queue`, `id<MTLLibrary> _library` and
   an `std::unordered_map<std::string, id<MTLComputePipelineState>>` directly as
   members. Under MRC, `id` is just a pointer — POD, no special class machinery. The
   objects come from `New`/`alloc` (owned, +1) and are deliberately never released
   (process-lifetime singleton).
2. **Raw object in a side table** — `allocator.mm` stores each `MTLBuffer` "raw in the
   side table and `[release]`s in `free()`", keyed by `reinterpret_cast<uintptr_t>` of
   the contents pointer.
3. **Thread-local cache owning +1** — `gemm.mm`'s
   `thread_local std::map<GemmKey, MPSMatrixMultiplication*>`; entries from
   `[[… alloc] init…]` are owned and never released ("mm is owned by the thread-local
   cache; do not release").

No `void*` + `__bridge` indirection exists in this backend — it's unnecessary because
ObjC types never cross into ARC or pure-C++ TUs.

**If ARC were ever enabled:** per the clang spec, ObjC++ class members with ownership
qualifiers (`__strong id`) are legal but make the type behave like a class with
non-trivial constructors/destructors/assignment — i.e. the containing C++ class becomes
non-POD, `memcpy`/`memset` on it becomes wrong, and containers must run real
constructors. The `unordered_map<string, id<…>>` member would then retain/release
through map operations. Switching this backend to ARC is a real migration, not a flag
flip — every explicit `retain`/`release` (e.g. `g_last_committed` in `device.mm`) would
have to be removed or bridged.

## Bridge casts (for reference — needed only at MRC↔ARC or CF boundaries)

From the clang spec, casting between retainable (ObjC) and non-retainable (`void*`/CF)
pointer types:

- `(__bridge T) op` — plain type pun. **No ownership transfer, no retain/release.**
- `(__bridge_retained T) op` — ObjC → non-retainable; ARC **retains** (+1) and the
  recipient must balance it (e.g. `CFRelease`).
- `(__bridge_transfer T) op` — non-retainable → ObjC; ARC takes over and will
  **release** the +1 at the end of the enclosing full-expression.
- Using `__bridge_retained`/`__bridge_transfer` "purely to convince ARC to emit an
  unbalanced retain or release … is poor form."

## Two silent-failure rules to keep in mind

- **Messaging `nil` is a no-op returning zero** — clang spec: when an operation's
  pointer "is null then the effect is a no-op." A nil `id<MTLComputePipelineState>`
  would make every encode call silently do nothing → garbage output, no crash. This is
  why `device.mm` throws `std::runtime_error` immediately on every nil-able creation
  (`MTLCreateSystemDefaultDevice`, `newCommandQueue`, `newFunctionWithName:`,
  pipeline/library creation) instead of letting nil propagate. Follow that pattern.
- **`NSError**`out-params** — Metal creation APIs take`error:&error`and signal
failure via a nil/false *return value*, not via the error object. Check the return
first, then read`error`for the message — exactly what`MetalContext::pipeline()`and`ensure_library()` do (`[[error localizedDescription] UTF8String]`into the
thrown message). Under MRC the returned`NSError\*`is autoreleased; the per-op pool
(see`autoreleasepool-in-long-loops.md`) or the throw path covers it here.

## MRC ownership cheat-sheet as practiced in this backend

| You called                                                                                                      | You got         | You must                                                                                                        |
| --------------------------------------------------------------------------------------------------------------- | --------------- | --------------------------------------------------------------------------------------------------------------- |
| `alloc`/`new`/`copy` (`newCommandQueue`, `alloc] init`)                                                         | +1 owned        | `release` when done (or own forever: caches)                                                                    |
| anything else (`commandBuffer`, `computeCommandEncoder`, `matrixDescriptorWithRows:…`, `stringWithUTF8String:`) | +0 autoreleased | nothing — but a pool must drain it (per-op pool), and `retain` if it must outlive the pool (`g_last_committed`) |

### Relevance to the CT2 Metal backend

- The MRC declaration and all three ownership patterns: `src/metal/device.mm`
  (`MetalContext`, `g_last_committed` retain/release), `src/metal/allocator.mm` (side
  table + `[release]`), `src/metal/gemm.mm` (thread-local +1 cache, `[a_matrix release]`
  after encode).
- The header split: `src/metal/utils.h` and `primitives.h` (C++-safe) vs
  `src/metal/device.h` (`.mm`-only); build wiring in `CMakeLists.txt` `WITH_METAL`
  block (`enable_language(OBJCXX)`, no ARC flag).
- The autoreleased-object lifetime rules interact with the per-op pool from commit
  `868d12d3` — see `autoreleasepool-in-long-loops.md` before adding any object that
  must survive `commit_command_buffer()`.
