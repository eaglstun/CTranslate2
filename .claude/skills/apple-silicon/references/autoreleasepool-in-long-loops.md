# Autorelease pools in long C++ loops (the Whisper 9 GB SIGKILL)

**The project-proven memory lesson:** Metal/MPS Objective-C objects returned at +0
(autoreleased) accumulate in the enclosing autorelease pool. A long-running C++ loop —
a transformer decode/encode loop calling Metal ops thousands of times — running on a
thread with **no run loop and no pool** never drains them. RSS climbs unboundedly while
every heap/allocator metric stays flat, until the OS kills the process.

Sources: clang ARC specification, <https://clang.llvm.org/docs/AutomaticReferenceCounting.html>
(autorelease pool semantics, `@autoreleasepool`); project commit `868d12d3` ("autorelease
pool", 2026-06-09) and `WHISPER_METAL_BRINGUP.md` / `METAL_BACKEND.md` for the measured
before/after.

## What autorelease actually means

From the clang ARC spec:

- An autorelease pool is **"a thread-local list of objects to call release on later"**;
  `autorelease` adds an object to the current thread's innermost pool.
- The behavior of `autorelease` "must be equivalent to sending `release` when one of the
  autorelease pools currently in scope is **popped**." No pop → no release → the object
  lives forever.
- `@autoreleasepool { ... }`: "Upon entry to this block, the current state of the
  autorelease pool is captured. When the block is exited normally … " the pool is popped
  back to that state. It compiles to `objc_autoreleasePoolPush`/`objc_autoreleasePoolPop`
  — a pointer push/pop, **cheap enough to put inside a per-op hot path** (this backend
  pays it per op with no measurable cost; see the perf numbers in
  `dispatch-overlap-and-perf-model.md`, all taken with the pool in place).

The Metal objects that bite: `[queue commandBuffer]`, `computeCommandEncoder`, and
`MPSMatrixDescriptor matrixDescriptorWithRows:…` are all convenience constructors (not
`alloc`/`new`/`copy`), so they return autoreleased (+0) objects.

## Why C++ call stacks don't drain pools

On the main thread of an app, the run loop pops a pool every event-loop turn — most ObjC
code gets draining "for free." CTranslate2 drives ops from **C++ worker threads**
(`src/thread_pool.cc`, `cpu::parallel_for` workers): no run loop, no implicit pool, and a
C++ `for` loop has no scope that ARC or the runtime associates with a pool. Every
autoreleased temporary from every op in the whole run piles up — as **wired memory**
(MTLBuffer-backed command-buffer machinery), not malloc heap.

## The Whisper fp16 incident (commit `868d12d3`)

fp16 Whisper-small transcribing a 730 s file on Metal (M4 Max), beam_size=5:

|          | before                   | after                          |
| -------- | ------------------------ | ------------------------------ |
| outcome  | SIGKILL at ~155 s audio  | done: 102 segments, 9411 chars |
| peak RSS | **9.07 GB and climbing** | **2.06 GB, flat**              |

What it was **not** (each ruled out before the fix, per `WHISPER_METAL_BRINGUP.md`):

- **Not unbounded heap** — heap RSS plateaued; the growth was wired memory.
- **Not command-buffer backlog** — a forced flush cadence didn't help.
- **Not the allocator** — `src/metal/allocator.mm` is direct `newBufferWithLength:` /
  `[buffer release]`, no autorelease in its path.

**Diagnostic signature to remember:** steady total-RSS climb with stable allocator
stats / flat malloc heap, on a process driving ObjC APIs from C++ threads ⇒ suspect an
undrained autorelease pool before anything else.

## Where the pools sit in this backend (what the code DOES)

The fix exploits the backend's op shape: every Metal op is a flat
`new_command_buffer()` → encode → `commit_command_buffer()` pair (`src/metal/device.mm`).
So the pool is **per op**, not per step or per flush:

- `new_command_buffer()` pushes a **thread-local** `NSAutoreleasePool` into
  `g_op_pool` _before_ creating the command buffer (`src/metal/device.mm`, the
  `g_op_pool` block).
- `commit_command_buffer()` drains it (`[g_op_pool release]`) _after_ commit — bracketing
  exactly that op's autoreleased temporaries (encoder, MPS descriptors, the command
  buffer's own +0).
- The committed buffer must survive the drain: it does, because
  `commit_command_buffer()` explicitly `[command_buffer retain]`s it into
  `g_last_committed` (the flush handle) before the pool pops.

Note the spelling: `device.mm` is compiled **without ARC** (see its header comment), so
the pool is a manual `NSAutoreleasePool` alloc/release rather than an `@autoreleasepool`
block — same runtime push/pop, but it can span two functions, which the
`new_command_buffer()`/`commit_command_buffer()` split requires. Thread-local because
ops run concurrently on worker threads (Conv1D's `parallel_for`), each needing its own
pool.

**Rule from `METAL_BACKEND.md`:** any future Metal code path that does **not** go
through the `new_command_buffer()`→`commit_command_buffer()` pair must wrap itself in
`@autoreleasepool` (or the manual equivalent). An op that creates a command buffer some
other way silently reopens the leak.

## Checklist when adding Metal/MPS code

1. Does the new path go through `new_command_buffer()`/`commit_command_buffer()`? If
   yes, it's covered. If no — wrap it.
2. Creating autoreleased objects (`commandBuffer`, encoders, `MPSMatrixDescriptor`,
   `NSString stringWithUTF8String:`, …) in a loop outside that pair? Pool per
   iteration.
3. Returning/caching an object past the pool's drain? It must hold a real retain
   (under MRC: explicit `[obj retain]`, like `g_last_committed`; the
   `MPSMatrixMultiplication` cache in `gemm.mm` holds +1 from `alloc`/`init` instead —
   alloc'd objects are owned, not autoreleased, and never enter the pool).
4. Validate with the signature: run the long workload, watch total RSS (not just heap).
   Flat ≈2 GB is healthy for Whisper-small fp16; a steady climb means a path escaped
   the pool.

### Relevance to the CT2 Metal backend

- The pool push/drain lives in `src/metal/device.mm` (`g_op_pool`, in
  `new_command_buffer()` / `commit_command_buffer()`), added in commit `868d12d3`; the
  design comment above `g_op_pool` is the in-tree summary of this file.
- The incident write-up with the measured table is in `WHISPER_METAL_BRINGUP.md`; the
  standing every-op-must-be-bracketed rule is in `METAL_BACKEND.md` (the autorelease
  section, ~line 115).
- `objcpp-interop-for-mm-files.md` covers the surrounding MRC ownership rules (why
  `alloc`'d cache entries don't need the pool); `dispatch-overlap-and-perf-model.md`
  explains the per-op command-buffer shape the pool brackets.
