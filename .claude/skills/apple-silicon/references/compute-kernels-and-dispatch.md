---
title: "Metal compute kernels & dispatch"
summary: >-
  The lifecycle of a Metal compute kernel from MSL source to GPU execution:
  the once-per-process object pipeline (MTLDevice, MTLLibrary/MTLFunction, the
  expensive-to-build MTLComputePipelineState, MTLCommandQueue) and the per-
  dispatch command-buffer/encoder/setBuffer/dispatchThreads sequence. It
  explains mandatory MSL address-space keywords, thread_position_in_grid
  replacing host loops, and threadgroup/grid sizing via
  maxTotalThreadsPerThreadgroup and threadExecutionWidth (32 on Apple GPUs).
  It contrasts dispatchThreads (non-uniform, no bounds check) against
  dispatchThreadgroups (manual, needs a guard). CT2 notes: device.mm runs
  steps 1-4 but compiles MSL at runtime lazily via newLibraryWithSource, hand
  kernels use dispatchThreads (hence no OOB guards), and every referenced
  buffer must be bound (the dummy-buffer gotcha).
---

Sources (Apple Developer Documentation):

- <https://developer.apple.com/documentation/metal/performing-calculations-on-a-gpu>
- <https://developer.apple.com/documentation/metal/calculating-threadgroup-and-grid-sizes>

How a compute kernel goes from MSL source to running on the GPU, and how to size
the grid of threads correctly.

## The object pipeline (once per process / per op)

1. **`MTLDevice`** — thin abstraction for one GPU. `MTLCreateSystemDefaultDevice()`.
   All other objects are created directly or indirectly from a device and are only
   usable with that device.
2. **`MTLLibrary` / `MTLFunction`** — `newDefaultLibrary` (Xcode-compiled `.metal`
   files) or `newLibraryWithSource:` (runtime compile). `newFunctionWithName:` gets
   a proxy for one `kernel` function — _not_ executable code yet.
3. **`MTLComputePipelineState`** — `newComputePipelineStateWithFunction:error:`
   finishes compiling the function **for this specific GPU**. Expensive; create
   once, reuse. Carries `maxTotalThreadsPerThreadgroup` and `threadExecutionWidth`.
4. **`MTLCommandQueue`** — `newCommandQueue`. Schedules command buffers. Create once.

## Per-dispatch

```
id<MTLCommandBuffer>        cb  = [queue commandBuffer];
id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
[enc setComputePipelineState:pso];
[enc setBuffer:bufA offset:0 atIndex:0];   // indices match MSL arg order, from 0
[enc setBuffer:bufB offset:0 atIndex:1];
[enc setBuffer:bufR offset:0 atIndex:2];
[enc dispatchThreads:gridSize threadsPerThreadgroup:tgSize];
[enc endEncoding];
[cb commit];               // async; Metal schedules onto the GPU
[cb waitUntilCompleted];   // or addCompletedHandler:, or poll .status
```

Buffer argument indices are assigned in the order arguments appear in the MSL
function signature, starting at 0. `offset:` lets one buffer hold multiple args.

## The MSL kernel

```metal
kernel void add_arrays(device const float* inA   [[buffer(0)]],
                       device const float* inB   [[buffer(1)]],
                       device float*       result[[buffer(2)]],
                       uint index [[thread_position_in_grid]])
{
    result[index] = inA[index] + inB[index];   // no for-loop: one thread per element
}
```

- `kernel` = a public compute function (the only functions the host can see; can't be
  called by other shaders).
- **Address space keywords are mandatory on every pointer.** `device` = persistent
  memory the GPU reads/writes. Others: `constant`, `threadgroup`, `thread`.
- The host for-loop is replaced by the **grid of threads**. `[[thread_position_in_grid]]`
  gives each thread its unique index (scalar for a 1D grid, `uint2`/`uint3` for 2D/3D).

## Threadgroups & grid sizing

Metal subdivides the grid into **threadgroups**, dispatched to different GPU cores.
Two pipeline-state properties drive sizing (both fixed after PSO creation, but two
PSOs on the same device may differ):

- **`maxTotalThreadsPerThreadgroup`** — max threads in one threadgroup; depends on
  the GPU _and_ the kernel's register/memory pressure.
- **`threadExecutionWidth`** — number of threads the GPU runs in lockstep (the SIMD
  width). Make threadgroup sizes a multiple of this.

Largest well-shaped 2D threadgroup:

```objective-c
NSUInteger w = pso.threadExecutionWidth;
NSUInteger h = pso.maxTotalThreadsPerThreadgroup / w;
MTLSize tg = MTLSizeMake(w, h, 1);
```

### Two ways to dispatch

**`dispatchThreads:threadsPerThreadgroup:`** (preferred, needs non-uniform threadgroup
support — all Apple Silicon has it). You pass the **total thread count**; Metal makes
non-uniform edge threadgroups so the grid matches your data exactly. **No bounds check
needed in the kernel.**

```objective-c
MTLSize grid = MTLSizeMake(arrayLength, 1, 1);
[enc dispatchThreads:grid threadsPerThreadgroup:tg];
```

**`dispatchThreadgroups:threadsPerThreadgroup:`** (older / manual). You pass the
**threadgroup count**, rounded up to cover the data — which can launch threads past the
end, so the kernel **must** guard:

```objective-c
MTLSize tgPerGrid = MTLSizeMake((width + w - 1)/w, (height + h - 1)/h, 1);
```

```metal
if (pos.x >= out.get_width() || pos.y >= out.get_height()) return;
```

Underusing threads (grid smaller than the work) wastes the GPU; the goal is one
thread per data element with minimal waste.

## Resource storage (see also storage-and-synchronization.md)

`newBufferWithLength:options:MTLResourceStorageModeShared` — CPU- and GPU-visible.
`buffer.contents` is a host pointer into the same memory the GPU uses (unified memory).

---

### Relevance to the CT2 Metal backend

- `src/metal/device.mm` does exactly steps 1–4, but compiles MSL at **runtime** via
  `newLibraryWithSource:` (embedded in `kernels/kernels_msl.h`) rather than a default
  library — and compilation is **lazy** (`ensure_library()`), so a single kernel that
  fails to compile doesn't take down device setup / allocation / MPS GEMM.
- The hand-written kernels (`ct2_softmax`, `ct2_rms_norm`, rotary, bias-add, …) use
  `dispatchThreads` (non-uniform) — that's why none of them carry an out-of-bounds
  guard. The row-reduction kernels use a **fixed 256-thread threadgroup** for tree
  reductions; keep that a multiple of `threadExecutionWidth` (32 on Apple GPUs).
- Buffers come from the shared-storage allocator (`src/metal/allocator.mm`); the
  `contents` pointer is what satisfies CT2's pointer-based `StorageView`/`Allocator`
  contract. Binding-index order in `primitives.mm` must match the MSL arg order — and
  remember the gotcha: **every referenced buffer must be bound**, hence the dummy
  buffer at unused indices when `lengths`/scalar operands are absent.
