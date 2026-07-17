---
title: "Storage and Synchronization on Apple GPUs"
summary: >-
  Explains Apple unified-memory storage modes (Shared: system memory, CPU+GPU,
  default, zero-copy contents pointer; Private: GPU-only; Memoryless: on-GPU
  tile memory, textures only) and CPU/GPU synchronization patterns (triple
  buffering with a pool of buffer instances, dispatch-semaphore throttling,
  completion handlers, immutable buffers, and asynchronous command-buffer
  commit observed via waitUntilCompleted/addCompletedHandler/status). For the
  CT2 Metal backend it establishes that the entire design rests on Shared
  storage so pointer-based CPU kernels run on Metal-resident data
  (allocator.mm always allocates Shared, never Private/Memoryless); command
  buffers commit asynchronously with a single global g_last_committed handle
  that metal::flush() waits on before any CPU access. The load-bearing lesson:
  that handle must be global not thread-local, because Conv1D's parallel_for
  issues GEMMs on worker threads that would otherwise never flush.
semantic_id: "zl1xxgvus-3Sg719Qt1rFochp2uT2quhDlf3kJS14s6KKXkH__8vGk94fk1F6r4ZpC0smszy_Z5-dazvRuxPpQ"
---

Two related concerns when sharing data between the CPU and an Apple GPU: how a resource is stored (which processor can touch its memory), and how to coordinate read/write timing so neither processor reads garbage or stalls.

## Resource Storage Modes / Unified Memory

Apple GPUs use a **unified memory** model: the CPU and GPU share system memory. Which processor can access a given resource — and at what cost — is determined by the resource's **storage mode**.

### The three modes

| Mode       | Constant                                   | Location                       | Access          | Use for                                                                                                  |
| ---------- | ------------------------------------------ | ------------------------------ | --------------- | -------------------------------------------------------------------------------------------------------- |
| Shared     | `MTLStorageModeShared` / `.shared`         | System memory                  | CPU **and** GPU | Data shared by both processors. **Default** for buffers and textures.                                    |
| Private    | `MTLStorageModePrivate` / `.private`       | System memory                  | GPU **only**    | Resources populated by the GPU (compute/render/blit): render targets, intermediaries, texture streaming. |
| Memoryless | `MTLStorageModeMemoryless` / `.memoryless` | Tile memory **inside** the GPU | GPU **only**    | Temporary per-pass **textures only** (e.g. a depth/stencil texture used within a single render pass).    |

Notes:

- **Shared** — on Apple Silicon, because memory is unified, there is no copy cost: the resource's `contents` pointer is the _same_ memory the GPU uses. Ideal whenever the CPU populates or updates the data.
- **Private** — use when only the GPU touches the resource.
- **Memoryless** — backed by on-GPU tile memory: higher bandwidth, lower latency, and less power than system memory. **Textures only — buffers cannot be memoryless.**

### Quick selection guide

- Populate / update on CPU → **Shared**
- GPU-exclusive → **Private**
- Populate on CPU + frequent GPU access → **Shared**
- Temporary texture used only within GPU passes → **Memoryless**

### Memoryless render target

```objc
// Buffers cannot be memoryless — textures only.
textureDescriptor.storageMode = MTLStorageModeMemoryless;
id<MTLTexture> tex = [device newTextureWithDescriptor:textureDescriptor];
// then set tex as a render pass attachment's texture
```

## Synchronizing CPU and GPU Work

Goal: manage data dependencies and avoid processor stalls by using **multiple instances** of shared resources.

### The data dependency problem

A data dependency exists whenever the CPU and GPU share a resource:

- GPU reads **before** the CPU finishes writing → undefined data.
- GPU reads **while** the CPU is still writing → incorrect data.

Both cases force a **stall**: each processor waits on the other.

### Solution: multiple buffer instances (triple buffering)

Keep a pool of buffer instances (`MaxFramesInFlight = 3`, "triple buffering"). The CPU writes the buffer for frame _n+1_ while the GPU reads the buffer for frame _n_. Cycle through them:

```objc
_currentBuffer = (_currentBuffer + 1) % MaxFramesInFlight;
```

This eliminates per-frame allocation and lets both processors run continuously.

### Throttle the CPU with a dispatch semaphore

Prevent the CPU from running too far ahead of the GPU:

```objc
// create (count = number of in-flight frames)
_inFlightSemaphore = dispatch_semaphore_create(MaxFramesInFlight);

// at frame start — decrement; CPU blocks here when count hits 0
dispatch_semaphore_wait(_inFlightSemaphore, DISPATCH_TIME_FOREVER);

// at frame end — signal from the command buffer completion handler
[commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> buffer) {
    dispatch_semaphore_signal(block_semaphore);
}];
```

The completion handler runs when the GPU finishes the command buffer.

### Command buffer commit is asynchronous

After committing, observe completion by one of:

- `waitUntilCompleted`,
- `addCompletedHandler:`, or
- polling the `.status` property.

### Optimization: immutable buffers

If the CPU always finishes writing before the GPU references a buffer, mark it immutable so Metal can optimize better:

```objc
pipelineStateDescriptor.vertexBuffers[i].mutability = MTLMutabilityImmutable;
```

### Key takeaways

- Triple buffering matches the Core Animation drawable limit.
- Semaphores synchronize work rates without blocking unnecessarily.
- Completion handlers signal GPU completion.
- Immutable buffers enable optimizations.
- Cycling buffers avoids per-frame alloc/free.

### Relevance to the CT2 Metal backend

- The ENTIRE Metal backend design rests on unified memory + Shared storage: a shared MTLBuffer's `contents` pointer is CPU-addressable, which is what lets CTranslate2's existing pointer-based CPU kernels run correctly on Metal-resident data (the METAL dispatch case is bound to the CPU implementation in src/device_dispatch.h). The allocator (src/metal/allocator.mm) always allocates Shared buffers.
- CT2 does NOT use Private or Memoryless: kernels and the CPU reference both need host-addressable memory, and there are no render passes.
- CT2's command buffers are committed ASYNCHRONOUSLY (per-op waitUntilCompleted was removed). A single global last-committed command buffer is tracked (g_last_committed under a mutex in src/metal/device.mm); metal::flush() waits on it and is called before ANY CPU access to Metal memory (every CPU-reference op via METAL_DEVICE_CASE, metal::synchronize(), and transitively via StorageView::to(CPU)). This is the same "let the GPU run ahead, synchronize only at the CPU read" idea as the semaphore/completion-handler pattern, minus the frame loop.
- LESSON from the codebase: the tracked command-buffer handle MUST be global, not thread-local — Conv1D's cpu::parallel_for issues GEMMs on worker threads whose batches would otherwise never get flushed. Per-op commit's FIFO queue gives cross-thread ordering for free; a shared/open command buffer breaks that and needs care.
- Raw host reads immediately after a metal:: op require an explicit metal::synchronize() (async model, like CUDA); reads via to_float32()/expect_storage_eq flush automatically.
