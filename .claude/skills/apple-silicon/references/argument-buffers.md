---
title: "Argument buffers — many resources behind one binding"
summary: >-
  Argument buffers pack references to many Metal resources (buffers, textures,
  samplers, inline constants) behind a single setBuffer binding, declared MSL-
  side as a struct. The doc explains the two-tier model: Tier 2 (all Apple
  Silicon) matches C-struct layout so you write MTLBuffer.gpuAddress directly
  with no MTLArgumentEncoder, while Tier 1/older OSes require an argument
  encoder and immutable CPU-accessible buffers. It covers residency via
  useResource and useHeap since argument-buffer-referenced resources are
  invisible to Metal's hazard tracking. The key CT2 verdict: the backend does
  not use them and does not need to (kernels bind at most ~6 buffers), and
  they would only matter paired with an indirect-command-buffer decode-
  batching redesign — measure first.
semantic_id: "zx4w1gmvqu3Tg-19BvkLPQMkJWuTy6qBChc2mJS_4t4qCHkOVcct3QX8H11f7KYYvA8sAMzp9Z5K_Qyvxvx_oQ"
---

Sources: https://developer.apple.com/documentation/metal/mtlargumentencoder,
https://developer.apple.com/documentation/metal/improving-cpu-performance-by-using-argument-buffers,
https://developer.apple.com/documentation/metal/mtldevice/argumentbufferssupport,
https://developer.apple.com/documentation/metal/mtlcomputecommandencoder/useresource(_:usage:),
https://developer.apple.com/documentation/metal/mtlcomputecommandencoder/useheap(_:),
https://developer.apple.com/documentation/metal/mtlbuffer/gpuaddress
(fetched via DocC JSON, 2026-06-11).

An argument buffer is an `MTLBuffer` whose contents are **references to other resources**
(buffers, textures, samplers, acceleration structures, plus inline constants). Declared in
MSL as a struct; bound to the kernel with a single `setBuffer` instead of one call per
resource — "argument buffers use less overhead than assigning each resource individually,"
especially for resource sets that don't change between dispatches (encode once, reuse).

Declared in MSL as a struct taken by reference (compute form, from the DocC article):

```metal
struct ArgumentBufferExample {
  device float4* e;   // buffer reference
  int g;              // inline constant
  // textures/samplers also allowed; unions are not
};
kernel void example(constant ArgumentBufferExample& args [[buffer(0)]]) { ... }
```

## Tier model

`MTLDevice.argumentBuffersSupport` → `MTLArgumentBuffersTier` (`.tier1` / `.tier2`).

- **Tier 2** (all Apple Silicon-era targets; C-struct layout on macOS 13+ / iOS 16+): the
  argument buffer "matches the memory layout of an equivalent C structure" — write
  `MTLBuffer.gpuAddress` (a `uint64`; add byte offsets directly) into the struct member
  yourself, no encoder object needed. This is the Metal 3 "bindless" simplification.
  Tier 2 buffers may be mutable and are pointer-indexable from MSL.
- **Tier 1 / older OSes**: the layout is private and GPU-specific — you must use an
  `MTLArgumentEncoder` (from `MTLFunction.makeArgumentEncoder(bufferIndex:)` or
  `MTLDevice.makeArgumentEncoder(arguments:)`): point it at the destination with
  `setArgumentBuffer(_:offset:)` (size it via `encodedLength`/`alignment`), then encode
  handles with `setBuffer(_:offset:index:)`, `setTexture(_:index:)`,
  `constantData(at:)`, etc. Tier 1 buffers must be immutable and CPU-accessible, and
  can't contain writable textures or pointers to other argument buffers.

## Residency: useResource / useHeap

Resources referenced only _through_ an argument buffer are invisible to Metal's binding
machinery, so you must declare them resident on the encoder before dispatch:

- `useResource(_:usage:)` — makes one resource resident for the rest of the pass and tells
  Metal where to apply hazard tracking. "You don't need to call this method if you bind a
  resource for compute kernels to access" — direct `setBuffer` bindings are exempt.
- `useHeap(_:)` — blankets every resource in a heap as resident **read-only**; anything
  written (and untracked resources generally) still needs `useResource`/fences.

### Relevance to the CT2 Metal backend

- **Not used, and the problem it solves doesn't exist here.** Every kernel dispatch in
  `src/metal/primitives.mm` / `src/metal/gemm.mm` binds at most ~6 real `MTLBuffer`s via
  `setBuffer:offset:atIndex:` plus a handful of `setBytes` scalars (highest index in the
  tree is 11, and 9–11 are scalars) — nowhere near a many-resources binding bottleneck,
  and no textures/samplers at all.
- **The trigger:** argument buffers become interesting only via the decode-loop encode-cost
  floor (`dispatch-overlap-and-perf-model.md`): if tiny decode ops were ever batched into
  reusable encodings (e.g. indirect command buffers, where Tier 2 argument buffers are the
  natural way to feed per-op operand sets), pre-encoded `gpuAddress` tables would replace
  per-dispatch `setBuffer` calls. Note the measured floor is dominated by per-op
  commit/scheduling, not binding calls — so this is paired with a dispatch-batching
  redesign or not at all. Measure first.
