# MTLBlitCommandEncoder — GPU-timeline buffer copies and fills

Sources: https://developer.apple.com/documentation/metal/mtlblitcommandencoder,
https://developer.apple.com/documentation/metal/mtlblitcommandencoder/copy(from:sourceoffset:to:destinationoffset:size:),
https://developer.apple.com/documentation/metal/mtlblitcommandencoder/fill(buffer:range:value:),
https://developer.apple.com/documentation/metal/mtlblitcommandencoder/synchronize(resource:)
(fetched via DocC JSON, 2026-06-11). Repo facts read from source the same day.

A blit encoder adds resource-movement commands to a command buffer: filling buffers with
repeating bytes, copying between buffers/textures, and (on managed-memory systems)
CPU↔GPU synchronization. Created with `[commandBuffer blitCommandEncoder]`
(`makeBlitCommandEncoder()`), finished with `endEncoding` — a separate encoder type that
lives on the same command buffer and timeline as compute encoders, and is ordered against
them by the same automatic hazard tracking that orders compute passes (tracked resources).

## The two calls a compute backend cares about

- **`copy(from:sourceOffset:to:destinationOffset:size:)`** — raw byte copy between two
  `MTLBuffer`s on the GPU timeline. **On macOS, `sourceOffset`, `destinationOffset`, and
  `size` must each be a multiple of 4** (any value on iOS/tvOS). Source and destination
  may be the _same_ buffer if the regions don't overlap ("if `size` is less than the
  distance between `sourceOffset` and `destinationOffset`").
- **`fill(buffer:range:value:)`** — sets every byte in `range` to one `UInt8` value.
  Range length must be > 0; the macOS multiple-of-4 rule applies to the range bounds too.
  Note this is a _byte_ pattern — it can zero a buffer or splat `0x7F`, but cannot fill a
  float value.

## `synchronize(resource:)` — irrelevant on this backend

Exists only for **managed**-storage resources, where CPU and GPU each hold a copy and the
blit makes GPU writes visible to the CPU copy. The CT2 Metal backend allocates everything
`MTLResourceStorageModeShared` (one physical allocation on unified memory — see
`resource-storage-modes-and-options.md`), so there is no second copy to synchronize and
nothing to call. CPU visibility here is a _scheduling_ problem (`flush()` /
`synchronize()` in `device.mm`), not a memory-coherence one.

## How the backend actually copies bytes today (verified inventory)

The backend uses **zero blits** (grep `BlitCommandEncoder` in `src/metal/` — no hits).
Three real copy paths exist:

1. **Contiguous copies = CPU-side `std::copy` through `contents`.**
   `StorageView::copy_from` (`src/storage_view.cc`) has a cross-device branch only for
   CUDA; on a Metal build it falls through to `DEVICE_DISPATCH(device,
primitives<D>::copy)`, and `device_dispatch.h` binds the METAL case to `Device::CPU` —
   so the copy is `std::copy` (`src/cpu/primitives.cc`, `primitives<Device::CPU>::copy`)
   over the Shared buffers' CPU-addressable pointers. This covers CPU→Metal weight upload
   (`src/models/model.cc`: `consume<char>` reads file bytes into a CPU `StorageView`,
   then `move_variables_to_device` → `variable.to(device)` → `copy_from`) **and**
   Metal→Metal tensor copies. Correctness of the latter rides on the backend's global
   `flush()`/`synchronize()` discipline — the CPU must not read a source buffer the GPU
   is still writing.
2. **Strided/indexed copies = custom compute kernels on the GPU timeline.**
   `ct2_strided_copy_bytes` (`src/metal/kernels/kernels_msl.h`, driven by
   `metal::strided_copy` in `src/metal/primitives.mm`) underlies Concat/Split/Slide
   (`src/ops/concat.cc`, `split.cc`, `slide.cc`); `ct2_gather_bytes` (`metal::gather`)
   underlies Gather (`src/ops/gather.cc`). These exist precisely because blit copies are
   **contiguous-only** — there is no strided or gather form of `copy(from:to:)`.
3. **Zero-fill = free at allocation, CPU loop otherwise.** New buffers arrive zeroed from
   `newBufferWithLength:` (see `mtlbuffer-api.md`); explicit `StorageView::zero()`/`fill`
   ride the CPU binding's loop over `contents`.

## When a blit would beat each path

- **vs. path 1 (the real candidate):** a blit keeps a large contiguous Metal→Metal copy
  on the **GPU timeline** — no need for the CPU to wait for pending GPU work before
  touching the bytes, and the copy overlaps with CPU encoding of the next op exactly like
  every other committed GPU op. Today's CPU `std::copy` of GPU-resident data is the same
  class of mid-pipeline CPU read the backend otherwise avoids.
- **vs. path 2:** only the degenerate fully-contiguous case could switch, and the byte
  kernels are already bandwidth-bound — no expected win.
- **The macOS multiple-of-4 rule is the gotcha** for int8 tensors at odd offsets — the
  same alignment class the int8 GEMV already guards explicitly
  (`int8-gemv-simdgroup-decode.md`); a blit path would need the same precondition checks
  with a kernel/CPU fallback.

### Relevance to the CT2 Metal backend

- Today: no blits anywhere in `src/metal/`. Copies are CPU `std::copy` via unified memory
  (`src/storage_view.cc` `copy_from` + `src/device_dispatch.h` METAL→CPU binding) or the
  two byte kernels in `src/metal/kernels/kernels_msl.h` (`ct2_strided_copy_bytes`,
  `ct2_gather_bytes` — hosts in `src/metal/primitives.mm`).
- The concrete trigger: a profile showing a large contiguous device-to-device copy that
  forces a `synchronize()` (KV-cache reorganization, beam reordering) — that is the copy
  a `makeBlitCommandEncoder` + `copy(from:to:)` moves onto the GPU timeline. Encode it on
  the same per-op command buffer from `new_command_buffer()` so the
  autorelease/commit discipline (`autoreleasepool-in-long-loops.md`) still holds.
- `fill(buffer:range:value:)` is the GPU-timeline zero; rarely needed since allocation
  already zero-fills.
- `synchronize(resource:)`: never — Shared-only backend, managed-only API.
