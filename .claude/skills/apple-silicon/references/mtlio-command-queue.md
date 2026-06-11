# MTLIOCommandQueue — fast resource loading, and why unified memory mostly neutralizes it

Sources: https://developer.apple.com/documentation/metal/mtliocommandqueue,
https://developer.apple.com/documentation/metal/mtliocommandbuffer,
https://developer.apple.com/documentation/metal/mtliofilehandle,
https://developer.apple.com/documentation/metal/mtldevice/makeiofilehandle(url:compressionmethod:),
https://developer.apple.com/documentation/metal/mtliocompressionmethod
(fetched via DocC JSON, 2026-06-11). Repo facts read from source the same day.

Metal 3's fast-resource-loading API (macOS 13+): a dedicated command-queue type that
schedules **file-system reads directly into Metal resources**, asynchronously, off the
loading thread. The pitch is skipping the app-managed staging copy between disk and GPU
memory.

## API surface (the parts a weights loader would use)

- `MTLIOCommandQueue` — from `device.makeIOCommandQueue(descriptor:)`
  (`MTLIOCommandQueueDescriptor`); supports `enqueueBarrier()`.
- `MTLIOFileHandle` — "represents a raw or compressed file"; from
  `device.makeIOFileHandle(url:)` or `makeIOFileHandle(url:compressionMethod:)`
  (compressed variant is macOS 14+). Codecs: `zlib`, `lzfse`, `lz4`, `lzma`, `lzBitmap`
  — decompression happens as part of the load.
- `MTLIOCommandBuffer` — holds the load commands:
  - `load(_:offset:size:sourceHandle:sourceHandleOffset:)` — file range → `MTLBuffer`
    range;
  - `loadBytes(_:size:sourceHandle:sourceHandleOffset:)` — file range → plain memory;
  - plus `addBarrier()`, `signalEvent(_:value:)`/`waitForEvent(_:value:)` (MTLSharedEvent
    handoff to the compute queue), `commit()`, `waitUntilCompleted()`, `status`/`error`.

Loaded resources are then used with commands on a regular `MTLCommandQueue` from the same
device; the event APIs exist to sequence that handoff.

## How CT2 model weights physically reach MTLBuffers today (verified)

`src/models/model.cc`, the load loop: for each variable, the loader `consume`s
name/shape/dtype from the `std::istream`, constructs a **CPU** `StorageView`, and
`consume<char>(model_file, num_bytes, variable.buffer())` reads the raw bytes straight
into it. After all variables are read (plus alias handling and possible dtype conversion
— `variable.to(float_dtype)` lives in this file), `move_variables_to_device` calls
`variable.to(device)` → `StorageView::copy_from` → on Metal that is a CPU-side
`std::copy` into the Shared `MTLBuffer`'s `contents` pointer (see
`blit-command-encoder.md` for that dispatch chain). So: **disk → CPU StorageView →
memcpy → MTLBuffer**, one staging copy, all on the loader thread.

## The honest analysis: what MTLIO would and wouldn't buy here

- **The headline win doesn't exist on Apple Silicon.** "Directly into GPU resources"
  matters on discrete-GPU systems with a PCIe hop. Here a Shared `MTLBuffer.contents` is
  ordinary CPU-writable memory — `istream::read` _into the destination buffer_ is already
  "direct". The staging copy above is removable **without MTLIO**: allocate the variable
  on `Device::METAL` first and read into its `contents`. That's the cheap version of
  everything MTLIO promises.
- **What MTLIO uniquely adds:** async loads queued off-thread (CT2 already loads once at
  startup, replicas share the model) and transparent decompression — but `model.bin` is
  uncompressed, and CT2's format interleaves per-variable headers with blobs, so loads
  would be one `load(...)` per variable anyway, driven by CPU-side header parsing.
- **What MTLIO cannot do:** the loader's inline work — dtype conversion to the effective
  compute type, aliases, tensor-parallel slicing (all in `model.cc`) — operates on bytes
  after they land; raw file→buffer DMA doesn't subsume any of it.

**Verdict:** if model-load latency or load-time RSS ever shows up as a measured problem,
the first lever is eliminating the CPU staging `StorageView` (load directly into the
Shared buffer), not adopting `MTLIOCommandQueue`. MTLIO becomes interesting only if CT2
ever ships compressed weight files.

### Relevance to the CT2 Metal backend

- `src/models/model.cc` — the load loop and `move_variables_to_device` are the path any
  load-time optimization rewires; the staging copy is there, not in `src/metal/`.
- `src/metal/allocator.mm` — Shared-mode allocation is what makes the "just read into
  `contents`" alternative valid (and is why this card is a pre-emption, not a proposal).
- Keep the backend's measure-first discipline (`dispatch-overlap-and-perf-model.md`):
  nobody has yet shown model load on a profile; this card exists so the MTLIO idea gets
  evaluated in one read instead of prototyped.
