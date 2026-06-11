# MTLBinaryArchive & PSO caching — what's left to win here (answer: nothing)

Sources: https://developer.apple.com/documentation/metal/mtlbinaryarchive,
https://developer.apple.com/documentation/metal/creating-binary-archives-from-device-built-pipeline-state-objects
(fetched via DocC JSON, 2026-06-11); warmup/cache numbers cited from
`pipeline-and-library-compilation.md` (this skill), which carries the original receipts.

`MTLBinaryArchive` (macOS 11+) is "a container for pipeline state descriptors and their
associated compiled shader code" — **compiled GPU binaries at the PSO level**. This is a
different level from a `.metallib`: a metallib skips the MSL→AIR _source_ compile; an
archive skips the AIR→GPU-binary _backend_ compile and PSO build. The project already
measured the metallib level dead; this card closes the archive level too.

## The API in one pass

- **Create/harvest:** `device.makeBinaryArchive(descriptor:)` with a
  `MTLBinaryArchiveDescriptor` whose `url` is `nil` (= new archive), then
  `addComputePipelineFunctions(descriptor:)` for each pipeline — **takes an
  `MTLComputePipelineDescriptor`**, not a bare function. `serialize(to:)` writes a
  `.binary.metallib` to disk.
- **Use:** load via a descriptor whose `url` points at the file, then list the archive in
  a pipeline descriptor's `binaryArchives` property before creating the PSO — Metal pulls
  the prebuilt binary instead of compiling.
- **Cross-GPU:** an archive built on-device contains one slice for the host GPU
  (`metal-lipo -archs` shows e.g. `applegpu_g13g`); `metal-source`/`metal-tt` re-target
  other GPUs from the embedded `mtlp-json` config script. Archives also retain an AIR
  slice (`air64_v26`) because "Metal may invalidate binaries when upgrading a device's
  operating system" — i.e. archives are not even durable across OS updates.

## The system's implicit cache already does this job

Per the receipt in `pipeline-and-library-compilation.md` (recorded in the
`src/metal/kernels/kernels_msl.h` header): macOS caches compiled shaders **by source hash
at the system level** — `newLibraryWithSource:` costs ~123 ms only on a truly cold cache,
~0.5 ms in every subsequent process. The repo's own PSO creation
(`newComputePipelineStateWithFunction:` in `src/metal/device.mm`, lazy per kernel, ~30
kernels) rides the same warm path and has never registered on a profile.

## Could an archive kill the ~493 ms first-MPS-GEMM warmup? No.

That warmup (measured; see `pipeline-and-library-compilation.md` /
`mps-matrix-multiplication.md`) is **MPS-internal pipeline compilation**:
`MPSMatrixMultiplication` builds its own pipelines inside the framework. This repo never
sees those pipeline descriptors, so it has nothing to call
`addComputePipelineFunctions(descriptor:)` on — there is no API handle by which an
app-side binary archive can pre-bake MPS's PSOs. The one warmup cost that matters here is
structurally out of reach.

## What adopting archives would actually cost

- `src/metal/device.mm` creates PSOs **function-based**
  (`newComputePipelineStateWithFunction:`); archives require switching every creation to
  descriptor-based (`MTLComputePipelineDescriptor` + `binaryArchives`).
- Plus the same ship-an-asset problem that killed the metallib idea: CT2 is a bare
  `.dylib` with no bundle to resolve an archive path from, and a harvest/serialize step
  in the build.

**Verdict (graveyard):** archives could only shave the repo's own already-sub-millisecond
warm-cache PSO builds, cannot touch the MPS warmup, and don't survive OS updates without
the AIR fallback recompiling anyway. There is nothing left for binary archives to win in
this backend. Don't re-dig — if first-run latency ever matters, the target is the MPS
warmup, and the lever there is warming it (one dummy GEMM at init), not archiving.

### Relevance to the CT2 Metal backend

- `src/metal/device.mm` — `ensure_library()` (runtime source compile, default options)
  and the function-based PSO cache are the two things an archive adoption would rewrite;
  both are currently fast-by-cache and not worth touching.
- `src/metal/kernels/kernels_msl.h` header comment — the measured metallib receipt this
  card extends one level down.
- `tests/metal_test.cc` — the benchmark harness already warms explicitly ("warmup (MPS
  pipeline compilation, allocator priming)"), which is the practical answer to first-run
  cost in every consumer.
