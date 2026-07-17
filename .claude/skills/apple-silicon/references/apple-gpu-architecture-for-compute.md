---
title: "Apple-GPU architecture for compute kernels (calibrate your mental model)"
summary: >-
  Calibrates a compute-kernel author's mental model of Apple GPUs,
  distinguishing provenance-labeled facts (Apple doc, spec, measured-here,
  marketing, unverified). It argues TBDR render machinery is off the compute
  path, leaving a simple model of many cores times 32-wide SIMD ALUs plus
  threadgroup memory over unified device memory; execution width is 32
  (threads_per_simdgroup, matching threadExecutionWidth). Load-bearing
  measured numbers: ~280 GB/s sustained weight reads on the Qwen lm_head int8
  GEMV against a 546 GB/s spec peak, and fp16 at ~1.55x fp32 for GEMM. Key
  structural fact: simdgroup_matrix is half/bfloat/float only with no int8
  WMMA, so int8 wins only in bandwidth-bound decode GEMV, never on ALU
  throughput, unlike CUDA dp4a/tensor cores.
---

Sources: Apple DocC article "Tailor your apps for Apple GPUs and tile-based deferred
rendering" (<https://developer.apple.com/documentation/metal/tailor-your-apps-for-apple-gpus-and-tile-based-deferred-rendering>,
fetched via DocC JSON 2026-06-11); MSL spec
(<https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf>) for
language-level facts; this repo's `METAL_BENCHMARKS.md` for every measured number.

Every claim is provenance-labeled: **[Apple doc]** / **[spec]** / **[measured here]** /
**[marketing]** / **[unverified]**. The point of this card: what a compute-kernel author
can actually _rely on_, vs what is render-lore or guesswork.

## TBDR heritage — and why compute kernels mostly don't care

**[Apple doc]** Apple GPUs are tile-based deferred renderers: the render destination is
split into tiles, each processed by a GPU core, with fast on-chip **tile memory** that
avoids device-memory traffic; imageblocks (structured tile-memory data) "are available to
both kernel and fragment functions." The article's content is overwhelmingly about
_render_ passes (hidden-surface removal, raster order groups, MSAA).

**[measured here / project experience]** For a pure-compute backend like CT2's, none of
the TBDR machinery is on the path: there are no render passes, so a compute kernel sees a
straightforward model — many cores × 32-wide SIMD ALUs + threadgroup memory + unified
device memory. The TBDR fact that _does_ matter indirectly: tile memory is the same
physical fast local store that backs `threadgroup` memory in compute, which is why staging
tiles in threadgroup memory (as `ct2_gemm_s8` does) is the canonical optimization.

## Execution width: 32

**[Apple doc + spec]** Threads execute in SIMD-groups of 32 on Apple GPUs —
`[[threads_per_simdgroup]]` reads 32 and equals
`MTLComputePipelineState.threadExecutionWidth` (see `simd-group-functions.md` for the
full lane-communication surface). Keep threadgroup sizes a multiple of 32; the backend's
fixed 256 = 8 SIMD-groups already is.

## Unified memory: zero-copy, but one bandwidth pool

- **[Apple doc]** CPU and GPU share one physical memory; a Shared-mode `MTLBuffer`'s
  `contents` pointer is CPU-addressable with no copies (the whole CT2 backend rides on
  this — see `storage-and-synchronization.md`).
- **[marketing]** Apple specs the M4 Max at **~546 GB/s** memory bandwidth. That is a
  peak interface number, not a sustained-kernel promise, and it is _shared_ with the CPU.
- **[measured here]** The int8 SIMD-group GEMV sustains **~280 GB/s** of weight reads on
  the Qwen lm_head (n=151936, k=896 at 0.49 ms) — about half of peak, "the right order"
  for a single streaming kernel (see `int8-gemv-simdgroup-decode.md`). Use ~half-of-peak
  as the realistic sustained budget when doing bandwidth-bound arithmetic.

## NO int8 matrix hardware

- **[spec §2.4]** `simdgroup_matrix` element types are **half / bfloat / float only** —
  there is no integer WMMA path (`simdgroup-matrix-functions.md` is the ground truth).
- **[measured here]** Consequence: an int32 MAC is plain ALU work (two ops), so the
  hand-tiled int8 GEMM runs **~2.4 T-MAC/s against a ~3.7 ALU ceiling**, while MPS fp16
  rides dedicated FMA pipelines to **~5.9 T-FMA/s** — large-m int8 is structurally
  ~3–5× slower than fp16 at the kernel level (`METAL_BENCHMARKS.md`, int8 section).
- **[context — NVIDIA-documented, not fetched here]** Calibration: CUDA GPUs have shipped
  `dp4a` (4-way int8 dot product) since SM 6.1 and int8 tensor cores since Volta/Turing,
  where int8 GEMM _beats_ fp16. Do not import that intuition to Apple GPUs — here int8
  wins only where it halves memory traffic (decode GEMV), never on ALU throughput.
- **[spec §7, Metal 4]** The 2025 surface adds `mpp::tensor_ops::matmul2d` with a
  documented char×char→int combination — see `metal4-tensors-and-mpp.md`; whether it maps
  to faster-than-ALU hardware is unpublished.

## fp16 vs fp32 ALU rate: unpublished — use the measured ratios

Apple does not publish per-precision ALU rates for its GPUs **[unverified — no Apple doc
found stating an fp16:fp32 ratio]**. What this project measured (M4 Max, MPS GEMM,
`METAL_BENCHMARKS.md`):

- n=2048 square GEMM: fp16 **11966** GFLOPS vs fp32 **7711** → **~1.55×**.
- Tiny-model e2e: fp16 ~804 ms vs fp32 ~1284 → ~1.6× (mixes bandwidth + ALU).
- fp16 buys ~nothing in overhead-bound decode (32→35 tok/s) — precision only pays where
  compute or bandwidth, not the per-op API floor, is the limit
  (`dispatch-overlap-and-perf-model.md`).

So budget fp16 at ~1.5× fp32 for GEMM-shaped work and treat any stronger claim as
unsourced. (Storage halving is the other, separate fp16 win.)

## Occupancy

Threadgroup-memory and register pressure bound how many threadgroups a core can keep in
flight; sizing rules, `maxTotalThreadsPerThreadgroup`, and the threadgroup-memory limits
live in `occupancy-and-threadgroup-memory.md` (companion reference). Grid/threadgroup
dispatch mechanics are in `compute-kernels-and-dispatch.md`.

### Relevance to the CT2 Metal backend

- The "straightforward ALU + unified memory" model is why the backend works at all:
  CPU-reference ops read GPU buffers directly (`src/allocator.cc`, `src/metal/allocator.mm`)
  and the dispatch shim binds METAL to CPU kernels (`src/device_dispatch.h`) — TBDR never
  enters the picture because `src/metal/` issues only compute encoders.
- The no-int8-matrix-units fact decided the int8 kernel design (`ct2_gemm_s8` /
  `ct2_gemv_s8` in `src/metal/kernels/kernels_msl.h`): win the bandwidth-bound GEMV
  regime, accept the ALU-bound tiled regime (`int8-gemm-kernel-design.md`).
- The ~280 GB/s sustained number is the sanity check for any future "this kernel should
  be bandwidth-bound" claim — compare against it, not against the 546 GB/s spec sheet.
- The 32-wide SIMD width is hard-coded into the GEMV's host/kernel coupling
  (8 SIMD-groups per threadgroup, `src/metal/primitives.mm`); see
  `int8-gemv-simdgroup-decode.md` for what breaks if either side changes.
