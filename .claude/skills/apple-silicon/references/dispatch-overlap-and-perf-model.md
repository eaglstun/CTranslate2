---
title: "Metal performance & dispatch-overlap model (the perf conclusions, with the graveyard)"
summary: >-
  The canonical home for the Metal backend's performance reasoning, distilling
  conclusions (especially dead ends) from METAL_BENCHMARKS.md. Its single
  mental model is the fixed ~0.03-0.04 ms per-op GPU-API floor versus CPU/GPU
  overlap, splitting workloads into compute-bound prefill (GPU wins, fp16
  batch-8 ~2.6x over CPU, 3.7x GEMM at n=2048) and overhead-bound
  autoregressive decode (tall-skinny m=batch GEMVs where the floor dominates
  and fp16 buys almost nothing). The load-bearing insight is that async per-op
  commit overlaps CPU encoding of op N+1 with GPU execution of op N, which is
  why the tempting command-buffer-reuse optimization backfired (-6% bs8
  decode, -23% bs8 prefill) and lives in the graveyard alongside net-zero
  MPSMatrixDescriptor caching. Records the real wins in landing order: async
  command-buffer batching (~20%), shape-keyed MPSMatrixMultiplication caching
  (~35%), and GPU fp16 Add for residuals (27x on that op). Also flags
  Concat/Split GPU graduation as correctness-only and lists open levers like
  .metallib, fp16 RMSNorm, and fused attention.
semantic_id: "yjk4xg-uu-2bAb19RN0pFKc4JWOTgqslClf-GpSdxnY6CnkG568LSwf5Whxj6a84JGsks87z3Z5-9aSrxrxuow"
---

**This is the canonical home for the Metal backend's performance reasoning.**
`METAL_BENCHMARKS.md` has the raw numbers and the journey narrative; this file distills
the _conclusions_ — especially the dead ends — so a future agent doesn't re-derive (or
worse, re-implement) something already proven not to work. Perf conclusions need ONE home,
not two: if a number here disagrees with the benchmarks doc, re-measure and fix both.

Source: `METAL_BENCHMARKS.md` (repo root), `src/metal/device.mm` (command-buffer
lifecycle), `src/metal/gemm.mm` (MPS object cache). Numbers are M4 Max (40-core GPU),
Release, Accelerate CPU baseline.

## The one mental model: per-op API floor vs CPU/GPU overlap

Every Metal op pays a **fixed per-op GPU-API cost** — create a command buffer, (for GEMM)
create `MPSMatrix` descriptors + an `MPSMatrixMultiplication`, encode, commit. Measured at
**~0.03–0.04 ms/op** after caching (see `benchmarking-and-profiling.md` for how that was
isolated). That floor is independent of how much math the op does.

Two regimes fall out of that single fact:

- **Prefill / big GEMMs (compute-bound)** — the op does enough FLOPs that the ~0.04 ms
  floor is noise. The GPU wins big. fp16 batch-8 prefill on Qwen2.5-0.5B is **~2.6× faster
  than CPU** and **4× faster than Metal fp32**.
- **Autoregressive decode / tiny ops (overhead-bound)** — each decode step runs at
  `m = batch`, so every GEMM is a tall-skinny matrix–_vector_ product down in the n<1024
  region where the floor dominates. The CPU (zero GPU API in its path) wins by ~2× on a
  real LLM, and by ~90× on a tiny transliteration model. **fp16 buys almost nothing here**
  (32→35, 60→61 tok/s) because the bottleneck is API overhead, not memory bandwidth.

The GEMM scaling table is the proof: CPU wins below n≈1024, and Metal fp16 hits a stable
**3.7× at n=2048** — the first _dependable_ GPU win. (n=1024 sits right on the crossover
and is a **variance trap**: re-measured 2026-06-09 it swung 0.85×–2.26× vs CPU across 4
back-to-back runs, 2.7× spread — too small to saturate the GPU, too big for overhead to
dominate, so don't quote it as a clean win. See `METAL_BENCHMARKS.md`.) A model's _total_
size doesn't move it between regimes — the _per-op_ shape does. A 500M-param model still
decodes one tiny op at a time.

## The CPU/GPU overlap principle — why per-op commit is already near-optimal

This is the load-bearing insight, and the reason the obvious optimization is a trap.

**Per-op commit lets the GPU run op N while the CPU encodes op N+1.** The command queue is
FIFO; committing each op asynchronously (no per-op `waitUntilCompleted`) keeps both
processors busy. The CPU races ahead encoding; the GPU chews through the committed queue.
You only synchronize at a CPU _read_ (`metal::flush()`).

So the per-op commit cost is **not** wasted serial time — it overlaps with GPU execution.
That is why the intuitive "batch all of a step's ops into one commit to amortize the
commit cost" _backfires_.

## ☠️ THE GRAVEYARD — do not re-dig these

### Command-buffer reuse (one commit per decode step) — TRIED, REVERTED, here's the coffin

The single most tempting "fix" for the decode floor. **It does not work. Do not
re-implement it.**

- **What was built:** a per-thread open command buffer; ops append instead of committing;
  `flush()` commits the calling thread's batch first; per-`parallel_for`-chunk commit to
  handle the Conv1D cross-thread orphan (the 8 Conv1D tests caught that exact case,
  as predicted).
- **It passed full parity.** Correctness was never the problem.
- **It measured neutral-to-negative on a real model (Qwen2.5-0.5B):** flat on bs1 decode,
  **−6% on bs8 decode, −23% on bs8 prefill.**
- **Why:** committing once per step **destroys CPU/GPU overlap.** One big batch leaves the
  GPU idle until the final commit, instead of streaming op-by-op. In GEMM-heavy regimes
  the lost overlap outweighs every microsecond of saved commit cost. **Commit count was
  never the bottleneck.** The per-op-commit floor (post-MM-cache) is already near-optimal;
  the remaining tiny-op decode cost is the GPU-API cost _itself_, which batching cannot
  remove without paying for it in lost overlap.

> ⚠️ If you read "decode is API-overhead-bound" and your first instinct is "so batch the
> command buffers" — that is the exact line of reasoning that was already followed to a
> −23% dead end. The overhead is real but it is _overlapped_. Stop here.
>
> (`METAL_BENCHMARKS.md` historically carried a stale line calling command-buffer reuse
> "the #1 lever for decode." It is not — it was the thing that got reverted. That
> contradiction is precisely why these conclusions now live in one place.)

### `MPSMatrixDescriptor` caching — net-zero, reverted

Caching the descriptors on top of the `MPSMatrixMultiplication` cache was measured net-zero
(the descriptor alloc was already cheap) and reverted. The expensive object to cache was the
multiplication, not the descriptor.

## ✅ THE WINS — what actually moved the needle (in landing order)

1. **Async command-buffer batching** (remove per-op `waitUntilCompleted`) — **~20% e2e.**
   This is what _creates_ the CPU/GPU overlap above. Flush only before a CPU read. The
   global last-committed buffer (`g_last_committed`, mutex, `src/metal/device.mm`) must be
   **global, not thread-local** — Conv1D's `cpu::parallel_for` issues GEMMs on worker
   threads whose batches would otherwise never flush.
2. **Cache `MPSMatrixMultiplication` by shape** — **~35% e2e** (fp32 2015→~1284, fp16
   1183→~804 on the tiny model). The decisive change. MPS takes operands only at _encode_
   time, so one object per shape is reusable; a decoder repeats a handful of shapes per
   layer/step. Encode dropped ~0.042→0.031 ms. See `mps-matrix-multiplication.md`.
3. **GPU elementwise `Add` for fp16 (residual connections)** — **27× on that op**, turned
   fp16 prefill from a loss into the headline win (bs8 1815→559 ms). The M1 `metal::add`
   kernel had silently never been wired into the `Add` op, so residuals ran on the CPU
   reference — fine in fp32 SIMD, catastrophic in software-emulated `half`, plus a flush
   per residual (48/forward). Found by profiling, not guessing (see
   `benchmarking-and-profiling.md`).

### Kept for correctness, but NOT a perf lever

- **Concat/Split graduated to GPU kernels** (KV-cache, every decode step). The hypothesis
  was that per-step CPU-reference flushes dominated decode. Graduating them to the GPU gave
  **no measurable e2e change** — disproving that hypothesis and pointing at the per-op floor
  instead. Kept anyway: it keeps the KV cache fully on-GPU, helps larger models, and is
  parity-verified. Just don't expect it to speed up decode.

## Open levers (unmeasured / secondary)

- **Offline `.metallib`** — removes first-use shader-compile cost. Unmeasured; one-time, so
  it helps startup latency, not steady-state throughput.
- **RMSNorm fp16** — profiler showed ~2.5× slower than fp32 (152→377 ms). Smaller than the
  `Add` bug was; worth a look, not the headline. Candidate for the SIMD-group reduction
  rewrite (see `simd-group-functions.md`).
- **Fused attention / `simdgroup_matrix` GEMM** — would attack the decode floor by issuing
  _fewer, bigger_ ops per step rather than fighting the floor head-on. The structurally
  honest direction, since the floor itself is irreducible per-op.

### Relevance to the CT2 Metal backend

- The whole async model lives in `src/metal/device.mm` (`g_last_committed`, `metal::flush()`,
  `metal::synchronize()`) and rests on Shared/unified memory — see
  `storage-and-synchronization.md` for the _mechanics_ (this file is the _why-it's-fast_).
- The MPS object cache lives in `src/metal/gemm.mm`; `mps-matrix-multiplication.md` explains
  why operands-at-encode-time makes shape-keyed caching valid.
- The decode floor is structural: it traces straight to the KV-cache `Concat` and the
  tall-skinny per-step GEMMs. The _structure_ of that decode step (RoPE + GQA + the
  per-step cache concat) is in the **`ct2-internals`** skill
  (`attention-and-kv-cache.md`) — read it to understand _which_ ops hit the floor and why
  there are so many tiny ones per step.
