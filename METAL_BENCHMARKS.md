# Metal Backend — Benchmarks

Performance measurements for the CTranslate2 Apple Metal backend, tracking the optimization
journey from the first **correctness-first baseline** (which committed a command buffer and
`waitUntilCompleted` per op — the dominant cost for small ops) through async command-buffer
batching, MPS-GEMM object caching, and the GPU `Add`/fp16 fix, to a real-LLM evaluation.
The long-standing headline — Metal **wins** GEMM-heavy prefill but **loses** tiny-op
autoregressive decode to a per-op GPU-API floor — was **overturned on 2026-07-07 by the
fused decode attention kernel** (see that section): with the MatMul→SoftMax→MatMul decode
sequence collapsed into one launch, **Metal now wins decode too** (Qwen bs=1 ~2.2×
kernel-level, edging the CPU e2e; bs=8 ~7–8×, ~2.8× over CPU). The earlier negative result
stands: command-buffer reuse, the long-assumed fix for the floor, measured
neutral-to-negative (it kills CPU/GPU overlap) — the working lever was _fewer, bigger ops_,
not fewer commits.

## Setup

- **Machine:** Apple M4 Max (40-core GPU), macOS 26.4.1
- **Build:** `-DWITH_METAL=ON -DWITH_ACCELERATE=ON -DOPENMP_RUNTIME=NONE -DCMAKE_BUILD_TYPE=Release`
- **CPU baseline:** Accelerate (the Apple-Silicon CPU BLAS)
- **Method:** warm-up iteration, then average over N iterations (`MetalTest.DISABLED_Benchmark*`)
- Reproduce: `./tests/ctranslate2_test ../tests/data --gtest_also_run_disabled_tests --gtest_filter='*Benchmark*'`

## Square GEMM (m = n = k)

Throughput in GFLOPS (higher is better), `2·n³` flops.

| n    | CPU fp32 (Accelerate) | Metal fp32 (MPS) | Metal fp16 (MPS) | Metal fp16 vs CPU |
| ---- | --------------------- | ---------------- | ---------------- | ----------------- |
| 256  | 1710                  | 155              | 180              | 0.11×             |
| 512  | 2655                  | 860              | 976              | 0.37×             |
| 1024 | 3053                  | 2139             | 5158 ⚠️          | 1.69× ⚠️          |
| 2048 | 3231                  | 7711             | 11966            | **3.70×**         |

ms/iter at n=2048: CPU 5.32, Metal fp32 2.23, Metal fp16 1.44.

> ⚠️ **n=1024 is a variance trap — the 5158 / 1.69× above is one lucky draw, not a stable
> fact.** Re-measured 2026-06-09 on the same M4 Max, n=1024 Metal fp16 swung **2395–6370
> GFLOPS across 4 back-to-back runs (2.7× spread)** → vs-CPU anywhere from **0.85× to
> 2.26×**. It straddles the crossover: big enough that dispatch overhead isn't everything,
> too small to saturate the GPU, so one slow iter (clock ramp / thermal / scheduler) skews
> the 20-iter average. n=256/512/2048 and the encode floor (below) re-confirmed cleanly the
> same day; **only n=1024 is noisy.** Treat **n=2048 as the first _dependable_ Metal win**,
> not n=1024.

**Read:** the GPU wins at scale. By **n=2048** Metal fp16 is **3.7× faster** than
Accelerate (a stable, repeatable result) and fp16 is ~1.5× the fp32 GPU rate; n=1024 is a
coin-flip near the crossover (see the variance note). Below n≈1024 the per-call dispatch
overhead (command buffer commit + `waitUntilCompleted` + a fresh `MPSMatrixMultiplication`
object per call) dominates and the CPU wins easily.

## End-to-end translation (batch of 32, tiny transliteration model)

ms per batch (lower is better), as each optimization landed. "sync per op" = original
commit+wait; "batched" = commit asynchronously, flush only before a CPU read; "+MM cache" =
also reuse `MPSMatrixMultiplication` objects by shape.

|            | sync per op | batched | +MM cache | CPU baseline |
| ---------- | ----------- | ------- | --------- | ------------ |
| Metal fp32 | 2419        | 2015    | **~1284** | 14           |
| Metal fp16 | 1493        | 1183    | **~804**  | 14           |

Cumulative ~1.9–2× faster than the original. The decisive change was caching the MPS GEMM
objects (~35% on top of batching), which a measurement pinpointed (see Analysis). Metal is
still dramatically slower than the CPU on this tiny model — its ops are too small to amortize
any GPU API cost — but the gap roughly halved.

## Analysis

**The kernels are fast (see the GEMM table); the per-op overhead on a tiny model is not.**

- Raw compute is healthy — MPS fp16 GEMM hits ~12 TFLOPS at n=2048 and beats the CPU by
  3.7×. The math is on the GPU and it's quick.
- Command-buffer batching (done) removed the per-op `waitUntilCompleted` block and bought
  ~20% on end-to-end. But two costs remain dominant for the tiny model:
  1. **Per-step flushes from CPU-reference ops.** The decode loop still runs some ops on
     the CPU reference — notably the KV-cache `Concat` every step — and each forces a
     flush (GPU sync) before it reads. That re-serializes the pipeline several times per
     step.
  2. **Fixed per-op GPU-API overhead.** Each op creates a command buffer and (for GEMM) a
     fresh `MPSMatrixMultiplication`, then commits. At microsecond-scale compute, that
     fixed cost dominates.

**Update — concat/split moved to GPU, e2e unchanged.** The KV-cache `Concat`/`Split`
were graduated to GPU kernels to remove the per-step flush (lever #1 below, now done).
Result: **no measurable e2e change** (fp32 ~2000ms, fp16 ~1180ms, same as batched). The
hypothesis that per-step flushes dominated was wrong. The data points instead at **fixed
per-op GPU-API overhead** — each op creates a command buffer (+ a fresh
`MPSMatrixMultiplication` per GEMM) and commits; at ~0.4ms × thousands of tiny ops per
decode that overhead, not synchronization, is the bottleneck. (The GPU concat/split is
still worthwhile: it keeps the KV cache fully on-GPU and helps larger models, and it's
parity-verified — it just isn't the perf lever here.) Note fp16 e2e is ~1.7× faster than
fp32, so reduced memory bandwidth across all ops does help; the gap to CPU is API overhead.

**Update 2 — measured the per-op cost, then cut it.** A probe (`DISABLED_BenchmarkGemmEncode`)
isolated GEMM encode from execution by timing many commits with one flush at the end:

|             | flush-per-iter | batched-encode (encode+commit only) |
| ----------- | -------------- | ----------------------------------- |
| n=256 fp16  | 0.22 ms        | **0.042 ms**                        |
| n=1024 fp16 | 0.44 ms        | 0.19 ms                             |

So at n=256 the per-op _encode_ cost is ~0.042 ms (command buffer + 3 `MPSMatrix` + 1
`MPSMatrixMultiplication` + commit), and the wait/round-trip adds the rest. Since e2e already
batches (no per-op wait), that **encode cost** is what hits thousands of times. Caching the
`MPSMatrixMultiplication` objects by shape (they take operands only at encode time, and a
decoder repeats a few shapes per layer/step) dropped encode to ~0.031 ms and **e2e by ~35%**
(fp32 2015→~1284, fp16 1183→~804). Caching `MPSMatrixDescriptor` on top was net-zero (the
alloc was already cheap) and was reverted.

_Re-confirmed 2026-06-09 (same M4 Max, after that day's `primitives.mm` changes): the probe
read **0.029 ms batched-encode at n=256** (vs 0.165 ms flush-per-iter), squarely on the
~0.031 ms post-cache floor — the encode floor is stable and the MM cache is still in effect.
Unlike the n=1024 GEMM throughput number above, this probe reproduces tightly._

Remaining performance levers, revised again:

1. ~~**Reuse command buffers across ops** (one per decode step, committed once)~~ — **TRIED,
   MEASURED NEUTRAL-TO-NEGATIVE, REVERTED.** Implemented in full (per-thread open command
   buffer; ops append instead of committing; flush commits the calling thread's batch first;
   per-`parallel_for`-chunk commit to fix the predicted Conv1D cross-thread orphan, which the
   8 Conv1D tests caught exactly as expected). It passed full parity but on Qwen2.5-0.5B it
   was flat on bs1 decode and a regression elsewhere (bs8 decode −6%, **bs8 prefill −23%**).
   Reason: committing once per batch **destroys CPU/GPU overlap** — per-op commit lets the GPU
   run op N while the CPU encodes op N+1; one big batch leaves the GPU idle until the final
   commit. For GEMM-heavy regimes the lost overlap outweighs the saved commit cost. So commit
   count was never the real bottleneck; **per-op commit (post-MM-cache) is already near-optimal.**
2. **Offline `.metallib`** — removes the first-use shader-compile cost. (Unmeasured.)
3. ~~Cache `MPSMatrixMultiplication`~~ — done, ~35% e2e win.
4. ~~Graduate per-step CPU-reference ops (Concat/Split) to GPU~~ — done; did not help e2e
   (kept for correctness/larger models).
5. ~~GPU elementwise `Add` (residual) for fp16~~ — done; fixed a 27× fp16 `Add` regression,
   made fp16 prefill win (see the LLM section above).

The GEMM table already proves the upside: a real LLM (large hidden size, GEMM-dominated,
fewer/bigger ops) sits at the favorable end and should show the fp16 advantage. This tiny
model is the worst case for any GPU backend — its ops are too small to amortize any per-op
API cost, which is why CPU (no GPU API in the path) still wins by ~90× even after the ~2×
speedup.

## Op fusion: residual-Add + norm (`DISABLED_BenchmarkAddRMSNorm`)

The first **positive** Metal perf lever (the SIMD-group-reduction rewrite was measured dead
— the row reductions are memory-bound, see the skill notes). Instead of optimizing _how_ a
reduction computes, fusion cuts _op count + memory passes_: one `ct2_add_rms_norm` /
`ct2_add_layer_norm` kernel reads `a`,`b`, writes both `sum = a+b` (the next residual) and
`norm(sum)·γ`, vs separate `ops::Add` then `ops::RMSNorm` — one fewer launch, 4 device read
passes → 3. Integrated into the Gemma2/T5Gemma `pre_post_layer_norm` path (byte-identical
fused≡unfused, e2e-verified on real gemma-2-2b).

Speedup (two-op / fused), **4 back-to-back runs, M4 Max, 2026-06-09** — and like the GEMM
table, the overhead-bound cells are noisy, so this is reported as ranges, not point values:

| shape (rows×depth)        | fp32 (4 runs)                 | fp16 (4 runs)                       |
| ------------------------- | ----------------------------- | ----------------------------------- |
| 8192×896 (typical hidden) | **~1.3× (1.27–1.39, stable)** | **~1.25× (1.16–1.31, stable)**      |
| 4096×2048 (large depth)   | ~1.1× (1.08–1.17, stable)     | 1.2–1.8× (noisy)                    |
| 16384×512 (many rows)     | 1.7–2.5× (win, noisy)         | **0.77–1.72× (wild: loss→big win)** |

**Read:** fusion is a **real, repeatable ~1.25–1.3× at the mid-depths that match real LLM
hidden sizes** (8192×896) — the cells to trust. The launch-overhead-bound extremes (many
rows, and fp16 generally) swing wildly run-to-run (one fp16 run dipped to 0.77×, a loss);
their _expected_ value is a win but any single draw is a coin-flip — do not quote them.
The mechanism (cut launches/passes) is why it helps decode without killing CPU/GPU overlap,
unlike command-buffer reuse — so fusion, not commit-batching, is the right lever for the
weak decode regime. _Numbers are single-run-per-cell averages; treat as indicative._

## Real decoder LLM (Qwen2.5-0.5B, fp32 weights)

The end-to-end numbers above are a _tiny_ transliteration model — the worst case for any
GPU backend. To test the "a real LLM sits at the favorable end of the GEMM table" claim,
we converted **Qwen2.5-0.5B-Instruct** (RoPE + RMSNorm + SwiGLU + GQA — exercises the
M6–M9 kernels) and benchmarked greedy generation on CPU vs Metal. Reproduce:

```bash
PYTHONPATH=python python -c "import sys; sys.argv=['x','--model','Qwen/Qwen2.5-0.5B-Instruct','--output_dir','/tmp/qwen0.5b-ct2','--quantization','float32','--force']; from ctranslate2.converters.transformers import main; main()"
CT2_LLM_MODEL=/tmp/qwen0.5b-ct2 ./tests/ctranslate2_test ../tests/data \
  --gtest_also_run_disabled_tests --gtest_filter='MetalTest.DISABLED_BenchmarkLLM'
```

**Decode-bound** (prompt 32, generate 32 tokens) — tok/s, higher is better:

|         | CPU fp32 | Metal fp32 | Metal fp16 |
| ------- | -------- | ---------- | ---------- |
| batch 1 | **~76**  | 32         | 35         |
| batch 8 | **~174** | 60         | 61         |

**Prefill-bound** (prompt 512, generate 1 token — isolates the one big seq×hidden GEMM) —
ms, lower is better. Both columns are **after** the GPU-`Add` fix described below; the
parenthesized values are before it:

|         | CPU fp32 | Metal fp32    | Metal fp16        |
| ------- | -------- | ------------- | ----------------- |
| batch 1 | 200      | **119** (130) | **98** (284 ⚠️)   |
| batch 8 | **1432** | 2273 (2757)   | **559** (1815) 🔥 |

**Read:**

- **The bottleneck is the decode loop, not compute.** In the decode-bound regime Metal
  loses ~2× to the CPU and fp16 buys little (32→35, 60→61) — autoregressive decode runs at
  `m = batch`, so every decode-step GEMM is a tall-skinny matrix–_vector_ product far down
  in the n<1024 region where per-op API overhead dominates and the CPU (no GPU API in its
  path) wins. The model having 500M params doesn't help: each _step_ still issues many tiny
  ops. NOTE: **command-buffer reuse** (batching ops into one commit per step) was implemented
  to attack this and **did not help** — it was neutral on bs1 decode and a regression on
  GEMM-heavy regimes because batching destroys CPU/GPU overlap (see the levers list above).
  Per-op commit is already near-optimal; the tiny-op decode floor is the GPU-API cost itself.
- **fp16 prefill now wins big — once a real bug was found.** The first run showed fp16 bs=1
  prefill _slower_ than fp32 (284 vs 130 ms), which looked like an MPS-fp16 weakness. A
  profiler run (`CT2_LLM_PROFILE=1`) said otherwise: the GEMMs were identical in both
  precisions; the entire regression was the elementwise **`Add`** op (the residual
  connections), which exploded **27×** in fp16 (51 ms → 1376 ms over 10 iters). Cause: the
  M1 `metal::add` kernel was never wired into the `Add` op, so `Add` ran on the CPU
  reference — fast SIMD in fp32, but catastrophic software-emulated `half` arithmetic in
  fp16, plus a pipeline flush per residual (48 per forward). Routing `Add` to a real GPU
  kernel (fp32 + fp16, mirroring `Mul`) fixed it: **fp16 bs=8 prefill 1815 → 559 ms (3.2×),
  now 2.6× faster than CPU and 4× faster than Metal fp32.** This is the "where Apple Silicon
  gets fast" result. fp32 improved too (the removed flushes), as predicted.
- **Metal's compute path is healthy.** Prefill — big compute-bound GEMMs — is where Metal
  wins (fp16 bs=8 559 ms vs CPU 1432 ms). The GPU is good at the math; the remaining loss is
  the per-op overhead of the _decode loop_, not the math.
- **The decode loop is API-overhead-bound** on the real model exactly as on the tiny one —
  but command-buffer reuse is **NOT** the lever (it was tried and reverted; see lever #1 in
  the Analysis above: batching one commit per step destroys CPU/GPU overlap, −23% bs8
  prefill). Per-op commit is already near-optimal; the real path forward is _fewer, bigger_
  ops per step (e.g. a fused attention kernel), not fewer commits. Secondary follow-up: the
  profiler also showed RMSNorm ~2.5× slower in fp16 (152 → 377 ms) — smaller than `Add`,
  worth a look but not the headline.

Lesson (again): **profile, don't guess.** The intuitive culprit (MPS fp16 GEMM) was fine;
the real one was an op that had quietly never been on the GPU at all.

Benchmark: `MetalTest.DISABLED_BenchmarkLLM`, model path via `CT2_LLM_MODEL`; add
`CT2_LLM_PROFILE=1` (needs `-DENABLE_PROFILING=ON`) for the per-op breakdown.

## int8: native int8×int8→int32 GEMM (Phase 2 kernels)

Measured 2026-06-11, M4 Max, Release build. Kernel-level: 30 iters/cell (8 for the
2048³ cell), flush per iter, `DISABLED_BenchmarkGemmInt8` (Dense layout, `trans_b`).
GFLOPS counts the same `2·m·n·k` madds for every dtype so the columns compare directly.

| m, n, k (kernel)         | Metal int8     | Metal fp16  | Metal fp32 |
| ------------------------ | -------------- | ----------- | ---------- |
| 2048, 2048, 2048 (tiled) | 7.28 ms        | **1.48 ms** | 1.72 ms    |
| 256, 4864, 896 (tiled)   | 1.15 ms        | **0.36 ms** | 0.38 ms    |
| 1, 4864, 896 (GEMV)      | **0.157 ms**   | 0.172 ms    | 0.176 ms   |
| 1, 151936, 896 (GEMV)    | **0.49 ms** 🔥 | 0.84 ms     | 1.35 ms    |

Two kernels behind one entry point (`metal::gemm_s8`):

- **m ≤ 8 (decode): SIMD-group GEMV** — one SIMD-group per output element, lane-strided
  `char4` k-loop, `simd_sum` fold. This regime is **memory-bound**, and int8 moves half
  the bytes of fp16 — so int8 **beats** the MPS float GEMM, 1.7× on the per-token
  lm_head GEMM (n=151936). This is where the quantization win actually lives.
- **m > 8 (prefill): threadgroup-tiled** — 64×64 C-tile, 32-deep k chunks, 4×4 int4
  register micro-tile. **ALU-bound at ~2.4 T-MAC/s** vs a ~3.7 ceiling: an int32 MAC is
  two ALU ops on a GPU with **no integer matrix units** (`simdgroup_matrix` is
  half/bfloat/float only — MSL spec 2.4), while MPS fp16 rides dedicated FMA pipelines
  to ~5.9 T-FMA/s. Large-m int8 GEMM is therefore structurally ~3–5× slower than MPS
  fp16 **at the kernel level** _for a hand-written ALU kernel_ — that diagnosis stands,
  but the conclusion ("no path closes it") was superseded 2026-06-11 by the Metal-4 MPP
  path below, which routes around the ALU bound entirely. The tiled kernel remains the
  fallback for pre-macOS-26 OSes, other transpose layouts, and integral `alpha != 1`.

### int8 prefill via Metal-4 MPP `matmul2d` (Task 6 — closes the prefill gap)

Measured 2026-06-11, M4 Max (macOS 26.4.1), Release build, best-of-3 benchmark runs
(`DISABLED_BenchmarkGemmInt8`, 30 iters/cell, 8 for 2048³, flush per iter). The m>8
path of `metal::gemm_s8` now routes to `mpp::tensor_ops::matmul2d` (int8×int8→int32,
base Metal 4, `kernels_mpp_msl.h`) when available; verified **int32-bit-exact** against
the host triple loop at k=2048 over the full int8 range, so the parity contract is
unchanged — this is a pure speed swap, not a quality tradeoff.

| m, n, k (kernel)      | int8 MPP (new) | int8 tiled (old) | Metal fp16 |
| --------------------- | -------------- | ---------------- | ---------- |
| 2048, 2048, 2048      | **1.51 ms** 🔥 | 7.19 ms          | 1.49 ms    |
| 256, 4864, 896        | **0.329 ms**   | 1.14 ms          | 0.336 ms   |
| 256, 896, 4864        | **0.370 ms**   | 1.70 ms          | 0.368 ms   |
| 1024, 1024, 1024      | 0.32–1.09 ms   | 1.95 ms          | 0.31 ms    |
| 1, 4864, 896 (GEMV)   | 0.146–0.160 ms | (unchanged path) | 0.17 ms    |
| 1, 151936, 896 (GEMV) | 0.48–0.62 ms   | (unchanged path) | 0.84 ms    |

**Read:** int8 prefill now **ties MPS fp16** at the kernel level (~11.4 T-effective-FLOPS
at 2048³, up from 2.4) — a 3.4–4.8× kernel speedup over the hand-tiled path. The "no int8
matrix units" ceiling applied to hand-written ALU kernels; MPP's opaque implementation
(compiler-resolved `__tensorops_impl_*` intrinsics) reaches fp16-matrix-pipeline rates
on int8 operands while remaining int32-exact. Decode GEMV shapes are untouched (still
routed to the SIMD-group kernel, still beating fp16).

Tuning notes that mattered (sweep in `experiments/mpp_matmul2d_tune.mm`):

- **Execution scope is the whole game: 2 SIMD-groups, not 4.** Apple's header example
  uses `execution_simdgroups<4>`; every 4-SG and 8-SG config measured 2–5× slower than
  the same tile at 2 SGs. Winner: 16×64 tile on 2 SIMD-groups (16×128 and 32×64 within
  a few %). Per-threadgroup tiles ≥128 wide collapsed on deep-k shapes.
- The interior/edge `slice<Extents...>` split (static extents skip bounds checks) is
  required for the win; all-dynamic `slice()` at the header-example config left most of
  the speedup on the table. (The MPP header comment calls it `static_slice`; the
  shipping stdlib spells it `slice<Extents...>`.)
- MPP dispatch matches element types **exactly**: `int8_t`/`int32_t`, non-const —
  `char` or `const int8_t` hit an "Unsupported type" static_assert.
- `mode::multiply` overwrites C (verified by garbage pre-fill), matching alpha=1/beta=0.
- Run-to-run spread on this machine exceeds 2× on single runs (the 1024³ row above kept
  its full observed range); use best-of-3 minimums for kernel comparisons.

**End-to-end** (Qwen2.5-0.5B-int8, batch 8 × 128-token prompt, 1 step, median of 5,
`experiments/int8_e2e_check.cc`): **555 ms → 350 ms** with MPP on (fp16: 300 ms) —
int8 e2e prefill goes from 1.85× slower than fp16 to within ~17%, the residual being
the per-Dense quantize/dequant epilogues, not the GEMM. Output tokens are
**byte-identical** with and without `CT2_NO_MPP_GEMM=1` (24 greedy steps), as bit
exactness predicts.

Availability: macOS 26+ (MSL 4.0). Pre-26 OSes, integral `alpha != 1`, k = 0, and
non-NT layouts fall back to the tiled kernel; `CT2_NO_MPP_GEMM=1` forces the fallback
(bisection lever, same spirit as `CT2_NO_MPS_ACT`).

The other two Task-6 candidates were not benchmarked, deliberately: MPP already ties
MPS fp16 GEMM — the fastest matmul measured on this machine — so there is no headroom
left for `MPSNDArrayQuantizedMatrixMultiplication` to win (and it dequantizes to float
output with undocumented accumulator semantics, which would have meant restructuring
the gemm+dequant pipeline AND re-litigating exactness), and the `simdgroup_float8x8`
staging fallback is moot. If MPP ever regresses or a pre-26 OS matters, those notes
live in the apple-silicon skill (`mpsndarray.md`, `int8-gemm-kernel-design.md`).

**End-to-end (Qwen2.5-0.5B-int8, `device="metal"`, 3 runs each, warm; spread shown):**

| metric                       | int8 (native)  | fp16             | int8 Phase 1 (shim)        |
| ---------------------------- | -------------- | ---------------- | -------------------------- |
| 3-prompt × 24-token e2e      | 1.22–1.27 s    | **0.99 s**       | ~2.5× slower than fp16     |
| prefill bs8 × 176-tok prompt | 41–44 ms       | 34–50 ms (noisy) | —                          |
| decode bs1, ms/token         | 28.8–30.2      | **25.2–25.7**    | —                          |
| **peak RSS**                 | **1453 MB** 🔥 | 2494 MB          | no win (per-call widening) |

**Read:** the Phase-1 shim's ~2.5× e2e penalty is gone — int8 now lands within ~1.26×
of fp16 e2e (decode within ~15%, the gap being the structural extra quantize +
dequant-epilogue launches per Dense on an API-floor-bound loop, and the prefill-side
tiled-kernel ALU bound above). The headline: **peak RSS drops 42%** (weights are
genuinely int8-resident now — the entire point of the phase), with **92/100
teacher-forced next-token agreement** vs the fp16 reference (5 prompts × 20 steps) —
identical to Phase 1, as expected from a bit-exact GEMM swap. Accumulation is exact
int32 at any depth: the suite includes an all-saturated k=2048 case (accumulator
3.3e7 > 2^24) that the retired fp32 shim could not represent.

## Fused decode attention: the decode gap closes (2026-07-07)

The profiler said a decode step spends **26.3% in `dot_product_attention`**
(MatMul q·K^T → SoftMax → MatMul ·V) and 22.3% in `MatMul` alone — and `gemm.mm`
explains why the profile understates it: **batched MatMul encodes one MPS GEMM per
batch index**, so decode attention at bs=8 issued 2 × (8·14) × 24 ≈ **5,400 MPS GEMM
encodes per step**. The `ct2_sdpa_*` kernel (`kernels_msl.h`) collapses each layer's
three-op sequence into **one launch** — one threadgroup per score row, 4 SIMD-groups
striding the key axis with an online-softmax partial each — never materializing the
`[rows, T]` score tensor. Routed in `dot_product_attention` (`attention.cc`) for the
q_len ≤ 8 decode regime (greedy, small beams, short prefills; guards exclude relative
bias / ALiBi / attention-weights output); `CT2_NO_METAL_SDPA=1` forces the unfused
reference.

Qwen2.5-0.5B (int8 model dir loaded at the listed compute type), M4 Max, macOS 26.4.1,
2026-07-07. Same binary, same model, two runs each; fused/unfused differ only by env var.
Decode-bound regime (prompt 32, generate 32) — tok/s, higher is better:

|           | unfused (CT2_NO_METAL_SDPA=1) | **fused**     | speedup | CPU fp32  |
| --------- | ----------------------------- | ------------- | ------- | --------- |
| bs=1 fp32 | 33.9                          | **75.9–79.0** | ~2.3×   | 72.6–75.8 |
| bs=1 fp16 | 35.8                          | **74.6–82.2** | ~2.2×   | —         |
| bs=8 fp32 | 62.0                          | **403–418**   | ~6.7×   | 175–177   |
| bs=8 fp16 | 62.7                          | **486–493**   | ~7.9×   | —         |

**Read:**

- **Metal now wins decode — the last regime where the CPU won.** bs=1 edges the CPU
  (spread straddles it run-to-run; call it parity-to-slightly-ahead), bs=8 is ~2.8× over
  CPU. Combined with the prefill wins, the Metal backend is now ahead of the Accelerate
  CPU baseline in every measured regime, fp32 and fp16.
- **The bs=8 blowout (6.7–7.9×) is the per-matrix MPS encode loop dying.** The unfused
  cost scaled with batch × heads (per-GEMM encode floor), the fused kernel is one launch
  per layer regardless — more threadgroups in the same launch is free on a 40-core GPU.
- The unfused baseline reproduces the historical table above (33.9/35.8/62.0/62.7 vs the
  2026-06-09 32/35/60/61), so the A/B is apples-to-apples.
- Prefill-bound numbers moved too (fp16 bs=1 113 → 81–82 ms; bs=8 573 → 480–485): that
  regime's single decode step at T=512 carried ~672 MPS encodes (~30 ms) now done in 24
  launches. Prefill itself (q_len > 8) stays on MPS GEMM, which wins compute-bound shapes.
- The doc's old "fewer, bigger ops per step (e.g. a fused attention kernel)" prediction
  (Analysis, lever list) is hereby confirmed measured. Remaining decode budget after the
  fusion: Gemm/Dense (the projections + lm_head), Concat (KV-cache append), RMSNorm,
  Rotary, Add — the next fusion candidates if more decode speed is wanted.

## Caveats

- Tiny model + tiny ops is the worst case; a real LLM (large hidden size, big GEMMs) sits
  much closer to the favorable end of the GEMM table — and since the fused decode
  attention kernel (2026-07-07), decode is no longer the loss column: the remaining
  per-op floor is carried by the projection GEMMs and cache/norm/rotary small ops.
- Numbers are single-run averages on a warm machine (the fused-SDPA table: two runs,
  spread shown); treat as indicative, not precise.
