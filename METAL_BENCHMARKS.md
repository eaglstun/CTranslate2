# Metal Backend — Benchmarks

Performance measurements for the CTranslate2 Apple Metal backend, tracking the optimization
journey from the first **correctness-first baseline** (which committed a command buffer and
`waitUntilCompleted` per op — the dominant cost for small ops) through async command-buffer
batching, MPS-GEMM object caching, and the GPU `Add`/fp16 fix, to a real-LLM evaluation. The
headline finding (see the Qwen2.5-0.5B section and Analysis): Metal **wins** the GEMM-heavy
prefill but **loses** tiny-op autoregressive decode to a fundamental per-op GPU-API floor —
and command-buffer reuse, the long-assumed fix for that floor, was implemented and measured
neutral-to-negative (it kills CPU/GPU overlap).

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
| 1024 | 3053                  | 2139             | 5158             | 1.69×             |
| 2048 | 3231                  | 7711             | 11966            | **3.70×**         |

ms/iter at n=2048: CPU 5.32, Metal fp32 2.23, Metal fp16 1.44.

**Read:** the GPU wins at scale. By n=1024 Metal fp16 overtakes the CPU; at n=2048 it is
**3.7× faster** than Accelerate and fp16 is ~1.5× the fp32 GPU rate. Below n≈1024 the
per-call dispatch overhead (command buffer commit + `waitUntilCompleted` + a fresh
`MPSMatrixMultiplication` object per call) dominates and the CPU wins easily.

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
- **Command-buffer reuse remains the #1 lever for decode**, confirmed by a real model: the
  decode loop is API-overhead-bound exactly as on the tiny model. Secondary follow-up: the
  profiler also showed RMSNorm ~2.5× slower in fp16 (152 → 377 ms) — smaller than `Add`,
  worth a look but not the headline.

Lesson (again): **profile, don't guess.** The intuitive culprit (MPS fp16 GEMM) was fine;
the real one was an op that had quietly never been on the GPU at all.

Benchmark: `MetalTest.DISABLED_BenchmarkLLM`, model path via `CT2_LLM_MODEL`; add
`CT2_LLM_PROFILE=1` (needs `-DENABLE_PROFILING=ON`) for the per-op breakdown.

## Caveats

- Tiny model + tiny ops is the worst case; a real LLM (large hidden size, big GEMMs) sits
  much closer to the favorable end of the GEMM table — but autoregressive **decode** is
  still tiny-op territory (batch-sized GEMMs), so the per-op-overhead bottleneck persists
  until command buffers are reused. Prefill (big GEMM) already wins at batch 1.
- Numbers are single-run averages on a warm machine; treat as indicative, not precise.
