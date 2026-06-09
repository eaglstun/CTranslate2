# Metal Backend — Benchmarks

Initial performance measurements for the CTranslate2 Apple Metal backend. These are a
**correctness-first baseline**, not an optimized result — the backend currently commits a
command buffer and `waitUntilCompleted` per op, which is the dominant cost for small ops
(see Analysis).

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

ms per batch (lower is better):

|            | ms/batch |
| ---------- | -------- |
| CPU fp32   | 14.5     |
| Metal fp32 | 2419     |
| Metal fp16 | 1493     |

**Read:** Metal is dramatically _slower_ here — and this is expected and informative. The
test model is tiny (a char-level transliteration Transformer) whose decode loop issues
many hundreds of _small_ ops. Every op currently does a synchronous GPU round-trip
(commit + block-until-complete), so the workload is pure latency overhead, not compute.
The CPU runs the same tiny ops with no dispatch barrier and wins by ~100×.

## Analysis

The two results tell one story: **the kernels are fast, the dispatch model is not.**

- Raw compute is healthy — MPS fp16 GEMM hits ~12 TFLOPS at n=2048 and beats the CPU
  by 3.7×. The math is on the GPU and it's quick.
- The bottleneck is **per-op synchronization**. Each op commits its own command buffer
  and calls `waitUntilCompleted`, forcing a CPU↔GPU round-trip per op. For large GEMMs
  the compute amortizes it; for a decode loop full of tiny ops it's ~all overhead.

This is a deliberate correctness-first baseline (M1–M10 prioritized "provably correct on
real models"). The performance work is well-scoped and independent of correctness:

1. **Batch command buffers** — encode many ops into one command buffer and synchronize
   once per layer/step instead of once per op. This is the single highest-impact change
   for end-to-end latency.
2. **Don't block per op** — only `waitUntilCompleted` when the CPU actually needs the
   data (e.g. before sampling reads logits). Let the GPU pipeline ops.
3. **Reuse MPS objects / pipeline state** — avoid allocating a new
   `MPSMatrixMultiplication` per GEMM call.
4. **Offline `.metallib`** — removes the first-use shader-compile cost.

Expectation after batching: the end-to-end gap should close sharply (the per-op barrier
is the 100× factor), and larger models — where ops are bigger and the GEMM share is
higher — should show the GPU/fp16 advantage that the n≥1024 GEMM numbers already prove.

## Caveats

- Tiny model + tiny ops is the worst case for the current design; a real LLM (large
  hidden size, big GEMMs) sits much closer to the favorable end of the GEMM table.
- Numbers are single-run averages on a warm machine; treat as indicative, not precise.
