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

ms per batch (lower is better). "Sync per op" is the original commit+wait design; "Batched"
removes the per-op block (commit asynchronously, flush only before a CPU read).

|            | sync per op | batched  | CPU baseline |
| ---------- | ----------- | -------- | ------------ |
| Metal fp32 | 2419        | **2015** | 14           |
| Metal fp16 | 1493        | **1183** | 14           |

**Read:** Metal is dramatically _slower_ here — expected and informative. The test model is
tiny (a char-level transliteration Transformer) whose decode loop issues many hundreds of
_small_ ops. Batching removed the per-op block and helped ~17–21%, but did not close the
gap, for two reasons covered in Analysis. The CPU runs the same tiny ops with no GPU API in
the way and wins by ~100×.

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

Remaining performance levers, in rough priority:

1. **Graduate the per-step CPU-reference ops to GPU kernels** (Concat/Split for the KV
   cache first) so a decode step batches without intermediate flushes.
2. **Reuse MPS objects / pipeline state** — avoid allocating a new
   `MPSMatrixMultiplication` per GEMM call; cache encoders.
3. **Offline `.metallib`** — removes the first-use shader-compile cost.

The GEMM table already proves the upside: a real LLM (large hidden size, GEMM-dominated,
fewer/bigger ops) sits at the favorable end and should show the fp16 advantage. This tiny
model is the worst case for any GPU backend.

## Caveats

- Tiny model + tiny ops is the worst case; a real LLM (large hidden size, big GEMMs) sits
  much closer to the favorable end of the GEMM table.
- Numbers are single-run averages on a warm machine; treat as indicative, not precise.
