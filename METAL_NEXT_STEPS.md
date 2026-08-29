# Metal Backend — Ranked Next Work

Status as of 2026-08-29, after M19 (fused QKV post-processing). This is the active,
ordered backlog. Completed investigations stay in `METAL_BENCHMARKS.md`; shipped features
stay in `METAL_BACKEND.md`; closed project plans are design records, not competing TODO
lists.

Two env-gated profilers are already in-tree: `CT2_AUTO_PROFILE=cpu|metal` and
`CT2_METAL_STATS=1`. Important A/B switches are `CT2_NO_METAL_SDPA=1`,
`CT2_NO_METAL_KV_CACHE=1`, `CT2_NO_METAL_QKV_FUSION=1`,
`CT2_NO_METAL_SAMPLING=1`, `CT2_NO_MPP_GEMM=1`, and `CT2_MPS_GEMV=1`
(experimental and off by default).

## Ranked list

1. **Close the classic encoder-decoder fp16 coverage/performance gap.** Qwen and Whisper
   run end-to-end in fp16, but the July OPUS-MT/NLLB sweep still found fp16 activation
   conversion churn and recommended fp32 or int8. Re-profile NLLB and OPUS-MT and graduate
   the highest-cost CPU-reference op actually observed. `Tile` is the leading architectural
   candidate because beam search replicates encoder state, but it is a hypothesis, not a
   conclusion. Other currently unrouted candidates include `Sub`, `Mean`, `Sum`, `MinMax`,
   and `AlibiAdd`. Success means fp16 is no longer a pessimization on at least NLLB batch 8+
   without regressing Qwen or Whisper.

2. **Fuse the int8 small-m GEMV epilogue, then validate it at 3B scale.** The post-M18
   profile explains the 0.5B anomaly: Quantize + GEMM + DequantizeGemmOutput consumes
   60–67% and the int8 diagnostic process issues about 42% more command buffers than float.
   Extend the small-m int8 GEMV to write fp16/fp32 output with scale, bias, and activation
   directly, eliminating the int32 output pass and dequant epilogue launch. Keep the exact
   int32-accumulation oracle. Then profile a 3B model (and 7B if memory permits) to establish
   the real size/batch crossover; do not generalize from 0.5B alone.

3. **Extend the capacity cache only where a real workload currently falls back.** The M18
   route intentionally excludes sliding-window attention, relative bias/ALiBi, attention
   output, FlashAttention, and merged-MQA layouts. Add one mode at a time only after a model
   or profile proves it hot. Sliding-window support is the most promising: a ring or paged
   layout can bound memory and avoid periodic compaction, but it also changes masking and
   logical-to-physical indexing and therefore needs dedicated wraparound and beam-reorder
   tests.

4. **Move shader compilation off the first inference.** Package an offline `.metallib`,
   locate it relative to the loaded library, and retain source compilation as a compatibility
   fallback. Measure cold model-load plus first-token latency; this is a startup win, not a
   steady-state throughput claim.

5. **Scope native bf16 as a separate compatibility project.** First inventory MPS/custom
   kernel dtype support and the minimum Apple GPU/macOS target, then prove load, GEMM,
   normalization, attention, and output parity on a model whose native weights are bf16.
   Do not advertise bf16 until a complete model path exists. AWQ int4 and multi-GPU remain
   lower priority than this.

## Next experiment to run

The immediate next action is item 1's NLLB/OPUS-MT profile. Start with fp32 and fp16 at
batch 1 and 8, use `CT2_AUTO_PROFILE=cpu|metal` to count conversion islands, and graduate
the highest-cost CPU-reference op the profile actually identifies. Do not implement `Tile`
solely from the architectural hypothesis.

Minimum matrix:

| Workload                | Compute    | Batch | Purpose                                                         |
| ----------------------- | ---------- | ----: | --------------------------------------------------------------- |
| NLLB-200 distilled 600M | fp32, fp16 |  1, 8 | identify fp16 conversion and fallback hotspots                  |
| OPUS-MT                 | fp32, fp16 |  1, 8 | distinguish shared encoder-decoder costs from NLLB-specific ops |

For every optimization: add direct fp32/fp16 parity where applicable, run the full Metal
suite, run a real-model token parity gate, and require a repeatable end-to-end win before
enabling it by default.

## Do not reopen without new evidence

- Whole-step command-buffer reuse: neutral at batch 1 and regressed GEMM-heavy workloads.
- Combining the two old Concat copies: parity-correct but slower; M18 replaced the layout.
- Defaulting to `MPSMatrixVectorMultiplication`: isolated GEMV wins did not survive the
  Qwen gate, so `CT2_MPS_GEMV=1` remains experimental.
- Reopening fused QKV dispatch tuning without a regression: M19's matched matrix was
  positive in every cell, so the route is default-on and the next profile owns priority.
- GPU Conv1D as a Whisper priority: the M17 profile put the whole encoder near 3% of the
  transcribe run.
- On-device beam selection or ICB replay for the old Whisper gap: M17 showed Gather and
  Transpose, not those mechanisms, caused it.

## Current coverage snapshot

Full fp16 model paths are proven for decoder-only Qwen and encoder-decoder Whisper. Native
Metal routes cover GEMM/MatMul, last-axis norms, RoPE, Gather, rank-≤4 Transpose for 1/2/4
byte elements, BiasAdd/activations, float Add/Mul, Concat/Split/Slide, int8 quantize/GEMM/
dequantize, sampling, fused decode SDPA, and capacity-strided KV append. The CPU-reference
fallback remains valid for fp32; an ungraduated fp16 op needs either a native half kernel or
an explicit fp16→fp32→fp16 compatibility island.
