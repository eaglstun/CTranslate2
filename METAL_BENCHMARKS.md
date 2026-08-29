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

## Real translation model — OPUS-MT en→de (2026-07-20, current build)

A real ~75M-param Marian encoder-decoder (`Helsinki-NLP/opus-mt-en-de`), converted with
`ct2-transformers-converter` and run through the **Python bindings** (`device="metal"`),
not the gtest rig. This is the current build — fused decode attention, MPS-GEMM cache, async
commits all present — so unlike the historical transliteration row above, **Metal now wins**.

- **Machine:** Apple M4 Max (40-core GPU), 64 GB, macOS 26.5.2
- **Package:** ctranslate2 4.8.1 (whisperX venv, linked to the `~/.local/ct2-metal` dylib)
- **Method:** beam=5, en→de, batch of repeated sentences; median of 5 timed passes after
  2 warm-ups. int8 / int8_float16 quantized at load from the fp32 model. Harness:
  `EasyNMT/ct2-bench/bench.py`.

Throughput in generated tok/s (higher is better), speedup vs CPU-fp32 (Accelerate) in
parens. CPU int8 is unavailable on the Accelerate build (no efficient int8 kernel) — it
`ValueError`s cleanly, as expected.

| batch | CPU fp32 | Metal fp32       | Metal fp16   | Metal int8       | Metal int8_fp16 |
| ----- | -------- | ---------------- | ------------ | ---------------- | --------------- |
| 1     | 163      | 203 (1.25×)      | 183 (1.12×)  | **217 (1.33×)**  | 184 (1.13×)     |
| 8     | 796      | 1037 (1.30×)     | 854 (1.07×)  | **1183 (1.49×)** | 771 (0.97×)     |
| 32    | 1680     | 2463 (1.47×)     | 1638 (0.98×) | 2493 (1.48×)     | 1512 (0.90×)    |
| 64    | 2010     | 3137 (**1.56×**) | 2039 (1.01×) | 2781 (1.38×)     | 1816 (0.90×)    |

**Read:** every Metal fp32/int8 config beats the CPU; the winner _depends on batch_, and
there's a clean crossover:

- **int8 (fp32 accumulation) is the small-batch champion** — 1.33× at bs=1, **1.49× at bs=8**,
  beating even Metal fp32. Decode-bound, skinny-matmul regime: int8 wins by shrinking memory
  traffic (the same "int8 GEMV beats fp16 on decode" effect seen on Qwen).
- **fp32 is the large-batch champion** — 1.56× at bs=64, where the GEMMs are fat enough that
  raw MPS fp32 throughput takes over and int8 (1.38×) falls behind.
- **Crossover ≈ batch 32** — int8 (1.48×) and fp32 (1.47×) tie.
- **fp16 activations are the consistent loser.** Both plain fp16 _and_ int8_float16 sag to
  ~0.90× at large batch — the int8 weights don't rescue int8_float16; the fp16 **activation**
  path is the poison. On a model this small the fp16 GEMM savings are tiny and get eaten by
  fp16↔fp32 conversion churn around the many small non-GEMM / CPU-reference ops. Mirror image
  of the Qwen/Whisper wins, where fp16 has real compute to save.

**Takeaway for small encoder-decoder translation models on Metal: `int8` for decode /
small-batch, `float32` for large-batch prefill, and never an fp16-activation compute type
(`float16` / `int8_float16`) — those are pessimizations here.**

## Bigger translation model — NLLB-200 distilled 600M en→de (2026-07-20)

Same rig and method as OPUS-MT above, ~8× the parameters (`facebook/nllb-200-distilled-600M`,
600M-param multilingual enc-dec; NLLB needs an explicit `eng_Latn`→`deu_Latn` target prefix —
harness `EasyNMT/ct2-bench/bench_nllb.py`). Output verified correct German before timing.
Run counts: bs 1/8 = 5 runs/2 warmup, bs 32 = 4/1, bs 64 = 3/1; all spreads < 5%.

Throughput in generated tok/s, speedup vs CPU-fp32 in parens:

| batch | CPU fp32 | Metal fp32      | Metal fp16  | Metal int8      | Metal int8_fp16 |
| ----- | -------- | --------------- | ----------- | --------------- | --------------- |
| 1     | 27       | **89 (3.26×)**  | 87 (3.21×)  | 79 (2.89×)      | 69 (2.54×)      |
| 8     | 143      | 278 (1.94×)     | 254 (1.77×) | **361 (2.52×)** | 269 (1.88×)     |
| 32    | 289      | 587 (2.03×)     | 438 (1.52×) | **633 (2.19×)** | 414 (1.43×)     |
| 64    | 318      | **696 (2.19×)** | 479 (1.51×) | 683 (2.15×)     | 440 (1.38×)     |

**Read:** the OPUS-MT rules of thumb _shift_ with model size — this is the more important
finding than any single number:

- **Metal wins much harder on the bigger model.** bs=1 Metal fp32 is **3.26×** here vs 1.25×
  on OPUS-MT. A 600M model's decode GEMV is fat enough that the GPU dominates even
  single-stream — no more squeaking past the CPU.
- **int8's sweet spot moved from bs=1 to mid-batch.** On OPUS-MT int8 owned bs=1; here it
  _lags_ at bs=1 (2.89×, since the bigger decode GEMV lets fp32/fp16 throughput win and int8's
  quant overhead isn't amortized) and instead wins the **mid-batch band — bs=8 a clear 2.52×**,
  narrowing to a tie with fp32 by bs=64.
- **fp16-activation types still lose, but the penalty now _grows with batch_** instead of being
  flat: fp16 ties fp32 at bs=1 (no batched activations to churn) then falls to 1.51× vs 2.19×
  at bs=64 — a clean fingerprint of fp16↔fp32 activation-conversion cost scaling with
  activation volume. int8_float16 is the worst config at every batch ≥ 8.

**Combined takeaway (OPUS-MT + NLLB): pick the Metal compute type by _both_ model size and
batch.** Small model → `int8` for decode. Big model → `float32` is the safe default, with
`int8` a real win in the mid-batch serving band (bs≈8–32). fp16-activation types
(`float16` / `int8_float16`) are never the right choice on either — the fp16 GEMM savings
don't cover the activation-conversion churn at translation-model scale.

## Decoder-only LLM — Qwen2.5-0.5B-Instruct decode (2026-07-20) — the pattern INVERTS

Same rig; decoder-only, so `ctranslate2.Generator`. Forced exactly 128 greedy tokens
(`min_length == max_length` suppresses early EOS) so every config does identical decode work;
report decode tok/s. **All configs load one fp32 model and quantize at load via `compute_type`**
— note the `--quantization float16` converter flag silently produced a float32 file for Qwen
(byte-identical to the fp32 dir; it worked correctly for OPUS-MT/NLLB), so load-time
quantization is the trustworthy path here. Harness: `EasyNMT/ct2-bench/bench_qwen.py`. Output
verified coherent before timing. Run counts: bs 1 = 5/2, bs 8/32 = 3/1; spreads < 5%.

Throughput in decode tok/s, speedup vs CPU-fp32 in parens:

| batch | CPU fp32 | Metal fp32  | Metal fp16       | Metal int8  | Metal int8_fp16 |
| ----- | -------- | ----------- | ---------------- | ----------- | --------------- |
| 1     | 78       | 92 (1.17×)  | **96 (1.23×)**   | 83 (1.06×)  | 86 (1.10×)      |
| 8     | 174      | 453 (2.60×) | **629 (3.62×)**  | 356 (2.05×) | 398 (2.29×)     |
| 32    | 472      | 847 (1.79×) | **1210 (2.56×)** | 787 (1.67×) | 1111 (2.36×)    |

**Read: the translation-model ranking flips completely.** On the enc-dec translation models
fp16-activation types were pessimizations and int8 won decode. On decoder-only Qwen:

- **fp16 wins at every batch** (1.23× / 3.62× / 2.56×) — the champion, not the poison.
- **int8 is now the _worst_ option** (1.06× / 2.05× / 1.67×) — the exact mirror of translation.
- **int8_float16 beats plain int8** — fp16 activations _help_ here, opposite of translation.

**Why the inversion — it points back at what the backend was tuned for.** Qwen is the model
the Metal path was optimized around: its hot decode ops — the fused single-launch SDPA (M16),
RMSNorm, RoPE — are all **fp16-native GPU kernels**, so fp16 activations stay on-device and pay
no conversion tax; fp16 then wins purely by halving weight/activation memory traffic in the
GEMV-bound decode loop. The Marian/M2M translation models route more ops through the
CPU-reference path, so _there_ fp16 activations trigger the fp16↔fp32 churn that sinks them.
Same backend, opposite verdict, decided entirely by **which ops have fused fp16 kernels.**

> **Nuance vs the int8 work:** this end-to-end result (fp16 > int8 on Qwen decode) refines the
> earlier _kernel-level_ finding that "int8 GEMV beats fp16 on decode" — true in isolation, but
> the full decode loop with its surrounding fused-fp16 ops lands on fp16. Different measurement,
> both correct.

### Qwen prefill — fp16 wins the GEMM-bound regime too

Prefill (long prompt, generate exactly 1 token; throughput = prompt*tokens·batch / sec) is the
compute-bound counterpart to decode — the regime where the \_translation* models flipped to
`float32`. On Qwen it does not flip: **fp16 wins prefill as well, and its lead grows with prompt
length.** Harness: `bench_qwen_prefill.py`, 5 runs / 2 warmup, spreads < 5%.

| workload           | CPU fp32 | Metal fp32   | Metal fp16       | Metal int8   | Metal int8_fp16 |
| ------------------ | -------- | ------------ | ---------------- | ------------ | --------------- |
| prefill 512, bs=1  | 2558     | 5915 (2.31×) | **7159 (2.80×)** | 5634 (2.20×) | 6456 (2.52×)    |
| prefill 512, bs=8  | 2921     | 7323 (2.51×) | **9466 (3.24×)** | 7143 (2.45×) | 8404 (2.88×)    |
| prefill 1024, bs=1 | 2258     | 6232 (2.76×) | **8078 (3.58×)** | 6022 (2.67×) | 7319 (3.24×)    |

Ranking is identical to decode — **fp16 > int8_fp16 > fp32 > int8** — fp16-activation on top,
weight precision secondary. The fp16 lead _widening_ with prompt length (2.80× → 3.58× as the
prompt goes 512 → 1024) is the tell: as prefill gets more GEMM-bound, the fp16-native fused
kernels pull _further_ ahead of fp32, not closer. **For a decoder-only LLM there is no regime —
decode or prefill — where you'd pick anything but `float16`.** This is the sharp contrast with
the enc-dec translation models, whose prefill flips to `float32`.

**Master takeaway across all three models: there is no global-best Metal compute type — it is
set by model _architecture_ first, then size and batch.** Decoder-only LLM → `float16` (both
decode _and_ prefill).
Small enc-dec translation → `int8` (decode) / `float32` (big batch). Large enc-dec → `float32`
default, `int8` mid-batch. The deciding factor is whether a model's hot ops have fused fp16
GPU kernels (LLM: yes → fp16 wins; classic enc-dec: partly → fp16 activations churn).

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

## Experimental float GEMV at m=1 (`DISABLED_BenchmarkMpsGemv`)

Measured 2026-08-28, M4 Max, Release build. `MPSMatrixVectorMultiplication` is a
dedicated alternative to expressing decode's `1×K · K×N` projections as a degenerate
`MPSMatrixMultiplication`. The implementation caches the GEMV object by
`{rows, columns, transpose, alpha, beta}` and preserves the per-op async commit model.
It is opt-in with `CT2_MPS_GEMV=1`; the decoder-only model gate below did not justify
making it the default.

The benchmark reports the best of four repetitions in one process. Values below are
flush-per-iteration milliseconds from two complete benchmark invocations; ranges show
the two minima rather than hiding machine variance.

| Dense shape (m=1) | fp16 GEMM   | fp16 GEMV       | fp32 GEMM   | fp32 GEMV       |
| ----------------- | ----------- | --------------- | ----------- | --------------- |
| n=896, k=896      | 0.146–0.156 | **0.129–0.134** | 0.133–0.142 | **0.102–0.107** |
| n=2688, k=896     | 0.126–0.135 | **0.113–0.118** | 0.129–0.140 | **0.118–0.125** |
| n=4864, k=896     | 0.135–0.138 | **0.112–0.122** | 0.150–0.172 | **0.126–0.137** |
| n=896, k=4864     | 0.137–0.147 | **0.110–0.112** | 0.148–0.179 | **0.123–0.124** |
| n=151936, k=896   | 0.686–0.689 | **0.638–0.657** | noisy       | noisy           |

**Read:** the dedicated kernel is a repeatable win on the per-layer projection shapes,
roughly 7–32% in this probe. The vocabulary projection is much closer; its fp32
flush-per-iteration cell had a large outlier in one invocation, while batched timings
were near parity (~1.17–1.21 ms), so no fp32 vocabulary claim is made.

The included translation benchmark (batch 32, three separate warm processes) moved by
median from **32.13→30.42 ms fp32** and **30.25→29.07 ms fp16** with the route enabled.
That encouraging result did not survive a real decoder-only model gate.

`MetalTest.DISABLED_BenchmarkLLMMpsGemv` loads Qwen2.5-0.5B once per precision, warms it,
then averages five forced 128-token batch-1 decodes. Two alternating baseline/GEMV
process pairs gave:

| Qwen decode | baseline range     | MPS GEMV range     | best-of-two result  |
| ----------- | ------------------ | ------------------ | ------------------- |
| fp32        | 3465.59–3715.32 ms | 3462.84–3615.11 ms | **parity** (−0.08%) |
| fp16        | 2614.33–2683.85 ms | 2661.07–2811.27 ms | **1.8% slower**     |

**Decision:** keep the route opt-in and deprioritized. The isolated 7–32% projection
wins are too small a share of the full decode loop to improve throughput, and fp16—the
preferred Qwen compute type—regresses slightly. This is another example of why a kernel
microbenchmark is a filter, not the promotion gate.

Direct fp32/fp16 parity, both matrix orientations, and nonzero alpha/beta are covered by
`MetalTest.MpsGemvMatchesHostReference`; all 31 Metal tests pass with the route enabled.
The real Qwen greedy-decode parity gate also matches CPU for all 24/24 tokens in fp32 and
fp16 with `CT2_MPS_GEMV=1`.

### Rejected follow-up: combine the two KV-cache Concat copies

The Qwen fp16 decode profile put `Concat` at 10.9%, and each two-input Metal Concat
encoded the old-cache and new-token copies in separate command buffers. Two implementations
were tested on 2026-08-28 and removed:

- One branch-selecting byte kernel copying both inputs in a single grid regressed the
  focused 128-token decode substantially; per-byte division/source selection erased the
  launch saving.
- Two existing strided-copy dispatches in one encoder preserved the efficient copy kernel
  but still failed the model gate: fp32 was flat and fp16 regressed. As with whole-step
  command-buffer reuse, combining commits interfered with the scheduling/overlap that the
  current per-copy path gets for free.

Both variants passed all six Metal Concat parity cases, including empty inputs and all
three concat axes. The failure was throughput, not correctness. Do not retry command-buffer
coalescing here; a future cache win needs a layout that can append in place without copying
the old history (for example a capacity-strided or paged cache), which is a state-layout
change rather than a Concat-kernel tweak.

### Implemented follow-up: capacity-strided append-in-place

That state-layout change was implemented on 2026-08-28 for the fused Metal decode path.
The cache is physically `[batch, heads, capacity, depth]`, begins at 64 timesteps, and
doubles geometrically. Logical length comes from the decoder offset. A common step writes
only the new K/V suffix with one byte-copy kernel; a growth step copies the live prefix and
appends the suffix into the new allocation in one launch. Fused SDPA consumes an explicit
K/V batch-head stride, so unused capacity never needs to be packed. Unsupported attention
modes retain the contiguous Concat path, with a compact-to-contiguous transition if a
caller switches modes. `CT2_NO_METAL_KV_CACHE=1` is the A/B escape hatch.

Focused Qwen2.5-0.5B batch-1 decode (prompt 32, forced 128 tokens, warmup then average of
five; same binary, two alternating cache-enabled/cache-disabled process pairs):

| compute | Concat-cache range | capacity-strided range | matched-pair improvement |
| ------- | ------------------ | ---------------------- | ------------------------ |
| fp32    | 3300.17–3612.89 ms | **2951.25–3218.07 ms** | **10.6–10.9%**           |
| fp16    | 2291.38–2425.06 ms | **2095.58–2310.78 ms** | **4.7–8.5%**             |

The real-model correctness gate matched CPU for all 24/24 greedy tokens in fp32 and fp16.
Primitive coverage verifies in-place append, grow, compact-to-contiguous, and SDPA over a
capacity stride. All 33 Metal tests pass; the full suite remains 299 passed / 3 skipped /
1 pre-existing CPU quantized-grouped-Conv1D failure.

### Post-M18 decode profile: cache copying is gone; two fusion targets remain

Re-profiled 2026-08-28 after capacity-strided append, using Qwen2.5-0.5B, prompt 32,
128 forced greedy tokens, and one warmup plus one profiled generation per cell. The
Release build had `ENABLE_PROFILING=ON`. Each cell ran in a separate process through the
`CT2_LLM_PROFILE=post_m18` mode in `MetalTest.DISABLED_BenchmarkLLM`.

The integrated profiler synchronizes Metal at every nested scope, so these percentages
rank work but do **not** predict an end-to-end speedup. In particular, parent scopes report
self-time after their named children are subtracted. `MultiHeadAttention self` is the
unnamed layout/cache/orchestration remainder around its profiled Dense, norm, RoPE, Split,
and attention children.

| compute / batch | GEMM | Quantize | Dequant GEMM epilogue | RMSNorm | RoPE | Add | fused SDPA | Split | MHA self |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fp32 / 1 | 39.3% | — | — | 9.6% | 8.8% | 8.5% | 7.7% | 6.0% | 5.4% |
| fp16 / 1 | 35.9% | — | — | 10.2% | 9.4% | 9.1% | 8.1% | 6.1% | 5.6% |
| int8 / 1 | 25.1% | 18.1% | 17.2% | 7.1% | 6.6% | 6.3% | 5.7% | 4.3% | 3.9% |
| fp32 / 8 | 48.5% | — | — | 8.2% | 7.5% | 6.8% | 5.5% | 4.7% | 5.3% |
| fp16 / 8 | 39.4% | — | — | 9.1% | 9.0% | 8.1% | 7.2% | 5.6% | 5.6% |
| int8 / 8 | 38.9% | 14.1% | 14.0% | 5.9% | 5.6% | 5.0% | 3.8% | 3.5% | 3.6% |

**Read:**

- The old cache `Concat` hotspot is absent. The remaining natural attention-side fusion
  boundary is QKV post-processing: `Split + RoPE + MHA self` is **17–21%** of fp32/fp16
  profile time. At one-token decode, a kernel can plausibly consume the fused QKV
  projection, rotate Q/K, emit Q, and write rotated K plus V directly into the
  capacity-strided cache. That attacks several launches and memory passes without batching
  command buffers.
- Float projection GEMMs remain the single largest bucket (36–48%). The isolated MPS GEMV
  experiment already failed its end-to-end gate, so this profile is not evidence to turn
  it on; a useful GEMM change needs a broader fusion boundary, not another API swap.
- int8 has a different bottleneck. `Quantize + GEMM + DequantizeGemmOutput` is **60.3% at
  batch 1 and 67.0% at batch 8**. The whole diagnostic process issued 169,550 command
  buffers for int8 versus 118,986 for fp32/fp16, about **42% more**. Fusing the small-m
  int8 GEMV's dequant/bias/activation epilogue is now the best concrete int8 kernel target;
  the 3B-scale gate still decides whether it matters enough end-to-end.
- Batch 8 increases the residual attention-side MPS `MatMul` share (0.3%→1.4–2.1%), but
  fused SDPA prevents the old batch×heads encode explosion. The overall ranking otherwise
  stays stable across batch sizes.

An attempted unprofiled rerun was discarded: a concurrent Simulator workload slowed the
first fp32 batch-1 cell to 30.96s versus the clean M18 range of 2.95–3.22s. That is machine
contention, not a recorded regression. Repeat the steady-state matrix on an idle GPU before
quoting new throughput; the op-ranking tables above completed and are the result of this
profiling session.

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
  attention and cache work: Gemm/Dense (the projections + lm_head), RMSNorm, Rotary, and
  Add — the next fusion candidates if more decode speed is wanted.

## Whisper beam-search decode: two kernels close an 8× gap (2026-07-17)

Whisper large-v3, 30s clip, faster-whisper `transcribe(beam_size=5)` via the whisperX
rig (`bench/bench_ct2.py`), M4 Max, macOS 26.4.1, 2026-07-17. Post-M16 baseline was
**39.2s** (0.76× realtime) vs CPU fp32 20.2s — Metal lost by 2×. Root-causing with the
new env-gated instrumentation (`CT2_AUTO_PROFILE`, `CT2_METAL_STATS` — see
`METAL_BACKEND.md`) found the entire gap in two kernels nobody had profiled:

1. **`Transpose` had no Metal routing** → CPU reference → `metal::flush()` per call.
   Beam search folds the beam into `split_heads`/`combine_heads`' time axis
   (`time == beam = 5 ≠ 1`), so decode hit the transpose path ~4×/layer × 32 layers ≈
   **128 full GPU queue drains per token** (~12,200 flushes/run, 36.8s total stall).
   Greedy LLM decode never sees this (`time==1` reshape fast-path) — which is why the
   Qwen benchmarks couldn't catch it. Fix: `ct2_transpose_b1/b2/b4` (generic rank ≤ 4
   permute, dispatched on element width, `transpose.cc` routes before generic dispatch).
2. **`ct2_gather_bytes` copied each gathered row with ONE thread in a serial byte
   loop.** The per-step beam KV-cache reorder gathers a handful of ~1 MB rows (32
   layers × 2 tensors × ~5 beams), so the GPU spent the decode copying megabytes with
   5 threads. With per-op sync, Gather measured **94.4s (80%)** of a 115s run. Fix:
   2D grid — one thread per 16-byte chunk per row.

Same binary flow (rebuild + `cmake --install` into the whisperX venv), transcript
byte-identical to CPU in every config (segments, timestamps, text):

| large-v3, 30s clip | before (2026-07-07 binary) | **after**                 | vs CPU          |
| ------------------ | -------------------------- | ------------------------- | --------------- |
| metal fp16         | 39.2s (0.76×RT)            | **4.6–4.7s (6.3–6.5×RT)** | **4.3× faster** |
| metal fp32         | —                          | 5.7s (5.3×RT)             | 3.5× faster     |
| metal int8         | —                          | 5.8s (5.2×RT)             | 3.5× faster     |
| cpu fp32           | 20.2s (1.49×RT)            | 20.2s (1.49×RT)           | baseline        |

Full 730s file, metal fp16: **206s (3.54× realtime)**, 5.1 GB peak RSS, completes
clean — vs 0.41× RT at bring-up (2026-06-09), an ~8.6× end-to-end improvement.

**Read:**

- The residual flush stats after the fix: 3,186 flushes/run (~26/token: sampler D2H,
  timestamp rules, beam bookkeeping), 803 stalled, **0.7s** total wait (was 36.8s).
- Qwen2.5-0.5B rechecked on the same binary: no regression — bs=1 within the M16
  spread (74.6/79.2 tok/s fp32/fp16), bs=8 slightly better (426.8/509.6 vs 403–418/
  486–493), prefill fp16 bs=1 86→73 ms (prefill transposes now native).
- The previously-planned levers (GEMM-epilogue fusion, on-device beam step, ICB
  replay, GPU Conv1D) are all deprioritized: encoder ≈ 3% of the run, and the per-op
  encode floor priced at ~30ms/token CPU-side against ~250ms of GPU-visible work that
  turned out to be the gather. Re-profile before building any of them.

## Caveats

- Tiny model + tiny ops is the worst case; a real LLM (large hidden size, big GEMMs) sits
  much closer to the favorable end of the GEMM table — and since the fused decode
  attention kernel (2026-07-07), decode is no longer the loss column: the remaining
  per-op floor is carried by the projection GEMMs and cache/norm/rotary small ops.
- Numbers are single-run averages on a warm machine (the fused-SDPA table: two runs,
  spread shown); treat as indicative, not precise.
