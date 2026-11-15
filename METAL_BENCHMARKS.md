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

| compute / batch |  GEMM | Quantize | Dequant GEMM epilogue | RMSNorm | RoPE |  Add | fused SDPA | Split | MHA self |
| --------------- | ----: | -------: | --------------------: | ------: | ---: | ---: | ---------: | ----: | -------: |
| fp32 / 1        | 39.3% |        — |                     — |    9.6% | 8.8% | 8.5% |       7.7% |  6.0% |     5.4% |
| fp16 / 1        | 35.9% |        — |                     — |   10.2% | 9.4% | 9.1% |       8.1% |  6.1% |     5.6% |
| int8 / 1        | 25.1% |    18.1% |                 17.2% |    7.1% | 6.6% | 6.3% |       5.7% |  4.3% |     3.9% |
| fp32 / 8        | 48.5% |        — |                     — |    8.2% | 7.5% | 6.8% |       5.5% |  4.7% |     5.3% |
| fp16 / 8        | 39.4% |        — |                     — |    9.1% | 9.0% | 8.1% |       7.2% |  5.6% |     5.6% |
| int8 / 8        | 38.9% |    14.1% |                 14.0% |    5.9% | 5.6% | 5.0% |       3.8% |  3.5% |     3.6% |

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

### M19: fused one-token QKV post-processing is a repeated end-to-end win

The post-M18 profile's attention fusion was implemented as one kernel that consumes the
flat QKV projection, rotates Q/K, emits Q, and writes K/V directly into the
capacity-strided cache. It replaces Split, two Rotary launches, and the separate cache
append/grow path. `CT2_NO_METAL_QKV_FUSION=1` restores the old sequence for A/B.

Qwen2.5-0.5B, prompt 32 / decode 128, Release build on M4 Max. Each reported process
warms once and averages three timed generations. Two matched off/on process pairs were
run per cell in opposite order; fp16 batch 1 received five pairs because it contradicted
the earlier single negative probe.

| Compute / batch | Separate path (ms) |  Fused path (ms) | Matched improvement |
| --------------- | -----------------: | ---------------: | ------------------: |
| fp32 / 1        |   2520.80, 2445.95 | 2303.42, 2276.36 |        **6.9–8.6%** |
| fp32 / 8        |   4057.52, 3860.60 | 3751.17, 3755.97 |        **2.7–7.6%** |
| fp16 / 1        |    1867.10–1947.52 |  1424.58–1545.04 |      **20.7–25.9%** |
| fp16 / 8        |   2318.99, 2314.72 | 2033.41, 1980.99 |      **12.3–14.4%** |
| int8 / 1        |   1727.42, 1729.62 | 1368.88, 1352.59 |      **20.8–21.8%** |
| int8 / 8        |   5333.09, 4211.86 | 3927.51, 4084.20 |       **3.0–26.4%** |

The int8 batch-8 cell was noisy, but reversing process order preserved a positive lower
bound. Direct append/grow fp32/fp16 parity passed, and the real-model decode gate produced
24/24 tokens identical to CPU for both Metal precisions. The repeated matrix therefore
clears the default-enable bar; no speculative dispatch tuning is warranted before the
next model-level profile.

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

## Head-to-head vs upstream `Device::MPS` (2026-11-13)

Upstream merged its own Apple GPU backend (`665e3fd0`, OpenNMT/CTranslate2#2077,
`-DWITH_MPS=ON`, `device="mps"`). This compares that build against this fork's `main`
(`c550401e`) on the same models and harnesses.

**Setup:** Apple M4 Max, 64 GB. Both libraries were built fresh, Release, with Accelerate,
`OPENMP_RUNTIME=NONE` and `ENABLE_PROFILING=OFF`, into separate prefixes and venvs. Inputs
come from `EasyNMT/ct2-bench` (OPUS-MT and NLLB use beam 5; Qwen generates 128 greedy
tokens). Each cell is 3 warmups plus 7 timed runs, and the median is shown. Two noisy
cells (marked †) were re-run 3 × 15 runs; all three re-runs agreed within 1%. Output
token sequences were hashed for parity.

**Upstream remaps compute types:** on MPS, `int8` and `int8_float16` resolve to
`float16`, so those MPS rows are fp16. Upstream's only native int8 path is `int8_float32`.

| Model / batch     | Best Metal (this fork) | Best MPS (upstream) | Winner         |
| ----------------- | ---------------------- | ------------------- | -------------- |
| OPUS-MT bs=1      | 279 (fp32)             | 294 (fp16)          | ≈ tie, MPS +5% |
| OPUS-MT bs=8      | 1048 (int8)            | 1378 (fp16)         | **MPS +31%**   |
| NLLB-600M bs=1    | 92 (fp16)              | 95 (fp16)           | tie            |
| NLLB-600M bs=8    | 318 (int8)             | 336 (fp16)          | MPS +6%        |
| Qwen2.5-0.5B bs=1 | 176† (fp16)            | 158† (fp16)         | **Metal +11%** |
| Qwen2.5-0.5B bs=8 | 1214 (fp16)            | 961 (fp16)          | **Metal +26%** |

Same-compute-type comparison (tok/s, Metal / MPS):

| Model / batch | fp32           | fp16             | int8_float32 (both native) |
| ------------- | -------------- | ---------------- | -------------------------- |
| OPUS bs=1     | 279 / 255      | 266 / 294        | 252 / 216                  |
| OPUS bs=8     | 881 / **1293** | 790† / **1280†** | 1048 / 1010                |
| NLLB bs=1     | **90 / 65**    | 92 / 95          | **72 / 53**                |
| NLLB bs=8     | 261 / 246      | 251 / **336**    | **318 / 182**              |
| Qwen bs=1     | 102 / 71       | **176† / 158†**  | **169 / 56**               |
| Qwen bs=8     | **857 / 656**  | **1214 / 961**   | **550 / 416**              |

CPU fp32 baselines on the same build: OPUS 164 / 824, NLLB 27 / 145, Qwen 85 / 167
(bs=1 / bs=8). MPS bf16 on Qwen measured 125 (bs=1) and 435 (bs=8), slower than its fp16.

### Whisper large-v3 (faster-whisper 1.2.0, beam 5, `whisperX/audio/clip30.wav`)

Each cell has 1 warmup and 5 timed runs (CPU: 3), measuring seconds for the 30 s clip
(lower is better). Both backends use the same `whisper_driver.py` call as
`whisperX/bench/bench_ct2.py`. The 730 s `westvet.m4a` file was no longer in the rig, so
long-form audio was not measured.

| Compute type                                 | Metal (this fork)               | MPS (upstream)                  | Winner                          |
| -------------------------------------------- | ------------------------------- | ------------------------------- | ------------------------------- |
| float16                                      | 4.63 s (4.60–4.67), 6.5× RT     | **3.74 s** (3.70–3.76), 8.0× RT | **MPS 1.24×**                   |
| float32                                      | 5.80 s (5.75–5.82)              | **4.47 s** (4.40–4.48)          | **MPS 1.30×**                   |
| int8_float32 (both native)                   | **5.67 s**, transcript drifts ‡ | 11.33 s, transcript exact       | Metal on speed, MPS on accuracy |
| `int8` (Metal → int8_float16, MPS → float16) | 5.87 s                          | 4.00 s                          | MPS (fp16 vs int8)              |

CPU fp32 took 18.58 s (1.6× RT). MPS bf16 took 9.75 s. Metal fp16 at 4.63 s reproduces
the M17 figure of 4.6 s, so this build has not regressed.

‡ Every row produced the same transcript as CPU fp32 (`be969f62b3`) **except Metal
`int8_float32`**: 85 words against 90, four small edits and one segment boundary merged.
That pattern points to quantization drift steering beam search, not corrupt output. MPS's
`int8_float32` stays byte-exact but is 2× slower. Metal `int8_float16`, the type plain
`int8` resolves to for this fp16 checkpoint, is exact.

**Read:**

- **Neither backend wins everywhere.** Metal leads on the decoder-only LLM and on native
  int8 speed. MPS leads on Whisper (about 1.25–1.3× in both fp16 and fp32) and on fp16
  encoder-decoder translation at batch size 8, where this fork's fp16 is a pessimization
  (see the OPUS/NLLB sections above). Upstream's fp16 enc-dec path has no such cost. The
  Whisper loss has a different cause; see "Why MPS wins Whisper" below.
- **MPS is far more consistent.** Its run-to-run spread is typically ≤3%, while Metal fp16
  OPUS bs=8 ranged from 684 to 927. A likely cause, not yet verified: upstream batches 16
  compute dispatches per command buffer and caps in-flight buffers at 8
  (`CT2_MPS_MAX_COMPUTE_OPS`, `CT2_MPS_MAX_INFLIGHT_COMMAND_BUFFERS`).
- **Parity:** every fp32 row matches the CPU token hash on both backends. Qwen fp16 is
  byte-identical across CPU, Metal and MPS. One divergence: Metal NLLB fp16 bs=8 differs
  from fp32 and CPU (`cece…` vs `05ee…`), while MPS matches. int8 hashes differ by design.

### Why MPS wins Whisper (diagnosed 2026-11-13)

Whisper large-v3 fp16 on `clip30.wav`, M4 Max. Measurements come from scratch-only
instrumentation (none of it is in this tree): summed `GPUEndTime − GPUStartTime` per command
buffer, buffers tagged by kernel name, and upstream run with `CT2_MPS_MAX_COMPUTE_OPS=1` so
each buffer holds exactly one kernel. `mediaanalysisd` used 100% CPU during the later runs,
so treat per-kernel figures as ±10%. Every conclusion below came from a same-binary
alternating A/B.

The 0.84 s per-pass gap (4.53 s vs 3.69 s) splits into 0.39 s more GPU work and 0.45 s
more GPU idle time:

| Per pass                               | Metal                                 | MPS    |
| -------------------------------------- | ------------------------------------- | ------ |
| Wall time                              | 4.53 s                                | 3.69 s |
| GPU busy (summed)                      | 3.84 s                                | 3.45 s |
| GPU idle                               | 0.69 s                                | 0.24 s |
| Command buffers                        | 111.7k                                | 54k    |
| Idle per command buffer                | 6.2 µs                                | 4.4 µs |
| CPU blocked in `[queue commandBuffer]` | 2.0 s (64-slot queue full: GPU-bound) | n/a    |

**Cause 1: cross-attention in `ct2_sdpa_half` (about +0.7 s of GPU time).** GPU time per kernel,
by job:

| Job (ms GPU per pass)   | Metal                                    | MPS                                                                |
| ----------------------- | ---------------------------------------- | ------------------------------------------------------------------ |
| Cross-attn (1500 keys)  | **1,873** (`sdpa`, 434 µs × 4,320)       | ≈1,100 (batched `tiled_gemm_f16` + softmax)                        |
| Self-attn (short keys)  | 114 (26 µs × 4,320)                      | in `gemv_row*`, ≈180                                               |
| KV cache + beam reorder | ≈180 (append/grow, gather, strided copy) | ≈566 (`concat2` 282 + gather 284)                                  |
| Projections + bias      | ≈1,400 (MPS + `bias_add` 100)            | ≈1,340 (`small_m5_gemm_nt_f16` with fused bias/residual/GELU, MPS) |
| LayerNorm + TopK        | ≈144                                     | ≈294                                                               |

The fused kernel uses one threadgroup per score row. In Whisper cross-attention that's
1 × 5 beams × 20 heads = 100 threadgroups, each walking 1,500 keys serially. It reads about
38 MB of K/V per call at roughly 90 GB/s effective, which is low occupancy and far from
bandwidth. Qwen decode never exposes this because it has more rows and fewer keys. Turning
it off (`CT2_NO_METAL_SDPA=1`) is not an option: the fallback is 4× slower overall
(18.6 s), because of the per-batch-index MPS encode from M16.

**Cause 2: twice as many command buffers (about +0.2–0.45 s, mostly GPU idle).** One
buffer per op (111.7k) against upstream's batched encoding (54k, 16 dispatches per buffer
and 8 in flight). Each buffer costs a fixed GPU-side gap. Forcing upstream to one op per
buffer cost it about 0.1–0.3 s per pass (noisy). Projections that run as a separate
`bias_add` dispatch add about 27k extra buffers per pass.

**Not the cause (ruled out by experiment):**

- **Queue drains.** A same-binary A/B that turned the 1,572 Metal→Metal `copy_from` drains
  into GPU blits cut flushes by 784 per pass and left wall time unchanged, with an
  identical transcript.
- **BiasAdd.** Its 2.9 s in the async profile is queue-backpressure attribution: it's the
  op that hits the full queue. Its kernel takes only about 100 ms of GPU time.
- **CPU encoding speed.** The CPU spends 2 s per pass blocked waiting for queue slots.

Where your own wins show up: the in-place KV cache (M18) beats upstream's `concat2` +
gather by about 0.39 s per pass, and LayerNorm and TopK are faster too. Fix candidates, by
estimated payoff: split-K / flash-decoding for `sdpa` when keys are many and rows few
(**done**: see "Split-K fused SDPA" below, Whisper fp16 4.61 → 3.35 s, now ahead of MPS); batching dispatches per command buffer while keeping overlap (**done**: see "Bounded
command-buffer batching" below, Whisper fp16 3.35 → 3.19 s); a fused bias/residual
epilogue for m=5 decode projections.

## Split-K fused SDPA: Whisper cross-attention at full GPU width (2026-11-13)

Apple M4 Max (40-core GPU), 64 GB, macOS 27.0, Release build with Accelerate,
`OPENMP_RUNTIME=NONE`, `ENABLE_PROFILING=OFF`. Every end-to-end figure is a same-binary
alternating A/B through `CT2_NO_METAL_SDPA_SPLIT=1` (off) against the default (split):
fresh process per arm, arm order reversed on odd rounds. `mediaanalysisd` and
`corespotlightd` were bursting to 100–160% CPU all afternoon, so the harness started a
run only on a quiet machine and discarded and repeated any run that ended with another
process above 20% of a core (14 discarded during the main matrix). Design and the M20
summary are in `METAL_BACKEND.md`; the plan and full results are in
`docs/metal/plan-sdpa-split-k.md`.

**Kernel-level sweep** (`DISABLED_BenchmarkSdpaSplit`, fp16, d_head 64, µs per call,
median of 5 bursts of 200 launches with one synchronize per burst; `s1` is the single-pass
kernel):

| Shape (rows × keys)                  | s1    | s2    | s3        | s4    | s8    | s16      | s24      | s32   | Heuristic picks |
| ------------------------------------ | ----- | ----- | --------- | ----- | ----- | -------- | -------- | ----- | --------------- |
| Whisper-L cross, beam 5: 100 × 1500  | 446.5 | 241.5 | **169.2** | 224.3 | 183.9 | 192.2    | 177.5    | 173.0 | 3               |
| Whisper-L cross, greedy: 20 × 1500   | 343.2 | 174.9 | 119.0     | 91.5  | 50.8  | **33.7** | 43.1     | 44.2  | 11              |
| Whisper-L cross, batch 4: 400 × 1500 | 944.9 | 741.5 | 805.9     | 818.4 | 722.4 | 658.5    | 649.2    | 653.1 | 1               |
| Qwen bs=1: 14 × 1024                 | 231.4 | 121.5 | 83.3      | 64.3  | 37.1  | 25.9     | **25.3** | 33.3  | 8               |
| Qwen bs=1: 14 × 4096                 | 938.6 | 475.1 | 321.8     | 244.3 | 129.7 | 73.6     | **57.1** | 78.2  | 22              |
| Qwen bs=8: 112 × 1024                | 268.8 | 145.3 | **106.5** | 140.5 | 131.6 | 116.5    | 115.1    | 114.1 | 2               |
| Qwen bs=1: 14 × 64                   | 19.2  | 14.5  | 12.9      | 11.4  | 11.0  | —        | —        | —     | 1               |
| NLLB cross, bs=8 beam 5: 640 × 32    | 21.7  | 26.4  | 27.1      | 27.2  | —     | —        | —        | —     | 1               |

fp32 follows the same shape (Whisper beam 5: 474.0 → 197.4 at 3 splits). The best split
count always lands where `rows × splits` is about 300–350: the single-pass kernel costs a
flat 69–72 µs from 256 to 352 rows (256 keys, fp16) and then climbs (368 rows 89 µs, 400
rows 120 µs, 512 rows 159 µs). That is one wave of 128-thread threadgroups on this GPU, so
the heuristic rounds `320 / rows` down. Past the wave, extra splits cost at most about 15%
against the best count and still beat single-pass at long keys; with short keys and many
rows (640 × 32) splitting loses, and the heuristic never splits there.

**Threshold sweep, end to end** (Whisper large-v3 fp16, clip30, beam 5, seconds per pass;
2 alternating rounds of 1 warmup + 3 passes under the quiet-machine gate; the two process
medians are shown, and every pass was within 0.02 s of its process median):

| Setting                                | Splits for cross-attn     | Round 1 | Round 2 |
| -------------------------------------- | ------------------------- | ------- | ------- |
| off (`CT2_NO_METAL_SDPA_SPLIT=1`)      | 1                         | 4.60    | 4.65    |
| `CT2_METAL_SDPA_SPLIT_TGS=200`         | 2                         | 3.67    | 3.68    |
| **default: target 320, min chunk 128** | 3                         | 3.37    | 3.38    |
| `CT2_METAL_SDPA_SPLIT_TGS=640`         | 6                         | 3.38    | 3.38    |
| `CT2_METAL_SDPA_SPLIT_TGS=1280`        | 11                        | 3.41    | 3.42    |
| `CT2_METAL_SDPA_SPLIT_MIN_CHUNK=32`    | 3, self-attention up to 3 | 3.34    | 3.34    |
| `CT2_METAL_SDPA_SPLIT_MIN_CHUNK=512`   | 2                         | 3.69    | 3.68    |

Three splits is the knee: two leaves 0.3 s on the table, six to eleven are flat to
slightly worse. An earlier ungated sweep also ran a 352-threadgroup target (3 splits,
3.34–3.35 s) and a 256-key minimum chunk (3 splits, 3.37 s). Letting self-attention split
as well (minimum chunk 32) measured 0.03–0.04 s (1%) faster in both rounds. That is
repeatable here but small, and it would also route short-context LLM decode to the split
path, so the default stays at 128.

**End-to-end A/B** (median of per-process medians; range is over process medians):

| Workload                      | off (single-pass)    | split (default)        | Change    | Output hash       |
| ----------------------------- | -------------------- | ---------------------- | --------- | ----------------- |
| Whisper large-v3 fp16, s/pass | 4.61 (4.58–4.62)     | **3.35** (3.34–3.88 †) | **1.38×** | `be969f62b3` both |
| Whisper large-v3 fp32, s/pass | 5.69 (5.69–5.77)     | **4.43** (4.37–4.56)   | **1.28×** | `be969f62b3` both |
| Qwen2.5-0.5B fp16 bs=1, tok/s | 177.2 (176.5–177.6)  | 177.2 (176.2–178.8)    | none      | `c1444f2d70` both |
| Qwen2.5-0.5B fp16 bs=8, tok/s | 1211.6 (1204–1316 ‡) | 1206.1 (1201–1320 ‡)   | none      | `f653792625` both |
| Qwen2.5-0.5B fp32 bs=1, tok/s | 150.1 (149.6–150.3)  | 150.1 (149.8–150.2)    | none      | `c1444f2d70` both |
| Qwen2.5-0.5B fp32 bs=8, tok/s | 858.6 (857.5–859.2)  | 856.2 (856.1–857.0)    | −0.3%     | `f653792625` both |
| OPUS-MT fp16 bs=1, tok/s      | 192.2 (190.7–198.8)  | 194.4 (194.3–194.9)    | none      | `1255ca4303` both |
| OPUS-MT fp16 bs=8, tok/s      | 849.6 (820.8–863.6)  | 844.6 (832.0–866.1)    | none      | `f1ed445777` both |
| NLLB-600M fp16 bs=1, tok/s    | 86.2 (86.0–86.6)     | 86.3 (85.4–86.6)       | none      | `8fba0a74d0` both |
| NLLB-600M fp16 bs=8, tok/s    | 256.4 (255.4–259.1)  | 256.1 (255.9–258.0)    | none      | `cece2ba05f` both |

Whisper: 5 process pairs × (1 warmup + 5 passes). Qwen (128 greedy tokens), OPUS-MT and
NLLB (beam 5): 3 process pairs × (3 warmups + 7 passes).

† Four of the five split processes had medians of 3.34–3.35 s. The fifth (3.88 s, passes
3.85–4.01) ran in the same round as the slowest fp32 split process; the gate only checks
load at the start and end of a run, so a mid-run burst is the likely cause, not verified.
Its slowest pass was still faster than the fastest off pass (4.56 s).

‡ Qwen fp16 bs=8 is bimodal on this rig in **both** arms (about 1,204 or about 1,318
tok/s per process), so the range is rig behaviour, not an effect of the change.

**Long-context LLM decode** (Qwen2.5-0.5B, 44-token prompt, **512** forced greedy tokens so
self-attention reaches 556 keys and the split path engages from 256 keys on; 3 process
pairs × (2 warmups + 5 passes), tok/s, median of process medians, range over them):

| Config    | off (single-pass)      | split (default)          | Change | Tokens                           |
| --------- | ---------------------- | ------------------------ | ------ | -------------------------------- |
| fp32 bs=1 | 127.8 (127.8–128.0)    | **146.5** (146.3–146.6)  | +14.6% | identical; both equal CPU fp32   |
| fp32 bs=8 | 788.4 (555.6 §–792.2)  | **855.3** (803.8–858.5)  | +8.5%  | identical; both equal CPU fp32   |
| fp16 bs=1 | 164.6 (159.0–170.6)    | **181.1** (181.0–182.0)  | +10.0% | differ from step 292 (see below) |
| fp16 bs=8 | 1200.0 (1196.0–1207.5) | **1321.5** (1317–1387 ¶) | +10.1% | differ from step 292 (see below) |

§ One off process read 555.6 (passes 527–576) with nothing flagged at its end; the other
two were 788.4 and 792.2. ¶ The 1,387 process was accepted on its fifth attempt with
`corespotlightd` busy at its end; the other two were 1,317 and 1,322.

**fp16 token parity at 512 tokens.** In fp32 the single-pass kernel, the split kernel and
the CPU produce the same 512 tokens at bs=1 and bs=8. In fp16 the two arms part ways, and
a CPU fp32 arbiter (`forward_batch` over the shared prefix) shows both are near-ties
decided by float summation order, not errors:

- **Single-pass fp16** leaves the fp32 sequence at generated step 292 (336 keys): fp32
  log-probs −2.3424 vs −2.3468 for the two candidates, a gap of 0.0044.
- **Split fp16, bs=1** (2 to 4 splits as the context grows) matches CPU fp32 for all 512
  tokens.
- **Split fp16, bs=8** (2 splits) matches fp32 until step 422 (466 keys), where the top
  fp32 token is the EOS that `min_length` suppresses and the next two candidates are
  0.0014 apart (−4.5361 vs −4.5375).

So the 128-token gate hashes are unchanged, and at 512 tokens the split path stays on the
fp32 sequence at least as long as the single-pass path does in every case measured. A
different split count is a different summation order, though, so fp16 greedy output past
a near-tie is not guaranteed to match between the two kernels.

**Per-kernel GPU time** (scratch-only instrumented build, summed
`GPUEndTime − GPUStartTime` per command buffer; one process per cell, 1 warmup + 4 passes,
quiet machine; 4,320 cross-attention and 4,320 self-attention `sdpa` launches per pass):

| Per pass                               | fp16 off         | fp16 split         | fp32 off         | fp32 split         |
| -------------------------------------- | ---------------- | ------------------ | ---------------- | ------------------ |
| Cross-attention `sdpa` (1,500 keys)    | 432 µs, 1,864 ms | **155 µs, 668 ms** | 445 µs, 1,920 ms | **160 µs, 691 ms** |
| Self-attention `sdpa` (short keys)     | 25.7 µs, 111 ms  | 25.7 µs, 111 ms    | 26.5 µs, 115 ms  | 26.4 µs, 114 ms    |
| All kernels, GPU busy                  | 3.82 s           | 2.62 s             | 4.72 s           | 3.49 s             |
| CPU blocked in `[queue commandBuffer]` | 2.06 s           | 0.88 s             | 2.74 s           | 1.60 s             |
| Wall time (instrumented)               | 4.65 s           | 3.38 s             | 5.79 s           | 4.36 s             |

**Read:**

- Cross-attention `sdpa` dropped 2.8× per call and 1.2 s of GPU time per pass in both
  precisions, and wall time followed (−1.26 s fp16 and fp32).
- Against upstream `Device::MPS`, re-measured the same afternoon without the quiet-machine
  gate (fp16: two processes of 5 passes, 3.69 s and 3.74 s; fp32: one process, 4.40 s; a
  background daemon was busy by the end of each): Metal fp16 at 3.35 s is now about 10%
  ahead, and Metal fp32 at 4.43 s is level. The morning's MPS figures were 3.69–3.74 s
  and 4.47 s. A third MPS fp32 process read 6.29 s straight after a Spotlight burst and
  was set aside.
- Qwen at 128 tokens, OPUS-MT and NLLB never reach the split path (fewer than 256 keys
  per row, or more than 160 rows), so "no change" there is by construction; the A/B
  confirms the refactored single-pass kernel is not slower and produces the same tokens.
  Once Qwen's context passes 256 keys the split path engages and is 8–15% faster end to
  end.
- The split path is more exposed to CPU contention than the single-pass path: under a
  `mediaanalysisd` burst the split arm lost 0.3–0.5 s per pass while the off arm did not
  move. A likely reason, not verified: the off arm spends 2.1 s per pass blocked on a full
  command queue, slack that hides a slower CPU, and the split arm has only 0.9 s of it.
- After the change the GPU is busy for 2.62 s of a 3.38 s pass. The remaining idle time
  is the one-command-buffer-per-op cost that the plan lists under "Not in this plan".

## Bounded command-buffer batching (2026-11-14)

Apple M4 Max (40-core GPU), 64 GB, macOS 27.0, Release build with Accelerate,
`OPENMP_RUNTIME=NONE`, `ENABLE_PROFILING=OFF`. Every end-to-end figure is a same-binary
alternating A/B through `CT2_NO_METAL_CB_BATCH=1` (off, one command buffer per op) against
the default (on): fresh process per arm, arm order reversed on odd rounds, started only on
a quiet machine and discarded and repeated if another process was above 20% of a core at
the end of the run (41 discarded during the final matrix). Design and the M21 summary are
in `METAL_BACKEND.md`; the plan and full results are in `docs/metal/plan-cb-batching.md`.

**End-to-end A/B**

Median of per-process medians; the range is over process medians.

| Workload                                  | off (per-op)           | on (default)               | Change        |
| ----------------------------------------- | ---------------------- | -------------------------- | ------------- |
| Whisper large-v3 fp16, s/pass             | 3.35 (3.35–3.38)       | **3.19** (3.16–3.58 †)     | **1.05×**     |
| Whisper large-v3 fp32, s/pass             | 4.44 (4.39–6.32 †)     | **4.30** (4.19–4.69 †)     | 1.03×         |
| Qwen decode 128, fp16 bs=1, tok/s         | 179.0 (177.4–181.8)    | **235.0** (234.4–237.5)    | **+31%**      |
| Qwen decode 128, fp16 bs=8, tok/s         | 1208.1 (1205.2–1211.9) | **1486.2** (1480.0–1495.0) | **+23%**      |
| Qwen decode 128, fp32 bs=1, tok/s         | 149.5 (149.0–150.3)    | **161.1** (161.0–161.7)    | +7.8%         |
| Qwen decode 128, fp32 bs=8, tok/s         | 857.4 (857.2–858.2)    | 873.2 (872.5–873.8)        | +1.8%         |
| Qwen decode 512, fp16 bs=1, tok/s         | 184.5 (183.0–187.7)    | **223.1** (219.4–223.9)    | **+21%**      |
| Qwen decode 512, fp16 bs=8, tok/s         | 1324.8 (1281.6–1326.0) | **1534.8** (1510.4–1539.0) | **+16%**      |
| Qwen decode 512, fp32 bs=1, tok/s         | 146.2 (146.2–146.4)    | **155.9** (155.2–156.1)    | +6.6%         |
| Qwen decode 512, fp32 bs=8, tok/s         | 854.8 (688.4 ‡–858.2)  | 870.7 (723.4 ‡–872.4)      | +1.9%         |
| Qwen prefill 128, fp16 bs=1, prompt tok/s | 3451.9 (3367.5–3454.5) | **3791.9** (3778.4–3828.4) | +9.8%         |
| Qwen prefill 128, fp16 bs=8               | 5886.8 (5782.7–5922.6) | 5945.2 (5860.2–6029.7)     | within spread |
| Qwen prefill 128, fp32 bs=1               | 3344.9 (3341.4–3345.3) | **3473.2** (3466.0–3482.8) | +3.8%         |
| Qwen prefill 128, fp32 bs=8               | 5381.3 (5362.4–5488.9) | 5451.8 (5378.4–5480.5)     | within spread |
| Qwen prefill 512, fp16 bs=1               | 7902.2 (7526.2–7937.9) | **8208.5** (8145.1–8231.9) | +3.9%         |
| Qwen prefill 512, fp16 bs=8               | 9938.1 (9891.2–9952.3) | 9945.1 (9906.8–9966.4)     | none          |
| Qwen prefill 512, fp32 bs=1               | 6689.3 (6656.1–6732.1) | 6718.5 (6690.6–6754.7)     | none          |
| Qwen prefill 512, fp32 bs=8               | 7573.3 (7541.1–7583.9) | 7593.8 (7570.3–7601.9)     | none          |
| OPUS-MT fp16 bs=1, tok/s                  | 184.5 (183.1–191.0)    | **292.0** (290.4–292.9)    | **+58%**      |
| OPUS-MT fp16 bs=8, tok/s                  | 822.6 (813.7–849.9)    | **994.2** (983.1–1041.4)   | **+21%**      |
| NLLB-600M fp16 bs=1, tok/s                | 86.3 (85.3–87.5)       | **108.8** (108.5–109.4)    | **+26%**      |
| NLLB-600M fp16 bs=8, tok/s                | 256.0 (255.4–256.6)    | **269.2** (268.6–269.3)    | +5.2%         |

Whisper: 5 process rounds × (1 warmup + 5 passes), beam 5, with a third arm running a
build of main `de415aca`: fp16 3.35 s (3.34–3.37), fp32 4.54 s (4.46–4.61). Qwen decode
128, OPUS-MT, NLLB: 3 rounds × (3 warmups + 7 passes). Qwen decode 512: 3 × (2 + 5).
Prefill is a prompt of 128 or 512 tokens per row plus one generated token, timed as prompt
tokens per second: 128 is 3 × (5 + 15), 512 is 3 × (3 + 7).

† One round was disturbed in each Whisper column. fp16: four on processes read 3.16–3.20
and one read 3.58. fp32: round 3 read 6.32 (off) and 4.69 (on) back to back; the other
four rounds were 4.39–4.53 (off) and 4.19–4.36 (on). The gate only samples load at the
start and end of a run, so a mid-run burst is the likely cause, not verified.

‡ Round 1 of Qwen 512 fp32 bs=8 was slow in both arms (688 and 723); the other two rounds
were 855–858 and 871–872.

The earlier sweeps, with the same harness, measured Whisper fp16 at 3.12–3.16 s for the
neighbouring settings (table below), so 3.19 s is at the slow end of what this change has
shown on this machine.

**Counters before and after**

One process per cell with `CT2_METAL_STATS=1` (which adds a completion handler per buffer),
1 warmup + 4 passes for Whisper, 2 + 8 for the others; figures are per pass.

| Per pass                       | Whisper fp16 off | Whisper fp16 on | Whisper fp32 off | Whisper fp32 on |
| ------------------------------ | ---------------- | --------------- | ---------------- | --------------- |
| Ops                            | 111.7k           | 111.7k          | 111.7k           | 111.7k          |
| Command buffers                | 111.7k           | **20.8k**       | 111.7k           | **26.1k**       |
| Ops per buffer                 | 1                | 5.4             | 1                | 4.3             |
| GPU busy (summed intervals)    | 2.62 s           | 2.49 s          | 3.49 s           | 3.41 s          |
| CPU blocked acquiring a buffer | 0.85 s           | 0.60 s          | 1.56 s           | 1.07 s          |
| Wall (this process)            | 3.37 s           | 3.18 s          | 4.44 s           | 4.22 s          |
| Peak memory footprint          | 4.47 GB          | 4.50 GB         | 8.65 GB          | 8.66 GB         |

| Whole process (10 passes)   | Buffers off → on | CPU blocked acquiring | GPU busy off → on |
| --------------------------- | ---------------- | --------------------- | ----------------- |
| Qwen decode 128, fp16 bs=1  | 441.0k → 64.1k   | 55 → 5 ms             | 5.43 → 4.81 s     |
| Qwen decode 128, fp16 bs=8  | 441.0k → 66.3k   | 74 → 5 ms             | 6.35 → 5.64 s     |
| Qwen prefill 512, fp16 bs=8 | 9.2k → 7.2k      | 54 → 8 ms             | 4.46 → 4.39 s     |
| NLLB-600M fp16 bs=1         | 76.0k → 15.8k    | 141 → 2 ms            | 1.40 → 1.38 s     |

Whisper is still GPU-bound after the change: the CPU still spends 0.6 s per pass waiting
for a queue slot, and the GPU is busy for 2.49 s of a 3.18 s pass. Peak memory did not
move (within 1%) in any cell.

"GPU busy" is the sum of each buffer's `GPUEndTime − GPUStartTime`. With one op per
buffer those intervals overlap (see the prefill finding below), so the off and on columns
are not measuring quite the same thing. Treat the busy figures as indicative; wall time is
the result.

**How the defaults were chosen**

Three sweeps, each 2 alternating rounds, fresh process per arm. Values are the two process
medians, or one where the two rounds were not comparable.

**Sweep 1: op count only** (no early-commit rules). Round 0 of the Whisper column ran with
a steady Firefox video load on the GPU (about 0.2 s slower in every arm, same ordering);
the quiet round is shown.

| Ops per buffer | Whisper fp16 s | Qwen dec fp16 bs=1 | Qwen dec fp16 bs=8 | Prefill 512 fp16 bs=8 | Prefill 512 fp16 bs=1 |
| -------------: | -------------- | ------------------ | ------------------ | --------------------- | --------------------- |
|        1 (off) | 3.36           | 180.1, 181.3       | 1209, 1207         | 9813, 9825            | 7844, 7902            |
|              2 | 3.23           | 213.8, 212.6       | 1403, 1386         | 9817, 9841            | 8174, 8151            |
|              4 | 3.15           | 226.9, 227.2       | 1458, 1453         | 9400, 9261            | 8085, 8018            |
|              8 | **3.12**       | **234.7, 234.5**   | **1453, 1457**     | 8463, 8499            | 7390, 6919            |
|             16 | 3.14           | 234.2, 234.5       | 1450, 1452         | 7861, 7887            | 6646, 6456            |
|             32 | 3.16           | 229.2, 229.7       | 1413, 1416         | 7676, 7703            | 5011, 5534            |
|             64 | 3.29           | 219.0, 220.3       | 1362, 1358         | 7666, 7704            | 6270, 5879            |
|            256 | 3.65           | 142.8, 147.5       | 933, 914           | 7546, 7584            | 5419, 5626            |

In-flight limit and encoder sharing at 8 ops per buffer, Whisper fp16 (quiet round):

| Setting                      | s/pass                                          |
| ---------------------------- | ----------------------------------------------- |
| in-flight 2                  | 3.93                                            |
| in-flight 4                  | 3.21                                            |
| in-flight 8                  | 3.19                                            |
| in-flight 16                 | 3.16                                            |
| in-flight 64 (queue default) | 3.12                                            |
| in-flight 256                | 3.12 (3.09 in the Firefox-free part of round 0) |
| separate encoder per op      | 3.10 (3.11)                                     |

Decode and Whisper peak at 8–16 ops and fall off from 32: the 2026-06 result, one commit
per step, is the far right of this curve. **Prefill regressed with op count** (bs=8: −14%
at 8, −20% at 16), which is the same loss the 2026-06 attempt recorded. An in-flight limit
below the queue default never helped and upstream's 8 cost 0.07 s; sharing the encoder
measured no different from one encoder per op.

**Sweep 2: add commit-when-GPU-idle** (still count only):

| Setting            | Prefill 512 fp16 bs=8 | Qwen dec fp16 bs=8 | Whisper fp16 s |
| ------------------ | --------------------- | ------------------ | -------------- |
| off                | 9882, 9892            | 1237, 1204         | 3.40, 3.37     |
| 8, no early commit | 8462, 8472            | 1455, 1452         | 3.14, 3.13     |
| 4                  | 9631, 9617            | 1453, 1456         | 3.16, 3.15     |
| 8                  | 9124, 9182            | **1495, 1494**     | **3.12, 3.12** |
| 16                 | 8803, 8779            | 1491, 1493         | 3.16, 3.16     |
| 32                 | 8638, 8733            | 1467, 1472         | 3.21, 3.21     |

The idle rule recovers about half the prefill loss and adds 3% to decode bs=8, but prefill
bs=8 is still 7% down at 8 ops.

**Why prefill loses.** `CT2_METAL_STATS=1` on prefill 512 fp16 bs=8 (12 passes): with one
buffer per op the summed GPU intervals were 5.33 s for 4.98 s of wall time, so consecutive
buffers overlap on the GPU. With 8 ops per buffer and no early commit the sum fell to
3.79 s while wall time rose to 5.77 s and the CPU's time blocked in `flush()` rose from
3.53 s to 4.41 s. The CPU was waiting and the GPU was doing less per unit of time. The
likely reading, not verified beyond these counters: the GPU pipelines separate command
buffers but runs the passes inside one buffer strictly one after another, which costs
nothing for small ops and a lot for ops of several hundred microseconds. Encoder sharing
is not the cause (same result with one encoder per op).

**Sweep 3: add the GPU-time target** (8 ops per buffer unless noted):

| Target µs      | Prefill 512 bs=8 | Prefill 512 bs=1 | Prefill 128 bs=8 | Qwen dec bs=1 | Qwen dec bs=8 | Whisper s  |
| -------------- | ---------------- | ---------------- | ---------------- | ------------- | ------------- | ---------- |
| off            | 9942, 9841       | 7912, 7939       | 5844, 5775       | 179.3, 177.9  | 1211, 1205    | 3.38, 3.37 |
| 0 (count only) | 9249, 9201       | 7112, 7557       | 5684, 5602       | 237.5, 235.5  | 1491, 1489    | 3.14, 3.11 |
| 50             | 9964, 9922       | 8195, 8149       | 5934, 5904       | 230.5, 230.9  | 1464, 1466    | 3.23, 3.21 |
| **100**        | 9856, 9868       | 8159, 8115       | 5896, 5701       | 234.6, 239.6  | 1484, 1484    | 3.18, 3.16 |
| 200            | 9941, 9883       | 8236, 8249       | 6008, 5897       | 235.4, 235.1  | 1487, 1490    | 3.16, 3.15 |
| 400            | 9743, 9729       | 7676, 7767       | 5804, 6047       | 239.1, 235.0  | 1485, 1490    | 3.12, 3.12 |
| 200, 16 ops    | 9903, 9891       | 7217, 8090       | 5928, 6068       | 236.4, 235.8  | 1501, 1501    | 3.19, 3.18 |
| 200, 4 ops     | 9941, 9886       | 8197, 8103       | 5895, 6091       | 227.5, 226.0  | 1452, 1453    | 3.17, 3.17 |

All fp16. Any target from 50 to 200 µs removes the prefill loss and keeps the decode gain.
The 128-token bs=1 prefill cell (35 ms per pass) was too noisy here to rank 100 against
200, so it was rerun alone, 5 rounds × (5 warmups + 15 passes):

| Prefill 128, bs=1 | off                | target 100             | target 200         | target 100, 4 ops  |
| ----------------- | ------------------ | ---------------------- | ------------------ | ------------------ |
| fp16              | 3515.1 (3447–3545) | **3821.4** (3798–3828) | 3759.9 (3750–3776) | 3782.1 (3768–3787) |
| fp32              | 3289.0 (3253–3323) | **3480.3** (3470–3481) | 3284.7 (3177–3293) | 3429.0 (3418–3447) |

100 µs wins both, and 200 µs gives nothing in fp32, so the default is 8 ops and 100 µs.
That costs Whisper about 0.02–0.04 s per pass against the best count-only setting.

**Read:**

- Batching 8 small ops per command buffer is worth 16–31% on fp16 LLM decode and 5–58% on
  fp16 translation (most at batch 1), where the CPU-side cost of creating and committing a
  buffer per op was on the critical path. Whisper, which is GPU-bound, gains 5%.
- The 2026-06 "command-buffer reuse" result was the far end of the op-count curve. The
  curve peaks at 8–16 ops and is negative by 256 for decode; for bs=8 512-token prefill it
  is negative from 4 ops on unless batches are sized by GPU time.
- After the change Whisper's GPU is busy for 2.49 s of a 3.18 s pass and the CPU still
  waits 0.6 s per pass for a queue slot. What is left is host sync points in the decode
  loop (2,356 flushes per pass), not command-buffer count.

## Caveats

- Tiny model + tiny ops is the worst case; a real LLM (large hidden size, big GEMMs) sits
  much closer to the favorable end of the GEMM table — and since the fused decode
  attention kernel (2026-07-07), decode is no longer the loss column: the remaining
  per-op floor is carried by the projection GEMMs and cache/norm/rotary small ops.
- Numbers are single-run averages on a warm machine (the fused-SDPA table: two runs,
  spread shown); treat as indicative, not precise.
