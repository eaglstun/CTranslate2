---
title: "Benchmarking & profiling the Metal backend (the methodology that found every win)"
summary: >-
  Documents the Metal-backend benchmarking harness that produced every number
  in METAL_BENCHMARKS.md: the DISABLED_Benchmark* Google Test cases in
  tests/metal_test.cc (BenchmarkGemm, BenchmarkGemmEncode, BenchmarkReduction,
  BenchmarkAddRMSNorm, BenchmarkTranslation, BenchmarkLLM), built Release with
  tests on and run via --gtest_also_run_disabled_tests. Its centerpiece is the
  probe-isolation trick in BenchmarkGemmEncode, which separates per-op encode
  cost from GPU round-trip by encoding N ops and flushing once, isolating the
  ~0.042 ms encode floor (cut to ~0.031 ms by the MPS-object cache). Covers
  the time_ms warmup primitive, the CT2_LLM_MODEL/CT2_LLM_PROFILE env knobs,
  ENABLE_PROFILING with PROFILE() scopes, and the cautionary tale where
  profiling exposed the elementwise Add op exploding 27x in fp16. Its
  discipline rules are always-Release-always-warm, isolate the decode/prefill
  regime, and pick the right probe for the question.
semantic_id: "yp4gxg2uum2TAfl9wPkZPKNwl6ORx68jDlX-OJa1xv4qBekGja-oVy-zfo0k648aqGc8c8V6nZ5W8ayvxjj8oQ"
---

Every number in `METAL_BENCHMARKS.md` — and every conclusion in
`dispatch-overlap-and-perf-model.md` — came out of this harness. The repeated lesson of
that doc is **"profile, don't guess"**; this is how you profile. Read this before chasing a
perf regression or claiming a speedup.

Source: `tests/metal_test.cc` (the `DISABLED_Benchmark*` cases), `src/profiler.cc` /
`include/ctranslate2/profiler.h`, `METAL_BENCHMARKS.md` reproduction commands. Line numbers
verified by grep on 2026-06-09 — re-grep the `DISABLED_Benchmark` symbol if they drift.

## The harness

Benchmarks live as `DISABLED_*` Google Test cases so a normal `ctranslate2_test` run skips
them. Build with tests on, then opt in explicitly:

```bash
cmake .. -DWITH_METAL=ON -DWITH_ACCELERATE=ON -DOPENMP_RUNTIME=NONE \
         -DBUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release   # Release matters for perf numbers
./tests/ctranslate2_test ../tests/data \
  --gtest_also_run_disabled_tests --gtest_filter='*Benchmark*'
```

`tests/metal_test.cc:344` `time_ms(iters, fn)` is the measurement primitive: it calls `fn`
**once as warmup** (MPS pipeline compile, allocator priming — never measure the first call)
then averages `iters` runs. Every benchmark uses it. The cases:

| Case (`MetalTest.DISABLED_…`) | What it isolates                                                                |
| ----------------------------- | ------------------------------------------------------------------------------- |
| `BenchmarkGemm` (:354)        | Square GEMM GFLOPS, CPU vs Metal fp32/fp16 — the crossover table                |
| `BenchmarkGemmEncode` (:383)  | **Per-op encode cost vs execution** — the probe-isolation trick (below)         |
| `BenchmarkReduction` (:414)   | softmax / rms_norm / layer_norm row reductions — A/B for the SIMD-group rewrite |
| `BenchmarkAddRMSNorm` (:448)  | the fused add+norm path                                                         |
| `BenchmarkTranslation` (:488) | tiny end-to-end model (worst case for any GPU backend)                          |
| `BenchmarkLLM` (:508)         | real decoder LLM, decode- and prefill-bound regimes                             |

## The probe-isolation trick (separate encode cost from execution cost)

This is the technique that found the floor. You cannot see the per-op API cost directly —
a normal benchmark flushes every iteration and measures encode **+** GPU round-trip
together. `BenchmarkGemmEncode` (`tests/metal_test.cc:383`) separates them:

- **flush-per-iter:** `mm(a,b,c); synchronize_device(METAL,0)` each iteration → encode + wait.
- **batched-encode:** encode all `iters` GEMMs, **one `synchronize_device` at the very end**
  → the floor set by command-buffer + `MPSMatrix` + `MPSMatrixMultiplication` + commit,
  with the GPU round-trip amortized away.

The gap between the two columns is the wait/round-trip; the batched number is the per-op
encode floor. This is exactly how the **~0.042 ms** encode cost was measured at n=256, and
how the MPS-object cache was shown to cut it to **~0.031 ms** (and e2e by ~35%). When you
add an optimization, run this probe before/after — it tells you whether you moved the
_encode floor_ or just the _execution time_, which are fixed by completely different things.

> The pattern generalizes: to measure the fixed per-op cost of _any_ Metal op, issue it N
> times committing each but flushing once at the end, and divide. Since the backend already
> batches commits in production (no per-op wait), that encode cost is what actually hits
> thousands of times per decode — it's the number that matters, not the flush-per-iter one.

## Profiling a real model (`CT2_LLM_PROFILE` — "profile, don't guess")

`BenchmarkLLM` (`tests/metal_test.cc:508`) runs greedy generation on a converted
decoder-only model and reports tok/s (decode-bound) and ms (prefill-bound). Environment
knobs (all `std::getenv`):

| Env var             | Effect                                                                      |
| ------------------- | --------------------------------------------------------------------------- |
| `CT2_LLM_MODEL`     | path to a converted decoder dir (e.g. Qwen2.5-0.5B); **unset → test skips** |
| `CT2_LLM_PROFILE=1` | dump the per-op execution profile (needs `-DENABLE_PROFILING=ON`)           |
| `CT2_LLM_PROMPT`    | override the prompt (decode-parity test; needs a real `<bos>` prompt)       |
| `CT2_GEMMA_MODEL`   | the Gemma2-specific case                                                    |

Convert a model first (from `METAL_BENCHMARKS.md`):

```bash
PYTHONPATH=python python -c "import sys; sys.argv=['x','--model','Qwen/Qwen2.5-0.5B-Instruct','--output_dir','/tmp/qwen0.5b-ct2','--quantization','float32','--force']; from ctranslate2.converters.transformers import main; main()"
CT2_LLM_MODEL=/tmp/qwen0.5b-ct2 CT2_LLM_PROFILE=1 ./tests/ctranslate2_test ../tests/data \
  --gtest_also_run_disabled_tests --gtest_filter='MetalTest.DISABLED_BenchmarkLLM'
```

Profiling requires `-DENABLE_PROFILING=ON` at cmake time — otherwise `init_profiling`
throws with a message telling you to enable it (`src/profiler.cc:9`). Ops are timed via
`PROFILE("name")` scopes (e.g. `PROFILE("MultiHeadAttention")` in `attention.cc`);
`dump_profiling` prints the per-function breakdown.

**Why this matters — the cautionary tale.** The first fp16 prefill run looked like an
MPS-fp16 weakness (284 vs 130 ms, _slower_ than fp32). The intuitive culprit was the GEMM.
`CT2_LLM_PROFILE=1` said the GEMMs were _identical_ in both precisions — the entire
regression was the elementwise **`Add`** op exploding **27×** in fp16 (51→1376 ms),
because it had silently never been routed to the GPU. The profiler found in one run what
no amount of staring at the GEMM would have. That fix is the headline 2.6×-vs-CPU result.

## Discipline

- **Always Release, always warm.** Numbers are single-run averages on a warm machine —
  indicative, not precise. State that when you report them.
- **Isolate the regime.** Decode-bound (prompt 32, gen 32) and prefill-bound (prompt 512,
  gen 1) are _different questions_ — a change can win one and lose the other (command-buffer
  reuse did exactly that). Never report a single e2e number as "the" result.
- **Pick the right probe.** End-to-end ms answers "is it faster"; `BenchmarkGemmEncode`
  answers "did I move the encode floor"; `CT2_LLM_PROFILE` answers "which op is to blame."
  Reaching for the wrong one is how you guess.

### Relevance to the CT2 Metal backend

- The conclusions these tools produced live in `dispatch-overlap-and-perf-model.md` (the
  floor, the overlap principle, the command-buffer-reuse graveyard). This file is _how to
  reproduce and extend_ them; that file is _what they mean_.
- A profiler hotspot on a norm/softmax/reduction → `simd-group-functions.md` (the reduction
  rewrite) and `math-functions-and-numeric-parity.md` (fp16 parity). A hotspot on an op
  that's secretly on the CPU reference (the `Add` story) → `op-graduation-playbook.md` (how
  to route it to the GPU).
