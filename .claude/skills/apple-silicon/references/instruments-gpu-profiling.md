---
title: "Instruments GPU profiling (the GPU-side view the CPU harness can't give)"
summary: >-
  Using Instruments and the xctrace CLI for GPU-side profiling that the CPU
  wall-clock harness cannot provide — the two load-bearing questions being
  whether CPU/GPU overlap is real (Metal System Trace timeline lanes) and
  whether ct2_gemm_s8 is ALU-bound (Metal GPU Counters limiter percentages).
  It documents the locally-verified template/instrument inventory (Metal
  System Trace template; Metal Application, Metal GPU Counters, Metal
  Performance Overview, os_signpost instruments), headless xctrace record
  recipes with --template/--instrument/--launch/--env flags, and how to label
  anonymous C++ encoder regions with os_signpost_interval_begin/end and
  os_signpost_id_generate. It honestly flags that Instruments was not actually
  run while writing the card; descriptions of trace contents come from Apple
  docs and must be verified in the UI on first use.
---

The repo's benchmark harness (`benchmarking-and-profiling.md`) measures _wall time from
the CPU side_ — it found the encode floor and the op-level wins, but it cannot say what
the GPU was doing in between, or _why_ a kernel is slow (ALU vs bandwidth vs occupancy).
Instruments is the tool for those questions. **Honesty note: Instruments was not run
while writing this card** — what's verified is the local tooling inventory (templates/
instruments listed via `xcrun xctrace` on the dev box, 2026-06-11), the `xctrace record`
CLI surface (from its own usage text), and the os_signpost API (Apple DocC). The
descriptions of what the trace _shows_ are from Apple's tool documentation and are
marked where unverifiable by fetch.

Sources: `xcrun xctrace list templates` / `list instruments` / `record` usage (run
locally); Apple DocC JSON for `os/os_signpost_interval_begin`, `os_signpost_interval_end`,
`os_signpost_id_generate`, `os/recording-performance-data`, `os/ossignposter`.

## What's installed here (verified on the dev box)

- Template: **`Metal System Trace`** (in `xcrun xctrace list templates`).
- Instruments (addable via `--instrument`): **`Metal Application`**, **`Metal GPU
Counters`**, **`Metal Performance Overview`**, **`Metal Resource Events`**, **`GPU`**,
  **`os_signpost`**. There is no standalone "GPU Counters" _template_ — add the
  `Metal GPU Counters` instrument to a recording.

## The two load-bearing questions Instruments would answer

1. **"Is the CPU/GPU overlap real?"** The whole perf model
   (`dispatch-overlap-and-perf-model.md`) rides on per-op commit letting the GPU run op
   N while the CPU encodes N+1 — inferred from timings (the −23% command-buffer-reuse
   revert), never _seen_. Metal System Trace shows CPU encode activity and GPU execution
   as separate timeline lanes per command buffer/encoder; a healthy decode step should
   show them interleaved, a regression shows GPU gaps. (Lane description per Apple's
   Instruments documentation — not fetchable as DocC JSON; verify against the actual UI
   on first use.)
2. **"Is `ct2_gemm_s8` ALU-bound?"** The int8 references conclude ALU-bound from
   arithmetic (no int8 matrix units, ~3–5× slower than MPS fp16 at large m —
   `int8-gemm-kernel-design.md`); the `Metal GPU Counters` instrument reports limiter
   percentages (ALU / memory / occupancy) per encoder and would turn that inference
   into a measurement.

## Headless capture with `xctrace` (CLI verified from local usage text)

```bash
# Record the int8 GEMM micro-benchmarks under Metal System Trace:
xcrun xctrace record --template 'Metal System Trace' \
  --output /tmp/ct2-gemm-s8.trace \
  --launch -- ./tests/ctranslate2_test ../tests/data \
       --gtest_also_run_disabled_tests --gtest_filter='*BenchmarkGemmS8*'

# Add the counters instrument for the limiter view:
xcrun xctrace record --template 'Metal System Trace' \
  --instrument 'Metal GPU Counters' \
  --time-limit 30s --output /tmp/ct2-counters.trace \
  --launch -- ./cli/ct2-translator ... --log_throughput

open /tmp/ct2-gemm-s8.trace        # inspect in the Instruments UI
```

Useful verified flags: `--env VAR=value` (launched process env — e.g.
`CT2_LLM_MODEL`), `--attach <pid|name>`, `--time-limit`, `--target-stdout -`.
`xctrace export` exists for XML extraction of some tables, but GPU-track export
coverage is unverified — plan on reading traces in the UI.

## Labeling C++ regions with os_signpost (so traces aren't anonymous)

A trace of `ctranslate2_test` shows nameless encoders. os_signpost intervals appear as
named lanes Instruments correlates with the GPU tracks (the `os_signpost` instrument is
in the local inventory). The C API (DocC-verified symbols, `<os/signpost.h>` /
`<os/log.h>`):

- `os_signpost_interval_begin(log, id, "name", ...)` — "marks the start of a time
  interval in your code using a signpost"; `os_signpost_interval_end` closes it.
- `os_signpost_id_generate(log)` — "creates a signpost identifier that's unique among
  signposts logged to a specified log" (needed when intervals of the same name nest or
  overlap, e.g. per decode step on worker threads).
- The log handle comes from `os_log_create("dev.ct2.metal", "decode")`; DocC's
  "recording-performance-data" article is the umbrella reference. (`OSSignposter` is
  the Swift wrapper — irrelevant for this C++ codebase.)

Worth wrapping when the need arises (temporary instrumentation, not committed):
`metal::flush()` (how long do CPU reads stall?), one decode step in
`DecodeParityLLM`-style harnesses, and `gemm_s8` host encode. Apple positions signposts
as cheap-when-disabled; cost not measured here.

## Suggested first session (the recipe, not a report)

1. Capture bs1 decode of Qwen2.5-0.5B int8 under Metal System Trace + Metal GPU
   Counters, 30 s.
2. Check the GPU lane for gaps during decode — quantifies the per-op floor's _visible_
   cost vs the overlap.
3. Read the limiter view for `ct2_gemv_s8` (expected: memory-bound — that's its design
   claim) and `ct2_gemm_s8` on a prefill capture (expected: ALU-bound).
4. Record findings into `METAL_BENCHMARKS.md` and the int8 references per the standing
   measurements-into-docs rule.

### Relevance to the CT2 Metal backend

- Complements the CPU-side methodology in `benchmarking-and-profiling.md`
  (`DISABLED_Benchmark*` in `tests/metal_test.cc`, `CT2_LLM_PROFILE`,
  `--log_throughput` in `cli/translator.cc`); GPU-side _in-process_ timing
  (`gpuStartTime`/`gpuEndTime`) is in `gpu-counters-and-timestamps.md` — Instruments is
  the whole-timeline, zero-code-change alternative.
- Would directly test the overlap claims in `dispatch-overlap-and-perf-model.md`
  (`src/metal/device.mm` async commit model) and the ALU-bound conclusion in
  `int8-gemm-kernel-design.md` (`ct2_gemm_s8` in `src/metal/kernels/kernels_msl.h`).
- For memory-bug hunts use `gpu-capture-and-shader-validation.md` (.gputrace + shader
  validation) instead — Instruments profiles, it doesn't validate.
