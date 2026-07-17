---
title: "Profiling infrastructure"
summary: >-
  Describes the integrated profiler behind --log_profiling: a compile-time-
  gated (ENABLE_PROFILING -> CT2_ENABLE_PROFILING) RAII scoped-timer system
  where PROFILE(NAME) expands to a zero-cost do{}while(0) when off and
  init_profiling throws loudly rather than producing empty output. In a
  profiling build PROFILE declares a stack ScopeProfiler (carried by ~55 files
  as the first line of an op's operator()); nesting is tracked via a
  thread_local current_scope whose destructor subtracts elapsed time from the
  parent's self-time, aggregation is by name across threads under a mutex, and
  both constructor and destructor call synchronize_stream so async GPU work is
  charged correctly but pipelining is serialized. dump_profiling sorts by
  self-time and prints %self/%total/%cum/name/self_ms with the denominator
  multiplied by num_threads. It also documents --log_throughput (generated
  target tokens of the best hypothesis over end-to-end wall time) and warns
  that on Metal the per-scope sync becomes a full metal::synchronize()
  destroying async-commit overlap, so trust the ranking not the magnitudes;
  profiling is C++/CLI only, not bound in Python.
---

The integrated profiler behind `--log_profiling`: a compile-time-gated, RAII scoped-timer
system with per-name aggregation, plus the `--log_throughput` tokens/sec metric. These two
CLI flags are the numbers maintainers use to gate performance changes (CLAUDE.md).

**Sources (all citations from real lines):**

- `include/ctranslate2/profiler.h` (the `PROFILE` macro + public API)
- `src/profiler.cc` (both builds: stub and real)
- `cli/translator.cc` (the CLI surface), `include/ctranslate2/translator.h` (`ExecutionStats`)
- `CMakeLists.txt:21,32-34` (`ENABLE_PROFILING` → `-DCT2_ENABLE_PROFILING`)

## 1. Compile flag and zero-cost when off

CMake `ENABLE_PROFILING` (default OFF, `CMakeLists.txt:21`) defines
`CT2_ENABLE_PROFILING` (`:32-34`). Without it:

- `PROFILE(NAME)` expands to `do {} while(0)` (`profiler.h:27`) — **zero cost**, the name
  string isn't even constructed.
- `init_profiling` **throws** "CTranslate2 was not compiled with profiling support"
  (`profiler.cc:9-12`); `dump_profiling` silently no-ops (`:14-15`). So `--log_profiling`
  on a normal build fails loudly at startup, not quietly with empty output.

`PROFILE_FUN` (`profiler.h:30`) is the file:function variant and survives both builds.

## 2. PROFILE() mechanics (profiling build)

`PROFILE("Add")` declares a stack `ScopeProfiler` (`profiler.h:12`); ~55 files in `src/`
carry one, typically as the first line of an op's `operator()` (`src/ops/add.cc:13`,
`concat.cc:18`, …). Mechanics in `src/profiler.cc`:

- A single global `Profiler` exists between `init_profiling`/`dump_profiling`
  (`profiler.cc:128-140`); if it's absent every `ScopeProfiler` is a no-op (`:147-148`).
- **Nesting** is tracked with a `thread_local ScopeProfiler* current_scope` (`:144`); the
  destructor adds the elapsed time to its own name and _subtracts_ it from the parent's
  self-time (`add_scope_time`, `:68-79`) — so the table separates time-in-scope from
  time-in-scope-and-callees.
- **Aggregation is by name across all threads** (mutex-guarded map, `:47-48,71`): two ops
  with the same `PROFILE` string accumulate together ("Times of profilers created in
  different threads with the same name are accumulated", `profiler.h:14`).
- **Device sync**: both constructor and destructor call
  `synchronize_stream(profiler->device())` (`:151,159`) so async GPU work is charged to
  the op that issued it. This is also why profiling distorts pipelined backends: it
  serializes the stream at every scope boundary.

## 3. Output format

`dump_profiling(os)` (`:81-124,135-140`) sorts by self-time descending and prints one row
per name:

```text
 %self  %total  %cum  name  self_ms
```

i.e. self-time ratio, self+callees ratio, running cumulative self ratio, the scope name
(left-padded to the longest), and self-time in ms (`:115-122`). The denominator is wall
time since `init_profiling` **multiplied by `num_threads`** (`:85-87`) — pass the worker
count (the CLI passes `translator_pool.num_replicas()`, `cli/translator.cc:180`) or the
percentages are inflated. `dump_profiling` destroys the profiler after printing (`:138`).

## 4. CLI surface and the throughput metric

`cli/translator.cc` (installed as `ct2-translator`):

- `--log_profiling` (`:24`) → `init_profiling(device, num_replicas)` before the run
  (`:178-180`) and `dump_profiling(std::cerr)` after (`:242-243`).
- `--log_throughput` (`:22`) → prints `stats.num_tokens / (stats.total_time_in_ms / 1000)`
  to stderr (`:252-253`). `ExecutionStats` (`translator.h:9-13`) is filled by the
  file-translation writer: `num_tokens += hypotheses[0].size()` per example
  (`translator.h:178-188`) — i.e. **generated target tokens of the best hypothesis**, over
  end-to-end wall time including batching/IO (`:190-208`). Higher is better; compare runs
  on identical inputs only, since tokenization and output length change the numerator.

Python exposure: none — `init_profiling`/`dump_profiling` are not bound in
`python/cpp/module.cc`; profiling is a C++/CLI-side tool.

## 5. Complementing per-op GPU timing

This profiler answers "where does wall time go across the whole engine" with op-name
granularity, at the price of a stream sync per scope. It cannot separate kernel time from
dispatch/commit overhead inside one op — for that the Metal work uses targeted
micro-benchmarks (`tests/metal_test.cc` `DISABLED_Benchmark*` cases) and signpost/capture
tooling; see the apple-silicon skill's `benchmarking-and-profiling.md`. Use both: this
table to find the hot op, the GPU tools to explain it.

### Relevance to the Metal backend

- The per-scope `synchronize_stream` (`profiler.cc:151,159`) maps to a full
  `metal::synchronize()` on Metal — profiling **destroys the async-commit overlap** that
  the backend's perf depends on, so absolute Metal numbers under `--log_profiling` are
  pessimistic; trust the _ranking_, not the magnitudes.
- A `metal::`-routed op still gets charged correctly (the `PROFILE` sits in the shared
  `operator()` before the device check — e.g. `softmax.cc` profiles before routing).
- For decode-loop analysis prefer `--log_throughput` deltas + the metal_test
  micro-benchmarks; reserve this profiler for CPU-reference-path hotspot hunting.
