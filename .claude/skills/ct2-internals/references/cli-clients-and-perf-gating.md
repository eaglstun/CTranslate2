# CLI client (ct2-translator) & the perf-gating workflow

The one CLI client — `cli/translator.cc`, built as `ct2-translator`
(`cli/CMakeLists.txt`: `OUTPUT_NAME ct2-translator`; `BUILD_CLI=ON` by default, needs
the `cxxopts` submodule) — and the maintainer recipe for gating a perf change with it.
`--log_throughput`/`--log_profiling` _mechanics_ (ExecutionStats, the profiler) are
`profiling-infrastructure.md` §4; this card owns the flag surface, the streaming loop,
and the recipe.

**Sources (all citations from real lines):**

- `cli/translator.cc` (the whole client is 257 lines)
- `include/ctranslate2/translator.h` (`translate_raw_text_file`, `consume_stream`)
- `src/translator.cc` (`translate_text_file` → space tokenization)
- `include/ctranslate2/replica_pool.h` (`consume_batches`)

## 1. The flag surface that matters

`cli/translator.cc` is a thin cxxopts shell over `Translator::translate_text_file` /
`score_text_file`. Groups (all defaults shown are the real cxxopts defaults):

- **General**: `--task translate|score` (`cli/translator.cc:18`), `--seed`,
  `--log_throughput` (`:22`), `--log_profiling` (`:24`).
- **Device**: `--device` (default `cpu`; help says "cpu, cuda, auto" at `:33` but
  `str_to_device` also accepts `metal` — `src/devices.cc:27`), `--device_index`
  (comma list), `--inter_threads` (replicas, default 1), `--intra_threads`
  (`parallelism-and-thread-config.md`), `--cpu_core_offset`.
- **Model**: `--model` (required, `:122-123`), `--compute_type` (default `default`,
  full list at `:43`), plus `--cpu_compute_type`/`--cuda_compute_type` overrides —
  note the override switch (`:133-142`) has only CPU and CUDA cases, so on Metal the
  plain `--compute_type` is the one that applies.
- **Data**: `--src`/`--tgt`/`--out` (stdin/stdout when unset, `:157-176`),
  `--batch_size` (default 32), `--batch_type examples|tokens`, `--read_batch_size`
  (0 = auto), `--max_queued_batches` (pool backpressure), `--max_input_length` (1024).
- **Translation**: the full `TranslationOptions` mirror (`:188-214`) — `--beam_size 2`,
  `--patience`, sampling trio, `--n_best`, `--with_score`, length/coverage/repetition
  penalties, `--no_repeat_ngram_size`, `--disable_unk`, `--suppress_sequences`
  (comma-separated sequences, escaped-space-separated tokens, split at `:210-214`),
  `--end_token`, `--prefix_bias_beta`, `--max/min_decoding_length`,
  `--replace_unknowns`, `--use_vmap`. Semantics/enforcement: the option table in
  `translator-and-seq2seq.md`.
- **Scoring**: `task=score` requires both `--src` and `--tgt` (`:224-226`), takes only
  `--max_input_length` (`:228-229`), and `--with_tokens_score` appends per-token
  log-probs; output lines are `normalized_score ||| tokens [||| t1 t2 …]`
  (`include/ctranslate2/translator.h:270-282`).

The two gating flags: `--log_profiling` → `init_profiling(device, num_replicas)` /
`dump_profiling(std::cerr)` (`:178-180`, `:242-243`; **throws at startup on a
non-`ENABLE_PROFILING` build**), and `--log_throughput` → prints
`stats.num_tokens / seconds` to stderr (`:252-253`). The numerator is task-dependent:
generated best-hypothesis tokens for translate
(`include/ctranslate2/translator.h:182`), scored target tokens for score (`:274`).

## 2. The streaming loop (what a throughput number actually times)

`translate_text_file(istream, ostream, …)` tokenizes by **whitespace**
(`split_tokens`/`join_tokens`, `src/translator.cc:108-127`) and delegates to the
header-template `translate_raw_text_file` (`include/ctranslate2/translator.h:165-210`),
which wall-clocks the whole pipeline (`t1`/`t2` around `consume_stream`,
`:190-208`) — so IO, tokenization, and batching are inside the denominator. Structure:

- `consume_stream` (`translator.h:314-344`) builds a `TextLineReader` per input
  (`ParallelBatchReader` zipping source+target when `--tgt` is set, `:325-331`).
- `consume_batches` (`include/ctranslate2/replica_pool.h:192`) is the prefetching
  driver: read `read_batch_size` examples (auto default = `max_batch_size * 16`,
  `:211`), `post_examples` onto the pool (length-sorting and splitting into
  `--batch_size` units happen there — `batching-and-length-sorting.md`), park the
  futures in a queue, and drain **in submission order** — non-blocking
  (`wait_for(0s)`) while reading, blocking at EOF (`pop_results`) — so output order
  always matches input order while compute overlaps reading.
- The writer lambda (`translator.h:178-188`) accumulates `ExecutionStats` and prints
  `n_best` hypotheses (`score ||| text` per line with `--with_score`).

## 3. Worked gating recipe (Apple Silicon flavor)

```bash
cmake .. -DWITH_MKL=OFF -DWITH_ACCELERATE=ON -DOPENMP_RUNTIME=NONE && make -j ct2-translator  # baseline + change builds
head -2000 wmt_test.txt > /tmp/fixed_input.txt                     # fixed data, big enough to amortize warmup
./cli/ct2-translator --model "$MODEL" --src /tmp/fixed_input.txt --out /dev/null \
    --device metal --compute_type float16 --batch_size 32 --beam_size 2 \
    --log_throughput 2>&1 >/dev/null | tail -1                     # tokens/sec on stderr
```

Rules that make the comparison valid: identical input file, flags, and model dir for
both builds (output length changes the numerator — never compare across `--beam_size`
or sampling settings); run ≥3 times and record the spread, not one number (standing
rule: measurements go into the doc with date/machine/run-count); keep `--out /dev/null`
so terminal IO doesn't pollute the timing. To find _where_ a regression lives, rebuild
with `-DENABLE_PROFILING=ON` and add `--log_profiling` — but read
`profiling-infrastructure.md` first: the per-scope stream sync makes the absolute
numbers pessimistic on async backends; gate on `--log_throughput`, _locate_ with the
profile.

There is no generation/whisper CLI — `cli/` contains only `translator.cc`, so gating a
decoder-only or Whisper change means a small Python driver over the wheel (e.g.
`tests/downstream/drivers/`) or the gtest micro-benchmarks.

### Relevance to the Metal backend

- `--device metal` works despite the help text (`str_to_device`, `src/devices.cc:27`);
  there is no `--metal_compute_type` override — use `--compute_type` directly.
- The file loop's read-ahead (`max_batch_size * 16`) keeps the pool saturated, so a
  Metal prefill-regime measurement reflects steady-state GPU throughput, not
  one-batch latency; small inputs measure warmup (pipeline caches, MPS object cache).
- `METAL_BENCHMARKS.md` numbers were gated exactly this way: same binary, fixed input,
  `--log_throughput` deltas; the metal_test `DISABLED_Benchmark*` cases cover the
  per-op level this CLI can't see.
