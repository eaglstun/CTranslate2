---
title: "The decoding loop & beam search (greedy/beam step structure, batch shrinking)"
summary: >-
  The token-generation driver in src/decoding.cc: decode() sets up prefix
  splitting, search strategy (GreedySearch when beam_size==1 and
  prefix_bias_beta==0 else BeamSearch), sampler, and logits processors. Walks
  the greedy per-step loop (decoder forward, DisableTokens/min-length/logits
  processors, in-place LogSoftMax only when return_scores, sampling with
  prefix override, token-append and finish check, then batch shrinking that
  drops finished rows via non_finished_index and an ops::Gather over every KV-
  cache entry). Covers beam search specifics: batch*beam flat layout,
  num_candidates=beam_size*2, expand_after_first_step, broadcast score
  accumulation, unflatten_ids beam-origin tracking, length/coverage penalties
  in finalize_hypothesis_score, and hard vs biased (prefix_bias_beta) prefix
  modes. The shrinking-m batch dimension and on-GPU Gather cache reorder are
  the Metal-relevant hooks.
---

CT2-architecture reference: the token-generation driver in `src/decoding.cc` — what a
decode _step_ actually does between the decoder forward and the next input token, and how
hypotheses finish and leave the batch. This is the orchestration above
`attention-and-kv-cache.md` (which owns the per-step op flow _inside_ the decoder).

Source: `src/decoding.cc`, `include/ctranslate2/decoding.h`. Line numbers verified by
read on 2026-06-11 — re-grep symbols before acting.

## Entry point: `decode()`

`decode()` (`decoding.cc:1306`) is what models call (e.g. `whisper.cc`,
`sequence_to_sequence.cc`). It takes a `layers::Decoder`, a `DecoderState` (the KV cache
map), per-batch `start_tokens`, `end_ids`, and a `DecodingOptions` struct
(`decoding.h:139-166`). It:

1. validates options (`validate_decoding_options`, `:1023`) — includes the
   top-p × top-k device cap (see `sampling-and-topk.md`);
2. splits each start sequence into a single start id + a **prefix** (`split_start_tokens`,
   `:999`) — everything after token 0 becomes `prefix_ids` (the "hard prompt");
3. builds the three pluggable pieces: `make_search_strategy` (`:1076` — `GreedySearch`
   when `beam_size == 1 && prefix_bias_beta == 0`, else `BeamSearch`), `make_sampler`
   (`:1066`), `make_logits_processors` (`:1090`, see `logits-processing.md`);
4. calls `SearchStrategy::search(...)` (interface at `decoding.h:32-52`), or per-example
   `decode_alternatives` (`:1122`) for `return_alternatives` mode.

## The greedy per-step loop (`GreedySearch::search`, `decoding.cc:732`)

Per step (`:844-971`), the canonical sequence is:

1. **decoder forward** (`:846`) — `decoder(start_step + step, sample_from, state, &logits)`
   produces `[cur_batch_size, vocab]` logits.
2. **token disabling** — a `DisableTokens` view over the logits (`:852`);
   `apply_min_length` (`:855`, defined `:384`) blocks `end_ids` until `min_length`; then
   every `LogitsProcessor` runs (`:863`); `disable_tokens.apply()` (`:866`) commits.
3. **log-softmax only if needed** — `ops::LogSoftMax` runs _in place_ only when
   `return_scores` (`:876-878`); otherwise raw logits go to the sampler (argmax is
   monotone-invariant).
4. **sampling** — `sampler(log_probs, best_ids, best_probs)` (`:880`); prefix override via
   `update_sample_with_prefix` (`:881`, defined `:21` — forces the prefix token while
   `step < prefix.size()`).
5. **token append + finish check** (`:899-953`) — hypotheses are plain
   `std::vector<size_t>` on the host; `is_finished` on EOS or `is_last_step` (`:377`).
   The streaming `_callback` (`DecodingStepResult`, `decoding.h:21`) fires here (`:925`)
   and can force-finish.
6. **batch shrinking** (`:962-970`) — finished rows are dropped: `non_finished_index`
   gathers `sample_from`/`alive_seq`, and `decoder.update_state(state, alive)` runs
   `ops::Gather` over **every cache entry** (`src/layers/decoder.cc:33-37`). This is why
   backends see a _shrinking batch dimension_ over a long decode.

`num_hypotheses > 1` with greedy = replicate the batch `num_hypotheses`× up front and
merge results (`:753-814`).

## Beam search (`BeamSearch::search`, `decoding.cc:424`)

Same skeleton with beam bookkeeping. The flat layout is `batch*beam` in dim 0
(`merge_batch_beam`/`split_batch_beam`, `decoding_utils.h:11-23`; `gather_beam_flat`,
`decoding.cc:15`).

- **Candidates**: `num_candidates = beam_size * 2` (`:449`) so EOS hits can be backfilled
  from secondary candidates (`:638-645`). `_max_candidates = round(beam_size * patience)`
  (`:355`, ctor `:420`).
- **First-step optimization**: `expand_after_first_step` (`:453`, CPU only) runs step 0 at
  batch size 1× before replicating state to `beam_size`× (`decoder.replicate_state`,
  `:471`); otherwise scores start as `[0, -inf, -inf, ...]` per beam
  (`initialize_beam_scores`, `:83`).
- **Score accumulation**: after `LogSoftMax` (`:545`), the running beam scores are added
  by broadcast (`add_depth_broadcast`, `:550-556`), log-probs are flattened to
  `[batch, beam*vocab]` (`:559`), and the sampler picks `num_candidates` (`:562`).
- **Beam origin tracking**: `unflatten_ids` (`:95`) decomposes flat ids into
  (beam, word) and emits `gather_indices`; `append_step_output` (`:117`) grows
  `alive_seq` (a `Concat` on dim 2) with a gather-reorder first; `decoder.update_state`
  (`:710`) gathers the whole KV cache by `gather_indices` — **the per-step cache reorder
  is an `ops::Gather` over every state tensor** (`decoder.cc:39-55`).
- **Finish & penalties**: a hypothesis registers when its token is EOS or the step limit
  hits (`:620-635`). Scores are finalized in `finalize_result` (`:237`) →
  `finalize_hypothesis_score` (`:189`): `score /= length^length_penalty` (`:194`) plus the
  coverage penalty from accumulated attention (`compute_coverage_penalty`, `:176`).
  Early exit (`allow_early_exit`, `:457`) only when both penalties are 0; else a batch
  needs `_max_candidates` finished hypotheses (`:651-657`).
- **Batch shrinking**: same pattern as greedy — `non_finished_index` → `keep_batches` →
  gather of ids/scores/seq/attention (`:691-706`) and the decoder state (`:710`).

### Prefix handling: hard vs biased

Two modes, chosen at `:483`: `use_hard_prefix` (note: the real name is `use_hard_prefix`,
not "use_hard_prompt") forces prefix tokens via `update_sample_with_prefix` (`:567-577`,
which also penalizes EOS in secondary beams at the first free step — issue #277
workaround, `:50-57`). With `prefix_bias_beta > 0`, `BiasedDecoder::decode`
(`:263-323`) instead _interpolates_: per beam still on-prefix, the softmax is discounted
by `(1-beta)` and `beta` is added to the prefix token's probability before the log
(`:302-318`); divergence is tracked per beam (`get_beams_divergence_from_prefix`, `:325`)
and biasing stops once all beams diverged (`:715-716`).

---

### Relevance to the Metal backend

- **This loop is the tiny-op regime.** One iteration = decoder forward (the dozen-plus ops
  in `attention-and-kv-cache.md`) + LogSoftMax + TopK + several Gathers, at `m = batch`.
  The per-op API floor analysis is the `apple-silicon` skill,
  `dispatch-overlap-and-perf-model.md`.
- **The per-step cache reorder is `ops::Gather`**, and Gather _does_ have a Metal kernel
  (`ct2_gather_bytes` in `src/metal/kernels/kernels_msl.h`, routed at
  `src/ops/gather.cc:90`) — beam reorder and batch shrinking stay on-GPU.
- All hypothesis bookkeeping (`results`, `alive_seq` reads, `topk_ids.at<int32_t>`) is
  host-side; sampled ids land on CPU every step (the Sampler contract — see
  `sampling-and-topk.md`). On Metal this is cheap reads over unified memory, but it is a
  per-step CPU↔GPU sync point.
- Batch shrinking means a backend must handle _shrinking_ `m` between steps without
  reallocation churn — the `StorageView` resize contract (`storageview.md`) is what makes
  that cheap.
