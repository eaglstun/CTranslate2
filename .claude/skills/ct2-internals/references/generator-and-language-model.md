---
title: "Generator & language models (decoder-only generate/score/forward surface)"
summary: >-
  Maps the decoder-only language-model runtime: the
  LanguageModel/SequenceGeneratorReplica/DecoderReplica class tree (adding
  vocabulary and a shared DecoderStateCache), the async-only Generator pool
  surface (generate_batch_async, score_batch_async, forward_batch_async with
  batch_id restoration), and the absence of a C++ generate_tokens versus the
  Python streaming path. It details DecoderReplica::run_generation's prompt
  handling: options mapping, static-prompt forward-once-then-copy_state
  (memoized in DecoderStateCache when cache_static_prompt), per-example
  common-prefix prefill in one batch forward, then decode(). Scoring is a non-
  iterative teacher-forced forward plus LogSoftMax and Gather with no decode
  loop. For Metal it notes run_generation straddles the prefill regime (GEMM-
  heavy, Metal wins ~2.6x) and the tiny-op decode regime, with KV cache
  staying device-resident; qwen_driver.py is the canonical consumer.
semantic_id: "Txy64wnqs22TwWlNQuNLnac4p-qSw7sBClViOVa8RnZqT9smza-JXq84fg01729bBm00-s_7zY5evwyTR7h95Q"
---

CT2-architecture reference: how a decoder-only LM runs — the `Generator` pool surface,
the `SequenceGeneratorReplica`/`DecoderReplica` class tree, prompt prefill (static
prompt vs per-example prompt), and scoring (forced decode + log-prob gather). The
search loop itself is `decoding-loop-and-beam-search.md`; the encoder-decoder
counterpart is `translator-and-seq2seq.md`.

Source: `src/models/language_model.cc`, `src/generator.cc`, `src/scoring.cc`,
`include/ctranslate2/models/language_model.h`, `include/ctranslate2/generator.h`,
`include/ctranslate2/generation.h`. Line numbers verified by read on 2026-06-11 —
re-grep symbols before acting.

## Class tree (`language_model.h`)

- `LanguageModel : Model` (`language_model.h:14`; `language_model.cc:8-36`) — adds the
  vocabulary (loaded from config `unk/bos/eos_token`, `:21-36`) and a shared
  `layers::DecoderStateCache` (`:9`) used by static-prompt caching.
- `SequenceGeneratorReplica : ModelReplica` (`language_model.h:33`) — the abstract
  runnable surface. Public `score`/`generate`/`forward` (`language_model.cc:39-102`)
  do device setup and delegate to the virtuals `run_scoring`/`skip_scoring`/
  `run_generation`/`forward(ids, lengths)` (`language_model.h:62-79`).
- `DecoderReplica : SequenceGeneratorReplica` (`language_model.h:87`) — the concrete
  implementation owning a `layers::Decoder` (`language_model.cc:105-110`). Created by
  `TransformerDecoderModel::as_sequence_generator` (`src/models/transformer.cc:111-119`).
- `SequenceEncoderReplica`/`EncoderReplica` (`language_model.h:114`/`:146`;
  `language_model.cc:302-403`) — the BERT-style encoder surface (last_hidden_state +
  optional pooler), sharing `LanguageModel` but not the generation machinery.

Note: `DecoderReplica` is **not** shared with seq2seq — the encoder-decoder path has
its own `EncoderDecoderReplica` (`sequence_to_sequence.h`). What the two actually share
is downstream: the free functions `decode()` (`src/decoding.cc`) and `score_sequences`
(`src/scoring.cc`), which both replica types call.

## The pool: `Generator` (`generator.h:10`, `src/generator.cc`)

`Generator : ReplicaPool<models::SequenceGeneratorReplica>`. The C++ surface is
async-only:

- `generate_batch_async` (`generator.cc:7`) — `post_examples` splits work into batches
  (`max_batch_size`/`BatchType`; see `batching-and-length-sorting.md`) and calls
  `replica.generate`. `restore_batch_ids_in_callback` (`generation.h:120`) rewraps the
  user callback so `GenerationStepResult.batch_id` maps back to the caller's example
  order after batch reordering.
- `score_batch_async` (`:26`) → `replica.score`.
- `forward_batch_async` (`:43`, 3 overloads) — single full forward returning raw logits
  or log-probs (`SequenceGeneratorReplica::forward`, `language_model.cc:81-102`:
  optional in-place `LogSoftMax`, then `synchronize_stream` before returning).

There is no C++ `generate_tokens`: the Python streaming generator is
`generator_generate_tokens` (`python/ctranslate2/extensions.py:270`), built on
`generate_batch` plus the per-step `callback` (`GenerationOptions.callback`,
`generation.h:77` — `std::function<bool(GenerationStepResult)>`, fired from the greedy
loop; returning `true` force-finishes; greedy-only in practice since beam hypotheses
aren't stable per step).

## Prompt handling in `DecoderReplica::run_generation` (`language_model.cc:149`)

Order of operations before the decode loop:

1. **Options mapping** (`:155-182`) — `GenerationOptions` fields copy 1:1 onto
   `DecodingOptions` (beam/penalties/sampling/num_hypotheses/...);
   `suppress_sequences` → `disable_sequences`, `disable_unk` → `disable_ids`,
   `end_token` resolved via `ResolveEndToken` (`:240`).
2. **Static prompt** (`:187-215`) — a _shared_ system prompt: forwarded once through
   the decoder (`(*_decoder)(0, static_prompt, static_state)`, `:207`), then the
   resulting KV state is copied to every example by `copy_state` (`:135-147` — plain
   copy for batch 1, `ops::Tile` on axis 0 otherwise). With `cache_static_prompt`
   (default true) the state is memoized in the model's `DecoderStateCache` keyed by the
   token ids (`:193-211`), so later batches skip the forward entirely.
   `start_step += static_prompt.size()` (`:214`).
3. **Per-example prompt prefill** (`:217-238`) — only when
   `include_prompt_in_result == false`: the _common_ prefix length
   (`min_prompt_length - 1`) across the batch is prefilled in **one batch forward**
   (`(*_decoder)(start_step, prompt, state)`, `:233`); the leftover per-example tokens
   stay in `start_ids` and are forced inside `decode()` as the hard prefix
   (`split_start_tokens` — see `decoding-loop-and-beam-search.md`). Note the prefill
   entry is just the decoder functor with a 2-D ids tensor — `forward_prompt` is a
   Whisper-specific wrapper name, not a generic API.
4. `decode(*_decoder, state, start_ids, end_ids, decoding_options)` (`:241`), then
   result post-processing: EOS strip (`:253-258`), re-prepending the start token when
   prompts are included and not BOS (`:261-266`).

## Scoring (`score` → `run_scoring` → `score_sequences`)

`score` (`language_model.cc:39`) uses `get_batch_results_helper` with
`skip_scoring` = sequences shorter than 2 tokens (`:129-133`). `run_scoring` (`:113`)
builds a **non-iterative** decoder state (`initial_state(false)`, `:120`) and calls the
shared `score_sequences` (`scoring.cc:6`):

- inputs = `seq[:-1]`, targets = `seq[1:]` (`scoring.cc:26-27`) — one teacher-forced
  batch forward (`:43`), **no decode loop**;
- in-place `LogSoftMax` (`:44`), then `ops::Gather(axis=-1, batch_dims=2)` picks each
  target token's log-prob (`:48`);
- scores move to CPU fp32 (`:50-53`) and are emitted per token from
  `ScoringOptions.offset` onward (`:61-64`).

## Downstream proof

`tests/downstream/drivers/qwen_driver.py` is the canonical minimal `Generator`
consumer: `ctranslate2.Generator(model_dir, device="metal", compute_type=...)` +
`generate_batch(prompt_tokens, max_length=20, sampling_topk=1,
include_prompt_in_result=False)` — the exact surface the int8-Metal validation rides
(golden capture fp16, teacher-forced agreement int8).

---

### Relevance to the Metal backend

- `run_generation` straddles both perf regimes: the static-prompt forward and the
  per-example prefill (`:207`, `:233`) are **prefill** (one big GEMM-heavy forward,
  where Metal wins ~2.6× on Qwen batch-8 fp16), while `decode()` is the **tiny-op
  decode regime** bounded by the per-op API floor.
- `copy_state`'s `ops::Tile` and the prefill forwards keep the KV cache device-resident
  — on Metal the cached state stays in `MTLBuffer`s across the static-prompt cache
  (`DecoderStateCache` holds Metal-resident `StorageView`s).
- `score_sequences` is pure prefill plus one Gather — scoring is the seq2seq/LM
  workload that benefits most from Metal as-is, with a single CPU sync at the
  `scores.to(Device::CPU)` copy (cheap over unified memory).
- The Qwen int8 results (decode GEMV beating fp16, RSS −42%, 92/100 agreement) were
  measured through exactly this path; see the int8-on-Metal memory and `METAL_BENCHMARKS.md`.
