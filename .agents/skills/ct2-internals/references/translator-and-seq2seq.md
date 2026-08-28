---
title: "Translator & seq2seq (TranslationOptions plumbing, encode→decode handoff)"
summary: >-
  Maps the encoder-decoder runtime: the Translator pool posting two streams
  (source and target_prefix) to
  SequenceToSequenceReplica/EncoderDecoderReplica, vocabulary/start-token
  loading (shared_vocabulary, decoder_start_token as BOS/EOS/null), and the
  run_translation flow of tokens-to-ids, encode(), the encode-to-decode
  handoff installing state memory/memory_lengths projected into cross-
  attention K/V at step 0, optional vmap output-layer restriction, decode(),
  and post-processing (EOS strip, attention trimming, replace_unknowns via
  attention argmax). A large table maps each TranslationOptions field to where
  it is copied and enforced (beam_size, patience, penalties, prefix_bias_beta,
  max/min_decoding_length, sampling, use_vmap, etc.). Scoring reuses the
  shared score_sequences teacher-forced forward. For Metal, encode is the
  prefill regime that wins while beam decode is the tiny-op regime;
  nllb_driver.py is the canonical consumer.
semantic_id: "yV6Q4gmrl22TwW1dwvFpvQUoo2uTz7sjClVgG4StwkZiH0kXbaeIXq94W4l4q8Y6pi0I-c_7zY5-vYy7hvBtpQ"
---

CT2-architecture reference: the encoder-decoder runtime path — `Translator` pool →
`SequenceToSequenceReplica`/`EncoderDecoderReplica` → `decode()` — and the practical
"what option does what, enforced where" card. The decode loop internals are
`decoding-loop-and-beam-search.md`; the decoder-only twin is
`generator-and-language-model.md`.

Source: `src/translator.cc`, `src/models/sequence_to_sequence.cc`,
`include/ctranslate2/translation.h`, `include/ctranslate2/models/sequence_to_sequence.h`.
Line numbers verified by read on 2026-06-11 — re-grep symbols before acting.

## The pool: `Translator` (`src/translator.cc`)

`translate_batch_async` (`translator.cc:7`, prefix overload `:15`) posts examples with
**two streams** — stream 0 = source, stream 1 = target_prefix
(`load_examples({source, target_prefix})`, `:22`) — onto
`SequenceToSequenceReplica::translate` via the free function `run_translation` (`:182`,
which also rewraps the step callback with `restore_batch_ids_in_callback`).
`score_batch_async` (`:30`) → `run_scoring` (`:172`) → `replica.score`. The sync
`translate_batch`/`score_batch` (`:45-82`) just `.get()` the futures. `translate_text_file`
/ `score_text_file` (`:84-169`) are stream wrappers (the `ct2-translator` CLI rides them).

## `SequenceToSequenceModel` (vocabularies & start token)

`load_vocabularies` (`sequence_to_sequence.cc:13`): `shared_vocabulary` collapses
source=target (`:20-25`); else `target_vocabulary` + `source_vocabulary` (or
`source_1_vocabulary`… for multi-feature models, `:36-44`). `vmap.txt`, if present,
becomes a `VocabularyMap` (`:52-57`). The **decoder start token** comes from config
(`initialize`, `:60-74`): `decoder_start_token` is BOS, EOS, or null
(user-supplied start tokens).

## translate flow (`EncoderDecoderReplica::run_translation`, `:304`)

`SequenceToSequenceReplica::translate` (`:112`) pads `target_prefix` to batch size and
runs `get_batch_results_helper`: empty-source examples are short-circuited by
`skip_translation` (`:434-470`, fabricates an empty/echoed result), the rest hit
`run_translation`:

1. **Tokens → ids** — `make_source_ids` (`:143`, applies `with_source_bos/eos` per
   source vocabulary) and `make_target_ids(…, is_prefix=true)` (`:168-186`): prefix
   mode prepends `decoder_start_token`, appends **no** EOS, and skips truncation;
   scoring mode appends EOS and allows `max_input_length + 1`.
2. **Encode** — `encode()` (`:216-233`): `make_sequence_inputs` per feature (lengths
   from feature 0) then `(*_encoder)(ids, memory_lengths, memory)`.
3. **Handoff** — the decoder state gets `state["memory"]` and
   `state["memory_lengths"]` (`:325-327`); the decoder projects them into per-layer
   cross-attention K/V at step 0 and erases `memory` afterwards
   (`layers/transformer.cc`, `state.erase("memory")` after step 0).
4. **Vmap restriction** — with `use_vmap` + a loaded map, candidate target ids from
   source n-grams shrink the output layer: `update_output_layer(…, restrict_ids)`
   (`:330-333`) — this is why `TransformerModel::is_packable` excludes `projection`.
5. **Decode** — options copied onto `DecodingOptions` (`:336-366`, table below), end
   token resolved (`:368`), `decode(*_decoder, state, target_ids, end_ids, …)` (`:369`).
   The `target_ids` prefix rides decode's hard-prefix (or `prefix_bias_beta` biased)
   machinery — see `decoding-loop-and-beam-search.md`.
6. **Post-process** (`:379-429`) — EOS strip (popping attention rows in sync,
   `:382-391`); attention vectors trimmed to the real source: resize to padded input,
   drop implicit BOS/EOS, resize to original length (`:402-412`);
   `replace_unknowns` substitutes each UNK with the source token at the attention
   argmax (`replace_unknown_tokens`, `:288-302`).

## Option → enforcement map (`TranslationOptions`, `translation.h:11`)

| Option (default)                              | Copied at (`sequence_to_sequence.cc`) | Enforced in                                                       |
| --------------------------------------------- | ------------------------------------- | ----------------------------------------------------------------- |
| `beam_size` (2)                               | `:337`                                | `make_search_strategy` — 1 → GreedySearch, else BeamSearch        |
| `patience` (1)                                | `:338`                                | `_max_candidates = beam_size * patience` (decoding.cc)            |
| `length_penalty` (1)                          | `:339`                                | `finalize_hypothesis_score`: `score /= length^lp`                 |
| `coverage_penalty` (0)                        | `:340`                                | `compute_coverage_penalty` over accumulated attention             |
| `repetition_penalty` / `no_repeat_ngram_size` | `:341-342`                            | logits processors (`make_logits_processors`)                      |
| `prefix_bias_beta` (0)                        | `:343`                                | >0 switches hard prefix → `BiasedDecoder` (also forces beam path) |
| `max_decoding_length` (256)                   | `:344` → `max_length`                 | `is_last_step` in the search loop                                 |
| `min_decoding_length` (1)                     | `:345` → `min_length`                 | `apply_min_length` (blocks EOS; not a logits processor)           |
| `sampling_topk/topp/temperature`              | `:346-348`                            | `make_sampler` (see `sampling-and-topk.md`)                       |
| `num_hypotheses` (1)                          | `:349`                                | beam result selection / greedy batch replication                  |
| `return_scores/attention/alternatives`        | `:350-353`                            | what `DecodingResult` carries back                                |
| `suppress_sequences` / `disable_unk`          | `:355-362`                            | `disable_sequences`/`disable_ids` → DisableTokens                 |
| `callback`                                    | `:363-366`                            | per-step `DecodingStepResult` (greedy loop)                       |
| `end_token` (model EOS)                       | `:368` `ResolveEndToken`              | `end_ids` finish check                                            |
| `use_vmap` (false)                            | `:331-333` (before decode)            | output-layer restriction via `update_output_layer`                |
| `max_input_length` (1024)                     | `:315-318` (tokenization)             | source/target truncation in `make_*_ids`                          |
| `replace_unknowns` (false)                    | `:352` (forces `return_attention`)    | post-process `:414-418`                                           |

## Scoring (`run_scoring`, `:235`)

Encode the source exactly as translation does, build a **non-iterative** state with
`memory`/`memory_lengths` (`:251-253`), build full targets
`<start> a b c </s>` (`make_target_ids` scoring mode), and call the shared
`score_sequences` (`src/scoring.cc:6` — one teacher-forced forward, `LogSoftMax`,
Gather of target ids; same function the LM path uses). `skip_scoring` (`:263`) handles
empty sources (zero scores) and prefix-less empty targets.

## `TranslationResult` (`translation.h:88-91`)

- `hypotheses` — `num_hypotheses` token sequences (always set).
- `scores` — length-normalized cumulative log-probs; **empty unless `return_scores`**.
- `attention` — per hypothesis, per target token, a float vector over _original_ source
  tokens; **populated only when `return_attention`** (computed also for
  `replace_unknowns`, then cleared at `:421-422`). Its content is the decoder's
  alignment heads (`alignment_layer`/`alignment_heads` wiring —
  `transformer-model-wiring.md`), cross-attention averaged via `ops::Mean`.

## Downstream proof

`tests/downstream/drivers/nllb_driver.py` is the canonical enc-dec consumer:
`ctranslate2.Translator(model_dir, device="metal", compute_type=...)` +
`translate_batch([source], target_prefix=[["fra_Latn"]], beam_size=4)` — NLLB's
target-language forcing is exactly the `target_prefix` → hard-prefix mechanics above
(int8 Metal validated 4/4 in the downstream rig).

---

### Relevance to the Metal backend

- The **encode step is the prefill regime** — one batch forward through the encoder
  stack, GEMM-heavy, where Metal wins; the beam-search decode that follows is the
  tiny-op regime (with beam_size multiplying `m`, slightly friendlier than greedy LM
  decode).
- The encode→decode handoff keeps `memory` device-resident: on Metal the encoder output
  and the projected cross-attention K/V cache stay in `MTLBuffer`s; beam replication
  skips `memory_*` entries (`TransformerDecoder::replicate_state`) so the encoder
  output is never duplicated per beam.
- `return_attention`/`replace_unknowns` pull attention to host-side
  `std::vector<float>` every result — a per-batch device→CPU copy (cheap over unified
  memory, but it forces attention tensors through the alignment-head Gather/Mean ops
  each step).
- The NLLB int8 downstream run is the live proof this whole path works quantized on
  `Device::METAL` (branch fable/int8-metal; see the int8-on-Metal memory).
