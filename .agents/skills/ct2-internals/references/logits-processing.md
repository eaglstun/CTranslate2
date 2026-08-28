---
title: "Logits processing (LogitsProcessor, DisableTokens, Whisper's timestamp rules)"
summary: >-
  CT2's logit-manipulation machinery between the decoder forward and token
  selection: the thin LogitsProcessor abstract class (apply with
  step/logits/DisableTokens/sequences/batch_offset/prefix, plus apply_first
  ordering and get_batch_index/get_sample_begin helpers), and the
  DisableTokens collector that writes float lowest directly for CPU logits but
  accumulates a sorted-unique flat-index list flushed by one indexed_fill for
  non-CPU devices. It enumerates the concrete processors (RepetitionPenalty,
  NoRepeatNgram, SuppressTokens, SuppressTokensBegin, SuppressSequences),
  notes min_length is a separate apply_min_length free function, and gives the
  make_logits_processors ordering. Whisper is the heaviest consumer:
  suppress_tokens/suppress_blank map to disable_ids, GetNoSpeechProbs is a
  read-only apply_first processor, and ApplyTimestampRules does real per-step
  LogSoftMax tensor math. On Metal, DisableTokens takes the non-CPU
  indexed_fill branch and ApplyTimestampRules upcasts fp16 log-probs to fp32.
semantic_id: "yxy45gnrom2b8a1dzMkrnbcuIWuTwosDDhewGxyvRlYOr3kGr4-tXo8w2o0u461aiC2s0sxyrQ5-vdzXjrz9pQ"
---

CT2-architecture reference: the logit-manipulation machinery that runs between the
decoder forward and token selection. The hook point is in the decode loop
(`decoding-loop-and-beam-search.md`); this file owns the processors themselves.

Source: `include/ctranslate2/decoding_utils.h`, `src/decoding_utils.cc`,
`src/decoding.cc` (ordering + min_length), `src/models/whisper.cc` (the heavy consumer).
Lines verified by read on 2026-06-11.

## The abstraction (it's real, but thin)

`LogitsProcessor` (`decoding_utils.h:74-103`) is an abstract class with one method:

```cpp
virtual void apply(dim_t step, StorageView& logits, DisableTokens& disable_tokens,
                   const StorageView& sequences,            // tokens generated so far
                   const std::vector<dim_t>& batch_offset,  // alive-row -> original batch id
                   const std::vector<std::vector<size_t>>* prefix) = 0;
```

plus `apply_first()` (`:78`, default false) which only controls ordering, and two
protected helpers: `get_batch_index` (`:90`, maps a beam-flattened row through
`batch_offset` — needed because the batch shrinks) and `get_sample_begin` (`:97`, the
first non-prefix step for a row). `sequences` is `[batch(*beam), time]` int32 and is
**only maintained when processors exist** (greedy: `decoding.cc:886-893`; beam:
`alive_seq` is merged batch×beam before the processor loop, `:519-524`).

A processor can act two ways: mutate `logits` directly (RepetitionPenalty does), or
register token bans on the shared **`DisableTokens`** collector.

## `DisableTokens` — batched masking, one fill at the end

`decoding_utils.h:36-71` + `decoding_utils.cc:10-35`. Constructed per step around the
logits. `add(batch_id, token_id)`:

- **CPU logits**: writes `float lowest` directly into the buffer (`decoding_utils.h:44-46`).
- **non-CPU**: accumulates a _sorted unique_ flat-index list (`:48-53`); `apply()`
  (`decoding_utils.cc:19-35`) then does one `primitives<D>::indexed_fill` with the index
  tensor moved to the device.

The loop calls `disable_tokens.apply()` once after all processors
(`decoding.cc:527`/`:866`), so N bans cost one kernel.

## What actually exists (real names)

All in `decoding_utils.cc`; the prompt-level names map 1:1 to options in
`DecodingOptions` (`decoding.h:139-166`):

- **`RepetitionPenalty`** (`:38-67`, option `repetition_penalty`) — gathers the scores of
  previously generated tokens (`ops::Gather` with batch_dims=1) and rescales them via
  `primitives<D>::penalize_previous_tokens`. Mutates logits; ignores `DisableTokens`.
- **`NoRepeatNgram`** (`:70-105`, option `no_repeat_ngram_size`) — host-side
  `std::search` over `sequences` for earlier occurrences of the current (n-1)-gram;
  disables each historical continuation token.
- **`SuppressTokens`** (`:156-169`, option `disable_ids`) — unconditional ban list.
- **`SuppressTokensBegin`** (`:172-194`, option `disable_ids_begin`) — bans only at
  `step == get_sample_begin(...)` (the first unconstrained step).
- **`SuppressSequences`** (`:108-153`, option `disable_sequences`) — ctor splits
  length-1 sequences into an always-ban list; longer ones ban the final token when the
  generated suffix matches the first n-1 tokens (`std::equal`, `:144-150`).

**`min_length` is NOT a LogitsProcessor** — it's the free function `apply_min_length`
(`decoding.cc:384-408`), run _before_ the processor loop, banning `end_ids` until the
minimum (prefix-aware when `return_prefix` is false).

## Ordering — `make_logits_processors` (`decoding.cc:1090-1120`)

1. user-supplied processors with `apply_first() == true`;
2. `RepetitionPenalty`; 3. `NoRepeatNgram`; 4. `SuppressTokens`;
3. `SuppressTokensBegin`; 6. `SuppressSequences`;
4. remaining user processors (`apply_first() == false`).

Within the step: `apply_min_length` → this list → `DisableTokens::apply()` →
(log-softmax) → sampler.

## Whisper: the heaviest consumer (`src/models/whisper.cc`)

Whisper plugs in entirely through this machinery: `options.suppress_tokens` becomes
`disable_ids` (with `-1` expanding to the model config's `suppress_ids`,
`src/models/whisper.cc:310-317`); `suppress_blank` becomes `disable_ids_begin` (`:319-322`);
`GetNoSpeechProbs` (`:197+`) is a _read-only_ processor with `apply_first() = true`
(`:209-211`) that softmaxes the first-step logits to extract P(no-speech).

**`ApplyTimestampRules`** (`src/models/whisper.cc:731`, registered `:332-341`) is a model-local
`LogitsProcessor` subclass and by far the heaviest processor: per step and per batch row
it (a) always bans `<|notimestamps|>`; (b) at the sample start forces a timestamp and
caps it at `max_initial_timestamp`; (c) enforces timestamps-in-pairs by inspecting the
last/penultimate generated tokens; (d) forbids non-monotonic timestamps by scanning the
sequence backwards; and (e) — the expensive part — calls `disable_tokens.apply()`
_mid-processor_, runs an `ops::LogSoftMax` over the full logits, and compares the
summed timestamp probability mass against the max text-token prob
(`should_sample_timestamp`) to force timestamp sampling. It is the one processor that
does real tensor math every step rather than just registering bans.

---

### Relevance to the Metal backend

- On Metal, `DisableTokens` takes the **non-CPU branch** (logits device ≠ CPU), so bans
  go through `primitives<D>::indexed_fill` — which the METAL dispatch case binds to the
  CPU primitive over unified memory. Correct, but it means logits must be coherent
  (flushed) before the write; the coherence point comes from the dispatch path
  (`apple-silicon` skill, `dispatch-overlap-and-perf-model.md`).
- Processors that read `sequences` (`NoRepeatNgram`, `SuppressSequences`,
  `ApplyTimestampRules`) do host-side pointer reads — free on unified memory.
- `ApplyTimestampRules` carries an explicit Metal accommodation: fp16 log-probs are
  upcast to fp32 before `should_sample_timestamp` because that reduction's dispatch has
  no fp16 case off-CUDA (`whisper.cc`, the `CT2_WITH_METAL` block inside `apply`) — part
  of the Whisper-on-Metal bringup.
- The per-step `LogSoftMax` inside `ApplyTimestampRules` runs on the Metal softmax kernel
  — one more tiny op per decode step in the regime documented in `apple-silicon`.
