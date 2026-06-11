# The DecoderState contract (state keys, lifecycle, who creates and reorders what)

CT2-architecture reference: `DecoderState` — the string→tensor map that carries the KV
cache and encoder memory across decode steps. This is the _contract_ (keys, ownership,
lifecycle); the per-step cache _growth mechanics_ are `attention-and-kv-cache.md` and the
loop that calls `update_state` is `decoding-loop-and-beam-search.md` (cross-refs only).

Source: `include/ctranslate2/layers/decoder.h`, `src/layers/decoder.cc`,
`src/layers/transformer.cc`, `src/layers/attention.cc`, plus the three model families in
`src/models/`. Line numbers verified by read on 2026-06-11 — re-grep symbols before acting.

## The type and the base-class surface

```cpp
// layers/decoder.h:14
using DecoderState = std::unordered_map<std::string, StorageView>;
```

`layers::Decoder` (`decoder.h`) declares: `initial_state(bool iterative_decoding = true)`
(pure virtual, `:23`), the two `update_state` overloads (greedy `:40`, beam `:43`),
`replicate_state(state, beam_size)` (`:49`), and the per-key predicate
`replicate_state(const std::string&)` (`:52`, default `true`, `decoder.cc:68-70`).
`batch_size(state)` is simply `state.begin()->second.dim(0)` (`decoder.cc:64-66`) — every
entry's dim 0 is the (flattened batch×beam) batch.

## The keys — who creates them, who fills them

**`TransformerDecoder::initial_state`** (`layers/transformer.cc:540-561`) creates, per
layer `i`:

- `self_keys_<i>` / `self_values_<i>` — the self-attention KV cache;
- `memory_keys_<i>` / `memory_values_<i>` — the _projected_ encoder-memory K/V, only when
  `_with_encoder_attention`.

All start **empty** — `StorageView(dtype, _device)` with `dtype = output_type()` (`:547`)
and the decoder's device. Empty-ness is the "first step" signal: self-attention moves its
first projection into the cache when `cached_keys->empty()` (`attention.cc:533`+, see
`attention-and-kv-cache.md`). With `iterative_decoding=false` the state is **empty `{}`**
— used for teacher-forced full-sequence forwards (scoring: `language_model.cc:120`,
`sequence_to_sequence.cc:251`, `models/whisper.cc:474`), where no cache is wanted.

**The model adds the encoder handoff** on top (not `initial_state`'s job):

- `memory` — encoder output `[batch, src_time, d_model]`
  (`sequence_to_sequence.cc:252`, `models/whisper.cc:252,475,606`);
- `memory_lengths` — int32 source lengths (`sequence_to_sequence.cc:253`). Whisper sets
  no `memory_lengths` (fixed 1500-frame encoder output).

Decoder-only LMs have neither: their state is exactly `self_keys_*`/`self_values_*`.

## Lifecycle of the cross-attention entries

`TransformerDecoder::decode` reads `state.at("memory")` **only at `step <= 0`**
(`layers/transformer.cc:708-715`). Each layer's cross-attention projects memory → K/V
once, on the empty-cache check in `process_cross_attention` (`attention.cc:368`, fill at
`:383-423`), into `memory_keys_<i>`/`memory_values_<i>`. After the first step the raw
memory is dead weight, so:

```cpp
// layers/transformer.cc:822-825
if (step == 0)
  state.erase("memory");   // projections were cached in the first step
```

`memory_lengths` survives (the mask is rebuilt per step, `:717-727`). Beam interplay:
memory K/V are **not** replicated per beam — `TransformerDecoder::replicate_state(name)`
returns false for `memory*` keys (`layers/transformer.cc:563-566`), and cross-attention
derives `beam_size = queries.dim(0) / cached_keys->dim(0)` instead (`attention.cc:433-434`).
So in a beam search, `self_*` entries have dim 0 = `batch*beam` while `memory_*` keep
dim 0 = `batch`.

## Reordering and shrinking (the Gather contract)

The search loop calls (mechanics in `decoding-loop-and-beam-search.md`):

- greedy batch-shrink: in-place `ops::Gather` over **every** entry (`decoder.cc:33-37`);
- beam reorder: `ops::Gather` by `beam_indices` for replicated keys, by `alive_batches`
  for the non-replicated `memory*` ones (`decoder.cc:39-55`);
- `replicate_state(state, beam_size)` → `repeat_batch` per replicated key (`decoder.cc:57-62`).

Contract for any new state entry: dim 0 must be the batch dim, and it must tolerate
arbitrary gather/shrink between steps. Keys are looked up by exact name per layer at
`layers/transformer.cc:760-767` — a typo'd key is a `std::out_of_range` from `state.at`.

## Shapes, dtype, device

Self-attention caches are `[batch, num_heads_kv(replicated to num_heads), time, d_head]`
growing on dim 2 — except the merged MQA layout caches on dim 1, and
`FlashMultiHeadAttention` uses `[batch, time, heads, d_head]` with `_cache_time_dim = 1`
and 512-row preallocated chunks (`layers/flash_attention.cc:13,71-95`; see
`flash-attention-integration.md`). Dtype is the decoder's `output_type()` — fp16 cache
for an int8_float16 model (activations stay float; only weights are int8). Device is the
decoder's; nothing in the contract is device-specific.

## Extras

- **`DecoderStateCache`** (`decoder.h:102-109`, `decoder.cc:142-151`) — a mutex-guarded
  prompt→state map used by `static_prompt` caching; `language_model.cc:135` `copy_state`
  deep-copies a cached state into a fresh one per batch (details in
  `generator-and-language-model.md`).
- Whisper reuses the plain `TransformerDecoder` state contract unchanged — its only
  specialization is placing `memory` from the audio encoder (`models/whisper.cc:251-252`).

---

### Relevance to the Metal backend

- State tensors live on `Device::METAL` as real `MTLBuffer`s (created empty by
  `initial_state`, first filled by the attention move/concat) — the cache stays
  GPU-resident across the whole decode.
- Every lifecycle operation is an op that already has a Metal route: the per-step growth
  is `Concat`, reorder/shrink is `Gather`, beam replication is `Tile` — no
  state-specific Metal code exists or is needed.
- The dtype rule matters for int8: caches are fp16/fp32 activations, so int8 quantization
  never touches `DecoderState` — only weights. RSS wins from int8 come from weights, not
  the KV cache.
