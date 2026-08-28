---
title: "Whisper model internals (encoder stem, prompt structure, align/DTW, no_speech)"
summary: >-
  Maps the Whisper model surface (encode/generate/detect_language/align) in
  src/models/whisper.cc and src/layers/whisper.cc. The encoder stem is two
  Conv1D+GELU layers (conv2 stride-2 gives the x2 turning 3000 mel frames into
  1500) then transpose, a learned PositionEmbedding, plain pre-norm
  transformer layers, and a final LayerNorm; the mel-input contract (80 or 128
  mels, 30s window) is derived from weights, and is_encoded lets callers pass
  either mel features or a prior encode() output. generate follows OpenAI's
  prompt layout (previous text, <|startoftranscript|>, language, task,
  optional <|notimestamps|>, forced text), splitting into a one-batch
  forward_prompt prefill (KV cache fill, no logits) plus the decode loop,
  appending ApplyTimestampRules when timestamps are implied. It also covers
  no_speech probability, detect_language (single decoder step, softmax over
  lang_ids), and align (forced decoding plus cross-attention DTW using config
  alignment_heads, MedianFilter, Mean over heads, then CPU negative_dtw). On
  Metal the conv stem is a CPU-reference island needing the conv-weight float
  guard, and ApplyTimestampRules carries the file's one CT2_WITH_METAL
  fp16->fp32 upcast branch.
semantic_id: "yxyy8omKt23Tgb1dQukrvCcuJ-PRzrspCldgGpS8wl5qK9kEjb8sX484Wo0u6-ZapG0s8t7ylY5-_YyTjnBtpw"
---

CT2-architecture reference: the Whisper model surface — `encode` / `generate` /
`detect_language` / `align` — and what each actually computes. The decode driver it
calls is `decoding-loop-and-beam-search.md`; the timestamp logits processor's mechanics
are `logits-processing.md` (pointers below, not duplicated).

Source: `src/models/whisper.cc`, `src/layers/whisper.cc`,
`include/ctranslate2/layers/whisper.h`, `include/ctranslate2/models/whisper.h`,
`python/ctranslate2/specs/whisper_spec.py`. Line numbers verified by read on
2026-06-11 — re-grep symbols before acting.

## Class surface

- `WhisperModel` (`models/whisper.cc:17-46`) — loads the vocabulary (BOS =
  `<|startoftranscript|>`, EOS/UNK = `<|endoftext|>`, `:25-34`); spec revision 3.
- `WhisperReplica` (`:59`) — owns a `layers::WhisperEncoder` + `layers::WhisperDecoder`
  and resolves the special token ids in its ctor (`:65-77`): `_sot_id`, `_eot_id`,
  `_no_timestamps_id`, `_no_speech_id` (with `<|nocaptions|>` fallback, `:70-71`),
  multilinguality = presence of the `""` token (`:72`), and
  `_num_languages = _no_speech_id - _sot_id - 5` derived from the fixed vocab layout
  documented in the comment at `:74-76`.
- `Whisper : ReplicaPool<WhisperReplica>` (`models/whisper.h:158`; `models/whisper.cc:645-728`) — the
  async pool surface: each method `post`s a closure onto a replica and returns
  `std::future`s. `WhisperOptions` (`models/whisper.h:11`) carries beam/sampling options plus
  Whisper-specific ones: `max_length = 448` (`:30`), `return_no_speech_prob` (`:48`),
  `max_initial_timestamp_index = 50` (`:51`), `suppress_blank` (`:54`).

faster-whisper and whisperX are thin Python consumers of exactly these four APIs — the
repo's `tests/downstream/drivers/{faster_whisper,whisperx}_driver.py` rig proves them
against goldens.

## The encoder stem (2×Conv1D+GELU, then a standard transformer)

`WhisperEncoder` (`layers/whisper.cc:8-23`): `_conv1` (stride 1, padding 1) and
`_conv2` (**stride 2**, padding 1), both with a baked-in GELU activation (`:6`,
`:9-12`), then transpose `{0,2,1}` to `[batch, time, depth]` (`:13`, applied `:52`), a
learned `PositionEmbedding` (`:14`), a stack of plain `TransformerEncoderLayer`s
(pre-norm, GELU, `:16-20`), and a final `LayerNorm` (`:21`). Forward order:
conv1 → conv2 → transpose → position add → layers → output norm (`:50-60`).

**Mel-input contract** — enforced at `:28-46`, derived from weights, not hardcoded:

- `input_size() = _conv1.input_size()` (`layers/whisper.h:26`) — the conv1 weight's input
  channels: 80 mels for classic Whisper, 128 for large-v3. `WhisperReplica` exposes it
  as `n_mels()` (`models/whisper.cc:73`, pool surface `:650`).
- `max_input_time() = max_output_time() * 2` (`layers/whisper.h:30`) where `max_output_time()`
  is the position-embedding table size (1500) — i.e. 3000 mel frames = the 30 s window.
  The stride-2 conv2 is the ×2.
- Features arrive `[batch, n_mels, time]`; output is `[batch, time/2, d_model]`.
  `is_encoded` (`layers/whisper.h:34-43`) detects an already-encoded input by shape, which is
  how `maybe_encode` (`models/whisper.cc:107`) lets callers pass either mel features **or** a
  previously returned `encode()` output to generate/align/detect_language.

`encode(features, to_cpu)` (`models/whisper.cc:80`) moves features to the model
device/dtype, runs the encoder, and either copies to CPU or `synchronize_stream`s.

## generate: prompt structure and the two-phase decode

`generate` (`models/whisper.cc:232`) takes per-example prompts that follow OpenAI's layout:
optional previous-text tokens, then `<|startoftranscript|>`, language token, task token
(`<|transcribe|>`/`<|translate|>`), optional `<|notimestamps|>`, then any forced text.
`check_prompts` (`:163`) requires the SOT index and prompt length to match across the
batch; `get_prompt_length` (`:154`) walks from SOT while tokens are in the special range
`[sot_id, no_timestamps_id]`.

The prompt is split at `prompt_length - 1` (`:268-271`): everything before becomes
`prompt_tokens` **prefilled in one batch forward** via
`WhisperDecoder::forward_prompt` (`layers/whisper.cc:64` — a `decode(step=0,
return_logits=false)` that fills the KV cache without computing logits), and the last
prompt token onward becomes `start_tokens` for the regular `decode()` loop with
`start_step = prompt size` (`:291-297`). Timestamps mode is implied by the last prompt
token not being `<|notimestamps|>` (`:332`), which appends the `ApplyTimestampRules`
logits processor (`:336-341`) — its per-step tensor math (timestamp pairing, the
sum-of-timestamp-probs vs max-text-prob test) is documented in
**`logits-processing.md`**; the class lives at `models/whisper.cc:731-860`.
`suppress_tokens`/`suppress_blank` map to `disable_ids`/`disable_ids_begin` from the
model config's `suppress_ids`/`suppress_ids_begin` (`:311-323`).

## no_speech detection

Two paths to the same number, the probability of `<|nospeech|>` at the SOT step
(`get_no_speech_probs_from_logits`, `:129` — SoftMax then a batched Gather):

- Prompt longer than just SOT: `forward_prompt` returns hidden states, and
  `compute_logits_for_steps` (`layers/whisper.cc:76` — Gather the SOT-step rows, then
  the output projection) produces logits only for that step (`models/whisper.cc:280-289`).
- SOT _is_ the start token: a `GetNoSpeechProbs` logits processor (`:194-230`,
  `apply_first() = true`) captures it at decode step 0 (`:326-330`).

## detect_language

`detect_language` (`:573`) runs a **single decoder step** on SOT (`:610`), gathers the
logits at the `lang_ids` from model config (`:590`, `:611`), softmaxes over just those
(`:612`), and returns per-language `(token, prob)` sorted descending (`:622-638`).

## align: DTW word timestamps

`align` (`:424`) is forced decoding + cross-attention DTW. Alignment heads come from the
**model config** `alignment_heads` (a list of `(layer, head)` pairs written by the
converter — `WhisperConfig`, `whisper_spec.py:16`); missing config throws a reconvert
error (`:440-446`). They're installed via
`TransformerDecoder::set_alignment_heads` (`:448`), so the decoder's `attention` output
concatenates exactly those heads. Then:

1. Build forced sequences `start_sequence + <|notimestamps|> + text + <|eot|>` (`:455-466`)
   and run **one batch decoder forward** (`:491`) for logits + attention weights.
2. Token probs: SoftMax masked to the text vocab (length mask at `_eot_id`, `:497-498`)
   then a batched Gather of each output token (`:501`).
3. `num_frames` are halved for conv2's stride (`:506`); per-example padding is removed
   (`remove_padding`, `:373`) and attention is softmaxed over real frames.
4. `compute_alignments` (`:387`): LayerNorm over the head axis (`:395`),
   `ops::MedianFilter` of width `median_filter_width` (`:392`, `:398`), `Mean` over
   heads (`:401`), then move to CPU (`:404-405` — "the remaining operations are not
   implemented on GPU") and run `negative_dtw` (`src/dtw.cc`) per example (`:418`) to
   get monotonic (text-token, frame) pairs.

## Spec shape

`WhisperSpec` (`whisper_spec.py:26`, name `"WhisperSpec"`, revision 3) =
`WhisperEncoderSpec` (`:68` — conv1, conv2, position_encodings, layer_norm, layers) +
a stock `TransformerDecoderSpec` with GELU and `scale_embeddings = False` (`:46-51`).
So the decoder is wired by `transformer-model-wiring.md`'s rules; only the encoder is
Whisper-specific. Registered in C++ as `register_model<WhisperModel>("WhisperSpec")`
(`src/models/model_factory.cc`).

---

### Relevance to the Metal backend

- **The conv stem is the CPU-reference island**: Conv1D has no Metal kernel, so the two
  stem convs run via the CPU path over unified memory — and the load-time **conv-weight
  float guard** (`weight-loading-and-conversion.md`, commit 351b1990) exists precisely
  because int8 Whisper must keep `conv1/conv2` weights float.
- Everything after the stem is the **prefill regime**: one big batch forward through
  GEMM-heavy encoder layers — where Metal wins. `generate`'s autoregressive loop is the
  tiny-op decode regime (per-op floor; see `apple-silicon`'s perf model).
- `ApplyTimestampRules` contains the one explicit `CT2_WITH_METAL` branch in this file
  (`models/whisper.cc:820-827`): fp16 log-probs are upcast to fp32 because
  `should_sample_timestamp` dispatches into fp32-only CPU-reference primitives.
- The fp16 Whisper bringup gotchas (missing `@autoreleasepool` SIGKILL, fp16→fp32
  upcasts for CPU-ref ops) are recorded in the Whisper-on-Metal memory; the structure
  they patch is the code mapped here.
