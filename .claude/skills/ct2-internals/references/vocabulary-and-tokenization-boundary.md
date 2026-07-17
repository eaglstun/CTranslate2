---
title: "Vocabulary & the tokenization boundary"
summary: >-
  The tokenization boundary: CT2 never tokenizes text — translate/generate
  take and return token strings, and the only string↔id conversion is
  Vocabulary::to_ids/to_tokens (subword tokenization is the caller's job). It
  covers the bidirectional Vocabulary index (VocabularyInfo special tokens
  sourced from config.json, from_text_file/from_json_file construction, unk
  appended if missing, to_id allow_unk fallback, the max_length truncation
  that preserves trailing EOS), how seq2seq/LM/Whisper models ship vocab
  files, and the optional VocabularyMap target-vocab restriction (vmap.txt
  n-gram rules). update_output_layer physically shrinks the output projection
  to |candidates| rows via select_weights with −1e10 pad bias. Metal notes:
  consumers tokenize in Python so the backend never sees strings, and an
  active vmap re-Gathers the output projection per batch, defeating cached MPS
  GEMM descriptors.
---

What the engine knows about text: an indexed token list and an optional target-vocab
restriction map. **CT2 never tokenizes text** — `translate_batch`/`generate_batch` take
token _strings_ (`std::vector<std::vector<std::string>>`) and return token strings; the
only string↔id conversion inside the engine is `Vocabulary::to_ids`/`to_tokens`.
Subword tokenization (SentencePiece, BPE, HF tokenizers) is the caller's job.

**Sources (all citations from real lines):**

- `include/ctranslate2/vocabulary.h` / `src/vocabulary.cc`
- `include/ctranslate2/vocabulary_map.h` / `src/vocabulary_map.cc`
- `src/models/sequence_to_sequence.cc` (vocab loading, vmap wiring), `src/models/model_reader.cc`
- `src/layers/decoder.cc` (`update_output_layer` — what the vmap does to the output layer)

## 1. Vocabulary

`Vocabulary` (`vocabulary.h:17-71`) is a bidirectional token↔id index:
`_id_to_token` (vector of pointers into the map's keys — tokens stored once,
`vocabulary.cc:51-54`) + `_token_to_id` hash map. `VocabularyInfo` carries the special
tokens with defaults `<unk>`/`<s>`/`</s>` (`vocabulary.h:10-14`); the real values come
from the model's `config.json` (`unk_token`/`bos_token`/`eos_token`,
`sequence_to_sequence.cc:15-18`). `unk_id()/bos_id()/eos_id()` are lookups of those
strings (`vocabulary.h:55-63`).

- Construction: `from_text_file` (one token per line; carriage returns are stripped only
  if _every_ line ends with one — some vocabs contain a literal `\r` token,
  `vocabulary.cc:9-30`) or `from_json_file` (a JSON string array, `:32-34`). **If the unk
  token is missing it is appended** as the last id (`vocabulary.cc:46-48`).
- `to_id(token, allow_unk)` falls back to the unk id; with `allow_unk=false` it throws on
  OOV instead (`vocabulary.cc:67-75`). `to_token(id)` throws on out-of-range (`:60-65`).
- Batch helpers: `to_tokens(batch_ids)` (`:81-95`) and `to_ids(batch_tokens, max_length,
prefix/suffix, allow_unk)` (`:108-147`). The `max_length` truncation has a subtlety: it
  preserves a trailing EOS — or EOS + final language-code token — in the last positions
  rather than chopping them (`vocabulary.cc:130-141`).

## 2. How models ship vocabularies

`load_vocabulary(model_reader, name, info)` tries `<name>.json` then `<name>.txt`
(`src/models/model_reader.cc:79-94`). Seq2seq loading order
(`sequence_to_sequence.cc:13-58`): `shared_vocabulary` (one file used for both sides) →
else `target_vocabulary` + (`source_vocabulary` | `source_1_vocabulary`,
`source_2_vocabulary`, … for multi-feature models). LMs/Whisper/wav2vec2 load a single
`vocabulary` file (`src/models/language_model.cc:33`, `src/models/whisper.cc:31`). The converter side of this —
including the shared-vocabulary collapse at save time — is in `model-binary-format.md`.

## 3. VocabularyMap — target-vocab restriction (translation only)

`VocabularyMap` (`vocabulary_map.h:19-32`) loads the optional `vmap.txt` from the model
directory (`sequence_to_sequence.cc:10,52-57`). File format: one rule per line,
`<source n-gram>\t<candidate tokens separated by spaces>` (parser:
`vocabulary_map.cc:7-50`; rules are bucketed by n-gram order, `:36-39`). An empty-string
key marks always-on candidates (`:46-49`); unk/bos/eos are always candidates (`:42-44`).

At translate time, **only when `options.use_vmap` is set** (`translation.h:61`) and the
model has a vmap: `get_candidates(source_tokens, target_prefix_ids)`
(`vocabulary_map.cc:52-75`) scans every source n-gram window against the rules and unions
the candidate sets (plus the target prefix ids), returning a **sorted** id vector.

What it does to the output layer — `Decoder::update_output_layer(size_multiple,
restrict_ids)` (`src/layers/decoder.cc:72-139`, called at `sequence_to_sequence.cc:330-333`):

- builds an int32 index of the candidate ids and calls
  `output_layer().select_weights(&index, extra_bias)` (`decoder.cc:126-133`) — the final
  projection physically **shrinks from vocab-size rows to |candidates| rows** (a Gather of
  the weight matrix), which is where the speedup comes from;
- pads `|candidates|` up to `preferred_size_multiple()` with id 0 plus an `extra_bias` of
  `-1e10` on the pad rows so softmax masks them (`:82-86,106-124`);
- maintains `_to_output_word_id`/`_to_original_word_id` so the decode loop translates
  between restricted and real ids (`:135-138`); empty `restrict_ids` resets to the full
  vocabulary (`:88-104`).

### Relevance to the Metal backend

- The tokens-in/tokens-out boundary means Whisper/NLLB/Qwen consumers (the downstream
  rig) do their own tokenization in Python; the Metal backend never sees strings.
- `update_output_layer`'s `select_weights` re-gathers the output projection **per batch**
  when a vmap is active — on Metal that's a weight-tensor Gather + a different GEMM `n`
  every call, defeating cached MPS GEMM descriptors and (for int8) the resident
  weight/scale pairing. None of the current Metal test models use a vmap.
- The `-1e10` pad bias rides the normal bias+activation epilogue — no Metal-specific
  masking; it reaches the GPU as ordinary bias data.
- `preferred_size_multiple()` padding exists for CPU int16/MKL alignment; Metal ignores it
  (`_preferred_size_multiple = 1` for the compute types Metal resolves to).
