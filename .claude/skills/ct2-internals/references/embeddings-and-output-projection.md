---
title: "Embeddings & the Output Projection"
summary: >-
  Covers the two bookends of a forward pass: the Embeddings layer (token ids
  to vectors) and the Dense output projection (hidden states to logits).
  Embeddings holds a {scope}/weight table plus optional weight_scale and an
  ops::Gather; a float table is a single gather while an int8/int16 table
  gathers int8 rows then runs ops::Dequantize (the table can legitimately be
  int8, cutting RSS). The sqrt(d) scale is NOT in the layer;
  build_embeddings_scale resolves it and it is applied as a plain ops::Mul
  after lookup. The projection is an ordinary Dense from scope
  decoder/projection running its GEMM with trans_b=true, storing weight as
  [vocab, depth] exactly like the embedding table, so weight tying is pure
  serialization (converter aliases the tensors, register_variable_alias shares
  one shared_ptr) with no runtime transpose. Vocabulary restriction hooks the
  projection via update_output_layer/select_weights, gathering selected rows
  into _partial buffers with -1e10 bias on padding rows. On Metal the
  projection is the dominant decode-step GEMM driving the small-m GEMV win.
---

The two bookends of every forward pass: token ids → vectors (`Embeddings`, a Gather) and
hidden states → logits (`Dense` as the projection), plus weight tying and the
vocabulary-restriction hook.

**Sources (read these, all citations below are from real lines):**

- `src/layers/common.cc` (`Embeddings`, `Dense::select_weights`) + `include/ctranslate2/layers/common.h`
- `src/layers/transformer.cc` (embeddings scale, `_proj`, `scale_outputs`, softcapping)
- `src/layers/decoder.cc` + `include/ctranslate2/layers/decoder.h` (`update_output_layer`)
- `src/models/model.cc`, `python/ctranslate2/specs/common_spec.py`, `src/vocabulary_map.cc`

---

## 1. Embeddings = Gather (+ Dequantize when the table is int8)

`Embeddings` (`src/layers/common.cc:53-86`) holds the table variable `{scope}/weight`,
the optional `{scope}/weight_scale`, and an `ops::Gather` member
(`include/ctranslate2/layers/common.h:61`). `operator()` (`common.cc:68-86`):

- **Float table**: a single `_gather_op(_embeddings, ids, output)` (`:83`).
- **int8/int16 table**: gather int8 rows, then `ops::Dequantize` — gathering the matching
  per-row scales first unless the scale is scalar (int16 global) (`:71-81`). So a
  quantized table costs one extra gather + dequant per step, but halves/quarters RSS.

**The table CAN be int8.** Three pieces agree: `EmbeddingsSpec` has an OPTIONAL
`weight_scale` (`python/ctranslate2/specs/common_spec.py:62-65`), which makes it
quantizable at convert time (`converter-quantization-and-fusion.md`); at load,
`Model::is_quantizable` is simply "name ends with `weight`" (`src/models/model.cc:287-289`),
so `set_compute_type` quantizes/dequantizes embeddings like any linear weight; and the
runtime path above consumes the result. Output dtype is always the compute type's float
(`_output_type = get_default_float_type(model.effective_compute_type())`, `common.cc:55`,
returned by `output_type()`, `:60-62`) — the gather/dequant produces float activations
regardless of table dtype.

**The √d scale is NOT in this layer.** `build_embeddings_scale`
(`src/layers/transformer.cc:384-405`) resolves `{scope}/scale_embeddings` (back-compat:
`embeddings/multiply_by_sqrt_depth`) into either `sqrt(output_size)` or an explicit
value, and the encoder/decoder applies it as a plain `ops::Mul` after the lookup
(`src/layers/transformer.cc:436-437` encoder; `:640-643` decoder, skipped at step 0 under
`start_from_zero_embedding`). `ParallelEmbeddings` (`common.cc:88+`) wraps multi-feature
inputs (concat or add merge).

## 2. Output projection: `Dense` as lm_head — tying needs no transpose

The decoder's projection is an ordinary `Dense` built from scope
`decoder/projection` (`src/layers/transformer.cc:516`). Logits sequence at the end of
`TransformerDecoder::operator()` (`src/layers/transformer.cc:845-870`): output norm → optional
`_project_out` → optional `scale_outputs` Mul (`:533-537`, `:853-854`; T5's
`d_model**-0.5` rides here) → `_proj(layer_in, *outputs)` (`:857`) → optional Gemma-style
`final_logit_softcapping` tanh squash (`:858-865`). Logits come out in `Dense`'s
`output_type()` — the effective float type. **No temperature here**: temperature is a
sampler-pipeline Mul (`sampling-and-topk.md`); the boundary stays clean.

**Tying mechanism.** `Dense` runs its GEMM with `trans_b=true` (`common.cc:291-297`), so
its weight is stored `[output, input]` = `[vocab, depth]` — exactly the embedding-table
layout. There is no runtime transpose for tying; it's pure serialization:

- The converter assigns the same tensor to both leaves
  (`spec.projection.weight = spec.embeddings.weight`,
  `python/ctranslate2/converters/transformers.py:1481` MPT,
  `converters/openai_gpt2.py:56`) — or HF already tied them and the element-wise alias
  dedup catches it (`model_spec.py::_alias_variables`, `:169-188`).
- At load, the alias record makes both names share one `shared_ptr<StorageView>`
  (`Model::register_variable_alias`, `src/models/model.cc:276-281`), and the loader also
  aliases `{alias}_scale`/`{alias}_zero` (`model.cc:763-772`) — so a tied + int8 table
  gives the projection its scale for free, and dtype conversion happens once.

## 3. Vocabulary restriction hooks the projection, not the model

`Decoder::update_output_layer(size_multiple, restrict_ids)` (`src/layers/decoder.cc:72-140`)
rebuilds the projection over a token subset: it pads the new size to a multiple
(`preferred_size_multiple`, tensor-core friendly), builds an int32 index, and calls
`output_layer().select_weights(&index, extra_bias)` — `output_layer()` being the pure
virtual `Dense&` accessor (`include/ctranslate2/layers/decoder.h:91`).
`Dense::select_weights` (`src/layers/common.cc:317-333`) **Gathers** the selected rows of
weight/bias (and qscale/compensation) into `_partial_*` buffers; padding rows get an
extra bias of `-1e10` so softmax masks them (`decoder.cc:108-123`). `select_weights(nullptr)`
restores the full vocab (`decoder.cc:95`).

The candidate ids come from `VocabularyMap::get_candidates`
(`src/vocabulary_map.cc:53`) when `use_vmap` is set
(`src/models/sequence_to_sequence.cc:330-333`); generators/Whisper call it with no
restriction just for the size padding (`src/models/language_model.cc:153`,
`src/models/whisper.cc:254`). Word ids are remapped back via `_to_original_word_id`
(`decoder.cc:135-138`). The vmap file itself (`vmap.txt`) is registered at convert time
(`specs-and-converters.md` §3).

---

### Relevance to the Metal backend

- Both bookends are pure op composition (Gather, Dequantize, Mul, Dense's
  quantize→gemm→dequantize) — the int8-Metal project changed **nothing** here; an int8
  embedding table on Metal runs the same `metal::dequantize_s8` kernel as any other
  dequant (`quantization-scheme-and-ops.md`).
- The projection is the single biggest GEMM of every decode step (`[*, depth] × [vocab,
depth]^T`) — it dominates the Metal GEMV win at small m (see `METAL_BENCHMARKS.md` and
  the `apple-silicon` perf model).
- `select_weights`' Gather rebuilds `_partial_weight` on device; on Metal that's a real
  GPU-resident copy of the restricted table — cheap, but it means vmap restriction also
  shrinks the dominant GEMM, same as on CUDA.
