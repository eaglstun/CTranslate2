# Transformer model wiring (spec config → constructed layer graph)

CT2-architecture reference: how a serialized Transformer model becomes a constructed
`TransformerEncoder`/`TransformerDecoder` object tree at load time — embeddings scale,
position-encoder selection, layer stack, final norm, output projection (incl. tied
embeddings). This is the assembly map; norm _placement_ execution is
`norm-placement-in-transformers.md` and attention internals are
`attention-and-kv-cache.md` (cross-ref, not duplicated here).

Source: `src/models/transformer.cc`, `src/layers/transformer.cc`,
`src/layers/common.cc` (Dense), `python/ctranslate2/specs/transformer_spec.py`,
`python/ctranslate2/specs/model_spec.py`. Line numbers verified by read on 2026-06-11 —
re-grep symbols before acting.

## Model classes → replicas (`src/models/transformer.cc`)

Three thin model classes; each builds the layer objects in its `as_*` factory:

- `TransformerModel::as_sequence_to_sequence` (`models/transformer.cc:82`) — constructs
  `layers::TransformerEncoder(*this, "encoder")` + `layers::TransformerDecoder(*this,
"decoder")` and wraps them in an `EncoderDecoderReplica` (`:85-89`). Spec revision 7
  (`:40`).
- `TransformerDecoderModel::as_sequence_generator` (`:111`) — decoder only →
  `DecoderReplica`. Revision 8 (`:97`).
- `TransformerEncoderModel::as_sequence_encoder` (`:134`) — encoder only →
  `EncoderReplica`.

The `"encoder"`/`"decoder"` strings are the **scope prefixes**: every layer ctor below
looks up its weights as `scope + "/..."` against the model's variable index. Legacy
plumbing: `map_v1_variable_name` (`:16`) renames OpenNMT-tf v1 variables;
`initialize` (`:61`) registers aliases (`encoder/num_heads` → `num_heads`, …) so
pre-revision-5 models still resolve scoped lookups. `is_linear_weight` (`:44`) =
quantizable and not under an `embeddings` scope; `is_packable` (`:49`) additionally
excludes `projection` (the output layer can be dynamically vocab-masked).

## Constructor wiring (`src/layers/transformer.cc`)

`TransformerEncoder` ctor (`layers/transformer.cc:408`) and `TransformerDecoder` ctor (`:490`)
resolve everything from model variables — presence/absence of a variable IS the config:

- **Embeddings scale** — `build_embeddings_scale` (`:384-405`): reads
  `scope/scale_embeddings` (falling back to the older
  `scope/embeddings/multiply_by_sqrt_depth`). The attribute is **either a bool flag or
  the actual value** (`:396`): int8-true → `sqrt(d_model)` (`:398`, the classic
  `x * sqrt(d_model)`); a float ≠ 1 → that value (`:399`, e.g. BART's `embed_scale`,
  `transformers.py:404`); int8-false / float 1 / absent → `nullptr` (no Mul at all).
  The spec defaults `scale_embeddings = True` (`transformer_spec.py:86`, `:228`) and
  converters for LLM families overwrite it to `False` (grep `scale_embeddings` in
  `converters/transformers.py`).
- **Position encoder** — `build_position_encoder` (`:372-382`): if
  `scope/position_encodings/encodings` exists → learned `PositionEmbedding`, else
  `SinusoidalPositionEncoder`. But it's **skipped entirely** when the first layer's
  attention has its own positional scheme (`:423-425`, `:512-514`:
  `has_positional_embeddings()` — RoPE/ALiBi/relative). The spec mirrors this: it only
  creates `position_encodings` when no relative/rotary/alibi option is set
  (`transformer_spec.py:235-241`).
- **`layernorm_embedding`** — optional norm right after embedding+position
  (`:414`, `:498`; applied at `:440-441`, `:652-653`); spec flag `layernorm_embedding`
  (`transformer_spec.py:91-92`, `:244-245`).
- **Layer stack** — `build_layers_list<TransformerEncoderLayer/...>` (`:417`, `:504`)
  over `scope + "/layer"` (i.e. `layer_0`, `layer_1`, …), passing `num_heads`,
  `pre_norm`, `activation`, flash-attention flag, and (decoder) the shared `Alibi`
  object (`make_alibi`, `:477`). Each layer ctor builds the optional sandwich /
  parallel-residual norms (`:69-72` encoder, `:169-187` decoder) — execution branches
  are `norm-placement-in-transformers.md`'s territory.
- **Final norm** — `_output_norm` = optional `scope/layer_norm` (`:415`, `:499`),
  applied after the last layer (`:470-471` encoder; `:846-847` decoder). The spec only
  creates it for `pre_norm and not no_final_norm` (`transformer_spec.py:89-90`, `:242-243`).
- **Decoder extras** — `_project_in`/`_project_out` (`:500-501`, spec `project_in_out`),
  `_start_from_zero_embedding` (`:495`), `_outputs_scale` from `scope/scale_outputs`
  (`:533-537` — e.g. T5's tied-embedding `d_model**-0.5`, `transformers.py:1252-1253`),
  `_sliding_window` (`:517`), `_final_logit_softcapping` (`:519`; applied as
  `tanh(logits/cap)*cap` at `:858-865`, Gemma2).
- **Alignment** — `alignment_layer`/`alignment_heads` attributes (`:521-531`): negative
  layer wraps from the end, `alignment_heads == 0` means all heads;
  `set_alignment_heads` (`:568`) records which layer's cross-attention feeds the
  returned `attention` tensor (averaged via `ops::Mean` at `:828-829`).

The decoder's forward order is `decode()` (`:624`): embeddings → zero-first-timestep →
scale Mul → project_in → position encoder (`offset = max(step, 0)`) →
layernorm_embedding → layer stack → output_norm → project_out → outputs_scale →
`_proj` (logits) → softcapping.

## Output projection & tied embeddings

`_proj` is a plain `Dense(model, scope + "/projection")` (`layers/transformer.cc:516`). The
`Dense` ctor (`common.cc:270`) resolves `scope + "/weight_packed"` then
`scope + "/weight"` (`get_linear_weight`, `common.cc:258`) — there is **no tying logic
in C++ at all**. Tying happens upstream:

1. The converter assigns the lm_head tensor onto `spec.decoder.projection`
   (`set_linear(spec.decoder.projection, model.lm_head)`, `transformers.py:306` et al.);
   for a tied model that tensor _is_ the embedding matrix.
2. `ModelSpec._alias_variables` (`model_spec.py:169`) finds element-wise-equal
   duplicates and serializes the later name as a string **alias** —
   `decoder/projection/weight` → alias of `decoder/embeddings/weight` (alphabetical
   order keeps the embeddings copy).
3. At load, `Model::load` registers each alias via `register_variable_alias`
   (`model.cc:766-771`, including the paired `_scale`/`_zero` names), so Dense's lookup
   of `decoder/projection/weight` resolves to the **same StorageView** as the embedding.

Subtlety: the surviving variable name contains `embeddings`, so
`is_linear_weight` (`models/transformer.cc:44`) is false for it — a tied projection weight is
treated as an embedding (no packing/linear-specific load transforms). Layouts agree
because CT2 linear weights are stored `[output, input]` and embeddings `[vocab, depth]`.

## Spec config keys → C++ reads (the scalar attributes)

Spec scalar attributes (`transformer_spec.py:79-94`, `:222-246`) are serialized as tiny
variables in `model.bin`, then read with `get_attribute_with_default` /
`get_flag_with_default` / `get_enum_value` (`models/model.h:136-148`):

| Spec attribute (dtype)        | C++ read site (`layers/transformer.cc`)                         |
| ----------------------------- | --------------------------------------------------------------- |
| `num_heads` (int16)           | `:412`/`:492` `get_attribute_with_default<int32_t>(…, 8)`       |
| `pre_norm` (bool→int8)        | `:421`/`:508` `get_flag_with_default(…, true)` → per-layer ctor |
| `activation` (int8 enum)      | `:422`/`:509` `get_enum_value<ops::ActivationType>`             |
| `embeddings_merge` (encoder)  | `:410` `get_enum_value<EmbeddingsMerge>`                        |
| `scale_embeddings` (flag/val) | `build_embeddings_scale` `:384`                                 |
| `alignment_layer/heads`       | `:521-524` (decoder ctor)                                       |
| `sliding_window` (int32)      | `:517`                                                          |
| `alibi`, `scale_alibi`, …     | `make_alibi` `:477-488`                                         |
| `final_logit_softcapping`     | `:519`                                                          |
| `start_from_zero_embedding`   | `:495`                                                          |

Structural options (`ffn_glu`, `rms_norm`, `num_heads_kv`, `parallel_residual`,
`pre_post_layer_norm`, rotary options…) are not scalar attributes — they manifest as the
_presence/shape of weight variables_ (e.g. `linear_0_noact` for GLU, the four sandwich
norms, fused-QKV widths) that the optional-layer builders detect.

---

### Relevance to the Metal backend

- All of this wiring runs **once at load on the CPU** — constructing the layer tree
  allocates the weight `StorageView`s on the model's device, so on `Device::METAL` every
  variable resolved here is already a real `MTLBuffer` (see `weight-loading-and-conversion.md`).
- The constructed graph determines the per-step op sequence Metal executes: embeddings
  Gather → scale Mul → norms → layers → output GEMM (`_proj`). The decoder's `_proj`
  over the full vocab is the **largest GEMM of every decode step** — the one place tiny-op
  decode still has real arithmetic intensity (the int8 GEMV win lives there).
- Tied embeddings mean the embedding table and the output projection are one buffer —
  on Metal that's one int8/fp16 `MTLBuffer` serving both a Gather and a GEMM; the
  `is_linear_weight` exclusion explains why a tied projection follows the _embedding_
  dtype path at load.
- `final_logit_softcapping` routes through `ops::Tanh` — the exact op whose Metal
  overflow-to-NaN bug broke Gemma2 (clamped in `kernels_msl.h`; see the Metal memory).
