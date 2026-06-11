# Encoder-only models & the audio encoders (Encoder, EncoderReplica, wav2vec2)

CT2-architecture reference: the encoder-only surface — the `Encoder` pool for BERT-style
text encoders and the two standalone audio encoders (`Wav2Vec2`, `Wav2Vec2Bert`). The
pool/job machinery itself is `replica-pools-and-async-api.md`; the encoder _inside_ a
seq2seq model is `translator-and-seq2seq.md` (cross-refs, not duplicated).

Source: `include/ctranslate2/encoder.h`, `src/encoder.cc`, `include/ctranslate2/encoding.h`,
`include/ctranslate2/models/language_model.h`, `src/models/language_model.cc`,
`src/layers/wav2vec2.cc`, `src/layers/wav2vec2bert.cc`, `python/ctranslate2/specs/`,
`python/ctranslate2/converters/transformers.py`. Line numbers verified by read on
2026-06-11 — re-grep symbols before acting.

## The text-encoder surface: `Encoder` → `SequenceEncoderReplica`

`Encoder : ReplicaPool<models::SequenceEncoderReplica>` (`encoder.h:9`) has exactly one
method family, `forward_batch_async`, in three input forms — string tokens, id vectors,
or a `StorageView` of ids + lengths (`encoder.h:13-24`, thin lambdas in `encoder.cc:6-36`).
All return `std::future<EncoderForwardOutput>`:

```cpp
// encoding.h:9-13
struct EncoderForwardOutput {
  StorageView last_hidden_state;            // [batch, time, d_model]
  std::optional<StorageView> pooler_output; // [batch, d_model], only if the model has a pooler
};
```

`SequenceEncoderReplica` (`models/language_model.h:114-142`) owns the overload funnel:
tokens → ids via the vocabulary, ids → padded `StorageView` via `make_sequence_inputs`
(`language_model.cc:309-314`), then the device-moving wrapper that ends with
`synchronize_stream(device)` before returning (`:335`) — outputs stay **on-device**
(the Python binding converts them lazily via `StorageView`'s array interface).

`EncoderReplica` (`language_model.h:146`, built by
`TransformerEncoderModel::as_sequence_encoder`, `models/transformer.cc:135`) implements
`forward_impl` (`language_model.cc:352-400`):

1. validate ids rank 2 + lengths size (`:356-365`);
2. if the encoder takes >1 input feature, append `token_type_ids` (zeros placeholder when
   not given, `:371-379`) — the BERT segment-embedding input;
3. run `layers::Encoder` → `last_hidden_state` (`:382-383`);
4. **pooler**: if `pooler_dense` exists, `ops::Gather(axis=1, batch_dims=1)` of the
   first token (CLS) then the pooler `Dense` with its activation (`:388-399`) — the
   activation comes from the `pooler_activation` attribute, default Tanh
   (`language_model.cc:333-336`, spec default `transformer_spec.py:778`).

**Which specs/converters target it:** `TransformerEncoderModelSpec`
(`transformer_spec.py:771`) wraps a `TransformerEncoderSpec` + optional pooler. Loaders in
`converters/transformers.py`: `DistilBertLoader` (`:3231`), `BertLoader` (`:3289`),
`XLMRobertaLoader` (`:3368`), `RobertaLoader` (`:3450`), `CamembertLoader` (`:3532`) —
i.e. the embedding-model family (BERT/XLM-R/sentence-transformers checkpoints).

## Wav2Vec2 (`src/layers/wav2vec2.cc`, `src/models/wav2vec2.cc`)

A CTC speech encoder. The replica surface is one call: `Wav2Vec2Replica::encode(features,
to_cpu)` returning a single `StorageView` (`models/wav2vec2.h:48-65`;
`Wav2Vec2 : ReplicaPool<Wav2Vec2Replica>`). `Wav2Vec2Encoder` (`layers/wav2vec2.cc:48-73`)
detects two converter generations by variable presence: an `_upgraded_model`
(`fp_projection/weight` exists, `:49`) embeds the whole HF graph — Conv1D feature
extractor (`Wav2Vec2LayerNormConvLayer` = conv→transpose→LayerNorm→GELU, `:7-28`; one
stride-5 layer + stride-2 stack), feature projection, a grouped positional conv embedding
(`Wav2Vec2PosConvLayer`, groups=16, `:30-46`), 24 standard `TransformerEncoderLayer`s, and
an optional `lm_head` producing CTC logits when `lm_head/weight` exists (`:50,107-109`).
Old conversions feed pre-extracted features straight to the transformer stack
(`:115-126`). Input is always rank-3 `[batch, frames, feat]`; output is hidden states or
CTC logits — decoding/SAD is the caller's job (spec: `specs/wav2vec2_spec.py`).

## Wav2Vec2-BERT (`src/layers/wav2vec2bert.cc`, `src/models/wav2vec2bert.cc`)

Same replica shape (`Wav2Vec2BertReplica::encode`, `models/wav2vec2bert.h:47-66`) but a
**Conformer** body: each `EncoderLayer` (`layers/wav2vec2bert.cc:6-33`) is the sandwich
half-step-FFN → self-attention → conv module (pointwise conv → depthwise conv groups=1024
→ norm → pointwise conv) → half-step-FFN, with the 0.5 FFN scaling done by an explicit
`ops::Mul` (`:42-47`). `Wav2Vec2BertEncoder` (`:166-217`) = feature projection →
`encoder_layers` → downsampling `adapter_layers` → optional `lm_head`. Activation is
Swish for encoder layers, ReLU for adapters (`:172-179`). This is the
faster-whisper/seamless feature-encoder lineage.

---

### Relevance to the Metal backend

- The text-encoder path is embeddings-Gather + transformer layers + one Gather/Dense
  pooler — all ops with Metal routes; `forward_batch_async` outputs stay device-resident,
  and the `synchronize_stream` before return is the coherence point.
- The wav2vec2 family leans on **Conv1D** (feature extractor, pos-conv, conformer conv
  module) — on Metal that's the CPU-reference/fp32-upcast island (`conv1d-op.md`), and
  the load-time conv-weight float guard applies to these models exactly as to Whisper.
- wav2vec2(-BERT) are the _only_ non-Whisper Conv1D users — when testing a conv change
  on Metal, they're the second consumer to check.
