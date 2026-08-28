---
title: "Position encodings (sinusoidal/learned, ALiBi, relative bias, RoPE variants)"
summary: >-
  The CT2 position-encoding family tree and where each comes from in the spec:
  additive PositionEncoder (learned PositionEmbedding from a model variable vs
  computed SinusoidalPositionEncoder starting at position 1), ALiBi (geometric
  per-head slopes added after the QK^T GEMM before softmax via ops::AlibiAdd),
  and the two distinct relative schemes — T5-style learned
  relative_attention_bias (log-bucketed, cached across layers) and Shaw-style
  relative_position_keys/values. It tabulates the RoPE variant/option surface
  (rotary_dim, interleave, base, and the Linear/Su/Llama3 scaling types with
  longrope mapping to Su). The additive encoder is built only when attention
  has no positional mechanism of its own. Metal note: RoPE has a native kernel
  with scaling variants upstream of it, while additive/ALiBi/relative-bias
  additions run the CPU reference over unified memory.
semantic_id: "il6mxglrm23Twz19R-0LXCc4o2-Ty6sDCpZ2G9zv5s42K3kHb6etX404Wk0_i49Tsi8sc9z7h456_Q2LBLhvpQ"
---

CT2-architecture reference: the position-encoding family. `attention-and-kv-cache.md`
owns _where RoPE is applied_ in the attention data flow (the `offset` trick); this file
owns the family tree — additive encoders, ALiBi, T5/Transformer-relative, and the RoPE
_variant/option_ surface — and where each comes from in the spec.

Source: `src/layers/common.cc`, `include/ctranslate2/layers/common.h`,
`src/layers/attention_layer.cc`, `include/ctranslate2/layers/attention_layer.h`,
`src/layers/attention.cc`, `src/layers/transformer.cc`,
`python/ctranslate2/specs/attention_spec.py`, `transformer_spec.py`,
`converters/transformers.py`. Lines verified by read on 2026-06-11.

## Additive input encoders: `PositionEncoder` (`common.h:89-120`)

Base class `PositionEncoder::operator()(input, index)` (`common.cc:155-178`): slices rows
`[index, index+time)` from a `[max_positions, depth]` table and **adds** them to the
embedded input via `primitives<D>::add_batch_broadcast` (`:173-177`). `index` is the
decode-step offset (decoder passes `step`, `src/layers/transformer.cc:650-651`). Two concrete types:

- **`PositionEmbedding`** (learned) — table is the model variable
  `{scope}/position_encodings/encodings` (`common.cc:186-189`); positions beyond the
  table throw (`:162-166`). Spec side: converters call
  `set_position_encodings` and the spec stores the matrix; used by BART/M2M100/GPT-2-style
  families.
- **`SinusoidalPositionEncoder`** — computed, OpenNMT-tf compatible: log-spaced
  timescales, `concat(sin | cos)` on the depth axis, **positions start at 1** (`i + 1`,
  `common.cc:220`); the table is built for 500 positions up front and lazily regrown
  (`:235-247`). Used when the model has _no_ encodings variable.

Wiring (`src/layers/transformer.cc:372-381`, `build_position_encoder`): if `{scope}/encodings`
exists → learned, else sinusoidal. Built **only when attention has no positional
mechanism of its own**: `has_positional_embeddings` = relative keys ‖ relative bias ‖
rotary ‖ alibi (`attention.h:48`), checked at `src/layers/transformer.cc:423-425` (encoder) and
`:512-514` (decoder).

## ALiBi (`attention_layer.cc:13-52`, `:346-366`)

- **Slopes** (`build_alibi`, `:13-52`): geometric per head, `base =
2^(-2^-(log2(closest_pow2)-3))` (`:20`), with the interleaved extra-slope rule for
  non-power-of-2 head counts (`:27-35`). Positions are `0..L-1` or `-(L-1)..0` depending
  on `use_positive_positions` (`:37-40`). The result is a `[1, heads, 1, key_len]` fp32
  tensor built on CPU then moved to the compute device/dtype (`:362`).
- **Addition point**: in `dot_product_attention`, **after** the QK^T score GEMM and
  before SoftMax — `alibi->apply(output, queries_scale)` (`attention.cc:264-265`) using
  the dedicated `ops::AlibiAdd` (`attention_layer.h:147`), which also handles masking
  semantics. `scale_alibi` multiplies the slopes by the query scale (`:361`).
- **Config**: model flags `{scope}/alibi`, `alibi_use_positive_positions`, `scale_alibi`
  read in `make_alibi` (`src/layers/transformer.cc:477-487`); spec booleans in
  `TransformerDecoderSpec` (`transformer_spec.py:134-136`, stored `:230-232`). Users:
  BLOOM (`converters/transformers.py:1376`, positive positions), MPT (`:1453`), Falcon/RW
  (`:3100-3112`, `scale_alibi=True`).

## Relative positions — two distinct schemes (`src/layers/attention.cc`)

Resolved as optional model variables in the `MultiHeadAttention` ctor
(`attention.cc:302-306`); max distances at `:319-331`.

- **T5-style `relative_attention_bias`**: a learned `[num_buckets, heads]` table.
  `get_relative_position_bucket` (`:51`) computes log-bucketed relative distances;
  `compute_relative_bias` (`:102`) gathers and transposes to `[heads, q_len, k_len]`,
  added to scores by `add_batch_broadcast` (`:257-261`) — and **cached across layers**
  via the `position_bias` out-param (`:237-246`; layer 0 computes, others reuse). Spec:
  `relative_attention_bias=True` → `relative_attention_bias` +
  `relative_attention_max_distance` (`attention_spec.py:60-62`); user: T5
  (`transformers.py:1244`).
- **Shaw-style `relative_position_keys`/`relative_position_values`** (OpenNMT
  `max_relative_position`): `make_relative_positions` (`:16`) builds a clipped relative
  index matrix; `add_relative_representations` (`:148-165`) gathers embedding rows and
  matmuls them into the scores (keys) and the context (values) — applied at `:218-223`
  and `:275-279`. Spec: `relative_position=True` (`attention_spec.py:56-58`). An
  **asymmetric** variant (`relative_asymmetric_position_keys` with separate
  left/right max positions, `:225-230`, ctor `:321-326`) exists for eole/OpenNMT-py
  models.

## RoPE — the variant/option surface

Application mechanics (offset, lazy sin/cos growth) are in `attention-and-kv-cache.md`.
Construction is `make_rotary_embeddings` (`attention_layer.cc:68-109`), reading model
attributes; scaling is realized in `RotaryEmbeddings::initialize`
(`attention_layer.cc:252-343`). Enum: `RotaryScalingType { None=-1, Linear, Su, Llama3 }`
(`attention_layer.h:72-77`), mirrored in `attention_spec.py:9-14`; HF `rope_scaling.type`
maps via `transformers.py:43-46` — note **"longrope" maps to Su**.

| Model attribute (`{scope}/...`)                                | Spec field (`attention_spec.py`)   | Effect in `initialize`                                                   |
| -------------------------------------------------------------- | ---------------------------------- | ------------------------------------------------------------------------ |
| `rotary_dim` (-1 = off; 0 = full head dim)                     | `rotary_dim` (`:79`)               | how many dims get rotated (`apply`, `:217`)                              |
| `rotary_interleave` (default true)                             | `rotary_interleave` (`:80`)        | GPT-NeoX pairwise vs GPT-J half-split layout (`:308-316`)                |
| `rotary_base` (default 10000)                                  | `rotary_base` (`:81`)              | inverse-frequency base (`:270`)                                          |
| `rotary_scaling_type`                                          | `rotary_scaling_type` (`:84`)      | selects the branch below                                                 |
| `rotary_scaling_factor` (Linear)                               | `rotary_scaling_factor` (`:85-88`) | positions divided by factor (`:301`)                                     |
| `rotary_scaling_{long,short}_factor` (Su/longrope)             | variables (`:89-91`)               | per-dim inv-freq divisors, long vs short context (`:257-267`)            |
| `rotary_{low,high}_freq_factor` (Llama3)                       | variables (`:92-94`)               | wavelength-smoothed frequency rescale (`:271-294`)                       |
| `original_max_position_embeddings` / `max_position_embeddings` | scalars (`:69-76`)                 | Su context switch + the sqrt-log magnitude scale on sin/cos (`:332-342`) |

Users: most decoder LLM converters (Llama/Mistral/Qwen/Gemma pass `rotary_dim=0,
interleave=False`; GPT-NeoX-family pass partial `rotary_dim` with interleave).

---

### Relevance to the Metal backend

- **RoPE has a native Metal kernel** (`ct2_rotary_float/half` in
  `src/metal/kernels/kernels_msl.h`) with a parity test in `tests/metal_test.cc`; all
  scaling variants are upstream of it (they only change the `_sin`/`_cos` tables, built
  once via generic ops), so a new scaling type needs no Metal work.
- Additive `PositionEncoder` runs through `primitives<D>::add_batch_broadcast` — on Metal
  that's the CPU reference over unified memory (no dedicated kernel); same for the T5
  relative-bias addition.
- ALiBi tables and relative-position index matrices are **built on CPU** then moved to
  the device (`attention_layer.cc:362`, `attention.cc:209-213`) — fine on unified memory,
  but they are per-length lazy rebuilds: a growth step mid-decode is a host-side stall.
- The `has_positional_embeddings` switch decides whether a Metal forward even contains
  the additive-encoder op — RoPE/ALiBi models skip it entirely.
