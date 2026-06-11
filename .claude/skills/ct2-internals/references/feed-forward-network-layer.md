# The FeedForwardNetwork layer (standard FFN vs GLU, shapes, fusion hooks)

CT2-architecture reference: `layers::FeedForwardNetwork` — the one transformer sublayer
not owned by another ref. Scope is the block structure and the gate+up representation;
the activation _formulas and fusion plumbing_ are `activation-ops.md`, norm _placement_
semantics are `norm-placement-in-transformers.md`, and `Dense` internals are
`dense-layer-and-quantized-linear.md` (cross-refs, not duplicated).

Source: `src/layers/transformer.cc:8-51`, `include/ctranslate2/layers/transformer.h:13-39`,
`python/ctranslate2/specs/transformer_spec.py:443-449`,
`python/ctranslate2/converters/transformers.py`. Line numbers verified by read on
2026-06-11 — re-grep symbols before acting.

## The members (header `layers/transformer.h:32-38`)

```cpp
const std::unique_ptr<const LayerNorm> _layer_norm;   // optional sublayer norm
const bool _pre_norm;
const ops::ActivationType _activation_type;
const Dense _ff1;                                     // linear_0  (carries the activation)
const std::unique_ptr<const Dense> _ff1_noact;        // linear_0_noact (GLU "up", optional)
const Dense _ff2;                                     // linear_1  (is_layer_out=true)
const bool _tensor_parallel;
```

Ctor (`layers/transformer.cc:8-19`): everything resolves from scoped variables under
`scope = ".../ffn"`. Two structural decisions live in the initializer list:

- `_ff1(model, scope + "/linear_0", &_activation_type)` (`:15`) — the activation is a
  _pointer handed to Dense_, so it executes inside `_ff1`'s GEMM/dequantize epilogue
  (the fusion hook; see `activation-ops.md` §3).
- `_ff1_noact = build_optional_layer<Dense>(..., "/linear_0_noact")` (`:16`) — **presence
  of this weight IS the GLU flag**. There is no `ffn_glu` attribute at runtime; the
  optional-layer builder detects the variable (cf. the structural-options note in
  `transformer-model-wiring.md`).
- `_ff2(model, scope + "/linear_1", nullptr, /*is_layer_out=*/true)` (`:17`) — no
  activation; the 4th arg marks it as a row-parallel output layer for tensor parallelism
  (`Dense` ctor signature `layers/common.h:128-131`; see `tensor-parallel.md`).

## `operator()` — the two variants in one body (`layers/transformer.cc:21-51`)

```cpp
if (_layer_norm && _pre_norm) { (*_layer_norm)(input, output); x = &output; }  // :23-26
_ff1(*x, inner);                          // :32  — activation fused into this GEMM
if (_ff1_noact) {                         // :33-37 — GLU: gate(act) * up(no act)
  (*_ff1_noact)(*x, linear);
  ops::Mul()(linear, inner, inner);       // elementwise gate
}
_ff2(inner, output, _layer_norm ? &input : nullptr);   // :39 — residual fused into Dense
if (_layer_norm && !_pre_norm) (*_layer_norm)(output, output);  // :49-50
```

Three things worth internalizing:

1. **Standard FFN** = `linear_1(act(linear_0(x)))`. **GLU FFN** = `linear_1(act(linear_0(x))
⊙ linear_0_noact(x))` — gate (`linear_0`, _with_ activation) times up (`linear_0_noact`,
   _without_), combined by a plain `ops::Mul` (`:36`). SwiGLU vs GeGLU is purely which
   `ActivationType` rides `_ff1` (Swish for llama-family, GELUTanh for Gemma).
2. **The residual is fused into `_ff2`**: `Dense::operator()` takes an optional `residual`
   (`common.h:134`) added in the bias/dequantize epilogue. It is passed _only when the FFN
   owns its norm_ (`_layer_norm ? &input : nullptr`, `:39`) — in the four-norm sandwich
   path the FFN has no local `layer_norm` variable, so the caller layer does the residual
   add itself (`norm-placement-in-transformers.md`).
3. With tensor parallelism, `_ff2` produces rank-partial sums and an
   `ops::ReduceAll(SUM)` allreduce follows (`:41-47`) — see `tensor-parallel.md`.

## Shapes

With `input [batch, time, d_model]` and weights stored `[output, input]`:
`linear_0`/`linear_0_noact` are `[d_ff, d_model]` → `inner [batch, time, d_ff]`;
`linear_1` is `[d_model, d_ff]` → `output [batch, time, d_model]`. The GLU `Mul` is
elementwise at `[batch, time, d_ff]`. Decode steps run the same code with `time == 1`.

## Spec & converter side

`FeedForwardSpec(glu=...)` (`transformer_spec.py:443-449`) declares `layer_norm`,
`linear_0`, `linear_1`, and adds `linear_0_noact` **only when `glu=True`** — which is how
the weight's presence becomes the runtime flag. Converter mappings (HF → spec):

- llama-family: `mlp.gate_proj → linear_0`, `mlp.up_proj → linear_0_noact`,
  `mlp.down_proj → linear_1` (`converters/transformers.py:1832-1840`, LlamaLoader;
  same pattern in GemmaLoader `:1584-1585`).
- T5: `ffn_glu=config.is_gated_act` (`:1243`); gated checkpoints map
  `DenseReluDense.wi_0/wi_1` to `linear_0`/`linear_0_noact` (`:1312-1315`).
- gate+up are **never fused** into one matrix at conversion (unlike QKV) — see
  `converter-quantization-and-fusion.md`.

---

### Relevance to the Metal backend

- The FFN is pure `Dense` + `Mul` + norm — every piece already routes to Metal (GEMM/
  GEMV incl. int8, fused bias+activation, `ops::Mul`, RMSNorm/LayerNorm), so the FFN
  needed **zero Metal-specific work**; int8 weight residency covers `linear_0`,
  `linear_0_noact`, and `linear_1` automatically.
- In a GLU model the FFN is _three_ large GEMMs per layer (vs two), which is why
  llama-family prefill is GEMM-bound and where the int8 GEMV decode win concentrates.
- The fused activation rides `_ff1`'s epilogue (`metal::bias_add` / the int8
  `ct2_dequant_gemm_out_*` kernels) — kernel-side specifics are the `apple-silicon` skill.
