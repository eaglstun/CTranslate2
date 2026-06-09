# Norm placement in CTranslate2 transformers (pre / post / pre-post sandwich)

CT2-architecture reference (not an Apple/Metal topic — but the norms work spans both,
so it lives here next to `math-functions-and-numeric-parity.md`, which covers the
kernel-side numerics). This file is about **where** a LayerNorm/RMSNorm sits in a
transformer block, not how it's computed.

**Line numbers below were verified by grep on 2026-06-09.** Model-family attributions
are intentionally NOT asserted — grep `pre_post_layer_norm=` / `pre_norm=` in the
converter to map a line to a family yourself; the subagent that first surfaced these
guessed some names.

## The mental model

`specs/` declares what placement a model has → `converters/` sets it per architecture →
`src/layers/transformer.cc` executes it. **Placement is pure CPU-side orchestration**:
it only decides the _order_ in which the norm op is called. None of these branches care
whether the norm ran on CPU or the Metal GPU — so the Metal backend (`rms_norm.cc` /
`layer_norm.cc` routing into the kernel) is orthogonal to all of it.

## 1. C++ runtime — `src/layers/transformer.cc` (+ `include/ctranslate2/layers/transformer.h`)

Where placement actually executes.

**Pre vs post, per sublayer** — `FeedForwardNetwork`, gated by a `_pre_norm` bool:

```cpp
// transformer.cc
:13   , _pre_norm(pre_norm)
:23   if (_layer_norm && _pre_norm) { ... }   // norm BEFORE the sublayer (pre-norm)
:49   if (_layer_norm && !_pre_norm) ...       // norm AFTER the sublayer (post-norm)
```

**Pre-post "sandwich" (norm both before AND after each sublayer)** — supported, and
detected at runtime by the presence of all four norm objects:

```cpp
// transformer.cc:88  "Check if using pre_post_layer_norm pattern (T5Gemma style)"
const bool pre_post_layer_norm = _input_layer_norm && _post_attention_layer_norm
                              && _pre_feedforward_layer_norm && _post_feedforward_layer_norm;
:91   if (pre_post_layer_norm) { ... }         // the sandwich execution path
```

The four `*_layer_norm` fields are built via `build_optional_layer<LayerNorm>` at
`:69` and `:72` (encoder layer) and `:171` / `:176` (decoder layer) — they're optional,
so absence = the model doesn't use the sandwich. The decoder also has a
**parallel-residual / shared-norm** variant in the same file, keyed on
`_shared_layer_norm` / `_input_layer_norm`.

## 2. Python specs — `python/ctranslate2/specs/transformer_spec.py`

The declarative layout the converter fills and the C++ loader reads — the source of
truth for a model's placement.

```python
:15   pre_norm: bool = True                  # encoder (docstring :42 "Enable the pre-norm ... architecture")
:122  pre_norm: bool = True                  # decoder
:35   pre_post_layer_norm: bool = False      # encoder (docstring :68 "Add post layer norm for each pre norm layer")
:146  pre_post_layer_norm: bool = False      # decoder
:144  parallel_residual: bool = False        # decoder
```

When `pre_post_layer_norm=True`, the spec creates four `common_spec.LayerNormSpec`
fields per layer (`input_/post_attention_/pre_feedforward_/post_feedforward_layer_norm`)
and `delattr`s the single per-sublayer `layer_norm` — matching the four-field detection
in the C++ layer above. Related: attention's own norm toggle is `has_norm` in
`attention_spec.py` (plus `qk_norm` / `v_norm`); the norm primitive is
`common_spec.LayerNormSpec`, which carries the `rms_norm` flag (RMSNorm vs LayerNorm).

## 3. Converters — `python/ctranslate2/converters/transformers.py`

Where each architecture _declares_ its placement. `pre_post_layer_norm=True` is set at
lines **1627, 1931, 2177, 3839, 3863, 4101** (verified). Standard pre-norm models pass
`pre_norm=True` and post-norm models (e.g. BERT-style) pass `pre_norm=False`. Grep
`pre_norm=` / `pre_post_layer_norm=` to see which loader sets what.

---

### Relevance to the CT2 Metal backend

- **None of this is Metal's concern.** Placement is CPU orchestration that calls the norm
  op N times in some order; the op then routes to the GPU kernel (or CPU reference) via
  the usual `if (x.device()==METAL)` branch in `rms_norm.cc` / `layer_norm.cc`. A
  pre-post sandwich just means the same `metal::rms_norm` / `metal::layer_norm` entry
  point fires more times per block — no new kernel needed.
- **Coverage caveat that DOES touch Metal:** `layer_norm.cc` only routes the common case
  (last-axis, with gamma+beta) to the GPU kernel; general-axis / no-affine norms fall to
  the CPU reference (see the milestone notes). All the placement variants here use
  last-axis affine norms, so they hit the GPU path — but if a new arch needs a no-affine
  or non-last-axis norm, that's a CPU-reference fallback, not a placement issue.
- So a norms task that's really about _placement_ (pre/post/sandwich/parallel-residual)
  is done entirely in the three files above and never opens `src/metal/`. A norms task
  about the _kernel_ (parity, fp16, reductions) is the inverse — see
  `math-functions-and-numeric-parity.md` and `simd-group-functions.md`.
