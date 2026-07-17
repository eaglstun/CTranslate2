---
title: "Converter-Side Weight Transforms: Quantization, Fusion, Aliases"
summary: >-
  What CTranslate2 converters do to weights beyond mapping them onto the spec
  tree: convert-time quantization (model_spec.py _quantize), weight fusion,
  and alias dedup. It gives the structural quantizability rule (a var is
  quantizable iff the spec has a sibling {name}_scale, so
  linear/embedding/conv weights qualify but norms/biases never do), the per-
  scheme math (int8 per-output-row symmetric 127/amax, int16 global scalar
  scale, float16/bfloat16 casts), and how AWQ bypasses _quantize entirely via
  detected quantization_config. It covers QKV fusion (fuse_linear
  concatenating along output rows, split by C++ after one GEMM), the un-fused
  SwiGLU gate+up, pre-quant fusion, and how the chosen scheme becomes the
  saved compute type read back by infer_compute_type. Metal note: an int8
  model converted once runs unchanged on CPU and Metal.
---

What converters do to weights BEYOND mapping them onto the spec tree
(`specs-and-converters.md` owns that pipeline) — convert-time quantization, weight
fusion, alias dedup, and how these choices constrain the runtime.

**Sources (read these, all citations below are from real lines):**

- `python/ctranslate2/specs/model_spec.py` (`_quantize`, `_alias_variables`, `optimize`)
- `python/ctranslate2/converters/utils.py` (`fuse_linear`, `fuse_linear_prequant`, …)
- `python/ctranslate2/converters/transformers.py` (loader-side fusion + AWQ entry)
- `src/models/model.cc` (`infer_compute_type`, alias registration)

---

## 1. Convert-time quantization: `_quantize`

`Converter.convert(quantization=...)` (the `--quantization` CLI flag,
`converters/converter.py:28,61`) ends in `spec.optimize(quantization)` →
`_alias_variables()` **then** `_quantize()` (`model_spec.py:262-274` — aliasing first, so
a tied weight is quantized once; the aliased attr is a string by then and skipped).

`_quantize` (`model_spec.py:191-260`) visits every variable. The eligibility rule is
structural, not name-based:

> **quantizable ⇔ the spec has a sibling `{name}_scale` attribute**
> (`is_quantizable = hasattr(spec, "%s_scale" % key)`, `model_spec.py:206`).

So `LinearSpec.weight`, `EmbeddingsSpec.weight`, and `Conv1DSpec.weight` are quantizable
(`common_spec.py:47,58,65`) — embeddings and conv included — while **norm gammas/betas,
biases, and position encodings never are** (no `_scale` sibling). Per scheme
(`ACCEPTED_MODEL_TYPES`, `model_spec.py:27-37`):

- `int8` / `int8_float32` / `int8_float16` / `int8_bfloat16` — per-output-row symmetric:
  `amax = max(|row|)`, `amax[amax==0] = 127`, `scale = 127/amax`, `value = rint(value*scale)`
  (`model_spec.py:228-242`). 3D conv kernels are reshaped to 2D around the scale
  computation (`:230-235,240-242`). Same math as the C++ `Quantize` op — see
  `quantization-scheme-and-ops.md` §1.
- `int16` — one **global** scalar scale `2**10 / amax(all)` (10-bit values → 20-bit
  products → 12 bits of headroom; `model_spec.py:210-226`).
- `float16` / `bfloat16` / `float32` — quantizable vars are just cast (`:245-246`).

Non-quantizable but float vars (`is_convertible`, `:207`) follow the scheme's float
dtype: fp16 for `float16`/`int8_float16`, bf16 for the bfloat16 pair, fp32 for
`float32`/`int16`/`int8_float32` (`:248-256`). This is what makes "int8_float16" mean
_int8 weights + fp16 everything-else_.

**AWQ does NOT enter here.** Pre-quantized checkpoints bypass `_quantize` entirely: the
loader detects `config.quantization_config` (`transformers.py:1715-1732`, AWQ
GEMM/GEMV versions only) and `set_linear` assigns the packed `qweight`/`scales`/`qzeros`
straight onto `weight`/`weight_scale`/`weight_zero` (`transformers.py:237-243`). The
`weight_zero` variable is what routes Dense to the AWQ path at runtime
(`quantization-scheme-and-ops.md` scope note).

## 2. Convert-time weight fusion

The runtime expects certain weights pre-fused; converters build them:

- **QKV fusion** — `utils.fuse_linear(spec, layers)` (`converters/utils.py:4-35`)
  concatenates weights along the output dim (rows) and synthesizes zero biases when only
  some inputs have one. The canonical user is `set_attention`: q/k/v into
  `spec.linear[0]` for self-attention; for cross-attention q alone and k+v fused
  (`transformers.py:379-398`). Llama-family loaders `torch.cat((qw, kw, vw))` directly
  (`:1581`, `:755`). The C++ side then `Split`s after one GEMM
  (`attention-and-kv-cache.md`).
- **Pre-quantized fusion** — `fuse_linear_prequant` (`utils.py:37-53`) concatenates
  `weight`/`weight_scale`/`weight_zero` together, along axis 1 for AWQ-GEMM packing vs 0
  for GEMV (`transformers.py:1818-1824`).
- **NOT fused: gate+up.** SwiGLU FFNs keep two separate leaves — `ffn.linear_0` (gate)
  and `ffn.linear_0_noact` (up), e.g. `transformers.py:1584-1585` — there is no gate+up
  concat in this codebase.
- Other in-converter weight surgery: `permute_for_sliced_rotary` (`utils.py:56-75`,
  reorders QK rows for the sliced-RoPE layout) and `smooth_activation` (SmoothQuant
  folding, `utils.py:78-103` — scales the norm down and the linear up).

## 3. Aliases: tied weights serialized once

`_alias_variables` (`model_spec.py:169-188`) compares all variables pairwise
(element-wise equality, since load-time transforms may have copied tensors) and replaces
the alphabetically-later duplicate with a **string alias** to the earlier name —
except names in `SKIP_CREATING_ALIAS` (rotary long/short factors, `model_spec.py:38`).
`model.bin` stores the alias table after the variables (`model-binary-format.md`); the
C++ loader points both names at one `shared_ptr<StorageView>` and **also aliases the
associated `_scale`/`_zero`** (`src/models/model.cc:763-772`) — which is why a tied,
quantized embedding/projection pair just works.

## 4. How convert-time choices constrain runtime compute_type

The quantization scheme chosen at convert time becomes the **saved** compute type: at
load, `Model::infer_compute_type` reads it back off the variables' dtypes (weight dtype +
other-float dtype → `data_type_to_compute_type`, `src/models/model.cc:370-385`). The
saved/requested/effective resolution — what AUTO picks, what up/down-conversions are
allowed per device capability — is `compute-type-resolution.md`; the actual
re-quantize/dequantize work at load (`ensure_dtype` reusing the same scheme,
`model.cc:303+`) is `weight-loading-and-conversion.md`. Two consequences worth naming:

- Converting with plain `float32` loses nothing: any quantization can be applied at load.
  Converting with `int8` is lossy-once — loading as float32 dequantizes the _quantized_
  values, not the originals.
- AWQ models additionally write `quantization_type` into `config.json`, picked up as
  `quant_method` at load (`model.cc:635-636`) — that path cannot be re-targeted by
  compute_type.

---

### Relevance to the Metal backend

- Everything here is target-agnostic: an `int8` model converted once ran unchanged on
  CPU and Metal during the int8 project — the per-row 127/amax weights+scales are exactly
  what `metal::gemm_s8`/`dequantize_gemm_output_s8` consume.
- The conv-weight quantization (§1) is _undone at load_ on Metal by the model.cc float
  guard (`conv1d-op.md` §4) — convert-time int8 conv weights are fine to ship.
- Environment gotcha from this project (one bullet's worth): the converter must come
  from the **venv-installed wheel** built against your C++ tree — a stale
  homebrew/site-packages `ct2-transformers-converter` shadows it and emits models your
  locally-built runtime mis-loads (details in the int8-metal project memory).
