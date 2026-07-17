---
title: "The HuggingFace converter: loader registry & ModelLoader anatomy"
summary: >-
  Dissects the internal architecture of
  python/ctranslate2/converters/transformers.py — the HuggingFace converter
  every new model goes through — centered on the _MODEL_LOADERS registry keyed
  by HF config class name (register_loader('Qwen2Config'), 42 stateless
  singleton loaders instantiated at import), resolved by
  TransformersConverter._load via AutoConfig with a hard error on a miss and
  no generic fallback (the version-coupling stale-install trap that bites
  Metal work since its runtime is always a local build). It lays out the
  ModelLoader contract and hook order (get_model_spec, set_config,
  get_vocabulary, set_vocabulary), architecture_name, the
  set_layer_norm/set_linear/set_embeddings weight idioms, get_rotary_params
  RoPE parsing, and the module-level activation/rope-scaling/quantization
  translation tables. Walks Qwen2Loader end-to-end (GQA num_heads_kv
  normalization, TransformerDecoderModelSpec.from_config keywords,
  set_decoder's per-layer q/k/v fuse_linear and gate/up/down mapping with
  delattr+gc.collect to keep peak RSS at one model), and notes adding a llama-
  family variant is often one new subclassed @register_loader class.
semantic_id: "S17y9wu-su3T4X19RvnLnSc8pyuSxKoDClVtHo68xt4qKWgEYb8tWg25eg0t740Rrm0k-837jY5uvQybxrzvsw"
---

The internal architecture of `python/ctranslate2/converters/transformers.py` — the
converter every new HF model goes through. `specs-and-converters.md` owns the overall
pipeline (spec tree, `Converter.convert()`, C++ factory); this card is the level below:
the loader registry, the `ModelLoader` contract, the version-coupling reality, and one
loader walked end-to-end (Qwen2).

**Sources (all citations from real lines):**

- `python/ctranslate2/converters/transformers.py` (~4265 lines, 42 `@register_loader` classes)
- `python/setup.py` (the `ct2-transformers-converter` console script)

## 1. The registry: one loader per HF _config class name_

`_MODEL_LOADERS = {}` (`transformers.py:54`) is keyed by the **name of the HF config
class**, not the model name: `register_loader("Qwen2Config")` etc. The decorator
_instantiates_ the class at import time (`_MODEL_LOADERS[config_name] = cls()`,
`transformers.py:57-64`) — loaders are stateless singletons, all 42 registered the
moment the module imports.

`TransformersConverter._load()` (`transformers.py:106-171`) resolves the loader:

1. `AutoConfig.from_pretrained(...)`, then `config_name = config.__class__.__name__`
   and `_MODEL_LOADERS.get(config_name)` (`:108-113`).
2. **Miss → hard error**: `"No conversion is registered for the model configuration X
(supported configurations are: ...)"` (`:115-120`).
3. The HF model class comes from `getattr(transformers, loader.architecture_name)`
   (`:122`), overridable via optional `get_model_class`/`get_model_kwargs` hooks on the
   loader (`:123-129`).
4. dtype: `--quantization float16/int8_float16` sets `load_as_float16` (`python/setup.py:118`
   script → `main()`, `transformers.py:3663`), i.e. `from_pretrained(dtype=torch.float16)`;
   otherwise the checkpoint's own `dtype`/`torch_dtype` (`:131-138`).
5. `spec = loader(model, tokenizer)` (`:159`), then optional SmoothQuant scales
   (`loader.smooth_activation`, `:161-165`) and `--copy_files` registration (`:167-169`).

### Version coupling (the stale-install trap)

**A loader exists ⇔ this converter version supports the architecture.** There is no
fallback path, no "generic transformer" loader — the registry lookup at `:113` is the
single gate. So `ct2-transformers-converter` from a stale install (old wheel shadowing
a fresh build) fails on any newer architecture with the `:115-120` error, _or worse_,
silently converts with an older loader whose spec options the local runtime disagrees
with. This project hit exactly this converting Qwen2 (see the int8-metal project memory
and `converter-quantization-and-fusion.md`'s venv gotcha): always check
`which ct2-transformers-converter` resolves into the venv whose wheel matches the C++
tree.

## 2. The `ModelLoader` contract (`transformers.py:199-286`)

`__call__(model, tokenizer)` (`:210-217`) fixes the hook order:

```python
spec = self.get_model_spec(model)          # abstract (:206-208) — the only required method
self.set_config(spec.config, model, tokenizer)   # default no-op (:230)
tokens = self.get_vocabulary(model, tokenizer)   # default: tokenizer vocab sorted by id (:219-225)
self.set_vocabulary(spec, tokens)                # default no-op (:227)
```

- `architecture_name` (property, default `None`, `:202-204`) — the `transformers`
  attribute naming the model class to instantiate (e.g. `"Qwen2ForCausalLM"`).
- The `set_*` weight idioms (`set_layer_norm`/`set_linear`/`set_embeddings`/
  `set_position_encodings`, `:233-257`) are direct attribute assignment onto leaf specs
  — covered in `specs-and-converters.md` §3; the AWQ branch of `set_linear`
  (qweight/scales/qzeros) in `converter-quantization-and-fusion.md` §1. One detail
  neither names: `set_linear` transposes `transformers.Conv1D` modules (`:245-246`,
  GPT-2-style linears stored transposed).
- `get_rotary_params(config, default_rope_theta)` (`:264-286`) — shared RoPE-config
  reader: parses `config.rope_scaling` (`type` or `rope_type` key), maps through
  `_SUPPORTED_ROPE_SCALING` (`:42-47`: linear/su/llama3, `"longrope"→Su`), raises
  `NotImplementedError` on unknown types, returns `(scaling_type, factor, rope_theta)`.

Three module-level translation tables feed every loader:
`_SUPPORTED_ACTIVATIONS` (`:30-40`, HF activation strings → `common_spec.Activation`;
note `gelu_fast`/`gelu_new`/`gelu_pytorch_tanh` all → `GELUTanh`, `silu`/`swish` →
`SWISH`), `_SUPPORTED_ROPE_SCALING` (`:42-47`), and `_SUPPORTED_QUANTIZATION`
(`:49-52`, AWQ `gemm`/`gemv` only).

## 3. Worked anatomy: `Qwen2Loader` (`transformers.py:2513-2651`)

`get_model_spec` (`:2519-2573`) shows the full decision surface a decoder-only loader owns:

- **GQA**: `num_heads_kv = getattr(config, "num_key_value_heads", num_heads)`,
  normalized to `None` when equal to `num_heads` (`:2522-2525`) — MHA models don't set
  the spec option at all.
- **RoPE**: `self.get_rotary_params(config, 10_000)` (`:2527-2529`).
- **Pre-quantized detection**: `config.quantization_config` → AWQ version lookup in
  `_SUPPORTED_QUANTIZATION`, else `Quantization.CT2` (`:2532-2551`).
- **The spec call** (`:2553-2569`): `TransformerDecoderModelSpec.from_config(num_layers,
num_heads, activation=SWISH, pre_norm=True, ffn_glu=True, rms_norm=True, rotary_dim=0,
rotary_interleave=False, rotary_scaling_*, rotary_base=rope_theta, num_heads_kv=...,
quant_*)` — every llama-family architectural choice is a keyword here.
- Then `set_decoder(spec.decoder, model.model)` and
  `set_linear(spec.decoder.projection, model.lm_head)` (`:2571-2572`).

`set_decoder` (`:2601-2651`) is the per-layer zip: `input_layernorm` →
`self_attention.layer_norm`, `post_attention_layernorm` → `ffn.layer_norm`
(`:2606-2612`); q/k/v into three temp `LinearSpec`s then `utils.fuse_linear` into
`self_attention.linear[0]` (or `fuse_linear_prequant` with the AWQ-GEMM-vs-GEMV concat
dim, `:2614-2631`); `o_proj` → `linear[1]`; gate/up/down →
`linear_0`/`linear_0_noact`/`linear_1` (`:2639-2647`). After each layer it
`delattr(layer, "self_attn")`/`delattr(layer, "mlp")` + `gc.collect()` (`:2649-2651`)
— the converter frees HF tensors as it goes so peak RSS ≈ one model, not two.

The other hooks: `get_vocabulary` pads to `config.vocab_size` with `<extra_id_%d>`
tokens (`:2575-2581`); `set_config` maps `bos_token` (falling back to the pad token),
`eos_token`, `unk_token` (fallback `""`), and `layer_norm_epsilon = rms_norm_eps`
(`:2586-2596`); `set_layer_norm` is overridden gamma-only for RMSNorm (`:2598-2599`).

Many loaders subclass other loaders instead of `ModelLoader` (e.g.
`MarianMTLoader(BartLoader)` `:421-422`, `WhisperLoader(BartLoader)` `:932-933`) and
override only what differs.

## 4. Adding a new architecture: the realistic diff surface

For a model that is "llama-family with a twist", the entire diff is **one new
`@register_loader("NewConfig")` class** — usually subclassing the nearest existing
loader and overriding `architecture_name` + the divergent `set_*`/`get_model_spec`
bits (compare `Qwen3Loader` right below Qwen2, `:2654`, which adds `head_dim` and
q/k norms via a `_set_self_attention` helper). No spec or C++ change is needed as long
as every structural choice maps to an existing `from_config` keyword. Only a genuinely
new structural option requires the full three-file checklist
(`specs-and-converters.md` §5: spec field + revision bump, loader, C++ layer wiring).

### Relevance to the Metal backend

- The converter is 100% device-agnostic — nothing here knows about Metal; a model
  converted once runs on CPU/CUDA/Metal unchanged.
- But the **version coupling bites Metal work hardest**: the Metal runtime is always a
  local build, so the converter must be the wheel built against the same tree
  (`python-bindings-architecture.md` §5, the downstream rig's force-reinstall).
- `Qwen2Loader` is the loader behind the int8-Metal canonical model (Qwen2.5-0.5B):
  the spec options it sets (`num_heads_kv`, `rms_norm`, `ffn_glu`, rotary) are exactly
  what `tests/downstream/drivers/qwen_driver.py` exercises on `Device::METAL`.
