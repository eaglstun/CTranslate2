# Model import pipeline: specs & converters

Sourced from:

- `python/ctranslate2/specs/model_spec.py` — `LayerSpec` / `ModelSpec` base, serialization
- `python/ctranslate2/specs/common_spec.py` — leaf specs (`LayerNormSpec`, `LinearSpec`, `EmbeddingsSpec`, `Conv1DSpec`) + enums
- `python/ctranslate2/specs/transformer_spec.py` — a concrete spec tree
- `python/ctranslate2/converters/converter.py` — base `Converter` + `convert()` flow
- `python/ctranslate2/converters/transformers.py` — a concrete `Converter` + `ModelLoader` `set_*` pattern
- `src/models/model_factory.cc` + `include/ctranslate2/models/model_factory.h` — C++ loader registry

## 1. Big picture / data flow

```
external checkpoint
  --(Converter._load → ModelLoader)-->  ModelSpec   (declarative weight + layer tree, in memory)
  --(spec.validate / optimize / save)-->  CT2 model dir  (model.bin + config.json + *_vocabulary.json [+ vmap.txt])
  --(C++ create_model via ModelFactory)-->  runnable Model
```

A converter reads the source framework's weights, instantiates a `ModelSpec` subclass whose Python attribute tree mirrors the network, assigns each tensor onto a leaf attribute, then `convert()` validates, optionally quantizes/aliases, and serializes to `model.bin`. At load time the C++ side reads the spec **name** string from `model.bin` and looks it up in a registry (`ModelFactory`) to construct the matching `Model` class. The format is versioned: `CURRENT_BINARY_VERSION = 6` (`model_spec.py:25`) plus a per-spec `revision`.

File map:

- Specs: `python/ctranslate2/specs/` (`model_spec.py`, `common_spec.py`, `attention_spec.py`, `transformer_spec.py`, `whisper_spec.py`, `wav2vec2_spec.py`, `wav2vec2bert_spec.py`)
- Converters: `python/ctranslate2/converters/` (`converter.py` base; `transformers.py`, `fairseq.py`, `marian.py`, `opennmt_py.py`, `opennmt_tf.py`, `opus_mt.py`, `openai_gpt2.py`, `eole_ct2.py`)
- C++ loader: `src/models/model_factory.cc`, `include/ctranslate2/models/model_factory.h`, and the `src/models/*.cc` model classes

## 2. Specs (declarative weight + layer layout)

A **`LayerSpec`** (`model_spec.py:98`) is a plain object whose non-underscore attributes are either weights (`None` = required-but-unset, `np.ndarray`/`torch.Tensor` once assigned, or the sentinel `OPTIONAL = "__optional"`) or **nested sub-specs** (other `LayerSpec` instances, or lists of them). The tree of attributes _is_ the model layout. It's a `FrozenAttr` (`model_spec.py:91`): after construction you can only set attributes that already exist — assigning an unknown name raises, which catches converter typos.

Traversal is generic: `visit_spec` (`model_spec.py:57`) recursively walks `__dict__`, descending into `LayerSpec` children and into lists (naming list elements `name_0`, `name_1`, …), building scoped names like `encoder/layer_3/self_attention/linear_0/weight`. This scope string becomes the serialized variable name.

Key `LayerSpec` methods:

- `validate()` (`model_spec.py:101`) — visits every attr; any still `None` is collected and raises `ValueError`. Also normalizes types here: `float64→float32`, Python `float→float32` scalar, `bool→int8`, `str→int8` UTF-8 byte array (unless it's `OPTIONAL`), wrapping ndarrays/tensors in `NumpyVariable`/`PyTorchVariable`.
- `optimize(quantization)` (`model_spec.py:262`) — calls `_alias_variables()` (dedup identical tensors, replacing the later one with a string alias to the earlier name; `model_spec.py:169`) then `_quantize()` (`model_spec.py:191`): a weight is quantizable when a sibling `<name>_scale` attribute exists (int8/int16 produce a scale; float16/bfloat16/float32 just cast). Accepted schemes in `ACCEPTED_MODEL_TYPES` (`model_spec.py:27`).
- `variables(prefix, ordered)` (`model_spec.py:143`) — flattens the tree to a `{scoped_name: value}` dict (skipping `OPTIONAL`).

A **`ModelSpec`** (`model_spec.py:323`) is the top-level `LayerSpec`. It adds `_config` (a `ModelConfig`) and `_files`, and is abstract via two properties a subclass must define:

- `name` (`model_spec.py:331`) — the spec-name string written into `model.bin` and used as the C++ registry key.
- `revision` (`model_spec.py:337`) — integer bumped whenever the weight layout changes.

`ModelSpec.save()` (`model_spec.py:364`) writes `model.bin` (via `_serialize`), `config.json`, and any registered files. `_serialize` (`model_spec.py:382`) is the binary format: u32 `CURRENT_BINARY_VERSION`, the spec name string, u32 `revision`, then the variable count and for each variable its name, shape, dtype id (`_dtype_to_type_id`, order matches `include/ctranslate2/types.h` `DataType`), byte length, and raw bytes; finally the alias table. Subclasses `SequenceToSequenceModelSpec` (`model_spec.py:464`) and `LanguageModelSpec` (`model_spec.py:578`) add vocabulary registration/validation and emit `*_vocabulary.json` (collapsing identical source/target vocabs to a single `shared`).

**Leaf specs** (the bottom of the tree), `common_spec.py`:

- `LayerNormSpec` (`common_spec.py:35`) — `gamma`, plus `beta` (LayerNorm) or `layer_norm_use_residual` OPTIONAL (RMS norm).
- `LinearSpec` (`common_spec.py:44`) — `weight` (required), `weight_scale`/`weight_zero`/`bias` OPTIONAL; `has_bias()` checks whether `bias` is still the OPTIONAL sentinel string.
- `Conv1DSpec` (`common_spec.py:55`), `EmbeddingsSpec` (`common_spec.py:62`, has `weight` + OPTIONAL `weight_scale`/`multiply_by_sqrt_depth`).
- Enums (`Activation`, `EmbeddingsMerge`, `Quantization`) are mirrored to C++ headers — comments name the counterpart, e.g. `Activation` matches `include/ctranslate2/ops/activation.h` (`common_spec.py:6`).

**Nesting example**, `transformer_spec.py`: `TransformerEncoderSpec` (`:10`) holds a list `self.layer = [TransformerEncoderLayerSpec(...) ...]` (`:96`), a `LayerNormSpec` (`:90`), an `embeddings` list of `EmbeddingsSpec` (`:83`), plus scalar config attrs stored as fixed-width numpy types (e.g. `self.num_heads = np.dtype("int16").type(num_heads)`, `:79`). Each `TransformerEncoderLayerSpec` (`:295`) nests `self.self_attention = MultiHeadAttentionSpec(...)` and `self.ffn = FeedForwardSpec(...)` (`:313`,`:328`). Optional sub-norms are added or removed dynamically (`delattr(self.self_attention, "layer_norm")`, `:342`). The concrete `TransformerSpec(SequenceToSequenceModelSpec)` returns `name = "TransformerSpec"` (`:582`) and `revision = 7` (`:586`).

## 3. Converters (Loader / Converter pattern)

Base class `Converter` (`converter.py:11`, abstract). The public entry is `convert()` (`converter.py:57`), which orchestrates:

1. `spec = self._load()` — the one abstract method (`converter.py:107`); returns a populated `ModelSpec` (or `None` → `NotImplementedError`).
2. optional `register_vocabulary_mapping(vmap)`,
3. `spec.validate()`,
4. `spec.optimize(quantization=...)`,
5. wipe/create `output_dir`, `spec.save(output_dir)`.

`declare_arguments` / `convert_from_args` (`converter.py:14`,`:40`) wire the shared CLI flags (`--output_dir`, `--vocab_mapping`, `--quantization`, `--force`) used by the `ct2-*-converter` console scripts.

A concrete converter implements `_load`. In `transformers.py`, `TransformersConverter._load()` (`:106`) reads the HF `AutoConfig`, looks up a registered loader by the config's class name (`_MODEL_LOADERS.get(config_name)`, `:113`), loads the HF model + tokenizer, then calls `spec = loader(model, tokenizer)` (`:159`). Loaders register themselves with the `@register_loader("<HFConfigName>")` decorator (`:57`), which instantiates the loader class into `_MODEL_LOADERS`.

The **`ModelLoader`** abstract base (`transformers.py:199`) is where weights actually move into the spec. `__call__` builds the spec, then calls `set_config`, `set_vocabulary`, and architecture-specific setters; the convention is small reusable `set_*` helpers that assign source tensors onto leaf-spec attributes by **direct attribute assignment**:

- `set_layer_norm(spec, module)` → `spec.gamma = module.weight; spec.beta = module.bias` (`:233`)
- `set_linear(spec, module, quant_type)` → `spec.weight = module.weight` (or `qweight`/`scales`/`qzeros` for AWQ), optional transpose, `spec.bias = module.bias` (`:237`)
- `set_embeddings(spec, module)` → `spec.weight = module.weight` (`:250`)
- `set_position_encodings(spec, module)` → `spec.encodings = module.weight` (`:253`)

Higher-level setters compose these by zipping spec sub-lists with source layers, e.g. `BartLoader.set_encoder` (`:332`) iterates `for layer_spec, layer in zip(spec.layer, encoder.layers): self.set_attention(...); self.set_linear(layer_spec.ffn.linear_0, layer.fc1); ...`. A loader names the HF spec it targets via `from_config` factory calls like `transformer_spec.TransformerSpec.from_config(...)` (`:296`). Other framework converters (`fairseq.py`, `marian.py`, `opennmt_*`, `opus_mt.py`, `openai_gpt2.py`, `eole_ct2.py`) follow the same `_load` → instantiate spec → assign-weights → return shape.

## 4. C++ loading

The serialized `name` string drives construction. `model_factory.cc` defines `register_supported_models()` (`:13`), which calls the templated `register_model<ModelClass>("<SpecName>"[, ctor args])` for each known spec name, e.g.:

```cpp
register_model<TransformerModel>("TransformerSpec");        // model_factory.cc:18
register_model<TransformerDecoderModel>("TransformerDecoderSpec");
register_model<TransformerEncoderModel>("TransformerEncoderSpec");
register_model<WhisperModel>("WhisperSpec");
register_model<Wav2Vec2Model>("Wav2Vec2Spec");
register_model<Wav2Vec2BertModel>("Wav2Vec2BertSpec");
```

Empty-string `""`, `"TransformerBase"`, `"TransformerBig"` are kept for backward compatibility (`model_factory.cc:14-17`), some with constructor args like `/*num_heads=*/8`.

The registry itself is a singleton `ModelFactory` (`model_factory.h:8`). `register_model` (`model_factory.h:16`) stores a `Builder` lambda (`std::function<std::shared_ptr<Model>()>`) keyed by name in `_registry`; `create_model(name)` (`model_factory.h:21`) looks the name up and invokes the builder, throwing `std::invalid_argument("Unknown model " + name)` on a miss. The free function `create_model` (`model_factory.cc:31`) runs `register_supported_models` once via `std::call_once` before delegating to the factory. The spec `revision` read from `model.bin` is checked by the model class against the revisions it supports (versioning is a stability guarantee per `docs/versioning.md`).

## 5. Adding a new architecture (practical checklist)

Per project CLAUDE.md, three coordinated changes:

1. **specs/** — add or extend a `ModelSpec`/`LayerSpec` subclass describing the weight + layer layout (reuse `LinearSpec`/`LayerNormSpec`/`EmbeddingsSpec` leaves). Set a stable `name` and bump `revision` if you change an existing layout.
2. **converters/** — add or extend a converter/`ModelLoader` (e.g. a new `@register_loader("...")` in `transformers.py`) that reads the source weights and assigns them onto the spec via the `set_*` helpers / direct attribute assignment, then returns the spec from `_load`.
3. **C++** — register the spec `name` in `src/models/model_factory.cc` (`register_model<YourModel>("YourSpec")`) and implement/extend the matching `src/models/*.cc` model class and any new `src/layers/*.cc` layers it needs.

### Relevance to the Metal backend

Specs, converters, and loading are entirely device-agnostic — a converted CT2 model directory and the `ModelFactory` construction path are identical regardless of target. A model loads onto `Device::METAL` exactly as it does on CPU; the device only changes _where ops execute_, not _how the model is imported_.
