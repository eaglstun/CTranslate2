# The CT2 model binary format

The serialized model directory the C++ loader reads and the Python converters write.

> **STABLE SURFACE.** Converted models are covered by the project's semantic-versioning
> backward-compatibility guarantee (`docs/versioning.md` lists "Converted models" first):
> every model ever produced by a converter must keep loading in newer CTranslate2
> releases. Forward compatibility is explicitly **not** guaranteed — the loader rejects
> models newer than itself (`check_version`, `model.cc:457-469`). Do not change the
> writer or reader casually; additions must be gated on a new binary version.

**Sources (both ends of the wire, all citations from real lines):**

- writer: `python/ctranslate2/specs/model_spec.py` (`ModelSpec._serialize`, `save`)
- reader: `src/models/model.cc` (`Model::load`, the `consume<T>` helpers)
- `include/ctranslate2/models/model.h` (`current_binary_version`)
- `docs/versioning.md`

---

## 1. What lives where in a model directory

`ModelSpec.save` (`model_spec.py:364-380`) writes:

- **`model.bin`** — all weights + aliases, the binary described below
  (`_serialize`, `model_spec.py:382-414`; reader constant `binary_file`, `model.cc:19`).
- **`config.json`** — the `ModelConfig` as pretty JSON (`save_as_json`,
  `model_spec.py:311-320`); parsed into `model->config` at load (`model.cc:613-617`).
  Tokens (`bos_token`…), behavior flags, and `quantization_type` live here.
- **vocabulary JSON files** — written by the spec subclasses, not the base:
  seq2seq specs save `source_vocabulary.json` / `target_vocabulary.json`, **collapsed to a
  single `shared_vocabulary.json` when all vocabularies are identical**
  (`SequenceToSequenceModelSpec.save`, `model_spec.py:538-549`); language models save
  `vocabulary.json` (`model_spec.py:612-617`); `_save_vocabulary` is a plain JSON token
  list (`model_spec.py:620-624`).
- **extra registered files** — copied verbatim via `register_file`
  (`model_spec.py:354-362`), e.g. the vocabulary map `vmap.txt` (`model_spec.py:504-510`).

Scalar model attributes are **not** in config.json — they are serialized as variables
(bools → int8, strings → int8 arrays; `LayerSpec.validate`, `model_spec.py:114-133`) and
read back through `Model::get_attribute` (`model.h:127-141`).

## 2. `model.bin` layout

Writer (`_serialize`, `model_spec.py:392-414`) and reader (`Model::load`,
`model.cc:586-658` + `763-774`) agree on, in order (all little-endian `struct` packing;
strings are `uint16 length+1`, UTF-8 bytes, then a NUL — `_write_string`,
`model_spec.py:394-397`, mirrored by `consume<std::string>`, `model.cc:80-87`):

```text
uint32   binary_version            # CURRENT_BINARY_VERSION = 6 (model_spec.py:25; model.h:20)
string   spec name                 # [v>=2] e.g. "TransformerSpec" -> create_model (model.cc:596-605)
uint32   spec_revision             # [v>=2]
uint32   num_variables
repeat num_variables:
  string   name                    # e.g. "encoder/layer_0/ffn/linear_0/weight"
  uint8    rank
  uint32   dim[rank]
  uint8    dtype_id                # [v>=4] index into the DataType enum
  uint32   num_bytes
  bytes    data[num_bytes]         # raw row-major buffer
uint32   num_aliases               # [v>=3]
repeat num_aliases:
  string   alias
  string   variable_name           # the target holding the actual data
```

- `dtype_id` order is pinned to the C++ `DataType` enum: `float32, int8, int16, int32,
float16, bfloat16` (`_dtype_to_type_id`, `model_spec.py:280-289`; enum at
  `types.h:16-23`). Adding an enum value mid-list would corrupt every existing model —
  append only.
- Pre-v4 files stored `uint8 item_size` + `uint32 num_elements` instead; the reader keeps
  the "old (and flawed)" item-size→dtype guess (`get_dtype_from_item_size`,
  `model.cc:426-438`, dispatched at `model.cc:647-655`).
- The reader streams each variable's bytes directly into a `StorageView` buffer
  (`model.cc:657-658`); truncated files fail with position info (`report_stream_error`,
  `model.cc:43-51`).

## 3. Two version numbers, two jobs

- **binary version** (file-format): bumped when the byte layout or load-time semantics
  change. Feature gates in the reader: ≥2 spec name/revision (`model.cc:598-603`), ≥3
  aliases (`model.cc:764`), ≥4 explicit dtype ids (`model.cc:647`), ≥5
  round-before-cast quantization (`round_before_cast_in_quantization`, `model.h:87-89`).
  Current is 6 on both ends (`model_spec.py:25`, `model.h:20`).
- **spec revision** (per-architecture weights layout): `ModelSpec.revision`
  (`model_spec.py:336-343`) is "incremented each time the weights layout of the model is
  changed (e.g. a weight is renamed)". The C++ counterpart is
  `Model::current_spec_revision` (default 1, `model.cc:166-168`), overridden per model.
  `check_version` runs on **both** numbers (`model.cc:593-594`, `611`): saved ≤ current,
  i.e. new code reads old models, never the reverse. Handling an old revision means the
  model subclass remaps names — the `register_variable*` hooks are virtual for exactly
  this (`model.h:164-168`).

## 4. Aliases — dedup on disk, shared pointers in memory

The converter dedups identical tensors before writing: `_alias_variables`
(`model_spec.py:169-189`) compares variables pairwise (element-wise `equal`, keeping the
alphabetically-first copy) and replaces duplicates' values with the **name string** of the
survivor; `_serialize` splits string-valued entries into the alias table
(`model_spec.py:384-390`, written at `411-414`). Typical case: tied input/output
embeddings stored once. (`SKIP_CREATING_ALIAS` exempts the rotary scaling factors,
`model_spec.py:38`.) At load, an alias becomes a second map entry sharing the target's
`shared_ptr<StorageView>` (`register_variable_alias`, `model.cc:276-281`), and the loader
speculatively aliases `{alias}_scale` / `{alias}_zero` too (`model.cc:769-772`) so
quantized aliased weights keep their scales.

## 5. Quantization is baked in by the writer

`ModelSpec.optimize(quantization=...)` runs alias detection then `_quantize`
(`model_spec.py:262-273`): weights with a sibling `{name}_scale` attribute get per-row
int8 scales `127 / amax(|row|)` (conv 3D weights reshaped to 2D around the scale
computation, `model_spec.py:229-242`) or the int16 global scale (`model_spec.py:209-221`).
Accepted type strings are `ACCEPTED_MODEL_TYPES` (`model_spec.py:27-36`). The C++ loader
reproduces "the same quantization logic as in model_spec.py" when it must re-quantize at
load (`model.cc:325-328`) — the two ends are deliberately twinned; change one, change both.

### Relevance to the Metal backend

- The format is fully device-agnostic: the same `model.bin` (including int8 models with
  their `_scale` variables) loads on CPU, CUDA, and Metal with **zero format changes** —
  the int8-on-Metal branch touched capability flags and load-time conversion
  (`compute-type-resolution.md`, `weight-loading-and-conversion.md`), never the wire format.
- Loading is entirely host-side stream I/O into `StorageView` buffers; on Metal the later
  `to(device)` move lands bytes in real `MTLBuffer`s, and unified memory keeps the
  host-pointer contract intact (see the `apple-silicon` skill).
- Whether a stored int8 weight _stays_ int8 on Metal is decided after parsing, in
  `set_compute_type` — including the conv-weight float guard.
