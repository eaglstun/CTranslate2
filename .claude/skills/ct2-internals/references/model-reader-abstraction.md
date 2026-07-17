---
title: "ModelReader abstraction (directory vs in-memory models)"
summary: >-
  Explains the ModelReader interface that decouples model loading from the
  filesystem: two virtuals, get_model_id (a display string for logs/errors)
  and get_file(filename, binary) returning an istream or nullptr, wrapped by
  the throwing get_required_file. Contrasts the two implementations,
  ModelFileReader (joins model_dir + filename, opens an ifstream) and
  ModelMemoryReader (user register_file with each file's bytes as a
  std::string, served zero-copy via an imemstream, ignoring the binary flag)
  for the embed-a-model-in-your-app path. Traces exactly what Model::load
  requests and in what order: the sole required model.bin, optional
  config.json, then the model->initialize hook where LM/Whisper request a
  vocabulary (json-then-txt fallback) and seq2seq requests
  shared/target/source vocabularies plus optional vmap.txt, noting there is no
  directory-listing API, only point lookups, which is what makes the map-
  backed memory reader sufficient. Covers the Python files= constructor arg
  building a ModelMemoryReader, contains_model, and the Metal note that reader
  choice is device-orthogonal (bytes land in host memory then move to
  Device::METAL) with a ~2x peak-RSS cost during load.
---

How `Model::load` gets its bytes: the small `ModelReader` interface that decouples
loading from the filesystem, its two implementations (`ModelFileReader`,
`ModelMemoryReader` — the embed-a-model-in-your-app path), and the file-request
contract a custom reader must satisfy. What's _inside_ model.bin/config.json is
`model-binary-format.md`; this card is only about how they're fetched.

**Sources (all citations from real lines):**

- `include/ctranslate2/models/model_reader.h` + `src/models/model_reader.cc`
- `src/models/model.cc` (`Model::load`, `contains_model`, `ModelLoader`)
- `src/models/language_model.cc`, `src/models/sequence_to_sequence.cc` (the
  `initialize` hooks that request vocabulary files)

## 1. The interface (`model_reader.h:14-27`)

Two virtuals: `get_model_id()` (a display string — used only in error messages and the
"Loaded model X on device Y" log, `src/models/model.cc:872-875`) and
`get_file(filename, binary=false)` returning a `unique_ptr<std::istream>` or
**nullptr** when the file doesn't exist. The non-virtual `get_required_file` wraps it
and throws `"Unable to open file '...' in model '...'"` (`src/models/model_reader.cc:8-15`).
That's the whole contract: a reader is a named map from filename → istream.

- **`ModelFileReader`** (`model_reader.h:29-38`; `.cc:18-35`) — joins
  `_model_dir + "/" + filename`, opens an `ifstream` (binary mode when asked), returns
  nullptr if it didn't open. `get_model_id()` is the directory path.
- **`ModelMemoryReader`** (`model_reader.h:40-53`; `.cc:54-76`) — user calls
  `register_file(filename, content)` with each file's bytes as a `std::string`;
  `get_file` serves a zero-copy `imemstream` (a `std::streambuf` pointed at the stored
  string, `.cc:38-51`). The `binary` flag is ignored (`.cc:67-68`) — memory streams
  have no CRLF translation to disable. This is the embedded-model path: ship model.bin
  inside an app bundle / decrypt to memory, never touch disk.

## 2. What gets requested, in order

`Model::load(const std::string& path, …)` just wraps a `ModelFileReader`
(`src/models/model.cc:550-558`); the real loader takes a `ModelReader&` (`:560`):

1. **`model.bin`** — `get_required_file(binary_file, /*binary=*/true)` (`:586`; the
   constant is `"model.bin"`, `model.cc:19`). The only _required_ file.
2. **`config.json`** — `get_file(config_file)` (`:614`, constant at `:20`); optional —
   absent config just means no `model->config` JSON.
3. After variables load, **`model->initialize(model_reader)`** (`:779`) — a virtual
   hook (`include/ctranslate2/models/model.h:171`, default no-op) where model classes
   request their extra files:
   - LM / Whisper: `load_vocabulary(model_reader, "vocabulary", …)`
     (`language_model.cc:33`, `src/models/whisper.cc:31`). The helper tries `<name>.json` then
     falls back to `<name>.txt` (`model_reader.cc:79-94`).
   - Seq2seq: `"shared_vocabulary"`, else `"target_vocabulary"` +
     `"source_vocabulary"` (or `source_1_vocabulary`… multi-feature)
     (`sequence_to_sequence.cc:20-44`), plus optional `vmap.txt`
     (`get_file(vmap_file)`, `:53`).

So a custom reader must serve: `model.bin` (always), the vocabulary file(s) for the
model type (json or txt), and optionally `config.json` / `vmap.txt`. Nothing else is
ever requested — there is no directory listing API, only point lookups, which is what
makes the map-backed memory reader sufficient.

Related entry points: `contains_model(path)` is literally "does
`ModelFileReader(path).get_file("model.bin")` succeed" (`model.cc:810-812`), and the
replica-pool `ModelLoader` holds a `shared_ptr<ModelReader>` with ctors for both a
path and a custom reader (`include/ctranslate2/models/model.h:211-227`;
`model.cc:814-821`) — every pool (Translator/Generator/…) can therefore load from
memory.

## 3. Python surface

The bindings expose this as the `files` constructor argument: a dict of
`{filename: file-like}` is read into a `ModelMemoryReader` before constructing the
pool (`python/cpp/replica_pool.h:11-36`; see `python-bindings-architecture.md` §1).
There is no Python `ModelReader` subclassing — custom readers are a C++-embedding
feature.

### Relevance to the Metal backend

- Reader choice is orthogonal to device: the istream bytes land in host memory first
  and reach `Device::METAL` through the normal load pipeline
  (`weight-loading-and-conversion.md`'s device move) — an in-memory model loads onto
  Metal exactly like a directory one.
- The embedded use case (Mac/iOS app shipping a model in the bundle) is the natural
  Metal pairing: `ModelMemoryReader` + `device="metal"` needs no filesystem layout at
  runtime.
- One cost note: `ModelMemoryReader` holds every registered file as a `std::string`,
  so peak RSS during load ≈ model bytes ×2 (the blob + the materialized weights) until
  the reader is destroyed.
