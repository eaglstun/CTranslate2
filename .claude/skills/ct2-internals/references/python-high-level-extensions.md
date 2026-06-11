# Python high-level extensions (extensions.py)

The pure-Python conveniences monkey-patched onto the compiled `Translator`/`Generator`
classes — the file that answers "is this feature C++ or Python?". Everything listed
here is Python (`python/ctranslate2/extensions.py`); every _other_ method on those
classes is a pybind11 binding (`python-bindings-architecture.md`).

**Sources (all citations from real lines):**

- `python/ctranslate2/extensions.py` (the whole file, 589 lines)
- `python/ctranslate2/__init__.py` (registration at import)

## 1. Registration and the full surface

`register_extensions()` (`extensions.py:19-27`) runs at package import
(`python/ctranslate2/__init__.py:46-50`) and `setattr`s seven methods:

| Method                            | On         | Built on (C++ binding) | Def            |
| --------------------------------- | ---------- | ---------------------- | -------------- |
| `translate_iterable`              | Translator | `translate_batch`      | `:30`          |
| `score_iterable`                  | both       | `score_batch`          | `:91`, `:167`  |
| `generate_iterable`               | Generator  | `generate_batch`       | `:130`         |
| `generate_tokens`                 | both       | `*_batch` + callback   | `:204`, `:270` |
| `async_generate_tokens` (asyncio) | Generator  | `generate_batch`       | `:351`         |

That's the _entire_ file — there are no other public utilities in it.

## 2. `generate_tokens`: the callback → queue → generator bridge

Both `translator_generate_tokens` (`:204`) and `generator_generate_tokens` (`:270`)
funnel into `_generate_tokens` (`:473-524`). The mechanics:

- It forces `asynchronous=True`, `beam_size=1`, and installs its own `callback`
  (`:487-493`) — which is why streaming is **greedy-only** (the docstrings say so;
  beam hypotheses aren't stable per step).
- The C++ greedy loop invokes the callback per token from a **worker thread**
  (pybind11 re-acquires the GIL — `python-bindings-architecture.md` §2); `_callback`
  runs the user's callback (if any), puts the `GenerationStepResult` on a
  `queue.Queue`, and returns `generator_closed.is_set() or user_callback_result`
  (`:481-485`) — returning `True` force-finishes that batch entry in the decoder.
- A daemon thread drains the `AsyncResult`s purely to catch exceptions, pushing the
  exception object then a `None` sentinel onto the same queue (`:497-506`).
- The main thread is a plain generator: `queue.get()` → yield; `None` ends it,
  an `Exception` instance is re-raised (`:508-521`). **Early consumer exit**
  (`GeneratorExit`, `:519-521`) sets `generator_closed`, so the _next_ callback
  invocation returns `True` and the decode stops; `thread.join()` (`:524`) then waits
  for the job to actually terminate — breaking out of the loop is graceful, not
  instant.

Differences between the two fronts: the Generator variant forces
`include_prompt_in_result=False` and passes through `static_prompt`/
`cache_static_prompt` (`:344-347`); the Translator variant takes a single example
(`[source]`, `:250-253`) — no batching.

`generator_async_generate_tokens` (`:351`) wraps the same `_generate_tokens` in an
`AsyncGenerator` class (`:433-470`): a producer task copies steps into an
`asyncio.Queue` with an `await asyncio.sleep(0.0001)` per item (`:448-449` — the
comment admits it: without yielding control the event loop never runs the consumer).

## 3. `*_iterable`: streamed batching over the async API

`_process_iterable` (`:527-554`) is the shared driver: zip the input iterables
(`itertools.zip_longest`, `:534` — a length mismatch surfaces as `None` and raises in
`_batch_iterator:584-585`), force `asynchronous=True`, and prefetch with
`read_batch_size = max_batch_size * 16` (`:544`) — deliberately the same read-ahead
the C++ file loop uses (`include/ctranslate2/replica_pool.h:211`). Results live in a
`collections.deque` of async handles; it yields `popleft().result()` only while
`queue[0].done()` (`:550-551`), so order is preserved and the consumer never blocks on
an out-of-order future while there's reading left to do.

`_batch_iterator` (`:557-589`) implements `batch_type`: `"examples"` cuts at
`batch_size` items; `"tokens"` tracks the running max example length and cuts when
`(count + 1) * max_length > batch_size` (`:570-576`) — a padded-token estimate, same
philosophy as the C++ `BatchType::Tokens` fill (`batching-and-length-sorting.md`).

## 4. C++ or Python? (the recurring question, settled)

- `generate_tokens`, `async_generate_tokens`, `*_iterable` — **Python**, this file.
  (`generator-and-language-model.md` already proved this for `generate_tokens`.)
- `translate_batch`, `generate_batch`, `score_batch`, their `_async` forms, file
  translation, `unload_model`/`load_model` — **C++ bindings** (`python/cpp/*.cc`).
- Consequence: a bug in token streaming order/shutdown is debugged in Python; a bug in
  the per-step _content_ (the `GenerationStepResult` fields) is the C++ callback path
  (`decoding-loop-and-beam-search.md`).

### Relevance to the Metal backend

- These extensions are device-blind — `device="metal"` models stream tokens through
  the exact same queue bridge; nothing here syncs or touches buffers.
- The per-token callback fires after each decode step, i.e. inside the tiny-op decode
  regime — the Python queue hop adds latency per token but runs concurrently with the
  next step (worker thread), so it doesn't change the per-op GPU-API floor story.
- `_process_iterable`'s 16× prefetch is what keeps a Metal replica fed during
  prefill-heavy batch scoring/translation from Python — same saturation logic as the
  CLI loop (`cli-clients-and-perf-gating.md` §2).
