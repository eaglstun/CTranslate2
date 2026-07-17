---
name: ct2-internals
description: >-
  CTranslate2 engine internals — the device/dtype-agnostic architecture that sits below
  any compute backend. Use when working in src/ops, src/layers, src/models, the dispatch
  machinery, StorageView, or the Python specs/converters import pipeline: how an op is
  structured and dispatched, how tensors are represented, how transformer blocks are
  wired (norm placement etc.), and how external checkpoints become CT2 models. Pull the
  matching reference before reasoning about engine structure; verify line numbers still
  hold before acting on them.
---

# CTranslate2 internals

Deep-dive references for the CTranslate2 engine's architecture — the parts that are true
regardless of CPU/CUDA/Metal. `CLAUDE.md` has the one-paragraph layer map (kernels →
primitives → ops → layers → models → replicas); this skill is the next level down, with
verified file:line citations into the real code.

**This is engine structure, not a compute backend.** Backend-specific material lives
elsewhere: the Apple **Metal** GPU backend is the `apple-silicon` skill (MSL kernels,
MPS, the op-graduation procedure). When a task is "how does CT2 do X" → here; when it's
"how do I make X run on the GPU" → `apple-silicon`.

## References

- **[references/dispatch-and-op-implementation.md](references/dispatch-and-op-implementation.md)**
  — The 4-file op pattern (flag-free header in `include/` → `.cc` checks+dispatch →
  `_cpu.cc` → `_gpu.cu`) and the dispatch macros (`DEVICE_DISPATCH`, `TYPE_DISPATCH`,
  `DEVICE_AND_TYPE_DISPATCH`, `DEVICE_AND_FLOAT_DISPATCH`, `NON_FLOAT_CASE`) that resolve
  `<D, T>` at runtime, with the SoftMax walkthrough. _Read when adding/editing an op or
  understanding how device+dtype selection works._

- **[references/storageview.md](references/storageview.md)**
  — `StorageView`, the core row-major buffer-with-shape (no math semantics); dtype and
  device resolved at runtime; the resize/reserve allocation contract (smaller never
  reallocs; allocation churn is a bug) and caching allocators. _Read when touching tensor
  storage, allocation, or anything perf-sensitive about buffers._

- **[references/norm-placement-in-transformers.md](references/norm-placement-in-transformers.md)**
  — Where a norm sits in a transformer block: pre / post (`FeedForwardNetwork::_pre_norm`),
  pre-post "sandwich" (four-norm auto-detect), parallel-residual — across `specs/` →
  `converters/` → `src/layers/transformer.cc`. Placement is CPU orchestration. _Read for
  a norms task about block structure; the kernel/numerics side is the `apple-silicon`
  skill._

- **[references/attention-and-kv-cache.md](references/attention-and-kv-cache.md)**
  — `MultiHeadAttention` structure: fused QKV projection → `Split`, the
  `split_heads`/`combine_heads` layout transforms, **GQA/MQA** head grouping via
  `replicate_heads` (Tile, not a copy-loop), **RoPE** (`RotaryEmbeddings::apply`, the
  `offset` trick), and the **KV cache grown one step per token** by `Concat` on the time dim
  (plus sliding-window `Slide`). The structural why-so-many-tiny-ops behind the decode loop.
  _Read for the decode-step data flow, GQA/RoPE, or before touching the cache; the per-op
  perf consequence is the `apple-silicon` skill (`dispatch-overlap-and-perf-model.md`)._

- **[references/specs-and-converters.md](references/specs-and-converters.md)**
  — The model import pipeline: external checkpoint → converter → spec (declarative
  weight/layer layout) → serialized CT2 model → C++ loader via `model_factory.cc`. The
  LayerSpec/ModelSpec tree, the converter `set_*` pattern, model-type registration, and
  the "add a new architecture" checklist. _Read when adding model support or tracing how
  weights load._

### Quantization & model loading (int8 project, 2026-06)

- **[references/quantization-scheme-and-ops.md](references/quantization-scheme-and-ops.md)**
  — CT2's int8 scheme: symmetric per-row `scale = 127/amax`, no zero-point (the `_qzero`/AWQ
  path is a different scheme, scoped out); dynamic activation scales vs static
  `{scope}/weight_scale`; the Quantize ctor options (`int16_scale_type`, `shift_to_uint8`,
  `round_before_cast`) and the TWO Dequantize overloads — the gemm-output form owns
  scales+bias+activation over the int32 accumulator. _Read before touching anything quantized._

- **[references/gemm-op-and-dtype-dispatch.md](references/gemm-op-and-dtype-dispatch.md)**
  — `ops::Gemm` end-to-end: the dtype switch (int8 → `compute<D, int8_t, int32_t>`), the
  unconditional `apply_bias_and_activation` epilogue, `compute` → `primitives<D>::gemm`
  (MKL/DNNL/Ruy on CPU, cublasGemmEx on CUDA), the integer alpha/beta contract
  (convention from Dense, NOT an assert — backends truncate/emulate/guard differently),
  and the MKL u8-shift compensation story. _Read before touching any matmul path._

- **[references/dense-layer-and-quantized-linear.md](references/dense-layer-and-quantized-linear.md)**
  — `Dense::operator()` orchestration: `_quantized_gemm` (a weight-dtype bit) selects
  quantize→gemm→dequantize vs plain GEMM vs AWQ; member state resolved from model
  variables (`weight_scale`, `weight_zero`, `weight_compensation`); bias+activation ride
  Dequantize when quantized, Gemm when not. Device-agnostic by construction — the int8-Metal
  project shipped with zero diff to this file. _Read before touching any linear layer._

- **[references/compute-type-resolution.md](references/compute-type-resolution.md)**
  — How requested compute*type ("auto"/"int8"/"int8_float16"…) becomes effective per-weight
  dtypes: the saved/requested/effective triple, the `mayiuse*\*`capability queries, the`resolve*compute_type` fallback table (what AUTO picks per device; plain "int8" never
silently un-quantizes), and the Python surface (`get_supported_compute_types`,
`Generator(compute_type=...)`). \_Read before touching types.cc resolution or capability flags.*

- **[references/weight-loading-and-conversion.md](references/weight-loading-and-conversion.md)**
  — The model-load weight pipeline in `src/models/model.cc`: binary read → `register_variable`
  index → `set_compute_type`/`ensure_dtype` quantize/dequantize/cast with `{name}_scale`
  pairing → device move + synchronize. Centerpiece: the **conv-weight float guard**
  (CUDA/Metal/DNNL have no quantized conv) and the int8-Whisper load crash it prevents
  (commit 351b1990). _Read before changing load-time conversion or capability flags._

- **[references/model-binary-format.md](references/model-binary-format.md)**
  — The serialized model directory, documented from BOTH ends (`model_spec.py::_serialize`
  writer ↔ `Model::load` reader): model.bin record layout, binary version 6 vs per-spec
  revision, the backward-compat guarantee (STABLE SURFACE), alias dedup, config.json vs
  vocabulary files (shared*vocabulary collapse). \_Read before touching serialization.*

### Runtime core

- **[references/allocators-and-caching.md](references/allocators-and-caching.md)**
  — The `Allocator` abstraction behind `get_allocator(device)`: the pointer-based contract,
  the 64-byte-aligned CPU allocator (MKL variant), the two CUDA allocators (cub
  CachingDeviceAllocator — bin 4/3/12, 200MB cap, `CT2_CUDA_CACHING_ALLOCATOR_CONFIG` —
  vs `cudaMallocAsync`, chosen by `CT2_CUDA_ALLOCATOR`), and who owns the pointer
  (StorageView). _Read before touching allocation or memory-churn perf._

- **[references/devices-and-device-management.md](references/devices-and-device-management.md)**
  — `Device` enum, `str_to_device` ("auto" resolution order), per-backend
  get/set*device_index (CUDA = thread-global `cudaSetDevice`), `ScopedDeviceSetter`,
  `synchronize_device` vs `synchronize_stream` semantics, and how `device:index` reaches
  each replica via `ReplicaWorker::initialize`. \_Read for device plumbing or sync semantics.*

- **[references/primitives-layer.md](references/primitives-layer.md)**
  — The `primitives<Device>` struct: family-by-family survey of the BLAS-like interface
  (fill/copy, reductions, elementwise+broadcast, transpose, activations, gemm),
  explicit-instantiation-per-device (why a new Device case is expensive),
  `cross_device_primitives` copy, and the CPU two-level split (primitives = orchestration,
  kernels.cc = ISA inner loops). _Read when deciding where new array math belongs._

- **[references/cpu-isa-dispatch-and-kernels.md](references/cpu-isa-dispatch-and-kernels.md)**
  — Runtime ISA selection: `CT2_FORCE_CPU_ISA` (AVX512 is env-only, never auto), the
  CMake trick that copies kernels.cc per ISA with different flags (`-DUSE_NEON` on arm64),
  `TARGET_ISA` stamping + `CPU_ISA_DISPATCH`, the `Vec<T,ISA>` widths (NEON 4 / AVX 8 /
  AVX512 16), and the separate GEMM-backend priority (MKL→DNNL→Accelerate→OpenBLAS→Ruy).
  _Read before touching CPU kernels or vec headers._

- **[references/parallelism-and-thread-config.md](references/parallelism-and-thread-config.md)**
  — inter*threads (replica ThreadPool, queue backpressure 4×workers) vs intra_threads
  (`cpu::parallel_for`, GRAIN_SIZE 32768, no-nesting rule); the OpenMP-vs-BS::thread_pool
  runtimes (`OPENMP_RUNTIME=NONE` → BS pool, this machine's Metal builds), and
  `set_num_threads` plumbing (default min(4, hw); set per worker thread in
  `ReplicaWorker::initialize`). \_Read for any thread-count or CPU-perf question.*

### Op families

- **[references/activation-ops.md](references/activation-ops.md)**
  — `ActivationType` (enum order is FIXED — serialized + reused as kernel selectors),
  `get_activation_op` (one `GELU` op carries all three approximations), the exact formulas
  (erf vs tanh vs sigmoid GELU, from `src/cpu/kernels.cc` functors), and the three
  application sites (Gemm epilogue `apply_bias_and_activation`, dequantize gemm-output,
  FFN's `_ff1` pointer) that make fusion possible. Plus the converter mapping (gelu:
  BERT/Whisper; gelu*pytorch_tanh: Gemma2/3; silu: llama-family). \_Read before touching
  activations or their fusion path.*

- **[references/softmax-and-masking.md](references/softmax-and-masking.md)**
  — SoftMax/LogSoftMax: always last-dim, the int32 _lengths_ mask (one valid-length per
  row, padding written as exact 0 — not exp(-inf); built by `prepare_length_mask`),
  in-place forms, max-subtract CPU kernel, and where the 1/√d scale lives: folded into the
  QK^T **MatMul alpha** (`queries_scale`), never a SoftMax param. _Read for attention
  masking or softmax numerics._

- **[references/norm-ops.md](references/norm-ops.md)**
  — LayerNorm (axis ctor param, eps default 1e-5, gamma+beta, the outer/axis/inner general
  kernel) vs RMSNorm (gamma-only, last-axis only, eps default 1e-6). Epsilon sits
  **inside the sqrt** in both. Beta-presence selects the op at load; Gemma's (1+gamma) is
  the runtime `layer_norm_use_residual` flag — NOT baked into stored weights. _Read for
  norm numerics/parity; placement is norm-placement-in-transformers.md._

- **[references/shape-manipulation-ops.md](references/shape-manipulation-ops.md)**
  — The decode-loop data movers: Concat (KV-cache append), Split (QKV un-fuse; `no_copy`
  axis-0 views), Transpose (perm copy, rank 2-4 only), Tile (GQA `replicate_heads`), Slide
  (sliding-window cache trim), Gather (axis+batch*dims; embedding lookup + beam reorder,
  with the strictly-increasing in-place fast path), Squeeze/Unsqueeze (pure metadata).
  \_Read for the decode-step plumbing.*

- **[references/elementwise-and-bias-ops.md](references/elementwise-and-bias-ops.md)**
  — Add/Sub/Mul/Min/Max: broadcasting is **scalar-b or same-size flat, nothing else** (no
  shape checks in the elementwise branch — caller's contract), aliasing-safe in-place.
  `BiasAdd` is the separate axis-broadcast op carrying bias+residual+activation for
  fusion; `apply_bias_and_activation` is the glue from every GEMM/dequantize epilogue.
  _Read before touching elementwise or the bias path._

### Decode machinery

- **[references/decoding-loop-and-beam-search.md](references/decoding-loop-and-beam-search.md)**
  — The token-generation driver in `src/decoding.cc`: `decode()` →
  `GreedySearch`/`BeamSearch::search`, the per-step sequence (forward →
  DisableTokens/processors → LogSoftMax-if-needed → sampler → append), beam bookkeeping
  (`unflatten_ids`, gather-based cache reorder, length/coverage penalties), **batch
  shrinking** as hypotheses finish, and hard-prefix vs `BiasedDecoder` modes. _Read before
  touching the decode driver; the per-op perf consequence is `apple-silicon`._

- **[references/sampling-and-topk.md](references/sampling-and-topk.md)**
  — `Sampler`/`BestSampler`/`RandomSampler` (`src/sampling.cc`): the filter pipeline order
  (top-k → temperature Mul → `ops::TopPMask` → Multinomial/GumbelMax → gather-back), the
  TopK op contract (axis -1 only, values+indices `{batch,k}`), and the RNG story
  (`get_random_generator` thread*local mt19937; CPU≠CUDA streams). \_Sampling runs CPU-side
  on Metal over unified memory.*

- **[references/logits-processing.md](references/logits-processing.md)**
  — The `LogitsProcessor` machinery (`decoding_utils.{h,cc}`): the `DisableTokens`
  collector (CPU direct-write vs device `indexed_fill`), the five built-ins
  (RepetitionPenalty, NoRepeatNgram, SuppressTokens/Begin/Sequences), the fixed ordering
  in `make_logits_processors`, and Whisper's `ApplyTimestampRules` (the one processor
  doing real tensor math per step). _min_length is NOT a processor — it's
  `apply_min_length` in decoding.cc._

- **[references/batching-and-length-sorting.md](references/batching-and-length-sorting.md)**
  — `rebatch_input` (longest-first sort, the two documented reasons), `BatchType`
  tokens-vs-examples fill, the promise-indexed-by-`example_index` order restoration in
  `ReplicaPool::post_examples`, and the `Padder` gather-based padding removal
  (`allow_padding_removal`: never for fp16 off-CPU). _Read before touching batching, the
  replica pool plumbing, or padded shapes._

- **[references/position-encodings.md](references/position-encodings.md)**
  — The position-encoding family: additive `PositionEncoder` (Sinusoidal — positions start
  at 1 — vs learned `PositionEmbedding`), ALiBi (slope construction, `ops::AlibiAdd` after
  the score GEMM), T5 `relative_attention_bias` (bucketed, cached across layers) vs
  Shaw-style relative keys/values, and the full RoPE option table (`rotary_dim`/
  interleave/base, scaling None/Linear/Su/Llama3 — "longrope"→Su). _RoPE apply mechanics
  stay in attention-and-kv-cache.md._

### Models

- **[references/transformer-model-wiring.md](references/transformer-model-wiring.md)**
  — From spec config to constructed layer graph: the `as_sequence_to_sequence`/
  `as_sequence_generator` factories, the encoder/decoder ctors resolving everything from
  scoped variables (`build_embeddings_scale` — flag _or_ value; `build_position_encoder`
  skipped when attention has RoPE/ALiBi), final norm, **tied embeddings** (converter alias
  dedup → `register_variable_alias`; zero tying logic in C++), and the spec-attribute →
  `get_attribute_with_default` table. _Read before wiring or tracing model assembly._

- **[references/whisper-model-internals.md](references/whisper-model-internals.md)**
  — The Whisper surface: encode/generate/detect*language/align on `WhisperReplica`, the
  2×Conv1D+GELU stem, prompt structure + the `forward_prompt` prefill/decode split,
  no_speech via `GetNoSpeechProbs`, and align's LayerNorm→MedianFilter→Mean→CPU-DTW
  pipeline with config.json alignment heads. \_ApplyTimestampRules mechanics stay in
  logits-processing.md.*

- **[references/generator-and-language-model.md](references/generator-and-language-model.md)**
  — Decoder-only runtime: `Generator : ReplicaPool<SequenceGeneratorReplica>` (async-only
  C++ surface; `generate_tokens` is a Python extension over the step callback), the two
  prefill paths in `run_generation` (cached **static_prompt** Tile-copied per batch vs
  common-prefix forward), and scoring = `score_sequences` teacher-forced forward +
  LogSoftMax + Gather. _Qwen downstream driver = canonical consumer._

- **[references/translator-and-seq2seq.md](references/translator-and-seq2seq.md)**
  — The seq2seq practical card: `translate_batch` → `EncoderDecoderReplica::run_translation`,
  encode→decode handoff via `state["memory"]`, `make_target_ids` prefix vs scoring modes,
  the full TranslationOptions→DecodingOptions enforcement table (incl. `use_vmap`
  output-layer restriction), and when `TranslationResult.attention` is populated. _NLLB
  downstream driver = the enc-dec proof._

### Infrastructure, tests & bindings

- **[references/replica-pools-and-async-api.md](references/replica-pools-and-async-api.md)**
  — The header-only `ReplicaPool<Replica>` template behind Translator/Generator/Encoder/
  Whisper: Job/JobQueue/Worker mechanics, `BatchJob`'s promise-per-result +
  exception*ptr-fans-out contract, `ModelLoader` (replicas on one device **share** the
  const Model; cross-device copies), and the streaming `callback` option (greedy-only).
  \_Read for pool lifecycle; thread counts and rebatch live in their own refs.*

- **[references/python-bindings-architecture.md](references/python-bindings-architecture.md)**
  — `python/cpp/*.cc`: `ReplicaPoolHelper` (inter→num*replicas_per_device, intra→pool
  config), the three GIL release points (`py::call_guard`, `AsyncResult::result()`,
  ctor/dtor), StorageView via `__array_interface__`/`__cuda_array_interface__` (NOT
  DLPack; Python Device enum has no metal), and the CTRANSLATE2_ROOT /
  rebuild-lib-then-wheel linkage rule. \_Read before touching bindings or wheel builds.*

- **[references/vocabulary-and-tokenization-boundary.md](references/vocabulary-and-tokenization-boundary.md)**
  — `Vocabulary` (token↔id, unk auto-appended, EOS-preserving truncation) and the
  tokens-in/tokens-out boundary (CT2 never tokenizes); `VocabularyMap`/`vmap.txt`
  target-vocab restriction → `Decoder::update_output_layer` physically shrinks the output
  projection via `select_weights`. _Read for vocab plumbing or the vmap feature._

- **[references/profiling-infrastructure.md](references/profiling-infrastructure.md)**
  — `ENABLE_PROFILING`/`PROFILE()` RAII scoped timers: cross-thread by-name aggregation,
  parent self-time subtraction, **stream sync at every scope boundary** (distorts async
  backends), the %self/%total/%cum dump format, and `--log_throughput` = best-hypothesis
  tokens / wall time. `init_profiling` THROWS on a non-profiling build. _Read before
  perf-gating a change._

- **[references/logging-and-env-config.md](references/logging-and-env-config.md)**
  — spdlog wiring (`CT2_VERBOSE` −3…3, default 0=Warning) and the complete grepped
  env-var table (CT2*FORCE_CPU_ISA, CT2_USE_MKL, CT2_PACKED_GEMM, CT2_CUDA*\* ×5,
  OMP*NUM_THREADS). The operational debugging card. \_Read before reaching for an env knob
  — several folklore vars don't exist (and `CT2_NO_MPS_ACT` was removed with the Gemma2
  fix).*

- **[references/ops-test-suite-structure.md](references/ops-test-suite-structure.md)**
  — The C++ test suite: one gtest binary (data dir = argv[1]), `OpDeviceTest`/
  `OpDeviceFPTest` value-parameterized over `Device`/`FloatType{device,dtype,error}`,
  instantiations gated by compile-time `#ifdef` (CPU fp32 1e-5; CUDA fp16 1e-2; METAL
  fp32-only + `GTEST_SKIP()` fixtures), `expect_storage_eq` (to-CPU copy + abs-eps only),
  and the 5-step recipe for an op test that covers all devices free. _The oracle — read
  before adding/judging tests._

- **[references/cuda-backend-structure.md](references/cuda-backend-structure.md)**
  — The CUDA backend as the reference GPU backend: shared infra in `src/cuda/` (per-op
  kernels are `src/ops/*_gpu.cu`, NOT `src/cuda/`), thread*local stream + cuBLAS/cuDNN
  handle per host thread, the `cublasGemmEx` dtype table (int8 = `CUDA_R_8I`→`CUDA_R_32I`,
  compensation param ignored), and how CUDA is a real `DEVICE_CASE` (no "CUDA_CASE" macro
  exists). The three properties int8-Metal mirrored, cited both sides. \_Read before
  structuring any new backend work.*

### Weights & projections

- **[references/embeddings-and-output-projection.md](references/embeddings-and-output-projection.md)**
  — The bookends: `Embeddings` = Gather (+ gathered-scale Dequantize when the table is
  int8 — it CAN be; spec+load+runtime all agree), the √d scale applied by encoder/decoder
  Mul not the layer, `Dense` as lm*head (`decoder/projection`, trans_b means [vocab,depth]
  = embedding layout so tying needs no transpose), and vocab restriction via
  `update_output_layer`. \_Temperature is sampling, not here.*

- **[references/conv1d-op.md](references/conv1d-op.md)**
  — Conv1D: ctor (stride/padding/dilation/groups + fused activation), CPU =
  im2col-transposed+GEMM by default vs DNNL direct, CUDA = cuDNN, and the dtype matrix:
  exactly ONE backend runs int8 conv — CPU-without-DNNL; the model.cc load guard forces
  conv weights float on CUDA/Metal/DNNL. Users: Whisper stem, wav2vec2(-BERT) only.
  _Bridge: Metal's fp16 upcast island + mps-convolution-options.md for graduation._

- **[references/converter-quantization-and-fusion.md](references/converter-quantization-and-fusion.md)**
  — What converters do to weights beyond layout: `_quantize` (eligibility = sibling
  `{name}_scale` attr exists — embeddings/conv ARE quantizable, norms/biases never; int8
  per-row 127/amax), AWQ bypasses `_quantize` via `set_linear` qweight/scales/qzeros,
  `fuse_linear` QKV concat (gate+up is NOT fused — separate `linear_0`/`linear_0_noact`),
  alias-before-quantize ordering, and how the saved scheme becomes `infer_compute_type` at
  load. _Cross-refs compute-type-resolution + weight-loading._

### Layers, features & tooling

- **[references/feed-forward-network-layer.md](references/feed-forward-network-layer.md)**
  — `FeedForwardNetwork`: standard 2-linear FFN vs GLU (gate = `linear_0` _with_
  activation, up = `linear_0_noact`, combined by plain `ops::Mul` — weight presence IS
  the flag, no runtime attribute), the residual fused into `_ff2`'s Dense epilogue only
  when the FFN owns its norm, and the converter gate/up mapping (llama
  `mlp.gate_proj/up_proj`, T5 `wi_0/wi_1`). _Activation formulas stay in
  activation-ops.md; placement in norm-placement._

- **[references/decoder-state-contract.md](references/decoder-state-contract.md)**
  — The `DecoderState` map contract: per-layer `self_keys_<i>`/`self_values_<i>` (+
  `memory_keys_/memory_values_<i>` for enc-dec) created **empty** by `initial_state`
  (emptiness = first-step signal), the model-placed `memory`/`memory_lengths` handoff and
  the `state.erase("memory")` after step 0, why `memory*` keys are never beam-replicated,
  and the dim-0-is-batch Gather contract. _Update mechanics live in decoding-loop; growth
  in attention-and-kv-cache._

- **[references/encoder-models-and-wav2vec2.md](references/encoder-models-and-wav2vec2.md)**
  — The encoder-only surface: `Encoder` pool → `SequenceEncoderReplica::forward`
  returning `EncoderForwardOutput{last_hidden_state, optional pooler_output}` (CLS-gather
  - pooler Dense), the BERT/XLM-R/Roberta loaders behind `TransformerEncoderModelSpec`,
    and the two audio encoders: `Wav2Vec2` (conv feature extractor + optional CTC
    `lm_head`) and `Wav2Vec2Bert` (Conformer sandwich + adapters). _Pool machinery is
    replica-pools._

- **[references/flash-attention-integration.md](references/flash-attention-integration.md)**
  — `WITH_FLASH_ATTN`: vendored FA2 sm80 kernels (fp16/bf16, hdim 32-256), the
  `FlashMultiHeadAttention` layer chosen at ctor time (self-attention ONLY; cross-attn
  stays composed), the structural deltas (heads-last layout, cache time-dim 1 with
  512-row preallocated chunks, RoPE inside the kernel at decode, `is_causal` instead of a
  lengths mask), and what's unsupported (ALiBi passed `nullptr`, no attention-weights
  output, CPU op throws). _CUDA-only — Metal's attention stays op-composed._

- **[references/tensor-parallel.md](references/tensor-parallel.md)**
  — `WITH_TENSOR_PARALLEL`: one MPI process per rank (`ScopedMPISetter` MPI*Init/NCCL
  comm), load-time **name-classified** weight sharding (`classify_variable`:
  column-parallel QKV/FFN-in; row-parallel output projections — NO spec markers), the
  per-sublayer `ops::ReduceAll(SUM)` allreduce points, and the don't-break list (rename a
  spec weight → silent missharding). \_CUDA-only — throws off-CUDA.*

- **[references/transformers-converter-loaders.md](references/transformers-converter-loaders.md)**
  — Inside the HF converter (`converters/transformers.py`): the `@register_loader`
  registry keyed by HF _config class name_ (42 singleton loaders), the `ModelLoader` hook
  order (`get_model_spec` → `set_config` → vocab), the loader-exists ⇔ arch-supported
  version coupling (the stale-install Qwen2 trap), and `Qwen2Loader` walked end-to-end
  (GQA normalize, RoPE params, QKV fuse). _Read before adding/extending an HF loader._

- **[references/cli-clients-and-perf-gating.md](references/cli-clients-and-perf-gating.md)**
  — `ct2-translator` (the only CLI; `cli/translator.cc`): the full flag surface
  (--compute*type overrides have no Metal case; --device accepts "metal" despite the help
  text), the streaming loop (`TextLineReader` → `consume_batches` 16× read-ahead, ordered
  futures drain), and the worked --log_throughput gating recipe (fixed input, ≥3 runs).
  \_Flag mechanics live in profiling-infrastructure.md; this is the operator card.*

- **[references/python-high-level-extensions.md](references/python-high-level-extensions.md)**
  — `extensions.py`, the complete "is it C++ or Python?" card: the seven monkey-patched
  methods (`generate_tokens`, `async_generate_tokens`, `*_iterable`), the
  callback→queue.Queue→generator bridge (forced greedy + asynchronous, daemon
  exception-drain thread), and `_process_iterable`'s 16× prefetch mirroring the C++ file
  loop. _Read before touching token streaming or batch iterables._

- **[references/model-reader-abstraction.md](references/model-reader-abstraction.md)**
  — `ModelReader`: filename→istream contract (`get_file` nullptr-on-miss /
  `get_required_file` throws), `ModelFileReader` vs `ModelMemoryReader` (zero-copy
  imemstream — the embed-in-app path, Python `files=` arg), and the exact request
  sequence: model.bin (only required file) → config.json → `initialize()` vocab/vmap
  pulls. _Short card; contents of those files are model-binary-format.md._

- **[references/downstream-validation-harness.md](references/downstream-validation-harness.md)**
  — The OTHER oracle: `scripts/validate-downstream.sh` + `tests/downstream/` — build →
  `cmake --install` to a pinned prefix → wheel via `CTRANSLATE2_ROOT` → uv-pip
  force-reinstall (+rpath fix) into 4 consumer venvs (whisperX/faster-whisper/Qwen/NLLB)
  → diff vs fp16-on-Metal goldens (WER/agreement/char-sim, quant-error tolerances).
  _The loose end-to-end gate — the 2026-06-11 int8 run (4/4) and the conv-guard catch._

## Conventions for this skill

- **Add references freely — growing this corpus is the point, not a chore.** When you
  trace out how a subsystem actually works and it isn't already covered here, write it up
  as a new `references/<topic>.md` rather than leaving the finding buried in a
  conversation. A new sibling doc is cheap and compounds; a lost finding gets re-derived.
- Each reference cites the source files it was built from (top of file) with real
  file:line references, and ends with a brief `### Relevance to the Metal backend`
  bridge to the `apple-silicon` skill where the two intersect.
- Keep SKILL.md lean: one-line pointers only. Detail lives in `references/`.
- **Line numbers drift.** These cite a snapshot; re-grep the symbol (not the line) before
  acting on a citation. Prefer quoting a function/macro name the reader can find.
- To add a reference: read the actual code, cite file:line, stay device-agnostic (push
  backend specifics to `apple-silicon`), add a one-line pointer above.
- **Every reference carries YAML frontmatter and no body `# H1`.** The first line is
  `---`, then a quoted `title` and a one-paragraph `summary`, then `---`, then the
  source-cited notes. The title lives in frontmatter only (no `# H1` atop the body). The
  `summary` is what the semantic-ID search embeds, so make it one dense, discriminating
  paragraph — name the specific ops, layers, files, and mechanisms the doc is really
  about, not generic filler.
- **After writing a reference, stamp its semantic ID:**
  `python3 .claude/scripts/stamp_semantic_ids.py .claude/skills/ct2-internals/references/<topic>.md`
  (run with no path to re-mint the whole corpus, `--health` to check bit health). This
  fills in the `semantic_id` frontmatter field that powers "find related".
- **Before trusting any `file:line` here, run `bash scripts/audit-citations.sh`** (`-q`
  for problems-only). It flags missing files, out-of-range lines, and ambiguous basenames;
  it CANNOT see content drift (a line that moved a few rows), so it prints each cited line
  for a fast eyeball. A "verified on DATE" note is worthless once the file is touched again —
  only a fresh run counts. (`transformer.cc`'s citations silently drifted ~8 lines this way.)
