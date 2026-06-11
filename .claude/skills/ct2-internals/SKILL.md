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

## Conventions for this skill

- Each reference cites the source files it was built from (top of file) with real
  file:line references, and ends with a brief `### Relevance to the Metal backend`
  bridge to the `apple-silicon` skill where the two intersect.
- Keep SKILL.md lean: one-line pointers only. Detail lives in `references/`.
- **Line numbers drift.** These cite a snapshot; re-grep the symbol (not the line) before
  acting on a citation. Prefer quoting a function/macro name the reader can find.
- To add a reference: read the actual code, cite file:line, stay device-agnostic (push
  backend specifics to `apple-silicon`), add a one-line pointer above.
- **Before trusting any `file:line` here, run `bash scripts/audit-citations.sh`** (`-q`
  for problems-only). It flags missing files, out-of-range lines, and ambiguous basenames;
  it CANNOT see content drift (a line that moved a few rows), so it prints each cited line
  for a fast eyeball. A "verified on DATE" note is worthless once the file is touched again —
  only a fresh run counts. (`transformer.cc`'s citations silently drifted ~8 lines this way.)
