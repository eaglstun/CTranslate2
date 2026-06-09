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

- **[references/specs-and-converters.md](references/specs-and-converters.md)**
  — The model import pipeline: external checkpoint → converter → spec (declarative
  weight/layer layout) → serialized CT2 model → C++ loader via `model_factory.cc`. The
  LayerSpec/ModelSpec tree, the converter `set_*` pattern, model-type registration, and
  the "add a new architecture" checklist. _Read when adding model support or tracing how
  weights load._

## Conventions for this skill

- Each reference cites the source files it was built from (top of file) with real
  file:line references, and ends with a brief `### Relevance to the Metal backend`
  bridge to the `apple-silicon` skill where the two intersect.
- Keep SKILL.md lean: one-line pointers only. Detail lives in `references/`.
- **Line numbers drift.** These cite a snapshot; re-grep the symbol (not the line) before
  acting on a citation. Prefer quoting a function/macro name the reader can find.
- To add a reference: read the actual code, cite file:line, stay device-agnostic (push
  backend specifics to `apple-silicon`), add a one-line pointer above.
