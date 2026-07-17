---
title: "Norm Ops (LayerNorm & RMSNorm)"
summary: >-
  The two normalization ops, ops::LayerNorm and ops::RMSNorm, at the
  numerics/contract level. LayerNorm (axis, epsilon) has affine/no-affine/in-
  place forms, an outer/axis/inner decomposition, a last-axis fast path
  computing var = max(sum_sq/n - mean^2, 0) clamped with epsilon inside the
  sqrt, and a general multi-axis form. RMSNorm is gamma-only, always last
  axis, inv_rms = 1/sqrt(mean(x^2)+epsilon) with epsilon INSIDE the sqrt (the
  parity-bug hotspot), and a use_residual flag applying (1+gamma) for Gemma at
  runtime rather than baking it into stored weights. Explains that
  layers::LayerNorm picks RMSNorm vs LayerNorm by whether a beta variable
  exists, sets epsilon defaults (1e-5 with beta, 1e-6 without), and exposes a
  fused add_norm. Notes the Metal-only fused
  ct2_add_rms_norm/ct2_add_layer_norm kernels measured 1.2-1.9x faster.
---

The two normalization ops: constructor params, gamma/beta vs gamma-only, the general
multi-axis LayerNorm form, RMSNorm's formula (epsilon is INSIDE the sqrt — parity bugs live
here), and the Gemma-style (1+gamma) flag. Norm **placement** in transformer blocks is
`norm-placement-in-transformers.md`; this file is the ops' numerics and contracts.

**Sources (read these, all citations below are from real lines):**

- `include/ctranslate2/ops/layer_norm.h`, `src/ops/layer_norm.cc`
- `include/ctranslate2/ops/rms_norm.h`, `src/ops/rms_norm.cc`, `src/ops/rms_norm_cpu.cc`
- `src/cpu/kernels.cc` — the formulas
- `src/layers/common.cc` — the `layers::LayerNorm` wrapper that picks between the two ops
- `python/ctranslate2/converters/transformers.py` — the Gemma flag

---

## 1. ops::LayerNorm

`LayerNorm(const dim_t axis = -1, const float epsilon = 1e-5)` (`layer_norm.h:10`). Three
public forms (`layer_norm.h:13-19`): with affine `(beta, gamma, input, output)`, without
affine `(input, output)`, and in-place `(input)`. All funnel into a private nullable-pointer
overload (`layer_norm.cc:33-76`).

The op pre-computes the **outer_size / axis_size / inner_size** decomposition
(`layer_norm.cc:40-48`: negative axis resolved against rank, then dims before/after the axis
multiplied out) and hands all three to `compute<D, T>` (`layer_norm.h:27-35`). The CPU side
has two kernels:

- **Last-axis fast path** `layer_norm` (`src/cpu/kernels.cc:462-496`): per row,
  one pass accumulates `sum` and `sum_squares`; then
  `var = max(sum_sq/n - mean^2, 0)` (clamped — the E[x²]−E[x]² form can go slightly
  negative), `rstd = 1/sqrt(var + epsilon)`, and
  `y = (x - mean) * rstd * gamma[j] + beta[j]` (`kernels.cc:487-493`). Epsilon is added to the
  **variance, inside the sqrt**.
- **General axis form** `layer_norm_axis` (`kernels.cc:498-541`): loops
  `outer × inner`, striding `k * inner_size` along the normalized axis. gamma/beta are
  optional here — without them it writes the plain standardized value (`kernels.cc:527-537`).
  This is the path for `axis != -1` or the affine-less overloads.

## 2. ops::RMSNorm

`RMSNorm(const float epsilon = 1e-6, const bool use_residual = false)` (`rms_norm.h:10`).
Gamma-only, no beta, **always last axis** (`rms_norm_cpu.cc:12-13` collapses to
`batch_size × depth`; there is no axis parameter).

Formula (`src/cpu/kernels.cc:543-573`):

```
inv_rms = 1 / sqrt(sum(x^2)/depth + epsilon)        // kernels.cc:561 — eps INSIDE the sqrt
y[j]    = x[j] * inv_rms * gamma[j]                 // default
y[j]    = x[j] * inv_rms * (1 + gamma[j])           // use_residual == true (Gemma)
```

No mean subtraction (that's the point of RMSNorm). The epsilon-inside-sqrt placement matches
HF/PyTorch `rms_norm`; an implementation that adds epsilon _outside_ the sqrt (or to the rms
rather than the mean-square) diverges most on near-zero rows — verify this exact line when
porting to a new backend.

## 3. How a model selects between them — `layers::LayerNorm`

The layer wrapper (`src/layers/common.cc:449-476`) decides per weight scope:

- `_beta = get_variable_if_exists(scope + "/beta")` — **if beta exists it's LayerNorm, if
  only gamma exists it's RMSNorm** (`common.cc:468-475`). Converters signal RMSNorm simply by
  not writing a beta.
- **Epsilon** comes from the model config key `layer_norm_epsilon`; if absent the default is
  `1e-5` with beta, `1e-6` without (`common.cc:453-457`). Converters set it from the HF
  config (e.g. `config.layer_norm_epsilon = model.config.rms_norm_eps` in the Gemma loader).
- **Gemma (1+gamma)**: `_use_residual` reads the model flag `{scope}/layer_norm_use_residual`
  (`common.cc:452`) and is passed to the RMSNorm op (`common.cc:473`). The flag is set at
  convert time by the Gemma loaders —
  `spec.layer_norm_use_residual = True` in `GemmaLoader` (`transformers.py:1560`),
  `Gemma2Loader` (`:1657`), `Gemma3Loader` (`:1990`). So the +1 is applied **at runtime by
  the kernel, NOT baked into the stored gamma** — the serialized weights keep HF's
  zero-centered gamma. Any backend kernel must honor the flag or Gemma silently degrades.
- The wrapper also exposes a fused `add_norm(a, b, sum_out, normed_out)`
  (`common.cc:478-518`): residual add + norm in one call, with a generic fallback of
  `ops::Add` followed by `operator()` (`common.cc:515-517`). Callers are the Gemma2-style
  pre/post sandwich blocks (`src/layers/transformer.cc:116`, `:284`).

### Relevance to the Metal backend

- Both ops are graduated: `metal::layer_norm` handles the common last-axis + affine case
  (other axes / missing affine fall back to CPU-ref — `layer_norm.cc:50-65`), and
  `metal::rms_norm` takes `use_residual` through (`rms_norm.cc:25-38`). Kernels
  `ct2_layer_norm_*`, `ct2_rms_norm_*` in `src/metal/kernels/kernels_msl.h:338,212`.
- The fused add+norm variants are Metal-only kernels (`ct2_add_rms_norm_*`,
  `ct2_add_layer_norm_*`, `kernels_msl.h:274,406`), dispatched from
  `layers::LayerNorm::add_norm` (`common.cc:480-513`) — measured 1.2–1.9× over the unfused
  pair; everywhere else uses the unfused fallback.
- Parity contract for any new backend: epsilon inside the sqrt (both ops), variance clamped
  at 0 (LayerNorm), and the `(1 + gamma)` flag — these are exactly the spots where fp16 GPU
  output drifts from the CPU reference.
