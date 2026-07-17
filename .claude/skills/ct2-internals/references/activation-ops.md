---
title: "Activation Ops"
summary: >-
  CT2's ActivationType enum (ReLU, GELUTanh, Swish, GELU, GELUSigmoid, Tanh,
  Sigmoid — fixed integer order, reused as raw kernel selectors) and the
  concrete ops it maps to via get_activation_op, where all three GELU variants
  are one GELU op parameterized by Approximation. It gives the exact CPU
  vectorized formulas from src/cpu/kernels.cc (erf GELU, tanh-approx with the
  sqrt(2/pi) constant, sigmoid quick_gelu with 1.702, SiLU/Swish) and
  documents the three fusion application sites — the Gemm epilogue
  apply_bias_and_activation, Dequantize::dequantize_gemm_output, and the
  FeedForwardNetwork's first linear (with the SwiGLU/GeGLU gate via Mul). It
  maps HF activation names per model family (BERT/Whisper erf-GELU, Gemma2
  GELUTanh+GeGLU, llama Swish) and notes the Metal tanh-overflow NaN clamp to
  [-15,15] that fixed the Gemma2 pad-collapse bug.
---

`ActivationType`, the concrete activation ops it maps to, the exact formulas, and the three
places activations get _applied_ (gemm epilogue, dequantize gemm-output, FFN) — the plumbing
that lets a backend fuse them.

**Sources (read these, all citations below are from real lines):**

- `include/ctranslate2/ops/activation.h` — the enum + `get_activation_op`
- `src/ops/activation.cc` — the mapping to concrete ops
- `include/ctranslate2/ops/gelu.h`, `src/ops/gelu.cc` — one GELU op, three approximations
- `src/cpu/kernels.cc` — the actual math (vectorized functors)
- `src/ops/gemm.cc`, `src/ops/dequantize_cpu.cc`, `src/layers/transformer.cc` — application sites
- `python/ctranslate2/specs/common_spec.py`, `python/ctranslate2/converters/transformers.py` — model-side mapping

---

## 1. The enum and the op mapping

`ActivationType` (`include/ctranslate2/ops/activation.h:9-17`) — note the comment at line 8:
**"This enum order should remain fixed."** The integer values are serialized via the Python
spec enum and also reused as raw kernel selectors (e.g. the Metal `ct2_apply_activation` code):

```cpp
enum class ActivationType { ReLU, GELUTanh, Swish, GELU, GELUSigmoid, Tanh, Sigmoid };
//                            0       1       2      3       4          5      6
```

The Python mirror is `common_spec.Activation` (`python/ctranslate2/specs/common_spec.py:7-16`),
with a comment requiring it to match the C++ enum.

`get_activation_op(ActivationType)` (`src/ops/activation.cc:12-44`) returns a reference to a
function-local `static const` op instance. There is **no separate GELUTanh/GELUSigmoid op**:
all three GELU variants are the single `GELU` op constructed with an `Approximation`
(`activation.cc:18-29`, `include/ctranslate2/ops/gelu.h:10-16` — `None`, `Tanh`, `Sigmoid`).
The op's `compute` switches to `primitives<D>::gelu` / `gelu_tanh` / `gelu_sigmoid`
(`gelu.h:22-34`).

## 2. The formulas (CPU vectorized kernels, `src/cpu/kernels.cc`)

All are unary float functors used by `vectorized_unary_transform`:

| Variant           | Functor                                    | Formula                                                                           |
| ----------------- | ------------------------------------------ | --------------------------------------------------------------------------------- |
| ReLU              | `relu_func` (`kernels.cc:147-152`)         | `max(x, 0)`                                                                       |
| GELU (exact, erf) | `gelu_func` (`kernels.cc:154-163`)         | `0.5 * x * (1 + erf(x / sqrt(2)))` — the `0.7071067811865475` constant is 1/√2    |
| GELUTanh          | `gelu_tanh_func` (`kernels.cc:165-177`)    | `0.5 * x * (1 + tanh(0.7978845608 * (x + 0.044715 * x^3)))` — 0.79788… is √(2/π)  |
| GELUSigmoid       | `gelu_sigmoid_func` (`kernels.cc:179-185`) | `x / (1 + exp(-1.702 * x))` = `x * sigmoid(1.702 x)` (HF "quick_gelu")            |
| Swish / SiLU      | `swish_func` (`kernels.cc:194-199`)        | `x / (1 + exp(-x))` = `x * sigmoid(x)`                                            |
| Sigmoid           | `sigmoid_func` (`kernels.cc:187-192`)      | `1 / (1 + exp(-x))`                                                               |
| Tanh              | `tanh_func` (`kernels.cc:201-204`)         | `tanh(x)` — the op calls `primitives<D>::tanh` directly (`src/ops/tanh.cc:28-29`) |

## 3. Where activations get applied (the fusion plumbing)

An activation is almost never a free-standing graph node; it rides as a `const ActivationType*`
parameter on the producing op, so backends can fuse it:

1. **GEMM epilogue** — `apply_bias_and_activation(x, bias, activation_type, residual, axis)`
   (`src/ops/gemm.cc:16-30`) is called **unconditionally** at the end of `Gemm::operator()`
   (`src/ops/gemm.cc:150`). With a bias it builds a `BiasAdd` carrying the activation (one fused
   bias+residual+activation pass); without one it applies `Add` (residual) then
   `get_activation_op(...)` in place.
2. **Dequantize gemm-output form** — when the linear layer is quantized, bias+activation move
   onto `Dequantize::dequantize_gemm_output` instead: `src/ops/dequantize_cpu.cc:66,83,98`.
   The fast `!transpose_a && transpose_b` path fuses scale\*dequant+bias+activation per row in
   one kernel (`src/cpu/kernels.cc:703-709` selects `gelu_func`/`gelu_tanh_func`/… inside
   `dequantize_gemm_output_row`). See `quantization-scheme-and-ops.md`.
3. **FFN** — `FeedForwardNetwork` stores `_activation_type` and hands a pointer to its first
   linear: `_ff1(model, scope + "/linear_0", &_activation_type)`
   (`src/layers/transformer.cc:14-15`), so the activation is computed inside the `_ff1` GEMM/
   dequantize epilogue. For GLU variants (SwiGLU/GeGLU) the gate path `linear_0_noact` has no
   activation and is combined by `ops::Mul()` (`src/layers/transformer.cc:33-37`).
4. **Conv1D** also applies its activation through the same helper
   (`src/ops/conv1d_cpu.cc:138`).

## 4. Which models use which

Converter mapping from HF activation names: `_SUPPORTED_ACTIVATIONS`
(`python/ctranslate2/converters/transformers.py:30-41`): `gelu`/`gelu_python` → GELU,
`gelu_fast`/`gelu_new`/`gelu_pytorch_tanh` → GELUTanh, `quick_gelu` → GELUSigmoid,
`relu` → RELU, `silu`/`swish` → SWISH.

- **GELU (erf)**: BERT/BART/Whisper (activation read from the HF config, which says `gelu`).
- **GELUTanh**: Gemma2 — its loader defaults `hidden_activation` to `"gelu_pytorch_tanh"` and
  picks GELUTanh unless the config says plain `gelu` (`transformers.py:1594` +
  ~20 lines below; same pattern in Gemma3 around `transformers.py:1915-1923`), combined with
  `ffn_glu=True` (GeGLU).
- **Swish/SiLU**: the llama family — `LlamaLoader` hardcodes `Activation.SWISH` with
  `ffn_glu=True` (`transformers.py:1739` area); Mistral/Qwen-style loaders do the same.
- **Tanh**: the BERT classification pooler (`pooling_activation=common_spec.Activation.Tanh`,
  `transformers.py:3310`, default in `specs/transformer_spec.py:778`).
- **GELUSigmoid**: HF `quick_gelu` and Marian's `"gelu"` (`converters/marian.py:14`).
- **Sigmoid**: present in the enum for completeness/pooling options; no transformers-loader
  maps to it in this tree.

### Relevance to the Metal backend

- All seven variants run on the GPU: each activation op (`relu.cc`, `gelu.cc`, `swish.cc`,
  `tanh.cc`, `sigmoid.cc`) routes `Device::METAL` fp32/fp16 to `metal::activation(x, y, size,
act)` before generic dispatch — `src/ops/gelu.cc:22-35` shows the int-code mapping
  (Tanh-approx → 1, Sigmoid-approx → 4, exact → 3), matching the `ActivationType` values.
- The fused forms are also on GPU: `metal::bias_add` takes the activation code
  (`src/ops/bias_add.cc:25-43`) and the int8 path fuses it into `ct2_dequant_gemm_out_*`
  (`src/metal/kernels/kernels_msl.h:720,732`).
- Numerics gotcha: Metal `tanh()` overflows to NaN for large |x|; the MSL kernels clamp the
  argument to [-15, 15] via `ct2_tanh_safe` (`kernels_msl.h:509-526`). This was the Gemma2
  `<pad>`-collapse bug — GELUTanh's cubic makes the argument huge in deep layers.
- Kernel-side details live in the apple-silicon skill (`math-functions-and-numeric-parity.md`).
