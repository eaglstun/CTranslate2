---
title: "The Dense Layer & Quantized Linear Orchestration"
summary: >-
  How layers::Dense orchestrates the choice between a plain GEMM and the
  quantize-to-gemm-to-dequantize pipeline entirely from loaded model state,
  with zero device- or backend-specific code. The constructor resolves members
  by scope-name lookup (_weight, _bias, _qscale from weight_scale, _qzero for
  AWQ, _u8_shift_compensation only on MKL/DNNL CPU) and sets the single
  selector _quantized_gemm from a dtype test (INT16 or INT8 weight).
  operator() is a three-way branch: the quantized path runs _quantize_op then
  int32 _gemm_op then a fused _dequantize_op epilogue carrying bias and
  activation, the _qzero+_qscale path is AWQ (asymmetric packed int4), and the
  else path is one Gemm. The doc's proof of the layering is that the entire
  int8-Metal project changed this file zero lines — all Metal int8 work landed
  in the quantize/dequantize/gemm op routing and src/metal/.
---

How `layers::Dense` decides between plain GEMM and the quantize→gemm→dequantize pipeline,
entirely from model state, with zero device- or backend-specific code.

**Sources (read these, all citations below are from real lines):**

- `include/ctranslate2/layers/common.h` (the `Dense` class, `:125-155`)
- `src/layers/common.cc` (`Dense` ctor + `operator()`, `:270-446`)
- `src/cpu/backend.cc` (`prefer_u8s8s32_gemm`), plus the op references this builds on:
  `quantization-scheme-and-ops.md`, `gemm-op-and-dtype-dispatch.md`

---

## 1. Member state — everything is decided at construction, from model variables

The real members (`include/ctranslate2/layers/common.h:136-154`): `_packed_weight`,
`_weight`, `_bias`, `_qscale`, `_qzero`, `_u8_shift_compensation`, the four `_partial_*`
StorageViews, `_output_type`, `_quant_method`, `_quantized_gemm`, and the three
pre-constructed ops `_gemm_op` / `_quantize_op` / `_dequantize_op`.

The ctor (`src/layers/common.cc:270-307`) resolves them all by **looking up model
variables by scope name**:

- `_weight` ← `{scope}/weight_packed` if present (pre-packed CPU weights), else
  `{scope}/weight` (`get_linear_weight`, `common.cc:258-268`).
- `_bias` ← `{scope}/bias` (optional, `:276`).
- `_qscale` ← `{scope}/weight_scale` (optional, `:277`) — the static per-output-row weight
  scales written by the converter (see `quantization-scheme-and-ops.md` §1).
- `_qzero` ← `{scope}/weight_zero` (optional, `:278`) — present only for AWQ models.
- `_u8_shift_compensation` ← `{scope}/weight_compensation`, loaded **only** when
  `weight.device() == CPU && weight.dtype() == INT8 && cpu::prefer_u8s8s32_gemm()`
  (`common.cc:279-283`) — i.e. only for the MKL/DNNL u8s8s32 CPU backends
  (`src/cpu/backend.cc:96-99`); on every signed-int8 backend it stays null.
- **`_quantized_gemm = (_weight.dtype() == INT16 || INT8)`** (`common.cc:290`) — the
  single bit that selects the pipeline. It's a _dtype_ test on the loaded weight; nothing
  about devices.

The three ops are configured once, consistently with that bit:

- `_gemm_op` (`common.cc:291-297`): `alpha=1, beta=0, trans_a=false, trans_b=true`
  (weights are stored `[output_dim, input_dim]`), and — key detail — the activation is
  passed **only when not quantized**: `_quantized_gemm ? nullptr : activation_type`.
- `_quantize_op` (`common.cc:298-302`): int16 scale type from
  `model.use_global_int16_scale()`; `shift_to_uint8 = bool(_u8_shift_compensation)` (the
  u8 shift exists iff the compensation term does); `round_before_cast` from
  `model.round_before_cast_in_quantization()`.
- `_dequantize_op` (`common.cc:303`): gets the activation type — in the quantized
  pipeline the activation (and bias) are fused into the dequantize epilogue, since an
  int32 accumulator can't take a float bias.

## 2. `operator()` — the decision tree

`Dense::operator()` (`src/layers/common.cc:343-446`):

1. **Partial-weight overrides** (`:345-350`): each of weight/bias/qscale/compensation is
   swapped for its `_partial_*` counterpart when non-empty. The partials are filled by
   `select_weights` (`common.cc:317-341`), which `Gather`s a row subset of the weight,
   bias, compensation, and (non-scalar) qscale — used for restricted target vocabularies.
   Because the qscale is per-output-row, selecting rows of the weight just selects the
   same rows of the scale (`:333-334`) — the quantization scheme survives slicing.
2. Tensor-parallel bookkeeping (`:352-356`, plus the `affected_by_tp` gather/slide block
   `:364-391`) — orthogonal to quantization, skipped here.
3. **The three-way branch**:

```cpp
// src/layers/common.cc:357-405 (condensed)
if (_quantized_gemm) {
  StorageView qinput(_weight.dtype(), device);        // int8 activations
  StorageView qinput_scale(_qscale->dtype(), device); // per-row dynamic scales
  StorageView qoutput(DataType::INT32, device);       // int32 accumulator
  _quantize_op(input, qinput, qinput_scale);                       // :393
  _gemm_op(qinput, *weight, qoutput, compensation);                // :396
  _dequantize_op(qoutput, qinput_scale, *qscale,
                 /*trans_a=*/false, /*trans_b=*/true,
                 output, bias);                                    // :397-403
  if (residual) ops::Add()(*residual, output, output);             // :404-405
} else if (_qzero && _qscale) {
  // AWQ (asymmetric, packed int4) — a DIFFERENT scheme: GemmAwq / GemvAwq /
  // DequantizeAwq, switched on model.quant_method()                  :406-442
} else {
  _gemm_op(input, *weight, output, nullptr, bias, residual);       // :444
}
```

- **Quantized path** (`:357-405`): exactly the pipeline diagram in
  `quantization-scheme-and-ops.md` §4. Bias + activation ride the `_dequantize_op` call
  (its gemm-output overload); `compensation` is forwarded into Gemm and is null everywhere
  except the MKL/DNNL CPU case.
- **Plain path** (`:443-445`): one Gemm call; bias, activation (ctor) and residual are
  fused into Gemm's own `apply_bias_and_activation` epilogue (`src/ops/gemm.cc:150`).
- The temporaries are local `StorageView`s constructed with the right dtype/device and
  sized by the ops themselves — the caching allocator absorbs the churn (see
  `storageview.md`).

## 3. Device/dtype agnosticism — verified, not asserted

Nothing in `Dense` mentions a device beyond `input.device()` for temporaries (`:358`).
Quantize/Gemm/Dequantize each do their own device+dtype dispatch (and their own targeted
GPU routing) below this layer. The evidence that the layering works: the entire int8-Metal
project (branch `fable/int8-metal`, 8 commits from base `cf7ad783` through native int8
GPU GEMM/GEMV, downstream validation, etc.) shipped with

```text
git diff cf7ad783..HEAD -- src/layers/common.cc   # → empty (verified 2026-06-11)
```

**zero changes to this file.** All Metal int8 work landed in `src/ops/{quantize,dequantize,gemm}.cc`
routing and `src/metal/`. (The Metal code that _is_ in `common.cc` —
`LayerNorm::add_norm`, `:478-518` — is earlier op-fusion work, not quantization plumbing.)

## 4. The other wrappers in this file (one-liners)

- **`Embeddings`** shares the _simple_ dequantize plumbing: for int8/int16 embedding
  tables it Gathers the quantized rows (and, for non-scalar scales, the matching scale
  rows) and runs `ops::Dequantize()` (`common.cc:71-84`). No Quantize, no GEMM.
- **`Conv1D`** just forwards its `{scope}/weight_scale` pointer into the Conv1D op
  (`common.cc:531`, `:546-551`); the op handles dequantization internally.
- **`LayerNorm`** has no quantization plumbing.

---

### Relevance to the Metal backend

- This file is the **proof of the layering**: int8 on Metal required no `Dense` changes —
  the per-op targeted routing (`quantize.cc` / `gemm.cc` / `dequantize.cc`) slid in under
  an unchanged orchestrator (§3 above).
- `_u8_shift_compensation` is constructed null on Metal (the `device() == Device::CPU`
  test at `common.cc:279-283`), so the quantized path naturally calls the GPU GEMM with
  `compensation == nullptr` — matching the Metal route's `!a_shift_compensation` guard
  (`src/ops/gemm.cc:113`).
- The fused dequantize epilogue Dense relies on (scales+bias+activation in one call,
  `common.cc:397-403`) maps to a single MSL kernel launch
  (`metal::dequantize_gemm_output_s8`) — one of the wins that made int8 decode beat fp16;
  see the `apple-silicon` skill and `METAL_BENCHMARKS.md`.
