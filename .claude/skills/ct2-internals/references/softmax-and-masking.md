---
title: "SoftMax & Masking"
summary: >-
  The SoftMax/LogSoftMax op in CT2: always reducing over the last dimension
  (every other dim collapsed into batch_size, no axis parameter), four
  overloads including in-place and two masked forms, and float-only generic
  dispatch (DEVICE_AND_FLOAT_DISPATCH) with fp16/bf16 only via CUDA or the
  Metal route. The lengths mask is not an additive -inf matrix but an int32
  tensor of one valid length per softmax row, where positions past the length
  are written as exactly 0 and zero-length rows skipped; lengths come from
  AttentionLayer::prepare_length_mask (causal capping plus decode step
  offset). It documents the max-subtract numerical pattern (log path never
  materializing exponentials) and the key fact that the 1/sqrt(d) query scale
  is folded into the QK^T MatMul alpha, not into SoftMax. On Metal, fp32/fp16
  route to metal::softmax with the masked contract holding bit-for-bit.
---

The SoftMax/LogSoftMax op: last-dim semantics, the optional lengths mask (how padding is
excluded), in-place forms, and where attention folds in the 1/√d scale (spoiler: into the
QK^T MatMul alpha, **not** into SoftMax).

**Sources (read these, all citations below are from real lines):**

- `include/ctranslate2/ops/softmax.h` — interface (also the 4-file-pattern exemplar in
  `dispatch-and-op-implementation.md`; dispatch mechanics are owned there, not re-explained here)
- `src/ops/softmax.cc` — checks + dispatch
- `src/ops/softmax_cpu.cc`, `src/cpu/kernels.cc` — the CPU kernel
- `src/layers/attention.cc`, `src/layers/attention_layer.cc` — the main caller
- `src/decoding.cc`, `src/models/language_model.cc` — LogSoftMax callers

---

## 1. Op semantics

`SoftMax(bool log = false)` (`include/ctranslate2/ops/softmax.h:8-23`); `LogSoftMax` is just `SoftMax(/*log=*/true)`
(`ops/softmax.h:26-29`, `softmax.cc:12-14`).

- **Always over the last dimension.** `depth = x.dim(-1)`; every other dimension is collapsed
  into `batch_size = x.size() / depth` (`softmax.cc:36`, `softmax_cpu.cc:12-13`). There is no
  axis parameter — callers arrange the reduction dim last.
- **Four overloads** (`ops/softmax.h:12-16`): in-place `(StorageView& x)` (calls itself with
  `output = x`, `softmax.cc:20-22`), out-of-place `(x, y)`, and two masked forms taking
  `lengths` by reference or nullable pointer. In-place is the common decoding form
  (`ops::LogSoftMax()(logits)`).
- **dtype**: generic dispatch is `DEVICE_AND_FLOAT_DISPATCH` (`softmax.cc:65`) — float only on
  CPU (`softmax_cpu.cc:29` instantiates just `float`), fp16/bf16 only via CUDA (or the Metal
  targeted route).

## 2. The lengths mask

The mask is **not** an additive -inf matrix. It is an `int32` tensor with **one valid-length
per softmax row**: `lengths->size()` must equal the collapsed `batch_size`
(`softmax.cc:41-48`, throws otherwise). In the CPU kernel (`src/cpu/kernels.cc:403-459`):

- the row's reduction size becomes `size = lengths[i]` (`kernels.cc:418-419`),
- positions `j >= size` are **directly written as 0** in the output (`kernels.cc:421-424`) —
  padding gets exactly zero probability, not exp(-inf),
- a row with `size == 0` is skipped entirely (all zeros, `kernels.cc:426-428`).

Where lengths come from: `AttentionLayer::prepare_length_mask`
(`src/layers/attention_layer.cc:152-174`) expands per-batch sequence lengths to a
`{batch, num_heads, num_queries}` int32 tensor (one entry per attention row;
`mask_future` makes it causal by capping each query's length, and decode adds the step offset —
`src/layers/transformer.cc:687-696`). That tensor is the `values_lengths` handed to
`ops::SoftMax()(output, values_lengths, attn)` in `dot_product_attention`
(`src/layers/attention.cc:268`).

## 3. The numerical pattern (max-subtract)

`kernels.cc:431-456`: compute `x_max = reduce_max(x, size)`, then `exp(x - x_max)`
vectorized. Plain softmax then divides by `exp_sum` (one multiply by `1/exp_sum`,
`kernels.cc:452-456`). The log path never materializes the exponentials: it map-reduces the
exp-sum and writes `y = x - x_max - log(exp_sum)` in one `add` pass (`kernels.cc:441-451`).
Rows are parallelized with `parallel_for` over `batch_size` (`kernels.cc:411`).

## 4. Who calls it, and where the 1/√d scale lives

- **Attention** (`dot_product_attention`, `attention.cc:216-268`): the query scale is folded
  into the **QK^T MatMul as alpha**, not pre-multiplied onto Q and not a SoftMax parameter:
  `const ops::MatMul keys_matmul(false, /*trans_b=*/true, queries_scale)` (`attention.cc:216`;
  same in the merged path at `attention.cc:781-784`). `queries_scale` defaults to
  `1/sqrt(d_head)` but is a loadable model attribute `{scope}/queries_scale`
  (`attention_layer.cc:131-133`) — models like Gemma override it. ALiBi biases are added
  scaled by the same factor (`attention.cc:264-265`). SoftMax itself is scale-free.
- **Decoding**: `ops::LogSoftMax()(logits)` in-place to get log-probs for sampling/scoring
  (`src/decoding.cc:545`, `:877`; beam search per-beam at `:320`).
- **Generator API**: normalized logits for callers (`src/models/language_model.cc:97`).
- `TopPMask` (nucleus sampling) embeds its own softmax-like normalization separately
  (`src/ops/topp_mask.cc`) — not this op.

### Relevance to the Metal backend

- SoftMax is a graduated Metal op: fp32 + fp16 route to `metal::softmax` before generic
  dispatch (`softmax.cc:50-63`), including the lengths mask (passed as the raw `int32*`).
  Kernels: `ct2_softmax_float` / `ct2_softmax_half` (`src/metal/kernels/kernels_msl.h:56,121`).
- The masked-row contract (zeros written for `j >= len`, zero-length rows skipped) must hold
  bit-for-bit on the GPU — parity is checked by `tests/ops_test.cc` parameterized over
  `Device::METAL` plus `tests/metal_test.cc`.
- The fp16 path matters because `DEVICE_AND_FLOAT_DISPATCH` would otherwise _reject_ fp16 on
  a non-CUDA device — the targeted route is what makes fp16 softmax legal on Metal.
- This file is the op walked through in `dispatch-and-op-implementation.md`; read that for the
  dispatch macros, and the apple-silicon `op-graduation-playbook.md` for the GPU-side recipe.
