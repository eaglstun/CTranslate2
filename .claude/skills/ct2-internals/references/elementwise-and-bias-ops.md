---
title: "Elementwise & Bias Ops"
summary: >-
  The binary elementwise family (Add, Sub, Mul, Min, Max) and BiasAdd in CT2:
  broadcasting is scalar-or-nothing (b is a scalar or a and b are flat same-
  size arrays — no numpy broadcasting, no shape checks, so mismatches read out
  of bounds), header-resident compute, DEVICE_AND_TYPE_DISPATCH over any
  dtype, and safe aliasing/in-place used pervasively for residual adds.
  BiasAdd is the axis-broadcast and fusion carrier (add_batch_broadcast for
  axis -1, add_block_broadcast for Conv1D's axis -2, then residual and
  activation), glued behind linear layers by apply_bias_and_activation.
  Structural call sites include post-attention/post-FFN residual adds, the
  SwiGLU/GeGLU gate Mul, and scalar Muls. On Metal, Add/Mul/BiasAdd are
  graduated (fused bias+residual+activation) while Sub/Min/Max fall to the CPU
  reference; wiring one GPU Add kernel gave 27x on that op and flipped fp16
  prefill to a win.
---

The binary elementwise family (Add, Sub, Mul, Min/Max) and BiasAdd: what broadcasting CT2
actually supports (NOT numpy), in-place/aliasing patterns, and where these ops sit
structurally — residual adds, the SwiGLU gate, the unfused bias path.

**Sources (read these, all citations below are from real lines):**

- `include/ctranslate2/ops/add.h`, `sub.h`, `mul.h`, `min_max.h` — compute lives in the headers
- `src/ops/add.cc`, `sub.cc`, `mul.cc` — dispatch (+ Metal routing where present)
- `include/ctranslate2/ops/bias_add.h`, `src/ops/bias_add.cc`, `src/ops/bias_add_cpu.cc`
- `src/ops/gemm.cc` — `apply_bias_and_activation`, the glue
- Call sites: `src/layers/transformer.cc`

---

## 1. Binary ops: broadcasting is scalar-or-nothing

`Add`, `Sub`, `Mul`, `Min`, `Max` are all the same shape: a `BinaryOp` whose `compute` lives
in the header and supports exactly **two** operand patterns (`add.h:13-21`, identical in
`sub.h:14-21`, `mul.h:14-21`, `min_max.h:14-37`):

```cpp
c.resize_as(a);
if (b.is_scalar())
  primitives<D>::add(b.data<T>()[0], a.data<T>(), c.data<T>(), c.size());  // scalar + tensor
else
  primitives<D>::add(a.data<T>(), b.data<T>(), c.data<T>(), c.size());     // same-size elementwise
```

- **No numpy broadcasting.** Either `b` is a scalar (a 0-rank/size-1 StorageView — note
  scalars like `StorageView(_layer_scalar).to(dtype)` are built on the host) or `a` and `b`
  are treated as flat arrays of `c.size()` elements. There are **no shape checks** in the
  non-scalar branch — a mismatched `b` is silently read out of bounds, not an error. Shapes
  are the caller's contract.
- Vector-vs-matrix broadcast (a bias along an axis) is NOT this family's job — that is
  `BiasAdd` (§2) or `primitives<D>::add_batch_broadcast` used directly (e.g. the relative
  attention bias add at `src/layers/attention.cc:257-261`).
- Dispatch is `DEVICE_AND_TYPE_DISPATCH` (`add.cc:32`, `sub.cc:10`, `mul.cc:32`) — these work
  on any dtype incl. int32, unlike the float-only activation/norm ops.
- **Aliasing/in-place**: `c` may alias `a` or `b`; the kernels are pure elementwise loops and
  callers rely on it everywhere, e.g. `ops::Add()(context, output, output)`
  (`src/layers/transformer.cc:127`). `resize_as` on an already-correct shape is a no-op, so
  no reallocation happens.

## 2. BiasAdd — the axis-broadcast + fusion carrier

`BiasAdd(const ActivationType* activation = nullptr, dim_t axis = -1)`
(`bias_add.h:10-28`), signature `(value, bias, output, residual = nullptr)`. Semantics
(`bias_add_cpu.cc:6-34`):

- axis `-1` (default): `add_batch_broadcast` — bias of length `dim(-1)` added to every row.
- other axes: `add_block_broadcast` with the computed inner `width` (`bias_add_cpu.cc:17-28`)
  — used by Conv1D, whose bias rides axis `-2` (`src/ops/conv1d_cpu.cc:138`).
- then `residual` is added (full-shape `Add`) and the activation applied in place
  (`bias_add_cpu.cc:30-33`).

It exists as its own op (rather than Add + activation) precisely so a backend can fuse
bias+residual+activation in one pass. CPU instantiates `float` only (`bias_add_cpu.cc:43`).

**The glue**: `apply_bias_and_activation(x, bias, activation_type, residual, axis)`
(`src/ops/gemm.cc:16-30`) — with a bias it constructs `BiasAdd`; without one it falls back to
`Add` (residual) + standalone activation. Called unconditionally from the Gemm epilogue
(`src/ops/gemm.cc:150`) and from dequantize-gemm-output (`src/ops/dequantize_cpu.cc:66,98`). So the
bias path is _usually_ fused behind the linear layer; a bare `BiasAdd`/`Add` only appears
when a layer applies bias separately.

## 3. Where they appear structurally

- **Residual adds**: post-attention and post-FFN — `ops::Add()(output, input, output)`
  (`src/layers/transformer.cc:252` self-attn residual, `:277` cross-attn, `:292` FFN in the Gemma2
  sandwich, `:332-333` both adds of the parallel-residual block). The _standard_ FFN residual
  instead rides the `_ff2` epilogue as the `residual` parameter:
  `_ff2(inner, output, _layer_norm ? &input : nullptr)` (`src/layers/transformer.cc:39`) — zero extra op.
- **SwiGLU/GeGLU gate**: `ops::Mul()(linear, inner, inner)` multiplies the no-activation
  branch with the activated branch (`src/layers/transformer.cc:33-37`).
- **Scalar Muls**: embedding scale (`src/layers/transformer.cc:437`), Gemma per-layer scalar
  (`src/layers/transformer.cc:296`), logit softcapping divide/multiply pair (`src/layers/transformer.cc:862-864`).
- **Min/Max**: clamping utilities (e.g. logit softcap composition, length penalties); same
  scalar-or-elementwise contract.
- **Sub**: rare — mostly internal normalizations (e.g. log-prob shifts in decoding helpers).

### Relevance to the Metal backend

- `Add` and `Mul` are graduated for fp32/fp16, including the scalar-operand form — the scalar
  may live in host memory, so it is read **by value** and passed into the kernel rather than
  by pointer (`add.cc:14-31`, `mul.cc:14-31`; kernels `ct2_add_*`, `ct2_mul_*` at
  `kernels_msl.h:33,42,594,603`).
- `BiasAdd` is graduated as a _fused_ kernel — bias + residual + activation in one dispatch
  for the last-axis case (`bias_add.cc:25-43`, `ct2_bias_add_*` at `kernels_msl.h:542`);
  other axes fall back to CPU-ref.
- `Sub`, `Min`, `Max` have **no** Metal routing (`sub.cc:8-11` has no METAL block): they run
  the CPU reference on unified memory via `METAL_DEVICE_CASE`. Cheap and rare enough that
  nobody has graduated them.
- The fp16 Add story: because every residual connection is a standalone `ops::Add`, wiring
  one GPU kernel into the op fixed all of them at once — **27× on that op** (the fp16 CPU
  reference is software-emulated `half`, plus a flush per residual), flipping fp16 prefill
  from a loss to the headline win (bs8 1815→559 ms). Measurement and the per-op dispatch
  floor: apple-silicon skill, `dispatch-overlap-and-perf-model.md` (lever 3).
