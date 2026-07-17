---
title: "Shape-Manipulation Ops"
summary: >-
  The data-movement op family the decode loop lives on, with semantics and one
  load-bearing call site each: Concat (re-materializes the KV cache every
  decode step), Split (un-fuses the QKV projection after one GEMM, GQA-aware
  sizes), Transpose (the {0,2,1,3} head layout flip, a materialized copy not a
  stride trick), Tile (GQA head replication in replicate_heads), Slide
  (sliding-window attention dropping the oldest timestep), Gather (embedding
  lookup and beam/batch reordering), and metadata-only Squeeze/Unsqueeze.
  Metal relevance: Concat/Split/Slide route to the generic
  ct2_strided_copy_bytes kernel keeping KV-cache append off the CPU, Gather
  has a kernel for the axis==batch_dims case, but Transpose and Tile have no
  Metal routing and run the CPU reference on unified memory.
---

The data-movement family the decode loop lives on: Concat, Split, Transpose, Tile, Slide,
Gather, Squeeze/Unsqueeze. A map of semantics + the one load-bearing call site each — not an
encyclopedia. The structural why-so-many-tiny-ops story is `attention-and-kv-cache.md`.

**Sources (read these, all citations below are from real lines):**

- `src/ops/concat.cc`, `split.cc`, `transpose.cc` (+ `include/ctranslate2/ops/transpose.h`),
  `tile.cc`, `slide.cc`, `gather.cc`
- `include/ctranslate2/ops/squeeze.h`, `unsqueeze.h` (header-only)
- Call sites: `src/layers/attention.cc`, `src/layers/decoder.cc`, `src/layers/common.cc`,
  `src/decoding.cc`

---

## Concat — `Concat(int axis)`

Variadic `(const std::vector<const StorageView*>& inputs, StorageView& output)`. Negative
axis allowed; output shape = inputs' shape with `dim(axis)` summed (`concat.cc:12-29`). All
inputs must share rank and non-axis dims (asserted).

**Load-bearing site: the KV cache.** Each decode step appends one timestep:
`ops::Concat concat_op(_cache_time_dim)` then `concat_op({&tmp, &keys_proj}, *cached_keys)`
(`src/layers/attention.cc:538-543`) — the cache is _re-materialized_ every step (the old cache
moves into a scratch view, then old+new are concatenated). Also merges decoder self K/V with
encoder memory K/V along time (`attention.cc:740-742`).

## Split — `Split(axis [, sizes], no_copy)`

Inverse of Concat. Two ctors (`split.cc:14-27`): even split across N outputs (axis dim must
divide) or explicit per-output sizes (`std::vector<dim_t>` summing to `dim(axis)`,
validated at `split.cc:49-62`). Convenience 2-way/3-way overloads (`split.cc:29-42`).
`no_copy=true` returns zero-copy **views** into the input — only legal on axis 0
(`split.cc:109-112`, views built at `split.cc:70-73`).

**Load-bearing site: un-fusing the QKV projection.** Self-attention does one fused GEMM then
`ops::Split(2, {_num_heads * _d_head, _num_heads_kv * _d_head, _num_heads_kv * _d_head})`
(GQA-aware sizes, `attention.cc:482`) or an even `ops::Split(1)(fused_proj, q, k, v)`
(`attention.cc:512`).

## Transpose — `Transpose(perm)`

Permutation copy (not a stride trick — output is materialized). Empty perm = reverse all
dims; identity perms short-circuit to `y = x` (`transpose.cc:19-40`). Rank 2/3/4 only,
dispatching to `primitives<D>::transpose_2d/3d/4d` (`include/ctranslate2/ops/transpose.h:17-31`);
rank ≤ 1 is a copy (`transpose.cc:14-17`). The 2D case is the
classic matrix transpose fast path (`transpose.h:19-21`).

**Load-bearing site:** the head layout flip `{0, 2, 1, 3}` used by both `split_heads` and
`combine_heads` — a single static op instance (`attention.cc:167`).

## Tile — `Tile(axis, num_tiles)`

Repeats the tensor along `axis`: output `dim(axis) *= num_tiles`, implemented as
`outer_size × inner_size` block copies (`tile.cc:14-37`). In-place overload moves through a
clone (`tile.cc:39-42`).

**Load-bearing site: GQA head replication.** `replicate_heads` expands KV heads to match query
heads: `expand_dims(2)`, `ops::Tile(2, repeats)(x)`, reshape (`attention.cc:288-292`) — a copy,
but one Tile, not a loop.

## Slide — `Slide(axis, index, size, no_copy)`

The windowing op: output = input with `dim(axis)` replaced by `[index, index+size)`
(`slide.cc:14-38`). `no_copy` views, axis-0 only, same restriction as Split
(`slide.cc:68-71`).

**Load-bearing site: sliding-window attention.** During decode, once the cache exceeds
`_sliding_window`, the oldest timestep is dropped: `ops::Slide slide_op(2, 1,
cached_keys->shape()[2] - 1)` (`attention.cc:545-552`); after prefill the cache is trimmed to
the last `_sliding_window` tokens (`attention.cc:584-592`). Also slices per-rank position
bias under tensor parallelism (`attention.cc:251-254`).

## Gather — `Gather(axis = 0, batch_dims = 0)`

ONNX-style: output shape = `input(indices)` shape + data dims after `axis`
(`gather.cc:14-21`). Indices are always `int32`. With `batch_dims > 0` the leading dims of
data and indices must match and gathering happens per batch (checked at `gather.cc:69-84`).
The in-place overload has a fast path when axis 0, no batch dims, and indices are strictly
increasing — it compacts rows without a clone (`gather.cc:23-45,53-57`); otherwise it moves
through a clone.

**Two load-bearing sites:**

- **Embedding lookup**: `Embeddings::operator()` is a Gather of rows from the (possibly
  quantized) embedding matrix (`src/layers/common.cc:68-85` — int8/int16 embeddings gather
  then `Dequantize`).
- **Beam/batch reordering**: beam search reorders every decoder state and the KV cache by
  indices — `gather_beam_flat` (`src/decoding.cc:13-17`) and
  `DecoderState` reordering `ops::Gather()(value, beam_indices)`
  (`src/layers/decoder.cc:35-53`); finished-hypothesis pruning gathers the live batches
  (`decoding.cc:683-700`).

## Squeeze / Unsqueeze — header-only metadata ops

No kernels, no dispatch: both just compute a new shape and `reshape`
(`include/ctranslate2/ops/squeeze.h`, `unsqueeze.h`). Squeeze validates the squeezed dims are
1 (throws otherwise); the out-of-place forms `shallow_copy` first — **no data movement
ever**. `StorageView::expand_dims` is the inline equivalent used by `replicate_heads`.

### Relevance to the Metal backend

- Concat, Split, and Slide all route Metal-resident data to one generic GPU kernel,
  `metal::strided_copy` (`ct2_strided_copy_bytes`, `kernels_msl.h:873`) — see the Metal
  blocks in `concat.cc:31-55`, `split.cc:80-104`, `slide.cc:41-63`. This is what keeps the
  per-token KV-cache append off the CPU.
- Gather has a GPU kernel for the `axis == batch_dims` case (`gather.cc:89-106`,
  `ct2_gather_bytes` at `kernels_msl.h:886`) — covers embedding lookup and beam reorder;
  other axes fall through to CPU-ref.
- Transpose and Tile have **no** Metal routing: they run the CPU reference on unified memory
  via `METAL_DEVICE_CASE` (`transpose.cc:42`, `tile.cc:35-36`). Tile only matters at GQA
  setup; Transpose (head split/combine) is a known per-step CPU cost.
- Squeeze/Unsqueeze are metadata-only and device-irrelevant.
- Per-op dispatch cost of this family in the decode loop is the apple-silicon skill's
  `dispatch-overlap-and-perf-model.md`.
