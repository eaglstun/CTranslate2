---
title: "Attention & the KV cache in CTranslate2 (RoPE, GQA, per-step cache growth)"
summary: >-
  Explains CTranslate2's MultiHeadAttention structure in
  attention.cc/attention_layer.cc: the split_heads/combine_heads reshape-and-
  transpose helpers moving between [batch, time, depth] and [batch, num_heads,
  time, d_head] (a pure reshape at time==1 decode steps), the fused QKV
  projection carved by ops::Split whose widths encode GQA/MQA head configs,
  and replicate_heads broadcasting fewer KV heads via expand_dims + ops::Tile
  + reshape rather than a copy loop. Details RotaryEmbeddings::apply rotating
  only Q and K after projection using the sequence-position offset with
  lazily-grown sin/cos tables, and the heart of decode: the KV cache grown one
  row per token by ops::Concat on the time dim (dim 2, or dim 1 in the merged
  MQA layout), with sliding-window Slide trims for bounded caches. Its load-
  bearing point is that each decode step issues a dozen-plus tiny ops (fused-
  QKV GEMM, Split, reshapes, Tile, RoPE, cache Concat, score/value GEMMs,
  combine, output GEMM) at m=batch, which is the structural reason Metal
  decode is API-overhead-bound; also flags Concat/Split GPU graduation as
  parity-verified but e2e-neutral.
---

CT2-architecture reference: how `MultiHeadAttention` projects, reshapes, rotates, and
**grows the KV cache one step at a time**. This is the structural side of the decode loop —
the part that explains _why_ a decode step issues so many tiny ops (the thing the Metal
perf story is about). Device-agnostic: every op here routes to CPU or Metal the usual way.

Source: `src/layers/attention.cc`, `src/layers/attention_layer.cc`,
`include/ctranslate2/layers/attention.h`. **Line numbers verified by grep on 2026-06-09 —
re-grep the symbol, not the line, before acting (this file is the worst offender for drift
since `attention.cc` is 844 lines of reshaping).**

## The shapes, and the two layouts

A token activation is `[batch, time, depth]` with `depth = num_heads * d_head`. Attention
math needs it split per-head as `[batch, num_heads, time, d_head]`. Two static helpers in
`attention.cc` do nothing but move between these:

- `split_heads` (`:795`) — `[batch, time, depth]` → `[batch, num_heads, time, d_head]`.
  For `time == 1` (a decode step) it's a pure `reshape` (`:811`); for `time > 1` it
  reshapes then **transposes** (`:813-816`). Optionally un-pads and de-interleaves beam.
- `combine_heads` (`:820`) — the inverse, transpose-then-reshape (`:829-835`).

`d_head` is `head_dim` if the model gives one, else `d_model / num_heads`
(`attention_layer.cc:126`). Keep this split/combine pair in mind: most of the op count in a
decode step is these reshapes/transposes plus the GQA replication below, not the GEMMs.

## Fused QKV projection → Split

`_linear[0]` projects to a single fused tensor (`attention.cc:465`), then an `ops::Split`
carves out Q, K, V. The split widths encode the head config — this is where GQA shows up:

```cpp
// self-attention, GQA/MQA branch — attention.cc:482
const ops::Split split_op(2, {_num_heads * _d_head,        // Q: full heads
                              _num_heads_kv * _d_head,      // K: fewer heads
                              _num_heads_kv * _d_head});    // V: fewer heads
```

When `_num_heads_kv == _num_heads` (vanilla MHA) it's the simpler `:511-512` path:
`split_heads(fused, 3*_num_heads)` then `ops::Split(1)`.

## GQA / MQA — `replicate_heads` (Tile, not a copy-loop)

`_num_heads_kv` (`attention_layer.cc:135`) is the number of K/V heads; `_multi_query`
forces it to 1. Three branches, all in `process_qkv` / `operator()`:

- **MHA** (`_num_heads_kv == _num_heads`): nothing to replicate.
- **MQA** (`_num_heads_kv == 1`, `attention.cc:386`): one K/V head shared by all queries.
- **GQA** (`1 < _num_heads_kv < _num_heads`, `:398`): each K/V head serves a _group_ of
  `_num_heads / _num_heads_kv` query heads.

For MQA/GQA the K and V heads are broadcast up to `_num_heads` by `replicate_heads`
(`:288`), which is **`expand_dims` + `ops::Tile` + `reshape`** — not a hand-rolled copy.
qk_norm / v_norm (the optional per-head `_q_norm`/`_k_norm`/`_v_norm`,
`attention.cc:342-366`) are applied **before** replication, while heads are still split,
because gamma is sized `d_head`.

> Note `_merge_time_and_head_dims` / `_cache_time_dim` (`:306-310`): the multi-query +
> no-relative-bias case keeps time and head dims merged and caches on dim 1 instead of
> dim 2, to avoid extra transposes. The `if (_merge_time_and_head_dims)` branches
> (`:485`, `:519`, `:527`) exist only to thread that layout through RoPE and the cache.

## RoPE — `RotaryEmbeddings::apply`

Applied to Q and K **after** projection/split, **before** the cache concat
(`attention.cc:518-525`):

```cpp
_rotary_embeddings->apply(queries_proj, offset);
_rotary_embeddings->apply(keys_proj, offset);
```

`offset` is the current sequence position — that's how a decode step rotates the single new
token to its absolute position without re-rotating the cache. `apply`
(`attention_layer.cc:213`) lazily grows its `_sin`/`_cos` tables when
`offset + max_time` exceeds what's cached (`:219`), slices out the `[offset, offset+time)`
rows, and calls the `_rotary_op`. Scaling variants (Llama3, Su/longrope, linear) are set up
in `make_rotary_embeddings` (`attention_layer.cc:68`) and `initialize` (`:252`). Only K and
Q are rotated — V never is.

## The KV cache — grown one step per token (`Concat` on the time dim)

This is the heart of the decode loop. `cached_keys`/`cached_values` are passed in by the
decoder and **owned across steps**. Per step (`attention.cc:533-554`):

```cpp
if (cached_keys->empty()) {                 // first step: cache IS the projection
  *cached_keys   = std::move(keys_proj);
  *cached_values = std::move(values_proj);
} else {                                    // every later step: grow by concatenation
  const ops::Concat concat_op(_cache_time_dim);   // dim 2 (or 1 if merged layout)
  tmp = std::move(*cached_keys);
  concat_op({&tmp, &keys_proj}, *cached_keys);     // [.., T, ..] ++ [.., 1, ..] → [.., T+1, ..]
  concat_op({&tmp, &values_proj}, *cached_values);
}
```

After the concat, `keys_proj`/`values_proj` `shallow_copy` the full cache
(`:557-560`) and feed `dot_product_attention` (`:562`). **The implication for perf:** every
single decode step does a `Concat` (cache grows by one row) plus the split/replicate/RoPE
ops above — many tiny ops, each paying the per-op floor. That's the structural reason
decode is overhead-bound; see the Metal bridge below.

### Sliding-window attention

If `_sliding_window > 0` (`attention_layer.cc:140`) the cache is bounded. Two `ops::Slide`
trims:

- **Generation** (`attention.cc:545-552`): after the concat, if the cache exceeds the
  window, slide off the oldest row (`Slide(2, 1, len-1)`).
- **Prefill** (`:584-592`): after attention, keep only the last `sliding_window` tokens.

`prefilling` is `(_sliding_window > 0 && values_lengths)` (`:469`). This is the path the
Metal-Gemma2 investigation flagged as the prime suspect (see the Metal-backend memory).

## Where the pieces are configured

- Head counts, `d_head`, scale, RoPE, sliding window: `attention_layer.cc:126-140`
  (the `AttentionLayer` ctor). Q/K/V norms and relative-position bias:
  `attention.cc:302-313`.
- `dot_product_attention` (`:170`+) does the score GEMM, optional relative/alibi bias,
  `SoftMax` (`:268`), and the value GEMM (`:273`). Relative-position attention (T5-style
  buckets) and ALiBi live here too but are orthogonal to the RoPE/GQA decode path.

---

### Relevance to the Metal backend

- **This file explains the op count that the Metal perf story is about.** Each decode step
  issues: fused-QKV GEMM, Split, split*heads (reshape), GQA `replicate_heads` (Tile), RoPE,
  the KV-cache **`Concat`**, score GEMM, SoftMax, value GEMM, combine_heads, output GEMM —
  a dozen-plus \_tiny* ops at `m = batch`. That's why Metal decode is API-overhead-bound and
  why the per-op floor dominates. The reasoning is in the **`apple-silicon`** skill,
  `dispatch-overlap-and-perf-model.md`.
- The per-step **`Concat`/`Split`** were graduated to Metal GPU kernels to keep the cache
  on-GPU — parity-verified but measured _neutral_ on e2e decode (it disproved the
  "per-step flush dominates" hypothesis). Structure here; that perf conclusion there.
- RoPE has a Metal kernel with its own parity test (`tests/metal_test.cc`, the rotary
  case); the fp16 numeric tolerance for it is a `math-functions-and-numeric-parity.md`
  concern in the `apple-silicon` skill, not a structural one.
