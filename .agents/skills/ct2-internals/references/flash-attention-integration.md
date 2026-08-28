---
title: "Flash Attention 2 integration (WITH_FLASH_ATTN)"
summary: >-
  How the vendored Flash Attention 2 CUDA kernels integrate as a parallel
  FlashMultiHeadAttention layer (selected at construction alongside
  MultiHeadAttention, both deriving from AttentionLayer) rather than a patch
  to the composed attention. Build-gated by WITH_FLASH_ATTN (head dims 32-256,
  fp16/bf16, sm80, HIP-incompatible) and runtime-gated to Ampere+ and self-
  attention only; the user surface is a flash_attention=False load flag, not a
  compute_type. The layer replaces the QK^T/mask/softmax/×V sequence with one
  ops::FlashAttention call, keeps heads last via reshape (no transpose), grows
  the KV cache in preallocated 512-row chunks instead of per-step Concat,
  moves RoPE into the kernel at decode, and uses is_causal instead of a
  lengths mask. It explicitly does not support cross-attention, CPU/non-CUDA,
  fp32/int8, or ALiBi (wired but nullptr-disabled), so Metal attention stays
  on the composed path — flash is CUDA-only by construction.
semantic_id: "yh6w9g2Ps23T4a190NgrPaMko2vTyjEjWhdyGpTVYs4MWVkGZ4-N388w-h0syoQapi0kus76h45OfKUPj6Q9pQ"
---

CT2-architecture reference: how the vendored Flash Attention 2 CUDA kernels integrate —
a parallel attention _layer_, not a patch to `MultiHeadAttention`. The composed attention
it replaces is `attention-and-kv-cache.md`; this file is the flash-specific delta.

Source: `CMakeLists.txt`, `src/layers/flash_attention.cc`,
`include/ctranslate2/layers/flash_attention.h`, `src/ops/flash_attention*.{cc,cu}`,
`src/models/model.cc`. Line numbers verified by read on 2026-06-11 — re-grep symbols
before acting.

## Build & gating

- `WITH_FLASH_ATTN` (`CMakeLists.txt:26`, default OFF) defines `CT2_WITH_FLASH_ATTN`
  (`:638`) and compiles the vendored FA2 forward kernels in
  `src/ops/flash-attention/flash_fwd[_split]_hdim{32..256}_{fp16,bf16}_sm80.cu`
  (`:640-706`) — head dims 32–256, fp16/bf16 only, sm80 codegen. Incompatible with HIP
  (`:751-752`, FATAL_ERROR).
- Runtime gates in `models::ModelLoader::load` (`model.cc:840-855`): requires
  `dprops.major >= 8` (Ampere+) or throws; and `Model::set_compute_type` throws unless
  the effective float dtype is fp16/bf16 (`model.cc:194-195`). The layer ctor adds:
  **self-attention only** (`layers/flash_attention.cc:15`, `ERROR_CHECK`) — cross-attention in
  enc-dec models stays on the composed path.
- User surface: a plain `flash_attention=False` ctor arg on every Python pool
  (`python/cpp/generator.cc:146`, translator/whisper/encoder likewise) →
  `ReplicaPool` config (`replica_pool.h`) → `Model::_use_flash_attention`
  (`model.cc:608`). Not a compute_type — a load flag.

## Where it plugs in

`TransformerEncoderLayer`/`TransformerDecoderLayer` ctors pick the class at construction:
`use_flash_attention ? FlashMultiHeadAttention : MultiHeadAttention`
(`src/layers/transformer.cc:54-66`, decoder `:162`). Both derive from `AttentionLayer`
(`attention_layer.h`), so the layer call sites don't change.

## What the flash layer does differently (`layers/flash_attention.cc:18-137`)

It keeps the projections (`_linear[0]` fused QKV, `:45`; split `:51-64`) and the output
projection + residual (`:128`), but replaces the whole QK^T → mask → SoftMax → ×V op
sequence with **one fused `ops::FlashAttention` call** (`:113-115`). Structural deltas:

- **Layout**: heads stay last — `[batch, time, num_heads, d_head]` via pure reshape, no
  transpose (`split_heads`, `:139-155`); the KV cache time dim is **1**, not 2 (`:13`).
- **Cache growth**: instead of per-step `Concat` of one row, the cache is extended in
  preallocated chunks of `_offset_free_space = 512` rows (`layers/flash_attention.h:50`,
  `layers/flash_attention.cc:75-84`) and the kernel writes the new K/V in place at `offset` —
  amortizing the concat. Sliding-window trims use the same `Slide` pattern (`:86-93`,
  prefill `:117-125`).
- **RoPE moves into the kernel** for decode steps: at `offset > 0` the layer passes
  `cos/sin` tables + interleave flag (`:102-109`) and FA2 rotates Q/K internally; at
  prefill it pre-applies `_rotary_embeddings->apply` (`:66-69`).
- **Masking**: no `lengths` mask tensor — causality is the op's `is_causal` ctor flag
  (default true, `ops/flash_attention.h:9`) plus the sliding-window width; padding is
  expected to be removed by the `Padder` upstream.
- The op itself is the standard 4-file pattern: `ops/flash_attention.cc:28-31`
  `DEVICE_DISPATCH`es to `compute<D>`; the CPU specialization **throws**
  (`flash_attention_cpu.cc:21`), and the `.cu` without `CT2_WITH_FLASH_ATTN` throws
  "Flash attention 2 is not supported" (`flash_attention_gpu.cu:364`). GQA/MQA is
  handled natively by the kernel (no `replicate_heads` Tile; head ratios + the
  `seqlenq_ngroups_swapped` decode trick, `flash_attention_gpu.cu:242`); head size is
  padded to a multiple of 8 (`:258`).

## What it does NOT support (from the code guards)

Cross-attention (`layers/flash_attention.cc:15`); CPU and non-CUDA devices; fp32/int8 compute
(fp16/bf16 only); pre-Ampere GPUs; ROCm. **ALiBi is wired but disabled** — the layer
passes `nullptr` for the alibi argument with a literal `/*alibli*/` comment
(`layers/flash_attention.cc:115`). T5-style relative position bias and returning attention
weights for alignment follow the composed path (flash returns no per-head score matrix —
`return_normalized_attention` is accepted but there's no softmax tensor to extract).
Models needing those must load with `flash_attention=false`.

---

### Relevance to the Metal backend

- Flash Attention is **CUDA-only by construction** (sm80 kernel sources, HIP fatal,
  CPU-throws op) — Metal's attention stays the op-composed path in
  `attention-and-kv-cache.md`; there is no flash shim to route.
- The flash layer is still a useful _design reference_ for Metal attention work: the
  chunked-preallocation KV cache (512-row `_offset_free_space`) is the in-tree
  alternative to per-step `Concat` if cache-append overhead ever becomes the measured
  bottleneck (it wasn't — see the Metal perf memory).
- A model loaded with `flash_attention=true` on Metal would fail the CUDA-only check at
  load (`model.cc:840-855` leaves `supports_flash_attention=false` off-CUDA) — nothing
  Metal-specific to guard.
