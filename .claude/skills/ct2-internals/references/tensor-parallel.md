---
title: "Tensor parallelism (WITH_TENSOR_PARALLEL: NCCL + MPI weight sharding)"
summary: >-
  The WITH_TENSOR_PARALLEL multi-GPU mode (NCCL + MPI, CUDA-only, inert
  otherwise): one process per rank under mpirun, with ScopedMPISetter running
  MPI_Init and lazy ncclCommInitRank, and getNRanks/getCurRank returning 1/0
  single-GPU. Explains Megatron-style weight sharding at load via
  classify_variable and split_variables: column-parallel splits output dim 0
  (fused QKV attention linear_0, FFN linear_0/linear_0_noact, interleave-aware
  with an MQA variant) and row-parallel splits input dim 1 (output
  projections, producing rank-partial sums). The two collectives are
  ops::ReduceAll(SUM) after each row-parallel projection and ops::GatherAll in
  the lm-head Dense. Lists the contributor invariants (name-based
  classification, reduce points, getNRanks-divided head geometry) and notes it
  is entirely dead code on Metal, where parallelism means replicas, not
  sharding.
---

CT2-architecture reference, survey level: the multi-GPU tensor-parallel mode — what
exists, where it lives, and what a change to Dense/attention/FFN must not break. CUDA-only
feature; everything here is inert unless built with `WITH_TENSOR_PARALLEL` _and_ launched
under MPI with `tensor_parallel=true`.

Source: `CMakeLists.txt`, `include/ctranslate2/devices.h`, `src/devices.cc`,
`src/models/model.cc`, `src/layers/common.cc`, `src/layers/attention_layer.cc`,
`src/layers/transformer.cc`, `src/ops/nccl_ops*`. Line numbers verified by read on
2026-06-11 — re-grep symbols before acting.

## Build & process model

`WITH_TENSOR_PARALLEL` (`CMakeLists.txt:25`, OFF) links NCCL + MPI and defines
`CT2_WITH_TENSOR_PARALLEL` (`:542`); incompatible with HIP (`:717-718`). One **process per
rank** (mpirun), each rank owning one GPU. `ScopedMPISetter` (`devices.h:60-79`) is the
lifecycle object: first construction runs `MPI_Init` (`devices.cc:193-196`) and registers
`MPI_Finalize` via `atexit` (`:214`, `:267`); `getNcclComm()` lazily does
`ncclCommInitRank` with an id broadcast over MPI (`:240-251`). Static accessors
`getNRanks()`/`getCurRank()` are queried _throughout the layers_ — they return 1/0 in a
non-MPI run, which is how the same binary stays correct single-GPU.

Hard constraints at load (`model.cc:832-837`): device must be CUDA (throws otherwise);
`device_index` is overridden by the MPI local rank (`model.cc:574-579`).

## Weight sharding at load (`Model::load`, `model.cc:660-742`)

Every variable is classified by name — `classify_variable` (`model.cc:471`, enum
`VARIABLE_TYPE` `:21-41`) — and split with `split_variables`, each rank keeping its slice
(`:738-739`). The two parallel styles, by Megatron convention:

- **Column-parallel** (split the _output_ dim 0): attention `linear_0` (fused QKV) and FFN
  `linear_0`/`linear_0_noact` weights, scales, and biases. The QKV case is
  interleave-aware: it splits Q and K and V each into `world_size` parts and re-concats
  one (Qᵢ,Kᵢ,Vᵢ) bundle per rank (`:680-710`), with an MQA variant driven by the
  `multi_query_attention` config flag (`:624-633`) and a 2-way variant for enc-dec
  cross-attention `linear_1` K/V (`:712-729`).
- **Row-parallel** (split the _input_ dim 1): the sublayer output projections —
  `SELF_ATTN_LINEAR_1`, `ATTN_LINEAR_2`, `FFN_LINEAR_1` weights (`:669-676`).
  Their outputs are rank-partial sums.

Everything else (`OTHERS` — embeddings, norms, projection) is replicated... except the
`default:` branch splits any _other_ classified scale on dim 0 (`:731-736`).

## Runtime: divided heads + two collective ops

Layer ctors divide their geometry by rank count: `_num_heads`, `_d_model`, and
`_num_heads_kv` are all `SAFE_DIVIDE`d (`attention_layer.cc:120-138`); mask building in
the decoder does the same (`layers/transformer.cc:719-721`). The collectives are ordinary
ops wrapping NCCL (`include/ctranslate2/ops/nccl_ops.h:7,26`; `ncclAllReduce` /
`ncclAllGather` on the current stream, `nccl_ops_gpu.cu:60,77`; CPU versions throw):

- **`ops::ReduceAll(SUM)` after each row-parallel projection** — attention output
  (`attention.cc:603-608`, flash `layers/flash_attention.cc:129-134`) and FFN output
  (`layers/transformer.cc:41-47`). These are THE allreduce points: one per sublayer.
- **`ops::GatherAll` in `Dense::operator()` for the lm head** — `_is_layer_out` Denses
  (`projection`, marked via the ctor's 4th arg, `common.h:131`) behave specially
  (`common.cc:352-356`): only rank 0 applies bias/residual, and the quantized path
  allgathers + transposes + re-slices the _input_ so each rank quantizes against the full
  activation before using its weight slice (`common.cc:366-391`).

`Model::tensor_parallel()` (`models/model.h`) is the runtime bit layers consult
(`FeedForwardNetwork` ctor `layers/transformer.cc:18`, `AttentionLayer` ctor
`attention_layer.cc:120`).

## What a contributor must not break

- **Name-based classification**: `classify_variable` matches scoped variable names
  (`linear_0`/`linear_1`/`linear_2` under `*attention*`/`ffn` scopes) — renaming spec
  weights or fusing layers differently silently changes sharding.
- **The reduce points**: any new code path around an attention/FFN output projection must
  keep the `ReduceAll` after the row-parallel GEMM (and the rank-0-only bias rule), or
  multi-GPU outputs silently diverge.
- **`getNRanks()`-divided geometry**: head-count math added to attention must use the
  divided `_num_heads`, not the spec value.
- Not vestigial: maintained surface (`docs/parallel.md`, `tensor_parallel` args across
  the Python pools, `ScopedMPISetter` reads even in `Dense`'s hot path — a no-op
  `getNRanks()==1` check single-GPU).

---

### Relevance to the Metal backend

- **Non-Metal by definition**: load throws for `tensor_parallel` off-CUDA
  (`model.cc:832-834`), NCCL/MPI never initialize, and `getNRanks()` returns 1 — every
  `_tensor_parallel` branch in the layers is dead code on Metal.
- The one Metal-relevant touchpoint: layers Metal cares about (`Dense`,
  `FeedForwardNetwork`, attention) carry TP branches inline — when reading or editing
  those `operator()`s, the `if (_tensor_parallel)` / `affected_by_tp` blocks can be
  ignored, but must be left intact.
- Apple-Silicon multi-GPU doesn't exist (one SoC GPU), so there is no plausible Metal
  port of this machinery — parallelism on Metal means replicas, not tensor sharding.
