---
title: "Batching & length sorting (rebatch_input, batch_type, Padder, order restoration)"
summary: >-
  How CT2 turns user inputs into padding-efficient batches and restores
  original order, driven by rebatch_input in src/batch_reader.cc, which sorts
  examples longest-to-shortest so similar lengths batch together and finished
  rows drop off the array end during decode. Defines Example/Batch/BatchType
  (Examples vs Tokens counting), the variable- vs fixed-increment fill
  strategies, and the readers (TextLineReader, VectorReader,
  ParallelBatchReader). Order restoration is free because
  ReplicaPool::post_examples pre-creates one promise per example in input
  order and moves them into batches by example_index. Details the Padder,
  which removes padding around position-independent FFN/projection regions via
  ops::Gather index maps, gated by allow_padding_removal (CPU always, GPU only
  when compute_type is not FLOAT16), so Metal fp16 runs keep padded shapes.
---

CT2-architecture reference: how user inputs become efficiently-shaped batches and how
results come back in the original order. The driver is padding waste: a batch runs at the
length of its _longest_ example, so CT2 (a) groups similar-length examples, and (b) when
it can't, removes the padding around the GEMM-heavy regions.

Source: `include/ctranslate2/batch_reader.h`, `src/batch_reader.cc`,
`include/ctranslate2/replica_pool.h`, `src/translator.cc`,
`include/ctranslate2/padder.h`, `src/padder.cc`, `src/layers/transformer.cc`.
Lines verified by read on 2026-06-11.

## The vocabulary: Example, Batch, BatchType

- `Example` (`batch_reader.h:21-49`) — one input as N parallel token streams (source [+
  target]); `length()` is stream 0's length.
- `BatchType` (`:12-15`) — `Examples` or `Tokens`: is `max_batch_size` counted in
  sentences or in tokens? (`str_to_batch_type`, `batch_reader.cc:10`.)
- `Batch` (`batch_reader.h:135-148`) — the examples **plus `example_index`**, each
  example's position in the original input. This vector is the order-restoration key.

## Filling a batch: `BatchReader::get_next` (`batch_reader.cc:82-103`)

`get_batch_size_increment` (`:18-26`): an example costs 1 (`Examples`) or its length
(`Tokens`). Two fill strategies, switched by `consider_padding`:

- `fill_batch_with_variable_increment` (`:60-80`) — sum of raw increments ≤
  `max_batch_size`. Used when reading ahead from a stream (no padding yet).
- `fill_batch_with_fixed_increment` (`:38-58`) — cost is
  `(count+1) * max_length_so_far`, i.e. the **padded** token count. Used by
  `rebatch_input` (`:210`, "the batch size increment per example is always fixed because
  padding is required").

Readers: `TextLineReader` (stream + tokenizer, `batch_reader.h:82-104`), `VectorReader`,
and `ParallelBatchReader` (zips source/target streams, `:123-132`).

## The sort: `rebatch_input` (`batch_reader.cc:174-227`)

The central reordering function (free function `ctranslate2::rebatch_input` — the real
name; there is no `ctranslate2::rebatch`). It sorts examples **longest → shortest**
(`:194-199`) for two documented reasons (comment `:187-193`):

1. similar-length examples batch together → less padding waste;
2. **shorter sentences finish decoding first**, and the decode loop removes finished rows
   (see `decoding-loop-and-beam-search.md`) — sorting short-last means removals happen at
   the _end_ of the arrays, which is cheaper for the in-place gathers.

It then slices the sorted stream into `Batch`es via the fixed-increment fill, recording
`example_index` per batch (`:216-221`).

## The flow: ReplicaPool → translate_batch

`Translator::translate_batch_async` (`src/translator.cc:16-21`) →
`ReplicaPool::post_examples` (`replica_pool.h:173-189`):

1. one `std::promise`/`std::future` per example, **created in original input order**
   (`:162-166`);
2. `rebatch_input(examples, max_batch_size, batch_type)` (`:179`);
3. for each batch, the matching promises are _moved_ into the batch job by
   `batch.example_index` (`:182-183`) and posted to a worker.

**Order restoration is therefore free**: no post-hoc unsort — each result fulfills the
promise that already sits at its original index. The streaming variant
`consume_batches` (`:191-226`) reads `read_batch_size` examples ahead (default
`max_batch_size * 16`, `:210-211`) so the length sort sees a wide window, then pops
futures FIFO into the result writer.

## The Padder — removing padding _inside_ the model

When one batch still mixes lengths, `Padder` (`padder.h:10-34`, `src/padder.cc`) removes
padding around position-independent regions (FFN/projections):

- Ctor (`padder.cc:7-54`) builds two int32 index maps from the lengths: `_padded_to_flat`
  (keep only real tokens) and `_flat_to_padded` (re-expand; pad slots point at the row's
  last real token, `:39-41`). No-op if nothing is padded (`:21-22`). It can also pad the
  flat batch to a multiple (`pad_batch_to_multiple`, `:46-49` — used for int16/alignment).
- `remove_padding` (`:56-64`): reshape `[batch, time, d]` → `[batch*time, d]` then
  `ops::Gather` by `_padded_to_flat` — the tensor _shrinks_ to the real token count.
- `add_padding` (`:66-74`): inverse gather + reshape.

Gate: `Padder::allow_padding_removal(device, compute_type)` (`padder.h:12-15`) — CPU
always; **GPU only when compute_type ≠ FLOAT16**. Used by the encoder
(`src/layers/transformer.cc:446-451`) and decoder prefill (`:663-712`, including a separate
`memory_padder` for cross-attention memory). Attention itself re-adds padding around the
score computation; the savings are in the linear layers.

---

### Relevance to the Metal backend

- Everything above is **host-side orchestration** — `rebatch_input`, the promise
  plumbing, and the Padder index-map construction are device-agnostic CPU code; the only
  device work is the Padder's `ops::Gather`, which has a Metal kernel
  (`ct2_gather_bytes`, routed at `src/ops/gather.cc:90`).
- `allow_padding_removal` returns **false for fp16 on any non-CPU device** — so a Metal
  fp16 run keeps padded shapes through the encoder/prefill (the rule was written for CUDA
  Tensor-Core alignment but Metal inherits it). int8/fp32 Metal runs do get padding
  removal.
- The longest-first sort shapes what Metal sees in prefill benchmarks: batch GEMM sizes
  are near-uniform within a batch by construction — relevant when reading
  `METAL_BENCHMARKS.md` prefill numbers.
- Batch shrink during decode (reason 2 of the sort) is the same shrinking-`m` behavior
  flagged in `decoding-loop-and-beam-search.md`'s Metal bridge.
