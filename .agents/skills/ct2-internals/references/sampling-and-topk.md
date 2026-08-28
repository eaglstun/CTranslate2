---
title: "Sampling & TopK (BestSampler/RandomSampler, TopPMask, temperature, RNG)"
summary: >-
  Details how a row of scores becomes a token id: the Sampler strategy
  interface (outputs must be CPU tensors), make_sampler choosing BestSampler
  (greedy TopK when sampling_topk==1 or temperature==0) versus RandomSampler,
  and RandomSampler's ordered filtering pipeline (top-k restrict, temperature
  Mul on log-scores, TopPMask nucleus masking, then SoftMax+Multinomial or
  LogSoftMax+GumbelMax draw, then Gather remap). It documents the TopK op
  (axis=-1 only, returns values and int32 indices, ties by index), TopPMask as
  a shape-preserving masking op with a CUDA class cap of 8192 (256x32)
  enforced by validate_decoding_options, and the RNG story: a thread_local
  mt19937 lazily seeded per thread, with CPU and CUDA using entirely different
  RNGs so outputs are not cross-device reproducible. On Metal, sampling runs
  the CPU reference over unified memory after a synchronize flush, matching
  CPU determinism exactly.
semantic_id: "Shywxghvqm2Sg719UtE5FSVoI6-bgqo6Chd3GZy0RtYqBfkOLb8tWg044l1_pwd4Ai8smsw7zR5-9Qyzxvz5pQ"
---

CT2-architecture reference: how a row of (log-)scores becomes a token id. The decode loop
(`decoding-loop-and-beam-search.md`) hands the sampler post-processor logits; everything
here is the selection machinery below that hand-off.

Source: `src/sampling.cc`, `include/ctranslate2/sampling.h`,
`include/ctranslate2/ops/topk.h`, `src/ops/topk{,_cpu}.cc`,
`include/ctranslate2/ops/topp_mask.h`, `src/ops/topp_mask{,_cpu,_gpu}.{cc,cu}`,
`src/random.cc`. Lines verified by read on 2026-06-11.

## The `Sampler` interface

`Sampler` (`sampling.h:8-22`) is a tiny strategy class: `operator()(scores, sampled_ids,
sampled_scores, num_samples=1)`. The hard contract is in `Sampler::operator()`
(`sampling.cc:7-22`): **outputs must be CPU tensors** (it throws otherwise, `:11-12`).
If the scores live on a device, sampling runs on-device and the results are copied back
(`:16-20`) — so every decode step ends with a device→CPU copy of `[batch, k]` ids/scores.

`make_sampler` (`src/decoding.cc:1066-1074`) picks the concrete class:
**`BestSampler`** when `sampling_topk == 1` _or_ `sampling_temperature == 0` (temperature
0 = greedy), else **`RandomSampler(topk, topp, temperature)`**.

## `BestSampler` — deterministic argmax/top-k

`sampling.cc:25-32`: literally `ops::TopK(num_samples)(scores, sampled_scores,
sampled_ids)`. Beam search uses this with `num_samples = 2*beam_size`
(`decoding.cc:562`); greedy with `num_samples = 1`.

## `RandomSampler` — the filtering pipeline (`sampling.cc:41-101`)

Order matters; each stage rebinds `final_scores`:

1. **top-k restrict** (`:54-57`): if `0 < _from_topk < vocab`, `ops::TopK(_from_topk)`
   shrinks the candidate set to `[batch, k]` (and keeps `top_ids` to map back later).
2. **temperature** (`:69-74`): `ops::Mul` by `1/_temperature` — applied to the
   (possibly top-k-reduced) _log-scores_, before top-p and before the softmax.
3. **top-p / nucleus** (`:76-82`): `ops::TopPMask(_topp)` — a dedicated **op**, not
   sampler-side logic (the name in the code is `TopPMask`).
4. **draw** (`:86-96`): `num_samples == 1` → `ops::SoftMax` + `ops::Multinomial`;
   `num_samples > 1` → `ops::LogSoftMax` + `ops::GumbelMax` (the Gumbel-max trick,
   because Multinomial samples _with_ replacement).
5. **id remap** (`:98-100`): two `ops::Gather` calls map sampled positions back to
   original vocabulary ids and fetch their original scores.

## The TopK op

`ops::TopK` (`topk.h:8-21`) takes `k` at construction; **only `axis = -1` is supported**
(ctor throws otherwise, `topk.cc:14-15`). `operator()` (`topk.cc:18`) resizes both
outputs to `{batch_size, k}` — it returns **values and indices** (int32). CPU impl
(`topk_cpu.cc`): `k == 1` is a `std::max_element` scan (`:23-31`); general `k` is
`std::iota` + `std::partial_sort` per row (`:33-54`), parallelized with
`cpu::parallel_for`. Ties resolve by index order (first occurrence wins).

## TopPMask — nucleus filtering as a masking op

`ops::TopPMask(p, mask_value = -inf)` (`topp_mask.h:10-27`) does _not_ shrink the tensor:
it writes `mask_value` over tokens outside the nucleus, preserving shape. `operator()`
(`topp_mask.cc:20-45`) first computes its own `ops::SoftMax` of the input (`:26-27`),
then dispatches the kernel. CPU kernel (`topp_mask_cpu.cc:12-40`): sort ids by descending
prob, keep a token while the cumulative prob _before_ it is `< p` (so the token that
crosses the threshold is kept), mask the rest.

**Capacity limit**: `TopPMask::max_num_classes(device)` — unlimited on CPU
(`topp_mask_cpu.cc:58-61`), `256 threads × 32 items = 8192` classes on CUDA
(`topp_mask_gpu.cu:15`, `:123-126`, block-radix-sort based). That's why
`validate_decoding_options` (`decoding.cc:1056-1063`) requires `sampling_topk <=
max_num_classes` whenever `sampling_topp < 1` — on GPU you must top-k down to ≤8192
candidates before top-p can run.

## The RNG story

`src/random.cc` (11-line header `include/ctranslate2/random.h`):

- `set_random_seed(seed)` sets a global atomic; `get_random_seed()` returns it, or a
  fresh `std::random_device` draw if never set (`:14-16`).
- `get_random_generator()` (`:18-21`) returns a **`thread_local std::mt19937`** seeded
  _lazily on first use per thread_ with `get_random_seed()`.

Determinism caveats: (1) every replica worker thread has its _own_ generator — same seed
⇒ identical streams per thread, but which thread runs which batch is scheduling-dependent
in a multi-replica pool; (2) setting the seed after a thread's generator was created has
no effect on that thread; (3) **CPU and CUDA draw from different RNGs entirely** — CPU
`Multinomial` uses `std::discrete_distribution` (`multinomial_cpu.cc:14-23`), CPU
`GumbelMax` adds `-log(uniform)` noise (`gumbel_max_cpu.cc:14-24`), while CUDA uses
`curandStatePhilox4_32_10_t` states initialized from the same seed
(`src/cuda/random.cu:14-30`, `:51`) — so sampled outputs are _not_ reproducible across
devices, only within one.

---

### Relevance to the Metal backend

- **Sampling runs CPU-side on Metal.** `TopK` and `TopPMask` have explicit Metal fp16
  routes that call the already-instantiated **CPU reference over unified memory**, after a
  `metal::synchronize()` flush of the async GPU queue (`topk.cc:24-35`,
  `topp_mask.cc:31-42`). `Multinomial`/`GumbelMax` run their CPU kernels via the
  METAL→CPU dispatch binding. No sampling kernel exists in `src/metal/kernels/`.
- The `Sampler` CPU-output contract makes this near-free on Apple Silicon: ids/scores are
  already CPU-addressable; the real cost is the synchronize barrier per step — part of
  the per-op floor story in `apple-silicon`'s `dispatch-overlap-and-perf-model.md`.
- `TopPMask::max_num_classes` is effectively unlimited on Metal (the CPU-path value),
  but `validate_decoding_options` queries it per device — worth knowing if Metal ever
  grows a native top-p kernel with a class cap.
- RNG determinism on Metal == CPU determinism (same `std::mt19937`), so Metal sampling
  matches CPU sampling exactly for a given seed/thread — handy for parity tests.
