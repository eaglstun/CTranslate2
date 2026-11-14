# Plan: split-K (flash-decoding) fused SDPA for long-key decode attention

Status: **implemented, default-on** on branch `fable/metal-splitk-sdpa` (2026-11-13; local
only, not merged). All three success criteria and the stretch goal were met; see
"Results". Written 2026-11-13. Owner: delegated to a Fable 5.2 agent on branch
`fable/metal-splitk-sdpa`. Evidence and numbers: `METAL_BENCHMARKS.md` → "Head-to-head vs
upstream `Device::MPS`" → "Why MPS wins Whisper".

## Why

Upstream's `Device::MPS` backend (`665e3fd0`, #2077) transcribes Whisper large-v3 fp16
about 1.24× faster than this fork's Metal backend: 3.69 s vs 4.53 s on
`whisperX/audio/clip30.wav`, beam 5, M4 Max. The diagnosis attributes the largest share of
that gap to one kernel:

- **`ct2_sdpa_half` in cross-attention costs 1,873 ms of GPU time per pass, about 434 µs per
  call across 4,320 calls.** Self-attention (short keys) costs only 26 µs per call.
  Upstream does the same work as batched `tiled_gemm_f16` plus softmax in about 1.1 s.
- **The cause is the launch shape: one threadgroup per score row**
  (`src/metal/primitives.mm` `sdpa_impl`, `kernels_msl.h` `ct2_sdpa_impl`). Whisper
  cross-attention has 1 × 5 beams × 20 heads = **100 rows**, each walking **1,500 keys**
  serially across 4 SIMD-groups. That's about 38 MB of K/V per call at roughly 90 GB/s
  effective, with low occupancy and far from the bandwidth roofline.
- **Qwen decode is fine** because it has more rows and fewer keys. This kernel is what made
  Metal win LLM decode (M16), so **it must not regress.**

A second, smaller cause, one command buffer per op, is **out of scope** here. See "Not in
this plan".

## Goal

Make long-key, few-row decode attention use the GPU's full width by splitting the key axis
across several threadgroups per row and merging the partial results with a log-sum-exp
combine (flash-decoding). Keep the current single-pass kernel for every shape where it
already wins.

**Success:**

1. Cross-attention `sdpa` GPU time per call drops at least 2× (≤ ~200 µs).
2. Whisper large-v3 fp16 on clip30 gets a repeatable end-to-end win against the current
   kernel, with a byte-identical transcript.
3. No Qwen decode regression at bs=1 or bs=8, fp16 or fp32.

**Stretch:** Metal beats MPS's 3.69 s on Whisper fp16.

## Design

### Kernels (in `src/metal/kernels/kernels_msl.h`)

1. **`ct2_sdpa_split_{float,half}`, the partial pass.**
   - Grid: `num_rows × num_splits` threadgroups. Split `s` of row `r` processes keys
     `[s*chunk, min((s+1)*chunk, len))`, with the same online-softmax loop as
     `ct2_sdpa_impl`.
   - It writes a float partial `(m, l, acc[depth])` for `(r, s)` into a scratch buffer,
     merging its own SIMD-groups through threadgroup memory as today.
   - Splits that start at or past the row's `len` write the empty partial
     (`m = -INFINITY, l = 0`) and return.
   - Prefer to factor shared logic out of `ct2_sdpa_impl` rather than copy it.
2. **`ct2_sdpa_combine_{float,half}`, the merge pass.**
   - One threadgroup (or SIMD-group) per row. It reads all `num_splits` partials and computes
     `M = max m_s`, `L = Σ l_s·exp(m_s − M)`, `out = Σ acc_s·exp(m_s − M) / L`, then writes
     T.
   - `len == 0` rows still produce **exact zeros**, as in the existing SoftMax contract.

Both dispatches go into **one compute encoder in one command buffer**. Serial dispatch
order gives the partial → combine dependency. Don't add a second command buffer, because
buffer count is the other cost centre.

### Host routing (in `src/metal/primitives.mm` `sdpa_impl`)

- Pick the split path when `num_keys` is large and rows are few. Start the heuristic from
  `target_tgs ≈ 4–8 × GPU core count` (or a fixed ~256 if the core count isn't easily
  queryable), `num_splits = clamp(ceil(target_tgs / num_rows), 1, ceil(len_max / min_chunk))`,
  `min_chunk ≈ 128` keys. Use the single-pass kernel when `num_splits == 1`. **Tune the
  thresholds by measurement, not by guess,** and record the sweep.
- The scratch buffer is `num_rows × num_splits × (depth + 2)` floats. Allocate it through the
  existing Metal allocator/caching path. Per-call `newBuffer` would bring back allocation
  churn, which the repo treats as a bug.
- Keep every existing contract: `kv_stride` (capacity-strided KV cache from M18),
  `rows_per_bh`, per-row `lengths`, `depth ≤ CT2_SDPA_MAX_D`, fp32 and fp16, and float
  accumulation.
- Add an A/B kill-switch, **`CT2_NO_METAL_SDPA_SPLIT=1`**, that forces the single-pass
  kernel, following the existing `CT2_NO_METAL_*` pattern. List it in `METAL_NEXT_STEPS.md`'s
  switch list.

Alternatives Fable may evaluate if split-K disappoints. Measure before choosing; don't do
all of them:

- A key-parallel inner loop (lanes over keys instead of a `simd_sum` per key). The current
  loop does a full SIMD reduction for every key at depth 64.
- Wider threadgroups per row (more SIMD-groups) as a cheaper first step.

## Correctness gates (all required before default-on)

1. Extend `tests/metal_test.cc` (`run_sdpa_case`, `SdpaFusedParityWithReference`,
   `SdpaFusedMaskedAndBeamParity`) with cases that **exercise the split path**:
   - Whisper shape (batch 5, heads 20, q_len 1, keys 1500, depth 64).
   - Keys not a multiple of the chunk, and keys smaller than one chunk.
   - Per-row lengths, including `len = 0` and lengths that end inside or before a chunk.
   - `kv_stride > keys*depth` (capacity-strided cache).
   - depth 80 and 256, in both fp32 and fp16.
   - Also force-test the split path on small shapes with a test hook or env var, so the
     merge logic gets covered on tiny inputs.
2. **Run the full Metal test suite**, not just the new tests. It must stay green, apart
   from the known pre-existing skips (2 Conv1D).
3. **Real-model token parity:**
   - Whisper clip30 fp16 and fp32 transcripts hash `be969f62b3`, identical to CPU and to
     the current kernel.
   - Qwen2.5-0.5B greedy 128-token hashes unchanged: fp16 `c1444f2d70` (bs=1) and
     `f653792625` (bs=8).

## Performance gates

Use alternating same-binary A/B with the kill-switch. Before trusting small deltas, check
`ps -Ao pcpu,comm -r | head` for background load: `mediaanalysisd` at 100% CPU skewed
2026-11-13's later runs.

| Workload                         | Config               | Must                                     |
| -------------------------------- | -------------------- | ---------------------------------------- |
| Whisper large-v3, clip30, beam 5 | fp16, fp32           | repeatable win; report sec/pass + spread |
| Qwen2.5-0.5B decode, 128 tokens  | fp16, fp32 × bs=1, 8 | no regression beyond spread              |
| OPUS-MT / NLLB-600M, beam 5      | fp16 × bs=1, 8       | no regression (sanity)                   |

Also report per-kernel GPU time for `sdpa`, split into cross-attention and self-attention,
before and after the change.

## Rig (already built on this machine; reuse it)

Scratch root: `/private/tmp/claude-501/-Users-eeaglstun-Documents-dev-CTranslate2/2d6b8352-a889-43d1-87ff-5f9a8b7a361d/scratchpad`, called `$S` below.

- **Build:**
  `cmake -S <worktree> -B $S/fable-build -DCMAKE_BUILD_TYPE=Release -DWITH_MKL=OFF -DWITH_ACCELERATE=ON -DOPENMP_RUNTIME=NONE -DWITH_METAL=ON -DBUILD_TESTS=ON -DBUILD_CLI=OFF -DCMAKE_INSTALL_PREFIX=$S/fable-prefix`,
  then `cmake --build … -j8 && cmake --install …`.
  - **zsh trap:** pass flags as a real array (`"${FLAGS[@]}"`). An unquoted `$FLAGS` string
    doesn't word-split in zsh, and CMake then silently falls back to Intel OpenMP.
  - Use `-DENABLE_PROFILING=ON` only for profile runs, not for timing.
- **Tests:** `$S/fable-build/tests/ctranslate2_test <worktree>/tests/data` (add
  `--gtest_filter='*Sdpa*'` while iterating).
- **Python:** reuse `$S/venv-mine` (Python 3.10, faster-whisper 1.2.0, transformers 4.57.6,
  this fork's bindings). The bindings are ABI-compatible, so point
  `DYLD_LIBRARY_PATH=$S/fable-prefix/lib` at the new library. Don't rebuild the wheel unless
  a public header changes.
- **Whisper:** `$S/whisper_driver.py large-v3 metal float16 /Users/eeaglstun/Documents/dev/whisperX/audio/clip30.wav --runs 5 --warmup 1`
  prints one JSON line (sec_med, spread, out_hash).
- **Translation and Qwen:** `$S/driver.py {opus|nllb|qwen} metal <ctype> <batch> --runs 7 --warmup 3`.
- **Upstream reference** (optional, to check the stretch goal): `$S/venv-up` +
  `DYLD_LIBRARY_PATH=$S/prefix-up/lib`, device `mps`.
- **Per-kernel GPU timing:** `$S/diag-instrumentation.patch` is the scratch-only patch that
  produced the diagnosis. It adds a `[kgpu]` per-kernel GPU-time table and `gpu_time_ms` to
  `CT2_METAL_STATS=1`, plus `metal::set_profile_tag()`. Apply it to a **separate** scratch
  build for measurement only. **Never commit it.** Its `storage_view.cc` blit experiment is
  irrelevant here (it was ruled out).

## Not in this plan (report, don't build)

- **Command-buffer batching** (several dispatches per buffer, with in-flight limits as
  upstream does). It's worth roughly 0.2–0.45 s on Whisper, but it touches every op and has a
  prior reverted attempt.
- **A fused bias/residual epilogue for m=5 decode projections.**
- **Sharing cross-attention K/V reads across beams.** CT2 replicates encoder output per
  beam, so the kernel can't see that the rows share data.
- **Merging upstream or porting to `Device::MPS`.**

## Deliverables

- Commits on `fable/metal-splitk-sdpa` (local only, no push): kernels, routing,
  kill-switch, tests.
- This file, copied into the branch, with the "Results" section below filled in: parity
  hashes, the test-suite tally, the A/B tables with date, machine, run count and spread, the
  threshold sweep, and per-kernel before/after.
- A short final report: what shipped, whether it's default-on, numbers, and anything that
  regressed or surprised.

## Results

Filled in 2026-11-13 by the implementing agent (Claude Fable 5.2). Machine: Apple M4 Max
(40-core GPU), 64 GB, macOS 27.0. Build: Release, Accelerate, `OPENMP_RUNTIME=NONE`,
`WITH_METAL=ON`, `ENABLE_PROFILING=OFF`. The longer tables (kernel-level sweep, every A/B
row) are in `METAL_BENCHMARKS.md` → "Split-K fused SDPA"; this section is the record of
what was decided and why.

### Outcome against the goals

| Goal                                                      | Result                                                             |
| --------------------------------------------------------- | ------------------------------------------------------------------ |
| 1. Cross-attention `sdpa` GPU time per call drops ≥ 2×    | **Met.** 432 → 155 µs fp16 (2.8×), 445 → 160 µs fp32               |
| 2. Repeatable Whisper fp16 win, byte-identical transcript | **Met.** 4.61 → 3.35 s per pass (1.38×); `be969f62b3` in every run |
| 3. No Qwen decode regression at bs=1/8, fp16/fp32         | **Met.** Unchanged at 128 tokens; 8–15% faster at 512 tokens       |
| Stretch: beat MPS's 3.69 s on Whisper fp16                | **Met.** 3.35 s against 3.69–3.74 s re-measured the same afternoon |

The split path is **default-on**. `CT2_NO_METAL_SDPA_SPLIT=1` forces the single-pass kernel.

### What was built

- **Kernels** (`src/metal/kernels/kernels_msl.h`): the online-softmax loop, the
  threadgroup publish and the log-sum-exp merge were factored out of `ct2_sdpa_impl` into
  `ct2_sdpa_scan`, `ct2_sdpa_publish` and `ct2_sdpa_merge`. `ct2_sdpa_split_{float,half}`
  uses them over one chunk of keys per `(split, row)` threadgroup and writes
  `{m, l, acc[depth]}` floats. `ct2_sdpa_combine_{float,half}` is one thread per output
  element (grid `depth × rows`). An empty split writes `{-INFINITY, 0, 0…}`; `len == 0`
  rows give exact zeros.
- **Host** (`src/metal/primitives.mm`): one `sdpa_impl` encodes either the single-pass
  dispatch or partial + combine in **one encoder, one command buffer**. The partials
  scratch is a `thread_local` buffer from the Metal allocator that grows geometrically, so
  steady state allocates nothing. Sharing it between consecutive async launches is safe
  because every launch goes to the one serial command queue and each command buffer writes
  its partials before its own combine reads them.
- **API** (`src/metal/primitives.h`): `metal::sdpa` is unchanged for callers
  (`attention.cc` was not touched). `metal::sdpa_num_splits(rows, keys)` is the heuristic;
  `metal::sdpa_split(..., num_splits)` forces a count and is what the tests and the
  benchmark call.
- **Switches**: `CT2_NO_METAL_SDPA_SPLIT=1` (kill-switch, listed in
  `METAL_NEXT_STEPS.md`), plus `CT2_METAL_SDPA_SPLIT_TGS` and
  `CT2_METAL_SDPA_SPLIT_MIN_CHUNK` to override the two thresholds.

### Heuristic and the sweep that chose it

`splits = clamp(floor(320 / rows), 1, floor(keys / 128))`. Two things differ from the
starting point in "Host routing":

- **Round down, not up.** `DISABLED_BenchmarkSdpaSplit` shows per-call time is flat while
  `rows × splits` stays within about 350 threadgroups of 128 threads and gets worse just
  past it: the single-pass kernel costs 69–72 µs from 256 to 352 rows, 89 µs at 368, 120 µs
  at 400. For Whisper beam 5 that makes 3 splits (300 threadgroups, 169 µs) better than 4
  (400 threadgroups, 224 µs). `ceil` would have picked 4.
- **Target 320**, not 256: 320 is 8 threadgroups for each of the 40 GPU cores and sits
  inside the measured flat region. The core count was not queried at run time, so on a
  smaller Apple GPU the same target oversubscribes. On this machine oversubscribing cost
  at most about 15% against the best count and still beat single-pass at long keys, but
  **no other GPU was measured**.

End-to-end threshold sweep, Whisper large-v3 fp16, seconds per pass, 2 alternating rounds
of 1 warmup + 3 passes under the quiet-machine gate (both process medians shown):

| Setting                       | Cross-attn splits         | Round 1 | Round 2 |
| ----------------------------- | ------------------------- | ------- | ------- |
| off                           | 1                         | 4.60    | 4.65    |
| target 200                    | 2                         | 3.67    | 3.68    |
| **target 320, min chunk 128** | 3                         | 3.37    | 3.38    |
| target 640                    | 6                         | 3.38    | 3.38    |
| target 1280                   | 11                        | 3.41    | 3.42    |
| min chunk 32                  | 3, self-attention up to 3 | 3.34    | 3.34    |
| min chunk 512                 | 2                         | 3.69    | 3.68    |

A minimum chunk of 32 was 1% faster in both rounds because Whisper self-attention then
splits too. It was not adopted: the gain is small, and it would move short-context LLM
decode onto the split path, where a Qwen sweep at 128 tokens (min chunk 16, 32, 64, 128
against off; 2 rounds × 5 passes) showed no difference in either direction.

### Correctness

- **New tests** (`tests/metal_test.cc`): `SdpaSplitParityWithReference` (Whisper shape
  5 × 20 × 1 × 1500 × 64 through the production route and at 2, 3, 12 and 16 splits; keys
  not a multiple of the chunk; chunks of 1 and 2 keys; fewer keys than splits; a single
  key; depth 80 and 256; beam-shaped q), `SdpaSplitMaskedAndCapacityStrideParity` (per-row
  lengths of 0, below the SIMD-group count, inside a chunk and exactly on a chunk boundary;
  one key per split; `kv_stride > keys × depth` with poisoned slack rows; Whisper shape
  with a memory mask; scores near −144 so a wrong empty-partial or an unscaled merge
  underflows), and `SdpaSplitRouting` (the heuristic and the kill-switch). The existing
  Sdpa tests also pin the single-pass kernel. Every case runs fp32 (1e-5) and fp16 (2e-2)
  against the unfused CPU reference.
- **Mutation check**: five single-line bugs injected into the split and combine kernels
  (ignore per-row length, drop the merge weights, off-by-one chunk start, finite
  placeholder for the empty partial, wrong partial stride) all fail the Sdpa tests. The
  empty-partial mutant survived the first version of the tests, which is why the
  very-negative-score cases exist.
- **Suite**: `*Metal*:*METAL*` 111 passed, 2 skipped (the known Conv1D pair), with the
  split path on and with `CT2_NO_METAL_SDPA_SPLIT=1`. Full suite 303 of 307 passed, 3
  skipped, 1 failed: `CPU/OpDeviceFPTest.Conv1DGroupNoBiasQuantized/float32`, the
  pre-existing Accelerate-build failure recorded in `METAL_BACKEND.md`.
- **Real-model parity**: Whisper clip30 fp16 and fp32 `be969f62b3` in every run of both
  arms (same as CPU). Qwen2.5-0.5B 128 greedy tokens `c1444f2d70` (bs=1) and `f653792625`
  (bs=8) in fp16 and fp32, both arms. OPUS-MT and NLLB hashes identical between arms.
- **Beyond the gate, 512 Qwen tokens**: fp32 output is identical across the single-pass
  kernel, the split kernel and the CPU at bs=1 and bs=8. In **fp16 the two kernels
  diverge**, and a CPU fp32 arbiter shows both divergences are near-ties: single-pass fp16
  leaves the fp32 sequence at step 292 (top-2 log-prob gap 0.0044); split fp16 matches fp32
  for all 512 tokens at bs=1, and at bs=8 leaves it at step 422, where fp32's top token is
  the EOS that `min_length` suppresses and the next two are 0.0014 apart. A different split
  count is a different float summation order, so fp16 greedy output is not guaranteed to
  be token-identical between the kernels past a near-tie.

### Performance

Same-binary alternating A/B (off = `CT2_NO_METAL_SDPA_SPLIT=1`), fresh process per arm,
order reversed on odd rounds. `mediaanalysisd` and `corespotlightd` were bursting to
100–160% CPU all afternoon, so runs started only on a quiet machine and any run that ended
with another process above 20% of a core was discarded and repeated. Median of process
medians; range over process medians.

| Workload (runs)                              | off                   | split                    | Change    |
| -------------------------------------------- | --------------------- | ------------------------ | --------- |
| Whisper large-v3 fp16, s/pass (5 × 5 passes) | 4.61 (4.58–4.62)      | **3.35** (3.34–3.88 †)   | **1.38×** |
| Whisper large-v3 fp32, s/pass (5 × 5)        | 5.69 (5.69–5.77)      | **4.43** (4.37–4.56)     | **1.28×** |
| Qwen fp16 bs=1, 128 tok, tok/s (3 × 7)       | 177.2 (176.5–177.6)   | 177.2 (176.2–178.8)      | none      |
| Qwen fp16 bs=8, 128 tok (3 × 7)              | 1211.6 (1204–1316 ‡)  | 1206.1 (1201–1320 ‡)     | none      |
| Qwen fp32 bs=1, 128 tok (3 × 7)              | 150.1 (149.6–150.3)   | 150.1 (149.8–150.2)      | none      |
| Qwen fp32 bs=8, 128 tok (3 × 7)              | 858.6 (857.5–859.2)   | 856.2 (856.1–857.0)      | −0.3%     |
| Qwen fp16 bs=1, 512 tok (3 × 5)              | 164.6 (159.0–170.6)   | **181.1** (181.0–182.0)  | +10.0%    |
| Qwen fp16 bs=8, 512 tok (3 × 5)              | 1200.0 (1196–1208)    | **1321.5** (1317–1387 §) | +10.1%    |
| Qwen fp32 bs=1, 512 tok (3 × 5)              | 127.8 (127.8–128.0)   | **146.5** (146.3–146.6)  | +14.6%    |
| Qwen fp32 bs=8, 512 tok (3 × 5)              | 788.4 (555.6 §–792.2) | **855.3** (803.8–858.5)  | +8.5%     |
| OPUS-MT fp16 bs=1, beam 5 (3 × 7)            | 192.2 (190.7–198.8)   | 194.4 (194.3–194.9)      | none      |
| OPUS-MT fp16 bs=8 (3 × 7)                    | 849.6 (820.8–863.6)   | 844.6 (832.0–866.1)      | none      |
| NLLB-600M fp16 bs=1, beam 5 (3 × 7)          | 86.2 (86.0–86.6)      | 86.3 (85.4–86.6)         | none      |
| NLLB-600M fp16 bs=8 (3 × 7)                  | 256.4 (255.4–259.1)   | 256.1 (255.9–258.0)      | none      |

† Four of five split processes read 3.34–3.35 s; the fifth read 3.88 s (passes 3.85–4.01)
in the same round as the slowest fp32 split process. A mid-run background burst is the
likely cause but was not verified. ‡ Qwen fp16 bs=8 at 128 tokens is bimodal in both arms
(about 1,204 or about 1,318 per process). § One off process read 555.6 with nothing
flagged; one split process (1,387) was accepted with `corespotlightd` busy at its end.

Qwen at 128 tokens, OPUS-MT and NLLB never take the split path (fewer than 256 keys per
row, or more than 160 rows), so those rows show that the refactored single-pass kernel did
not get slower.

Per-kernel GPU time, Whisper, scratch-only instrumented build (`diag-instrumentation.patch`
applied to a copy of the tree, never committed), 1 warmup + 4 passes per cell, 4,320
launches of each kind per pass:

| Per pass                               | fp16 off         | fp16 split         | fp32 off         | fp32 split         |
| -------------------------------------- | ---------------- | ------------------ | ---------------- | ------------------ |
| Cross-attention `sdpa`                 | 432 µs, 1,864 ms | **155 µs, 668 ms** | 445 µs, 1,920 ms | **160 µs, 691 ms** |
| Self-attention `sdpa`                  | 25.7 µs, 111 ms  | 25.7 µs, 111 ms    | 26.5 µs, 115 ms  | 26.4 µs, 114 ms    |
| All kernels, GPU busy                  | 3.82 s           | 2.62 s             | 4.72 s           | 3.49 s             |
| CPU blocked in `[queue commandBuffer]` | 2.06 s           | 0.88 s             | 2.74 s           | 1.60 s             |

### Surprises and what is left

- **The wave size sets the answer.** The split count that wins is the one that fills one
  wave of threadgroups, and one more split past it is worse than several more. This is why
  the heuristic rounds down.
- **Long-context LLM decode gains too**, 8–15% on Qwen at 512 tokens. The plan expected
  "no regression" there, not a win.
- **The split arm is more sensitive to CPU contention.** Under a `mediaanalysisd` burst it
  lost 0.3–0.5 s per pass while the off arm did not move. Benchmarks of this path need a
  quiet machine or a gate like the one used here.
- **Not built, as the plan said:** command-buffer batching, the bias epilogue, beam K/V
  sharing. After this change the GPU is busy 2.62 s of a 3.38 s fp16 pass, so the
  per-command-buffer gap is now the largest remaining Whisper cost.
- **Candidates this work exposed, not built:** (a) scale the 320 target by GPU core count,
  which needs measurements on a second Apple GPU; (b) splitting when rows already exceed
  one wave and keys are long (400 rows × 1,500 keys: 945 → 640–660 µs at 16 or more splits
  in the kernel benchmark; the heuristic never splits there today, and no end-to-end
  workload was measured); (c) a key-parallel inner loop, since the split kernel still runs
  one `simd_sum` and two `exp` per key on every lane.
