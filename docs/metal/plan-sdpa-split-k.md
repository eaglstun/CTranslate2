# Plan: split-K (flash-decoding) fused SDPA for long-key decode attention

Status: **open**, written 2026-11-13. Owner: delegated to a Fable 5.2 agent on branch
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

_(to be filled in by the implementing agent)_
