# Metal Whisper — RESOLVED: the 2x gap was two kernels, not the architecture

**Status (2026-07-17, M17):** CLOSED — and inverted. Whisper large-v3, 30s clip,
beam-5 decode: Metal fp16 **4.6s (6.4× realtime)** vs CPU fp32 **20.2s (1.49×)** —
Metal is now **4.3× faster than CPU**, with a byte-identical transcript (segments,
timestamps, text) in fp16, fp32, and int8. The 730s file runs at 3.54× RT, 5.1 GB
peak RSS. The original plan below was ranked by priors; Step 0's profile invalidated
almost all of it. Measurements recorded in `METAL_BENCHMARKS.md`; milestone write-up
in `METAL_BACKEND.md` (M17).

## What Step 0 actually found

The doc's working hypothesis (per-op encode floor + beam-loop drains + Conv1D anchor)
was wrong in the ranking. Instrumentation added for this investigation — both env-gated,
committed, reusable:

- `CT2_AUTO_PROFILE=cpu|metal` (`src/profiler.cc`): auto-init the built-in profiler at
  library load, dump to stderr at exit. Works through the Python wheel with no rebuild.
  `cpu` keeps GPU submission async (wall-clock attribution — queue stalls land in the
  scope that calls `flush()`); `metal` syncs every scope (per-op GPU-inclusive time).
- `CT2_METAL_STATS=1` (`src/metal/device.mm`): command-buffer / flush / stalled-flush
  counters + total stall time, dumped at exit.

Attribution of the 39.2s (encoder ~3%, decoder forwards ~14%, `beam_search` self 78%
≈ flush-stall time 36.8s):

1. **`Transpose` had no Metal routing** → CPU reference → `metal::flush()` (full queue
   drain) per call. Beam search folds the beam into the time axis of
   `split_heads`/`combine_heads` (`time == beam = 5 ≠ 1` → real transpose; greedy
   decode takes the `time==1` reshape fast-path, which is why Qwen benchmarks never
   saw it). ~4/layer × 32 layers ≈ **128 queue drains per token**, ~12,200
   flushes/run. Fixed: `ct2_transpose_b1/b2/b4` generic rank ≤ 4 permute (dispatched
   on element width), routed in `src/ops/transpose.cc`. −2.9s, flushes → 3,186.
2. **`ct2_gather_bytes` copied each gathered row with ONE thread in a serial byte
   loop.** The per-step beam KV-cache reorder gathers a handful of ~1 MB rows (32
   layers × K and V); the GPU spent essentially the whole decode doing 5-thread
   megabyte copies. Under per-op sync it measured **94.4s = 80%** of the run. Fixed:
   2D grid, one thread per 16-byte chunk. −31.6s. **This was the gap.**

After both: flush wait 36.8s → **0.7s**; Qwen decode/prefill rechecked same-binary, no
regression (bs=8 slightly better, prefill transposes now native).

## Scorecard for the original ranked plan

| Original item            | Verdict from the numbers                                                                |
| ------------------------ | --------------------------------------------------------------------------------------- |
| 0. Profile first         | **The whole ballgame.** Both root causes were absent from the ranked list.              |
| 1. Extend decode fusion  | Not needed for the 2x. Still valid future work; re-profile first.                       |
| 2. On-device beam step   | Not needed. Residual host readbacks: ~26 flushes/token, 0.7s total.                     |
| 3. GPU Conv1D            | Deprioritized by data: whole encoder ≈ 3% of the run.                                   |
| 4. ICB replay            | Not needed. The "encode floor" priced at ~30ms/token CPU-side; the wait was the gather. |
| 5. fp16 decoder coverage | Unchanged; still the path to a fully-native fp16 model.                                 |

## If more Whisper speed is wanted later

Re-profile first (`CT2_AUTO_PROFILE`, `CT2_METAL_STATS` make this a one-command step).
Known remaining costs, none load-bearing today: ~26 flushes/token from sampler D2H +
`ApplyTimestampRules` (fp16→fp32 upcast each step) + beam bookkeeping; the per-op
encode floor (~1,000 command buffers/token); MPS-encode-per-batch-index in unfused
batched MatMul (the general trap noted in the M16 write-up).

> The whisperx skill's SKILL.md "2.4x + wrong, experiment closed" verdict was stale and
> has been updated to the M17 numbers.
