# Plan: bounded command-buffer batching for the Metal backend

Status: **implemented, default-on** on branch `fable/metal-cb-batching` (2026-11-14; local
only, not merged). Every goal was met; see "Results". Owner: delegated to a Fable 5.2
agent. Evidence for the starting point: `METAL_BENCHMARKS.md` → "Head-to-head vs upstream
`Device::MPS`" → "Why MPS wins Whisper", and the M20 "Split-K fused SDPA" section.

## Why

After split-K SDPA (M20), Whisper large-v3 fp16 on clip30 took 3.35 s per pass with the
GPU busy for only about 2.62 s of it. The backend committed **one command buffer per op**:
about 111.7k buffers per pass against upstream `Device::MPS`'s 54k, a fixed GPU-side gap
per buffer, and a CPU that spent 0.88 s per pass blocked in `[queue commandBuffer]` on the
64-buffer queue limit.

A first attempt at this in 2026-06 ("command-buffer reuse") kept one buffer open per
thread and committed **once per decode step or forward pass**. It was correct and it was
slower: prefill bs=8 fp16 −23%, decode bs=8 −6%, because the GPU sat idle until the final
commit. It also hit a real cross-thread bug: with `OPENMP_RUNTIME=NONE`, float Conv1D
issues a Metal GEMM per `cpu::parallel_for` chunk on BS_thread_pool workers, and their
per-thread batches were never committed.

## Goal

Batch several ops per command buffer **without** giving up CPU/GPU overlap.

1. Whisper large-v3 fp16 clip30: a repeatable win over main (3.35 s per pass), transcript
   `be969f62b3` unchanged in fp16 and fp32.
2. No regression in Qwen2.5-0.5B prefill or decode (fp16/fp32, bs=1/8, 128 and 512
   tokens), nor in OPUS-MT / NLLB-600M fp16 bs=1/8, beyond spread.
3. Report command buffers per pass, GPU busy time, and CPU time blocked acquiring a
   command buffer, before and after.

A neutral or negative result was an acceptable outcome, to be reported and left off.

## Design

### One shared open batch, in issue order

Every Metal op is a `metal::CommandScope` (`src/metal/device.h`), an RAII object that
replaces the old `new_command_buffer()` / `commit_command_buffer()` pair at all 28 call
sites (25 compute kernels in `primitives.mm`, 3 MPS paths in `gemm.mm`).

- There is **one open command buffer for the whole process**, not one per thread. A scope
  takes `g_commit_mutex` in its constructor and releases it in `commit()`, so ops from
  different threads are appended in the order their scopes were entered and never
  interleave inside an op.
- Consecutive compute ops share **one compute encoder**. Dispatches of one encoder run
  serially and in order. Encoder state is not reset between ops, so each op must set
  everything its kernel reads; every existing site already did.
- MPS (`encodeToCommandBuffer:`) cannot share an open encoder. `command_buffer()` ends the
  shared encoder first; the next compute op opens a new one in the same buffer.
- The scope pushes an `NSAutoreleasePool` in its constructor and drains it in `commit()`
  (or its destructor), on the issuing thread, exactly as before. The open buffer, its
  encoder and the last committed buffer are explicitly retained, so they outlive the op's
  pool and can be ended or committed from another thread.
- A scope destroyed without `commit()` (an exception while encoding) releases the lock and
  the pool.
- No per-op allocation: the batch state is four globals.

### Invariants

1. **Issue order = encode order = commit order = GPU execution order.** One mutex covers
   scope entry to scope commit, buffers are committed inside that mutex to a single FIFO
   queue, and the passes of one buffer run in encode order.
2. **`g_last_committed` is the most recently committed buffer**, so its completion implies
   every earlier buffer completed.
3. **`metal::flush()` = commit the open buffer, then wait for `g_last_committed`.** After it
   returns, every op committed before the call has executed.

Invariant 3 is why no host read point changed. Every one of them already went through
`metal::flush()`: `METAL_DEVICE_CASE` (`src/device_dispatch.h`, before any CPU-reference
op), `synchronize_stream` / `synchronize_device`, device→host `StorageView` copies, the
sampling ops' `metal::synchronize()`, and the replica idle hook. Invariant 1 is why the
Conv1D bug cannot recur: a worker thread's ops are in the same batch as everyone else's,
so the reading thread's flush commits them, and a later op on another thread cannot run
before them. There is nothing to commit at `parallel_for` or thread-pool boundaries.

When a batch is committed affects timing only; the invariants hold for any policy.

### When a batch is committed

After each op (`batch_is_due_locked()` in `src/metal/device.mm`), the open buffer is
committed, without waiting, if any of these holds:

- **Full:** it holds `CT2_METAL_CB_OPS` ops (default 8).
- **Enough GPU work:** `ops × (moving average of measured GPU time per op)` reached
  `CT2_METAL_CB_TARGET_US` (default 100 µs). The average comes from a completion handler
  on each buffer (`GPUEndTime − GPUStartTime` divided by its op count, weight 1/8). Large
  ops therefore get a buffer each, which is the per-op behaviour.
- **GPU idle:** the last committed buffer has already completed. Holding ops back from an
  idle GPU only loses overlap.

The in-flight bound is the command queue's own limit (64 uncompleted buffers, the open one
included); `CT2_METAL_CB_MAX_INFLIGHT` creates the queue with a different limit.

### Switches

| Variable                       | Default | Effect                                                          |
| ------------------------------ | ------: | --------------------------------------------------------------- |
| `CT2_NO_METAL_CB_BATCH=1`      |     off | One command buffer per op, the pre-M21 path                     |
| `CT2_METAL_CB_OPS`             |       8 | Maximum ops per command buffer (1 = per-op)                     |
| `CT2_METAL_CB_TARGET_US`       |     100 | Estimated GPU time that commits a batch early; 0 = count only   |
| `CT2_METAL_CB_MAX_INFLIGHT`    |  0 (64) | Queue limit on uncompleted buffers; 0 keeps the queue default   |
| `CT2_METAL_CB_EAGER=0`         |      on | Turns off the commit-when-GPU-idle rule                         |
| `CT2_METAL_CB_SHARE_ENCODER=0` |      on | Diagnostic: one compute encoder per op inside the shared buffer |

`CT2_METAL_STATS=1` now also prints `ops`, `partial_batch_flushes`, `acquire_wait_ms`
(time inside `[queue commandBuffer]`), `gpu_busy_ms` (summed per-buffer GPU interval) and
the last GPU-time-per-op estimate. The timed fields are collected only when stats are on.

### Not done

- Upstream's per-thread streams with a registry, explicit `synchronize("reason")` labels
  and per-stream error reporting. One ordered batch made them unnecessary here.
- `commandBufferWithUnretainedReferences`. Buffers are freed while in-flight commands
  still reference them (the allocator does not cache), so it would be a use-after-free.

## Results

Apple M4 Max (40-core GPU), 64 GB, macOS 27.0, 2026-11-14. Release, Accelerate,
`OPENMP_RUNTIME=NONE`, `ENABLE_PROFILING=OFF`. All timing is a same-binary alternating
A/B through `CT2_NO_METAL_CB_BATCH=1` (off) against the default (on), fresh process per
arm, arm order reversed on odd rounds, under the quiet-machine gate from the M20 work (a
run is discarded and repeated if another process is above 20% of a core when it ends; 41
discarded during the final matrix). The off arm was also checked against a build of main
itself.

### Outcome against the goals

| Goal                                          | Result                                                              |
| --------------------------------------------- | ------------------------------------------------------------------- |
| Whisper fp16 win over 3.35 s, same transcript | **3.35 → 3.19 s** (1.05×), `be969f62b3` in fp16 and fp32, every run |
| No Qwen prefill or decode regression          | Prefill 0 to +10%; decode **+2% to +31%**; hashes identical         |
| No OPUS-MT / NLLB regression                  | **+5% to +58%**; hashes identical                                   |
| Counters before and after                     | Below                                                               |

Default-on.

### Correctness

- Full C++ suite, batching on: 307 passed, 3 skipped, 1 failed
  (`CPU/OpDeviceFPTest.Conv1DGroupNoBiasQuantized/float32`, the known Accelerate failure).
  With `CT2_NO_METAL_CB_BATCH=1`: the same tally.
- `*Metal*:*METAL*`: 117 passed, 2 skipped (the two known Conv1D skips), in both modes, and
  also with both early-commit rules off at 8 ops per buffer and at a batch size that never
  fills (so only flushes commit).
- New tests in `tests/metal_test.cc`:
  - `CommandBatchManySmallOpsThenHostRead`: dependent chains of 1, 15, 16, 17 and 1,003
    ops, then one read.
  - `CommandBatchHostReadMidBatch`: a read through `StorageView::to(CPU)`, with no explicit
    synchronize, after every burst length from 0 to 40.
  - `CommandBatchMixedMpsAndComputeOps`: MPS GEMM, Add and ReLU alternating in one chain,
    fp32 and fp16, against the CPU.
  - `CommandBatchOpsIssuedFromWorkerThreads`: six threads issue chains and exit without
    flushing, a second wave consumes their outputs, and the main thread reads.
  - `CommandBatchConv1DParallelForGemm`: grouped float Conv1D (GEMM per `parallel_for`
    chunk) against the CPU.
  - `CommandBatchEnvMatrix`: reruns the five tests above in child processes under 11
    settings, because the knobs are read once per process: kill-switch, N=1, N=2, N=3
    with separate encoders, N=8, in-flight 1 and 2, a batch that never fills, and each
    early-commit rule on its own.
- Mutation check: a build whose `flush()` skipped the open batch failed
  `CommandBatchEnvMatrix` in all six batched settings it applied to and passed the two
  per-op settings.
- Real-model output hashes are identical between arms in every cell of the matrix below:
  Whisper `be969f62b3` (fp16, fp32), Qwen 128 greedy tokens `c1444f2d70` (bs=1) and
  `f653792625` (bs=8) in fp16 and fp32, Qwen 512 tokens, both prefill lengths, OPUS-MT
  `1255ca4303` / `f1ed445777`, NLLB `8fba0a74d0` / `cece2ba05f`.

### End-to-end A/B

Median of per-process medians; the range is over process medians.

| Workload                                  | off (per-op)           | on (default)               | Change        |
| ----------------------------------------- | ---------------------- | -------------------------- | ------------- |
| Whisper large-v3 fp16, s/pass             | 3.35 (3.35–3.38)       | **3.19** (3.16–3.58 †)     | **1.05×**     |
| Whisper large-v3 fp32, s/pass             | 4.44 (4.39–6.32 †)     | **4.30** (4.19–4.69 †)     | 1.03×         |
| Qwen decode 128, fp16 bs=1, tok/s         | 179.0 (177.4–181.8)    | **235.0** (234.4–237.5)    | **+31%**      |
| Qwen decode 128, fp16 bs=8, tok/s         | 1208.1 (1205.2–1211.9) | **1486.2** (1480.0–1495.0) | **+23%**      |
| Qwen decode 128, fp32 bs=1, tok/s         | 149.5 (149.0–150.3)    | **161.1** (161.0–161.7)    | +7.8%         |
| Qwen decode 128, fp32 bs=8, tok/s         | 857.4 (857.2–858.2)    | 873.2 (872.5–873.8)        | +1.8%         |
| Qwen decode 512, fp16 bs=1, tok/s         | 184.5 (183.0–187.7)    | **223.1** (219.4–223.9)    | **+21%**      |
| Qwen decode 512, fp16 bs=8, tok/s         | 1324.8 (1281.6–1326.0) | **1534.8** (1510.4–1539.0) | **+16%**      |
| Qwen decode 512, fp32 bs=1, tok/s         | 146.2 (146.2–146.4)    | **155.9** (155.2–156.1)    | +6.6%         |
| Qwen decode 512, fp32 bs=8, tok/s         | 854.8 (688.4 ‡–858.2)  | 870.7 (723.4 ‡–872.4)      | +1.9%         |
| Qwen prefill 128, fp16 bs=1, prompt tok/s | 3451.9 (3367.5–3454.5) | **3791.9** (3778.4–3828.4) | +9.8%         |
| Qwen prefill 128, fp16 bs=8               | 5886.8 (5782.7–5922.6) | 5945.2 (5860.2–6029.7)     | within spread |
| Qwen prefill 128, fp32 bs=1               | 3344.9 (3341.4–3345.3) | **3473.2** (3466.0–3482.8) | +3.8%         |
| Qwen prefill 128, fp32 bs=8               | 5381.3 (5362.4–5488.9) | 5451.8 (5378.4–5480.5)     | within spread |
| Qwen prefill 512, fp16 bs=1               | 7902.2 (7526.2–7937.9) | **8208.5** (8145.1–8231.9) | +3.9%         |
| Qwen prefill 512, fp16 bs=8               | 9938.1 (9891.2–9952.3) | 9945.1 (9906.8–9966.4)     | none          |
| Qwen prefill 512, fp32 bs=1               | 6689.3 (6656.1–6732.1) | 6718.5 (6690.6–6754.7)     | none          |
| Qwen prefill 512, fp32 bs=8               | 7573.3 (7541.1–7583.9) | 7593.8 (7570.3–7601.9)     | none          |
| OPUS-MT fp16 bs=1, tok/s                  | 184.5 (183.1–191.0)    | **292.0** (290.4–292.9)    | **+58%**      |
| OPUS-MT fp16 bs=8, tok/s                  | 822.6 (813.7–849.9)    | **994.2** (983.1–1041.4)   | **+21%**      |
| NLLB-600M fp16 bs=1, tok/s                | 86.3 (85.3–87.5)       | **108.8** (108.5–109.4)    | **+26%**      |
| NLLB-600M fp16 bs=8, tok/s                | 256.0 (255.4–256.6)    | **269.2** (268.6–269.3)    | +5.2%         |

Whisper: 5 process rounds × (1 warmup + 5 passes), beam 5, with a third arm running a
build of main `de415aca`: fp16 3.35 s (3.34–3.37), fp32 4.54 s (4.46–4.61). Qwen decode
128, OPUS-MT, NLLB: 3 rounds × (3 warmups + 7 passes). Qwen decode 512: 3 × (2 + 5).
Prefill is a prompt of 128 or 512 tokens per row plus one generated token, timed as prompt
tokens per second: 128 is 3 × (5 + 15), 512 is 3 × (3 + 7).

† One round was disturbed in each Whisper column. fp16: four on processes read 3.16–3.20
and one read 3.58. fp32: round 3 read 6.32 (off) and 4.69 (on) back to back; the other
four rounds were 4.39–4.53 (off) and 4.19–4.36 (on). The gate only samples load at the
start and end of a run, so a mid-run burst is the likely cause, not verified.

‡ Round 1 of Qwen 512 fp32 bs=8 was slow in both arms (688 and 723); the other two rounds
were 855–858 and 871–872.

The earlier sweeps, with the same harness, measured Whisper fp16 at 3.12–3.16 s for the
neighbouring settings (table below), so 3.19 s is at the slow end of what this change has
shown on this machine.

### Counters before and after

One process per cell with `CT2_METAL_STATS=1` (which adds a completion handler per buffer),
1 warmup + 4 passes for Whisper, 2 + 8 for the others; figures are per pass.

| Per pass                       | Whisper fp16 off | Whisper fp16 on | Whisper fp32 off | Whisper fp32 on |
| ------------------------------ | ---------------- | --------------- | ---------------- | --------------- |
| Ops                            | 111.7k           | 111.7k          | 111.7k           | 111.7k          |
| Command buffers                | 111.7k           | **20.8k**       | 111.7k           | **26.1k**       |
| Ops per buffer                 | 1                | 5.4             | 1                | 4.3             |
| GPU busy (summed intervals)    | 2.62 s           | 2.49 s          | 3.49 s           | 3.41 s          |
| CPU blocked acquiring a buffer | 0.85 s           | 0.60 s          | 1.56 s           | 1.07 s          |
| Wall (this process)            | 3.37 s           | 3.18 s          | 4.44 s           | 4.22 s          |
| Peak memory footprint          | 4.47 GB          | 4.50 GB         | 8.65 GB          | 8.66 GB         |

| Whole process (10 passes)   | Buffers off → on | CPU blocked acquiring | GPU busy off → on |
| --------------------------- | ---------------- | --------------------- | ----------------- |
| Qwen decode 128, fp16 bs=1  | 441.0k → 64.1k   | 55 → 5 ms             | 5.43 → 4.81 s     |
| Qwen decode 128, fp16 bs=8  | 441.0k → 66.3k   | 74 → 5 ms             | 6.35 → 5.64 s     |
| Qwen prefill 512, fp16 bs=8 | 9.2k → 7.2k      | 54 → 8 ms             | 4.46 → 4.39 s     |
| NLLB-600M fp16 bs=1         | 76.0k → 15.8k    | 141 → 2 ms            | 1.40 → 1.38 s     |

Whisper is still GPU-bound after the change: the CPU still spends 0.6 s per pass waiting
for a queue slot, and the GPU is busy for 2.49 s of a 3.18 s pass. Peak memory did not
move (within 1%) in any cell.

"GPU busy" is the sum of each buffer's `GPUEndTime − GPUStartTime`. With one op per
buffer those intervals overlap (see the prefill finding below), so the off and on columns
are not measuring quite the same thing. Treat the busy figures as indicative; wall time is
the result.

### How the defaults were chosen

Three sweeps, each 2 alternating rounds, fresh process per arm. Values are the two process
medians, or one where the two rounds were not comparable.

**Sweep 1: op count only** (no early-commit rules). Round 0 of the Whisper column ran with
a steady Firefox video load on the GPU (about 0.2 s slower in every arm, same ordering);
the quiet round is shown.

| Ops per buffer | Whisper fp16 s | Qwen dec fp16 bs=1 | Qwen dec fp16 bs=8 | Prefill 512 fp16 bs=8 | Prefill 512 fp16 bs=1 |
| -------------: | -------------- | ------------------ | ------------------ | --------------------- | --------------------- |
|        1 (off) | 3.36           | 180.1, 181.3       | 1209, 1207         | 9813, 9825            | 7844, 7902            |
|              2 | 3.23           | 213.8, 212.6       | 1403, 1386         | 9817, 9841            | 8174, 8151            |
|              4 | 3.15           | 226.9, 227.2       | 1458, 1453         | 9400, 9261            | 8085, 8018            |
|              8 | **3.12**       | **234.7, 234.5**   | **1453, 1457**     | 8463, 8499            | 7390, 6919            |
|             16 | 3.14           | 234.2, 234.5       | 1450, 1452         | 7861, 7887            | 6646, 6456            |
|             32 | 3.16           | 229.2, 229.7       | 1413, 1416         | 7676, 7703            | 5011, 5534            |
|             64 | 3.29           | 219.0, 220.3       | 1362, 1358         | 7666, 7704            | 6270, 5879            |
|            256 | 3.65           | 142.8, 147.5       | 933, 914           | 7546, 7584            | 5419, 5626            |

In-flight limit and encoder sharing at 8 ops per buffer, Whisper fp16 (quiet round):

| Setting                      | s/pass                                          |
| ---------------------------- | ----------------------------------------------- |
| in-flight 2                  | 3.93                                            |
| in-flight 4                  | 3.21                                            |
| in-flight 8                  | 3.19                                            |
| in-flight 16                 | 3.16                                            |
| in-flight 64 (queue default) | 3.12                                            |
| in-flight 256                | 3.12 (3.09 in the Firefox-free part of round 0) |
| separate encoder per op      | 3.10 (3.11)                                     |

Decode and Whisper peak at 8–16 ops and fall off from 32: the 2026-06 result, one commit
per step, is the far right of this curve. **Prefill regressed with op count** (bs=8: −14%
at 8, −20% at 16), which is the same loss the 2026-06 attempt recorded. An in-flight limit
below the queue default never helped and upstream's 8 cost 0.07 s; sharing the encoder
measured no different from one encoder per op.

**Sweep 2: add commit-when-GPU-idle** (still count only):

| Setting            | Prefill 512 fp16 bs=8 | Qwen dec fp16 bs=8 | Whisper fp16 s |
| ------------------ | --------------------- | ------------------ | -------------- |
| off                | 9882, 9892            | 1237, 1204         | 3.40, 3.37     |
| 8, no early commit | 8462, 8472            | 1455, 1452         | 3.14, 3.13     |
| 4                  | 9631, 9617            | 1453, 1456         | 3.16, 3.15     |
| 8                  | 9124, 9182            | **1495, 1494**     | **3.12, 3.12** |
| 16                 | 8803, 8779            | 1491, 1493         | 3.16, 3.16     |
| 32                 | 8638, 8733            | 1467, 1472         | 3.21, 3.21     |

The idle rule recovers about half the prefill loss and adds 3% to decode bs=8, but prefill
bs=8 is still 7% down at 8 ops.

**Why prefill loses.** `CT2_METAL_STATS=1` on prefill 512 fp16 bs=8 (12 passes): with one
buffer per op the summed GPU intervals were 5.33 s for 4.98 s of wall time, so consecutive
buffers overlap on the GPU. With 8 ops per buffer and no early commit the sum fell to
3.79 s while wall time rose to 5.77 s and the CPU's time blocked in `flush()` rose from
3.53 s to 4.41 s. The CPU was waiting and the GPU was doing less per unit of time. The
likely reading, not verified beyond these counters: the GPU pipelines separate command
buffers but runs the passes inside one buffer strictly one after another, which costs
nothing for small ops and a lot for ops of several hundred microseconds. Encoder sharing
is not the cause (same result with one encoder per op).

**Sweep 3: add the GPU-time target** (8 ops per buffer unless noted):

| Target µs      | Prefill 512 bs=8 | Prefill 512 bs=1 | Prefill 128 bs=8 | Qwen dec bs=1 | Qwen dec bs=8 | Whisper s  |
| -------------- | ---------------- | ---------------- | ---------------- | ------------- | ------------- | ---------- |
| off            | 9942, 9841       | 7912, 7939       | 5844, 5775       | 179.3, 177.9  | 1211, 1205    | 3.38, 3.37 |
| 0 (count only) | 9249, 9201       | 7112, 7557       | 5684, 5602       | 237.5, 235.5  | 1491, 1489    | 3.14, 3.11 |
| 50             | 9964, 9922       | 8195, 8149       | 5934, 5904       | 230.5, 230.9  | 1464, 1466    | 3.23, 3.21 |
| **100**        | 9856, 9868       | 8159, 8115       | 5896, 5701       | 234.6, 239.6  | 1484, 1484    | 3.18, 3.16 |
| 200            | 9941, 9883       | 8236, 8249       | 6008, 5897       | 235.4, 235.1  | 1487, 1490    | 3.16, 3.15 |
| 400            | 9743, 9729       | 7676, 7767       | 5804, 6047       | 239.1, 235.0  | 1485, 1490    | 3.12, 3.12 |
| 200, 16 ops    | 9903, 9891       | 7217, 8090       | 5928, 6068       | 236.4, 235.8  | 1501, 1501    | 3.19, 3.18 |
| 200, 4 ops     | 9941, 9886       | 8197, 8103       | 5895, 6091       | 227.5, 226.0  | 1452, 1453    | 3.17, 3.17 |

All fp16. Any target from 50 to 200 µs removes the prefill loss and keeps the decode gain.
The 128-token bs=1 prefill cell (35 ms per pass) was too noisy here to rank 100 against
200, so it was rerun alone, 5 rounds × (5 warmups + 15 passes):

| Prefill 128, bs=1 | off                | target 100             | target 200         | target 100, 4 ops  |
| ----------------- | ------------------ | ---------------------- | ------------------ | ------------------ |
| fp16              | 3515.1 (3447–3545) | **3821.4** (3798–3828) | 3759.9 (3750–3776) | 3782.1 (3768–3787) |
| fp32              | 3289.0 (3253–3323) | **3480.3** (3470–3481) | 3284.7 (3177–3293) | 3429.0 (3418–3447) |

100 µs wins both, and 200 µs gives nothing in fp32, so the default is 8 ops and 100 µs.
That costs Whisper about 0.02–0.04 s per pass against the best count-only setting.

### Surprises and what is left

- **Batching by op count is not enough.** The early sweep reproduced the 2026-06 prefill
  regression at only 8 ops per buffer. What fixed it was sizing batches by measured GPU
  time, which no one had proposed going in.
- **The biggest wins are not Whisper.** Whisper is GPU-bound, so it gained 5%. The
  workloads where the CPU's per-buffer cost was on the critical path gained far more:
  OPUS-MT bs=1 +58%, Qwen fp16 bs=1 +31%, NLLB bs=1 +26%.
- **Upstream's in-flight limit of 8 does not transfer.** Here it costs time and saves no
  measurable memory, because resources are retained per command either way and the
  allocator does not cache.
- **Whisper still has idle GPU time**: busy 2.49 s of 3.18 s, with 2,356 flushes per pass
  (815 of them found work still pending). The remaining gap is host sync points inside the
  decode loop, not command-buffer count. Next candidates: find and remove the per-step
  CPU-reference ops that force those flushes, and the fused bias/residual epilogue for
  m=5 projections that the M20 diagnosis listed.
- **A latent hazard this change did not create but makes more deterministic:** a host
  _write_ into Metal memory (`StorageView::copy_from` from a CPU pointer) does not flush.
  If an uncommitted or unexecuted op still reads that buffer, it sees the new data.
  Nothing in the test suite or the six real-model workloads does this, and the same race
  existed with per-op commit under a full queue.
- The GPU-time estimate is one global moving average. A workload that alternates between
  very large and very small ops inside one sync interval will get batch sizes that suit
  neither perfectly. Whisper (a 155 µs cross-attention among 20 µs ops) is the closest
  case measured and it averages 4–5 ops per buffer.
- `CLAUDE.md` and `README.md` were not edited. Their Whisper figures (fp16 realtime factor
  and the CPU ratio) and the Qwen decode ratios are now conservative.
