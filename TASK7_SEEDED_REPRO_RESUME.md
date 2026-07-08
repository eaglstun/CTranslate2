# Task 7 resume — close the seeded-reproducibility gap

> **Branch:** `fable/int8-metal` · **Last checkpoint:** `e8b49597` (2026-06-11, Opus checkpoint of Fable's work).
> Working tree is **clean**. This file is a cold-start brief so a fresh worker can finish task 7 in one sitting.
> Task 7 = _"sampling ops on Metal — keep LLM decode on-GPU."_

## TL;DR

Task 7 is **~95% done and committed**. TopK, TopPMask, GumbelMax, and Multinomial are all ported to
Metal, routed at `operator()` (house pattern, no `DEVICE_CASE`), and verified:

- **TopK / TopPMask** — bit-parity vs CPU reference (incl. large-vocab).
- **GumbelMax / Gumbel noise** — distributional parity; also fixed a real hole (GumbelMax previously
  threw on Metal fp16).
- **Multinomial** — draws are valid and correctly distributed (`Float16MultinomialRuns`,
  `GumbelNoiseStatistics`).

Suite: **85 passed / 0 failed / 2 skipped (known CPU Conv1D) / 1 disabled.**

**There is exactly ONE open item to close task 7:** re-enable and pass
`DISABLED_MultinomialSeededReproducible` (`tests/metal_test.cc:396`). It is currently disabled — _not_
because sampling is wrong, but because **`set_random_seed()` doesn't reseed a live generator**, so the
test's "same seed → same draws" contract can't hold. Details + fix below.

## The one open item — root cause (it's in core RNG, not Metal)

The disabled test does this (`tests/metal_test.cc:396`):

```cpp
set_random_seed(1234);
const auto first  = draw_sequence();   // 50 Multinomial(1) draws on Device::METAL
set_random_seed(1234);
const auto second = draw_sequence();
EXPECT_EQ(first, second);
```

The Metal Multinomial path draws its uniforms from the CT2 **host** generator
(`src/ops/multinomial.cc:47-51`) and hands them to the kernel via `setBytes`. The kernel is pure
inverse-CDF given those uniforms — deterministic. So the **only** nondeterminism source is the host
uniform stream. Now look at the RNG core (`src/random.cc`):

```cpp
static std::atomic<unsigned int> g_seed(default_seed);

void set_random_seed(const unsigned int seed) {
  g_seed = seed;                                    // ← only sets the atomic
}

std::mt19937& get_random_generator() {
  static thread_local std::mt19937 generator(get_random_seed());  // ← seeded ONCE, lazily, on first use
  return generator;
}
```

**The generator is `static thread_local` and seeded exactly once**, at first use, from whatever
`g_seed` happened to be then. `set_random_seed()` updates the atomic but **never reseeds an
already-constructed generator.** So in the test, `first` and `second` are just _consecutive_ draws
from one continuous mt19937 stream — they differ, and `EXPECT_EQ` fails. Nothing Metal-specific;
`multinomial_cpu.cc` shares the same generator and the same limitation.

That's what "the host-seeded path isn't fully wired" meant in the checkpoint message. The Fable run
was cut off (account spend limit) mid-debug on exactly this.

## The fix

**Minimal (makes the test pass, single-threaded):** reseed the calling thread's live generator in
`set_random_seed` — `src/random.cc`:

```cpp
void set_random_seed(const unsigned int seed) {
  g_seed = seed;
  get_random_generator().seed(seed);   // reseed the live (calling-thread) generator
}
```

**Robust (correct for real decode, cross-thread):** CT2 draws sampling uniforms on **worker**
threads, while `set_random_seed` is typically called on the **main** thread — so the minimal fix
alone won't make a real generation run reproducible. Add an epoch check so any thread's generator
reseeds when the global seed changes:

```cpp
static std::atomic<uint64_t> g_seed_epoch(0);

void set_random_seed(const unsigned int seed) {
  g_seed = seed;
  g_seed_epoch.fetch_add(1);
}

std::mt19937& get_random_generator() {
  static thread_local std::mt19937 generator(get_random_seed());
  static thread_local uint64_t seen_epoch = 0;
  const uint64_t epoch = g_seed_epoch.load();
  if (epoch != seen_epoch) { generator.seed(get_random_seed()); seen_epoch = epoch; }
  return generator;
}
```

**Recommendation:** do the **robust** version — it fixes the test _and_ makes actual on-GPU sampled
decode reproducible under `set_random_seed`, which is the whole point of task 7. Note this is a change
to **shared core** (`src/random.cc`), not the Metal backend, so it touches every backend's seeding
semantics — treat it as such (see verification).

⚠️ **Confirm upstream intent.** CTranslate2 upstream may deliberately treat `set_random_seed` as
"seed for newly-created thread generators only." Grep the history / upstream for why it was atomic-only
before changing the contract; if there's a reason, scope the reseed so it can't regress documented
behavior. Flag it in the commit message either way.

## Verification checklist

1. Remove the `DISABLED_` prefix on `MetalTest.MultinomialSeededReproducible` (and delete the
   now-stale "not yet wired" comment block at `tests/metal_test.cc:391-395`).
2. Build + run the Metal suite; target **86 passed / 0 failed / 2 skipped / 0 disabled**.
3. **Regression guard for the core change:** run the full (non-Metal) test suite too — `set_random_seed`
   is shared. Confirm no CPU/CUDA sampling or seeding tests regress.
4. Sanity: with `CT2_NO_METAL_SAMPLING=1` set, the same test should pass on the CPU-reference path
   (proves the fix is in the RNG core, not the kernel).

## After it's green

- **Re-enable in the milestone doc:** `METAL_BACKEND.md` still says _"Status as of 2026-06-08"_ and its
  milestone list stops at **M14 (int8 prefill)** — task 7 / sampling ops isn't recorded there yet. Add
  an **M15 — sampling ops on Metal (TopK/TopPMask/GumbelMax/Multinomial, seeded-reproducible)** entry so
  the doc matches the branch.
- Commit on `fable/int8-metal` in the house style (end with the `Co-Authored-By: Claude ...` trailer;
  keep it local/unpushed unless told otherwise — the checkpoint was `Local only, not pushed`).

## Neighborhood context (NOT task 7 — so you don't chase ghosts)

- **Whisper already runs on Metal.** Despite `METAL_BACKEND.md` listing Conv1D as "Deferred," the
  `WHISPER_METAL_BRINGUP.md` **2026-06-09 update** fixed fp16 Whisper end-to-end (Conv1D + a missing
  autorelease pool): full 730s file, **1.01× RT, 2.06 GB flat RSS**, fp16 output byte-identical to CPU
  on a 30s clip. So whisperx-on-Metal is _functional today_ — the open Conv1D item is a **native/fast**
  Metal Conv1D (it currently upcasts fp16→fp32 and runs the CPU reference), i.e. a perf task, not a
  correctness blocker. Don't treat Whisper as broken.
- Genuinely still deferred (not task 7): flash-attention on Metal, native Conv1D, AWQ int4 GEMM, bf16,
  multi-GPU. See `METAL_BACKEND.md` "Deferred / out of scope."

## Conventions to respect (from METAL_BACKEND.md "Gotchas")

- **No `Device::METAL` as a real `DEVICE_CASE`** in `device_dispatch.h` — graduate ops via targeted
  routing at `operator()` (that's why the sampling ops are wired the way they are).
- `.mm` files use **manual retain/release, not ARC**.
- **All referenced Metal buffers must be bound** before dispatch.
- MPS is **row-major** like `StorageView` — no cuBLAS column-major swap.
- Kernel library compiles **lazily** on first pipeline use; a bad kernel surfaces as a runtime error
  from the first op that needs it, not at load.

---

_Authored by Claude Opus 4.8 (1M context) as a cold-start brief for a fresh worker, 2026-07-07.
Diagnosis traced to `src/random.cc:10-21`, `src/ops/multinomial.cc:31-59`, `tests/metal_test.cc:391-426`._
