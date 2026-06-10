# Postmortem: Gemma2 `<pad>` collapse on the Metal backend

**Status:** Resolved (2026-06-09), fix in `1d079129`.
**One-line cause:** Metal's `tanh(x)` overflows to `NaN` for large `x`; the GELU-tanh
activation kernel fed it the huge arguments that Gemma2's deep-layer activations produce.
**Fix:** clamp the `tanh` argument to `[-15, 15]` in `src/metal/kernels/kernels_msl.h`
(~10 lines).

---

## Symptom

Running real `google/gemma-2-2b` (converted to CT2 fp16) on `Device::METAL`:

```
CPU   fp32:  ▁ 2 0 1 9 - 2 0 2 0 ▁school ▁year ▁is ▁off ▁to ▁a ▁great ▁start ! ...
METAL fp32:  ▁ <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> ...   (forever)
```

The **first** generated token always matched CPU; every token after collapsed to `<pad>`,
identically in fp32 and fp16. The same model on CPU produced coherent text. The bug was
**pre-existing** and **Gemma2-specific** — Qwen2.5-0.5B decoded correctly on Metal,
fp32 and fp16, 24/24 vs CPU.

## What it was NOT (three sessions of wrong suspects)

The collapse-to-a-single-token symptom is a siren song for plausible-but-wrong theories.
Every one of these was investigated and killed:

| Suspect                                    | Why it was tempting                                 | Why it's wrong                                                                                                                            |
| ------------------------------------------ | --------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `final_logit_softcapping`                  | Gemma2 caps logits; missing cap → degenerate argmax | Not even set by `Gemma2Loader`; and runs correctly on unified memory anyway                                                               |
| `attn_logit_softcapping`                   | Same family, applied pre-softmax                    | **Not implemented in CT2 at all** — a feature missing from _both_ CPU and Metal can't cause a CPU-vs-Metal divergence                     |
| Alternating sliding-window attention       | The one Gemma2-shaped attention quirk               | `Gemma2Loader` never sets `sliding_window` per layer (that's Gemma3); CT2's window is KV-cache eviction that can't fire under 4096 tokens |
| `query_pre_attn_scalar`                    | Differs from `1/sqrt(head_dim)` in general          | Defaults correctly for 2b (`head_dim == scalar == 256`)                                                                                   |
| `(1+γ)` RMSNorm, the whole attention stack | Gemma2-distinctive                                  | Exonerated by the trace below — layers 0–22 are byte-identical to CPU                                                                     |

**Lesson 1 — read the converter, not the model card.** Three of the five suspects died
just by reading what `Gemma2Loader` (`python/ctranslate2/converters/transformers.py`)
actually sets, versus what HF Gemma2 _has_. CT2's converter omits soft-capping entirely;
the "Gemma2 features" you'd reach for from memory aren't all in the converted model.

**Lesson 2 — "stateless per-step op" reasoning is weak.** The original prime suspect
(soft-capping) was demoted with "it runs on step 1 too, but step 1 is correct, so it can't
be the cause." That logic feels airtight and isn't — a wrong-but-not-yet-catastrophic op can
produce a correct argmax on step 1 and diverge later. It happened to point the right way
here, but don't lean on it. Get data.

## How it was actually found

### 1. Localize with env-gated NaN tripwires (boundary reads only)

Added env-gated (`CT2_DUMP_LAYERS`) per-layer and per-sub-op checksum/NaN dumps to the
decode loop. The decisive observations, all at **decode step 2** (the token that collapses):

- Steps 0 and 1 match CPU exactly. The divergence is born at step 2.
- Decoder layers **0–22** are **byte-identical** CPU-vs-Metal (sum _and_ maxabs match).
- **Layer 23 output is NaN** (CPU finite). NaN then propagates → final RMSNorm NaN →
  logits NaN → `argmax` falls to index 0 = `<pad>` → collapse forever.
- The NaN reaches the _real_ sampled logits, so it is genuine GPU state, not a read artifact.

### 2. The trap: mid-pipeline CPU reads are unreliable

Trying to attribute the NaN to a specific op _inside_ layer 23, a raw GEMM-output probe
read **all-NaN for every GEMM** — including prefill GEMMs that demonstrably produce the
correct token. Even with an explicit `metal::synchronize()` before the read.

**Lesson 3 — on Metal, a CPU read of a freshly-committed MPS-GEMM output is not reliable**,
because ops commit asynchronously and `flush()`/`synchronize()` did not make that specific
just-committed result visible to a CPU read in that context. Layer-_boundary_ `to(CPU)`
reads (after the layer's last committed op) **are** reliable — that's why the per-layer
trace was trustworthy and the per-GEMM probe was garbage. Do not debug Metal NaNs with raw
post-op CPU probes; use boundary reads plus the bisection below.

### 3. Bisect by forcing op families to the CPU reference

The CPU reference path runs correctly on Metal's **unified-memory** pointers (Metal binds to
the CPU dispatch case), so forcing one op family to CPU-ref is a clean, known-good A/B:

| Experiment                                                            | Result              | Conclusion                           |
| --------------------------------------------------------------------- | ------------------- | ------------------------------------ |
| Zero the GEMM output buffer before MPS (`beta=0` stale-buffer theory) | still collapses     | not a stale-buffer × `beta` issue    |
| `synchronize()` after every GEMM                                      | still collapses     | not a simple async GEMM-output race  |
| **All fp32 GEMMs → CPU reference**                                    | **still collapses** | **MPS GEMM is exonerated**           |
| **fp32 GELU → CPU reference**                                         | **24/24, fixed**    | **the Metal GELU kernel is the bug** |

That last row is the whole ballgame: with the GELU activation on CPU, Gemma2 decodes
perfectly; with it on the Metal kernel, it collapses.

## Root cause

The Metal GELU-tanh kernel (`ct2_apply_activation`, case 1, in `kernels_msl.h`):

```c
case 1: { const float u = 0.7978845608028654f * (v + 0.044715f * v * v * v);
          return 0.5f * v * (1.0f + tanh(u)); }   // <-- tanh(u) returns NaN for large u
```

Metal's `tanh(x)` is computed as `(exp(2x) - 1) / (exp(2x) + 1)`. For large `|x|`,
`exp(2x)` overflows to `Inf`, and `Inf/Inf = NaN` — whereas `tanh` should saturate to ±1
(which CPU `std::tanh` does correctly).

GELU-tanh's argument is `~v³`. Gemma2 is famous for **large, growing deep-layer
activations** (and CT2's converter ships no soft-capping to tame them): by layer 23 the
gate pre-activation `v` is large enough that `u` overflows `tanh`'s internal `exp` →
`tanh` returns `NaN` → GELU returns `NaN` → the rest of the forward pass is poisoned.

This explains every facet of the symptom:

- **Gemma2-specific:** only its activations get large enough to overflow `tanh`.
- **Layer 23 / step 2:** the deepest, largest activations, at the step where `v` first
  crosses the overflow threshold.
- **Qwen2.5 fine:** smaller activations never push the argument that far.
- **fp32 and fp16 identical:** it's an argument-magnitude overflow, not a precision issue.

## The fix

`tanh` saturates to ±1 long before its argument can overflow the internal `exp`. So clamp
the argument: `tanh(±15)` already equals ±1.0 in float32, and `exp(2·15) = exp(30)` is
nowhere near the float32 ceiling.

```c
// Metal's tanh(x) computes (exp(2x)-1)/(exp(2x)+1); for large |x| exp(2x) overflows to Inf
// and Inf/Inf = NaN, whereas tanh mathematically saturates to +-1. tanh(+-15) already equals
// +-1.0 in float32, so clamping the argument to [-15,15] is exact for the saturated region
// and avoids the overflow.
inline float ct2_tanh_safe(float x) {
  return tanh(clamp(x, -15.0f, 15.0f));
}
```

Applied to case 1 (GELU-tanh) and case 5 (Tanh). Because the clamp is a no-op for `|u| < 15`,
it is numerically exact in the meaningful range — normal-range GELU and small-activation
models (e.g. Qwen) are provably unaffected.

## Validation

- `MetalTest.DISABLED_DecodeParityLLM` on `gemma-2-2b`: **fp32 24/24** and **fp16 24/24**
  vs CPU — coherent text — **PASSED**. (Run it with a real, BOS-led prompt; see below.)
- Full `MetalTest.*` suite: **15/15 PASSED**, including `Float16BiasAddGELUMatchesFloat32`.

```bash
CT2_LLM_MODEL=/path/to/gemma2-2b-ct2-f16 CT2_LLM_PROMPT="<bos> The" \
  ./tests/ctranslate2_test ../tests/data \
  --gtest_also_run_disabled_tests --gtest_filter='MetalTest.DISABLED_DecodeParityLLM'
```

## Transferable lessons

1. **Read the converter, not the model card** — verify what the CT2 model actually contains.
2. **Don't trust "the op runs on step 1 too" reasoning** — get a per-layer trace instead.
3. **Mid-pipeline CPU reads of MPS-GEMM output are unreliable on this backend.** Trust
   layer-boundary `to(CPU)` reads; do not trust raw post-op probes.
4. **CPU-reference bisection is the high-signal tool** for a Metal correctness bug: unified
   memory lets you flip one op family to known-good CPU and A/B it. It exonerated MPS GEMM
   and pinned the GELU kernel in two runs.
5. **`tanh`/`exp` on the GPU are overflow-traps.** A library `tanh` that's fine on CPU can
   return `NaN` for large arguments on Metal. Clamp arguments to math-equivalent safe ranges.

### Test-gate gotcha (carried over from earlier sessions)

A meaningless / BOS-less prompt makes _both_ backends degenerate into the same loop, so
`CPU == Metal` holds on garbage and the parity gate **false-passes**. Gate on a real,
model-appropriate prompt (Gemma2 needs a leading `<bos>`), never a filler prompt.

## Follow-ups

- `MetalTest.DISABLED_Gemma2PrePostParity` can now have its `EXPECT_EQ(metal32, cpu)` e2e
  assertion restored (its comment said to do so once this bug was fixed).
