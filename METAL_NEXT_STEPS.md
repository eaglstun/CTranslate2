# Metal Backend — Next Steps

Session handoff, 2026-07-20. Three ranked improvement targets, each grounded in the
compute-type benchmark sweep run this session (OPUS-MT 75M, NLLB-200 600M, Qwen2.5-0.5B —
see `METAL_BENCHMARKS.md`). Benchmark rig lives in `../EasyNMT/ct2-bench/`
(`bench.py`, `bench_nllb.py`, `bench_qwen.py`, `bench_qwen_prefill.py`), run through the
Metal-capable ctranslate2 4.8.1 in the whisperX venv
(`/Users/eeaglstun/Documents/dev/whisperX/.venv`, linked to `~/.local/ct2-metal`).

Two env-gated profilers are already in-tree: `CT2_AUTO_PROFILE=cpu|metal` and
`CT2_METAL_STATS=1`.

---

## Goal 1 (highest leverage) — close the encoder-decoder FP16 gap

**The finding.** On translation models (OPUS-MT, NLLB, M2M), FP16 is a _pessimization_ — it
loses to both `float32` and `int8`. On decoder-only Qwen, FP16 _wins_ every regime. Same
backend, opposite verdict. The mechanism: enc-dec models route some hot ops to the CPU
reference, forcing fp16→fp32→fp16 round-trips; the tax scales with activation volume (the
penalty grows with batch). Qwen's hot path (fused SDPA, RMSNorm, RoPE) is all fp16-native, so
it pays no tax. This is a coverage gap, not a law of physics.

**Why it's #1.** It's the difference between "Metal wins translation ~1.5×" and "Metal wins
translation like it wins LLMs (3×+)." FP16 is the knob users reach for by default, so a
backend where that knob makes translation _slower_ is a footgun. Whisper (also enc-dec) already
wins in FP16 after M17, so its hot ops are covered — the Marian/NLLB models must touch
something Whisper skips.

**Concrete first step (≈10 min, no code).** Profile the fp16 enc-dec path to name the guilty
op(s):

```bash
cd /Users/eeaglstun/Documents/dev/EasyNMT/ct2-bench
CT2_AUTO_PROFILE=metal CT2_METAL_STATS=1 \
  /Users/eeaglstun/Documents/dev/whisperX/.venv/bin/python bench_nllb.py \
  --batch 32 --beam 5 --runs 1 --warmup 0
```

Look for ops running on the CPU reference and/or fp16↔fp32 conversions in the hot loop.

**Prime suspects (hypotheses, confirm with the profiler — do NOT assume):**

- `tile` — beam search replicates encoder states across beams; enc-dec + beam-specific, and it
  has NO Metal routing (confirmed via `grep -L Device::METAL src/ops/*.cc`).
- A general-axis `softmax` / `layer_norm` variant these models hit that Whisper doesn't.
- Other unrouted ops in the list: `alibi_add`, `mean`, `sub`, `sum`, `min_max`.

**The fix pattern (established — same as Whisper M17).** For the guilty op: add a native Metal
kernel in `kernels/kernels_msl.h`, check `x.device() == Device::METAL` at the `operator()`
level and call the `metal::` entry point (return before generic dispatch), else fall through to
CPU reference. Verify parity with the `metal-parity-verifier` agent (fp32 + fp16), then re-run
`bench_nllb.py` to confirm fp16 crosses back above fp32.

**Success criteria.** NLLB fp16 ≥ fp32 at bs≥8 (ideally approaching the int8 line); no parity
regression on the op/decode suites.

---

## Goal 2 (runner-up) — the int8-loses-e2e anomaly

**The finding.** The `int8-metal-project` notes record that int8 decode GEMV _beats_ fp16 at
the kernel level. But end-to-end on Qwen this session, int8 was the _slowest_ Metal option
(decode 1.06×/2.05×/1.67× vs fp16 1.23×/3.62×/2.56×). A kernel that wins in isolation but loses
e2e = free performance hiding around it.

**Why it matters.** int8's ~42% RSS reduction (per the notes) is what lets _big_ models fit on
a Mac. Right now you pay memory to lose speed — nobody takes that trade. If int8 delivered its
kernel-level win e2e, you'd get both.

**Concrete first step.** Profile Qwen decode under int8 and compare against fp16:

```bash
CT2_AUTO_PROFILE=metal CT2_METAL_STATS=1 \
  /Users/eeaglstun/Documents/dev/whisperX/.venv/bin/python bench_qwen.py \
  --batch 8 --gen-len 64 --runs 1 --warmup 0 --model-dir qwen05-fp32
```

Questions to answer: is the fast int8 GEMV actually being dispatched in the real decode loop,
or is it falling back? How much time goes to `dequantize`/`quantize` around the GEMMs? Are the
non-GEMM ops (norms, SDPA) running int8-adjacent in fp32 and forcing conversions?

**Success criteria.** Either (a) int8 decode ≥ fp16 on Qwen, or (b) a documented, understood
reason it can't be (so the README guidance is provably right, not just empirically observed).

---

## Goal 3 (further out — needs a bigger test than this session ran)

Everything benchmarked so far is ≤600M. Two things only surface at scale:

**3a. int8 memory win at scale.** Convert a 3B–7B model (e.g. Qwen2.5-3B/7B) and re-run the
decode + prefill sweeps. Does int8's memory-traffic advantage finally pay off e2e under real
memory pressure? This is the natural continuation of Goal 2 and the real-world case for int8.

**3b. Native BF16 compute path.** The README advertises BF16, modern LLMs are natively bf16, and
M3+ GPUs support bf16 in hardware. We only benchmarked fp16 (which requires bf16→fp16
conversion and has range issues — recall the Gemma2 fp16-`tanh` NaN saga, `metal-gemma2-broken`
notes). A native bf16 path could be a _correctness_ win as much as a perf one: match
model-native precision, avoid the fp16 range clamps. This is a real project, not an afternoon —
scope it before committing.

**Success criteria.** 3a: int8 wins (or ties) fp16 e2e on a 3B+ model. 3b: bf16 compute type
runs end-to-end matching CPU, and removes at least one existing fp16 range workaround.

---

## Quick reference — what's already routed to Metal

Ops WITH a Metal path (as of this session): `add`, `bias_add`, `concat`, `conv1d`,
`dequantize`, `gather`, `gelu`, `gemm`, `gumbel_max`, `layer_norm`, `matmul`, `mul`,
`multinomial`, `quantize`, `relu`, `rms_norm`, `rotary`, `sigmoid`, `slide`, `softmax`,
`split`, `swish`, `tanh`, `topk`, `topp_mask`, `transpose`.

Ops with NO Metal routing (candidates for Goal 1): `tile`, `alibi_add`, `mean`, `sub`, `sum`,
`min_max`, `cos`, `sin`, `log`, `median_filter`. (Regenerate with
`for f in src/ops/*.cc; do grep -q "Device::METAL\|metal::" "$f" || echo "$(basename $f)"; done`
— ignore the `*_cpu.cc` entries, those are CPU halves of routed ops.)
