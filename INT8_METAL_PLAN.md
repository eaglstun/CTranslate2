# INT8 on Metal — implementation & validation plan

> **Status: SHIPPED (2026-07-07).** All phases landed — native int8×int8→int32 GEMM/GEMV,
> the Metal-4 MPP `matmul2d` prefill path, and the downstream validation harness. This
> file is retained as the **design record** (the "why" behind `tests/downstream/` and the
> resolved environment facts below), not as pending work; the shipped reality and numbers
> live in `METAL_BACKEND.md` (M11–M16) and `METAL_BENCHMARKS.md`. Original brief, 2026-06-11
> on the M4 Max, preserved verbatim below.
>
> Read `METAL_BACKEND.md` and the `apple-silicon` / `ct2-internals` skills before touching
> code — this plan assumes their design rules.

## Mission

Give the Metal backend a real **int8** path so quantized models keep their weights
int8-resident on the GPU. Right now an int8-converted model _loads_ on Metal but CT2
dequantizes the weights to fp16 because Metal has no int8 — so the headline win (≈2×
resident-memory cut for LLMs) is thrown away exactly when you need it.

### The trap (read this before "optimizing")

`MPSMatrixMultiplication` — the GEMM the whole backend rides on — is **float-only**
(`apple-silicon/references/mps-matrix-multiplication.md:73`: "INT8/INT16 GEMM are NOT
on MPS"). So:

- The tempting shortcut — _dequant int8→fp16, then reuse the cached fp16 MPS GEMM_ —
  works but **defeats the purpose**: to feed fp16 MPS you need the weights resident as
  fp16 during the multiply, so the memory win evaporates. The shim buys smaller model
  _files_ and faster _load_, not resident savings.
- **The real win requires a custom MSL int8×int8→int32 GEMM kernel.** No way around it.
  (This is also the single highest-value future kernel flagged in the apple-silicon
  skill TODO — `§6.8 simdgroup_matrix`.)

The shim is still useful — as a **Phase-1 scaffold** to get the plumbing + oracle green
before the hard kernel exists. It is a stepping stone, not the destination.

## Architecture — mirror the CPU int8 path

CT2's int8 scheme (traced from the CPU/CUDA implementations) is **symmetric, per-row,
dynamic-activation**:

```
fp16/fp32 activation
   │  Quantize          scale = 127 / amax(row)   → int8 activation + fp32 row-scale
   ▼
int8 activation × int8 weight   (weights pre-quantized at convert time + stored scales)
   │  GEMM              accumulate to INT32
   ▼
int32 accumulator
   │  Dequantize (gemm-output form): out = int32 / (a_scale · b_scale), + bias, + activation
   ▼
fp16/fp32 output
```

Key facts (do not re-derive):

- Symmetric, **no zero-point** (the `_qzero` path is AWQ — a different scheme, out of scope).
- Accumulate to **int32**, dequantize separately. Activation fn is applied in the dequant stage.
- Activation scales are **dynamic per-row at inference** (size = batch rows); weight
  scales are static, loaded from `{scope}/weight_scale`.
- The CPU s8→u8 shift + compensation term (MKL) is **NOT needed on Metal** — use signed
  int8 throughout (mirror the CUDA path, which is also signed int8 → int32 via cuBLAS).

### Four kernels to add (in `src/metal/`, MSL in `kernels/kernels_msl.h`)

| Kernel                   | Job                                                                 | Difficulty                          | Oracle                |
| ------------------------ | ------------------------------------------------------------------- | ----------------------------------- | --------------------- |
| `quantize_s8`            | per-row amax reduce → `scale=127/amax`, rescale+round+cast to int8  | low (reduction like softmax)        | op suite              |
| `dequantize` (simple)    | `out = int8 / scale`                                                | trivial                             | op suite              |
| `dequantize_gemm_output` | int32 → `/(a_scale·b_scale)` + bias + activation (all act variants) | medium                              | op suite              |
| **int8 GEMM**            | int8×int8 → **int32** accumulate, row-major, transpose flags        | **HIGH** — hand-tiled, the mountain | op suite + downstream |

### Hook points (route at `operator()`, return before generic dispatch — the CT2 Metal pattern)

- `src/ops/gemm.cc` — the `DataType::INT8` branch (`compute<D, int8_t, int32_t>`). Add a
  `Device::METAL` int8 check here, mirroring the existing fp32/fp16 `metal::gemm` branch.
- `src/ops/quantize.cc` — add a `Device::METAL` branch → `metal::quantize_s8`.
- `src/ops/dequantize.cc` — add a `Device::METAL` branch (both the simple and gemm-output overloads).
- New `src/metal/quantize.mm` / extend `src/metal/gemm.mm`; dispatch/pipeline wiring in
  `src/metal/primitives.mm` (follow the existing fused-norm kernels as a template).
- `src/layers/common.cc` `Dense::operator()` already orchestrates quantize → gemm →
  dequantize; it is device-agnostic and should need **no changes** if the ops route correctly.

## Phasing (climb under a green test the whole way)

**Phase 1 — plumbing + oracle (low risk).** Implement `quantize_s8`, `dequantize`,
`dequantize_gemm_output` as real Metal kernels. For the GEMM, use the **dequant-shim**
(int8→fp16 → existing cached fp16 MPS GEMM) as a temporary stand-in. Goal: a real
quantized model produces CPU-parity output on Metal, and `get_supported_compute_types("metal")`
starts reporting an int8 type. End state = green op suite + green downstream smoke.

**Phase 2 — the real kernel (the mountain).** Replace the shim with a hand-tiled
int8×int8→int32 MSL GEMM, **under the exact same passing tests**. Start with a correct
threadgroup-tiled kernel; consider `simdgroup_matrix` (apple-silicon TODO §6.8) only if
int8 is supported there and after correctness is locked. Now weights stay int8-resident
— the actual win. The Phase-1 oracle catches any regression the instant it appears.

## Verification — two stacked oracles

| Oracle              | Checks                                      | Tightness                                      | Where                                                                                          |
| ------------------- | ------------------------------------------- | ---------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| **C++ op suite**    | Metal int8 ≈ **CPU int8**, bit-level        | tight — real kernel-correctness gate           | `tests/ops_test.cc` etc., parameterized over `Device::METAL`; CPU int8 reference exists in C++ |
| **Downstream diff** | Metal int8 ≈ **fp16 reference**, end-to-end | loose — absorbs quant error, catches _garbage_ | the 4 consumers below                                                                          |

Note: the downstream golden is **fp16 reference output, with a quant-error tolerance**
(WER delta for Whisper, token-match-rate for the LLM, BLEU/char-match for translation) —
NOT bit-exact int8-on-CPU, because this machine's Accelerate CPU build only exposes
`float32` (`get_supported_compute_types("cpu") == {'float32'}`). The bit-tight int8-vs-int8
check lives in the C++ suite where the CPU int8 reference is available.

## Downstream validation harness (lives in THIS repo; consumers stay in their own folders)

Coupling is through the **installed library + rebuilt wheel**, never source colocation.
Build → `cmake --install` to a **pinned per-run prefix** (e.g. `~/.local/ct2-metal`) →
rebuild the wheel (`CTRANSLATE2_ROOT=<prefix>`) → `pip install --force-reinstall` into
each consumer venv → run each consumer's canonical job → diff vs golden.

Add: `scripts/validate-downstream.sh` + `tests/downstream/projects.toml` (path, venv,
run command, golden, tolerance). Capture goldens from **current `main`, fp16, before any
int8-Metal change**.

### The four consumers (materials verified present 2026-06-11)

| Consumer         | Location                                                                       | Venv                            | Model                                               | Notes                                          |
| ---------------- | ------------------------------------------------------------------------------ | ------------------------------- | --------------------------------------------------- | ---------------------------------------------- |
| whisperX         | `/Users/eeaglstun/Documents/AI/whisperX`                                       | `./.venv` (has Metal ct2 4.8.0) | auto-downloads CT2 Whisper                          | `audio/` has test inputs                       |
| faster-whisper   | `/Users/eeaglstun/Documents/AI/faster-whisper`                                 | **none yet — create one**       | auto-downloads CT2 Whisper                          | fresh clone; `uv venv` + install + Metal wheel |
| Qwen2.5 LLM      | model: `/Users/eeaglstun/Documents/AI/ct2-models/qwen2.5-0.5b-int8`            | reuse whisperX venv             | int8 (converted ✓, loads+generates on Metal fp16 ✓) | most on-point int8 target                      |
| NLLB translation | model: `/Users/eeaglstun/Documents/AI/ct2-models/nllb-200-distilled-600M-int8` | reuse whisperX venv             | int8 (converted ✓)                                  | enc-dec architecture breadth                   |

Drivers for the two model-only consumers (Qwen, NLLB) are small Python scripts — write
them in the harness. The Qwen smoke pattern that already works:

```python
gen = ctranslate2.Generator(model_dir, device="metal", compute_type="float16")  # → "int8" once it lands
tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
ids = tok("The capital of France is").input_ids
res = gen.generate_batch([tok.convert_ids_to_tokens(ids)], max_length=24, sampling_topk=1)
```

## Skill expansion (≥50 quality references each)

Grow `apple-silicon` (currently ~8 refs) and `ct2-internals` to **≥50 genuinely-relevant
references each**. Hard rule from the skill's own philosophy: **every new reference keeps
its "Relevance to the CT2 Metal backend" section and earns its place** — 50 must not
become 50 ways to dilute. Much of this is a **byproduct** of the GEMM work
(`simdgroup_matrix`, atomics, int8 conversion/packing, integer functions, quantized-matmul
notes); write those as you go, then deliberately round out adjacent API surface. Sources:
DocC JSON endpoint for Metal/MPS API, the vendored MSL spec PDF for kernel-side functions
(recipe in `apple-silicon/SKILL.md`). Run `scripts/audit-citations.sh` after.

## Environment facts (resolved — don't re-discover)

- **Build (Apple Silicon + Metal):** `cmake .. -DWITH_METAL=ON -DWITH_MKL=OFF
-DWITH_ACCELERATE=ON -DOPENMP_RUNTIME=NONE -DBUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release`.
- **Python selects Metal** via `device="metal"` (C++ `str_to_device`, `src/devices.cc:27`);
  `"auto"` picks Metal if a GPU is present. Bindings need no change.
- **Converter gotcha:** the homebrew `ct2-transformers-converter` is STALE (no Qwen2; NLLB
  hits `M2M100Encoder has no embed_scale`). Use the **whisperX venv's** 4.8.0 converter
  (`transformers` 4.57.6) — it converts both cleanly.
- `get_supported_compute_types("metal")` currently `{'float32','float16'}` — int8 appearing
  there is the graduation smoke signal.
- First MPS GEMM call pays a one-time ~493ms pipeline warmup, then is fast.

## Guardrails

- Perf-critical C++/Metal: "a misplaced pointer costs hours." Keep changes within the area
  understood; respect the op-routing pattern (no `Device::METAL` `DEVICE_CASE`).
- **Do not re-dig measured-dead perf levers** (command-buffer reuse: tried, −23% bs8, reverted —
  see `apple-silicon/references/dispatch-overlap-and-perf-model.md`).
- Pin the install prefix per run; parallel worktrees must NOT share one prefix.
- Backward compat (model format + public API) is a guarantee — don't break it.
- **Git: forks/worktrees/local commits are expected and fine. NEVER push.** No `git push`,
  no remote operations, no PR creation — all work stays local for review.
