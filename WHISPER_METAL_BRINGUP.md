# Whisper / faster-whisper bring-up on the Metal backend — findings

First end-to-end attempt to run **OpenAI Whisper (via faster-whisper / CTranslate2)** on
the `Device::METAL` backend. The translation transformer path is well covered by
`tests/metal_test.cc`, but Whisper exercises code the Metal suite never touches — most
importantly the **Conv1D audio encoder**. This is a field report of what happened, with
exact repro and root cause tied to source lines, so the gaps can go on the roadmap.

**TL;DR:** the bindings build and load cleanly, the Metal device is recognized, and
`get_supported_compute_types("metal")` returns `{float16, float32}`. But no Whisper
configuration is currently usable: **fp16 throws in Conv1D**, **fp32 large-v3 is SIGKILLed
during the encoder forward pass**, and **fp32 small is correct but ~5× slower than CPU**
and is SIGKILLed on long audio. All three map cleanly onto already-known roadmap gaps
(Conv1D is "Deferred / out of scope"; perf work is "after correctness").

> **UPDATE 2026-06-09 — failure #1 (fp16 Conv1D throw) is FIXED.** `src/ops/conv1d.cc` now
> detects fp16 on Metal and upcasts to fp32, runs the proven CPU-reference conv over unified
> memory, and downcasts back (the CPU reference is fp32-only). Covered by
> `MetalTest.Float16Conv1DMatchesFloat32`. This unblocks the fp16 encoder reaching Conv1D;
> the **SIGKILL (#2/#3) is still the open usability blocker** and needs the Python/audio rig
>
> - a Metal allocations profiler to confirm command-buffer accumulation vs a single oversized
>   staging alloc. (Item #4, CPU int8 unavailable in the MKL-less build, is also still open —
>   it surfaces as `CPU/OpDeviceFPTest.Conv1DGroupNoBiasQuantized` failing with "No INT8 GEMM
>   backend for CPU".)

## Environment

|                 |                                                                                                                                     |
| --------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| Machine         | Apple M4 Max, 64 GB unified memory                                                                                                  |
| OS              | macOS 26.4.1 (build 25E253)                                                                                                         |
| CTranslate2     | `v4.8.0-30-gab9b7569`, built `-DWITH_METAL=ON -DWITH_MKL=OFF -DWITH_ACCELERATE=ON -DOPENMP_RUNTIME=NONE -DCMAKE_BUILD_TYPE=Release` |
| Python bindings | built from `python/` against the above lib (`CTRANSLATE2_ROOT` + rpath to the staged `lib/`), Python 3.10.19                        |
| faster-whisper  | 1.2.0 (`ctranslate2<5,>=4.0`, so 4.8.0 satisfies the pin unchanged)                                                                 |
| Test audio      | 730 s (~12 min) mono speech; 30 s clip cut from it for crash isolation                                                              |

The bindings build is the upstream `python/setup.py` with no source changes. faster-whisper
passes the `device` string straight through to `ctranslate2.models.Whisper`, so `"metal"`
reaches `str_to_device` unmodified — no integration shim was needed to _select_ the device.

## What works

- **Bindings + device.** `import ctranslate2` loads the Metal dylib (via rpath), reports
  `4.8.0`, and `get_supported_compute_types("metal") == {'float16', 'float32'}`.
- **Model load on Metal.** `WhisperModel("large-v3", device="metal", compute_type="float32")`
  loads in ~3.9 s with no error. The crash is strictly in the forward pass, not load.
- **CPU reference path unchanged.** `device="cpu", compute_type="float32"` on this build
  matches the stock 4.6.0 wheel to within noise (small: 78 s vs 81 s on the 12-min file).

## What fails

### 1. fp16 — `Conv1D only supports float types` (hard throw)

```
WhisperModel("small", device="metal", compute_type="float16").transcribe(audio)
  → faster_whisper/transcribe.py … encode()
  → ValueError: Conv1D only supports float types
```

**Root cause (confirmed in source).** `src/ops/conv1d.cc:51` dispatches via
`DEVICE_AND_FLOAT_DISPATCH`. In `src/dispatch.h` the `float16`/`bfloat16` cases are
hardcoded to CUDA:

```cpp
// src/dispatch.h
case DataType::FLOAT16: {
    if (DEVICE != Device::CUDA)
        throw std::invalid_argument(NAME " only supports float types");
    constexpr Device D = Device::CUDA;
    ...
```

So a `float16` input on `Device::METAL` can never reach the (CPU-reference) Conv1D kernel —
it hits the `!= Device::CUDA` throw first. This is exactly the item the roadmap lists under
fp16 work: _"Optionally a `Device::METAL`-aware `DEVICE_AND_FLOAT_DISPATCH` in
`src/dispatch.h` (it hardcodes `Device::CUDA` for fp16/bf16)."_ Whisper makes it
non-optional: the encoder's first op is Conv1D, so **fp16 is unreachable for any model with
a conv stem (Whisper, Wav2Vec2)**, which is also the precision where Apple Silicon would
actually be fast.

### 2. fp32 large-v3 — SIGKILL during the encoder forward pass

```
WhisperModel("large-v3", device="metal", compute_type="float32")  → LOADED ok in 3.9s
  .transcribe(clip30.wav)  → prints "encoding..." → process killed (exit 137 / SIGKILL)
```

Dies inside the **encoder forward pass**, before the first segment is yielded, on a **30 s**
clip, on a 64 GB machine — so this is not weights-don't-fit OOM (large-v3 fp32 ≈ 3 GB). It
looks like a runaway allocation / unbounded buffer growth in the encode path. Candidates,
in roadmap terms:

- The encoder runs ~1500 frames through Conv1D + 32 attention/FFN blocks with **no
  batching at this layer**; if every `metal::` op commits its own command buffer and the
  shared-storage buffers are not released until a `flush()`, peak resident Metal memory can
  balloon over a long op chain. (The roadmap notes per-op async commit was deliberately kept
  over single-commit-per-step; the interaction with a very long encoder chain may be the
  issue.)
- Conv1D on Metal falls to the CPU reference over unified memory; the im2col/quantized
  staging buffers in `conv1d_cpu.cc` over a 1500-wide sequence at d_model=1280 may be the
  allocation that tips it over.

This one needs a contributor with a Metal memory profiler (Instruments → Allocations /
`MTLHeap` counters) to confirm whether it's command-buffer accumulation or a single bad
allocation. Happy to re-run any instrumented build.

### 3. fp32 small — correct but ~5× slower than CPU; SIGKILL on long audio

| input      | metal fp32                       | cpu fp32       |
| ---------- | -------------------------------- | -------------- |
| 30 s clip  | 16.6 s (1.8× RT), output correct | 3.x s          |
| 730 s file | **SIGKILL (exit 137)**           | 81 s (9.0× RT) |

`small` _completes correctly_ on the 30 s clip — so the Metal path is numerically fine — but
at **1.8× realtime vs CPU's ~9×**. Expected for a tiny model: decoder GEMMs are small enough
that MPS per-op dispatch overhead dominates, and the encoder Conv1D is on CPU anyway. On the
full 12-min file it accumulates into the same SIGKILL as #2, consistent with unbounded
growth over the decode loop rather than a fixed working set.

### 4. Build-config side effect — CPU int8 unavailable

`get_supported_compute_types("cpu")` on this build is `{'float32'}` only — no `int8`. This
is the `-DWITH_MKL=OFF` build (no INT8 GEMM backend), not a Metal bug, but worth flagging:
faster-whisper's default CPU mode is `int8`, so a Metal-enabled wheel built this way silently
loses the fast quantized CPU fallback. A note in `METAL_BACKEND.md`'s Build section ("Metal
builds are MKL-less → CPU int8 is unavailable; use float32 or build with an int8-capable CPU
GEMM") would save the next person the surprise.

## Benchmark numbers

Engine isolated — `WhisperModel.transcribe(beam_size=5)`, generator fully consumed, no
alignment/VAD/diarization. Same code path across all rows; only model/device/compute vary.

| ct2   | model    | device | compute | transcribe | speed  | note                                            |
| ----- | -------- | ------ | ------- | ---------- | ------ | ----------------------------------------------- |
| 4.6.0 | small    | cpu    | int8    | 68.2 s     | 10.7×  | stock wheel baseline                            |
| 4.6.0 | small    | cpu    | float32 | 81.1 s     | 9.0×   |                                                 |
| 4.6.0 | large-v3 | cpu    | int8    | 638.8 s    | 1.14×  |                                                 |
| 4.8.0 | small    | cpu    | float32 | 77.6 s     | 9.4×   | Metal build, CPU path — matches stock           |
| 4.8.0 | small    | metal  | float16 | —          | —      | ❌ ValueError: Conv1D only supports float types |
| 4.8.0 | small    | metal  | float32 | 16.6 s\*   | 1.8×\* | \*30 s clip; SIGKILL on full 12 min             |
| 4.8.0 | large-v3 | cpu    | float32 | 19.0 s\*   | 1.6×\* | \*30 s clip                                     |
| 4.8.0 | large-v3 | metal  | float32 | —          | —      | ❌ SIGKILL during encode (even 30 s)            |

## Reproduction

```bash
# 1. Build the bindings against a WITH_METAL=ON lib
cmake --install build --prefix ~/.local/ct2-metal
cd python
CTRANSLATE2_ROOT=~/.local/ct2-metal LDFLAGS="-Wl,-rpath,$HOME/.local/ct2-metal/lib" \
  pip install --no-build-isolation --force-reinstall --no-deps .

# 2. Sanity
python -c "import ctranslate2 as c; print(c.__version__, c.get_supported_compute_types('metal'))"
# -> 4.8.0 {'float16', 'float32'}

# 3. Drive Whisper (bench_ct2.py below)
python bench_ct2.py small    metal float16   # ValueError: Conv1D only supports float types
python bench_ct2.py large-v3 metal float32 clip30.wav   # SIGKILL during encode
python bench_ct2.py small    metal float32 clip30.wav   # works, ~1.8x RT
```

<details>
<summary><code>bench_ct2.py</code> (isolates the ASR engine; same path for every config)</summary>

```python
import sys, time
import ctranslate2
from faster_whisper import WhisperModel

model_size   = sys.argv[1] if len(sys.argv) > 1 else "small"
device       = sys.argv[2] if len(sys.argv) > 2 else "cpu"
compute_type = sys.argv[3] if len(sys.argv) > 3 else "int8"
audio        = sys.argv[4] if len(sys.argv) > 4 else "audio.wav"

print(f"ct2={ctranslate2.__version__} model={model_size} device={device} compute={compute_type}", flush=True)
t0 = time.perf_counter()
model = WhisperModel(model_size, device=device, compute_type=compute_type)
load_s = time.perf_counter() - t0
t1 = time.perf_counter()
segments, info = model.transcribe(audio, beam_size=5)
n_seg = sum(1 for _ in segments)          # generator is where the work happens
transcribe_s = time.perf_counter() - t1
print(f"load {load_s:.2f}s  transcribe {transcribe_s:.2f}s  "
      f"({info.duration/transcribe_s:.2f}x RT, {n_seg} segs)")
```

</details>

## Suggested priorities (from a Whisper consumer's seat)

1. **Make `DEVICE_AND_FLOAT_DISPATCH` Metal-aware for fp16** (or route Conv1D fp16 to the
   CPU-reference float path by upcasting). This single change unblocks the entire fp16
   encoder and is the highest-leverage item — fp16 is the point of the backend on Apple GPUs.
2. **Bound encoder memory** — confirm whether the SIGKILL is command-buffer accumulation
   (insert periodic `metal::flush()` / buffer reclamation across the encoder op chain) or a
   single oversized Conv1D staging allocation. Instruments Allocations trace would settle it.
3. **A GPU Conv1D kernel** (the deferred item) — until then the audio encoder is CPU-bound
   regardless of decoder speed, capping any Whisper win.
4. **Doc note** that Metal (MKL-less) builds drop CPU int8.

Until 1–3 land, faster-whisper should stay on the CPU/CUDA wheels; the Metal backend is a
strong correctness result for the translation transformer but not yet a Whisper accelerator.
Glad to retest instrumented builds on the M4 Max.
