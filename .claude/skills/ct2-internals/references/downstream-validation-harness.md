# The Downstream Validation Harness — Real Consumers as the Loose Oracle

End-to-end validation against the four real consumers of this library (whisperX,
faster-whisper, and direct `Generator`/`Translator` use). The C++ gtest suite
(`ops-test-suite-structure.md`) is the **tight** oracle — bit/eps-level op parity; this
rig is the **loose** one — "does a real app still produce sane output through the
installed wheel" with a quant-error-sized tolerance. It catches garbage, not ULP drift
(`tests/downstream/README.md:4-7`).

**Sources (read these, all citations below are from real lines):**

- `scripts/validate-downstream.sh`, `tests/downstream/README.md`, `tests/downstream/projects.json`
- `tests/downstream/compare.py`, the four `tests/downstream/drivers/*_driver.py`
- `tests/downstream/results/RESULTS.md` (the 2026-06-11 int8 run)

---

## 1. Architecture: couple through the INSTALLED library, never source colocation

The pipeline (`scripts/validate-downstream.sh:4-7`): build the C++ lib from THIS
worktree → `cmake --install` to a pinned prefix → rebuild the Python wheel with
`CTRANSLATE2_ROOT=$PREFIX` (`:50-57`) → `uv pip install --force-reinstall` the wheel
into each consumer venv (`:88`) → run each consumer's canonical job → diff vs golden
(`:138-139`). Consumers never see the source tree — they exercise exactly what a user
installs, including the wheel-link rule from `python-bindings-architecture.md`.

Two modes: `--capture-goldens` forces `COMPUTE=float16` and writes driver output straight
to `goldens/` (`:32`, `:114-115`); plain invocation runs `COMPUTE=int8` candidates into
`results/` and compares (`:116-124`). `--skip-build` / `--skip-install` / `--only NAME` /
`--compute-type CT` speed up iteration (`:15`). Verdicts accumulate in
`results/verdicts.jsonl` (`:96-97`); exit status is the fail count flag (`:152`).

**Why the prefix is pinned and dedicated** (`~/.local/ct2-metal-downstream`,
`tests/downstream/projects.json:3`): the wheel's extension resolves `libctranslate2`
through an rpath into `$PREFIX/lib` patched in post-install — `fix_rpath` runs
`install_name_tool -add_rpath` + ad-hoc `codesign` on `_ext*.so` in each venv
(`scripts/validate-downstream.sh:64-74`). The venvs load the dylib from that prefix _at
runtime_, so a parallel worktree installing over the same prefix would silently swap the
library under every consumer venv mid-experiment. One worktree's harness, one prefix.

## 2. `projects.json` schema and the four consumers

Top level: `prefix`, `wheel_python` (interpreter that builds the wheel), `venvs`
(name → path), `bootstrap` (auto-create a missing venv: `uv venv --python` + editable
install of the consumer source, `scripts/validate-downstream.sh:79-86`), `consumers`.
Each consumer entry: `name`, `venv`, `driver`, `args`, `metric`, `tolerance`, `golden`,
optional `pass_golden` (`projects.json:15-53`). Only qwen2.5 sets `pass_golden: true`
(`:42`) — its driver needs the golden tokens at run time (teacher forcing), not just at
compare time (`scripts/validate-downstream.sh:117`).

| consumer       | venv                             | exercises                                                                                                                                                                                                                                                                         |
| -------------- | -------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| whisperx       | `~/Documents/AI/whisperX/.venv`  | conv stem + encoder, batched VAD-segmented inference (prefill-heavy). CT2 `WhisperModel` built on `device="metal"` and injected via `model=` because whisperX's torch-side VAD rejects "metal" (`tests/downstream/drivers/whisperx_driver.py:3-9,25-32`), `batch_size=8` (`:34`). |
| faster_whisper | own venv (bootstrapped editable) | same clip, sequential beam-5 decode (`tests/downstream/drivers/faster_whisper_driver.py:18-19`) — the enc-dec decode-loop path.                                                                                                                                                   |
| qwen2.5        | reuses whisperx venv             | decoder-only `Generator`, Qwen2.5-0.5B-int8, 5 prompts × 20 greedy steps (`tests/downstream/drivers/qwen_driver.py:15-22`).                                                                                                                                                       |
| nllb           | reuses whisperx venv             | encoder-decoder `Translator`, NLLB-600M-int8, one fixed eng_Latn→fra_Latn sentence, beam 4 with `target_prefix` (`tests/downstream/drivers/nllb_driver.py:7-12,32-34`).                                                                                                           |

## 3. Tolerance semantics (`tests/downstream/compare.py`) and the golden rule

Three metrics, one stdlib-only comparator that prints a one-line JSON verdict and exits
0/1 (`tests/downstream/compare.py:4-12`):

- **wer** (whisperx ≤0.10, faster_whisper ≤0.10) — word error rate via Levenshtein on
  lowercased, punctuation-stripped word lists (`compare.py:22-39`, pass at `:56-58`).
- **agreement** (qwen2.5 ≥0.90) — teacher-forced next-token match rate. Computed by the
  _driver_ (it needs the model: at every step t the model sees prompt + golden[:t] and
  its greedy token is compared to golden[t], `tests/downstream/drivers/qwen_driver.py:72-99`);
  `compare.py:59-61` just thresholds `cand["agreement"]`.
- **char_similarity** (nllb ≥0.90) — `difflib.SequenceMatcher` ratio on output text
  (`compare.py:62-66`).

**Golden-capture rule:** goldens are **fp16-on-Metal from the same build**, captured
BEFORE validating the change under test — NOT bit-exact int8 replays. int8 symmetric
per-row quantization legitimately flips occasional argmax decisions; the tolerances
absorb that while a broken kernel scores ~0 on every metric
(`tests/downstream/README.md:32-35`). Subtlety: Qwen/NLLB are pre-converted int8 CT2
models, so their fp16 golden runs _dequantize-at-load_ — the diff isolates the Metal
int8 **kernels**, not the conversion (`README.md:37-43`). Whisper models auto-download
(`Systran/faster-whisper-small`) and quantize fp16→int8 at load.

## 4. Operational gotchas (from README/RESULTS, learned the hard way)

- `pip` inside these venvs is bypassed — installs go through
  `uv pip install --python <venv>/bin/python` (`scripts/validate-downstream.sh:85-88`).
- The wheel does NOT embed the library; without the `fix_rpath` `install_name_tool`
  step (+ re-codesign) the extension fails to find `libctranslate2.dylib` (`:64-74`).
  Sanity check after install: import + `get_supported_compute_types('metal')` (`:90`).
- Venvs are reused across consumers (qwen + nllb piggyback on whisperx,
  `projects.json:37,46`); only faster_whisper has a `bootstrap` entry (`:9-14`).
- Model paths are machine-pinned: clip at `~/Documents/AI/whisperX/audio/clip30.wav`,
  int8 models in `~/Documents/AI/ct2-models/` (`projects.json:20,38,48`).
- Goldens are **committed** (`tests/downstream/goldens/`); `results/` is per-run scratch.

## 5. Worked example: the 2026-06-11 int8 run — 4/4 PASS

From `tests/downstream/results/RESULTS.md` (M4 Max, branch `fable/int8-metal`, single
full run, goldens fp16-on-Metal from the same build):

| consumer       | metric                   | int8 value | tolerance |
| -------------- | ------------------------ | ---------- | --------- |
| whisperx       | WER vs fp16              | **0.000**  | ≤ 0.10    |
| faster_whisper | WER vs fp16              | **0.071**  | ≤ 0.10    |
| qwen2.5        | teacher-forced agreement | **0.900**  | ≥ 0.90    |
| nllb           | char similarity vs fp16  | **1.000**  | ≥ 0.90    |

whisperx word-identical; faster_whisper 6 word-edits in 85 (filler dropped — classic
beam divergence from quantization-shifted logits); qwen 90/100 (per-prompt 19, 16, 19,
19, 17/20 — consistent with Phase 2's 92/100 on a different prompt set, and exactly at
the floor); nllb byte-identical French.

**The bug this rig caught** (RESULTS.md "Bug the harness caught"): Whisper int8 crashed
at load — once Metal advertises int8, model loading quantized **conv** weights too, but
Metal has no quantized convolution (Conv1D runs via the CPU reference). Fix: add
`Device::METAL` to the conv-float guard in `Model::set_compute_type`'s weight pass
(`src/models/model.cc:212-220`) so conv weights stay `float_dtype` while Dense stays
int8 — exactly the existing CUDA/DNNL behavior. Full story in
`weight-loading-and-conversion.md`; the gtest suite alone could not have caught it (it
has no int8 Whisper-on-Metal load).

---

### Relevance to the Metal backend

- This rig is what signed off the int8-Metal work end-to-end (4/4 green above) — and is
  the **graduation gate** for any future backend change that survives the op suite:
  tight oracle first (`ops-test-suite-structure.md`), then this.
- The conv-weight float-guard bug was found _here_, not in gtest — load-time
  quantization policy only bites when a real consumer loads a real conv model on Metal.
- Re-capture goldens (`--capture-goldens`, fp16) from the same build as the candidate —
  comparing across builds conflates the change under test with unrelated drift.
