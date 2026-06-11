# Downstream validation harness

End-to-end check that the Metal int8 path produces sane output in the four real
consumers, per `INT8_METAL_PLAN.md`. The C++ op suite is the bit-tight oracle
(Metal int8 vs CPU int8); this harness is the loose oracle — Metal int8 vs an
**fp16-on-Metal golden** with a quant-error tolerance. It catches garbage, not
ULP drift.

## Run

```bash
scripts/validate-downstream.sh --capture-goldens   # record fp16 goldens (same build)
scripts/validate-downstream.sh                     # run int8, diff vs goldens
```

The harness: builds the lib from this worktree → `cmake --install` to the
pinned prefix in `projects.json` (`~/.local/ct2-metal-downstream`) → rebuilds
the wheel (`CTRANSLATE2_ROOT=<prefix>`) → force-reinstalls into each consumer
venv (`uv pip`, + `install_name_tool -add_rpath`) → runs each consumer's
canonical job → compares vs golden. `--skip-build` / `--skip-install` /
`--only NAME` speed up iteration.

## Consumers, metrics, tolerances

| consumer       | job                                        | metric                                   | pass when |
| -------------- | ------------------------------------------ | ---------------------------------------- | --------- |
| whisperx       | batched VAD transcription, 30s clip, small | WER vs fp16 golden                       | ≤ 0.10    |
| faster_whisper | beam-5 sequential transcription, same clip | WER vs fp16 golden                       | ≤ 0.10    |
| qwen2.5        | 5 prompts × 20 steps, Qwen2.5-0.5B-int8    | teacher-forced next-token agreement      | ≥ 0.90    |
| nllb           | fixed eng→fra sentence, NLLB-600M-int8     | char similarity (difflib) vs fp16 golden | ≥ 0.90    |

Tolerance rationale: int8 symmetric per-row quantization perturbs logits
enough to flip occasional argmax decisions (Phase 2 measured 92/100 Qwen
agreement); the tolerances bound that expected drift while still failing hard
on a broken kernel (garbage output scores ~0 on every metric).

Whisper models auto-download (`Systran/faster-whisper-small`, cached);
`compute_type="int8"` quantizes the fp16 weights at load. The Qwen/NLLB models
are pre-converted int8 CT2 models in `~/Documents/AI/ct2-models/`; their fp16
golden runs dequantize at load (`compute_type="float16"`), so the Qwen golden
is "int8-converted weights, fp16 math" — the diff isolates the Metal int8
_kernels_, not the conversion.

Goldens live in `goldens/` (committed); per-run candidates and verdicts in
`results/`.
