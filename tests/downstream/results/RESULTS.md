# Downstream validation — int8-on-Metal vs fp16 goldens

**Run:** 2026-06-11, M4 Max (64 GB, macOS 26.4.1), branch `fable/int8-metal`
(harness at 46656f80 + the conv-weight load fix committed with this file).
Single full-harness run (`scripts/validate-downstream.sh --skip-build
--skip-install` after build+install+wheel+reinstall in the same session);
goldens are fp16-on-Metal from the **same build**. Prefix
`~/.local/ct2-metal-downstream`, wheel
`ctranslate2-4.8.0-cp310-cp310-macosx_11_0_arm64.whl`.

## Verdict: 4/4 PASS

| consumer       | job                                        | metric                   | int8 value | tolerance | pass |
| -------------- | ------------------------------------------ | ------------------------ | ---------- | --------- | ---- |
| whisperx       | batched VAD transcription, 30s clip, small | WER vs fp16              | **0.000**  | ≤ 0.10    | ✅   |
| faster_whisper | beam-5 transcription, same clip            | WER vs fp16              | **0.071**  | ≤ 0.10    | ✅   |
| qwen2.5        | 5 prompts × 20 steps, Qwen2.5-0.5B-int8    | teacher-forced agreement | **0.900**  | ≥ 0.90    | ✅   |
| nllb           | eng→fra fixed sentence, NLLB-600M-int8     | char similarity vs fp16  | **1.000**  | ≥ 0.90    | ✅   |

Raw verdicts in `verdicts.jsonl`; per-consumer outputs in `*.int8.json`;
goldens in `../goldens/`.

## Detail

- **whisperx** — int8 transcript is word-identical to the fp16 golden
  (409 chars). The batched VAD path feeds 30s windows, so this run leans on
  the tiled int8 GEMM (prefill regime).
- **faster_whisper** — 6 word-edits over 85 golden words (0.0706): int8 drops
  conversational filler ("like", "and so you know") and inserts one "and".
  Content is semantically identical; classic beam-search divergence from
  quantization-shifted logits.
- **qwen2.5** — 90/100 teacher-forced next-token matches
  (per-prompt 19, 16, 19, 19, 17/20), consistent with the 92/100 measured at
  Phase 2 with a different prompt set. Exactly at the ≥0.90 baseline.
- **nllb** — int8 French output is byte-identical to fp16:
  "Le renard brun rapide saute sur le chien paresseux, et l'agriculteur
  endormi ne remarque même pas le tumulte de son champ."

## Bug the harness caught (fixed in this commit)

Whisper int8 on Metal crashed at load:
`ValueError: Conversion from int8 to float32 is not yet implemented`
(`StorageView::to`, reached from `model.encode`). Cause: once
`get_supported_compute_types("metal")` includes int8, model loading quantizes
**conv** weights to int8 too — but Metal has no quantized convolution (Conv1D
runs via the CPU reference, which has no int8 conv on this MKL-less build).
CUDA and DNNL already special-case this in `src/models/model.cc`
(`update_weight`); the fix adds `Device::METAL` to the same guard, so conv
weights stay in `float_dtype` while Dense weights stay int8-resident —
exactly what CUDA does for int8 Whisper. Qwen/NLLB (no conv) were unaffected.

Regression check after the fix: gtest `--gtest_filter='*METAL*'` 73 passed /
2 skipped / 0 failed (skips = the known Conv1D dilation + grouped-quantized
CPU-reference gaps); `--gtest_filter='*Metal*'` 22/22.
