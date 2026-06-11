#!/usr/bin/env python
"""whisperX consumer driver: transcribe a fixed clip on Device::METAL.

whisperX's load_model() builds its torch-side VAD with torch.device(device),
which rejects "metal" — so the CT2 WhisperModel is constructed separately on
Metal and injected via model=, while the pipeline (VAD etc.) stays on CPU.
This exercises whisperX's batched VAD-segmented inference path, distinct from
faster-whisper's sequential beam decode.
"""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compute-type", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default="small")
    ap.add_argument("--audio", required=True)
    args = ap.parse_args()

    import whisperx
    from whisperx.asr import WhisperModel  # whisperX's batched subclass

    wm = WhisperModel(args.model, device="metal", compute_type=args.compute_type)
    pipe = whisperx.load_model(
        args.model,
        device="cpu",
        compute_type=args.compute_type,
        model=wm,
        language="en",
    )
    audio = whisperx.load_audio(args.audio)
    result = pipe.transcribe(audio, batch_size=8)
    text = " ".join(s["text"].strip() for s in result["segments"]).strip()

    with open(args.output, "w") as f:
        json.dump(
            {
                "consumer": "whisperx",
                "compute_type": args.compute_type,
                "device": "metal",
                "model": args.model,
                "audio": args.audio,
                "text": text,
                "segments": [
                    {"start": s["start"], "end": s["end"], "text": s["text"].strip()}
                    for s in result["segments"]
                ],
            },
            f,
            indent=2,
        )
    print(f"whisperx[{args.compute_type}]: {len(text)} chars")


if __name__ == "__main__":
    main()
