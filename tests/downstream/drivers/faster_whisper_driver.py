#!/usr/bin/env python
"""faster-whisper consumer driver: sequential beam-search transcription on
Device::METAL (same fixed clip as the whisperX driver, different code path)."""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compute-type", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default="small")
    ap.add_argument("--audio", required=True)
    args = ap.parse_args()

    from faster_whisper import WhisperModel

    model = WhisperModel(args.model, device="metal", compute_type=args.compute_type)
    segments, info = model.transcribe(args.audio, beam_size=5, language="en")
    segs = [
        {"start": s.start, "end": s.end, "text": s.text.strip()} for s in segments
    ]
    text = " ".join(s["text"] for s in segs).strip()

    with open(args.output, "w") as f:
        json.dump(
            {
                "consumer": "faster_whisper",
                "compute_type": args.compute_type,
                "device": "metal",
                "model": args.model,
                "audio": args.audio,
                "duration": info.duration,
                "text": text,
                "segments": segs,
            },
            f,
            indent=2,
        )
    print(f"faster_whisper[{args.compute_type}]: {len(text)} chars")


if __name__ == "__main__":
    main()
