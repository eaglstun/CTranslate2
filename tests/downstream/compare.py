#!/usr/bin/env python3
"""Compare a candidate run against a golden, per the consumer's metric.

Metrics:
  wer             — word error rate of candidate text vs golden text
                    (lowercased, punctuation stripped); pass if <= tolerance.
  agreement       — teacher-forced next-token match rate, computed by the
                    driver itself (needs the model); pass if >= tolerance.
  char_similarity — difflib.SequenceMatcher ratio on output text;
                    pass if >= tolerance.

Prints a one-line JSON verdict and exits 0 (pass) / 1 (fail).
Runs on any python3 — stdlib only.
"""
import argparse
import difflib
import json
import re
import sys


def normalize_words(text):
    return re.sub(r"[^\w\s']", "", text.lower()).split()


def wer(ref, hyp):
    """Word error rate via Levenshtein distance on word lists."""
    r, h = normalize_words(ref), normalize_words(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(
                d[j] + 1,
                d[j - 1] + 1,
                prev + (r[i - 1] != h[j - 1]),
            )
            prev, d[j] = d[j], cur
    return d[len(h)] / max(len(r), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", required=True,
                    choices=["wer", "agreement", "char_similarity"])
    ap.add_argument("--tolerance", type=float, required=True)
    ap.add_argument("--golden", required=True)
    ap.add_argument("--candidate", required=True)
    args = ap.parse_args()

    with open(args.golden) as f:
        golden = json.load(f)
    with open(args.candidate) as f:
        cand = json.load(f)

    if args.metric == "wer":
        value = wer(golden["text"], cand["text"])
        ok = value <= args.tolerance
    elif args.metric == "agreement":
        value = cand["agreement"]
        ok = value >= args.tolerance
    else:  # char_similarity
        value = difflib.SequenceMatcher(
            None, golden["text"], cand["text"]
        ).ratio()
        ok = value >= args.tolerance

    print(json.dumps({
        "consumer": cand.get("consumer"),
        "metric": args.metric,
        "value": round(value, 4),
        "tolerance": args.tolerance,
        "golden_compute_type": golden.get("compute_type"),
        "candidate_compute_type": cand.get("compute_type"),
        "pass": ok,
    }))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
