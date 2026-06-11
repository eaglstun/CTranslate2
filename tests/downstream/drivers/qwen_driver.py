#!/usr/bin/env python
"""Qwen2.5-0.5B-int8 consumer driver (Phase-2 methodology: 5 prompts x 20 steps).

Golden capture (fp16, no --golden): greedy-generate 20 tokens per prompt and
record the token ids — that sequence IS the golden.

Validation (int8, --golden): teacher-forced next-token agreement — at every
step t the model sees prompt + golden[:t] and its greedy next token is compared
against golden[t]. The agreement rate is computed here (it needs the model),
so this driver emits the metric value directly; compare.py just thresholds it.
"""
import argparse
import json

PROMPTS = [
    "The capital of France is",
    "In a distant future, humanity has",
    "The chemical symbol for gold is",
    "To reverse a linked list in C, you",
    "Once upon a time, there was a",
]
STEPS = 20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compute-type", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--golden", default=None)
    args = ap.parse_args()

    import ctranslate2
    import transformers

    gen = ctranslate2.Generator(
        args.model_dir, device="metal", compute_type=args.compute_type
    )
    tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    prompt_tokens = [
        tok.convert_ids_to_tokens(tok(p).input_ids) for p in PROMPTS
    ]

    out = {
        "consumer": "qwen2.5",
        "compute_type": args.compute_type,
        "device": "metal",
        "model_dir": args.model_dir,
        "prompts": PROMPTS,
        "steps": STEPS,
    }

    if args.golden is None:
        # Golden capture: greedy continuation, STEPS new tokens per prompt.
        res = gen.generate_batch(
            prompt_tokens,
            max_length=STEPS,
            sampling_topk=1,
            include_prompt_in_result=False,
        )
        out["golden_tokens"] = [r.sequences_ids[0][:STEPS] for r in res]
        out["golden_texts"] = [tok.decode(ids) for ids in out["golden_tokens"]]
        print(f"qwen2.5[{args.compute_type}]: captured "
              f"{sum(len(t) for t in out['golden_tokens'])} golden tokens")
    else:
        with open(args.golden) as f:
            golden = json.load(f)["golden_tokens"]
        matches = 0
        total = 0
        per_prompt = []
        predicted = [[] for _ in PROMPTS]
        for t in range(STEPS):
            batch = [
                pt + tok.convert_ids_to_tokens(g[:t])
                for pt, g in zip(prompt_tokens, golden)
                if t < len(g)
            ]
            res = gen.generate_batch(
                batch,
                max_length=1,
                sampling_topk=1,
                include_prompt_in_result=False,
            )
            bi = 0
            for pi, g in enumerate(golden):
                if t >= len(g):
                    continue
                pred = res[bi].sequences_ids[0][0]
                predicted[pi].append(pred)
                matches += int(pred == g[t])
                total += 1
                bi += 1
        for pi, g in enumerate(golden):
            m = sum(int(p == gt) for p, gt in zip(predicted[pi], g))
            per_prompt.append({"prompt": PROMPTS[pi], "matches": m, "steps": len(g)})
        out["agreement"] = matches / total if total else 0.0
        out["matches"] = matches
        out["total"] = total
        out["per_prompt"] = per_prompt
        out["predicted_tokens"] = predicted
        print(f"qwen2.5[{args.compute_type}]: teacher-forced agreement "
              f"{matches}/{total} = {out['agreement']:.3f}")

    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
