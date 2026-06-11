#!/usr/bin/env python
"""NLLB-200-distilled-600M-int8 consumer driver: fixed eng_Latn -> fra_Latn
sentence translated on Device::METAL (encoder-decoder architecture breadth)."""
import argparse
import json

SENTENCE = (
    "The quick brown fox jumps over the lazy dog, and the sleepy farmer "
    "never even notices the commotion in his field."
)
SRC_LANG = "eng_Latn"
TGT_LANG = "fra_Latn"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compute-type", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model-dir", required=True)
    args = ap.parse_args()

    import ctranslate2
    import transformers

    translator = ctranslate2.Translator(
        args.model_dir, device="metal", compute_type=args.compute_type
    )
    tok = transformers.AutoTokenizer.from_pretrained(
        "facebook/nllb-200-distilled-600M", src_lang=SRC_LANG
    )
    source = tok.convert_ids_to_tokens(tok(SENTENCE).input_ids)
    res = translator.translate_batch(
        [source], target_prefix=[[TGT_LANG]], beam_size=4
    )
    target_tokens = res[0].hypotheses[0]
    if target_tokens and target_tokens[0] == TGT_LANG:
        target_tokens = target_tokens[1:]
    text = tok.decode(
        tok.convert_tokens_to_ids(target_tokens), skip_special_tokens=True
    ).strip()

    with open(args.output, "w") as f:
        json.dump(
            {
                "consumer": "nllb",
                "compute_type": args.compute_type,
                "device": "metal",
                "model_dir": args.model_dir,
                "source": SENTENCE,
                "src_lang": SRC_LANG,
                "tgt_lang": TGT_LANG,
                "text": text,
            },
            f,
            indent=2,
        )
    print(f"nllb[{args.compute_type}]: {text}")


if __name__ == "__main__":
    main()
