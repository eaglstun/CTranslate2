#!/usr/bin/env python3
"""Stamp 512-bit semantic IDs into the skill reference docs' frontmatter.

Gives each reference file under .claude/skills/*/references/ a `semantic_id` whose
bits ARE the meaning of the doc, so "find related" is an XOR + popcount over the
corpus with no vector database. This is a deliberately simple, self-contained
scheme — NOT the ~/.claude semantic-ids skill's 192-bit mean-centered format.

Scheme
------
  1. embed  "search_document: {title}\n\n{summary}"  with nomic-embed-text (Ollama)
  2. take the first 512 of nomic's 768 dims (nomic is Matryoshka-trained, so the
     leading dims are valid on their own)
  3. binarize by RAW SIGN: component > 0 -> 1 bit, else 0  (no mean-centering)
  4. pack 512 bits MSB-first into 64 bytes; base64url, no padding -> 86 chars
  5. write as a single-line quoted `semantic_id` scalar (prettier does not wrap
     quoted frontmatter scalars, so this is formatter-safe)

Deterministic and idempotent: same title+summary -> same ID; an existing
`semantic_id` line is replaced, not duplicated. Requires Ollama serving
nomic-embed-text on localhost:11434 and PyYAML. Standard library otherwise.

Note on raw sign: it leaves ~50 embedding dims consistently-signed across the
corpus ("dead bits"), and mean pairwise Hamming lands well under the ~256 chance
value, so the space is used inefficiently — but nearest-neighbour retrieval still
returns semantically correct neighbours. `--health` prints these numbers. If you
ever want crisper separation, switch step 3 to sign(v - frozen_mean); that needs a
frozen per-dimension mean stored to disk that must NEVER be recomputed on a grown
corpus (doing so silently invalidates every previously-issued ID).

Usage
-----
  python3 .claude/scripts/stamp_semantic_ids.py            # stamp all, print health
  python3 .claude/scripts/stamp_semantic_ids.py --health   # only report, do not write
  python3 .claude/scripts/stamp_semantic_ids.py path/to/one.md [more.md ...]
"""
import base64
import glob
import itertools
import json
import os
import sys
import urllib.request

import yaml

DIMS = 512
MODEL = "nomic-embed-text"
OLLAMA = "http://localhost:11434/api/embed"
# repo root = two levels up from this file (.claude/scripts/ -> repo)
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GLOB = os.path.join(REPO, ".claude/skills/*/references/*.md")


def embed(texts):
    out = []
    for i in range(0, len(texts), 32):
        chunk = texts[i:i + 32]
        body = json.dumps({"model": MODEL,
                           "input": ["search_document: " + t for t in chunk]}).encode()
        req = urllib.request.Request(OLLAMA, body, {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            out.extend(json.load(r)["embeddings"])
    return out


def read_frontmatter(raw, path):
    if not raw.startswith("---\n"):
        raise SystemExit("no frontmatter: " + path)
    end = raw.find("\n---\n", 4)
    if end == -1:
        raise SystemExit("unterminated frontmatter: " + path)
    return yaml.safe_load(raw[4:end]), end


def pack_id(vec):
    bits = [1 if v > 0 else 0 for v in vec[:DIMS]]
    if len(bits) < DIMS:
        raise SystemExit("embedding has %d dims, need >= %d" % (len(vec), DIMS))
    b = bytearray(DIMS // 8)
    for i, bit in enumerate(bits):
        if bit:
            b[i // 8] |= 1 << (7 - (i % 8))   # MSB-first within each byte
    return base64.urlsafe_b64encode(bytes(b)).decode().rstrip("="), bits


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    health_only = "--health" in sys.argv[1:]
    files = ([os.path.join(REPO, a) if not os.path.isabs(a) else a for a in args]
             if args else sorted(glob.glob(GLOB)))

    texts, raws, ends = [], [], []
    for f in files:
        raw = open(f, encoding="utf-8").read()
        fm, end = read_frontmatter(raw, f)
        texts.append(fm["title"] + "\n\n" + fm["summary"])
        raws.append(raw)
        ends.append(end)

    embs = embed(texts)
    all_bits = []
    for f, raw, end, e in zip(files, raws, ends, embs):
        sid, bits = pack_id(e)
        all_bits.append(bits)
        if health_only:
            continue
        head_lines = [ln for ln in raw[:end].split("\n")
                      if not ln.startswith("semantic_id:")]
        head_lines.append('semantic_id: "%s"' % sid)
        open(f, "w", encoding="utf-8").write("\n".join(head_lines) + raw[end:])

    n = len(all_bits)
    if n > 1:
        dead = sum(1 for j in range(DIMS)
                   if len({all_bits[i][j] for i in range(n)}) == 1)
        tot = cnt = 0
        for a, b in itertools.combinations(range(n), 2):
            tot += sum(all_bits[a][j] ^ all_bits[b][j] for j in range(DIMS))
            cnt += 1
        verb = "checked" if health_only else "stamped"
        print("%s %d files; dead_bits=%d/%d; mean_pair_distance=%.1f (chance=%d)"
              % (verb, n, dead, DIMS, tot / cnt, DIMS // 2))
    else:
        print(("checked" if health_only else "stamped") + " %d file(s)" % n)


if __name__ == "__main__":
    main()
