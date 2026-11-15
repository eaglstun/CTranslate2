---
topic_id: "v2:BIFA"
topic_path: "ct2-internals/position-encodings"
semantic_id: "JQryQhiEHqS8Y92JQkQ1aDMm5nFv8AAO"
related_ids:
  - "IVLTyoiEBuS8Iw2IWE5tzLMxpznH4AAL"
  - "AwLfy4gBBu48O87IcGdtrDUrrjFDMAAJ"
---
# CTranslate2 internals references

Deep-dive, source-cited notes on the CTranslate2 engine's **device/dtype-agnostic architecture** —
ops & dispatch, `StorageView`, transformer block wiring, the specs/converters import pipeline, and
the CUDA backend structure. Each file cites the real source files (with file:line) it was built
from; **line numbers drift — re-grep the symbol, not the line, before acting on a citation.**

These were moved here from the `ct2-internals` skill's local `references/`. The **annotated index**
(a one-line "read when…" pointer per file) lives in that skill: `skills/ct2-internals/SKILL.md`.
`scripts/audit-citations.sh` there sanity-checks these citations (its `REF_DIR` points here).

Siblings: the Apple **Metal** GPU backend is `apple-silicon` (`references/apple-silicon/`); NVIDIA
**CUDA** backend doc research is the `cuda-references` agent (`references/cuda/`). Rule of thumb —
"how does CT2 do X" is here; "how do I make X run on the GPU" is the backend shelves.

## Conventions (moved from the skill's SKILL.md)

Relative paths below are written from the skill directory (`skills/ct2-internals/`); from
there this shelf is `../../references/ct2-internals/`.

- Each reference cites the source files it was built from (top of file) with real
  file:line references, and ends with a brief `### Relevance to the Metal backend`
  bridge to the `apple-silicon` skill where the two intersect.
- Keep SKILL.md lean: one-line pointers only. Detail lives in the shared
  `references/ct2-internals/` (reached as `../../references/ct2-internals/` from this skill).
- **Line numbers drift.** These cite a snapshot; re-grep the symbol (not the line) before
  acting on a citation. Prefer quoting a function/macro name the reader can find.
- To add a reference: read the actual code, cite file:line, stay device-agnostic (push
  backend specifics to `apple-silicon`), add a one-line pointer above.
- **Before trusting any `file:line` here, run `bash scripts/audit-citations.sh`** (`-q`
  for problems-only). It flags missing files, out-of-range lines, and ambiguous basenames;
  it CANNOT see content drift (a line that moved a few rows), so it prints each cited line
  for a fast eyeball. A "verified on DATE" note is worthless once the file is touched again —
  only a fresh run counts. (`transformer.cc`'s citations silently drifted ~8 lines this way.)
