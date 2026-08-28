# ct2-internals skill

Deep-dive references for the CTranslate2 engine's device/dtype-agnostic architecture —
ops & dispatch, `StorageView`, transformer block structure, and the specs/converters
model-import pipeline.

Entry point is [SKILL.md](SKILL.md); references live in [references/](references/), each
sourced from the actual code with file:line citations. `CLAUDE.md` has the high-level
layer map; this skill is the level below it.

Sibling skill: `apple-silicon` holds the **Metal GPU backend** specifics (MSL kernels,
MPS, op graduation, numeric parity). Rule of thumb — "how does CT2 do X" lives here;
"how do I make X run on the GPU" lives there.

`scripts/` and `sources/` are reserved; empty for now.
