# Metal Backend — Where Does This Work Go?

Context: the Metal GPU backend (fp32/fp16 end-to-end, CPU-parity on the op suite) is
heavily AI-assisted. CTranslate2's `CONTRIBUTING.md` (lines 32–36) requires that any
contribution be (a) explicitly disclosed as AI-assisted, (b) fully understood by the
contributor, and (c) defensible line-by-line. A whole backend dropped as one PR is too
much surface for a maintainer to vet and will be declined regardless of authorship.

So the question isn't "AI or not" — it's "what's the highest-value home for this work."
Ranked options below.

## 1. Maintain it as a public fork (most likely the real answer)

Out-of-tree GPU backends are normal. Ship `CTranslate2-metal`, people on Apple Silicon
who want it build from the fork, keep it rebased on upstream. Zero gatekeeping, full
control of the roadmap, genuinely useful artifact. Cost: ongoing rebase/maintenance —
but the understanding burden was already ours. For a _separate_ backend this is often
the architecturally correct home, not a consolation prize.

## 2. Open a GitHub Discussion / issue FIRST (do this first — cheap, decides everything)

Not a PR. A message: "Working Metal backend, fp32/fp16 end-to-end, parity with CPU on
the op suite, here's the design — is there appetite for upstreaming, and in what shape?"
Costs an afternoon, tells us whether option #4 is even open. Maintainers respond very
differently to "can we talk about this" than to a surprise 50-file diff.

## 3. Contribute the pieces that DO fit the policy

Line 36 explicitly invites docs / examples / integrations:

- `METAL_BACKEND.md` design writeup
- Apple Silicon build-recipe docs
- benchmarks doc
  Low-risk, high-welcome, plants a flag, builds trust for a future backend conversation.

## 4. Salami-slice the upstream attempt (only if #2 says yes)

Don't submit the backend — submit the smallest self-contained piece defensible without
notes. E.g. the `device_dispatch.h` METAL→CPU binding trick: tight, clever, explainable.
Land trust in small bites. Slow path; only worth it with a maintainer's signal.

## 5. Write it up publicly

A blog post / technical writeup ("adding a Metal backend to CTranslate2 with
unified-memory tricks") is arguably more portfolio value than a PR buried in someone
else's repo. The per-op command-buffer-sync bottleneck story alone is good content.

This already has a home: ai.ericeaglstun.com has a `deep-dives/` section
(`~/Documents/web/ericeaglstun-ai/content/deep-dives/`) already running an Apple-Silicon
porting series — `porting-ml-to-apple-silicon.md`, `reviving-pulse-apple-silicon.md`,
`audiocraft-apple-silicon.md`. The CTranslate2 Metal backend slots in as the next entry
in that exact series and can link straight into the existing glossary terms (`metal`,
`mps`, `cuda`, `tensor`). The scaffolding is already standing — it's a deep dive with a
slot pre-cut for it, not a hypothetical.

## Recommendation

- Do **#2 this week** — cheap, unlocks everything else.
- Assume **#1 is where this lives** regardless. Fork = correct architecture for a
  separate backend.

## Non-negotiable if going upstream

Must be able to defend every load-bearing choice cold:

- the unified-memory pointer contract
- why METAL binds to the CPU dispatch case (avoiding ~50 dispatch-site instantiations)
- the `allocator.cc` / `devices.cc` early-returns for `Device::METAL`
  In this codebase a wrong pointer really does eat a weekend — that's why the rule exists,
  and it's not aimed at a contributor who can explain these.

## Next step

Draft the Discussion post for #2 when ready (a few days out).
