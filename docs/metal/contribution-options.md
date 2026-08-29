# Metal Backend — Open Follow-ups

The fork-vs-upstream question is settled: the Metal backend lives as a **public fork**,
rebased on upstream, PR'd into the fork's own `main` (the architecturally correct home
for a separate GPU backend). What remains are two open, optional follow-ups — neither
blocks anything.

Status refreshed 2026-08-28 after M18. These remain optional non-engine follow-ups; the
ranked engineering backlog lives in [`METAL_NEXT_STEPS.md`](../../METAL_NEXT_STEPS.md).

## 1. Deep-dive writeup (portfolio value)

The perf-investigation arc is good content: profiling a real LLM and finding the `Add`
op had silently never been on the GPU (27× fp16 blowup); trying command-buffer reuse —
the "obvious" lever — and **measuring it neutral-to-negative** because batching kills
CPU/GPU overlap; then the M16 fused-attention and M17 Whisper wins where the real
culprits (per-batch-index MPS encodes; a CPU-reference `Transpose` draining the queue
~128×/token; a single-threaded gather) were nothing the priors predicted; then M18
showed why changing the KV-cache layout beat both attempted Concat micro-optimizations.
The through-line — _measure, don't guess_ — is a better story than a clean win.

Home is already scaffolded: `ai.ericeaglstun.com` has a `deep-dives/` Apple-Silicon
porting series (`~/Documents/web/ericeaglstun-ai/content/deep-dives/`:
`porting-ml-to-apple-silicon.md`, `reviving-pulse-apple-silicon.md`,
`audiocraft-apple-silicon.md`). The CTranslate2 Metal backend slots in as the next
entry and links straight into the existing glossary terms (`metal`, `mps`, `cuda`,
`tensor`).

## 2. Upstream Discussion (only if there's appetite)

If upstreaming is ever worth pursuing, open a GitHub **Discussion first — not a PR**:
"Working Metal backend, fp32/fp16/int8 end-to-end, parity with CPU on the op suite,
here's the design — is there appetite, and in what shape?" Costs an afternoon and tells
you whether a salami-sliced upstream attempt is even open. Per
[`CONTRIBUTING.md`](../../CONTRIBUTING.md) the work
must be disclosed as AI-assisted and every load-bearing choice defensible cold — the
unified-memory pointer contract, why METAL binds to the CPU dispatch case (avoiding ~50
dispatch-site instantiations), and the `allocator.cc` / `devices.cc` early-returns for
`Device::METAL`. All three are documented in
[`METAL_BACKEND.md`](../../METAL_BACKEND.md).
