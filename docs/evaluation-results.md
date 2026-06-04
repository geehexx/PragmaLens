# Evaluation Results

This page summarizes the current evaluation state after the latest hardening
passes. It is intentionally short and factual so humans can compare the
documented process with the current artifacts.

## Current Baselines

- Benchmark corpus: `ragtruth-15592-mini`
- Benchmark size: 2 cases
- Semantic golden set: `pragmalens-golden-2026-06`
- Semantic golden set size: 2 cases
- CI live-verifier lane: MiniCheck only
- Manual live smoke lane: MiniCheck and CrossEncoder, opt-in only

## What the Current Checks Cover

- Schema and contract export parity.
- Benchmark corpus approval gating.
- Golden-set ambiguity coverage for promise, review commitment, antecedent
  resolution, and team-membership scope.
- Stage timing traces and manifest generation.
- Live runtime availability for the opt-in smoke lane.

## Property-Based Coverage Added in This Tranche

- Span offset and text-length round-trips.
- Benchmark approval and corpus-id gating.
- Golden-set annotation round-trips.
- Verifier calibration case loading.
- Verifier calibration recommendation behavior for evidence-backed examples.

## Current Operational Findings

- The shared CI runner does not reliably fetch the CrossEncoder model from
  Hugging Face, so that path stays manual. The live loader is pinned to
  revision `f2f24f9fce8fc5b34aedf861f5c819c6ba0cf4f5` and caches under
  `.local_state/crossencoder-cache` by default.
- The CPU-safe CI verifier lane is stable when it only exercises MiniCheck.
- The offline verification lane and lane-alignment gate are green on the latest
  branch state.

## Research Notes for Future Expansion

If the next tranche broadens evaluation beyond the current tiny baselines, the
first external corpora to compare are FEVER, FEVEROUS, SciFact, and a larger
RAGTruth slice. Each needs explicit provenance and approval metadata before it
is promoted into the tracked fixtures.
