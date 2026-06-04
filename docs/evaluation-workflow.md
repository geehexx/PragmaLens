# Evaluation Workflow

PragmaLens keeps evaluation intentionally layered so contract tests, benchmark
runs, and live smoke checks each answer a different question.

## Test Layers

- Contract tests validate schema, serialization, and model invariants.
- Evaluation tests validate the benchmark corpus, approval metadata, golden-set
  refreshes, and CLI wiring.
- Integration tests validate the report, trace, and manifest artifacts emitted
  by the pipeline.
- Live smoke tests validate optional runtime backends only when the required
  local models and dependencies are installed.
- Default CI live-verifier coverage currently exercises MiniCheck only; the
  CrossEncoder live-smoke path remains manual because hosted model downloads
  are not reliably reproducible on the shared runner. The loader is pinned to
  revision `f2f24f9fce8fc5b34aedf861f5c819c6ba0cf4f5` and caches under
  `.local_state/crossencoder-cache` by default.

## Current Baselines

- Approved benchmark corpus: `ragtruth-15592-mini`
- Approved benchmark size: 2 cases
- Approved semantic golden set: `pragmalens-golden-2026-06`
- Approved semantic golden set size: 2 cases
- Stage timing traces are emitted in `trace/stage_timings.json` and the total
  runtime is recorded in `trace/trace_manifest.json`.

For current outcomes and the latest operational summary, see
[docs/evaluation-results.md](evaluation-results.md).

These baselines are intentionally small so that they stay reviewable and can be
used as durable regression gates rather than as a substitute for large-scale
model evaluation.

## What the Current Tests Guarantee

- Corpus benchmarks refuse unapproved or mismatched corpus metadata.
- The golden set keeps promise, review commitment, antecedent resolution, and
  team-membership ambiguity explicit.
- Span contracts reject inverted or zero-width spans.
- Report and trace artifacts keep the selected backend, timings, and manifest
  paths aligned.

## Property-Based Coverage

The test suite now uses property-based checks where the contract is simple but
the input space is broad:

- `SpanRef` round-trips across many valid offset/text combinations.
- Golden-set annotations round-trip across many valid label/resolution values.
- Corpus benchmark gating rejects any non-approved metadata status and any
  corpus-id mismatch.

This replaces brittle one-off examples with assertions that better represent
the actual invariants.

## Research Notes for Future Corpus Expansion

The current corpus and golden set are sufficient for the present tranche. If a
future tranche broadens evaluation coverage, the main candidates to compare are:

- FEVER for general fact verification with supported/refuted/NEI labels.
- FEVEROUS for claim verification with mixed structured and unstructured
  evidence.
- SciFact for scientific claim verification with evidence-backed labels.
- RAGTruth for retrieval-augmented hallucination analysis and targeted
  regression slices.

Any new corpus should be added only with explicit license/provenance review and
an approval sidecar that matches the checked-in fixture.

For the extraction-specific tuning plan and manual verification matrix, see
[docs/extraction-tuning-plan.md](extraction-tuning-plan.md).

## Refresh Commands

```bash
uv run python scripts/refresh_captured_assets.py benchmark-fixtures
uv run python scripts/refresh_captured_assets.py golden-set
uv run pytest -q tests/unit/evaluation/test_benchmark.py tests/unit/evaluation/test_golden_set.py
uv run pytest -q tests/integration/offline/test_cli.py tests/integration/offline/test_report_artifacts.py
```
