# PragmaLens v0.1

PragmaLens is a Python natural-language evidence and discourse audit engine, aligned to the v5 implementation package. Current code is a v0.1 scaffold with a deterministic offline default verifier, captured fixture lanes for extraction, and optional live smoke lanes for extractor and verifier runtimes.

## v0.1 Scope Constraints

- Input: Markdown (`.md`) and plain text (`.txt`) only.
- No PII core in v0.1.
- No public plugin system in v0.1.
- No PDF/DOCX parsing in v0.1.
- No web retrieval/external truth lookup in v0.1.

## Stage Graph (Current)

- `normalize_document`
- `segment_and_index_spans`
- `spacy_substrate`
- `langextract_discourse` (captured fixture mode for offline CI)
- `gliner2_candidates` (captured fixture mode for offline CI)
- `evidence_normalizer`
- `verify_claims` (selected verifier verdict contract, with optional same-batch comparison artifact)
- `synthesize_findings`
- `render_reports_and_traces`

## Quickstart

```bash
uv venv -p python3.12 .venv
source .venv/bin/activate
uv sync
uv run pragmalens schema export --out schemas/pragmalens_report.schema.json --model report
uv run pragmalens run --input /path/to/input.md --report-out out/report.json --report-md-out out/report.md --manifest-out out/run_manifest.json --verifier-backend offline
```

## CLI

- `pragmalens schema export --out <path> --model report|manifest|profile|evidence_candidate|span_ref|verification_score|verification_verdict|verifier_batch_comparison|verifier_calibration_profile`
- `pragmalens run --input <markdown_or_txt> --report-out <path> [--report-md-out <path>] --manifest-out <path> [--trace-dir trace] [--profile default] [--verifier-backend offline|minicheck|crossencoder_nli]`

## Quality Gates

```bash
uv sync --frozen
uv run nox -s repo-layout lint types offline-verify build
```

## Pull Requests

Use [`.github/pull_request_template.md`](.github/pull_request_template.md) for new
PRs. Keep the body scoped to the branch diff, list the exact verification
commands you ran, and summarize any automated-review feedback from CodeRabbit,
GitHub Copilot, or Qodo when available.

Coverage is a real gate. The configured report must stay at or above 90%.
Docstring coverage is also a real gate. The configured report must stay at or above 88%.
Import hygiene is a real gate. Ruff `PLC0415` blocks nested imports, and `lint-imports`
keeps package boundaries explicit.

## Hook Setup

```bash
uv run lefthook install
```

RTK is optional and local-only. Product commands, hooks, and CI must not depend
on RTK being installed.

## Nox Sessions

```bash
uv run nox -l
uv run nox -s repo-layout lint types offline-verify build
uv run nox -s dev-fast
uv run nox -s dev-live
```

## Security Scan

- Local binary (if installed):

```bash
gitleaks dir . --config .gitleaks.toml
gitleaks git --config .gitleaks.toml
```

- CI uses `.github/workflows/ci.yml` and `.gitleaks.toml`.

## Offline-first and live smoke separation

- Default product runs and default tests are offline and use captured fixtures for LangExtract and GLiNER2.
- Live smoke tests are marked `live_smoke` and skipped by default.
- Live verifier backends are optional runtime selections. The product CLI can select them explicitly with `--verifier-backend` or through `PRAGMALENS_VERIFIER`.
- The verifier defaults to the offline baseline unless `--verifier-backend` or `PRAGMALENS_VERIFIER` selects MiniCheck or CrossEncoder NLI.
- The verification stage now supports a typed same-batch comparison surface for offline baseline, MiniCheck, and CrossEncoder NLI when a comparison harness is injected programmatically. The resulting artifact lands in `trace/verification.json` and, when enabled, in `report.verification_comparison`.
- Live smoke opt-in:

```bash
PRAGMALENS_ENABLE_LIVE_SMOKE=1 uv run pytest -q -m live_smoke
```

The live smoke lane expects the live dependencies and local model assets to be
installed; missing prerequisites fail the manual-live run rather than being
silently skipped.

- Local runtime bootstrap for the current live smoke lane:

```bash
uv run python -m spacy download en_core_web_sm
uv run python -m spacy validate
```

`uv sync` installs the minimal default development environment only. Live and QA tooling are opt-in so the default install stays cheap and offline-first.

- Default development install:

```bash
uv sync
```

- Fast impacted-local lane:

```bash
uv sync --group qa
uv run nox -s dev-fast
```

- Live smoke lane:

```bash
uv sync --group live
uv run nox -s dev-live
```

- Full local development with both opt-in groups:

```bash
uv sync --group live --group qa
```

- LangExtract live smoke defaults to a local Ollama model when available.
  Current default: `qwen3.5:0.8b`
- Override live runtime models with environment variables:
  - `PRAGMALENS_SPACY_MODEL`
  - `PRAGMALENS_GLINER2_MODEL`
  - `PRAGMALENS_LANGEXTRACT_PROVIDER=ollama|gemini|auto`
  - `PRAGMALENS_LANGEXTRACT_MODEL`
  - `PRAGMALENS_OLLAMA_URL`
  - `PRAGMALENS_VERIFIER=offline|minicheck|crossencoder_nli`
  - `PRAGMALENS_MINICHECK_MODEL`
  - `PRAGMALENS_MINICHECK_CACHE_DIR`
  - `PRAGMALENS_CROSSENCODER_MODEL`

Live/model tests remain outside default CI. Default CI stays offline and deterministic.

## License

PragmaLens is released under the Apache License 2.0. See [LICENSE](LICENSE).
