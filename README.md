# PragmaLens v0.1 (PR-01..PR-04 Bootstrap)

PragmaLens is a Python natural-language evidence and discourse audit engine bootstrap, aligned to the v5 implementation package.

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

## Quickstart

```bash
uv venv -p python3.12 .venv
source .venv/bin/activate
uv sync
uv run pragmalens schema export --out schemas/pragmalens_report.schema.json --model report
uv run pragmalens run --input /path/to/input.md --report-out out/report.json --manifest-out out/manifest.json
```

## CLI

- `pragmalens schema export --out <path> --model report|manifest|span_ref`
- `pragmalens run --input <markdown_or_txt> --report-out <path> --manifest-out <path> [--trace-dir trace] [--profile default]`

## Quality Gates

```bash
uv lock --check
uv sync --frozen
uv run ruff check .
uv run ruff format --check .
uv run mypy pragmalens tests
uv run pytest -q -m "not live_smoke"
```

## Hook Setup

```bash
uv run lefthook install
```

## Security Scan

- Local binary (if installed):

```bash
gitleaks dir . --config .gitleaks.toml
gitleaks git --config .gitleaks.toml
```

- CI uses `.github/workflows/ci.yml` and `.gitleaks.toml`.

## Offline-first and live smoke separation

- Default tests are offline and use captured fixtures for LangExtract and GLiNER2.
- Live smoke tests are marked `live_smoke` and skipped by default.
- MiniCheck primary target install path (live smoke only):

```bash
pip install "minicheck @ git+https://github.com/Liyan06/MiniCheck.git@main"
```

MiniCheck is not executed in default CI/offline tests.
