# PragmaLens v0.1

PragmaLens is a Python natural-language evidence and discourse audit engine, aligned to the v5 implementation package. Current code is a v0.1 scaffold with offline fixture lanes and a conservative verifier contract.

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
- `verify_claims` (offline baseline verdict contract)
- `synthesize_findings`
- `render_reports_and_traces`

## Quickstart

```bash
uv venv -p python3.12 .venv
source .venv/bin/activate
uv sync
uv run pragmalens schema export --out schemas/pragmalens_report.schema.json --model report
uv run pragmalens run --input /path/to/input.md --report-out out/report.json --report-md-out out/report.md --manifest-out out/run_manifest.json
```

## CLI

- `pragmalens schema export --out <path> --model report|manifest|profile|span_ref|verification_verdict`
- `pragmalens run --input <markdown_or_txt> --report-out <path> [--report-md-out <path>] --manifest-out <path> [--trace-dir trace] [--profile default]`

## Quality Gates

```bash
uv lock --check
uv sync --frozen
uv run python scripts/check_product_repo_layout.py
uv run ruff check .
uv run ruff format --check .
uv run mypy src/pragmalens tests
uv run interrogate src/pragmalens
uv run coverage run -m pytest -q -m "not live_smoke"
uv run coverage report --skip-empty
uv run python scripts/check_schema_parity.py
uv run python scripts/check_stage_graph.py
```

Coverage is a real gate. The configured report must stay at or above 90%.

## Hook Setup

```bash
uv run lefthook install
```

## Local Codex Overlay

Heavy Codex planning and operator-only control artifacts live outside this
product repo. See [docs/local-codex-bootstrap.md](docs/local-codex-bootstrap.md)
to restore the local overlay from the private sibling control repo.

RTK is optional and local-only. Product commands, hooks, and CI must not depend
on RTK being installed.

## Nox Sessions

```bash
uv run nox -l
uv run nox -s lint types tests build
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

- Default tests are offline and use captured fixtures for LangExtract and GLiNER2.
- Live smoke tests are marked `live_smoke` and skipped by default.
- Live smoke opt-in:

```bash
PRAGMALENS_ENABLE_LIVE_SMOKE=1 uv run pytest -q -m live_smoke
```

- Local runtime bootstrap for the current live smoke lane:

```bash
uv run python -m spacy download en_core_web_sm
uv run python -m spacy validate
uv pip install "minicheck @ git+https://github.com/Liyan06/MiniCheck.git@main"
```

- LangExtract live smoke defaults to a local Ollama model when available.
  Current default: `qwen3.5:0.8b`
- Override live runtime models with environment variables:
  - `PRAGMALENS_SPACY_MODEL`
  - `PRAGMALENS_GLINER2_MODEL`
  - `PRAGMALENS_LANGEXTRACT_PROVIDER=ollama|gemini|auto`
  - `PRAGMALENS_LANGEXTRACT_MODEL`
  - `PRAGMALENS_OLLAMA_URL`
  - `PRAGMALENS_MINICHECK_MODEL`
  - `PRAGMALENS_MINICHECK_CACHE_DIR`

Live/model tests remain outside default CI. Default CI stays offline and deterministic.
