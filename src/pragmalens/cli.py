"""CLI entrypoints for pipeline runs, schema export, and optional verifier selection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import typer

from pragmalens.core import derive_document_id, run_pipeline
from pragmalens.evaluation import (
    render_corpus_benchmark_summary,
    run_corpus_benchmark_files,
)
from pragmalens.schema import export_schema

app = typer.Typer(help="PragmaLens CLI")
schema_app = typer.Typer(help="Schema commands")
app.add_typer(schema_app, name="schema")


@schema_app.command("export")
def schema_export(
    out: str = typer.Option(..., help="Output path for JSON schema"),
    model: str = typer.Option(
        "report",
        help=(
            "Model name: report|manifest|profile|settings|evidence_candidate|span_ref|"
            "verification_score|verification_verdict|verifier_batch_comparison|"
            "verifier_calibration_profile"
        ),
    ),
) -> None:
    """Export a JSON schema for one of the supported contract models."""
    dest = export_schema(model_name=model, out_path=out)
    typer.echo(str(dest))


@app.command("run")
def run(
    input: str = typer.Option(..., "--input", help="Input markdown or text file"),
    report_out: str = typer.Option(..., "--report-out", help="Output report JSON path"),
    report_md_out: str | None = typer.Option(
        None, "--report-md-out", help="Output report Markdown path"
    ),
    manifest_out: str = typer.Option(..., "--manifest-out", help="Output manifest JSON path"),
    trace_dir: str = typer.Option("trace", "--trace-dir", help="Trace artifact directory"),
    profile: str = typer.Option("default", "--profile", help="Pipeline profile name"),
    verifier_backend: str | None = typer.Option(
        None,
        "--verifier-backend",
        help=(
            "Verifier backend: offline|minicheck|crossencoder_nli. "
            "Defaults to PRAGMALENS_VERIFIER, then offline."
        ),
    ),
) -> None:
    """Run the PragmaLens pipeline against one Markdown or text input."""
    src = Path(input)
    if src.suffix.lower() not in {".md", ".txt"}:
        raise typer.BadParameter("input must be markdown (.md) or plain text (.txt)")

    text = src.read_text(encoding="utf-8")
    document_id = derive_document_id(str(src))

    report, manifest = run_pipeline(
        text=text,
        document_id=document_id,
        input_path=str(src),
        report_path=report_out,
        trace_dir=Path(trace_dir),
        profile_name=profile,
        verifier_backend=verifier_backend,
    )

    report_path = Path(report_out)
    report_md_path = Path(report_md_out) if report_md_out else report_path.with_suffix(".md")
    manifest_path = Path(manifest_out)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_md_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    report_path.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8"
    )
    report_md_path.write_text(
        _render_markdown_report(report.model_dump(mode="json")), encoding="utf-8"
    )
    manifest_path.write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8"
    )
    typer.echo("ok")


@app.command("benchmark")
def benchmark(
    corpus_batch: str = typer.Option(..., "--corpus-batch", help="Approved corpus batch JSON"),
    corpus_metadata: str = typer.Option(
        ..., "--corpus-metadata", help="Local corpus approval metadata JSON"
    ),
    report_out: str = typer.Option(..., "--report-out", help="Output calibration report JSON path"),
    metadata_out: str = typer.Option(
        ..., "--metadata-out", help="Output benchmark sidecar metadata JSON path"
    ),
    summary_out: str | None = typer.Option(
        None, "--summary-out", help="Optional Markdown summary path"
    ),
    selected_backend: str = typer.Option(
        "offline",
        "--selected-backend",
        help="Benchmark backend: offline|minicheck|crossencoder_nli",
    ),
    run_id: str | None = typer.Option(
        None,
        "--run-id",
        help="Optional run identifier; defaults to corpus/backend",
    ),
) -> None:
    """Run a corpus-gated benchmark and emit report plus sidecar metadata."""
    report, metadata, recommendation = run_corpus_benchmark_files(
        corpus_batch_path=corpus_batch,
        approval_metadata_path=corpus_metadata,
        selected_backend=selected_backend,
        run_id=run_id,
    )

    report_path = Path(report_out)
    metadata_path = Path(metadata_out)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8"
    )
    metadata_path.write_text(
        json.dumps(metadata.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8"
    )

    if summary_out is not None:
        summary_path = Path(summary_out)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            render_corpus_benchmark_summary(report, metadata, recommendation), encoding="utf-8"
        )
    typer.echo("ok")


def _render_markdown_report(payload: dict[str, object]) -> str:
    """Render a minimal Markdown summary for the generated report payload."""
    findings_value = payload.get("findings", [])
    findings = findings_value if isinstance(findings_value, list) else []
    lines = [
        f"# PragmaLens Report: {payload['document_id']}",
        "",
        f"- Run ID: `{payload['run_id']}`",
        f"- Findings: {len(findings) if isinstance(findings, list) else 0}",
        "",
        "## Findings",
        "",
    ]
    if not findings:
        lines.append("No findings were produced by the current offline verifier.")
        return "\n".join(lines) + "\n"

    for finding in findings:
        if not isinstance(finding, dict):
            continue
        finding = cast(dict[str, Any], finding)
        lines.append(
            f"- `{finding.get('finding_id', 'finding')}` "
            f"({finding.get('verdict', 'unknown')}): {finding.get('summary', '')}"
        )
        question = finding.get("question")
        if question:
            lines.append(f"  - Question: {question}")
        actionability = finding.get("actionability")
        if actionability:
            lines.append(f"  - Actionability: {actionability}")
        evidence_ids = finding.get("evidence_ids")
        if isinstance(evidence_ids, list) and evidence_ids:
            lines.append(f"  - Evidence: {', '.join(str(item) for item in evidence_ids)}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    app()
