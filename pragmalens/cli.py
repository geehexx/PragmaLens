from __future__ import annotations

import json
from pathlib import Path

import typer

from pragmalens.core import derive_document_id, run_pipeline
from pragmalens.schema import export_schema

app = typer.Typer(help="PragmaLens CLI")
schema_app = typer.Typer(help="Schema commands")
app.add_typer(schema_app, name="schema")


@schema_app.command("export")
def schema_export(
    out: str = typer.Option(..., help="Output path for JSON schema"),
    model: str = typer.Option("report", help="Model name: report|manifest|span_ref"),
) -> None:
    dest = export_schema(model_name=model, out_path=out)
    typer.echo(str(dest))


@app.command("run")
def run(
    input: str = typer.Option(..., "--input", help="Input markdown or text file"),
    report_out: str = typer.Option(..., "--report-out", help="Output report JSON path"),
    manifest_out: str = typer.Option(..., "--manifest-out", help="Output manifest JSON path"),
    trace_dir: str = typer.Option("trace", "--trace-dir", help="Trace artifact directory"),
    profile: str = typer.Option("default", "--profile", help="Pipeline profile name"),
) -> None:
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
    )

    report_path = Path(report_out)
    manifest_path = Path(manifest_out)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    report_path.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8"
    )
    manifest_path.write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8"
    )
    typer.echo("ok")


if __name__ == "__main__":
    app()
