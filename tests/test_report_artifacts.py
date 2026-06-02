import json
from pathlib import Path

from typer.testing import CliRunner

from pragmalens.cli import app


def test_cli_emits_report_manifest_markdown_and_trace_with_shared_run_id(tmp_path: Path) -> None:
    source = tmp_path / "sample.md"
    report = tmp_path / "report.json"
    report_md = tmp_path / "report.md"
    manifest = tmp_path / "run_manifest.json"
    trace = tmp_path / "trace"
    source.write_text("The team should ship the verifier if evidence is present.", encoding="utf-8")

    result = CliRunner().invoke(
        app,
        [
            "run",
            "--input",
            str(source),
            "--report-out",
            str(report),
            "--report-md-out",
            str(report_md),
            "--manifest-out",
            str(manifest),
            "--trace-dir",
            str(trace),
        ],
    )

    assert result.exit_code == 0, result.output
    report_payload = json.loads(report.read_text(encoding="utf-8"))
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    trace_manifest_payload = json.loads((trace / "trace_manifest.json").read_text(encoding="utf-8"))
    assert report_payload["run_id"] == manifest_payload["run_id"]
    assert trace_manifest_payload["run_id"] == report_payload["run_id"]
    assert manifest_payload["artifacts"]["trace_dir"] == str(trace)
    assert manifest_payload["artifacts"]["trace_manifest_json"] == str(
        trace / "trace_manifest.json"
    )
    assert report_md.exists()
    assert (trace / "stage_results.json").exists()
    assert (trace / "verification.json").exists()
    assert (trace / "findings.json").exists()
    assert "verification.json" in trace_manifest_payload["files"]
