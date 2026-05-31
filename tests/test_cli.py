import json

from typer.testing import CliRunner

from pragmalens.cli import app


def test_cli_run_produces_report_and_manifest(tmp_path) -> None:
    src = tmp_path / "sample.md"
    src.write_text("# Plan\n\nA claim.", encoding="utf-8")

    report_out = tmp_path / "out" / "report.json"
    manifest_out = tmp_path / "out" / "manifest.json"

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "run",
            "--input",
            str(src),
            "--report-out",
            str(report_out),
            "--manifest-out",
            str(manifest_out),
        ],
    )

    assert result.exit_code == 0
    assert report_out.exists()
    assert manifest_out.exists()

    report = json.loads(report_out.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_out.read_text(encoding="utf-8"))

    assert report["document_id"] == "sample"
    assert report["findings"] == []
    assert manifest["verifier_stub"] is True


def test_cli_schema_export(tmp_path) -> None:
    out = tmp_path / "schemas" / "manifest.schema.json"
    runner = CliRunner()
    result = runner.invoke(app, ["schema", "export", "--out", str(out), "--model", "manifest"])
    assert result.exit_code == 0
    assert out.exists()
