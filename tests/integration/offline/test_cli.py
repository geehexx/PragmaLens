import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from pragmalens.cli import app
from pragmalens.models import (
    NeutralReport,
    RunManifest,
    VerificationStatus,
    VerificationVerdict,
    VerifierBatchComparison,
    VerifierCandidateComparison,
)


def test_cli_run_produces_report_and_manifest(tmp_path: Path) -> None:
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
            "--verifier-backend",
            "offline",
        ],
    )

    assert result.exit_code == 0
    assert report_out.exists()
    assert manifest_out.exists()

    report = json.loads(report_out.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_out.read_text(encoding="utf-8"))

    assert report["document_id"] == "sample"
    assert report["findings"] == []
    assert "verify_claims" in manifest["stages_requested"]


def test_cli_schema_export(tmp_path: Path) -> None:
    out = tmp_path / "schemas" / "settings.schema.json"
    runner = CliRunner()
    result = runner.invoke(app, ["schema", "export", "--out", str(out), "--model", "settings"])
    assert result.exit_code == 0
    assert out.exists()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["title"] == "PragmaLensSettings"
    assert "spacy_model" in data["properties"]
    assert "crossencoder_cache_dir" in data["properties"]
    assert "crossencoder_revision" in data["properties"]


def test_cli_run_persists_default_signal_ensemble_comparison_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    src = tmp_path / "sample.md"
    src.write_text("# Plan\n\nA claim.", encoding="utf-8")

    report_out = tmp_path / "out" / "report.json"
    manifest_out = tmp_path / "out" / "manifest.json"

    report = NeutralReport(
        run_id="run-sample",
        document_id="sample",
        findings=[],
        candidates=[],
        verification=[
            VerificationVerdict(
                candidate_id="candidate-1",
                status=VerificationStatus.SUPPORTED,
                rationale="ensemble fixture",
                backend="signal_ensemble",
            )
        ],
        verification_comparison=VerifierBatchComparison(
            selected_backend="minicheck",
            total_candidates=1,
            records=[
                VerifierCandidateComparison(
                    candidate_id="candidate-1",
                    selected_verdict=VerificationVerdict(
                        candidate_id="candidate-1",
                        status=VerificationStatus.SUPPORTED,
                        rationale="selected fixture",
                        backend="minicheck",
                    ),
                    backend_verdicts=[],
                    disagreement=False,
                )
            ],
        ),
    )
    manifest = RunManifest(
        run_id="run-sample",
        document_id="sample",
        input_path=str(src),
        report_path=str(report_out),
        stages_requested=["verify_claims"],
        stage_health={"verify_claims": "ok"},
        artifacts={"verification_comparison_json": str(tmp_path / "trace" / "comparison.json")},
    )
    monkeypatch.setattr("pragmalens.cli.run_pipeline", lambda **_: (report, manifest))

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
    saved_report = json.loads(report_out.read_text(encoding="utf-8"))
    saved_manifest = json.loads(manifest_out.read_text(encoding="utf-8"))
    assert saved_report["verification"][0]["backend"] == "signal_ensemble"
    assert saved_report["verification_comparison"]["selected_backend"] == "minicheck"
    assert "verification_comparison_json" in saved_manifest["artifacts"]


def test_cli_benchmark_produces_calibration_report_and_sidecar(
    tmp_path: Path,
) -> None:
    fixtures = Path("tests/fixtures")
    report_out = tmp_path / "benchmark" / "report.json"
    metadata_out = tmp_path / "benchmark" / "metadata.json"
    summary_out = tmp_path / "benchmark" / "summary.md"

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "benchmark",
            "--corpus-batch",
            str(fixtures / "corpus_benchmark_ragtruth.json"),
            "--corpus-metadata",
            str(fixtures / "corpus_approval_ragtruth.json"),
            "--report-out",
            str(report_out),
            "--metadata-out",
            str(metadata_out),
            "--summary-out",
            str(summary_out),
            "--selected-backend",
            "offline",
            "--run-id",
            "benchmark-run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert report_out.exists()
    assert metadata_out.exists()
    assert summary_out.exists()

    report = json.loads(report_out.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_out.read_text(encoding="utf-8"))
    summary = summary_out.read_text(encoding="utf-8")

    assert report["run_id"] == "benchmark-run"
    assert {summary_entry["backend"] for summary_entry in report["summaries"]} == {"offline"}
    assert metadata["corpus_id"] == "ragtruth-15592-mini"
    assert metadata["selected_backend"] == "offline"
    assert metadata["approval_status"] == "approved"
    assert metadata["case_count"] == 2
    assert metadata["approval_evidence"] == [
        "upstream dataset source_id 15592 from RAGTruth source_info.jsonl",
        (
            "upstream response id 0 is supported and response id 2 contains the explicit "
            "2022 hallucination"
        ),
    ]
    assert summary.startswith("# Corpus Benchmark: RAGTruth")
