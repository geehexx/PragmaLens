import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from pragmalens.cli import app
from pragmalens.models import EvidenceCandidate, VerificationStatus, VerificationVerdict


class _UnsupportedVerifier:
    def verify(
        self,
        candidates: list[EvidenceCandidate],
        *,
        document_id: str,
        text: str,
    ) -> list[VerificationVerdict]:
        del document_id, text
        return [
            VerificationVerdict(
                candidate_id=candidate.candidate_id,
                status=VerificationStatus.UNSUPPORTED,
                rationale="insufficient supporting evidence",
                evidence_ids=candidate.evidence_refs,
                backend="offline",
            )
            for candidate in candidates
        ]


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
            "--verifier-backend",
            "offline",
        ],
    )

    assert result.exit_code == 0, result.output
    report_payload = json.loads(report.read_text(encoding="utf-8"))
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    stage_timings_payload = json.loads((trace / "stage_timings.json").read_text(encoding="utf-8"))
    trace_manifest_payload = json.loads((trace / "trace_manifest.json").read_text(encoding="utf-8"))
    assert report_payload["run_id"] == manifest_payload["run_id"]
    assert trace_manifest_payload["run_id"] == report_payload["run_id"]
    assert manifest_payload["artifacts"]["trace_dir"] == str(trace)
    assert manifest_payload["artifacts"]["trace_manifest_json"] == str(
        trace / "trace_manifest.json"
    )
    assert manifest_payload["artifacts"]["stage_timings_json"] == str(trace / "stage_timings.json")
    assert report_md.exists()
    assert (trace / "stage_results.json").exists()
    assert (trace / "stage_timings.json").exists()
    assert (trace / "verification.json").exists()
    assert (trace / "findings.json").exists()
    assert trace_manifest_payload["total_duration_ms"] >= round(
        sum(item["duration_ms"] for item in stage_timings_payload["stages"]),
        3,
    )
    assert trace_manifest_payload["total_duration_ms"] == stage_timings_payload["total_duration_ms"]
    assert "verification.json" in trace_manifest_payload["files"]
    assert "stage_timings.json" in trace_manifest_payload["files"]


def test_cli_renders_synthesized_question_and_actionability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "sample.md"
    report = tmp_path / "report.json"
    report_md = tmp_path / "report.md"
    manifest = tmp_path / "run_manifest.json"
    trace = tmp_path / "trace"
    source.write_text("The team should ship the verifier if evidence is present.", encoding="utf-8")

    monkeypatch.setattr(
        "pragmalens.core.build_default_verifier_runtime",
        lambda backend=None: (_UnsupportedVerifier(), None),
    )

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
            "--verifier-backend",
            "offline",
        ],
    )

    assert result.exit_code == 0, result.output
    report_payload = json.loads(report.read_text(encoding="utf-8"))
    report_md_text = report_md.read_text(encoding="utf-8")
    assert report_payload["findings"][0]["question"]
    assert report_payload["findings"][0]["actionability"]
    assert "Question:" in report_md_text
    assert "Actionability:" in report_md_text
