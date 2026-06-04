from __future__ import annotations

import json
from pathlib import Path

import pytest

from pragmalens.core import run_pipeline
from pragmalens.models import EvidenceCandidate, VerificationStatus, VerificationVerdict
from pragmalens.pipeline.stage_graph import (
    REQUIRED_STAGE_IDS,
    default_v01_stages,
    validate_stage_graph,
)
from pragmalens.pipeline.text import is_valid_span
from pragmalens.verifier import OfflineBaselineVerifier, VerifierBackend
from pragmalens.verifier_comparison import VerifierComparisonHarness


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


class _SupportedVerifier:
    backend = VerifierBackend.MINICHECK

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
                status=VerificationStatus.SUPPORTED,
                rationale="signal ensemble fixture",
                evidence_ids=candidate.evidence_refs,
                backend="signal_ensemble",
            )
            for candidate in candidates
        ]


def test_default_stage_graph_validates() -> None:
    """Keep the runtime stage graph aligned with the declared v0.1 sequence."""
    stages = default_v01_stages()
    validate_stage_graph(stages)
    assert [stage.id for stage in stages] == REQUIRED_STAGE_IDS


def test_run_pipeline_outputs_spacy_artifacts_and_valid_cues(tmp_path: Path) -> None:
    """Verify the current default pipeline emits substrate traces and trace metadata."""
    text = "If we must ship now, we should not delay [1]."
    report_out = tmp_path / "out" / "report.json"
    trace_dir = tmp_path / "trace"

    report, manifest = run_pipeline(
        text=text,
        document_id="doc-1",
        input_path="/tmp/doc.md",
        report_path=str(report_out),
        trace_dir=trace_dir,
    )

    assert report.findings == []
    assert manifest.stage_health["spacy_substrate"] == "ok"
    assert manifest.stages_requested == REQUIRED_STAGE_IDS

    spacy_artifact = json.loads((trace_dir / "spacy_substrate.json").read_text(encoding="utf-8"))
    assert spacy_artifact["sentences"]
    assert spacy_artifact["tokens"]
    assert spacy_artifact["cues"]

    for cue in spacy_artifact["cues"]:
        assert is_valid_span(text, cue["start_char"], cue["end_char"])


def test_run_pipeline_builds_default_report_shape(tmp_path: Path) -> None:
    """Verify the default report contract still has the expected empty baseline shape."""
    report, _ = run_pipeline(
        text="# Doc\n\nClaim",
        document_id="doc-123",
        input_path="/tmp/doc.md",
        report_path=str(tmp_path / "report.json"),
        trace_dir=tmp_path / "trace",
    )

    payload = report.model_dump(mode="json")
    assert payload["report_version"] == "0.1"
    assert payload["document_id"] == "doc-123"
    assert payload["source_format"] == "markdown_or_text"
    assert payload["findings"] == []
    assert payload["candidates"]
    assert payload["warnings"] == []


def test_run_pipeline_surfaces_question_and_actionability_for_unsupported_verdict(
    tmp_path: Path,
) -> None:
    report, _ = run_pipeline(
        text="# Doc\n\nClaim",
        document_id="doc-123",
        input_path="/tmp/doc.md",
        report_path=str(tmp_path / "report.json"),
        trace_dir=tmp_path / "trace",
        verifier=_UnsupportedVerifier(),
    )

    payload = report.model_dump(mode="json")
    assert payload["findings"][0]["question"]
    assert payload["findings"][0]["actionability"]


def test_run_pipeline_uses_default_verifier_runtime_for_signal_ensemble(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = VerifierComparisonHarness(
        selected_backend=VerifierBackend.MINICHECK,
        verifiers=[OfflineBaselineVerifier(), _SupportedVerifier()],
    )
    monkeypatch.setattr(
        "pragmalens.core.build_default_verifier_runtime",
        lambda backend=None: (_SupportedVerifier(), harness),
    )

    report, manifest = run_pipeline(
        text="# Doc\n\nClaim",
        document_id="doc-ensemble",
        input_path="/tmp/doc.md",
        report_path=str(tmp_path / "report.json"),
        trace_dir=tmp_path / "trace",
    )

    assert report.verification
    assert report.verification[0].backend == "signal_ensemble"
    assert report.verification_comparison is not None
    assert report.verification_comparison.selected_backend == VerifierBackend.MINICHECK
    assert "verification_comparison_json" in manifest.artifacts
