"""Corpus benchmark automation built on top of verifier calibration helpers."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from pragmalens.models import SpanRef, VerificationStatus, VerifierCalibrationRecommendation
from pragmalens.verifier import (
    CrossEncoderNliVerifier,
    MiniCheckVerifier,
    OfflineBaselineVerifier,
    SignalEnsembleVerifier,
    VerifierAdapter,
    VerifierBackend,
    build_verifier_adapter,
)
from pragmalens.verifier_calibration import (
    VerifierCalibrationCase,
    VerifierCalibrationReport,
    compare_verifier_backends,
)
from pragmalens.verifier_comparison import CalibrationExample, recommend_calibration

ApprovalStatus = Literal["approved", "pending", "rejected"]
GoldenAnnotationLabel = Literal[
    "promise",
    "review_commitment",
    "antecedent_resolution",
    "team_membership",
]


class CorpusBenchmarkBatch(BaseModel):
    """Benchmark cases grouped under one approved corpus identifier."""

    corpus_id: str = Field(min_length=1)
    cases: list[VerifierCalibrationCase] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_cases(self) -> CorpusBenchmarkBatch:
        if not self.cases:
            raise ValueError("benchmark corpus must include at least one case")
        return self


class CorpusApprovalMetadata(BaseModel):
    """Local approval evidence supplied alongside a benchmark corpus batch."""

    corpus_id: str = Field(min_length=1)
    corpus_name: str = Field(min_length=1)
    corpus_version: str = Field(min_length=1)
    source_uri: str = Field(min_length=1)
    approval_status: ApprovalStatus
    approval_evidence: list[str] = Field(default_factory=list)


class GoldenSetAnnotation(BaseModel):
    """Explicit semantic annotation used by the internal golden set."""

    label: GoldenAnnotationLabel
    span: SpanRef
    resolved_to: str | None = None
    notes: list[str] = Field(default_factory=list)


class GoldenSetCase(BaseModel):
    """One labeled semantic regression case for the internal golden set."""

    case_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    document_text: str = Field(min_length=1)
    annotations: list[GoldenSetAnnotation] = Field(default_factory=list)


class GoldenSetBatch(BaseModel):
    """Approved semantic-gold set grouped under one internal identifier."""

    golden_set_id: str = Field(min_length=1)
    cases: list[GoldenSetCase] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_cases(self) -> GoldenSetBatch:
        if not self.cases:
            raise ValueError("golden set must include at least one case")
        return self


class CorpusBenchmarkRunMetadata(BaseModel):
    """Compact sidecar metadata written next to the benchmark report."""

    run_id: str = Field(min_length=1)
    corpus_id: str = Field(min_length=1)
    corpus_name: str = Field(min_length=1)
    corpus_version: str = Field(min_length=1)
    source_uri: str = Field(min_length=1)
    selected_backend: str = Field(min_length=1)
    approval_status: ApprovalStatus
    approval_evidence: list[str] = Field(default_factory=list)
    case_count: int = Field(ge=1)


def load_corpus_benchmark_batch(path: str | Path) -> CorpusBenchmarkBatch:
    """Load a benchmark batch from a JSON object on disk."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("benchmark corpus file must contain a JSON object")
    return CorpusBenchmarkBatch.model_validate(payload)


def load_corpus_approval_metadata(path: str | Path) -> CorpusApprovalMetadata:
    """Load corpus approval evidence from a JSON object on disk."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("corpus approval file must contain a JSON object")
    return CorpusApprovalMetadata.model_validate(payload)


def load_golden_set_batch(path: str | Path) -> GoldenSetBatch:
    """Load a semantic golden set from a JSON object on disk."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("golden set file must contain a JSON object")
    return GoldenSetBatch.model_validate(payload)


def run_corpus_benchmark(
    batch: CorpusBenchmarkBatch,
    approval: CorpusApprovalMetadata,
    *,
    selected_backend: VerifierBackend | str | None = None,
    selected_verifier: VerifierAdapter | None = None,
    run_id: str | None = None,
) -> tuple[
    VerifierCalibrationReport,
    CorpusBenchmarkRunMetadata,
    VerifierCalibrationRecommendation,
]:
    """Run one approved corpus benchmark and return report plus sidecar metadata."""
    backend = _resolve_selected_backend(selected_backend, selected_verifier)
    if selected_verifier is not None:
        verifier_backend = _resolve_selected_backend(None, selected_verifier)
        if verifier_backend != backend:
            raise ValueError("selected backend must match the selected verifier")
    _validate_corpus_gate(batch, approval)
    verifier = selected_verifier or build_verifier_adapter(backend)
    benchmark_verifier = _benchmark_verifier_for(verifier)
    calibration_cases = list(batch.cases)
    resolved_run_id = run_id or f"{batch.corpus_id}-{backend.value}"
    report = _compare_benchmark_backends(
        calibration_cases, benchmark_verifier, backend, resolved_run_id
    )
    recommendation = recommend_calibration(
        benchmark_verifier,
        _to_calibration_examples(calibration_cases),
        evidence_source=approval.source_uri,
    )
    metadata = CorpusBenchmarkRunMetadata(
        run_id=resolved_run_id,
        corpus_id=batch.corpus_id,
        corpus_name=approval.corpus_name,
        corpus_version=approval.corpus_version,
        source_uri=approval.source_uri,
        selected_backend=backend.value,
        approval_status=approval.approval_status,
        approval_evidence=list(approval.approval_evidence),
        case_count=len(calibration_cases),
    )
    return report, metadata, recommendation


def run_corpus_benchmark_files(
    *,
    corpus_batch_path: str | Path,
    approval_metadata_path: str | Path,
    selected_backend: VerifierBackend | str | None = None,
    selected_verifier: VerifierAdapter | None = None,
    run_id: str | None = None,
) -> tuple[
    VerifierCalibrationReport,
    CorpusBenchmarkRunMetadata,
    VerifierCalibrationRecommendation,
]:
    """Load benchmark inputs from disk and execute the corpus benchmark."""
    batch = load_corpus_benchmark_batch(corpus_batch_path)
    approval = load_corpus_approval_metadata(approval_metadata_path)
    return run_corpus_benchmark(
        batch,
        approval,
        selected_backend=selected_backend,
        selected_verifier=selected_verifier,
        run_id=run_id,
    )


def render_corpus_benchmark_summary(
    report: VerifierCalibrationReport,
    metadata: CorpusBenchmarkRunMetadata,
    recommendation: VerifierCalibrationRecommendation,
) -> str:
    """Render a compact Markdown summary for human review."""
    lines = [
        f"# Corpus Benchmark: {metadata.corpus_name}",
        "",
        f"- Run ID: `{metadata.run_id}`",
        f"- Corpus ID: `{metadata.corpus_id}`",
        f"- Selected backend: `{metadata.selected_backend}`",
        f"- Approval status: `{metadata.approval_status}`",
        f"- Cases: {metadata.case_count}",
        "",
        "## Approval Evidence",
    ]
    for item in metadata.approval_evidence:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Backend Summaries",
        ]
    )
    for summary in report.summaries:
        lines.append(
            f"- `{summary.backend}`: exact match {summary.exact_match_count}/{summary.total}, "
            f"accuracy {summary.exact_match_rate:.3f}"
        )
    lines.extend(
        [
            "",
            "## Calibration Recommendation",
            f"- Backend: `{recommendation.backend}`",
            f"- Model: `{recommendation.model_name or 'n/a'}`",
            f"- Evaluated examples: {recommendation.metrics.evaluated_examples}",
            f"- Accuracy: {recommendation.metrics.accuracy:.3f}",
        ]
    )
    thresholds = recommendation.calibration.thresholds
    threshold_lines = [
        ("support_probability_min", thresholds.support_probability_min),
        ("support_probability_max", thresholds.support_probability_max),
        ("contradiction_probability_min", thresholds.contradiction_probability_min),
        ("entailment_probability_min", thresholds.entailment_probability_min),
        ("neutral_probability_max", thresholds.neutral_probability_max),
    ]
    lines.append("")
    lines.append("## Thresholds")
    for name, value in threshold_lines:
        lines.append(f"- {name}: {value if value is not None else 'n/a'}")
    return "\n".join(lines) + "\n"


def _validate_corpus_gate(batch: CorpusBenchmarkBatch, approval: CorpusApprovalMetadata) -> None:
    """Reject benchmark runs that do not have explicit local corpus approval."""
    if batch.corpus_id != approval.corpus_id:
        raise ValueError("corpus id mismatch between benchmark batch and approval metadata")
    if approval.approval_status != "approved":
        raise ValueError("corpus approval status must be approved")
    if not approval.approval_evidence:
        raise ValueError("corpus approval evidence is required")


def _resolve_selected_backend(
    selected_backend: VerifierBackend | str | None, selected_verifier: VerifierAdapter | None
) -> VerifierBackend:
    """Resolve the benchmark backend from an explicit selector or the injected verifier."""
    if selected_backend is not None:
        return VerifierBackend(selected_backend)
    if isinstance(selected_verifier, SignalEnsembleVerifier):
        if selected_verifier.selected_backend is None:
            raise ValueError("signal ensemble verifier does not declare a selected backend")
        return selected_verifier.selected_backend
    if selected_verifier is not None:
        backend = getattr(selected_verifier, "backend", None)
        if isinstance(backend, VerifierBackend):
            return backend
        if isinstance(backend, str):
            return VerifierBackend(backend)
    return VerifierBackend.OFFLINE


def _benchmark_verifier_for(verifier: VerifierAdapter) -> VerifierAdapter:
    """Resolve the concrete backend adapter that benchmark helpers should compare."""
    if isinstance(verifier, SignalEnsembleVerifier):
        return verifier.selected_backend_adapter()
    return verifier


def _compare_benchmark_backends(
    cases: Sequence[VerifierCalibrationCase],
    verifier: VerifierAdapter,
    backend: VerifierBackend,
    run_id: str,
) -> VerifierCalibrationReport:
    """Build a calibration report with the selected backend plus the offline baseline."""
    if isinstance(verifier, OfflineBaselineVerifier) or backend is VerifierBackend.OFFLINE:
        return compare_verifier_backends(cases, offline_verifier=verifier, run_id=run_id)
    if isinstance(verifier, MiniCheckVerifier):
        return compare_verifier_backends(
            cases,
            offline_verifier=OfflineBaselineVerifier(),
            minicheck_scorer=verifier.scorer,
            run_id=run_id,
        )
    if isinstance(verifier, CrossEncoderNliVerifier):
        return compare_verifier_backends(
            cases,
            offline_verifier=OfflineBaselineVerifier(),
            crossencoder_model=verifier.model,
            run_id=run_id,
        )
    raise ValueError(f"unsupported benchmark verifier: {type(verifier)!r}")


def _to_calibration_examples(
    cases: Sequence[VerifierCalibrationCase],
) -> list[CalibrationExample]:
    """Convert benchmark cases into calibration examples for threshold recommendation."""
    return [
        CalibrationExample(
            example_id=case.case_id,
            document_id=case.document_id,
            text=case.document_text,
            claim_text=case.claim_text,
            expected_status=_expected_status_from_case(case),
        )
        for case in cases
    ]


def _expected_status_from_case(case: VerifierCalibrationCase) -> VerificationStatus:
    """Infer the gold status for a calibration example from the benchmark case."""
    return case.gold_status
