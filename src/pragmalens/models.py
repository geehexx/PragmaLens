"""Pydantic contracts for reports, candidates, findings, and manifests."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class CandidateStatus(StrEnum):
    """Lifecycle states for extracted evidence candidates."""

    VALID = "valid"
    QUARANTINED = "quarantined"
    DUPLICATE = "duplicate"
    REJECTED = "rejected"


class VerificationStatus(StrEnum):
    """Verifier outcomes for normalized candidates."""

    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    VERIFIER_ERROR = "verifier_error"


class VerificationScore(BaseModel):
    """Normalized score payload captured from one verifier backend."""

    predicted_label: str | None = None
    support_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    contradiction_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    entailment_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    neutral_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    normalized: bool = True


class VerifierCalibrationThresholds(BaseModel):
    """Explicit threshold fields used to map backend scores into verdicts."""

    support_probability_min: float | None = Field(default=None, ge=0.0, le=1.0)
    support_probability_max: float | None = Field(default=None, ge=0.0, le=1.0)
    contradiction_probability_min: float | None = Field(default=None, ge=0.0, le=1.0)
    entailment_probability_min: float | None = Field(default=None, ge=0.0, le=1.0)
    neutral_probability_max: float | None = Field(default=None, ge=0.0, le=1.0)


class VerifierCalibrationProfile(BaseModel):
    """Evidence-backed threshold profile for one verifier backend."""

    calibration_id: str = Field(min_length=1)
    backend: str = Field(min_length=1)
    evidence_source: str = Field(min_length=1)
    sample_size: int = Field(ge=1)
    thresholds: VerifierCalibrationThresholds = Field(default_factory=VerifierCalibrationThresholds)
    notes: list[str] = Field(default_factory=list)


class SpanRef(BaseModel):
    """Offset-based reference into a source document."""

    document_id: str = Field(min_length=1)
    start_char: int = Field(ge=0)
    end_char: int = Field(gt=0)
    text: str = Field(default="")

    @model_validator(mode="after")
    def _validate_offsets(self) -> SpanRef:
        """Reject inverted or zero-width spans."""
        if self.end_char <= self.start_char:
            raise ValueError("end_char must be greater than start_char")
        return self

    @model_validator(mode="after")
    def _validate_text_length(self) -> SpanRef:
        """Ensure embedded text matches the declared span width when present."""
        if self.text and len(self.text) != (self.end_char - self.start_char):
            raise ValueError("text length must match span width when text is provided")
        return self


class EvidenceCandidate(BaseModel):
    """Normalized candidate produced by one or more extraction stages."""

    candidate_id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    status: CandidateStatus = CandidateStatus.VALID
    span: SpanRef
    confidence: float | None = None
    provenance: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    relations: list[dict[str, Any]] = Field(default_factory=list)
    attributes: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _default_evidence_refs(self) -> EvidenceCandidate:
        """Backfill evidence refs from provenance when callers omit them."""
        if not self.evidence_refs:
            self.evidence_refs = list(self.provenance)
        return self


class VerificationVerdict(BaseModel):
    """Verifier result for a single evidence candidate."""

    candidate_id: str = Field(min_length=1)
    status: VerificationStatus
    rationale: str = Field(min_length=1)
    evidence_ids: list[str] = Field(default_factory=list)
    backend: str = Field(default="offline", min_length=1)
    model_name: str | None = None
    score: VerificationScore | None = None
    calibration_id: str | None = None
    error: str | None = None


class VerifierCandidateComparison(BaseModel):
    """Per-candidate same-batch verdict comparison across verifier backends."""

    candidate_id: str = Field(min_length=1)
    selected_verdict: VerificationVerdict
    backend_verdicts: list[VerificationVerdict] = Field(default_factory=list)
    disagreement: bool = False


class VerifierBatchComparison(BaseModel):
    """Typed comparison artifact covering one candidate batch across backends."""

    selected_backend: str = Field(min_length=1)
    total_candidates: int = Field(ge=0)
    records: list[VerifierCandidateComparison] = Field(default_factory=list)


class VerifierCalibrationMetrics(BaseModel):
    """Summary metrics produced while fitting a calibration profile."""

    evaluated_examples: int = Field(ge=0)
    correct: int = Field(ge=0)
    accuracy: float = Field(ge=0.0, le=1.0)


class VerifierCalibrationRecommendation(BaseModel):
    """Recommended calibration profile plus the evidence that supports it."""

    backend: str = Field(min_length=1)
    model_name: str | None = None
    calibration: VerifierCalibrationProfile
    metrics: VerifierCalibrationMetrics


class Finding(BaseModel):
    """User-facing finding synthesized from verifier output."""

    finding_id: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    verdict: VerificationStatus
    summary: str = Field(min_length=1)
    evidence_ids: list[str] = Field(default_factory=list)


class NeutralReport(BaseModel):
    """Top-level offline report contract emitted by the product pipeline."""

    report_version: str = "0.1"
    run_id: str = Field(min_length=1)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    document_id: str = Field(min_length=1)
    source_format: str = Field(default="markdown_or_text")
    findings: list[Finding] = Field(default_factory=list)
    candidates: list[EvidenceCandidate] = Field(default_factory=list)
    verification: list[VerificationVerdict] = Field(default_factory=list)
    verification_comparison: VerifierBatchComparison | None = None
    warnings: list[str] = Field(default_factory=list)


class RunManifest(BaseModel):
    """Execution manifest describing one pipeline run and its artifacts."""

    run_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    input_path: str = Field(min_length=1)
    report_path: str = Field(min_length=1)
    profile_name: str = Field(default="default")
    prompt_hash: str | None = None
    model_registry: dict[str, str] = Field(default_factory=dict)
    stages_requested: list[str] = Field(default_factory=list)
    stage_health: dict[str, str] = Field(default_factory=dict)
    artifacts: dict[str, str] = Field(default_factory=dict)

    @field_validator("stages_requested")
    @classmethod
    def _stages_non_empty(cls, value: list[str]) -> list[str]:
        """Require at least one recorded stage in the run manifest."""
        if not value:
            raise ValueError("stages_requested must not be empty")
        return value
