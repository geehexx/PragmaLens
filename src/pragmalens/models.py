from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class CandidateStatus(StrEnum):
    VALID = "valid"
    QUARANTINED = "quarantined"
    DUPLICATE = "duplicate"
    REJECTED = "rejected"


class VerificationStatus(StrEnum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    VERIFIER_ERROR = "verifier_error"


class SpanRef(BaseModel):
    document_id: str = Field(min_length=1)
    start_char: int = Field(ge=0)
    end_char: int = Field(gt=0)
    text: str = Field(default="")

    @model_validator(mode="after")
    def _validate_offsets(self) -> SpanRef:
        if self.end_char <= self.start_char:
            raise ValueError("end_char must be greater than start_char")
        return self

    @model_validator(mode="after")
    def _validate_text_length(self) -> SpanRef:
        if self.text and len(self.text) != (self.end_char - self.start_char):
            raise ValueError("text length must match span width when text is provided")
        return self


class EvidenceCandidate(BaseModel):
    candidate_id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    status: CandidateStatus = CandidateStatus.VALID
    span: SpanRef
    confidence: float | None = None
    provenance: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    relations: list[dict[str, Any]] = Field(default_factory=list)
    attributes: dict[str, Any] = Field(default_factory=dict)


class VerificationVerdict(BaseModel):
    candidate_id: str = Field(min_length=1)
    status: VerificationStatus
    rationale: str = Field(min_length=1)
    evidence_ids: list[str] = Field(default_factory=list)
    error: str | None = None


class Finding(BaseModel):
    finding_id: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    verdict: VerificationStatus
    summary: str = Field(min_length=1)
    evidence_ids: list[str] = Field(default_factory=list)


class NeutralReport(BaseModel):
    report_version: str = "0.1"
    run_id: str = Field(min_length=1)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    document_id: str = Field(min_length=1)
    source_format: str = Field(default="markdown_or_text")
    findings: list[Finding] = Field(default_factory=list)
    candidates: list[EvidenceCandidate] = Field(default_factory=list)
    verification: list[VerificationVerdict] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class RunManifest(BaseModel):
    run_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    input_path: str = Field(min_length=1)
    report_path: str = Field(min_length=1)
    profile_name: str = Field(default="default")
    prompt_hash: str | None = None
    model_registry: dict[str, str] = Field(default_factory=dict)
    stages_requested: list[str] = Field(default_factory=lambda: ["pr01_normalize"])
    stage_health: dict[str, str] = Field(default_factory=lambda: {"pr01_normalize": "ok"})
    artifacts: dict[str, str] = Field(default_factory=dict)
    verifier_stub: bool = False

    @field_validator("stages_requested")
    @classmethod
    def _stages_non_empty(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("stages_requested must not be empty")
        return value
