"""Verifier comparison, calibration, and recommendation helpers."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, Field

from pragmalens.models import (
    EvidenceCandidate,
    SpanRef,
    VerificationScore,
    VerificationStatus,
    VerificationVerdict,
    VerifierBatchComparison,
    VerifierCalibrationMetrics,
    VerifierCalibrationProfile,
    VerifierCalibrationRecommendation,
    VerifierCalibrationThresholds,
    VerifierCandidateComparison,
)
from pragmalens.verifier import (
    CrossEncoderNliVerifier,
    MiniCheckVerifier,
    OfflineBaselineVerifier,
    VerifierAdapter,
    VerifierBackend,
)


class CalibrationExample(BaseModel):
    """Evidence-backed calibration example used for threshold fitting."""

    example_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    claim_text: str = Field(min_length=1)
    expected_status: VerificationStatus


class VerifierComparisonHarness:
    """Compare multiple verifier backends against the same candidate batch."""

    def __init__(
        self,
        *,
        selected_backend: VerifierBackend,
        verifiers: Sequence[VerifierAdapter],
    ) -> None:
        """Store the selected backend and preserve the verifier ordering."""
        self.selected_backend = selected_backend
        self._verifiers = list(verifiers)
        backends = [_backend_for_verifier(verifier) for verifier in self._verifiers]
        if len(set(backends)) != len(backends):
            raise ValueError("duplicate verifier backends are not supported in comparisons")
        if self.selected_backend not in backends:
            raise ValueError(f"selected backend {self.selected_backend!r} is not configured")

    def compare(
        self,
        candidates: Sequence[EvidenceCandidate],
        *,
        document_id: str,
        text: str,
        selected_verdicts: Sequence[VerificationVerdict] | None = None,
    ) -> VerifierBatchComparison:
        """Compare the same candidate batch across every configured backend."""
        selected_backend_verdicts = (
            list(selected_verdicts)
            if selected_verdicts is not None
            else self._selected_backend_verdicts(
                list(candidates), document_id=document_id, text=text
            )
        )
        if len(selected_backend_verdicts) != len(candidates):
            raise ValueError("selected backend returned wrong verdict count for comparison")
        verdicts_by_backend = {
            _backend_for_verifier(verifier): verifier.verify(
                list(candidates), document_id=document_id, text=text
            )
            for verifier in self._verifiers
        }
        for backend, verdicts in verdicts_by_backend.items():
            _validate_verdict_batch(candidates, verdicts, backend)
        records = []
        for idx, candidate in enumerate(candidates):
            backend_verdicts: list[VerificationVerdict] = []
            for verifier in self._verifiers:
                backend = _backend_for_verifier(verifier)
                verdict = (
                    selected_backend_verdicts[idx]
                    if backend is self.selected_backend and selected_verdicts is not None
                    else verdicts_by_backend[backend][idx]
                )
                backend_verdicts.append(verdict)
            selected_verdict = selected_backend_verdicts[idx]
            records.append(
                VerifierCandidateComparison(
                    candidate_id=candidate.candidate_id,
                    selected_verdict=selected_verdict,
                    backend_verdicts=backend_verdicts,
                    disagreement=any(
                        verdict.status is not selected_verdict.status
                        for verdict in backend_verdicts
                    ),
                )
            )
        return VerifierBatchComparison(
            selected_backend=self.selected_backend,
            total_candidates=len(candidates),
            records=records,
        )

    def _selected_backend_verdicts(
        self, candidates: list[EvidenceCandidate], *, document_id: str, text: str
    ) -> list[VerificationVerdict]:
        """Return the selected backend's verdicts from the configured verifier list."""
        for verifier in self._verifiers:
            if _backend_for_verifier(verifier) is self.selected_backend:
                verdicts = verifier.verify(candidates, document_id=document_id, text=text)
                _validate_verdict_batch(candidates, verdicts, self.selected_backend)
                return verdicts
        raise ValueError(f"selected backend {self.selected_backend!r} is not configured")


def load_verifier_calibration_examples(path: str | Path) -> list[CalibrationExample]:
    """Load calibration examples from a JSON list on disk."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("calibration example file must contain a JSON list")
    return [CalibrationExample.model_validate(item) for item in payload]


def recommend_calibration(
    verifier: VerifierAdapter,
    examples: Sequence[CalibrationExample],
    *,
    evidence_source: str,
) -> VerifierCalibrationRecommendation:
    """Recommend a calibration profile from evidence-backed examples."""
    if not examples:
        raise ValueError("calibration requires at least one example")

    backend = _backend_for_verifier(verifier)
    scores: list[VerificationScore | None] = []
    correct = 0
    for example in examples:
        candidate = _example_to_candidate(example)
        verdict = verifier.verify([candidate], document_id=example.document_id, text=example.text)
        _validate_verdict_batch([candidate], verdict, backend)
        scores.append(verdict[0].score)
        if verdict[0].status is example.expected_status:
            correct += 1
    metrics = VerifierCalibrationMetrics(
        evaluated_examples=len(examples),
        correct=correct,
        accuracy=(correct / len(examples)) if examples else 0.0,
    )
    thresholds = VerifierCalibrationThresholds()
    score_values = [score for score in scores if score is not None]
    if score_values:
        support_scores = [
            score.support_probability
            for example, score in zip(examples, scores, strict=True)
            if example.expected_status is VerificationStatus.SUPPORTED
            and score is not None
            and score.support_probability is not None
        ]
        nonsupport_scores = [
            score.support_probability
            for example, score in zip(examples, scores, strict=True)
            if example.expected_status is not VerificationStatus.SUPPORTED
            and score is not None
            and score.support_probability is not None
        ]
        thresholds.support_probability_min = min(support_scores) if support_scores else None
        thresholds.support_probability_max = max(nonsupport_scores) if nonsupport_scores else None
        if thresholds.support_probability_min is None and (
            thresholds.support_probability_max is not None
        ):
            thresholds.support_probability_min = thresholds.support_probability_max
        if thresholds.support_probability_max is None and (
            thresholds.support_probability_min is not None
        ):
            thresholds.support_probability_max = thresholds.support_probability_min
    calibration = VerifierCalibrationProfile(
        calibration_id=f"{backend.value}-calibration",
        backend=backend.value,
        evidence_source=evidence_source,
        sample_size=len(examples),
        thresholds=thresholds,
        notes=[],
    )
    return VerifierCalibrationRecommendation(
        backend=backend.value,
        model_name=getattr(verifier, "model_name", None),
        calibration=calibration,
        metrics=metrics,
    )


def _backend_for_verifier(verifier: VerifierAdapter) -> VerifierBackend:
    """Infer the backend enum for one verifier implementation."""
    if isinstance(verifier, OfflineBaselineVerifier):
        return VerifierBackend.OFFLINE
    if isinstance(verifier, MiniCheckVerifier):
        return VerifierBackend.MINICHECK
    if isinstance(verifier, CrossEncoderNliVerifier):
        return VerifierBackend.CROSSENCODER_NLI
    backend = getattr(verifier, "backend", None)
    if isinstance(backend, VerifierBackend):
        return backend
    if isinstance(backend, str):
        return VerifierBackend(backend)
    raise ValueError(f"unsupported verifier adapter: {type(verifier)!r}")


def _validate_verdict_batch(
    candidates: Sequence[EvidenceCandidate],
    verdicts: Sequence[VerificationVerdict],
    backend: VerifierBackend,
) -> None:
    """Reject backend outputs that do not align with the requested batch."""
    if len(verdicts) != len(candidates):
        raise ValueError(f"{backend.value} returned wrong verdict count")
    expected_ids = [candidate.candidate_id for candidate in candidates]
    actual_ids = [verdict.candidate_id for verdict in verdicts]
    if actual_ids != expected_ids:
        raise ValueError(f"{backend.value} returned verdicts for unexpected candidate ids")


def _example_to_candidate(example: CalibrationExample) -> EvidenceCandidate:
    """Convert a calibration example into a verifier candidate."""
    return EvidenceCandidate(
        candidate_id=example.example_id,
        label="claim",
        kind="claim",
        span=SpanRef(
            document_id=example.document_id,
            start_char=0,
            end_char=len(example.claim_text),
            text=example.claim_text,
        ),
        provenance=[example.example_id],
        evidence_refs=[example.example_id],
    )
