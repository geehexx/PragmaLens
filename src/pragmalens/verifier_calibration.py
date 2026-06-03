"""Verifier comparison and calibration helpers."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from pydantic import BaseModel, Field

from pragmalens.models import CandidateStatus, EvidenceCandidate, SpanRef, VerificationStatus
from pragmalens.verifier import (
    CrossEncoderModel,
    MiniCheckScorer,
    OfflineBaselineVerifier,
    VerifierAdapter,
)


class VerifierCalibrationCase(BaseModel):
    """Single claim/document pair with a gold verification label."""

    case_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    document_text: str = Field(min_length=1)
    claim_text: str = Field(min_length=1)
    gold_status: VerificationStatus


class VerifierCalibrationObservation(BaseModel):
    """Backend-level observation for one calibration case."""

    case_id: str = Field(min_length=1)
    backend: str = Field(min_length=1)
    gold_status: VerificationStatus
    predicted_status: VerificationStatus
    support_score: float | None = None
    support_threshold: float | None = None
    binary_support_prediction: bool | None = None
    binary_support_correct: bool | None = None
    rationale: str = Field(min_length=1)


class VerifierCalibrationSummary(BaseModel):
    """Aggregate calibration metrics for one backend."""

    backend: str = Field(min_length=1)
    total: int = Field(ge=0)
    exact_match_count: int = Field(ge=0)
    exact_match_rate: float = Field(ge=0.0, le=1.0)
    supported_gold_count: int = Field(ge=0)
    support_score_count: int = Field(ge=0)
    recommended_support_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    support_binary_accuracy: float | None = Field(default=None, ge=0.0, le=1.0)
    support_binary_f1: float | None = Field(default=None, ge=0.0, le=1.0)
    support_score_min: float | None = Field(default=None, ge=0.0, le=1.0)
    support_score_max: float | None = Field(default=None, ge=0.0, le=1.0)
    disagreement_case_ids: list[str] = Field(default_factory=list)


class VerifierCalibrationReport(BaseModel):
    """Combined calibration output across one or more verifier backends."""

    run_id: str = Field(min_length=1)
    cases: list[VerifierCalibrationCase] = Field(default_factory=list)
    observations: list[VerifierCalibrationObservation] = Field(default_factory=list)
    summaries: list[VerifierCalibrationSummary] = Field(default_factory=list)


def load_verifier_calibration_cases(path: str | Path) -> list[VerifierCalibrationCase]:
    """Load calibration cases from a JSON list on disk."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("calibration case file must contain a JSON list")
    return [VerifierCalibrationCase.model_validate(item) for item in payload]


def compare_verifier_backends(
    cases: Sequence[VerifierCalibrationCase],
    *,
    offline_verifier: VerifierAdapter | None = None,
    minicheck_scorer: MiniCheckScorer | None = None,
    crossencoder_model: CrossEncoderModel | None = None,
    crossencoder_label_mapping: Sequence[str] = (
        "contradiction",
        "entailment",
        "neutral",
    ),
    run_id: str = "verifier-calibration",
) -> VerifierCalibrationReport:
    """Compare all available verifier backends on the same calibration batch."""
    observations: list[VerifierCalibrationObservation] = []
    summaries: list[VerifierCalibrationSummary] = []
    if offline_verifier is None:
        offline_verifier = OfflineBaselineVerifier()

    if cases:
        offline_obs = _compare_offline_backend(offline_verifier, cases)
        observations.extend(offline_obs)
        summaries.append(_summarize_backend("offline", offline_obs))

    if minicheck_scorer is not None and cases:
        minicheck_obs = _compare_minicheck_backend(minicheck_scorer, cases)
        observations.extend(minicheck_obs)
        summaries.append(_summarize_backend("minicheck", minicheck_obs))

    if crossencoder_model is not None and cases:
        crossencoder_obs = _compare_crossencoder_backend(
            crossencoder_model,
            cases,
            label_mapping=crossencoder_label_mapping,
        )
        observations.extend(crossencoder_obs)
        summaries.append(_summarize_backend("crossencoder_nli", crossencoder_obs))

    return VerifierCalibrationReport(
        run_id=run_id,
        cases=list(cases),
        observations=observations,
        summaries=summaries,
    )


def _compare_offline_backend(
    verifier: VerifierAdapter, cases: Sequence[VerifierCalibrationCase]
) -> list[VerifierCalibrationObservation]:
    candidates_by_case = [_case_to_candidate(case) for case in cases]
    observations: list[VerifierCalibrationObservation] = []
    for case, candidate in zip(cases, candidates_by_case, strict=True):
        verdict = verifier.verify(
            [candidate], document_id=case.document_id, text=case.document_text
        )
        if len(verdict) != 1:
            raise ValueError("offline verifier returned wrong verdict count for calibration case")
        observations.append(
            VerifierCalibrationObservation(
                case_id=case.case_id,
                backend="offline",
                gold_status=case.gold_status,
                predicted_status=verdict[0].status,
                rationale=verdict[0].rationale,
                support_score=None,
                support_threshold=None,
                binary_support_prediction=None,
                binary_support_correct=None,
            )
        )
    return observations


def _compare_minicheck_backend(
    scorer: MiniCheckScorer, cases: Sequence[VerifierCalibrationCase]
) -> list[VerifierCalibrationObservation]:
    docs = [case.document_text for case in cases]
    claims = [case.claim_text for case in cases]
    pred_labels, raw_prob, _, _ = scorer.score(docs=docs, claims=claims)
    if len(pred_labels) != len(cases) or len(raw_prob) != len(cases):
        raise ValueError("MiniCheck scorer returned wrong result count for calibration cases")
    scores = [float(prob) for prob in raw_prob]
    threshold = _recommend_support_threshold(
        [case.gold_status is VerificationStatus.SUPPORTED for case in cases],
        scores,
    )
    observations: list[VerifierCalibrationObservation] = []
    for case, pred_label, support_score in zip(cases, pred_labels, scores, strict=True):
        predicted_status = (
            VerificationStatus.SUPPORTED
            if _coerce_supported_label(pred_label)
            else VerificationStatus.UNSUPPORTED
        )
        binary_support_prediction = support_score >= threshold
        gold_supported = case.gold_status is VerificationStatus.SUPPORTED
        observations.append(
            VerifierCalibrationObservation(
                case_id=case.case_id,
                backend="minicheck",
                gold_status=case.gold_status,
                predicted_status=predicted_status,
                rationale=f"MiniCheck support score {support_score:.3f}",
                support_score=support_score,
                support_threshold=threshold,
                binary_support_prediction=binary_support_prediction,
                binary_support_correct=binary_support_prediction == gold_supported,
            )
        )
    return observations


def _compare_crossencoder_backend(
    model: CrossEncoderModel,
    cases: Sequence[VerifierCalibrationCase],
    *,
    label_mapping: Sequence[str],
) -> list[VerifierCalibrationObservation]:
    pairs = [(case.document_text, case.claim_text) for case in cases]
    try:
        score_rows = model.predict(pairs, apply_softmax=True)
    except TypeError:
        score_rows = model.predict(pairs)
    if len(score_rows) != len(cases):
        raise ValueError("CrossEncoder model returned wrong result count for calibration cases")
    label_mapping = tuple(label_mapping)
    entailment_index = _label_index(label_mapping, "entailment")
    scores: list[float] = []
    for row in score_rows:
        if len(row) <= entailment_index:
            raise ValueError("CrossEncoder entailment score missing from calibration row")
        scores.append(float(row[entailment_index]))
    threshold = _recommend_support_threshold(
        [case.gold_status is VerificationStatus.SUPPORTED for case in cases],
        scores,
    )
    observations: list[VerifierCalibrationObservation] = []
    for case, row, support_score in zip(cases, score_rows, scores, strict=True):
        label = _resolve_nli_label(row, label_mapping=label_mapping)
        predicted_status = _status_for_nli_label(label)
        binary_support_prediction = support_score >= threshold
        gold_supported = case.gold_status is VerificationStatus.SUPPORTED
        observations.append(
            VerifierCalibrationObservation(
                case_id=case.case_id,
                backend="crossencoder_nli",
                gold_status=case.gold_status,
                predicted_status=predicted_status,
                rationale=f"CrossEncoder predicted {label}",
                support_score=support_score,
                support_threshold=threshold,
                binary_support_prediction=binary_support_prediction,
                binary_support_correct=binary_support_prediction == gold_supported,
            )
        )
    return observations


def _summarize_backend(
    backend: str, observations: Sequence[VerifierCalibrationObservation]
) -> VerifierCalibrationSummary:
    """Aggregate calibration metrics for a backend-specific observation set."""
    total = len(observations)
    exact_match_count = sum(1 for obs in observations if obs.predicted_status is obs.gold_status)
    support_score_values = [
        obs.support_score for obs in observations if obs.support_score is not None
    ]
    support_gold_count = sum(
        1 for obs in observations if obs.gold_status is VerificationStatus.SUPPORTED
    )
    support_threshold = observations[0].support_threshold if observations else None
    support_accuracy = None
    support_f1 = None
    if support_score_values:
        support_score_observations = [obs for obs in observations if obs.support_score is not None]
        gold_supports = [
            obs.gold_status is VerificationStatus.SUPPORTED for obs in support_score_observations
        ]
        support_score_values = [
            cast(float, obs.support_score) for obs in support_score_observations
        ]
        support_accuracy, support_f1 = _support_metrics(
            support_score_values,
            gold_supports,
        )
        support_threshold = _recommend_support_threshold(
            gold_supports,
            support_score_values,
        )
    disagreement_case_ids = [
        obs.case_id for obs in observations if obs.predicted_status is not obs.gold_status
    ]
    return VerifierCalibrationSummary(
        backend=backend,
        total=total,
        exact_match_count=exact_match_count,
        exact_match_rate=(exact_match_count / total) if total else 0.0,
        supported_gold_count=support_gold_count,
        support_score_count=len(support_score_values),
        recommended_support_threshold=support_threshold,
        support_binary_accuracy=support_accuracy,
        support_binary_f1=support_f1,
        support_score_min=min(support_score_values) if support_score_values else None,
        support_score_max=max(support_score_values) if support_score_values else None,
        disagreement_case_ids=disagreement_case_ids,
    )


def _recommend_support_threshold(
    gold_supports: Sequence[bool], support_scores: Sequence[float]
) -> float:
    """Pick the support threshold that best matches gold support labels."""
    thresholds = [index / 20 for index in range(1, 20)]
    best_threshold = 0.5
    best_key = (-1.0, -1.0, -1.0)
    for threshold in thresholds:
        accuracy, f1 = _support_metrics(support_scores, gold_supports, threshold=threshold)
        key = (f1, accuracy, -abs(threshold - 0.5))
        if key > best_key:
            best_key = key
            best_threshold = threshold
    return best_threshold


def _support_metrics(
    support_scores: Sequence[float],
    gold_supports: Sequence[bool],
    *,
    threshold: float = 0.5,
) -> tuple[float, float]:
    """Compute accuracy and F1 for a binary support threshold."""
    if len(support_scores) != len(gold_supports):
        raise ValueError("support score and gold-support lengths do not match")
    if not support_scores:
        return 0.0, 0.0
    predicted_supports = [score >= threshold for score in support_scores]
    tp = sum(pred and gold for pred, gold in zip(predicted_supports, gold_supports, strict=True))
    tn = sum(
        (not pred) and (not gold)
        for pred, gold in zip(predicted_supports, gold_supports, strict=True)
    )
    fp = sum(
        pred and (not gold) for pred, gold in zip(predicted_supports, gold_supports, strict=True)
    )
    fn = sum(
        (not pred) and gold for pred, gold in zip(predicted_supports, gold_supports, strict=True)
    )
    accuracy = (tp + tn) / len(support_scores)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 0.0
    if precision + recall:
        f1 = 2 * precision * recall / (precision + recall)
    return accuracy, f1


def _case_to_candidate(case: VerifierCalibrationCase) -> EvidenceCandidate:
    """Convert a calibration case into the product's verification candidate shape."""
    return EvidenceCandidate(
        candidate_id=case.case_id,
        label="claim",
        kind="claim",
        status=CandidateStatus.VALID,
        span=SpanRef(
            document_id=case.document_id,
            start_char=0,
            end_char=len(case.claim_text),
            text=case.claim_text,
        ),
        provenance=[case.case_id],
        evidence_refs=[case.case_id],
    )


def _coerce_supported_label(value: object) -> bool:
    """Interpret MiniCheck-style binary labels."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return int(value) == 1
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "supported", "entailment"}:
            return True
        if normalized in {"0", "false", "unsupported", "contradiction"}:
            return False
    raise ValueError(f"Unsupported MiniCheck label value: {value!r}")


def _status_for_nli_label(label: str) -> VerificationStatus:
    """Map contradiction/entailment/neutral labels into product verdicts."""
    normalized = label.strip().lower()
    if normalized == "entailment":
        return VerificationStatus.SUPPORTED
    if normalized == "contradiction":
        return VerificationStatus.UNSUPPORTED
    if normalized == "neutral":
        return VerificationStatus.INSUFFICIENT_EVIDENCE
    raise ValueError(f"Unsupported NLI label: {label!r}")


def _resolve_nli_label(row: Sequence[float], *, label_mapping: Sequence[str]) -> str:
    """Resolve the winning label for one NLI score row."""
    if len(row) == 0:
        raise ValueError("CrossEncoder model returned an empty score row")
    max_index = max(range(len(row)), key=lambda index: row[index])
    try:
        return label_mapping[max_index]
    except IndexError as exc:
        raise ValueError("CrossEncoder label mapping does not match score width") from exc


def _label_index(label_mapping: Sequence[str], target: str) -> int:
    """Return the configured index for a label or fail fast."""
    try:
        return tuple(label_mapping).index(target)
    except ValueError as exc:
        raise ValueError(f"CrossEncoder label mapping does not include {target!r}") from exc
