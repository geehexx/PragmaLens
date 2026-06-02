from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from pragmalens.models import EvidenceCandidate, VerificationStatus, VerificationVerdict


class VerifierAdapter(Protocol):
    """Batch verifier boundary used by the verification stage."""

    def verify(
        self,
        candidates: list[EvidenceCandidate],
        *,
        document_id: str,
        text: str,
    ) -> list[VerificationVerdict]: ...


class MiniCheckScorer(Protocol):
    """Minimal MiniCheck-like scorer surface used by the adapter."""

    def score(
        self,
        *,
        docs: list[str],
        claims: list[str],
    ) -> tuple[Sequence[Any], Sequence[Any], Any, Any]: ...


class CrossEncoderModel(Protocol):
    """Minimal CrossEncoder-like inference surface used by the adapter."""

    def predict(self, pairs: list[tuple[str, str]]) -> Sequence[Sequence[float]]: ...


class OfflineBaselineVerifier:
    """Deterministic offline verifier used until live adapters are introduced."""

    def verify(
        self,
        candidates: list[EvidenceCandidate],
        *,
        document_id: str,
        text: str,
    ) -> list[VerificationVerdict]:
        """Return baseline insufficient-evidence verdicts for each candidate."""
        del document_id, text
        return [
            VerificationVerdict(
                candidate_id=candidate.candidate_id,
                status=VerificationStatus.INSUFFICIENT_EVIDENCE,
                rationale=(
                    "offline baseline verifier requires external evidence before support claims"
                ),
                evidence_ids=candidate.evidence_refs,
            )
            for candidate in candidates
        ]


class MiniCheckVerifier:
    """Adapter for MiniCheck-style binary document/claim verification."""

    def __init__(self, scorer: MiniCheckScorer) -> None:
        """Capture an injected MiniCheck-compatible scorer."""
        self._scorer = scorer

    def verify(
        self,
        candidates: list[EvidenceCandidate],
        *,
        document_id: str,
        text: str,
    ) -> list[VerificationVerdict]:
        """Map MiniCheck-style binary outputs into verifier verdicts."""
        del document_id
        docs = [text for _ in candidates]
        claims = [candidate.span.text for candidate in candidates]
        pred_labels, raw_prob, _, _ = self._scorer.score(docs=docs, claims=claims)
        if len(pred_labels) != len(candidates) or len(raw_prob) != len(candidates):
            raise ValueError("MiniCheck scorer returned wrong result count")
        verdicts: list[VerificationVerdict] = []
        for candidate, pred_label, prob in zip(candidates, pred_labels, raw_prob, strict=True):
            supported = _coerce_supported_label(pred_label)
            verdicts.append(
                VerificationVerdict(
                    candidate_id=candidate.candidate_id,
                    status=(
                        VerificationStatus.SUPPORTED
                        if supported
                        else VerificationStatus.UNSUPPORTED
                    ),
                    rationale=f"MiniCheck provisional baseline scored support={float(prob):.3f}",
                    evidence_ids=candidate.evidence_refs,
                )
            )
        return verdicts


class CrossEncoderNliVerifier:
    """Adapter for NLI-style CrossEncoder models with contradiction/entailment labels."""

    def __init__(
        self,
        model: CrossEncoderModel,
        *,
        label_mapping: Sequence[str] = ("contradiction", "entailment", "neutral"),
    ) -> None:
        """Store the injected CrossEncoder-like model and label order."""
        self._model = model
        self._label_mapping = tuple(label_mapping)

    def verify(
        self,
        candidates: list[EvidenceCandidate],
        *,
        document_id: str,
        text: str,
    ) -> list[VerificationVerdict]:
        """Map CrossEncoder NLI scores into verifier verdicts."""
        del document_id
        pairs = [(text, candidate.span.text) for candidate in candidates]
        score_rows = self._model.predict(pairs)
        if len(score_rows) != len(candidates):
            raise ValueError("CrossEncoder model returned wrong result count")
        verdicts: list[VerificationVerdict] = []
        for candidate, row in zip(candidates, score_rows, strict=True):
            label = self._resolve_nli_label(row)
            verdicts.append(
                VerificationVerdict(
                    candidate_id=candidate.candidate_id,
                    status=_status_for_nli_label(label),
                    rationale=f"CrossEncoder provisional baseline predicted {label}",
                    evidence_ids=candidate.evidence_refs,
                )
            )
        return verdicts

    def _resolve_nli_label(self, row: Sequence[float]) -> str:
        """Resolve the winning label for one NLI score row."""
        if not row:
            raise ValueError("CrossEncoder model returned an empty score row")
        max_index = max(range(len(row)), key=lambda index: row[index])
        try:
            return self._label_mapping[max_index]
        except IndexError as exc:
            raise ValueError("CrossEncoder label mapping does not match score width") from exc


def _coerce_supported_label(value: Any) -> bool:
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
