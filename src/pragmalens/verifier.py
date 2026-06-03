"""Verifier adapters and runtime builders for offline and optional live backends."""

from __future__ import annotations

import os
from collections.abc import Sequence
from enum import StrEnum
from typing import Any, Protocol

from pragmalens.models import EvidenceCandidate, VerificationStatus, VerificationVerdict


class VerifierBackend(StrEnum):
    """Supported verifier runtime selections."""

    OFFLINE = "offline"
    MINICHECK = "minicheck"
    CROSSENCODER_NLI = "crossencoder_nli"


class VerifierAdapter(Protocol):
    """Batch verifier boundary used by the verification stage."""

    def verify(
        self,
        candidates: list[EvidenceCandidate],
        *,
        document_id: str,
        text: str,
    ) -> list[VerificationVerdict]:
        """Return one verifier verdict per candidate in the original request order."""
        ...


class MiniCheckScorer(Protocol):
    """Minimal MiniCheck-like scorer surface used by the adapter."""

    def score(
        self,
        *,
        docs: list[str],
        claims: list[str],
    ) -> tuple[Sequence[Any], Sequence[Any], Any, Any]:
        """Score one or more document/claim pairs."""
        ...


class CrossEncoderModel(Protocol):
    """Minimal CrossEncoder-like inference surface used by the adapter."""

    def predict(self, pairs: list[tuple[str, str]]) -> Sequence[Sequence[float]]:
        """Return per-label score rows for each text/claim pair."""
        ...


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
        if len(row) == 0:
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


def build_verifier_adapter(
    backend: VerifierBackend | str,
    *,
    model_name: str | None = None,
    cache_dir: str | None = None,
    label_mapping: Sequence[str] | None = None,
) -> VerifierAdapter:
    """Build a verifier adapter from a backend selector and optional runtime config."""
    selected = VerifierBackend(backend)
    if selected is VerifierBackend.OFFLINE:
        return OfflineBaselineVerifier()
    if selected is VerifierBackend.MINICHECK:
        return _build_minicheck_from_runtime(
            model_name=model_name or "roberta-large",
            cache_dir=cache_dir or ".local_state/minicheck-cache",
        )
    if selected is VerifierBackend.CROSSENCODER_NLI:
        return _build_crossencoder_from_runtime(
            model_name=model_name or "cross-encoder/nli-deberta-v3-base",
            label_mapping=label_mapping or ("contradiction", "entailment", "neutral"),
        )
    raise ValueError(f"Unsupported verifier backend: {backend!r}")


def build_verifier_from_env() -> VerifierAdapter:
    """Build a verifier adapter from environment configuration."""
    backend = os.environ.get("PRAGMALENS_VERIFIER", VerifierBackend.OFFLINE)
    if backend == VerifierBackend.MINICHECK:
        return build_verifier_adapter(
            backend,
            model_name=os.environ.get("PRAGMALENS_MINICHECK_MODEL", "roberta-large"),
            cache_dir=os.environ.get(
                "PRAGMALENS_MINICHECK_CACHE_DIR",
                ".local_state/minicheck-cache",
            ),
        )
    if backend == VerifierBackend.CROSSENCODER_NLI:
        return build_verifier_adapter(
            backend,
            model_name=os.environ.get(
                "PRAGMALENS_CROSSENCODER_MODEL",
                "cross-encoder/nli-deberta-v3-base",
            ),
        )
    return build_verifier_adapter(VerifierBackend.OFFLINE)


def _build_minicheck_from_runtime(*, model_name: str, cache_dir: str) -> VerifierAdapter:
    """Instantiate a MiniCheck-backed verifier from real runtime dependencies."""
    from minicheck.minicheck import MiniCheck

    return MiniCheckVerifier(MiniCheck(model_name=model_name, cache_dir=cache_dir))


def _build_crossencoder_from_runtime(
    *, model_name: str, label_mapping: Sequence[str]
) -> VerifierAdapter:
    """Instantiate a CrossEncoder-backed verifier from real runtime dependencies."""
    from sentence_transformers import CrossEncoder

    return CrossEncoderNliVerifier(CrossEncoder(model_name), label_mapping=label_mapping)
