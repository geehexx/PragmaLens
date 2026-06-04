"""Verifier adapters and runtime builders for offline and optional live backends."""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from functools import lru_cache
from importlib import import_module
from math import exp
from typing import TYPE_CHECKING, Any, Protocol

from pragmalens.live_runtime import load_crossencoder_model
from pragmalens.models import (
    EvidenceCandidate,
    VerificationScore,
    VerificationStatus,
    VerificationVerdict,
    VerifierCalibrationProfile,
    VerifierCalibrationThresholds,
)
from pragmalens.settings import load_settings

if TYPE_CHECKING:
    from pragmalens.verifier_comparison import VerifierComparisonHarness


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

    def predict(
        self,
        pairs: list[tuple[str, str]],
        *,
        apply_softmax: bool = False,
    ) -> Sequence[Sequence[float]]:
        """Return per-label score rows for each text/claim pair."""
        ...


def _backend_for_verifier(verifier: VerifierAdapter) -> VerifierBackend:
    """Infer a concrete backend enum from one verifier implementation."""
    backend = getattr(verifier, "backend", None)
    if isinstance(backend, VerifierBackend):
        return backend
    if isinstance(backend, str):
        return VerifierBackend(backend)
    raise ValueError(f"unsupported verifier adapter: {type(verifier)!r}")


class OfflineBaselineVerifier:
    """Deterministic offline verifier used until live adapters are introduced."""

    backend = VerifierBackend.OFFLINE
    model_name = "offline-baseline"
    calibration = VerifierCalibrationProfile(
        calibration_id="offline-baseline-v0_1",
        backend=VerifierBackend.OFFLINE,
        evidence_source="deterministic product baseline",
        sample_size=1,
        thresholds=VerifierCalibrationThresholds(),
        notes=["offline baseline always emits insufficient_evidence verdicts"],
    )

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
                backend=self.backend,
                model_name=self.model_name,
                calibration_id=self.calibration.calibration_id,
            )
            for candidate in candidates
        ]


class SignalEnsembleVerifier:
    """Conservative composite verifier over multiple backend verdict streams."""

    backend = "signal_ensemble"
    model_name = "signal-ensemble"

    def __init__(
        self,
        verifiers: Sequence[VerifierAdapter],
        *,
        selected_backend: VerifierBackend | str | None = None,
    ) -> None:
        """Preserve the configured verifier order for deterministic combination."""
        self._verifiers = list(verifiers)
        if not self._verifiers:
            raise ValueError("signal ensemble requires at least one verifier")
        self.selected_backend = (
            VerifierBackend(selected_backend) if selected_backend is not None else None
        )
        if self.selected_backend is not None:
            configured_backends = [_backend_for_verifier(verifier) for verifier in self._verifiers]
            if self.selected_backend not in configured_backends:
                raise ValueError(f"selected backend {self.selected_backend!r} is not configured")
        self._last_verdicts_by_backend: dict[VerifierBackend, list[VerificationVerdict]] = {}

    def verify(
        self,
        candidates: list[EvidenceCandidate],
        *,
        document_id: str,
        text: str,
    ) -> list[VerificationVerdict]:
        """Combine backend verdicts using the current conservative precedence rules."""
        batches_by_backend: dict[VerifierBackend, list[VerificationVerdict]] = {}
        batches = []
        for verifier in self._verifiers:
            backend = _backend_for_verifier(verifier)
            batch = self._validated_verdict_batch(
                verifier.verify(candidates, document_id=document_id, text=text),
                candidates,
            )
            batches_by_backend[backend] = batch
            batches.append(batch)
        self._last_verdicts_by_backend = batches_by_backend
        verdicts: list[VerificationVerdict] = []
        for index, candidate in enumerate(candidates):
            candidate_verdicts = [batch[index] for batch in batches]
            selected_status = _merge_verification_statuses(
                [verdict.status for verdict in candidate_verdicts]
            )
            verdicts.append(
                VerificationVerdict(
                    candidate_id=candidate.candidate_id,
                    status=selected_status,
                    rationale=_build_ensemble_rationale(candidate_verdicts),
                    evidence_ids=_merge_evidence_ids(candidate, candidate_verdicts),
                    backend=self.backend,
                    model_name=self.model_name,
                    score=_selected_score(candidate_verdicts, selected_status),
                )
            )
        return verdicts

    def selected_backend_adapter(self) -> VerifierAdapter:
        """Expose the concrete backend selected for comparison and calibration."""
        if self.selected_backend is None:
            raise ValueError("signal ensemble does not declare a selected backend")
        for verifier in self._verifiers:
            if _backend_for_verifier(verifier) is self.selected_backend:
                return verifier
        raise ValueError(f"selected backend {self.selected_backend!r} is not configured")

    def selected_backend_verdicts(self) -> list[VerificationVerdict]:
        """Return the cached verdicts for the selected concrete backend."""
        if self.selected_backend is None:
            raise ValueError("signal ensemble does not declare a selected backend")
        verdicts = self._last_verdicts_by_backend.get(self.selected_backend)
        if verdicts is None:
            raise ValueError("signal ensemble has no cached selected backend verdicts")
        return list(verdicts)

    @staticmethod
    def _validated_verdict_batch(
        verdicts: list[VerificationVerdict], candidates: list[EvidenceCandidate]
    ) -> list[VerificationVerdict]:
        """Reject backend outputs that do not align with the requested candidate batch."""
        _validate_verdict_batch(candidates, verdicts)
        return verdicts


class MiniCheckVerifier:
    """Adapter for MiniCheck-style binary document/claim verification."""

    backend = VerifierBackend.MINICHECK

    def __init__(
        self,
        scorer: MiniCheckScorer,
        *,
        model_name: str = "roberta-large",
        calibration: VerifierCalibrationProfile | None = None,
    ) -> None:
        """Capture an injected MiniCheck-compatible scorer."""
        self._scorer = scorer
        self.model_name = model_name
        self.calibration = calibration or _default_minicheck_calibration()

    @property
    def scorer(self) -> MiniCheckScorer:
        """Expose the injected scorer for calibration and benchmark helpers."""
        return self._scorer

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
            score = VerificationScore(
                predicted_label="supported" if supported else "unsupported",
                support_probability=float(prob),
            )
            verdicts.append(
                VerificationVerdict(
                    candidate_id=candidate.candidate_id,
                    status=_status_for_minicheck(score, self.calibration),
                    rationale=f"MiniCheck scored support={float(prob):.3f}",
                    evidence_ids=candidate.evidence_refs,
                    backend=self.backend,
                    model_name=self.model_name,
                    score=score,
                    calibration_id=self.calibration.calibration_id,
                )
            )
        return verdicts


class CrossEncoderNliVerifier:
    """Adapter for NLI-style CrossEncoder models with contradiction/entailment labels."""

    backend = VerifierBackend.CROSSENCODER_NLI

    def __init__(
        self,
        model: CrossEncoderModel,
        *,
        model_name: str = "cross-encoder/nli-deberta-v3-base",
        label_mapping: Sequence[str] = ("contradiction", "entailment", "neutral"),
        calibration: VerifierCalibrationProfile | None = None,
    ) -> None:
        """Store the injected CrossEncoder-like model and label order."""
        self._model = model
        self._label_mapping = tuple(label_mapping)
        self.model_name = model_name
        self.calibration = calibration or _default_crossencoder_calibration()

    @property
    def model(self) -> CrossEncoderModel:
        """Expose the injected model for calibration and benchmark helpers."""
        return self._model

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
        score_rows = self._predict_score_rows(pairs)
        if len(score_rows) != len(candidates):
            raise ValueError("CrossEncoder model returned wrong result count")
        verdicts: list[VerificationVerdict] = []
        for candidate, row in zip(candidates, score_rows, strict=True):
            score = self._build_score(row)
            label = self._resolve_nli_label(row)
            verdicts.append(
                VerificationVerdict(
                    candidate_id=candidate.candidate_id,
                    status=_status_for_crossencoder(score, self.calibration, label=label),
                    rationale=f"CrossEncoder predicted {label}",
                    evidence_ids=candidate.evidence_refs,
                    backend=self.backend,
                    model_name=self.model_name,
                    score=score,
                    calibration_id=self.calibration.calibration_id,
                )
            )
        return verdicts

    def _predict_score_rows(self, pairs: list[tuple[str, str]]) -> Sequence[Sequence[float]]:
        """Prefer backend-side softmax, but remain compatible with narrow test doubles."""
        try:
            return self._model.predict(pairs, apply_softmax=True)
        except TypeError:
            return self._model.predict(pairs)

    def _build_score(self, row: Sequence[float]) -> VerificationScore:
        """Normalize one NLI score row into explicit probability fields."""
        probabilities = _normalize_score_row(row)
        label = self._resolve_nli_label(probabilities)
        values = {
            label_name: probabilities[index] for index, label_name in enumerate(self._label_mapping)
        }
        return VerificationScore(
            predicted_label=label,
            contradiction_probability=values.get("contradiction"),
            entailment_probability=values.get("entailment"),
            neutral_probability=values.get("neutral"),
            normalized=True,
        )

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


def _status_for_minicheck(
    score: VerificationScore, calibration: VerifierCalibrationProfile
) -> VerificationStatus:
    """Map MiniCheck support probabilities into product verdicts."""
    probability = score.support_probability
    if probability is None:
        raise ValueError("MiniCheck verdict missing support_probability")
    thresholds = calibration.thresholds
    if (
        thresholds.support_probability_min is not None
        and probability >= thresholds.support_probability_min
    ):
        return VerificationStatus.SUPPORTED
    if (
        thresholds.support_probability_max is not None
        and probability <= thresholds.support_probability_max
    ):
        return VerificationStatus.UNSUPPORTED
    return VerificationStatus.INSUFFICIENT_EVIDENCE


def _status_for_crossencoder(
    score: VerificationScore,
    calibration: VerifierCalibrationProfile,
    *,
    label: str,
) -> VerificationStatus:
    """Map normalized NLI probabilities into product verdicts."""
    thresholds = calibration.thresholds
    entailment = score.entailment_probability
    contradiction = score.contradiction_probability
    neutral = score.neutral_probability
    if entailment is None or contradiction is None or neutral is None:
        raise ValueError("CrossEncoder verdict missing normalized probabilities")
    if (
        thresholds.entailment_probability_min is not None
        and entailment >= thresholds.entailment_probability_min
        and entailment >= contradiction
        and entailment >= neutral
    ):
        return VerificationStatus.SUPPORTED
    if (
        thresholds.contradiction_probability_min is not None
        and contradiction >= thresholds.contradiction_probability_min
        and contradiction >= entailment
        and contradiction >= neutral
    ):
        return VerificationStatus.UNSUPPORTED
    if label.strip().lower() == "neutral":
        return VerificationStatus.INSUFFICIENT_EVIDENCE
    return VerificationStatus.INSUFFICIENT_EVIDENCE


def _default_minicheck_calibration() -> VerifierCalibrationProfile:
    """Return the current provisional MiniCheck decision thresholds."""
    return VerifierCalibrationProfile(
        calibration_id="minicheck-provisional-v0_1",
        backend=VerifierBackend.MINICHECK,
        evidence_source=(
            "tests/live_smoke/test_runtime_backends.py::"
            "test_minicheck_live_smoke_scores_supported_vs_unsupported_claims"
        ),
        sample_size=2,
        thresholds=VerifierCalibrationThresholds(
            support_probability_min=0.75,
            support_probability_max=0.25,
        ),
        notes=["conservative provisional gap around the live smoke support boundary"],
    )


def _default_crossencoder_calibration() -> VerifierCalibrationProfile:
    """Return the current provisional CrossEncoder NLI decision thresholds."""
    return VerifierCalibrationProfile(
        calibration_id="crossencoder-nli-provisional-v0_1",
        backend=VerifierBackend.CROSSENCODER_NLI,
        evidence_source=(
            "tests/live_smoke/test_runtime_backends.py::"
            "test_crossencoder_live_smoke_maps_supported_and_unsupported_claims"
        ),
        sample_size=2,
        thresholds=VerifierCalibrationThresholds(
            entailment_probability_min=0.5,
            contradiction_probability_min=0.5,
        ),
        notes=["provisional thresholds derived from the current two-example live smoke lane"],
    )


def _normalize_score_row(row: Sequence[float]) -> list[float]:
    """Convert raw multi-class scores into probabilities without double-normalizing."""
    if len(row) == 0:
        raise ValueError("CrossEncoder model returned an empty score row")
    values = [float(value) for value in row]
    total = sum(values)
    if all(0.0 <= value <= 1.0 for value in values) and abs(total - 1.0) <= 1e-6:
        return values
    max_value = max(values)
    exp_values = [exp(value - max_value) for value in values]
    exp_total = sum(exp_values)
    if exp_total == 0.0:
        raise ValueError("CrossEncoder model returned a degenerate score row")
    return [value / exp_total for value in exp_values]


def build_verifier_adapter(
    backend: VerifierBackend | str,
    *,
    model_name: str | None = None,
    cache_dir: str | None = None,
    label_mapping: Sequence[str] | None = None,
) -> VerifierAdapter:
    """Build a verifier adapter from a backend selector and optional runtime config."""
    try:
        selected = VerifierBackend(backend)
    except ValueError as exc:
        raise ValueError(f"Unsupported verifier backend: {backend!r}") from exc
    if selected is VerifierBackend.OFFLINE:
        return OfflineBaselineVerifier()
    if selected is VerifierBackend.MINICHECK:
        settings = load_settings() if model_name is None or cache_dir is None else None
        return _build_minicheck_from_runtime(
            model_name=model_name or (settings.minicheck_model if settings else "roberta-large"),
            cache_dir=cache_dir
            or (
                str(settings.minicheck_cache_dir)
                if settings is not None
                else ".local_state/minicheck-cache"
            ),
        )
    if selected is VerifierBackend.CROSSENCODER_NLI:
        settings = load_settings() if model_name is None or cache_dir is None else None
        return _build_crossencoder_from_runtime(
            model_name=model_name
            or (
                settings.crossencoder_model
                if settings is not None
                else "cross-encoder/nli-deberta-v3-base"
            ),
            cache_dir=cache_dir
            or (
                str(settings.crossencoder_cache_dir)
                if settings is not None
                else ".local_state/crossencoder-cache"
            ),
            revision=(
                settings.crossencoder_revision
                if settings is not None
                else "f2f24f9fce8fc5b34aedf861f5c819c6ba0cf4f5"
            ),
            label_mapping=label_mapping or ("contradiction", "entailment", "neutral"),
        )
    raise ValueError(f"Unsupported verifier backend: {backend!r}")


def build_default_verifier_runtime(
    backend: VerifierBackend | str | None = None,
) -> tuple[VerifierAdapter, VerifierComparisonHarness | None]:
    """Build the default verifier and optional comparison harness for one backend choice."""
    selected = _resolve_backend_selector(backend)
    if selected is VerifierBackend.OFFLINE:
        return OfflineBaselineVerifier(), None

    offline = OfflineBaselineVerifier()
    live = build_verifier_adapter(selected)
    verifiers = [offline, live]
    comparison_module = import_module("pragmalens.verifier_comparison")
    return (
        SignalEnsembleVerifier(verifiers, selected_backend=selected),
        comparison_module.VerifierComparisonHarness(
            selected_backend=selected,
            verifiers=verifiers,
        ),
    )


def build_verifier_from_env() -> VerifierAdapter:
    """Build a verifier adapter from environment configuration."""
    return build_verifier_adapter(_resolve_backend_selector(None))


def _build_minicheck_from_runtime(*, model_name: str, cache_dir: str) -> VerifierAdapter:
    """Instantiate a MiniCheck-backed verifier from real runtime dependencies."""
    return _load_minicheck_verifier(model_name=model_name, cache_dir=cache_dir)


def _build_crossencoder_from_runtime(
    *, model_name: str, cache_dir: str, revision: str, label_mapping: Sequence[str]
) -> VerifierAdapter:
    """Instantiate a CrossEncoder-backed verifier from real runtime dependencies."""
    return _load_crossencoder_verifier(
        model_name=model_name,
        cache_dir=cache_dir,
        revision=revision,
        label_mapping=tuple(label_mapping),
    )


@lru_cache(maxsize=8)
def _load_minicheck_verifier(*, model_name: str, cache_dir: str) -> VerifierAdapter:
    """Load and cache a MiniCheck-backed verifier instance."""
    minicheck_module = import_module("minicheck.minicheck")
    return MiniCheckVerifier(
        minicheck_module.MiniCheck(model_name=model_name, cache_dir=cache_dir),
        model_name=model_name,
    )


@lru_cache(maxsize=8)
def _load_crossencoder_verifier(
    *, model_name: str, cache_dir: str, revision: str, label_mapping: tuple[str, ...]
) -> VerifierAdapter:
    """Load and cache a CrossEncoder-backed verifier instance."""
    return CrossEncoderNliVerifier(
        load_crossencoder_model(model_name=model_name, cache_dir=cache_dir, revision=revision),
        model_name=model_name,
        label_mapping=label_mapping,
    )


def _resolve_backend_selector(backend: VerifierBackend | str | None) -> VerifierBackend:
    """Resolve an explicit backend selector or the default environment selector."""
    raw_backend = backend if backend is not None else load_settings().verifier_backend
    try:
        return VerifierBackend(raw_backend)
    except ValueError as exc:
        raise ValueError(f"Unsupported verifier backend: {raw_backend!r}") from exc


def _merge_verification_statuses(statuses: Sequence[VerificationStatus]) -> VerificationStatus:
    """Collapse multiple verifier statuses into one product-facing verdict."""
    has_supported = any(status is VerificationStatus.SUPPORTED for status in statuses)
    has_unsupported = any(status is VerificationStatus.UNSUPPORTED for status in statuses)
    if has_supported and has_unsupported:
        return VerificationStatus.INSUFFICIENT_EVIDENCE
    if has_supported:
        return VerificationStatus.SUPPORTED
    if has_unsupported:
        return VerificationStatus.UNSUPPORTED
    if any(status is VerificationStatus.INSUFFICIENT_EVIDENCE for status in statuses):
        return VerificationStatus.INSUFFICIENT_EVIDENCE
    if any(status is VerificationStatus.VERIFIER_ERROR for status in statuses):
        return VerificationStatus.VERIFIER_ERROR
    return VerificationStatus.INSUFFICIENT_EVIDENCE


def _build_ensemble_rationale(verdicts: Sequence[VerificationVerdict]) -> str:
    """Render one concise rationale string from member backend outcomes."""
    return "; ".join(f"{verdict.backend}={verdict.status.value}" for verdict in verdicts)


def _merge_evidence_ids(
    candidate: EvidenceCandidate,
    verdicts: Sequence[VerificationVerdict],
) -> list[str]:
    """Preserve candidate evidence order while unioning member evidence ids."""
    merged = list(candidate.evidence_refs)
    for verdict in verdicts:
        for evidence_id in verdict.evidence_ids:
            if evidence_id not in merged:
                merged.append(evidence_id)
    return merged


def _validate_verdict_batch(
    candidates: Sequence[EvidenceCandidate],
    verdicts: Sequence[VerificationVerdict],
) -> None:
    """Reject backend outputs that do not align with the requested candidate batch."""
    if len(verdicts) != len(candidates):
        raise ValueError("signal ensemble verifier returned wrong verdict count")
    expected_ids = [candidate.candidate_id for candidate in candidates]
    actual_ids = [verdict.candidate_id for verdict in verdicts]
    if actual_ids != expected_ids:
        raise ValueError("signal ensemble verifier returned verdicts for unexpected ids")


def _selected_score(
    verdicts: Sequence[VerificationVerdict], selected_status: VerificationStatus
) -> VerificationScore | None:
    """Carry forward a representative score when the ensemble keeps one status."""
    for verdict in verdicts:
        if verdict.status is selected_status and verdict.score is not None:
            return verdict.score
    return None
