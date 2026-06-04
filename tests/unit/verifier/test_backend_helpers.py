import pytest

from pragmalens.models import (
    EvidenceCandidate,
    SpanRef,
    VerificationScore,
    VerificationStatus,
    VerificationVerdict,
    VerifierCalibrationProfile,
    VerifierCalibrationThresholds,
)
from pragmalens.verifier import (
    CrossEncoderNliVerifier,
    SignalEnsembleVerifier,
    VerifierBackend,
    _status_for_crossencoder,
    _status_for_minicheck,
    build_default_verifier_runtime,
    build_verifier_adapter,
)


def _candidate() -> EvidenceCandidate:
    return EvidenceCandidate(
        candidate_id="candidate-1",
        label="claim",
        kind="claim",
        span=SpanRef(document_id="doc", start_char=0, end_char=4, text="Need"),
        provenance=["fixture"],
        evidence_refs=["source-a"],
    )


class _FallbackCrossEncoderModel:
    def predict(
        self,
        pairs: list[tuple[str, str]],
        *,
        apply_softmax: bool = False,
    ) -> list[list[float]]:
        assert pairs == [("Need evidence.", "Need")]
        assert apply_softmax is True
        return [[2.0, 5.0, 1.0]]


class _StaticVerifier:
    def __init__(self, backend: str, statuses: list[VerificationStatus]) -> None:
        self.backend = backend
        self._statuses = statuses

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
                status=status,
                rationale=f"{self.backend} -> {status.value}",
                evidence_ids=candidate.evidence_refs,
                backend=self.backend,
            )
            for candidate, status in zip(candidates, self._statuses, strict=True)
        ]


def test_crossencoder_verifier_falls_back_to_plain_predict_and_normalizes_scores() -> None:
    verdicts = CrossEncoderNliVerifier(_FallbackCrossEncoderModel()).verify(
        [_candidate()], document_id="doc", text="Need evidence."
    )

    assert verdicts[0].status is VerificationStatus.SUPPORTED
    assert verdicts[0].score is not None
    assert verdicts[0].score.normalized is True
    assert verdicts[0].score.entailment_probability is not None
    assert verdicts[0].score.entailment_probability > 0.0


def test_status_helpers_reject_missing_probability_fields() -> None:
    calibration = VerifierCalibrationProfile(
        calibration_id="calibration-fixture",
        backend=VerifierBackend.MINICHECK,
        evidence_source="tests/unit/verifier/test_backend_helpers.py",
        sample_size=1,
        thresholds=VerifierCalibrationThresholds(),
    )

    with pytest.raises(ValueError, match="support_probability"):
        _status_for_minicheck(VerificationScore(), calibration)

    with pytest.raises(ValueError, match="normalized probabilities"):
        _status_for_crossencoder(VerificationScore(), calibration, label="neutral")


def test_build_verifier_adapter_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="Unsupported verifier backend"):
        build_verifier_adapter("unknown-backend")


def test_signal_ensemble_verifier_combines_signals_conservatively() -> None:
    candidates = [
        _candidate(),
        EvidenceCandidate(
            candidate_id="candidate-2",
            label="claim",
            kind="claim",
            span=SpanRef(document_id="doc", start_char=0, end_char=6, text="claims"),
            provenance=["fixture"],
            evidence_refs=["source-b"],
        ),
        EvidenceCandidate(
            candidate_id="candidate-3",
            label="claim",
            kind="claim",
            span=SpanRef(document_id="doc", start_char=0, end_char=5, text="error"),
            provenance=["fixture"],
            evidence_refs=["source-c"],
        ),
    ]
    verifier = SignalEnsembleVerifier(
        [
            _StaticVerifier(
                VerifierBackend.OFFLINE,
                [
                    VerificationStatus.INSUFFICIENT_EVIDENCE,
                    VerificationStatus.INSUFFICIENT_EVIDENCE,
                    VerificationStatus.VERIFIER_ERROR,
                ],
            ),
            _StaticVerifier(
                VerifierBackend.MINICHECK,
                [
                    VerificationStatus.SUPPORTED,
                    VerificationStatus.UNSUPPORTED,
                    VerificationStatus.INSUFFICIENT_EVIDENCE,
                ],
            ),
            _StaticVerifier(
                VerifierBackend.CROSSENCODER_NLI,
                [
                    VerificationStatus.UNSUPPORTED,
                    VerificationStatus.UNSUPPORTED,
                    VerificationStatus.VERIFIER_ERROR,
                ],
            ),
        ]
    )

    verdicts = verifier.verify(candidates, document_id="doc", text="Need claims")

    assert [verdict.status for verdict in verdicts] == [
        VerificationStatus.INSUFFICIENT_EVIDENCE,
        VerificationStatus.UNSUPPORTED,
        VerificationStatus.INSUFFICIENT_EVIDENCE,
    ]
    assert all(verdict.backend == "signal_ensemble" for verdict in verdicts)
    assert "minicheck=supported" in verdicts[0].rationale
    assert "minicheck=unsupported" in verdicts[1].rationale
    assert "crossencoder_nli=unsupported" in verdicts[0].rationale


def test_signal_ensemble_verifier_retains_errors_only_without_stronger_verdicts() -> None:
    verdicts = SignalEnsembleVerifier(
        [
            _StaticVerifier(VerifierBackend.MINICHECK, [VerificationStatus.VERIFIER_ERROR]),
            _StaticVerifier(VerifierBackend.CROSSENCODER_NLI, [VerificationStatus.VERIFIER_ERROR]),
        ]
    ).verify([_candidate()], document_id="doc", text="Need")

    assert verdicts[0].status is VerificationStatus.VERIFIER_ERROR


def test_build_default_verifier_runtime_wraps_live_backend_in_signal_ensemble(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "pragmalens.verifier.build_verifier_adapter",
        lambda backend, **_: _StaticVerifier(str(backend), [VerificationStatus.SUPPORTED]),
    )

    verifier, comparison_harness = build_default_verifier_runtime(VerifierBackend.MINICHECK)

    assert isinstance(verifier, SignalEnsembleVerifier)
    assert comparison_harness is not None
    assert comparison_harness.selected_backend == VerifierBackend.MINICHECK


def test_build_default_verifier_runtime_keeps_offline_default_simple() -> None:
    verifier, comparison_harness = build_default_verifier_runtime()

    assert comparison_harness is None
    assert getattr(verifier, "backend", None) is VerifierBackend.OFFLINE
