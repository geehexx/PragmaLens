import pytest

from pragmalens.models import (
    EvidenceCandidate,
    SpanRef,
    VerificationScore,
    VerificationStatus,
    VerifierCalibrationProfile,
    VerifierCalibrationThresholds,
)
from pragmalens.verifier import (
    CrossEncoderNliVerifier,
    VerifierBackend,
    _status_for_crossencoder,
    _status_for_minicheck,
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
