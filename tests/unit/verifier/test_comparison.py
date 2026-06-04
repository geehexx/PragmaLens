import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pragmalens.models import (
    CandidateStatus,
    EvidenceCandidate,
    SpanRef,
    VerificationStatus,
)
from pragmalens.pipeline.reporting import build_report_and_manifest
from pragmalens.pipeline.runtime import RunContext, StageResult
from pragmalens.pipeline.verification import VerifyClaimsStage
from pragmalens.verifier import (
    CrossEncoderNliVerifier,
    MiniCheckVerifier,
    OfflineBaselineVerifier,
    VerifierBackend,
)
from pragmalens.verifier_comparison import (
    CalibrationExample,
    VerifierComparisonHarness,
    load_verifier_calibration_examples,
    recommend_calibration,
)


class BatchMiniCheckScorer:
    def __init__(self, probabilities: dict[str, float]) -> None:
        self._probabilities = probabilities

    def score(
        self,
        *,
        docs: list[str],
        claims: list[str],
    ) -> tuple[list[int], list[float], None, None]:
        del docs
        probs = [self._probabilities[claim] for claim in claims]
        labels = [1 if prob >= 0.5 else 0 for prob in probs]
        return labels, probs, None, None


class BatchCrossEncoderModel:
    def __init__(self, rows: dict[str, list[float]]) -> None:
        self._rows = rows

    def predict(
        self,
        pairs: list[tuple[str, str]],
        *,
        apply_softmax: bool = False,
    ) -> list[list[float]]:
        assert apply_softmax is True
        return [self._rows[claim] for _, claim in pairs]


def _candidate(claim: str, *, candidate_id: str = "candidate-1") -> EvidenceCandidate:
    return EvidenceCandidate(
        candidate_id=candidate_id,
        label="claim",
        kind="claim",
        status=CandidateStatus.VALID,
        span=SpanRef(document_id="doc", start_char=0, end_char=len(claim), text=claim),
        provenance=["fixture"],
        evidence_refs=["source-a"],
    )


def test_comparison_harness_records_same_batch_results_for_all_backends() -> None:
    candidate = _candidate("Need evidence.")
    harness = VerifierComparisonHarness(
        selected_backend=VerifierBackend.MINICHECK,
        verifiers=[
            OfflineBaselineVerifier(),
            MiniCheckVerifier(
                BatchMiniCheckScorer({"Need evidence.": 0.91}),
                model_name="mini-fixture",
            ),
            CrossEncoderNliVerifier(
                BatchCrossEncoderModel({"Need evidence.": [0.05, 0.9, 0.05]}),
                model_name="ce-fixture",
            ),
        ],
    )

    comparison = harness.compare([candidate], document_id="doc", text="Need evidence.")

    assert comparison.selected_backend == VerifierBackend.MINICHECK
    assert comparison.total_candidates == 1
    assert comparison.records[0].candidate_id == "candidate-1"
    assert [verdict.backend for verdict in comparison.records[0].backend_verdicts] == [
        VerifierBackend.OFFLINE,
        VerifierBackend.MINICHECK,
        VerifierBackend.CROSSENCODER_NLI,
    ]
    assert comparison.records[0].selected_verdict.status is VerificationStatus.SUPPORTED
    assert comparison.records[0].backend_verdicts[1].score is not None
    assert comparison.records[0].backend_verdicts[2].score is not None


def test_comparison_harness_rejects_duplicate_backends() -> None:
    candidate = _candidate("Need evidence.")

    with pytest.raises(ValueError, match="duplicate verifier backends"):
        VerifierComparisonHarness(
            selected_backend=VerifierBackend.OFFLINE,
            verifiers=[OfflineBaselineVerifier(), OfflineBaselineVerifier()],
        ).compare([candidate], document_id="doc", text="Need evidence.")


def test_comparison_harness_can_use_selected_verdicts_override() -> None:
    candidate = _candidate("Need evidence.")
    verifier = MiniCheckVerifier(
        BatchMiniCheckScorer({"Need evidence.": 0.91}),
        model_name="mini-fixture",
    )
    harness = VerifierComparisonHarness(
        selected_backend=VerifierBackend.MINICHECK,
        verifiers=[OfflineBaselineVerifier(), verifier],
    )
    selected_verdicts = verifier.verify([candidate], document_id="doc", text="Need evidence.")

    comparison = harness.compare(
        [candidate],
        document_id="doc",
        text="Need evidence.",
        selected_verdicts=selected_verdicts,
    )

    assert comparison.records[0].selected_verdict is selected_verdicts[0]


def test_recommend_calibration_returns_evidence_backed_minicheck_thresholds() -> None:
    verifier = MiniCheckVerifier(
        BatchMiniCheckScorer(
            {
                "supported claim": 0.91,
                "unsupported claim": 0.08,
                "uncertain claim": 0.54,
            }
        ),
        model_name="mini-fixture",
    )
    examples = [
        CalibrationExample(
            example_id="ex-supported",
            document_id="doc",
            text="supported claim",
            claim_text="supported claim",
            expected_status=VerificationStatus.SUPPORTED,
        ),
        CalibrationExample(
            example_id="ex-unsupported",
            document_id="doc",
            text="unsupported claim",
            claim_text="unsupported claim",
            expected_status=VerificationStatus.UNSUPPORTED,
        ),
        CalibrationExample(
            example_id="ex-uncertain",
            document_id="doc",
            text="uncertain claim",
            claim_text="uncertain claim",
            expected_status=VerificationStatus.INSUFFICIENT_EVIDENCE,
        ),
    ]

    recommendation = recommend_calibration(
        verifier,
        examples,
        evidence_source="tests/unit/verifier/test_comparison.py",
    )

    assert recommendation.backend == VerifierBackend.MINICHECK
    assert recommendation.calibration.sample_size == 3
    assert recommendation.calibration.evidence_source == "tests/unit/verifier/test_comparison.py"
    assert recommendation.calibration.thresholds.support_probability_min is not None
    assert recommendation.calibration.thresholds.support_probability_max is not None
    assert (
        recommendation.calibration.thresholds.support_probability_max
        < recommendation.calibration.thresholds.support_probability_min
    )
    assert recommendation.metrics.accuracy == pytest.approx(1.0)


def test_verify_claims_stage_records_comparison_payload_and_report_surface(
    tmp_path: Path,
) -> None:
    candidate = _candidate("Need evidence.")
    context = RunContext(
        document_id="doc",
        input_path="doc.md",
        text="Need evidence.",
        candidates=[candidate],
    )
    verifier = MiniCheckVerifier(
        BatchMiniCheckScorer({"Need evidence.": 0.91}),
        model_name="mini-fixture",
    )
    harness = VerifierComparisonHarness(
        selected_backend=VerifierBackend.MINICHECK,
        verifiers=[
            OfflineBaselineVerifier(),
            verifier,
            CrossEncoderNliVerifier(
                BatchCrossEncoderModel({"Need evidence.": [0.05, 0.9, 0.05]}),
                model_name="ce-fixture",
            ),
        ],
    )

    result = VerifyClaimsStage(verifier=verifier, comparison_harness=harness).run(context)

    assert isinstance(result, StageResult)
    comparison_payload = context.artifacts["verify_claims"]["comparison"]
    assert comparison_payload["selected_backend"] == VerifierBackend.MINICHECK
    assert (
        comparison_payload["records"][0]["selected_verdict"]["backend"] == VerifierBackend.MINICHECK
    )

    report, _ = build_report_and_manifest(
        context,
        input_path="doc.md",
        report_path=str(tmp_path / "report.json"),
        trace_dir=tmp_path / "trace",
        stage_results=[result],
    )

    assert report.verification_comparison is not None
    assert (
        report.verification_comparison.records[0].selected_verdict.backend
        == VerifierBackend.MINICHECK
    )


def test_load_verifier_calibration_examples(tmp_path: Path) -> None:
    path = tmp_path / "examples.json"
    path.write_text(
        """
[
  {
    "example_id": "ex-1",
    "document_id": "doc",
    "text": "supported claim",
    "claim_text": "supported claim",
    "expected_status": "supported"
  }
]
""".strip(),
        encoding="utf-8",
    )

    examples = load_verifier_calibration_examples(path)

    assert examples[0].example_id == "ex-1"
    assert examples[0].expected_status is VerificationStatus.SUPPORTED


@given(
    example_id=st.text(min_size=1, max_size=16),
    document_id=st.text(min_size=1, max_size=16),
    text=st.text(min_size=1, max_size=24),
    claim_text=st.text(min_size=1, max_size=24),
    expected_status=st.sampled_from(list(VerificationStatus)),
)
@settings(max_examples=40)
def test_load_verifier_calibration_examples_round_trips_generated_payloads(
    example_id: str,
    document_id: str,
    text: str,
    claim_text: str,
    expected_status: VerificationStatus,
) -> None:
    payload = {
        "example_id": example_id,
        "document_id": document_id,
        "text": text,
        "claim_text": claim_text,
        "expected_status": expected_status.value,
    }
    with TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "examples.json"
        path.write_text(json.dumps([payload]), encoding="utf-8")
        examples = load_verifier_calibration_examples(path)

    assert examples[0].model_dump(mode="json") == payload
