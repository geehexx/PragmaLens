import json
from pathlib import Path

from pragmalens.models import VerificationStatus
from pragmalens.verifier import OfflineBaselineVerifier
from pragmalens.verifier_calibration import (
    VerifierCalibrationCase,
    compare_verifier_backends,
    load_verifier_calibration_cases,
)


class _MiniCheckScorer:
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


class _CrossEncoderModel:
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


def _cases() -> list[VerifierCalibrationCase]:
    return [
        VerifierCalibrationCase(
            case_id="supported",
            document_id="doc",
            document_text="supported claim",
            claim_text="supported claim",
            gold_status=VerificationStatus.SUPPORTED,
        ),
        VerifierCalibrationCase(
            case_id="unsupported",
            document_id="doc",
            document_text="unsupported claim",
            claim_text="unsupported claim",
            gold_status=VerificationStatus.UNSUPPORTED,
        ),
    ]


def test_load_verifier_calibration_cases(tmp_path: Path) -> None:
    path = tmp_path / "cases.json"
    path.write_text(
        json.dumps(
            [
                {
                    "case_id": "supported",
                    "document_id": "doc",
                    "document_text": "supported claim",
                    "claim_text": "supported claim",
                    "gold_status": "supported",
                }
            ]
        ),
        encoding="utf-8",
    )

    cases = load_verifier_calibration_cases(path)

    assert cases[0].case_id == "supported"
    assert cases[0].gold_status is VerificationStatus.SUPPORTED


def test_compare_verifier_backends_returns_summary_for_each_backend() -> None:
    report = compare_verifier_backends(
        _cases(),
        offline_verifier=OfflineBaselineVerifier(),
        minicheck_scorer=_MiniCheckScorer(
            {
                "supported claim": 0.91,
                "unsupported claim": 0.08,
            }
        ),
        crossencoder_model=_CrossEncoderModel(
            {
                "supported claim": [0.05, 0.9, 0.05],
                "unsupported claim": [0.9, 0.05, 0.05],
            }
        ),
        run_id="calibration-run",
    )

    assert report.run_id == "calibration-run"
    assert {summary.backend for summary in report.summaries} == {
        "offline",
        "minicheck",
        "crossencoder_nli",
    }
    assert len(report.observations) == 6
    assert report.summaries[1].recommended_support_threshold is not None
    assert report.summaries[2].recommended_support_threshold is not None
    assert report.summaries[1].exact_match_rate == 1.0


def test_compare_verifier_backends_can_use_only_offline_backend() -> None:
    report = compare_verifier_backends(_cases(), run_id="offline-only")

    assert report.run_id == "offline-only"
    assert {summary.backend for summary in report.summaries} == {"offline"}
