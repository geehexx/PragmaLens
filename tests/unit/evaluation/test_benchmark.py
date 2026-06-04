from __future__ import annotations

from pathlib import Path

import pytest

from pragmalens.evaluation import (
    ApprovalStatus,
    CorpusApprovalMetadata,
    CorpusBenchmarkBatch,
    load_corpus_approval_metadata,
    load_corpus_benchmark_batch,
    render_corpus_benchmark_summary,
    run_corpus_benchmark,
    run_corpus_benchmark_files,
)
from pragmalens.models import VerificationStatus
from pragmalens.verifier import (
    CrossEncoderNliVerifier,
    MiniCheckVerifier,
    OfflineBaselineVerifier,
)
from pragmalens.verifier_calibration import VerifierCalibrationCase

_FIXTURES = Path("tests/fixtures")


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


def _batch() -> CorpusBenchmarkBatch:
    return CorpusBenchmarkBatch(
        corpus_id="CORPUS-CAND-001",
        cases=[
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
        ],
    )


def _approval(
    status: ApprovalStatus = "approved",
    corpus_id: str = "CORPUS-CAND-001",
) -> CorpusApprovalMetadata:
    return CorpusApprovalMetadata(
        corpus_id=corpus_id,
        corpus_name="RAGTruth",
        corpus_version="2024-02",
        source_uri="https://github.com/ParticleMedia/RAGTruth",
        approval_status=status,
        approval_evidence=[
            "public repository README and MIT license",
            "open-access ACL paper and dataset release history",
        ],
    )


def test_load_corpus_benchmark_batch_round_trips_fixture() -> None:
    batch = load_corpus_benchmark_batch(_FIXTURES / "corpus_benchmark_ragtruth.json")

    assert batch.corpus_id == "ragtruth-15592-mini"
    assert [case.case_id for case in batch.cases] == [
        "ragtruth-15592-r0-supported",
        "ragtruth-15592-r2-unsupported",
    ]
    assert batch.cases[0].document_id == "ragtruth-15592"
    assert (
        batch.cases[0].claim_text == "Anne and Margot Frank probably did not survive to March 1945."
    )
    assert (
        batch.cases[1].claim_text
        == "Anne and Margot Frank are believed to have died before February 7, 2022."
    )


def test_load_corpus_approval_metadata_round_trips_fixture() -> None:
    approval = load_corpus_approval_metadata(_FIXTURES / "corpus_approval_ragtruth.json")

    assert approval.approval_status == "approved"
    assert approval.approval_evidence == [
        "upstream dataset source_id 15592 from RAGTruth source_info.jsonl",
        (
            "upstream response id 0 is supported and response id 2 contains the explicit "
            "2022 hallucination"
        ),
    ]
    assert approval.corpus_id == "ragtruth-15592-mini"
    assert (
        approval.source_uri
        == "https://raw.githubusercontent.com/ParticleMedia/RAGTruth/main/dataset/source_info.jsonl"
    )


def test_run_corpus_benchmark_rejects_unapproved_corpus() -> None:
    with pytest.raises(ValueError, match="approved"):
        run_corpus_benchmark(
            _batch(),
            _approval(status="pending"),
            selected_verifier=OfflineBaselineVerifier(),
        )


def test_run_corpus_benchmark_rejects_corpus_id_mismatch() -> None:
    with pytest.raises(ValueError, match="corpus id mismatch"):
        run_corpus_benchmark(
            _batch(),
            _approval(corpus_id="other"),
            selected_verifier=OfflineBaselineVerifier(),
        )


def test_run_corpus_benchmark_offline_is_deterministic() -> None:
    report, metadata, recommendation = run_corpus_benchmark(
        _batch(),
        _approval(),
        selected_verifier=OfflineBaselineVerifier(),
        run_id="benchmark-run",
    )
    report_again, metadata_again, recommendation_again = run_corpus_benchmark(
        _batch(),
        _approval(),
        selected_verifier=OfflineBaselineVerifier(),
        run_id="benchmark-run",
    )

    assert report.model_dump(mode="json") == report_again.model_dump(mode="json")
    assert metadata.model_dump(mode="json") == metadata_again.model_dump(mode="json")
    assert recommendation.model_dump(mode="json") == recommendation_again.model_dump(mode="json")
    assert metadata.selected_backend == "offline"
    assert metadata.case_count == 2
    assert {summary.backend for summary in report.summaries} == {"offline"}


def test_run_corpus_benchmark_can_compare_selected_live_backend() -> None:
    report, metadata, recommendation = run_corpus_benchmark(
        _batch(),
        _approval(),
        selected_backend="minicheck",
        selected_verifier=MiniCheckVerifier(
            _MiniCheckScorer(
                {
                    "supported claim": 0.91,
                    "unsupported claim": 0.08,
                }
            ),
            model_name="mini-fixture",
        ),
        run_id="mini-benchmark",
    )

    assert metadata.selected_backend == "minicheck"
    assert {summary.backend for summary in report.summaries} == {"offline", "minicheck"}
    assert recommendation.backend == "minicheck"


def test_run_corpus_benchmark_can_compare_crossencoder_live_backend() -> None:
    report, metadata, recommendation = run_corpus_benchmark(
        _batch(),
        _approval(),
        selected_backend="crossencoder_nli",
        selected_verifier=CrossEncoderNliVerifier(
            _CrossEncoderModel(
                {
                    "supported claim": [0.05, 0.9, 0.05],
                    "unsupported claim": [0.9, 0.05, 0.05],
                }
            ),
            model_name="ce-fixture",
        ),
        run_id="crossencoder-benchmark",
    )

    assert metadata.selected_backend == "crossencoder_nli"
    assert {summary.backend for summary in report.summaries} == {"offline", "crossencoder_nli"}
    assert recommendation.backend == "crossencoder_nli"


def test_render_corpus_benchmark_summary_includes_core_fields() -> None:
    report, metadata, recommendation = run_corpus_benchmark(
        _batch(),
        _approval(),
        selected_verifier=OfflineBaselineVerifier(),
        run_id="benchmark-run",
    )

    summary = render_corpus_benchmark_summary(report, metadata, recommendation)

    assert "Corpus Benchmark: RAGTruth" in summary
    assert "Run ID: `benchmark-run`" in summary
    assert "Selected backend: `offline`" in summary


def test_run_corpus_benchmark_files_loads_inputs_and_emits_results(tmp_path: Path) -> None:
    report, metadata, recommendation = run_corpus_benchmark_files(
        corpus_batch_path=_FIXTURES / "corpus_benchmark_ragtruth.json",
        approval_metadata_path=_FIXTURES / "corpus_approval_ragtruth.json",
        run_id="files-run",
    )

    assert report.run_id == "files-run"
    assert metadata.run_id == "files-run"
    assert metadata.case_count == 2
    assert recommendation.backend == "offline"
    assert metadata.corpus_id == "ragtruth-15592-mini"
