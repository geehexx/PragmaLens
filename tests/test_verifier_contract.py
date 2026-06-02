import pytest

from pragmalens.models import (
    CandidateStatus,
    EvidenceCandidate,
    SpanRef,
    VerificationStatus,
    VerificationVerdict,
)
from pragmalens.stages import RunContext, StageResult, VerifyClaimsStage


@pytest.mark.parametrize("status", list(VerificationStatus))
def test_verification_verdict_states_serialize(status: VerificationStatus) -> None:
    verdict = VerificationVerdict(
        candidate_id="candidate-1",
        status=status,
        rationale="contract fixture",
        evidence_ids=["fixture"],
        error="boom" if status is VerificationStatus.VERIFIER_ERROR else None,
    )

    assert verdict.model_dump(mode="json")["status"] == status.value


def test_verifier_stage_records_empty_verdicts_without_silent_failure() -> None:
    context = RunContext(document_id="doc", input_path="doc.md", text="No candidates.")

    result = VerifyClaimsStage().run(context)

    assert isinstance(result, StageResult)
    assert result.status == "ok"
    assert context.metadata["verification"] == []
    assert context.artifacts["verify_claims"] == {"verdicts": [], "skipped_candidates": []}


def test_verifier_stage_skips_non_valid_candidates() -> None:
    context = RunContext(
        document_id="doc",
        input_path="doc.md",
        text="No candidates.",
        candidates=[
            EvidenceCandidate(
                candidate_id="candidate-1",
                label="claim",
                kind="claim",
                status=CandidateStatus.QUARANTINED,
                span=SpanRef(document_id="doc", start_char=0, end_char=1, text="N"),
                provenance=["fixture"],
            )
        ],
    )

    result = VerifyClaimsStage().run(context)

    assert isinstance(result, StageResult)
    assert result.status == "ok"
    assert context.metadata["verification"] == []
    assert len(context.artifacts["verify_claims"]["skipped_candidates"]) == 1
    assert "verifier_skipped_non_valid:candidate-1" in context.warnings
