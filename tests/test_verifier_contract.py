import pytest

from pragmalens.models import VerificationStatus, VerificationVerdict
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
    assert context.artifacts["verify_claims"] == {"verdicts": []}
