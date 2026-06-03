"""Pipeline stages for verifier invocation and finding synthesis."""

from __future__ import annotations

from pragmalens.models import (
    CandidateStatus,
    EvidenceCandidate,
    Finding,
    VerificationStatus,
    VerificationVerdict,
)
from pragmalens.pipeline.runtime import RunContext, StageResult
from pragmalens.verifier import OfflineBaselineVerifier, VerifierAdapter


class VerifyClaimsStage:
    """Emit verifier verdicts for normalized candidates."""

    id = "verify_claims"
    version = "0.1"
    input_contract = "merged_candidates"
    output_contract = "verification_verdicts"

    def __init__(self, verifier: VerifierAdapter | None = None) -> None:
        """Use the default offline adapter unless a verifier is injected."""
        self._verifier = verifier or OfflineBaselineVerifier()

    def run(self, context: RunContext) -> StageResult:
        """Populate verification verdicts without silent adapter failures."""
        skipped = [
            candidate
            for candidate in context.candidates
            if candidate.status is not CandidateStatus.VALID
        ]
        valid_candidates = [
            candidate
            for candidate in context.candidates
            if candidate.status is CandidateStatus.VALID
        ]
        try:
            verdicts = self._verifier.verify(
                valid_candidates,
                document_id=context.document_id,
                text=context.text,
            )
            self._validate_verdicts(valid_candidates, verdicts)
        except Exception as exc:
            verdicts = self._fallback_error_verdicts(valid_candidates, error=str(exc))
            if valid_candidates:
                context.warnings.append("verifier_adapter_failed")
        if skipped:
            context.warnings.extend(
                [f"verifier_skipped_non_valid:{candidate.candidate_id}" for candidate in skipped]
            )
        context.artifacts[self.id] = {
            "verdicts": [verdict.model_dump(mode="json") for verdict in verdicts],
            "skipped_candidates": [candidate.model_dump(mode="json") for candidate in skipped],
        }
        context.metadata["verification"] = verdicts
        return StageResult(stage_id=self.id, status="ok")

    @staticmethod
    def _validate_verdicts(
        candidates: list[EvidenceCandidate], verdicts: list[VerificationVerdict]
    ) -> None:
        """Reject adapter outputs that do not align with the requested batch."""
        if len(verdicts) != len(candidates):
            raise ValueError("verifier returned wrong verdict count")
        expected_ids = [candidate.candidate_id for candidate in candidates]
        actual_ids = [verdict.candidate_id for verdict in verdicts]
        if actual_ids != expected_ids:
            raise ValueError("verifier returned verdicts for unexpected candidate ids")

    @staticmethod
    def _fallback_error_verdicts(
        candidates: list[EvidenceCandidate], *, error: str
    ) -> list[VerificationVerdict]:
        """Produce deterministic verifier-error verdicts after adapter failure."""
        return [
            VerificationVerdict(
                candidate_id=candidate.candidate_id,
                status=VerificationStatus.VERIFIER_ERROR,
                rationale="verifier adapter failed; emitted fallback error verdict",
                evidence_ids=candidate.evidence_refs,
                error=error,
            )
            for candidate in candidates
        ]


class SynthesizeFindingsStage:
    """Convert verifier outcomes into user-facing findings."""

    id = "synthesize_findings"
    version = "0.1"
    input_contract = "verification_verdicts"
    output_contract = "findings"

    def run(self, context: RunContext) -> StageResult:
        """Generate synthesized findings for actionable negative verdicts."""
        verdicts = context.metadata.get("verification", [])
        findings = [
            Finding(
                finding_id=f"finding-{idx + 1}",
                candidate_id=verdict.candidate_id,
                verdict=verdict.status,
                summary=verdict.rationale,
                evidence_ids=verdict.evidence_ids,
            )
            for idx, verdict in enumerate(verdicts)
            if verdict.status in {VerificationStatus.UNSUPPORTED, VerificationStatus.VERIFIER_ERROR}
        ]
        context.artifacts[self.id] = {
            "findings": [finding.model_dump(mode="json") for finding in findings]
        }
        context.metadata["findings"] = findings
        return StageResult(stage_id=self.id, status="ok")
