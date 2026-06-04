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
from pragmalens.verifier import (
    OfflineBaselineVerifier,
    SignalEnsembleVerifier,
    VerifierAdapter,
)
from pragmalens.verifier_comparison import VerifierComparisonHarness


class VerifyClaimsStage:
    """Emit verifier verdicts for normalized candidates."""

    id = "verify_claims"
    version = "0.1"
    input_contract = "merged_candidates"
    output_contract = "verification_verdicts"

    def __init__(
        self,
        verifier: VerifierAdapter | None = None,
        comparison_harness: VerifierComparisonHarness | None = None,
    ) -> None:
        """Use the default offline adapter unless a verifier is injected."""
        self._verifier = verifier or OfflineBaselineVerifier()
        self._comparison_harness = comparison_harness

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
        comparison_payload = None
        if self._comparison_harness is not None and valid_candidates:
            try:
                selected_verdicts = verdicts
                if isinstance(self._verifier, SignalEnsembleVerifier):
                    selected_verdicts = self._verifier.selected_backend_verdicts()
                comparison = self._comparison_harness.compare(
                    valid_candidates,
                    document_id=context.document_id,
                    text=context.text,
                    selected_verdicts=selected_verdicts,
                )
                context.metadata["verification_comparison"] = comparison
                comparison_payload = comparison.model_dump(mode="json")
            except Exception as exc:
                context.warnings.append("verifier_comparison_failed")
                comparison_payload = {"error": str(exc)}
        if skipped:
            context.warnings.extend(
                [f"verifier_skipped_non_valid:{candidate.candidate_id}" for candidate in skipped]
            )
        artifact: dict[str, object] = {
            "verdicts": [verdict.model_dump(mode="json") for verdict in verdicts],
            "skipped_candidates": [candidate.model_dump(mode="json") for candidate in skipped],
        }
        if comparison_payload is not None:
            artifact["comparison"] = comparison_payload
        context.artifacts[self.id] = artifact
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
                backend="verifier_error",
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
            _build_finding(
                finding_id=f"finding-{idx + 1}",
                verdict=verdict,
            )
            for idx, verdict in enumerate(verdicts)
            if verdict.status in {VerificationStatus.UNSUPPORTED, VerificationStatus.VERIFIER_ERROR}
        ]
        context.artifacts[self.id] = {
            "findings": [finding.model_dump(mode="json") for finding in findings]
        }
        context.metadata["findings"] = findings
        return StageResult(stage_id=self.id, status="ok")


def _build_finding(*, finding_id: str, verdict: VerificationVerdict) -> Finding:
    """Build one synthesized finding with deterministic guidance text."""
    if verdict.status is VerificationStatus.VERIFIER_ERROR:
        question = "Can the verifier be rerun with the required live dependencies installed?"
        actionability = (
            "Fix the verifier runtime or install the missing backend before rerunning this claim."
        )
    else:
        question = "What supporting evidence would let this claim be verified?"
        actionability = "Collect corroborating evidence or revise the claim before promoting it."

    return Finding(
        finding_id=finding_id,
        candidate_id=verdict.candidate_id,
        verdict=verdict.status,
        summary=verdict.rationale,
        question=question,
        actionability=actionability,
        evidence_ids=verdict.evidence_ids,
    )
