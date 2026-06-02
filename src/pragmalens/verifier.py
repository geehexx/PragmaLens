from __future__ import annotations

from typing import Protocol

from pragmalens.models import EvidenceCandidate, VerificationStatus, VerificationVerdict


class VerifierAdapter(Protocol):
    """Batch verifier boundary used by the verification stage."""

    def verify(
        self,
        candidates: list[EvidenceCandidate],
        *,
        document_id: str,
        text: str,
    ) -> list[VerificationVerdict]: ...


class OfflineBaselineVerifier:
    """Deterministic offline verifier used until live adapters are introduced."""

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
            )
            for candidate in candidates
        ]
