"""Normalization helpers for captured LangExtract output."""

from __future__ import annotations

from typing import Any

from pragmalens.models import CandidateStatus, EvidenceCandidate, SpanRef


def _quarantine_span(document_id: str) -> SpanRef:
    """Return a placeholder span used for quarantined extraction records."""
    return SpanRef(document_id=document_id, start_char=0, end_char=1, text=" ")


def normalize_langextract_records(
    records: list[dict[str, Any]],
    text: str,
    *,
    document_id: str = "doc",
    source: str = "langextract",
) -> tuple[list[EvidenceCandidate], list[EvidenceCandidate], dict[str, Any]]:
    """Normalize captured LangExtract records into valid and quarantined candidates."""
    valid: list[EvidenceCandidate] = []
    quarantined: list[EvidenceCandidate] = []

    fixture_meta = records[0].get("_meta", {}) if records and isinstance(records[0], dict) else {}

    for idx, rec in enumerate(records):
        if "_meta" in rec:
            continue

        interval = rec.get("char_interval")
        extraction_text = rec.get("extraction_text", "")
        label = rec.get("label", "claim")
        kind = rec.get("kind", "claim")

        if interval is None:
            quarantined.append(
                EvidenceCandidate(
                    candidate_id=f"lx-q-{idx}",
                    label=label,
                    kind=kind,
                    status=CandidateStatus.QUARANTINED,
                    span=_quarantine_span(document_id),
                    provenance=[source],
                    evidence_refs=[source],
                    warnings=["missing_char_interval"],
                    attributes={"raw": rec},
                )
            )
            continue

        try:
            start, end = int(interval[0]), int(interval[1])
        except Exception:
            quarantined.append(
                EvidenceCandidate(
                    candidate_id=f"lx-q-{idx}",
                    label=label,
                    kind=kind,
                    status=CandidateStatus.QUARANTINED,
                    span=_quarantine_span(document_id),
                    provenance=[source],
                    evidence_refs=[source],
                    warnings=["invalid_char_interval"],
                    attributes={"raw": rec},
                )
            )
            continue

        if start < 0 or end <= start or end > len(text):
            quarantined.append(
                EvidenceCandidate(
                    candidate_id=f"lx-q-{idx}",
                    label=label,
                    kind=kind,
                    status=CandidateStatus.QUARANTINED,
                    span=_quarantine_span(document_id),
                    provenance=[source],
                    evidence_refs=[source],
                    warnings=["invalid_char_interval"],
                    attributes={"raw": rec},
                )
            )
            continue

        span_text = text[start:end]
        if extraction_text and extraction_text != span_text:
            quarantined.append(
                EvidenceCandidate(
                    candidate_id=f"lx-q-{idx}",
                    label=label,
                    kind=kind,
                    status=CandidateStatus.QUARANTINED,
                    span=_quarantine_span(document_id),
                    provenance=[source],
                    evidence_refs=[source],
                    warnings=["extraction_text_mismatch"],
                    attributes={"raw": rec},
                )
            )
            continue

        valid.append(
            EvidenceCandidate(
                candidate_id=f"lx-{idx}",
                label=label,
                kind=kind,
                status=CandidateStatus.VALID,
                span=SpanRef(
                    document_id=document_id, start_char=start, end_char=end, text=span_text
                ),
                provenance=[source],
                evidence_refs=[source],
                attributes={"raw": rec},
            )
        )

    meta = {
        "provider": fixture_meta.get("provider", "captured"),
        "model": fixture_meta.get("model", "captured"),
        "prompt_hash": fixture_meta.get("prompt_hash", "captured-fixture"),
        "example_set": fixture_meta.get("example_set", "captured-fixture"),
    }
    return valid, quarantined, meta
