"""Candidate deduplication and conflict-trace helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from pragmalens.models import CandidateStatus, EvidenceCandidate, SpanRef


@dataclass
class CandidateNormalizationResult:
    """Structured merge/quarantine result for evidence normalization."""

    candidates: list[EvidenceCandidate]
    quarantined: list[EvidenceCandidate]
    duplicates: list[EvidenceCandidate]
    duplicate_events: list[dict[str, Any]]
    conflicts: list[dict[str, Any]]
    stats: dict[str, Any]


def merge_and_dedupe_candidates(candidates: list[EvidenceCandidate]) -> list[EvidenceCandidate]:
    """Merge duplicate candidates while preserving provenance and conflict warnings."""
    return normalize_candidate_set(candidates).candidates


def normalize_candidate_set(
    candidates: list[EvidenceCandidate],
    quarantined: list[EvidenceCandidate] | None = None,
) -> CandidateNormalizationResult:
    """Return merged candidates plus duplicate/conflict/quarantine trace details."""
    merged: dict[tuple[str, int, int, str, str], EvidenceCandidate] = {}
    span_labels: dict[tuple[str, int, int], set[str]] = {}
    span_kinds: dict[tuple[str, int, int], set[str]] = {}
    duplicate_candidates: list[EvidenceCandidate] = []
    duplicate_events: list[dict[str, Any]] = []
    input_source_counts = _count_candidate_sources(candidates)

    for candidate in candidates:
        span_key = (candidate.span.document_id, candidate.span.start_char, candidate.span.end_char)
        span_labels.setdefault(span_key, set()).add(candidate.label)
        span_kinds.setdefault(span_key, set()).add(candidate.kind)

        key = (
            candidate.span.document_id,
            candidate.span.start_char,
            candidate.span.end_char,
            candidate.label,
            candidate.kind,
        )
        if key not in merged:
            _ensure_evidence_refs(candidate)
            merged[key] = candidate
            continue

        existing = merged[key]
        _ensure_evidence_refs(existing)
        _ensure_evidence_refs(candidate)
        existing.provenance = sorted(set(existing.provenance + candidate.provenance))
        existing.evidence_refs = sorted(set(existing.evidence_refs + candidate.evidence_refs))
        existing.warnings = sorted(set(existing.warnings + candidate.warnings))
        relation_keys = {
            json.dumps(relation, sort_keys=True)
            for relation in existing.relations + candidate.relations
        }
        existing.relations = [json.loads(relation) for relation in sorted(relation_keys)]

        duplicate = candidate.model_copy(deep=True)
        duplicate.status = CandidateStatus.DUPLICATE
        duplicate.warnings = sorted(set([*duplicate.warnings, "absorbed_duplicate"]))
        duplicate.attributes["duplicate_of"] = existing.candidate_id
        duplicate_candidates.append(duplicate)
        duplicate_events.append(
            {
                "candidate_id": duplicate.candidate_id,
                "duplicate_of": existing.candidate_id,
                "span": _span_payload(duplicate.span),
                "label": duplicate.label,
                "kind": duplicate.kind,
                "sources": _candidate_sources(duplicate),
                "winner_sources": _candidate_sources(existing),
                "evidence_refs": list(duplicate.evidence_refs),
                "warnings": list(duplicate.warnings),
            }
        )

    result = list(merged.values())
    conflicts: list[dict[str, Any]] = []
    grouped_result: dict[tuple[str, int, int], list[EvidenceCandidate]] = {}
    for candidate in result:
        span_key = (candidate.span.document_id, candidate.span.start_char, candidate.span.end_char)
        grouped_result.setdefault(span_key, []).append(candidate)
        labels = sorted(span_labels.get(span_key, set()))
        kinds = sorted(span_kinds.get(span_key, set()))
        if len(labels) > 1 and "label_conflict" not in candidate.warnings:
            candidate.warnings.append("label_conflict")
            candidate.attributes["conflicting_labels"] = labels
        if len(kinds) > 1 and "kind_conflict" not in candidate.warnings:
            candidate.warnings.append("kind_conflict")
            candidate.attributes["conflicting_kinds"] = kinds
        candidate.attributes["merged_provenance"] = list(candidate.provenance)

    quarantined_candidates = list(quarantined or [])
    conflicts = _build_conflicts(grouped_result, span_labels, span_kinds)
    return CandidateNormalizationResult(
        candidates=result,
        quarantined=quarantined_candidates,
        duplicates=duplicate_candidates,
        duplicate_events=duplicate_events,
        conflicts=conflicts,
        stats={
            "input_candidates": len(candidates),
            "valid_candidates": len(result),
            "quarantined_candidates": len(quarantined_candidates),
            "duplicate_candidates": len(duplicate_candidates),
            "conflict_count": len(conflicts),
            "conflict_spans": len(conflicts),
            "input_candidates_by_source": input_source_counts,
            "quarantined_candidates_by_source": _count_candidate_sources(quarantined_candidates),
            "duplicate_candidates_by_source": _count_candidate_sources(duplicate_candidates),
            "quarantine_warning_counts": _count_warning_reasons(quarantined_candidates),
        },
    )


def _build_conflicts(
    grouped_result: dict[tuple[str, int, int], list[EvidenceCandidate]],
    span_labels: dict[tuple[str, int, int], set[str]],
    span_kinds: dict[tuple[str, int, int], set[str]],
) -> list[dict[str, Any]]:
    """Build trace payloads for label/kind conflicts that survive normalization."""
    conflicts: list[dict[str, Any]] = []
    for span_key, related in grouped_result.items():
        labels = sorted(span_labels.get(span_key, set()))
        kinds = sorted(span_kinds.get(span_key, set()))
        if len(labels) <= 1 and len(kinds) <= 1:
            continue
        conflicts.append(
            {
                "span": {
                    "document_id": span_key[0],
                    "start_char": span_key[1],
                    "end_char": span_key[2],
                },
                "candidate_ids": [candidate.candidate_id for candidate in related],
                "warnings": sorted(
                    {
                        warning
                        for candidate in related
                        for warning in candidate.warnings
                        if warning in {"label_conflict", "kind_conflict"}
                    }
                ),
                "conflicting_labels": labels,
                "conflicting_kinds": kinds,
                "sources": sorted(
                    {source for candidate in related for source in _candidate_sources(candidate)}
                ),
                "evidence_refs": sorted(
                    {ref for candidate in related for ref in candidate.evidence_refs}
                ),
            }
        )
    return conflicts


def _ensure_evidence_refs(candidate: EvidenceCandidate) -> None:
    """Populate a canonical deduped evidence-ref list on the candidate model."""
    evidence_refs = candidate.evidence_refs or candidate.provenance
    candidate.evidence_refs = sorted(set(str(ref) for ref in evidence_refs))


def _candidate_sources(candidate: EvidenceCandidate) -> list[str]:
    """Return the best available source list for one candidate."""
    return sorted(set(str(source) for source in (candidate.evidence_refs or candidate.provenance)))


def _count_candidate_sources(candidates: list[EvidenceCandidate]) -> dict[str, int]:
    """Count how many candidates reference each source."""
    counts: dict[str, int] = {}
    for candidate in candidates:
        for source in _candidate_sources(candidate):
            counts[source] = counts.get(source, 0) + 1
    return dict(sorted(counts.items()))


def _count_warning_reasons(candidates: list[EvidenceCandidate]) -> dict[str, int]:
    """Count quarantine or normalization warning reasons across candidates."""
    counts: dict[str, int] = {}
    for candidate in candidates:
        for warning in candidate.warnings:
            counts[warning] = counts.get(warning, 0) + 1
    return dict(sorted(counts.items()))


def _span_payload(span: SpanRef) -> dict[str, Any]:
    """Serialize a span for trace payloads."""
    return {
        "document_id": span.document_id,
        "start_char": span.start_char,
        "end_char": span.end_char,
        "text": span.text,
    }
