from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pragmalens.models import CandidateStatus, EvidenceCandidate, SpanRef


@dataclass
class GLiNER2NormalizationResult:
    """Structured normalization result for captured GLiNER2 entities."""

    valid: list[EvidenceCandidate]
    quarantined: list[EvidenceCandidate]


@dataclass
class CandidateNormalizationResult:
    """Structured merge/quarantine result for evidence normalization."""

    candidates: list[EvidenceCandidate]
    quarantined: list[EvidenceCandidate]
    duplicates: list[EvidenceCandidate]
    duplicate_events: list[dict[str, Any]]
    conflicts: list[dict[str, Any]]
    stats: dict[str, Any]


def load_json(path: str | Path) -> Any:
    """Load a JSON payload from disk."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load newline-delimited JSON records from disk."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _quarantine_span(document_id: str) -> SpanRef:
    """Return a placeholder span used for quarantined extraction records."""
    return SpanRef(document_id=document_id, start_char=0, end_char=1, text=" ")


def _quarantine_gliner2_entity(
    entity: Any,
    *,
    idx: int,
    document_id: str,
    source: str,
    warning: str,
) -> EvidenceCandidate:
    """Build a quarantined candidate for malformed GLiNER2 entity payloads."""
    raw = entity if isinstance(entity, dict) else {"raw_entity": entity}
    raw_label = raw.get("label", "unknown") if isinstance(raw, dict) else "unknown"
    raw_kind = raw.get("kind", "claim") if isinstance(raw, dict) else "claim"
    label = raw_label if isinstance(raw_label, str) and raw_label else "unknown"
    kind = raw_kind if isinstance(raw_kind, str) and raw_kind else "claim"
    return EvidenceCandidate(
        candidate_id=f"gl-q-{idx}",
        label=label,
        kind=kind,
        status=CandidateStatus.QUARANTINED,
        span=_quarantine_span(document_id),
        provenance=[source],
        evidence_refs=[source],
        warnings=[warning],
        attributes={"raw": raw},
    )


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


def normalize_gliner2_output(
    payload: dict[str, Any],
    text: str,
    *,
    document_id: str = "doc",
    source: str = "gliner2",
    label_map: dict[str, str] | None = None,
) -> list[EvidenceCandidate]:
    """Normalize captured GLiNER2 output into product evidence candidates."""
    return normalize_gliner2_output_with_quarantine(
        payload,
        text,
        document_id=document_id,
        source=source,
        label_map=label_map,
    ).valid


def normalize_gliner2_output_with_quarantine(
    payload: dict[str, Any],
    text: str,
    *,
    document_id: str = "doc",
    source: str = "gliner2",
    label_map: dict[str, str] | None = None,
) -> GLiNER2NormalizationResult:
    """Normalize captured GLiNER2 output and quarantine malformed entities."""
    candidates: list[EvidenceCandidate] = []
    quarantined: list[EvidenceCandidate] = []
    entities = payload.get("entities", [])
    relations = payload.get("relations", [])
    mapping = label_map or {}

    for idx, ent in enumerate(entities):
        if not isinstance(ent, dict):
            quarantined.append(
                _quarantine_gliner2_entity(
                    ent,
                    idx=idx,
                    document_id=document_id,
                    source=source,
                    warning="invalid_entity_payload",
                )
            )
            continue
        try:
            start = int(ent["start_char"])
            end = int(ent["end_char"])
        except KeyError:
            quarantined.append(
                _quarantine_gliner2_entity(
                    ent,
                    idx=idx,
                    document_id=document_id,
                    source=source,
                    warning="missing_char_interval",
                )
            )
            continue
        except (TypeError, ValueError):
            quarantined.append(
                _quarantine_gliner2_entity(
                    ent,
                    idx=idx,
                    document_id=document_id,
                    source=source,
                    warning="invalid_char_interval",
                )
            )
            continue
        if start < 0 or end <= start or end > len(text):
            quarantined.append(
                _quarantine_gliner2_entity(
                    ent,
                    idx=idx,
                    document_id=document_id,
                    source=source,
                    warning="invalid_char_interval",
                )
            )
            continue
        raw_label_value = ent.get("label", "unknown")
        raw_label = (
            raw_label_value if isinstance(raw_label_value, str) and raw_label_value else "unknown"
        )
        mapped_label_value = mapping.get(raw_label, raw_label)
        mapped_label = (
            mapped_label_value
            if isinstance(mapped_label_value, str) and mapped_label_value
            else raw_label
        )
        raw_kind = ent.get("kind", "claim")
        kind = raw_kind if isinstance(raw_kind, str) and raw_kind else "claim"
        relation_keys = {
            json.dumps(relation, sort_keys=True)
            for relation in relations
            if relation.get("head") == idx or relation.get("tail") == idx
        }
        candidates.append(
            EvidenceCandidate(
                candidate_id=f"gl-{idx}",
                label=mapped_label,
                kind=kind,
                status=CandidateStatus.VALID,
                span=SpanRef(
                    document_id=document_id, start_char=start, end_char=end, text=text[start:end]
                ),
                confidence=float(ent.get("confidence", 0.0)),
                provenance=[source],
                evidence_refs=[source],
                relations=[json.loads(relation) for relation in sorted(relation_keys)],
                attributes={"raw": ent, "raw_label": raw_label},
            )
        )
    return GLiNER2NormalizationResult(valid=candidates, quarantined=quarantined)


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

    for cand in candidates:
        span_key = (cand.span.document_id, cand.span.start_char, cand.span.end_char)
        span_labels.setdefault(span_key, set()).add(cand.label)
        span_kinds.setdefault(span_key, set()).add(cand.kind)

        key = (
            cand.span.document_id,
            cand.span.start_char,
            cand.span.end_char,
            cand.label,
            cand.kind,
        )
        if key not in merged:
            _ensure_evidence_refs(cand)
            merged[key] = cand
            continue

        existing = merged[key]
        _ensure_evidence_refs(existing)
        _ensure_evidence_refs(cand)
        existing.provenance = sorted(set(existing.provenance + cand.provenance))
        existing.evidence_refs = sorted(set(existing.evidence_refs + cand.evidence_refs))
        existing.warnings = sorted(set(existing.warnings + cand.warnings))
        relation_keys = {
            json.dumps(rel, sort_keys=True) for rel in existing.relations + cand.relations
        }
        merged_relations = sorted(relation_keys)
        existing.relations = [json.loads(relation) for relation in merged_relations]
        duplicate = cand.model_copy(deep=True)
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
    for cand in result:
        span_key = (cand.span.document_id, cand.span.start_char, cand.span.end_char)
        grouped_result.setdefault(span_key, []).append(cand)
        labels = sorted(span_labels.get(span_key, set()))
        kinds = sorted(span_kinds.get(span_key, set()))
        if len(labels) > 1 and "label_conflict" not in cand.warnings:
            cand.warnings.append("label_conflict")
            cand.attributes["conflicting_labels"] = labels
        if len(kinds) > 1 and "kind_conflict" not in cand.warnings:
            cand.warnings.append("kind_conflict")
            cand.attributes["conflicting_kinds"] = kinds
        cand.attributes["merged_provenance"] = list(cand.provenance)

    quarantined_candidates = list(quarantined or [])
    quarantined_source_counts = _count_candidate_sources(quarantined_candidates)
    duplicate_source_counts = _count_candidate_sources(duplicate_candidates)
    quarantine_warning_counts = _count_warning_reasons(quarantined_candidates)
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
            "quarantined_candidates_by_source": quarantined_source_counts,
            "duplicate_candidates_by_source": duplicate_source_counts,
            "quarantine_warning_counts": quarantine_warning_counts,
        },
    )


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
