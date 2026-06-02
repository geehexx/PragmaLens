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
    conflicts: list[dict[str, Any]]
    stats: dict[str, int]


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
        start = int(ent["start_char"])
        end = int(ent["end_char"])
        if start < 0 or end <= start or end > len(text):
            quarantined.append(
                EvidenceCandidate(
                    candidate_id=f"gl-q-{idx}",
                    label=ent.get("label", "unknown"),
                    kind=ent.get("kind", "claim"),
                    status=CandidateStatus.QUARANTINED,
                    span=_quarantine_span(document_id),
                    provenance=[source],
                    evidence_refs=[source],
                    warnings=["invalid_char_interval"],
                    attributes={"raw": ent},
                )
            )
            continue
        raw_label = ent.get("label", "unknown")
        mapped_label = mapping.get(raw_label, raw_label)
        relation_keys = {
            json.dumps(relation, sort_keys=True)
            for relation in relations
            if relation.get("head") == idx or relation.get("tail") == idx
        }
        candidates.append(
            EvidenceCandidate(
                candidate_id=f"gl-{idx}",
                label=mapped_label,
                kind=ent.get("kind", "claim"),
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

    result = list(merged.values())
    conflicts: list[dict[str, Any]] = []
    for cand in result:
        span_key = (cand.span.document_id, cand.span.start_char, cand.span.end_char)
        labels = sorted(span_labels.get(span_key, set()))
        kinds = sorted(span_kinds.get(span_key, set()))
        if len(labels) > 1 and "label_conflict" not in cand.warnings:
            cand.warnings.append("label_conflict")
            cand.attributes["conflicting_labels"] = labels
        if len(kinds) > 1 and "kind_conflict" not in cand.warnings:
            cand.warnings.append("kind_conflict")
            cand.attributes["conflicting_kinds"] = kinds
        cand.attributes["merged_provenance"] = list(cand.provenance)
        if "label_conflict" in cand.warnings or "kind_conflict" in cand.warnings:
            conflicts.append(
                {
                    "candidate_id": cand.candidate_id,
                    "warnings": list(cand.warnings),
                    "conflicting_labels": cand.attributes.get("conflicting_labels", []),
                    "conflicting_kinds": cand.attributes.get("conflicting_kinds", []),
                }
            )

    quarantined_candidates = list(quarantined or [])
    return CandidateNormalizationResult(
        candidates=result,
        quarantined=quarantined_candidates,
        duplicates=duplicate_candidates,
        conflicts=conflicts,
        stats={
            "input_candidates": len(candidates),
            "valid_candidates": len(result),
            "quarantined_candidates": len(quarantined_candidates),
            "duplicate_candidates": len(duplicate_candidates),
            "conflict_count": len(conflicts),
        },
    )


def _ensure_evidence_refs(candidate: EvidenceCandidate) -> None:
    """Populate a canonical deduped evidence-ref list on the candidate model."""
    evidence_refs = candidate.evidence_refs or candidate.provenance
    candidate.evidence_refs = sorted(set(str(ref) for ref in evidence_refs))
