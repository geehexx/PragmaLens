from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pragmalens.models import CandidateStatus, EvidenceCandidate, SpanRef


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
    candidates: list[EvidenceCandidate] = []
    entities = payload.get("entities", [])
    relations = payload.get("relations", [])
    mapping = label_map or {}

    for idx, ent in enumerate(entities):
        start = int(ent["start_char"])
        end = int(ent["end_char"])
        if start < 0 or end <= start or end > len(text):
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
                relations=[json.loads(relation) for relation in sorted(relation_keys)],
                attributes={"raw": ent, "raw_label": raw_label},
            )
        )
    return candidates


def merge_and_dedupe_candidates(candidates: list[EvidenceCandidate]) -> list[EvidenceCandidate]:
    """Merge duplicate candidates while preserving provenance and conflict warnings."""
    merged: dict[tuple[str, int, int, str, str], EvidenceCandidate] = {}
    span_labels: dict[tuple[str, int, int], set[str]] = {}
    span_kinds: dict[tuple[str, int, int], set[str]] = {}

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
            merged[key] = cand
            continue

        existing = merged[key]
        existing.provenance = sorted(set(existing.provenance + cand.provenance))
        existing.warnings = sorted(set(existing.warnings + cand.warnings))
        relation_keys = {
            json.dumps(rel, sort_keys=True) for rel in existing.relations + cand.relations
        }
        merged_relations = sorted(relation_keys)
        existing.relations = [json.loads(relation) for relation in merged_relations]

    result = list(merged.values())
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
    return result
