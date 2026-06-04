"""Normalization helpers for captured GLiNER2 output."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from pragmalens.models import CandidateStatus, EvidenceCandidate, SpanRef


@dataclass
class GLiNER2NormalizationResult:
    """Structured normalization result for captured GLiNER2 entities."""

    valid: list[EvidenceCandidate]
    quarantined: list[EvidenceCandidate]


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
    entities = list(_iter_entities(payload))
    relations = payload.get("relations", [])
    mapping = label_map or {}

    for idx, (ent, raw_label_hint) in enumerate(entities):
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
        start_value = ent.get("start_char", ent.get("start"))
        end_value = ent.get("end_char", ent.get("end"))
        if start_value is None or end_value is None:
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
        try:
            start = int(start_value)
            end = int(end_value)
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
        raw_label_value = raw_label_hint or ent.get("label", "unknown")
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


def _iter_entities(payload: dict[str, Any]) -> list[tuple[Any, str | None]]:
    """Yield GLiNER2 entities from either the live label map or a flat list."""
    entities = payload.get("entities", [])
    if isinstance(entities, dict):
        flattened: list[tuple[Any, str | None]] = []
        for label, label_entities in entities.items():
            if not isinstance(label_entities, list):
                label_entities = [label_entities]
            for entity in label_entities:
                flattened.append((entity, label if isinstance(label, str) else None))
        return flattened
    if isinstance(entities, list):
        return [(entity, None) for entity in entities]
    return []
