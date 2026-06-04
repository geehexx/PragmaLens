"""Fixture-backed extraction pipeline helpers used by offline tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pragmalens.extraction import (
    load_json,
    load_jsonl,
    normalize_candidate_set,
    normalize_gliner2_output_with_quarantine,
    normalize_langextract_records,
    resolve_data_path,
)
from pragmalens.models import EvidenceCandidate
from pragmalens.profiles import load_profile


def run_captured_extraction_pipeline(
    *,
    text: str,
    langextract_jsonl: str | Path,
    gliner2_json: str | Path,
    trace_dir: str | Path,
    document_id: str = "captured-input",
    label_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run the fixture-backed extraction pipeline used by offline tests."""
    lx_records = load_jsonl(langextract_jsonl)
    gl_payload = load_json(gliner2_json)
    resolved_label_map = label_map if label_map is not None else load_profile().label_map

    lx_valid, lx_quarantined, lx_meta = normalize_langextract_records(
        lx_records, text, document_id=document_id
    )
    gl_normalized = normalize_gliner2_output_with_quarantine(
        gl_payload,
        text,
        document_id=document_id,
        label_map=resolved_label_map,
    )
    normalization = normalize_candidate_set(
        lx_valid + gl_normalized.valid,
        quarantined=lx_quarantined + gl_normalized.quarantined,
    )

    trace = Path(trace_dir)
    trace.mkdir(parents=True, exist_ok=True)
    (trace / "langextract_raw.jsonl").write_text(
        resolve_data_path(langextract_jsonl).read_text(encoding="utf-8"), encoding="utf-8"
    )
    (trace / "langextract_candidates.json").write_text(
        json.dumps([c.model_dump(mode="json") for c in lx_valid], indent=2) + "\n", encoding="utf-8"
    )
    (trace / "langextract_quarantined.json").write_text(
        json.dumps([c.model_dump(mode="json") for c in lx_quarantined], indent=2) + "\n",
        encoding="utf-8",
    )
    (trace / "gliner2_raw.json").write_text(
        resolve_data_path(gliner2_json).read_text(encoding="utf-8"), encoding="utf-8"
    )
    (trace / "gliner2_candidates.json").write_text(
        json.dumps([c.model_dump(mode="json") for c in gl_normalized.valid], indent=2) + "\n",
        encoding="utf-8",
    )
    (trace / "gliner2_quarantined.json").write_text(
        json.dumps([c.model_dump(mode="json") for c in gl_normalized.quarantined], indent=2) + "\n",
        encoding="utf-8",
    )
    (trace / "evidence_normalization.json").write_text(
        json.dumps(
            {
                "candidates": [c.model_dump(mode="json") for c in normalization.candidates],
                "quarantined": [c.model_dump(mode="json") for c in normalization.quarantined],
                "duplicates": [c.model_dump(mode="json") for c in normalization.duplicates],
                "duplicate_events": normalization.duplicate_events,
                "conflicts": normalization.conflicts,
                "stats": normalization.stats,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return {
        "langextract_meta": lx_meta,
        "valid_candidates": normalization.candidates,
        "quarantined_candidates": normalization.quarantined,
        "duplicate_candidates": normalization.duplicates,
    }


def candidates_to_report_payload(candidates: list[EvidenceCandidate]) -> list[dict[str, Any]]:
    """Serialize evidence candidates into JSON-compatible report payloads."""
    return [c.model_dump(mode="json") for c in candidates]
