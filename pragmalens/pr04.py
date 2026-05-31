from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pragmalens.extraction import (
    load_json,
    load_jsonl,
    merge_and_dedupe_candidates,
    normalize_gliner2_output,
    normalize_langextract_records,
)
from pragmalens.models import EvidenceCandidate


def run_captured_extraction(
    *,
    text: str,
    langextract_jsonl: str | Path,
    gliner2_json: str | Path,
    trace_dir: str | Path,
) -> dict[str, Any]:
    lx_records = load_jsonl(langextract_jsonl)
    gl_payload = load_json(gliner2_json)

    lx_valid, lx_quarantined, lx_meta = normalize_langextract_records(lx_records, text)
    gl_candidates = normalize_gliner2_output(gl_payload, text)
    merged = merge_and_dedupe_candidates(lx_valid + gl_candidates)

    trace = Path(trace_dir)
    trace.mkdir(parents=True, exist_ok=True)
    (trace / "langextract_raw.jsonl").write_text(Path(langextract_jsonl).read_text(encoding="utf-8"), encoding="utf-8")
    (trace / "langextract_candidates.json").write_text(
        json.dumps([c.model_dump(mode="json") for c in lx_valid], indent=2) + "\n", encoding="utf-8"
    )
    (trace / "langextract_quarantined.json").write_text(
        json.dumps([c.model_dump(mode="json") for c in lx_quarantined], indent=2) + "\n", encoding="utf-8"
    )
    (trace / "gliner2_raw.json").write_text(Path(gliner2_json).read_text(encoding="utf-8"), encoding="utf-8")
    (trace / "gliner2_candidates.json").write_text(
        json.dumps([c.model_dump(mode="json") for c in gl_candidates], indent=2) + "\n", encoding="utf-8"
    )
    (trace / "evidence_normalization.json").write_text(
        json.dumps([c.model_dump(mode="json") for c in merged], indent=2) + "\n", encoding="utf-8"
    )

    return {
        "langextract_meta": lx_meta,
        "valid_candidates": merged,
        "quarantined_candidates": lx_quarantined,
    }


def candidates_to_report_payload(candidates: list[EvidenceCandidate]) -> list[dict[str, Any]]:
    return [c.model_dump(mode="json") for c in candidates]
