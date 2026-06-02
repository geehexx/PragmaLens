from pathlib import Path
from typing import Any

from pragmalens.extraction import (
    merge_and_dedupe_candidates,
    normalize_gliner2_output,
    normalize_langextract_records,
)
from pragmalens.pipeline.captured_extraction import run_captured_extraction_pipeline


def test_langextract_quarantines_missing_char_interval() -> None:
    text = "When we act"
    records: list[dict[str, Any]] = [
        {
            "char_interval": [0, 4],
            "extraction_text": "When",
            "label": "condition",
            "kind": "condition",
        },
        {"char_interval": None, "extraction_text": "x", "label": "claim", "kind": "claim"},
    ]

    valid, quarantined, _ = normalize_langextract_records(records, text)
    assert len(valid) == 1
    assert len(quarantined) == 1
    assert "missing_char_interval" in quarantined[0].warnings


def test_gliner2_relations_preserved() -> None:
    text = "Alice signs plan"
    payload = {
        "entities": [
            {"start_char": 0, "end_char": 5, "label": "agent", "kind": "entity", "confidence": 0.9},
            {
                "start_char": 6,
                "end_char": 11,
                "label": "action",
                "kind": "claim",
                "confidence": 0.8,
            },
        ],
        "relations": [{"head": 1, "tail": 0, "label": "owned_by"}],
    }
    candidates = normalize_gliner2_output(payload, text)
    assert len(candidates) == 2
    action = next(c for c in candidates if c.label == "action")
    assert action.relations


def test_merge_dedupe_preserves_provenance() -> None:
    text = "When AI acts"
    records = [
        {
            "char_interval": [0, 4],
            "extraction_text": "When",
            "label": "condition",
            "kind": "condition",
        }
    ]
    valid, _, _ = normalize_langextract_records(records, text)
    dup = valid[0].model_copy(deep=True)
    dup.provenance = ["gliner2"]

    merged = merge_and_dedupe_candidates([valid[0], dup])
    assert len(merged) == 1
    assert sorted(merged[0].provenance) == ["gliner2", "langextract"]
    assert merged[0].attributes["merged_provenance"] == ["gliner2", "langextract"]


def test_merge_dedupe_surfaces_label_conflicts_and_dedupes_relations() -> None:
    text = "Alice signs"
    payload = {
        "entities": [
            {"start_char": 0, "end_char": 5, "label": "agent", "kind": "entity", "confidence": 0.9},
            {"start_char": 0, "end_char": 5, "label": "owner", "kind": "entity", "confidence": 0.7},
        ],
        "relations": [
            {"head": 0, "tail": 0, "label": "self"},
            {"head": 0, "tail": 0, "label": "self"},
        ],
    }
    candidates = normalize_gliner2_output(payload, text)

    merged = merge_and_dedupe_candidates(candidates)

    assert len(merged) == 2
    assert max(len(candidate.relations) for candidate in merged) == 1
    for candidate in merged:
        assert "label_conflict" in candidate.warnings
        assert candidate.attributes["conflicting_labels"] == ["agent", "owner"]


def test_captured_pipeline_writes_traces(tmp_path: Path) -> None:
    text = "When AI acts"
    out = run_captured_extraction_pipeline(
        text=text,
        langextract_jsonl="tests/fixtures/langextract_sample.jsonl",
        gliner2_json="tests/fixtures/gliner2_sample.json",
        trace_dir=tmp_path / "trace",
    )

    assert out["langextract_meta"]["prompt_hash"] == "sha256:fixture-pr03-v1"
    assert (tmp_path / "trace" / "langextract_candidates.json").exists()
    assert (tmp_path / "trace" / "gliner2_candidates.json").exists()
    assert (tmp_path / "trace" / "evidence_normalization.json").exists()
