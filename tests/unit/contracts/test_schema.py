import json
from pathlib import Path

from pragmalens.schema import export_schema


def test_schema_export_report(tmp_path: Path) -> None:
    out = tmp_path / "report.schema.json"
    path = export_schema("report", str(out))
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["title"] == "NeutralReport"
    assert "properties" in data


def test_schema_export_evidence_candidate_includes_evidence_refs(tmp_path: Path) -> None:
    out = tmp_path / "evidence_candidate.schema.json"
    path = export_schema("evidence_candidate", str(out))
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["title"] == "EvidenceCandidate"
    assert "evidence_refs" in data["properties"]


def test_schema_export_invalid_model(tmp_path: Path) -> None:
    try:
        export_schema("unknown", str(tmp_path / "x.json"))
    except ValueError as exc:
        assert "Unknown model" in str(exc)
    else:
        raise AssertionError("expected ValueError")
