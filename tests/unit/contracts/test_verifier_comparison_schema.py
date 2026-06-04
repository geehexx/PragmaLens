import json
from pathlib import Path

from pragmalens.schema import export_schema


def test_schema_export_verifier_batch_comparison(tmp_path: Path) -> None:
    out = tmp_path / "verifier_batch_comparison.schema.json"
    path = export_schema("verifier_batch_comparison", str(out))
    data = json.loads(path.read_text(encoding="utf-8"))

    assert data["title"] == "VerifierBatchComparison"
    assert "selected_backend" in data["properties"]
    assert "records" in data["properties"]
