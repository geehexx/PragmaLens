from pragmalens.core import normalize_document


def test_neutral_report_shape_defaults() -> None:
    report = normalize_document(text="# Doc\n\nClaim", document_id="doc-123")

    payload = report.model_dump(mode="json")
    assert payload["report_version"] == "0.1"
    assert payload["document_id"] == "doc-123"
    assert payload["source_format"] == "markdown_or_text"
    assert payload["findings"] == []
    assert payload["candidates"] == []
    assert payload["warnings"] == []
