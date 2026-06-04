from pragmalens.extraction import normalize_langextract_records


def test_ungrounded_langextract_records_are_quarantined() -> None:
    records = [
        {"char_interval": None, "extraction_text": "ungrounded", "label": "claim", "kind": "claim"}
    ]

    valid, quarantined, meta = normalize_langextract_records(
        records, "grounded text", document_id="doc"
    )

    assert valid == []
    assert len(quarantined) == 1
    assert quarantined[0].status == "quarantined"
    assert quarantined[0].warnings == ["missing_char_interval"]
    assert meta["provider"] == "captured"
