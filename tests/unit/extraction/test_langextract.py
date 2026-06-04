from __future__ import annotations

import pytest

from pragmalens.extraction import normalize_langextract_records


def _record(*, interval, text: str, label: str = "claim", kind: str = "claim") -> dict[str, object]:
    return {
        "char_interval": interval,
        "extraction_text": text,
        "label": label,
        "kind": kind,
    }


@pytest.mark.parametrize(
    "records,text,expected_valid,expected_quarantine_warnings,expected_meta",
    [
        (
            [
                {
                    "char_interval": None,
                    "extraction_text": "ungrounded",
                    "label": "claim",
                    "kind": "claim",
                }
            ],
            "grounded text",
            0,
            ["missing_char_interval"],
            {"provider": "captured", "model": "captured"},
        ),
        (
            [_record(interval=(0, 4), text="good"),],
            "good text",
            1,
            [],
            {"provider": "captured", "model": "captured"},
        ),
        (
            [_record(interval=(0, 4), text="mismatch")],
            "good text",
            0,
            ["extraction_text_mismatch"],
            {"provider": "captured", "model": "captured"},
        ),
        (
            [
                {
                    "_meta": {
                        "provider": "provider-x",
                        "model": "model-y",
                        "prompt_hash": "hash-z",
                        "example_set": "fixture-a",
                    }
                },
                _record(interval=(0, 4), text="good"),
            ],
            "good text",
            1,
            [],
            {
                "provider": "provider-x",
                "model": "model-y",
                "prompt_hash": "hash-z",
                "example_set": "fixture-a",
            },
        ),
    ],
)
def test_langextract_normalization_covers_quarantine_and_provider_metadata(
    records: list[dict[str, object]],
    text: str,
    expected_valid: int,
    expected_quarantine_warnings: list[str],
    expected_meta: dict[str, str],
) -> None:
    valid, quarantined, meta = normalize_langextract_records(records, text, document_id="doc")

    assert len(valid) == expected_valid
    assert len(quarantined) == len(expected_quarantine_warnings)
    if quarantined:
        assert quarantined[0].warnings == expected_quarantine_warnings
    for key, value in expected_meta.items():
        assert meta[key] == value
