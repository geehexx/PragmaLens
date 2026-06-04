from __future__ import annotations

from typing import Any, cast

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pragmalens.extraction import normalize_langextract_records


def _record(
    *, interval: tuple[int, int], text: str, label: str = "claim", kind: str = "claim"
) -> dict[str, object]:
    return {
        "char_interval": interval,
        "extraction_text": text,
        "label": label,
        "kind": kind,
    }


@st.composite
def _valid_record_and_text(draw: Any) -> tuple[dict[str, object], str]:
    text = draw(st.text(min_size=1, max_size=20))
    start = draw(st.integers(min_value=0, max_value=len(text) - 1))
    end = draw(st.integers(min_value=start + 1, max_value=len(text)))
    label = draw(st.sampled_from(["claim", "evidence", "actor"]))
    kind = draw(st.sampled_from(["claim", "evidence", "actor"]))
    record = _record(
        interval=(start, end),
        text=text[start:end],
        label=label,
        kind=kind,
    )
    return record, text


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
            [
                _record(interval=(0, 4), text="good"),
            ],
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


@given(payload_and_text=_valid_record_and_text())
@settings(max_examples=40)
def test_langextract_normalization_round_trips_valid_records(
    payload_and_text: tuple[dict[str, object], str],
) -> None:
    record, text = payload_and_text

    valid, quarantined, meta = normalize_langextract_records([record], text, document_id="doc")

    assert len(valid) == 1
    assert quarantined == []
    assert meta["provider"] == "captured"
    assert meta["model"] == "captured"
    assert valid[0].span.text == record["extraction_text"]
    char_interval = cast(tuple[int, int], record["char_interval"])
    assert valid[0].span.start_char == char_interval[0]
    assert valid[0].span.end_char == char_interval[1]
