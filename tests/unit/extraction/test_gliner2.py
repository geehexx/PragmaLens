from typing import cast

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pragmalens.extraction import normalize_gliner2_output, normalize_gliner2_output_with_quarantine
from pragmalens.profiles import ProfileModel


def test_gliner2_candidates_normalize_span_confidence_and_label_map() -> None:
    payload = {
        "entities": [
            {
                "text": "team",
                "label": "agent",
                "start_char": 4,
                "end_char": 8,
                "confidence": 0.91,
            }
        ]
    }

    candidates = normalize_gliner2_output(
        payload,
        "The team will ship.",
        document_id="doc",
        label_map={"agent": "commitment_owner"},
    )

    assert len(candidates) == 1
    assert candidates[0].label == "commitment_owner"
    assert candidates[0].span.text == "team"
    assert candidates[0].confidence == 0.91


def test_model_profile_requires_fixture_paths_for_offline_default() -> None:
    with pytest.raises(ValueError):
        ProfileModel(name="local", langextract_fixture="", gliner2_fixture="", label_map={})


def test_gliner2_invalid_span_is_quarantined() -> None:
    payload = {
        "entities": [
            {"text": "team", "label": "agent", "start_char": -1, "end_char": 8, "confidence": 0.91}
        ]
    }

    normalized = normalize_gliner2_output_with_quarantine(
        payload, "The team will ship.", document_id="doc"
    )

    assert normalized.valid == []
    assert len(normalized.quarantined) == 1
    assert normalized.quarantined[0].status == "quarantined"
    assert normalized.quarantined[0].warnings == ["invalid_char_interval"]


def test_gliner2_live_captured_payload_flattens_label_groups() -> None:
    payload = {
        "entities": {
            "agent": [
                {"text": "Alice", "confidence": 0.9956095814704895, "start": 0, "end": 5},
            ],
            "action": [
                {"text": "review", "confidence": 0.7919526696205139, "start": 58, "end": 64},
                {
                    "text": "ship the report",
                    "confidence": 0.6691595911979675,
                    "start": 18,
                    "end": 33,
                },
            ],
            "condition": [],
        }
    }

    candidates = normalize_gliner2_output(
        payload,
        "Alice promised to ship the report tomorrow. The team will review the draft before Friday.",
        document_id="doc",
        label_map={"agent": "commitment_owner", "action": "commitment_action"},
    )

    assert [candidate.label for candidate in candidates] == [
        "commitment_owner",
        "commitment_action",
        "commitment_action",
    ]
    assert [candidate.span.text for candidate in candidates] == [
        "Alice",
        "review",
        "ship the report",
    ]


@st.composite
def _valid_gliner2_payload(draw):
    text = draw(st.text(min_size=1, max_size=20))
    start = draw(st.integers(min_value=0, max_value=len(text) - 1))
    end = draw(st.integers(min_value=start + 1, max_value=len(text)))
    confidence = draw(
        st.floats(
            min_value=0.5,
            max_value=1.0,
            allow_nan=False,
            allow_infinity=False,
        )
    )
    return (
        {
            "entities": [
                {
                    "text": text[start:end],
                    "label": "agent",
                    "start_char": start,
                    "end_char": end,
                    "confidence": confidence,
                }
            ]
        },
        text,
    )


@pytest.mark.parametrize(
    ("entity", "warning"),
    [
        ({"label": "agent", "end_char": 8}, "missing_char_interval"),
        ({"label": "agent", "start_char": "x", "end_char": 8}, "invalid_char_interval"),
        ("not-a-dict", "invalid_entity_payload"),
    ],
)
def test_gliner2_malformed_entities_are_quarantined(entity: object, warning: str) -> None:
    payload = {"entities": [entity]}

    normalized = normalize_gliner2_output_with_quarantine(
        payload, "The team will ship.", document_id="doc"
    )

    assert normalized.valid == []
    assert len(normalized.quarantined) == 1
    assert normalized.quarantined[0].warnings == [warning]


@given(payload_and_text=_valid_gliner2_payload())
@settings(max_examples=40)
def test_gliner2_valid_payload_round_trips_span_and_label_map(
    payload_and_text: tuple[dict[str, object], str],
) -> None:
    payload, text = payload_and_text
    entities = cast(list[dict[str, object]], payload["entities"])
    entity = entities[0]

    candidates = normalize_gliner2_output(
        payload,
        text,
        document_id="doc",
        label_map={"agent": "commitment_owner"},
    )

    assert len(candidates) == 1
    assert candidates[0].label == "commitment_owner"
    assert candidates[0].span.text == cast(str, entity["text"])
    assert candidates[0].span.start_char == cast(int, entity["start_char"])
    assert candidates[0].span.end_char == cast(int, entity["end_char"])
