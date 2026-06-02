import pytest

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
