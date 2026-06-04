import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from pragmalens.models import SpanRef


@given(
    document_id=st.text(min_size=1, max_size=20),
    text=st.text(min_size=1, max_size=20),
)
@settings(max_examples=50)
def test_valid_span_roundtrip(document_id: str, text: str) -> None:
    span = SpanRef(document_id=document_id, start_char=0, end_char=len(text), text=text)

    payload = span.model_dump(mode="json")
    assert payload["end_char"] - payload["start_char"] == len(payload["text"])
    assert SpanRef.model_validate(payload).model_dump(mode="json") == payload


@pytest.mark.parametrize(
    "start,end",
    [
        (5, 5),
        (6, 5),
        (-1, 2),
    ],
)
def test_invalid_spans_rejected(start: int, end: int) -> None:
    with pytest.raises(ValidationError):
        SpanRef(document_id="doc-1", start_char=start, end_char=end, text="")


def test_text_length_mismatch_rejected() -> None:
    with pytest.raises(ValidationError):
        SpanRef(document_id="doc-1", start_char=0, end_char=4, text="abc")
