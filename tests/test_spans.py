import pytest

from pragmalens.models import SpanRef


def test_valid_span_roundtrip() -> None:
    text = "claim"
    span = SpanRef(document_id="doc-1", start_char=0, end_char=5, text=text)
    assert span.end_char - span.start_char == len(span.text)


@pytest.mark.parametrize(
    "start,end",
    [
        (5, 5),
        (6, 5),
        (-1, 2),
    ],
)
def test_invalid_spans_rejected(start: int, end: int) -> None:
    with pytest.raises(Exception):
        SpanRef(document_id="doc-1", start_char=start, end_char=end, text="")


def test_text_length_mismatch_rejected() -> None:
    with pytest.raises(Exception):
        SpanRef(document_id="doc-1", start_char=0, end_char=4, text="abc")
