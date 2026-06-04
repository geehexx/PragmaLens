from __future__ import annotations

from pathlib import Path
from typing import cast

from hypothesis import given, settings
from hypothesis import strategies as st

from pragmalens.evaluation import (
    GoldenAnnotationLabel,
    GoldenSetAnnotation,
    load_corpus_approval_metadata,
    load_golden_set_batch,
)
from pragmalens.models import SpanRef

_FIXTURES = Path("tests/fixtures/golden")


def test_load_golden_set_batch_round_trips_fixture() -> None:
    batch = load_golden_set_batch(_FIXTURES / "pragmalens_golden_set.json")

    assert batch.golden_set_id == "pragmalens-golden-2026-06"
    assert [case.case_id for case in batch.cases] == [
        "recommendation-not-promise",
        "promise-review-ambiguity",
    ]
    assert [annotation.label for annotation in batch.cases[0].annotations] == [
        "promise",
        "review_commitment",
        "team_membership",
    ]
    assert batch.cases[0].annotations[0].resolved_to == "recommendation"
    assert batch.cases[0].annotations[2].resolved_to == "ambiguous"
    assert [annotation.label for annotation in batch.cases[1].annotations] == [
        "promise",
        "review_commitment",
        "antecedent_resolution",
        "team_membership",
    ]
    assert batch.cases[1].annotations[2].resolved_to == "Alice"
    assert batch.cases[1].annotations[3].notes == [
        "Do not assume Alice is part of the team from the previous sentence."
    ]


def test_load_golden_set_approval_metadata_round_trips_fixture() -> None:
    approval = load_corpus_approval_metadata(_FIXTURES / "pragmalens_golden_set.approval.json")

    assert approval.corpus_id == "pragmalens-golden-2026-06"
    assert approval.corpus_name == "PragmaLens Golden Set"
    assert approval.approval_status == "approved"
    assert approval.source_uri == "urn:pragmalens:golden-set"


@given(
    label=st.sampled_from(
        [
            "promise",
            "review_commitment",
            "antecedent_resolution",
            "team_membership",
        ]
    ),
    text=st.text(min_size=1, max_size=40),
    resolved_to=st.one_of(st.none(), st.text(min_size=1, max_size=20)),
    notes=st.lists(st.text(min_size=1, max_size=40), max_size=3),
)
@settings(max_examples=40)
def test_golden_set_annotation_round_trips_explicit_semantics(
    label: str,
    text: str,
    resolved_to: str | None,
    notes: list[str],
) -> None:
    annotation = GoldenSetAnnotation(
        label=cast(GoldenAnnotationLabel, label),
        span=SpanRef(
            document_id="golden",
            start_char=0,
            end_char=len(text),
            text=text,
        ),
        resolved_to=resolved_to,
        notes=notes,
    )

    payload = annotation.model_dump(mode="json")
    assert GoldenSetAnnotation.model_validate(payload).model_dump(mode="json") == payload
