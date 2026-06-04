from __future__ import annotations

from pathlib import Path

from pragmalens.evaluation import (
    load_corpus_approval_metadata,
    load_golden_set_batch,
)

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
