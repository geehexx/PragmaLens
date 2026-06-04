from __future__ import annotations

import os

import pytest

from pragmalens.live_runtime import (
    langextract_provider_config as _langextract_provider_config,
)
from pragmalens.live_runtime import (
    load_gliner2_model as _load_gliner2_model,
)
from pragmalens.live_runtime import (
    load_minicheck_scorer as _load_minicheck_scorer,
)
from pragmalens.live_runtime import (
    load_spacy_pipeline as _load_spacy_pipeline,
)
from pragmalens.models import CandidateStatus, EvidenceCandidate, SpanRef, VerificationStatus
from pragmalens.verifier import build_verifier_adapter


def _require_live_smoke_enabled() -> None:
    """Require an explicit opt-in before touching live runtimes or remote services."""
    if os.environ.get("PRAGMALENS_ENABLE_LIVE_SMOKE") != "1":
        pytest.skip("Set PRAGMALENS_ENABLE_LIVE_SMOKE=1 to run real live smoke tests.")


@pytest.mark.live_smoke
def test_spacy_live_smoke_uses_real_pipeline() -> None:
    _require_live_smoke_enabled()

    nlp = _load_spacy_pipeline()
    doc = nlp("Ship it. Do not wait.")

    assert [sent.text for sent in doc.sents] == ["Ship it.", "Do not wait."]
    assert doc[0].pos_ == "VERB"


@pytest.mark.live_smoke
@pytest.mark.slow
def test_gliner2_live_smoke_extracts_entities_with_spans() -> None:
    _require_live_smoke_enabled()

    extractor = _load_gliner2_model()
    result = extractor.extract_entities(
        "Apple CEO Tim Cook announced iPhone 15 in Cupertino.",
        ["company", "person", "product", "location"],
        include_confidence=True,
        include_spans=True,
    )

    assert result["entities"]["company"][0]["text"] == "Apple"
    assert result["entities"]["person"][0]["text"] == "Tim Cook"
    assert result["entities"]["product"][0]["text"] == "iPhone 15"
    assert result["entities"]["location"][0]["start"] == 42
    assert result["entities"]["company"][0]["confidence"] > 0.5


@pytest.mark.live_smoke
@pytest.mark.slow
def test_langextract_live_smoke_extracts_grounded_commitment() -> None:
    _require_live_smoke_enabled()

    lx = pytest.importorskip("langextract")
    config = _langextract_provider_config()
    example = lx.data.ExampleData(
        text="Alice promised to ship the report tomorrow.",
        extractions=[
            lx.data.Extraction(extraction_class="actor", extraction_text="Alice"),
            lx.data.Extraction(
                extraction_class="commitment",
                extraction_text="promised to ship the report tomorrow",
            ),
        ],
    )
    kwargs: dict[str, str] = {"model_id": config["model_id"]}
    if config["provider"] == "ollama":
        kwargs["model_url"] = config["model_url"]

    result = lx.extract(
        text_or_documents="Bob promised to send the draft tonight.",
        prompt_description="Extract the actor and the commitment phrase.",
        examples=[example],
        **kwargs,
    )

    classes = {extraction.extraction_class for extraction in result.extractions}
    assert {"actor", "commitment"} <= classes
    assert any(extraction.extraction_text == "Bob" for extraction in result.extractions)
    assert any("promised" in extraction.extraction_text for extraction in result.extractions)
    assert all(extraction.char_interval is not None for extraction in result.extractions)


@pytest.mark.live_smoke
@pytest.mark.slow
def test_minicheck_live_smoke_scores_supported_vs_unsupported_claims() -> None:
    _require_live_smoke_enabled()

    scorer = _load_minicheck_scorer()
    doc = (
        "A group of students gather in the school library to study for their upcoming final exams."
    )
    pred_label, raw_prob, _, _ = scorer.score(
        docs=[doc, doc],
        claims=[
            "The students are preparing for an examination.",
            "The students are on vacation.",
        ],
    )

    assert pred_label == [1, 0]
    assert raw_prob[0] > 0.5
    assert raw_prob[1] < 0.5


@pytest.mark.live_smoke
@pytest.mark.slow
def test_crossencoder_live_smoke_maps_supported_and_unsupported_claims() -> None:
    _require_live_smoke_enabled()

    verifier = build_verifier_adapter(
        "crossencoder_nli",
        model_name=os.environ.get(
            "PRAGMALENS_CROSSENCODER_MODEL",
            "cross-encoder/nli-deberta-v3-base",
        ),
    )
    candidates = [
        EvidenceCandidate(
            candidate_id="candidate-supported",
            label="claim",
            kind="claim",
            status=CandidateStatus.VALID,
            span=SpanRef(
                document_id="doc",
                start_char=0,
                end_char=20,
                text="A man eats something",
            ),
            provenance=["fixture"],
            evidence_refs=["ref-supported"],
        ),
        EvidenceCandidate(
            candidate_id="candidate-unsupported",
            label="claim",
            kind="claim",
            status=CandidateStatus.VALID,
            span=SpanRef(
                document_id="doc",
                start_char=0,
                end_char=36,
                text="A man is driving down a lonely road.",
            ),
            provenance=["fixture"],
            evidence_refs=["ref-unsupported"],
        ),
    ]
    verdicts = verifier.verify(
        candidates,
        document_id="doc",
        text="A man is eating pizza",
    )

    assert verdicts[0].status is VerificationStatus.SUPPORTED
    assert verdicts[1].status is VerificationStatus.UNSUPPORTED
    assert verdicts[0].evidence_ids == ["ref-supported"]
    assert verdicts[1].evidence_ids == ["ref-unsupported"]
