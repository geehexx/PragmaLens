from __future__ import annotations

import os
import shutil
import subprocess
from functools import lru_cache
from typing import Any

import langextract as lx
import pytest
import spacy
from gliner2 import GLiNER2
from minicheck.minicheck import MiniCheck  # type: ignore[import-untyped]
from spacy.language import Language


def _require_live_smoke_enabled() -> None:
    """Require an explicit opt-in before touching live runtimes or remote services."""
    if os.environ.get("PRAGMALENS_ENABLE_LIVE_SMOKE") != "1":
        pytest.skip("Set PRAGMALENS_ENABLE_LIVE_SMOKE=1 to run real live smoke tests.")


@lru_cache(maxsize=1)
def _load_spacy_pipeline() -> Language:
    """Load the configured spaCy pipeline for live smoke verification."""
    model_name = os.environ.get("PRAGMALENS_SPACY_MODEL", "en_core_web_sm")
    return spacy.load(model_name)


@lru_cache(maxsize=1)
def _load_gliner2_model() -> GLiNER2:
    """Load the configured GLiNER2 model once for the live smoke lane."""
    model_name = os.environ.get("PRAGMALENS_GLINER2_MODEL", "fastino/gliner2-base-v1")
    return GLiNER2.from_pretrained(model_name)


def _langextract_provider_config() -> dict[str, str]:
    """Choose the best available live LangExtract backend for this machine."""
    provider = os.environ.get("PRAGMALENS_LANGEXTRACT_PROVIDER", "auto")
    if provider == "ollama":
        return _ollama_langextract_config()
    if provider == "gemini":
        return _gemini_langextract_config()
    if _ollama_model_available(_ollama_model_name()):
        return _ollama_langextract_config()
    if os.environ.get("LANGEXTRACT_API_KEY") or os.environ.get("GEMINI_API_KEY"):
        return _gemini_langextract_config()
    pytest.skip(
        "No live LangExtract backend available. Provide an Ollama model or a working Gemini key."
    )


def _ollama_langextract_config() -> dict[str, str]:
    """Build the local Ollama-backed LangExtract configuration."""
    if shutil.which("ollama") is None:
        pytest.skip("Ollama is not installed; cannot run local LangExtract live smoke.")
    model_name = _ollama_model_name()
    if not _ollama_model_available(model_name):
        pytest.skip(f"Ollama model '{model_name}' is not available for LangExtract live smoke.")
    return {
        "provider": "ollama",
        "model_id": model_name,
        "model_url": os.environ.get("PRAGMALENS_OLLAMA_URL", "http://localhost:11434"),
    }


def _gemini_langextract_config() -> dict[str, str]:
    """Build the Gemini-backed LangExtract configuration."""
    if not (os.environ.get("LANGEXTRACT_API_KEY") or os.environ.get("GEMINI_API_KEY")):
        pytest.skip(
            "Gemini-backed LangExtract live smoke requires LANGEXTRACT_API_KEY or GEMINI_API_KEY."
        )
    return {
        "provider": "gemini",
        "model_id": os.environ.get("PRAGMALENS_LANGEXTRACT_MODEL", "gemini-3.5-flash"),
    }


def _ollama_model_name() -> str:
    """Return the default local model for LangExtract live smoke."""
    return os.environ.get("PRAGMALENS_LANGEXTRACT_MODEL", "qwen3.5:0.8b")


@lru_cache(maxsize=4)
def _ollama_model_available(model_name: str) -> bool:
    """Return whether the requested Ollama model appears in `ollama list`."""
    if shutil.which("ollama") is None:
        return False
    result = subprocess.run(
        ["ollama", "list"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and model_name in result.stdout


@lru_cache(maxsize=1)
def _load_minicheck_scorer() -> Any:
    """Load the configured MiniCheck scorer once for the live smoke lane."""
    model_name = os.environ.get("PRAGMALENS_MINICHECK_MODEL", "roberta-large")
    cache_dir = os.environ.get("PRAGMALENS_MINICHECK_CACHE_DIR", ".local_state/minicheck-cache")
    return MiniCheck(model_name=model_name, cache_dir=cache_dir)


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
