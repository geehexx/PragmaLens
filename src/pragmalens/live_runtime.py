"""Shared live-runtime loaders for smoke tests and captured fixture refreshes."""

from __future__ import annotations

import importlib
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

from pragmalens.settings import load_settings


@lru_cache(maxsize=1)
def load_spacy_pipeline() -> Any:
    """Load the configured spaCy pipeline for live smoke verification."""
    settings = load_settings()
    try:
        spacy = importlib.import_module("spacy")
    except ImportError as exc:
        raise RuntimeError(
            "Install spaCy with `uv sync --group live` before running live smoke tests."
        ) from exc
    model_name = settings.spacy_model
    model_path = Path(model_name)
    if not (spacy.util.is_package(model_name) or (model_path / "config.cfg").is_file()):
        raise RuntimeError(
            "Install the spaCy model with "
            f"`uv run python -m spacy download {model_name}` before running live smoke tests."
        )
    return spacy.load(model_name)


@lru_cache(maxsize=1)
def load_gliner2_model() -> Any:
    """Load the configured GLiNER2 model once for the live smoke lane."""
    settings = load_settings()
    try:
        gliner2_module = importlib.import_module("gliner2")
    except ImportError as exc:
        raise RuntimeError(
            "Install GLiNER2 with `uv sync --group live` before running live smoke tests."
        ) from exc
    model_name = settings.gliner2_model
    return gliner2_module.GLiNER2.from_pretrained(model_name)


def langextract_provider_config() -> dict[str, str]:
    """Choose the best available live LangExtract backend for this machine."""
    settings = load_settings()
    provider = settings.langextract_provider
    if provider == "ollama":
        return _ollama_langextract_config()
    if provider == "gemini":
        return _gemini_langextract_config()
    if _ollama_model_available(_ollama_model_name()):
        return _ollama_langextract_config()
    if settings.langextract_api_key or settings.gemini_api_key:
        return _gemini_langextract_config()
    raise RuntimeError(
        "No live LangExtract backend available. Provide an Ollama model or a working Gemini key."
    )


def load_minicheck_scorer() -> Any:
    """Load the configured MiniCheck scorer once for the live smoke lane."""
    settings = load_settings()
    try:
        minicheck_module = importlib.import_module("minicheck.minicheck")
    except ImportError as exc:
        raise RuntimeError(
            "Install MiniCheck with `uv sync --group live` before running live smoke tests."
        ) from exc
    return minicheck_module.MiniCheck(
        model_name=settings.minicheck_model,
        cache_dir=str(settings.minicheck_cache_dir),
    )


def _ollama_langextract_config() -> dict[str, str]:
    """Build the local Ollama-backed LangExtract configuration."""
    settings = load_settings()
    if shutil.which("ollama") is None:
        raise RuntimeError("Ollama is not installed; cannot run local LangExtract live smoke.")
    model_name = settings.langextract_model
    if not _ollama_model_available(model_name):
        raise RuntimeError(
            f"Ollama model '{model_name}' is not available for LangExtract live smoke."
        )
    return {
        "provider": "ollama",
        "model_id": model_name,
        "model_url": settings.ollama_url,
    }


def _gemini_langextract_config() -> dict[str, str]:
    """Build the Gemini-backed LangExtract configuration."""
    settings = load_settings()
    if not (settings.langextract_api_key or settings.gemini_api_key):
        raise RuntimeError(
            "Gemini-backed LangExtract live smoke requires LANGEXTRACT_API_KEY or GEMINI_API_KEY."
        )
    return {
        "provider": "gemini",
        "model_id": settings.langextract_model,
    }


def _ollama_model_name() -> str:
    """Return the default local model for LangExtract live smoke."""
    return load_settings().langextract_model


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
