from __future__ import annotations

from pathlib import Path

import pytest

from pragmalens.settings import PragmaLensSettings, load_settings


def test_load_settings_defaults_without_env(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_runtime_env(monkeypatch)

    settings = load_settings()

    assert settings == PragmaLensSettings()
    assert settings.minicheck_cache_dir == Path(".local_state/minicheck-cache")
    assert settings.crossencoder_cache_dir == Path(".local_state/crossencoder-cache")
    assert settings.crossencoder_revision == "f2f24f9fce8fc5b34aedf861f5c819c6ba0cf4f5"


def test_load_settings_applies_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_runtime_env(monkeypatch)
    monkeypatch.setenv("PRAGMALENS_SPACY_MODEL", "en_core_web_md")
    monkeypatch.setenv("PRAGMALENS_GLINER2_MODEL", "fastino/demo-model")
    monkeypatch.setenv("PRAGMALENS_LANGEXTRACT_PROVIDER", "ollama")
    monkeypatch.setenv("PRAGMALENS_LANGEXTRACT_MODEL", "qwen3.5:1.5b")
    monkeypatch.setenv("PRAGMALENS_OLLAMA_URL", "http://localhost:11435")
    monkeypatch.setenv("LANGEXTRACT_API_KEY", "langextract-key")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    monkeypatch.setenv("PRAGMALENS_VERIFIER", "minicheck")
    monkeypatch.setenv("PRAGMALENS_MINICHECK_MODEL", "mini-model")
    monkeypatch.setenv("PRAGMALENS_MINICHECK_CACHE_DIR", "/tmp/minicheck-cache")
    monkeypatch.setenv("PRAGMALENS_CROSSENCODER_MODEL", "ce-model")
    monkeypatch.setenv("PRAGMALENS_CROSSENCODER_CACHE_DIR", "/tmp/crossencoder-cache")
    monkeypatch.setenv(
        "PRAGMALENS_CROSSENCODER_REVISION",
        "f2f24f9fce8fc5b34aedf861f5c819c6ba0cf4f5",
    )

    settings = load_settings()

    assert settings.spacy_model == "en_core_web_md"
    assert settings.gliner2_model == "fastino/demo-model"
    assert settings.langextract_provider == "ollama"
    assert settings.langextract_model == "qwen3.5:1.5b"
    assert settings.ollama_url == "http://localhost:11435"
    assert settings.langextract_api_key == "langextract-key"
    assert settings.gemini_api_key == "gemini-key"
    assert settings.verifier_backend == "minicheck"
    assert settings.minicheck_model == "mini-model"
    assert settings.minicheck_cache_dir == Path("/tmp/minicheck-cache")
    assert settings.crossencoder_model == "ce-model"
    assert settings.crossencoder_cache_dir == Path("/tmp/crossencoder-cache")
    assert settings.crossencoder_revision == "f2f24f9fce8fc5b34aedf861f5c819c6ba0cf4f5"


def test_load_settings_rejects_invalid_verifier_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_runtime_env(monkeypatch)
    monkeypatch.setenv("PRAGMALENS_VERIFIER", "not-a-backend")

    with pytest.raises(ValueError, match="invalid PragmaLens runtime settings"):
        load_settings()


def _clear_runtime_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in [
        "PRAGMALENS_SPACY_MODEL",
        "PRAGMALENS_GLINER2_MODEL",
        "PRAGMALENS_LANGEXTRACT_PROVIDER",
        "PRAGMALENS_LANGEXTRACT_MODEL",
        "PRAGMALENS_OLLAMA_URL",
        "LANGEXTRACT_API_KEY",
        "GEMINI_API_KEY",
        "PRAGMALENS_VERIFIER",
        "PRAGMALENS_MINICHECK_MODEL",
        "PRAGMALENS_MINICHECK_CACHE_DIR",
        "PRAGMALENS_CROSSENCODER_MODEL",
        "PRAGMALENS_CROSSENCODER_CACHE_DIR",
        "PRAGMALENS_CROSSENCODER_REVISION",
    ]:
        monkeypatch.delenv(name, raising=False)
