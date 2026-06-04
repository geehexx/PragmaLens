from __future__ import annotations

import importlib
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import pragmalens.live_runtime as live_runtime


@pytest.fixture(autouse=True)
def _clear_caches() -> None:
    live_runtime.load_spacy_pipeline.cache_clear()
    live_runtime.load_gliner2_model.cache_clear()
    live_runtime._ollama_model_available.cache_clear()


def test_load_spacy_pipeline_uses_package_model(monkeypatch: pytest.MonkeyPatch) -> None:
    spacy = SimpleNamespace(
        util=SimpleNamespace(is_package=lambda name: name == "en_core_web_sm"),
        load=lambda name: f"spacy:{name}",
    )
    monkeypatch.setattr(importlib, "import_module", lambda name: spacy if name == "spacy" else None)
    monkeypatch.setenv("PRAGMALENS_SPACY_MODEL", "en_core_web_sm")

    assert live_runtime.load_spacy_pipeline() == "spacy:en_core_web_sm"


def test_load_spacy_pipeline_uses_local_config_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model_dir = tmp_path / "spacy-model"
    model_dir.mkdir()
    (model_dir / "config.cfg").write_text("[model]\n", encoding="utf-8")
    spacy = SimpleNamespace(
        util=SimpleNamespace(is_package=lambda _: False),
        load=lambda name: f"spacy:{name}",
    )
    monkeypatch.setattr(importlib, "import_module", lambda name: spacy if name == "spacy" else None)
    monkeypatch.setenv("PRAGMALENS_SPACY_MODEL", str(model_dir))

    assert live_runtime.load_spacy_pipeline() == f"spacy:{model_dir}"


def test_load_spacy_pipeline_requires_install(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: (_ for _ in ()).throw(ImportError("missing")) if name == "spacy" else None,
    )

    with pytest.raises(RuntimeError, match="Install spaCy"):
        live_runtime.load_spacy_pipeline()


def test_load_gliner2_model_loads_configured_model(monkeypatch: pytest.MonkeyPatch) -> None:
    gliner2 = SimpleNamespace(
        GLiNER2=SimpleNamespace(from_pretrained=lambda name: f"gliner2:{name}"),
    )
    monkeypatch.setattr(
        importlib, "import_module", lambda name: gliner2 if name == "gliner2" else None
    )
    monkeypatch.setenv("PRAGMALENS_GLINER2_MODEL", "fastino/demo-model")

    assert live_runtime.load_gliner2_model() == "gliner2:fastino/demo-model"


def test_load_gliner2_model_requires_install(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: (_ for _ in ()).throw(ImportError("missing")) if name == "gliner2" else None,
    )

    with pytest.raises(RuntimeError, match="Install GLiNER2"):
        live_runtime.load_gliner2_model()


def test_load_minicheck_scorer_loads_configured_model(monkeypatch: pytest.MonkeyPatch) -> None:
    minicheck = SimpleNamespace(MiniCheck=lambda model_name, cache_dir: (model_name, cache_dir))
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: minicheck if name == "minicheck.minicheck" else None,
    )
    monkeypatch.setenv("PRAGMALENS_MINICHECK_MODEL", "mini-model")
    monkeypatch.setenv("PRAGMALENS_MINICHECK_CACHE_DIR", "/tmp/mini-cache")

    assert live_runtime.load_minicheck_scorer() == ("mini-model", "/tmp/mini-cache")


def test_load_minicheck_scorer_requires_install(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: (
            (_ for _ in ()).throw(ImportError("missing")) if name == "minicheck.minicheck" else None
        ),
    )

    with pytest.raises(RuntimeError, match="Install MiniCheck"):
        live_runtime.load_minicheck_scorer()


def test_langextract_provider_config_covers_all_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    def _ollama_available(model_name: str) -> bool:
        return model_name == "ollama-model"

    monkeypatch.setattr(live_runtime, "_ollama_langextract_config", lambda: {"provider": "ollama"})
    monkeypatch.setattr(live_runtime, "_gemini_langextract_config", lambda: {"provider": "gemini"})
    monkeypatch.setattr(live_runtime, "_ollama_model_name", lambda: "ollama-model")
    monkeypatch.setattr(live_runtime, "_ollama_model_available", _ollama_available)

    monkeypatch.setenv("PRAGMALENS_LANGEXTRACT_PROVIDER", "ollama")
    assert live_runtime.langextract_provider_config() == {"provider": "ollama"}

    monkeypatch.setenv("PRAGMALENS_LANGEXTRACT_PROVIDER", "gemini")
    assert live_runtime.langextract_provider_config() == {"provider": "gemini"}

    monkeypatch.delenv("PRAGMALENS_LANGEXTRACT_PROVIDER", raising=False)
    assert live_runtime.langextract_provider_config() == {"provider": "ollama"}

    monkeypatch.setattr(live_runtime, "_ollama_model_available", lambda _: False)
    monkeypatch.setenv("LANGEXTRACT_API_KEY", "key")
    assert live_runtime.langextract_provider_config() == {"provider": "gemini"}

    monkeypatch.delenv("LANGEXTRACT_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="No live LangExtract backend available"):
        live_runtime.langextract_provider_config()


def test_ollama_langextract_config_checks_binary_and_model(monkeypatch: pytest.MonkeyPatch) -> None:
    def _ollama_available(model_name: str) -> bool:
        return model_name == "qwen3.5:0.8b"

    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/ollama")
    monkeypatch.setattr(live_runtime, "_ollama_model_name", lambda: "qwen3.5:0.8b")
    monkeypatch.setattr(live_runtime, "_ollama_model_available", _ollama_available)
    monkeypatch.setenv("PRAGMALENS_OLLAMA_URL", "http://localhost:11434")

    assert live_runtime._ollama_langextract_config() == {
        "provider": "ollama",
        "model_id": "qwen3.5:0.8b",
        "model_url": "http://localhost:11434",
    }


def test_ollama_langextract_config_requires_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: None)

    with pytest.raises(RuntimeError, match="Ollama is not installed"):
        live_runtime._ollama_langextract_config()


def test_ollama_model_available_checks_command_output(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: None)
    assert live_runtime._ollama_model_available("qwen3.5:0.8b") is False

    class _Result:
        returncode = 0
        stdout = "qwen3.5:0.8b\n"

    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/ollama")
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: _Result())
    live_runtime._ollama_model_available.cache_clear()
    assert live_runtime._ollama_model_available("qwen3.5:0.8b") is True


def test_gemini_langextract_config_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LANGEXTRACT_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="Gemini-backed LangExtract"):
        live_runtime._gemini_langextract_config()

    monkeypatch.setenv("PRAGMALENS_LANGEXTRACT_MODEL", "gemini-test")
    monkeypatch.setenv("LANGEXTRACT_API_KEY", "key")
    assert live_runtime._gemini_langextract_config() == {
        "provider": "gemini",
        "model_id": "gemini-test",
    }
