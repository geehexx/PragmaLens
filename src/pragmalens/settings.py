"""Typed runtime settings for live and verifier backends."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class PragmaLensSettings(BaseModel):
    """Validated runtime configuration resolved from environment variables."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    spacy_model: str = Field(default="en_core_web_sm", min_length=1)
    gliner2_model: str = Field(default="fastino/gliner2-base-v1", min_length=1)
    langextract_provider: Literal["auto", "ollama", "gemini"] = "auto"
    langextract_model: str = Field(default="qwen3.5:0.8b", min_length=1)
    ollama_url: str = Field(default="http://localhost:11434", min_length=1)
    langextract_api_key: str | None = None
    gemini_api_key: str | None = None
    verifier_backend: Literal["offline", "minicheck", "crossencoder_nli"] = "offline"
    minicheck_model: str = Field(default="roberta-large", min_length=1)
    minicheck_cache_dir: Path = Field(default_factory=lambda: Path(".local_state/minicheck-cache"))
    crossencoder_model: str = Field(default="cross-encoder/nli-deberta-v3-base", min_length=1)
    crossencoder_cache_dir: Path = Field(
        default_factory=lambda: Path(".local_state/crossencoder-cache")
    )
    crossencoder_revision: str = Field(
        default="f2f24f9fce8fc5b34aedf861f5c819c6ba0cf4f5",
        min_length=1,
    )

    @classmethod
    def from_env(cls) -> PragmaLensSettings:
        """Load runtime settings from the current process environment."""
        try:
            payload: dict[str, Any] = {
                "spacy_model": os.environ.get("PRAGMALENS_SPACY_MODEL", "en_core_web_sm"),
                "gliner2_model": os.environ.get(
                    "PRAGMALENS_GLINER2_MODEL",
                    "fastino/gliner2-base-v1",
                ),
                "langextract_provider": os.environ.get(
                    "PRAGMALENS_LANGEXTRACT_PROVIDER",
                    "auto",
                ),
                "langextract_model": os.environ.get(
                    "PRAGMALENS_LANGEXTRACT_MODEL",
                    "qwen3.5:0.8b",
                ),
                "ollama_url": os.environ.get("PRAGMALENS_OLLAMA_URL", "http://localhost:11434"),
                "langextract_api_key": os.environ.get("LANGEXTRACT_API_KEY"),
                "gemini_api_key": os.environ.get("GEMINI_API_KEY"),
                "verifier_backend": os.environ.get("PRAGMALENS_VERIFIER", "offline"),
                "minicheck_model": os.environ.get("PRAGMALENS_MINICHECK_MODEL", "roberta-large"),
                "minicheck_cache_dir": Path(
                    os.environ.get(
                        "PRAGMALENS_MINICHECK_CACHE_DIR",
                        ".local_state/minicheck-cache",
                    )
                ),
                "crossencoder_model": os.environ.get(
                    "PRAGMALENS_CROSSENCODER_MODEL",
                    "cross-encoder/nli-deberta-v3-base",
                ),
                "crossencoder_cache_dir": Path(
                    os.environ.get(
                        "PRAGMALENS_CROSSENCODER_CACHE_DIR",
                        ".local_state/crossencoder-cache",
                    )
                ),
                "crossencoder_revision": os.environ.get(
                    "PRAGMALENS_CROSSENCODER_REVISION",
                    "f2f24f9fce8fc5b34aedf861f5c819c6ba0cf4f5",
                ),
            }
            return cls.model_validate(payload)
        except ValidationError as exc:
            raise ValueError("invalid PragmaLens runtime settings") from exc


@lru_cache(maxsize=1)
def load_settings() -> PragmaLensSettings:
    """Return the cached runtime settings snapshot for this process."""
    return PragmaLensSettings.from_env()
