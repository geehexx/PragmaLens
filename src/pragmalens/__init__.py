"""PragmaLens product package."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

__all__ = [
    "PragmaLensSettings",
    "load_settings",
    "models",
]

if TYPE_CHECKING:
    from pragmalens.settings import PragmaLensSettings, load_settings


def __getattr__(name: str) -> Any:
    """Lazily expose package-level conveniences without eager dependencies."""
    if name == "models":
        module = import_module("pragmalens.models")
        globals()[name] = module
        return module
    if name in {"PragmaLensSettings", "load_settings"}:
        module = import_module("pragmalens.settings")
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
