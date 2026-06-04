from __future__ import annotations

import builtins
import importlib
import sys
from typing import Any, cast

import pytest

from pragmalens import PragmaLensSettings, load_settings, models


def test_package_root_exports_settings_api() -> None:
    assert PragmaLensSettings.__name__ == "PragmaLensSettings"
    assert load_settings.__name__ == "load_settings"
    assert hasattr(models, "EvidenceCandidate")


def test_package_root_import_does_not_require_pydantic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def _guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "pydantic" or name.startswith("pydantic."):
            raise AssertionError("package root import should not require pydantic")
        return cast(Any, real_import)(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "pragmalens", raising=False)
    monkeypatch.delitem(sys.modules, "pragmalens.models", raising=False)
    monkeypatch.delitem(sys.modules, "pragmalens.settings", raising=False)
    monkeypatch.setattr(builtins, "__import__", _guarded_import)

    module = importlib.import_module("pragmalens")

    assert module.__name__ == "pragmalens"
