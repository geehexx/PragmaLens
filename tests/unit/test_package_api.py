from __future__ import annotations

from pragmalens import PragmaLensSettings, load_settings, models


def test_package_root_exports_settings_api() -> None:
    assert PragmaLensSettings.__name__ == "PragmaLensSettings"
    assert load_settings.__name__ == "load_settings"
    assert hasattr(models, "EvidenceCandidate")
