from __future__ import annotations

from collections.abc import Iterator

import pytest

from pragmalens.settings import load_settings


@pytest.fixture(autouse=True)
def _clear_runtime_settings_cache() -> Iterator[None]:
    """Keep runtime settings reads isolated across tests."""
    load_settings.cache_clear()
    yield
    load_settings.cache_clear()
