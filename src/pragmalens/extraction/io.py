"""File loading helpers for captured extraction fixtures and trace inputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PACKAGE_FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"


def resolve_data_path(path: str | Path) -> Path:
    """Resolve a data path from the working tree or package fixture directory."""
    candidate = Path(path)
    if candidate.exists():
        return candidate
    packaged = PACKAGE_FIXTURES_DIR / candidate
    if packaged.exists():
        return packaged
    return candidate


def load_json(path: str | Path) -> Any:
    """Load a JSON payload from disk."""
    return json.loads(resolve_data_path(path).read_text(encoding="utf-8"))


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load newline-delimited JSON records from disk."""
    lines = resolve_data_path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]
