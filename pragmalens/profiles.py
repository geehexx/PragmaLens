from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Profile:
    name: str
    langextract_fixture: str
    gliner2_fixture: str
    label_map: dict[str, str]


DEFAULT_PROFILE = {
    "name": "default",
    "langextract_fixture": "tests/fixtures/langextract_sample.jsonl",
    "gliner2_fixture": "tests/fixtures/gliner2_sample.json",
    "label_map": {
        "agent": "commitment_owner",
        "action": "commitment_action",
        "condition": "condition",
    },
}


def load_profile(name: str = "default") -> Profile:
    profiles_dir = Path("pragmalens/fixtures")
    path = profiles_dir / f"{name}.profile.json"
    payload: dict[str, Any] = (
        json.loads(path.read_text(encoding="utf-8")) if path.exists() else DEFAULT_PROFILE
    )

    return Profile(
        name=payload.get("name", name),
        langextract_fixture=payload["langextract_fixture"],
        gliner2_fixture=payload["gliner2_fixture"],
        label_map=payload.get("label_map", {}),
    )
