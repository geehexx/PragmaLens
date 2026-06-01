from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class ProfileModel(BaseModel):
    name: str = Field(min_length=1)
    langextract_fixture: str = Field(min_length=1)
    gliner2_fixture: str = Field(min_length=1)
    label_map: dict[str, str]


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
    profiles_dir = Path(__file__).resolve().parent / "fixtures"
    path = profiles_dir / f"{name}.profile.json"
    payload: dict[str, Any] = (
        json.loads(path.read_text(encoding="utf-8")) if path.exists() else DEFAULT_PROFILE
    )

    model = ProfileModel(
        name=payload.get("name", name),
        langextract_fixture=payload["langextract_fixture"],
        gliner2_fixture=payload["gliner2_fixture"],
        label_map=payload.get("label_map", {}),
    )
    return Profile(
        name=model.name,
        langextract_fixture=model.langextract_fixture,
        gliner2_fixture=model.gliner2_fixture,
        label_map=model.label_map,
    )
