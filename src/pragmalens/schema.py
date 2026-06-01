from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from pragmalens.models import NeutralReport, RunManifest, SpanRef, VerificationVerdict
from pragmalens.profiles import ProfileModel

MODEL_MAP: dict[str, type[BaseModel]] = {
    "report": NeutralReport,
    "manifest": RunManifest,
    "profile": ProfileModel,
    "span_ref": SpanRef,
    "verification_verdict": VerificationVerdict,
}


def export_schema(model_name: str, out_path: str) -> Path:
    if model_name not in MODEL_MAP:
        raise ValueError(f"Unknown model '{model_name}'. Expected one of: {', '.join(MODEL_MAP)}")

    model_cls = MODEL_MAP[model_name]
    schema = model_cls.model_json_schema()
    dest = Path(out_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return dest
