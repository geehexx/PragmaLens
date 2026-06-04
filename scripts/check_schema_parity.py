#!/usr/bin/env python3
"""Verify exported contract schemas stay aligned with the checked-in copies."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from pragmalens.models import (
    EvidenceCandidate,
    NeutralReport,
    RunManifest,
    VerificationVerdict,
)
from pragmalens.profiles import ProfileModel

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "schemas" / "v5" / "contract_requirements.json"


def fail(msg: str) -> None:
    """Exit with a failing status and a human-readable error message."""
    print(f"FAIL: {msg}")
    raise SystemExit(1)


def main() -> None:
    """Compare runtime-exported schemas with the committed schema artifacts."""
    if not CONTRACT.exists():
        fail(f"missing v5 schema contract: {CONTRACT}")

    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))["models"]
    models = {
        "evidence_candidate": EvidenceCandidate,
        "report": NeutralReport,
        "manifest": RunManifest,
        "profile": ProfileModel,
        "verification_verdict": VerificationVerdict,
    }
    for name, model in models.items():
        fields = set(model.model_json_schema().get("properties", {}))
        required = set(contract[name]["required_fields"])
        missing_fields = sorted(required - fields)
        if missing_fields:
            fail(f"{name} schema missing required fields: {missing_fields}")

    samples = {
        "evidence_candidate": {
            "candidate_id": "candidate-1",
            "label": "claim",
            "kind": "claim",
            "status": "valid",
            "span": {
                "document_id": "doc",
                "start_char": 0,
                "end_char": 4,
                "text": "text",
            },
            "provenance": ["fixture"],
            "evidence_refs": ["fixture"],
        },
        "report": {
            "run_id": "run-doc",
            "document_id": "doc",
            "findings": [],
            "candidates": [],
            "verification": [],
            "warnings": [],
        },
        "manifest": {
            "run_id": "run-doc",
            "document_id": "doc",
            "input_path": "doc.md",
            "report_path": "report.json",
            "stages_requested": ["normalize_document"],
            "stage_health": {"normalize_document": "ok"},
            "artifacts": {"report_json": "report.json"},
        },
        "profile": {
            "name": "default",
            "langextract_fixture": "tests/fixtures/langextract_sample.jsonl",
            "gliner2_fixture": "tests/fixtures/gliner2_sample.json",
            "label_map": {},
        },
        "verification_verdict": {
            "candidate_id": "candidate-1",
            "status": "insufficient_evidence",
            "rationale": "fixture",
            "evidence_ids": [],
        },
    }
    for name, sample in samples.items():
        try:
            models[name].model_validate(sample)
        except ValidationError as exc:
            fail(f"{name} sample failed validation: {exc}")

    print("PASS: generated Pydantic schemas satisfy the v5 required-field contract")
    print("PASS: candidate/report/manifest/profile/verdict samples validate")


if __name__ == "__main__":
    main()
