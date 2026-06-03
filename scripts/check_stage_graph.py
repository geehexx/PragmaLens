#!/usr/bin/env python3
"""Verify the runtime stage graph matches the declared v0.1 sequence."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

REQUIRED = [
    "normalize_document",
    "segment_and_index_spans",
    "spacy_substrate",
    "langextract_discourse",
    "gliner2_candidates",
    "evidence_normalizer",
    "verify_claims",
    "synthesize_findings",
    "render_reports_and_traces",
]


def fail(msg: str) -> None:
    """Exit with a failing status and a human-readable error message."""
    print(f"FAIL: {msg}")
    raise SystemExit(1)


def main() -> None:
    """Compare the runtime stage graph with the required canonical order."""
    from pragmalens.pipeline.stage_graph import default_v01_stages

    actual = [stage.id for stage in default_v01_stages()]
    if actual != REQUIRED:
        fail(f"runtime v0.1 stage graph mismatch. expected={REQUIRED} actual={actual}")

    print("PASS: runtime v0.1 graph equals the required target graph")
    for stage_id in REQUIRED:
        print(f" - {stage_id}")


if __name__ == "__main__":
    main()
