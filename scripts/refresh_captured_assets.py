#!/usr/bin/env python3
"""Refresh captured fixtures and benchmark slices from real upstream sources."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path

from pragmalens.evaluation import (
    load_corpus_approval_metadata,
    load_golden_set_batch,
    run_corpus_benchmark_files,
)
from pragmalens.extraction import resolve_data_path
from pragmalens.live_runtime import langextract_provider_config, load_gliner2_model
from pragmalens.pipeline.captured_extraction import run_captured_extraction_pipeline
from pragmalens.profiles import load_profile

ROOT = Path(__file__).resolve().parents[1]
CAPTURED_DIR = ROOT / "src" / "pragmalens" / "fixtures" / "captured"
BENCHMARK_BATCH = ROOT / "tests" / "fixtures" / "corpus_benchmark_ragtruth.json"
BENCHMARK_APPROVAL = ROOT / "tests" / "fixtures" / "corpus_approval_ragtruth.json"
GOLDEN_SET_DIR = ROOT / "tests" / "fixtures" / "golden"
GOLDEN_SET_BATCH = GOLDEN_SET_DIR / "pragmalens_golden_set.json"
GOLDEN_SET_APPROVAL = GOLDEN_SET_DIR / "pragmalens_golden_set.approval.json"
RAGTRUTH_BASE = "https://raw.githubusercontent.com/ParticleMedia/RAGTruth/main/dataset"
RAGTRUTH_SOURCE_INFO_URL = f"{RAGTRUTH_BASE}/source_info.jsonl"
RAGTRUTH_RESPONSE_URL = f"{RAGTRUTH_BASE}/response.jsonl"
LIVE_CAPTURE_NAME = "pragmalens-live-capture:v1"
LIVE_CAPTURE_INPUT = CAPTURED_DIR / "input.md"
LIVE_PROMPT_DESCRIPTION = "Extract the actor and the commitment phrase."
LIVE_PROMPT_HASH = "sha256:e38173b7d773c7f5c7dd3fc2a23ab7f541691449cbed44aefc7bcc39451dff48"
LIVE_EXAMPLE_TEXT = "Alice promised to ship the report tomorrow."
BENCHMARK_CORPUS_ID = "ragtruth-15592-mini"
BENCHMARK_DOCUMENT_ID = "ragtruth-15592"
BENCHMARK_SOURCE_ID = "15592"
GOLDEN_SET_ID = "pragmalens-golden-2026-06"
GOLDEN_SET_APPROVAL_SOURCE = "urn:pragmalens:golden-set"

GOLDEN_SET_BATCH_PAYLOAD: dict[str, object] = {
    "golden_set_id": GOLDEN_SET_ID,
    "cases": [
        {
            "case_id": "recommendation-not-promise",
            "document_id": "golden-001",
            "document_text": "The team should ship the verifier if evidence is present.",
            "annotations": [
                {
                    "label": "promise",
                    "span": {
                        "document_id": "golden-001",
                        "start_char": 9,
                        "end_char": 56,
                        "text": "should ship the verifier if evidence is present",
                    },
                    "resolved_to": "recommendation",
                    "notes": ["Modal 'should' marks a recommendation, not a promise."],
                },
                {
                    "label": "review_commitment",
                    "span": {
                        "document_id": "golden-001",
                        "start_char": 16,
                        "end_char": 33,
                        "text": "ship the verifier",
                    },
                    "resolved_to": "review_commitment",
                    "notes": ["Keep the review action separate from the modal recommendation."],
                },
                {
                    "label": "team_membership",
                    "span": {
                        "document_id": "golden-001",
                        "start_char": 0,
                        "end_char": 8,
                        "text": "The team",
                    },
                    "resolved_to": "ambiguous",
                    "notes": ["Do not infer any individual membership from sentence adjacency."],
                },
            ],
        },
        {
            "case_id": "promise-review-ambiguity",
            "document_id": "golden-002",
            "document_text": (
                "Alice promised to ship the report tomorrow. The team will review the draft "
                "before Friday."
            ),
            "annotations": [
                {
                    "label": "promise",
                    "span": {
                        "document_id": "golden-002",
                        "start_char": 6,
                        "end_char": 42,
                        "text": "promised to ship the report tomorrow",
                    },
                    "resolved_to": "promise",
                    "notes": ["This is an explicit promise, not a recommendation."],
                },
                {
                    "label": "review_commitment",
                    "span": {
                        "document_id": "golden-002",
                        "start_char": 53,
                        "end_char": 88,
                        "text": "will review the draft before Friday",
                    },
                    "resolved_to": "commitment",
                    "notes": ["The review action is a separate commitment from the promise."],
                },
                {
                    "label": "antecedent_resolution",
                    "span": {
                        "document_id": "golden-002",
                        "start_char": 0,
                        "end_char": 5,
                        "text": "Alice",
                    },
                    "resolved_to": "Alice",
                    "notes": ["The antecedent for the promise is explicit."],
                },
                {
                    "label": "team_membership",
                    "span": {
                        "document_id": "golden-002",
                        "start_char": 44,
                        "end_char": 52,
                        "text": "The team",
                    },
                    "resolved_to": "ambiguous",
                    "notes": [
                        "Do not assume Alice is part of the team from the previous sentence."
                    ],
                },
            ],
        },
    ],
}

GOLDEN_SET_APPROVAL_PAYLOAD: dict[str, object] = {
    "corpus_id": GOLDEN_SET_ID,
    "corpus_name": "PragmaLens Golden Set",
    "corpus_version": "2026-06",
    "source_uri": GOLDEN_SET_APPROVAL_SOURCE,
    "approval_status": "approved",
    "approval_evidence": [
        "internal semantic regression cases derived from current promise/review ambiguity gaps",
        "manual review confirmed each annotation is explicit and reviewable",
    ],
}


def _fail(message: str) -> None:
    print(f"FAIL: {message}")
    raise SystemExit(1)


def _pass(message: str) -> None:
    print(f"PASS: {message}")


def _read_jsonl(url: str) -> list[dict[str, object]]:
    with urllib.request.urlopen(url) as response:
        rows = [json.loads(line) for line in response.read().decode("utf-8").splitlines()]
    return rows


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _promote_files(paths: list[Path], dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for path in paths:
        shutil.copy2(path, dest / path.name)


def refresh_live_fixtures() -> None:
    """Regenerate the package-captured live extraction fixtures from live runtimes."""
    if os.environ.get("PRAGMALENS_ENABLE_LIVE_SMOKE") != "1":
        _fail("set PRAGMALENS_ENABLE_LIVE_SMOKE=1 before refreshing live capture fixtures")

    try:
        langextract = importlib.import_module("langextract")
    except ImportError as exc:
        raise RuntimeError(
            "Install live dependencies with `uv sync --group live` before refreshing captures."
        ) from exc

    text = resolve_data_path(LIVE_CAPTURE_INPUT).read_text(encoding="utf-8")
    config = langextract_provider_config()
    examples = [
        langextract.data.ExampleData(
            text=LIVE_EXAMPLE_TEXT,
            extractions=[
                langextract.data.Extraction(extraction_class="actor", extraction_text="Alice"),
                langextract.data.Extraction(
                    extraction_class="commitment",
                    extraction_text="promised to ship the report tomorrow",
                ),
            ],
        )
    ]
    kwargs: dict[str, str] = {"model_id": config["model_id"]}
    if config["provider"] == "ollama":
        kwargs["model_url"] = config["model_url"]

    extractor_result = langextract.extract(
        text_or_documents=text,
        prompt_description=LIVE_PROMPT_DESCRIPTION,
        examples=examples,
        **kwargs,
    )
    gliner2_model = load_gliner2_model()
    gliner2_payload = gliner2_model.extract_entities(
        text,
        ["agent", "action", "condition"],
        include_confidence=True,
        include_spans=True,
    )
    if not isinstance(gliner2_payload, dict):
        raise RuntimeError("GLiNER2 live output must be a JSON object")

    with tempfile.TemporaryDirectory(prefix="pragmalens-live-capture-") as tmp:
        staging = Path(tmp)
        langextract_path = staging / "langextract.jsonl"
        gliner2_path = staging / "gliner2.json"
        manifest_path = staging / "capture_manifest.json"

        langextract_records = [
            {
                "_meta": {
                    "provider": config["provider"],
                    "model": config["model_id"],
                    "prompt_hash": LIVE_PROMPT_HASH,
                    "example_set": LIVE_CAPTURE_NAME,
                }
            }
        ]
        for extraction in extractor_result.extractions:
            interval = extraction.char_interval
            langextract_records.append(
                {
                    "char_interval": list(interval) if interval is not None else None,
                    "extraction_text": extraction.extraction_text,
                    "label": extraction.extraction_class,
                    "kind": extraction.extraction_class,
                }
            )
        langextract_path.write_text(
            "\n".join(json.dumps(record) for record in langextract_records) + "\n",
            encoding="utf-8",
        )
        _write_json(gliner2_path, gliner2_payload)
        _write_json(
            manifest_path,
            {
                "capture_name": LIVE_CAPTURE_NAME,
                "input_path": str(LIVE_CAPTURE_INPUT.relative_to(ROOT)),
                "document_id": "captured-input",
                "langextract": {
                    "provider": config["provider"],
                    "model": config["model_id"],
                    "prompt_description": LIVE_PROMPT_DESCRIPTION,
                    "example_set": LIVE_CAPTURE_NAME,
                    "prompt_hash": LIVE_PROMPT_HASH,
                },
                "gliner2": {
                    "model": os.environ.get("PRAGMALENS_GLINER2_MODEL", "fastino/gliner2-base-v1"),
                    "labels": ["agent", "action", "condition"],
                },
                "outputs": ["langextract.jsonl", "gliner2.json"],
            },
        )

        result = run_captured_extraction_pipeline(
            text=text,
            langextract_jsonl=langextract_path,
            gliner2_json=gliner2_path,
            trace_dir=staging / "trace",
            document_id="captured-input",
            label_map=load_profile().label_map,
        )
        if result["quarantined_candidates"]:
            raise RuntimeError(
                "captured live fixtures produced quarantined candidates; refuse to promote"
            )

        _promote_files([langextract_path, gliner2_path, manifest_path], CAPTURED_DIR)
    _pass("refreshed live capture fixtures")


def refresh_benchmark_fixtures() -> None:
    """Regenerate the approved RAGTruth benchmark slice from upstream dataset rows."""
    source_rows = _read_jsonl(RAGTRUTH_SOURCE_INFO_URL)
    response_rows = _read_jsonl(RAGTRUTH_RESPONSE_URL)
    source_row = next(
        (row for row in source_rows if row.get("source_id") == BENCHMARK_SOURCE_ID),
        None,
    )
    if source_row is None:
        _fail(f"missing upstream source_id {BENCHMARK_SOURCE_ID}")
    response0 = next(
        (
            row
            for row in response_rows
            if row.get("id") == "0" and row.get("source_id") == BENCHMARK_SOURCE_ID
        ),
        None,
    )
    response2 = next(
        (
            row
            for row in response_rows
            if row.get("id") == "2" and row.get("source_id") == BENCHMARK_SOURCE_ID
        ),
        None,
    )
    if response0 is None or response2 is None:
        _fail("missing upstream benchmark responses 0 or 2 for source_id 15592")
    if response0.get("labels") != []:
        _fail("upstream response 0 is no longer the clean supported example")
    if not any(
        label.get("text") == "February 7, 2022."
        for label in response2.get("labels", [])
        if isinstance(label, dict)
    ):
        _fail("upstream response 2 no longer exposes the expected hallucinated date")
    source_text = str(source_row["source_info"])
    if "probably did not survive to March 1945" not in source_text:
        _fail("upstream source text no longer contains the supported claim anchor")
    if "symptoms before February 7" not in source_text:
        _fail("upstream source text no longer contains the unsupported claim anchor")

    batch_payload = {
        "corpus_id": BENCHMARK_CORPUS_ID,
        "cases": [
            {
                "case_id": "ragtruth-15592-r0-supported",
                "document_id": BENCHMARK_DOCUMENT_ID,
                "document_text": source_text,
                "claim_text": "Anne and Margot Frank probably did not survive to March 1945.",
                "gold_status": "supported",
            },
            {
                "case_id": "ragtruth-15592-r2-unsupported",
                "document_id": BENCHMARK_DOCUMENT_ID,
                "document_text": source_text,
                "claim_text": (
                    "Anne and Margot Frank are believed to have died before February 7, 2022."
                ),
                "gold_status": "unsupported",
            },
        ],
    }
    approval_payload = {
        "corpus_id": BENCHMARK_CORPUS_ID,
        "corpus_name": "RAGTruth",
        "corpus_version": "2024-02",
        "source_uri": RAGTRUTH_SOURCE_INFO_URL,
        "approval_status": "approved",
        "approval_evidence": [
            "upstream dataset source_id 15592 from RAGTruth source_info.jsonl",
            (
                "upstream response id 0 is supported and response id 2 contains the explicit "
                "2022 hallucination"
            ),
        ],
    }

    with tempfile.TemporaryDirectory(prefix="pragmalens-benchmark-") as tmp:
        staging = Path(tmp)
        batch_path = staging / BENCHMARK_BATCH.name
        approval_path = staging / BENCHMARK_APPROVAL.name
        _write_json(batch_path, batch_payload)
        _write_json(approval_path, approval_payload)

        report, metadata, recommendation = run_corpus_benchmark_files(
            corpus_batch_path=batch_path,
            approval_metadata_path=approval_path,
            selected_backend="offline",
            run_id=f"{BENCHMARK_CORPUS_ID}-offline",
        )
        if metadata.case_count != 2:
            raise RuntimeError("benchmark validation did not load the expected two cases")
        if report.run_id != f"{BENCHMARK_CORPUS_ID}-offline":
            raise RuntimeError("benchmark validation run id mismatch")
        del recommendation

        _promote_files([batch_path, approval_path], ROOT / "tests" / "fixtures")
    _pass("refreshed benchmark fixtures")


def refresh_golden_set() -> None:
    """Regenerate the internal semantic golden set fixture and approval sidecar."""
    with tempfile.TemporaryDirectory(prefix="pragmalens-golden-set-") as tmp:
        staging = Path(tmp)
        batch_path = staging / GOLDEN_SET_BATCH.name
        approval_path = staging / GOLDEN_SET_APPROVAL.name
        _write_json(batch_path, GOLDEN_SET_BATCH_PAYLOAD)
        _write_json(approval_path, GOLDEN_SET_APPROVAL_PAYLOAD)

        batch = load_golden_set_batch(batch_path)
        approval = load_corpus_approval_metadata(approval_path)
        if batch.golden_set_id != approval.corpus_id:
            raise RuntimeError("golden set approval corpus id mismatch")
        if approval.source_uri != GOLDEN_SET_APPROVAL_SOURCE:
            raise RuntimeError("golden set approval source uri mismatch")
        if approval.approval_status != "approved":
            raise RuntimeError("golden set approval must remain approved")

        _promote_files([batch_path, approval_path], GOLDEN_SET_DIR)
    _pass("refreshed golden set fixtures")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the refresh modes."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    subparsers.add_parser("live-fixtures", help="Refresh captured live extraction fixtures")
    subparsers.add_parser("benchmark-fixtures", help="Refresh approved benchmark fixture slices")
    subparsers.add_parser("golden-set", help="Refresh the internal semantic golden set")
    return parser


def main() -> None:
    """Dispatch to the requested refresh mode."""
    args = build_parser().parse_args()
    if args.mode == "live-fixtures":
        refresh_live_fixtures()
        return
    if args.mode == "benchmark-fixtures":
        refresh_benchmark_fixtures()
        return
    if args.mode == "golden-set":
        refresh_golden_set()
        return
    raise AssertionError(f"unhandled mode: {args.mode}")


if __name__ == "__main__":
    main()
