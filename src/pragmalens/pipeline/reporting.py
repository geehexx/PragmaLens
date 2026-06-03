"""Pipeline reporting helpers for manifests and JSON trace emission."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pragmalens.models import NeutralReport, RunManifest
from pragmalens.pipeline.runtime import RunContext, StageResult


class RenderReportsAndTracesStage:
    """Record that final rendering is handled by the CLI orchestration layer."""

    id = "render_reports_and_traces"
    version = "0.1"
    input_contract = "findings + candidates + stage_results"
    output_contract = "report_json + report_md + run_manifest + trace"

    def run(self, context: RunContext) -> StageResult:
        """Store renderer metadata for downstream trace emission."""
        context.artifacts[self.id] = {"renderer": "cli_orchestrated"}
        return StageResult(stage_id=self.id, status="ok")


def build_report_and_manifest(
    context: RunContext,
    input_path: str,
    report_path: str,
    trace_dir: Path,
    stage_results: list[StageResult],
) -> tuple[NeutralReport, RunManifest]:
    """Build the final report and manifest from completed stage state."""
    report = NeutralReport(
        run_id=f"run-{context.document_id}",
        document_id=context.document_id,
        findings=context.metadata.get("findings", []),
        candidates=context.candidates,
        verification=context.metadata.get("verification", []),
        warnings=context.warnings,
    )
    manifest = RunManifest(
        run_id=f"run-{context.document_id}",
        document_id=context.document_id,
        input_path=input_path,
        report_path=report_path,
        profile_name=context.profile_name,
        prompt_hash=context.metadata.get("langextract", {}).get("prompt_hash"),
        model_registry={
            "langextract": context.metadata.get("langextract", {}).get("model", "captured"),
            "gliner2": "captured",
        },
        stages_requested=[result.stage_id for result in stage_results],
        stage_health={result.stage_id: result.status for result in stage_results},
        artifacts={
            "report_json": report_path,
            "run_manifest_json": str(Path(report_path).with_name("run_manifest.json")),
            "trace_dir": str(trace_dir),
            "trace_manifest_json": str(trace_dir / "trace_manifest.json"),
        },
    )
    return report, manifest


def write_traces(trace_dir: Path, context: RunContext, stage_results: list[StageResult]) -> None:
    """Write JSON trace artifacts for each major pipeline surface."""
    trace_dir.mkdir(parents=True, exist_ok=True)

    def dump(name: str, payload: Any) -> None:
        """Serialize one trace payload into the trace directory."""
        (trace_dir / name).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    dump("normalized_document.json", context.artifacts.get("normalize_document", {}))
    dump("span_index.json", context.artifacts.get("segment_and_index_spans", {}))
    dump("spacy_substrate.json", context.artifacts.get("spacy_substrate", {}))
    dump("langextract_candidates.json", context.artifacts.get("langextract_discourse", {}))
    dump("gliner2_candidates.json", context.artifacts.get("gliner2_candidates", {}))
    dump("evidence_normalization.json", context.artifacts.get("evidence_normalizer", {}))
    dump("verification.json", context.artifacts.get("verify_claims", {}))
    dump("findings.json", context.artifacts.get("synthesize_findings", {}))
    dump("report_rendering.json", context.artifacts.get("render_reports_and_traces", {}))
    dump(
        "quarantined_candidates.json",
        [candidate.model_dump(mode="json") for candidate in context.quarantined_candidates],
    )
    dump("stage_results.json", [result.__dict__ for result in stage_results])
    dump(
        "trace_manifest.json",
        {
            "run_id": f"run-{context.document_id}",
            "document_id": context.document_id,
            "trace_dir": str(trace_dir),
            "files": sorted(path.name for path in trace_dir.iterdir() if path.is_file()),
        },
    )
