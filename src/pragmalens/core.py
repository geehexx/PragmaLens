"""Top-level pipeline orchestration helpers used by the CLI and tests."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

from pragmalens.models import NeutralReport, RunManifest
from pragmalens.pipeline.reporting import (
    build_report_and_manifest,
    finalize_trace_durations,
    write_traces,
)
from pragmalens.pipeline.runtime import PipelineRunner, RunContext
from pragmalens.pipeline.stage_graph import default_v01_stages, validate_stage_graph
from pragmalens.profiles import load_profile
from pragmalens.verifier import VerifierAdapter, build_default_verifier_runtime
from pragmalens.verifier_comparison import VerifierComparisonHarness


def derive_document_id(input_path: str) -> str:
    """Derive a stable document identifier from the input path."""
    return Path(input_path).stem or "document"


def run_pipeline(
    *,
    text: str,
    document_id: str,
    input_path: str,
    report_path: str,
    trace_dir: Path,
    profile_name: str = "default",
    verifier: VerifierAdapter | None = None,
    verifier_backend: str | None = None,
    comparison_harness: VerifierComparisonHarness | None = None,
) -> tuple[NeutralReport, RunManifest]:
    """Execute the default v0.1 pipeline and emit traces plus contracts."""
    pipeline_started = perf_counter()
    profile = load_profile(profile_name)
    context = RunContext(
        document_id=document_id,
        input_path=input_path,
        text=text,
        profile_name=profile_name,
        profile=profile,
    )
    selected_verifier = verifier
    selected_comparison_harness = comparison_harness
    if selected_verifier is None:
        selected_verifier, default_comparison_harness = build_default_verifier_runtime(
            verifier_backend
        )
        if selected_comparison_harness is None:
            selected_comparison_harness = default_comparison_harness
    stages = default_v01_stages(
        verifier=selected_verifier,
        comparison_harness=selected_comparison_harness,
    )
    validate_stage_graph(stages)
    runner = PipelineRunner(stages)
    results = runner.run(context)
    write_traces(trace_dir=trace_dir, context=context, stage_results=results)
    report, manifest = build_report_and_manifest(
        context,
        input_path=input_path,
        report_path=report_path,
        trace_dir=trace_dir,
        stage_results=results,
    )
    finalize_trace_durations(trace_dir, (perf_counter() - pipeline_started) * 1000.0)
    return report, manifest
