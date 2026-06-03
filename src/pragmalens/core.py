"""Top-level pipeline orchestration helpers used by the CLI and tests."""

from __future__ import annotations

from pathlib import Path

from pragmalens.models import NeutralReport, RunManifest
from pragmalens.pipeline.reporting import build_report_and_manifest, write_traces
from pragmalens.pipeline.runtime import PipelineRunner, RunContext
from pragmalens.pipeline.stage_graph import default_v01_stages, validate_stage_graph
from pragmalens.profiles import load_profile
from pragmalens.verifier import VerifierAdapter, build_verifier_adapter, build_verifier_from_env


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
) -> tuple[NeutralReport, RunManifest]:
    """Execute the default v0.1 pipeline and emit traces plus contracts."""
    profile = load_profile(profile_name)
    context = RunContext(
        document_id=document_id,
        input_path=input_path,
        text=text,
        profile_name=profile_name,
        profile=profile,
    )
    selected_verifier = verifier
    if selected_verifier is None:
        selected_verifier = (
            build_verifier_adapter(verifier_backend)
            if verifier_backend is not None
            else build_verifier_from_env()
        )
    stages = default_v01_stages(verifier=selected_verifier)
    validate_stage_graph(stages)
    runner = PipelineRunner(stages)
    results = runner.run(context)
    write_traces(trace_dir=trace_dir, context=context, stage_results=results)
    return build_report_and_manifest(
        context,
        input_path=input_path,
        report_path=report_path,
        trace_dir=trace_dir,
        stage_results=results,
    )
