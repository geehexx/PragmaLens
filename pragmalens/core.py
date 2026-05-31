from __future__ import annotations

from pathlib import Path

from pragmalens.models import NeutralReport, RunManifest
from pragmalens.profiles import load_profile
from pragmalens.stages import (
    PipelineRunner,
    RunContext,
    build_report_and_manifest,
    default_pr04_stages,
    validate_stage_graph,
    write_traces,
)


def normalize_document(text: str, document_id: str) -> NeutralReport:
    return NeutralReport(document_id=document_id, findings=[], candidates=[], warnings=[])


def build_manifest(input_path: str, report_path: str, document_id: str) -> RunManifest:
    return RunManifest(run_id=f"run-{document_id}", document_id=document_id, input_path=input_path, report_path=report_path)


def derive_document_id(input_path: str) -> str:
    return Path(input_path).stem or "document"


def run_pipeline(
    *, text: str, document_id: str, input_path: str, report_path: str, trace_dir: Path, profile_name: str = "default"
) -> tuple[NeutralReport, RunManifest]:
    profile = load_profile(profile_name)
    context = RunContext(document_id=document_id, input_path=input_path, text=text, profile_name=profile_name, profile=profile)
    stages = default_pr04_stages()
    validate_stage_graph(stages)
    runner = PipelineRunner(stages)
    results = runner.run(context)
    write_traces(trace_dir=trace_dir, context=context, stage_results=results)
    return build_report_and_manifest(context, input_path=input_path, report_path=report_path, stage_results=results)


def run_pr02_pipeline(text: str, document_id: str, input_path: str, report_path: str, trace_dir: Path) -> tuple[NeutralReport, RunManifest]:
    return run_pipeline(
        text=text,
        document_id=document_id,
        input_path=input_path,
        report_path=report_path,
        trace_dir=trace_dir,
        profile_name="default",
    )
