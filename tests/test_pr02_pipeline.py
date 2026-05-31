import json

from pragmalens.core import run_pr02_pipeline
from pragmalens.stages import default_pr02_stages, is_valid_span, validate_pr02_graph


def test_pr02_graph_validates() -> None:
    stages = default_pr02_stages()
    validate_pr02_graph(stages)


def test_pr02_pipeline_outputs_spacy_artifacts_and_valid_cues(tmp_path) -> None:
    text = "If we must ship now, we should not delay [1]."
    report_out = tmp_path / "out" / "report.json"
    trace_dir = tmp_path / "trace"

    report, manifest = run_pr02_pipeline(
        text=text,
        document_id="doc-1",
        input_path="/tmp/doc.md",
        report_path=str(report_out),
        trace_dir=trace_dir,
    )

    assert report.findings == []
    assert manifest.stage_health["spacy_substrate"] == "ok"

    spacy_artifact = json.loads((trace_dir / "spacy_substrate.json").read_text(encoding="utf-8"))
    assert spacy_artifact["sentences"]
    assert spacy_artifact["tokens"]
    assert spacy_artifact["cues"]

    for cue in spacy_artifact["cues"]:
        assert is_valid_span(text, cue["start_char"], cue["end_char"])


def test_spacy_stage_emits_no_findings_only_artifacts(tmp_path) -> None:
    report, _ = run_pr02_pipeline(
        text="We may proceed when validated.",
        document_id="doc-2",
        input_path="/tmp/doc2.md",
        report_path=str(tmp_path / "r.json"),
        trace_dir=tmp_path / "trace",
    )
    assert report.findings == []
