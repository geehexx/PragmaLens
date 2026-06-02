from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Protocol

import spacy

from pragmalens.extraction import (
    load_json,
    load_jsonl,
    merge_and_dedupe_candidates,
    normalize_gliner2_output,
    normalize_langextract_records,
)
from pragmalens.models import (
    EvidenceCandidate,
    Finding,
    NeutralReport,
    RunManifest,
    VerificationStatus,
    VerificationVerdict,
)
from pragmalens.profiles import Profile


@dataclass
class RunContext:
    """Mutable state shared across pipeline stages during one run."""

    document_id: str
    input_path: str
    text: str
    profile_name: str = "default"
    profile: Profile | None = None
    artifacts: dict[str, Any] = field(default_factory=dict)
    candidates: list[EvidenceCandidate] = field(default_factory=list)
    quarantined_candidates: list[EvidenceCandidate] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StageResult:
    """Status record returned by one pipeline stage."""

    stage_id: str
    status: str
    warnings: list[str] = field(default_factory=list)


class PipelineStage(Protocol):
    """Protocol implemented by each pipeline stage."""

    id: str
    version: str
    input_contract: str
    output_contract: str

    def run(self, context: RunContext) -> StageResult: ...


class NormalizeDocumentStage:
    """Normalize line endings and record basic document metadata."""

    id = "normalize_document"
    version = "0.1"
    input_contract = "raw_text"
    output_contract = "normalized_text"

    def run(self, context: RunContext) -> StageResult:
        """Normalize document text before downstream indexing."""
        normalized = context.text.replace("\r\n", "\n").replace("\r", "\n")
        context.text = normalized
        context.artifacts[self.id] = {"document_id": context.document_id, "length": len(normalized)}
        return StageResult(stage_id=self.id, status="ok")


class SegmentAndIndexSpansStage:
    """Build sentence and paragraph span indexes from normalized text."""

    id = "segment_and_index_spans"
    version = "0.1"
    input_contract = "normalized_text"
    output_contract = "sentence_and_paragraph_spans"

    def __init__(self) -> None:
        """Initialize a lightweight sentence segmenter for indexing work."""
        nlp = spacy.blank("en")
        nlp.add_pipe("sentencizer")
        self._nlp = nlp

    def run(self, context: RunContext) -> StageResult:
        """Emit sentence and paragraph spans for later stages."""
        doc = self._nlp(context.text)
        sentences = [
            {"start_char": s.start_char, "end_char": s.end_char, "text": s.text} for s in doc.sents
        ]

        paragraphs: list[dict[str, Any]] = []
        offset = 0
        for block in context.text.split("\n\n"):
            start = context.text.find(block, offset)
            end = start + len(block)
            if block.strip():
                paragraphs.append({"start_char": start, "end_char": end, "text": block})
            offset = end

        context.artifacts[self.id] = {"sentences": sentences, "paragraphs": paragraphs}
        return StageResult(stage_id=self.id, status="ok")


class SpacySubstrateStage:
    """Emit token-level substrate artifacts and simple deterministic cues."""

    id = "spacy_substrate"
    version = "0.1"
    input_contract = "normalized_text + span_index"
    output_contract = "tokens + cues"

    MODAL: ClassVar[set[str]] = {"must", "shall", "should", "may", "might", "will"}
    CONDITION: ClassVar[set[str]] = {"if", "unless", "when", "provided"}
    NEGATION: ClassVar[set[str]] = {"not", "no", "never"}

    def __init__(self) -> None:
        """Initialize the minimal spaCy pipeline needed for substrate traces."""
        nlp = spacy.blank("en")
        nlp.add_pipe("sentencizer")
        self._nlp = nlp

    def run(self, context: RunContext) -> StageResult:
        """Produce token, sentence, and cue artifacts from the current text."""
        doc = self._nlp(context.text)
        tokens: list[dict[str, Any]] = []
        cues: list[dict[str, Any]] = []

        for t in doc:
            tokens.append(
                {
                    "text": t.text,
                    "lower": t.lower_,
                    "start_char": t.idx,
                    "end_char": t.idx + len(t.text),
                }
            )
            lower = t.lower_
            if lower in self.MODAL:
                cues.append(_cue("modal", t.idx, t.idx + len(t.text), t.text))
            if lower in self.CONDITION:
                cues.append(_cue("condition", t.idx, t.idx + len(t.text), t.text))
            if lower in self.NEGATION:
                cues.append(_cue("negation", t.idx, t.idx + len(t.text), t.text))

        for m in re.finditer(r"\[[0-9]+\]|\([0-9]+\)", context.text):
            cues.append(
                _cue("citation_like", m.start(), m.end(), context.text[m.start() : m.end()])
            )

        context.artifacts[self.id] = {
            "sentences": [
                {"start_char": s.start_char, "end_char": s.end_char, "text": s.text}
                for s in doc.sents
            ],
            "tokens": tokens,
            "cues": cues,
        }
        return StageResult(stage_id=self.id, status="ok")


class LangExtractCapturedStage:
    """Load fixture-backed LangExtract output into normalized candidates."""

    id = "langextract_discourse"
    version = "0.1"
    input_contract = "normalized_text + profile"
    output_contract = "langextract_candidates + quarantined"

    def run(self, context: RunContext) -> StageResult:
        """Normalize captured LangExtract records for offline pipeline runs."""
        assert context.profile is not None
        records = load_jsonl(context.profile.langextract_fixture)
        valid, quarantined, meta = normalize_langextract_records(
            records, context.text, document_id=context.document_id
        )
        context.metadata["langextract"] = meta
        context.candidates.extend(valid)
        context.quarantined_candidates.extend(quarantined)
        context.artifacts[self.id] = {
            "candidates": [c.model_dump(mode="json") for c in valid],
            "quarantined": [c.model_dump(mode="json") for c in quarantined],
            "meta": meta,
        }
        return StageResult(stage_id=self.id, status="ok")


class GLiNER2CapturedStage:
    """Load fixture-backed GLiNER2 output into normalized candidates."""

    id = "gliner2_candidates"
    version = "0.1"
    input_contract = "normalized_text + profile"
    output_contract = "gliner2_candidates"

    def run(self, context: RunContext) -> StageResult:
        """Normalize captured GLiNER2 entities for offline pipeline runs."""
        assert context.profile is not None
        payload = load_json(context.profile.gliner2_fixture)
        candidates = normalize_gliner2_output(
            payload,
            context.text,
            document_id=context.document_id,
            label_map=context.profile.label_map,
        )
        context.candidates.extend(candidates)
        context.artifacts[self.id] = {"candidates": [c.model_dump(mode="json") for c in candidates]}
        return StageResult(stage_id=self.id, status="ok")


class EvidenceNormalizerStage:
    """Merge duplicate candidates and surface label conflicts."""

    id = "evidence_normalizer"
    version = "0.1"
    input_contract = "raw_candidates"
    output_contract = "merged_candidates"

    def run(self, context: RunContext) -> StageResult:
        """Deduplicate and normalize evidence candidates collected so far."""
        merged = merge_and_dedupe_candidates(context.candidates)
        context.candidates = merged
        context.artifacts[self.id] = {"candidates": [c.model_dump(mode="json") for c in merged]}
        return StageResult(stage_id=self.id, status="ok")


class VerifyClaimsStage:
    """Emit the offline baseline verifier contract for each candidate."""

    id = "verify_claims"
    version = "0.1"
    input_contract = "merged_candidates"
    output_contract = "verification_verdicts"

    def run(self, context: RunContext) -> StageResult:
        """Populate baseline verification verdicts without live evidence lookup."""
        verdicts = [
            VerificationVerdict(
                candidate_id=c.candidate_id,
                status=VerificationStatus.INSUFFICIENT_EVIDENCE,
                rationale=(
                    "offline baseline verifier requires external evidence before support claims"
                ),
                evidence_ids=c.provenance,
            )
            for c in context.candidates
        ]
        context.artifacts[self.id] = {"verdicts": [v.model_dump(mode="json") for v in verdicts]}
        context.metadata["verification"] = verdicts
        return StageResult(stage_id=self.id, status="ok")


class SynthesizeFindingsStage:
    """Convert verifier outcomes into user-facing findings."""

    id = "synthesize_findings"
    version = "0.1"
    input_contract = "verification_verdicts"
    output_contract = "findings"

    def run(self, context: RunContext) -> StageResult:
        """Generate synthesized findings for actionable negative verdicts."""
        verdicts = context.metadata.get("verification", [])
        findings = [
            Finding(
                finding_id=f"finding-{idx + 1}",
                candidate_id=v.candidate_id,
                verdict=v.status,
                summary=v.rationale,
                evidence_ids=v.evidence_ids,
            )
            for idx, v in enumerate(verdicts)
            if v.status in {VerificationStatus.UNSUPPORTED, VerificationStatus.VERIFIER_ERROR}
        ]
        context.artifacts[self.id] = {"findings": [f.model_dump(mode="json") for f in findings]}
        context.metadata["findings"] = findings
        return StageResult(stage_id=self.id, status="ok")


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


class PipelineRunner:
    """Execute a linear list of stages against one run context."""

    def __init__(self, stages: list[PipelineStage]) -> None:
        """Store the stage sequence to execute."""
        self.stages = stages

    def run(self, context: RunContext) -> list[StageResult]:
        """Run every stage in order and collect per-stage results."""
        return [stage.run(context) for stage in self.stages]


def _cue(kind: str, start: int, end: int, text: str) -> dict[str, Any]:
    """Build a compact cue payload for the substrate trace."""
    return {"kind": kind, "start_char": start, "end_char": end, "text": text}


def default_pr04_stages() -> list[PipelineStage]:
    """Return the historical PR-04 stage sequence."""
    return [
        NormalizeDocumentStage(),
        SegmentAndIndexSpansStage(),
        SpacySubstrateStage(),
        LangExtractCapturedStage(),
        GLiNER2CapturedStage(),
        EvidenceNormalizerStage(),
    ]


def default_v01_stages() -> list[PipelineStage]:
    """Return the current full v0.1 stage graph."""
    return [
        NormalizeDocumentStage(),
        SegmentAndIndexSpansStage(),
        SpacySubstrateStage(),
        LangExtractCapturedStage(),
        GLiNER2CapturedStage(),
        EvidenceNormalizerStage(),
        VerifyClaimsStage(),
        SynthesizeFindingsStage(),
        RenderReportsAndTracesStage(),
    ]


def default_pr02_stages() -> list[PipelineStage]:
    """Return the historical PR-02 stage sequence."""
    return [
        NormalizeDocumentStage(),
        SegmentAndIndexSpansStage(),
        SpacySubstrateStage(),
    ]


def validate_pr02_graph(stages: list[PipelineStage]) -> None:
    """Validate that a stage list matches the historical PR-02 graph."""
    expected = ["normalize_document", "segment_and_index_spans", "spacy_substrate"]
    actual = [s.id for s in stages]
    if actual != expected:
        raise ValueError(f"Invalid stage graph. expected={expected} actual={actual}")


def validate_stage_graph(stages: list[PipelineStage]) -> None:
    """Validate that a stage list matches the current full v0.1 graph."""
    expected = [
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
    actual = [s.id for s in stages]
    if actual != expected:
        raise ValueError(f"Invalid stage graph. expected={expected} actual={actual}")


def build_report_and_manifest(
    context: RunContext, input_path: str, report_path: str, stage_results: list[StageResult]
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
        stages_requested=[r.stage_id for r in stage_results],
        stage_health={r.stage_id: r.status for r in stage_results},
        artifacts={
            "report_json": report_path,
            "run_manifest_json": str(Path(report_path).with_name("run_manifest.json")),
            "trace_dir": "trace",
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
        [c.model_dump(mode="json") for c in context.quarantined_candidates],
    )
    dump("stage_results.json", [r.__dict__ for r in stage_results])


def is_valid_span(text: str, start: int, end: int) -> bool:
    """Return whether a character interval is valid for the given text."""
    return not (start < 0 or end <= start or end > len(text))
