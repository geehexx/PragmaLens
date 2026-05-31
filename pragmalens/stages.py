from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import spacy

from pragmalens.extraction import load_json, load_jsonl, merge_and_dedupe_candidates, normalize_gliner2_output, normalize_langextract_records
from pragmalens.models import EvidenceCandidate, NeutralReport, RunManifest
from pragmalens.profiles import Profile


@dataclass
class RunContext:
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
    stage_id: str
    status: str
    warnings: list[str] = field(default_factory=list)


class PipelineStage(Protocol):
    id: str
    version: str
    input_contract: str
    output_contract: str

    def run(self, context: RunContext) -> StageResult: ...


class NormalizeDocumentStage:
    id = "normalize_document"
    version = "0.1"
    input_contract = "raw_text"
    output_contract = "normalized_text"

    def run(self, context: RunContext) -> StageResult:
        normalized = context.text.replace("\r\n", "\n").replace("\r", "\n")
        context.text = normalized
        context.artifacts[self.id] = {"document_id": context.document_id, "length": len(normalized)}
        return StageResult(stage_id=self.id, status="ok")


class SegmentAndIndexSpansStage:
    id = "segment_and_index_spans"
    version = "0.1"
    input_contract = "normalized_text"
    output_contract = "sentence_and_paragraph_spans"

    def __init__(self) -> None:
        nlp = spacy.blank("en")
        nlp.add_pipe("sentencizer")
        self._nlp = nlp

    def run(self, context: RunContext) -> StageResult:
        doc = self._nlp(context.text)
        sentences = [{"start_char": s.start_char, "end_char": s.end_char, "text": s.text} for s in doc.sents]

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
    id = "spacy_substrate"
    version = "0.1"
    input_contract = "normalized_text + span_index"
    output_contract = "tokens + cues"

    MODAL = {"must", "shall", "should", "may", "might", "will"}
    CONDITION = {"if", "unless", "when", "provided"}
    NEGATION = {"not", "no", "never"}

    def __init__(self) -> None:
        nlp = spacy.blank("en")
        nlp.add_pipe("sentencizer")
        self._nlp = nlp

    def run(self, context: RunContext) -> StageResult:
        doc = self._nlp(context.text)
        tokens: list[dict[str, Any]] = []
        cues: list[dict[str, Any]] = []

        for t in doc:
            tokens.append({"text": t.text, "lower": t.lower_, "start_char": t.idx, "end_char": t.idx + len(t.text)})
            lower = t.lower_
            if lower in self.MODAL:
                cues.append(_cue("modal", t.idx, t.idx + len(t.text), t.text))
            if lower in self.CONDITION:
                cues.append(_cue("condition", t.idx, t.idx + len(t.text), t.text))
            if lower in self.NEGATION:
                cues.append(_cue("negation", t.idx, t.idx + len(t.text), t.text))

        for m in re.finditer(r"\[[0-9]+\]|\([0-9]+\)", context.text):
            cues.append(_cue("citation_like", m.start(), m.end(), context.text[m.start():m.end()]))

        context.artifacts[self.id] = {
            "sentences": [{"start_char": s.start_char, "end_char": s.end_char, "text": s.text} for s in doc.sents],
            "tokens": tokens,
            "cues": cues,
        }
        return StageResult(stage_id=self.id, status="ok")


class LangExtractCapturedStage:
    id = "langextract_discourse"
    version = "0.1"
    input_contract = "normalized_text + profile"
    output_contract = "langextract_candidates + quarantined"

    def run(self, context: RunContext) -> StageResult:
        assert context.profile is not None
        records = load_jsonl(context.profile.langextract_fixture)
        valid, quarantined, meta = normalize_langextract_records(records, context.text, document_id=context.document_id)
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
    id = "gliner2_candidates"
    version = "0.1"
    input_contract = "normalized_text + profile"
    output_contract = "gliner2_candidates"

    def run(self, context: RunContext) -> StageResult:
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
    id = "evidence_normalizer"
    version = "0.1"
    input_contract = "raw_candidates"
    output_contract = "merged_candidates"

    def run(self, context: RunContext) -> StageResult:
        merged = merge_and_dedupe_candidates(context.candidates)
        context.candidates = merged
        context.artifacts[self.id] = {"candidates": [c.model_dump(mode="json") for c in merged]}
        return StageResult(stage_id=self.id, status="ok")


class PipelineRunner:
    def __init__(self, stages: list[PipelineStage]) -> None:
        self.stages = stages

    def run(self, context: RunContext) -> list[StageResult]:
        return [stage.run(context) for stage in self.stages]


def _cue(kind: str, start: int, end: int, text: str) -> dict[str, Any]:
    return {"kind": kind, "start_char": start, "end_char": end, "text": text}


def default_pr04_stages() -> list[PipelineStage]:
    return [
        NormalizeDocumentStage(),
        SegmentAndIndexSpansStage(),
        SpacySubstrateStage(),
        LangExtractCapturedStage(),
        GLiNER2CapturedStage(),
        EvidenceNormalizerStage(),
    ]


def default_pr02_stages() -> list[PipelineStage]:
    return [
        NormalizeDocumentStage(),
        SegmentAndIndexSpansStage(),
        SpacySubstrateStage(),
    ]


def validate_pr02_graph(stages: list[PipelineStage]) -> None:
    expected = ["normalize_document", "segment_and_index_spans", "spacy_substrate"]
    actual = [s.id for s in stages]
    if actual != expected:
        raise ValueError(f"Invalid stage graph. expected={expected} actual={actual}")


def validate_stage_graph(stages: list[PipelineStage]) -> None:
    expected = [
        "normalize_document",
        "segment_and_index_spans",
        "spacy_substrate",
        "langextract_discourse",
        "gliner2_candidates",
        "evidence_normalizer",
    ]
    actual = [s.id for s in stages]
    if actual != expected:
        raise ValueError(f"Invalid stage graph. expected={expected} actual={actual}")


def build_report_and_manifest(context: RunContext, input_path: str, report_path: str, stage_results: list[StageResult]) -> tuple[NeutralReport, RunManifest]:
    report = NeutralReport(document_id=context.document_id, findings=[], candidates=context.candidates, warnings=context.warnings)
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
    )
    return report, manifest


def write_traces(trace_dir: Path, context: RunContext, stage_results: list[StageResult]) -> None:
    trace_dir.mkdir(parents=True, exist_ok=True)
    def dump(name: str, payload: Any) -> None:
        (trace_dir / name).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    dump("normalized_document.json", context.artifacts.get("normalize_document", {}))
    dump("span_index.json", context.artifacts.get("segment_and_index_spans", {}))
    dump("spacy_substrate.json", context.artifacts.get("spacy_substrate", {}))
    dump("langextract_candidates.json", context.artifacts.get("langextract_discourse", {}))
    dump("gliner2_candidates.json", context.artifacts.get("gliner2_candidates", {}))
    dump("evidence_normalization.json", context.artifacts.get("evidence_normalizer", {}))
    dump("quarantined_candidates.json", [c.model_dump(mode="json") for c in context.quarantined_candidates])
    dump("stage_results.json", [r.__dict__ for r in stage_results])


def is_valid_span(text: str, start: int, end: int) -> bool:
    return not (start < 0 or end <= start or end > len(text))
