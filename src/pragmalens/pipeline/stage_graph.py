"""Canonical v0.1 stage graph assembly and validation helpers."""

from __future__ import annotations

from pragmalens.pipeline.extraction import (
    EvidenceNormalizerStage,
    GLiNER2CapturedStage,
    LangExtractCapturedStage,
)
from pragmalens.pipeline.reporting import RenderReportsAndTracesStage
from pragmalens.pipeline.runtime import PipelineStage
from pragmalens.pipeline.text import (
    NormalizeDocumentStage,
    SegmentAndIndexSpansStage,
    SpacySubstrateStage,
)
from pragmalens.pipeline.verification import SynthesizeFindingsStage, VerifyClaimsStage
from pragmalens.verifier import VerifierAdapter

REQUIRED_STAGE_IDS = [
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


def default_v01_stages(*, verifier: VerifierAdapter | None = None) -> list[PipelineStage]:
    """Return the current full v0.1 stage graph."""
    return [
        NormalizeDocumentStage(),
        SegmentAndIndexSpansStage(),
        SpacySubstrateStage(),
        LangExtractCapturedStage(),
        GLiNER2CapturedStage(),
        EvidenceNormalizerStage(),
        VerifyClaimsStage(verifier=verifier),
        SynthesizeFindingsStage(),
        RenderReportsAndTracesStage(),
    ]


def validate_stage_graph(stages: list[PipelineStage]) -> None:
    """Validate that a stage list matches the current full v0.1 graph."""
    actual = [stage.id for stage in stages]
    if actual != REQUIRED_STAGE_IDS:
        raise ValueError(f"Invalid stage graph. expected={REQUIRED_STAGE_IDS} actual={actual}")
