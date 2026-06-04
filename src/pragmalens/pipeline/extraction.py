"""Pipeline stages that load captured extraction artifacts and normalize candidates."""

from __future__ import annotations

from pragmalens.extraction import (
    load_json,
    load_jsonl,
    normalize_candidate_set,
    normalize_gliner2_output_with_quarantine,
    normalize_langextract_records,
)
from pragmalens.pipeline.runtime import RunContext, StageResult


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
            "candidates": [candidate.model_dump(mode="json") for candidate in valid],
            "quarantined": [candidate.model_dump(mode="json") for candidate in quarantined],
            "meta": meta,
        }
        return StageResult(stage_id=self.id, status="ok")


class GLiNER2CapturedStage:
    """Load fixture-backed GLiNER2 output into normalized candidates."""

    id = "gliner2_candidates"
    version = "0.1"
    input_contract = "normalized_text + profile"
    output_contract = "gliner2_candidates + quarantined_candidates"

    def run(self, context: RunContext) -> StageResult:
        """Normalize captured GLiNER2 entities for offline pipeline runs."""
        assert context.profile is not None
        payload = load_json(context.profile.gliner2_fixture)
        normalized = normalize_gliner2_output_with_quarantine(
            payload,
            context.text,
            document_id=context.document_id,
            label_map=context.profile.label_map,
        )
        context.candidates.extend(normalized.valid)
        context.quarantined_candidates.extend(normalized.quarantined)
        context.artifacts[self.id] = {
            "candidates": [candidate.model_dump(mode="json") for candidate in normalized.valid],
            "quarantined": [
                candidate.model_dump(mode="json") for candidate in normalized.quarantined
            ],
        }
        return StageResult(stage_id=self.id, status="ok")


class EvidenceNormalizerStage:
    """Merge duplicate candidates and surface label conflicts."""

    id = "evidence_normalizer"
    version = "0.1"
    input_contract = "raw_candidates"
    output_contract = "merged_candidates"

    def run(self, context: RunContext) -> StageResult:
        """Deduplicate and normalize evidence candidates collected so far."""
        normalization = normalize_candidate_set(
            context.candidates, quarantined=context.quarantined_candidates
        )
        context.candidates = normalization.candidates
        context.artifacts[self.id] = {
            "candidates": [
                candidate.model_dump(mode="json") for candidate in normalization.candidates
            ],
            "quarantined": [
                candidate.model_dump(mode="json") for candidate in normalization.quarantined
            ],
            "duplicates": [
                candidate.model_dump(mode="json") for candidate in normalization.duplicates
            ],
            "duplicate_events": normalization.duplicate_events,
            "conflicts": normalization.conflicts,
            "stats": normalization.stats,
        }
        if normalization.conflicts:
            context.warnings.extend(
                [
                    f"normalization_conflict:{','.join(item['candidate_ids'])}"
                    for item in normalization.conflicts
                ]
            )
        return StageResult(stage_id=self.id, status="ok")
