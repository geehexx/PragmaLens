"""Pipeline stages for text normalization, segmentation, and spaCy substrate traces."""

from __future__ import annotations

import re
from typing import Any, ClassVar

import spacy

from pragmalens.pipeline.runtime import RunContext, StageResult


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
            {
                "start_char": sentence.start_char,
                "end_char": sentence.end_char,
                "text": sentence.text,
            }
            for sentence in doc.sents
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

        for token in doc:
            tokens.append(
                {
                    "text": token.text,
                    "lower": token.lower_,
                    "start_char": token.idx,
                    "end_char": token.idx + len(token.text),
                }
            )
            lower = token.lower_
            if lower in self.MODAL:
                cues.append(_cue("modal", token.idx, token.idx + len(token.text), token.text))
            if lower in self.CONDITION:
                cues.append(_cue("condition", token.idx, token.idx + len(token.text), token.text))
            if lower in self.NEGATION:
                cues.append(_cue("negation", token.idx, token.idx + len(token.text), token.text))

        for match in re.finditer(r"\[[0-9]+\]|\([0-9]+\)", context.text):
            cues.append(
                _cue(
                    "citation_like",
                    match.start(),
                    match.end(),
                    context.text[match.start() : match.end()],
                )
            )

        context.artifacts[self.id] = {
            "sentences": [
                {
                    "start_char": sentence.start_char,
                    "end_char": sentence.end_char,
                    "text": sentence.text,
                }
                for sentence in doc.sents
            ],
            "tokens": tokens,
            "cues": cues,
        }
        return StageResult(stage_id=self.id, status="ok")


def is_valid_span(text: str, start: int, end: int) -> bool:
    """Return whether a character interval is valid for the given text."""
    return not (start < 0 or end <= start or end > len(text))


def _cue(kind: str, start: int, end: int, text: str) -> dict[str, Any]:
    """Build a compact cue payload for the substrate trace."""
    return {"kind": kind, "start_char": start, "end_char": end, "text": text}
