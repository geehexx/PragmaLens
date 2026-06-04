"""Shared runtime state and protocol definitions for the pipeline stage graph."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from pragmalens.models import EvidenceCandidate
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

    def run(self, context: RunContext) -> StageResult:
        """Execute the stage against the shared run context."""
        ...


class PipelineRunner:
    """Execute a linear list of stages against one run context."""

    def __init__(self, stages: list[PipelineStage]) -> None:
        """Store the stage sequence to execute."""
        self.stages = stages

    def run(self, context: RunContext) -> list[StageResult]:
        """Run every stage in order and collect per-stage results."""
        return [stage.run(context) for stage in self.stages]
