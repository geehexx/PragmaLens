import pytest

from pragmalens.models import (
    CandidateStatus,
    EvidenceCandidate,
    SpanRef,
    VerificationStatus,
    VerificationVerdict,
)
from pragmalens.pipeline.runtime import RunContext, StageResult
from pragmalens.pipeline.verification import SynthesizeFindingsStage, VerifyClaimsStage
from pragmalens.verifier import (
    CrossEncoderNliVerifier,
    MiniCheckVerifier,
    OfflineBaselineVerifier,
    VerifierBackend,
    build_verifier_adapter,
    build_verifier_from_env,
)


@pytest.mark.parametrize("status", list(VerificationStatus))
def test_verification_verdict_states_serialize(status: VerificationStatus) -> None:
    verdict = VerificationVerdict(
        candidate_id="candidate-1",
        status=status,
        rationale="contract fixture",
        evidence_ids=["fixture"],
        error="boom" if status is VerificationStatus.VERIFIER_ERROR else None,
    )

    assert verdict.model_dump(mode="json")["status"] == status.value


def test_verifier_stage_records_empty_verdicts_without_silent_failure() -> None:
    context = RunContext(document_id="doc", input_path="doc.md", text="No candidates.")

    result = VerifyClaimsStage().run(context)

    assert isinstance(result, StageResult)
    assert result.status == "ok"
    assert context.metadata["verification"] == []
    assert context.artifacts["verify_claims"] == {"verdicts": [], "skipped_candidates": []}


def test_verifier_stage_skips_non_valid_candidates() -> None:
    context = RunContext(
        document_id="doc",
        input_path="doc.md",
        text="No candidates.",
        candidates=[
            EvidenceCandidate(
                candidate_id="candidate-1",
                label="claim",
                kind="claim",
                status=CandidateStatus.QUARANTINED,
                span=SpanRef(document_id="doc", start_char=0, end_char=1, text="N"),
                provenance=["fixture"],
            )
        ],
    )

    result = VerifyClaimsStage().run(context)

    assert isinstance(result, StageResult)
    assert result.status == "ok"
    assert context.metadata["verification"] == []
    assert len(context.artifacts["verify_claims"]["skipped_candidates"]) == 1
    assert "verifier_skipped_non_valid:candidate-1" in context.warnings


def test_evidence_candidate_backfills_evidence_refs_from_provenance() -> None:
    candidate = EvidenceCandidate(
        candidate_id="candidate-1",
        label="claim",
        kind="claim",
        span=SpanRef(document_id="doc", start_char=0, end_char=1, text="N"),
        provenance=["fixture"],
    )

    assert candidate.evidence_refs == ["fixture"]


def test_verifier_stage_uses_evidence_refs_for_valid_candidates() -> None:
    context = RunContext(
        document_id="doc",
        input_path="doc.md",
        text="Need evidence.",
        candidates=[
            EvidenceCandidate(
                candidate_id="candidate-1",
                label="claim",
                kind="claim",
                status=CandidateStatus.VALID,
                span=SpanRef(document_id="doc", start_char=0, end_char=4, text="Need"),
                provenance=["fixture"],
                evidence_refs=["source-a", "source-b"],
            )
        ],
    )

    result = VerifyClaimsStage().run(context)

    assert isinstance(result, StageResult)
    assert result.status == "ok"
    assert context.metadata["verification"][0].evidence_ids == ["source-a", "source-b"]
    assert context.artifacts["verify_claims"]["verdicts"][0]["evidence_ids"] == [
        "source-a",
        "source-b",
    ]


class RaisingVerifier:
    def verify(
        self,
        candidates: list[EvidenceCandidate],
        *,
        document_id: str,
        text: str,
    ) -> list[VerificationVerdict]:
        del candidates, document_id, text
        raise RuntimeError("boom")


class WrongCountVerifier:
    def verify(
        self,
        candidates: list[EvidenceCandidate],
        *,
        document_id: str,
        text: str,
    ) -> list[VerificationVerdict]:
        del candidates, document_id, text
        return []


class FakeMiniCheckScorer:
    def score(
        self,
        *,
        docs: list[str],
        claims: list[str],
    ) -> tuple[list[int], list[float], None, None]:
        assert docs == ["Need evidence."]
        assert claims == ["Need"]
        return [1], [0.97], None, None


class FakeCrossEncoderModel:
    def predict(
        self,
        pairs: list[tuple[str, str]],
        *,
        apply_softmax: bool = False,
    ) -> list[list[float]]:
        assert apply_softmax is True
        assert pairs == [("Need evidence.", "Need")]
        return [[0.1, 0.8, 0.1]]


class FakeMiniCheckUnsupportedScorer:
    def score(
        self,
        *,
        docs: list[str],
        claims: list[str],
    ) -> tuple[list[str], list[float], None, None]:
        assert docs == ["Need evidence."]
        assert claims == ["Need"]
        return ["unsupported"], [0.02], None, None


class FakeMiniCheckBadLabelScorer:
    def score(
        self,
        *,
        docs: list[str],
        claims: list[str],
    ) -> tuple[list[str], list[float], None, None]:
        assert docs == ["Need evidence."]
        assert claims == ["Need"]
        return ["maybe"], [0.5], None, None


class FakeMiniCheckWrongCountScorer:
    def score(
        self,
        *,
        docs: list[str],
        claims: list[str],
    ) -> tuple[list[int], list[float], None, None]:
        assert docs == ["Need evidence."]
        assert claims == ["Need"]
        return [1, 0], [0.9], None, None


class FakeCrossEncoderContradictionModel:
    def predict(
        self,
        pairs: list[tuple[str, str]],
        *,
        apply_softmax: bool = False,
    ) -> list[list[float]]:
        assert apply_softmax is True
        assert pairs == [("Need evidence.", "Need")]
        return [[0.9, 0.05, 0.05]]


class FakeCrossEncoderNeutralModel:
    def predict(
        self,
        pairs: list[tuple[str, str]],
        *,
        apply_softmax: bool = False,
    ) -> list[list[float]]:
        assert apply_softmax is True
        assert pairs == [("Need evidence.", "Need")]
        return [[0.1, 0.2, 0.7]]


class FakeCrossEncoderEmptyRowModel:
    def predict(
        self,
        pairs: list[tuple[str, str]],
        *,
        apply_softmax: bool = False,
    ) -> list[list[float]]:
        assert apply_softmax is True
        assert pairs == [("Need evidence.", "Need")]
        return [[]]


class FakeCrossEncoderWideRowModel:
    def predict(
        self,
        pairs: list[tuple[str, str]],
        *,
        apply_softmax: bool = False,
    ) -> list[list[float]]:
        assert apply_softmax is True
        assert pairs == [("Need evidence.", "Need")]
        return [[0.1, 0.2, 0.3, 0.4]]


def test_verifier_stage_emits_error_verdicts_when_adapter_raises() -> None:
    context = RunContext(
        document_id="doc",
        input_path="doc.md",
        text="Need evidence.",
        candidates=[
            EvidenceCandidate(
                candidate_id="candidate-1",
                label="claim",
                kind="claim",
                status=CandidateStatus.VALID,
                span=SpanRef(document_id="doc", start_char=0, end_char=4, text="Need"),
                provenance=["fixture"],
                evidence_refs=["source-a"],
            )
        ],
    )

    result = VerifyClaimsStage(verifier=RaisingVerifier()).run(context)

    assert isinstance(result, StageResult)
    assert result.status == "ok"
    assert context.metadata["verification"][0].status is VerificationStatus.VERIFIER_ERROR
    assert context.metadata["verification"][0].evidence_ids == ["source-a"]
    assert "verifier_adapter_failed" in context.warnings


def test_verifier_stage_emits_error_verdicts_on_cardinality_mismatch() -> None:
    context = RunContext(
        document_id="doc",
        input_path="doc.md",
        text="Need evidence.",
        candidates=[
            EvidenceCandidate(
                candidate_id="candidate-1",
                label="claim",
                kind="claim",
                status=CandidateStatus.VALID,
                span=SpanRef(document_id="doc", start_char=0, end_char=4, text="Need"),
                provenance=["fixture"],
                evidence_refs=["source-a"],
            )
        ],
    )

    result = VerifyClaimsStage(verifier=WrongCountVerifier()).run(context)

    assert isinstance(result, StageResult)
    assert result.status == "ok"
    assert context.metadata["verification"][0].status is VerificationStatus.VERIFIER_ERROR
    assert context.metadata["verification"][0].error == "verifier returned wrong verdict count"
    assert context.artifacts["verify_claims"]["verdicts"][0]["evidence_ids"] == ["source-a"]


def test_synthesize_findings_stage_adds_questions_and_actionability() -> None:
    context = RunContext(
        document_id="doc",
        input_path="doc.md",
        text="Need evidence.",
        metadata={
            "verification": [
                VerificationVerdict(
                    candidate_id="candidate-1",
                    status=VerificationStatus.UNSUPPORTED,
                    rationale="verifier found insufficient support",
                    evidence_ids=["source-a"],
                ),
                VerificationVerdict(
                    candidate_id="candidate-2",
                    status=VerificationStatus.VERIFIER_ERROR,
                    rationale="verifier adapter failed",
                    evidence_ids=["source-b"],
                    error="boom",
                ),
            ]
        },
    )

    result = SynthesizeFindingsStage().run(context)

    assert isinstance(result, StageResult)
    findings = context.metadata["findings"]
    assert len(findings) == 2
    assert findings[0].question == "What supporting evidence would let this claim be verified?"
    assert findings[0].actionability == (
        "Collect corroborating evidence or revise the claim before promoting it."
    )
    assert findings[1].question == (
        "Can the verifier be rerun with the required live dependencies installed?"
    )
    assert findings[1].actionability == (
        "Fix the verifier runtime or install the missing backend before rerunning this claim."
    )
    assert context.artifacts["synthesize_findings"]["findings"][0]["question"] == (
        findings[0].question
    )


def test_minicheck_verifier_maps_binary_scores() -> None:
    candidate = EvidenceCandidate(
        candidate_id="candidate-1",
        label="claim",
        kind="claim",
        status=CandidateStatus.VALID,
        span=SpanRef(document_id="doc", start_char=0, end_char=4, text="Need"),
        provenance=["fixture"],
        evidence_refs=["source-a"],
    )

    verdicts = MiniCheckVerifier(FakeMiniCheckScorer()).verify(
        [candidate], document_id="doc", text="Need evidence."
    )

    assert verdicts[0].status is VerificationStatus.SUPPORTED
    assert verdicts[0].evidence_ids == ["source-a"]
    assert "0.970" in verdicts[0].rationale


def test_crossencoder_verifier_maps_entailment_scores() -> None:
    candidate = EvidenceCandidate(
        candidate_id="candidate-1",
        label="claim",
        kind="claim",
        status=CandidateStatus.VALID,
        span=SpanRef(document_id="doc", start_char=0, end_char=4, text="Need"),
        provenance=["fixture"],
        evidence_refs=["source-a"],
    )

    verdicts = CrossEncoderNliVerifier(FakeCrossEncoderModel()).verify(
        [candidate], document_id="doc", text="Need evidence."
    )

    assert verdicts[0].status is VerificationStatus.SUPPORTED
    assert verdicts[0].evidence_ids == ["source-a"]
    assert verdicts[0].rationale.endswith("entailment")


def test_minicheck_verifier_maps_unsupported_scores() -> None:
    candidate = EvidenceCandidate(
        candidate_id="candidate-1",
        label="claim",
        kind="claim",
        status=CandidateStatus.VALID,
        span=SpanRef(document_id="doc", start_char=0, end_char=4, text="Need"),
        provenance=["fixture"],
        evidence_refs=["source-a"],
    )

    verdicts = MiniCheckVerifier(FakeMiniCheckUnsupportedScorer()).verify(
        [candidate], document_id="doc", text="Need evidence."
    )

    assert verdicts[0].status is VerificationStatus.UNSUPPORTED
    assert "0.020" in verdicts[0].rationale


def test_minicheck_verifier_rejects_unknown_labels() -> None:
    candidate = EvidenceCandidate(
        candidate_id="candidate-1",
        label="claim",
        kind="claim",
        status=CandidateStatus.VALID,
        span=SpanRef(document_id="doc", start_char=0, end_char=4, text="Need"),
        provenance=["fixture"],
        evidence_refs=["source-a"],
    )

    with pytest.raises(ValueError, match="Unsupported MiniCheck label value"):
        MiniCheckVerifier(FakeMiniCheckBadLabelScorer()).verify(
            [candidate], document_id="doc", text="Need evidence."
        )


def test_minicheck_verifier_rejects_result_count_mismatch() -> None:
    candidate = EvidenceCandidate(
        candidate_id="candidate-1",
        label="claim",
        kind="claim",
        status=CandidateStatus.VALID,
        span=SpanRef(document_id="doc", start_char=0, end_char=4, text="Need"),
        provenance=["fixture"],
        evidence_refs=["source-a"],
    )

    with pytest.raises(ValueError, match="MiniCheck scorer returned wrong result count"):
        MiniCheckVerifier(FakeMiniCheckWrongCountScorer()).verify(
            [candidate], document_id="doc", text="Need evidence."
        )


def test_crossencoder_verifier_maps_contradiction_scores() -> None:
    candidate = EvidenceCandidate(
        candidate_id="candidate-1",
        label="claim",
        kind="claim",
        status=CandidateStatus.VALID,
        span=SpanRef(document_id="doc", start_char=0, end_char=4, text="Need"),
        provenance=["fixture"],
        evidence_refs=["source-a"],
    )

    verdicts = CrossEncoderNliVerifier(FakeCrossEncoderContradictionModel()).verify(
        [candidate], document_id="doc", text="Need evidence."
    )

    assert verdicts[0].status is VerificationStatus.UNSUPPORTED
    assert verdicts[0].rationale.endswith("contradiction")


def test_crossencoder_verifier_maps_neutral_scores() -> None:
    candidate = EvidenceCandidate(
        candidate_id="candidate-1",
        label="claim",
        kind="claim",
        status=CandidateStatus.VALID,
        span=SpanRef(document_id="doc", start_char=0, end_char=4, text="Need"),
        provenance=["fixture"],
        evidence_refs=["source-a"],
    )

    verdicts = CrossEncoderNliVerifier(FakeCrossEncoderNeutralModel()).verify(
        [candidate], document_id="doc", text="Need evidence."
    )

    assert verdicts[0].status is VerificationStatus.INSUFFICIENT_EVIDENCE
    assert verdicts[0].rationale.endswith("neutral")


def test_crossencoder_verifier_rejects_empty_score_rows() -> None:
    candidate = EvidenceCandidate(
        candidate_id="candidate-1",
        label="claim",
        kind="claim",
        status=CandidateStatus.VALID,
        span=SpanRef(document_id="doc", start_char=0, end_char=4, text="Need"),
        provenance=["fixture"],
        evidence_refs=["source-a"],
    )

    with pytest.raises(ValueError, match="empty score row"):
        CrossEncoderNliVerifier(FakeCrossEncoderEmptyRowModel()).verify(
            [candidate], document_id="doc", text="Need evidence."
        )


def test_crossencoder_verifier_rejects_label_mapping_mismatch() -> None:
    candidate = EvidenceCandidate(
        candidate_id="candidate-1",
        label="claim",
        kind="claim",
        status=CandidateStatus.VALID,
        span=SpanRef(document_id="doc", start_char=0, end_char=4, text="Need"),
        provenance=["fixture"],
        evidence_refs=["source-a"],
    )

    with pytest.raises(ValueError, match="label mapping does not match score width"):
        CrossEncoderNliVerifier(FakeCrossEncoderWideRowModel()).verify(
            [candidate], document_id="doc", text="Need evidence."
        )


def test_build_verifier_adapter_returns_offline_baseline() -> None:
    verifier = build_verifier_adapter(VerifierBackend.OFFLINE)

    assert isinstance(verifier, OfflineBaselineVerifier)


def test_build_verifier_from_env_defaults_to_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PRAGMALENS_VERIFIER", raising=False)

    verifier = build_verifier_from_env()

    assert isinstance(verifier, OfflineBaselineVerifier)


def test_build_verifier_from_env_uses_minicheck_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    marker = object()

    def fake_builder(*, model_name: str, cache_dir: str) -> object:
        assert model_name == "mini-model"
        assert cache_dir == "/tmp/minicheck"
        return marker

    monkeypatch.setattr("pragmalens.verifier._build_minicheck_from_runtime", fake_builder)
    monkeypatch.setenv("PRAGMALENS_VERIFIER", "minicheck")
    monkeypatch.setenv("PRAGMALENS_MINICHECK_MODEL", "mini-model")
    monkeypatch.setenv("PRAGMALENS_MINICHECK_CACHE_DIR", "/tmp/minicheck")

    verifier = build_verifier_from_env()

    assert verifier is marker


def test_build_verifier_from_env_uses_crossencoder_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = object()

    def fake_builder(*, model_name: str, label_mapping: tuple[str, ...]) -> object:
        assert model_name == "ce-model"
        assert label_mapping == ("contradiction", "entailment", "neutral")
        return marker

    monkeypatch.setattr("pragmalens.verifier._build_crossencoder_from_runtime", fake_builder)
    monkeypatch.setenv("PRAGMALENS_VERIFIER", "crossencoder_nli")
    monkeypatch.setenv("PRAGMALENS_CROSSENCODER_MODEL", "ce-model")

    verifier = build_verifier_from_env()

    assert verifier is marker
