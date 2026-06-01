from pragmalens.stages import RunContext, SpacySubstrateStage


def test_spacy_substrate_extracts_sentences_and_offsets() -> None:
    context = RunContext(document_id="doc", input_path="doc.md", text="Ship it. Do not wait.")

    SpacySubstrateStage().run(context)

    artifact = context.artifacts["spacy_substrate"]
    assert [s["text"] for s in artifact["sentences"]] == ["Ship it.", "Do not wait."]
    assert {"kind": "negation", "start_char": 12, "end_char": 15, "text": "not"} in artifact["cues"]
