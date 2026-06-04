from collections.abc import Callable

import pytest

import pragmalens.live_runtime as live_runtime
from pragmalens.verifier import (
    VerifierBackend,
    _build_crossencoder_from_runtime,
    _load_crossencoder_verifier,
    _load_minicheck_verifier,
)


class FakeMiniCheck:
    def __init__(self, *, model_name: str, cache_dir: str) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir


class FakeCrossEncoder:
    def __init__(self, model_name: str, *, cache_folder: str, revision: str) -> None:
        self.model_name = model_name
        self.cache_folder = cache_folder
        self.revision = revision


class FakeMiniCheckModule:
    def __init__(self, factory: Callable[..., FakeMiniCheck]) -> None:
        self.MiniCheck = factory


class FakeSentenceTransformersModule:
    def __init__(
        self,
        factory: Callable[..., FakeCrossEncoder],
    ) -> None:
        self.CrossEncoder = factory


def test_minicheck_loader_is_cached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str]] = []

    def fake_minicheck(*, model_name: str, cache_dir: str) -> FakeMiniCheck:
        calls.append((model_name, cache_dir))
        return FakeMiniCheck(model_name=model_name, cache_dir=cache_dir)

    monkeypatch.setattr(
        "pragmalens.verifier.import_module",
        lambda name: FakeMiniCheckModule(fake_minicheck),
    )
    _load_minicheck_verifier.cache_clear()

    first = _load_minicheck_verifier(model_name="mini", cache_dir="/tmp/cache")
    second = _load_minicheck_verifier(model_name="mini", cache_dir="/tmp/cache")

    assert first is second
    assert calls == [("mini", "/tmp/cache")]


def test_crossencoder_loader_accepts_sequence_keys_and_is_cached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_crossencoder(model_name: str, *, cache_folder: str, revision: str) -> FakeCrossEncoder:
        calls.append(model_name)
        return FakeCrossEncoder(model_name, cache_folder=cache_folder, revision=revision)

    monkeypatch.setattr(
        "pragmalens.live_runtime.importlib.import_module",
        lambda name: FakeSentenceTransformersModule(fake_crossencoder),
    )
    live_runtime.load_crossencoder_model.cache_clear()
    _load_crossencoder_verifier.cache_clear()

    first = _build_crossencoder_from_runtime(
        model_name="cross",
        cache_dir="/tmp/cross",
        revision="rev-1",
        label_mapping=["contradiction", "entailment", "neutral"],
    )
    second = _build_crossencoder_from_runtime(
        model_name="cross",
        cache_dir="/tmp/cross",
        revision="rev-1",
        label_mapping=["contradiction", "entailment", "neutral"],
    )

    assert first is second
    assert calls == ["cross"]
    assert getattr(first, "backend", None) is VerifierBackend.CROSSENCODER_NLI
    assert getattr(getattr(first, "model", None), "cache_folder", None) == "/tmp/cross"
    assert getattr(getattr(first, "model", None), "revision", None) == "rev-1"
