import pytest


@pytest.mark.live_smoke
def test_langextract_live_smoke_placeholder() -> None:
    pytest.skip("Live LangExtract provider smoke test is intentionally excluded from default offline CI.")


@pytest.mark.live_smoke
def test_gliner2_live_smoke_placeholder() -> None:
    pytest.skip("Live GLiNER2 model smoke test is intentionally excluded from default offline CI.")


@pytest.mark.live_smoke
def test_minicheck_live_smoke_placeholder() -> None:
    pytest.skip(
        "MiniCheck live smoke requires VCS install: pip install \"minicheck @ git+https://github.com/Liyan06/MiniCheck.git@main\""
    )
