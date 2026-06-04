"""Canonical local and CI verification lanes for the PragmaLens repo."""

from __future__ import annotations

import nox

nox.options.default_venv_backend = "uv"
nox.options.sessions = ["repo-layout", "lint", "types", "offline-verify", "build"]


def _run_uv(session: nox.Session, *args: str) -> None:
    session.run("uv", *args, external=True)


@nox.session(name="repo-layout")
def repo_layout(session: nox.Session) -> None:
    """Verify product-repo boundary rules and local overlay exclusions."""
    _run_uv(session, "run", "python", "scripts/check_product_repo_layout.py")
    _run_uv(session, "run", "python", "scripts/check_lane_alignment.py")


@nox.session
def lint(session: nox.Session) -> None:
    """Run formatter-compatible lint checks for tracked code."""
    _run_uv(session, "run", "ruff", "check", ".")
    _run_uv(session, "run", "lint-imports")
    _run_uv(session, "run", "ruff", "format", "--check", ".")


@nox.session
def types(session: nox.Session) -> None:
    """Run static typing and docstring coverage checks."""
    _run_uv(session, "run", "mypy", "src/pragmalens", "tests")
    _run_uv(session, "run", "interrogate", "src/pragmalens")


@nox.session(name="dev-fast")
def dev_fast(session: nox.Session) -> None:
    """Run the fastest local impacted-test lane."""
    _run_uv(session, "run", "--group", "qa", "pytest", "-q", "-m", "not live_smoke", "--testmon")


@nox.session(name="dev-full")
def dev_full(session: nox.Session) -> None:
    """Run the full offline pytest lane without coverage aggregation."""
    _run_uv(session, "run", "pytest", "-q", "-m", "not live_smoke")


@nox.session(name="offline-verify")
def offline_verify(session: nox.Session) -> None:
    """Run the canonical offline verification lane used by CI and hooks."""
    _run_uv(session, "lock", "--check")
    _run_uv(session, "run", "python", "-m", "compileall", "-q", "src", "tests")
    _run_uv(session, "run", "coverage", "run", "-m", "pytest", "-q", "-m", "not live_smoke")
    _run_uv(session, "run", "coverage", "report", "--skip-empty")
    _run_uv(session, "run", "python", "scripts/check_schema_parity.py")
    _run_uv(session, "run", "python", "scripts/check_stage_graph.py")


@nox.session(name="dev-live", default=False)
def dev_live(session: nox.Session) -> None:
    """Run opt-in live smoke tests only."""
    _run_uv(session, "run", "--group", "live", "pytest", "-q", "-m", "live_smoke")


@nox.session(name="live-verifier")
def live_verifier(session: nox.Session) -> None:
    """Run the CPU-safe live verifier slice used by default CI."""
    session.env["PRAGMALENS_ENABLE_LIVE_SMOKE"] = "1"
    session.env["CUDA_VISIBLE_DEVICES"] = ""
    _run_uv(session, "run", "--group", "live", "python", "-m", "nltk.downloader", "punkt_tab")
    _run_uv(
        session,
        "run",
        "--group",
        "live",
        "pytest",
        "tests/live_smoke/test_runtime_backends.py",
        "-k",
        "minicheck or crossencoder",
    )


@nox.session
def build(session: nox.Session) -> None:
    """Build distributable artifacts for the current working tree."""
    _run_uv(session, "run", "python", "-m", "build")
