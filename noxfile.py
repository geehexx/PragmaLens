from __future__ import annotations

import nox

nox.options.default_venv_backend = "uv"
nox.options.sessions = ["lint", "types", "tests", "build"]


def _run_uv(session: nox.Session, *args: str) -> None:
    session.run("uv", *args, external=True)


@nox.session
def lint(session: nox.Session) -> None:
    _run_uv(session, "run", "ruff", "check", ".")
    _run_uv(session, "run", "ruff", "format", "--check", ".")


@nox.session
def types(session: nox.Session) -> None:
    _run_uv(session, "run", "mypy", "src/pragmalens", "tests")
    _run_uv(session, "run", "interrogate", "src/pragmalens")


@nox.session(name="dev-fast")
def dev_fast(session: nox.Session) -> None:
    _run_uv(session, "run", "pytest", "-q", "-m", "not live_smoke", "--testmon")


@nox.session(name="dev-full")
def dev_full(session: nox.Session) -> None:
    _run_uv(session, "run", "pytest", "-q", "-m", "not live_smoke")


@nox.session
def tests(session: nox.Session) -> None:
    _run_uv(session, "run", "coverage", "run", "-m", "pytest", "-q", "-m", "not live_smoke")
    _run_uv(session, "run", "coverage", "report", "--skip-empty")
    _run_uv(session, "run", "python", "scripts/check_product_repo_layout.py")
    _run_uv(session, "run", "python", "scripts/check_schema_parity.py")
    _run_uv(session, "run", "python", "scripts/check_stage_graph.py")


@nox.session
def build(session: nox.Session) -> None:
    _run_uv(session, "run", "python", "-m", "compileall", "-q", "src", "tests")
    _run_uv(session, "run", "python", "-m", "build")


@nox.session(name="dev-live", default=False)
def dev_live(session: nox.Session) -> None:
    _run_uv(session, "run", "pytest", "-q", "-m", "live_smoke")


@nox.session(name="service-integration", default=False)
def service_integration(session: nox.Session) -> None:
    _run_uv(session, "run", "pytest", "-q", "-m", "service_integration")
