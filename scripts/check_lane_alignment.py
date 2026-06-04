#!/usr/bin/env python3
"""Verify the repo's install, hook, CI, and README lane definitions stay aligned."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
README = ROOT / "README.md"
NOXFILE = ROOT / "noxfile.py"
LEFTHOOK = ROOT / "lefthook.yml"
CI = ROOT / ".github" / "workflows" / "ci.yml"
DEV_FAST_PATTERN = re.compile(
    r'_run_uv\(session,\s*"run",\s*"--group",\s*"qa",\s*"pytest",\s*'
    r'"-q",\s*"-m",\s*"not live_smoke",\s*"--testmon"\)'
)
DEV_LIVE_PATTERN = re.compile(
    r'_run_uv\(session,\s*"run",\s*"--group",\s*"live",\s*"pytest",\s*'
    r'"-q",\s*"-m",\s*"live_smoke"\)'
)
LIVE_VERIFIER_SNIPPETS = (
    '"--group", "live"',
    '"pytest"',
    '"tests/live_smoke/test_runtime_backends.py"',
    '"-k"',
    '"minicheck or crossencoder"',
)


def fail(msg: str) -> None:
    """Exit with a failing status and a human-readable error message."""
    print(f"FAIL: {msg}")
    raise SystemExit(1)


def ok(msg: str) -> None:
    """Print a successful check message."""
    print(f"PASS: {msg}")


def _read(path: Path) -> str:
    if not path.exists():
        fail(f"missing file: {path}")
    return path.read_text(encoding="utf-8")


def _marker_names(markers: list[str]) -> set[str]:
    names = set()
    for marker in markers:
        names.add(marker.split(":", 1)[0].strip())
    return names


def _ensure_contains(text: str, snippet: str, message: str) -> None:
    if snippet not in text:
        fail(message)


def _check_pyproject() -> None:
    data = tomllib.loads(_read(PYPROJECT))
    tool_uv = data.get("tool", {}).get("uv", {})
    default_groups = tool_uv.get("default-groups")
    if default_groups != ["dev"]:
        fail(f"expected tool.uv.default-groups = ['dev'], got {default_groups!r}")
    ok("default uv install surface stays minimal")

    pytest_ini = data.get("tool", {}).get("pytest", {}).get("ini_options", {})
    marker_names = _marker_names(pytest_ini.get("markers", []))
    expected_markers = {"live_smoke", "slow"}
    if marker_names != expected_markers:
        fail(f"pytest marker set mismatch: expected {expected_markers}, got {marker_names}")
    ok("pytest marker set matches the current lane model")

    ruff_select = data.get("tool", {}).get("ruff", {}).get("lint", {}).get("select", [])
    if "PLC0415" not in ruff_select:
        fail("ruff no longer enforces nested-import hygiene with PLC0415")
    ok("ruff enforces nested-import hygiene")

    import_linter = data.get("tool", {}).get("importlinter", {})
    if import_linter.get("root_packages") != ["pragmalens"]:
        fail("import-linter root package configuration is missing or changed")
    if not import_linter.get("contracts"):
        fail("import-linter layered architecture contract is missing")
    ok("import-linter enforces the package structure contract")


def _check_noxfile() -> None:
    nox_text = _read(NOXFILE)
    _ensure_contains(
        nox_text,
        'nox.options.sessions = ["repo-layout", "lint", "types", "offline-verify", "build"]',
        "nox default sessions no longer match the documented canonical set",
    )
    if not DEV_FAST_PATTERN.search(nox_text):
        fail("dev-fast nox session is not pinned to the qa group")
    if not DEV_LIVE_PATTERN.search(nox_text):
        fail("dev-live nox session is not pinned to the live group")
    if "service-integration" in nox_text:
        fail("dead service-integration nox session still exists")
    if "lint-imports" not in nox_text:
        fail("nox lint session no longer runs import-linter")
    if not all(snippet in nox_text for snippet in LIVE_VERIFIER_SNIPPETS):
        fail("live-verifier nox session is not pinned to the CPU-safe live verifier slice")
    ok("nox sessions match the trimmed lane surface")


def _check_lefthook() -> None:
    lefthook_text = _read(LEFTHOOK)
    for token in ("repo-layout:", "ruff:", "format:", "nox-types:"):
        if token not in lefthook_text:
            fail(f"lefthook entry missing: {token}")
    for token in ("nox-offline-verify:", "nox-build:"):
        if token in lefthook_text:
            fail(f"heavy pre-push lane still present in lefthook: {token}")
    ok("lefthook stays cheap and bounded")


def _check_readme() -> None:
    readme_text = _read(README)
    required_snippets = [
        "`uv sync` installs the minimal default development environment only.",
        "uv sync --group qa",
        "uv sync --group live",
        "uv run nox -s dev-fast",
        "uv run nox -s live-verifier",
        "uv run nox -s dev-live",
        "CPU-safe live-verifier lane",
        "lint-imports",
    ]
    missing = [snippet for snippet in required_snippets if snippet not in readme_text]
    if missing:
        fail(f"README missing lane/install guidance: {missing}")
    if "Live/model tests remain outside default CI" in readme_text:
        fail("README still contains stale default-CI wording for live/model tests")
    ok("README documents the minimal default install and opt-in lanes")


def _check_ci() -> None:
    ci_text = _read(CI)
    required_ci_snippets = [
        "repo-layout:",
        "lint-types:",
        "offline-verify:",
        "secrets:",
        "uv run nox -s repo-layout",
        "uv run nox -s lint types",
        "uv run nox -s offline-verify build",
        "uv run nox -s live-verifier",
    ]
    missing_ci = [snippet for snippet in required_ci_snippets if snippet not in ci_text]
    if missing_ci:
        fail(f"CI no longer matches the canonical lane set: {missing_ci}")
    ok("CI matches the canonical lane set")


def main() -> None:
    """Check the lane-definition surfaces against a single canonical policy."""
    _check_pyproject()
    _check_noxfile()
    _check_lefthook()
    _check_readme()
    _check_ci()

    print("PASS: lane alignment is internally consistent")


if __name__ == "__main__":
    main()
