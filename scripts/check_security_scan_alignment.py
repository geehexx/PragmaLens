#!/usr/bin/env python3
"""Verify the security-scan workflow, pin, and README stay aligned."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CI = ROOT / ".github" / "workflows" / "ci.yml"
README = ROOT / "README.md"

EXPECTED_ACTION = "DariuszPorowski/github-action-gitleaks@a027585cb1f2780aa441f20f4d0aabf8b1a0d90e"
EXPECTED_TREE_SCAN_SNIPPET = "no_git: true"
EXPECTED_CI_TREE_ONLY_SNIPPET = "CI runs a tree-only scan"
EXPECTED_LOCAL_TREE_SNIPPET = "Local tree scan:"
EXPECTED_LOCAL_HISTORY_SNIPPET = "Optional local history scan:"
EXPECTED_LOCAL_ONLY_SNIPPET = "history scan is local-only"


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


def _check_ci(ci_text: str) -> None:
    if EXPECTED_ACTION not in ci_text:
        fail(f"expected pinned gitleaks action not found: {EXPECTED_ACTION}")
    if EXPECTED_TREE_SCAN_SNIPPET not in ci_text:
        fail("CI secrets job is missing the tree-only gitleaks mode")
    if re.search(r"fetch-depth:\s*0", ci_text):
        fail("CI secrets job still fetches full history for a tree-only scan")
    ok("CI uses the pinned tree-only gitleaks action")


def _check_readme(readme_text: str) -> None:
    required_snippets = [
        EXPECTED_CI_TREE_ONLY_SNIPPET,
        EXPECTED_ACTION,
        EXPECTED_LOCAL_TREE_SNIPPET,
        EXPECTED_LOCAL_HISTORY_SNIPPET,
        EXPECTED_LOCAL_ONLY_SNIPPET,
    ]
    missing = [snippet for snippet in required_snippets if snippet not in readme_text]
    if missing:
        fail(f"README security-scan guidance is missing: {missing}")
    ok("README distinguishes tree-only CI from optional local history scanning")


def main() -> None:
    """Compare the security-scan policy surfaces against the canonical policy."""
    ci_text = _read(CI)
    readme_text = _read(README)
    _check_ci(ci_text)
    _check_readme(readme_text)
    print("PASS: security-scan alignment is internally consistent")


if __name__ == "__main__":
    main()
