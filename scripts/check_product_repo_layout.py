#!/usr/bin/env python3
from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP_DOC = ROOT / "docs" / "local-codex-bootstrap.md"
PRIVATE_PATTERNS = ["AGENTS.md", "RTK.md", ".codex", ".agents"]


def fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    raise SystemExit(1)


def ok(msg: str) -> None:
    print(f"PASS: {msg}")


def main() -> None:
    result = subprocess.run(
        ["git", "ls-files", *PRIVATE_PATTERNS],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    tracked = [line for line in result.stdout.splitlines() if line]
    if tracked:
        fail(f"private control-plane files are still tracked: {tracked}")
    ok("private control-plane files are not tracked in the product repo")

    if not BOOTSTRAP_DOC.exists():
        fail(f"missing local bootstrap doc: {BOOTSTRAP_DOC}")
    ok("local Codex bootstrap doc exists")


if __name__ == "__main__":
    main()
