#!/usr/bin/env python3
from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP_DOC = ROOT / "docs" / "local-codex-bootstrap.md"
PRIVATE_PATTERNS = ["AGENTS.md", "RTK.md", ".codex", ".agents"]
PRIVATE_REPO_NAMES = {"pragmalens-codex", ".codex", ".agents"}


def fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    raise SystemExit(1)


def ok(msg: str) -> None:
    print(f"PASS: {msg}")


def _git_ls_files(*paths: str) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", *paths],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return [line for line in result.stdout.splitlines() if line]


def _tracked_files() -> list[str]:
    return _git_ls_files()


def _assert_no_private_symlink_targets() -> None:
    repo_root = ROOT.resolve()
    offenders: list[str] = []
    for rel_path in _tracked_files():
        path = ROOT / rel_path
        if not path.is_symlink():
            continue
        target = path.resolve(strict=False)
        escapes_repo = not str(target).startswith(f"{repo_root}/") and target != repo_root
        mentions_private_repo = any(part in PRIVATE_REPO_NAMES for part in target.parts)
        if escapes_repo or mentions_private_repo:
            offenders.append(f"{rel_path} -> {target}")
    if offenders:
        fail(
            "tracked symlinks escape the product repo or resolve into private control state: "
            f"{offenders}"
        )
    ok("tracked files do not include symlinks into private control state")


def main() -> None:
    tracked = _git_ls_files(*PRIVATE_PATTERNS)
    if tracked:
        fail(f"private control-plane files are still tracked: {tracked}")
    ok("private control-plane files are not tracked in the product repo")
    _assert_no_private_symlink_targets()

    if not BOOTSTRAP_DOC.exists():
        fail(f"missing local bootstrap doc: {BOOTSTRAP_DOC}")
    ok("local Codex bootstrap doc exists")


if __name__ == "__main__":
    main()
