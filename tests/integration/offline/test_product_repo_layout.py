from __future__ import annotations

import subprocess
from pathlib import Path


def test_private_control_plane_files_are_not_tracked() -> None:
    result = subprocess.run(
        ["git", "ls-files", "AGENTS.md", "RTK.md", ".codex", ".agents"],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        check=True,
    )

    tracked = [line for line in result.stdout.splitlines() if line]
    assert tracked == []


def test_tracked_files_do_not_include_private_symlinks() -> None:
    result = subprocess.run(
        ["uv", "run", "python", "scripts/check_product_repo_layout.py"],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        check=True,
    )
    assert "tracked files do not include symlinks into private control state" in result.stdout


def test_local_codex_bootstrap_doc_exists() -> None:
    assert Path("docs/local-codex-bootstrap.md").exists()


def test_removed_transitional_files_are_absent() -> None:
    assert not Path("src/pragmalens/pr04.py").exists()
