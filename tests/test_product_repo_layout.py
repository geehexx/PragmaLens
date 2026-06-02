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


def test_local_codex_bootstrap_doc_exists() -> None:
    assert Path("docs/local-codex-bootstrap.md").exists()
