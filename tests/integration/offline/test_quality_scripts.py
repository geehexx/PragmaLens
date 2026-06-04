from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _run_script(path: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, path],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        check=False,
    )


def test_schema_parity_script_passes() -> None:
    result = _run_script("scripts/check_schema_parity.py")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS:" in result.stdout


def test_stage_graph_script_passes() -> None:
    result = _run_script("scripts/check_stage_graph.py")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS:" in result.stdout


def test_lane_alignment_script_passes() -> None:
    result = _run_script("scripts/check_lane_alignment.py")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS:" in result.stdout


def test_security_scan_alignment_script_passes() -> None:
    result = _run_script("scripts/check_security_scan_alignment.py")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS:" in result.stdout
