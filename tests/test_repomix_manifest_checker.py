import hashlib
import subprocess
import sys
from pathlib import Path

CHECKER = Path(".codex/plan_tools/check_repomix_manifest.py")


def _write_manifest(tmp_path: Path, xml_path: Path, sha: str, size: int, count: int) -> Path:
    manifest = tmp_path / "manifest.md"
    manifest.write_text(
        "\n".join(
            [
                "profile: D-executable-audit",
                "## Default Package",
                f"- Path: `{xml_path}`",
                "- Format: XML, parsable-style, split output",
                f"- Size: {size} bytes",
                f"- SHA256: {sha}",
                f"- File count: {count}",
                "## Include / Exclude Baseline",
                "## Compression Marker Policy",
                "## Validation Commands",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return manifest


def test_repomix_manifest_checker_fails_on_stale_hash(tmp_path: Path) -> None:
    xml = tmp_path / "package.xml"
    xml.write_text('<repomix><files><file path="pyproject.toml">x</file></files></repomix>')
    manifest = _write_manifest(tmp_path, xml, "0" * 64, xml.stat().st_size, 1)

    result = subprocess.run(
        [sys.executable, str(CHECKER), str(manifest)],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert "sha256 mismatch" in result.stdout


def test_repomix_manifest_checker_passes_matching_package(tmp_path: Path) -> None:
    xml = tmp_path / "package.xml"
    xml.write_text(
        "<repomix><files>"
        '<file path="pyproject.toml">x</file>'
        '<file path="AGENTS.md">x</file>'
        '<file path="RTK.md">x</file>'
        '<file path=".codex/superpowers/v0.1/ACTIVE.md">x</file>'
        '<file path=".codex/superpowers/v0.1/registry/gates.md">x</file>'
        '<file path=".codex/plan_tools/check_repomix_manifest.py">x</file>'
        "</files></repomix>",
        encoding="utf-8",
    )
    sha = hashlib.sha256(xml.read_bytes()).hexdigest()
    manifest = _write_manifest(tmp_path, xml, sha, xml.stat().st_size, 6)

    result = subprocess.run(
        [sys.executable, str(CHECKER), str(manifest)],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "manifest hash, byte count, and file count match" in result.stdout
