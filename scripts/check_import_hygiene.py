#!/usr/bin/env python3
"""Verify imports stay at module top level in product and policy code."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCAN_ROOTS = [
    ROOT / "src" / "pragmalens",
    ROOT / "tests",
    ROOT / "scripts",
    ROOT / ".codex" / "plan_tools",
]


class ImportVisitor(ast.NodeVisitor):
    """Collect nested import statements from one parsed module."""

    def __init__(self, path: Path) -> None:
        """Bind the current file path and initialize the traversal state."""
        self._path = path
        self._parents: list[ast.AST] = []
        self.offenders: list[str] = []

    def generic_visit(self, node: ast.AST) -> None:
        """Walk the AST while collecting nested import statements."""
        self._parents.append(node)
        if isinstance(node, (ast.Import, ast.ImportFrom)) and any(
            isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Try))
            for parent in self._parents[:-1]
        ):
            self.offenders.append(
                f"{self._path}:{node.lineno}:{node.col_offset} nested import statement"
            )
        super().generic_visit(node)
        self._parents.pop()


def fail(msg: str) -> None:
    """Exit with a failing status and a human-readable error message."""
    print(f"FAIL: {msg}")
    raise SystemExit(1)


def main() -> None:
    """Reject imports nested inside functions or try blocks."""
    offenders: list[str] = []
    for root in SCAN_ROOTS:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            visitor = ImportVisitor(path)
            visitor.visit(tree)
            offenders.extend(visitor.offenders)
    if offenders:
        fail("nested import statements found: " + "; ".join(offenders[:12]))
    print("PASS: imports stay at module top level in scanned product/policy code")


if __name__ == "__main__":
    main()
