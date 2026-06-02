"""Backward-compatible shim for older imports.

Prefer `pragmalens.pipeline.captured_extraction`.
"""

from typing import Any

from pragmalens.pipeline.captured_extraction import (  # noqa: F401
    candidates_to_report_payload,
    run_captured_extraction_pipeline,
)


def run_captured_extraction(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Backward-compatible wrapper around the captured extraction entrypoint."""
    return run_captured_extraction_pipeline(*args, **kwargs)
