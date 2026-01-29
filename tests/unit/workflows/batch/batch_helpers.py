"""Helper functions for batch workflow tests.

These functions are defined at module level to be picklable for ProcessPoolExecutor tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def simple_analysis(filepath: str | Path) -> dict[str, Any]:
    """Simple analysis that returns file size."""
    return {"size": len(Path(filepath).read_text())}


def failing_analysis(filepath: str | Path) -> dict[str, Any]:
    """Analysis that always fails."""
    raise ValueError("Test error")


def conditional_fail_analysis(filepath: str | Path) -> dict[str, Any]:
    """Analysis that fails on file_1."""
    if "file_1" in str(filepath):
        raise RuntimeError("File 1 failed")
    return {"result": "success"}


def conditional_fail_file_0(filepath: str | Path) -> dict[str, Any]:
    """Analysis that fails on file_0."""
    if "file_0" in str(filepath):
        raise RuntimeError("Test error")
    return {"status": "ok"}


def name_analysis(filepath: str | Path) -> dict[str, Any]:
    """Analysis that returns filename."""
    return {"name": Path(filepath).name}


def analysis_with_config(
    filepath: str | Path,
    multiplier: int = 1,
    threshold: float = 0.5,
    mode: str = "fast",
) -> dict[str, Any]:
    """Analysis that uses config parameters."""
    size = len(Path(filepath).read_text())
    return {
        "result": size * multiplier,
        "threshold": threshold,
        "mode": mode,
    }


def analysis_returns_index(filepath: str | Path) -> dict[str, Any]:
    """Extract index from filename."""
    return {"index": int(Path(filepath).stem.split("_")[1])}


def analysis_returns_processed(filepath: str | Path) -> dict[str, Any]:
    """Returns processed flag."""
    return {"processed": True}


def analysis_returns_status_ok(filepath: str | Path) -> dict[str, Any]:
    """Returns status ok."""
    return {"status": "ok"}


def analysis_returns_int(filepath: str | Path) -> int:
    """Returns non-dict (int)."""
    return 42  # type: ignore[return-value]


def slow_analysis(filepath: str | Path) -> dict[str, Any]:
    """Slow analysis for timeout testing."""
    import time

    time.sleep(0.5)
    return {"status": "ok"}


def quick_analysis(filepath: str | Path) -> dict[str, Any]:
    """Quick analysis for timeout testing."""
    import time

    time.sleep(0.05)
    return {"status": "completed"}
