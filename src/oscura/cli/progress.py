"""Progress reporting utilities for CLI commands.

Provides progress bars, status messages, and ETA calculations for long-running
operations.


Example:
    >>> progress = ProgressReporter(stages=3)
    >>> progress.start_stage("Loading data")
    >>> # ... work ...
    >>> progress.complete_stage()
"""

from __future__ import annotations

import sys
import time
from typing import Any

try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


class ProgressReporter:
    """Progress reporter for multi-stage operations.

    Displays progress bars (if tqdm available) or status messages for
    long-running CLI operations.

    Args:
        quiet: If True, suppress all output.
        stages: Number of stages in operation.
        use_tqdm: Force tqdm usage (default: auto-detect).

    Example:
        >>> reporter = ProgressReporter(stages=3)
        >>> reporter.start_stage("Stage 1")
        >>> time.sleep(1)
        >>> reporter.complete_stage()
        >>> reporter.start_stage("Stage 2")
        >>> time.sleep(1)
        >>> reporter.complete_stage()
        >>> reporter.finish()
    """

    def __init__(
        self,
        quiet: bool = False,
        stages: int = 1,
        use_tqdm: bool | None = None,
    ) -> None:
        """Initialize progress reporter.

        Args:
            quiet: Suppress all output.
            stages: Total number of stages.
            use_tqdm: Force tqdm usage (None = auto-detect).
        """
        self.quiet = quiet
        self.total_stages = stages
        self.current_stage = 0
        self.stage_name = ""
        self.start_time = time.time()
        self.stage_start = time.time()

        # Determine if we can use tqdm
        if use_tqdm is None:
            self.use_tqdm = TQDM_AVAILABLE and not quiet and sys.stdout.isatty()
        else:
            self.use_tqdm = use_tqdm and TQDM_AVAILABLE and not quiet

        self.pbar: Any = None
        if self.use_tqdm:
            self.pbar = tqdm(total=stages, desc="Progress", unit="stage")

    def start_stage(self, name: str) -> None:
        """Start a new stage.

        Args:
            name: Stage name/description.
        """
        self.current_stage += 1
        self.stage_name = name
        self.stage_start = time.time()

        if not self.quiet:
            if self.use_tqdm and self.pbar:
                self.pbar.set_description(f"{name}")
            else:
                timestamp = time.strftime("%H:%M:%S")
                print(
                    f"[{timestamp}] [{self.current_stage}/{self.total_stages}] {name}...",
                    file=sys.stderr,
                )

    def complete_stage(self) -> None:
        """Mark current stage as complete."""
        stage_duration = time.time() - self.stage_start

        if not self.quiet:
            if self.use_tqdm and self.pbar:
                self.pbar.update(1)
            else:
                timestamp = time.strftime("%H:%M:%S")
                print(
                    f"[{timestamp}] {self.stage_name} completed ({stage_duration:.1f}s)",
                    file=sys.stderr,
                )

    def update_progress(self, current: int, total: int, message: str = "") -> None:
        """Update progress within current stage.

        Args:
            current: Current progress value.
            total: Total progress value.
            message: Optional progress message.
        """
        if not self.quiet and not self.use_tqdm:
            pct = (current / total * 100) if total > 0 else 0
            msg = f"{message} " if message else ""
            print(f"  {msg}{current}/{total} ({pct:.1f}%)", file=sys.stderr, end="\r")

    def finish(self) -> None:
        """Finish progress reporting."""
        total_duration = time.time() - self.start_time

        if not self.quiet:
            if self.use_tqdm and self.pbar:
                self.pbar.close()
            else:
                timestamp = time.strftime("%H:%M:%S")
                print(
                    f"[{timestamp}] All stages complete ({total_duration:.1f}s total)",
                    file=sys.stderr,
                )

    def __enter__(self) -> ProgressReporter:
        """Context manager entry."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Context manager exit."""
        self.finish()
