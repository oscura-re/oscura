"""Memory usage logging and progress tracking for Oscura operations.

This module provides detailed memory profiling logs for debugging and monitoring
memory usage during long-running operations.


Example:
    >>> from oscura.core.memory_progress import MemoryLogger
    >>> logger = MemoryLogger("analysis.log", format="csv")
    >>> with logger:
    ...     for i in range(1000):
    ...         # Perform work
    ...         logger.log_operation("fft", iteration=i)
    >>> stats = logger.get_summary()

References:
    psutil documentation for memory monitoring
"""

from __future__ import annotations

import csv
import json
import os
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TextIO

from oscura.utils.memory import get_available_memory, get_memory_pressure

if TYPE_CHECKING:
    from collections.abc import Iterator


@dataclass
class MemoryLogEntry:
    """Single memory log entry.


    Attributes:
        timestamp: Time of log entry (seconds since epoch).
        operation: Name of operation being performed.
        iteration: Iteration number (if applicable).
        memory_used: Process memory usage (bytes).
        memory_peak: Peak memory since start (bytes).
        memory_available: System available memory (bytes).
        memory_pressure: Memory pressure (0.0-1.0).
        eta_seconds: Estimated time to completion (seconds).
        message: Optional descriptive message.
    """

    timestamp: float
    operation: str
    iteration: int | None
    memory_used: int
    memory_peak: int
    memory_available: int
    memory_pressure: float
    eta_seconds: float
    message: str


class MemoryLogger:
    """Logger for detailed memory profiling during operations.


    Logs memory usage at each operation with timestamps and metadata.
    Supports CSV and JSON output formats.

    Args:
        log_file: Path to output log file.
        format: Output format ('csv' or 'json').
        auto_flush: Flush after each write (default: True).
        enable_console: Also print to console (default: False).

    Example:
        >>> logger = MemoryLogger("memory.csv", format="csv")
        >>> with logger:
        ...     for i in range(100):
        ...         # Do work
        ...         logger.log_operation("processing", iteration=i)
        >>> print(logger.get_summary())

    References:
        MEM-025: Memory Usage Logging
    """

    def __init__(
        self,
        log_file: str | Path,
        *,
        format: Literal["csv", "json"] = "csv",
        auto_flush: bool = True,
        enable_console: bool = False,
    ):
        """Initialize memory logger.

        Args:
            log_file: Path to log file.
            format: Output format ('csv' or 'json').
            auto_flush: Flush after each entry.
            enable_console: Print to console as well.
        """
        self.log_file = Path(log_file)
        self.format = format
        self.auto_flush = auto_flush
        self.enable_console = enable_console

        # State
        self._entries: list[MemoryLogEntry] = []
        self._file_handle: TextIO | None = None
        self._csv_writer: Any = None
        self._start_time = 0.0
        self._start_memory = 0
        self._peak_memory = 0

        # Create directory if needed
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def __enter__(self) -> MemoryLogger:
        """Enter context and initialize logging."""
        self._start_time = time.time()
        self._start_memory = self._get_process_memory()
        self._peak_memory = self._start_memory

        # Open log file
        self._file_handle = open(self.log_file, "w", newline="")

        # Initialize CSV writer if needed
        if self.format == "csv":
            self._csv_writer = csv.DictWriter(
                self._file_handle,
                fieldnames=[
                    "timestamp",
                    "operation",
                    "iteration",
                    "memory_used",
                    "memory_peak",
                    "memory_available",
                    "memory_pressure",
                    "eta_seconds",
                    "message",
                ],
            )
            self._csv_writer.writeheader()
            if self.auto_flush:
                self._file_handle.flush()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore[no-untyped-def]
        """Exit context and finalize logging."""
        # Note: exc_val and exc_tb intentionally unused but required for Python 3.11+ compatibility
        # Write summary for JSON format
        if self.format == "json" and self._file_handle:
            summary = {
                "entries": [asdict(entry) for entry in self._entries],
                "summary": self._get_summary_dict(),
            }
            json.dump(summary, self._file_handle, indent=2)

        # Close file
        if self._file_handle:
            self._file_handle.close()
            self._file_handle = None

    def log_operation(
        self,
        operation: str,
        *,
        iteration: int | None = None,
        eta_seconds: float = 0.0,
        message: str = "",
    ) -> None:
        """Log memory usage for an operation.


        Args:
            operation: Name of operation.
            iteration: Iteration number (optional).
            eta_seconds: Estimated time to completion.
            message: Optional descriptive message.

        Example:
            >>> logger.log_operation("fft", iteration=100, eta_seconds=5.2)

        References:
            MEM-025: Memory Usage Logging
        """
        # Get memory metrics
        memory_used = self._get_process_memory()
        memory_available = get_available_memory()
        memory_pressure = get_memory_pressure()

        # Update peak
        self._peak_memory = max(self._peak_memory, memory_used)

        # Create entry
        entry = MemoryLogEntry(
            timestamp=time.time(),
            operation=operation,
            iteration=iteration,
            memory_used=memory_used,
            memory_peak=self._peak_memory,
            memory_available=memory_available,
            memory_pressure=memory_pressure,
            eta_seconds=eta_seconds,
            message=message,
        )

        self._entries.append(entry)

        # Write to file
        if self._file_handle and self.format == "csv":
            self._csv_writer.writerow(asdict(entry))
            if self.auto_flush:
                self._file_handle.flush()

        # Console output
        if self.enable_console:
            print(self._format_entry(entry))

    def log_progress(
        self,
        operation: str,
        current: int,
        total: int,
        *,
        message: str = "",
    ) -> None:
        """Log memory usage with progress information.


        Convenience method that calculates ETA from progress.

        Args:
            operation: Name of operation.
            current: Current progress value.
            total: Total progress value.
            message: Optional message.

        Example:
            >>> for i in range(1000):
            ...     logger.log_progress("analysis", i + 1, 1000)

        References:
            MEM-024: Memory-Aware Progress Callback
            MEM-025: Memory Usage Logging
        """
        # Calculate ETA
        elapsed = time.time() - self._start_time
        eta = elapsed / current * (total - current) if current > 0 else 0.0

        self.log_operation(
            operation,
            iteration=current,
            eta_seconds=eta,
            message=message,
        )

    def get_summary(self) -> str:
        """Get human-readable summary of memory usage.

        Returns:
            Formatted summary string.

        Example:
            >>> logger = MemoryLogger("test.log")
            >>> with logger:
            ...     logger.log_operation("test")
            >>> print(logger.get_summary())

        References:
            MEM-025: Memory Usage Logging
        """
        summary = self._get_summary_dict()

        return (
            f"Memory Usage Summary:\n"
            f"  Entries: {summary['entry_count']}\n"
            f"  Duration: {summary['duration']:.2f}s\n"
            f"  Start Memory: {summary['start_memory'] / 1e9:.2f} GB\n"
            f"  Peak Memory: {summary['peak_memory'] / 1e9:.2f} GB\n"
            f"  Delta: {summary['memory_delta'] / 1e9:.2f} GB\n"
            f"  Min Available: {summary['min_available'] / 1e9:.2f} GB\n"
            f"  Max Pressure: {summary['max_pressure'] * 100:.1f}%\n"
        )

    def get_entries(self) -> list[MemoryLogEntry]:
        """Get all logged entries.

        Returns:
            List of memory log entries.

        References:
            MEM-025: Memory Usage Logging
        """
        return self._entries.copy()

    def _get_summary_dict(self) -> dict:  # type: ignore[type-arg]
        """Get summary statistics as dictionary."""
        if not self._entries:
            return {
                "entry_count": 0,
                "duration": 0.0,
                "start_memory": self._start_memory,
                "peak_memory": self._peak_memory,
                "memory_delta": 0,
                "min_available": 0,
                "max_pressure": 0.0,
            }

        duration = self._entries[-1].timestamp - self._entries[0].timestamp
        min_available = min(e.memory_available for e in self._entries)
        max_pressure = max(e.memory_pressure for e in self._entries)

        return {
            "entry_count": len(self._entries),
            "duration": duration,
            "start_memory": self._start_memory,
            "peak_memory": self._peak_memory,
            "memory_delta": self._peak_memory - self._start_memory,
            "min_available": min_available,
            "max_pressure": max_pressure,
        }

    def _format_entry(self, entry: MemoryLogEntry) -> str:
        """Format entry for console output."""
        elapsed = entry.timestamp - self._start_time
        return (
            f"[{elapsed:7.2f}s] {entry.operation:20s} | "
            f"Used: {entry.memory_used / 1e9:6.2f} GB | "
            f"Peak: {entry.memory_peak / 1e9:6.2f} GB | "
            f"Avail: {entry.memory_available / 1e9:6.2f} GB | "
            f"Pressure: {entry.memory_pressure * 100:5.1f}%"
            + (f" | {entry.message}" if entry.message else "")
        )

    def _get_process_memory(self) -> int:
        """Get current process memory usage in bytes."""
        try:
            import psutil

            process = psutil.Process()
            return process.memory_info().rss  # type: ignore[no-any-return]
        except ImportError:
            # Fallback: estimate from system memory
            from oscura.utils.memory import get_total_memory

            return get_total_memory() - get_available_memory()


@contextmanager
def log_memory(
    log_file: str | Path,
    *,
    format: Literal["csv", "json"] = "csv",
    enable_console: bool = False,
) -> Iterator[MemoryLogger]:
    """Context manager for memory logging.


    Convenience function that wraps MemoryLogger.

    Args:
        log_file: Path to log file.
        format: Output format ('csv' or 'json').
        enable_console: Print to console.

    Yields:
        MemoryLogger instance.

    Example:
        >>> with log_memory("analysis.csv") as logger:
        ...     for i in range(1000):
        ...         # Do work
        ...         logger.log_progress("processing", i + 1, 1000)

    References:
        MEM-025: Memory Usage Logging
    """
    logger = MemoryLogger(log_file, format=format, enable_console=enable_console)
    with logger:
        yield logger


def create_progress_callback_with_logging(
    logger: MemoryLogger,
    operation: str,
) -> callable:  # type: ignore[valid-type]
    """Create progress callback that logs to MemoryLogger.


    Returns a callback function compatible with progress tracking APIs
    that automatically logs memory usage.

    Args:
        logger: MemoryLogger instance to log to.
        operation: Name of operation.

    Returns:
        Progress callback function.

    Example:
        >>> logger = MemoryLogger("test.log")
        >>> callback = create_progress_callback_with_logging(logger, "fft")
        >>> callback(50, 100, "Processing")

    References:
        MEM-024: Memory-Aware Progress Callback
        MEM-025: Memory Usage Logging
    """

    def callback(current: int, total: int, message: str) -> None:
        """Progress callback with memory logging."""
        logger.log_progress(operation, current, total, message=message)

    return callback


def enable_memory_logging_from_cli(
    log_file: str | Path | None = None,
) -> MemoryLogger | None:
    """Enable memory logging from CLI flag.


    Checks for --log-memory CLI flag and returns logger if enabled.

    Args:
        log_file: Override log file path (default: auto-generate).

    Returns:
        MemoryLogger if enabled, None otherwise.

    Example:
        >>> logger = enable_memory_logging_from_cli()
        >>> if logger:
        ...     with logger:
        ...         # Operations are logged
        ...         logger.log_operation("test")

    References:
        MEM-025: Memory Usage Logging
    """
    # Check environment variable
    if os.environ.get("TK_LOG_MEMORY", "").lower() not in ("1", "true", "yes"):
        return None

    # Generate log file name if not provided
    if log_file is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_file = Path(f"oscura_memory_{timestamp}.csv")

    return MemoryLogger(log_file, format="csv", enable_console=False)


__all__ = [
    "MemoryLogEntry",
    "MemoryLogger",
    "create_progress_callback_with_logging",
    "enable_memory_logging_from_cli",
    "log_memory",
]
