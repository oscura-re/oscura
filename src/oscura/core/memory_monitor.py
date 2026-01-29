"""Memory monitoring and OOM prevention for Oscura.

This module provides runtime memory monitoring to prevent out-of-memory crashes
and gracefully handle memory exhaustion scenarios.


Example:
    >>> from oscura.core.memory_monitor import monitor_memory, MemoryMonitor
    >>> with MemoryMonitor('spectrogram', max_memory="4GB") as monitor:
    ...     for i in range(1000):
    ...         # Perform work
    ...         monitor.check(i)  # Check memory periodically
    ...     stats = monitor.get_stats()
    >>> print(f"Peak memory: {stats['peak'] / 1e9:.2f} GB")

References:
    psutil documentation for memory monitoring
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from oscura.core.config.memory import get_memory_config
from oscura.utils.memory import get_available_memory, get_max_memory

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


@dataclass
class MemorySnapshot:
    """Snapshot of memory state at a point in time.

    Attributes:
        timestamp: Time of snapshot (seconds since epoch).
        available: Available system memory (bytes).
        process_rss: Process resident set size (bytes).
        process_vms: Process virtual memory size (bytes).
        pressure: Memory pressure (0.0-1.0).
    """

    timestamp: float
    available: int
    process_rss: int
    process_vms: int
    pressure: float


class MemoryMonitor:
    """Context manager for monitoring memory usage during operations.


    Monitors memory usage during long-running operations and aborts
    before system crashes if memory pressure becomes critical.

    Attributes:
        operation: Name of the operation being monitored.
        max_memory: Maximum allowed memory (None = use global config).
        check_interval: How often to check memory (number of iterations).
        abort_on_critical: Whether to abort when critical threshold reached.

    Example:
        >>> with MemoryMonitor('fft', max_memory="2GB") as monitor:
        ...     result = compute_fft(data)
        ...     stats = monitor.get_stats()
        >>> print(f"Peak: {stats['peak'] / 1e6:.1f} MB")

    Raises:
        MemoryError: If memory usage approaches critical limit.
    """

    def __init__(
        self,
        operation: str,
        *,
        max_memory: int | str | None = None,
        check_interval: int = 100,
        abort_on_critical: bool = True,
    ):
        """Initialize memory monitor.

        Args:
            operation: Name of operation being monitored.
            max_memory: Maximum memory limit (bytes, string like "4GB", or None for auto).
            check_interval: Check memory every N iterations.
            abort_on_critical: Abort operation if critical threshold reached.
        """
        self.operation = operation
        self.check_interval = check_interval
        self.abort_on_critical = abort_on_critical

        # Parse max_memory
        if max_memory is None:
            self.max_memory = get_max_memory()
        elif isinstance(max_memory, str):
            from oscura.core.config.memory import _parse_memory_string

            self.max_memory = _parse_memory_string(max_memory)
        else:
            self.max_memory = int(max_memory)

        # State
        self.start_memory = 0
        self.peak_memory = 0
        self.current_memory = 0
        self._iteration = 0
        self._snapshots: list[MemorySnapshot] = []
        self._start_time = 0.0

    def __enter__(self) -> MemoryMonitor:
        """Enter context and record starting memory."""
        self.start_memory = self._get_process_memory()
        self.peak_memory = self.start_memory
        self.current_memory = self.start_memory
        self._start_time = time.time()

        # Take initial snapshot
        self._take_snapshot()

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Exit context and finalize monitoring."""
        # Note: exc_val and exc_tb intentionally unused but required for Python 3.11+ compatibility
        # Take final snapshot
        self._take_snapshot()

    def check(self, iteration: int | None = None) -> None:
        """Check memory usage and raise error if limit approached.


        Args:
            iteration: Current iteration number (for periodic checking).

        Raises:
            MemoryError: If memory usage exceeds critical threshold.

        Example:
            >>> with MemoryMonitor('operation') as monitor:
            ...     for i in range(10000):
            ...         # Do work
            ...         monitor.check(i)  # Check every 100 iterations
        """
        self._iteration += 1

        # Only check periodically to reduce overhead
        if iteration is not None and iteration % self.check_interval != 0:
            return

        self.current_memory = self._get_process_memory()
        self.peak_memory = max(self.peak_memory, self.current_memory)

        # Check against available memory and thresholds
        available = get_available_memory()
        config = get_memory_config()

        # Calculate pressure
        pressure = 1.0 - (available / self.max_memory) if self.max_memory > 0 else 0.0

        # Take snapshot if significant time passed
        if self._snapshots and (time.time() - self._snapshots[-1].timestamp) > 1.0:
            self._take_snapshot()

        # Check critical threshold
        if self.abort_on_critical and pressure >= config.critical_threshold:
            raise MemoryError(
                f"Critical memory pressure during {self.operation}. "
                f"Available: {available / 1e9:.2f} GB, "
                f"Pressure: {pressure * 100:.1f}%, "
                f"Limit: {self.max_memory / 1e9:.2f} GB. "
                f"Operation aborted to prevent system crash. "
                f"Suggestion: Reduce dataset size, increase memory limit, "
                f"or use chunked processing."
            )

    def get_stats(self) -> dict[str, int | float]:
        """Get memory statistics for this monitoring session.

        Returns:
            Dictionary with memory statistics including:
            - start: Starting memory (bytes)
            - current: Current memory (bytes)
            - peak: Peak memory usage (bytes)
            - delta: Memory increase since start (bytes)
            - duration: Monitoring duration (seconds)

        Example:
            >>> with MemoryMonitor('operation') as monitor:
            ...     # ... do work ...
            ...     stats = monitor.get_stats()
            >>> print(f"Peak: {stats['peak'] / 1e6:.1f} MB")
        """
        duration = time.time() - self._start_time if self._start_time > 0 else 0.0

        return {
            "start": self.start_memory,
            "current": self.current_memory,
            "peak": self.peak_memory,
            "delta": self.peak_memory - self.start_memory,
            "duration": duration,
        }

    def get_snapshots(self) -> list[MemorySnapshot]:
        """Get all memory snapshots taken during monitoring.

        Returns:
            List of MemorySnapshot objects.

        Example:
            >>> with MemoryMonitor('operation') as monitor:
            ...     # ... work ...
            ...     pass
            >>> for snap in monitor.get_snapshots():
            ...     print(f"t={snap.timestamp:.1f}s: {snap.available/1e9:.2f} GB available")
        """
        return self._snapshots.copy()

    def _get_process_memory(self) -> int:
        """Get current process memory usage in bytes.

        Returns:
            Resident set size (RSS) in bytes.
        """
        try:
            import psutil

            process = psutil.Process()
            return process.memory_info().rss  # type: ignore[no-any-return]
        except ImportError:
            # Fallback: estimate from system memory
            from oscura.utils.memory import get_total_memory

            return get_total_memory() - get_available_memory()

    def _take_snapshot(self) -> None:
        """Take a snapshot of current memory state."""
        try:
            import psutil

            process = psutil.Process()
            mem_info = process.memory_info()
            available = get_available_memory()

            from oscura.utils.memory import get_memory_pressure

            pressure = get_memory_pressure()

            snapshot = MemorySnapshot(
                timestamp=time.time(),
                available=available,
                process_rss=mem_info.rss,
                process_vms=mem_info.vms,
                pressure=pressure,
            )
            self._snapshots.append(snapshot)
        except ImportError:
            # Skip snapshots if psutil not available
            pass


@contextmanager
def monitor_memory(
    operation: str,
    *,
    max_memory: int | str | None = None,
    check_interval: int = 100,
) -> Iterator[MemoryMonitor]:
    """Context manager for monitoring memory usage.


    Convenience function that wraps MemoryMonitor.

    Args:
        operation: Name of operation being monitored.
        max_memory: Maximum memory limit.
        check_interval: Check memory every N iterations.

    Yields:
        MemoryMonitor instance.

    Example:
        >>> with monitor_memory('spectrogram', max_memory="4GB") as mon:
        ...     for i in range(1000):
        ...         # Work
        ...         mon.check(i)
    """
    monitor = MemoryMonitor(
        operation,
        max_memory=max_memory,
        check_interval=check_interval,
    )
    with monitor:
        yield monitor


@dataclass
class ProgressWithMemory:
    """Progress information with memory metrics.


    Attributes:
        current: Current progress value.
        total: Total progress value.
        eta_seconds: Estimated time to completion (seconds).
        memory_used: Current memory usage (bytes).
        memory_peak: Peak memory usage (bytes).
        memory_available: Available system memory (bytes).
        operation: Name of operation.
    """

    current: int
    total: int
    eta_seconds: float
    memory_used: int
    memory_peak: int
    memory_available: int
    operation: str

    @property
    def percent(self) -> float:
        """Progress percentage (0.0-100.0)."""
        if self.total == 0:
            return 100.0
        return (self.current / self.total) * 100.0

    @property
    def memory_pressure(self) -> float:
        """Memory pressure (0.0-1.0)."""
        from oscura.utils.memory import get_memory_pressure

        return get_memory_pressure()

    def format_progress(self) -> str:
        """Format progress as human-readable string.

        Returns:
            Formatted progress string with memory info.

        Example:
            >>> progress = ProgressWithMemory(42, 100, 5.0, 1.2e9, 2.1e9, 6e9, "fft")
            >>> print(progress.format_progress())
            42.0% | 1.20 GB used | 2.10 GB peak | 6.00 GB avail | ETA 5s
        """
        return (
            f"{self.percent:.1f}% | "
            f"{self.memory_used / 1e9:.2f} GB used | "
            f"{self.memory_peak / 1e9:.2f} GB peak | "
            f"{self.memory_available / 1e9:.2f} GB avail | "
            f"ETA {self.eta_seconds:.0f}s"
        )


class ProgressMonitor:
    """Combined progress and memory monitoring.


    Tracks both operation progress and memory usage, providing
    unified progress updates with memory metrics.

    Example:
        >>> monitor = ProgressMonitor('spectrogram', total=1000)
        >>> for i in range(1000):
        ...     # Work
        ...     monitor.update(i)
        ...     if i % 100 == 0:
        ...         progress = monitor.get_progress()
        ...         print(progress.format_progress())
    """

    def __init__(
        self,
        operation: str,
        total: int,
        *,
        callback: Callable[[ProgressWithMemory], None] | None = None,
        update_interval: int = 1,
    ):
        """Initialize progress monitor.

        Args:
            operation: Name of operation.
            total: Total number of items to process.
            callback: Optional callback function called on each update.
            update_interval: Call callback every N updates.
        """
        self.operation = operation
        self.total = total
        self.callback = callback
        self.update_interval = update_interval
        self.current = 0
        self._start_time = time.time()
        self._memory_monitor = MemoryMonitor(operation, check_interval=1)
        self._update_count = 0

    def update(self, current: int | None = None) -> None:
        """Update progress.

        Args:
            current: Current progress value (if None, increments by 1).
        """
        if current is not None:
            self.current = current
        else:
            self.current += 1

        self._update_count += 1

        # Check memory
        self._memory_monitor.check(self._update_count)

        # Call callback if interval reached
        if self.callback and self._update_count % self.update_interval == 0:
            progress = self.get_progress()
            self.callback(progress)

    def get_progress(self) -> ProgressWithMemory:
        """Get current progress with memory metrics.

        Returns:
            ProgressWithMemory instance.
        """
        elapsed = time.time() - self._start_time
        eta = elapsed / self.current * (self.total - self.current) if self.current > 0 else 0.0

        stats = self._memory_monitor.get_stats()

        return ProgressWithMemory(
            current=self.current,
            total=self.total,
            eta_seconds=eta,
            memory_used=stats["current"],  # type: ignore[arg-type]
            memory_peak=stats["peak"],  # type: ignore[arg-type]
            memory_available=get_available_memory(),
            operation=self.operation,
        )


__all__ = [
    "MemoryMonitor",
    "MemorySnapshot",
    "ProgressMonitor",
    "ProgressWithMemory",
    "monitor_memory",
]
