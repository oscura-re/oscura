"""Performance timing and monitoring infrastructure.

This module provides automatic performance timing for operations,
metric collection, and performance summary reporting.


Example:
    >>> from oscura.core.performance import timed, get_performance_summary
    >>> @timed(threshold=0.1)
    ... def slow_operation(data):
    ...     result = expensive_computation(data)
    ...     return result
    >>> summary = get_performance_summary()

References:
    - Python time.perf_counter() for high-resolution timing
    - Performance monitoring best practices
"""

from __future__ import annotations

import functools
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class PerformanceRecord:
    """Record of a timed operation.

    Attributes:
        operation: Name of the operation (typically function name).
        duration: Duration in seconds.
        timestamp: ISO 8601 timestamp of when operation completed.
        metadata: Optional metadata about the operation.

    References:
        LOG-006: Automatic Performance Timing
    """

    operation: str
    duration: float
    timestamp: str
    metadata: dict[str, Any] = field(default_factory=dict)


class PerformanceCollector:
    """Collects and aggregates performance timing data.

    Thread-safe collector for recording and analyzing operation timings.
    Provides summary statistics per operation type.

    Example:
        >>> collector = PerformanceCollector()
        >>> collector.record("fft_computation", 1.23, samples=1000000)
        >>> collector.record("fft_computation", 1.45, samples=2000000)
        >>> summary = collector.get_summary()
        >>> print(summary["fft_computation"]["mean"])
        1.34

    References:
        LOG-006: Automatic Performance Timing
    """

    def __init__(self) -> None:
        """Initialize the performance collector."""
        self.records: list[PerformanceRecord] = []

    def record(
        self,
        operation: str,
        duration: float,
        **metadata: Any,
    ) -> None:
        """Record a performance measurement.

        Args:
            operation: Name of the operation being timed.
            duration: Duration in seconds (use time.perf_counter()).
            **metadata: Additional metadata about the operation.

        Example:
            >>> collector.record("load_trace", 0.5, file_size_mb=100)
            >>> collector.record("compute_fft", 2.3, samples=10000000)

        References:
            LOG-006: Automatic Performance Timing
        """
        self.records.append(
            PerformanceRecord(
                operation=operation,
                duration=duration,
                timestamp=datetime.now(UTC).isoformat(),
                metadata=metadata,
            )
        )

    def get_summary(self) -> dict[str, dict[str, float]]:
        """Get summary statistics per operation.

        Returns:
            Dictionary mapping operation names to statistics:
            - count: Number of times operation was performed
            - mean: Mean duration in seconds
            - std: Standard deviation of duration
            - min: Minimum duration
            - max: Maximum duration
            - total: Total time spent in operation

        Example:
            >>> summary = collector.get_summary()
            >>> fft_stats = summary["fft_computation"]
            >>> print(f"FFT avg: {fft_stats['mean']:.3f}s")

        References:
            LOG-006: Automatic Performance Timing
        """
        by_op: dict[str, list[float]] = defaultdict(list)
        for record in self.records:
            by_op[record.operation].append(record.duration)

        summary: dict[str, dict[str, float]] = {}
        for op, durations in by_op.items():
            arr: NDArray[np.float64] = np.array(durations)
            summary[op] = {
                "count": float(len(arr)),
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "total": float(np.sum(arr)),
            }
        return summary

    def clear(self) -> None:
        """Clear all performance records.

        Useful for resetting measurements between test runs or
        analysis sessions.

        Example:
            >>> collector.clear()
        """
        self.records.clear()

    def get_records(
        self,
        operation: str | None = None,
        since: datetime | None = None,
    ) -> list[PerformanceRecord]:
        """Get performance records with optional filtering.

        Args:
            operation: Filter by operation name, or None for all.
            since: Filter records after this timestamp, or None for all.

        Returns:
            List of matching performance records.

        Example:
            >>> from datetime import datetime, timezone, timedelta
            >>> one_hour_ago = datetime.now(timezone.utc) - timedelta(hours=1)
            >>> recent = collector.get_records(since=one_hour_ago)
        """
        records = self.records

        if operation is not None:
            records = [r for r in records if r.operation == operation]

        if since is not None:
            since_iso = since.isoformat()
            records = [r for r in records if r.timestamp >= since_iso]

        return records


# Global performance collector
_global_collector = PerformanceCollector()


F = TypeVar("F", bound=Callable[..., Any])


def timed(
    threshold: float | None = None,
    log_level: str = "DEBUG",
    collect: bool = True,
) -> Callable[[F], F]:
    """Decorator to time and log function execution.

    Automatically times function execution using time.perf_counter()
    and optionally logs if duration exceeds threshold.

    Args:
        threshold: Only log if duration exceeds this (seconds). None logs all.
        log_level: Log level for timing messages (DEBUG, INFO, WARNING, etc.).
        collect: Whether to collect timing in global PerformanceCollector.

    Returns:
        Decorated function.

    Example:
        >>> @timed(threshold=0.1)
        ... def slow_function(data):
        ...     time.sleep(0.2)
        ...     return process(data)

        >>> @timed(log_level="INFO")
        ... def important_operation():
        ...     return compute_result()

    References:
        LOG-006: Automatic Performance Timing
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                duration = time.perf_counter() - start

                # Record in global collector
                if collect:
                    _global_collector.record(func.__qualname__, duration)

                # Log if threshold met
                if threshold is None or duration >= threshold:
                    # Lazy import to avoid circular dependency
                    from oscura.core.logging import get_logger

                    logger = get_logger(func.__module__ or "oscura")
                    log_func = getattr(logger, log_level.lower())
                    log_func(
                        f"Function {func.__qualname__} completed",
                        extra={"duration_seconds": round(duration, 6)},
                    )

        return wrapper  # type: ignore[return-value]

    return decorator


def get_performance_summary() -> dict[str, dict[str, float]]:
    """Get summary of collected performance data from global collector.

    Returns:
        Dictionary of performance statistics per operation.

    Example:
        >>> summary = get_performance_summary()
        >>> for op, stats in summary.items():
        ...     print(f"{op}: {stats['mean']:.3f}s avg, {stats['count']:.0f} calls")

    References:
        LOG-006: Automatic Performance Timing
    """
    return _global_collector.get_summary()


def clear_performance_data() -> None:
    """Clear all collected performance data from global collector.

    Example:
        >>> clear_performance_data()
    """
    _global_collector.clear()


def get_performance_records(
    operation: str | None = None,
    since: datetime | None = None,
) -> list[PerformanceRecord]:
    """Get performance records from global collector.

    Args:
        operation: Filter by operation name, or None for all.
        since: Filter records after this timestamp, or None for all.

    Returns:
        List of matching performance records.

    Example:
        >>> records = get_performance_records(operation="fft_computation")
        >>> for record in records:
        ...     print(f"{record.timestamp}: {record.duration:.3f}s")
    """
    return _global_collector.get_records(operation=operation, since=since)


class PerformanceContext:
    """Context manager for timing a block of code.

    Example:
        >>> with PerformanceContext("data_loading") as ctx:
        ...     data = load_large_file()
        >>> print(f"Loading took {ctx.duration:.3f}s")

    References:
        LOG-006: Automatic Performance Timing
    """

    def __init__(
        self,
        operation: str,
        log_threshold: float | None = None,
        collect: bool = True,
    ):
        """Initialize performance context.

        Args:
            operation: Name of the operation being timed.
            log_threshold: Log if duration exceeds threshold (seconds).
            collect: Whether to collect in global collector.
        """
        self.operation = operation
        self.log_threshold = log_threshold
        self.collect = collect
        self.start_time: float = 0.0
        self.duration: float = 0.0

    def __enter__(self) -> PerformanceContext:
        """Start timing."""
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        """Stop timing and record."""
        self.duration = time.perf_counter() - self.start_time

        if self.collect:
            _global_collector.record(self.operation, self.duration)

        if self.log_threshold is None or self.duration >= self.log_threshold:
            from oscura.core.logging import get_logger

            logger = get_logger("oscura.performance")
            logger.debug(
                f"Operation {self.operation} completed",
                extra={"duration_seconds": round(self.duration, 6)},
            )
