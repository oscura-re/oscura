"""Performance profiling for signal analysis operations.

This module provides profiling utilities for measuring and analyzing
performance of signal processing operations.
"""

from __future__ import annotations

import functools
import logging
import statistics
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

logger = logging.getLogger(__name__)

__all__ = [
    "OperationProfile",
    "ProfileReport",
    "Profiler",
    "profile",
]


@dataclass
class OperationProfile:
    """Profile data for a single operation.

    Attributes:
        name: Operation name
        calls: Number of calls
        total_time: Total time in seconds
        min_time: Minimum time
        max_time: Maximum time
        times: List of individual times
        memory_peak: Peak memory usage (bytes)
        input_size: Input data size

    References:
        API-012: Performance Profiling API
    """

    name: str
    calls: int = 0
    total_time: float = 0.0
    min_time: float = float("inf")
    max_time: float = 0.0
    times: list[float] = field(default_factory=list)
    memory_peak: int = 0
    input_size: int = 0

    @property
    def mean_time(self) -> float:
        """Average time per call."""
        return self.total_time / self.calls if self.calls > 0 else 0.0

    @property
    def std_time(self) -> float:
        """Standard deviation of times."""
        if len(self.times) < 2:
            return 0.0
        return statistics.stdev(self.times)

    @property
    def throughput(self) -> float:
        """Throughput in items per second."""
        if self.total_time > 0 and self.input_size > 0:
            return (self.input_size * self.calls) / self.total_time
        return 0.0

    def record(self, elapsed: float, size: int = 0) -> None:
        """Record a timing.

        Args:
            elapsed: Elapsed time in seconds
            size: Input size
        """
        self.calls += 1
        self.total_time += elapsed
        self.min_time = min(self.min_time, elapsed)
        self.max_time = max(self.max_time, elapsed)
        self.times.append(elapsed)
        if size > 0:
            self.input_size = size

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "calls": self.calls,
            "total_time": self.total_time,
            "mean_time": self.mean_time,
            "min_time": self.min_time if self.min_time != float("inf") else 0,
            "max_time": self.max_time,
            "std_time": self.std_time,
            "throughput": self.throughput,
        }


@dataclass
class ProfileReport:
    """Complete profiling report.

    Attributes:
        profiles: Dictionary of operation profiles
        start_time: Report start time
        end_time: Report end time
        total_operations: Total number of operations

    References:
        API-012: Performance Profiling API
    """

    profiles: dict[str, OperationProfile] = field(default_factory=dict)
    start_time: float = 0.0
    end_time: float = 0.0
    total_operations: int = 0

    @property
    def total_time(self) -> float:
        """Total profiled time."""
        return sum(p.total_time for p in self.profiles.values())

    @property
    def wall_time(self) -> float:
        """Wall clock time."""
        return self.end_time - self.start_time if self.end_time > 0 else 0.0

    def get_slowest(self, n: int = 5) -> list[OperationProfile]:
        """Get slowest operations.

        Args:
            n: Number of operations

        Returns:
            List of slowest operation profiles
        """
        sorted_profiles = sorted(self.profiles.values(), key=lambda p: p.total_time, reverse=True)
        return sorted_profiles[:n]

    def get_most_called(self, n: int = 5) -> list[OperationProfile]:
        """Get most frequently called operations.

        Args:
            n: Number of operations

        Returns:
            List of most called operation profiles
        """
        sorted_profiles = sorted(self.profiles.values(), key=lambda p: p.calls, reverse=True)
        return sorted_profiles[:n]

    def summary(self) -> str:
        """Generate text summary.

        Returns:
            Summary string
        """
        lines = [
            "Performance Profile Report",
            "=" * 50,
            f"Total operations: {self.total_operations}",
            f"Total profiled time: {self.total_time:.4f}s",
            f"Wall clock time: {self.wall_time:.4f}s",
            "",
            "Slowest Operations:",
            "-" * 30,
        ]

        for profile in self.get_slowest():
            lines.append(
                f"  {profile.name}: "
                f"{profile.total_time:.4f}s ({profile.calls} calls, "
                f"{profile.mean_time * 1000:.2f}ms avg)"
            )

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_time": self.total_time,
            "wall_time": self.wall_time,
            "total_operations": self.total_operations,
            "profiles": {name: profile.to_dict() for name, profile in self.profiles.items()},
        }


class Profiler:
    """Performance profiler for signal analysis operations.

    Tracks timing and performance metrics for operations.

    Example:
        >>> profiler = Profiler()
        >>> with profiler.profile("fft"):
        ...     result = np.fft.fft(data)
        >>> report = profiler.report()
        >>> print(report.summary())

    References:
        API-012: Performance Profiling API
    """

    _instance: Profiler | None = None

    def __init__(self) -> None:
        """Initialize profiler."""
        self._profiles: dict[str, OperationProfile] = {}
        self._start_time: float = 0.0
        self._enabled: bool = True
        self._stack: list[str] = []

    @classmethod
    def get_instance(cls) -> Profiler:
        """Get global profiler instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def enable(self) -> None:
        """Enable profiling."""
        self._enabled = True

    def disable(self) -> None:
        """Disable profiling."""
        self._enabled = False

    def reset(self) -> None:
        """Reset all profiles."""
        self._profiles.clear()
        self._start_time = 0.0

    @contextmanager
    def profile(self, name: str, input_size: int = 0) -> Iterator[None]:
        """Context manager for profiling code block.

        Args:
            name: Operation name
            input_size: Input data size

        Yields:
            None

        Example:
            >>> with profiler.profile("fft"):
            ...     result = compute_fft(data)
        """
        if not self._enabled:
            yield
            return

        if self._start_time == 0:
            self._start_time = time.perf_counter()

        if name not in self._profiles:
            self._profiles[name] = OperationProfile(name)

        self._stack.append(name)
        start = time.perf_counter()

        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self._profiles[name].record(elapsed, input_size)
            self._stack.pop()

    def record(self, name: str, elapsed: float, input_size: int = 0) -> None:
        """Manually record a timing.

        Args:
            name: Operation name
            elapsed: Elapsed time
            input_size: Input size
        """
        if not self._enabled:
            return

        if name not in self._profiles:
            self._profiles[name] = OperationProfile(name)

        self._profiles[name].record(elapsed, input_size)

    def get_profile(self, name: str) -> OperationProfile | None:
        """Get profile for operation.

        Args:
            name: Operation name

        Returns:
            Operation profile or None
        """
        return self._profiles.get(name)

    def report(self) -> ProfileReport:
        """Generate profiling report.

        Returns:
            Profile report
        """
        return ProfileReport(
            profiles=self._profiles.copy(),
            start_time=self._start_time,
            end_time=time.perf_counter(),
            total_operations=sum(p.calls for p in self._profiles.values()),
        )


def profile(name: str | None = None, input_size_arg: str | None = None) -> Callable:  # type: ignore[type-arg]
    """Decorator for profiling functions.

    Args:
        name: Profile name (defaults to function name)
        input_size_arg: Argument name for input size

    Returns:
        Decorated function

    Example:
        >>> @profile()
        >>> def compute_fft(data, nfft=None):
        ...     return np.fft.fft(data, n=nfft)

    References:
        API-012: Performance Profiling API
    """

    def decorator(func: Callable) -> Callable:  # type: ignore[type-arg]
        profile_name = name or func.__name__

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            profiler = Profiler.get_instance()

            # Determine input size
            input_size = 0
            if input_size_arg:
                if input_size_arg in kwargs:
                    data = kwargs[input_size_arg]
                elif args:
                    data = args[0]
                else:
                    data = None

                if hasattr(data, "__len__"):
                    input_size = len(data)

            with profiler.profile(profile_name, input_size):
                return func(*args, **kwargs)

        return wrapper

    return decorator


# Convenience functions
def get_profiler() -> Profiler:
    """Get global profiler instance.

    Returns:
        Global Profiler instance

    References:
        API-012: Performance Profiling API
    """
    return Profiler.get_instance()


def enable_profiling() -> None:
    """Enable global profiling."""
    get_profiler().enable()


def disable_profiling() -> None:
    """Disable global profiling."""
    get_profiler().disable()


def reset_profiling() -> None:
    """Reset global profiler."""
    get_profiler().reset()


def get_profile_report() -> ProfileReport:
    """Get global profile report.

    Returns:
        Profile report
    """
    return get_profiler().report()
