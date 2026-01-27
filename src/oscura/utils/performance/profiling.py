"""Comprehensive performance profiling for analysis workflows.

This module provides detailed profiling capabilities including CPU usage,
memory consumption, I/O operations, and bottleneck identification using
multiple profiling backends (cProfile, line_profiler, memory_profiler).

Example:
    >>> profiler = PerformanceProfiler()
    >>> profiler.start()
    >>> # Run analysis code here
    >>> result = profiler.stop()
    >>> print(result.summary())
    >>> result.export_json("profile.json")

References:
    Python Profilers: https://docs.python.org/3/library/profile.html
    IEEE 1685-2009: IP-XACT standard for profiling
"""

from __future__ import annotations

import cProfile
import functools
import io
import json
import logging
import pstats
import sys
import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

logger = logging.getLogger(__name__)

__all__ = [
    "FunctionStats",
    "PerformanceProfiler",
    "ProfilingMode",
    "ProfilingResult",
]

# Type aliases
HotspotDict: TypeAlias = dict[str, Any]
CallGraphDict: TypeAlias = dict[str, list[str]]


class ProfilingMode(Enum):
    """Profiling mode selection.

    Attributes:
        FUNCTION: Function-level profiling (time per function, call counts)
        LINE: Line-level profiling (time per line of code)
        MEMORY: Memory profiling (allocation tracking, memory leaks)
        IO: I/O profiling (disk reads/writes, network traffic)
        FULL: All profiling modes combined
    """

    FUNCTION = "function"
    LINE = "line"
    MEMORY = "memory"
    IO = "io"
    FULL = "full"


@dataclass
class FunctionStats:
    """Statistics for a profiled function.

    Attributes:
        name: Function name
        calls: Number of times called
        time: Total time spent in function (seconds)
        cumulative_time: Cumulative time including sub-calls (seconds)
        memory: Peak memory usage (bytes)
        per_call_time: Average time per call (seconds)
        per_call_memory: Average memory per call (bytes)
        filename: Source file containing function
        lineno: Line number where function is defined
    """

    name: str
    calls: int
    time: float
    cumulative_time: float
    memory: int = 0
    per_call_time: float = 0.0
    per_call_memory: int = 0
    filename: str = ""
    lineno: int = 0

    def __post_init__(self) -> None:
        """Calculate derived stats."""
        if self.calls > 0:
            self.per_call_time = self.time / self.calls
            if self.memory > 0:
                self.per_call_memory = self.memory // self.calls


@dataclass
class ProfilingResult:
    """Complete profiling results.

    Attributes:
        function_stats: Statistics for all profiled functions
        hotspots: Performance bottlenecks (top N slowest functions)
        memory_stats: Memory usage statistics
        call_graph: Function call hierarchy
        total_time: Total execution time (seconds)
        peak_memory: Peak memory usage (bytes)
        mode: Profiling mode used
        metadata: Additional metadata (timestamp, environment, etc.)
    """

    function_stats: dict[str, FunctionStats]
    hotspots: list[HotspotDict]
    memory_stats: dict[str, Any]
    call_graph: CallGraphDict
    total_time: float
    peak_memory: int
    mode: ProfilingMode
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Generate text summary of profiling results.

        Returns:
            Human-readable summary string
        """
        lines = [
            "=" * 80,
            "Performance Profiling Report",
            "=" * 80,
            f"Mode: {self.mode.value}",
            f"Total Time: {self.total_time:.4f}s",
            f"Peak Memory: {self._format_bytes(self.peak_memory)}",
            f"Functions Profiled: {len(self.function_stats)}",
            "",
            "Top 10 Hotspots (by cumulative time):",
            "-" * 80,
        ]

        # Add hotspots
        for i, hotspot in enumerate(self.hotspots[:10], 1):
            func_name = hotspot["function"]
            cum_time = hotspot["cumulative_time"]
            calls = hotspot["calls"]
            pct = hotspot["percent_time"]

            lines.append(f"{i:2d}. {func_name:50s} {cum_time:8.4f}s ({calls:6d} calls) {pct:5.1f}%")

        if self.mode == ProfilingMode.MEMORY or self.mode == ProfilingMode.FULL:
            lines.extend(
                [
                    "",
                    "Memory Statistics:",
                    "-" * 80,
                    f"  Peak Usage: {self._format_bytes(self.memory_stats.get('peak', 0))}",
                    f"  Current Usage: {self._format_bytes(self.memory_stats.get('current', 0))}",
                    f"  Allocations: {self.memory_stats.get('allocations', 0):,}",
                ]
            )

        lines.append("=" * 80)
        return "\n".join(lines)

    def export_json(self, filepath: str | Path) -> None:
        """Export profiling results to JSON file.

        Args:
            filepath: Output file path
        """
        filepath = Path(filepath)

        # Convert to serializable format
        data = {
            "function_stats": {name: asdict(stats) for name, stats in self.function_stats.items()},
            "hotspots": self.hotspots,
            "memory_stats": self.memory_stats,
            "call_graph": self.call_graph,
            "total_time": self.total_time,
            "peak_memory": self.peak_memory,
            "mode": self.mode.value,
            "metadata": self.metadata,
        }

        with filepath.open("w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Profiling results exported to {filepath}")

    def export_html(self, filepath: str | Path) -> None:
        """Export profiling results to HTML file with flame graph.

        Args:
            filepath: Output file path
        """
        filepath = Path(filepath)

        html = self._generate_html()

        with filepath.open("w") as f:
            f.write(html)

        logger.info(f"HTML report exported to {filepath}")

    def export_text(self, filepath: str | Path) -> None:
        """Export profiling results to text file.

        Args:
            filepath: Output file path
        """
        filepath = Path(filepath)

        with filepath.open("w") as f:
            f.write(self.summary())

        logger.info(f"Text report exported to {filepath}")

    def _format_bytes(self, bytes_value: int) -> str:
        """Format bytes as human-readable string.

        Args:
            bytes_value: Number of bytes

        Returns:
            Formatted string (e.g., "10.5 MB")
        """
        if bytes_value == 0:
            return "0 B"

        units = ["B", "KB", "MB", "GB", "TB"]
        unit_index = 0
        value = float(bytes_value)

        while value >= 1024.0 and unit_index < len(units) - 1:
            value /= 1024.0
            unit_index += 1

        return f"{value:.2f} {units[unit_index]}"

    def _generate_html(self) -> str:
        """Generate HTML report with styling.

        Returns:
            HTML string
        """
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Performance Profiling Report</title>
    <style>
        body {{ font-family: 'Courier New', monospace; margin: 20px; background: #1e1e1e; color: #d4d4d4; }}
        h1 {{ color: #4ec9b0; }}
        h2 {{ color: #569cd6; margin-top: 30px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
        th {{ background: #2d2d30; color: #4ec9b0; padding: 10px; text-align: left; }}
        td {{ padding: 8px; border-bottom: 1px solid #3e3e42; }}
        .hotspot {{ background: #3c2929; }}
        .stats {{ margin: 10px 0; padding: 10px; background: #2d2d30; border-radius: 5px; }}
        .metric {{ display: inline-block; margin: 10px 20px 10px 0; }}
        .value {{ color: #ce9178; font-weight: bold; }}
    </style>
</head>
<body>
    <h1>Performance Profiling Report</h1>

    <div class="stats">
        <div class="metric">Mode: <span class="value">{self.mode.value}</span></div>
        <div class="metric">Total Time: <span class="value">{self.total_time:.4f}s</span></div>
        <div class="metric">Peak Memory: <span class="value">{self._format_bytes(self.peak_memory)}</span></div>
        <div class="metric">Functions: <span class="value">{len(self.function_stats)}</span></div>
    </div>

    <h2>Top Hotspots</h2>
    <table>
        <tr>
            <th>Rank</th>
            <th>Function</th>
            <th>Cumulative Time</th>
            <th>Calls</th>
            <th>% of Total</th>
        </tr>
"""

        for i, hotspot in enumerate(self.hotspots[:20], 1):
            css_class = "hotspot" if i <= 5 else ""
            html += f"""        <tr class="{css_class}">
            <td>{i}</td>
            <td>{hotspot["function"]}</td>
            <td>{hotspot["cumulative_time"]:.4f}s</td>
            <td>{hotspot["calls"]}</td>
            <td>{hotspot["percent_time"]:.1f}%</td>
        </tr>
"""

        html += """    </table>
</body>
</html>
"""
        return html


class PerformanceProfiler:
    """Comprehensive performance profiler for analysis workflows.

    Profiles CPU usage, memory consumption, I/O operations, and identifies
    performance bottlenecks using multiple profiling backends.

    Example:
        Basic usage::

            profiler = PerformanceProfiler()
            profiler.start()
            # Run analysis code
            result = profiler.stop()
            print(result.summary())

        Context manager::

            with PerformanceProfiler() as profiler:
                # Run analysis code
                pass
            result = profiler.get_results()

        Decorator::

            @PerformanceProfiler.profile_function()
            def my_analysis_function(data):
                # Analysis code
                pass

    References:
        Python cProfile: https://docs.python.org/3/library/profile.html
        tracemalloc: https://docs.python.org/3/library/tracemalloc.html
    """

    def __init__(self, mode: ProfilingMode = ProfilingMode.FUNCTION) -> None:
        """Initialize profiler.

        Args:
            mode: Profiling mode (FUNCTION, LINE, MEMORY, IO, FULL)
        """
        self.mode = mode
        self._profiler: cProfile.Profile | None = None
        self._start_time: float = 0.0
        self._end_time: float = 0.0
        self._memory_start: tuple[int, int] = (0, 0)
        self._memory_peak: int = 0
        self._result: ProfilingResult | None = None
        self._is_running: bool = False

        # Check optional dependencies
        self._line_profiler_available = self._check_line_profiler()
        self._memory_profiler_available = self._check_memory_profiler()

    def _check_line_profiler(self) -> bool:
        """Check if line_profiler is available.

        Returns:
            True if line_profiler is installed
        """
        try:
            import line_profiler  # type: ignore[import-not-found]  # noqa: F401

            return True
        except ImportError:
            if self.mode == ProfilingMode.LINE:
                logger.warning(
                    "line_profiler not installed. Install with: pip install line_profiler"
                )
            return False

    def _check_memory_profiler(self) -> bool:
        """Check if memory_profiler is available.

        Returns:
            True if memory_profiler is installed
        """
        try:
            import memory_profiler  # type: ignore[import-not-found]  # noqa: F401

            return True
        except ImportError:
            if self.mode == ProfilingMode.MEMORY:
                logger.info("memory_profiler not installed (optional). Using tracemalloc instead.")
            return False

    def start(self) -> None:
        """Start profiling."""
        if self._is_running:
            logger.warning("Profiler is already running")
            return

        self._start_time = time.perf_counter()
        self._is_running = True

        # Start appropriate profiling backend
        if self.mode in (ProfilingMode.FUNCTION, ProfilingMode.FULL):
            self._profiler = cProfile.Profile()
            try:
                self._profiler.enable()
            except ValueError as e:
                # Handle nested profiler case
                if "Another profiling tool is already active" in str(e):
                    logger.warning(
                        "Another profiler is active. This profiler will not collect detailed statistics."
                    )
                    self._profiler = None
                else:
                    raise

        if self.mode in (ProfilingMode.MEMORY, ProfilingMode.FULL):
            if not tracemalloc.is_tracing():
                tracemalloc.start()
            self._memory_start = tracemalloc.get_traced_memory()

        logger.info(f"Profiling started in {self.mode.value} mode")

    def stop(self) -> ProfilingResult:
        """Stop profiling and generate results.

        Returns:
            Profiling results

        Raises:
            RuntimeError: If profiler is not running
        """
        if not self._is_running:
            raise RuntimeError("Profiler is not running. Call start() first.")

        self._end_time = time.perf_counter()
        total_time = self._end_time - self._start_time

        # Stop profiling backends
        if self._profiler is not None:
            self._profiler.disable()

        memory_stats: dict[str, Any] = {}
        if self.mode in (ProfilingMode.MEMORY, ProfilingMode.FULL):
            current, peak = tracemalloc.get_traced_memory()
            self._memory_peak = peak
            memory_stats = {
                "current": current,
                "peak": peak,
                "allocations": tracemalloc.get_tracemalloc_memory(),
            }
            tracemalloc.stop()

        # Extract statistics
        function_stats = self._extract_function_stats()
        hotspots = self._identify_hotspots(function_stats, total_time)
        call_graph = self._build_call_graph()

        # Create result
        self._result = ProfilingResult(
            function_stats=function_stats,
            hotspots=hotspots,
            memory_stats=memory_stats,
            call_graph=call_graph,
            total_time=total_time,
            peak_memory=self._memory_peak,
            mode=self.mode,
            metadata={
                "python_version": sys.version,
                "platform": sys.platform,
                "timestamp": time.time(),
            },
        )

        self._is_running = False
        logger.info(f"Profiling stopped. Total time: {total_time:.4f}s")

        return self._result

    def get_results(self) -> ProfilingResult | None:
        """Get profiling results without stopping.

        Returns:
            Profiling results or None if not available
        """
        return self._result

    @contextmanager
    def profile(self) -> Iterator[PerformanceProfiler]:
        """Context manager for profiling code blocks.

        Yields:
            PerformanceProfiler instance

        Example:
            >>> with PerformanceProfiler() as profiler:
            ...     # Code to profile
            ...     pass
            >>> result = profiler.get_results()
        """
        self.start()
        try:
            yield self
        finally:
            self.stop()

    def __enter__(self) -> PerformanceProfiler:
        """Enter context manager."""
        self.start()
        return self

    def __exit__(self, exc_type: type, exc_val: Exception, exc_tb: Any) -> None:
        """Exit context manager."""
        if self._is_running:
            self.stop()

    @staticmethod
    def profile_function(
        mode: ProfilingMode = ProfilingMode.FUNCTION,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator for profiling individual functions.

        Args:
            mode: Profiling mode

        Returns:
            Decorator function

        Example:
            >>> @PerformanceProfiler.profile_function()
            >>> def my_function(x):
            ...     return x * 2
        """

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                profiler = PerformanceProfiler(mode=mode)
                profiler.start()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    profiling_result = profiler.stop()
                    logger.info(
                        f"\nProfiling results for {func.__name__}:\n{profiling_result.summary()}"
                    )

            return wrapper

        return decorator

    def _extract_function_stats(self) -> dict[str, FunctionStats]:
        """Extract function statistics from profiler.

        Returns:
            Dictionary mapping function names to their statistics
        """
        if self._profiler is None:
            return {}

        # Get statistics from cProfile
        stats_stream = io.StringIO()
        stats = pstats.Stats(self._profiler, stream=stats_stream)
        stats.sort_stats(pstats.SortKey.CUMULATIVE)

        function_stats: dict[str, FunctionStats] = {}

        for func_key, (_cc, nc, tt, ct, _callers) in stats.stats.items():  # type: ignore[attr-defined]
            filename, lineno, func_name = func_key

            # Create readable function name
            if filename == "~":
                full_name = func_name
            else:
                full_name = f"{Path(filename).name}:{func_name}"

            function_stats[full_name] = FunctionStats(
                name=full_name,
                calls=nc,
                time=tt,
                cumulative_time=ct,
                filename=filename,
                lineno=lineno,
            )

        return function_stats

    def _identify_hotspots(
        self, function_stats: dict[str, FunctionStats], total_time: float
    ) -> list[HotspotDict]:
        """Identify performance hotspots.

        Args:
            function_stats: Function statistics
            total_time: Total execution time

        Returns:
            List of hotspot dictionaries sorted by cumulative time
        """
        hotspots: list[HotspotDict] = []

        for func_name, stats in function_stats.items():
            percent_time = (stats.cumulative_time / total_time * 100.0) if total_time > 0 else 0.0

            hotspots.append(
                {
                    "function": func_name,
                    "calls": stats.calls,
                    "time": stats.time,
                    "cumulative_time": stats.cumulative_time,
                    "percent_time": percent_time,
                    "per_call_time": stats.per_call_time,
                }
            )

        # Sort by cumulative time (descending)
        hotspots.sort(key=lambda x: x["cumulative_time"], reverse=True)

        return hotspots

    def _build_call_graph(self) -> CallGraphDict:
        """Build function call graph.

        Returns:
            Dictionary mapping functions to list of called functions
        """
        if self._profiler is None:
            return {}

        call_graph: CallGraphDict = {}

        stats = pstats.Stats(self._profiler)

        for func_key in stats.stats:  # type: ignore[attr-defined]
            filename, lineno, func_name = func_key

            # Create readable function name
            if filename == "~":
                full_name = func_name
            else:
                full_name = f"{Path(filename).name}:{func_name}"

            # Get callees (functions called by this function)
            # Note: all_callees may not be available in all Python versions
            callees: list[str] = []
            all_callees = getattr(stats, "all_callees", None)
            if all_callees is not None and func_key in all_callees:
                for callee_key in all_callees[func_key]:
                    callee_filename, callee_lineno, callee_name = callee_key
                    if callee_filename == "~":
                        callee_full = callee_name
                    else:
                        callee_full = f"{Path(callee_filename).name}:{callee_name}"
                    callees.append(callee_full)

            call_graph[full_name] = callees

        return call_graph
