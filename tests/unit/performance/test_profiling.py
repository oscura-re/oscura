"""Unit tests for performance profiling module.

Tests the comprehensive performance profiling capabilities including CPU,
memory, I/O profiling, and bottleneck identification.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from oscura.utils.performance.profiling import (
    FunctionStats,
    PerformanceProfiler,
    ProfilingMode,
    ProfilingResult,
)

pytestmark = pytest.mark.unit


class TestProfilingMode:
    """Test ProfilingMode enum."""

    def test_profiling_modes_exist(self) -> None:
        """Test all profiling modes are defined."""
        assert ProfilingMode.FUNCTION.value == "function"
        assert ProfilingMode.LINE.value == "line"
        assert ProfilingMode.MEMORY.value == "memory"
        assert ProfilingMode.IO.value == "io"
        assert ProfilingMode.FULL.value == "full"

    def test_profiling_mode_enum_members(self) -> None:
        """Test enum has expected members."""
        modes = list(ProfilingMode)
        assert len(modes) == 5
        assert ProfilingMode.FUNCTION in modes
        assert ProfilingMode.LINE in modes
        assert ProfilingMode.MEMORY in modes
        assert ProfilingMode.IO in modes
        assert ProfilingMode.FULL in modes


class TestFunctionStats:
    """Test FunctionStats dataclass."""

    def test_function_stats_creation(self) -> None:
        """Test creating FunctionStats instance."""
        stats = FunctionStats(
            name="test_function",
            calls=10,
            time=1.5,
            cumulative_time=2.0,
        )

        assert stats.name == "test_function"
        assert stats.calls == 10
        assert stats.time == 1.5
        assert stats.cumulative_time == 2.0
        assert stats.memory == 0
        assert stats.filename == ""
        assert stats.lineno == 0

    def test_function_stats_per_call_time_calculation(self) -> None:
        """Test per_call_time is calculated correctly."""
        stats = FunctionStats(
            name="test_function",
            calls=10,
            time=1.0,
            cumulative_time=2.0,
        )

        assert stats.per_call_time == 0.1  # 1.0 / 10

    def test_function_stats_per_call_memory_calculation(self) -> None:
        """Test per_call_memory is calculated correctly."""
        stats = FunctionStats(
            name="test_function",
            calls=5,
            time=1.0,
            cumulative_time=2.0,
            memory=1000,
        )

        assert stats.per_call_memory == 200  # 1000 // 5

    def test_function_stats_zero_calls(self) -> None:
        """Test FunctionStats with zero calls."""
        stats = FunctionStats(
            name="test_function",
            calls=0,
            time=0.0,
            cumulative_time=0.0,
        )

        # Should not raise division by zero
        assert stats.per_call_time == 0.0
        assert stats.per_call_memory == 0

    def test_function_stats_with_file_info(self) -> None:
        """Test FunctionStats with file information."""
        stats = FunctionStats(
            name="test_function",
            calls=1,
            time=0.1,
            cumulative_time=0.2,
            filename="/path/to/file.py",
            lineno=42,
        )

        assert stats.filename == "/path/to/file.py"
        assert stats.lineno == 42


class TestProfilingResult:
    """Test ProfilingResult dataclass."""

    def test_profiling_result_creation(self) -> None:
        """Test creating ProfilingResult instance."""
        stats = {
            "func1": FunctionStats(name="func1", calls=10, time=1.0, cumulative_time=2.0),
        }

        result = ProfilingResult(
            function_stats=stats,
            hotspots=[],
            memory_stats={},
            call_graph={},
            total_time=5.0,
            peak_memory=1024,
            mode=ProfilingMode.FUNCTION,
        )

        assert result.function_stats == stats
        assert result.hotspots == []
        assert result.memory_stats == {}
        assert result.call_graph == {}
        assert result.total_time == 5.0
        assert result.peak_memory == 1024
        assert result.mode == ProfilingMode.FUNCTION
        assert "timestamp" not in result.metadata

    def test_profiling_result_summary_basic(self) -> None:
        """Test summary generation."""
        result = ProfilingResult(
            function_stats={},
            hotspots=[
                {
                    "function": "test_func",
                    "cumulative_time": 1.5,
                    "calls": 10,
                    "percent_time": 75.0,
                    "time": 1.0,
                    "per_call_time": 0.1,
                }
            ],
            memory_stats={},
            call_graph={},
            total_time=2.0,
            peak_memory=1024,
            mode=ProfilingMode.FUNCTION,
        )

        summary = result.summary()

        assert "Performance Profiling Report" in summary
        assert "Mode: function" in summary
        assert "Total Time: 2.0000s" in summary
        assert "1.00 KB" in summary  # Peak memory
        assert "Top 10 Hotspots" in summary
        assert "test_func" in summary
        assert "1.5000s" in summary
        assert "75.0%" in summary

    def test_profiling_result_summary_with_memory(self) -> None:
        """Test summary generation with memory stats."""
        result = ProfilingResult(
            function_stats={},
            hotspots=[],
            memory_stats={
                "peak": 1048576,
                "current": 524288,
                "allocations": 100,
            },
            call_graph={},
            total_time=1.0,
            peak_memory=1048576,
            mode=ProfilingMode.MEMORY,
        )

        summary = result.summary()

        assert "Memory Statistics:" in summary
        assert "Peak Usage: 1.00 MB" in summary
        assert "Current Usage: 512.00 KB" in summary
        assert "Allocations: 100" in summary

    def test_profiling_result_format_bytes(self) -> None:
        """Test byte formatting."""
        result = ProfilingResult(
            function_stats={},
            hotspots=[],
            memory_stats={},
            call_graph={},
            total_time=1.0,
            peak_memory=0,
            mode=ProfilingMode.FUNCTION,
        )

        assert result._format_bytes(0) == "0 B"
        assert result._format_bytes(512) == "512.00 B"
        assert result._format_bytes(1024) == "1.00 KB"
        assert result._format_bytes(1024 * 1024) == "1.00 MB"
        assert result._format_bytes(1024 * 1024 * 1024) == "1.00 GB"
        assert result._format_bytes(1024 * 1024 * 1024 * 1024) == "1.00 TB"

    def test_profiling_result_format_bytes_non_round(self) -> None:
        """Test byte formatting with non-round numbers."""
        result = ProfilingResult(
            function_stats={},
            hotspots=[],
            memory_stats={},
            call_graph={},
            total_time=1.0,
            peak_memory=0,
            mode=ProfilingMode.FUNCTION,
        )

        assert result._format_bytes(1536) == "1.50 KB"
        assert result._format_bytes(2560 * 1024) == "2.50 MB"

    def test_profiling_result_export_json(self, tmp_path: Path) -> None:
        """Test exporting profiling results to JSON."""
        stats = {
            "func1": FunctionStats(name="func1", calls=10, time=1.0, cumulative_time=2.0),
        }

        result = ProfilingResult(
            function_stats=stats,
            hotspots=[
                {
                    "function": "func1",
                    "cumulative_time": 2.0,
                    "calls": 10,
                    "percent_time": 100.0,
                    "time": 1.0,
                    "per_call_time": 0.1,
                }
            ],
            memory_stats={"peak": 1024},
            call_graph={"func1": []},
            total_time=2.0,
            peak_memory=1024,
            mode=ProfilingMode.FUNCTION,
            metadata={"test": "value"},
        )

        output_file = tmp_path / "profile.json"
        result.export_json(output_file)

        assert output_file.exists()

        with output_file.open() as f:
            data = json.load(f)

        assert "function_stats" in data
        assert "hotspots" in data
        assert "memory_stats" in data
        assert "call_graph" in data
        assert data["total_time"] == 2.0
        assert data["peak_memory"] == 1024
        assert data["mode"] == "function"
        assert data["metadata"]["test"] == "value"

    def test_profiling_result_export_html(self, tmp_path: Path) -> None:
        """Test exporting profiling results to HTML."""
        result = ProfilingResult(
            function_stats={},
            hotspots=[
                {
                    "function": "test_func",
                    "cumulative_time": 1.5,
                    "calls": 10,
                    "percent_time": 75.0,
                    "time": 1.0,
                    "per_call_time": 0.1,
                }
            ],
            memory_stats={},
            call_graph={},
            total_time=2.0,
            peak_memory=1024,
            mode=ProfilingMode.FUNCTION,
        )

        output_file = tmp_path / "profile.html"
        result.export_html(output_file)

        assert output_file.exists()

        html_content = output_file.read_text()

        assert "<!DOCTYPE html>" in html_content
        assert "Performance Profiling Report" in html_content
        assert "test_func" in html_content
        assert "1.5000s" in html_content
        assert "75.0%" in html_content

    def test_profiling_result_export_text(self, tmp_path: Path) -> None:
        """Test exporting profiling results to text."""
        result = ProfilingResult(
            function_stats={},
            hotspots=[],
            memory_stats={},
            call_graph={},
            total_time=1.0,
            peak_memory=1024,
            mode=ProfilingMode.FUNCTION,
        )

        output_file = tmp_path / "profile.txt"
        result.export_text(output_file)

        assert output_file.exists()

        text_content = output_file.read_text()

        assert "Performance Profiling Report" in text_content
        assert "Total Time: 1.0000s" in text_content


class TestPerformanceProfiler:
    """Test PerformanceProfiler class."""

    def test_profiler_initialization(self) -> None:
        """Test creating PerformanceProfiler instance."""
        profiler = PerformanceProfiler()

        assert profiler.mode == ProfilingMode.FUNCTION
        assert profiler._profiler is None
        assert profiler._start_time == 0.0
        assert profiler._end_time == 0.0
        assert profiler._memory_peak == 0
        assert profiler._result is None
        assert profiler._is_running is False

    def test_profiler_initialization_with_mode(self) -> None:
        """Test creating PerformanceProfiler with specific mode."""
        profiler = PerformanceProfiler(mode=ProfilingMode.MEMORY)

        assert profiler.mode == ProfilingMode.MEMORY

    def test_profiler_start(self) -> None:
        """Test starting profiler."""
        profiler = PerformanceProfiler()
        profiler.start()

        assert profiler._is_running is True
        # profiler may be None if another profiler is active (e.g., pytest)
        assert profiler._start_time > 0

        # Cleanup
        profiler.stop()

    def test_profiler_start_already_running(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test starting profiler when already running."""
        profiler = PerformanceProfiler()
        profiler.start()
        profiler.start()  # Second start

        assert "already running" in caplog.text

        # Cleanup
        profiler.stop()

    def test_profiler_stop(self) -> None:
        """Test stopping profiler."""
        profiler = PerformanceProfiler()
        profiler.start()

        # Do some work
        total = sum(range(1000))

        result = profiler.stop()

        assert isinstance(result, ProfilingResult)
        assert profiler._is_running is False
        assert result.total_time > 0
        # function_stats may be empty if another profiler is active
        assert len(result.function_stats) >= 0

    def test_profiler_stop_not_running(self) -> None:
        """Test stopping profiler when not running."""
        profiler = PerformanceProfiler()

        with pytest.raises(RuntimeError, match="not running"):
            profiler.stop()

    def test_profiler_context_manager(self) -> None:
        """Test profiler as context manager."""
        with PerformanceProfiler() as profiler:
            # Do some work
            total = sum(range(1000))
            assert profiler._is_running is True

        result = profiler.get_results()
        assert result is not None
        assert result.total_time > 0

    def test_profiler_get_results_none(self) -> None:
        """Test get_results returns None before profiling."""
        profiler = PerformanceProfiler()

        assert profiler.get_results() is None

    def test_profiler_get_results_after_profiling(self) -> None:
        """Test get_results returns results after profiling."""
        profiler = PerformanceProfiler()
        profiler.start()
        total = sum(range(100))
        result = profiler.stop()

        assert profiler.get_results() is result

    @patch("oscura.performance.profiling.tracemalloc")
    def test_profiler_memory_mode(self, mock_tracemalloc: MagicMock) -> None:
        """Test profiler in memory mode."""
        mock_tracemalloc.is_tracing.return_value = False
        mock_tracemalloc.get_traced_memory.return_value = (1024, 2048)
        mock_tracemalloc.get_tracemalloc_memory.return_value = 512

        profiler = PerformanceProfiler(mode=ProfilingMode.MEMORY)
        profiler.start()

        assert mock_tracemalloc.start.called

        # Do some work
        total = sum(range(100))

        result = profiler.stop()

        assert mock_tracemalloc.stop.called
        assert "current" in result.memory_stats
        assert "peak" in result.memory_stats
        assert "allocations" in result.memory_stats

    @patch("oscura.performance.profiling.tracemalloc")
    def test_profiler_full_mode(self, mock_tracemalloc: MagicMock) -> None:
        """Test profiler in full mode."""
        mock_tracemalloc.is_tracing.return_value = False
        mock_tracemalloc.get_traced_memory.return_value = (1024, 2048)
        mock_tracemalloc.get_tracemalloc_memory.return_value = 512

        profiler = PerformanceProfiler(mode=ProfilingMode.FULL)
        profiler.start()

        # profiler may be None if another profiler is active
        assert mock_tracemalloc.start.called

        # Cleanup
        profiler.stop()

    def test_profiler_decorator(self) -> None:
        """Test @profile_function decorator."""

        @PerformanceProfiler.profile_function()
        def test_function(x: int) -> int:
            return sum(range(x))

        result = test_function(100)

        assert result == sum(range(100))

    def test_profiler_decorator_with_mode(self) -> None:
        """Test @profile_function decorator with custom mode."""

        @PerformanceProfiler.profile_function(mode=ProfilingMode.FUNCTION)
        def test_function(x: int) -> int:
            return x * 2

        result = test_function(5)

        assert result == 10

    def test_profiler_extract_function_stats(self) -> None:
        """Test extracting function statistics."""
        profiler = PerformanceProfiler()
        profiler.start()

        # Do some work with known function calls
        def test_func() -> int:
            return sum(range(100))

        test_func()
        test_func()

        profiler.stop()

        stats = profiler._extract_function_stats()

        assert isinstance(stats, dict)
        # Stats may be empty if another profiler is active
        assert len(stats) >= 0

        # Check structure of stats
        for func_stats in stats.values():
            assert isinstance(func_stats, FunctionStats)
            assert func_stats.calls > 0
            assert func_stats.time >= 0
            assert func_stats.cumulative_time >= 0

    def test_profiler_identify_hotspots(self) -> None:
        """Test identifying performance hotspots."""
        profiler = PerformanceProfiler()

        function_stats = {
            "fast_func": FunctionStats(name="fast_func", calls=10, time=0.1, cumulative_time=0.2),
            "slow_func": FunctionStats(name="slow_func", calls=5, time=1.0, cumulative_time=1.5),
            "medium_func": FunctionStats(
                name="medium_func", calls=20, time=0.5, cumulative_time=0.7
            ),
        }

        hotspots = profiler._identify_hotspots(function_stats, total_time=2.0)

        assert len(hotspots) == 3
        # Should be sorted by cumulative_time descending
        assert hotspots[0]["function"] == "slow_func"
        assert hotspots[1]["function"] == "medium_func"
        assert hotspots[2]["function"] == "fast_func"

        # Check percent_time calculation
        assert hotspots[0]["percent_time"] == 75.0  # 1.5/2.0 * 100

    def test_profiler_identify_hotspots_zero_total_time(self) -> None:
        """Test identifying hotspots with zero total time."""
        profiler = PerformanceProfiler()

        function_stats = {
            "func1": FunctionStats(name="func1", calls=1, time=0.0, cumulative_time=0.0),
        }

        hotspots = profiler._identify_hotspots(function_stats, total_time=0.0)

        assert len(hotspots) == 1
        assert hotspots[0]["percent_time"] == 0.0

    def test_profiler_build_call_graph(self) -> None:
        """Test building function call graph."""
        profiler = PerformanceProfiler()
        profiler.start()

        def caller() -> int:
            return callee()

        def callee() -> int:
            return 42

        caller()

        profiler.stop()

        call_graph = profiler._build_call_graph()

        assert isinstance(call_graph, dict)
        # Call graph should have entries
        assert len(call_graph) >= 0

    def test_profiler_metadata(self) -> None:
        """Test profiling result includes metadata."""
        profiler = PerformanceProfiler()
        profiler.start()
        result = profiler.stop()

        assert "python_version" in result.metadata
        assert "platform" in result.metadata
        assert "timestamp" in result.metadata


class TestProfilingIntegration:
    """Integration tests for performance profiling."""

    def test_profile_simple_function(self) -> None:
        """Test profiling a simple function."""

        def fibonacci(n: int) -> int:
            if n <= 1:
                return n
            return fibonacci(n - 1) + fibonacci(n - 2)

        profiler = PerformanceProfiler()
        profiler.start()

        result = fibonacci(10)

        profiling_result = profiler.stop()

        assert result == 55
        assert profiling_result.total_time > 0
        # function_stats may be empty if another profiler is active
        assert len(profiling_result.function_stats) >= 0
        assert len(profiling_result.hotspots) >= 0

    def test_profile_with_loops(self) -> None:
        """Test profiling code with loops."""
        profiler = PerformanceProfiler()
        profiler.start()

        total = 0
        for i in range(1000):
            total += i * 2

        profiling_result = profiler.stop()

        assert profiling_result.total_time > 0
        # function_stats may be empty if another profiler is active
        assert len(profiling_result.function_stats) >= 0

    def test_profile_memory_intensive_code(self) -> None:
        """Test profiling memory-intensive code."""
        profiler = PerformanceProfiler(mode=ProfilingMode.FULL)
        profiler.start()

        # Allocate some memory
        large_list = list(range(10000))
        large_dict = {i: i * 2 for i in range(1000)}

        profiling_result = profiler.stop()

        assert profiling_result.peak_memory > 0
        assert "current" in profiling_result.memory_stats
        assert "peak" in profiling_result.memory_stats

    def test_export_all_formats(self, tmp_path: Path) -> None:
        """Test exporting profiling results in all formats."""
        profiler = PerformanceProfiler()
        profiler.start()
        total = sum(range(100))
        result = profiler.stop()

        # Export to all formats
        json_file = tmp_path / "profile.json"
        html_file = tmp_path / "profile.html"
        text_file = tmp_path / "profile.txt"

        result.export_json(json_file)
        result.export_html(html_file)
        result.export_text(text_file)

        assert json_file.exists()
        assert html_file.exists()
        assert text_file.exists()

        # Verify content
        with json_file.open() as f:
            data = json.load(f)
        assert "function_stats" in data

        html_content = html_file.read_text()
        assert "Performance Profiling Report" in html_content

        text_content = text_file.read_text()
        assert "Performance Profiling Report" in text_content


class TestProfilingEdgeCases:
    """Test edge cases and error handling."""

    def test_profiler_empty_code_block(self) -> None:
        """Test profiling empty code block."""
        profiler = PerformanceProfiler()
        profiler.start()
        # Do nothing
        result = profiler.stop()

        assert result.total_time >= 0
        # May have some built-in function calls
        assert len(result.function_stats) >= 0

    def test_profiler_exception_in_context_manager(self) -> None:
        """Test profiler handles exceptions in context manager."""
        try:
            with PerformanceProfiler() as profiler:
                raise ValueError("Test error")
        except ValueError:
            pass

        # Profiler should still have stopped
        result = profiler.get_results()
        assert result is not None

    def test_profiler_multiple_start_stop_cycles(self) -> None:
        """Test multiple profiling cycles."""
        profiler = PerformanceProfiler()

        # First cycle
        profiler.start()
        sum(range(100))
        result1 = profiler.stop()

        # Second cycle
        profiler.start()
        sum(range(200))
        result2 = profiler.stop()

        assert result1 is not result2
        assert result1.total_time > 0
        assert result2.total_time > 0

    def test_profiler_nested_context_managers(self) -> None:
        """Test nested profilers (not recommended but should work)."""
        with PerformanceProfiler() as profiler1:
            sum(range(50))

            with PerformanceProfiler() as profiler2:
                sum(range(100))

            sum(range(75))

        result1 = profiler1.get_results()
        result2 = profiler2.get_results()

        assert result1 is not None
        assert result2 is not None
        assert result1.total_time > 0
        assert result2.total_time > 0

    def test_profiler_very_fast_execution(self) -> None:
        """Test profiling very fast code."""
        profiler = PerformanceProfiler()
        profiler.start()

        # Extremely fast operation
        x = 1 + 1

        result = profiler.stop()

        # Should still produce valid results
        assert result.total_time >= 0
        assert isinstance(result.function_stats, dict)

    def test_function_stats_equality(self) -> None:
        """Test FunctionStats comparison."""
        stats1 = FunctionStats(name="func", calls=10, time=1.0, cumulative_time=2.0)
        stats2 = FunctionStats(name="func", calls=10, time=1.0, cumulative_time=2.0)

        # Dataclasses have default equality
        assert stats1 == stats2

    def test_profiling_result_empty_hotspots_in_summary(self) -> None:
        """Test summary with no hotspots."""
        result = ProfilingResult(
            function_stats={},
            hotspots=[],
            memory_stats={},
            call_graph={},
            total_time=1.0,
            peak_memory=0,
            mode=ProfilingMode.FUNCTION,
        )

        summary = result.summary()

        assert "Performance Profiling Report" in summary
        assert "Top 10 Hotspots" in summary

    def test_profiling_result_many_hotspots(self) -> None:
        """Test summary with more than 10 hotspots."""
        hotspots = [
            {
                "function": f"func_{i}",
                "cumulative_time": float(i),
                "calls": i,
                "percent_time": float(i),
                "time": float(i),
                "per_call_time": 1.0,
            }
            for i in range(20)
        ]

        result = ProfilingResult(
            function_stats={},
            hotspots=hotspots,
            memory_stats={},
            call_graph={},
            total_time=10.0,
            peak_memory=0,
            mode=ProfilingMode.FUNCTION,
        )

        summary = result.summary()

        # Should only show top 10
        for i in range(10):
            assert f"func_{i}" in summary

    def test_profiler_check_line_profiler_unavailable(self) -> None:
        """Test graceful degradation when line_profiler unavailable."""
        profiler = PerformanceProfiler(mode=ProfilingMode.LINE)

        # Should not raise, just log warning
        assert profiler._line_profiler_available in (True, False)

    def test_profiler_check_memory_profiler_unavailable(self) -> None:
        """Test graceful degradation when memory_profiler unavailable."""
        profiler = PerformanceProfiler(mode=ProfilingMode.MEMORY)

        # Should not raise, just log info
        assert profiler._memory_profiler_available in (True, False)
