"""Tests for performance timing and monitoring.

Tests the performance timing infrastructure (LOG-006).
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from oscura.core.performance import (
    PerformanceCollector,
    PerformanceContext,
    PerformanceRecord,
    clear_performance_data,
    get_performance_records,
    get_performance_summary,
    timed,
)

pytestmark = [pytest.mark.unit, pytest.mark.core]


class TestPerformanceRecord:
    """Test PerformanceRecord dataclass."""

    def test_create_basic(self) -> None:
        """Test creating basic performance record."""
        record = PerformanceRecord(
            operation="test_op",
            duration=1.23,
            timestamp="2025-01-01T12:00:00Z",
        )
        assert record.operation == "test_op"
        assert record.duration == 1.23
        assert record.timestamp == "2025-01-01T12:00:00Z"
        assert record.metadata == {}

    def test_create_with_metadata(self) -> None:
        """Test creating record with metadata."""
        metadata = {"samples": 1000, "algorithm": "fft"}
        record = PerformanceRecord(
            operation="compute",
            duration=2.5,
            timestamp="2025-01-01T12:00:00Z",
            metadata=metadata,
        )
        assert record.metadata["samples"] == 1000
        assert record.metadata["algorithm"] == "fft"


class TestPerformanceCollector:
    """Test PerformanceCollector class."""

    def test_init(self) -> None:
        """Test collector initialization."""
        collector = PerformanceCollector()
        assert collector.records == []

    def test_record_single(self) -> None:
        """Test recording single measurement."""
        collector = PerformanceCollector()
        collector.record("operation1", 1.5)
        assert len(collector.records) == 1
        assert collector.records[0].operation == "operation1"
        assert collector.records[0].duration == 1.5

    def test_record_multiple(self) -> None:
        """Test recording multiple measurements."""
        collector = PerformanceCollector()
        collector.record("op1", 1.0)
        collector.record("op2", 2.0)
        collector.record("op1", 1.5)
        assert len(collector.records) == 3

    def test_record_with_metadata(self) -> None:
        """Test recording with metadata."""
        collector = PerformanceCollector()
        collector.record("fft", 2.3, samples=10000, window="hann")
        record = collector.records[0]
        assert record.metadata["samples"] == 10000
        assert record.metadata["window"] == "hann"

    def test_get_summary_single_operation(self) -> None:
        """Test summary for single operation."""
        collector = PerformanceCollector()
        collector.record("op1", 1.0)
        collector.record("op1", 2.0)
        collector.record("op1", 3.0)

        summary = collector.get_summary()
        assert "op1" in summary
        assert summary["op1"]["count"] == 3
        assert summary["op1"]["mean"] == 2.0
        assert summary["op1"]["total"] == 6.0

    def test_get_summary_multiple_operations(self) -> None:
        """Test summary for multiple operations."""
        collector = PerformanceCollector()
        collector.record("op1", 1.0)
        collector.record("op2", 5.0)
        collector.record("op1", 3.0)

        summary = collector.get_summary()
        assert "op1" in summary
        assert "op2" in summary
        assert summary["op1"]["count"] == 2
        assert summary["op2"]["count"] == 1

    def test_get_summary_statistics(self) -> None:
        """Test summary statistics calculation."""
        collector = PerformanceCollector()
        collector.record("op", 1.0)
        collector.record("op", 2.0)
        collector.record("op", 3.0)
        collector.record("op", 4.0)
        collector.record("op", 5.0)

        summary = collector.get_summary()
        stats = summary["op"]
        assert stats["count"] == 5
        assert stats["mean"] == 3.0
        assert stats["total"] == 15.0
        assert stats["min"] == 1.0
        assert stats["max"] == 5.0
        assert "std" in stats

    def test_get_summary_empty(self) -> None:
        """Test summary when no records."""
        collector = PerformanceCollector()
        summary = collector.get_summary()
        assert summary == {}

    def test_clear(self) -> None:
        """Test clearing records."""
        collector = PerformanceCollector()
        collector.record("op1", 1.0)
        collector.record("op2", 2.0)
        assert len(collector.records) == 2

        collector.clear()
        assert len(collector.records) == 0

    def test_get_records_all(self) -> None:
        """Test getting all records."""
        collector = PerformanceCollector()
        collector.record("op1", 1.0)
        collector.record("op2", 2.0)

        records = collector.get_records()
        assert len(records) == 2

    def test_get_records_filtered(self) -> None:
        """Test getting filtered records."""
        collector = PerformanceCollector()
        collector.record("op1", 1.0)
        collector.record("op2", 2.0)
        collector.record("op1", 1.5)

        records = collector.get_records(operation="op1")
        assert len(records) == 2
        assert all(r.operation == "op1" for r in records)

    def test_get_records_by_operation(self) -> None:
        """Test getting records filtered by operation."""
        collector = PerformanceCollector()
        collector.record("op1", 1.0)
        collector.record("op2", 2.0)
        collector.record("op1", 1.5)

        records = collector.get_records(operation="op1")
        assert len(records) == 2

    def test_get_records_since(self) -> None:
        """Test getting records filtered by timestamp."""
        from datetime import UTC, datetime

        collector = PerformanceCollector()
        # Record first operation
        collector.record("op1", 1.0)

        # Get current time after first record
        cutoff = datetime.now(UTC)

        # Small delay to ensure timestamp difference
        import time

        time.sleep(0.01)

        # Record second operation
        collector.record("op2", 2.0)

        # Should only get op2 (after cutoff)
        records = collector.get_records(since=cutoff)
        assert len(records) == 1
        assert records[0].operation == "op2"


class TestTimedDecorator:
    """Test @timed decorator."""

    @patch("oscura.core.performance._global_collector")
    def test_basic_timing(self, mock_collector: MagicMock) -> None:
        """Test basic function timing."""

        @timed()
        def test_func() -> int:
            return 42

        result = test_func()
        assert result == 42
        assert mock_collector.record.called

    @patch("oscura.core.performance._global_collector")
    def test_preserves_function_name(self, mock_collector: MagicMock) -> None:
        """Test that decorator preserves function name."""

        @timed()
        def my_function() -> None:
            pass

        assert my_function.__name__ == "my_function"

    @patch("oscura.core.performance._global_collector")
    def test_preserves_docstring(self, mock_collector: MagicMock) -> None:
        """Test that decorator preserves docstring."""

        @timed()
        def documented_func() -> None:
            """This is a docstring."""

        assert documented_func.__doc__ == "This is a docstring."

    @patch("oscura.core.performance._global_collector")
    def test_with_args_and_kwargs(self, mock_collector: MagicMock) -> None:
        """Test timing function with arguments."""

        @timed()
        def add(a: int, b: int = 10) -> int:
            return a + b

        result = add(5, b=7)
        assert result == 12

    @patch("oscura.core.performance._global_collector")
    def test_with_threshold(self, mock_collector: MagicMock) -> None:
        """Test timing with threshold."""

        @timed(threshold=0.1)
        def slow_func() -> None:
            time.sleep(0.15)

        slow_func()
        assert mock_collector.record.called

    @patch("oscura.core.performance._global_collector")
    def test_with_collect_false(self, mock_collector: MagicMock) -> None:
        """Test that collect=False doesn't record."""

        @timed(collect=False)
        def func() -> None:
            pass

        func()
        # Should not record when collect=False
        assert not mock_collector.record.called

    @patch("oscura.core.performance._global_collector")
    def test_uses_qualname(self, mock_collector: MagicMock) -> None:
        """Test that decorator uses function's __qualname__."""

        @timed()
        def my_function() -> None:
            pass

        my_function()
        call_args = mock_collector.record.call_args
        # __qualname__ includes class/module prefix if any
        assert "my_function" in call_args[0][0]

    @patch("oscura.core.performance._global_collector")
    def test_records_duration(self, mock_collector: MagicMock) -> None:
        """Test that duration is recorded."""

        @timed()
        def timed_sleep() -> None:
            time.sleep(0.01)

        timed_sleep()
        call_args = mock_collector.record.call_args
        duration = call_args[0][1]
        assert duration >= 0.01  # At least the sleep time

    @patch("oscura.core.performance._global_collector")
    def test_with_exception(self, mock_collector: MagicMock) -> None:
        """Test timing when function raises exception."""

        @timed()
        def failing_func() -> None:
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            failing_func()

        # Should still record timing even with exception
        assert mock_collector.record.called


class TestGlobalFunctions:
    """Test global performance functions."""

    @patch("oscura.core.performance._global_collector")
    def test_get_performance_summary(self, mock_collector: MagicMock) -> None:
        """Test get_performance_summary()."""
        mock_collector.get_summary.return_value = {"op1": {"count": 5}}
        summary = get_performance_summary()
        assert summary == {"op1": {"count": 5}}
        mock_collector.get_summary.assert_called_once()

    @patch("oscura.core.performance._global_collector")
    def test_clear_performance_data(self, mock_collector: MagicMock) -> None:
        """Test clear_performance_data()."""
        clear_performance_data()
        mock_collector.clear.assert_called_once()

    @patch("oscura.core.performance._global_collector")
    def test_get_performance_records_all(self, mock_collector: MagicMock) -> None:
        """Test get_performance_records() without filter."""
        mock_records = [MagicMock(), MagicMock()]
        mock_collector.get_records.return_value = mock_records
        records = get_performance_records()
        assert records == mock_records

    @patch("oscura.core.performance._global_collector")
    def test_get_performance_records_filtered(self, mock_collector: MagicMock) -> None:
        """Test get_performance_records() with filter."""
        get_performance_records(operation="op1")
        mock_collector.get_records.assert_called_once_with(operation="op1", since=None)


class TestPerformanceContext:
    """Test PerformanceContext context manager."""

    @patch("oscura.core.performance._global_collector")
    def test_basic_context(self, mock_collector: MagicMock) -> None:
        """Test basic context manager usage."""
        with PerformanceContext("operation1"):
            pass  # Do some work

        assert mock_collector.record.called
        call_args = mock_collector.record.call_args
        assert call_args[0][0] == "operation1"

    @patch("oscura.core.performance._global_collector")
    def test_context_records_duration(self, mock_collector: MagicMock) -> None:
        """Test that context records duration."""
        with PerformanceContext("timed_operation"):
            time.sleep(0.01)

        call_args = mock_collector.record.call_args
        duration = call_args[0][1]
        assert duration >= 0.01

    @patch("oscura.core.performance._global_collector")
    def test_context_with_collect_false(self, mock_collector: MagicMock) -> None:
        """Test context with collect=False doesn't record."""
        with PerformanceContext("operation", collect=False):
            time.sleep(0.01)

        # Should not record when collect=False
        assert not mock_collector.record.called

    @patch("oscura.core.performance._global_collector")
    def test_context_with_exception(self, mock_collector: MagicMock) -> None:
        """Test context when exception occurs."""
        try:
            with PerformanceContext("operation"):
                raise ValueError("Test error")
        except ValueError:
            pass

        # Should still record timing even with exception
        assert mock_collector.record.called

    @patch("oscura.core.performance._global_collector")
    def test_context_with_log_threshold(self, mock_collector: MagicMock) -> None:
        """Test context with log_threshold."""
        with PerformanceContext("fast_op", log_threshold=1.0):
            pass  # Very fast

        # Should still record (log_threshold only affects logging, not recording)
        assert mock_collector.record.called

    @patch("oscura.core.performance._global_collector")
    def test_context_returns_context_object(self, mock_collector: MagicMock) -> None:
        """Test that context manager returns itself."""
        with PerformanceContext("op") as ctx:
            assert isinstance(ctx, PerformanceContext)


class TestCorePerformanceIntegration:
    """Integration tests for performance monitoring."""

    def test_full_workflow(self) -> None:
        """Test complete performance monitoring workflow."""
        # Clear any existing data
        clear_performance_data()

        # Define timed functions
        @timed()
        def operation1() -> int:
            time.sleep(0.01)
            return 42

        @timed()
        def operation2() -> None:
            time.sleep(0.01)

        # Execute operations
        result1 = operation1()
        operation2()
        operation1()  # Second call

        assert result1 == 42

        # Check summary
        summary = get_performance_summary()
        # Functions are recorded by their __qualname__ (includes class name)
        # Check counts by looking for the right operation
        op1_key = next(k for k in summary if "operation1" in k)
        op2_key = next(k for k in summary if "operation2" in k)
        assert summary[op1_key]["count"] == 2
        assert summary[op2_key]["count"] == 1

    def test_context_workflow(self) -> None:
        """Test workflow using context managers."""
        clear_performance_data()

        with PerformanceContext("load_data"):
            time.sleep(0.01)

        with PerformanceContext("process_data"):
            time.sleep(0.01)

        records = get_performance_records()
        assert len(records) == 2

        summary = get_performance_summary()
        assert "load_data" in summary
        assert "process_data" in summary

    def test_mixed_timing_methods(self) -> None:
        """Test using both decorator and context manager."""
        clear_performance_data()

        @timed()
        def func1() -> None:
            time.sleep(0.01)

        func1()

        with PerformanceContext("manual_timing"):
            time.sleep(0.01)

        summary = get_performance_summary()
        assert len(summary) == 2

    def test_record_filtering(self) -> None:
        """Test filtering performance records."""
        clear_performance_data()

        @timed()
        def op1() -> None:
            pass

        @timed()
        def op2() -> None:
            pass

        op1()
        op2()
        op1()

        # Get all records first to find the qualname
        all_records = get_performance_records()
        assert len(all_records) == 3

        # Find the actual operation name used (includes qualname)
        op1_name = next(r.operation for r in all_records if "op1" in r.operation)

        # Get only op1 records by qualname
        op1_records = get_performance_records(operation=op1_name)
        assert len(op1_records) == 2
        assert all(r.operation == op1_name for r in op1_records)

    def test_collect_parameter(self) -> None:
        """Test that collect parameter controls recording."""
        clear_performance_data()

        @timed(collect=True)
        def recorded_op() -> None:
            time.sleep(0.01)

        @timed(collect=False)
        def not_recorded_op() -> None:
            pass

        recorded_op()
        not_recorded_op()

        summary = get_performance_summary()
        # Only recorded_op should be in summary
        assert any("recorded_op" in key for key in summary)
        assert not any("not_recorded_op" in key for key in summary)
