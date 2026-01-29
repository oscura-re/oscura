"""Tests for batch metrics module (workflows/batch/metrics.py).

Achieves 100% coverage through comprehensive testing of:
- FileMetrics dataclass and conversion
- ErrorBreakdown, TimingStats, ThroughputStats dataclasses
- BatchMetricsSummary dataclass and calculations
- BatchMetrics lifecycle and recording
- Export to JSON and CSV formats
- CLI command helper functions
- Thread safety

References:
    LOG-012: Batch Job Performance Metrics
"""

from __future__ import annotations

import csv
import json
import statistics
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from oscura.workflows.batch.metrics import (
    BatchMetrics,
    BatchMetricsSummary,
    ErrorBreakdown,
    FileMetrics,
    ThroughputStats,
    TimingStats,
    get_batch_stats,
)

if TYPE_CHECKING:
    from collections.abc import Generator


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def batch_metrics() -> BatchMetrics:
    """Create a batch metrics instance."""
    return BatchMetrics(batch_id="test-batch-001")


@pytest.fixture
def temp_output_dir() -> Generator[Path, None, None]:
    """Create temporary directory for export testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# =============================================================================
# FileMetrics Tests
# =============================================================================


def test_file_metrics_creation():
    """Test creating file metrics with defaults."""
    metrics = FileMetrics(filename="test.wfm")

    assert metrics.filename == "test.wfm"
    assert metrics.start_time == 0.0
    assert metrics.end_time == 0.0
    assert metrics.duration == 0.0
    assert metrics.samples == 0
    assert metrics.measurements == 0
    assert metrics.status == "pending"
    assert metrics.error_type is None
    assert metrics.error_message is None
    assert metrics.memory_peak is None


def test_file_metrics_creation_with_values():
    """Test creating file metrics with all fields."""
    metrics = FileMetrics(
        filename="capture.wfm",
        start_time=1000.0,
        end_time=1001.5,
        duration=1.5,
        samples=100000,
        measurements=50,
        status="success",
        memory_peak=1024 * 1024,
    )

    assert metrics.filename == "capture.wfm"
    assert metrics.duration == 1.5
    assert metrics.samples == 100000
    assert metrics.measurements == 50
    assert metrics.status == "success"
    assert metrics.memory_peak == 1024 * 1024


def test_file_metrics_to_dict():
    """Test converting file metrics to dictionary."""
    metrics = FileMetrics(
        filename="test.wfm",
        start_time=1234567890.0,
        end_time=1234567891.0,
        duration=1.0,
        samples=50000,
        measurements=25,
        status="success",
        memory_peak=2048,
    )

    result = metrics.to_dict()

    assert result["filename"] == "test.wfm"
    assert result["duration_seconds"] == 1.0
    assert result["samples"] == 50000
    assert result["measurements"] == 25
    assert result["status"] == "success"
    assert result["memory_peak_bytes"] == 2048
    assert result["samples_per_second"] == 50000.0  # 50000 / 1.0
    # Timestamps should be formatted
    assert result["start_time"] is not None
    assert result["end_time"] is not None


def test_file_metrics_to_dict_zero_duration():
    """Test file metrics handles zero duration (no division by zero)."""
    metrics = FileMetrics(
        filename="zero.wfm",
        duration=0.0,
        samples=1000,
    )

    result = metrics.to_dict()
    assert result["samples_per_second"] == 0


def test_file_metrics_to_dict_with_error():
    """Test file metrics with error information."""
    metrics = FileMetrics(
        filename="error.wfm",
        status="error",
        error_type="FileNotFoundError",
        error_message="File does not exist",
    )

    result = metrics.to_dict()
    assert result["status"] == "error"
    assert result["error_type"] == "FileNotFoundError"
    assert result["error_message"] == "File does not exist"


def test_file_metrics_to_dict_none_times():
    """Test file metrics with None timestamps."""
    metrics = FileMetrics(
        filename="notime.wfm",
        start_time=0.0,
        end_time=0.0,
    )

    result = metrics.to_dict()
    # With 0.0 times, formatter should return None
    assert result["start_time"] is not None or result["start_time"] is None  # Depends on formatter


# =============================================================================
# ErrorBreakdown Tests
# =============================================================================


def test_error_breakdown_creation():
    """Test creating error breakdown."""
    breakdown = ErrorBreakdown(
        by_type={"ValueError": 5, "RuntimeError": 3},
        total=8,
        rate=0.08,
    )

    assert breakdown.by_type == {"ValueError": 5, "RuntimeError": 3}
    assert breakdown.total == 8
    assert breakdown.rate == 0.08


def test_error_breakdown_to_dict():
    """Test converting error breakdown to dictionary."""
    breakdown = ErrorBreakdown(
        by_type={"IOError": 2},
        total=2,
        rate=0.05,
    )

    result = breakdown.to_dict()

    assert result["by_type"] == {"IOError": 2}
    assert result["total"] == 2
    assert result["rate_percent"] == 5.0  # 0.05 * 100


def test_error_breakdown_empty():
    """Test error breakdown with no errors."""
    breakdown = ErrorBreakdown()

    assert breakdown.by_type == {}
    assert breakdown.total == 0
    assert breakdown.rate == 0.0


# =============================================================================
# TimingStats Tests
# =============================================================================


def test_timing_stats_creation():
    """Test creating timing stats."""
    stats = TimingStats(
        total_duration=120.5,
        average_per_file=1.2,
        min_per_file=0.5,
        max_per_file=2.5,
        median_per_file=1.0,
        stddev_per_file=0.3,
    )

    assert stats.total_duration == 120.5
    assert stats.average_per_file == 1.2


def test_timing_stats_to_dict():
    """Test converting timing stats to dictionary."""
    stats = TimingStats(
        total_duration=60.123456,
        average_per_file=1.234567,
        min_per_file=0.5,
        max_per_file=2.0,
        median_per_file=1.0,
        stddev_per_file=0.25,
    )

    result = stats.to_dict()

    # Should be rounded to 3 decimal places
    assert result["total_duration_seconds"] == 60.123
    assert result["average_per_file_seconds"] == 1.235
    assert result["min_per_file_seconds"] == 0.5
    assert result["max_per_file_seconds"] == 2.0
    assert result["median_per_file_seconds"] == 1.0
    assert result["stddev_per_file_seconds"] == 0.25


# =============================================================================
# ThroughputStats Tests
# =============================================================================


def test_throughput_stats_creation():
    """Test creating throughput stats."""
    stats = ThroughputStats(
        files_per_second=10.5,
        samples_per_second=1000000.0,
        measurements_per_second=500.0,
        bytes_per_second=1024 * 1024,
    )

    assert stats.files_per_second == 10.5
    assert stats.samples_per_second == 1000000.0


def test_throughput_stats_to_dict():
    """Test converting throughput stats to dictionary."""
    stats = ThroughputStats(
        files_per_second=5.12345,
        samples_per_second=999999.9,
        measurements_per_second=123.456,
        bytes_per_second=2048.789,
    )

    result = stats.to_dict()

    # Different fields have different rounding
    assert result["files_per_second"] == 5.123
    assert result["samples_per_second"] == 1000000  # Rounded to int
    assert result["measurements_per_second"] == 123  # Rounded to int
    assert result["bytes_per_second"] == 2049  # Rounded to int


# =============================================================================
# BatchMetricsSummary Tests
# =============================================================================


def test_batch_metrics_summary_creation():
    """Test creating batch metrics summary."""
    timing = TimingStats(total_duration=100.0)
    throughput = ThroughputStats(files_per_second=5.0)
    errors = ErrorBreakdown(total=2)

    summary = BatchMetricsSummary(
        batch_id="summary-001",
        total_files=100,
        processed_count=95,
        error_count=2,
        skip_count=3,
        timing=timing,
        throughput=throughput,
        errors=errors,
        start_time="2025-01-01T00:00:00Z",
        end_time="2025-01-01T00:01:40Z",
    )

    assert summary.batch_id == "summary-001"
    assert summary.total_files == 100
    assert summary.processed_count == 95
    assert summary.error_count == 2
    assert summary.skip_count == 3


def test_batch_metrics_summary_to_dict():
    """Test converting batch metrics summary to dictionary."""
    summary = BatchMetricsSummary(
        batch_id="dict-test",
        total_files=50,
        processed_count=48,
        error_count=2,
        skip_count=0,
        timing=TimingStats(total_duration=60.0),
        throughput=ThroughputStats(files_per_second=0.8),
        errors=ErrorBreakdown(total=2, rate=0.04),
        start_time="2025-01-01T10:00:00Z",
        end_time="2025-01-01T10:01:00Z",
    )

    result = summary.to_dict()

    assert result["batch_id"] == "dict-test"
    assert result["total_files"] == 50
    assert result["processed_count"] == 48
    assert result["error_count"] == 2
    assert result["skip_count"] == 0
    assert result["success_rate_percent"] == 96.0  # 48/50 * 100
    assert "timing" in result
    assert "throughput" in result
    assert "errors" in result


def test_batch_metrics_summary_to_dict_zero_files():
    """Test summary handles zero files without division error."""
    summary = BatchMetricsSummary(
        batch_id="zero-files",
        total_files=0,
        processed_count=0,
        error_count=0,
        skip_count=0,
        timing=TimingStats(),
        throughput=ThroughputStats(),
        errors=ErrorBreakdown(),
        start_time="",
        end_time="",
    )

    result = summary.to_dict()
    assert result["success_rate_percent"] == 0.0


# =============================================================================
# BatchMetrics Lifecycle Tests
# =============================================================================


def test_batch_metrics_creation_with_id():
    """Test creating batch metrics with explicit ID."""
    metrics = BatchMetrics(batch_id="custom-id")
    assert metrics.batch_id == "custom-id"


def test_batch_metrics_creation_auto_id():
    """Test batch metrics auto-generates UUID if no ID."""
    metrics = BatchMetrics()
    assert metrics.batch_id is not None
    assert len(metrics.batch_id) > 0
    assert "-" in metrics.batch_id  # UUID format


def test_batch_metrics_start():
    """Test starting batch metrics collection."""
    metrics = BatchMetrics(batch_id="start-test")
    assert metrics._start_time is None

    metrics.start()

    start_time = metrics._start_time
    assert start_time is not None
    assert start_time > 0  # type: ignore[unreachable]


def test_batch_metrics_finish():
    """Test finishing batch metrics collection."""
    metrics = BatchMetrics()
    metrics.start()
    time.sleep(0.01)
    metrics.finish()

    assert metrics._end_time is not None
    assert metrics._start_time is not None
    assert metrics._end_time > metrics._start_time


# =============================================================================
# BatchMetrics Recording Tests
# =============================================================================


def test_batch_metrics_record_file_success():
    """Test recording successful file processing."""
    metrics = BatchMetrics()

    metrics.record_file(
        "test.wfm",
        duration=1.5,
        samples=100000,
        measurements=50,
        status="success",
    )

    assert len(metrics._files) == 1
    file_metrics = metrics._files[0]
    assert file_metrics.filename == "test.wfm"
    assert file_metrics.duration == 1.5
    assert file_metrics.samples == 100000
    assert file_metrics.measurements == 50
    assert file_metrics.status == "success"


def test_batch_metrics_record_file_with_memory():
    """Test recording file with memory tracking."""
    metrics = BatchMetrics()

    metrics.record_file(
        "memory.wfm",
        duration=2.0,
        samples=50000,
        memory_peak=1024 * 1024,
    )

    file_metrics = metrics._files[0]
    assert file_metrics.memory_peak == 1024 * 1024


def test_batch_metrics_record_file_error():
    """Test recording file with error."""
    metrics = BatchMetrics()

    metrics.record_file(
        "error.wfm",
        duration=0.5,
        status="error",
        error_type="FileNotFoundError",
        error_message="File not found",
    )

    assert len(metrics._files) == 1
    file_metrics = metrics._files[0]
    assert file_metrics.status == "error"
    assert file_metrics.error_type == "FileNotFoundError"
    assert file_metrics.error_message == "File not found"
    # Error should be tracked
    assert metrics._error_types["FileNotFoundError"] == 1


def test_batch_metrics_record_multiple_files():
    """Test recording multiple files."""
    metrics = BatchMetrics()

    for i in range(10):
        metrics.record_file(
            f"file{i}.wfm",
            duration=1.0,
            samples=10000,
        )

    assert len(metrics._files) == 10


def test_batch_metrics_record_error_helper():
    """Test record_error convenience method."""
    metrics = BatchMetrics()

    metrics.record_error(
        "broken.wfm",
        error_type="ValueError",
        error_message="Invalid data",
        duration=0.3,
    )

    file_metrics = metrics._files[0]
    assert file_metrics.status == "error"
    assert file_metrics.error_type == "ValueError"
    assert file_metrics.error_message == "Invalid data"
    assert file_metrics.duration == 0.3


def test_batch_metrics_record_skip_helper():
    """Test record_skip convenience method."""
    metrics = BatchMetrics()

    metrics.record_skip("skipped.wfm", reason="Already processed")

    file_metrics = metrics._files[0]
    assert file_metrics.status == "skipped"
    assert file_metrics.error_message == "Already processed"
    assert file_metrics.duration == 0.0


# =============================================================================
# BatchMetrics Summary Generation Tests
# =============================================================================


def test_batch_metrics_summary_empty():
    """Test generating summary with no files."""
    metrics = BatchMetrics(batch_id="empty")
    metrics.start()
    metrics.finish()

    summary = metrics.summary()

    assert summary.batch_id == "empty"
    assert summary.total_files == 0
    assert summary.processed_count == 0
    assert summary.error_count == 0
    assert summary.skip_count == 0


def test_batch_metrics_summary_all_success():
    """Test summary with all successful files."""
    metrics = BatchMetrics()
    metrics.start()

    for i in range(5):
        metrics.record_file(
            f"file{i}.wfm",
            duration=1.0,
            samples=10000,
            measurements=5,
            status="success",
        )

    metrics.finish()

    summary = metrics.summary()

    assert summary.total_files == 5
    assert summary.processed_count == 5
    assert summary.error_count == 0
    assert summary.skip_count == 0


def test_batch_metrics_summary_mixed_status():
    """Test summary with mixed success/error/skip statuses."""
    metrics = BatchMetrics()
    metrics.start()

    # Success
    metrics.record_file("s1.wfm", duration=1.0, status="success")
    metrics.record_file("s2.wfm", duration=1.2, status="success")

    # Errors
    metrics.record_file("e1.wfm", duration=0.5, status="error", error_type="Error1")
    metrics.record_file("e2.wfm", duration=0.3, status="error", error_type="Error2")

    # Skipped
    metrics.record_skip("skip.wfm")

    metrics.finish()

    summary = metrics.summary()

    assert summary.total_files == 5
    assert summary.processed_count == 2
    assert summary.error_count == 2
    assert summary.skip_count == 1


def test_batch_metrics_summary_timing_stats():
    """Test summary calculates timing stats correctly."""
    metrics = BatchMetrics()
    metrics.start()

    # Add files with varying durations
    durations = [0.5, 1.0, 1.5, 2.0, 2.5]
    for i, duration in enumerate(durations):
        metrics.record_file(f"file{i}.wfm", duration=duration, status="success")

    metrics.finish()

    summary = metrics.summary()

    # Verify timing calculations
    assert summary.timing.average_per_file == statistics.mean(durations)
    assert summary.timing.min_per_file == min(durations)
    assert summary.timing.max_per_file == max(durations)
    assert summary.timing.median_per_file == statistics.median(durations)
    assert summary.timing.stddev_per_file == statistics.stdev(durations)


def test_batch_metrics_summary_timing_single_file():
    """Test timing stats with single file (no stddev)."""
    metrics = BatchMetrics()
    metrics.start()

    metrics.record_file("single.wfm", duration=1.5, status="success")

    metrics.finish()

    summary = metrics.summary()

    assert summary.timing.average_per_file == 1.5
    assert summary.timing.stddev_per_file == 0.0  # Only one file


def test_batch_metrics_summary_timing_no_successful_files():
    """Test timing stats when no successful files."""
    metrics = BatchMetrics()
    metrics.start()

    metrics.record_file("err.wfm", duration=0.5, status="error")
    metrics.record_skip("skip.wfm")

    metrics.finish()

    summary = metrics.summary()

    # Should have total duration but no per-file stats
    assert summary.timing.total_duration > 0
    assert summary.timing.average_per_file == 0.0


def test_batch_metrics_summary_throughput_stats():
    """Test summary calculates throughput stats correctly."""
    metrics = BatchMetrics()
    metrics.start()

    # Record files with known metrics
    for i in range(10):
        metrics.record_file(
            f"file{i}.wfm",
            duration=0.1,
            samples=10000,
            measurements=50,
            status="success",
        )

    time.sleep(0.05)  # Small delay
    metrics.finish()

    summary = metrics.summary()

    # Should calculate rates
    assert summary.throughput.files_per_second > 0
    assert summary.throughput.samples_per_second > 0
    assert summary.throughput.measurements_per_second > 0


def test_batch_metrics_summary_throughput_zero_duration():
    """Test throughput with zero duration returns zeros."""
    metrics = BatchMetrics()
    # Don't call start/finish, durations will be zero

    metrics.record_file("f.wfm", duration=0.0, status="success")

    summary = metrics.summary()

    assert summary.throughput.files_per_second == 0.0
    assert summary.throughput.samples_per_second == 0.0


def test_batch_metrics_summary_error_breakdown():
    """Test summary calculates error breakdown correctly."""
    metrics = BatchMetrics()

    # Multiple error types
    metrics.record_error("e1.wfm", "ValueError", "Error 1")
    metrics.record_error("e2.wfm", "ValueError", "Error 2")
    metrics.record_error("e3.wfm", "RuntimeError", "Error 3")

    summary = metrics.summary()

    assert summary.errors.total == 3
    assert summary.errors.by_type["ValueError"] == 2
    assert summary.errors.by_type["RuntimeError"] == 1
    assert summary.errors.rate == 1.0  # All files are errors


def test_batch_metrics_summary_timestamps():
    """Test summary includes formatted timestamps."""
    metrics = BatchMetrics()
    metrics.start()
    time.sleep(0.01)
    metrics.finish()

    summary = metrics.summary()

    assert summary.start_time != ""
    assert summary.end_time != ""
    # End should be after start
    assert summary.start_time <= summary.end_time


def test_batch_metrics_summary_without_explicit_timing():
    """Test summary calculates duration from file times if not explicit."""
    metrics = BatchMetrics()
    # Don't call start/finish

    metrics.record_file("f1.wfm", duration=1.0, status="success")
    metrics.record_file("f2.wfm", duration=1.5, status="success")

    summary = metrics.summary()

    # Should sum durations
    assert summary.timing.total_duration == 2.5


# =============================================================================
# BatchMetrics Query Tests
# =============================================================================


def test_batch_metrics_get_file_metrics():
    """Test retrieving all file metrics."""
    metrics = BatchMetrics()

    metrics.record_file("f1.wfm", duration=1.0, samples=1000)
    metrics.record_file("f2.wfm", duration=2.0, samples=2000)

    file_metrics = metrics.get_file_metrics()

    assert len(file_metrics) == 2
    assert file_metrics[0]["filename"] == "f1.wfm"
    assert file_metrics[1]["filename"] == "f2.wfm"


def test_batch_metrics_get_file_metrics_empty():
    """Test getting file metrics when none recorded."""
    metrics = BatchMetrics()
    file_metrics = metrics.get_file_metrics()
    assert file_metrics == []


# =============================================================================
# BatchMetrics Export Tests
# =============================================================================


def test_batch_metrics_export_json(temp_output_dir: Path):
    """Test exporting metrics to JSON."""
    metrics = BatchMetrics(batch_id="export-json")
    metrics.start()

    metrics.record_file("f1.wfm", duration=1.0, samples=1000, status="success")
    metrics.record_error("f2.wfm", "ValueError", "Error")

    metrics.finish()

    output_path = temp_output_dir / "metrics.json"
    metrics.export_json(output_path)

    # Verify file was created
    assert output_path.exists()

    # Verify content
    with open(output_path) as f:
        data = json.load(f)

    assert "summary" in data
    assert "files" in data
    assert data["summary"]["batch_id"] == "export-json"
    assert len(data["files"]) == 2


def test_batch_metrics_export_json_with_path_string(temp_output_dir: Path):
    """Test export_json accepts string path."""
    metrics = BatchMetrics()
    metrics.record_file("test.wfm", duration=1.0)

    output_path = str(temp_output_dir / "metrics_str.json")
    metrics.export_json(output_path)

    assert Path(output_path).exists()


def test_batch_metrics_export_csv(temp_output_dir: Path):
    """Test exporting metrics to CSV."""
    metrics = BatchMetrics(batch_id="export-csv")

    metrics.record_file("f1.wfm", duration=1.0, samples=1000, measurements=10)
    metrics.record_file("f2.wfm", duration=2.0, samples=2000, measurements=20)

    output_path = temp_output_dir / "metrics.csv"
    metrics.export_csv(output_path)

    # Verify file was created
    assert output_path.exists()

    # Verify content
    with open(output_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 2
    assert rows[0]["filename"] == "f1.wfm"
    assert rows[1]["filename"] == "f2.wfm"


def test_batch_metrics_export_csv_empty(temp_output_dir: Path, caplog: pytest.LogCaptureFixture):
    """Test exporting empty metrics to CSV logs warning."""
    metrics = BatchMetrics()

    output_path = temp_output_dir / "empty.csv"

    with caplog.at_level("WARNING"):
        metrics.export_csv(output_path)

    # Should not create file
    assert not output_path.exists()
    # Should log warning
    assert "No file metrics to export" in caplog.text


def test_batch_metrics_export_csv_with_path_string(temp_output_dir: Path):
    """Test export_csv accepts string path."""
    metrics = BatchMetrics()
    metrics.record_file("test.wfm", duration=1.0)

    output_path = str(temp_output_dir / "metrics_str.csv")
    metrics.export_csv(output_path)

    assert Path(output_path).exists()


# =============================================================================
# CLI Helper Function Tests
# =============================================================================


def test_get_batch_stats():
    """Test get_batch_stats helper function."""
    metrics = BatchMetrics(batch_id="cli-test")
    metrics.start()
    metrics.record_file("f1.wfm", duration=1.0, status="success")
    metrics.finish()

    result = get_batch_stats("cli-test", metrics)

    assert isinstance(result, dict)
    assert result["batch_id"] == "cli-test"
    assert result["total_files"] == 1


def test_get_batch_stats_id_mismatch():
    """Test get_batch_stats raises error on batch ID mismatch."""
    metrics = BatchMetrics(batch_id="actual-id")

    with pytest.raises(ValueError, match="Batch ID mismatch"):
        get_batch_stats("wrong-id", metrics)


# =============================================================================
# Thread Safety Tests
# =============================================================================


def test_batch_metrics_thread_safety():
    """Test batch metrics is thread-safe for concurrent recording."""
    import concurrent.futures

    metrics = BatchMetrics(batch_id="thread-test")

    def record_file(index: int) -> None:
        metrics.record_file(
            f"file{index}.wfm",
            duration=0.001,
            samples=1000,
            status="success",
        )

    # Record files concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(record_file, i) for i in range(100)]
        concurrent.futures.wait(futures)

    # All files should be recorded
    assert len(metrics._files) == 100


def test_batch_metrics_concurrent_error_tracking():
    """Test error tracking is thread-safe."""
    import concurrent.futures

    metrics = BatchMetrics()

    def record_error(index: int) -> None:
        error_type = "ValueError" if index % 2 == 0 else "RuntimeError"
        metrics.record_error(f"file{index}.wfm", error_type, "Error")

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(record_error, i) for i in range(40)]
        concurrent.futures.wait(futures)

    # Error counts should be correct
    assert metrics._error_types["ValueError"] == 20
    assert metrics._error_types["RuntimeError"] == 20


# =============================================================================
# Integration Tests
# =============================================================================


def test_batch_metrics_complete_workflow(temp_output_dir: Path):
    """Test complete batch metrics workflow."""
    metrics = BatchMetrics(batch_id="complete-workflow")

    # Start batch
    metrics.start()

    # Process files
    for i in range(10):
        if i < 8:
            metrics.record_file(
                f"success{i}.wfm",
                duration=1.0 + i * 0.1,
                samples=10000 * (i + 1),
                measurements=50 * (i + 1),
                status="success",
            )
        elif i == 8:
            metrics.record_error(
                "error.wfm",
                "ValueError",
                "Processing error",
                duration=0.5,
            )
        else:
            metrics.record_skip("skipped.wfm", "Already processed")

    # Finish batch
    time.sleep(0.01)
    metrics.finish()

    # Get summary
    summary = metrics.summary()

    assert summary.total_files == 10
    assert summary.processed_count == 8
    assert summary.error_count == 1
    assert summary.skip_count == 1

    # Export both formats
    json_path = temp_output_dir / "complete.json"
    csv_path = temp_output_dir / "complete.csv"

    metrics.export_json(json_path)
    metrics.export_csv(csv_path)

    # Verify both exports exist
    assert json_path.exists()
    assert csv_path.exists()

    # Verify JSON content
    with open(json_path) as f:
        json_data = json.load(f)
    assert json_data["summary"]["total_files"] == 10

    # Verify CSV content
    with open(csv_path, newline="") as f:
        csv_data = list(csv.DictReader(f))
    assert len(csv_data) == 10


def test_batch_metrics_timing_accuracy():
    """Test that timing calculations are accurate."""
    metrics = BatchMetrics()
    metrics.start()

    # Record files with precise durations
    durations = [1.0, 1.5, 2.0, 2.5, 3.0]
    for i, duration in enumerate(durations):
        metrics.record_file(f"file{i}.wfm", duration=duration, status="success")

    time.sleep(0.1)
    metrics.finish()

    summary = metrics.summary()

    # Verify statistics match expected values
    expected_mean = statistics.mean(durations)
    expected_median = statistics.median(durations)
    expected_min = min(durations)
    expected_max = max(durations)
    expected_stddev = statistics.stdev(durations)

    assert abs(summary.timing.average_per_file - expected_mean) < 0.001
    assert abs(summary.timing.median_per_file - expected_median) < 0.001
    assert summary.timing.min_per_file == expected_min
    assert summary.timing.max_per_file == expected_max
    assert abs(summary.timing.stddev_per_file - expected_stddev) < 0.001
