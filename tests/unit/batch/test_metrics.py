"""Unit tests for batch metrics module.

Tests LOG-012: Batch Job Performance Metrics
"""

import json
import tempfile
from pathlib import Path

import pytest

from oscura.batch.metrics import (
    BatchMetrics,
    BatchMetricsSummary,
    ErrorBreakdown,
    FileMetrics,
    ThroughputStats,
    TimingStats,
    get_batch_stats,
)

pytestmark = pytest.mark.unit


class TestFileMetrics:
    """Test FileMetrics dataclass."""

    def test_basic_creation(self) -> None:
        """Test creating basic file metrics."""
        metrics = FileMetrics(
            filename="test.wfm",
            duration=1.5,
            samples=100000,
            status="success",
        )

        assert metrics.filename == "test.wfm"
        assert metrics.duration == 1.5
        assert metrics.samples == 100000
        assert metrics.status == "success"

    def test_to_dict(self) -> None:
        """Test dictionary conversion."""
        metrics = FileMetrics(
            filename="test.wfm",
            start_time=1000.0,
            end_time=1001.5,
            duration=1.5,
            samples=100000,
            measurements=10,
            status="success",
        )

        result = metrics.to_dict()

        assert result["filename"] == "test.wfm"
        assert result["duration_seconds"] == 1.5
        assert result["samples"] == 100000
        assert result["measurements"] == 10
        assert result["status"] == "success"
        assert result["samples_per_second"] == pytest.approx(66666.67, rel=0.01)

    def test_error_metrics(self) -> None:
        """Test error file metrics."""
        metrics = FileMetrics(
            filename="broken.wfm",
            duration=0.1,
            status="error",
            error_type="FileNotFoundError",
            error_message="File not found",
        )

        result = metrics.to_dict()

        assert result["status"] == "error"
        assert result["error_type"] == "FileNotFoundError"
        assert result["error_message"] == "File not found"


class TestTimingStats:
    """Test TimingStats dataclass."""

    def test_basic_creation(self) -> None:
        """Test creating timing stats."""
        stats = TimingStats(
            total_duration=10.0,
            average_per_file=0.5,
            min_per_file=0.1,
            max_per_file=1.0,
            median_per_file=0.4,
            stddev_per_file=0.2,
        )

        assert stats.total_duration == 10.0
        assert stats.average_per_file == 0.5

    def test_to_dict(self) -> None:
        """Test dictionary conversion."""
        stats = TimingStats(
            total_duration=10.0,
            average_per_file=0.5,
        )

        result = stats.to_dict()

        assert result["total_duration_seconds"] == 10.0
        assert result["average_per_file_seconds"] == 0.5


class TestThroughputStats:
    """Test ThroughputStats dataclass."""

    def test_basic_creation(self) -> None:
        """Test creating throughput stats."""
        stats = ThroughputStats(
            files_per_second=2.0,
            samples_per_second=200000.0,
            measurements_per_second=20.0,
        )

        assert stats.files_per_second == 2.0
        assert stats.samples_per_second == 200000.0

    def test_to_dict(self) -> None:
        """Test dictionary conversion."""
        stats = ThroughputStats(
            files_per_second=2.5,
            samples_per_second=250000.0,
        )

        result = stats.to_dict()

        assert result["files_per_second"] == 2.5
        assert result["samples_per_second"] == 250000.0


class TestErrorBreakdown:
    """Test ErrorBreakdown dataclass."""

    def test_basic_creation(self) -> None:
        """Test creating error breakdown."""
        errors = ErrorBreakdown(
            by_type={"FileNotFoundError": 2, "ValueError": 1},
            total=3,
            rate=0.15,
        )

        assert errors.total == 3
        assert errors.by_type["FileNotFoundError"] == 2

    def test_to_dict(self) -> None:
        """Test dictionary conversion."""
        errors = ErrorBreakdown(
            by_type={"FileNotFoundError": 2},
            total=2,
            rate=0.1,
        )

        result = errors.to_dict()

        assert result["total"] == 2
        assert result["rate_percent"] == 10.0
        assert result["by_type"]["FileNotFoundError"] == 2


class TestBatchMetrics:
    """Test BatchMetrics class."""

    def test_basic_creation(self) -> None:
        """Test creating batch metrics with auto-generated ID."""
        metrics = BatchMetrics()

        assert metrics.batch_id is not None
        assert len(metrics.batch_id) == 36  # UUID format

    def test_custom_batch_id(self) -> None:
        """Test creating batch metrics with custom ID."""
        metrics = BatchMetrics(batch_id="custom-001")

        assert metrics.batch_id == "custom-001"

    def test_start_finish(self) -> None:
        """Test start and finish timing."""
        metrics = BatchMetrics(batch_id="test-001")
        metrics.start()
        metrics.finish()

        summary = metrics.summary()
        assert summary.timing.total_duration >= 0

    def test_record_file(self) -> None:
        """Test recording file metrics."""
        metrics = BatchMetrics(batch_id="test-001")
        metrics.start()

        metrics.record_file(
            "file1.wfm",
            duration=0.5,
            samples=100000,
            measurements=10,
        )
        metrics.record_file(
            "file2.wfm",
            duration=0.3,
            samples=50000,
            measurements=5,
        )

        metrics.finish()
        summary = metrics.summary()

        assert summary.total_files == 2
        assert summary.processed_count == 2
        assert summary.error_count == 0

    def test_record_error(self) -> None:
        """Test recording file errors."""
        metrics = BatchMetrics(batch_id="test-001")
        metrics.start()

        metrics.record_file(
            "good.wfm",
            duration=0.5,
            samples=100000,
        )
        metrics.record_error(
            "bad.wfm",
            error_type="FileNotFoundError",
            error_message="File not found",
            duration=0.1,
        )

        metrics.finish()
        summary = metrics.summary()

        assert summary.total_files == 2
        assert summary.processed_count == 1
        assert summary.error_count == 1
        assert summary.errors.by_type["FileNotFoundError"] == 1

    def test_record_skip(self) -> None:
        """Test recording skipped files."""
        metrics = BatchMetrics(batch_id="test-001")
        metrics.start()

        metrics.record_file("processed.wfm", duration=0.5)
        metrics.record_skip("skipped.wfm", reason="Already processed")

        metrics.finish()
        summary = metrics.summary()

        assert summary.total_files == 2
        assert summary.processed_count == 1
        assert summary.skip_count == 1

    def test_timing_statistics(self) -> None:
        """Test timing statistics calculation."""
        metrics = BatchMetrics(batch_id="test-001")
        metrics.start()

        # Record files with known durations
        metrics.record_file("file1.wfm", duration=0.5, samples=1000)
        metrics.record_file("file2.wfm", duration=0.3, samples=1000)
        metrics.record_file("file3.wfm", duration=0.4, samples=1000)

        metrics.finish()
        summary = metrics.summary()

        assert summary.timing.min_per_file == pytest.approx(0.3, rel=0.01)
        assert summary.timing.max_per_file == pytest.approx(0.5, rel=0.01)
        assert summary.timing.average_per_file == pytest.approx(0.4, rel=0.01)
        assert summary.timing.median_per_file == pytest.approx(0.4, rel=0.01)

    def test_throughput_calculation(self) -> None:
        """Test throughput statistics calculation."""
        metrics = BatchMetrics(batch_id="test-001")
        metrics.start()

        # Record files
        metrics.record_file("file1.wfm", duration=0.5, samples=100000, measurements=10)
        metrics.record_file("file2.wfm", duration=0.5, samples=100000, measurements=10)

        metrics.finish()
        summary = metrics.summary()

        # Should have processed 2 files with 200000 samples total
        assert summary.throughput.files_per_second > 0
        assert summary.throughput.samples_per_second > 0
        assert summary.throughput.measurements_per_second > 0

    def test_get_file_metrics(self) -> None:
        """Test getting file metrics list."""
        metrics = BatchMetrics(batch_id="test-001")
        metrics.record_file("file1.wfm", duration=0.5, samples=1000)
        metrics.record_file("file2.wfm", duration=0.3, samples=2000)

        file_metrics = metrics.get_file_metrics()

        assert len(file_metrics) == 2
        assert file_metrics[0]["filename"] == "file1.wfm"
        assert file_metrics[1]["filename"] == "file2.wfm"

    def test_export_json(self) -> None:
        """Test exporting metrics to JSON."""
        metrics = BatchMetrics(batch_id="test-001")
        metrics.start()
        metrics.record_file("file1.wfm", duration=0.5, samples=1000)
        metrics.finish()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            output_path = Path(f.name)

        try:
            metrics.export_json(output_path)

            assert output_path.exists()

            with open(output_path) as f:
                data = json.load(f)

            assert "summary" in data
            assert "files" in data
            assert data["summary"]["batch_id"] == "test-001"
            assert len(data["files"]) == 1
        finally:
            output_path.unlink()

    def test_export_csv(self) -> None:
        """Test exporting metrics to CSV."""
        metrics = BatchMetrics(batch_id="test-001")
        metrics.start()
        metrics.record_file("file1.wfm", duration=0.5, samples=1000)
        metrics.record_file("file2.wfm", duration=0.3, samples=2000)
        metrics.finish()

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            output_path = Path(f.name)

        try:
            metrics.export_csv(output_path)

            assert output_path.exists()

            content = output_path.read_text()
            lines = content.strip().split("\n")

            # Header + 2 data rows
            assert len(lines) == 3
            assert "filename" in lines[0]
            assert "file1.wfm" in lines[1]
            assert "file2.wfm" in lines[2]
        finally:
            output_path.unlink()

    def test_thread_safety(self) -> None:
        """Test thread-safe recording."""
        import threading

        metrics = BatchMetrics(batch_id="test-001")
        metrics.start()

        def record_files(prefix: str) -> None:
            for i in range(10):
                metrics.record_file(f"{prefix}_{i}.wfm", duration=0.01, samples=100)

        threads = [threading.Thread(target=record_files, args=(f"t{i}",)) for i in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        metrics.finish()
        summary = metrics.summary()

        # Should have 50 files (5 threads x 10 files each)
        assert summary.total_files == 50


class TestBatchMetricsSummary:
    """Test BatchMetricsSummary dataclass."""

    def test_to_dict(self) -> None:
        """Test dictionary conversion."""
        summary = BatchMetricsSummary(
            batch_id="test-001",
            total_files=10,
            processed_count=8,
            error_count=2,
            skip_count=0,
            timing=TimingStats(total_duration=5.0),
            throughput=ThroughputStats(files_per_second=2.0),
            errors=ErrorBreakdown(total=2, rate=0.2),
            start_time="2025-01-01T00:00:00Z",
            end_time="2025-01-01T00:00:05Z",
        )

        result = summary.to_dict()

        assert result["batch_id"] == "test-001"
        assert result["total_files"] == 10
        assert result["processed_count"] == 8
        assert result["error_count"] == 2
        assert result["success_rate_percent"] == 80.0
        assert "timing" in result
        assert "throughput" in result
        assert "errors" in result


class TestGetBatchStats:
    """Test get_batch_stats function."""

    def test_get_stats(self) -> None:
        """Test getting batch stats."""
        metrics = BatchMetrics(batch_id="job-001")
        metrics.start()
        metrics.record_file("file1.wfm", duration=0.5, samples=1000)
        metrics.finish()

        stats = get_batch_stats("job-001", metrics)

        assert stats["batch_id"] == "job-001"
        assert stats["total_files"] == 1

    def test_batch_id_mismatch(self) -> None:
        """Test error on batch ID mismatch."""
        metrics = BatchMetrics(batch_id="job-001")

        with pytest.raises(ValueError, match="mismatch"):
            get_batch_stats("wrong-id", metrics)


class TestMetricsIntegration:
    """Integration tests for batch metrics."""

    def test_full_workflow(self) -> None:
        """Test complete metrics workflow."""
        # Create metrics collector
        metrics = BatchMetrics(batch_id="integration-test")

        # Start batch
        metrics.start()

        # Process some files
        files = [
            ("capture1.wfm", 0.5, 100000, 10),
            ("capture2.wfm", 0.3, 50000, 5),
            ("capture3.wfm", 0.7, 150000, 15),
        ]

        for filename, duration, samples, measurements in files:
            metrics.record_file(
                filename,
                duration=duration,
                samples=samples,
                measurements=measurements,
            )

        # Record an error
        metrics.record_error(
            "broken.wfm",
            error_type="ValueError",
            error_message="Invalid format",
        )

        # Record a skip
        metrics.record_skip("duplicate.wfm", reason="Already processed")

        # Finish batch
        metrics.finish()

        # Get summary
        summary = metrics.summary()

        # Verify counts
        assert summary.total_files == 5
        assert summary.processed_count == 3
        assert summary.error_count == 1
        assert summary.skip_count == 1

        # Verify timing
        assert summary.timing.total_duration > 0
        assert summary.timing.average_per_file > 0

        # Verify throughput
        assert summary.throughput.files_per_second > 0
        assert summary.throughput.samples_per_second > 0

        # Verify error breakdown
        assert summary.errors.total == 1
        assert summary.errors.by_type["ValueError"] == 1

        # Verify export
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "metrics.json"
            csv_path = Path(tmpdir) / "metrics.csv"

            metrics.export_json(json_path)
            metrics.export_csv(csv_path)

            assert json_path.exists()
            assert csv_path.exists()
