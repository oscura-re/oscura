"""Batch job performance metrics collection and export.

This module provides comprehensive metrics collection for batch processing jobs
including throughput, timing, error statistics, and export capabilities.


Example:
    >>> from oscura.workflows.batch.metrics import BatchMetrics
    >>> metrics = BatchMetrics(batch_id="job-001")
    >>> metrics.start()
    >>> metrics.record_file("file1.wfm", duration=0.5, samples=100000)
    >>> metrics.record_file("file2.wfm", duration=0.3, samples=50000)
    >>> summary = metrics.summary()
    >>> metrics.export_json("metrics.json")

References:
"""

from __future__ import annotations

import csv
import json
import statistics
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from oscura.core.logging import format_timestamp, get_logger

logger = get_logger(__name__)


@dataclass
class FileMetrics:
    """Metrics for a single file in a batch job.

    Attributes:
        filename: Path to the file.
        start_time: Processing start time (epoch seconds).
        end_time: Processing end time (epoch seconds).
        duration: Processing duration in seconds.
        samples: Number of samples processed.
        measurements: Number of measurements computed.
        status: Processing status (success, error, skipped).
        error_type: Error type if status is 'error'.
        error_message: Error message if status is 'error'.
        memory_peak: Peak memory usage in bytes (if tracked).

    References:
        LOG-012: Batch Job Performance Metrics
    """

    filename: str
    start_time: float = 0.0
    end_time: float = 0.0
    duration: float = 0.0
    samples: int = 0
    measurements: int = 0
    status: str = "pending"
    error_type: str | None = None
    error_message: str | None = None
    memory_peak: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "filename": self.filename,
            "start_time": format_timestamp(datetime.fromtimestamp(self.start_time, tz=UTC))
            if self.start_time
            else None,
            "end_time": format_timestamp(datetime.fromtimestamp(self.end_time, tz=UTC))
            if self.end_time
            else None,
            "duration_seconds": self.duration,
            "samples": self.samples,
            "measurements": self.measurements,
            "status": self.status,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "memory_peak_bytes": self.memory_peak,
            "samples_per_second": self.samples / self.duration if self.duration > 0 else 0,
        }


@dataclass
class ErrorBreakdown:
    """Breakdown of errors by type.

    Attributes:
        by_type: Count of errors grouped by error type.
        total: Total number of errors.
        rate: Error rate as percentage.

    References:
        LOG-012: Error breakdown requirement
    """

    by_type: dict[str, int] = field(default_factory=dict)
    total: int = 0
    rate: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "by_type": self.by_type,
            "total": self.total,
            "rate_percent": round(self.rate * 100, 2),
        }


@dataclass
class TimingStats:
    """Timing statistics for batch processing.

    Attributes:
        total_duration: Total wall-clock duration in seconds.
        average_per_file: Average processing time per file.
        min_per_file: Minimum processing time per file.
        max_per_file: Maximum processing time per file.
        median_per_file: Median processing time per file.
        stddev_per_file: Standard deviation of processing times.

    References:
        LOG-012: Timing metrics requirement
    """

    total_duration: float = 0.0
    average_per_file: float = 0.0
    min_per_file: float = 0.0
    max_per_file: float = 0.0
    median_per_file: float = 0.0
    stddev_per_file: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "total_duration_seconds": round(self.total_duration, 3),
            "average_per_file_seconds": round(self.average_per_file, 3),
            "min_per_file_seconds": round(self.min_per_file, 3),
            "max_per_file_seconds": round(self.max_per_file, 3),
            "median_per_file_seconds": round(self.median_per_file, 3),
            "stddev_per_file_seconds": round(self.stddev_per_file, 3),
        }


@dataclass
class ThroughputStats:
    """Throughput statistics for batch processing.

    Attributes:
        files_per_second: Processing rate in files per second.
        samples_per_second: Processing rate in samples per second.
        measurements_per_second: Rate of measurements computed per second.
        bytes_per_second: Data processing rate (if tracked).

    References:
        LOG-012: Throughput metrics requirement
    """

    files_per_second: float = 0.0
    samples_per_second: float = 0.0
    measurements_per_second: float = 0.0
    bytes_per_second: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "files_per_second": round(self.files_per_second, 3),
            "samples_per_second": round(self.samples_per_second, 0),
            "measurements_per_second": round(self.measurements_per_second, 0),
            "bytes_per_second": round(self.bytes_per_second, 0),
        }


@dataclass
class BatchMetricsSummary:
    """Complete summary of batch processing metrics.

    Attributes:
        batch_id: Unique batch job identifier.
        total_files: Total number of files in the batch.
        processed_count: Number of successfully processed files.
        error_count: Number of files with errors.
        skip_count: Number of skipped files.
        timing: Timing statistics.
        throughput: Throughput statistics.
        errors: Error breakdown.
        start_time: Batch start time (ISO 8601).
        end_time: Batch end time (ISO 8601).

    References:
        LOG-012: Batch Job Performance Metrics
    """

    batch_id: str
    total_files: int
    processed_count: int
    error_count: int
    skip_count: int
    timing: TimingStats
    throughput: ThroughputStats
    errors: ErrorBreakdown
    start_time: str
    end_time: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "batch_id": self.batch_id,
            "total_files": self.total_files,
            "processed_count": self.processed_count,
            "error_count": self.error_count,
            "skip_count": self.skip_count,
            "success_rate_percent": round(
                (self.processed_count / self.total_files * 100) if self.total_files > 0 else 0, 2
            ),
            "timing": self.timing.to_dict(),
            "throughput": self.throughput.to_dict(),
            "errors": self.errors.to_dict(),
            "start_time": self.start_time,
            "end_time": self.end_time,
        }


class BatchMetrics:
    """Batch job performance metrics collector.

    Collects and aggregates performance metrics for batch processing jobs
    including throughput, timing, and error statistics.

    Example:
        >>> metrics = BatchMetrics(batch_id="analysis-001")
        >>> metrics.start()
        >>> for file in files:
        ...     metrics.record_file(file, duration=0.5, samples=100000)
        >>> metrics.finish()
        >>> summary = metrics.summary()
        >>> metrics.export_json("metrics.json")

    References:
        LOG-012: Batch Job Performance Metrics
    """

    def __init__(self, batch_id: str | None = None) -> None:
        """Initialize batch metrics collector.

        Args:
            batch_id: Unique batch job identifier. Auto-generated if None.
        """
        import uuid

        self.batch_id = batch_id or str(uuid.uuid4())
        self._files: list[FileMetrics] = []
        self._lock = threading.Lock()
        self._start_time: float | None = None
        self._end_time: float | None = None
        self._error_types: dict[str, int] = {}

    def start(self) -> None:
        """Mark batch job as started.

        Records the start time for duration calculation.
        """
        self._start_time = time.time()
        logger.info(
            "Batch metrics collection started",
            extra={"batch_id": self.batch_id},
        )

    def finish(self) -> None:
        """Mark batch job as finished.

        Records the end time and logs the completion summary.
        """
        self._end_time = time.time()
        summary = self.summary()
        logger.info(
            "Batch metrics collection finished",
            extra={
                "batch_id": self.batch_id,
                "total_files": summary.total_files,
                "processed": summary.processed_count,
                "errors": summary.error_count,
                "duration": summary.timing.total_duration,
            },
        )

    def record_file(
        self,
        filename: str,
        *,
        duration: float,
        samples: int = 0,
        measurements: int = 0,
        status: str = "success",
        error_type: str | None = None,
        error_message: str | None = None,
        memory_peak: int | None = None,
    ) -> None:
        """Record metrics for a processed file.

        Args:
            filename: Path to the processed file.
            duration: Processing duration in seconds.
            samples: Number of samples processed.
            measurements: Number of measurements computed.
            status: Processing status (success, error, skipped).
            error_type: Error type if status is 'error'.
            error_message: Error message if status is 'error'.
            memory_peak: Peak memory usage in bytes.

        References:
            LOG-012: Per-file metrics tracking
        """
        now = time.time()
        file_metrics = FileMetrics(
            filename=filename,
            start_time=now - duration,
            end_time=now,
            duration=duration,
            samples=samples,
            measurements=measurements,
            status=status,
            error_type=error_type,
            error_message=error_message,
            memory_peak=memory_peak,
        )

        with self._lock:
            self._files.append(file_metrics)
            if status == "error" and error_type:
                self._error_types[error_type] = self._error_types.get(error_type, 0) + 1

    def record_error(
        self,
        filename: str,
        error_type: str,
        error_message: str,
        duration: float = 0.0,
    ) -> None:
        """Record a file processing error.

        Args:
            filename: Path to the file that failed.
            error_type: Type of error (e.g., "FileNotFoundError").
            error_message: Detailed error message.
            duration: Processing duration before error.

        References:
            LOG-012: Error tracking
        """
        self.record_file(
            filename,
            duration=duration,
            status="error",
            error_type=error_type,
            error_message=error_message,
        )

    def record_skip(self, filename: str, reason: str = "") -> None:
        """Record a skipped file.

        Args:
            filename: Path to the skipped file.
            reason: Reason for skipping.

        References:
            LOG-012: Skip count tracking
        """
        self.record_file(
            filename,
            duration=0.0,
            status="skipped",
            error_message=reason,
        )

    def summary(self) -> BatchMetricsSummary:
        """Generate batch processing metrics summary.

        Returns:
            BatchMetricsSummary with aggregated statistics.

        References:
            LOG-012: Batch Job Performance Metrics
        """
        with self._lock:
            files = list(self._files)
            error_types = dict(self._error_types)

        # Count by status
        processed_count = sum(1 for f in files if f.status == "success")
        error_count = sum(1 for f in files if f.status == "error")
        skip_count = sum(1 for f in files if f.status == "skipped")
        total_files = len(files)

        # Collect durations for successful files
        durations = [f.duration for f in files if f.status == "success" and f.duration > 0]

        # Calculate timing stats
        if durations:
            timing = TimingStats(
                total_duration=self._end_time - self._start_time
                if self._start_time and self._end_time
                else sum(durations),
                average_per_file=statistics.mean(durations),
                min_per_file=min(durations),
                max_per_file=max(durations),
                median_per_file=statistics.median(durations),
                stddev_per_file=statistics.stdev(durations) if len(durations) > 1 else 0.0,
            )
        else:
            timing = TimingStats(
                total_duration=self._end_time - self._start_time
                if self._start_time and self._end_time
                else 0.0
            )

        # Calculate throughput stats
        total_samples = sum(f.samples for f in files if f.status == "success")
        total_measurements = sum(f.measurements for f in files if f.status == "success")

        if timing.total_duration > 0:
            throughput = ThroughputStats(
                files_per_second=processed_count / timing.total_duration,
                samples_per_second=total_samples / timing.total_duration,
                measurements_per_second=total_measurements / timing.total_duration,
            )
        else:
            throughput = ThroughputStats()

        # Error breakdown
        errors = ErrorBreakdown(
            by_type=error_types,
            total=error_count,
            rate=error_count / total_files if total_files > 0 else 0.0,
        )

        # Timestamps
        start_time = (
            format_timestamp(datetime.fromtimestamp(self._start_time, tz=UTC))
            if self._start_time
            else ""
        )
        end_time = (
            format_timestamp(datetime.fromtimestamp(self._end_time, tz=UTC))
            if self._end_time
            else ""
        )

        return BatchMetricsSummary(
            batch_id=self.batch_id,
            total_files=total_files,
            processed_count=processed_count,
            error_count=error_count,
            skip_count=skip_count,
            timing=timing,
            throughput=throughput,
            errors=errors,
            start_time=start_time,
            end_time=end_time,
        )

    def get_file_metrics(self) -> list[dict[str, Any]]:
        """Get metrics for all files.

        Returns:
            List of file metric dictionaries.
        """
        with self._lock:
            return [f.to_dict() for f in self._files]

    def export_json(self, path: str | Path) -> None:
        """Export metrics to JSON file.

        Args:
            path: Output file path.

        References:
            LOG-012: Export batch metrics as JSON
        """
        path = Path(path)
        summary = self.summary()

        output = {
            "summary": summary.to_dict(),
            "files": self.get_file_metrics(),
        }

        with open(path, "w") as f:
            json.dump(output, f, indent=2, default=str)

        logger.info(f"Batch metrics exported to {path}")

    def export_csv(self, path: str | Path) -> None:
        """Export per-file metrics to CSV file.

        Args:
            path: Output file path.

        References:
            LOG-012: Export batch metrics as CSV
        """
        path = Path(path)
        files = self.get_file_metrics()

        if not files:
            logger.warning("No file metrics to export")
            return

        # Get all keys from first file
        fieldnames = list(files[0].keys())

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(files)

        logger.info(f"Batch metrics CSV exported to {path}")


def get_batch_stats(batch_id: str, metrics: BatchMetrics) -> dict[str, Any]:
    """Get statistics for a batch job.

    CLI command implementation: oscura batch stats <batch_id>

    Args:
        batch_id: Batch job identifier.
        metrics: BatchMetrics instance.

    Returns:
        Dictionary with batch statistics.

    Raises:
        ValueError: If batch ID does not match the metrics instance.

    References:
        LOG-012: CLI command requirement
    """
    if metrics.batch_id != batch_id:
        raise ValueError(f"Batch ID mismatch: expected {batch_id}, got {metrics.batch_id}")

    return metrics.summary().to_dict()


__all__ = [
    "BatchMetrics",
    "BatchMetricsSummary",
    "ErrorBreakdown",
    "FileMetrics",
    "ThroughputStats",
    "TimingStats",
    "get_batch_stats",
]
