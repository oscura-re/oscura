"""Aggregate logging for batch processing operations.

This module provides consolidated logging for parallel batch workers
with job-level summaries and per-file tracking.


Example:
    >>> from oscura.workflows.batch.logging import BatchLogger
    >>> logger = BatchLogger(batch_id="job-001")
    >>> with logger.file_context("capture1.wfm") as file_log:
    ...     file_log.info("Processing file")
    ...     result = analyze(file)
    >>> logger.summary()

References:
    LOG-011,
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

from oscura.core.logging import format_timestamp, get_logger


@dataclass
class FileLogEntry:
    """Log entry for a single file in a batch job.

    Attributes:
        file_id: Unique identifier for this file.
        filename: Path to the file.
        start_time: Processing start time.
        end_time: Processing end time.
        status: Processing status (pending, processing, success, error).
        error_message: Error message if status is 'error'.
        log_messages: List of log messages for this file.

    References:
        LOG-011: Aggregate Logging for Batch Processing
    """

    file_id: str
    filename: str
    start_time: float | None = None
    end_time: float | None = None
    status: str = "pending"
    error_message: str | None = None
    log_messages: list[dict[str, Any]] = field(default_factory=list)

    @property
    def duration(self) -> float | None:
        """Get processing duration in seconds."""
        if self.start_time is not None and self.end_time is not None:
            return self.end_time - self.start_time
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "file_id": self.file_id,
            "filename": self.filename,
            "start_time": format_timestamp(datetime.fromtimestamp(self.start_time, tz=UTC))
            if self.start_time
            else None,
            "end_time": format_timestamp(datetime.fromtimestamp(self.end_time, tz=UTC))
            if self.end_time
            else None,
            "duration_seconds": self.duration,
            "status": self.status,
            "error_message": self.error_message,
            "log_count": len(self.log_messages),
        }


@dataclass
class BatchSummary:
    """Summary of batch processing results.

    Attributes:
        batch_id: Unique identifier for the batch job.
        total_files: Total number of files in the batch.
        success_count: Number of successfully processed files.
        error_count: Number of files that failed.
        total_duration: Total processing time in seconds.
        errors_by_type: Count of errors grouped by type.

    References:
        LOG-011: Aggregate Logging for Batch Processing
    """

    batch_id: str
    total_files: int
    success_count: int
    error_count: int
    total_duration: float
    start_time: str
    end_time: str
    errors_by_type: dict[str, int] = field(default_factory=dict)
    files_per_second: float = 0.0
    average_duration_per_file: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "batch_id": self.batch_id,
            "total_files": self.total_files,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "success_rate": self.success_count / self.total_files if self.total_files > 0 else 0.0,
            "total_duration_seconds": self.total_duration,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "files_per_second": self.files_per_second,
            "average_duration_per_file": self.average_duration_per_file,
            "errors_by_type": self.errors_by_type,
        }


class FileLogger:
    """Logger for individual file processing within a batch.

    Provides logging methods that automatically tag logs with
    batch_id and file_id for aggregation.

    References:
        LOG-011: Aggregate Logging for Batch Processing
    """

    def __init__(
        self,
        entry: FileLogEntry,
        batch_id: str,
        parent_logger: logging.Logger,
    ):
        """Initialize file logger.

        Args:
            entry: FileLogEntry for this file.
            batch_id: Batch job identifier.
            parent_logger: Parent logger for output.
        """
        self._entry = entry
        self._batch_id = batch_id
        self._logger = parent_logger

    def _log(self, level: int, message: str, **kwargs: Any) -> None:
        """Log a message with batch/file context."""
        log_entry = {
            "timestamp": format_timestamp(),
            "level": logging.getLevelName(level),
            "message": message,
            "batch_id": self._batch_id,
            "file_id": self._entry.file_id,
            "filename": self._entry.filename,
            **kwargs,
        }
        self._entry.log_messages.append(log_entry)
        self._logger.log(
            level,
            message,
            extra={
                "batch_id": self._batch_id,
                "file_id": self._entry.file_id,
                **kwargs,
            },
        )

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message."""
        self._log(logging.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message."""
        self._log(logging.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message."""
        self._log(logging.WARNING, message, **kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        """Log error message."""
        self._log(logging.ERROR, message, **kwargs)


class BatchLogger:
    """Aggregate logger for batch processing operations.

    Consolidates logs from parallel batch workers with job-level
    summaries and per-file tracking.

    Example:
        >>> logger = BatchLogger(batch_id="job-001")
        >>> with logger.file_context("capture1.wfm") as file_log:
        ...     file_log.info("Loading file")
        ...     result = analyze(file)
        ...     file_log.info("Analysis complete", result=result)
        >>> summary = logger.summary()

    References:
        LOG-011: Aggregate Logging for Batch Processing
        LOG-013: Batch Job Correlation ID and Lineage
    """

    def __init__(
        self,
        batch_id: str | None = None,
        logger_name: str = "oscura.batch",
    ):
        """Initialize batch logger.

        Args:
            batch_id: Unique batch job identifier. Auto-generated if None.
            logger_name: Name for the underlying logger.
        """
        self.batch_id = batch_id or str(uuid.uuid4())
        self._logger = get_logger(logger_name)
        self._files: dict[str, FileLogEntry] = {}
        self._lock = threading.Lock()
        self._start_time: float | None = None
        self._end_time: float | None = None
        self._error_types: dict[str, int] = defaultdict(int)

    def start(self) -> None:
        """Mark batch job as started.

        Records the start time for duration calculation.
        """
        self._start_time = time.time()
        self._logger.info(
            "Batch job started",
            extra={"batch_id": self.batch_id},
        )

    def finish(self) -> None:
        """Mark batch job as finished.

        Records the end time and logs the completion summary.
        """
        self._end_time = time.time()
        summary = self.summary()
        self._logger.info(
            "Batch job completed",
            extra={
                "batch_id": self.batch_id,
                "total_files": summary.total_files,
                "success_count": summary.success_count,
                "error_count": summary.error_count,
                "duration": summary.total_duration,
            },
        )

    def register_file(self, filename: str) -> str:
        """Register a file for processing.

        Args:
            filename: Path to the file.

        Returns:
            Unique file_id for this file.
        """
        file_id = str(uuid.uuid4())
        with self._lock:
            self._files[file_id] = FileLogEntry(
                file_id=file_id,
                filename=filename,
            )
        return file_id

    @contextmanager
    def file_context(self, filename: str) -> Iterator[FileLogger]:
        """Context manager for file processing.

        Automatically tracks start/end time and status.

        Args:
            filename: Path to the file being processed.

        Yields:
            FileLogger for logging within this file's context.

        Raises:
            Exception: Re-raises any exception from the processing context.

        Example:
            >>> with batch_logger.file_context("data.wfm") as log:
            ...     log.info("Processing started")
            ...     result = process_file("data.wfm")
        """
        file_id = self.register_file(filename)
        entry = self._files[file_id]

        entry.start_time = time.time()
        entry.status = "processing"

        file_logger = FileLogger(entry, self.batch_id, self._logger)

        try:
            yield file_logger
            entry.status = "success"
        except Exception as e:
            entry.status = "error"
            entry.error_message = str(e)
            error_type = type(e).__name__
            with self._lock:
                self._error_types[error_type] += 1
            file_logger.error(f"Processing failed: {e}", exception_type=error_type)
            raise
        finally:
            entry.end_time = time.time()

    def mark_success(self, file_id: str) -> None:
        """Mark a file as successfully processed.

        Args:
            file_id: File identifier from register_file.
        """
        with self._lock:
            if file_id in self._files:
                self._files[file_id].status = "success"
                self._files[file_id].end_time = time.time()

    def mark_error(self, file_id: str, error: str, error_type: str = "Unknown") -> None:
        """Mark a file as failed.

        Args:
            file_id: File identifier from register_file.
            error: Error message.
            error_type: Type of error for aggregation.
        """
        with self._lock:
            if file_id in self._files:
                self._files[file_id].status = "error"
                self._files[file_id].error_message = error
                self._files[file_id].end_time = time.time()
                self._error_types[error_type] += 1

    def summary(self) -> BatchSummary:
        """Generate batch processing summary.

        Returns:
            BatchSummary with aggregated statistics.

        References:
            LOG-011: Aggregate Logging for Batch Processing
        """
        with self._lock:
            files = list(self._files.values())
            errors_by_type = dict(self._error_types)

        total_files = len(files)
        success_count = sum(1 for f in files if f.status == "success")
        error_count = sum(1 for f in files if f.status == "error")

        # Calculate timing
        start_time = self._start_time or (
            min((f.start_time for f in files if f.start_time), default=0)
        )
        end_time = self._end_time or (max((f.end_time for f in files if f.end_time), default=0))
        total_duration = end_time - start_time if start_time and end_time else 0.0

        # Calculate per-file metrics
        durations = [f.duration for f in files if f.duration is not None]
        avg_duration = sum(durations) / len(durations) if durations else 0.0
        files_per_second = total_files / total_duration if total_duration > 0 else 0.0

        return BatchSummary(
            batch_id=self.batch_id,
            total_files=total_files,
            success_count=success_count,
            error_count=error_count,
            total_duration=total_duration,
            start_time=format_timestamp(datetime.fromtimestamp(start_time, tz=UTC))
            if start_time
            else "",
            end_time=format_timestamp(datetime.fromtimestamp(end_time, tz=UTC)) if end_time else "",
            errors_by_type=errors_by_type,
            files_per_second=files_per_second,
            average_duration_per_file=avg_duration,
        )

    def get_file_logs(self, file_id: str) -> list[dict[str, Any]]:
        """Get all log messages for a specific file.

        Args:
            file_id: File identifier.

        Returns:
            List of log message dictionaries.
        """
        with self._lock:
            if file_id in self._files:
                return list(self._files[file_id].log_messages)
            return []

    def get_all_files(self) -> list[dict[str, Any]]:
        """Get summary information for all files.

        Returns:
            List of file summary dictionaries.
        """
        with self._lock:
            return [f.to_dict() for f in self._files.values()]

    def get_errors(self) -> list[dict[str, Any]]:
        """Get all files that encountered errors.

        Returns:
            List of error file dictionaries with error details.
        """
        with self._lock:
            return [
                {
                    **f.to_dict(),
                    "logs": f.log_messages,
                }
                for f in self._files.values()
                if f.status == "error"
            ]


def aggregate_batch_logs(
    batch_loggers: list[BatchLogger],
) -> dict[str, Any]:
    """Aggregate logs from multiple batch loggers.

    Combines summaries from multiple batch jobs into a single
    aggregate report.

    Args:
        batch_loggers: List of BatchLogger instances.

    Returns:
        Aggregated summary dictionary.

    References:
        LOG-011: Aggregate Logging for Batch Processing
    """
    total_files = 0
    total_success = 0
    total_errors = 0
    total_duration = 0.0
    all_errors_by_type: dict[str, int] = defaultdict(int)
    batch_summaries = []

    for logger in batch_loggers:
        summary = logger.summary()
        batch_summaries.append(summary.to_dict())
        total_files += summary.total_files
        total_success += summary.success_count
        total_errors += summary.error_count
        total_duration += summary.total_duration
        for error_type, count in summary.errors_by_type.items():
            all_errors_by_type[error_type] += count

    return {
        "aggregate": {
            "total_batches": len(batch_loggers),
            "total_files": total_files,
            "total_success": total_success,
            "total_errors": total_errors,
            "total_duration_seconds": total_duration,
            "overall_success_rate": total_success / total_files if total_files > 0 else 0.0,
            "errors_by_type": dict(all_errors_by_type),
        },
        "batches": batch_summaries,
    }


__all__ = [
    "BatchLogger",
    "BatchSummary",
    "FileLogEntry",
    "FileLogger",
    "aggregate_batch_logs",
]
