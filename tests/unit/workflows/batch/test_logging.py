"""Tests for batch logging module (workflows/batch/logging.py).

Achieves 100% coverage through comprehensive testing of:
- FileLogEntry dataclass and properties
- BatchSummary dataclass and conversion methods
- FileLogger logging at all levels
- BatchLogger lifecycle and context management
- Error tracking and aggregation
- Multi-batch aggregation

References:
    LOG-011: Aggregate Logging for Batch Processing
    LOG-013: Batch Job Correlation ID and Lineage
"""

from __future__ import annotations

import logging
import time

import pytest

from oscura.workflows.batch.logging import (
    BatchLogger,
    BatchSummary,
    FileLogEntry,
    FileLogger,
    aggregate_batch_logs,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def batch_logger() -> BatchLogger:
    """Create a batch logger instance."""
    return BatchLogger(batch_id="test-batch-001")


@pytest.fixture
def file_entry() -> FileLogEntry:
    """Create a sample file log entry."""
    return FileLogEntry(
        file_id="file-001",
        filename="test_file.wfm",
    )


# =============================================================================
# FileLogEntry Tests
# =============================================================================


def test_file_log_entry_creation():
    """Test creating a file log entry with defaults."""
    entry = FileLogEntry(
        file_id="file-123",
        filename="capture.wfm",
    )

    assert entry.file_id == "file-123"
    assert entry.filename == "capture.wfm"
    assert entry.start_time is None
    assert entry.end_time is None
    assert entry.status == "pending"
    assert entry.error_message is None
    assert entry.log_messages == []


def test_file_log_entry_duration_none_when_no_times():
    """Test duration property returns None when times not set."""
    entry = FileLogEntry(file_id="f1", filename="test.wfm")
    assert entry.duration is None


def test_file_log_entry_duration_none_when_only_start_time():
    """Test duration returns None when only start time is set."""
    entry = FileLogEntry(file_id="f1", filename="test.wfm")
    entry.start_time = time.time()
    assert entry.duration is None


def test_file_log_entry_duration_none_when_only_end_time():
    """Test duration returns None when only end time is set."""
    entry = FileLogEntry(file_id="f1", filename="test.wfm")
    entry.end_time = time.time()
    assert entry.duration is None


def test_file_log_entry_duration_calculated():
    """Test duration calculation when both times are set."""
    entry = FileLogEntry(file_id="f1", filename="test.wfm")
    start = time.time()
    time.sleep(0.01)  # Small delay
    end = time.time()

    entry.start_time = start
    entry.end_time = end

    duration = entry.duration
    assert duration is not None
    assert duration > 0
    assert duration < 1.0  # Should be very short


def test_file_log_entry_to_dict_with_times():
    """Test converting entry to dict with all fields populated."""
    entry = FileLogEntry(
        file_id="file-001",
        filename="test.wfm",
        start_time=1234567890.0,
        end_time=1234567891.5,
        status="success",
        error_message=None,
    )

    result = entry.to_dict()

    assert result["file_id"] == "file-001"
    assert result["filename"] == "test.wfm"
    assert result["duration_seconds"] == 1.5
    assert result["status"] == "success"
    assert result["error_message"] is None
    assert result["log_count"] == 0
    # Timestamps should be ISO formatted
    assert result["start_time"] is not None
    assert result["end_time"] is not None


def test_file_log_entry_to_dict_without_times():
    """Test converting entry to dict when times are None."""
    entry = FileLogEntry(
        file_id="file-002",
        filename="missing.wfm",
    )

    result = entry.to_dict()

    assert result["file_id"] == "file-002"
    assert result["filename"] == "missing.wfm"
    assert result["start_time"] is None
    assert result["end_time"] is None
    assert result["duration_seconds"] is None


def test_file_log_entry_with_error():
    """Test file entry with error status and message."""
    entry = FileLogEntry(
        file_id="file-err",
        filename="broken.wfm",
        status="error",
        error_message="File not found",
    )

    assert entry.status == "error"
    assert entry.error_message == "File not found"

    result = entry.to_dict()
    assert result["status"] == "error"
    assert result["error_message"] == "File not found"


# =============================================================================
# BatchSummary Tests
# =============================================================================


def test_batch_summary_creation():
    """Test creating a batch summary."""
    summary = BatchSummary(
        batch_id="batch-001",
        total_files=100,
        success_count=95,
        error_count=5,
        total_duration=120.5,
        start_time="2025-01-01T00:00:00Z",
        end_time="2025-01-01T00:02:00Z",
        errors_by_type={"FileNotFoundError": 3, "ValueError": 2},
        files_per_second=0.83,
        average_duration_per_file=1.27,
    )

    assert summary.batch_id == "batch-001"
    assert summary.total_files == 100
    assert summary.success_count == 95
    assert summary.error_count == 5


def test_batch_summary_to_dict():
    """Test converting batch summary to dictionary."""
    summary = BatchSummary(
        batch_id="batch-002",
        total_files=50,
        success_count=48,
        error_count=2,
        total_duration=60.0,
        start_time="2025-01-01T10:00:00Z",
        end_time="2025-01-01T10:01:00Z",
        errors_by_type={"RuntimeError": 2},
        files_per_second=0.8,
        average_duration_per_file=1.25,
    )

    result = summary.to_dict()

    assert result["batch_id"] == "batch-002"
    assert result["total_files"] == 50
    assert result["success_count"] == 48
    assert result["error_count"] == 2
    assert result["success_rate"] == 48 / 50
    assert result["total_duration_seconds"] == 60.0
    assert result["files_per_second"] == 0.8
    assert result["average_duration_per_file"] == 1.25
    assert result["errors_by_type"] == {"RuntimeError": 2}


def test_batch_summary_to_dict_zero_files():
    """Test batch summary with zero files to avoid division by zero."""
    summary = BatchSummary(
        batch_id="batch-empty",
        total_files=0,
        success_count=0,
        error_count=0,
        total_duration=0.0,
        start_time="",
        end_time="",
    )

    result = summary.to_dict()
    assert result["success_rate"] == 0.0  # Should handle division by zero


# =============================================================================
# FileLogger Tests
# =============================================================================


def test_file_logger_debug(file_entry: FileLogEntry, caplog: pytest.LogCaptureFixture):
    """Test file logger debug method."""
    logger = logging.getLogger("test-logger")
    file_logger = FileLogger(file_entry, "batch-001", logger)

    with caplog.at_level(logging.DEBUG, logger="test-logger"):
        file_logger.debug("Debug message", extra_field="value")

    # Check log entry was added to file entry
    assert len(file_entry.log_messages) == 1
    log_msg = file_entry.log_messages[0]
    assert log_msg["level"] == "DEBUG"
    assert log_msg["message"] == "Debug message"
    assert log_msg["batch_id"] == "batch-001"
    assert log_msg["file_id"] == "file-001"
    assert log_msg["filename"] == "test_file.wfm"
    assert log_msg["extra_field"] == "value"


def test_file_logger_info(file_entry: FileLogEntry, caplog: pytest.LogCaptureFixture):
    """Test file logger info method."""
    logger = logging.getLogger("test-logger")
    file_logger = FileLogger(file_entry, "batch-002", logger)

    with caplog.at_level(logging.INFO, logger="test-logger"):
        file_logger.info("Info message")

    assert len(file_entry.log_messages) == 1
    assert file_entry.log_messages[0]["level"] == "INFO"
    assert file_entry.log_messages[0]["message"] == "Info message"


def test_file_logger_warning(file_entry: FileLogEntry, caplog: pytest.LogCaptureFixture):
    """Test file logger warning method."""
    logger = logging.getLogger("test-logger")
    file_logger = FileLogger(file_entry, "batch-003", logger)

    with caplog.at_level(logging.WARNING, logger="test-logger"):
        file_logger.warning("Warning message")

    assert len(file_entry.log_messages) == 1
    assert file_entry.log_messages[0]["level"] == "WARNING"
    assert file_entry.log_messages[0]["message"] == "Warning message"


def test_file_logger_error(file_entry: FileLogEntry, caplog: pytest.LogCaptureFixture):
    """Test file logger error method."""
    logger = logging.getLogger("test-logger")
    file_logger = FileLogger(file_entry, "batch-004", logger)

    with caplog.at_level(logging.ERROR, logger="test-logger"):
        file_logger.error("Error message")

    assert len(file_entry.log_messages) == 1
    assert file_entry.log_messages[0]["level"] == "ERROR"
    assert file_entry.log_messages[0]["message"] == "Error message"


def test_file_logger_multiple_messages(file_entry: FileLogEntry):
    """Test logging multiple messages accumulates in entry."""
    logger = logging.getLogger("test-logger")
    file_logger = FileLogger(file_entry, "batch-005", logger)

    file_logger.debug("First")
    file_logger.info("Second")
    file_logger.warning("Third")
    file_logger.error("Fourth")

    assert len(file_entry.log_messages) == 4
    assert file_entry.log_messages[0]["message"] == "First"
    assert file_entry.log_messages[1]["message"] == "Second"
    assert file_entry.log_messages[2]["message"] == "Third"
    assert file_entry.log_messages[3]["message"] == "Fourth"


# =============================================================================
# BatchLogger Lifecycle Tests
# =============================================================================


def test_batch_logger_creation_with_batch_id():
    """Test creating batch logger with explicit batch ID."""
    logger = BatchLogger(batch_id="custom-batch-id")
    assert logger.batch_id == "custom-batch-id"


def test_batch_logger_creation_auto_id():
    """Test batch logger auto-generates UUID if no ID provided."""
    logger = BatchLogger()
    assert logger.batch_id is not None
    assert len(logger.batch_id) > 0
    # Should be UUID format
    assert "-" in logger.batch_id


def test_batch_logger_start():
    """Test marking batch job as started."""
    logger = BatchLogger(batch_id="batch-start")
    assert logger._start_time is None

    logger.start()

    start_time = logger._start_time
    assert start_time is not None
    assert start_time > 0  # type: ignore[unreachable]


def test_batch_logger_finish():
    """Test marking batch job as finished."""
    logger = BatchLogger(batch_id="batch-finish")
    logger.start()
    time.sleep(0.01)
    logger.finish()

    assert logger._end_time is not None
    assert logger._start_time is not None
    assert logger._end_time > logger._start_time


def test_batch_logger_register_file():
    """Test registering a file for processing."""
    logger = BatchLogger()

    file_id = logger.register_file("test1.wfm")

    assert file_id is not None
    assert file_id in logger._files
    assert logger._files[file_id].filename == "test1.wfm"
    assert logger._files[file_id].status == "pending"


def test_batch_logger_register_multiple_files():
    """Test registering multiple files gets unique IDs."""
    logger = BatchLogger()

    id1 = logger.register_file("file1.wfm")
    id2 = logger.register_file("file2.wfm")

    assert id1 != id2
    assert len(logger._files) == 2


# =============================================================================
# BatchLogger Context Manager Tests
# =============================================================================


def test_batch_logger_file_context_success():
    """Test file context manager for successful processing."""
    logger = BatchLogger(batch_id="ctx-success")

    with logger.file_context("success.wfm") as file_log:
        file_log.info("Processing file")

    # Check file was registered
    assert len(logger._files) == 1

    # Get the file entry
    file_id = next(iter(logger._files.keys()))
    entry = logger._files[file_id]

    assert entry.filename == "success.wfm"
    assert entry.status == "success"
    assert entry.start_time is not None
    assert entry.end_time is not None
    assert entry.duration is not None
    assert entry.error_message is None
    assert len(entry.log_messages) == 1


def test_batch_logger_file_context_error():
    """Test file context manager handles errors correctly."""
    logger = BatchLogger(batch_id="ctx-error")

    with pytest.raises(ValueError, match="Test error"):
        with logger.file_context("error.wfm") as file_log:
            file_log.info("Starting")
            raise ValueError("Test error")

    # Check error was recorded
    file_id = next(iter(logger._files.keys()))
    entry = logger._files[file_id]

    assert entry.status == "error"
    assert entry.error_message == "Test error"
    assert "ValueError" in logger._error_types
    assert logger._error_types["ValueError"] == 1


def test_batch_logger_file_context_multiple_errors():
    """Test tracking multiple error types."""
    logger = BatchLogger()

    # First error
    with pytest.raises(ValueError):
        with logger.file_context("file1.wfm"):
            raise ValueError("Error 1")

    # Second error of same type
    with pytest.raises(ValueError):
        with logger.file_context("file2.wfm"):
            raise ValueError("Error 2")

    # Different error type
    with pytest.raises(RuntimeError):
        with logger.file_context("file3.wfm"):
            raise RuntimeError("Error 3")

    assert logger._error_types["ValueError"] == 2
    assert logger._error_types["RuntimeError"] == 1


def test_batch_logger_file_context_timing():
    """Test file context tracks timing correctly."""
    logger = BatchLogger()

    with logger.file_context("timing.wfm") as file_log:
        time.sleep(0.02)  # Small delay
        file_log.info("Processing")

    file_id = next(iter(logger._files.keys()))
    entry = logger._files[file_id]

    assert entry.duration is not None
    assert entry.duration >= 0.02  # At least the sleep time


# =============================================================================
# BatchLogger Manual Marking Tests
# =============================================================================


def test_batch_logger_mark_success():
    """Test manually marking a file as successful."""
    logger = BatchLogger()
    file_id = logger.register_file("manual.wfm")

    # Initially pending
    assert logger._files[file_id].status == "pending"

    logger.mark_success(file_id)

    assert logger._files[file_id].status == "success"
    assert logger._files[file_id].end_time is not None


def test_batch_logger_mark_success_invalid_id():
    """Test marking success with invalid file ID (should be no-op)."""
    logger = BatchLogger()
    logger.mark_success("invalid-id")  # Should not raise


def test_batch_logger_mark_error():
    """Test manually marking a file as error."""
    logger = BatchLogger()
    file_id = logger.register_file("error.wfm")

    logger.mark_error(file_id, "Custom error", error_type="CustomError")

    assert logger._files[file_id].status == "error"
    assert logger._files[file_id].error_message == "Custom error"
    assert logger._files[file_id].end_time is not None
    assert logger._error_types["CustomError"] == 1


def test_batch_logger_mark_error_invalid_id():
    """Test marking error with invalid file ID (should be no-op)."""
    logger = BatchLogger()
    logger.mark_error("invalid-id", "error", "ErrorType")  # Should not raise


# =============================================================================
# BatchLogger Summary Tests
# =============================================================================


def test_batch_logger_summary_empty():
    """Test summary with no files."""
    logger = BatchLogger(batch_id="empty-batch")
    logger.start()
    logger.finish()

    summary = logger.summary()

    assert summary.batch_id == "empty-batch"
    assert summary.total_files == 0
    assert summary.success_count == 0
    assert summary.error_count == 0


def test_batch_logger_summary_with_files():
    """Test summary with processed files."""
    logger = BatchLogger(batch_id="summary-batch")
    logger.start()

    # Process successful files
    with logger.file_context("file1.wfm"):
        pass

    with logger.file_context("file2.wfm"):
        pass

    # Process error file
    with pytest.raises(ValueError):
        with logger.file_context("file3.wfm"):
            raise ValueError("Error")

    logger.finish()

    summary = logger.summary()

    assert summary.batch_id == "summary-batch"
    assert summary.total_files == 3
    assert summary.success_count == 2
    assert summary.error_count == 1
    assert summary.total_duration > 0
    assert summary.start_time != ""
    assert summary.end_time != ""
    assert "ValueError" in summary.errors_by_type


def test_batch_logger_summary_timing_without_explicit_start_end():
    """Test summary calculates timing from file times if not explicitly set."""
    logger = BatchLogger()

    # Process files without calling start/finish
    with logger.file_context("file1.wfm"):
        time.sleep(0.01)

    with logger.file_context("file2.wfm"):
        time.sleep(0.01)

    summary = logger.summary()

    # Should calculate from file times
    assert summary.total_duration > 0
    assert summary.files_per_second > 0
    assert summary.average_duration_per_file > 0


def test_batch_logger_summary_performance_metrics():
    """Test summary calculates performance metrics correctly."""
    logger = BatchLogger()
    logger.start()

    # Create multiple files with known timing
    for i in range(5):
        with logger.file_context(f"file{i}.wfm"):
            time.sleep(0.01)

    logger.finish()

    summary = logger.summary()

    assert summary.files_per_second > 0
    assert summary.average_duration_per_file > 0


# =============================================================================
# BatchLogger Query Tests
# =============================================================================


def test_batch_logger_get_file_logs():
    """Test retrieving logs for a specific file."""
    logger = BatchLogger()

    with logger.file_context("query.wfm") as file_log:
        file_log.info("Message 1")
        file_log.debug("Message 2")

    file_id = next(iter(logger._files.keys()))
    logs = logger.get_file_logs(file_id)

    assert len(logs) == 2
    assert logs[0]["message"] == "Message 1"
    assert logs[1]["message"] == "Message 2"


def test_batch_logger_get_file_logs_invalid_id():
    """Test getting logs for invalid file ID returns empty list."""
    logger = BatchLogger()
    logs = logger.get_file_logs("invalid-id")
    assert logs == []


def test_batch_logger_get_all_files():
    """Test retrieving summary of all files."""
    logger = BatchLogger()

    with logger.file_context("file1.wfm"):
        pass

    with pytest.raises(ValueError):
        with logger.file_context("file2.wfm"):
            raise ValueError("Error")

    all_files = logger.get_all_files()

    assert len(all_files) == 2
    assert all_files[0]["filename"] == "file1.wfm"
    assert all_files[0]["status"] == "success"
    assert all_files[1]["filename"] == "file2.wfm"
    assert all_files[1]["status"] == "error"


def test_batch_logger_get_errors():
    """Test retrieving only error files."""
    logger = BatchLogger()

    # Success
    with logger.file_context("success.wfm"):
        pass

    # Errors
    with pytest.raises(ValueError):
        with logger.file_context("error1.wfm"):
            raise ValueError("Error 1")

    with pytest.raises(RuntimeError):
        with logger.file_context("error2.wfm"):
            raise RuntimeError("Error 2")

    errors = logger.get_errors()

    assert len(errors) == 2
    assert errors[0]["status"] == "error"
    assert errors[1]["status"] == "error"
    # Should include logs
    assert "logs" in errors[0]
    assert "logs" in errors[1]


# =============================================================================
# Multi-Batch Aggregation Tests
# =============================================================================


def test_aggregate_batch_logs_empty():
    """Test aggregating empty batch list."""
    result = aggregate_batch_logs([])

    assert result["aggregate"]["total_batches"] == 0
    assert result["aggregate"]["total_files"] == 0
    assert result["batches"] == []


def test_aggregate_batch_logs_single_batch():
    """Test aggregating single batch."""
    logger = BatchLogger(batch_id="agg-1")
    logger.start()

    with logger.file_context("file1.wfm"):
        pass

    logger.finish()

    result = aggregate_batch_logs([logger])

    assert result["aggregate"]["total_batches"] == 1
    assert result["aggregate"]["total_files"] == 1
    assert result["aggregate"]["total_success"] == 1
    assert result["aggregate"]["total_errors"] == 0


def test_aggregate_batch_logs_multiple_batches():
    """Test aggregating multiple batches."""
    logger1 = BatchLogger(batch_id="agg-1")
    logger1.start()
    with logger1.file_context("file1.wfm"):
        pass
    with logger1.file_context("file2.wfm"):
        pass
    logger1.finish()

    logger2 = BatchLogger(batch_id="agg-2")
    logger2.start()
    with logger2.file_context("file3.wfm"):
        pass
    with pytest.raises(ValueError):
        with logger2.file_context("file4.wfm"):
            raise ValueError("Error")
    logger2.finish()

    result = aggregate_batch_logs([logger1, logger2])

    assert result["aggregate"]["total_batches"] == 2
    assert result["aggregate"]["total_files"] == 4
    assert result["aggregate"]["total_success"] == 3
    assert result["aggregate"]["total_errors"] == 1
    assert result["aggregate"]["overall_success_rate"] == 3 / 4
    assert "ValueError" in result["aggregate"]["errors_by_type"]
    assert len(result["batches"]) == 2


def test_aggregate_batch_logs_combines_error_types():
    """Test aggregation combines error types across batches."""
    logger1 = BatchLogger()
    with pytest.raises(ValueError):
        with logger1.file_context("f1"):
            raise ValueError("E1")

    logger2 = BatchLogger()
    with pytest.raises(ValueError):
        with logger2.file_context("f2"):
            raise ValueError("E2")
    with pytest.raises(RuntimeError):
        with logger2.file_context("f3"):
            raise RuntimeError("E3")

    result = aggregate_batch_logs([logger1, logger2])

    errors = result["aggregate"]["errors_by_type"]
    assert errors["ValueError"] == 2
    assert errors["RuntimeError"] == 1


def test_aggregate_batch_logs_success_rate_zero_division():
    """Test aggregation handles zero files correctly."""
    logger1 = BatchLogger()
    logger2 = BatchLogger()

    result = aggregate_batch_logs([logger1, logger2])

    assert result["aggregate"]["overall_success_rate"] == 0.0


# =============================================================================
# Thread Safety Tests
# =============================================================================


def test_batch_logger_thread_safety():
    """Test batch logger is thread-safe for concurrent file operations."""
    import concurrent.futures

    logger = BatchLogger(batch_id="thread-test")

    def process_file(index: int) -> None:
        with logger.file_context(f"file{index}.wfm") as file_log:
            file_log.info(f"Processing {index}")
            time.sleep(0.001)

    # Process files concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(process_file, i) for i in range(50)]
        concurrent.futures.wait(futures)

    # All files should be recorded
    assert len(logger._files) == 50
    all_success = all(f.status == "success" for f in logger._files.values())
    assert all_success


def test_batch_logger_concurrent_error_tracking():
    """Test error tracking is thread-safe."""
    import concurrent.futures

    logger = BatchLogger()

    def process_with_error(index: int) -> None:
        try:
            with logger.file_context(f"file{index}.wfm"):
                if index % 2 == 0:
                    raise ValueError("Even error")
                else:
                    raise RuntimeError("Odd error")
        except (ValueError, RuntimeError):
            pass

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(process_with_error, i) for i in range(20)]
        concurrent.futures.wait(futures)

    # Error counts should be correct
    assert logger._error_types["ValueError"] == 10
    assert logger._error_types["RuntimeError"] == 10
