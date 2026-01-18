import pytest

"""Tests for memory progress logging module.

Tests requirements:
"""

import json
import tempfile
import time
from pathlib import Path

from oscura.core.memory_progress import (
    MemoryLogEntry,
    MemoryLogger,
    create_progress_callback_with_logging,
    enable_memory_logging_from_cli,
    log_memory,
)

pytestmark = pytest.mark.unit


class TestMemoryLogger:
    """Tests for MemoryLogger (MEM-025)."""

    def test_csv_logging(self):
        """Test CSV format logging."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
            log_file = Path(tmp.name)

        try:
            logger = MemoryLogger(log_file, format="csv")
            with logger:
                logger.log_operation("test_op", iteration=0, message="Start")
                logger.log_operation("test_op", iteration=1, message="Continue")
                logger.log_operation("test_op", iteration=2, message="End")

            # Check file exists and has content
            assert log_file.exists()
            content = log_file.read_text()
            assert "timestamp" in content
            assert "test_op" in content
            assert "Start" in content

            # Check entry count
            lines = content.strip().split("\n")
            assert len(lines) >= 4  # header + 3 entries

        finally:
            if log_file.exists():
                log_file.unlink()

    def test_json_logging(self):
        """Test JSON format logging."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            log_file = Path(tmp.name)

        try:
            logger = MemoryLogger(log_file, format="json")
            with logger:
                logger.log_operation("test_op", iteration=0)
                logger.log_operation("test_op", iteration=1)

            # Check file exists and is valid JSON
            assert log_file.exists()
            data = json.loads(log_file.read_text())
            assert "entries" in data
            assert "summary" in data
            assert len(data["entries"]) == 2

        finally:
            if log_file.exists():
                log_file.unlink()

    def test_log_operation(self):
        """Test logging individual operations."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
            log_file = Path(tmp.name)

        try:
            logger = MemoryLogger(log_file)
            with logger:
                logger.log_operation("fft", iteration=10, eta_seconds=5.0, message="Processing")

                entries = logger.get_entries()
                assert len(entries) == 1

                entry = entries[0]
                assert entry.operation == "fft"
                assert entry.iteration == 10
                assert entry.eta_seconds == 5.0
                assert entry.message == "Processing"
                assert entry.memory_used > 0
                assert entry.memory_available > 0
                assert 0.0 <= entry.memory_pressure <= 1.0

        finally:
            if log_file.exists():
                log_file.unlink()

    def test_log_progress(self):
        """Test logging with progress information."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
            log_file = Path(tmp.name)

        try:
            logger = MemoryLogger(log_file)
            with logger:
                # Simulate progress
                total = 100
                for i in range(5):
                    logger.log_progress("analysis", i + 1, total, message=f"Step {i}")
                    time.sleep(0.01)  # Small delay for ETA calculation

                entries = logger.get_entries()
                assert len(entries) == 5

                # Check iterations are sequential
                for i, entry in enumerate(entries):
                    assert entry.iteration == i + 1
                    assert entry.operation == "analysis"

        finally:
            if log_file.exists():
                log_file.unlink()

    def test_get_summary(self):
        """Test summary generation."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
            log_file = Path(tmp.name)

        try:
            logger = MemoryLogger(log_file)
            with logger:
                for i in range(10):
                    logger.log_operation("test", iteration=i)

                summary = logger.get_summary()
                assert "Entries: 10" in summary
                assert "Duration:" in summary
                assert "Peak Memory:" in summary
                assert "GB" in summary

        finally:
            if log_file.exists():
                log_file.unlink()

    def test_memory_tracking(self):
        """Test that memory is tracked correctly."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
            log_file = Path(tmp.name)

        try:
            logger = MemoryLogger(log_file)
            with logger:
                # Log several operations
                for i in range(5):
                    logger.log_operation("test", iteration=i)

                entries = logger.get_entries()

                # Check all entries have valid memory values
                for entry in entries:
                    assert entry.memory_used > 0
                    assert entry.memory_peak >= entry.memory_used
                    assert entry.memory_available > 0
                    assert 0.0 <= entry.memory_pressure <= 1.0

                # Peak should be monotonically increasing or stable
                peaks = [e.memory_peak for e in entries]
                for i in range(1, len(peaks)):
                    assert peaks[i] >= peaks[i - 1]

        finally:
            if log_file.exists():
                log_file.unlink()

    def test_console_output(self, capsys):
        """Test console output when enabled."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
            log_file = Path(tmp.name)

        try:
            logger = MemoryLogger(log_file, enable_console=True)
            with logger:
                logger.log_operation("test", message="Test message")

            captured = capsys.readouterr()
            assert "test" in captured.out or "Test message" in captured.out

        finally:
            if log_file.exists():
                log_file.unlink()


class TestLogMemoryContext:
    """Tests for log_memory context manager."""

    def test_context_manager(self):
        """Test log_memory context manager."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
            log_file = Path(tmp.name)

        try:
            with log_memory(log_file) as logger:
                logger.log_operation("test1")
                logger.log_operation("test2")

            # File should exist after context exit
            assert log_file.exists()
            content = log_file.read_text()
            assert "test1" in content
            assert "test2" in content

        finally:
            if log_file.exists():
                log_file.unlink()


class TestProgressCallbackWithLogging:
    """Tests for create_progress_callback_with_logging (MEM-024)."""

    def test_callback_integration(self):
        """Test progress callback integration."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
            log_file = Path(tmp.name)

        try:
            logger = MemoryLogger(log_file)
            with logger:
                callback = create_progress_callback_with_logging(logger, "operation")

                # Simulate progress callbacks
                callback(25, 100, "Quarter done")
                callback(50, 100, "Half done")
                callback(100, 100, "Complete")

                entries = logger.get_entries()
                assert len(entries) == 3

                # Check progress is logged
                assert entries[0].iteration == 25
                assert entries[1].iteration == 50
                assert entries[2].iteration == 100

        finally:
            if log_file.exists():
                log_file.unlink()


class TestCLIIntegration:
    """Tests for CLI integration (MEM-025)."""

    def test_enable_from_cli_disabled(self):
        """Test that logger is None when not enabled."""
        import os

        # Ensure env var is not set
        os.environ.pop("TK_LOG_MEMORY", None)

        logger = enable_memory_logging_from_cli()
        assert logger is None

    def test_enable_from_cli_enabled(self):
        """Test that logger is created when enabled."""
        import os

        try:
            # Enable via env var
            os.environ["TK_LOG_MEMORY"] = "1"

            logger = enable_memory_logging_from_cli()
            assert logger is not None
            assert isinstance(logger, MemoryLogger)

            # Clean up
            with logger:
                logger.log_operation("test")

        finally:
            os.environ.pop("TK_LOG_MEMORY", None)
            # Clean up log file if created
            if logger and logger.log_file.exists():
                logger.log_file.unlink()

    def test_custom_log_file(self):
        """Test custom log file path."""
        import os

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
            log_file = Path(tmp.name)

        try:
            os.environ["TK_LOG_MEMORY"] = "1"

            logger = enable_memory_logging_from_cli(log_file)
            assert logger is not None
            assert logger.log_file == log_file

            with logger:
                logger.log_operation("test")

            assert log_file.exists()

        finally:
            os.environ.pop("TK_LOG_MEMORY", None)
            if log_file.exists():
                log_file.unlink()


class TestMemoryLogEntry:
    """Tests for MemoryLogEntry dataclass."""

    def test_entry_creation(self):
        """Test creating log entry."""
        entry = MemoryLogEntry(
            timestamp=time.time(),
            operation="test",
            iteration=5,
            memory_used=1000000,
            memory_peak=2000000,
            memory_available=8000000000,
            memory_pressure=0.25,
            eta_seconds=10.5,
            message="Testing",
        )

        assert entry.operation == "test"
        assert entry.iteration == 5
        assert entry.memory_used == 1000000
        assert entry.memory_pressure == 0.25


class TestUtilsMemoryProgressEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_logger(self):
        """Test logger with no entries."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
            log_file = Path(tmp.name)

        try:
            logger = MemoryLogger(log_file)
            with logger:
                pass  # No operations logged

            summary = logger.get_summary()
            assert "Entries: 0" in summary

        finally:
            if log_file.exists():
                log_file.unlink()

    def test_directory_creation(self):
        """Test that parent directories are created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "subdir" / "test.csv"

            logger = MemoryLogger(log_file)
            with logger:
                logger.log_operation("test")

            assert log_file.exists()
            assert log_file.parent.exists()
