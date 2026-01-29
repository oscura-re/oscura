"""Tests for memory usage logging and progress tracking.

Tests the memory logging infrastructure (MEM-024, MEM-025).
"""

from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from oscura.core.memory_progress import (
    MemoryLogEntry,
    MemoryLogger,
    create_progress_callback_with_logging,
    enable_memory_logging_from_cli,
    log_memory,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.core,
    pytest.mark.usefixtures("reset_logging_state"),
]


class TestMemoryLogEntry:
    """Test MemoryLogEntry dataclass."""

    def test_create_entry(self) -> None:
        """Test creating memory log entry."""
        entry = MemoryLogEntry(
            timestamp=1234567890.0,
            operation="test_op",
            iteration=100,
            memory_used=512 * 1024**2,
            memory_peak=600 * 1024**2,
            memory_available=4 * 1024**3,
            memory_pressure=0.25,
            eta_seconds=30.0,
            message="Processing batch",
        )
        assert entry.timestamp == 1234567890.0
        assert entry.operation == "test_op"
        assert entry.iteration == 100
        assert entry.message == "Processing batch"


class TestMemoryLogger:
    """Test MemoryLogger class."""

    def test_init_csv_format(self, tmp_path: Path) -> None:
        """Test initialization with CSV format."""
        log_file = tmp_path / "test.csv"
        logger = MemoryLogger(log_file, format="csv")
        assert logger.log_file == log_file
        assert logger.format == "csv"
        assert logger.auto_flush is True
        assert logger.enable_console is False

    def test_init_json_format(self, tmp_path: Path) -> None:
        """Test initialization with JSON format."""
        log_file = tmp_path / "test.json"
        logger = MemoryLogger(log_file, format="json", auto_flush=False, enable_console=True)
        assert logger.format == "json"
        assert logger.auto_flush is False
        assert logger.enable_console is True

    def test_init_creates_directory(self, tmp_path: Path) -> None:
        """Test that logger creates parent directories."""
        log_file = tmp_path / "subdir" / "test.csv"
        logger = MemoryLogger(log_file)
        assert log_file.parent.exists()

    @patch("oscura.core.memory_progress.MemoryLogger._get_process_memory")
    def test_context_enter_csv(self, mock_memory: MagicMock, tmp_path: Path) -> None:
        """Test entering context with CSV format."""
        mock_memory.return_value = 100 * 1024**2
        log_file = tmp_path / "test.csv"
        logger = MemoryLogger(log_file, format="csv")

        with logger as log:
            assert log is logger
            assert logger._start_time > 0
            assert logger._start_memory == 100 * 1024**2
            assert logger._file_handle is not None

        # Verify CSV header was written
        with open(log_file) as f:
            reader = csv.DictReader(f)
            assert "timestamp" in reader.fieldnames  # type: ignore[operator]

    @patch("oscura.core.memory_progress.MemoryLogger._get_process_memory")
    def test_context_exit_json(self, mock_memory: MagicMock, tmp_path: Path) -> None:
        """Test exiting context with JSON format."""
        mock_memory.return_value = 100 * 1024**2
        log_file = tmp_path / "test.json"
        logger = MemoryLogger(log_file, format="json")

        with logger:
            pass  # Empty context

        # Verify JSON was written
        with open(log_file) as f:
            data = json.load(f)
            assert "entries" in data
            assert "summary" in data

    @patch("oscura.core.memory_progress.get_memory_pressure")
    @patch("oscura.core.memory_progress.get_available_memory")
    @patch("oscura.core.memory_progress.MemoryLogger._get_process_memory")
    def test_log_operation_csv(
        self,
        mock_process_mem: MagicMock,
        mock_available: MagicMock,
        mock_pressure: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test logging operation with CSV format."""
        mock_process_mem.return_value = 200 * 1024**2
        mock_available.return_value = 6 * 1024**3
        mock_pressure.return_value = 0.15

        log_file = tmp_path / "test.csv"
        logger = MemoryLogger(log_file, format="csv")

        with logger:
            logger.log_operation("test_op", iteration=1, eta_seconds=5.0, message="Testing")

        # Verify entry was logged
        assert len(logger._entries) == 1
        entry = logger._entries[0]
        assert entry.operation == "test_op"
        assert entry.iteration == 1
        assert entry.eta_seconds == 5.0
        assert entry.message == "Testing"

        # Verify CSV was written
        with open(log_file) as f:
            rows = list(csv.DictReader(f))
            assert len(rows) == 1
            assert rows[0]["operation"] == "test_op"

    @patch("oscura.core.memory_progress.get_memory_pressure")
    @patch("oscura.core.memory_progress.get_available_memory")
    @patch("oscura.core.memory_progress.MemoryLogger._get_process_memory")
    @patch("builtins.print")
    def test_log_operation_console(
        self,
        mock_print: MagicMock,
        mock_process_mem: MagicMock,
        mock_available: MagicMock,
        mock_pressure: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test logging with console output enabled."""
        mock_process_mem.return_value = 200 * 1024**2
        mock_available.return_value = 6 * 1024**3
        mock_pressure.return_value = 0.15

        log_file = tmp_path / "test.csv"
        logger = MemoryLogger(log_file, enable_console=True)

        with logger:
            logger.log_operation("test_op", message="Console test")

        # Verify console output was printed
        assert mock_print.called
        call_args = mock_print.call_args[0][0]
        assert "test_op" in call_args
        assert "Console test" in call_args

    @patch("oscura.core.memory_progress.get_memory_pressure")
    @patch("oscura.core.memory_progress.get_available_memory")
    @patch("oscura.core.memory_progress.MemoryLogger._get_process_memory")
    def test_log_progress(
        self,
        mock_process_mem: MagicMock,
        mock_available: MagicMock,
        mock_pressure: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test log_progress method."""
        mock_process_mem.return_value = 200 * 1024**2
        mock_available.return_value = 6 * 1024**3
        mock_pressure.return_value = 0.15

        log_file = tmp_path / "test.csv"
        logger = MemoryLogger(log_file)

        with logger:
            time.sleep(0.01)  # Ensure elapsed > 0
            logger.log_progress("processing", current=50, total=100, message="Half done")

        assert len(logger._entries) == 1
        entry = logger._entries[0]
        assert entry.operation == "processing"
        assert entry.iteration == 50
        assert entry.eta_seconds > 0  # ETA calculated
        assert entry.message == "Half done"

    @patch("oscura.core.memory_progress.MemoryLogger._get_process_memory")
    def test_get_summary(self, mock_memory: MagicMock, tmp_path: Path) -> None:
        """Test get_summary method."""
        mock_memory.return_value = 100 * 1024**2
        log_file = tmp_path / "test.csv"
        logger = MemoryLogger(log_file)

        with logger:
            pass

        summary = logger.get_summary()
        assert "Memory Usage Summary" in summary
        assert "Entries: 0" in summary
        assert "Duration:" in summary
        assert "GB" in summary

    @patch("oscura.core.memory_progress.get_memory_pressure")
    @patch("oscura.core.memory_progress.get_available_memory")
    @patch("oscura.core.memory_progress.MemoryLogger._get_process_memory")
    def test_get_entries(
        self,
        mock_process_mem: MagicMock,
        mock_available: MagicMock,
        mock_pressure: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test get_entries returns copy."""
        mock_process_mem.return_value = 100 * 1024**2
        mock_available.return_value = 6 * 1024**3
        mock_pressure.return_value = 0.1

        log_file = tmp_path / "test.csv"
        logger = MemoryLogger(log_file)

        with logger:
            logger.log_operation("op1")
            logger.log_operation("op2")

        entries = logger.get_entries()
        assert len(entries) == 2
        assert entries is not logger._entries  # Verify it's a copy


class TestLogMemory:
    """Test log_memory context manager."""

    @patch("oscura.core.memory_progress.MemoryLogger._get_process_memory")
    def test_log_memory_context(self, mock_memory: MagicMock, tmp_path: Path) -> None:
        """Test log_memory context manager."""
        mock_memory.return_value = 100 * 1024**2
        log_file = tmp_path / "test.csv"

        with log_memory(log_file, format="csv", enable_console=False) as logger:
            assert isinstance(logger, MemoryLogger)
            assert logger.format == "csv"


class TestCreateProgressCallback:
    """Test create_progress_callback_with_logging."""

    @patch("oscura.core.memory_progress.get_memory_pressure")
    @patch("oscura.core.memory_progress.get_available_memory")
    @patch("oscura.core.memory_progress.MemoryLogger._get_process_memory")
    def test_create_callback(
        self,
        mock_process_mem: MagicMock,
        mock_available: MagicMock,
        mock_pressure: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test creating progress callback."""
        mock_process_mem.return_value = 100 * 1024**2
        mock_available.return_value = 6 * 1024**3
        mock_pressure.return_value = 0.1

        log_file = tmp_path / "test.csv"
        logger = MemoryLogger(log_file)

        with logger:
            callback = create_progress_callback_with_logging(logger, "test_operation")
            callback(50, 100, "Progress message")

        # Verify log entry was created
        assert len(logger._entries) == 1
        entry = logger._entries[0]
        assert entry.operation == "test_operation"
        assert entry.iteration == 50
        assert entry.message == "Progress message"


class TestEnableMemoryLoggingFromCLI:
    """Test enable_memory_logging_from_cli."""

    def test_cli_disabled(self) -> None:
        """Test when CLI logging is disabled."""
        with patch.dict(os.environ, {}, clear=True):
            logger = enable_memory_logging_from_cli()
            assert logger is None

    def test_cli_enabled(self) -> None:
        """Test when CLI logging is enabled."""
        with patch.dict(os.environ, {"TK_LOG_MEMORY": "1"}):
            logger = enable_memory_logging_from_cli()
            assert logger is not None
            assert isinstance(logger, MemoryLogger)
            assert logger.format == "csv"

    def test_cli_enabled_with_custom_path(self, tmp_path: Path) -> None:
        """Test CLI logging with custom path."""
        log_file = tmp_path / "custom.csv"
        with patch.dict(os.environ, {"TK_LOG_MEMORY": "true"}):
            logger = enable_memory_logging_from_cli(log_file=log_file)
            assert logger is not None
            assert logger.log_file == log_file


class TestCoreMemoryProgressIntegration:
    """Integration tests for memory progress logging."""

    @patch("oscura.core.memory_progress.get_memory_pressure")
    @patch("oscura.core.memory_progress.get_available_memory")
    @patch("oscura.core.memory_progress.MemoryLogger._get_process_memory")
    def test_full_workflow_csv(
        self,
        mock_process_mem: MagicMock,
        mock_available: MagicMock,
        mock_pressure: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test complete CSV logging workflow."""
        mock_process_mem.return_value = 100 * 1024**2
        mock_available.return_value = 6 * 1024**3
        mock_pressure.return_value = 0.1

        log_file = tmp_path / "workflow.csv"

        with MemoryLogger(log_file) as logger:
            for i in range(10):
                logger.log_progress("processing", i + 1, 10)

        # Verify CSV file
        with open(log_file) as f:
            rows = list(csv.DictReader(f))
            assert len(rows) == 10
            assert rows[0]["operation"] == "processing"
            assert rows[0]["iteration"] == "1"

        # Get summary
        summary = logger.get_summary()
        assert "Entries: 10" in summary

    @patch("oscura.core.memory_progress.get_memory_pressure")
    @patch("oscura.core.memory_progress.get_available_memory")
    @patch("oscura.core.memory_progress.MemoryLogger._get_process_memory")
    def test_full_workflow_json(
        self,
        mock_process_mem: MagicMock,
        mock_available: MagicMock,
        mock_pressure: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test complete JSON logging workflow."""
        mock_process_mem.return_value = 100 * 1024**2
        mock_available.return_value = 6 * 1024**3
        mock_pressure.return_value = 0.1

        log_file = tmp_path / "workflow.json"

        with MemoryLogger(log_file, format="json") as logger:
            for i in range(5):
                logger.log_operation("compute", iteration=i)

        # Verify JSON file
        with open(log_file) as f:
            data = json.load(f)
            assert "entries" in data
            assert "summary" in data
            assert len(data["entries"]) == 5
            assert data["summary"]["entry_count"] == 5
