"""Tests for structured logging infrastructure.

Tests the logging framework (LOG-001 through LOG-008).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from oscura.core.logging import (
    LogConfig,
    StructuredFormatter,
    configure_logging,
    format_timestamp,
    get_logger,
    set_log_level,
)

pytestmark = [pytest.mark.unit, pytest.mark.core]


class TestLogConfig:
    """Test LogConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration."""
        config = LogConfig()
        assert config.level == "INFO"
        assert config.format == "text"
        assert config.console_output is True
        assert config.file_output is False

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = LogConfig(
            level="DEBUG",
            format="json",
            console_output=False,
            file_output=True,
            file_path="/tmp/test.log",
        )
        assert config.level == "DEBUG"
        assert config.format == "json"
        assert config.console_output is False
        assert config.file_output is True
        assert config.file_path == "/tmp/test.log"


class TestStructuredFormatter:
    """Test StructuredFormatter class."""

    def test_format_json(self) -> None:
        """Test JSON formatting."""
        formatter = StructuredFormatter(fmt="json")
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        data = json.loads(output)
        assert data["message"] == "Test message"
        assert data["level"] == "INFO"
        assert data["module"] == "test"

    def test_format_logfmt(self) -> None:
        """Test logfmt formatting."""
        formatter = StructuredFormatter(fmt="logfmt")
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        assert "level=INFO" in output
        assert 'message="Test message"' in output
        assert "module=test" in output

    def test_format_text(self) -> None:
        """Test text formatting."""
        formatter = StructuredFormatter(fmt="text")
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        assert "INFO" in output
        assert "Test message" in output


class TestConfigureLogging:
    """Test configure_logging function."""

    def test_configure_console_only(self) -> None:
        """Test configuring console-only logging."""
        configure_logging(level="DEBUG", format="text")
        logger = get_logger("oscura.test")
        assert logger.level == logging.DEBUG or logger.getEffectiveLevel() == logging.DEBUG

    def test_configure_json_format(self) -> None:
        """Test configuring JSON format."""
        configure_logging(level="INFO", format="json")
        logger = get_logger("oscura.test2")
        # Just verify it doesn't crash
        assert logger is not None

    def test_configure_file_output(self, tmp_path: Path) -> None:
        """Test configuring file output."""
        log_file = tmp_path / "test.log"
        configure_logging(
            level="INFO",
            format="text",
            handlers={
                "file": {
                    "filename": str(log_file),
                }
            },
        )
        logger = get_logger("oscura.test3")
        logger.info("Test message")
        # Verify log file was created
        assert log_file.exists()


class TestGetLogger:
    """Test get_logger function."""

    def test_get_logger_returns_logger(self) -> None:
        """Test that get_logger returns a logger."""
        logger = get_logger("oscura.module")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "oscura.module"

    def test_get_logger_same_instance(self) -> None:
        """Test that same name returns same logger."""
        logger1 = get_logger("oscura.same")
        logger2 = get_logger("oscura.same")
        assert logger1 is logger2


class TestSetLogLevel:
    """Test set_log_level function."""

    def test_set_global_level(self) -> None:
        """Test setting global log level."""
        set_log_level("WARNING")
        logger = get_logger("oscura")
        # Level should be WARNING or higher
        assert logger.getEffectiveLevel() >= logging.WARNING

    def test_set_module_level(self) -> None:
        """Test setting module-specific log level."""
        set_log_level("DEBUG", module="oscura.specific")
        logger = get_logger("oscura.specific")
        assert logger.level == logging.DEBUG


class TestFormatTimestamp:
    """Test format_timestamp function."""

    def test_format_iso8601(self) -> None:
        """Test ISO 8601 timestamp formatting."""
        from datetime import UTC, datetime

        dt = datetime.now(UTC)
        result = format_timestamp(dt, "iso8601")
        # Should contain 'T' for ISO format
        assert "T" in result
        assert "Z" in result

    def test_format_unix(self) -> None:
        """Test Unix timestamp formatting."""
        from datetime import UTC, datetime

        dt = datetime.now(UTC)
        result = format_timestamp(dt, "unix")
        # Should be numeric string
        assert result.replace(".", "").replace("-", "").isdigit()


class TestCompressingRotatingFileHandler:
    """Test CompressingRotatingFileHandler class."""

    def test_init(self, tmp_path: Path) -> None:
        """Test handler initialization."""
        from oscura.core.logging import CompressingRotatingFileHandler

        filename = str(tmp_path / "test.log")
        handler = CompressingRotatingFileHandler(
            filename, maxBytes=1000, backupCount=3, compress=True
        )
        assert handler.compress is True
        assert handler.backupCount == 3
        handler.close()

    def test_doRollover_without_compression(self, tmp_path: Path) -> None:
        """Test rollover without compression."""
        from oscura.core.logging import CompressingRotatingFileHandler

        filename = str(tmp_path / "test.log")
        handler = CompressingRotatingFileHandler(
            filename, maxBytes=100, backupCount=2, compress=False
        )
        handler.setFormatter(logging.Formatter("%(message)s"))

        # Write enough to trigger rollover
        for i in range(20):
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="test.py",
                lineno=10,
                msg=f"Long message {i}" * 10,
                args=(),
                exc_info=None,
            )
            handler.emit(record)

        handler.close()

        # Should have created rotated files
        assert Path(filename).exists()

    def test_doRollover_with_compression(self, tmp_path: Path) -> None:
        """Test rollover with compression."""
        from oscura.core.logging import CompressingRotatingFileHandler

        filename = str(tmp_path / "test.log")
        handler = CompressingRotatingFileHandler(
            filename, maxBytes=100, backupCount=2, compress=True
        )
        handler.setFormatter(logging.Formatter("%(message)s"))

        # Write enough to trigger rollover
        for i in range(20):
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="test.py",
                lineno=10,
                msg=f"Long message {i}" * 10,
                args=(),
                exc_info=None,
            )
            handler.emit(record)

        handler.close()

        # Should have compressed rotated file
        assert Path(f"{filename}.1.gz").exists()


class TestCompressingTimedRotatingFileHandler:
    """Test CompressingTimedRotatingFileHandler class."""

    def test_init(self, tmp_path: Path) -> None:
        """Test handler initialization."""
        from oscura.core.logging import CompressingTimedRotatingFileHandler

        filename = str(tmp_path / "timed.log")
        handler = CompressingTimedRotatingFileHandler(
            filename,
            when="midnight",
            interval=1,
            backupCount=5,
            compress=True,
            max_age="30d",
        )
        assert handler.compress is True
        assert handler.max_age == "30d"
        assert handler._max_age_seconds == 30 * 86400
        handler.close()

    def test_parse_max_age_days(self, tmp_path: Path) -> None:
        """Test parsing max_age in days."""
        from oscura.core.logging import CompressingTimedRotatingFileHandler

        filename = str(tmp_path / "timed.log")
        handler = CompressingTimedRotatingFileHandler(filename, max_age="7d")
        assert handler._max_age_seconds == 7 * 86400
        handler.close()

    def test_parse_max_age_hours(self, tmp_path: Path) -> None:
        """Test parsing max_age in hours."""
        from oscura.core.logging import CompressingTimedRotatingFileHandler

        filename = str(tmp_path / "timed.log")
        handler = CompressingTimedRotatingFileHandler(filename, max_age="24h")
        assert handler._max_age_seconds == 24 * 3600
        handler.close()

    def test_parse_max_age_minutes(self, tmp_path: Path) -> None:
        """Test parsing max_age in minutes."""
        from oscura.core.logging import CompressingTimedRotatingFileHandler

        filename = str(tmp_path / "timed.log")
        handler = CompressingTimedRotatingFileHandler(filename, max_age="60m")
        assert handler._max_age_seconds == 60 * 60
        handler.close()

    def test_parse_max_age_invalid(self, tmp_path: Path) -> None:
        """Test parsing invalid max_age format."""
        import warnings

        from oscura.core.logging import CompressingTimedRotatingFileHandler

        filename = str(tmp_path / "timed.log")
        # The handler should raise ValueError with invalid max_age
        # Suppress ResourceWarning as file is opened before validation fails
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            with pytest.raises(ValueError, match="Invalid max_age format"):
                handler = CompressingTimedRotatingFileHandler(filename, max_age="30x")

    def test_cleanup_old_files(self, tmp_path: Path) -> None:
        """Test cleanup of old files."""
        import os
        import time

        from oscura.core.logging import CompressingTimedRotatingFileHandler

        filename = str(tmp_path / "timed.log")

        # Create some old files
        old_file = tmp_path / "timed.log.2020-01-01"
        old_file.write_text("old content")
        # Set modification time to old
        os.utime(str(old_file), (time.time() - 86400 * 60, time.time() - 86400 * 60))

        handler = None
        try:
            handler = CompressingTimedRotatingFileHandler(
                filename, when="H", interval=1, backupCount=5, max_age="30d"
            )

            # Trigger cleanup
            handler._cleanup_old_files()

            # Old file should be removed
            assert not old_file.exists()
        finally:
            if handler:
                handler.close()

    def test_delete_old_files(self, tmp_path: Path) -> None:
        """Test deletion of files exceeding backup count."""
        from oscura.core.logging import CompressingTimedRotatingFileHandler

        filename = str(tmp_path / "timed.log")

        # Create several rotated files
        for i in range(10):
            rotated = tmp_path / f"timed.log.{i}"
            rotated.write_text(f"content {i}")

        handler = CompressingTimedRotatingFileHandler(filename, when="H", interval=1, backupCount=3)

        # Trigger deletion
        handler._delete_old_files()

        handler.close()

        # Should only keep 3 most recent files
        remaining = list(tmp_path.glob("timed.log.*"))
        assert len(remaining) <= 3


class TestStructuredFormatterAdvanced:
    """Advanced tests for StructuredFormatter."""

    def test_format_with_correlation_id(self) -> None:
        """Test formatting with correlation ID."""
        from oscura.core.correlation import set_correlation_id
        from oscura.core.logging import StructuredFormatter

        set_correlation_id("test-correlation-id")
        formatter = StructuredFormatter(fmt="json")
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        output = formatter.format(record)
        data = json.loads(output)

        assert "correlation_id" in data
        assert data["correlation_id"] == "test-correlation-id"

    def test_format_with_extra_fields(self) -> None:
        """Test formatting with extra fields."""
        formatter = StructuredFormatter(fmt="json")
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        # Add extra field
        record.user_id = "user123"
        record.request_id = "req456"

        output = formatter.format(record)
        data = json.loads(output)

        assert data["user_id"] == "user123"
        assert data["request_id"] == "req456"

    def test_format_with_exception(self) -> None:
        """Test formatting with exception info."""
        formatter = StructuredFormatter(fmt="json")
        try:
            raise ValueError("Test error")
        except ValueError:
            import sys

            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=10,
            msg="Error occurred",
            args=(),
            exc_info=exc_info,
        )

        output = formatter.format(record)
        data = json.loads(output)

        assert "exception" in data
        assert "ValueError" in data["exception"]
        assert "Test error" in data["exception"]

    def test_format_timestamp_iso8601(self) -> None:
        """Test ISO 8601 timestamp formatting."""
        formatter = StructuredFormatter(fmt="json", timestamp_format="iso8601")
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test",
            args=(),
            exc_info=None,
        )

        output = formatter.format(record)
        data = json.loads(output)

        assert "timestamp" in data
        assert "T" in data["timestamp"]
        assert data["timestamp"].endswith("Z")

    def test_format_timestamp_iso8601_local(self) -> None:
        """Test ISO 8601 local timestamp formatting."""
        formatter = StructuredFormatter(fmt="json", timestamp_format="iso8601_local")
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test",
            args=(),
            exc_info=None,
        )

        output = formatter.format(record)
        data = json.loads(output)

        assert "timestamp" in data
        assert "T" in data["timestamp"]

    def test_format_timestamp_unix(self) -> None:
        """Test Unix timestamp formatting."""
        formatter = StructuredFormatter(fmt="json", timestamp_format="unix")
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test",
            args=(),
            exc_info=None,
        )

        output = formatter.format(record)
        data = json.loads(output)

        assert "timestamp" in data
        # Unix timestamp should be numeric string
        float(data["timestamp"])  # Should not raise

    def test_format_logfmt_quoted_values(self) -> None:
        """Test logfmt formatting with quoted values."""
        formatter = StructuredFormatter(fmt="logfmt")
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Message with spaces",
            args=(),
            exc_info=None,
        )

        output = formatter.format(record)
        assert 'message="Message with spaces"' in output

    def test_format_text_with_extra(self) -> None:
        """Test text formatting with extra fields."""
        formatter = StructuredFormatter(fmt="text")
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        record.user_id = "user123"

        output = formatter.format(record)
        assert "user_id=user123" in output


class TestErrorContextCapture:
    """Test ErrorContextCapture class."""

    def test_from_exception(self) -> None:
        """Test creating context from exception."""
        from oscura.core.logging import ErrorContextCapture

        try:
            raise ValueError("Test error")
        except ValueError as e:
            context = ErrorContextCapture.from_exception(e)

        assert context.exc_type == ValueError
        assert str(context.exc_value) == "Test error"
        assert context.exc_traceback is not None

    def test_to_dict_basic(self) -> None:
        """Test converting to dict."""
        from oscura.core.logging import ErrorContextCapture

        try:
            raise ValueError("Test error")
        except ValueError as e:
            context = ErrorContextCapture.from_exception(e)
            data = context.to_dict(include_locals=False)

        assert data["exception_type"] == "ValueError"
        assert data["exception_message"] == "Test error"
        assert "traceback" in data

    def test_to_dict_with_locals(self) -> None:
        """Test converting to dict with local variables."""
        from oscura.core.logging import ErrorContextCapture

        def test_func() -> None:
            local_var = "test_value"
            raise ValueError("Test error")

        try:
            test_func()
        except ValueError as e:
            context = ErrorContextCapture.from_exception(e)
            data = context.to_dict(include_locals=True)

        assert "frames" in data
        assert len(data["frames"]) > 0

    def test_to_dict_with_cause(self) -> None:
        """Test exception with cause."""
        from oscura.core.logging import ErrorContextCapture

        try:
            try:
                raise ValueError("Original error")
            except ValueError as e:
                raise RuntimeError("Wrapped error") from e
        except RuntimeError as e:
            context = ErrorContextCapture.from_exception(e)
            data = context.to_dict()

        assert "caused_by" in data
        assert data["caused_by"]["type"] == "ValueError"
        assert data["caused_by"]["message"] == "Original error"

    def test_to_dict_with_context(self) -> None:
        """Test additional context."""
        from oscura.core.logging import ErrorContextCapture

        try:
            raise ValueError("Test error")
        except ValueError as e:
            context = ErrorContextCapture.from_exception(
                e, additional_context={"user_id": "user123", "action": "test_action"}
            )
            data = context.to_dict()

        assert "context" in data
        assert data["context"]["user_id"] == "user123"
        assert data["context"]["action"] == "test_action"

    def test_filter_sensitive_data(self) -> None:
        """Test sensitive data filtering."""
        from oscura.core.logging import ErrorContextCapture

        try:
            raise ValueError("Test")
        except ValueError as e:
            context = ErrorContextCapture.from_exception(e)

        data = {"password": "secret123", "api_key": "key123", "normal_field": "value"}
        filtered = context._filter_sensitive_data(data)

        assert filtered["password"] == "***REDACTED***"
        assert filtered["api_key"] == "***REDACTED***"
        assert filtered["normal_field"] == "value"


class TestLogException:
    """Test log_exception function."""

    def test_log_exception_basic(self) -> None:
        """Test basic exception logging."""
        from oscura.core.logging import log_exception

        logger = MagicMock()

        try:
            raise ValueError("Test error")
        except ValueError as e:
            log_exception(e, logger=logger, include_locals=False)

        assert logger.exception.called

    def test_log_exception_with_context(self) -> None:
        """Test exception logging with context."""
        from oscura.core.logging import log_exception

        logger = MagicMock()

        try:
            raise ValueError("Test error")
        except ValueError as e:
            log_exception(e, logger=logger, context={"user_id": "user123"})

        assert logger.exception.called

    def test_log_exception_with_locals(self) -> None:
        """Test exception logging with locals."""
        from oscura.core.logging import log_exception

        logger = MagicMock()

        def test_func() -> None:
            local_var = "test"
            raise ValueError("Test error")

        try:
            test_func()
        except ValueError as e:
            log_exception(e, logger=logger, include_locals=True)

        assert logger.exception.called


class TestFormatTimestampAdvanced:
    """Test format_timestamp function - advanced cases."""

    def test_format_default(self) -> None:
        """Test default formatting."""
        result = format_timestamp()
        assert "T" in result
        assert result.endswith("Z")

    def test_format_with_datetime(self) -> None:
        """Test formatting with specific datetime."""
        from datetime import UTC, datetime

        dt = datetime(2025, 1, 15, 12, 30, 45, tzinfo=UTC)
        result = format_timestamp(dt, "iso8601")
        assert "2025-01-15" in result
        assert "12:30:45" in result

    def test_format_iso8601_local(self) -> None:
        """Test ISO 8601 local formatting."""
        from datetime import UTC, datetime

        dt = datetime(2025, 1, 15, 12, 30, 45, tzinfo=UTC)
        result = format_timestamp(dt, "iso8601_local")
        assert "T" in result
        assert "2025-01-15" in result

    def test_format_unix(self) -> None:
        """Test Unix timestamp formatting."""
        from datetime import UTC, datetime

        dt = datetime(2025, 1, 15, 12, 30, 45, tzinfo=UTC)
        result = format_timestamp(dt, "unix")
        # Should be numeric
        float(result)

    def test_format_invalid(self) -> None:
        """Test invalid format."""
        from datetime import UTC, datetime

        dt = datetime.now(UTC)
        with pytest.raises(ValueError, match="Unknown timestamp format"):
            format_timestamp(dt, "invalid")  # type: ignore


class TestConfigureLoggingAdvanced:
    """Advanced tests for configure_logging."""

    def test_configure_with_console_handler(self) -> None:
        """Test configuring with console handler."""
        import logging

        # Clear any existing handlers
        root_logger = logging.getLogger("oscura")
        for h in root_logger.handlers[:]:
            h.close()
            root_logger.removeHandler(h)

        configure_logging(
            level="INFO",
            format="json",
            handlers={"console": {"level": "DEBUG"}},
        )
        logger = get_logger("oscura.console_test")
        assert logger is not None

        # Cleanup
        for h in root_logger.handlers[:]:
            h.close()
            root_logger.removeHandler(h)

    def test_configure_with_time_based_rotation(self, tmp_path: Path) -> None:
        """Test configuring with time-based rotation."""
        import logging

        # Clear any existing handlers
        root_logger = logging.getLogger("oscura")
        for h in root_logger.handlers[:]:
            h.close()
            root_logger.removeHandler(h)

        log_file = tmp_path / "timed.log"
        configure_logging(
            level="INFO",
            format="text",
            handlers={
                "file": {
                    "filename": str(log_file),
                    "when": "midnight",
                    "interval": 1,
                    "backup_count": 7,
                }
            },
        )
        logger = get_logger("oscura.timed_test")
        logger.info("Test message")
        assert log_file.exists()

        # Cleanup
        for h in root_logger.handlers[:]:
            h.close()
            root_logger.removeHandler(h)

    def test_configure_with_compression(self, tmp_path: Path) -> None:
        """Test configuring with compression."""
        import logging

        # Clear any existing handlers
        root_logger = logging.getLogger("oscura")
        for h in root_logger.handlers[:]:
            h.close()
            root_logger.removeHandler(h)

        log_file = tmp_path / "compressed.log"
        configure_logging(
            level="INFO",
            format="text",
            handlers={
                "file": {
                    "filename": str(log_file),
                    "max_bytes": 1000,
                    "compress": True,
                }
            },
        )
        logger = get_logger("oscura.compressed_test")
        logger.info("Test message")

        # Cleanup
        for h in root_logger.handlers[:]:
            h.close()
            root_logger.removeHandler(h)

    def test_configure_with_max_age(self, tmp_path: Path) -> None:
        """Test configuring with max_age."""
        import logging

        # Clear any existing handlers
        root_logger = logging.getLogger("oscura")
        for h in root_logger.handlers[:]:
            h.close()
            root_logger.removeHandler(h)

        log_file = tmp_path / "aged.log"
        configure_logging(
            level="INFO",
            format="text",
            handlers={
                "file": {
                    "filename": str(log_file),
                    "when": "H",
                    "max_age": "7d",
                }
            },
        )
        logger = get_logger("oscura.aged_test")
        logger.info("Test message")

        # Cleanup
        for h in root_logger.handlers[:]:
            h.close()
            root_logger.removeHandler(h)

    def test_reconfigure_removes_old_handlers(self) -> None:
        """Test that reconfiguring removes old handlers."""
        import logging

        # Clear any existing handlers
        root_logger = logging.getLogger("oscura")
        for h in root_logger.handlers[:]:
            h.close()
            root_logger.removeHandler(h)

        # First configuration
        configure_logging(level="INFO", format="text")
        logger1 = get_logger("oscura")
        handler_count1 = len(logger1.handlers)

        # Reconfigure
        configure_logging(level="DEBUG", format="json")
        logger2 = get_logger("oscura")
        handler_count2 = len(logger2.handlers)

        # Should have same number of handlers (old ones removed)
        assert handler_count1 == handler_count2

        # Cleanup
        for h in root_logger.handlers[:]:
            h.close()
            root_logger.removeHandler(h)


class TestGetLoggerAdvanced:
    """Advanced tests for get_logger."""

    def test_get_logger_auto_configures(self) -> None:
        """Test that get_logger auto-configures if needed."""
        from oscura.core import logging as logging_module

        # Reset configuration flag
        logging_module._logging_configured = False

        logger = get_logger("oscura.auto")
        assert logger is not None
        assert logging_module._logging_configured is True

    def test_get_logger_adds_prefix(self) -> None:
        """Test that logger name is prefixed."""
        logger = get_logger("mymodule")
        assert logger.name == "oscura.mymodule"

    def test_get_logger_no_double_prefix(self) -> None:
        """Test that prefix is not added twice."""
        logger = get_logger("oscura.mymodule")
        assert logger.name == "oscura.mymodule"


class TestInitLogging:
    """Test _init_logging function."""

    def test_init_logging_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test initialization from environment variables."""
        from oscura.core import logging as logging_module

        monkeypatch.setenv("OSCURA_LOG_LEVEL", "DEBUG")
        monkeypatch.setenv("OSCURA_LOG_FORMAT", "json")

        # Reset and re-initialize
        logging_module._logging_configured = False
        logging_module._init_logging()

        assert logging_module._logging_configured is True

    def test_init_logging_invalid_format(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test initialization with invalid format."""
        from oscura.core import logging as logging_module

        monkeypatch.setenv("OSCURA_LOG_LEVEL", "INFO")
        monkeypatch.setenv("OSCURA_LOG_FORMAT", "invalid")

        # Reset and re-initialize
        logging_module._logging_configured = False
        logging_module._init_logging()

        # Should still configure with default format
        assert logging_module._logging_configured is True


class TestCoreLoggingIntegration:
    """Integration tests for logging."""

    def test_complete_workflow(self, tmp_path: Path) -> None:
        """Test complete logging workflow."""
        log_file = tmp_path / "workflow.log"

        # Configure logging
        configure_logging(
            level="DEBUG",
            format="json",
            handlers={
                "file": {
                    "filename": str(log_file),
                }
            },
        )

        # Get logger and log messages
        logger = get_logger("oscura.workflow")
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")

        # Verify log file exists and has content
        assert log_file.exists()
        content = log_file.read_text()
        assert len(content) > 0

    def test_correlation_workflow(self) -> None:
        """Test logging with correlation IDs."""
        from oscura.core.correlation import set_correlation_id

        configure_logging(level="INFO", format="json")
        logger = get_logger("oscura.correlation_test")

        # Set correlation ID directly
        set_correlation_id("test-correlation")
        logger.info("Message with correlation")

        # Clear correlation ID
        set_correlation_id(None)

    def test_performance_workflow(self) -> None:
        """Test logging with performance timing."""
        from oscura.core.performance import timed

        configure_logging(level="INFO", format="text")

        @timed()
        def timed_function() -> int:
            return 42

        result = timed_function()
        assert result == 42

    def test_error_logging_workflow(self, tmp_path: Path) -> None:
        """Test error logging workflow."""
        from oscura.core.logging import log_exception

        log_file = tmp_path / "errors.log"
        configure_logging(
            level="ERROR",
            format="json",
            handlers={"file": {"filename": str(log_file)}},
        )

        logger = get_logger("oscura.error_test")

        try:
            raise ValueError("Test error")
        except ValueError as e:
            log_exception(e, logger=logger, include_locals=True)

        # Verify error was logged
        assert log_file.exists()
        content = log_file.read_text()
        assert "ValueError" in content
