"""Tests for core logging module.

Tests requirements:
- LOG-005: ISO 8601 Timestamps
"""

import logging
import os
import tempfile
import time
from pathlib import Path

import pytest

from oscura.core.logging import (
    CompressingRotatingFileHandler,
    CompressingTimedRotatingFileHandler,
    CorrelationContext,
    StructuredFormatter,
    configure_logging,
    get_correlation_id,
    get_logger,
    get_performance_summary,
    log_exception,
    set_correlation_id,
    set_log_level,
    timed,
    with_correlation_id,
)

pytestmark = [pytest.mark.unit, pytest.mark.core]


@pytest.fixture(autouse=True)
def reset_logging():
    """Reset logging state before and after each test to prevent cross-test contamination."""
    # Setup: Clear handlers before test
    root_logger = logging.getLogger()
    oscura_logger = logging.getLogger("oscura")

    # Save original state
    original_root_handlers = root_logger.handlers.copy()
    original_root_level = root_logger.level
    original_oscura_handlers = oscura_logger.handlers.copy()
    original_oscura_level = oscura_logger.level

    yield

    # Teardown: Remove all handlers added during test
    for handler in root_logger.handlers[:]:
        handler.close()
        root_logger.removeHandler(handler)

    for handler in oscura_logger.handlers[:]:
        handler.close()
        oscura_logger.removeHandler(handler)

    # Restore original handlers
    for handler in original_root_handlers:
        if handler not in root_logger.handlers:
            root_logger.addHandler(handler)

    for handler in original_oscura_handlers:
        if handler not in oscura_logger.handlers:
            oscura_logger.addHandler(handler)

    # Restore original levels
    root_logger.setLevel(original_root_level)
    oscura_logger.setLevel(original_oscura_level)


class TestConfigureLogging:
    """Tests for logging configuration (LOG-001)."""

    def test_configure_default(self):
        """Test default configuration."""
        configure_logging()
        logger = get_logger("test")
        assert logger is not None

    def test_configure_json_format(self):
        """Test JSON format configuration."""
        configure_logging(format="json", level="DEBUG")
        logger = get_logger("test.json")
        # Just verify it doesn't crash
        logger.info("Test message")

    def test_configure_logfmt(self):
        """Test logfmt format configuration."""
        configure_logging(format="logfmt", level="DEBUG")
        logger = get_logger("test.logfmt")
        logger.info("Test message")


class TestGetLogger:
    """Tests for logger retrieval."""

    def test_get_logger(self):
        """Test getting a logger."""
        logger = get_logger("oscura.test")
        assert logger is not None
        assert isinstance(logger, logging.Logger)

    def test_logger_hierarchy(self):
        """Test logger namespace hierarchy."""
        logger = get_logger("test.module")
        # Should be under oscura namespace
        assert logger.name.startswith("oscura")


class TestSetLogLevel:
    """Tests for log level setting (LOG-002)."""

    def test_set_global_level(self):
        """Test setting global log level."""
        set_log_level("DEBUG")
        get_logger("test.level")
        # Should be able to log at DEBUG level

    def test_set_module_level(self):
        """Test setting module-specific level."""
        set_log_level("ERROR", "oscura.test")
        get_logger("oscura.test.child")


class TestLogRotation:
    """Tests for log rotation (LOG-003)."""

    def setup_method(self):
        """Reset logging state before each test."""
        # Clear all handlers from root logger
        root = logging.getLogger()
        for handler in root.handlers[:]:
            handler.close()
            root.removeHandler(handler)

    def test_compressing_rotating_handler_init(self):
        """Test CompressingRotatingFileHandler initialization.

        References:
            LOG-003: Size-based rotation with compression
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.log"
            handler = CompressingRotatingFileHandler(
                str(log_path),
                maxBytes=1024,
                backupCount=3,
                compress=True,
            )
            assert handler.compress is True
            assert handler.backupCount == 3
            handler.close()

    def test_compressing_timed_handler_init(self):
        """Test CompressingTimedRotatingFileHandler initialization.

        References:
            LOG-003: Time-based rotation with compression
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.log"
            handler = CompressingTimedRotatingFileHandler(
                str(log_path),
                when="midnight",
                interval=1,
                backupCount=30,
                compress=True,
                max_age="30d",
            )
            assert handler.compress is True
            assert handler.max_age == "30d"
            assert handler._max_age_seconds == 30 * 86400
            handler.close()

    def test_parse_max_age_days(self):
        """Test max_age parsing for days.

        References:
            LOG-003: Retention age configuration
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.log"
            handler = CompressingTimedRotatingFileHandler(
                str(log_path),
                when="H",
                max_age="7d",
            )
            assert handler._max_age_seconds == 7 * 86400
            handler.close()

    def test_parse_max_age_hours(self):
        """Test max_age parsing for hours.

        References:
            LOG-003: Retention age configuration
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.log"
            handler = CompressingTimedRotatingFileHandler(
                str(log_path),
                when="H",
                max_age="24h",
            )
            assert handler._max_age_seconds == 24 * 3600
            handler.close()

    def test_parse_max_age_minutes(self):
        """Test max_age parsing for minutes.

        References:
            LOG-003: Retention age configuration
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.log"
            handler = CompressingTimedRotatingFileHandler(
                str(log_path),
                when="H",
                max_age="60m",
            )
            assert handler._max_age_seconds == 60 * 60
            handler.close()

    def test_configure_time_based_rotation(self):
        """Test configuring time-based log rotation.

        References:
            LOG-003: Time-based rotation via configure_logging
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "app.log"
            configure_logging(
                level="DEBUG",
                handlers={
                    "file": {
                        "filename": str(log_path),
                        "when": "midnight",
                        "interval": 1,
                        "backup_count": 30,
                        "compress": True,
                        "max_age": "30d",
                    }
                },
            )
            logger = get_logger("test.rotation")
            logger.info("Test time-based rotation")
            # Verify log file was created
            assert log_path.exists()

    def test_configure_size_based_rotation_with_compression(self):
        """Test configuring size-based rotation with compression.

        References:
            LOG-003: Size-based rotation with compression
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "app.log"
            configure_logging(
                level="DEBUG",
                handlers={
                    "file": {
                        "filename": str(log_path),
                        "max_bytes": 10_000_000,
                        "backup_count": 5,
                        "compress": True,
                    }
                },
            )
            logger = get_logger("test.size_rotation")
            logger.info("Test size-based rotation with compression")
            assert log_path.exists()


class TestCorrelationId:
    """Tests for correlation ID (LOG-004)."""

    def test_get_correlation_id_default(self):
        """Test default correlation ID is None."""
        # Reset correlation ID
        set_correlation_id("")
        # Get returns None or empty
        cid = get_correlation_id()
        assert cid is None or cid == ""

    def test_set_correlation_id(self):
        """Test setting correlation ID."""
        set_correlation_id("test-123")
        assert get_correlation_id() == "test-123"

    def test_correlation_context(self):
        """Test correlation context manager."""
        with CorrelationContext("ctx-456") as cid:
            assert cid == "ctx-456"
            assert get_correlation_id() == "ctx-456"

    def test_correlation_auto_generate(self):
        """Test auto-generated correlation ID."""
        with CorrelationContext() as cid:
            assert cid is not None
            assert len(cid) > 0


class TestWithCorrelationId:
    """Tests for correlation ID decorator."""

    def test_decorator_sets_id(self):
        """Test decorator sets correlation ID."""

        @with_correlation_id("decorator-test")
        def my_function():
            return get_correlation_id()

        result = my_function()
        assert result == "decorator-test"


class TestStructuredFormatter:
    """Tests for structured formatter (LOG-005)."""

    def test_json_format(self):
        """Test JSON output format."""
        formatter = StructuredFormatter("json")
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        assert '"level": "INFO"' in output
        assert '"message": "Test message"' in output

    def test_logfmt_format(self):
        """Test logfmt output format."""
        formatter = StructuredFormatter("logfmt")
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        assert "level=INFO" in output

    def test_text_format(self):
        """Test plain text output format."""
        formatter = StructuredFormatter("text")
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        assert "INFO" in output
        assert "Test message" in output

    def test_iso8601_timestamp(self):
        """Test ISO 8601 timestamp format."""
        formatter = StructuredFormatter("json", "iso8601")
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        # Should contain ISO format timestamp
        assert "T" in output  # ISO 8601 date-time separator


class TestTimedDecorator:
    """Tests for performance timing (LOG-006)."""

    def test_timed_decorator(self):
        """Test timing decorator."""

        @timed()
        def slow_function():
            time.sleep(0.01)
            return "done"

        result = slow_function()
        assert result == "done"

    def test_timed_with_threshold(self):
        """Test timing decorator with threshold."""

        @timed(threshold=1.0)  # Only log if > 1 second
        def fast_function():
            return "fast"

        result = fast_function()
        assert result == "fast"

    def test_performance_summary(self):
        """Test performance summary."""

        @timed()
        def counted_function():
            pass

        counted_function()
        counted_function()

        summary = get_performance_summary()
        # Should have recorded the function
        assert isinstance(summary, dict)


class TestLogException:
    """Tests for exception logging."""

    def test_log_exception(self):
        """Test logging an exception."""
        logger = get_logger("test.exception")
        try:
            raise ValueError("Test error")
        except ValueError as e:
            # Should not raise
            log_exception(e, logger, context={"key": "value"})


class TestEnvironmentConfiguration:
    """Tests for environment-based configuration."""

    def setup_method(self):
        """Reset logging state before each test."""
        # Clear all handlers from root logger
        root = logging.getLogger()
        for handler in root.handlers[:]:
            handler.close()
            root.removeHandler(handler)

    def test_log_level_from_env(self):
        """Test log level from environment variable."""
        original = os.environ.get("OSCURA_LOG_LEVEL")
        try:
            os.environ["OSCURA_LOG_LEVEL"] = "DEBUG"
            # Reconfigure
            configure_logging()
        finally:
            if original:
                os.environ["OSCURA_LOG_LEVEL"] = original
            elif "OSCURA_LOG_LEVEL" in os.environ:
                del os.environ["OSCURA_LOG_LEVEL"]
