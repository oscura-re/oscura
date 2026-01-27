"""Structured logging infrastructure for Oscura.

This module provides structured logging with JSON/logfmt support,
hierarchical loggers, log rotation, and error context capture.


Example:
    >>> from oscura.core.logging import configure_logging, get_logger
    >>> configure_logging(format='json', level='INFO')
    >>> logger = get_logger('oscura.loaders')
    >>> logger.info("Loading trace", file="data.bin", size_mb=1024)

References:
    Python logging module best practices
    LOG-001 through LOG-008 requirements
"""

from __future__ import annotations

import gzip
import json
import logging
import logging.handlers
import os
import shutil
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

# Global logging configuration
_logging_configured = False
_root_logger_name = "oscura"


@dataclass
class LogConfig:
    """Logging configuration.

    Attributes:
        level: Default log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        format: Output format (json, logfmt, text).
        timestamp_format: Timestamp format (iso8601, iso8601_local, unix, custom).
        custom_timestamp_format: Custom timestamp format string.
        console_output: Enable console output to stderr.
        file_output: Enable file output.
        file_path: Path to log file.
        max_bytes: Maximum log file size before rotation.
        backup_count: Number of rotated log files to keep.
        compress: Compress rotated log files.
        when: Time-based rotation interval type ('midnight', 'H', 'D', 'W0'-'W6').
        interval: Interval for time-based rotation.
        max_age: Maximum age for log files (e.g., '30d').

    References:
        LOG-001: Structured Logging Framework
        LOG-003: Automatic Log Rotation and Retention Policies
    """

    level: str = "INFO"
    format: Literal["json", "logfmt", "text"] = "text"
    timestamp_format: Literal["iso8601", "iso8601_local", "unix", "custom"] = "iso8601"
    custom_timestamp_format: str | None = None
    console_output: bool = True
    file_output: bool = False
    file_path: str | None = None
    max_bytes: int = 10_000_000  # 10 MB
    backup_count: int = 5
    compress: bool = False
    when: str | None = None  # Time-based rotation: 'midnight', 'H', 'D', 'W0'-'W6'
    interval: int = 1  # Interval for time-based rotation
    max_age: str | None = None  # Maximum age for log files (e.g., '30d')


# Default configuration
_config = LogConfig()


class CompressingRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """RotatingFileHandler that compresses rotated files with gzip.

    Extends standard RotatingFileHandler to optionally compress log files
    when they are rotated, saving disk space for historical logs.

    Args:
        filename: Path to log file.
        mode: File open mode.
        maxBytes: Maximum file size before rotation.
        backupCount: Number of backup files to keep.
        encoding: File encoding.
        compress: Whether to gzip compress rotated files.

    References:
        LOG-003: Automatic Log Rotation and Retention Policies
    """

    def __init__(
        self,
        filename: str,
        mode: str = "a",
        maxBytes: int = 0,
        backupCount: int = 0,
        encoding: str | None = None,
        compress: bool = False,
    ):
        """Initialize compressing rotating file handler.

        Args:
            filename: Path to log file.
            mode: File open mode.
            maxBytes: Maximum file size before rotation.
            backupCount: Number of backup files to keep.
            encoding: File encoding.
            compress: Whether to gzip compress rotated files.
        """
        super().__init__(filename, mode, maxBytes, backupCount, encoding)
        self.compress = compress

    def doRollover(self) -> None:
        """Perform rollover and optionally compress the rotated file.

        References:
            LOG-003: Automatic Log Rotation and Retention Policies
        """
        # Standard rollover
        super().doRollover()

        # Compress the rolled file if compression is enabled
        if self.compress and self.backupCount > 0:
            # The most recently rotated file is .1
            rotated_file = f"{self.baseFilename}.1"
            compressed_file = f"{rotated_file}.gz"

            if Path(rotated_file).exists():
                # Compress the file
                with open(rotated_file, "rb") as f_in:
                    with gzip.open(compressed_file, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)

                # Remove the uncompressed file
                Path(rotated_file).unlink()


class CompressingTimedRotatingFileHandler(logging.handlers.TimedRotatingFileHandler):
    """TimedRotatingFileHandler that compresses rotated files with gzip.

    Extends standard TimedRotatingFileHandler to optionally compress log files
    when they are rotated, saving disk space for historical logs.

    Args:
        filename: Path to log file.
        when: Type of interval ('midnight', 'H', 'D', 'W0'-'W6').
        interval: Number of intervals between rotations.
        backupCount: Number of backup files to keep.
        encoding: File encoding.
        compress: Whether to gzip compress rotated files.
        max_age: Maximum age for log files (e.g., '30d').

    References:
        LOG-003: Automatic Log Rotation and Retention Policies
    """

    def __init__(
        self,
        filename: str,
        when: str = "midnight",
        interval: int = 1,
        backupCount: int = 0,
        encoding: str | None = None,
        compress: bool = False,
        max_age: str | None = None,
    ):
        """Initialize compressing timed rotating file handler.

        Args:
            filename: Path to log file.
            when: Type of interval ('midnight', 'H', 'D', 'W0'-'W6').
            interval: Number of intervals between rotations.
            backupCount: Number of backup files to keep.
            encoding: File encoding.
            compress: Whether to gzip compress rotated files.
            max_age: Maximum age for log files (e.g., '30d').
        """
        super().__init__(filename, when, interval, backupCount, encoding=encoding)
        self.compress = compress
        self.max_age = max_age
        self._max_age_seconds = self._parse_max_age(max_age) if max_age else None

    def _parse_max_age(self, max_age: str) -> int:
        """Parse max_age string to seconds.

        Args:
            max_age: Age string like '30d', '7d', '24h'.

        Returns:
            Number of seconds.

        Raises:
            ValueError: If max_age format is invalid.
        """
        if max_age.endswith("d"):
            return int(max_age[:-1]) * 86400  # days to seconds
        elif max_age.endswith("h"):
            return int(max_age[:-1]) * 3600  # hours to seconds
        elif max_age.endswith("m"):
            return int(max_age[:-1]) * 60  # minutes to seconds
        else:
            raise ValueError(f"Invalid max_age format: {max_age}. Use 'd', 'h', or 'm' suffix.")

    def doRollover(self) -> None:
        """Perform rollover and optionally compress the rotated file.

        Also cleans up files older than max_age if specified.

        References:
            LOG-003: Automatic Log Rotation and Retention Policies
        """
        # Close stream before rollover
        if self.stream is not None:
            self.stream.close()
            self.stream = None  # type: ignore[assignment]

        # Determine the file that just got rotated
        current_time = int(self.rolloverAt - self.interval)
        time_tuple = time.gmtime(current_time) if self.utc else time.localtime(current_time)
        dfn = self.rotation_filename(self.baseFilename + "." + self.suffix % time_tuple[:6])

        # Handle the existing rotated file
        if Path(dfn).exists():
            Path(dfn).unlink()

        # Rotate the current file
        self.rotate(self.baseFilename, dfn)

        # Compress if enabled
        if self.compress and Path(dfn).exists():
            compressed_file = f"{dfn}.gz"
            with open(dfn, "rb") as f_in, gzip.open(compressed_file, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
            Path(dfn).unlink()

        # Clean up old files based on max_age
        if self._max_age_seconds:
            self._cleanup_old_files()

        # Delete old files based on backupCount
        if self.backupCount > 0:
            self._delete_old_files()

        # Set next rollover time
        new_rollover_at = self.computeRollover(current_time)
        while new_rollover_at <= self.rolloverAt:
            new_rollover_at = new_rollover_at + self.interval
        self.rolloverAt = new_rollover_at

        # Open new log file
        if not self.delay:
            self.stream = self._open()

    def _cleanup_old_files(self) -> None:
        """Remove log files older than max_age.

        References:
            LOG-003: Automatic Log Rotation and Retention Policies
        """
        if not self._max_age_seconds:
            return

        now = datetime.now().timestamp()
        base_path = Path(self.baseFilename)
        log_dir = base_path.parent
        base_name = base_path.name

        for log_file in log_dir.glob(f"{base_name}.*"):
            try:
                file_age = now - log_file.stat().st_mtime
                if file_age > self._max_age_seconds:
                    log_file.unlink()
            except OSError:
                pass  # Ignore errors during cleanup

    def _delete_old_files(self) -> None:
        """Delete files exceeding backup count.

        References:
            LOG-003: Automatic Log Rotation and Retention Policies
        """
        base_path = Path(self.baseFilename)
        log_dir = base_path.parent
        base_name = base_path.name

        # Get all rotated files
        rotated_files = sorted(
            log_dir.glob(f"{base_name}.*"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        # Remove files beyond backup count
        for old_file in rotated_files[self.backupCount :]:
            try:
                old_file.unlink()
            except OSError:
                pass


class StructuredFormatter(logging.Formatter):
    """Formatter that produces structured log output (JSON or logfmt).

    Supports multiple output formats with ISO 8601 timestamps and
    automatic correlation ID injection.

    Args:
        fmt: Output format (json, logfmt, text).
        timestamp_format: Timestamp format (iso8601, iso8601_local, unix).

    References:
        LOG-001: Structured Logging Framework
        LOG-005: ISO 8601 Timestamps
    """

    def __init__(
        self,
        fmt: Literal["json", "logfmt", "text"] = "text",
        timestamp_format: str = "iso8601",
    ):
        """Initialize structured formatter.

        Args:
            fmt: Output format.
            timestamp_format: Timestamp format.
        """
        super().__init__()
        self.fmt = fmt
        self.timestamp_format = timestamp_format

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured output.

        Args:
            record: Log record to format.

        Returns:
            Formatted log string.
        """
        # Get timestamp
        timestamp = self._format_timestamp(record.created)

        # Build structured data
        data = {
            "timestamp": timestamp,
            "level": record.levelname,
            "module": record.name,
            "message": record.getMessage(),
        }

        # Add correlation ID if present
        try:
            from oscura.core.correlation import get_correlation_id

            corr_id = get_correlation_id()
            if corr_id:
                data["correlation_id"] = corr_id
        except ImportError:
            pass  # Correlation module not yet loaded

        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in (
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "exc_info",
                "exc_text",
                "stack_info",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
                "message",
                "asctime",
            ):
                data[key] = value

        # Add exception info if present
        if record.exc_info:
            data["exception"] = self.formatException(record.exc_info)

        if self.fmt == "json":
            return json.dumps(data, default=str)
        elif self.fmt == "logfmt":
            return self._format_logfmt(data)
        else:
            # Plain text format
            extra = " ".join(
                f"{k}={v}"
                for k, v in data.items()
                if k not in ("timestamp", "level", "module", "message")
            )
            base = f"{timestamp} [{record.levelname}] {record.name}: {record.getMessage()}"
            if extra:
                base += f" | {extra}"
            return base

    def _format_timestamp(self, created: float) -> str:
        """Format timestamp according to configuration.

        Args:
            created: Timestamp as float (seconds since epoch).

        Returns:
            Formatted timestamp string.

        References:
            LOG-005: ISO 8601 Timestamps
        """
        dt = datetime.fromtimestamp(created, tz=UTC)
        if self.timestamp_format == "iso8601":
            # ISO 8601 with microseconds: 2025-12-20T15:30:45.123456Z
            return dt.strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"
        elif self.timestamp_format == "iso8601_local":
            dt_local = datetime.fromtimestamp(created)
            return dt_local.strftime("%Y-%m-%dT%H:%M:%S.%f")
        elif self.timestamp_format == "unix":
            return str(created)
        else:
            return dt.strftime(self.timestamp_format)

    def _format_logfmt(self, data: dict) -> str:  # type: ignore[type-arg]
        """Format data as logfmt (key=value pairs).

        Args:
            data: Dictionary to format.

        Returns:
            Logfmt formatted string.
        """
        parts = []
        for key, value in data.items():
            if isinstance(value, str) and (" " in value or '"' in value):
                # Quote values with spaces
                value_str = f'"{value.replace(chr(34), chr(92) + chr(34))}"'
            else:
                value_str = str(value)
            parts.append(f"{key}={value_str}")
        return " ".join(parts)


def configure_logging(
    *,
    level: str = "INFO",
    format: Literal["json", "logfmt", "text"] = "text",
    timestamp_format: Literal["iso8601", "iso8601_local", "unix"] = "iso8601",
    handlers: dict[str, dict[str, Any]] | None = None,
) -> None:
    """Configure Oscura logging.

    Sets up structured logging with the specified format and handlers.
    Supports both size-based and time-based log rotation with optional
    gzip compression.

    Args:
        level: Default log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        format: Output format (json, logfmt, or text).
        timestamp_format: Timestamp format (iso8601, iso8601_local, unix).
        handlers: Dict of handler configurations. Supported handlers:
            - 'console': Console output to stderr.
                - level: Log level for this handler.
            - 'file': File output with rotation.
                - filename: Path to log file.
                - level: Log level for this handler.
                - max_bytes: Max file size before rotation (size-based).
                - backup_count: Number of rotated files to keep.
                - compress: Gzip compress rotated files.
                - when: Time-based rotation ('midnight', 'H', 'D', 'W0'-'W6').
                - interval: Interval for time-based rotation.
                - max_age: Max age for log files (e.g., '30d').

    Example:
        >>> # Size-based rotation with compression
        >>> configure_logging(handlers={
        ...     'file': {'filename': 'app.log', 'max_bytes': 10e6, 'compress': True}
        ... })
        >>> # Time-based daily rotation
        >>> configure_logging(handlers={
        ...     'file': {'filename': 'app.log', 'when': 'midnight', 'backup_count': 30}
        ... })
        >>> # Combined: time-based with max_age cleanup
        >>> configure_logging(handlers={
        ...     'file': {'filename': 'app.log', 'when': 'midnight',
        ...              'compress': True, 'max_age': '30d'}
        ... })

    References:
        LOG-001: Structured Logging Framework
        LOG-002: Hierarchical Log Levels
        LOG-003: Automatic Log Rotation and Retention Policies
    """
    global _logging_configured, _config

    # Update config
    _config.level = level
    _config.format = format
    _config.timestamp_format = timestamp_format

    # Get or create root logger
    root_logger = logging.getLogger(_root_logger_name)
    root_logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers and close them to prevent resource leaks
    _cleanup_existing_handlers(root_logger)

    # Create formatter
    formatter = StructuredFormatter(format, timestamp_format)

    # Add handlers
    if handlers:
        _add_configured_handlers(root_logger, handlers, formatter, level)
    else:
        _add_default_console_handler(root_logger, formatter, level)

    _logging_configured = True


def _cleanup_existing_handlers(logger: logging.Logger) -> None:
    """Remove and close existing handlers.

    Args:
        logger: Logger to cleanup.
    """
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)


def _add_configured_handlers(
    logger: logging.Logger,
    handlers: dict[str, dict[str, Any]],
    formatter: StructuredFormatter,
    default_level: str,
) -> None:
    """Add configured handlers to logger.

    Args:
        logger: Logger to add handlers to.
        handlers: Handler configuration dict.
        formatter: Formatter to use.
        default_level: Default log level.
    """
    for name, config in handlers.items():
        if name == "console":
            _add_console_handler(logger, config, formatter, default_level)
        elif name == "file":
            _add_file_handler(logger, config, formatter)


def _add_console_handler(
    logger: logging.Logger,
    config: dict[str, Any],
    formatter: StructuredFormatter,
    default_level: str,
) -> None:
    """Add console handler to logger.

    Args:
        logger: Logger to add handler to.
        config: Handler configuration.
        formatter: Formatter to use.
        default_level: Default log level.
    """
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(getattr(logging, config.get("level", default_level).upper()))
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def _add_file_handler(
    logger: logging.Logger,
    config: dict[str, Any],
    formatter: StructuredFormatter,
) -> None:
    """Add file handler to logger with rotation support.

    Args:
        logger: Logger to add handler to.
        config: Handler configuration.
        formatter: Formatter to use.
    """
    filename = config.get("filename", "oscura.log")
    handler_level = config.get("level", "DEBUG")
    backup_count = int(config.get("backup_count", 5))
    compress = config.get("compress", False)

    # Check if time-based rotation is requested
    when = config.get("when")
    if when:
        # Time-based rotation
        interval = int(config.get("interval", 1))
        max_age = config.get("max_age")
        time_handler = CompressingTimedRotatingFileHandler(
            filename,
            when=when,
            interval=interval,
            backupCount=backup_count,
            compress=compress,
            max_age=max_age,
        )
        time_handler.setLevel(getattr(logging, handler_level.upper()))
        time_handler.setFormatter(formatter)
        logger.addHandler(time_handler)
    else:
        # Size-based rotation
        max_bytes = int(config.get("max_bytes", 10_000_000))
        size_handler = CompressingRotatingFileHandler(
            filename,
            maxBytes=max_bytes,
            backupCount=backup_count,
            compress=compress,
        )
        size_handler.setLevel(getattr(logging, handler_level.upper()))
        size_handler.setFormatter(formatter)
        logger.addHandler(size_handler)


def _add_default_console_handler(
    logger: logging.Logger,
    formatter: StructuredFormatter,
    level: str,
) -> None:
    """Add default console handler when no handlers configured.

    Args:
        logger: Logger to add handler to.
        formatter: Formatter to use.
        level: Log level.
    """
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(getattr(logging, level.upper()))
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name.

    Returns a logger under the oscura namespace with proper
    configuration.

    Args:
        name: Logger name (e.g., 'oscura.loaders.binary').

    Returns:
        Configured logging.Logger instance.

    Example:
        >>> logger = get_logger('oscura.analyzers.spectral')
        >>> logger.info("Computing FFT", samples=1000000)

    References:
        LOG-001: Structured Logging Framework
    """
    if not _logging_configured:
        # Auto-configure with defaults
        configure_logging()

    # Ensure name is under oscura namespace
    if not name.startswith(_root_logger_name):
        name = f"{_root_logger_name}.{name}"

    return logging.getLogger(name)


def set_log_level(level: str, module: str | None = None) -> None:
    """Set log level globally or for a specific module.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        module: Module name to set level for, or None for global.

    Example:
        >>> set_log_level('DEBUG')  # Global
        >>> set_log_level('DEBUG', 'oscura.loaders')  # Module-specific

    References:
        LOG-002: Hierarchical Log Levels
    """
    logger = logging.getLogger(module) if module else logging.getLogger(_root_logger_name)

    logger.setLevel(getattr(logging, level.upper()))


class ErrorContextCapture:
    """Captures rich error context including stack traces and local variables.

    Provides detailed error information for debugging including:
    - Full stack trace
    - Local variables at each frame
    - Exception chain
    - System information

    Example:
        >>> try:
        ...     risky_operation()
        ... except Exception as exc:
        ...     context = ErrorContextCapture.from_exception(exc, include_locals=True)
        ...     logger.error("Operation failed", extra=context.to_dict())

    References:
        LOG-008: Rich Error Context with Stack Traces
        CORE-006: Helpful exception messages
    """

    def __init__(
        self,
        exc_type: type[BaseException],
        exc_value: BaseException,
        exc_traceback: Any,
        additional_context: dict[str, Any] | None = None,
    ):
        """Initialize error context capture.

        Args:
            exc_type: Exception type.
            exc_value: Exception instance.
            exc_traceback: Exception traceback.
            additional_context: Additional context to include.
        """
        self.exc_type = exc_type
        self.exc_value = exc_value
        self.exc_traceback = exc_traceback
        self.additional_context = additional_context or {}

    @classmethod
    def from_exception(
        cls,
        exc: BaseException,
        include_locals: bool = True,
        additional_context: dict[str, Any] | None = None,
    ) -> ErrorContextCapture:
        """Create error context from an exception.

        Args:
            exc: Exception to capture.
            include_locals: Whether to include local variables.
            additional_context: Additional context to include.

        Returns:
            ErrorContextCapture instance.
        """
        exc_type = type(exc)
        exc_value = exc
        exc_traceback = exc.__traceback__
        return cls(exc_type, exc_value, exc_traceback, additional_context)

    def to_dict(self, include_locals: bool = True) -> dict[str, Any]:
        """Convert error context to dictionary.

        Args:
            include_locals: Whether to include local variables.

        Returns:
            Dictionary with error context.

        References:
            LOG-008: Rich Error Context
        """
        result: dict[str, Any] = {
            "exception_type": self.exc_type.__name__,
            "exception_module": self.exc_type.__module__,
            "exception_message": str(self.exc_value),
            "traceback": traceback.format_exception(
                self.exc_type, self.exc_value, self.exc_traceback
            ),
        }

        # Add exception chain
        if hasattr(self.exc_value, "__cause__") and self.exc_value.__cause__:
            result["caused_by"] = {
                "type": type(self.exc_value.__cause__).__name__,
                "message": str(self.exc_value.__cause__),
            }

        # Add local variables if requested
        if include_locals and self.exc_traceback:
            frames = []
            tb = self.exc_traceback
            while tb is not None:
                frame = tb.tb_frame
                frames.append(
                    {
                        "filename": frame.f_code.co_filename,
                        "function": frame.f_code.co_name,
                        "lineno": tb.tb_lineno,
                        "locals": self._filter_sensitive_data(
                            {k: repr(v) for k, v in frame.f_locals.items()}
                        ),
                    }
                )
                tb = tb.tb_next
            result["frames"] = frames

        # Add additional context
        if self.additional_context:
            result["context"] = self.additional_context

        return result

    def _filter_sensitive_data(self, data: dict[str, str]) -> dict[str, str]:
        """Filter sensitive data from local variables.

        Args:
            data: Dictionary of local variables.

        Returns:
            Filtered dictionary with sensitive data redacted.

        References:
            LOG-008: Rich Error Context (sensitive data filtering)
        """
        sensitive_keys = {
            "password",
            "passwd",
            "pwd",
            "secret",
            "token",
            "api_key",
            "apikey",
            "auth",
            "authorization",
        }

        filtered = {}
        for key, value in data.items():
            key_lower = key.lower()
            if any(sensitive in key_lower for sensitive in sensitive_keys):
                filtered[key] = "***REDACTED***"
            else:
                filtered[key] = value
        return filtered


def log_exception(
    exc: BaseException,
    logger: logging.Logger | None = None,
    context: dict[str, Any] | None = None,
    include_locals: bool = False,
) -> None:
    """Log an exception with full context.

    Captures rich error context including stack traces, exception chain,
    and optionally local variables for debugging.

    Args:
        exc: The exception to log.
        logger: Logger to use (default: root oscura logger).
        context: Additional context to include.
        include_locals: Whether to include local variables from stack frames.

    Example:
        >>> try:
        ...     result = complex_computation(data)
        ... except Exception as e:
        ...     log_exception(e, context={"data_size": len(data)})

    References:
        LOG-008: Rich Error Context with Stack Traces
        CORE-006: Helpful exception messages
    """
    if logger is None:
        logger = get_logger("oscura")

    # Capture error context
    error_context = ErrorContextCapture.from_exception(
        exc, include_locals=include_locals, additional_context=context
    )

    # Convert to dict and log
    context_dict = error_context.to_dict(include_locals=include_locals)

    # Log with exception info
    logger.exception("Exception occurred", extra=context_dict)


def format_timestamp(
    dt: datetime | None = None,
    format: Literal["iso8601", "iso8601_local", "unix"] = "iso8601",
) -> str:
    """Format a timestamp according to LOG-005 requirements.

    Args:
        dt: Datetime to format, or None for current time.
        format: Format to use (iso8601, iso8601_local, unix).

    Returns:
        Formatted timestamp string.

    Raises:
        ValueError: If format is unknown.

    Example:
        >>> ts = format_timestamp()
        >>> print(ts)  # 2025-12-20T15:30:45.123456Z

    References:
        LOG-005: ISO 8601 Timestamps
    """
    if dt is None:
        dt = datetime.now(UTC)

    if format == "iso8601":
        return dt.strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"
    elif format == "iso8601_local":
        dt_local = dt.astimezone()
        return dt_local.strftime("%Y-%m-%dT%H:%M:%S.%f")
    elif format == "unix":
        return str(dt.timestamp())
    else:
        raise ValueError(f"Unknown timestamp format: {format}")


# Initialize logging on module import (with defaults)
def _init_logging() -> None:
    """Initialize logging with environment variable configuration.

    Reads:
        OSCURA_LOG_LEVEL: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        OSCURA_LOG_FORMAT: Log format (json, logfmt, text)

    References:
        LOG-001: Structured Logging Framework
        LOG-002: Hierarchical Log Levels
    """
    level = os.environ.get("OSCURA_LOG_LEVEL", "WARNING")
    log_format = os.environ.get("OSCURA_LOG_FORMAT", "text")

    if log_format in ("json", "logfmt", "text"):
        configure_logging(level=level, format=log_format)  # type: ignore[arg-type]
    else:
        configure_logging(level=level)


# Auto-initialize on import
_init_logging()

# Re-export correlation and performance functions for convenience
# These provide LOG-004 and LOG-006 functionality through this module
from oscura.core.correlation import (
    CorrelationContext,
    generate_correlation_id,
    get_correlation_id,
    set_correlation_id,
    with_correlation_id,
)
from oscura.core.performance import (
    PerformanceContext,
    PerformanceRecord,
    clear_performance_data,
    get_performance_records,
    get_performance_summary,
    timed,
)

__all__ = [
    "CompressingRotatingFileHandler",
    "CompressingTimedRotatingFileHandler",
    "CorrelationContext",
    # Error handling (LOG-008)
    "ErrorContextCapture",
    "LogConfig",
    "PerformanceContext",
    "PerformanceRecord",
    "StructuredFormatter",
    "clear_performance_data",
    # Logging configuration (LOG-001, LOG-002, LOG-003)
    "configure_logging",
    # Timestamps (LOG-005)
    "format_timestamp",
    "generate_correlation_id",
    # Correlation ID (LOG-004)
    "get_correlation_id",
    "get_logger",
    "get_performance_records",
    "get_performance_summary",
    "log_exception",
    "set_correlation_id",
    "set_log_level",
    # Performance timing (LOG-006)
    "timed",
    "with_correlation_id",
]
