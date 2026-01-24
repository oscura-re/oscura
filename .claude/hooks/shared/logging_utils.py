#!/usr/bin/env python3
"""
Shared logging utilities for Claude hooks.

Provides standardized logging across all hooks with consistent formatting,
including optional JSON structured logging.

Version: 2.0.0
Created: 2026-01-19
Updated: 2026-01-22
"""

import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Standard log format for all hooks
LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
DATE_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


class JSONFormatter(logging.Formatter):
    """Format log records as JSON lines.

    Outputs structured JSON for machine parsing and log aggregation.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON.

        Args:
            record: Log record to format

        Returns:
            JSON string with timestamp, level, logger, message, and extra fields
        """
        log_entry: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add extra fields if present
        if hasattr(record, "extra_fields"):
            log_entry.update(record.extra_fields)

        return json.dumps(log_entry)


def get_hook_logger(
    name: str,
    log_file: Path | None = None,
    level: int = logging.INFO,
    console: bool = False,
    use_json: bool = False,
) -> logging.Logger:
    """Get standardized logger for hook.

    Args:
        name: Logger name (typically __name__ or hook filename)
        log_file: Path to log file (defaults to .claude/hooks/hooks.log)
        level: Logging level (default: INFO)
        console: Also log to console/stderr (default: False)
        use_json: If True, log as JSON lines; if False, plain text (default: False)

    Returns:
        Configured logger instance

    Example:
        # Plain text logging
        logger = get_hook_logger(__name__)
        logger.info("Hook started")

        # JSON structured logging
        logger = get_hook_logger(__name__, use_json=True)
        logger.info("Hook started", extra={"extra_fields": {"hook": "validate_path"}})
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # Create formatter
    if use_json:
        formatter: logging.Formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    # File handler
    if log_file is None:
        # Default log file
        project_dir = Path.cwd()
        log_file = project_dir / ".claude" / "hooks" / "hooks.log"

    log_file.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler (optional)
    if console:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


def log_hook_start(
    logger: logging.Logger, hook_name: str, context: dict[str, Any] | None = None
) -> None:
    """Log hook start with context.

    Args:
        logger: Logger instance
        hook_name: Name of hook
        context: Optional context dict (tool_name, file_path, etc.)
    """
    if context:
        logger.info(f"{hook_name} started - context: {context}")
    else:
        logger.info(f"{hook_name} started")


def log_hook_end(
    logger: logging.Logger, hook_name: str, success: bool, duration_ms: float | None = None
) -> None:
    """Log hook completion.

    Args:
        logger: Logger instance
        hook_name: Name of hook
        success: Whether hook succeeded
        duration_ms: Optional execution time in milliseconds
    """
    status = "SUCCESS" if success else "FAILED"
    if duration_ms is not None:
        logger.info(f"{hook_name} {status} - duration: {duration_ms:.2f}ms")
    else:
        logger.info(f"{hook_name} {status}")


def log_structured_event(
    logger: logging.Logger, event_type: str, message: str, **extra_fields: Any
) -> None:
    """Log structured event with extra fields.

    Args:
        logger: Logger instance
        event_type: Type of event (e.g., "hook_enforcement", "agent_launch")
        message: Human-readable message
        **extra_fields: Additional fields to include in JSON

    Example:
        logger = get_hook_logger(__name__, use_json=True)
        log_structured_event(
            logger,
            "route_decision",
            "Routed to code_reviewer",
            agent="code_reviewer",
            complexity=45,
            keywords=["review", "audit"]
        )
    """
    extra_fields["event_type"] = event_type
    logger.info(message, extra={"extra_fields": extra_fields})


def log_to_file(file_path: Path, level: str, message: str) -> None:
    """Simple file logging without logger instance.

    Args:
        file_path: Log file path
        level: Log level (INFO, WARNING, ERROR)
        message: Message to log

    Example:
        log_to_file(Path(".claude/hooks/errors.log"), "ERROR", "Hook failed")
    """
    timestamp = datetime.now(UTC).isoformat()
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "a") as f:
        f.write(f"[{timestamp}] [{level}] {message}\n")


class HookLogger:
    """Context manager for hook logging with automatic start/end logging.

    Example:
        with HookLogger("validate_path") as logger:
            logger.info("Validating path")
            # ... do work ...
            # Automatically logs success/failure on exit
    """

    def __init__(
        self,
        hook_name: str,
        log_file: Path | None = None,
        fail_on_exception: bool = True,
    ):
        """Initialize hook logger.

        Args:
            hook_name: Name of hook
            log_file: Log file path
            fail_on_exception: Whether to log as failure on exception
        """
        self.hook_name = hook_name
        self.logger = get_hook_logger(hook_name, log_file)
        self.fail_on_exception = fail_on_exception
        self.start_time = 0.0
        self.success = True

    def __enter__(self) -> logging.Logger:
        """Enter context - log start."""
        import time

        self.start_time = time.time()
        self.logger.info(f"{self.hook_name} started")
        return self.logger

    def __exit__(self, exc_type: type | None, exc_val: Exception | None, exc_tb: Any) -> bool:
        """Exit context - log end."""
        import time

        duration_ms = (time.time() - self.start_time) * 1000

        if exc_type is not None:
            if self.fail_on_exception:
                self.logger.error(f"{self.hook_name} FAILED - {exc_val}", exc_info=True)
                self.logger.info(f"{self.hook_name} duration: {duration_ms:.2f}ms")
                return False  # Re-raise exception
            else:
                self.logger.warning(f"{self.hook_name} exception (ignored) - {exc_val}")

        status = "SUCCESS" if self.success else "FAILED"
        self.logger.info(f"{self.hook_name} {status} - duration: {duration_ms:.2f}ms")
        return True  # Suppress exception if fail_on_exception=False

    def set_failed(self) -> None:
        """Mark this hook execution as failed."""
        self.success = False
