#!/usr/bin/env python3
"""
Shared logging utilities for Claude hooks.

Provides standardized logging across all hooks with consistent formatting.

Version: 1.0.0
Created: 2026-01-19
"""

import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Standard log format for all hooks
LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
DATE_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def get_hook_logger(
    name: str,
    log_file: Path | None = None,
    level: int = logging.INFO,
    console: bool = False,
) -> logging.Logger:
    """Get standardized logger for hook.

    Args:
        name: Logger name (typically __name__ or hook filename)
        log_file: Path to log file (defaults to .claude/hooks/hooks.log)
        level: Logging level (default: INFO)
        console: Also log to console/stderr (default: False)

    Returns:
        Configured logger instance

    Example:
        logger = get_hook_logger(__name__)
        logger.info("Hook started")
        logger.error(f"Failed: {error}")
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # Create formatter
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
