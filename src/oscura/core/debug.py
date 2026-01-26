"""Debug mode and verbosity control infrastructure.

This module provides programmatic debug mode control with multiple
verbosity levels for troubleshooting and diagnostics.


Example:
    >>> from oscura.core.debug import enable_debug, is_debug_enabled, DebugContext
    >>> enable_debug(level='verbose')
    >>> if is_debug_enabled():
    ...     logger.debug("Extra diagnostic information")
    >>> with DebugContext(level='trace'):
    ...     # Temporarily increase verbosity
    ...     analyze_complex_signal()

References:
    - Debugging and diagnostic best practices
    - Python logging levels integration
"""

from __future__ import annotations

import contextvars
import logging
import os
from enum import IntEnum
from typing import Any, Literal

# Context variable for debug state
_debug_level: contextvars.ContextVar[int] = contextvars.ContextVar("debug_level", default=0)


class DebugLevel(IntEnum):
    """Debug verbosity levels.

    Levels:
        DISABLED: No debug output (0)
        MINIMAL: Basic debug info (1)
        NORMAL: Standard debug info (2)
        VERBOSE: Detailed debug info (3)
        TRACE: Very detailed trace info (4)

    References:
        LOG-007: Programmatic Debug Mode
    """

    DISABLED = 0
    MINIMAL = 1
    NORMAL = 2
    VERBOSE = 3
    TRACE = 4


# String to enum mapping
_LEVEL_MAP: dict[str, DebugLevel] = {
    "disabled": DebugLevel.DISABLED,
    "minimal": DebugLevel.MINIMAL,
    "normal": DebugLevel.NORMAL,
    "verbose": DebugLevel.VERBOSE,
    "trace": DebugLevel.TRACE,
}


def enable_debug(
    level: Literal["minimal", "normal", "verbose", "trace"] = "normal",
) -> None:
    """Enable debug mode with specified verbosity level.

    Sets the global debug level and adjusts logging configuration
    accordingly. Higher levels include all output from lower levels.

    Args:
        level: Debug verbosity level.

    Example:
        >>> enable_debug(level='verbose')
        >>> # Now all debug logging at verbose level is active

        >>> enable_debug(level='trace')
        >>> # Extremely detailed trace logging

    References:
        LOG-007: Programmatic Debug Mode
    """
    debug_level = _LEVEL_MAP[level]
    _debug_level.set(debug_level)

    # Adjust logging level based on debug level
    from oscura.core.logging import set_log_level

    if debug_level == DebugLevel.TRACE:
        set_log_level("DEBUG")  # Most verbose
    elif debug_level >= DebugLevel.NORMAL:
        set_log_level("DEBUG")
    elif debug_level == DebugLevel.MINIMAL:
        set_log_level("INFO")
    else:
        set_log_level("WARNING")


def disable_debug() -> None:
    """Disable debug mode.

    Resets debug level to DISABLED and sets logging to WARNING.

    Example:
        >>> disable_debug()
        >>> # Debug output is now suppressed

    References:
        LOG-007: Programmatic Debug Mode
    """
    _debug_level.set(DebugLevel.DISABLED)

    from oscura.core.logging import set_log_level

    set_log_level("WARNING")


def is_debug_enabled(min_level: DebugLevel = DebugLevel.MINIMAL) -> bool:
    """Check if debug mode is enabled at or above the specified level.

    Args:
        min_level: Minimum debug level to check for.

    Returns:
        True if debug is enabled at or above min_level.

    Example:
        >>> enable_debug(level='verbose')
        >>> is_debug_enabled()
        True
        >>> is_debug_enabled(DebugLevel.TRACE)
        False

    References:
        LOG-007: Programmatic Debug Mode
    """
    current = _debug_level.get()
    return current >= min_level


def get_debug_level() -> DebugLevel:
    """Get the current debug level.

    Returns:
        Current debug level enum value.

    Example:
        >>> level = get_debug_level()
        >>> if level >= DebugLevel.VERBOSE:
        ...     print("Verbose debugging enabled")

    References:
        LOG-007: Programmatic Debug Mode
    """
    return DebugLevel(_debug_level.get())


class DebugContext:
    """Context manager for temporary debug level changes.

    Temporarily sets a debug level for the duration of a code block,
    then restores the previous level.

    Args:
        level: Debug level to set within the context.

    Example:
        >>> # Normal debug level
        >>> with DebugContext(level='trace'):
        ...     # Temporarily enable trace-level debugging
        ...     analyze_signal(complex_data)
        >>> # Back to normal debug level

    References:
        LOG-007: Programmatic Debug Mode
    """

    def __init__(
        self,
        level: Literal["disabled", "minimal", "normal", "verbose", "trace"],
    ):
        """Initialize debug context.

        Args:
            level: Debug level to set within the context.
        """
        self.level = _LEVEL_MAP[level]
        self.token: contextvars.Token | None = None  # type: ignore[type-arg]
        self.previous_log_level: str | None = None

    def __enter__(self) -> DebugContext:
        """Enter the debug context and set new level."""
        # Save current debug level
        self.token = _debug_level.set(self.level)

        # Adjust logging
        from oscura.core.logging import get_logger

        root_logger = get_logger("oscura")
        self.previous_log_level = logging.getLevelName(root_logger.level)

        # Set appropriate log level for debug level
        from oscura.core.logging import set_log_level

        if self.level in (DebugLevel.TRACE, DebugLevel.VERBOSE, DebugLevel.NORMAL):
            set_log_level("DEBUG")
        elif self.level == DebugLevel.MINIMAL:
            set_log_level("INFO")
        else:
            set_log_level("WARNING")

        return self

    def __exit__(self, *args: Any) -> None:
        """Exit the debug context and restore previous level."""
        # Restore debug level
        if self.token:
            _debug_level.reset(self.token)

        # Restore logging level
        if self.previous_log_level:
            from oscura.core.logging import set_log_level

            set_log_level(self.previous_log_level)


# Backward compatibility alias (deprecated, use DebugContext)
debug_context = DebugContext


def should_log_debug(min_level: DebugLevel = DebugLevel.NORMAL) -> bool:
    """Check if debug logging should occur at specified level.

    Helper function for conditional debug logging.

    Args:
        min_level: Minimum level required for logging.

    Returns:
        True if current debug level meets or exceeds min_level.

    Example:
        >>> if should_log_debug(DebugLevel.VERBOSE):
        ...     logger.debug("Detailed diagnostic: %s", expensive_computation())

    References:
        LOG-007: Programmatic Debug Mode
    """
    return is_debug_enabled(min_level)


def configure_debug_from_env() -> None:
    """Configure debug mode from environment variables.

    Reads OSCURA_DEBUG environment variable and sets debug level
    accordingly.

    Environment Variables:
        OSCURA_DEBUG: Debug level (minimal, normal, verbose, trace)

    Example:
        >>> import os
        >>> os.environ['OSCURA_DEBUG'] = 'verbose'
        >>> configure_debug_from_env()

    References:
        LOG-007: Programmatic Debug Mode
    """
    debug_env = os.environ.get("OSCURA_DEBUG", "").lower()
    if debug_env in _LEVEL_MAP:
        enable_debug(level=debug_env)  # type: ignore[arg-type]


def debug_log(
    logger: logging.Logger,
    message: str,
    min_level: DebugLevel = DebugLevel.NORMAL,
    **kwargs: Any,
) -> None:
    """Conditionally log debug message based on debug level.

    Only logs if current debug level meets or exceeds min_level.

    Args:
        logger: Logger to use for output.
        message: Message to log.
        min_level: Minimum debug level required.
        **kwargs: Additional keyword arguments for logger.

    Example:
        >>> from oscura.core.logging import get_logger
        >>> logger = get_logger(__name__)
        >>> debug_log(logger, "Processing FFT", DebugLevel.VERBOSE, samples=1000)

    References:
        LOG-007: Programmatic Debug Mode
    """
    if is_debug_enabled(min_level):
        logger.debug(message, **kwargs)


# Auto-configure from environment on import
configure_debug_from_env()
