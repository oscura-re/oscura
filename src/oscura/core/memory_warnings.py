"""Memory warning threshold system for Oscura.

This module provides configurable memory pressure warnings with
automatic alerts and optional operation cancellation.


Example:
    >>> from oscura.core.memory_warnings import check_memory_warnings, MemoryWarningLevel
    >>> level = check_memory_warnings()
    >>> if level == MemoryWarningLevel.CRITICAL:
    ...     print("Critical memory pressure!")

References:
    See oscura.config.memory for threshold configuration.
"""

from __future__ import annotations

import warnings
from enum import Enum
from typing import TYPE_CHECKING

from oscura.core.config.memory import get_memory_config
from oscura.utils.memory import get_available_memory, get_memory_pressure

if TYPE_CHECKING:
    from collections.abc import Callable


class MemoryWarningLevel(Enum):
    """Memory warning levels based on pressure thresholds.

    Attributes:
        OK: Memory usage is normal (below warn threshold).
        WARNING: Memory usage is elevated (above warn threshold).
        CRITICAL: Memory usage is critical (above critical threshold).
    """

    OK = "ok"
    WARNING = "warning"
    CRITICAL = "critical"


def check_memory_warnings() -> MemoryWarningLevel:
    """Check current memory pressure against configured thresholds.


    Returns:
        Current memory warning level.

    Example:
        >>> level = check_memory_warnings()
        >>> if level != MemoryWarningLevel.OK:
        ...     print(f"Memory pressure: {level.value}")
    """
    pressure = get_memory_pressure()
    config = get_memory_config()

    if pressure >= config.critical_threshold:
        return MemoryWarningLevel.CRITICAL
    elif pressure >= config.warn_threshold:
        return MemoryWarningLevel.WARNING
    else:
        return MemoryWarningLevel.OK


def emit_memory_warning(force: bool = False) -> None:
    """Emit warning if memory pressure exceeds thresholds.


    Args:
        force: If True, always emit warning regardless of level.

    Example:
        >>> emit_memory_warning()  # Emits warning if pressure high
    """
    level = check_memory_warnings()
    pressure = get_memory_pressure()
    available = get_available_memory()

    if force or level != MemoryWarningLevel.OK:
        if level == MemoryWarningLevel.CRITICAL:
            warnings.warn(
                f"CRITICAL memory pressure: {pressure * 100:.1f}% utilized. "
                f"Only {available / 1e9:.2f} GB available. "
                "Consider closing applications or canceling operations.",
                ResourceWarning,
                stacklevel=2,
            )
        elif level == MemoryWarningLevel.WARNING:
            warnings.warn(
                f"High memory pressure: {pressure * 100:.1f}% utilized. "
                f"{available / 1e9:.2f} GB available. "
                "Monitor memory usage closely.",
                ResourceWarning,
                stacklevel=2,
            )


def check_and_abort_if_critical(operation: str = "operation") -> None:
    """Check memory and abort if critical threshold exceeded.


    Args:
        operation: Name of operation to abort.

    Raises:
        MemoryError: If memory pressure is critical.

    Example:
        >>> try:
        ...     check_and_abort_if_critical('spectrogram')
        ...     # ... perform operation ...
        ... except MemoryError:
        ...     print("Operation aborted due to critical memory")
    """
    level = check_memory_warnings()

    if level == MemoryWarningLevel.CRITICAL:
        pressure = get_memory_pressure()
        available = get_available_memory()
        config = get_memory_config()

        raise MemoryError(
            f"Critical memory pressure ({pressure * 100:.1f}% > {config.critical_threshold * 100:.0f}%). "
            f"Only {available / 1e9:.2f} GB available. "
            f"Operation '{operation}' aborted to prevent system crash. "
            "Free memory or increase critical threshold to proceed."
        )


class MemoryWarningMonitor:
    """Context manager for monitoring memory warnings during operations.


    Monitors memory pressure during an operation and emits warnings
    when thresholds are crossed.

    Example:
        >>> with MemoryWarningMonitor('spectrogram', check_interval=100):
        ...     for i in range(1000):
        ...         # ... perform work ...
        ...         pass
    """

    def __init__(
        self,
        operation: str,
        *,
        check_interval: int = 100,
        abort_on_critical: bool = True,
    ):
        """Initialize memory warning monitor.

        Args:
            operation: Name of operation being monitored.
            check_interval: Check memory every N iterations.
            abort_on_critical: Abort if critical threshold reached.
        """
        self.operation = operation
        self.check_interval = check_interval
        self.abort_on_critical = abort_on_critical
        self._iteration = 0
        self._warned = False
        self._critical_warned = False

    def __enter__(self) -> MemoryWarningMonitor:
        """Enter context and perform initial check."""
        # Initial check
        emit_memory_warning()
        return self

    def __exit__(self, exc_type: type, exc_val: Exception, exc_tb: object) -> None:
        """Exit context."""
        # Note: exc_val and exc_tb intentionally unused but required for Python 3.11+ compatibility

    def check(self, iteration: int | None = None) -> None:
        """Check memory pressure and emit warnings.

        Args:
            iteration: Current iteration number (for periodic checking).

        Raises:
            MemoryError: If critical threshold exceeded and abort_on_critical=True.
        """
        self._iteration += 1

        # Only check periodically
        if iteration is not None and iteration % self.check_interval != 0:
            return

        level = check_memory_warnings()
        pressure = get_memory_pressure()
        available = get_available_memory()

        if level == MemoryWarningLevel.CRITICAL:
            if not self._critical_warned:
                warnings.warn(
                    f"Critical memory pressure during {self.operation}: "
                    f"{pressure * 100:.1f}% utilized, {available / 1e9:.2f} GB available.",
                    ResourceWarning,
                    stacklevel=2,
                )
                self._critical_warned = True

            if self.abort_on_critical:
                raise MemoryError(
                    f"Critical memory pressure during {self.operation}. "
                    f"Operation aborted to prevent system crash. "
                    f"Pressure: {pressure * 100:.1f}%, Available: {available / 1e9:.2f} GB"
                )

        elif level == MemoryWarningLevel.WARNING:
            if not self._warned:
                warnings.warn(
                    f"High memory pressure during {self.operation}: "
                    f"{pressure * 100:.1f}% utilized, {available / 1e9:.2f} GB available.",
                    ResourceWarning,
                    stacklevel=2,
                )
                self._warned = True


def register_memory_warning_callback(callback: Callable[[MemoryWarningLevel], None]) -> None:
    """Register a callback for memory warnings.

    The callback will be invoked whenever memory warnings are checked
    via emit_memory_warning() or MemoryWarningMonitor.

    Args:
        callback: Function that accepts MemoryWarningLevel.

    Example:
        >>> def my_callback(level):
        ...     if level == MemoryWarningLevel.CRITICAL:
        ...         print("CRITICAL MEMORY!")
        >>> register_memory_warning_callback(my_callback)
    """
    _warning_callbacks.append(callback)


def clear_memory_warning_callbacks() -> None:
    """Clear all registered memory warning callbacks.

    Example:
        >>> clear_memory_warning_callbacks()
    """
    _warning_callbacks.clear()


# Global list of warning callbacks
_warning_callbacks: list[Callable[[MemoryWarningLevel], None]] = []


def _invoke_callbacks(level: MemoryWarningLevel) -> None:
    """Invoke all registered callbacks with current warning level.

    Args:
        level: Current memory warning level.
    """
    for callback in _warning_callbacks:
        try:
            callback(level)
        except Exception as e:
            warnings.warn(
                f"Memory warning callback failed: {e}",
                RuntimeWarning,
                stacklevel=2,
            )


def format_memory_warning(level: MemoryWarningLevel) -> str:
    """Format memory warning message.

    Args:
        level: Memory warning level.

    Returns:
        Formatted warning message.

    Example:
        >>> msg = format_memory_warning(MemoryWarningLevel.WARNING)
        >>> print(msg)
        High memory pressure: 75.0% utilized...
    """
    pressure = get_memory_pressure()
    available = get_available_memory()
    config = get_memory_config()

    if level == MemoryWarningLevel.CRITICAL:
        return (
            f"CRITICAL memory pressure: {pressure * 100:.1f}% utilized "
            f"(threshold: {config.critical_threshold * 100:.0f}%). "
            f"Only {available / 1e9:.2f} GB available."
        )
    elif level == MemoryWarningLevel.WARNING:
        return (
            f"High memory pressure: {pressure * 100:.1f}% utilized "
            f"(threshold: {config.warn_threshold * 100:.0f}%). "
            f"{available / 1e9:.2f} GB available."
        )
    else:
        return f"Memory OK: {pressure * 100:.1f}% utilized, {available / 1e9:.2f} GB available."


__all__ = [
    "MemoryWarningLevel",
    "MemoryWarningMonitor",
    "check_and_abort_if_critical",
    "check_memory_warnings",
    "clear_memory_warning_callbacks",
    "emit_memory_warning",
    "format_memory_warning",
    "register_memory_warning_callback",
]
