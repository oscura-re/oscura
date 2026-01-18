"""Pre-flight memory checking for Oscura operations.

This module provides automatic memory verification before executing
memory-intensive operations to prevent OOM crashes.


Example:
    >>> from oscura.core.memory_check import check_operation_memory, require_memory
    >>> check = check_operation_memory('spectrogram', samples=1e9, nperseg=4096)
    >>> if not check.sufficient:
    ...     print(check.recommendation)

References:
    See oscura.utils.memory for memory estimation functions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from oscura.utils.memory import (
    MemoryCheck,
    MemoryCheckError,
    check_memory_available,
    require_memory,
)

if TYPE_CHECKING:
    from collections.abc import Callable


# Operations that automatically perform memory checks
_AUTO_CHECK_OPERATIONS = {
    "fft",
    "psd",
    "spectrogram",
    "eye_diagram",
    "correlate",
    "filter",
    "stft",
    "cwt",
    "dwt",
}

# Global flag to bypass memory checks (use with caution)
_force_memory = False


def set_force_memory(enabled: bool) -> None:
    """Enable or disable forced memory bypass.


    Args:
        enabled: If True, bypass memory checks (dangerous).

    Warning:
        Bypassing memory checks can lead to system crashes.
        Only use when you are certain the operation will succeed.

    Example:
        >>> set_force_memory(True)  # Bypass all memory checks
        >>> # ... perform operation ...
        >>> set_force_memory(False)  # Re-enable checks
    """
    global _force_memory
    _force_memory = enabled


def is_force_memory() -> bool:
    """Check if memory checks are bypassed.

    Returns:
        True if memory checks are disabled.
    """
    return _force_memory


def check_operation_memory(
    operation: str,
    samples: int | float | None = None,
    **kwargs: Any,
) -> MemoryCheck:
    """Check if sufficient memory is available for an operation.


    This is a convenience wrapper around utils.memory.check_memory_available
    with automatic bypass support.

    Args:
        operation: Operation name (fft, psd, spectrogram, etc.).
        samples: Number of samples to process.
        **kwargs: Additional operation-specific parameters.

    Returns:
        MemoryCheck with sufficiency status and recommendations.

    Example:
        >>> check = check_operation_memory('fft', samples=1e9, nfft=8192)
        >>> if not check.sufficient:
        ...     print(f"Insufficient memory: {check.recommendation}")
    """
    # Bypass check if forced
    if _force_memory:
        return MemoryCheck(
            sufficient=True,
            available=0,
            required=0,
            recommendation="Memory check bypassed (--force-memory enabled)",
        )

    return check_memory_available(operation, samples, **kwargs)


def auto_check_memory(
    operation: str,
    samples: int | float | None = None,
    **kwargs: Any,
) -> None:
    """Automatically check memory and raise error if insufficient.


    This function is called automatically by operations that support
    memory checking (fft, psd, spectrogram, etc.).

    Args:
        operation: Operation name.
        samples: Number of samples.
        **kwargs: Additional parameters.

    Example:
        >>> try:
        ...     auto_check_memory('spectrogram', samples=1e9, nperseg=4096)
        ... except MemoryCheckError as e:
        ...     print(f"Memory check failed: {e}")
        ...     print(f"Suggestion: {e.recommendation}")

    Note:
        May raise MemoryCheckError if insufficient memory and not forced.
    """
    # Skip check if operation doesn't require it
    if operation not in _AUTO_CHECK_OPERATIONS:
        return

    # Bypass if forced
    if _force_memory:
        return

    # Perform check
    require_memory(operation, samples, **kwargs)


def with_memory_check(func: Callable) -> Callable:  # type: ignore[type-arg]
    """Decorator to add automatic memory checking to a function.


    The decorated function must accept 'samples' as a keyword argument
    and should have an 'operation' attribute or name that matches
    a supported operation type.

    Args:
        func: Function to decorate.

    Returns:
        Decorated function with memory checking.

    Example:
        >>> @with_memory_check
        ... def my_fft(signal, samples=None, **kwargs):
        ...     # ... FFT implementation ...
        ...     pass
        >>> my_fft.operation = 'fft'  # Specify operation type
    """

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # Extract operation name
        operation = getattr(func, "operation", func.__name__)

        # Extract samples if available
        samples = kwargs.get("samples")
        if samples is None and len(args) > 0:
            # Try to infer from first argument
            try:
                import numpy as np

                if isinstance(args[0], np.ndarray):
                    samples = len(args[0])
            except (ImportError, TypeError):
                pass

        # Perform check
        if operation in _AUTO_CHECK_OPERATIONS and not _force_memory:
            auto_check_memory(operation, samples, **kwargs)

        # Call original function
        return func(*args, **kwargs)

    # Preserve function metadata
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    wrapper.__module__ = func.__module__

    return wrapper


def register_auto_check_operation(operation: str) -> None:
    """Register an operation for automatic memory checking.

    Args:
        operation: Operation name to register.

    Example:
        >>> register_auto_check_operation('custom_transform')
        >>> # Now custom_transform will automatically check memory
    """
    _AUTO_CHECK_OPERATIONS.add(operation)


def unregister_auto_check_operation(operation: str) -> None:
    """Unregister an operation from automatic memory checking.

    Args:
        operation: Operation name to unregister.

    Example:
        >>> unregister_auto_check_operation('custom_transform')
        >>> # Now custom_transform will not automatically check memory
    """
    _AUTO_CHECK_OPERATIONS.discard(operation)


def _reset_auto_check_operations() -> None:
    """Reset AUTO_CHECK_OPERATIONS to default state.

    WARNING: This is for testing only. It resets the global state to defaults.
    """
    global _AUTO_CHECK_OPERATIONS
    _AUTO_CHECK_OPERATIONS = {
        "fft",
        "psd",
        "spectrogram",
        "eye_diagram",
        "correlate",
        "filter",
        "stft",
        "cwt",
        "dwt",
    }


def get_auto_check_operations() -> set[str]:
    """Get set of operations that automatically check memory.

    Returns:
        Set of operation names.

    Example:
        >>> ops = get_auto_check_operations()
        >>> print(f"Auto-checked operations: {', '.join(sorted(ops))}")
    """
    return _AUTO_CHECK_OPERATIONS.copy()


__all__ = [
    "MemoryCheck",
    "MemoryCheckError",
    "auto_check_memory",
    "check_operation_memory",
    "get_auto_check_operations",
    "is_force_memory",
    "register_auto_check_operation",
    "set_force_memory",
    "unregister_auto_check_operation",
    "with_memory_check",
]
