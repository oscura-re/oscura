"""Memory-safe guards for Oscura analysis.

This module provides memory guards and resource limiting utilities to prevent
out-of-memory conditions during analysis operations.


Example:
    >>> from oscura.core.memory_guard import MemoryGuard, check_memory_available
    >>> if check_memory_available(500):  # Need at least 500MB
    ...     with MemoryGuard(max_mb=1000, name="fft") as guard:
    ...         result = compute_fft(data)
    ...         if not guard.check():
    ...             raise MemoryError("Exceeded memory limit")

References:
    See oscura.core.memory_monitor for runtime monitoring.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

logger = logging.getLogger(__name__)


def get_memory_usage_mb() -> float:
    """Get current process memory usage in MB.

    Returns:
        Current resident set size (RSS) in megabytes.

    Example:
        >>> mem_mb = get_memory_usage_mb()
        >>> print(f"Current process using {mem_mb:.1f} MB")
    """
    try:
        import psutil

        process = psutil.Process(os.getpid())
        return float(process.memory_info().rss / (1024 * 1024))
    except ImportError:
        # Fallback for systems without psutil
        logger.debug("psutil not available, memory usage tracking disabled")
        return 0.0


def check_memory_available(required_mb: float = 100) -> bool:
    """Check if sufficient memory is available.

    Args:
        required_mb: Required available memory in megabytes.

    Returns:
        True if sufficient memory is available.

    Example:
        >>> if not check_memory_available(1000):
        ...     print("Warning: Less than 1GB available")
        ...     # Reduce batch size or chunk operations
    """
    try:
        import psutil

        available = float(psutil.virtual_memory().available / (1024 * 1024))
        return bool(available > required_mb)
    except ImportError:
        # Assume OK if we can't check
        return True


class MemoryGuard:
    """Context manager for memory-safe operations.


    Monitors memory usage within a context and raises warnings/errors
    if limits are exceeded.

    Attributes:
        max_mb: Maximum memory limit in megabytes.
        name: Operation name for logging.
        start_mem: Starting memory usage (MB).

    Example:
        >>> with MemoryGuard(max_mb=2000, name="spectrogram") as guard:
        ...     # Perform memory-intensive operation
        ...     for chunk in data_chunks:
        ...         process_chunk(chunk)
        ...         if not guard.check():
        ...             break  # Stop before exceeding limit
        >>> stats = guard.get_stats()
        >>> print(f"Peak: {stats['peak_mb']:.1f} MB, Delta: {stats['delta_mb']:.1f} MB")
    """

    def __init__(self, max_mb: float = 1000, name: str = "operation"):
        """Initialize memory guard.

        Args:
            max_mb: Maximum memory increase allowed in megabytes.
            name: Operation name for logging and error messages.
        """
        self.max_mb = max_mb
        self.name = name
        self.start_mem = 0.0
        self._peak_mem = 0.0

    def __enter__(self) -> MemoryGuard:
        """Enter context and record starting memory."""
        self.start_mem = get_memory_usage_mb()
        self._peak_mem = self.start_mem
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Exit context and report memory usage."""
        # Note: exc_val and exc_tb intentionally unused but required for Python 3.11+ compatibility
        end_mem = get_memory_usage_mb()
        delta = end_mem - self.start_mem

        if delta > self.max_mb:
            logger.warning(
                f"{self.name} used {delta:.1f} MB (limit: {self.max_mb:.1f} MB). "
                f"Consider reducing batch size or enabling chunked processing."
            )

        # Update peak
        self._peak_mem = max(self._peak_mem, end_mem)

    def check(self) -> bool:
        """Check if within memory limit.

        Returns:
            True if within limit, False if limit exceeded.

        Example:
            >>> with MemoryGuard(max_mb=500) as guard:
            ...     for i in range(1000):
            ...         # Do work
            ...         if i % 100 == 0 and not guard.check():
            ...             raise MemoryError("Memory limit exceeded")
        """
        current = get_memory_usage_mb()
        self._peak_mem = max(self._peak_mem, current)
        delta = current - self.start_mem

        if delta > self.max_mb:
            logger.warning(
                f"{self.name}: Memory usage {delta:.1f} MB exceeds limit {self.max_mb:.1f} MB"
            )
            return False

        return True

    def get_stats(self) -> dict[str, float]:
        """Get memory statistics for this guard.

        Returns:
            Dictionary with keys:
                - start_mb: Starting memory
                - current_mb: Current memory
                - peak_mb: Peak memory
                - delta_mb: Memory increase since start
                - limit_mb: Configured limit

        Example:
            >>> with MemoryGuard(max_mb=1000, name="test") as guard:
            ...     # ... work ...
            ...     pass
            >>> stats = guard.get_stats()
            >>> print(f"Used {stats['delta_mb']:.1f} / {stats['limit_mb']:.1f} MB")
        """
        current = get_memory_usage_mb()
        return {
            "start_mb": self.start_mem,
            "current_mb": current,
            "peak_mb": self._peak_mem,
            "delta_mb": current - self.start_mem,
            "limit_mb": self.max_mb,
        }


def safe_array_size(shape: tuple[int, ...], dtype_bytes: int = 8) -> int:
    """Calculate array size in bytes, checking for overflow.


    Args:
        shape: Array shape tuple.
        dtype_bytes: Bytes per element (default: 8 for float64).

    Returns:
        Total array size in bytes.

    Raises:
        OverflowError: If array size would overflow.

    Example:
        >>> size = safe_array_size((1000, 1000, 8), dtype_bytes=8)
        >>> print(f"Array would use {size / 1e6:.1f} MB")
        >>> # Check if safe to allocate
        >>> if can_allocate(size):
        ...     arr = np.zeros((1000, 1000, 8))
    """
    try:
        import numpy as np

        total_elements = np.prod(shape)

        # Check for overflow in element count
        if total_elements > sys.maxsize // dtype_bytes:
            raise OverflowError(f"Array size too large: {shape}")

        size = int(total_elements) * dtype_bytes
        return size

    except (OverflowError, ValueError) as e:
        raise OverflowError(f"Array dimensions {shape} would cause overflow") from e


def can_allocate(size_bytes: int) -> bool:
    """Check if allocation is safe given available memory.


    Args:
        size_bytes: Requested allocation size in bytes.

    Returns:
        True if allocation is safe (with 2x safety margin).

    Example:
        >>> import numpy as np
        >>> shape = (10000, 10000)
        >>> size = safe_array_size(shape, dtype_bytes=8)
        >>> if can_allocate(size):
        ...     arr = np.zeros(shape)
        ... else:
        ...     print("Not enough memory, use chunked processing")
    """
    size_mb = size_bytes / (1024 * 1024)

    # Check with 2x safety margin
    return check_memory_available(size_mb * 2)


def get_safe_chunk_size(
    total_samples: int,
    dtype_bytes: int = 8,
    max_chunk_mb: float = 100,
) -> int:
    """Calculate safe chunk size for processing large datasets.


    Args:
        total_samples: Total number of samples to process.
        dtype_bytes: Bytes per sample (default: 8 for float64).
        max_chunk_mb: Maximum chunk size in megabytes.

    Returns:
        Chunk size in samples that fits within memory limit.

    Example:
        >>> total = 1_000_000_000  # 1 billion samples
        >>> chunk_size = get_safe_chunk_size(total, max_chunk_mb=100)
        >>> print(f"Process in chunks of {chunk_size:,} samples")
        >>> for i in range(0, total, chunk_size):
        ...     chunk = data[i:i+chunk_size]
        ...     process(chunk)
    """
    max_bytes = max_chunk_mb * 1024 * 1024
    max_samples = max_bytes // dtype_bytes

    # Ensure at least 1000 samples per chunk, but not more than total
    chunk_size = max(1000, min(max_samples, total_samples))

    return int(chunk_size)


__all__ = [
    "MemoryGuard",
    "can_allocate",
    "check_memory_available",
    "get_memory_usage_mb",
    "get_safe_chunk_size",
    "safe_array_size",
]
