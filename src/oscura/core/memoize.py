"""Memory-safe memoization decorators for Oscura analyzer functions.

This module provides lightweight memoization decorators optimized for analyzer
functions that process numpy arrays. Unlike the full OscuraCache, these
decorators provide simple in-memory caching with bounded size.


Example:
    >>> from oscura.core.memoize import memoize_analysis
    >>> @memoize_analysis(maxsize=32)
    ... def expensive_fft(signal, nperseg):
    ...     return scipy.fft.fft(signal, n=nperseg)
    >>> result = expensive_fft(signal_array, 1024)  # Computed
    >>> result = expensive_fft(signal_array, 1024)  # Cached

References:
    functools.lru_cache for standard Python memoization
    hashlib for stable array hashing
"""

from __future__ import annotations

import hashlib
from functools import wraps
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from collections.abc import Callable

T = TypeVar("T")


def array_hash(arr: NDArray[Any], sample_size: int = 10000) -> str:
    """Create stable hash for numpy array.

    Uses first `sample_size` bytes of array data to create a hash key.
    This is faster than hashing the entire array while maintaining
    good cache hit rates for typical analysis workflows.

    Args:
        arr: Numpy array to hash.
        sample_size: Number of bytes to sample for hashing (default: 10KB).

    Returns:
        16-character hex hash string.

    Example:
        >>> arr = np.arange(1000000, dtype=np.float32)
        >>> hash1 = array_hash(arr)
        >>> hash2 = array_hash(arr)
        >>> assert hash1 == hash2
    """
    # Use shape, dtype, and sample of data for hash
    hash_obj = hashlib.sha256()

    # Include shape and dtype
    hash_obj.update(str(arr.shape).encode())
    hash_obj.update(str(arr.dtype).encode())

    # Sample first N bytes of data
    data_bytes = arr.tobytes()[:sample_size]
    hash_obj.update(data_bytes)

    return hash_obj.hexdigest()[:16]


def memoize_analysis(maxsize: int = 32) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for memoizing analysis functions with numpy arrays.

    Automatically hashes numpy array arguments for cache keys.
    Memory-safe with bounded cache size using LRU eviction.


    Args:
        maxsize: Maximum number of cached results (default: 32).

    Returns:
        Decorator function.

    Example:
        >>> @memoize_analysis(maxsize=16)
        ... def detect_edges(signal, threshold):
        ...     # Expensive edge detection...
        ...     return edges
        >>> # First call computes
        >>> edges1 = detect_edges(signal_array, 0.5)
        >>> # Second call uses cache
        >>> edges2 = detect_edges(signal_array, 0.5)
        >>> assert edges1 is edges2

    Note:
        Cache is stored per-function. Use OscuraCache from core.cache
        for persistent cross-function caching.

    References:
        PERF-001: Performance optimization requirements
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        cache: dict[str, T] = {}
        cache_order: list[str] = []  # Track insertion order for LRU

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            # Build cache key from args
            key_parts: list[str] = []

            for arg in args:
                if isinstance(arg, np.ndarray):
                    key_parts.append(f"arr_{len(arg)}_{array_hash(arg)}")
                else:
                    key_parts.append(str(arg))

            for k, v in sorted(kwargs.items()):
                if isinstance(v, np.ndarray):
                    key_parts.append(f"{k}=arr_{len(v)}_{array_hash(v)}")
                else:
                    key_parts.append(f"{k}={v}")

            cache_key = ":".join(key_parts)

            # Check cache
            if cache_key in cache:
                # Move to end (most recently used)
                cache_order.remove(cache_key)
                cache_order.append(cache_key)
                return cache[cache_key]

            # Compute result
            result = func(*args, **kwargs)

            # Evict oldest if at capacity
            if len(cache) >= maxsize:
                oldest = cache_order.pop(0)
                del cache[oldest]

            # Store result
            cache[cache_key] = result
            cache_order.append(cache_key)

            return result

        def cache_clear() -> None:
            """Clear all cached results."""
            cache.clear()
            cache_order.clear()

        def cache_info() -> dict[str, Any]:
            """Get cache statistics.

            Returns:
                Dictionary with cache size and maxsize.
            """
            return {"size": len(cache), "maxsize": maxsize}

        # Attach utility methods
        wrapper.cache_clear = cache_clear  # type: ignore[attr-defined]
        wrapper.cache_info = cache_info  # type: ignore[attr-defined]

        return wrapper

    return decorator


__all__ = [
    "array_hash",
    "memoize_analysis",
]
