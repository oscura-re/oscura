"""Extended memory management utilities for Oscura.

Additional memory management features including context managers,
LRU caching, and HDF5 lazy loading support.


Example:
    >>> from oscura.utils.memory_extensions import ResourceManager, LRUCache
    >>> with ResourceManager(large_array) as data:
    ...     result = process(data)
    >>> # Data automatically cleaned up

References:
    Python resource management patterns
    functools.lru_cache documentation
"""

from __future__ import annotations

import gc
import hashlib
import os
import time
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

T = TypeVar("T")


# =============================================================================
# Context Managers for Resource Cleanup (MEM-019)
# =============================================================================


class ResourceManager:
    """Context manager for large data resources with automatic cleanup.

    Ensures prompt memory release when done with large datasets.

    Args:
        resource: Resource to manage (array, file handle, etc.).
        cleanup_func: Optional cleanup function to call on exit.

    Example:
        >>> import numpy as np
        >>> with ResourceManager(np.zeros(1000000)) as data:
        ...     result = process(data)
        >>> # Data is automatically released

    References:
        MEM-019: Explicit Resource Cleanup
    """

    def __init__(
        self,
        resource: Any,
        cleanup_func: Callable[[Any], None] | None = None,
    ) -> None:
        self._resource = resource
        self._cleanup_func = cleanup_func

    def __enter__(self) -> Any:
        """Enter context, return resource."""
        return self._resource

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context, cleanup resource."""
        # Note: exc_val and exc_tb intentionally unused but required for Python 3.11+ compatibility
        if self._cleanup_func is not None:
            self._cleanup_func(self._resource)

        # Delete reference
        self._resource = None

        # Force garbage collection
        gc.collect()


class ArrayManager(ResourceManager):
    """Context manager specifically for numpy arrays.

    Example:
        >>> import numpy as np
        >>> with ArrayManager(np.zeros((10000, 10000))) as arr:
        ...     result = np.sum(arr)
        >>> # Array memory is released
    """

    def __init__(self, array: NDArray[Any]) -> None:
        super().__init__(array, cleanup_func=lambda x: None)


# =============================================================================
# LRU Cache for Intermediate Results (MEM-021, MEM-029)
# =============================================================================


class LRUCache(Generic[T]):
    """Least-Recently-Used cache with memory-based eviction.

    Caches intermediate results with automatic eviction when
    memory limit is exceeded.

    Args:
        max_memory_bytes: Maximum cache size in bytes.
        max_entries: Maximum number of cache entries (default unlimited).

    Example:
        >>> cache = LRUCache(max_memory_bytes=1_000_000_000)  # 1 GB
        >>> cache.put("key1", large_array, size_bytes=800_000_000)
        >>> result = cache.get("key1")
        >>> cache.clear()

    References:
        MEM-021: Intermediate Result Eviction
        MEM-029: LRU Cache for Intermediate Results
    """

    def __init__(
        self,
        max_memory_bytes: int,
        max_entries: int | None = None,
    ) -> None:
        self._max_memory = max_memory_bytes
        self._max_entries = max_entries
        self._cache: OrderedDict[str, tuple[T, int, float]] = OrderedDict()
        self._current_memory: int = 0
        self._hits: int = 0
        self._misses: int = 0

    def get(self, key: str) -> T | None:
        """Get cached value by key.

        Args:
            key: Cache key.

        Returns:
            Cached value or None if not found.
        """
        if key in self._cache:
            # Move to end (most recently used)
            value, size, _ = self._cache.pop(key)
            self._cache[key] = (value, size, time.time())
            self._hits += 1
            return value
        else:
            self._misses += 1
            return None

    def put(self, key: str, value: T, size_bytes: int | None = None) -> None:
        """Put value in cache.

        Args:
            key: Cache key.
            value: Value to cache.
            size_bytes: Size in bytes. If None, estimated from value.
        """
        # Estimate size if not provided
        if size_bytes is None:
            size_bytes = self._estimate_size(value)

        # Check if single item exceeds max memory
        if size_bytes > self._max_memory:
            # Don't cache items larger than max memory
            return

        # Evict until we have space
        while (
            self._current_memory + size_bytes > self._max_memory
            or (self._max_entries and len(self._cache) >= self._max_entries)
        ) and len(self._cache) > 0:
            self._evict_oldest()

        # Remove if key already exists
        if key in self._cache:
            _, old_size, _ = self._cache.pop(key)
            self._current_memory -= old_size

        # Add new entry
        self._cache[key] = (value, size_bytes, time.time())
        self._current_memory += size_bytes

    def _evict_oldest(self) -> None:
        """Evict least recently used entry."""
        if len(self._cache) > 0:
            _, (_, size, _) = self._cache.popitem(last=False)
            self._current_memory -= size

    def clear(self) -> None:
        """Clear entire cache."""
        self._cache.clear()
        self._current_memory = 0

    def _estimate_size(self, value: T) -> int:
        """Estimate size of cached value."""
        if isinstance(value, np.ndarray):
            return int(value.nbytes)
        elif isinstance(value, list | tuple):
            # Rough estimate for sequences
            return sum(self._estimate_size(item) for item in value)
        elif isinstance(value, dict):
            return sum(self._estimate_size(k) + self._estimate_size(v) for k, v in value.items())
        else:
            # Fallback: assume 1KB for unknown types
            return 1024

    def stats(self) -> dict[str, int | float]:
        """Get cache statistics.

        Returns:
            Dictionary with cache stats.
        """
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0.0

        return {
            "entries": len(self._cache),
            "memory_bytes": self._current_memory,
            "memory_mb": self._current_memory / (1024 * 1024),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
        }

    def __len__(self) -> int:
        """Number of cached entries."""
        return len(self._cache)


# Global result cache
_result_cache: LRUCache[Any] | None = None


def get_result_cache() -> LRUCache[Any]:
    """Get global result cache.

    Returns:
        Global LRU cache instance.

    Example:
        >>> cache = get_result_cache()
        >>> cache.put("fft_result", fft_data, size_bytes=8000000)
        >>> result = cache.get("fft_result")
    """
    global _result_cache
    if _result_cache is None:
        # Default: 2 GB cache
        max_cache_size = int(os.environ.get("TK_CACHE_SIZE", 2 * 1024 * 1024 * 1024))
        _result_cache = LRUCache(max_memory_bytes=max_cache_size)
    return _result_cache


def clear_cache() -> None:
    """Clear the global result cache.

    Example:
        >>> clear_cache()
        >>> # All cached results released
    """
    cache = get_result_cache()
    cache.clear()


def show_cache_stats() -> dict[str, int | float]:
    """Show statistics for the global cache.

    Returns:
        Dictionary with cache statistics.

    Example:
        >>> stats = show_cache_stats()
        >>> print(f"Hit rate: {stats['hit_rate']*100:.1f}%")
        >>> print(f"Memory used: {stats['memory_mb']:.1f} MB")
    """
    cache = get_result_cache()
    return cache.stats()


def cache_key(*args: Any, **kwargs: Any) -> str:
    """Generate cache key from arguments.

    Note:
        Uses MD5 for cache key generation only (not for security).
        MD5 is appropriate here for non-cryptographic checksums.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Hash-based cache key.

    Example:
        >>> key = cache_key("fft", samples=1000, nfft=2048)
        >>> # Use key for caching
    """
    # Create stable string representation
    parts = [str(arg) for arg in args]
    parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
    key_str = "|".join(parts)

    # Hash for consistent key (MD5 used for cache keys only, not security)
    return hashlib.md5(key_str.encode(), usedforsecurity=False).hexdigest()


# =============================================================================
# HDF5 Lazy Loading (MEM-017)
# =============================================================================


def load_hdf5_lazy(
    file_path: str,
    dataset_path: str = "/data",
) -> Any:
    """Load HDF5 dataset as lazy h5py.Dataset (not fully in memory).

    Enables partial loading via slicing without loading entire dataset.

    Args:
        file_path: Path to HDF5 file.
        dataset_path: Path to dataset within file (default "/data").

    Returns:
        h5py.Dataset object (lazy, not loaded until accessed).

    Raises:
        ImportError: If h5py is not available.
        FileNotFoundError: If file does not exist.
        KeyError: If dataset not found in file.

    Example:
        >>> # Load dataset lazily
        >>> dataset = load_hdf5_lazy("large_file.h5", "/signals/ch1")
        >>> # Only load specific range (not entire file)
        >>> chunk = dataset[1000:2000]
        >>> print(f"Chunk shape: {chunk.shape}")

    References:
        MEM-017: HDF5 Chunked Dataset Access
    """
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py required for lazy HDF5 loading. Install with: pip install h5py")

    from pathlib import Path

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Open file in read mode
    # Note: File handle should be kept open for lazy access
    # User is responsible for closing the file handle
    f = h5py.File(file_path, "r")

    if dataset_path not in f:
        available = list(f.keys())
        f.close()
        raise KeyError(
            f"Dataset '{dataset_path}' not found in HDF5 file. "
            f"Available datasets: {', '.join(available)}"
        )

    dataset = f[dataset_path]

    return dataset


class LazyHDF5Array:
    """Wrapper for lazy HDF5 dataset access with context management.

    Provides automatic file handle cleanup and numpy-like slicing.

    Args:
        file_path: Path to HDF5 file.
        dataset_path: Path to dataset within file.

    Example:
        >>> with LazyHDF5Array("data.h5", "/signals/ch1") as arr:
        ...     # Only loads specific slice
        ...     chunk = arr[1000:2000]
        ...     print(f"Shape: {arr.shape}, dtype: {arr.dtype}")
        >>> # File automatically closed

    References:
        MEM-017: HDF5 Chunked Dataset Access
        MEM-019: Explicit Resource Cleanup
    """

    def __init__(self, file_path: str, dataset_path: str = "/data"):
        self._file_path = file_path
        self._dataset_path = dataset_path
        self._file = None
        self._dataset = None

    def __enter__(self) -> LazyHDF5Array:
        """Open HDF5 file and dataset."""
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py required. Install with: pip install h5py")

        self._file = h5py.File(self._file_path, "r")
        self._dataset = self._file[self._dataset_path]  # type: ignore[index]
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Close HDF5 file."""
        # Note: exc_val and exc_tb intentionally unused but required for Python 3.11+ compatibility
        if self._file is not None:
            self._file.close()  # type: ignore[unreachable]
            self._file = None
            self._dataset = None

    def __getitem__(self, key: Any) -> NDArray[Any]:
        """Get item/slice from dataset (triggers partial load)."""
        if self._dataset is None:
            raise RuntimeError("LazyHDF5Array must be used as context manager")
        return np.asarray(self._dataset[key])  # type: ignore[unreachable]

    @property
    def shape(self) -> tuple[int, ...]:
        """Dataset shape."""
        if self._dataset is None:
            raise RuntimeError("LazyHDF5Array must be used as context manager")
        return self._dataset.shape  # type: ignore[unreachable]

    @property
    def dtype(self) -> np.dtype[Any]:
        """Dataset dtype."""
        if self._dataset is None:
            raise RuntimeError("LazyHDF5Array must be used as context manager")
        return self._dataset.dtype  # type: ignore[unreachable]

    @property
    def size(self) -> int:
        """Total number of elements."""
        if self._dataset is None:
            raise RuntimeError("LazyHDF5Array must be used as context manager")
        return self._dataset.size  # type: ignore[unreachable]

    def __len__(self) -> int:
        """Length of first dimension."""
        if self._dataset is None:
            raise RuntimeError("LazyHDF5Array must be used as context manager")
        return len(self._dataset)  # type: ignore[unreachable]


__all__ = [
    "ArrayManager",
    "LRUCache",
    "LazyHDF5Array",
    "ResourceManager",
    "cache_key",
    "clear_cache",
    "get_result_cache",
    "load_hdf5_lazy",
    "show_cache_stats",
]
