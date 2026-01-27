"""Persistent LRU cache with disk spillover for Oscura intermediate results.

This module provides memory-bounded caching for expensive operations like FFT,
spectrograms, and filtered traces with automatic disk spillover when memory
limits are exceeded.


Example:
    >>> from oscura.core.cache import OscuraCache, get_cache
    >>> cache = get_cache(max_memory="2GB")
    >>> result = cache.get_or_compute("fft_key", compute_fft, signal, 1024)
    >>> cache.show_stats()
    Cache Statistics: 42 hits, 15 misses (73.7% hit rate)

References:
    Python functools.lru_cache
    Python pickle for serialization
"""

from __future__ import annotations

import contextlib
import hashlib
import hmac
import logging
import pickle
import secrets
import tempfile
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np

from oscura.core.exceptions import SecurityError

if TYPE_CHECKING:
    from collections.abc import Callable


logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class CacheEntry:
    """Single cache entry with metadata.

    Attributes:
        key: Cache key (hash of inputs).
        value: Cached value (in memory).
        disk_path: Path to disk file if spilled.
        size_bytes: Size of cached value in bytes.
        created_at: Creation timestamp.
        last_accessed: Last access timestamp.
        access_count: Number of times accessed.
        in_memory: True if value is in memory.
    """

    key: str
    value: Any | None
    disk_path: Path | None
    size_bytes: int
    created_at: float
    last_accessed: float
    access_count: int
    in_memory: bool


@dataclass
class CacheStats:
    """Cache statistics.


    Attributes:
        hits: Number of cache hits.
        misses: Number of cache misses.
        evictions: Number of entries evicted.
        disk_spills: Number of entries spilled to disk.
        current_memory: Current memory usage (bytes).
        current_entries: Number of entries in cache.
        disk_entries: Number of entries on disk.
    """

    hits: int
    misses: int
    evictions: int
    disk_spills: int
    current_memory: int
    current_entries: int
    disk_entries: int

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate (0.0-1.0)."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def __str__(self) -> str:
        """Format stats as human-readable string."""
        return (
            f"Cache Statistics:\n"
            f"  Hits: {self.hits}\n"
            f"  Misses: {self.misses}\n"
            f"  Hit Rate: {self.hit_rate * 100:.1f}%\n"
            f"  Evictions: {self.evictions}\n"
            f"  Disk Spills: {self.disk_spills}\n"
            f"  Memory Usage: {self.current_memory / 1e9:.2f} GB\n"
            f"  Entries (Memory): {self.current_entries}\n"
            f"  Entries (Disk): {self.disk_entries}\n"
        )


class OscuraCache:
    """LRU cache with disk spillover for intermediate results.


    Caches expensive computation results with automatic memory management.
    When memory limit is exceeded, least-recently-used entries are spilled
    to disk. Automatic cleanup on exit.

    Args:
        max_memory: Maximum memory for cache (bytes or string like "2GB").
        cache_dir: Directory for disk cache (default: /tmp/oscura_cache).
        auto_cleanup: Clean up disk cache on exit (default: True).

    Example:
        >>> cache = OscuraCache(max_memory="1GB")
        >>> result = cache.get_or_compute("key", expensive_func, arg1, arg2)
        >>> stats = cache.get_stats()
        >>> cache.clear()

    Security Note:
        Cache files use pickle serialization. Cache directory should be in
        a secure location with appropriate permissions. Do not share cache
        directories across security boundaries. The cache is intended for
        single-user, local computation only.

    References:
        MEM-031: Persistent Cache (Disk-Based)
        MEM-029: LRU Cache for Intermediate Results
    """

    def __init__(
        self,
        max_memory: int | str = "2GB",
        *,
        cache_dir: str | Path | None = None,
        auto_cleanup: bool = True,
    ):
        """Initialize cache.

        Args:
            max_memory: Maximum memory (bytes or string).
            cache_dir: Directory for disk cache.
            auto_cleanup: Clean up on exit.
        """
        # Parse max_memory
        if isinstance(max_memory, str):
            self.max_memory = self._parse_memory_string(max_memory)
        else:
            self.max_memory = int(max_memory)

        # Set up cache directory
        if cache_dir is None:
            self.cache_dir = Path(tempfile.gettempdir()) / "oscura_cache"
        else:
            self.cache_dir = Path(cache_dir)

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.auto_cleanup = auto_cleanup

        # Cache storage (LRU via OrderedDict)
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()

        # Thread lock for thread-safe operations (MEM-031)
        self._lock = threading.RLock()

        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._disk_spills = 0
        self._current_memory = 0

        # Security: HMAC signing key for cache integrity (SEC-003 fix)
        self._cache_key = self._load_or_create_cache_key()

    def __enter__(self) -> OscuraCache:
        """Enter context."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore[no-untyped-def]
        """Exit context and clean up if enabled."""
        # Note: exc_val and exc_tb intentionally unused but required for Python 3.11+ compatibility
        if self.auto_cleanup:
            self.clear()

    def get(self, key: str) -> Any | None:
        """Get value from cache.


        Args:
            key: Cache key.

        Returns:
            Cached value or None if not found.

        Example:
            >>> value = cache.get("my_key")
            >>> if value is None:
            ...     value = compute_value()
            ...     cache.put("my_key", value)

        References:
            MEM-031: Persistent Cache (Disk-Based)
        """
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            # Cache hit
            self._hits += 1
            entry = self._cache[key]

            # Update access metadata
            entry.last_accessed = time.time()
            entry.access_count += 1

            # Move to end (most recently used)
            self._cache.move_to_end(key)

            # Load from disk if needed
            if not entry.in_memory:
                try:
                    entry.value = self._load_from_disk(entry.disk_path)  # type: ignore[arg-type]
                    entry.in_memory = True
                    self._current_memory += entry.size_bytes

                    # Check if we need to spill to make room
                    self._ensure_memory_limit()
                except SecurityError:
                    # Re-raise security errors (tampered data)
                    raise
                except (OSError, pickle.UnpicklingError):
                    # Remove corrupted entry from cache for non-security errors
                    del self._cache[key]
                    self._misses += 1
                    return None

            return entry.value

    def put(self, key: str, value: Any) -> None:
        """Put value in cache.


        Args:
            key: Cache key.
            value: Value to cache.

        Example:
            >>> cache.put("result_key", computed_result)

        References:
            MEM-031: Persistent Cache (Disk-Based)
        """
        # Calculate size outside lock (potentially expensive)
        size_bytes = self._estimate_size(value)

        with self._lock:
            # Remove old entry if exists
            if key in self._cache:
                old_entry = self._cache[key]
                self._current_memory -= old_entry.size_bytes
                if old_entry.disk_path and old_entry.disk_path.exists():
                    old_entry.disk_path.unlink()
                del self._cache[key]

            # Create new entry
            entry = CacheEntry(
                key=key,
                value=value,
                disk_path=None,
                size_bytes=size_bytes,
                created_at=time.time(),
                last_accessed=time.time(),
                access_count=0,
                in_memory=True,
            )

            # Add to cache
            self._cache[key] = entry
            self._current_memory += size_bytes

            # Ensure memory limit
            self._ensure_memory_limit()

    def get_or_compute(
        self,
        key: str,
        compute_fn: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """Get cached value or compute and cache it.


        Args:
            key: Cache key.
            compute_fn: Function to compute value if not cached.
            args: Arguments to compute_fn.
            kwargs: Keyword arguments to compute_fn.

        Returns:
            Cached or computed value.

        Example:
            >>> result = cache.get_or_compute("fft_key", np.fft.fft, signal)

        References:
            MEM-029: LRU Cache for Intermediate Results
        """
        value = self.get(key)
        if value is not None:
            return value  # type: ignore[no-any-return]

        # Compute value
        value = compute_fn(*args, **kwargs)

        # Cache it
        self.put(key, value)

        return value

    def clear(self) -> None:
        """Clear all cache entries and disk files.


        Example:
            >>> cache.clear()

        References:
            MEM-031: Persistent Cache (Disk-Based)
        """
        with self._lock:
            # Remove disk files
            for entry in self._cache.values():
                if entry.disk_path and entry.disk_path.exists():
                    with contextlib.suppress(OSError):
                        entry.disk_path.unlink()

            # Clear cache
            self._cache.clear()
            self._current_memory = 0

        # Try to remove cache directory if empty (outside lock, not critical)
        try:
            if self.cache_dir.exists() and not any(self.cache_dir.iterdir()):
                self.cache_dir.rmdir()
        except OSError:
            pass

    def get_stats(self) -> CacheStats:
        """Get cache statistics.


        Returns:
            CacheStats object.

        Example:
            >>> stats = cache.get_stats()
            >>> print(f"Hit rate: {stats.hit_rate * 100:.1f}%")

        References:
            MEM-031: Persistent Cache (Disk-Based)
        """
        with self._lock:
            disk_entries = sum(1 for e in self._cache.values() if not e.in_memory)
            return CacheStats(
                hits=self._hits,
                misses=self._misses,
                evictions=self._evictions,
                disk_spills=self._disk_spills,
                current_memory=self._current_memory,
                current_entries=len(self._cache),
                disk_entries=disk_entries,
            )

    def show_stats(self) -> None:
        """Print cache statistics.


        Example:
            >>> cache.show_stats()
            Cache Statistics: 42 hits, 15 misses (73.7% hit rate)

        References:
            MEM-031: Persistent Cache (Disk-Based)
        """
        stats = self.get_stats()
        print(stats)

    def compute_key(self, *args: Any, **kwargs: Any) -> str:
        """Compute cache key from arguments.

        Creates a hash key from arbitrary arguments for cache lookups.

        Args:
            args: Positional arguments.
            kwargs: Keyword arguments.

        Returns:
            Hash key string.

        Example:
            >>> key = cache.compute_key("operation", param1=10, param2="value")

        References:
            MEM-029: LRU Cache for Intermediate Results
        """
        # Create hashable representation
        hash_obj = hashlib.sha256()

        # Hash positional args
        for arg in args:
            hash_obj.update(self._make_hashable(arg))

        # Hash keyword args (sorted for consistency)
        for k in sorted(kwargs.keys()):
            hash_obj.update(k.encode())
            hash_obj.update(self._make_hashable(kwargs[k]))

        return hash_obj.hexdigest()

    def _ensure_memory_limit(self) -> None:
        """Ensure cache memory usage is within limit."""
        while self._current_memory > self.max_memory and self._cache:
            # Evict least recently used entry
            key, entry = self._cache.popitem(last=False)

            if entry.in_memory:
                # Spill to disk if not already there
                if entry.disk_path is None:
                    entry.disk_path = self._spill_to_disk(key, entry.value)
                    entry.in_memory = False
                    entry.value = None
                    self._disk_spills += 1

                    # Put back in cache (on disk)
                    self._cache[key] = entry
                    self._cache.move_to_end(key, last=False)

                self._current_memory -= entry.size_bytes

            self._evictions += 1

    def _spill_to_disk(self, key: str, value: Any) -> Path:
        """Write value to disk with HMAC signature.

        Args:
            key: Cache key.
            value: Value to write.

        Returns:
            Path to disk file.

        Security:
            SEC-003 fix: Writes HMAC-SHA256 signature + pickled data.
            Format: [32 bytes signature][pickled data]
            Signature computed over pickled data using self._cache_key.

        References:
            MEM-031: Persistent Cache (Disk-Based)
        """
        disk_path = self.cache_dir / f"{key}.pkl"

        # Serialize data
        data = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)

        # Compute HMAC-SHA256 signature
        signature = hmac.new(self._cache_key, data, hashlib.sha256).digest()

        # Write signature + data
        with open(disk_path, "wb") as f:
            f.write(signature)  # First 32 bytes
            f.write(data)  # Rest is pickled data

        return disk_path

    def _load_from_disk(self, disk_path: Path) -> Any:
        """Load value from disk with HMAC verification.

        Args:
            disk_path: Path to disk file.

        Returns:
            Loaded value.

        Raises:
            SecurityError: If HMAC verification fails (tampered cache file).

        Security:
            SEC-003 fix: Verifies HMAC-SHA256 signature before unpickling.
            Prevents code execution from tampered cache files.
            Uses constant-time comparison (hmac.compare_digest).

        References:
            MEM-031: Persistent Cache (Disk-Based)
        """
        try:
            with open(disk_path, "rb") as f:
                signature = f.read(32)  # SHA256 = 32 bytes
                data = f.read()

            # Check if file is too short (corrupted, not tampered)
            if len(signature) < 32:
                logger.warning(f"Cache file too short (corrupted): {disk_path.name}")
                disk_path.unlink(missing_ok=True)
                raise OSError(f"Cache file corrupted (too short): {disk_path.name}")

            # Verify HMAC signature
            expected_signature = hmac.new(self._cache_key, data, hashlib.sha256).digest()

            if not hmac.compare_digest(signature, expected_signature):
                logger.error(f"Cache integrity check failed for {disk_path.name}")
                # Delete corrupted cache file
                disk_path.unlink(missing_ok=True)
                raise SecurityError(
                    f"Cache file integrity verification failed: {disk_path.name}. "
                    "File may have been tampered with and has been removed."
                )

            # Deserialize only after HMAC verification
            return pickle.loads(data)

        except SecurityError:
            raise  # Re-raise security errors
        except Exception as e:
            logger.warning(f"Failed to load cache file {disk_path.name}: {e}")
            # Clean up corrupted file
            disk_path.unlink(missing_ok=True)
            raise

    def _estimate_size(self, value: Any) -> int:
        """Estimate size of value in bytes."""
        if isinstance(value, np.ndarray):
            return value.nbytes
        elif isinstance(value, list | tuple):
            return sum(self._estimate_size(item) for item in value)
        elif isinstance(value, dict):
            return sum(self._estimate_size(k) + self._estimate_size(v) for k, v in value.items())
        else:
            # Fallback: use pickle size
            try:
                return len(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
            except (TypeError, pickle.PicklingError):
                return 1024  # Default estimate

    def _make_hashable(self, obj: Any) -> bytes:
        """Convert object to hashable bytes."""
        if isinstance(obj, np.ndarray):
            # Use array bytes for hashing
            return obj.tobytes()
        elif isinstance(obj, str | bytes):
            return obj.encode() if isinstance(obj, str) else obj
        elif isinstance(obj, int | float | bool):
            return str(obj).encode()
        elif isinstance(obj, list | tuple):
            return b"".join(self._make_hashable(item) for item in obj)
        else:
            # Fallback: use string representation
            return str(obj).encode()

    def _parse_memory_string(self, memory_str: str) -> int:
        """Parse memory string like '2GB' to bytes."""
        memory_str = memory_str.strip().upper()

        if memory_str.endswith("GB"):
            return int(float(memory_str[:-2]) * 1e9)
        elif memory_str.endswith("MB"):
            return int(float(memory_str[:-2]) * 1e6)
        elif memory_str.endswith("KB"):
            return int(float(memory_str[:-2]) * 1e3)
        else:
            return int(memory_str)

    def _load_or_create_cache_key(self) -> bytes:
        """Load or create HMAC signing key for cache integrity.

        Returns:
            256-bit signing key.

        Security:
            SEC-003 fix: Protects cached pickle files from tampering.
            Key is persistent per cache directory and stored with 0o600 permissions.
            Each cache directory has its own unique key.

        References:
            https://owasp.org/www-project-top-ten/
        """
        key_file = self.cache_dir / ".cache_key"

        # Load existing key
        if key_file.exists():
            with open(key_file, "rb") as f:
                return f.read()

        # Create new 256-bit key
        key = secrets.token_bytes(32)

        # Save with restrictive permissions
        with open(key_file, "wb") as f:
            f.write(key)

        # Set owner read/write only (0o600)
        key_file.chmod(0o600)

        logger.info(f"Created new cache signing key: {key_file}")
        return key


# Global cache instance
_global_cache: OscuraCache | None = None


def get_cache(
    max_memory: int | str = "2GB",
    *,
    cache_dir: str | Path | None = None,
) -> OscuraCache:
    """Get or create global cache instance.


    Args:
        max_memory: Maximum memory for cache.
        cache_dir: Cache directory.

    Returns:
        Global OscuraCache instance.

    Example:
        >>> cache = get_cache(max_memory="1GB")
        >>> result = cache.get_or_compute("key", compute_fn, args)

    References:
        MEM-031: Persistent Cache (Disk-Based)
    """
    global _global_cache

    if _global_cache is None:
        _global_cache = OscuraCache(max_memory, cache_dir=cache_dir)

    return _global_cache


def clear_cache() -> None:
    """Clear global cache.

    Example:
        >>> clear_cache()

    References:
        MEM-031: Persistent Cache (Disk-Based)
    """
    global _global_cache

    if _global_cache is not None:
        _global_cache.clear()
        _global_cache = None


def show_cache_stats() -> None:
    """Show global cache statistics.


    Example:
        >>> show_cache_stats()
        Cache Statistics: 42 hits, 15 misses (73.7% hit rate)

    References:
        MEM-031: Persistent Cache (Disk-Based)
    """
    if _global_cache is not None:
        _global_cache.show_stats()
    else:
        print("Cache not initialized")


__all__ = [
    "CacheEntry",
    "CacheStats",
    "OscuraCache",
    "clear_cache",
    "get_cache",
    "show_cache_stats",
]
