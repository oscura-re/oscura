"""Caching layer for expensive computations in signal analysis.

This module provides a comprehensive caching system for expensive operations
like FFT, correlation, and protocol decoding. Supports multiple backends
(memory, disk, Redis) with automatic key generation, TTL expiration, LRU
eviction, and cache statistics.

Example:
    >>> cache = CacheManager(backend="memory", max_size_mb=100)
    >>> @cache.cached(ttl=3600)
    ... def expensive_fft(signal: np.ndarray) -> np.ndarray:
    ...     return np.fft.fft(signal)
    >>> result = expensive_fft(my_signal)  # Computed and cached
    >>> result = expensive_fft(my_signal)  # Retrieved from cache

References:
    Cache algorithms: https://en.wikipedia.org/wiki/Cache_replacement_policies
    Redis protocol: https://redis.io/docs/reference/protocol-spec/
"""

from __future__ import annotations

import functools
import hashlib
import hmac
import json
import logging
import pickle
import secrets
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np

from oscura.core.exceptions import SecurityError

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

__all__ = [
    "CacheBackend",
    "CacheEntry",
    "CacheManager",
    "CachePolicy",
    "CacheStats",
]

# Type aliases
CacheKey: TypeAlias = str
CacheValue: TypeAlias = Any


class CacheBackend(Enum):
    """Cache storage backend.

    Attributes:
        MEMORY: In-memory cache using OrderedDict (LRU)
        DISK: Disk-based cache using pickle files
        REDIS: Distributed cache using Redis (optional, graceful degradation)
        MULTI_LEVEL: Memory cache with disk fallback
    """

    MEMORY = "memory"
    DISK = "disk"
    REDIS = "redis"
    MULTI_LEVEL = "multi_level"


class EvictionPolicy(Enum):
    """Cache eviction policy.

    Attributes:
        LRU: Least Recently Used
        LFU: Least Frequently Used
        FIFO: First In First Out
        SIZE_BASED: Evict when size limit reached
    """

    LRU = "lru"
    LFU = "lfu"
    FIFO = "fifo"
    SIZE_BASED = "size_based"


@dataclass
class CacheEntry:
    """Cached data entry with metadata.

    Attributes:
        key: Cache key (hash of function + args)
        value: Cached value
        timestamp: Creation time (Unix timestamp)
        access_count: Number of times accessed
        ttl: Time-to-live in seconds (None = no expiration)
        size_bytes: Approximate size in bytes
        last_access: Last access time (Unix timestamp)
    """

    key: str
    value: Any
    timestamp: float
    access_count: int = 0
    ttl: float | None = None
    size_bytes: int = 0
    last_access: float = field(default_factory=time.time)

    def is_expired(self) -> bool:
        """Check if entry has expired based on TTL.

        Returns:
            True if expired, False otherwise
        """
        if self.ttl is None:
            return False
        return (time.time() - self.timestamp) > self.ttl

    def touch(self) -> None:
        """Update access metadata."""
        self.access_count += 1
        self.last_access = time.time()


@dataclass
class CacheStats:
    """Cache performance statistics.

    Attributes:
        hits: Number of cache hits
        misses: Number of cache misses
        hit_rate: Hit rate (hits / (hits + misses))
        size_mb: Current cache size in megabytes
        entry_count: Number of cached entries
        evictions: Number of entries evicted
        expired: Number of entries expired
        backend: Cache backend type
    """

    hits: int = 0
    misses: int = 0
    hit_rate: float = 0.0
    size_mb: float = 0.0
    entry_count: int = 0
    evictions: int = 0
    expired: int = 0
    backend: str = "memory"

    def __post_init__(self) -> None:
        """Calculate derived statistics."""
        total = self.hits + self.misses
        self.hit_rate = self.hits / total if total > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Export statistics to dictionary.

        Returns:
            Dictionary with all statistics
        """
        return asdict(self)

    def to_json(self, filepath: str | Path) -> None:
        """Export statistics to JSON file.

        Args:
            filepath: Output JSON file path
        """
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


@dataclass
class CachePolicy:
    """Cache behavior policy.

    Attributes:
        ttl: Default time-to-live in seconds (None = no expiration)
        max_size_mb: Maximum cache size in megabytes
        eviction: Eviction policy when size limit reached
        serialize_numpy: Whether to pickle numpy arrays
        compress: Whether to compress cached data
        version: Cache version (invalidate when changed)
    """

    ttl: float | None = 3600.0  # 1 hour default
    max_size_mb: float = 100.0
    eviction: EvictionPolicy = EvictionPolicy.LRU
    serialize_numpy: bool = True
    compress: bool = False
    version: str = "1.0"


class CacheManager:
    """Multi-backend cache manager for expensive computations.

    Manages caching with automatic key generation, TTL expiration, size-based
    eviction, and performance statistics. Supports memory, disk, and Redis backends.

    Args:
        backend: Cache storage backend
        cache_dir: Directory for disk cache (default: ~/.cache/oscura)
        policy: Cache behavior policy
        redis_url: Redis connection URL (for REDIS backend)

    Example:
        >>> cache = CacheManager(backend="memory", policy=CachePolicy(max_size_mb=50))
        >>> @cache.cached(ttl=1800)
        ... def compute_fft(signal: np.ndarray) -> np.ndarray:
        ...     return np.fft.fft(signal)
        >>> result = compute_fft(signal)  # Cached
        >>> stats = cache.get_stats()
        >>> print(f"Hit rate: {stats.hit_rate:.2%}")
    """

    def __init__(
        self,
        backend: str | CacheBackend = CacheBackend.MEMORY,
        cache_dir: str | Path | None = None,
        policy: CachePolicy | None = None,
        redis_url: str | None = None,
    ) -> None:
        """Initialize cache manager.

        Args:
            backend: Cache storage backend
            cache_dir: Directory for disk cache
            policy: Cache behavior policy
            redis_url: Redis connection URL
        """
        self.backend = CacheBackend(backend) if isinstance(backend, str) else backend
        self.policy = policy or CachePolicy()

        # Cache directory setup
        if cache_dir is None:
            self.cache_dir = Path.home() / ".cache" / "oscura" / "performance"
        else:
            self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize storage
        self._memory_cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._disk_cache_index: dict[str, Path] = {}
        self._redis_client: Any = None

        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._expired = 0

        # Security: HMAC signing key for cache integrity (SEC-003 fix)
        self._cache_key = self._load_or_create_cache_key()

        # Initialize backend
        if self.backend == CacheBackend.REDIS:
            self._init_redis(redis_url)
        elif self.backend == CacheBackend.DISK:
            self._load_disk_index()

    def _init_redis(self, redis_url: str | None) -> None:
        """Initialize Redis connection with graceful degradation.

        Args:
            redis_url: Redis connection URL
        """
        try:
            import redis  # type: ignore[import-not-found]

            self._redis_client = redis.from_url(redis_url or "redis://localhost:6379")
            self._redis_client.ping()
            logger.info("Redis cache backend initialized")
        except ImportError:
            logger.warning("Redis not available, falling back to memory cache")
            self.backend = CacheBackend.MEMORY
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}, falling back to memory cache")
            self.backend = CacheBackend.MEMORY

    def _load_disk_index(self) -> None:
        """Load disk cache index from cache directory."""
        index_file = self.cache_dir / "cache_index.json"
        if index_file.exists():
            try:
                with open(index_file) as f:
                    data = json.load(f)
                    self._disk_cache_index = {k: Path(v) for k, v in data.items()}
                logger.info(f"Loaded disk cache index: {len(self._disk_cache_index)} entries")
            except Exception as e:
                logger.warning(f"Failed to load disk cache index: {e}")
                self._disk_cache_index = {}

    def _save_disk_index(self) -> None:
        """Save disk cache index to cache directory."""
        index_file = self.cache_dir / "cache_index.json"
        try:
            with open(index_file, "w") as f:
                data = {k: str(v) for k, v in self._disk_cache_index.items()}
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save disk cache index: {e}")

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

    def _generate_key(self, func_name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
        """Generate deterministic cache key from function and arguments.

        Args:
            func_name: Function name
            args: Positional arguments
            kwargs: Keyword arguments

        Returns:
            SHA256 hex digest cache key
        """
        key_parts = [self.policy.version, func_name]

        # Hash arguments
        for arg in args:
            key_parts.append(self._hash_value(arg))

        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}={self._hash_value(v)}")

        key_str = ":".join(key_parts)
        return hashlib.sha256(key_str.encode()).hexdigest()

    def _hash_value(self, value: Any) -> str:
        """Hash a single value (supports numpy arrays).

        Note:
            Uses MD5 for cache key generation only (not for security).
            MD5 is appropriate here for non-cryptographic checksums.

        Args:
            value: Value to hash

        Returns:
            Hex digest of value
        """
        if isinstance(value, np.ndarray):
            # Hash array data (MD5 used for cache keys only, not security)
            return hashlib.md5(value.tobytes(), usedforsecurity=False).hexdigest()
        elif isinstance(value, (list, tuple)):
            # Hash sequences
            return hashlib.md5(
                str([self._hash_value(v) for v in value]).encode(), usedforsecurity=False
            ).hexdigest()
        elif isinstance(value, dict):
            # Hash dicts
            items = sorted((k, self._hash_value(v)) for k, v in value.items())
            return hashlib.md5(str(items).encode(), usedforsecurity=False).hexdigest()
        else:
            # Hash other types via string representation
            return hashlib.md5(str(value).encode(), usedforsecurity=False).hexdigest()

    def _estimate_size(self, value: Any) -> int:
        """Estimate memory size of value in bytes.

        Args:
            value: Value to measure

        Returns:
            Approximate size in bytes
        """
        if isinstance(value, np.ndarray):
            return value.nbytes
        elif isinstance(value, (str, bytes)):
            return len(value)
        elif isinstance(value, (list, tuple)):
            return sum(self._estimate_size(v) for v in value)
        elif isinstance(value, dict):
            return sum(self._estimate_size(k) + self._estimate_size(v) for k, v in value.items())
        else:
            # Rough estimate using pickle
            try:
                return len(pickle.dumps(value))
            except Exception:
                return 0

    def _get_cache_size_mb(self) -> float:
        """Calculate current cache size in megabytes.

        Returns:
            Cache size in MB
        """
        if self.backend == CacheBackend.MEMORY or self.backend == CacheBackend.MULTI_LEVEL:
            total_bytes = sum(entry.size_bytes for entry in self._memory_cache.values())
            return total_bytes / (1024 * 1024)
        elif self.backend == CacheBackend.DISK:
            total_bytes = sum(
                path.stat().st_size for path in self._disk_cache_index.values() if path.exists()
            )
            return total_bytes / (1024 * 1024)
        return 0.0

    def _evict_if_needed(self) -> None:
        """Evict entries if cache size exceeds limit."""
        while self._get_cache_size_mb() > self.policy.max_size_mb:
            if self.backend == CacheBackend.MEMORY or self.backend == CacheBackend.MULTI_LEVEL:
                if not self._memory_cache:
                    break

                if self.policy.eviction == EvictionPolicy.LRU:
                    # Remove least recently used (first in OrderedDict)
                    self._memory_cache.popitem(last=False)
                elif self.policy.eviction == EvictionPolicy.FIFO:
                    # Remove oldest entry
                    self._memory_cache.popitem(last=False)
                else:
                    # Default to LRU
                    self._memory_cache.popitem(last=False)

                self._evictions += 1
            elif self.backend == CacheBackend.DISK:
                if not self._disk_cache_index:
                    break

                # Remove oldest file
                oldest_key = next(iter(self._disk_cache_index))
                oldest_path = self._disk_cache_index[oldest_key]
                if oldest_path.exists():
                    oldest_path.unlink()
                del self._disk_cache_index[oldest_key]
                self._evictions += 1
            else:
                break

    def _memory_get(self, key: str) -> CacheEntry | None:
        """Get entry from memory cache.

        Args:
            key: Cache key

        Returns:
            Cache entry or None if not found/expired
        """
        if key not in self._memory_cache:
            return None

        entry = self._memory_cache[key]

        # Check expiration
        if entry.is_expired():
            del self._memory_cache[key]
            self._expired += 1
            return None

        # Update LRU order
        self._memory_cache.move_to_end(key)
        entry.touch()
        return entry

    def _memory_set(self, key: str, entry: CacheEntry) -> None:
        """Set entry in memory cache.

        Args:
            key: Cache key
            entry: Cache entry
        """
        self._memory_cache[key] = entry
        self._memory_cache.move_to_end(key)
        self._evict_if_needed()

    def _disk_get(self, key: str) -> CacheEntry | None:
        """Get entry from disk cache with HMAC verification.

        Args:
            key: Cache key

        Returns:
            Cache entry or None if not found/expired

        Raises:
            SecurityError: If HMAC verification fails (tampered cache file)

        Security:
            SEC-003 fix: Verifies HMAC-SHA256 signature before unpickling.
            Prevents code execution from tampered cache files.
            Uses constant-time comparison (hmac.compare_digest).
        """
        if key not in self._disk_cache_index:
            return None

        cache_file = self._disk_cache_index[key]
        if not cache_file.exists():
            del self._disk_cache_index[key]
            return None

        try:
            with open(cache_file, "rb") as f:
                signature = f.read(32)  # SHA256 = 32 bytes
                data = f.read()

            # Verify HMAC signature
            expected_signature = hmac.new(self._cache_key, data, hashlib.sha256).digest()

            if not hmac.compare_digest(signature, expected_signature):
                logger.error(f"Cache integrity check failed for {key}")
                # Delete corrupted cache file
                cache_file.unlink()
                del self._disk_cache_index[key]
                raise SecurityError(
                    f"Cache file integrity verification failed: {key}. "
                    "File may have been tampered with and has been removed."
                )

            # Deserialize only after HMAC verification
            loaded_entry: CacheEntry = pickle.loads(data)

            # Check expiration
            if loaded_entry.is_expired():
                cache_file.unlink()
                del self._disk_cache_index[key]
                self._expired += 1
                return None

            loaded_entry.touch()
            return loaded_entry

        except SecurityError:
            raise  # Re-raise security errors
        except Exception as e:
            logger.warning(f"Failed to load cache entry {key}: {e}")
            if cache_file.exists():
                cache_file.unlink()
            del self._disk_cache_index[key]
            return None

    def _disk_set(self, key: str, entry: CacheEntry) -> None:
        """Set entry in disk cache with HMAC signature.

        Args:
            key: Cache key
            entry: Cache entry

        Security:
            SEC-003 fix: Writes HMAC-SHA256 signature + pickled data.
            Format: [32 bytes signature][pickled data]
            Signature computed over pickled data using self._cache_key.
        """
        cache_file = self.cache_dir / f"{key}.pkl"
        try:
            # Serialize entry
            data = pickle.dumps(entry, protocol=pickle.HIGHEST_PROTOCOL)

            # Compute HMAC-SHA256 signature
            signature = hmac.new(self._cache_key, data, hashlib.sha256).digest()

            # Write signature + data
            with open(cache_file, "wb") as f:
                f.write(signature)  # First 32 bytes
                f.write(data)  # Rest is pickled data

            self._disk_cache_index[key] = cache_file
            self._evict_if_needed()
            self._save_disk_index()
        except Exception as e:
            logger.warning(f"Failed to save cache entry {key}: {e}")

    def get(self, key: str) -> Any | None:
        """Retrieve value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        entry = None

        if self.backend == CacheBackend.MEMORY:
            entry = self._memory_get(key)
        elif self.backend == CacheBackend.DISK:
            entry = self._disk_get(key)
        elif self.backend == CacheBackend.MULTI_LEVEL:
            # Try memory first, then disk
            entry = self._memory_get(key)
            if entry is None:
                entry = self._disk_get(key)
                # Promote to memory cache
                if entry is not None:
                    self._memory_set(key, entry)
        elif self.backend == CacheBackend.REDIS and self._redis_client:
            try:
                data = self._redis_client.get(key)
                if data:
                    # Security: Only loading from trusted Redis cache
                    loaded_entry: CacheEntry = pickle.loads(data)
                    if not loaded_entry.is_expired():
                        loaded_entry.touch()
                        entry = loaded_entry
                    else:
                        self._redis_client.delete(key)
                        entry = None
                        self._expired += 1
            except Exception as e:
                logger.warning(f"Redis get failed: {e}")

        if entry:
            self._hits += 1
            return entry.value
        else:
            self._misses += 1
            return None

    def set(self, key: str, value: Any, ttl: float | None = None) -> None:
        """Store value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (None uses policy default)
        """
        entry = CacheEntry(
            key=key,
            value=value,
            timestamp=time.time(),
            ttl=ttl if ttl is not None else self.policy.ttl,
            size_bytes=self._estimate_size(value),
        )

        if self.backend == CacheBackend.MEMORY:
            self._memory_set(key, entry)
        elif self.backend == CacheBackend.DISK:
            self._disk_set(key, entry)
        elif self.backend == CacheBackend.MULTI_LEVEL:
            self._memory_set(key, entry)
            # Also save to disk for persistence
            self._disk_set(key, entry)
        elif self.backend == CacheBackend.REDIS and self._redis_client:
            try:
                data = pickle.dumps(entry)
                if entry.ttl:
                    self._redis_client.setex(key, int(entry.ttl), data)
                else:
                    self._redis_client.set(key, data)
            except Exception as e:
                logger.warning(f"Redis set failed: {e}")

    def invalidate(self, pattern: str | None = None) -> int:
        """Invalidate cache entries by key pattern.

        Args:
            pattern: Key pattern to match (None = clear all)

        Returns:
            Number of entries invalidated
        """
        if pattern is None:
            return self._clear_all_caches()
        else:
            return self._clear_by_pattern(pattern)

    def _clear_all_caches(self) -> int:
        """Clear all cache entries.

        Returns:
            Number of entries invalidated.
        """
        invalidated = 0

        # Clear memory cache
        if self.backend in (CacheBackend.MEMORY, CacheBackend.MULTI_LEVEL):
            invalidated += len(self._memory_cache)
            self._memory_cache.clear()

        # Clear disk cache
        if self.backend in (CacheBackend.DISK, CacheBackend.MULTI_LEVEL):
            invalidated += self._clear_disk_cache()

        # Clear Redis cache
        if self.backend == CacheBackend.REDIS and self._redis_client:
            self._clear_redis_cache()

        return invalidated

    def _clear_disk_cache(self) -> int:
        """Clear all disk cache entries.

        Returns:
            Number of entries cleared.
        """
        for cache_file in self._disk_cache_index.values():
            if cache_file.exists():
                cache_file.unlink()

        count = len(self._disk_cache_index)
        self._disk_cache_index.clear()
        self._save_disk_index()
        return count

    def _clear_redis_cache(self) -> None:
        """Clear all Redis cache entries."""
        try:
            self._redis_client.flushdb()
        except Exception as e:
            logger.warning(f"Redis flush failed: {e}")

    def _clear_by_pattern(self, pattern: str) -> int:
        """Clear cache entries matching pattern.

        Args:
            pattern: Pattern to match in keys.

        Returns:
            Number of entries invalidated.
        """
        invalidated = 0

        # Clear memory cache by pattern
        if self.backend in (CacheBackend.MEMORY, CacheBackend.MULTI_LEVEL):
            invalidated += self._clear_memory_by_pattern(pattern)

        # Clear disk cache by pattern
        if self.backend in (CacheBackend.DISK, CacheBackend.MULTI_LEVEL):
            invalidated += self._clear_disk_by_pattern(pattern)

        return invalidated

    def _clear_memory_by_pattern(self, pattern: str) -> int:
        """Clear memory cache entries matching pattern.

        Args:
            pattern: Pattern to match.

        Returns:
            Number of entries cleared.
        """
        keys_to_remove = [k for k in self._memory_cache if pattern in k]
        for k in keys_to_remove:
            del self._memory_cache[k]
        return len(keys_to_remove)

    def _clear_disk_by_pattern(self, pattern: str) -> int:
        """Clear disk cache entries matching pattern.

        Args:
            pattern: Pattern to match.

        Returns:
            Number of entries cleared.
        """
        keys_to_remove = [k for k in self._disk_cache_index if pattern in k]
        for k in keys_to_remove:
            cache_file = self._disk_cache_index[k]
            if cache_file.exists():
                cache_file.unlink()
            del self._disk_cache_index[k]

        if keys_to_remove:
            self._save_disk_index()

        return len(keys_to_remove)

    def get_stats(self) -> CacheStats:
        """Get cache performance statistics.

        Returns:
            Cache statistics including hit rate and size
        """
        return CacheStats(
            hits=self._hits,
            misses=self._misses,
            size_mb=self._get_cache_size_mb(),
            entry_count=len(self._memory_cache) + len(self._disk_cache_index)
            if self.backend == CacheBackend.MULTI_LEVEL
            else (
                len(self._memory_cache)
                if self.backend == CacheBackend.MEMORY
                else len(self._disk_cache_index)
            ),
            evictions=self._evictions,
            expired=self._expired,
            backend=self.backend.value,
        )

    def cached(
        self, ttl: float | None = None, key_prefix: str = ""
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator for automatic function result caching.

        Args:
            ttl: Time-to-live in seconds (None uses policy default)
            key_prefix: Prefix for cache keys (useful for versioning)

        Returns:
            Decorated function with caching

        Example:
            >>> cache = CacheManager()
            >>> @cache.cached(ttl=3600)
            ... def expensive_computation(x: np.ndarray) -> np.ndarray:
            ...     return np.fft.fft(x)
        """

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                # Generate cache key
                func_name = f"{key_prefix}{func.__module__}.{func.__name__}"
                cache_key = self._generate_key(func_name, args, kwargs)

                # Try to get from cache
                cached_value = self.get(cache_key)
                if cached_value is not None:
                    logger.debug(f"Cache hit for {func_name}")
                    return cached_value

                # Compute and cache
                logger.debug(f"Cache miss for {func_name}, computing...")
                result = func(*args, **kwargs)
                self.set(cache_key, result, ttl=ttl)
                return result

            return wrapper

        return decorator


# Global cache instance for convenience
_global_cache: CacheManager | None = None


def get_global_cache() -> CacheManager:
    """Get or create global cache instance.

    Returns:
        Global CacheManager instance
    """
    global _global_cache
    if _global_cache is None:
        _global_cache = CacheManager()
    return _global_cache


def cache(
    ttl: float | None = None, key_prefix: str = ""
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Convenience decorator using global cache.

    Args:
        ttl: Time-to-live in seconds
        key_prefix: Prefix for cache keys

    Returns:
        Decorated function with caching

    Example:
        >>> @cache(ttl=3600)
        ... def expensive_fft(signal: np.ndarray) -> np.ndarray:
        ...     return np.fft.fft(signal)
    """
    return get_global_cache().cached(ttl=ttl, key_prefix=key_prefix)
