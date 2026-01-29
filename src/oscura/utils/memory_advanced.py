"""Advanced memory management utilities.

This module provides advanced memory management features including
quality modes, cache management, garbage collection, and streaming
backpressure support.
"""

from __future__ import annotations

import contextlib
import gc
import hashlib
import hmac
import json
import logging
import pickle
import secrets
import tempfile
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import numpy as np

from oscura.core.exceptions import SecurityError

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = logging.getLogger(__name__)

__all__ = [
    "AdaptiveMeasurementSelector",
    "BackpressureController",
    "CacheEntry",
    "CacheInvalidationStrategy",
    "DiskCache",
    "GCController",
    "MemoryLogger",
    "MultiChannelMemoryManager",
    "QualityMode",
    "QualityModeConfig",
    "WSLSwapChecker",
    "adaptive_measurements",
    "gc_aggressive",
    "get_wsl_memory_limits",
]


# =============================================================================
# =============================================================================


class QualityMode(Enum):
    """Quality mode for memory-constrained scenarios.

    References:
        MEM-014: Quality vs Memory Trade-offs
    """

    PREVIEW = "preview"  # Max speed, low memory, approximate
    BALANCED = "balanced"  # Default, moderate quality
    HIGH_QUALITY = "high"  # Accurate, may require more memory


@dataclass
class QualityModeConfig:
    """Configuration for quality modes.

    Attributes:
        mode: Quality mode
        downsample_factor: Downsampling factor for preview mode
        nfft_factor: FFT size reduction factor
        overlap_factor: Overlap reduction factor
        enable_caching: Enable intermediate caching
        use_approximations: Use faster approximations

    References:
        MEM-014: Quality vs Memory Trade-offs
    """

    mode: QualityMode = QualityMode.BALANCED
    downsample_factor: int = 1
    nfft_factor: float = 1.0
    overlap_factor: float = 1.0
    enable_caching: bool = True
    use_approximations: bool = False

    @classmethod
    def for_mode(cls, mode: QualityMode | str) -> QualityModeConfig:
        """Get configuration for specified quality mode.

        Args:
            mode: Quality mode

        Returns:
            Appropriate configuration
        """
        if isinstance(mode, str):
            mode = QualityMode(mode.lower())

        if mode == QualityMode.PREVIEW:
            return cls(
                mode=mode,
                downsample_factor=8,
                nfft_factor=0.25,
                overlap_factor=0.5,
                enable_caching=False,
                use_approximations=True,
            )
        elif mode == QualityMode.HIGH_QUALITY:
            return cls(
                mode=mode,
                downsample_factor=1,
                nfft_factor=2.0,
                overlap_factor=1.5,
                enable_caching=True,
                use_approximations=False,
            )
        else:  # BALANCED
            return cls(
                mode=mode,
                downsample_factor=1,
                nfft_factor=1.0,
                overlap_factor=1.0,
                enable_caching=True,
                use_approximations=False,
            )


# Global quality mode
_current_quality_mode = QualityMode.BALANCED


def set_quality_mode(mode: QualityMode | str) -> None:
    """Set global quality mode.

    Args:
        mode: Quality mode to use
    """
    global _current_quality_mode
    if isinstance(mode, str):
        mode = QualityMode(mode.lower())
    _current_quality_mode = mode
    logger.info(f"Quality mode set to: {mode.value}")


def get_quality_mode() -> QualityMode:
    """Get current quality mode."""
    return _current_quality_mode


def get_quality_config() -> QualityModeConfig:
    """Get configuration for current quality mode."""
    return QualityModeConfig.for_mode(_current_quality_mode)


# =============================================================================
# =============================================================================


class GCController:
    """Garbage collection controller.

    Controls when and how garbage collection occurs based on
    memory pressure and operation completion.

    References:
        MEM-020: Garbage Collection Triggers
    """

    def __init__(self, aggressive: bool = False) -> None:
        """Initialize GC controller.

        Args:
            aggressive: Enable aggressive GC mode
        """
        self._aggressive = aggressive
        self._collection_count = 0
        self._bytes_collected = 0

    @property
    def aggressive(self) -> bool:
        """Check if aggressive mode enabled."""
        return self._aggressive

    @aggressive.setter
    def aggressive(self, value: bool) -> None:
        """Set aggressive mode."""
        self._aggressive = value

    def collect(self) -> int:
        """Perform garbage collection.

        Returns:
            Number of objects collected
        """
        if self._aggressive:
            # Full collection with all generations
            collected = gc.collect(generation=2)
        else:
            # Standard collection
            collected = gc.collect()

        self._collection_count += 1
        self._bytes_collected += collected
        logger.debug(f"GC collected {collected} objects")
        return collected

    def collect_after_operation(self) -> None:
        """Collect garbage after a large operation."""
        if self._aggressive:
            # Force immediate collection
            gc.collect()
            gc.collect()  # Second pass for circular references

    def get_stats(self) -> dict[str, Any]:
        """Get GC statistics.

        Returns:
            Dict with GC statistics
        """
        return {
            "collection_count": self._collection_count,
            "bytes_collected": self._bytes_collected,
            "aggressive_mode": self._aggressive,
            "gc_threshold": gc.get_threshold(),
            "gc_count": gc.get_count(),
        }


# Global GC controller
_gc_controller = GCController()


def gc_aggressive(enable: bool = True) -> None:
    """Enable/disable aggressive garbage collection.

    Args:
        enable: Whether to enable aggressive GC
    """
    _gc_controller.aggressive = enable


def force_gc() -> int:
    """Force garbage collection.

    Returns:
        Number of objects collected
    """
    return _gc_controller.collect()


# =============================================================================
# =============================================================================


class WSLSwapChecker:
    """WSL swap availability checker.

    Detects WSL environment and applies conservative memory
    estimates accounting for limited swap.

    References:
        MEM-023: WSL Swap Awareness
    """

    def __init__(self) -> None:
        """Initialize WSL checker."""
        self._is_wsl = self._detect_wsl()
        self._wslconfig_parsed = False
        self._wslconfig_memory: int | None = None
        self._wslconfig_swap: int | None = None

    def _detect_wsl(self) -> bool:
        """Detect if running in WSL."""
        try:
            with open("/proc/version") as f:
                version = f.read().lower()
                return "microsoft" in version or "wsl" in version
        except FileNotFoundError:
            return False

    @property
    def is_wsl(self) -> bool:
        """Check if running in WSL."""
        return self._is_wsl

    def get_wsl_memory_limit(self) -> int | None:
        """Get WSL memory limit from .wslconfig if available.

        Returns:
            Memory limit in bytes, or None if not configured
        """
        if not self._is_wsl:
            return None

        if self._wslconfig_parsed:
            return self._wslconfig_memory

        # Try to parse .wslconfig
        wslconfig_path = Path.home() / ".wslconfig"
        if wslconfig_path.exists():
            try:
                content = wslconfig_path.read_text()
                for line in content.split("\n"):
                    if line.strip().lower().startswith("memory="):
                        value = line.split("=")[1].strip()
                        self._wslconfig_memory = self._parse_size(value)
                    elif line.strip().lower().startswith("swap="):
                        value = line.split("=")[1].strip()
                        self._wslconfig_swap = self._parse_size(value)
            except Exception as e:
                logger.warning(f"Failed to parse .wslconfig: {e}")

        self._wslconfig_parsed = True
        return self._wslconfig_memory

    def get_wsl_swap_limit(self) -> int | None:
        """Get WSL swap limit.

        Returns:
            Swap limit in bytes, or None if not configured
        """
        if not self._is_wsl:
            return None

        if not self._wslconfig_parsed:
            self.get_wsl_memory_limit()

        return self._wslconfig_swap

    def _parse_size(self, size_str: str) -> int:
        """Parse size string like '8GB' to bytes."""
        size_str = size_str.strip().upper()
        multipliers = {
            "K": 1024,
            "KB": 1024,
            "M": 1024**2,
            "MB": 1024**2,
            "G": 1024**3,
            "GB": 1024**3,
            "T": 1024**4,
            "TB": 1024**4,
        }

        for suffix, mult in multipliers.items():
            if size_str.endswith(suffix):
                num = float(size_str[: -len(suffix)])
                return int(num * mult)

        return int(size_str)

    def get_safe_memory(self) -> int:
        """Get safe memory limit for WSL.

        Returns:
            Recommended maximum memory to use
        """
        if not self._is_wsl:
            import psutil

            return int(psutil.virtual_memory().available * 0.8)

        # WSL has minimal swap by default - use physical RAM only
        import psutil

        total = psutil.virtual_memory().total

        # Check .wslconfig limit
        wsl_limit = self.get_wsl_memory_limit()
        if wsl_limit is not None:
            total = min(total, wsl_limit)

        # Use 50% of available for safety (WSL can be unpredictable)
        available = psutil.virtual_memory().available
        return min(int(available * 0.5), int(total * 0.5))


def get_wsl_memory_limits() -> dict[str, int | None]:
    """Get WSL memory limits.

    Returns:
        Dict with memory and swap limits
    """
    checker = WSLSwapChecker()
    return {
        "is_wsl": checker.is_wsl,
        "memory_limit": checker.get_wsl_memory_limit(),
        "swap_limit": checker.get_wsl_swap_limit(),
        "safe_memory": checker.get_safe_memory(),
    }


# =============================================================================
# =============================================================================


@dataclass
class MemoryLogEntry:
    """Single memory log entry.

    Attributes:
        timestamp: Entry timestamp
        operation: Operation name
        memory_used: Memory used in bytes
        memory_peak: Peak memory
        duration: Operation duration in seconds
    """

    timestamp: float
    operation: str
    memory_used: int
    memory_peak: int
    duration: float


class MemoryLogger:
    """Memory usage logger for debugging.

    Logs memory usage at each operation for analysis.

    References:
        MEM-025: Memory Usage Logging
    """

    def __init__(
        self,
        log_file: str | Path | None = None,
        format: str = "csv",
    ) -> None:
        """Initialize memory logger.

        Args:
            log_file: Output file path
            format: Output format ('csv' or 'json')
        """
        self._log_file = Path(log_file) if log_file else None
        self._format = format
        self._entries: list[MemoryLogEntry] = []
        self._enabled = False
        self._peak_memory = 0
        self._start_memory = 0
        self._lock = threading.Lock()

    def enable(self) -> None:
        """Enable memory logging."""
        self._enabled = True
        import psutil

        process = psutil.Process()
        self._start_memory = process.memory_info().rss
        self._peak_memory = self._start_memory
        logger.info("Memory logging enabled")

    def disable(self) -> None:
        """Disable memory logging."""
        self._enabled = False
        self.flush()

    def log_operation(
        self,
        operation: str,
        duration: float = 0.0,
    ) -> None:
        """Log memory for an operation.

        Args:
            operation: Operation name
            duration: Operation duration
        """
        if not self._enabled:
            return

        import psutil

        process = psutil.Process()
        memory_used = process.memory_info().rss
        self._peak_memory = max(self._peak_memory, memory_used)

        entry = MemoryLogEntry(
            timestamp=time.time(),
            operation=operation,
            memory_used=memory_used,
            memory_peak=self._peak_memory,
            duration=duration,
        )

        with self._lock:
            self._entries.append(entry)

    def flush(self) -> None:
        """Write log to file."""
        if self._log_file is None or not self._entries:
            return

        with self._lock:
            entries = self._entries.copy()
            self._entries.clear()

        if self._format == "csv":
            self._write_csv(entries)
        else:
            self._write_json(entries)

    def _write_csv(self, entries: list[MemoryLogEntry]) -> None:
        """Write entries as CSV."""
        import csv

        assert self._log_file is not None
        mode = "a" if self._log_file.exists() else "w"
        with open(self._log_file, mode, newline="") as f:
            writer = csv.writer(f)
            if mode == "w":
                writer.writerow(
                    ["timestamp", "operation", "memory_used", "memory_peak", "duration"]
                )
            for entry in entries:
                writer.writerow(
                    [
                        entry.timestamp,
                        entry.operation,
                        entry.memory_used,
                        entry.memory_peak,
                        entry.duration,
                    ]
                )

    def _write_json(self, entries: list[MemoryLogEntry]) -> None:
        """Write entries as JSON."""
        assert self._log_file is not None
        data = [
            {
                "timestamp": e.timestamp,
                "operation": e.operation,
                "memory_used": e.memory_used,
                "memory_peak": e.memory_peak,
                "duration": e.duration,
            }
            for e in entries
        ]

        with open(self._log_file, "a") as f:
            f.writelines(json.dumps(entry) + "\n" for entry in data)

    def get_summary(self) -> dict[str, Any]:
        """Get logging summary.

        Returns:
            Summary statistics
        """
        return {
            "entries_logged": len(self._entries),
            "peak_memory_bytes": self._peak_memory,
            "start_memory_bytes": self._start_memory,
            "memory_growth_bytes": self._peak_memory - self._start_memory,
            "enabled": self._enabled,
        }


# =============================================================================
# =============================================================================


class AdaptiveMeasurementSelector:
    """Adaptive measurement selection for large files.

    Disables memory-intensive measurements for very large files
    and suggests alternatives.

    References:
        MEM-028: Adaptive Measurement Selection
    """

    # Default size thresholds (in samples)
    THRESHOLDS = {
        "eye_diagram": 1e8,  # 100M samples
        "spectrogram": 5e8,  # 500M samples
        "full_correlation": 1e9,  # 1B samples
        "wavelet": 2e8,  # 200M samples
    }

    def __init__(
        self,
        file_size_samples: int,
        enable_all: bool = False,
    ) -> None:
        """Initialize selector.

        Args:
            file_size_samples: Number of samples in file
            enable_all: Override to enable all measurements
        """
        self._size = file_size_samples
        self._enable_all = enable_all

    def is_enabled(self, measurement: str) -> bool:
        """Check if measurement is enabled for current file size.

        Args:
            measurement: Measurement name

        Returns:
            True if measurement should be enabled
        """
        if self._enable_all:
            return True

        threshold = self.THRESHOLDS.get(measurement, float("inf"))
        return self._size < threshold

    def get_recommendations(self) -> dict[str, str]:
        """Get recommendations for disabled measurements.

        Returns:
            Dict mapping disabled measurement to recommendation
        """
        recommendations = {}

        for measurement, threshold in self.THRESHOLDS.items():
            if self._size >= threshold:
                size_gb = self._size * 8 / 1e9  # Assume float64
                if measurement == "eye_diagram":
                    recommendations[measurement] = (
                        f"File size ({size_gb:.1f} GB) exceeds threshold. "
                        f"Use --roi START:END to specify time range, or "
                        f"--enable-all to force processing."
                    )
                elif measurement == "spectrogram":
                    recommendations[measurement] = (
                        "File too large for full spectrogram. "
                        "Use chunked_spectrogram() or downsample data."
                    )
                elif measurement == "full_correlation":
                    recommendations[measurement] = (
                        f"Correlation on {size_gb:.1f} GB requires chunked approach. "
                        f"Use correlate_chunked() instead."
                    )

        return recommendations


def adaptive_measurements(
    samples: int,
    enable_all: bool = False,
) -> AdaptiveMeasurementSelector:
    """Create adaptive measurement selector.

    Args:
        samples: Number of samples
        enable_all: Override to enable all

    Returns:
        Selector instance
    """
    return AdaptiveMeasurementSelector(samples, enable_all)


# =============================================================================
# =============================================================================


T = TypeVar("T")


@dataclass
class CacheEntry(Generic[T]):
    """Cache entry with metadata.

    Attributes:
        key: Cache key
        value: Cached value
        created_at: Creation timestamp
        accessed_at: Last access timestamp
        source_hash: Hash of source data
        params_hash: Hash of parameters
        ttl_seconds: Time-to-live in seconds

    References:
        MEM-030: Cache Invalidation Strategy
    """

    key: str
    value: T
    created_at: float
    accessed_at: float
    source_hash: str
    params_hash: str
    ttl_seconds: float = 3600.0  # 1 hour default

    @property
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.ttl_seconds <= 0:
            return False
        return time.time() - self.created_at > self.ttl_seconds

    @property
    def age_seconds(self) -> float:
        """Get entry age in seconds."""
        return time.time() - self.created_at


class CacheInvalidationStrategy:
    """Cache invalidation strategy manager.

    Manages cache invalidation based on data changes,
    parameter changes, and time-to-live.

    References:
        MEM-030: Cache Invalidation Strategy
    """

    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: float = 3600.0,
    ) -> None:
        """Initialize cache.

        Args:
            max_size: Maximum entries
            default_ttl: Default TTL in seconds
        """
        self._cache: OrderedDict[str, CacheEntry[Any]] = OrderedDict()
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def _compute_hash(self, data: Any) -> str:
        """Compute hash of data for comparison.

        Note:
            Uses MD5 for cache invalidation checksums only (not for security).
            MD5 is appropriate here for non-cryptographic data comparison.
        """
        if isinstance(data, np.ndarray):
            # Sample first 1KB for performance (cache invalidation only, not security)
            return hashlib.md5(data.tobytes()[:1024], usedforsecurity=False).hexdigest()
        elif isinstance(data, dict | list):
            return hashlib.md5(
                json.dumps(data, sort_keys=True).encode(), usedforsecurity=False
            ).hexdigest()
        else:
            return hashlib.md5(str(data).encode(), usedforsecurity=False).hexdigest()

    def get(
        self,
        key: str,
        source_data: Any = None,
        params: dict[str, Any] | None = None,
    ) -> tuple[Any, bool]:
        """Get value from cache.

        Args:
            key: Cache key
            source_data: Source data to validate against
            params: Parameters to validate against

        Returns:
            (value, hit) tuple
        """
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None, False

            entry = self._cache[key]

            # Check expiration
            if entry.is_expired:
                del self._cache[key]
                self._misses += 1
                return None, False

            # Check source data change
            if source_data is not None:
                source_hash = self._compute_hash(source_data)
                if source_hash != entry.source_hash:
                    del self._cache[key]
                    self._misses += 1
                    return None, False

            # Check parameter change
            if params is not None:
                params_hash = self._compute_hash(params)
                if params_hash != entry.params_hash:
                    del self._cache[key]
                    self._misses += 1
                    return None, False

            # Update access time and move to end
            entry.accessed_at = time.time()
            self._cache.move_to_end(key)
            self._hits += 1
            return entry.value, True

    def set(
        self,
        key: str,
        value: Any,
        source_data: Any = None,
        params: dict[str, Any] | None = None,
        ttl: float | None = None,
    ) -> None:
        """Set cache value.

        Args:
            key: Cache key
            value: Value to cache
            source_data: Source data for invalidation
            params: Parameters for invalidation
            ttl: Time-to-live (uses default if None)
        """
        with self._lock:
            # Evict if at capacity
            while len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)

            entry = CacheEntry(
                key=key,
                value=value,
                created_at=time.time(),
                accessed_at=time.time(),
                source_hash=self._compute_hash(source_data) if source_data is not None else "",
                params_hash=self._compute_hash(params) if params is not None else "",
                ttl_seconds=ttl if ttl is not None else self._default_ttl,
            )
            self._cache[key] = entry

    def invalidate(self, key: str) -> bool:
        """Invalidate specific cache entry.

        Args:
            key: Key to invalidate

        Returns:
            True if entry existed
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def invalidate_by_source(self, source_data: Any) -> int:
        """Invalidate all entries with matching source.

        Args:
            source_data: Source data to match

        Returns:
            Number of entries invalidated
        """
        source_hash = self._compute_hash(source_data)
        count = 0
        with self._lock:
            keys_to_remove = [k for k, v in self._cache.items() if v.source_hash == source_hash]
            for key in keys_to_remove:
                del self._cache[key]
                count += 1
        return count

    def clear(self) -> int:
        """Clear all cache entries.

        Returns:
            Number of entries cleared
        """
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            return count

    def cleanup_expired(self) -> int:
        """Remove expired entries.

        Returns:
            Number of entries removed
        """
        count = 0
        with self._lock:
            keys_to_remove = [k for k, v in self._cache.items() if v.is_expired]
            for key in keys_to_remove:
                del self._cache[key]
                count += 1
        return count

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Cache statistics
        """
        with self._lock:
            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self._hits / (self._hits + self._misses)
                if (self._hits + self._misses) > 0
                else 0,
                "default_ttl": self._default_ttl,
            }


# =============================================================================
# =============================================================================


class DiskCache:
    """Disk-based cache for large intermediates.

    Spills cache to disk when memory limit exceeded.

    References:
        MEM-031: Persistent Cache (Disk-Based)
    """

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        max_memory_mb: int = 1024,
        max_disk_mb: int = 10240,
        ttl_hours: float = 1.0,
    ) -> None:
        """Initialize disk cache.

        Args:
            cache_dir: Cache directory (default: temp dir)
            max_memory_mb: Max in-memory cache size in MB
            max_disk_mb: Max on-disk cache size in MB
            ttl_hours: Time-to-live in hours
        """
        self._cache_dir = (
            Path(cache_dir) if cache_dir else Path(tempfile.gettempdir()) / "oscura_cache"
        )
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._max_memory = max_memory_mb * 1024 * 1024
        self._max_disk = max_disk_mb * 1024 * 1024
        self._ttl_seconds = ttl_hours * 3600
        self._memory_cache: OrderedDict[str, tuple[Any, int]] = OrderedDict()
        self._memory_used = 0
        self._lock = threading.Lock()

        # Security: HMAC signing key for cache integrity (SEC-003 fix)
        self._cache_key = self._load_or_create_cache_key()

    def _get_cache_path(self, key: str) -> Path:
        """Get cache file path for key."""
        key_hash = hashlib.sha256(key.encode()).hexdigest()[:16]
        return self._cache_dir / f"{key_hash}.cache"

    def _estimate_size(self, value: Any) -> int:
        """Estimate memory size of value."""
        if isinstance(value, np.ndarray):
            return value.nbytes
        else:
            return len(pickle.dumps(value))

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
        key_file = self._cache_dir / ".cache_key"

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

    def get(self, key: str) -> tuple[Any, bool]:
        """Get value from cache.

        Args:
            key: Cache key

        Returns:
            (value, hit) tuple
        """
        # Check memory cache first
        with self._lock:
            if key in self._memory_cache:
                value, size = self._memory_cache[key]
                self._memory_cache.move_to_end(key)
                return value, True

        # Check disk cache
        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            # Check TTL
            if time.time() - cache_path.stat().st_mtime > self._ttl_seconds:
                cache_path.unlink()
                return None, False

            try:
                with open(cache_path, "rb") as f:
                    signature = f.read(32)  # SHA256 = 32 bytes
                    data = f.read()

                # Verify HMAC signature (SEC-003 fix)
                expected_signature = hmac.new(self._cache_key, data, hashlib.sha256).digest()

                if not hmac.compare_digest(signature, expected_signature):
                    logger.error(f"Cache integrity check failed for {key}")
                    # Delete corrupted cache file
                    cache_path.unlink()
                    raise SecurityError(
                        f"Cache file integrity verification failed: {key}. "
                        "File may have been tampered with and has been removed."
                    )

                # Deserialize only after HMAC verification
                value = pickle.loads(data)

                # Promote to memory cache if space
                size = self._estimate_size(value)
                if size < self._max_memory:
                    self._add_to_memory(key, value, size)

                return value, True

            except SecurityError:
                raise  # Re-raise security errors
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
                return None, False

        return None, False

    def _add_to_memory(self, key: str, value: Any, size: int) -> None:
        """Add to memory cache, evicting if needed."""
        with self._lock:
            # Evict until we have space
            while self._memory_used + size > self._max_memory and self._memory_cache:
                evict_key, (evict_value, evict_size) = self._memory_cache.popitem(last=False)
                self._memory_used -= evict_size
                # Spill to disk
                self._write_to_disk(evict_key, evict_value)

            self._memory_cache[key] = (value, size)
            self._memory_used += size

    def _write_to_disk(self, key: str, value: Any) -> None:
        """Write value to disk cache with HMAC signature.

        Security:
            SEC-003 fix: Writes HMAC-SHA256 signature + pickled data.
            Format: [32 bytes signature][pickled data]
            Signature computed over pickled data using self._cache_key.
        """
        # Check disk space
        self._cleanup_disk()

        cache_path = self._get_cache_path(key)
        try:
            # Serialize value
            data = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)

            # Compute HMAC-SHA256 signature
            signature = hmac.new(self._cache_key, data, hashlib.sha256).digest()

            # Write signature + data
            with open(cache_path, "wb") as f:
                f.write(signature)  # First 32 bytes
                f.write(data)  # Rest is pickled data

        except Exception as e:
            logger.warning(f"Failed to write cache: {e}")

    def _cleanup_disk(self) -> None:
        """Clean up disk cache if over limit."""
        try:
            total_size = sum(f.stat().st_size for f in self._cache_dir.glob("*.cache"))
            if total_size > self._max_disk:
                # Remove oldest files
                files = sorted(self._cache_dir.glob("*.cache"), key=lambda f: f.stat().st_mtime)
                for f in files:
                    if total_size <= self._max_disk * 0.8:
                        break
                    total_size -= f.stat().st_size
                    f.unlink()
        except Exception as e:
            logger.warning(f"Disk cleanup error: {e}")

    def set(self, key: str, value: Any) -> None:
        """Set cache value.

        Args:
            key: Cache key
            value: Value to cache
        """
        size = self._estimate_size(value)
        if size < self._max_memory:
            self._add_to_memory(key, value, size)
        else:
            # Too large for memory, write directly to disk
            self._write_to_disk(key, value)

    def clear(self) -> None:
        """Clear all caches."""
        with self._lock:
            self._memory_cache.clear()
            self._memory_used = 0

        for f in self._cache_dir.glob("*.cache"):
            with contextlib.suppress(Exception):
                f.unlink()


# =============================================================================
# =============================================================================


class BackpressureController:
    """Backpressure controller for streaming.

    Manages slow consumer handling for real-time streaming.

    References:
        MEM-032: Backpressure for Streaming
    """

    def __init__(
        self,
        buffer_size: int = 10000,
        drop_oldest: bool = True,
        warn_threshold: float = 0.8,
    ) -> None:
        """Initialize backpressure controller.

        Args:
            buffer_size: Maximum buffer size
            drop_oldest: Drop oldest on overflow (vs pause)
            warn_threshold: Warning threshold (0-1)
        """
        self._buffer: list[Any] = []
        self._buffer_size = buffer_size
        self._drop_oldest = drop_oldest
        self._warn_threshold = warn_threshold
        self._dropped_count = 0
        self._paused = False
        self._lock = threading.Lock()

    @property
    def is_paused(self) -> bool:
        """Check if acquisition should be paused."""
        return self._paused

    @property
    def buffer_usage(self) -> float:
        """Get buffer usage ratio (0-1)."""
        return len(self._buffer) / self._buffer_size

    @property
    def dropped_count(self) -> int:
        """Get number of dropped samples."""
        return self._dropped_count

    def push(self, data: Any) -> bool:
        """Push data to buffer.

        Args:
            data: Data to buffer

        Returns:
            True if accepted, False if dropped
        """
        with self._lock:
            if len(self._buffer) >= self._buffer_size:
                if self._drop_oldest:
                    self._buffer.pop(0)
                    self._dropped_count += 1
                    logger.warning(
                        f"Buffer overflow: dropped frame ({self._dropped_count} total dropped)"
                    )
                else:
                    self._paused = True
                    return False

            self._buffer.append(data)

            # Check warning threshold
            if self.buffer_usage > self._warn_threshold:
                logger.warning(
                    f"Buffer usage at {self.buffer_usage * 100:.0f}% - "
                    f"analysis slower than acquisition"
                )

            return True

    def pop(self) -> Any | None:
        """Pop data from buffer.

        Returns:
            Data or None if empty
        """
        with self._lock:
            if not self._buffer:
                return None

            data = self._buffer.pop(0)
            self._paused = False
            return data

    def pop_all(self) -> list[Any]:
        """Pop all data from buffer.

        Returns:
            List of all buffered data
        """
        with self._lock:
            data = self._buffer.copy()
            self._buffer.clear()
            self._paused = False
            return data

    def signal_backpressure(self) -> None:
        """Signal backpressure to source."""
        with self._lock:
            self._paused = True

    def get_stats(self) -> dict[str, Any]:
        """Get backpressure statistics.

        Returns:
            Statistics dict
        """
        return {
            "buffer_size": len(self._buffer),
            "buffer_capacity": self._buffer_size,
            "usage_ratio": self.buffer_usage,
            "dropped_count": self._dropped_count,
            "is_paused": self._paused,
        }


# =============================================================================
# =============================================================================


class MultiChannelMemoryManager:
    """Multi-channel memory management.

    Manages memory for multi-channel processing, enabling
    sequential or subset processing for bounded memory.

    References:
        MEM-033: Multi-Channel Memory Management
    """

    def __init__(
        self,
        max_memory_mb: int = 4096,
        bytes_per_sample: int = 8,
    ) -> None:
        """Initialize manager.

        Args:
            max_memory_mb: Maximum memory in MB
            bytes_per_sample: Bytes per sample (default 8 for float64)
        """
        self._max_memory = max_memory_mb * 1024 * 1024
        self._bytes_per_sample = bytes_per_sample

    def estimate_channel_memory(
        self,
        samples_per_channel: int,
        num_channels: int,
    ) -> int:
        """Estimate memory for channels.

        Args:
            samples_per_channel: Samples per channel
            num_channels: Number of channels

        Returns:
            Estimated memory in bytes
        """
        return samples_per_channel * num_channels * self._bytes_per_sample

    def can_load_all(
        self,
        samples_per_channel: int,
        num_channels: int,
    ) -> bool:
        """Check if all channels can be loaded at once.

        Args:
            samples_per_channel: Samples per channel
            num_channels: Number of channels

        Returns:
            True if all can be loaded
        """
        required = self.estimate_channel_memory(samples_per_channel, num_channels)
        return required < self._max_memory

    def get_channel_batches(
        self,
        samples_per_channel: int,
        channel_indices: list[int],
    ) -> list[list[int]]:
        """Get channel batches for sequential processing.

        Args:
            samples_per_channel: Samples per channel
            channel_indices: Channel indices to process

        Returns:
            List of channel batches
        """
        memory_per_channel = samples_per_channel * self._bytes_per_sample

        # How many channels can we load at once?
        channels_at_once = max(1, int(self._max_memory / memory_per_channel))

        # Create batches
        batches = []
        for i in range(0, len(channel_indices), channels_at_once):
            batches.append(channel_indices[i : i + channels_at_once])

        return batches

    def suggest_subset(
        self,
        samples_per_channel: int,
        total_channels: int,
    ) -> dict[str, Any]:
        """Suggest channel subset for memory-bounded analysis.

        Args:
            samples_per_channel: Samples per channel
            total_channels: Total available channels

        Returns:
            Suggestion dict
        """
        required = self.estimate_channel_memory(samples_per_channel, total_channels)

        if required < self._max_memory:
            return {
                "can_load_all": True,
                "suggested_channels": list(range(total_channels)),
                "memory_required_gb": required / 1e9,
            }

        # Calculate how many channels we can handle
        memory_per_channel = samples_per_channel * self._bytes_per_sample
        max_channels = max(1, int(self._max_memory / memory_per_channel))

        return {
            "can_load_all": False,
            "max_channels_at_once": max_channels,
            "suggested_channels": list(range(min(max_channels, total_channels))),
            "memory_required_gb": required / 1e9,
            "memory_limit_gb": self._max_memory / 1e9,
            "recommendation": (
                f"Process channels in batches of {max_channels}, "
                f"or use --channels 0,1,2,... to select subset"
            ),
        }

    def iterate_channels(
        self,
        samples_per_channel: int,
        channel_indices: list[int],
    ) -> Iterator[list[int]]:
        """Iterate over channel batches.

        Args:
            samples_per_channel: Samples per channel
            channel_indices: Channels to process

        Yields:
            Channel index batches
        """
        batches = self.get_channel_batches(samples_per_channel, channel_indices)
        yield from batches
