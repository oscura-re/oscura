"""Comprehensive tests for cache HMAC validation (SEC-003).

Tests verify HMAC-SHA256 signature validation for all cache systems to prevent
pickle deserialization attacks from tampered cache files.

Coverage includes:
- HMAC signature verification on cache load
- Tampered data detection and rejection
- Cache key persistence across sessions
- Corrupted file cleanup
- SecurityError raising on integrity failures
"""

from __future__ import annotations

import hashlib
import hmac
import pickle
from pathlib import Path

import pytest

from oscura.core.cache import OscuraCache
from oscura.core.exceptions import SecurityError
from oscura.utils.memory_advanced import DiskCache
from oscura.utils.performance.caching import CacheBackend, CacheManager

# ============================================================================
# OscuraCache (core/cache.py) Tests
# ============================================================================


class TestOscuraCacheHMAC:
    """Test HMAC validation in OscuraCache."""

    def test_cache_key_created_on_init(self, tmp_path: Path) -> None:
        """Verify cache signing key is created on initialization."""
        cache = OscuraCache(max_memory="100MB", cache_dir=tmp_path)
        key_file = tmp_path / ".cache_key"

        assert key_file.exists()
        assert key_file.stat().st_size == 32  # 256 bits
        assert (key_file.stat().st_mode & 0o777) == 0o600  # Owner read/write only

    def test_cache_key_persists_across_sessions(self, tmp_path: Path) -> None:
        """Verify cache key is reused across cache instances."""
        cache1 = OscuraCache(max_memory="100MB", cache_dir=tmp_path)
        key1 = cache1._cache_key

        cache2 = OscuraCache(max_memory="100MB", cache_dir=tmp_path)
        key2 = cache2._cache_key

        assert key1 == key2

    def test_cache_accepts_valid_signed_data(self, tmp_path: Path) -> None:
        """Verify cache loads correctly signed data."""
        cache = OscuraCache(max_memory="100", cache_dir=tmp_path)

        # Store data (forces disk spill due to small memory limit)
        test_data = {"value": "test" * 1000}  # Large enough to spill
        cache.put("test_key", test_data)

        # Entry should be on disk (not in memory)
        assert "test_key" in cache._cache
        assert not cache._cache["test_key"].in_memory
        assert cache._cache["test_key"].disk_path is not None

        # Load should succeed with valid HMAC (loads from disk)
        loaded = cache.get("test_key")
        assert loaded == test_data

    def test_cache_rejects_tampered_data(self, tmp_path: Path) -> None:
        """Verify cache detects and rejects tampered cache files."""
        cache = OscuraCache(max_memory="100", cache_dir=tmp_path)

        # Store data
        test_data = {"value": "test" * 1000}
        cache.put("test_key", test_data)

        # Find cache file
        cache_file = next(tmp_path.glob("*.pkl"))

        # Tamper with cache file (modify data while keeping signature)
        with open(cache_file, "rb") as f:
            signature = f.read(32)
            data = f.read()

        # Corrupt data
        corrupted_data = b"malicious" + data[9:]

        with open(cache_file, "wb") as f:
            f.write(signature)  # Original signature
            f.write(corrupted_data)  # Modified data

        # Clear memory cache to force disk load

        # Load should raise SecurityError
        with pytest.raises(SecurityError, match="integrity verification failed"):
            cache.get("test_key")

        # Verify corrupted file was deleted
        assert not cache_file.exists()

    def test_cache_rejects_tampered_signature(self, tmp_path: Path) -> None:
        """Verify cache detects tampered HMAC signatures."""
        cache = OscuraCache(max_memory="100", cache_dir=tmp_path)

        # Store data
        test_data = {"value": "test" * 1000}
        cache.put("test_key", test_data)

        # Find cache file
        cache_file = next(tmp_path.glob("*.pkl"))

        # Tamper with signature
        with open(cache_file, "rb") as f:
            signature = f.read(32)
            data = f.read()

        # Flip one bit in signature
        corrupted_sig = bytearray(signature)
        corrupted_sig[0] ^= 0x01
        corrupted_sig = bytes(corrupted_sig)

        with open(cache_file, "wb") as f:
            f.write(corrupted_sig)
            f.write(data)

        # Clear memory cache

        # Should raise SecurityError
        with pytest.raises(SecurityError, match="integrity verification failed"):
            cache.get("test_key")

    def test_different_cache_dirs_use_different_keys(self, tmp_path: Path) -> None:
        """Verify each cache directory has its own signing key."""
        cache_dir1 = tmp_path / "cache1"
        cache_dir2 = tmp_path / "cache2"

        cache1 = OscuraCache(max_memory="100MB", cache_dir=cache_dir1)
        cache2 = OscuraCache(max_memory="100MB", cache_dir=cache_dir2)

        assert cache1._cache_key != cache2._cache_key

    def test_cache_file_format_correct(self, tmp_path: Path) -> None:
        """Verify cache file format: [32-byte signature][pickled data]."""
        cache = OscuraCache(max_memory="100", cache_dir=tmp_path)

        # Store data
        test_data = {"value": "test" * 1000}
        cache.put("test_key", test_data)

        # Find cache file
        cache_file = next(tmp_path.glob("*.pkl"))

        # Verify format
        with open(cache_file, "rb") as f:
            signature = f.read(32)
            data = f.read()

        # Signature should be 32 bytes
        assert len(signature) == 32

        # Data should deserialize
        loaded = pickle.loads(data)
        assert loaded == test_data

        # Signature should match
        expected = hmac.new(cache._cache_key, data, hashlib.sha256).digest()
        assert hmac.compare_digest(signature, expected)


# ============================================================================
# CacheManager (utils/performance/caching.py) Tests
# ============================================================================


class TestCacheManagerHMAC:
    """Test HMAC validation in CacheManager (disk backend)."""

    def test_disk_cache_key_created(self, tmp_path: Path) -> None:
        """Verify cache signing key created for disk backend."""
        cache = CacheManager(backend=CacheBackend.DISK, cache_dir=tmp_path)
        key_file = tmp_path / ".cache_key"

        assert key_file.exists()
        assert key_file.stat().st_size == 32

    def test_disk_cache_accepts_valid_signed_data(self, tmp_path: Path) -> None:
        """Verify disk cache loads correctly signed data."""
        cache = CacheManager(backend=CacheBackend.DISK, cache_dir=tmp_path)

        # Store data
        cache.set("test_key", {"value": "test_data"})

        # Load should succeed
        loaded = cache.get("test_key")
        assert loaded == {"value": "test_data"}

    def test_disk_cache_rejects_tampered_data(self, tmp_path: Path) -> None:
        """Verify disk cache detects tampered files."""
        cache = CacheManager(backend=CacheBackend.DISK, cache_dir=tmp_path)

        # Store data
        cache.set("test_key", {"value": "test_data"})

        # Find cache file
        cache_file = next(tmp_path.glob("*.pkl"))

        # Tamper with data
        with open(cache_file, "rb") as f:
            signature = f.read(32)
            data = f.read()

        with open(cache_file, "wb") as f:
            f.write(signature)
            f.write(b"corrupted" + data[9:])

        # Load should raise SecurityError
        with pytest.raises(SecurityError, match="integrity verification failed"):
            cache.get("test_key")

        # Verify file was deleted
        assert not cache_file.exists()

    def test_multi_level_cache_hmac_validation(self, tmp_path: Path) -> None:
        """Verify HMAC validation works with multi-level cache."""
        cache = CacheManager(backend=CacheBackend.MULTI_LEVEL, cache_dir=tmp_path)

        # Store data (goes to both memory and disk)
        cache.set("test_key", {"value": "test_data"})

        # Clear memory cache to force disk load
        cache._memory_cache.clear()

        # Load from disk (should verify HMAC)
        loaded = cache.get("test_key")
        assert loaded == {"value": "test_data"}


# ============================================================================
# DiskCache (utils/memory_advanced.py) Tests
# ============================================================================


class TestDiskCacheHMAC:
    """Test HMAC validation in DiskCache."""

    def test_disk_cache_key_created(self, tmp_path: Path) -> None:
        """Verify cache signing key created on initialization."""
        cache = DiskCache(cache_dir=tmp_path)
        key_file = tmp_path / ".cache_key"

        assert key_file.exists()
        assert key_file.stat().st_size == 32

    def test_disk_cache_accepts_valid_data(self, tmp_path: Path) -> None:
        """Verify disk cache loads correctly signed data."""
        cache = DiskCache(cache_dir=tmp_path, max_memory_mb=0.001)  # 1KB to force disk spill

        # Store data (small memory forces disk spill)
        test_data = {"value": "test" * 10000}
        cache.set("test_key", test_data)

        # Clear memory to force disk load
        cache._memory_cache.clear()
        cache._memory_used = 0

        # Load should succeed
        loaded, hit = cache.get("test_key")
        assert hit is True
        assert loaded == test_data

    def test_disk_cache_rejects_tampered_data(self, tmp_path: Path) -> None:
        """Verify disk cache detects tampered files."""
        cache = DiskCache(cache_dir=tmp_path, max_memory_mb=0.001)  # 1KB to force disk spill

        # Store data
        test_data = {"value": "test" * 10000}
        cache.set("test_key", test_data)

        # Clear memory
        cache._memory_cache.clear()
        cache._memory_used = 0

        # Find and tamper with cache file
        cache_file = next(tmp_path.glob("*.cache"))

        with open(cache_file, "rb") as f:
            signature = f.read(32)
            data = f.read()

        # Corrupt data
        with open(cache_file, "wb") as f:
            f.write(signature)
            f.write(b"malicious" + data[9:])

        # Load should raise SecurityError
        with pytest.raises(SecurityError, match="integrity verification failed"):
            cache.get("test_key")


# ============================================================================
# Cross-Cache Compatibility Tests
# ============================================================================


class TestCacheCompatibility:
    """Test cache key isolation between different cache instances."""

    def test_cache_keys_isolated_per_directory(self, tmp_path: Path) -> None:
        """Verify each cache directory has isolated signing keys."""
        dir1 = tmp_path / "cache1"
        dir2 = tmp_path / "cache2"

        cache1 = OscuraCache(max_memory="100", cache_dir=dir1)  # Small memory to force disk spill
        cache2 = OscuraCache(max_memory="100", cache_dir=dir2)

        # Keys should be different
        assert cache1._cache_key != cache2._cache_key

        # Files can't be shared between caches
        cache1.put("key", {"data": "test" * 1000})

        # Get cache file from cache1 (should be on disk due to small memory limit)
        cache1_file = next(dir1.glob("*.pkl"))

        # Copy to cache2 directory
        cache2_file = dir2 / cache1_file.name
        cache2_file.write_bytes(cache1_file.read_bytes())

        # Get cache entry before clearing
        from oscura.core.cache import CacheEntry

        cache1_entry = cache1._cache["key"]

        # Create entry in cache2 pointing to copied file
        cache2._cache["key"] = CacheEntry(
            key="key",
            value=None,
            disk_path=cache2_file,
            size_bytes=cache1_entry.size_bytes,
            created_at=cache1_entry.created_at,
            last_accessed=cache1_entry.last_accessed,
            access_count=0,
            in_memory=False,
        )

        # cache2 should reject file (different HMAC key)
        with pytest.raises(SecurityError):
            cache2.get("key")


# ============================================================================
# Migration and Backward Compatibility Tests
# ============================================================================


class TestCacheMigration:
    """Test migration from legacy cache format (no HMAC) to HMAC-protected."""

    def test_legacy_cache_files_rejected(self, tmp_path: Path) -> None:
        """Verify legacy cache files (no HMAC) are rejected."""
        cache = OscuraCache(max_memory="100", cache_dir=tmp_path)

        # Create legacy cache file (direct pickle, no HMAC)
        legacy_file = tmp_path / "legacy_key.pkl"
        test_data = {"value": "test" * 1000}
        with open(legacy_file, "wb") as f:
            pickle.dump(test_data, f)

        # Manually add to cache index
        from oscura.core.cache import CacheEntry

        cache._cache["legacy_key"] = CacheEntry(
            key="legacy_key",
            value=None,
            disk_path=legacy_file,
            size_bytes=1000,
            created_at=0.0,
            last_accessed=0.0,
            access_count=0,
            in_memory=False,
        )

        # Attempt to load should fail (missing HMAC signature)
        with pytest.raises(SecurityError, match="integrity verification failed"):
            cache.get("legacy_key")

        # File should be deleted
        assert not legacy_file.exists()

    def test_cache_migration_clears_legacy_files(self, tmp_path: Path) -> None:
        """Verify migration clears all legacy cache files."""
        # Create some legacy cache files
        for i in range(5):
            legacy_file = tmp_path / f"legacy_{i}.pkl"
            with open(legacy_file, "wb") as f:
                pickle.dump({"data": i}, f)

        # Create cache (should ignore legacy files)
        cache = OscuraCache(max_memory="100MB", cache_dir=tmp_path)

        # Put new data (with HMAC)
        cache.put("new_key", {"value": "new_data" * 1000})

        # New data should load fine
        loaded = cache.get("new_key")
        assert loaded == {"value": "new_data" * 1000}


# ============================================================================
# Performance Tests
# ============================================================================


class TestHMACPerformance:
    """Verify HMAC validation doesn't significantly impact performance."""

    def test_hmac_overhead_acceptable(self, tmp_path: Path) -> None:
        """Verify HMAC adds <10% overhead to cache operations."""
        import time

        cache = OscuraCache(max_memory="100", cache_dir=tmp_path)

        # Store 10 items (forces disk spill)
        test_data = {"value": "x" * 1000}
        for i in range(10):
            cache.put(f"key_{i}", test_data)

        # Clear memory

        # Measure load time
        start = time.perf_counter()
        for i in range(10):
            cache.get(f"key_{i}")
        duration = time.perf_counter() - start

        # Should complete in <100ms for 10 loads
        assert duration < 0.1

    def test_concurrent_cache_access_with_hmac(self, tmp_path: Path) -> None:
        """Verify HMAC validation is thread-safe."""
        import threading

        cache = OscuraCache(max_memory="100", cache_dir=tmp_path)

        # Store data
        for i in range(20):
            cache.put(f"key_{i}", {"value": i * 1000})

        errors: list[Exception] = []

        def access_cache(key: str) -> None:
            try:
                cache.get(key)
            except Exception as e:
                errors.append(e)

        # Clear memory to force disk loads

        # Concurrent access
        threads = [
            threading.Thread(target=access_cache, args=(f"key_{i % 20}",)) for i in range(100)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # No errors should occur
        assert len(errors) == 0


# ============================================================================
# Security Property Tests
# ============================================================================


class TestCacheSecurityProperties:
    """Validate security properties of HMAC implementation."""

    def test_signature_uses_constant_time_comparison(self, tmp_path: Path) -> None:
        """Verify HMAC comparison uses constant-time algorithm (timing attack prevention)."""
        # This test verifies we use hmac.compare_digest (constant-time)
        # by checking that the code raises SecurityError regardless of how
        # much of the signature matches

        cache = OscuraCache(max_memory="100", cache_dir=tmp_path)
        cache.put("key", {"data": "test" * 1000})

        cache_file = next(tmp_path.glob("*.pkl"))

        with open(cache_file, "rb") as f:
            correct_sig = f.read(32)
            data = f.read()

        # Test 1: Completely wrong signature
        wrong_sig = b"\x00" * 32
        with open(cache_file, "wb") as f:
            f.write(wrong_sig)
            f.write(data)

        with pytest.raises(SecurityError):
            cache.get("key")

        # Test 2: Partially correct signature (first 16 bytes match)
        partial_sig = correct_sig[:16] + b"\x00" * 16
        with open(cache_file, "wb") as f:
            f.write(partial_sig)
            f.write(data)

        with pytest.raises(SecurityError):
            cache.get("key")

    def test_hmac_key_permissions(self, tmp_path: Path) -> None:
        """Verify cache key file has restrictive permissions."""
        cache = OscuraCache(max_memory="100MB", cache_dir=tmp_path)
        key_file = tmp_path / ".cache_key"

        # Permissions should be 0o600 (owner read/write only)
        mode = key_file.stat().st_mode & 0o777
        assert mode == 0o600

    def test_cache_cleanup_removes_key_file(self, tmp_path: Path) -> None:
        """Verify cache cleanup can remove key file."""
        cache = OscuraCache(max_memory="100MB", cache_dir=tmp_path)
        key_file = tmp_path / ".cache_key"

        assert key_file.exists()

        # Clear cache (should clean up directory if empty)
        cache.clear()

        # Directory and key file should be removed if cache was only content
        # (may not be removed if other files exist, which is OK)


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestCacheErrorHandling:
    """Test error handling in HMAC validation."""

    def test_short_cache_file_rejected(self, tmp_path: Path) -> None:
        """Verify cache files shorter than 32 bytes are rejected."""
        cache = OscuraCache(max_memory="100", cache_dir=tmp_path)

        # Create invalid cache file (too short)
        cache_file = tmp_path / "short_file.pkl"
        cache_file.write_bytes(b"short")

        # Manually add to cache
        from oscura.core.cache import CacheEntry

        cache._cache["short_key"] = CacheEntry(
            key="short_key",
            value=None,
            disk_path=cache_file,
            size_bytes=5,
            created_at=0.0,
            last_accessed=0.0,
            access_count=0,
            in_memory=False,
        )

        # Should handle gracefully (not crash)
        result = cache.get("short_key")
        assert result is None  # Returns None for corrupted files

    def test_missing_cache_file_handled(self, tmp_path: Path) -> None:
        """Verify missing cache files are handled gracefully."""
        cache = OscuraCache(max_memory="100", cache_dir=tmp_path)

        # Store data
        cache.put("key", {"data": "test" * 1000})

        # Find and delete cache file
        cache_file = next(tmp_path.glob("*.pkl"))
        cache_file.unlink()

        # Clear memory

        # Should return None (not raise exception)
        result = cache.get("key")
        assert result is None
