"""Unit tests for caching layer.

Tests the comprehensive caching system including memory/disk backends,
TTL expiration, LRU eviction, and cache statistics.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from oscura.utils.performance.caching import (
    CacheBackend,
    CacheEntry,
    CacheManager,
    CachePolicy,
    CacheStats,
    EvictionPolicy,
    cache,
    get_global_cache,
)

pytestmark = pytest.mark.unit


class TestCacheBackend:
    """Test CacheBackend enum."""

    def test_cache_backends_exist(self) -> None:
        """Test all cache backends are defined."""
        assert CacheBackend.MEMORY.value == "memory"
        assert CacheBackend.DISK.value == "disk"
        assert CacheBackend.REDIS.value == "redis"
        assert CacheBackend.MULTI_LEVEL.value == "multi_level"

    def test_cache_backend_enum_members(self) -> None:
        """Test enum has expected members."""
        backends = list(CacheBackend)
        assert len(backends) == 4
        assert CacheBackend.MEMORY in backends
        assert CacheBackend.DISK in backends
        assert CacheBackend.REDIS in backends
        assert CacheBackend.MULTI_LEVEL in backends


class TestEvictionPolicy:
    """Test EvictionPolicy enum."""

    def test_eviction_policies_exist(self) -> None:
        """Test all eviction policies are defined."""
        assert EvictionPolicy.LRU.value == "lru"
        assert EvictionPolicy.LFU.value == "lfu"
        assert EvictionPolicy.FIFO.value == "fifo"
        assert EvictionPolicy.SIZE_BASED.value == "size_based"


class TestCacheEntry:
    """Test CacheEntry dataclass."""

    def test_cache_entry_creation(self) -> None:
        """Test creating CacheEntry instance."""
        entry = CacheEntry(
            key="test_key",
            value={"result": 42},
            timestamp=time.time(),
            ttl=3600.0,
        )

        assert entry.key == "test_key"
        assert entry.value == {"result": 42}
        assert entry.ttl == 3600.0
        assert entry.access_count == 0
        assert entry.size_bytes == 0

    def test_cache_entry_is_expired_no_ttl(self) -> None:
        """Test entry with no TTL never expires."""
        entry = CacheEntry(
            key="test_key",
            value="test_value",
            timestamp=time.time() - 10000,
            ttl=None,
        )

        assert not entry.is_expired()

    def test_cache_entry_is_expired_within_ttl(self) -> None:
        """Test entry within TTL is not expired."""
        entry = CacheEntry(
            key="test_key",
            value="test_value",
            timestamp=time.time(),
            ttl=3600.0,
        )

        assert not entry.is_expired()

    def test_cache_entry_is_expired_beyond_ttl(self) -> None:
        """Test entry beyond TTL is expired."""
        entry = CacheEntry(
            key="test_key",
            value="test_value",
            timestamp=time.time() - 7200,
            ttl=3600.0,
        )

        assert entry.is_expired()

    def test_cache_entry_touch(self) -> None:
        """Test touch updates access metadata."""
        entry = CacheEntry(
            key="test_key",
            value="test_value",
            timestamp=time.time(),
        )

        initial_count = entry.access_count
        initial_access = entry.last_access

        time.sleep(0.01)
        entry.touch()

        assert entry.access_count == initial_count + 1
        assert entry.last_access > initial_access


class TestCacheStats:
    """Test CacheStats dataclass."""

    def test_cache_stats_creation(self) -> None:
        """Test creating CacheStats instance."""
        stats = CacheStats(
            hits=100,
            misses=20,
            size_mb=15.5,
            entry_count=50,
        )

        assert stats.hits == 100
        assert stats.misses == 20
        assert stats.size_mb == 15.5
        assert stats.entry_count == 50
        assert stats.evictions == 0
        assert stats.expired == 0

    def test_cache_stats_hit_rate_calculation(self) -> None:
        """Test hit rate is calculated correctly."""
        stats = CacheStats(hits=80, misses=20)
        assert stats.hit_rate == 0.8  # 80 / 100

    def test_cache_stats_hit_rate_no_requests(self) -> None:
        """Test hit rate with no requests is 0."""
        stats = CacheStats(hits=0, misses=0)
        assert stats.hit_rate == 0.0

    def test_cache_stats_to_dict(self) -> None:
        """Test exporting stats to dictionary."""
        stats = CacheStats(hits=50, misses=10, size_mb=5.0, entry_count=20)
        stats_dict = stats.to_dict()

        assert stats_dict["hits"] == 50
        assert stats_dict["misses"] == 10
        assert stats_dict["size_mb"] == 5.0
        assert stats_dict["entry_count"] == 20
        assert stats_dict["hit_rate"] == pytest.approx(0.833, rel=0.01)

    def test_cache_stats_to_json(self, tmp_path: Path) -> None:
        """Test exporting stats to JSON file."""
        stats = CacheStats(hits=50, misses=10, size_mb=5.0, entry_count=20)
        json_file = tmp_path / "stats.json"

        stats.to_json(json_file)

        assert json_file.exists()
        with open(json_file) as f:
            data = json.load(f)
            assert data["hits"] == 50
            assert data["misses"] == 10


class TestCachePolicy:
    """Test CachePolicy dataclass."""

    def test_cache_policy_defaults(self) -> None:
        """Test default cache policy values."""
        policy = CachePolicy()

        assert policy.ttl == 3600.0
        assert policy.max_size_mb == 100.0
        assert policy.eviction == EvictionPolicy.LRU
        assert policy.serialize_numpy is True
        assert policy.compress is False
        assert policy.version == "1.0"

    def test_cache_policy_custom_values(self) -> None:
        """Test custom cache policy values."""
        policy = CachePolicy(
            ttl=1800.0,
            max_size_mb=50.0,
            eviction=EvictionPolicy.FIFO,
            version="2.0",
        )

        assert policy.ttl == 1800.0
        assert policy.max_size_mb == 50.0
        assert policy.eviction == EvictionPolicy.FIFO
        assert policy.version == "2.0"


class TestCacheManager:
    """Test CacheManager class."""

    def test_cache_manager_memory_backend(self, tmp_path: Path) -> None:
        """Test CacheManager with memory backend."""
        cache = CacheManager(backend="memory", cache_dir=tmp_path)

        assert cache.backend == CacheBackend.MEMORY
        assert cache.cache_dir == tmp_path

    def test_cache_manager_disk_backend(self, tmp_path: Path) -> None:
        """Test CacheManager with disk backend."""
        cache = CacheManager(backend="disk", cache_dir=tmp_path)

        assert cache.backend == CacheBackend.DISK
        assert cache.cache_dir == tmp_path

    def test_cache_manager_memory_set_get(self, tmp_path: Path) -> None:
        """Test setting and getting values from memory cache."""
        cache = CacheManager(backend="memory", cache_dir=tmp_path)

        cache.set("test_key", {"result": 42})
        value = cache.get("test_key")

        assert value == {"result": 42}

    def test_cache_manager_memory_get_miss(self, tmp_path: Path) -> None:
        """Test cache miss returns None."""
        cache = CacheManager(backend="memory", cache_dir=tmp_path)

        value = cache.get("nonexistent_key")

        assert value is None

    def test_cache_manager_disk_set_get(self, tmp_path: Path) -> None:
        """Test setting and getting values from disk cache."""
        cache = CacheManager(backend="disk", cache_dir=tmp_path)

        cache.set("test_key", {"result": 42})
        value = cache.get("test_key")

        assert value == {"result": 42}

        # Verify file was created
        cache_files = list(tmp_path.glob("*.pkl"))
        assert len(cache_files) > 0

    def test_cache_manager_ttl_expiration(self, tmp_path: Path) -> None:
        """Test TTL expiration removes entries."""
        cache = CacheManager(backend="memory", cache_dir=tmp_path)

        # Set with 0.05 second TTL
        cache.set("test_key", "test_value", ttl=0.05)

        # Should be available immediately
        assert cache.get("test_key") == "test_value"

        # Wait for expiration
        time.sleep(0.1)

        # Should be expired
        assert cache.get("test_key") is None

    def test_cache_manager_lru_eviction(self, tmp_path: Path) -> None:
        """Test LRU eviction when size limit reached."""
        # Small cache size to trigger eviction
        policy = CachePolicy(max_size_mb=0.001, eviction=EvictionPolicy.LRU)
        cache = CacheManager(backend="memory", cache_dir=tmp_path, policy=policy)

        # Add entries until eviction occurs
        for i in range(10):
            cache.set(f"key_{i}", {"data": "x" * 1000})

        stats = cache.get_stats()
        # Some entries should have been evicted
        assert stats.evictions > 0
        assert stats.entry_count < 10

    def test_cache_manager_invalidate_all(self, tmp_path: Path) -> None:
        """Test invalidating all cache entries."""
        cache = CacheManager(backend="memory", cache_dir=tmp_path)

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")

        invalidated = cache.invalidate()

        assert invalidated == 3
        assert cache.get("key1") is None
        assert cache.get("key2") is None
        assert cache.get("key3") is None

    def test_cache_manager_invalidate_pattern(self, tmp_path: Path) -> None:
        """Test invalidating cache entries by pattern."""
        cache = CacheManager(backend="memory", cache_dir=tmp_path)

        cache.set("test_key1", "value1")
        cache.set("test_key2", "value2")
        cache.set("other_key", "value3")

        invalidated = cache.invalidate(pattern="test_")

        assert invalidated == 2
        assert cache.get("test_key1") is None
        assert cache.get("test_key2") is None
        assert cache.get("other_key") == "value3"

    def test_cache_manager_get_stats(self, tmp_path: Path) -> None:
        """Test getting cache statistics."""
        cache = CacheManager(backend="memory", cache_dir=tmp_path)

        cache.set("key1", "value1")
        cache.set("key2", "value2")

        # Trigger hit
        cache.get("key1")

        # Trigger miss
        cache.get("nonexistent")

        stats = cache.get_stats()

        assert stats.hits == 1
        assert stats.misses == 1
        assert stats.entry_count == 2
        assert stats.hit_rate == 0.5
        assert stats.backend == "memory"

    def test_cache_manager_numpy_array_caching(self, tmp_path: Path) -> None:
        """Test caching numpy arrays."""
        cache = CacheManager(backend="memory", cache_dir=tmp_path)

        array = np.array([1, 2, 3, 4, 5])
        cache.set("numpy_key", array)

        retrieved = cache.get("numpy_key")

        assert isinstance(retrieved, np.ndarray)
        np.testing.assert_array_equal(retrieved, array)

    def test_cache_manager_hash_numpy_arrays(self, tmp_path: Path) -> None:
        """Test consistent hashing of numpy arrays."""
        cache = CacheManager(backend="memory", cache_dir=tmp_path)

        array1 = np.array([1, 2, 3])
        array2 = np.array([1, 2, 3])

        hash1 = cache._hash_value(array1)
        hash2 = cache._hash_value(array2)

        # Same data should produce same hash
        assert hash1 == hash2

    def test_cache_manager_hash_different_types(self, tmp_path: Path) -> None:
        """Test hashing different data types."""
        cache = CacheManager(backend="memory", cache_dir=tmp_path)

        # All should produce valid hashes
        assert len(cache._hash_value(42)) > 0
        assert len(cache._hash_value("string")) > 0
        assert len(cache._hash_value([1, 2, 3])) > 0
        assert len(cache._hash_value({"key": "value"})) > 0
        assert len(cache._hash_value(np.array([1, 2]))) > 0

    def test_cache_manager_estimate_size(self, tmp_path: Path) -> None:
        """Test size estimation for different types."""
        cache = CacheManager(backend="memory", cache_dir=tmp_path)

        # Numpy array
        array = np.zeros(1000, dtype=np.float64)
        assert cache._estimate_size(array) == 8000  # 1000 * 8 bytes

        # String
        string = "x" * 100
        assert cache._estimate_size(string) == 100

        # List
        lst = [1, 2, 3]
        assert cache._estimate_size(lst) > 0

    def test_cache_manager_generate_key_consistency(self, tmp_path: Path) -> None:
        """Test cache key generation is deterministic."""
        cache = CacheManager(backend="memory", cache_dir=tmp_path)

        key1 = cache._generate_key("func", (1, 2), {"arg": 3})
        key2 = cache._generate_key("func", (1, 2), {"arg": 3})

        assert key1 == key2

    def test_cache_manager_generate_key_different_args(self, tmp_path: Path) -> None:
        """Test different arguments produce different keys."""
        cache = CacheManager(backend="memory", cache_dir=tmp_path)

        key1 = cache._generate_key("func", (1, 2), {})
        key2 = cache._generate_key("func", (1, 3), {})

        assert key1 != key2

    def test_cache_manager_decorator_caching(self, tmp_path: Path) -> None:
        """Test decorator for automatic caching."""
        cache = CacheManager(backend="memory", cache_dir=tmp_path)

        call_count = 0

        @cache.cached(ttl=3600)
        def expensive_function(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        # First call - should compute
        result1 = expensive_function(5)
        assert result1 == 10
        assert call_count == 1

        # Second call - should use cache
        result2 = expensive_function(5)
        assert result2 == 10
        assert call_count == 1  # Not incremented

        # Different argument - should compute
        result3 = expensive_function(10)
        assert result3 == 20
        assert call_count == 2

    def test_cache_manager_decorator_with_numpy(self, tmp_path: Path) -> None:
        """Test decorator caching with numpy arrays."""
        cache = CacheManager(backend="memory", cache_dir=tmp_path)

        call_count = 0

        @cache.cached()
        def compute_fft(signal: np.ndarray) -> np.ndarray:
            nonlocal call_count
            call_count += 1
            return np.fft.fft(signal)

        signal = np.array([1, 2, 3, 4, 5])

        # First call - should compute
        result1 = compute_fft(signal)
        assert call_count == 1

        # Second call with same array - should use cache
        result2 = compute_fft(signal)
        assert call_count == 1
        np.testing.assert_array_almost_equal(result1, result2)

    def test_cache_manager_multi_level_backend(self, tmp_path: Path) -> None:
        """Test multi-level cache (memory + disk)."""
        cache = CacheManager(backend="multi_level", cache_dir=tmp_path)

        cache.set("test_key", "test_value")

        # Should be in memory
        assert cache.get("test_key") == "test_value"

        # Clear memory cache
        cache._memory_cache.clear()

        # Should still be retrievable from disk
        assert cache.get("test_key") == "test_value"

        # Should be promoted back to memory
        assert "test_key" in cache._memory_cache

    def test_cache_manager_redis_fallback(self, tmp_path: Path) -> None:
        """Test Redis backend falls back to memory if unavailable."""
        # Redis should fail if not installed/running
        cache = CacheManager(backend="redis", cache_dir=tmp_path)

        # Should fall back to memory backend
        assert cache.backend == CacheBackend.MEMORY

    def test_cache_manager_disk_index_persistence(self, tmp_path: Path) -> None:
        """Test disk cache index is persisted."""
        # Create cache and add entry
        cache1 = CacheManager(backend="disk", cache_dir=tmp_path)
        cache1.set("persistent_key", "persistent_value")

        # Create new cache instance
        cache2 = CacheManager(backend="disk", cache_dir=tmp_path)

        # Should load index and retrieve value
        assert cache2.get("persistent_key") == "persistent_value"


class TestGlobalCache:
    """Test global cache convenience functions."""

    def test_get_global_cache(self) -> None:
        """Test getting global cache instance."""
        cache1 = get_global_cache()
        cache2 = get_global_cache()

        # Should return same instance
        assert cache1 is cache2

    def test_global_cache_decorator(self) -> None:
        """Test global cache decorator."""
        call_count = 0

        @cache(ttl=3600)
        def test_function(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        result1 = test_function(5)
        assert result1 == 10
        assert call_count == 1

        result2 = test_function(5)
        assert result2 == 10
        assert call_count == 1  # Cached


class TestCacheIntegration:
    """Integration tests for caching with real-world scenarios."""

    def test_fft_caching_speedup(self, tmp_path: Path) -> None:
        """Test caching provides speedup for FFT operations."""
        cache_manager = CacheManager(backend="memory", cache_dir=tmp_path)

        @cache_manager.cached(ttl=3600)
        def cached_fft(signal: np.ndarray) -> np.ndarray:
            return np.fft.fft(signal)

        # Large signal for measurable timing
        signal = np.random.randn(10000)

        # First call - compute
        start1 = time.time()
        result1 = cached_fft(signal)
        time1 = time.time() - start1

        # Second call - cached
        start2 = time.time()
        result2 = cached_fft(signal)
        time2 = time.time() - start2

        # Cached call should be faster
        assert time2 < time1

        # Results should be identical
        np.testing.assert_array_almost_equal(result1, result2)

    def test_protocol_decoding_cache(self, tmp_path: Path) -> None:
        """Test caching for protocol decoding results."""
        cache_manager = CacheManager(backend="disk", cache_dir=tmp_path)

        @cache_manager.cached(ttl=7200)
        def decode_protocol(data: bytes) -> dict[str, Any]:
            # Simulate expensive decoding
            time.sleep(0.01)
            return {
                "type": "UART",
                "baudrate": 9600,
                "messages": len(data),
            }

        data = b"test_protocol_data"

        # First decode
        result1 = decode_protocol(data)
        assert result1["type"] == "UART"

        # Second decode should be faster
        start = time.time()
        result2 = decode_protocol(data)
        elapsed = time.time() - start

        assert result2 == result1
        assert elapsed < 0.005  # Should be much faster than 0.01s

    def test_cache_statistics_accuracy(self, tmp_path: Path) -> None:
        """Test cache statistics are accurate across operations."""
        cache_manager = CacheManager(backend="memory", cache_dir=tmp_path)

        @cache_manager.cached()
        def test_func(x: int) -> int:
            return x * 2

        # Generate hits and misses
        test_func(1)  # Miss
        test_func(1)  # Hit
        test_func(2)  # Miss
        test_func(2)  # Hit
        test_func(1)  # Hit

        stats = cache_manager.get_stats()

        assert stats.hits == 3
        assert stats.misses == 2
        assert stats.hit_rate == 0.6  # 3/5
        assert stats.entry_count == 2

    def test_cache_size_management(self, tmp_path: Path) -> None:
        """Test cache respects size limits."""
        policy = CachePolicy(max_size_mb=0.01)  # Very small limit
        cache_manager = CacheManager(backend="memory", cache_dir=tmp_path, policy=policy)

        # Add large entries
        for i in range(20):
            large_array = np.zeros(1000)
            cache_manager.set(f"key_{i}", large_array)

        stats = cache_manager.get_stats()

        # Should have evicted entries to stay under limit
        assert stats.size_mb <= policy.max_size_mb * 1.5  # Allow some overhead
        assert stats.evictions > 0

    def test_eviction_policy_fifo(self, tmp_path: Path) -> None:
        """Test FIFO eviction policy."""
        policy = CachePolicy(max_size_mb=0.001, eviction=EvictionPolicy.FIFO)
        cache_manager = CacheManager(backend="memory", cache_dir=tmp_path, policy=policy)

        # Add entries to trigger FIFO eviction
        for i in range(10):
            cache_manager.set(f"key_{i}", {"data": "x" * 1000})

        stats = cache_manager.get_stats()
        assert stats.evictions > 0

    def test_hash_list_value(self, tmp_path: Path) -> None:
        """Test hashing list values."""
        cache_manager = CacheManager(backend="memory", cache_dir=tmp_path)

        hash1 = cache_manager._hash_value([1, 2, 3])
        hash2 = cache_manager._hash_value([1, 2, 3])
        assert hash1 == hash2

        hash3 = cache_manager._hash_value([1, 2, 4])
        assert hash1 != hash3

    def test_hash_dict_value(self, tmp_path: Path) -> None:
        """Test hashing dict values."""
        cache_manager = CacheManager(backend="memory", cache_dir=tmp_path)

        hash1 = cache_manager._hash_value({"a": 1, "b": 2})
        hash2 = cache_manager._hash_value({"a": 1, "b": 2})
        assert hash1 == hash2

    def test_redis_set_failure(self, tmp_path: Path) -> None:
        """Test Redis set with connection failure."""
        cache_manager = CacheManager(backend="memory", cache_dir=tmp_path)
        # Simulate Redis backend without client
        cache_manager.backend = CacheBackend.REDIS
        cache_manager._redis_client = None

        # Should not raise exception
        cache_manager.set("test_key", "test_value")

    def test_redis_get_failure(self, tmp_path: Path) -> None:
        """Test Redis get with connection failure."""
        cache_manager = CacheManager(backend="memory", cache_dir=tmp_path)
        # Simulate Redis backend without client
        cache_manager.backend = CacheBackend.REDIS
        cache_manager._redis_client = None

        # Should return None gracefully
        result = cache_manager.get("test_key")
        assert result is None

    def test_disk_eviction(self, tmp_path: Path) -> None:
        """Test disk cache eviction."""
        policy = CachePolicy(max_size_mb=0.001, eviction=EvictionPolicy.LRU)
        cache_manager = CacheManager(backend="disk", cache_dir=tmp_path, policy=policy)

        # Add entries to trigger disk eviction
        for i in range(10):
            cache_manager.set(f"key_{i}", {"data": "x" * 1000})

        stats = cache_manager.get_stats()
        assert stats.evictions > 0

    def test_disk_get_corrupted_file(self, tmp_path: Path) -> None:
        """Test disk get with corrupted cache file."""
        cache_manager = CacheManager(backend="disk", cache_dir=tmp_path)

        # Create corrupted cache file
        cache_key = "corrupted_key"
        cache_file = tmp_path / f"{cache_key}.pkl"
        cache_file.write_text("corrupted data")

        # Add to index
        cache_manager._disk_cache_index[cache_key] = cache_file

        # Should handle gracefully
        result = cache_manager.get(cache_key)
        assert result is None
        assert cache_key not in cache_manager._disk_cache_index

    def test_decorator_with_kwargs(self, tmp_path: Path) -> None:
        """Test decorator with keyword arguments."""
        cache_manager = CacheManager(backend="memory", cache_dir=tmp_path)

        call_count = 0

        @cache_manager.cached()
        def test_function(x: int, y: int = 10) -> int:
            nonlocal call_count
            call_count += 1
            return x + y

        result1 = test_function(5, y=15)
        assert result1 == 20
        assert call_count == 1

        result2 = test_function(5, y=15)
        assert result2 == 20
        assert call_count == 1  # Cached

        result3 = test_function(5, y=20)
        assert result3 == 25
        assert call_count == 2  # Different kwargs

    def test_size_estimation_edge_cases(self, tmp_path: Path) -> None:
        """Test size estimation for edge cases."""
        cache_manager = CacheManager(backend="memory", cache_dir=tmp_path)

        # Object that can't be pickled
        class UnpicklableObject:
            def __init__(self) -> None:
                self.func = lambda x: x

        obj = UnpicklableObject()
        size = cache_manager._estimate_size(obj)
        assert size == 0  # Should return 0 for unpicklable objects

    def test_disk_cache_missing_file(self, tmp_path: Path) -> None:
        """Test disk cache with missing file."""
        cache_manager = CacheManager(backend="disk", cache_dir=tmp_path)

        # Add entry to index but delete file
        cache_key = "missing_key"
        cache_file = tmp_path / f"{cache_key}.pkl"
        cache_manager._disk_cache_index[cache_key] = cache_file

        # Should handle missing file gracefully
        result = cache_manager.get(cache_key)
        assert result is None
        assert cache_key not in cache_manager._disk_cache_index

    def test_memory_cache_size_calculation(self, tmp_path: Path) -> None:
        """Test cache size calculation for memory backend."""
        cache_manager = CacheManager(backend="memory", cache_dir=tmp_path)

        # Add entries with known sizes
        cache_manager.set("key1", "x" * 1000)  # 1000 bytes
        cache_manager.set("key2", np.zeros(500, dtype=np.float64))  # 4000 bytes

        size_mb = cache_manager._get_cache_size_mb()
        assert size_mb > 0

    def test_disk_cache_size_calculation(self, tmp_path: Path) -> None:
        """Test cache size calculation for disk backend."""
        cache_manager = CacheManager(backend="disk", cache_dir=tmp_path)

        # Add entries
        cache_manager.set("key1", "test_value")
        cache_manager.set("key2", np.zeros(100))

        size_mb = cache_manager._get_cache_size_mb()
        assert size_mb > 0

    def test_invalidate_disk_pattern(self, tmp_path: Path) -> None:
        """Test invalidating disk cache entries by pattern."""
        cache_manager = CacheManager(backend="disk", cache_dir=tmp_path)

        cache_manager.set("test_key1", "value1")
        cache_manager.set("test_key2", "value2")
        cache_manager.set("other_key", "value3")

        invalidated = cache_manager.invalidate(pattern="test_")

        assert invalidated == 2
        assert cache_manager.get("test_key1") is None
        assert cache_manager.get("test_key2") is None
        assert cache_manager.get("other_key") == "value3"

    def test_redis_with_ttl(self, tmp_path: Path) -> None:
        """Test Redis backend set with TTL."""
        cache_manager = CacheManager(backend="memory", cache_dir=tmp_path)

        # Simulate Redis backend with mock client
        mock_redis = MagicMock()
        cache_manager.backend = CacheBackend.REDIS
        cache_manager._redis_client = mock_redis

        # Set with TTL
        cache_manager.set("test_key", "test_value", ttl=3600)

        # Verify setex was called with TTL
        mock_redis.setex.assert_called_once()

    def test_redis_with_no_ttl(self, tmp_path: Path) -> None:
        """Test Redis backend set without TTL."""
        cache_manager = CacheManager(backend="memory", cache_dir=tmp_path)

        # Simulate Redis backend with mock client
        mock_redis = MagicMock()
        cache_manager.backend = CacheBackend.REDIS
        cache_manager._redis_client = mock_redis

        # Create policy with no TTL
        cache_manager.policy.ttl = None

        # Set without TTL
        cache_manager.set("test_key", "test_value")

        # Verify set was called (not setex)
        mock_redis.set.assert_called_once()

    def test_multi_level_entry_count(self, tmp_path: Path) -> None:
        """Test entry count for multi-level cache."""
        cache_manager = CacheManager(backend="multi_level", cache_dir=tmp_path)

        cache_manager.set("key1", "value1")
        cache_manager.set("key2", "value2")

        stats = cache_manager.get_stats()

        # Should count entries in both memory and disk
        assert stats.entry_count >= 2
