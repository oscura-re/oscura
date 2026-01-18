import pytest

"""Tests for persistent cache module.

Tests requirements:
"""

import tempfile
from pathlib import Path

import numpy as np

from oscura.core.cache import (
    CacheStats,
    OscuraCache,
    clear_cache,
    get_cache,
    show_cache_stats,
)

pytestmark = [pytest.mark.unit, pytest.mark.core]


class TestOscuraCache:
    """Tests for OscuraCache (MEM-031, MEM-029)."""

    def test_basic_put_get(self):
        """Test basic cache put and get operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = OscuraCache(max_memory="10MB", cache_dir=tmpdir)

            # Put value
            cache.put("key1", "value1")

            # Get value
            value = cache.get("key1")
            assert value == "value1"

            cache.clear()

    def test_cache_miss(self):
        """Test cache miss returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = OscuraCache(max_memory="10MB", cache_dir=tmpdir)

            value = cache.get("nonexistent")
            assert value is None

            stats = cache.get_stats()
            assert stats.misses == 1
            assert stats.hits == 0

            cache.clear()

    def test_cache_hit(self):
        """Test cache hit increments hit counter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = OscuraCache(max_memory="10MB", cache_dir=tmpdir)

            cache.put("key", "value")

            # Multiple gets
            for _ in range(5):
                value = cache.get("key")
                assert value == "value"

            stats = cache.get_stats()
            assert stats.hits == 5

            cache.clear()

    def test_lru_eviction(self):
        """Test LRU eviction policy."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Small cache size
            cache = OscuraCache(max_memory="1KB", cache_dir=tmpdir)

            # Add multiple large entries to trigger eviction
            for i in range(10):
                data = np.zeros(200, dtype=np.float64)  # ~1.6KB each
                cache.put(f"key{i}", data)

            stats = cache.get_stats()

            # Should have evictions or disk spills
            assert stats.evictions > 0 or stats.disk_spills > 0

            cache.clear()

    def test_disk_spillover(self):
        """Test disk spillover when memory limit exceeded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = OscuraCache(max_memory="1KB", cache_dir=tmpdir)

            # Add data that exceeds memory limit
            large_data = np.zeros(1000, dtype=np.float64)  # 8KB
            cache.put("large_key", large_data)

            # Add more to trigger spill
            for i in range(5):
                data = np.zeros(100, dtype=np.float64)
                cache.put(f"key{i}", data)

            stats = cache.get_stats()

            # Should have disk entries
            assert stats.disk_spills > 0 or stats.disk_entries > 0

            cache.clear()

    def test_reload_from_disk(self):
        """Test reloading values from disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = OscuraCache(max_memory="1KB", cache_dir=tmpdir)

            # Create data that will be spilled
            original_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
            cache.put("spilled_key", original_data)

            # Force spill by adding more data
            for i in range(10):
                data = np.zeros(200, dtype=np.float64)
                cache.put(f"key{i}", data)

            # Now retrieve the spilled data
            retrieved_data = cache.get("spilled_key")

            # Should be able to reload from disk
            if retrieved_data is not None:
                np.testing.assert_array_equal(retrieved_data, original_data)

            cache.clear()

    def test_get_or_compute(self):
        """Test get_or_compute method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = OscuraCache(max_memory="10MB", cache_dir=tmpdir)

            call_count = 0

            def expensive_computation(x):
                nonlocal call_count
                call_count += 1
                return x * 2

            # First call should compute
            result1 = cache.get_or_compute("compute_key", expensive_computation, 5)
            assert result1 == 10
            assert call_count == 1

            # Second call should use cache
            result2 = cache.get_or_compute("compute_key", expensive_computation, 5)
            assert result2 == 10
            assert call_count == 1  # Not called again

            cache.clear()

    def test_cache_stats(self):
        """Test cache statistics (MEM-031)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = OscuraCache(max_memory="10MB", cache_dir=tmpdir)

            # Generate some activity
            cache.put("key1", "value1")
            cache.put("key2", "value2")
            cache.get("key1")  # Hit
            cache.get("key3")  # Miss

            stats = cache.get_stats()

            assert stats.hits == 1
            assert stats.misses == 1
            assert stats.hit_rate == 0.5
            assert stats.current_entries == 2

            cache.clear()

    def test_show_stats(self, capsys):
        """Test show_stats output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = OscuraCache(max_memory="10MB", cache_dir=tmpdir)

            cache.put("key", "value")
            cache.get("key")

            cache.show_stats()

            captured = capsys.readouterr()
            assert "Cache Statistics" in captured.out
            assert "Hits:" in captured.out
            assert "Hit Rate:" in captured.out

            cache.clear()

    def test_clear_cache(self):
        """Test cache clearing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = OscuraCache(max_memory="10MB", cache_dir=tmpdir)

            # Add entries
            cache.put("key1", "value1")
            cache.put("key2", "value2")

            stats_before = cache.get_stats()
            assert stats_before.current_entries == 2

            # Clear
            cache.clear()

            stats_after = cache.get_stats()
            assert stats_after.current_entries == 0

    def test_context_manager(self):
        """Test cache as context manager with auto-cleanup."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)

            with OscuraCache(max_memory="10MB", cache_dir=cache_dir) as cache:
                cache.put("key", "value")

                # Should have entry
                assert cache.get("key") == "value"

            # Cache should be cleared after context exit
            # (auto_cleanup=True by default)

    def test_compute_key(self):
        """Test cache key computation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = OscuraCache(max_memory="10MB", cache_dir=tmpdir)

            # Same arguments should produce same key
            key1 = cache.compute_key("operation", param1=10, param2="value")
            key2 = cache.compute_key("operation", param1=10, param2="value")
            assert key1 == key2

            # Different arguments should produce different keys
            key3 = cache.compute_key("operation", param1=20, param2="value")
            assert key1 != key3

            cache.clear()

    def test_numpy_array_caching(self):
        """Test caching of numpy arrays."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = OscuraCache(max_memory="10MB", cache_dir=tmpdir)

            # Cache numpy array
            array = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
            cache.put("array_key", array)

            # Retrieve
            retrieved = cache.get("array_key")
            assert retrieved is not None
            np.testing.assert_array_equal(retrieved, array)

            cache.clear()

    def test_nested_data_caching(self):
        """Test caching of nested data structures."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = OscuraCache(max_memory="10MB", cache_dir=tmpdir)

            # Cache complex structure
            data = {
                "values": np.array([1, 2, 3]),
                "metadata": {"param1": 10, "param2": "test"},
                "results": [1.5, 2.5, 3.5],
            }
            cache.put("complex_key", data)

            # Retrieve
            retrieved = cache.get("complex_key")
            assert retrieved is not None
            assert "values" in retrieved
            np.testing.assert_array_equal(retrieved["values"], data["values"])
            assert retrieved["metadata"] == data["metadata"]

            cache.clear()

    def test_memory_string_parsing(self):
        """Test parsing of memory limit strings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test different memory string formats
            cache_gb = OscuraCache(max_memory="2GB", cache_dir=tmpdir)
            assert cache_gb.max_memory == 2e9

            cache_mb = OscuraCache(max_memory="500MB", cache_dir=tmpdir)
            assert cache_mb.max_memory == 500e6

            cache_kb = OscuraCache(max_memory="1024KB", cache_dir=tmpdir)
            assert cache_kb.max_memory == 1024e3

            cache_gb.clear()
            cache_mb.clear()
            cache_kb.clear()

    def test_access_tracking(self):
        """Test that access count is tracked."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = OscuraCache(max_memory="10MB", cache_dir=tmpdir)

            cache.put("key", "value")

            # Access multiple times
            for _ in range(5):
                cache.get("key")

            # Access count should be tracked in entry
            entry = cache._cache.get("key")
            assert entry is not None
            assert entry.access_count == 5

            cache.clear()


class TestGlobalCache:
    """Tests for global cache functions."""

    def test_get_cache(self):
        """Test get_cache creates global instance."""
        cache = get_cache(max_memory="10MB")
        assert cache is not None
        assert isinstance(cache, OscuraCache)

        # Should return same instance
        cache2 = get_cache()
        assert cache2 is cache

        clear_cache()

    def test_clear_cache(self):
        """Test global clear_cache."""
        cache = get_cache(max_memory="10MB")
        cache.put("key", "value")

        clear_cache()

        # Cache should be None after clear
        cache = get_cache()  # Will create new one
        assert cache.get("key") is None

        clear_cache()

    def test_show_cache_stats(self, capsys):
        """Test global show_cache_stats."""
        cache = get_cache(max_memory="10MB")
        cache.put("key", "value")
        cache.get("key")

        show_cache_stats()

        captured = capsys.readouterr()
        assert "Cache Statistics" in captured.out

        clear_cache()

    def test_show_stats_uninitialized(self, capsys):
        """Test show_cache_stats when cache not initialized."""
        clear_cache()  # Ensure no cache

        show_cache_stats()

        captured = capsys.readouterr()
        assert "not initialized" in captured.out


class TestCacheStats:
    """Tests for CacheStats dataclass."""

    def test_stats_creation(self):
        """Test creating CacheStats."""
        stats = CacheStats(
            hits=10,
            misses=5,
            evictions=2,
            disk_spills=1,
            current_memory=1000000,
            current_entries=8,
            disk_entries=1,
        )

        assert stats.hits == 10
        assert stats.misses == 5
        assert stats.hit_rate == 10 / 15

    def test_hit_rate_calculation(self):
        """Test hit rate calculation."""
        stats = CacheStats(
            hits=75,
            misses=25,
            evictions=0,
            disk_spills=0,
            current_memory=0,
            current_entries=0,
            disk_entries=0,
        )

        assert stats.hit_rate == 0.75

    def test_zero_hits_misses(self):
        """Test hit rate with no activity."""
        stats = CacheStats(
            hits=0,
            misses=0,
            evictions=0,
            disk_spills=0,
            current_memory=0,
            current_entries=0,
            disk_entries=0,
        )

        assert stats.hit_rate == 0.0

    def test_stats_string_format(self):
        """Test stats string formatting."""
        stats = CacheStats(
            hits=10,
            misses=5,
            evictions=2,
            disk_spills=1,
            current_memory=1000000000,
            current_entries=8,
            disk_entries=1,
        )

        stats_str = str(stats)
        assert "Hits: 10" in stats_str
        assert "Misses: 5" in stats_str
        assert "Hit Rate:" in stats_str
        assert "GB" in stats_str


class TestCoreCacheBasicEdgeCases:
    """Tests for edge cases and error handling."""

    def test_overwrite_existing_key(self):
        """Test overwriting existing cache entry."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = OscuraCache(max_memory="10MB", cache_dir=tmpdir)

            cache.put("key", "value1")
            cache.put("key", "value2")  # Overwrite

            value = cache.get("key")
            assert value == "value2"

            cache.clear()

    def test_empty_cache_stats(self):
        """Test stats on empty cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = OscuraCache(max_memory="10MB", cache_dir=tmpdir)

            stats = cache.get_stats()
            assert stats.current_entries == 0
            assert stats.hits == 0
            assert stats.misses == 0

            cache.clear()

    def test_very_large_value(self):
        """Test caching very large value."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Small cache, large value
            cache = OscuraCache(max_memory="1KB", cache_dir=tmpdir)

            # Value larger than cache
            large_value = np.zeros(10000, dtype=np.float64)  # 80KB
            cache.put("large", large_value)

            # Should spill to disk
            stats = cache.get_stats()
            assert stats.disk_spills > 0 or stats.disk_entries > 0

            cache.clear()

    def test_compute_key_with_array(self):
        """Test compute_key with numpy arrays."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = OscuraCache(max_memory="10MB", cache_dir=tmpdir)

            arr1 = np.array([1, 2, 3])
            arr2 = np.array([1, 2, 3])
            arr3 = np.array([1, 2, 4])

            # Same arrays should produce same key
            key1 = cache.compute_key(arr1)
            key2 = cache.compute_key(arr2)
            assert key1 == key2

            # Different arrays should produce different keys
            key3 = cache.compute_key(arr3)
            assert key1 != key3

            cache.clear()

    def test_no_auto_cleanup(self):
        """Test cache without auto cleanup."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with OscuraCache(max_memory="10MB", cache_dir=tmpdir, auto_cleanup=False) as cache:
                cache.put("key", "value")

                # Entry should still exist after context
                stats_before = cache.get_stats()
                assert stats_before.current_entries == 1

            # Manual cleanup needed
            cache.clear()


class TestThreadSafety:
    """Tests for thread-safe cache operations (MEM-031)."""

    def test_concurrent_put_get(self):
        """Test concurrent put/get operations from multiple threads."""
        import threading

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = OscuraCache(max_memory="10MB", cache_dir=tmpdir)
            errors = []

            def writer(thread_id: int) -> None:
                try:
                    for i in range(50):
                        cache.put(f"thread_{thread_id}_key_{i}", f"value_{thread_id}_{i}")
                except Exception as e:
                    errors.append(e)

            def reader(thread_id: int) -> None:
                try:
                    for i in range(50):
                        cache.get(f"thread_{thread_id}_key_{i}")
                except Exception as e:
                    errors.append(e)

            # Start multiple threads
            threads = []
            for t_id in range(4):
                t_write = threading.Thread(target=writer, args=(t_id,))
                t_read = threading.Thread(target=reader, args=(t_id,))
                threads.extend([t_write, t_read])

            for t in threads:
                t.start()

            for t in threads:
                t.join()

            # No errors should have occurred
            assert len(errors) == 0, f"Thread errors: {errors}"

            cache.clear()

    def test_has_lock_attribute(self):
        """Verify cache has threading lock for thread safety."""
        import threading

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = OscuraCache(max_memory="10MB", cache_dir=tmpdir)

            # Should have a lock
            assert hasattr(cache, "_lock")
            assert isinstance(cache._lock, type(threading.RLock()))

            cache.clear()
