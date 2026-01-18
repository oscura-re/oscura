"""Tests for persistent LRU cache with disk spillover.

Tests the caching infrastructure (MEM-029, MEM-031).
"""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from oscura.core.cache import (
    CacheStats,
    OscuraCache,
    clear_cache,
    get_cache,
    show_cache_stats,
)

pytestmark = [pytest.mark.unit, pytest.mark.core]


class TestCacheStats:
    """Test CacheStats dataclass."""

    def test_create_stats(self) -> None:
        """Test creating cache stats."""
        stats = CacheStats(
            hits=100,
            misses=20,
            evictions=5,
            disk_spills=3,
            current_memory=1024**3,
            current_entries=50,
            disk_entries=3,
        )
        assert stats.hits == 100
        assert stats.misses == 20
        assert stats.hit_rate == pytest.approx(100 / 120)

    def test_hit_rate_zero_total(self) -> None:
        """Test hit rate with no hits or misses."""
        stats = CacheStats(0, 0, 0, 0, 0, 0, 0)
        assert stats.hit_rate == 0.0

    def test_str_formatting(self) -> None:
        """Test string formatting."""
        stats = CacheStats(10, 5, 2, 1, 500 * 1024**2, 10, 1)
        output = str(stats)
        assert "Hits: 10" in output
        assert "Misses: 5" in output
        assert "Hit Rate:" in output
        assert "GB" in output


class TestOscuraCache:
    """Test OscuraCache class."""

    def test_init_with_string_memory(self, tmp_path: Path) -> None:
        """Test initialization with string memory."""
        cache = OscuraCache("500MB", cache_dir=tmp_path)
        assert cache.max_memory == 500 * 1e6
        assert cache.cache_dir == tmp_path

    def test_init_with_int_memory(self, tmp_path: Path) -> None:
        """Test initialization with integer memory."""
        cache = OscuraCache(1024**3, cache_dir=tmp_path)
        assert cache.max_memory == 1024**3

    def test_init_creates_cache_dir(self, tmp_path: Path) -> None:
        """Test that cache creates directory."""
        cache_dir = tmp_path / "cache_test"
        cache = OscuraCache("1GB", cache_dir=cache_dir)
        assert cache_dir.exists()

    def test_context_manager(self, tmp_path: Path) -> None:
        """Test context manager auto-cleanup."""
        with OscuraCache("1GB", cache_dir=tmp_path / "test1") as cache:
            cache.put("key1", "value1")
            assert cache.get("key1") == "value1"

        # After exit, cache should be cleared due to auto_cleanup
        assert len(cache._cache) == 0

    def test_context_manager_no_cleanup(self, tmp_path: Path) -> None:
        """Test context manager without auto cleanup."""
        cache = OscuraCache("1GB", cache_dir=tmp_path / "test2", auto_cleanup=False)
        with cache:
            cache.put("key1", "value1")

        # Cache should still have entry
        assert len(cache._cache) == 1

    def test_get_miss(self, tmp_path: Path) -> None:
        """Test cache miss."""
        cache = OscuraCache("1GB", cache_dir=tmp_path)
        result = cache.get("nonexistent")
        assert result is None
        assert cache._misses == 1

    def test_get_hit(self, tmp_path: Path) -> None:
        """Test cache hit."""
        cache = OscuraCache("1GB", cache_dir=tmp_path)
        cache.put("key1", "value1")
        result = cache.get("key1")
        assert result == "value1"
        assert cache._hits == 1

    def test_get_updates_lru(self, tmp_path: Path) -> None:
        """Test that get updates LRU order."""
        cache = OscuraCache("1GB", cache_dir=tmp_path)
        cache.put("key1", "value1")
        cache.put("key2", "value2")

        # Access key1
        cache.get("key1")

        # key1 should be at end (most recent)
        keys = list(cache._cache.keys())
        assert keys[-1] == "key1"

    def test_put_new_value(self, tmp_path: Path) -> None:
        """Test putting new value."""
        cache = OscuraCache("1GB", cache_dir=tmp_path)
        cache.put("key1", "value1")
        assert "key1" in cache._cache
        assert cache.get("key1") == "value1"

    def test_put_replaces_existing(self, tmp_path: Path) -> None:
        """Test that put replaces existing value."""
        cache = OscuraCache("1GB", cache_dir=tmp_path)
        cache.put("key1", "value1")
        cache.put("key1", "value2")
        assert cache.get("key1") == "value2"
        assert len(cache._cache) == 1

    def test_get_or_compute_cached(self, tmp_path: Path) -> None:
        """Test get_or_compute with cached value."""
        cache = OscuraCache("1GB", cache_dir=tmp_path)
        compute_fn = MagicMock(return_value="computed")

        cache.put("key1", "cached_value")
        result = cache.get_or_compute("key1", compute_fn)

        assert result == "cached_value"
        assert not compute_fn.called

    def test_get_or_compute_miss(self, tmp_path: Path) -> None:
        """Test get_or_compute with cache miss."""
        cache = OscuraCache("1GB", cache_dir=tmp_path)
        compute_fn = MagicMock(return_value="computed_value")

        result = cache.get_or_compute("key1", compute_fn, "arg1", kwarg1="val1")

        assert result == "computed_value"
        compute_fn.assert_called_once_with("arg1", kwarg1="val1")
        assert cache.get("key1") == "computed_value"

    def test_clear(self, tmp_path: Path) -> None:
        """Test clearing cache."""
        cache = OscuraCache("1GB", cache_dir=tmp_path)
        cache.put("key1", "value1")
        cache.put("key2", "value2")

        cache.clear()

        assert len(cache._cache) == 0
        assert cache._current_memory == 0

    def test_get_stats(self, tmp_path: Path) -> None:
        """Test getting statistics."""
        cache = OscuraCache("1GB", cache_dir=tmp_path)
        cache.put("key1", "value1")
        cache.get("key1")  # Hit
        cache.get("key2")  # Miss

        stats = cache.get_stats()
        assert stats.hits == 1
        assert stats.misses == 1
        assert stats.current_entries == 1

    def test_show_stats(self, tmp_path: Path) -> None:
        """Test show_stats prints output."""
        cache = OscuraCache("1GB", cache_dir=tmp_path)

        with patch("builtins.print") as mock_print:
            cache.show_stats()
            assert mock_print.called

    def test_compute_key_simple(self, tmp_path: Path) -> None:
        """Test compute_key with simple args."""
        cache = OscuraCache("1GB", cache_dir=tmp_path)
        key1 = cache.compute_key("arg1", "arg2")
        key2 = cache.compute_key("arg1", "arg2")
        key3 = cache.compute_key("arg1", "different")

        assert key1 == key2  # Same args = same key
        assert key1 != key3  # Different args = different key

    def test_compute_key_with_kwargs(self, tmp_path: Path) -> None:
        """Test compute_key with kwargs."""
        cache = OscuraCache("1GB", cache_dir=tmp_path)
        key1 = cache.compute_key("arg", param1=10, param2="test")
        key2 = cache.compute_key("arg", param2="test", param1=10)  # Different order

        assert key1 == key2  # kwargs are sorted

    def test_compute_key_numpy(self, tmp_path: Path) -> None:
        """Test compute_key with numpy arrays."""
        cache = OscuraCache("1GB", cache_dir=tmp_path)
        arr1 = np.array([1, 2, 3])
        arr2 = np.array([1, 2, 3])
        arr3 = np.array([1, 2, 4])

        key1 = cache.compute_key(arr1)
        key2 = cache.compute_key(arr2)
        key3 = cache.compute_key(arr3)

        assert key1 == key2
        assert key1 != key3

    def test_disk_spillover(self, tmp_path: Path) -> None:
        """Test disk spillover when memory limit exceeded."""
        # Memory limit that allows one large item + small item
        cache = OscuraCache(5000, cache_dir=tmp_path)  # 5000 bytes

        # Add large value (4000 bytes) - fits in memory
        large_data1 = np.zeros(500, dtype=np.float64)  # 4000 bytes
        cache.put("key1", large_data1)

        # Add small value (80 bytes) - both fit
        small_data = np.zeros(10, dtype=np.float64)  # 80 bytes
        cache.put("key2", small_data)

        # Add another large value - should spill key1 to disk but keep it in cache
        large_data2 = np.ones(500, dtype=np.float64)  # 4000 bytes
        cache.put("key3", large_data2)

        # Should have triggered disk spill
        assert cache._disk_spills > 0

        # Verify key1 is still in cache but on disk
        assert "key1" in cache._cache
        assert not cache._cache["key1"].in_memory

        # Verify we can retrieve key1 from disk
        result = cache.get("key1")
        np.testing.assert_array_equal(result, large_data1)

    def test_estimate_size_numpy(self, tmp_path: Path) -> None:
        """Test size estimation for numpy arrays."""
        cache = OscuraCache("1GB", cache_dir=tmp_path)
        arr = np.zeros(100, dtype=np.float64)
        size = cache._estimate_size(arr)
        assert size == 800  # 100 * 8 bytes

    def test_estimate_size_list(self, tmp_path: Path) -> None:
        """Test size estimation for lists."""
        cache = OscuraCache("1GB", cache_dir=tmp_path)
        data = [1, 2, 3, 4, 5]
        size = cache._estimate_size(data)
        assert size > 0

    def test_estimate_size_dict(self, tmp_path: Path) -> None:
        """Test size estimation for dicts."""
        cache = OscuraCache("1GB", cache_dir=tmp_path)
        data = {"a": 1, "b": 2}
        size = cache._estimate_size(data)
        assert size > 0

    def test_parse_memory_string_gb(self, tmp_path: Path) -> None:
        """Test parsing GB memory string."""
        cache = OscuraCache("2GB", cache_dir=tmp_path)
        assert cache.max_memory == 2e9

    def test_parse_memory_string_mb(self, tmp_path: Path) -> None:
        """Test parsing MB memory string."""
        cache = OscuraCache("500MB", cache_dir=tmp_path)
        assert cache.max_memory == 500e6

    def test_parse_memory_string_kb(self, tmp_path: Path) -> None:
        """Test parsing KB memory string."""
        cache = OscuraCache("1024KB", cache_dir=tmp_path)
        assert cache.max_memory == 1024e3

    def test_lru_eviction_order(self, tmp_path: Path) -> None:
        """Test that LRU eviction works correctly."""
        cache = OscuraCache(7000, cache_dir=tmp_path)  # Enough for ~1.75 large items

        # Add two items that fit in memory
        cache.put("key1", np.zeros(400, dtype=np.float64))  # 3200 bytes
        time.sleep(0.01)
        cache.put("key2", np.zeros(400, dtype=np.float64))  # 3200 bytes

        # Access key1 to make it more recent than key2
        cache.get("key1")

        # Add key3 - should spill key2 (least recent) since memory = 6400 + 3200 = 9600 > 7000
        cache.put("key3", np.zeros(400, dtype=np.float64))

        # key1 and key3 should still be in memory
        assert cache._cache["key1"].in_memory
        assert cache._cache["key3"].in_memory

        # key2 should be spilled to disk but still in cache
        assert "key2" in cache._cache
        assert not cache._cache["key2"].in_memory


class TestGlobalCache:
    """Test global cache functions."""

    def test_get_cache_creates_instance(self, tmp_path: Path) -> None:
        """Test get_cache creates global instance."""
        from oscura.core import cache as cache_module

        cache_module._global_cache = None  # Reset

        cache1 = get_cache("1GB", cache_dir=tmp_path / "global1")
        cache2 = get_cache("1GB", cache_dir=tmp_path / "global2")

        # Should return same instance
        assert cache1 is cache2

    def test_clear_cache(self, tmp_path: Path) -> None:
        """Test clear_cache clears global cache."""
        from oscura.core import cache as cache_module

        cache_module._global_cache = None  # Reset

        cache = get_cache("1GB", cache_dir=tmp_path / "global2")
        cache.put("key1", "value1")

        clear_cache()

        assert cache_module._global_cache is None

    def test_show_cache_stats_initialized(self, tmp_path: Path) -> None:
        """Test show_cache_stats with initialized cache."""
        from oscura.core import cache as cache_module

        cache_module._global_cache = None  # Reset

        get_cache("1GB", cache_dir=tmp_path / "global3")

        with patch("builtins.print") as mock_print:
            show_cache_stats()
            assert mock_print.called

    def test_show_cache_stats_not_initialized(self) -> None:
        """Test show_cache_stats without cache."""
        from oscura.core import cache as cache_module

        cache_module._global_cache = None  # Reset

        with patch("builtins.print") as mock_print:
            show_cache_stats()
            mock_print.assert_called_with("Cache not initialized")


class TestCoreCacheIntegration:
    """Integration tests for cache."""

    def test_full_workflow(self, tmp_path: Path) -> None:
        """Test complete caching workflow."""
        cache = OscuraCache("10MB", cache_dir=tmp_path)

        # Compute and cache result
        def expensive_computation(n: int) -> np.ndarray:
            return np.arange(n)

        key = cache.compute_key("computation", n=1000)
        result1 = cache.get_or_compute(key, expensive_computation, 1000)
        result2 = cache.get_or_compute(key, expensive_computation, 1000)

        # Second call should be cached
        np.testing.assert_array_equal(result1, result2)

        stats = cache.get_stats()
        assert stats.hits == 1  # Second call was cached
        assert stats.current_entries == 1

    def test_disk_spillover_workflow(self, tmp_path: Path) -> None:
        """Test workflow with disk spillover."""
        cache = OscuraCache(13000, cache_dir=tmp_path)  # Enough for ~3 large items

        # Add 4 large arrays to trigger spillover
        arrays = []
        for i in range(4):
            arr = np.full(500, i, dtype=np.float64)  # 4000 bytes each
            cache.put(f"key{i}", arr)
            arrays.append(arr)

        # Some should be spilled to disk
        assert cache._disk_spills > 0

        # Most recent items should be in memory (key1, key2, key3)
        assert cache._cache["key3"].in_memory  # Most recent

        # key0 should be spilled to disk
        assert "key0" in cache._cache
        assert not cache._cache["key0"].in_memory
        assert cache._cache["key0"].disk_path is not None

        # Verify we can retrieve from disk
        result = cache.get("key0")
        np.testing.assert_array_equal(result, arrays[0])

    def test_thread_safe_access(self, tmp_path: Path) -> None:
        """Test basic thread safety."""
        import threading

        cache = OscuraCache("1GB", cache_dir=tmp_path)
        errors = []

        def worker(thread_id: int) -> None:
            try:
                for i in range(10):
                    key = f"thread{thread_id}_key{i}"
                    cache.put(key, f"value{i}")
                    result = cache.get(key)
                    assert result == f"value{i}"
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert cache.get_stats().current_entries == 30
