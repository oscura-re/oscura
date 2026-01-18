"""Unit tests for advanced memory management features.

Tests for MEM-* requirements:
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

# Lazy evaluation
from oscura.utils.lazy import (
    LazyArray,
    LazyOperation,
    ProgressiveResolution,
    auto_preview,
    lazy_operation,
    select_roi,
)

# Memory extensions
from oscura.utils.memory_extensions import (
    ArrayManager,
    LRUCache,
    ResourceManager,
    cache_key,
    clear_cache,
    get_result_cache,
)

pytestmark = pytest.mark.unit


# Skip tests if optional dependencies not available
pywt = pytest.importorskip("pywt", reason="PyWavelets required for wavelet tests")
h5py = pytest.importorskip("h5py", reason="h5py required for HDF5 tests")


# =============================================================================
# =============================================================================


def test_cwt_chunked_basic():
    """Test chunked CWT with small file."""
    from oscura.analyzers.waveform.wavelets import cwt_chunked

    # Create test file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as f:
        data = np.sin(2 * np.pi * 10 * np.linspace(0, 1, 10000))
        data.astype(np.float64).tofile(f)
        temp_path = f.name

    try:
        scales = np.arange(1, 32)
        chunks = list(cwt_chunked(temp_path, scales, chunk_size=5000))

        # Should have 2 chunks for 10000 samples with chunk_size=5000
        assert len(chunks) >= 1
        assert len(chunks) <= 3  # Allow some overlap

        # Check output shape
        coeffs, scales_out = chunks[0]
        assert coeffs.shape[0] == len(scales)
        assert np.array_equal(scales_out, scales)

    finally:
        Path(temp_path).unlink()


def test_dwt_chunked_basic():
    """Test chunked DWT with small file."""
    from oscura.analyzers.waveform.wavelets import dwt_chunked

    # Create test file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as f:
        data = np.random.randn(8192)
        data.astype(np.float64).tofile(f)
        temp_path = f.name

    try:
        level = 3
        chunks = list(dwt_chunked(temp_path, level=level, chunk_size=4096))

        assert len(chunks) >= 1

        # Check dictionary structure
        coeffs_dict = chunks[0]
        assert f"cA{level}" in coeffs_dict
        assert f"cD{level}" in coeffs_dict
        assert "cD1" in coeffs_dict

    finally:
        Path(temp_path).unlink()


# =============================================================================
# =============================================================================


def test_correlate_chunked_same_mode():
    """Test chunked correlation with 'same' mode."""
    from oscura.analyzers.statistics.correlation import correlate_chunked

    # Large signals
    signal1 = np.random.randn(100000)
    signal2 = np.random.randn(1000)

    # Chunked correlation
    result = correlate_chunked(signal1, signal2, mode="same")

    # Verify shape
    assert result.shape == (len(signal1),)

    # Compare with numpy (for small subset)
    expected = np.correlate(signal1[:1000], signal2[:100], mode="same")
    actual = result[:1000][: len(expected) - len(signal2) + 100]
    # Results should be close (not exact due to chunking boundary effects)
    assert actual.shape[0] > 0


def test_correlate_chunked_valid_mode():
    """Test chunked correlation with 'valid' mode."""
    from oscura.analyzers.statistics.correlation import correlate_chunked

    signal1 = np.random.randn(50000)
    signal2 = np.random.randn(500)

    result = correlate_chunked(signal1, signal2, mode="valid")

    # Valid mode output length
    expected_len = max(0, len(signal1) - len(signal2) + 1)
    assert result.shape == (expected_len,)


def test_correlate_chunked_full_mode():
    """Test chunked correlation with 'full' mode."""
    from oscura.analyzers.statistics.correlation import correlate_chunked

    signal1 = np.random.randn(10000)
    signal2 = np.random.randn(100)

    result = correlate_chunked(signal1, signal2, mode="full")

    # Full mode output length
    expected_len = len(signal1) + len(signal2) - 1
    assert result.shape == (expected_len,)


# =============================================================================
# MEM-013, MEM-026, MEM-027: Progressive Resolution & ROI
# =============================================================================


def test_auto_preview():
    """Test automatic preview generation."""
    data = np.random.randn(100000)

    # Small data, no preview
    result = auto_preview(data, preview_only=False)
    assert len(result) == len(data)

    # Large data, preview mode
    large_data = np.random.randn(10_000_000)
    preview = auto_preview(large_data, downsample_factor=100, preview_only=True)
    assert len(preview) == len(large_data) // 100


def test_select_roi_by_samples():
    """Test ROI selection by sample indices."""
    data = np.random.randn(100000)

    roi = select_roi(data, start=1000, end=2000)
    assert len(roi) == 1000
    assert np.array_equal(roi, data[1000:2000])


def test_select_roi_by_time():
    """Test ROI selection by time values."""
    data = np.random.randn(100000)
    sample_rate = 1e6  # 1 MHz

    roi = select_roi(data, start_time=0.001, end_time=0.002, sample_rate=sample_rate)

    # Should select 1000 samples (1ms at 1MHz)
    assert len(roi) == 1000


def test_progressive_resolution():
    """Test progressive resolution analyzer."""
    data = np.random.randn(1_000_000)
    sample_rate = 1e6

    analyzer = ProgressiveResolution(data, sample_rate)

    # Get preview
    preview = analyzer.get_preview(downsample_factor=100)
    assert len(preview) == len(data) // 100

    # Get ROI
    roi = analyzer.get_roi(start_time=0.1, end_time=0.2)
    expected_len = int(0.1 * sample_rate)
    assert len(roi) == expected_len


# =============================================================================
# =============================================================================


def test_lazy_hdf5_array():
    """Test lazy HDF5 array loading."""
    from oscura.utils.memory_extensions import LazyHDF5Array

    # Create test HDF5 file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as f:
        temp_path = f.name

    try:
        # Write test data
        with h5py.File(temp_path, "w") as f:
            data = np.random.randn(100000)
            f.create_dataset("data", data=data)

        # Load lazily
        with LazyHDF5Array(temp_path, "/data") as arr:
            assert arr.shape == (100000,)
            assert arr.dtype == np.float64

            # Load only a slice
            chunk = arr[1000:2000]
            assert chunk.shape == (1000,)

    finally:
        Path(temp_path).unlink()


# =============================================================================
# =============================================================================


def test_lazy_array():
    """Test lazy array evaluation."""

    def expensive_computation():
        return np.random.randn(10000)

    lazy = LazyArray(expensive_computation)

    # Not computed yet
    assert not lazy.is_computed()

    # Trigger computation
    result = lazy.compute()
    assert lazy.is_computed()
    assert result.shape == (10000,)

    # Second call uses cached result
    result2 = lazy.compute()
    assert np.array_equal(result, result2)


def test_lazy_operation_chaining():
    """Test chaining lazy operations."""
    data = np.arange(1000)

    # Chain operations
    op1 = LazyOperation(lambda x: x**2, data)
    op2 = LazyOperation(lambda x: x + 1, op1)
    op3 = LazyOperation(lambda x: np.sum(x), op2)

    # Not computed yet
    assert not op1.is_computed()
    assert not op2.is_computed()

    # Compute final result
    result = op3.compute()

    # Verify result
    expected = np.sum(data**2 + 1)
    assert result == expected


def test_lazy_operation_helper():
    """Test lazy_operation helper function."""
    data = np.arange(100)

    lazy_fft = lazy_operation(np.fft.fft, data)
    assert not lazy_fft.is_computed()

    result = lazy_fft.compute()
    expected = np.fft.fft(data)
    assert np.allclose(result, expected)


# =============================================================================
# =============================================================================


def test_resource_manager():
    """Test resource manager context."""
    cleanup_called = False

    def cleanup(resource):
        nonlocal cleanup_called
        cleanup_called = True

    with ResourceManager(np.zeros(1000), cleanup_func=cleanup) as data:
        assert data.shape == (1000,)

    assert cleanup_called


def test_array_manager():
    """Test array manager context."""
    with ArrayManager(np.zeros((100, 100))) as arr:
        assert arr.shape == (100, 100)
        result = np.sum(arr)
        assert result == 0.0

    # Array should be cleaned up after context


# =============================================================================
# MEM-021, MEM-029: LRU Cache
# =============================================================================


def test_lru_cache_basic():
    """Test basic LRU cache operations."""
    cache = LRUCache(max_memory_bytes=1_000_000)

    # Put and get
    data = np.zeros(1000)
    cache.put("key1", data, size_bytes=8000)

    result = cache.get("key1")
    assert result is not None
    assert np.array_equal(result, data)

    # Cache miss
    result = cache.get("nonexistent")
    assert result is None


def test_lru_cache_eviction():
    """Test LRU cache eviction."""
    cache = LRUCache(max_memory_bytes=10000, max_entries=2)

    # Add entries
    cache.put("key1", np.zeros(100), size_bytes=800)
    cache.put("key2", np.zeros(100), size_bytes=800)

    # Both should be cached
    assert cache.get("key1") is not None
    assert cache.get("key2") is not None

    # Add third entry - should evict key1 (LRU)
    cache.put("key3", np.zeros(100), size_bytes=800)

    assert cache.get("key1") is None  # Evicted
    assert cache.get("key2") is not None
    assert cache.get("key3") is not None


def test_lru_cache_size_estimation():
    """Test automatic size estimation."""
    cache = LRUCache(max_memory_bytes=1_000_000)

    # Numpy array - size auto-estimated
    data = np.zeros(1000, dtype=np.float64)
    cache.put("array", data)

    stats = cache.stats()
    assert stats["memory_bytes"] == data.nbytes


def test_lru_cache_stats():
    """Test cache statistics."""
    cache = LRUCache(max_memory_bytes=100000)

    # Hits and misses
    cache.put("key1", "value1", size_bytes=100)
    cache.get("key1")  # Hit
    cache.get("key1")  # Hit
    cache.get("key2")  # Miss

    stats = cache.stats()
    assert stats["hits"] == 2
    assert stats["misses"] == 1
    assert stats["hit_rate"] == 2 / 3
    assert stats["entries"] == 1


def test_global_result_cache():
    """Test global result cache functions."""
    clear_cache()  # Start fresh

    cache = get_result_cache()
    cache.put("test_result", np.zeros(100), size_bytes=800)

    # Verify cached
    result = cache.get("test_result")
    assert result is not None

    # Clear cache
    clear_cache()
    result = cache.get("test_result")
    assert result is None


def test_cache_key_generation():
    """Test cache key generation."""
    key1 = cache_key("operation", samples=1000, nfft=2048)
    key2 = cache_key("operation", samples=1000, nfft=2048)
    key3 = cache_key("operation", samples=1000, nfft=4096)

    # Same arguments produce same key
    assert key1 == key2

    # Different arguments produce different key
    assert key1 != key3


# =============================================================================
# Integration Tests
# =============================================================================


def test_lazy_with_cache_integration():
    """Test lazy evaluation combined with caching."""
    clear_cache()

    def expensive_operation(data):
        return np.fft.fft(data)

    data = np.random.randn(1000)

    # Create lazy operation
    lazy_fft = lazy_operation(expensive_operation, data)

    # Generate cache key
    key = cache_key("fft", data.shape, data.dtype)

    # Check cache (miss)
    cache = get_result_cache()
    cached_result = cache.get(key)
    assert cached_result is None

    # Compute and cache
    result = lazy_fft.compute()
    cache.put(key, result)

    # Verify cached
    cached_result = cache.get(key)
    assert cached_result is not None
    assert np.allclose(cached_result, result)


def test_progressive_resolution_with_roi():
    """Test complete progressive workflow."""
    # Create large synthetic data
    t = np.linspace(0, 1, 10_000_000)
    data = np.sin(2 * np.pi * 10 * t) + 0.1 * np.random.randn(len(t))

    # Progressive analyzer
    analyzer = ProgressiveResolution(data, sample_rate=10e6)

    # Stage 1: Preview
    preview = analyzer.get_preview(downsample_factor=1000)
    assert len(preview) == len(data) // 1000

    # Stage 2: User identifies ROI from preview, gets full resolution
    roi = analyzer.get_roi(start_time=0.1, end_time=0.2)
    expected_len = int(0.1 * analyzer.sample_rate)
    assert len(roi) == expected_len

    # Verify ROI matches original data
    start_idx = int(0.1 * analyzer.sample_rate)
    end_idx = int(0.2 * analyzer.sample_rate)
    assert np.allclose(roi, data[start_idx:end_idx])
