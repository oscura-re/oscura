"""Tests for memory management module.

Tests requirements:
"""

import os

import numpy as np
import pytest

from oscura.utils.memory import (
    DownsamplingRecommendation,
    MemoryCheck,
    MemoryCheckError,
    MemoryConfig,
    MemoryEstimate,
    MemoryMonitor,
    ProgressInfo,
    check_memory_available,
    configure_memory,
    detect_wsl,
    estimate_memory,
    get_available_memory,
    get_max_memory,
    get_memory_config,
    get_memory_info,
    get_memory_pressure,
    get_swap_available,
    get_total_memory,
    require_memory,
    set_max_memory,
    suggest_downsampling,
)

pytestmark = pytest.mark.unit


class TestMemoryEstimation:
    """Tests for memory estimation (MEM-001)."""

    def test_estimate_fft(self):
        """Test FFT memory estimation."""
        estimate = estimate_memory("fft", samples=1000000, nfft=1024)
        assert isinstance(estimate, MemoryEstimate)
        assert estimate.total > 0
        assert estimate.data > 0
        assert estimate.operation == "fft"

    def test_estimate_spectrogram(self):
        """Test spectrogram memory estimation."""
        estimate = estimate_memory("spectrogram", samples=1000000, nperseg=256, noverlap=128)
        assert estimate.total > 0
        assert estimate.output > 0

    def test_estimate_psd(self):
        """Test PSD memory estimation."""
        estimate = estimate_memory("psd", samples=1000000, nperseg=1024)
        assert estimate.total > 0

    def test_estimate_correlation(self):
        """Test correlation memory estimation."""
        estimate = estimate_memory("correlate", samples=100000)
        assert estimate.total > 0
        assert estimate.intermediate > 0

    def test_estimate_filter(self):
        """Test filter memory estimation."""
        estimate = estimate_memory("filter", samples=100000, filter_order=8)
        assert estimate.total > 0

    def test_estimate_with_channels(self):
        """Test memory estimation with multiple channels."""
        est_1ch = estimate_memory("fft", samples=1000000, channels=1)
        est_4ch = estimate_memory("fft", samples=1000000, channels=4)
        assert est_4ch.data >= est_1ch.data * 4

    def test_estimate_dtype(self):
        """Test memory estimation with different dtypes."""
        est_f64 = estimate_memory("fft", samples=1000000, dtype="float64")
        est_f32 = estimate_memory("fft", samples=1000000, dtype="float32")
        assert est_f64.data > est_f32.data


class TestMemoryQuery:
    """Tests for memory query functions (MEM-002)."""

    def test_get_total_memory(self):
        """Test getting total system memory."""
        total = get_total_memory()
        assert total > 0
        # Should be at least 100 MB (reasonable minimum)
        assert total > 100 * 1024 * 1024

    def test_get_available_memory(self):
        """Test getting available memory."""
        available = get_available_memory()
        assert available > 0
        assert available <= get_total_memory()

    def test_get_memory_pressure(self):
        """Test getting memory pressure."""
        pressure = get_memory_pressure()
        assert 0 <= pressure <= 1

    def test_get_swap_available(self):
        """Test getting swap availability."""
        swap = get_swap_available()
        assert swap >= 0

    def test_detect_wsl(self):
        """Test WSL detection."""
        is_wsl = detect_wsl()
        assert isinstance(is_wsl, bool)

    def test_get_memory_info(self):
        """Test comprehensive memory info."""
        info = get_memory_info()
        assert "total" in info
        assert "available" in info
        assert "pressure_pct" in info
        assert "wsl" in info


class TestMemoryCheck:
    """Tests for memory availability check (MEM-003)."""

    def test_check_small_operation(self):
        """Test check for small operation (should pass)."""
        check = check_memory_available("fft", samples=1000)
        assert isinstance(check, MemoryCheck)
        assert check.sufficient
        assert check.recommendation

    def test_check_large_operation(self):
        """Test check for potentially large operation."""
        # This might pass or fail depending on system memory
        check = check_memory_available("spectrogram", samples=1e9, nperseg=8192)
        assert isinstance(check, MemoryCheck)
        assert check.required > 0
        assert check.available > 0

    def test_require_memory_small(self):
        """Test require_memory for small operation."""
        # Should not raise for small operation
        require_memory("fft", samples=1000)

    def test_require_memory_huge(self):
        """Test require_memory for impossible operation."""
        with pytest.raises(MemoryCheckError, match="Insufficient memory for fft"):
            # 1 trillion samples = way too much
            require_memory("fft", samples=1e12)


class TestMemoryConfiguration:
    """Tests for memory configuration."""

    def test_set_max_memory_bytes(self):
        """Test setting max memory in bytes."""
        set_max_memory(4 * 1024 * 1024 * 1024)  # 4 GB
        assert get_max_memory() == 4 * 1024 * 1024 * 1024
        set_max_memory(None)  # Reset

    def test_set_max_memory_string(self):
        """Test setting max memory with string."""
        set_max_memory("4GB")
        assert get_max_memory() == 4e9
        set_max_memory("512MB")
        assert get_max_memory() == 512e6
        set_max_memory(None)  # Reset

    def test_memory_reserve_env(self):
        """Test memory reserve from environment."""
        original = os.environ.get("TK_MEMORY_RESERVE")
        try:
            os.environ["TK_MEMORY_RESERVE"] = "1GB"
            available = get_available_memory()
            # Should be reduced by reserve
            assert available > 0
        finally:
            if original:
                os.environ["TK_MEMORY_RESERVE"] = original
            elif "TK_MEMORY_RESERVE" in os.environ:
                del os.environ["TK_MEMORY_RESERVE"]


class TestMemoryConfig:
    """Tests for memory configuration (MEM-009, MEM-010, MEM-011)."""

    def test_memory_config_creation(self):
        """Test MemoryConfig creation."""
        config = MemoryConfig(max_memory=4e9, warn_threshold=0.7, critical_threshold=0.9)
        assert config.max_memory == 4e9
        assert config.warn_threshold == 0.7
        assert config.critical_threshold == 0.9

    def test_memory_config_validation(self):
        """Test MemoryConfig threshold validation."""
        with pytest.raises(ValueError, match="warn_threshold.*must be < critical_threshold"):
            # warn >= critical
            MemoryConfig(warn_threshold=0.9, critical_threshold=0.8)

        with pytest.raises(ValueError, match="warn_threshold must be 0\\.0-1\\.0"):
            # threshold out of range
            MemoryConfig(warn_threshold=1.5)

    def test_configure_memory_string(self):
        """Test configure_memory with string limits."""
        configure_memory(max_memory="4GB")
        config = get_memory_config()
        assert config.max_memory == 4e9

        configure_memory(max_memory="512MB")
        config = get_memory_config()
        assert config.max_memory == 512e6

    def test_configure_memory_thresholds(self):
        """Test configure_memory with thresholds."""
        configure_memory(warn_threshold=0.6, critical_threshold=0.85)
        config = get_memory_config()
        assert config.warn_threshold == 0.6
        assert config.critical_threshold == 0.85

    def test_configure_auto_degrade(self):
        """Test auto_degrade configuration."""
        configure_memory(auto_degrade=True)
        config = get_memory_config()
        assert config.auto_degrade is True

        configure_memory(auto_degrade=False)
        config = get_memory_config()
        assert config.auto_degrade is False


class TestDownsampling:
    """Tests for automatic downsampling (MEM-012)."""

    def test_suggest_downsampling_sufficient_memory(self):
        """Test suggest_downsampling when memory is sufficient."""
        rec = suggest_downsampling("fft", samples=1000, sample_rate=1e6)
        # Should be None for small operations
        assert rec is None

    def test_suggest_downsampling_needed(self):
        """Test suggest_downsampling when downsampling needed."""
        # Very large operation
        rec = suggest_downsampling(
            "spectrogram",
            samples=1e10,  # 10 billion samples
            sample_rate=1e9,
            nperseg=8192,
        )
        if rec is not None:
            assert isinstance(rec, DownsamplingRecommendation)
            assert rec.factor >= 2
            assert rec.factor in [2, 4, 8, 16]
            assert rec.new_sample_rate == 1e9 / rec.factor
            assert rec.message
            assert rec.required_memory > rec.available_memory

    def test_downsampling_recommendation_attributes(self):
        """Test DownsamplingRecommendation attributes."""
        rec = DownsamplingRecommendation(
            factor=4,
            required_memory=16e9,
            available_memory=4e9,
            new_sample_rate=250e6,
            message="Test message",
        )
        assert rec.factor == 4
        assert rec.required_memory == 16e9
        assert rec.available_memory == 4e9
        assert rec.new_sample_rate == 250e6


class TestMemoryMonitor:
    """Tests for graceful OOM handling (MEM-015)."""

    def test_memory_monitor_context(self):
        """Test MemoryMonitor as context manager."""
        with MemoryMonitor("test_operation") as monitor:
            assert monitor.operation == "test_operation"
            assert monitor.start_memory > 0
            # Do some work
            np.random.randn(1000)
            monitor.check(iteration=0)

    def test_memory_monitor_stats(self):
        """Test MemoryMonitor statistics."""
        with MemoryMonitor("test_operation") as monitor:
            # Allocate some memory
            np.random.randn(100000)
            monitor.check(iteration=0)
            stats = monitor.get_stats()

            assert "start" in stats
            assert "current" in stats
            assert "peak" in stats
            assert "delta" in stats
            assert stats["start"] > 0
            assert stats["peak"] >= stats["start"]

    def test_memory_monitor_max_memory(self):
        """Test MemoryMonitor with max_memory limit."""
        with MemoryMonitor("test", max_memory="1GB") as monitor:
            assert monitor.max_memory == 1e9

        with MemoryMonitor("test", max_memory=2e9) as monitor:
            assert monitor.max_memory == 2e9

    def test_memory_monitor_check_interval(self):
        """Test MemoryMonitor check interval."""
        with MemoryMonitor("test", check_interval=10) as monitor:
            # Should not check every iteration
            for i in range(100):
                monitor.check(iteration=i)
            # Should complete without error


class TestProgressInfo:
    """Tests for memory-aware progress (MEM-024)."""

    def test_progress_info_creation(self):
        """Test ProgressInfo creation."""
        info = ProgressInfo(
            current=50,
            total=100,
            eta_seconds=10.5,
            memory_used=1e9,
            memory_peak=1.5e9,
            operation="test",
        )
        assert info.current == 50
        assert info.total == 100
        assert info.percent == 50.0

    def test_progress_info_percent(self):
        """Test progress percentage calculation."""
        info = ProgressInfo(
            current=25,
            total=200,
            eta_seconds=5,
            memory_used=1e9,
            memory_peak=1e9,
            operation="test",
        )
        assert info.percent == 12.5

    def test_progress_info_format(self):
        """Test progress formatting."""
        info = ProgressInfo(
            current=75,
            total=100,
            eta_seconds=30,
            memory_used=2.5e9,
            memory_peak=3e9,
            operation="test",
        )
        formatted = info.format_progress()
        assert "75.0%" in formatted
        assert "2.50 GB" in formatted  # memory_used
        assert "3.00 GB" in formatted  # memory_peak
        assert "ETA" in formatted
        assert "30s" in formatted

    def test_progress_info_zero_total(self):
        """Test progress with zero total (edge case)."""
        info = ProgressInfo(
            current=0,
            total=0,
            eta_seconds=0,
            memory_used=0,
            memory_peak=0,
            operation="test",
        )
        # Should handle division by zero gracefully
        assert info.percent == 100.0
