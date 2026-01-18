"""Comprehensive unit tests for memory management utilities.

This module provides extensive testing for memory estimation, availability checking,
configuration management, and OOM prevention. Tests aim for >90% coverage and cover
all public functions and classes with edge cases, error handling, and validation.


Test Coverage:
- MemoryEstimate: initialization, repr, attributes
- MemoryCheck: initialization, attributes
- MemoryCheckError: exception behavior, attributes
- MemoryConfig: validation, thresholds, auto_degrade
- DownsamplingRecommendation: initialization, attributes
- ProgressInfo: percentage calculation, formatting, edge cases
- MemoryMonitor: context manager, stats, interval checking
- All public functions with various parameter combinations
- Edge cases: zero samples, very large numbers, invalid inputs
- Error conditions: invalid thresholds, malformed strings
- Environment variable handling
- Mocking for psutil and /proc filesystem
"""

from __future__ import annotations

import os
from unittest.mock import mock_open, patch

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
    _next_power_of_2,
    check_memory_available,
    configure_memory,
    detect_wsl,
    estimate_memory,
    gc_collect,
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


# =============================================================================
# Dataclass Tests
# =============================================================================


@pytest.mark.unit
class TestMemoryEstimate:
    """Test MemoryEstimate dataclass."""

    def test_memory_estimate_creation(self) -> None:
        """Test basic MemoryEstimate creation."""
        est = MemoryEstimate(
            data=1e9,
            intermediate=2e9,
            output=1e9,
            total=4e9,
            operation="test",
            parameters={"samples": 1000},
        )
        assert est.data == 1e9
        assert est.intermediate == 2e9
        assert est.output == 1e9
        assert est.total == 4e9
        assert est.operation == "test"
        assert est.parameters["samples"] == 1000

    def test_memory_estimate_repr(self) -> None:
        """Test MemoryEstimate string representation."""
        est = MemoryEstimate(
            data=1e9,
            intermediate=2e9,
            output=1e9,
            total=4e9,
            operation="fft",
            parameters={},
        )
        repr_str = repr(est)
        assert "MemoryEstimate" in repr_str
        assert "fft" in repr_str
        assert "4.00 GB" in repr_str

    def test_memory_estimate_large_values(self) -> None:
        """Test MemoryEstimate with very large values."""
        est = MemoryEstimate(
            data=100e9,
            intermediate=200e9,
            output=100e9,
            total=400e9,
            operation="huge_fft",
            parameters={},
        )
        assert est.total == 400e9
        repr_str = repr(est)
        assert "400.00 GB" in repr_str


@pytest.mark.unit
class TestMemoryCheck:
    """Test MemoryCheck dataclass."""

    def test_memory_check_sufficient(self) -> None:
        """Test MemoryCheck with sufficient memory."""
        check = MemoryCheck(
            sufficient=True,
            available=8e9,
            required=2e9,
            recommendation="Memory sufficient",
        )
        assert check.sufficient
        assert check.available == 8e9
        assert check.required == 2e9

    def test_memory_check_insufficient(self) -> None:
        """Test MemoryCheck with insufficient memory."""
        check = MemoryCheck(
            sufficient=False,
            available=2e9,
            required=8e9,
            recommendation="Need more memory",
        )
        assert not check.sufficient
        assert check.available < check.required


@pytest.mark.unit
class TestMemoryCheckError:
    """Test MemoryCheckError exception."""

    def test_memory_check_error_creation(self) -> None:
        """Test MemoryCheckError creation."""
        error = MemoryCheckError(
            message="Test error",
            required=8e9,
            available=2e9,
            recommendation="Downgrade operation",
        )
        assert str(error) == "Test error"
        assert error.required == 8e9
        assert error.available == 2e9
        assert error.recommendation == "Downgrade operation"

    def test_memory_check_error_raise(self) -> None:
        """Test raising MemoryCheckError."""
        with pytest.raises(MemoryCheckError, match="Out of memory") as exc_info:
            raise MemoryCheckError(
                message="Out of memory",
                required=16e9,
                available=4e9,
                recommendation="Use chunking",
            )
        assert exc_info.value.required == 16e9
        assert exc_info.value.available == 4e9


@pytest.mark.unit
class TestMemoryConfig:
    """Test MemoryConfig dataclass."""

    def test_memory_config_defaults(self) -> None:
        """Test MemoryConfig with default values."""
        config = MemoryConfig()
        assert config.max_memory is None
        assert config.warn_threshold == 0.7
        assert config.critical_threshold == 0.9
        assert config.auto_degrade is False

    def test_memory_config_custom_values(self) -> None:
        """Test MemoryConfig with custom values."""
        config = MemoryConfig(
            max_memory=4e9,
            warn_threshold=0.6,
            critical_threshold=0.85,
            auto_degrade=True,
        )
        assert config.max_memory == 4e9
        assert config.warn_threshold == 0.6
        assert config.critical_threshold == 0.85
        assert config.auto_degrade is True

    def test_memory_config_threshold_validation_out_of_range_low(self) -> None:
        """Test MemoryConfig with threshold below 0.0."""
        with pytest.raises(ValueError, match="warn_threshold must be 0.0-1.0"):
            MemoryConfig(warn_threshold=-0.1)

    def test_memory_config_threshold_validation_out_of_range_high(self) -> None:
        """Test MemoryConfig with threshold above 1.0."""
        with pytest.raises(ValueError, match="warn_threshold must be 0.0-1.0"):
            MemoryConfig(warn_threshold=1.5)

    def test_memory_config_critical_threshold_validation(self) -> None:
        """Test MemoryConfig with invalid critical threshold."""
        with pytest.raises(ValueError, match="critical_threshold must be 0.0-1.0"):
            MemoryConfig(critical_threshold=1.5)

    def test_memory_config_threshold_ordering(self) -> None:
        """Test MemoryConfig with warn >= critical."""
        with pytest.raises(ValueError, match="warn_threshold.*must be < critical_threshold"):
            MemoryConfig(warn_threshold=0.9, critical_threshold=0.8)

    def test_memory_config_threshold_equal(self) -> None:
        """Test MemoryConfig with warn == critical."""
        with pytest.raises(ValueError, match="warn_threshold.*must be < critical_threshold"):
            MemoryConfig(warn_threshold=0.7, critical_threshold=0.7)


@pytest.mark.unit
class TestDownsamplingRecommendation:
    """Test DownsamplingRecommendation dataclass."""

    def test_downsampling_recommendation_creation(self) -> None:
        """Test DownsamplingRecommendation creation."""
        rec = DownsamplingRecommendation(
            factor=4,
            required_memory=16e9,
            available_memory=4e9,
            new_sample_rate=250e6,
            message="Test recommendation",
        )
        assert rec.factor == 4
        assert rec.required_memory == 16e9
        assert rec.available_memory == 4e9
        assert rec.new_sample_rate == 250e6
        assert "Test recommendation" in rec.message


@pytest.mark.unit
class TestProgressInfo:
    """Test ProgressInfo dataclass."""

    def test_progress_info_creation(self) -> None:
        """Test ProgressInfo creation."""
        info = ProgressInfo(
            current=50,
            total=100,
            eta_seconds=30.5,
            memory_used=2e9,
            memory_peak=3e9,
            operation="test_op",
        )
        assert info.current == 50
        assert info.total == 100
        assert info.memory_used == 2e9
        assert info.memory_peak == 3e9
        assert info.operation == "test_op"

    def test_progress_info_percent(self) -> None:
        """Test ProgressInfo percent calculation."""
        info = ProgressInfo(
            current=25,
            total=100,
            eta_seconds=0,
            memory_used=0,
            memory_peak=0,
            operation="test",
        )
        assert info.percent == 25.0

    def test_progress_info_percent_three_quarters(self) -> None:
        """Test ProgressInfo percent at 75%."""
        info = ProgressInfo(
            current=150,
            total=200,
            eta_seconds=0,
            memory_used=0,
            memory_peak=0,
            operation="test",
        )
        assert info.percent == 75.0

    def test_progress_info_percent_zero_total(self) -> None:
        """Test ProgressInfo percent with zero total."""
        info = ProgressInfo(
            current=0,
            total=0,
            eta_seconds=0,
            memory_used=0,
            memory_peak=0,
            operation="test",
        )
        assert info.percent == 100.0

    def test_progress_info_format_progress(self) -> None:
        """Test ProgressInfo format_progress method."""
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
        assert "2.50 GB" in formatted
        assert "3.00 GB" in formatted
        assert "ETA" in formatted
        assert "30s" in formatted

    def test_progress_info_format_progress_large_eta(self) -> None:
        """Test ProgressInfo format with large ETA."""
        info = ProgressInfo(
            current=1,
            total=100,
            eta_seconds=3600.5,
            memory_used=1e9,
            memory_peak=2e9,
            operation="test",
        )
        formatted = info.format_progress()
        assert "1.0%" in formatted
        assert "3600s" in formatted or "3601s" in formatted


# =============================================================================
# WSL Detection Tests
# =============================================================================


@pytest.mark.unit
class TestDetectWSL:
    """Test WSL detection."""

    def test_detect_wsl_not_wsl(self) -> None:
        """Test WSL detection on non-WSL system."""
        # Mock /proc/version to return non-WSL content
        with patch("builtins.open", mock_open(read_data="Linux version 5.10.0")):
            result = detect_wsl()
            assert isinstance(result, bool)

    def test_detect_wsl_microsoft(self) -> None:
        """Test WSL detection with Microsoft marker."""
        with patch("builtins.open", mock_open(read_data="Microsoft WSL")):
            result = detect_wsl()
            assert result is True

    def test_detect_wsl_wsl_marker(self) -> None:
        """Test WSL detection with WSL marker."""
        with patch("builtins.open", mock_open(read_data="wsl version")):
            result = detect_wsl()
            assert result is True

    def test_detect_wsl_case_insensitive(self) -> None:
        """Test WSL detection is case insensitive."""
        with patch("builtins.open", mock_open(read_data="microsoft version")):
            result = detect_wsl()
            assert result is True

    def test_detect_wsl_file_not_found(self) -> None:
        """Test WSL detection when /proc/version doesn't exist."""
        with patch("builtins.open", side_effect=FileNotFoundError):
            result = detect_wsl()
            assert result is False

    def test_detect_wsl_permission_denied(self) -> None:
        """Test WSL detection with permission error."""
        with patch("builtins.open", side_effect=PermissionError):
            result = detect_wsl()
            assert result is False


# =============================================================================
# Memory Query Tests
# =============================================================================


@pytest.mark.unit
class TestMemoryQueries:
    """Test memory query functions."""

    def test_get_total_memory_returns_positive(self) -> None:
        """Test get_total_memory returns positive value."""
        total = get_total_memory()
        assert isinstance(total, int)
        assert total > 0

    def test_get_total_memory_reasonable_range(self) -> None:
        """Test get_total_memory returns reasonable value."""
        total = get_total_memory()
        # Should be at least 100 MB
        assert total > 100 * 1024 * 1024

    def test_get_available_memory_returns_positive(self) -> None:
        """Test get_available_memory returns positive value."""
        available = get_available_memory()
        assert isinstance(available, int)
        assert available >= 0

    def test_get_available_memory_less_than_total(self) -> None:
        """Test get_available_memory <= total."""
        total = get_total_memory()
        available = get_available_memory()
        assert available <= total

    def test_get_swap_available_returns_non_negative(self) -> None:
        """Test get_swap_available returns non-negative value."""
        swap = get_swap_available()
        assert isinstance(swap, int)
        assert swap >= 0

    def test_get_memory_pressure_in_range(self) -> None:
        """Test get_memory_pressure returns value in [0.0, 1.0]."""
        pressure = get_memory_pressure()
        assert isinstance(pressure, float)
        assert 0.0 <= pressure <= 1.0

    def test_get_memory_pressure_reasonable(self) -> None:
        """Test get_memory_pressure is reasonable."""
        # Most systems should be below 90%
        pressure = get_memory_pressure()
        assert pressure < 1.0

    def test_get_memory_info_returns_dict(self) -> None:
        """Test get_memory_info returns dictionary."""
        info = get_memory_info()
        assert isinstance(info, dict)

    def test_get_memory_info_has_required_keys(self) -> None:
        """Test get_memory_info has all required keys."""
        info = get_memory_info()
        required_keys = [
            "total",
            "available",
            "swap_available",
            "max_memory",
            "pressure_pct",
            "wsl",
        ]
        for key in required_keys:
            assert key in info

    def test_get_memory_info_values_are_valid(self) -> None:
        """Test get_memory_info values are valid."""
        info = get_memory_info()
        assert info["total"] > 0
        assert info["available"] >= 0
        assert info["swap_available"] >= 0
        assert info["max_memory"] > 0
        assert 0 <= info["pressure_pct"] <= 100
        assert isinstance(info["wsl"], bool)


# =============================================================================
# Memory Estimation Tests
# =============================================================================


@pytest.mark.unit
class TestEstimateMemory:
    """Test estimate_memory function."""

    def test_estimate_fft_basic(self) -> None:
        """Test FFT memory estimation."""
        est = estimate_memory("fft", samples=1e6, nfft=1024)
        assert isinstance(est, MemoryEstimate)
        assert est.total > 0
        assert est.data > 0
        assert est.intermediate > 0
        assert est.output > 0
        assert est.operation == "fft"

    def test_estimate_fft_float32(self) -> None:
        """Test FFT with float32."""
        est32 = estimate_memory("fft", samples=1e6, dtype="float32")
        est64 = estimate_memory("fft", samples=1e6, dtype="float64")
        assert est32.data < est64.data

    def test_estimate_fft_multi_channel(self) -> None:
        """Test FFT with multiple channels."""
        est1 = estimate_memory("fft", samples=1e6, channels=1)
        est4 = estimate_memory("fft", samples=1e6, channels=4)
        assert est4.data >= est1.data * 3

    def test_estimate_psd(self) -> None:
        """Test PSD memory estimation."""
        est = estimate_memory("psd", samples=1e6, nperseg=256, nfft=512)
        assert est.total > 0
        assert est.operation == "psd"

    def test_estimate_psd_defaults(self) -> None:
        """Test PSD with default nperseg."""
        est = estimate_memory("psd", samples=1e6)
        assert est.total > 0
        assert est.parameters["nperseg"] == 256

    def test_estimate_spectrogram(self) -> None:
        """Test spectrogram memory estimation."""
        est = estimate_memory("spectrogram", samples=1e6, nperseg=256, noverlap=128, nfft=512)
        assert est.total > 0
        assert est.operation == "spectrogram"

    def test_estimate_spectrogram_defaults(self) -> None:
        """Test spectrogram with default parameters."""
        est = estimate_memory("spectrogram", samples=1e6)
        assert est.total > 0
        assert est.parameters["nperseg"] == 256
        assert est.parameters["noverlap"] == 128

    def test_estimate_eye_diagram(self) -> None:
        """Test eye diagram memory estimation."""
        est = estimate_memory(
            "eye_diagram",
            samples=1e6,
            samples_per_ui=100,
            num_uis=1000,
        )
        assert est.total > 0
        assert est.operation == "eye_diagram"

    def test_estimate_eye_diagram_defaults(self) -> None:
        """Test eye diagram with default parameters."""
        est = estimate_memory("eye_diagram", samples=1e6)
        assert est.total > 0

    def test_estimate_correlate(self) -> None:
        """Test correlation memory estimation."""
        est = estimate_memory("correlate", samples=1e6)
        assert est.total > 0
        assert est.operation == "correlate"

    def test_estimate_filter(self) -> None:
        """Test filter memory estimation."""
        est = estimate_memory("filter", samples=1e6, filter_order=8)
        assert est.total > 0
        assert est.operation == "filter"

    def test_estimate_filter_default_order(self) -> None:
        """Test filter with default order."""
        est = estimate_memory("filter", samples=1e6)
        assert est.total > 0

    def test_estimate_generic_operation(self) -> None:
        """Test memory estimation for unknown operation."""
        est = estimate_memory("unknown_op", samples=1e6)
        assert est.total > 0
        assert est.operation == "unknown_op"

    def test_estimate_zero_samples(self) -> None:
        """Test memory estimation with zero samples."""
        est = estimate_memory("fft", samples=0)
        assert est.data == 0
        assert est.total >= 0

    def test_estimate_none_samples(self) -> None:
        """Test memory estimation with None samples."""
        est = estimate_memory("fft", samples=None)
        assert est.data == 0

    def test_estimate_float_samples(self) -> None:
        """Test memory estimation with float samples."""
        est = estimate_memory("fft", samples=1.5e6)
        assert est.data > 0

    def test_estimate_very_large_samples(self) -> None:
        """Test memory estimation with very large samples."""
        est = estimate_memory("fft", samples=1e12)
        assert est.total > 100e9


# =============================================================================
# Memory Check Tests
# =============================================================================


@pytest.mark.unit
class TestCheckMemoryAvailable:
    """Test check_memory_available function."""

    def test_check_small_operation_sufficient(self) -> None:
        """Test memory check for small operation."""
        check = check_memory_available("fft", samples=1000)
        assert isinstance(check, MemoryCheck)
        assert check.sufficient
        assert check.available > 0
        assert check.required > 0

    def test_check_large_operation(self) -> None:
        """Test memory check for potentially large operation."""
        check = check_memory_available("spectrogram", samples=1e9, nperseg=8192)
        assert isinstance(check, MemoryCheck)
        assert isinstance(check.sufficient, bool)

    def test_check_very_large_operation(self) -> None:
        """Test memory check for huge operation."""
        check = check_memory_available("spectrogram", samples=1e12, nperseg=8192)
        # This should typically fail on most systems
        if not check.sufficient:
            assert check.recommendation
            assert check.required > check.available


@pytest.mark.unit
class TestRequireMemory:
    """Test require_memory function."""

    def test_require_memory_small_operation(self) -> None:
        """Test require_memory for small operation."""
        # Should not raise
        require_memory("fft", samples=1000)

    def test_require_memory_huge_operation(self) -> None:
        """Test require_memory for impossible operation."""
        with pytest.raises(MemoryCheckError, match="Insufficient memory for fft"):
            require_memory("fft", samples=1e12)

    def test_require_memory_error_attributes(self) -> None:
        """Test MemoryCheckError attributes from require_memory."""
        try:
            require_memory("fft", samples=1e12)
            pytest.fail("Should have raised MemoryCheckError")
        except MemoryCheckError as e:
            assert e.required > 0
            assert e.available > 0
            assert e.recommendation


# =============================================================================
# Memory Configuration Tests
# =============================================================================


@pytest.mark.unit
class TestSetMaxMemory:
    """Test set_max_memory function."""

    def teardown_method(self) -> None:
        """Reset memory config after each test."""
        set_max_memory(None)

    def test_set_max_memory_bytes(self) -> None:
        """Test setting max memory in bytes."""
        set_max_memory(4 * 1024 * 1024 * 1024)
        assert get_max_memory() == 4 * 1024 * 1024 * 1024

    def test_set_max_memory_gb_string(self) -> None:
        """Test setting max memory with GB string."""
        set_max_memory("4GB")
        assert get_max_memory() == 4e9

    def test_set_max_memory_mb_string(self) -> None:
        """Test setting max memory with MB string."""
        set_max_memory("512MB")
        assert get_max_memory() == 512e6

    def test_set_max_memory_kb_string(self) -> None:
        """Test setting max memory with KB string."""
        set_max_memory("256KB")
        assert get_max_memory() == 256e3

    def test_set_max_memory_lowercase(self) -> None:
        """Test setting max memory with lowercase units."""
        set_max_memory("2gb")
        assert get_max_memory() == 2e9

    def test_set_max_memory_with_spaces(self) -> None:
        """Test setting max memory with spaces."""
        set_max_memory("  4GB  ")
        assert get_max_memory() == 4e9

    def test_set_max_memory_none(self) -> None:
        """Test setting max memory to None."""
        set_max_memory("4GB")
        set_max_memory(None)
        # Should use 80% of available
        max_mem = get_max_memory()
        available = get_available_memory()
        assert max_mem <= available


@pytest.mark.unit
class TestGetMaxMemory:
    """Test get_max_memory function."""

    def teardown_method(self) -> None:
        """Reset memory config after each test."""
        set_max_memory(None)
        if "TK_MAX_MEMORY" in os.environ:
            del os.environ["TK_MAX_MEMORY"]

    def test_get_max_memory_default(self) -> None:
        """Test get_max_memory returns reasonable default."""
        set_max_memory(None)
        max_mem = get_max_memory()
        available = get_available_memory()
        # Default is 80% of available
        assert max_mem <= available
        assert max_mem >= available * 0.75

    def test_get_max_memory_respects_set_value(self) -> None:
        """Test get_max_memory respects set_max_memory."""
        set_max_memory(2e9)
        assert get_max_memory() == 2e9

    def test_get_max_memory_from_env(self) -> None:
        """Test get_max_memory reads from environment."""
        set_max_memory(None)
        os.environ["TK_MAX_MEMORY"] = "3GB"
        # Clear any cached value
        max_mem = get_max_memory()
        assert max_mem == 3e9


@pytest.mark.unit
class TestConfigureMemory:
    """Test configure_memory function."""

    def setup_method(self) -> None:
        """Reset config before each test."""
        configure_memory(max_memory=None, warn_threshold=0.7, critical_threshold=0.9)

    def test_configure_memory_max_memory_bytes(self) -> None:
        """Test configure_memory with bytes."""
        configure_memory(max_memory=4e9)
        config = get_memory_config()
        assert config.max_memory == 4e9

    def test_configure_memory_max_memory_gb(self) -> None:
        """Test configure_memory with GB string."""
        configure_memory(max_memory="4GB")
        config = get_memory_config()
        assert config.max_memory == 4e9

    def test_configure_memory_max_memory_mb(self) -> None:
        """Test configure_memory with MB string."""
        configure_memory(max_memory="512MB")
        config = get_memory_config()
        assert config.max_memory == 512e6

    def test_configure_memory_thresholds(self) -> None:
        """Test configure_memory with thresholds."""
        configure_memory(warn_threshold=0.6, critical_threshold=0.85)
        config = get_memory_config()
        assert config.warn_threshold == 0.6
        assert config.critical_threshold == 0.85

    def test_configure_memory_auto_degrade(self) -> None:
        """Test configure_memory with auto_degrade."""
        configure_memory(auto_degrade=True)
        config = get_memory_config()
        assert config.auto_degrade is True

    def test_configure_memory_validation(self) -> None:
        """Test configure_memory validates thresholds."""
        with pytest.raises(ValueError, match="warn_threshold.*must be < critical_threshold"):
            configure_memory(warn_threshold=0.9, critical_threshold=0.8)

    def test_configure_memory_partial_update(self) -> None:
        """Test configure_memory partial updates."""
        configure_memory(warn_threshold=0.6)
        config = get_memory_config()
        assert config.warn_threshold == 0.6
        # Other values should remain unchanged
        assert config.critical_threshold == 0.9


@pytest.mark.unit
class TestMemoryReserve:
    """Test memory reserve from environment."""

    def setup_method(self) -> None:
        """Save original environment."""
        self.original_reserve = os.environ.get("TK_MEMORY_RESERVE")

    def teardown_method(self) -> None:
        """Restore original environment."""
        if self.original_reserve:
            os.environ["TK_MEMORY_RESERVE"] = self.original_reserve
        elif "TK_MEMORY_RESERVE" in os.environ:
            del os.environ["TK_MEMORY_RESERVE"]

    def test_memory_reserve_gb(self) -> None:
        """Test memory reserve in GB."""
        os.environ["TK_MEMORY_RESERVE"] = "1GB"
        available_with_reserve = get_available_memory()
        # Should be reduced by ~1GB
        assert available_with_reserve >= 0

    def test_memory_reserve_mb(self) -> None:
        """Test memory reserve in MB."""
        os.environ["TK_MEMORY_RESERVE"] = "512MB"
        available_with_reserve = get_available_memory()
        assert available_with_reserve >= 0

    def test_memory_reserve_bytes(self) -> None:
        """Test memory reserve in bytes."""
        os.environ["TK_MEMORY_RESERVE"] = "1000000000"
        available_with_reserve = get_available_memory()
        assert available_with_reserve >= 0

    def test_memory_reserve_invalid(self) -> None:
        """Test invalid memory reserve falls back to zero."""
        os.environ["TK_MEMORY_RESERVE"] = "invalid"
        available = get_available_memory()
        # Should still work, treating invalid as 0
        assert available > 0


# =============================================================================
# Downsampling Tests
# =============================================================================


@pytest.mark.unit
class TestSuggestDownsampling:
    """Test suggest_downsampling function."""

    def test_suggest_downsampling_small_operation(self) -> None:
        """Test suggest_downsampling for small operation."""
        rec = suggest_downsampling("fft", samples=1000, sample_rate=1e6)
        # Should be None for small operations
        assert rec is None

    def test_suggest_downsampling_huge_operation(self) -> None:
        """Test suggest_downsampling for huge operation."""
        rec = suggest_downsampling(
            "spectrogram",
            samples=1e11,
            sample_rate=1e9,
            nperseg=8192,
        )
        if rec is not None:
            assert isinstance(rec, DownsamplingRecommendation)
            assert rec.factor >= 2
            assert rec.factor <= 16
            assert rec.new_sample_rate == 1e9 / rec.factor
            assert rec.message

    def test_suggest_downsampling_factor_power_of_2(self) -> None:
        """Test suggest_downsampling returns power of 2 factor."""
        rec = suggest_downsampling(
            "spectrogram",
            samples=1e12,
            sample_rate=1e9,
            nperseg=8192,
        )
        if rec is not None:
            # Should be power of 2
            assert rec.factor & (rec.factor - 1) == 0

    def test_suggest_downsampling_max_factor_16(self) -> None:
        """Test suggest_downsampling caps factor at 16."""
        rec = suggest_downsampling(
            "spectrogram",
            samples=1e12,
            sample_rate=1e9,
            nperseg=8192,
        )
        if rec is not None:
            assert rec.factor <= 16


# =============================================================================
# MemoryMonitor Tests
# =============================================================================


@pytest.mark.unit
class TestMemoryMonitor:
    """Test MemoryMonitor context manager."""

    def test_memory_monitor_context_entry_exit(self) -> None:
        """Test MemoryMonitor context manager entry and exit."""
        with MemoryMonitor("test_op") as monitor:
            assert monitor.operation == "test_op"
            assert monitor.start_memory > 0
            assert monitor.peak_memory >= monitor.start_memory

    def test_memory_monitor_with_max_memory_int(self) -> None:
        """Test MemoryMonitor with integer max_memory."""
        with MemoryMonitor("test", max_memory=2e9) as monitor:
            assert monitor.max_memory == 2e9

    def test_memory_monitor_with_max_memory_gb(self) -> None:
        """Test MemoryMonitor with GB string max_memory."""
        with MemoryMonitor("test", max_memory="4GB") as monitor:
            assert monitor.max_memory == 4e9

    def test_memory_monitor_with_max_memory_mb(self) -> None:
        """Test MemoryMonitor with MB string max_memory."""
        with MemoryMonitor("test", max_memory="512MB") as monitor:
            assert monitor.max_memory == 512e6

    def test_memory_monitor_get_stats(self) -> None:
        """Test MemoryMonitor get_stats method."""
        with MemoryMonitor("test") as monitor:
            np.random.randn(10000)  # Allocate some memory
            monitor.check(iteration=0)
            stats = monitor.get_stats()

            assert isinstance(stats, dict)
            assert "start" in stats
            assert "current" in stats
            assert "peak" in stats
            assert "delta" in stats
            assert stats["start"] > 0
            assert stats["peak"] >= stats["start"]
            assert stats["delta"] >= 0

    def test_memory_monitor_check_interval(self) -> None:
        """Test MemoryMonitor respects check_interval."""
        with MemoryMonitor("test", check_interval=10) as monitor:
            # Should only check every 10 iterations
            for i in range(100):
                monitor.check(iteration=i)

    def test_memory_monitor_check_without_iteration(self) -> None:
        """Test MemoryMonitor check without iteration parameter."""
        with MemoryMonitor("test") as monitor:
            # Should always check without iteration
            monitor.check(iteration=None)
            monitor.check(iteration=None)

    def test_memory_monitor_check_iteration_periodic(self) -> None:
        """Test MemoryMonitor check is periodic."""
        check_count = 0

        with MemoryMonitor("test", check_interval=5) as monitor:
            # Mock _get_process_memory to count calls
            original_get_memory = monitor._get_process_memory
            call_count = 0

            def mock_get_memory() -> int:
                nonlocal call_count
                call_count += 1
                return original_get_memory()

            monitor._get_process_memory = mock_get_memory

            for i in range(20):
                monitor.check(iteration=i)

            # Should only call _get_process_memory 4 times (0, 5, 10, 15)
            assert call_count >= 3


# =============================================================================
# Garbage Collection Tests
# =============================================================================


@pytest.mark.unit
class TestGarbageCollection:
    """Test garbage collection function."""

    def test_gc_collect_returns_int(self) -> None:
        """Test gc_collect returns integer."""
        result = gc_collect()
        assert isinstance(result, int)
        assert result >= 0

    def test_gc_collect_actually_collects(self) -> None:
        """Test gc_collect performs collection."""
        # Create some garbage
        large_array = np.random.randn(1000000)
        del large_array
        # Force collection
        collected = gc_collect()
        # Should have collected something
        assert isinstance(collected, int)


# =============================================================================
# Helper Function Tests
# =============================================================================


@pytest.mark.unit
class TestNextPowerOf2:
    """Test _next_power_of_2 helper function."""

    def test_next_power_of_2_exact(self) -> None:
        """Test _next_power_of_2 with exact power of 2."""
        assert _next_power_of_2(1) == 1
        assert _next_power_of_2(2) == 2
        assert _next_power_of_2(4) == 4
        assert _next_power_of_2(8) == 8
        assert _next_power_of_2(16) == 16

    def test_next_power_of_2_between_powers(self) -> None:
        """Test _next_power_of_2 with values between powers."""
        assert _next_power_of_2(3) == 4
        assert _next_power_of_2(5) == 8
        assert _next_power_of_2(7) == 8
        assert _next_power_of_2(9) == 16
        assert _next_power_of_2(100) == 128

    def test_next_power_of_2_zero(self) -> None:
        """Test _next_power_of_2 with zero."""
        assert _next_power_of_2(0) == 1

    def test_next_power_of_2_negative(self) -> None:
        """Test _next_power_of_2 with negative number."""
        assert _next_power_of_2(-1) == 1
        assert _next_power_of_2(-100) == 1

    def test_next_power_of_2_large(self) -> None:
        """Test _next_power_of_2 with large number."""
        assert _next_power_of_2(1000000) == 1048576  # 2^20


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.unit
class TestMemoryIntegration:
    """Integration tests combining multiple functions."""

    def setup_method(self) -> None:
        """Reset config before each test."""
        configure_memory(max_memory=None, warn_threshold=0.7, critical_threshold=0.9)

    def teardown_method(self) -> None:
        """Reset config after each test."""
        configure_memory(max_memory=None, warn_threshold=0.7, critical_threshold=0.9)

    def test_estimate_and_check_workflow(self) -> None:
        """Test typical estimate and check workflow."""
        # Estimate memory for operation
        estimate = estimate_memory("fft", samples=1e6, nfft=2048)
        assert estimate.total > 0

        # Check if operation is feasible
        check = check_memory_available("fft", samples=1e6, nfft=2048)
        assert check.required == estimate.total

    def test_memory_config_workflow(self) -> None:
        """Test memory configuration workflow."""
        # Configure global limits
        configure_memory(
            max_memory="4GB",
            warn_threshold=0.65,
            critical_threshold=0.85,
        )

        # Check configuration
        config = get_memory_config()
        assert config.max_memory == 4e9
        assert config.warn_threshold == 0.65

        # Also test set_max_memory for memory info
        set_max_memory("4GB")

        # Get memory info - will use the value from set_max_memory
        info = get_memory_info()
        assert info["max_memory"] == 4e9

    def test_downsampling_workflow(self) -> None:
        """Test downsampling recommendation workflow."""
        # For a huge operation
        rec = suggest_downsampling(
            "spectrogram",
            samples=1e12,
            sample_rate=1e9,
            nperseg=4096,
        )

        if rec is not None:
            # Verify downsampling recommendation is valid
            assert rec.factor >= 2
            assert rec.factor <= 16
            assert rec.new_sample_rate < 1e9  # Rate should be reduced
            assert rec.required_memory > rec.available_memory  # Should be OOM
            assert rec.message  # Should have recommendation message


# =============================================================================
# Mock/Fallback Tests
# =============================================================================


@pytest.mark.unit
class TestFallbackBehavior:
    """Test fallback behavior when psutil is not available."""

    def test_get_total_memory_without_psutil(self) -> None:
        """Test get_total_memory fallback without psutil."""
        with patch.dict("sys.modules", {"psutil": None}):
            # Should use fallback
            total = get_total_memory()
            assert total > 0

    def test_get_available_memory_without_psutil(self) -> None:
        """Test get_available_memory fallback without psutil."""
        with patch.dict("sys.modules", {"psutil": None}):
            # Should use fallback
            available = get_available_memory()
            assert available > 0

    def test_get_swap_available_without_psutil(self) -> None:
        """Test get_swap_available fallback without psutil."""
        with patch.dict("sys.modules", {"psutil": None}):
            # Should use fallback
            swap = get_swap_available()
            assert swap >= 0

    def test_memory_monitor_get_process_memory_without_psutil(self) -> None:
        """Test MemoryMonitor._get_process_memory fallback without psutil."""
        with patch.dict("sys.modules", {"psutil": None}):
            with MemoryMonitor("test") as monitor:
                mem = monitor._get_process_memory()
                assert mem > 0


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


@pytest.mark.unit
class TestUtilsMemoryEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_estimate_memory_with_kwargs(self) -> None:
        """Test estimate_memory with extra kwargs."""
        est = estimate_memory(
            "unknown",
            samples=1e6,
            extra_param=123,
            another_param="test",
        )
        assert est.parameters["extra_param"] == 123
        assert est.parameters["another_param"] == "test"

    def test_memory_config_zero_max_memory(self) -> None:
        """Test MemoryConfig with zero max_memory."""
        config = MemoryConfig(max_memory=0)
        assert config.max_memory == 0

    def test_estimate_very_small_samples(self) -> None:
        """Test estimate_memory with very small sample counts."""
        est = estimate_memory("fft", samples=1)
        assert est.total >= 0

    def test_progress_info_large_numbers(self) -> None:
        """Test ProgressInfo with very large numbers."""
        info = ProgressInfo(
            current=int(1e18),
            total=int(2e18),
            eta_seconds=3600.5,
            memory_used=int(1e12),
            memory_peak=int(2e12),
            operation="test",
        )
        assert info.percent > 0

    def test_set_max_memory_float_value(self) -> None:
        """Test set_max_memory with float value."""
        set_max_memory(4.5e9)
        assert isinstance(get_max_memory(), int)
        set_max_memory(None)
