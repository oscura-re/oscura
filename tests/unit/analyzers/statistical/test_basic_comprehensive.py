"""Comprehensive tests for basic statistics module.

This module provides comprehensive coverage for:
- src/oscura/analyzers/statistics/basic.py
"""

from __future__ import annotations

import numpy as np
import pytest

from oscura.analyzers.statistics.basic import (
    basic_stats,
    percentiles,
    quartiles,
    running_stats,
    summary_stats,
    weighted_mean,
)
from oscura.core.types import TraceMetadata, WaveformTrace

pytestmark = [pytest.mark.unit, pytest.mark.analyzer, pytest.mark.statistical]


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def normal_data() -> np.ndarray:
    """Generate normally distributed data."""
    rng = np.random.default_rng(42)
    return rng.normal(loc=5.0, scale=2.0, size=1000)


@pytest.fixture
def waveform_trace(normal_data: np.ndarray) -> WaveformTrace:
    """Create WaveformTrace from normal data."""
    return WaveformTrace(
        data=normal_data,
        metadata=TraceMetadata(sample_rate=1e6),
    )


# =============================================================================
# Basic Stats Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.analyzer
class TestBasicStatsComprehensive:
    """Comprehensive tests for basic_stats function."""

    def test_basic_stats_ddof_parameter(self, normal_data: np.ndarray) -> None:
        """Test ddof parameter affects variance calculation."""
        stats_ddof0 = basic_stats(normal_data, ddof=0)
        stats_ddof1 = basic_stats(normal_data, ddof=1)

        # Variance with ddof=1 should be slightly larger
        assert stats_ddof1["variance"] > stats_ddof0["variance"]
        assert stats_ddof1["std"] > stats_ddof0["std"]

    def test_basic_stats_single_value(self) -> None:
        """Test basic_stats with single value."""
        data = np.array([5.0])
        stats = basic_stats(data)

        assert stats["mean"] == 5.0
        assert stats["variance"] == 0.0
        assert stats["std"] == 0.0
        assert stats["min"] == 5.0
        assert stats["max"] == 5.0
        assert stats["range"] == 0.0
        assert stats["count"] == 1

    def test_basic_stats_constant_data(self) -> None:
        """Test basic_stats with all identical values."""
        data = np.full(100, 7.0)
        stats = basic_stats(data)

        assert stats["mean"] == 7.0
        assert stats["variance"] == 0.0
        assert stats["range"] == 0.0

    def test_basic_stats_empty_array(self) -> None:
        """Test basic_stats behavior with empty array."""
        import warnings

        data = np.array([])

        # May raise error or return special values
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                stats = basic_stats(data)
                assert stats["count"] == 0
            except (ValueError, IndexError):
                pass  # Acceptable to raise error

    def test_basic_stats_negative_values(self) -> None:
        """Test basic_stats with negative numbers."""
        data = np.array([-5.0, -3.0, -1.0, -2.0])
        stats = basic_stats(data)

        assert stats["mean"] == pytest.approx(-2.75)
        assert stats["min"] == -5.0
        assert stats["max"] == -1.0


# =============================================================================
# Percentiles Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.analyzer
class TestPercentilesComprehensive:
    """Comprehensive tests for percentiles function."""

    def test_percentiles_fractional_values(self, normal_data: np.ndarray) -> None:
        """Test percentiles with fractional percentile values."""
        pct = percentiles(normal_data, [2.5, 97.5])

        assert "p2.5" in pct
        assert "p97.5" in pct
        assert pct["p2.5"] < pct["p97.5"]

    def test_percentiles_extreme_values(self, normal_data: np.ndarray) -> None:
        """Test percentiles at 0 and 100."""
        pct = percentiles(normal_data, [0, 100])

        assert pct["p0"] == np.min(normal_data)
        assert pct["p100"] == np.max(normal_data)

    def test_percentiles_single_percentile(self, normal_data: np.ndarray) -> None:
        """Test requesting single percentile."""
        pct = percentiles(normal_data, [50])

        assert "p50" in pct
        assert pct["p50"] == pytest.approx(np.median(normal_data))

    def test_percentiles_array_input(self) -> None:
        """Test percentiles with numpy array input."""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        pct = percentiles(data, [25, 75])

        assert "p25" in pct
        assert "p75" in pct


# =============================================================================
# Quartiles Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.analyzer
class TestQuartilesComprehensive:
    """Comprehensive tests for quartiles function."""

    def test_quartiles_fence_formula(self, normal_data: np.ndarray) -> None:
        """Test fence calculation formula."""
        q = quartiles(normal_data)

        expected_lower = q["q1"] - 1.5 * q["iqr"]
        expected_upper = q["q3"] + 1.5 * q["iqr"]

        assert q["lower_fence"] == pytest.approx(expected_lower)
        assert q["upper_fence"] == pytest.approx(expected_upper)

    def test_quartiles_with_trace(self, waveform_trace: WaveformTrace) -> None:
        """Test quartiles with WaveformTrace input."""
        q = quartiles(waveform_trace)

        assert "q1" in q
        assert "median" in q
        assert "q3" in q
        assert "iqr" in q


# =============================================================================
# Weighted Mean Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.analyzer
class TestWeightedMeanComprehensive:
    """Comprehensive tests for weighted_mean function."""

    def test_weighted_mean_uniform_weights(self, normal_data: np.ndarray) -> None:
        """Test weighted mean with uniform weights equals regular mean."""
        weights = np.ones(len(normal_data))
        wm = weighted_mean(normal_data, weights)
        regular_mean = np.mean(normal_data)

        assert wm == pytest.approx(regular_mean)

    def test_weighted_mean_no_weights(self, normal_data: np.ndarray) -> None:
        """Test weighted mean without weights."""
        wm = weighted_mean(normal_data, weights=None)
        regular_mean = np.mean(normal_data)

        assert wm == pytest.approx(regular_mean)

    def test_weighted_mean_different_weights(self) -> None:
        """Test weighted mean with varying weights."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        weights = np.array([1.0, 1.0, 1.0, 1.0, 5.0])  # Weight 5 heavily

        wm = weighted_mean(data, weights)

        # Should be closer to 5 than to simple mean (3)
        assert wm > 3.5

    def test_weighted_mean_with_trace(self, waveform_trace: WaveformTrace) -> None:
        """Test weighted mean with WaveformTrace input."""
        weights = np.linspace(0.5, 1.5, len(waveform_trace.data))
        wm = weighted_mean(waveform_trace, weights)

        assert isinstance(wm, float)


# =============================================================================
# Running Stats Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.analyzer
class TestRunningStatsComprehensive:
    """Comprehensive tests for running_stats function."""

    def test_running_stats_window_larger_than_data(self) -> None:
        """Test running stats when window is larger than data."""
        data = np.array([1, 2, 3, 4, 5], dtype=float)
        result = running_stats(data, window_size=10)

        # Window should be clamped to data length
        assert len(result["mean"]) == 1

    def test_running_stats_window_size_one(self, normal_data: np.ndarray) -> None:
        """Test running stats with window size of 1."""
        result = running_stats(normal_data, window_size=1)

        # Each window contains single value
        assert len(result["mean"]) == len(normal_data)
        assert np.all(result["std"] == 0)  # Single value has no variance

    def test_running_stats_values(self) -> None:
        """Test running stats produces correct values."""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        result = running_stats(data, window_size=3)

        # First window [1, 2, 3] has mean 2.0
        assert result["mean"][0] == pytest.approx(2.0)
        # First window has min 1, max 3
        assert result["min"][0] == pytest.approx(1.0)
        assert result["max"][0] == pytest.approx(3.0)

    def test_running_stats_with_trace(self, waveform_trace: WaveformTrace) -> None:
        """Test running stats with WaveformTrace input."""
        result = running_stats(waveform_trace, window_size=50)

        assert "mean" in result
        assert "std" in result
        assert "min" in result
        assert "max" in result


# =============================================================================
# Summary Stats Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.analyzer
class TestSummaryStatsComprehensive:
    """Comprehensive tests for summary_stats function."""

    def test_summary_stats_includes_all_metrics(self, normal_data: np.ndarray) -> None:
        """Test summary_stats includes all expected metrics."""
        summary = summary_stats(normal_data)

        # From basic_stats
        assert "mean" in summary
        assert "variance" in summary
        assert "std" in summary
        assert "min" in summary
        assert "max" in summary
        assert "range" in summary
        assert "count" in summary

        # From quartiles
        assert "q1" in summary
        assert "median" in summary
        assert "q3" in summary
        assert "iqr" in summary

        # Additional metrics
        assert "median_abs_dev" in summary
        assert "rms" in summary
        assert "peak_to_rms" in summary

    def test_summary_stats_mad_calculation(self, normal_data: np.ndarray) -> None:
        """Test median absolute deviation calculation."""
        summary = summary_stats(normal_data)

        median = np.median(normal_data)
        expected_mad = np.median(np.abs(normal_data - median))

        assert summary["median_abs_dev"] == pytest.approx(expected_mad)

    def test_summary_stats_rms_calculation(self) -> None:
        """Test RMS calculation."""
        data = np.array([3.0, 4.0])  # RMS = sqrt((9+16)/2) = sqrt(12.5) ≈ 3.536
        summary = summary_stats(data)

        expected_rms = np.sqrt(np.mean(data**2))
        assert summary["rms"] == pytest.approx(expected_rms, rel=1e-6)

    def test_summary_stats_peak_to_rms(self, normal_data: np.ndarray) -> None:
        """Test peak-to-RMS ratio."""
        summary = summary_stats(normal_data)

        expected_ratio = summary["max"] / summary["rms"]
        assert summary["peak_to_rms"] == pytest.approx(expected_ratio)

    def test_summary_stats_zero_rms(self) -> None:
        """Test summary stats with zero RMS (all zeros)."""
        data = np.zeros(100)
        summary = summary_stats(data)

        assert summary["rms"] == 0.0
        assert np.isnan(summary["peak_to_rms"])

    def test_summary_stats_with_trace(self, waveform_trace: WaveformTrace) -> None:
        """Test summary_stats with WaveformTrace input."""
        summary = summary_stats(waveform_trace)

        assert isinstance(summary, dict)
        assert len(summary) > 10  # Should have many metrics


# =============================================================================
# Edge Cases
# =============================================================================


@pytest.mark.unit
@pytest.mark.analyzer
class TestStatisticalBasicComprehensiveEdgeCases:
    """Tests for edge cases."""

    def test_nan_values(self) -> None:
        """Test handling of NaN values."""
        data = np.array([1.0, 2.0, np.nan, 4.0, 5.0])

        # Functions may propagate NaN or handle gracefully
        try:
            stats = basic_stats(data)
            # If it works, NaN should propagate
            assert "mean" in stats
        except (ValueError, RuntimeWarning):
            pass  # Acceptable to raise error

    def test_inf_values(self) -> None:
        """Test handling of infinity values."""
        data = np.array([1.0, 2.0, np.inf, 4.0, 5.0])

        try:
            stats = basic_stats(data)
            assert "mean" in stats
        except (ValueError, RuntimeWarning):
            pass

    def test_very_large_dataset(self) -> None:
        """Test performance with large dataset."""
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 1_000_000)

        stats = basic_stats(data)
        assert stats["count"] == 1_000_000

    def test_mixed_dtype_array(self) -> None:
        """Test with integer array (converted to float)."""
        data = np.array([1, 2, 3, 4, 5])
        stats = basic_stats(data)

        assert isinstance(stats["mean"], float)
