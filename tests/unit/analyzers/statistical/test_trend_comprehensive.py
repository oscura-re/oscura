"""Comprehensive tests for trend analysis module.

This module provides comprehensive coverage for:
- src/oscura/analyzers/statistics/trend.py
"""

from __future__ import annotations

import numpy as np
import pytest

from oscura.analyzers.statistics.trend import (
    TrendResult,
    change_point_detection,
    detect_drift_segments,
    detect_trend,
    detrend,
    moving_average,
    piecewise_linear_fit,
)
from oscura.core.types import TraceMetadata, WaveformTrace

pytestmark = [pytest.mark.unit, pytest.mark.analyzer, pytest.mark.statistical]


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def linear_trend_data() -> np.ndarray:
    """Generate data with linear trend."""
    rng = np.random.default_rng(42)
    t = np.linspace(0, 1, 1000)
    return 2 * t + 1 + 0.1 * rng.normal(0, 1, 1000)


@pytest.fixture
def no_trend_data() -> np.ndarray:
    """Generate data with no trend."""
    rng = np.random.default_rng(42)
    return rng.normal(0, 1, 1000)


@pytest.fixture
def linear_trace(linear_trend_data: np.ndarray) -> WaveformTrace:
    """Create WaveformTrace with linear trend."""
    return WaveformTrace(
        data=linear_trend_data,
        metadata=TraceMetadata(sample_rate=1000.0),
    )


# =============================================================================
# Detect Trend Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.analyzer
class TestDetectTrendComprehensive:
    """Comprehensive tests for detect_trend function."""

    def test_detect_trend_array_input(self, linear_trend_data: np.ndarray) -> None:
        """Test detect_trend with numpy array input."""
        result = detect_trend(linear_trend_data, sample_rate=1000.0)

        assert isinstance(result, TrendResult)
        assert result.slope == pytest.approx(2.0, rel=0.1)

    def test_detect_trend_array_requires_sample_rate(self, linear_trend_data: np.ndarray) -> None:
        """Test that array input requires sample_rate."""
        with pytest.raises(ValueError, match="sample_rate required"):
            detect_trend(linear_trend_data)

    def test_detect_trend_very_short_data(self) -> None:
        """Test detect_trend with very short data (< 3 points)."""
        data = np.array([1.0, 2.0])
        result = detect_trend(data, sample_rate=100.0)

        # Should return NaN values for insufficient data
        assert np.isnan(result.slope)
        assert np.isnan(result.p_value)
        assert not result.is_significant

    def test_detect_trend_constant_data(self) -> None:
        """Test detect_trend with constant data."""
        data = np.ones(100)
        result = detect_trend(data, sample_rate=100.0)

        # Slope should be zero or very small
        assert abs(result.slope) < 1e-10

    def test_detect_trend_significance_level(self, linear_trace: WaveformTrace) -> None:
        """Test custom significance level."""
        result_strict = detect_trend(linear_trace, significance_level=0.01)
        result_lenient = detect_trend(linear_trace, significance_level=0.10)

        # Both should detect the obvious trend
        assert result_strict.is_significant
        assert result_lenient.is_significant

    def test_detect_trend_negative_slope(self) -> None:
        """Test detecting negative trend."""
        t = np.linspace(0, 1, 100)
        data = -2 * t + 5  # Negative slope
        result = detect_trend(data, sample_rate=100.0)

        assert result.slope < 0
        assert result.slope == pytest.approx(-2.0, rel=0.05)


# =============================================================================
# Detrend Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.analyzer
class TestDetrendComprehensive:
    """Comprehensive tests for detrend function."""

    def test_detrend_constant_method(self, linear_trend_data: np.ndarray) -> None:
        """Test constant detrending (mean removal)."""
        detrended = detrend(linear_trend_data, method="constant")

        # Mean should be near zero
        assert abs(np.mean(detrended)) < 1e-10

    def test_detrend_linear_array_input(self, linear_trend_data: np.ndarray) -> None:
        """Test linear detrending with array input."""
        detrended = detrend(linear_trend_data, method="linear", sample_rate=1000.0)

        assert isinstance(detrended, np.ndarray)
        assert len(detrended) == len(linear_trend_data)

    def test_detrend_polynomial_order(self) -> None:
        """Test polynomial detrending with different orders."""
        t = np.linspace(0, 1, 100)
        data = t**3 + 0.1 * np.random.randn(100)  # Cubic trend

        # Order 1 won't remove cubic well
        detrended_1 = detrend(data, method="polynomial", order=1)
        # Order 3 should remove it well
        detrended_3 = detrend(data, method="polynomial", order=3)

        # Cubic fit should have lower variance
        assert np.var(detrended_3) < np.var(detrended_1)

    def test_detrend_unknown_method(self, linear_trend_data: np.ndarray) -> None:
        """Test detrend with unknown method raises error."""
        with pytest.raises(ValueError, match="Unknown method"):
            detrend(linear_trend_data, method="unknown")  # type: ignore[arg-type]

    def test_detrend_return_trend_sum(self, linear_trace: WaveformTrace) -> None:
        """Test that detrended + trend = original."""
        detrended, trend = detrend(linear_trace, return_trend=True)

        reconstructed = detrended + trend
        np.testing.assert_allclose(reconstructed, linear_trace.data, rtol=1e-10)

    def test_detrend_preserves_length(self, linear_trace: WaveformTrace) -> None:
        """Test detrend preserves data length."""
        detrended = detrend(linear_trace, method="linear")

        assert len(detrended) == len(linear_trace.data)


# =============================================================================
# Moving Average Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.analyzer
class TestMovingAverageComprehensive:
    """Comprehensive tests for moving_average function."""

    def test_moving_average_simple_calculation(self) -> None:
        """Test simple moving average calculation."""
        data = np.array([1, 2, 3, 4, 5], dtype=float)
        result = moving_average(data, window_size=3, method="simple")

        # Should have same length as input
        assert len(result) == len(data)

    def test_moving_average_exponential_alpha(self) -> None:
        """Test exponential moving average with different alpha."""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)

        # Higher alpha = more responsive
        ema_high = moving_average(data, window_size=5, method="exponential", alpha=0.8)
        ema_low = moving_average(data, window_size=5, method="exponential", alpha=0.2)

        # Both should be monotonic increasing for this data
        assert np.all(np.diff(ema_high) > 0)
        assert np.all(np.diff(ema_low) > 0)

    def test_moving_average_weighted_method(self) -> None:
        """Test weighted moving average."""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        result = moving_average(data, window_size=3, method="weighted")

        assert len(result) == len(data)

    def test_moving_average_unknown_method(self, no_trend_data: np.ndarray) -> None:
        """Test moving average with unknown method raises error."""
        with pytest.raises(ValueError, match="Unknown method"):
            moving_average(no_trend_data, window_size=10, method="unknown")  # type: ignore[arg-type]

    def test_moving_average_window_larger_than_data(self) -> None:
        """Test moving average when window exceeds data length."""
        data = np.array([1, 2, 3, 4, 5], dtype=float)
        result = moving_average(data, window_size=10, method="simple")

        # Should clamp window to data length
        assert len(result) == len(data)

    def test_moving_average_zero_window(self, no_trend_data: np.ndarray) -> None:
        """Test moving average with zero or negative window."""
        result = moving_average(no_trend_data, window_size=0, method="simple")

        # Should return copy of original data
        np.testing.assert_array_equal(result, no_trend_data)

    def test_moving_average_reduces_variance(self, no_trend_data: np.ndarray) -> None:
        """Test that moving average smooths noise."""
        smoothed = moving_average(no_trend_data, window_size=20, method="simple")

        # Smoothed should have lower variance
        assert np.var(smoothed) < np.var(no_trend_data)


# =============================================================================
# Drift Segments Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.analyzer
class TestDetectDriftSegmentsComprehensive:
    """Comprehensive tests for detect_drift_segments function."""

    def test_detect_drift_segments_array_input(self) -> None:
        """Test drift detection with array input."""
        # Create data with drift in middle
        data = np.concatenate(
            [
                np.zeros(300),
                np.linspace(0, 5, 300),  # Drift segment
                np.full(400, 5.0),
            ]
        )

        segments = detect_drift_segments(data, segment_size=200, sample_rate=1000.0)

        # Should detect at least one drift segment
        assert len(segments) >= 1

    def test_detect_drift_segments_threshold_slope(self) -> None:
        """Test threshold_slope parameter."""
        t = np.linspace(0, 1, 1000)
        data = 0.5 * t  # Gentle slope

        # High threshold should find nothing
        segments_high = detect_drift_segments(
            data,
            segment_size=200,
            threshold_slope=10.0,
            sample_rate=1000.0,
        )
        assert len(segments_high) == 0

        # Low threshold should find drift
        segments_low = detect_drift_segments(
            data,
            segment_size=200,
            threshold_slope=0.1,
            sample_rate=1000.0,
        )
        assert len(segments_low) >= 1

    def test_detect_drift_segments_result_structure(self) -> None:
        """Test result structure of drift segments."""
        t = np.linspace(0, 1, 1000)
        data = 2 * t

        segments = detect_drift_segments(data, segment_size=300, sample_rate=1000.0)

        for seg in segments:
            assert "start_sample" in seg
            assert "end_sample" in seg
            assert "start_time" in seg
            assert "end_time" in seg
            assert "slope" in seg
            assert "r_squared" in seg
            assert "p_value" in seg

    def test_detect_drift_segments_requires_sample_rate(
        self, linear_trend_data: np.ndarray
    ) -> None:
        """Test that array input requires sample_rate."""
        with pytest.raises(ValueError, match="sample_rate required"):
            detect_drift_segments(linear_trend_data, segment_size=100)

    def test_detect_drift_segments_very_short_segments(self) -> None:
        """Test with segments too short for regression."""
        data = np.random.randn(50)

        # Segments of size 5 with minimum 10 points needed
        segments = detect_drift_segments(data, segment_size=5, sample_rate=100.0)

        # Should find no segments (too short)
        assert len(segments) == 0


# =============================================================================
# Change Point Detection Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.analyzer
class TestChangePointDetectionComprehensive:
    """Comprehensive tests for change_point_detection function."""

    def test_change_point_detection_level_changes(self) -> None:
        """Test detection of level changes."""
        data = np.concatenate(
            [
                np.zeros(100),
                np.ones(100) * 5,
                np.zeros(100),
            ]
        )

        change_points = change_point_detection(data, min_segment_size=20)

        # Should detect changes around 100 and 200
        assert len(change_points) >= 1

    def test_change_point_detection_penalty_parameter(self) -> None:
        """Test penalty parameter effect."""
        data = np.concatenate([np.zeros(100), np.ones(100) * 5])

        # High penalty = fewer change points
        cp_high = change_point_detection(data, penalty=100.0)
        # Low penalty = more change points
        cp_low = change_point_detection(data, penalty=0.1)

        assert len(cp_high) <= len(cp_low)

    def test_change_point_detection_auto_penalty(self) -> None:
        """Test automatic penalty selection."""
        data = np.concatenate([np.zeros(100), np.ones(100) * 5])

        change_points = change_point_detection(data, penalty=None)

        # Should work and return some results
        assert isinstance(change_points, list)

    def test_change_point_detection_too_short(self) -> None:
        """Test with data too short for detection."""
        data = np.array([1.0, 2.0, 3.0])

        change_points = change_point_detection(data, min_segment_size=10)

        # Should return empty list
        assert len(change_points) == 0

    def test_change_point_detection_no_changes(self) -> None:
        """Test with constant data (no change points)."""
        data = np.ones(200)

        change_points = change_point_detection(data)

        # Should find no change points
        assert len(change_points) == 0


# =============================================================================
# Piecewise Linear Fit Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.analyzer
class TestPiecewiseLinearFitComprehensive:
    """Comprehensive tests for piecewise_linear_fit function."""

    def test_piecewise_fit_result_structure(self) -> None:
        """Test result structure."""
        data = np.linspace(0, 10, 1000) + np.random.randn(1000) * 0.1

        result = piecewise_linear_fit(data, n_segments=3, sample_rate=1000.0)

        assert "breakpoints" in result
        assert "segments" in result
        assert "fitted" in result
        assert "residuals" in result
        assert "rmse" in result

    def test_piecewise_fit_breakpoint_count(self) -> None:
        """Test number of breakpoints."""
        data = np.random.randn(1000)

        result = piecewise_linear_fit(data, n_segments=5, sample_rate=1000.0)

        # n_segments means n+1 breakpoints (including 0 and n)
        assert len(result["breakpoints"]) == 6
        assert len(result["segments"]) == 5

    def test_piecewise_fit_segment_fields(self) -> None:
        """Test segment fields."""
        data = np.linspace(0, 10, 100)

        result = piecewise_linear_fit(data, n_segments=2, sample_rate=100.0)

        for seg in result["segments"]:
            assert "slope" in seg
            assert "intercept" in seg
            assert "start" in seg
            assert "end" in seg

    def test_piecewise_fit_reconstruction(self) -> None:
        """Test fitted + residuals = original."""
        data = np.linspace(0, 10, 100) + np.random.randn(100) * 0.1

        result = piecewise_linear_fit(data, n_segments=3, sample_rate=100.0)

        reconstructed = result["fitted"] + result["residuals"]
        np.testing.assert_allclose(reconstructed, data, rtol=1e-10)

    def test_piecewise_fit_requires_sample_rate(self) -> None:
        """Test that array input requires sample_rate."""
        data = np.random.randn(100)

        with pytest.raises(ValueError, match="sample_rate required"):
            piecewise_linear_fit(data, n_segments=2)

    def test_piecewise_fit_single_segment(self) -> None:
        """Test with single segment (equivalent to linear fit)."""
        t = np.linspace(0, 1, 100)
        data = 2 * t + 1

        result = piecewise_linear_fit(data, n_segments=1, sample_rate=100.0)

        # Single segment should fit the line well
        assert result["rmse"] < 0.01


# =============================================================================
# Edge Cases
# =============================================================================


@pytest.mark.unit
@pytest.mark.analyzer
class TestStatisticalTrendComprehensiveEdgeCases:
    """Tests for edge cases."""

    def test_nan_handling(self) -> None:
        """Test handling of NaN values."""
        data = np.array([1.0, 2.0, np.nan, 4.0, 5.0])

        # May propagate NaN or raise error
        try:
            result = detect_trend(data, sample_rate=100.0)
            assert isinstance(result, TrendResult)
        except (ValueError, RuntimeWarning):
            pass

    def test_inf_handling(self) -> None:
        """Test handling of infinity values."""
        data = np.array([1.0, 2.0, np.inf, 4.0, 5.0])

        try:
            result = detect_trend(data, sample_rate=100.0)
            assert isinstance(result, TrendResult)
        except (ValueError, RuntimeWarning):
            pass

    def test_very_large_dataset(self) -> None:
        """Test with large dataset."""
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 100_000)

        result = detect_trend(data, sample_rate=1000.0)
        assert isinstance(result, TrendResult)
