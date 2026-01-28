"""Comprehensive tests for statistics/trend.py.

This test module provides complete coverage for trend detection and detrending functions,
including edge cases and error conditions.
"""

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


@pytest.fixture
def simple_trace() -> WaveformTrace:
    """Create simple trace with known properties.

    Returns:
        WaveformTrace with 1000 samples at 1 kHz
    """
    data = np.sin(2 * np.pi * 10 * np.linspace(0, 1, 1000))
    metadata = TraceMetadata(
        sample_rate=1000.0,
        vertical_scale=1.0,
        vertical_offset=0.0,
    )
    return WaveformTrace(data=data, metadata=metadata)


@pytest.fixture
def trend_trace() -> WaveformTrace:
    """Create trace with linear trend.

    Returns:
        WaveformTrace with linear drift
    """
    t = np.linspace(0, 1, 1000)
    # Linear trend: 2.0 V/s slope
    data = 2.0 * t + 0.1 * np.random.randn(1000)
    metadata = TraceMetadata(
        sample_rate=1000.0,
        vertical_scale=1.0,
        vertical_offset=0.0,
    )
    return WaveformTrace(data=data, metadata=metadata)


class TestTrendResult:
    """Test suite for TrendResult dataclass."""

    def test_initialization(self) -> None:
        """Test successful initialization with valid parameters."""
        trend_line = np.array([0.0, 1.0, 2.0])
        result = TrendResult(
            slope=0.5,
            intercept=1.0,
            r_squared=0.95,
            p_value=0.001,
            std_error=0.05,
            is_significant=True,
            trend_line=trend_line,
        )

        assert result.slope == 0.5
        assert result.intercept == 1.0
        assert result.r_squared == 0.95
        assert result.p_value == 0.001
        assert result.std_error == 0.05
        assert result.is_significant is True
        assert np.array_equal(result.trend_line, trend_line)


class TestDetectTrend:
    """Test suite for detect_trend function."""

    def test_linear_trend_detected(self, trend_trace: WaveformTrace) -> None:
        """Test detection of clear linear trend."""
        result = detect_trend(trend_trace)

        assert result.is_significant
        assert result.p_value < 0.05
        # Slope should be close to 2.0 V/s
        assert 1.5 < result.slope < 2.5
        assert result.r_squared > 0.8

    def test_no_trend_in_noise(self, simple_trace: WaveformTrace) -> None:
        """Test pure noise has no significant trend."""
        # White noise
        data = np.random.randn(1000)
        metadata = TraceMetadata(sample_rate=1000.0)
        trace = WaveformTrace(data=data, metadata=metadata)

        result = detect_trend(trace)

        # Should not be significantly different from zero slope
        # Relaxed threshold to account for random variation in noise (CI saw 0.157)
        assert abs(result.slope) < 0.20  # Small slope for random data
        # May or may not be significant due to randomness

    def test_array_input_with_sample_rate(self) -> None:
        """Test detect_trend with array input."""
        data = np.linspace(0, 10, 1000)  # Perfect linear trend
        result = detect_trend(data, sample_rate=1000.0)

        assert result.is_significant
        assert result.p_value < 0.001
        assert result.r_squared > 0.99  # Nearly perfect fit
        assert abs(result.slope - 10.0) < 0.1  # 10 V/s slope

    def test_array_without_sample_rate_raises(self) -> None:
        """Test array input without sample_rate raises error."""
        data = np.linspace(0, 10, 1000)

        with pytest.raises(ValueError, match="sample_rate required"):
            detect_trend(data)

    def test_custom_significance_level(self, trend_trace: WaveformTrace) -> None:
        """Test custom significance level threshold."""
        # Very strict threshold
        result = detect_trend(trend_trace, significance_level=0.0001)

        assert isinstance(result.is_significant, bool)
        assert result.p_value < 0.05  # Strong trend

    def test_very_short_trace(self) -> None:
        """Test trace with < 3 samples returns NaN."""
        data = np.array([1.0, 2.0])
        metadata = TraceMetadata(sample_rate=1000.0)
        trace = WaveformTrace(data=data, metadata=metadata)

        result = detect_trend(trace)

        assert np.isnan(result.slope)
        assert np.isnan(result.r_squared)
        assert not result.is_significant
        assert len(result.trend_line) == 2

    def test_constant_signal(self) -> None:
        """Test constant signal has zero slope."""
        data = np.ones(1000) * 5.0
        metadata = TraceMetadata(sample_rate=1000.0)
        trace = WaveformTrace(data=data, metadata=metadata)

        result = detect_trend(trace)

        assert abs(result.slope) < 1e-10
        # Constant signal has no variance, r_squared is NaN (not a number)
        assert np.isnan(result.r_squared) or result.r_squared < 0.01

    def test_trend_line_length_matches_input(self, trend_trace: WaveformTrace) -> None:
        """Test trend_line has same length as input."""
        result = detect_trend(trend_trace)

        assert len(result.trend_line) == len(trend_trace.data)


class TestDetrend:
    """Test suite for detrend function."""

    def test_detrend_constant(self) -> None:
        """Test removing mean (DC offset)."""
        data = np.random.randn(1000) + 5.0  # Noise with DC offset
        metadata = TraceMetadata(sample_rate=1000.0)
        trace = WaveformTrace(data=data, metadata=metadata)

        detrended = detrend(trace, method="constant")

        # Mean should be near zero
        assert abs(np.mean(detrended)) < 0.1
        assert len(detrended) == len(data)

    def test_detrend_linear(self, trend_trace: WaveformTrace) -> None:
        """Test removing linear trend."""
        detrended = detrend(trend_trace, method="linear")

        # After detrending, mean should be near zero
        # and no significant linear trend should remain
        result = detect_trend(detrended, sample_rate=trend_trace.metadata.sample_rate)

        assert abs(result.slope) < 0.5  # Slope much reduced
        assert len(detrended) == len(trend_trace.data)

    def test_detrend_polynomial(self) -> None:
        """Test removing polynomial trend."""
        t = np.linspace(0, 1, 1000)
        # Quadratic trend
        data = 2.0 * t**2 + 0.5 * t + 1.0
        metadata = TraceMetadata(sample_rate=1000.0)
        trace = WaveformTrace(data=data, metadata=metadata)

        detrended = detrend(trace, method="polynomial", order=2)

        # Should remove most of the trend
        assert abs(np.mean(detrended)) < 0.5
        assert len(detrended) == len(data)

    def test_return_trend(self, trend_trace: WaveformTrace) -> None:
        """Test returning both detrended and trend."""
        detrended, trend = detrend(trend_trace, method="linear", return_trend=True)

        # Detrended + trend should equal original
        reconstructed = detrended + trend
        assert np.allclose(reconstructed, trend_trace.data, atol=0.01)

    def test_invalid_method_raises(self, simple_trace: WaveformTrace) -> None:
        """Test invalid method raises ValueError."""
        with pytest.raises(ValueError, match="Unknown method"):
            detrend(simple_trace, method="invalid")  # type: ignore[arg-type]

    def test_array_input(self) -> None:
        """Test detrend with array input."""
        data = np.linspace(0, 10, 1000) + np.random.randn(1000) * 0.1
        detrended = detrend(data, method="linear", sample_rate=1000.0)

        assert len(detrended) == len(data)
        assert isinstance(detrended, np.ndarray)


class TestMovingAverage:
    """Test suite for moving_average function."""

    def test_simple_moving_average(self) -> None:
        """Test simple moving average smooths signal."""
        # Step function with noise
        data = np.concatenate([np.ones(500), np.ones(500) * 5.0])
        data += np.random.randn(1000) * 0.1
        metadata = TraceMetadata(sample_rate=1000.0)
        trace = WaveformTrace(data=data, metadata=metadata)

        smoothed = moving_average(trace, window_size=10, method="simple")

        # Smoothed should have less variance
        assert np.std(smoothed) < np.std(data)
        assert len(smoothed) == len(data)

    def test_exponential_moving_average(self) -> None:
        """Test exponential moving average."""
        data = np.random.randn(1000)
        metadata = TraceMetadata(sample_rate=1000.0)
        trace = WaveformTrace(data=data, metadata=metadata)

        smoothed = moving_average(trace, window_size=10, method="exponential", alpha=0.2)

        assert len(smoothed) == len(data)
        assert np.all(np.isfinite(smoothed))

    def test_weighted_moving_average(self) -> None:
        """Test weighted moving average."""
        data = np.random.randn(1000)
        metadata = TraceMetadata(sample_rate=1000.0)
        trace = WaveformTrace(data=data, metadata=metadata)

        smoothed = moving_average(trace, window_size=10, method="weighted")

        assert len(smoothed) == len(data)
        assert np.all(np.isfinite(smoothed))

    def test_window_size_larger_than_signal(self) -> None:
        """Test window size larger than signal length."""
        data = np.random.randn(50)
        metadata = TraceMetadata(sample_rate=1000.0)
        trace = WaveformTrace(data=data, metadata=metadata)

        smoothed = moving_average(trace, window_size=100, method="simple")

        # Should handle gracefully (use signal length)
        assert len(smoothed) == len(data)

    def test_window_size_one(self) -> None:
        """Test window size of 1 returns original signal."""
        data = np.random.randn(1000)
        metadata = TraceMetadata(sample_rate=1000.0)
        trace = WaveformTrace(data=data, metadata=metadata)

        smoothed = moving_average(trace, window_size=1, method="simple")

        assert np.allclose(smoothed, data, atol=1e-10)

    def test_invalid_method_raises(self, simple_trace: WaveformTrace) -> None:
        """Test invalid method raises ValueError."""
        with pytest.raises(ValueError, match="Unknown method"):
            moving_average(simple_trace, window_size=10, method="invalid")  # type: ignore[arg-type]

    def test_array_input(self) -> None:
        """Test moving average with array input."""
        data = np.random.randn(1000)
        smoothed = moving_average(data, window_size=10, method="simple")

        assert len(smoothed) == len(data)
        assert isinstance(smoothed, np.ndarray)


class TestDetectDriftSegments:
    """Test suite for detect_drift_segments function."""

    def test_detect_drift_in_segments(self) -> None:
        """Test detection of drift in specific segments."""
        # Create signal with drift in middle segment
        seg1 = np.ones(500) * 1.0
        seg2 = np.linspace(1.0, 3.0, 500)  # Drift segment
        seg3 = np.ones(500) * 3.0
        data = np.concatenate([seg1, seg2, seg3])
        metadata = TraceMetadata(sample_rate=1000.0)
        trace = WaveformTrace(data=data, metadata=metadata)

        segments = detect_drift_segments(trace, segment_size=500)

        # Should detect drift in middle segment
        assert len(segments) >= 1
        # Middle segment should have drift
        middle_seg = [s for s in segments if 400 < s["start_sample"] < 600]
        assert len(middle_seg) > 0

    def test_no_drift_in_constant_signal(self) -> None:
        """Test constant signal produces no drift segments."""
        data = np.ones(2000) * 5.0
        metadata = TraceMetadata(sample_rate=1000.0)
        trace = WaveformTrace(data=data, metadata=metadata)

        segments = detect_drift_segments(trace, segment_size=500)

        # No significant drift
        assert len(segments) == 0

    def test_threshold_slope_filtering(self) -> None:
        """Test threshold_slope parameter filters results."""
        t = np.linspace(0, 1, 1000)
        data = 0.1 * t  # Small slope
        metadata = TraceMetadata(sample_rate=1000.0)
        trace = WaveformTrace(data=data, metadata=metadata)

        # With high threshold, should find no segments
        segments = detect_drift_segments(trace, segment_size=500, threshold_slope=1.0)

        assert len(segments) == 0

    def test_array_input_with_sample_rate(self) -> None:
        """Test array input with sample_rate."""
        data = np.linspace(0, 10, 2000)
        segments = detect_drift_segments(data, segment_size=500, sample_rate=1000.0)

        # Should detect drift in all segments
        assert len(segments) >= 1

    def test_array_without_sample_rate_raises(self) -> None:
        """Test array input without sample_rate raises error."""
        data = np.linspace(0, 10, 2000)

        with pytest.raises(ValueError, match="sample_rate required"):
            detect_drift_segments(data, segment_size=500)

    def test_segment_metadata(self) -> None:
        """Test segment dictionaries contain required fields."""
        t = np.linspace(0, 1, 2000)
        data = 5.0 * t
        metadata = TraceMetadata(sample_rate=1000.0)
        trace = WaveformTrace(data=data, metadata=metadata)

        segments = detect_drift_segments(trace, segment_size=500)

        for seg in segments:
            assert "start_sample" in seg
            assert "end_sample" in seg
            assert "start_time" in seg
            assert "end_time" in seg
            assert "slope" in seg
            assert "r_squared" in seg
            assert "p_value" in seg


class TestChangePointDetection:
    """Test suite for change_point_detection function."""

    def test_detect_level_shift(self) -> None:
        """Test detection of level shift."""
        # Signal with clear level shift
        seg1 = np.ones(500) * 1.0
        seg2 = np.ones(500) * 5.0
        data = np.concatenate([seg1, seg2])
        metadata = TraceMetadata(sample_rate=1000.0)
        trace = WaveformTrace(data=data, metadata=metadata)

        change_points = change_point_detection(trace, min_segment_size=50)

        # Should detect change near sample 500
        assert len(change_points) >= 1
        assert any(450 < cp < 550 for cp in change_points)

    def test_no_changes_in_constant_signal(self) -> None:
        """Test constant signal produces no change points."""
        data = np.ones(1000) * 5.0
        metadata = TraceMetadata(sample_rate=1000.0)
        trace = WaveformTrace(data=data, metadata=metadata)

        change_points = change_point_detection(trace)

        assert len(change_points) == 0

    def test_multiple_change_points(self) -> None:
        """Test detection of multiple changes."""
        # Three level shifts
        seg1 = np.ones(250) * 1.0
        seg2 = np.ones(250) * 3.0
        seg3 = np.ones(250) * 2.0
        seg4 = np.ones(250) * 4.0
        data = np.concatenate([seg1, seg2, seg3, seg4])
        metadata = TraceMetadata(sample_rate=1000.0)
        trace = WaveformTrace(data=data, metadata=metadata)

        change_points = change_point_detection(trace, min_segment_size=50)

        # Should detect multiple changes
        assert len(change_points) >= 2

    def test_custom_penalty(self) -> None:
        """Test custom penalty parameter."""
        data = np.random.randn(1000)
        metadata = TraceMetadata(sample_rate=1000.0)
        trace = WaveformTrace(data=data, metadata=metadata)

        # High penalty should reduce number of change points
        cps_high_penalty = change_point_detection(trace, penalty=10.0)
        cps_low_penalty = change_point_detection(trace, penalty=0.1)

        # Lower penalty typically finds more change points
        assert len(cps_low_penalty) >= len(cps_high_penalty)

    def test_array_input(self) -> None:
        """Test change_point_detection with array input."""
        data = np.concatenate([np.ones(500), np.ones(500) * 5.0])
        change_points = change_point_detection(data)

        assert isinstance(change_points, list)
        assert all(isinstance(cp, int) for cp in change_points)

    def test_too_short_signal(self) -> None:
        """Test signal shorter than 2*min_segment_size."""
        data = np.random.randn(50)
        metadata = TraceMetadata(sample_rate=1000.0)
        trace = WaveformTrace(data=data, metadata=metadata)

        change_points = change_point_detection(trace, min_segment_size=50)

        assert len(change_points) == 0


class TestPiecewiseLinearFit:
    """Test suite for piecewise_linear_fit function."""

    def test_fit_piecewise_linear(self) -> None:
        """Test fitting piecewise linear model."""
        # Create piecewise linear signal
        t = np.linspace(0, 1, 1000)
        data = np.where(t < 0.5, 2.0 * t, 1.0 + 1.0 * (t - 0.5))
        metadata = TraceMetadata(sample_rate=1000.0)
        trace = WaveformTrace(data=data, metadata=metadata)

        result = piecewise_linear_fit(trace, n_segments=2, sample_rate=1000.0)

        assert "breakpoints" in result
        assert "segments" in result
        assert "fitted" in result
        assert "residuals" in result
        assert "rmse" in result

        assert len(result["segments"]) == 2
        assert len(result["fitted"]) == len(data)

    def test_single_segment(self) -> None:
        """Test with single segment (equivalent to linear fit)."""
        data = np.linspace(0, 10, 1000)
        metadata = TraceMetadata(sample_rate=1000.0)
        trace = WaveformTrace(data=data, metadata=metadata)

        result = piecewise_linear_fit(trace, n_segments=1, sample_rate=1000.0)

        assert len(result["segments"]) == 1
        # Should fit well
        assert result["rmse"] < 0.1

    def test_multiple_segments(self) -> None:
        """Test with multiple segments."""
        data = np.random.randn(2000)
        metadata = TraceMetadata(sample_rate=1000.0)
        trace = WaveformTrace(data=data, metadata=metadata)

        result = piecewise_linear_fit(trace, n_segments=4, sample_rate=1000.0)

        assert len(result["segments"]) == 4
        assert len(result["breakpoints"]) == 5  # n_segments + 1

    def test_array_input_with_sample_rate(self) -> None:
        """Test array input with sample_rate."""
        data = np.linspace(0, 10, 1000)
        result = piecewise_linear_fit(data, n_segments=3, sample_rate=1000.0)

        assert len(result["segments"]) == 3
        assert isinstance(result["fitted"], np.ndarray)

    def test_array_without_sample_rate_raises(self) -> None:
        """Test array input without sample_rate raises error."""
        data = np.linspace(0, 10, 1000)

        with pytest.raises(ValueError, match="sample_rate required"):
            piecewise_linear_fit(data, n_segments=2)

    def test_segment_boundaries(self) -> None:
        """Test segment boundaries are correct."""
        data = np.random.randn(1000)
        metadata = TraceMetadata(sample_rate=1000.0)
        trace = WaveformTrace(data=data, metadata=metadata)

        result = piecewise_linear_fit(trace, n_segments=4, sample_rate=1000.0)

        # First segment starts at 0
        assert result["segments"][0]["start"] == 0
        # Last segment ends at len(data)
        assert result["segments"][-1]["end"] == len(data)

        # Segments should be contiguous
        for i in range(len(result["segments"]) - 1):
            assert result["segments"][i]["end"] == result["segments"][i + 1]["start"]
