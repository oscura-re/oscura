"""Tests for correlation and trend analysis functions.

Tests for TASK-032 (Correlation and Trend Analysis).
"""

import numpy as np
import pytest

from oscura.analyzers.statistics.correlation import (
    CrossCorrelationResult,
    autocorrelation,
    coherence,
    correlation_coefficient,
    cross_correlation,
    find_periodicity,
)
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
# Correlation Fixtures
# =============================================================================


@pytest.fixture
def periodic_signal() -> WaveformTrace:
    """Generate a periodic signal for autocorrelation testing."""
    sample_rate = 10000.0
    duration = 1.0
    t = np.arange(0, duration, 1 / sample_rate)
    # 10 Hz sine wave
    data = np.sin(2 * np.pi * 10 * t)
    return WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))


@pytest.fixture
def delayed_signal(periodic_signal: WaveformTrace) -> WaveformTrace:
    """Generate a delayed version of the periodic signal."""
    delay_samples = 100  # 10 ms delay
    data = np.zeros_like(periodic_signal.data)
    data[delay_samples:] = periodic_signal.data[:-delay_samples]
    return WaveformTrace(data=data, metadata=periodic_signal.metadata)


@pytest.fixture
def correlated_signals() -> tuple[WaveformTrace, WaveformTrace]:
    """Generate two correlated signals."""
    sample_rate = 1000.0
    t = np.arange(0, 1, 1 / sample_rate)
    rng = np.random.default_rng(42)

    # Base signal
    base = np.sin(2 * np.pi * 5 * t)

    # Correlated signal = base + noise
    signal1 = base + 0.1 * rng.normal(0, 1, len(t))
    signal2 = base + 0.1 * rng.normal(0, 1, len(t))

    trace1 = WaveformTrace(data=signal1, metadata=TraceMetadata(sample_rate=sample_rate))
    trace2 = WaveformTrace(data=signal2, metadata=TraceMetadata(sample_rate=sample_rate))

    return trace1, trace2


@pytest.fixture
def uncorrelated_signals() -> tuple[WaveformTrace, WaveformTrace]:
    """Generate two uncorrelated signals."""
    rng = np.random.default_rng(42)
    sample_rate = 1000.0
    n = 1000

    signal1 = rng.normal(0, 1, n)
    signal2 = rng.normal(0, 1, n)

    trace1 = WaveformTrace(data=signal1, metadata=TraceMetadata(sample_rate=sample_rate))
    trace2 = WaveformTrace(data=signal2, metadata=TraceMetadata(sample_rate=sample_rate))

    return trace1, trace2


# =============================================================================
# Trend Fixtures
# =============================================================================


@pytest.fixture
def linear_trend_signal() -> WaveformTrace:
    """Generate a signal with linear trend."""
    sample_rate = 1000.0
    t = np.arange(0, 1, 1 / sample_rate)
    rng = np.random.default_rng(42)

    # Linear trend: y = 2*t + 1 + noise
    data = 2 * t + 1 + 0.1 * rng.normal(0, 1, len(t))

    return WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))


@pytest.fixture
def no_trend_signal() -> WaveformTrace:
    """Generate a signal with no trend."""
    sample_rate = 1000.0
    n = 1000
    rng = np.random.default_rng(42)

    data = rng.normal(0, 1, n)

    return WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))


@pytest.fixture
def drift_segments_signal() -> WaveformTrace:
    """Generate a signal with multiple drift segments."""
    sample_rate = 1000.0
    n = 1000
    rng = np.random.default_rng(42)

    data = np.zeros(n)
    # Segment 1: no drift (0-300)
    data[:300] = rng.normal(0, 0.1, 300)
    # Segment 2: positive drift (300-600)
    data[300:600] = np.linspace(0, 2, 300) + rng.normal(0, 0.1, 300)
    # Segment 3: negative drift (600-900)
    data[600:900] = np.linspace(2, 0, 300) + rng.normal(0, 0.1, 300)
    # Segment 4: no drift (900-1000)
    data[900:] = rng.normal(0, 0.1, 100)

    return WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))


# =============================================================================
# Autocorrelation Tests
# =============================================================================


class TestAutocorrelation:
    """Tests for autocorrelation function."""

    def test_autocorrelation_periodic(self, periodic_signal: WaveformTrace):
        """Test autocorrelation of periodic signal."""
        lag_times, acf = autocorrelation(periodic_signal, max_lag=500)

        assert len(lag_times) == len(acf)

        # Autocorrelation at lag 0 should be 1 (normalized)
        assert np.isclose(acf[0], 1.0)

        # Should have peaks at period (0.1s for 10 Hz)
        # Period = 100 samples at 1000 Hz sample rate
        period_samples = 100
        if period_samples < len(acf):
            # Should see peak around the period
            window = acf[period_samples - 10 : period_samples + 10]
            assert np.max(window) > 0.8

    def test_autocorrelation_decays(self, no_trend_signal: WaveformTrace):
        """Test that autocorrelation of white noise decays quickly."""
        _lag_times, acf = autocorrelation(no_trend_signal, max_lag=100)

        # Autocorrelation should decay quickly for white noise
        assert acf[0] == pytest.approx(1.0)
        # At lag > 0, should be near zero
        assert np.mean(np.abs(acf[10:])) < 0.2

    def test_autocorrelation_max_lag(self, periodic_signal: WaveformTrace):
        """Test max_lag parameter."""
        _lag_times, acf = autocorrelation(periodic_signal, max_lag=100)

        assert len(acf) == 101  # 0 to 100 inclusive


# =============================================================================
# Cross-Correlation Tests
# =============================================================================


class TestCrossCorrelation:
    """Tests for cross_correlation function."""

    def test_cross_correlation_delayed(
        self, periodic_signal: WaveformTrace, delayed_signal: WaveformTrace
    ):
        """Test cross-correlation detects delay."""
        result = cross_correlation(periodic_signal, delayed_signal, max_lag=200)

        assert isinstance(result, CrossCorrelationResult)

        # Peak lag should correspond to the delay (100 samples = 10 ms)
        expected_lag_time = 0.01  # 10 ms
        assert np.isclose(result.peak_lag_time, expected_lag_time, rtol=0.1)

        # Peak coefficient should be high
        assert abs(result.peak_coefficient) > 0.8

    def test_cross_correlation_identical(self, periodic_signal: WaveformTrace):
        """Test cross-correlation of identical signals."""
        result = cross_correlation(periodic_signal, periodic_signal)

        # Peak should be at lag 0
        assert result.peak_lag == 0
        # Peak coefficient should be 1.0
        assert np.isclose(result.peak_coefficient, 1.0, rtol=0.01)

    def test_cross_correlation_uncorrelated(
        self, uncorrelated_signals: tuple[WaveformTrace, WaveformTrace]
    ):
        """Test cross-correlation of uncorrelated signals."""
        trace1, trace2 = uncorrelated_signals
        result = cross_correlation(trace1, trace2)

        # Peak coefficient should be low
        assert abs(result.peak_coefficient) < 0.3


# =============================================================================
# Correlation Coefficient Tests
# =============================================================================


class TestCorrelationCoefficient:
    """Tests for correlation_coefficient function."""

    def test_perfect_correlation(self):
        """Test correlation of identical signals."""
        data = np.sin(np.linspace(0, 10, 100))
        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=100))

        r = correlation_coefficient(trace, trace)
        assert np.isclose(r, 1.0)

    def test_perfect_anticorrelation(self):
        """Test correlation of negated signals."""
        data = np.sin(np.linspace(0, 10, 100))
        trace1 = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=100))
        trace2 = WaveformTrace(data=-data, metadata=TraceMetadata(sample_rate=100))

        r = correlation_coefficient(trace1, trace2)
        assert np.isclose(r, -1.0)

    def test_high_correlation(self, correlated_signals: tuple[WaveformTrace, WaveformTrace]):
        """Test correlation of related signals."""
        trace1, trace2 = correlated_signals
        r = correlation_coefficient(trace1, trace2)

        # Should be highly correlated
        assert r > 0.9


# =============================================================================
# Find Periodicity Tests
# =============================================================================


class TestFindPeriodicity:
    """Tests for find_periodicity function."""

    def test_find_periodicity_basic(self, periodic_signal: WaveformTrace):
        """Test finding periodicity in periodic signal."""
        result = find_periodicity(periodic_signal)

        # 10 Hz signal -> 0.1s period
        expected_period = 0.1
        assert np.isclose(result["period_time"], expected_period, rtol=0.05)
        assert np.isclose(result["frequency"], 10.0, rtol=0.05)
        assert result["strength"] > 0.8

    def test_find_periodicity_harmonics(self, periodic_signal: WaveformTrace):
        """Test detection of harmonics."""
        result = find_periodicity(periodic_signal)

        # Pure sine should have harmonics at 2x, 3x period
        result["harmonics"]
        # May or may not detect harmonics depending on signal


# =============================================================================
# Coherence Tests
# =============================================================================


class TestCoherence:
    """Tests for coherence function."""

    def test_coherence_identical(self, periodic_signal: WaveformTrace):
        """Test coherence of identical signals."""
        _freq, coh = coherence(periodic_signal, periodic_signal)

        # Coherence should be 1.0 at all frequencies
        assert np.all(coh > 0.99)

    def test_coherence_correlated(self, correlated_signals: tuple[WaveformTrace, WaveformTrace]):
        """Test coherence of correlated signals."""
        trace1, trace2 = correlated_signals
        freq, coh = coherence(trace1, trace2)

        # Should have high coherence at signal frequency (5 Hz)
        # Find index closest to 5 Hz
        idx_5hz = np.argmin(np.abs(freq - 5))
        assert coh[idx_5hz] > 0.8


# =============================================================================
# Trend Detection Tests
# =============================================================================


class TestDetectTrend:
    """Tests for detect_trend function."""

    def test_detect_linear_trend(self, linear_trend_signal: WaveformTrace):
        """Test detecting linear trend."""
        result = detect_trend(linear_trend_signal)

        assert isinstance(result, TrendResult)

        # Slope should be around 2 (y = 2*t + 1)
        assert np.isclose(result.slope, 2.0, rtol=0.1)

        # Intercept should be around 1
        assert np.isclose(result.intercept, 1.0, rtol=0.2)

        # Should be significant
        assert result.is_significant
        assert result.p_value < 0.05
        assert result.r_squared > 0.9

    def test_detect_no_trend(self, no_trend_signal: WaveformTrace):
        """Test with signal having no trend."""
        result = detect_trend(no_trend_signal)

        # Slope should be near zero
        assert abs(result.slope) < 0.1

        # R-squared should be low
        assert result.r_squared < 0.1


# =============================================================================
# Detrend Tests
# =============================================================================


class TestDetrend:
    """Tests for detrend function."""

    def test_detrend_linear(self, linear_trend_signal: WaveformTrace):
        """Test removing linear trend."""
        detrended = detrend(linear_trend_signal, method="linear")

        # Detrended signal should have near-zero mean trend
        result = detect_trend(
            WaveformTrace(
                data=detrended,
                metadata=linear_trend_signal.metadata,
            )
        )
        assert abs(result.slope) < 0.01

    def test_detrend_constant(self, linear_trend_signal: WaveformTrace):
        """Test removing mean (DC offset)."""
        detrended = detrend(linear_trend_signal, method="constant")

        # Mean should be near zero
        assert abs(np.mean(detrended)) < 0.01

    def test_detrend_polynomial(self):
        """Test polynomial detrending."""
        sample_rate = 1000.0
        t = np.arange(0, 1, 1 / sample_rate)
        # Quadratic trend: y = t^2 + noise
        data = t**2 + 0.01 * np.random.randn(len(t))
        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        detrended = detrend(trace, method="polynomial", order=2)

        # Should remove the quadratic trend
        assert np.std(detrended) < 0.1

    def test_detrend_return_trend(self, linear_trend_signal: WaveformTrace):
        """Test returning both detrended and trend."""
        detrended, trend = detrend(linear_trend_signal, return_trend=True)

        # Original = detrended + trend
        reconstructed = detrended + trend
        assert np.allclose(reconstructed, linear_trend_signal.data, rtol=1e-10)


# =============================================================================
# Moving Average Tests
# =============================================================================


class TestMovingAverage:
    """Tests for moving_average function."""

    def test_moving_average_simple(self):
        """Test simple moving average."""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=1))

        smoothed = moving_average(trace, window_size=3, method="simple")

        # Check length
        assert len(smoothed) == len(data)

    def test_moving_average_exponential(self):
        """Test exponential moving average."""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=1))

        smoothed = moving_average(trace, window_size=3, method="exponential", alpha=0.3)

        assert len(smoothed) == len(data)
        # First value should be unchanged
        assert smoothed[0] == data[0]

    def test_moving_average_smoothing(self, no_trend_signal: WaveformTrace):
        """Test that moving average reduces variance."""
        smoothed = moving_average(no_trend_signal, window_size=10, method="simple")

        # Smoothed should have lower variance
        assert np.var(smoothed) < np.var(no_trend_signal.data)


# =============================================================================
# Drift Segment Detection Tests
# =============================================================================


class TestDriftSegments:
    """Tests for detect_drift_segments function."""

    def test_detect_drift_segments_basic(self, drift_segments_signal: WaveformTrace):
        """Test detecting drift segments."""
        segments = detect_drift_segments(
            drift_segments_signal, segment_size=200, threshold_slope=0.5
        )

        # Should detect at least the strong drift segments
        assert len(segments) >= 2

        for seg in segments:
            assert "start_sample" in seg
            assert "end_sample" in seg
            assert "slope" in seg
            assert "r_squared" in seg


# =============================================================================
# Change Point Detection Tests
# =============================================================================


class TestChangePointDetection:
    """Tests for change_point_detection function."""

    def test_detect_change_points(self):
        """Test detecting change points in signal level."""
        # Signal with abrupt level changes
        data = np.concatenate(
            [
                np.zeros(100),
                np.ones(100) * 5,
                np.zeros(100),
                np.ones(100) * 3,
            ]
        )
        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=1000))

        change_points = change_point_detection(trace, min_segment_size=20)

        # Should detect changes around 100, 200, 300
        assert len(change_points) >= 2


# =============================================================================
# Piecewise Linear Fit Tests
# =============================================================================


class TestPiecewiseLinearFit:
    """Tests for piecewise_linear_fit function."""

    def test_piecewise_fit_basic(self, drift_segments_signal: WaveformTrace):
        """Test piecewise linear fitting."""
        result = piecewise_linear_fit(drift_segments_signal, n_segments=4)

        assert "breakpoints" in result
        assert "segments" in result
        assert "fitted" in result
        assert "residuals" in result
        assert "rmse" in result

        # 4 segments means 5 breakpoints (including 0 and n)
        assert len(result["breakpoints"]) == 5
        assert len(result["segments"]) == 4

        # Fitted should match data length
        assert len(result["fitted"]) == len(drift_segments_signal.data)


# =============================================================================
# Edge Cases
# =============================================================================


class TestStatisticalCorrelationTrendEdgeCases:
    """Tests for edge cases."""

    def test_short_signal(self):
        """Test with very short signal."""
        data = np.array([1.0, 2.0, 3.0])
        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=1))

        detect_trend(trace)
        # Should not crash, may return NaN for some fields

    def test_constant_signal(self):
        """Test with constant signal."""
        data = np.ones(100)
        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=1))

        _lag_times, _acf = autocorrelation(trace)
        # Autocorrelation of constant is undefined or 0

    def test_array_input(self):
        """Test functions with numpy array input."""
        data = np.sin(np.linspace(0, 10, 100))

        # Functions that accept arrays directly
        _lag_times, acf = autocorrelation(data, sample_rate=100)
        assert len(acf) > 0
