"""Unit tests for filtering.filters convenience functions.

Tests all convenience filter functions from oscura.filtering.filters module.
Covers filter application, parameter validation, edge cases, and error handling.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from oscura.core.exceptions import AnalysisError
from oscura.core.types import TraceMetadata, WaveformTrace
from oscura.filtering import filters

pytestmark = pytest.mark.unit


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_rate() -> float:
    """Standard sample rate for tests."""
    return 10_000.0  # 10 kHz


@pytest.fixture
def test_trace(sample_rate: float) -> WaveformTrace:
    """Generate a test waveform trace with multiple frequency components."""
    n = 1000
    t = np.arange(n) / sample_rate
    # 100 Hz sine + 1 kHz noise + 50 Hz low frequency
    data = (
        np.sin(2 * np.pi * 100 * t)
        + 0.3 * np.sin(2 * np.pi * 1000 * t)
        + 0.2 * np.sin(2 * np.pi * 50 * t)
    )
    return WaveformTrace(
        data=data.astype(np.float64),
        metadata=TraceMetadata(sample_rate=sample_rate),
    )


@pytest.fixture
def short_trace(sample_rate: float) -> WaveformTrace:
    """Generate a short test trace for edge case testing."""
    n = 20
    t = np.arange(n) / sample_rate
    data = np.sin(2 * np.pi * 100 * t)
    return WaveformTrace(
        data=data.astype(np.float64),
        metadata=TraceMetadata(sample_rate=sample_rate),
    )


@pytest.fixture
def noisy_trace(sample_rate: float) -> WaveformTrace:
    """Generate a trace with impulse noise for median filter testing."""
    n = 1000
    t = np.arange(n) / sample_rate
    data = np.sin(2 * np.pi * 100 * t)
    # Add impulse noise
    data[100] = 10.0
    data[500] = -10.0
    data[800] = 8.0
    return WaveformTrace(
        data=data.astype(np.float64),
        metadata=TraceMetadata(sample_rate=sample_rate),
    )


@pytest.fixture
def pulse_template() -> NDArray[np.floating]:
    """Generate a pulse template for matched filter testing."""
    return np.array([0.0, 0.5, 1.0, 0.5, 0.0], dtype=np.float64)


# =============================================================================
# Low Pass Filter Tests
# =============================================================================


@pytest.mark.unit
class TestLowPassFilter:
    """Test low_pass convenience function."""

    def test_low_pass_basic(self, test_trace: WaveformTrace):
        """Test basic low-pass filtering."""
        result = filters.low_pass(test_trace, cutoff=500.0)

        assert isinstance(result, WaveformTrace)
        assert len(result.data) == len(test_trace.data)
        assert result.metadata == test_trace.metadata

    def test_low_pass_removes_high_freq(self, test_trace: WaveformTrace):
        """Test that low-pass removes high frequency components."""
        # Low cutoff should remove 1kHz component
        result = filters.low_pass(test_trace, cutoff=200.0)

        # Signal should be smoother (lower variance)
        assert np.var(result.data) < np.var(test_trace.data)

    def test_low_pass_different_orders(self, test_trace: WaveformTrace):
        """Test low-pass filter with different orders."""
        result_2 = filters.low_pass(test_trace, cutoff=500.0, order=2)
        result_8 = filters.low_pass(test_trace, cutoff=500.0, order=8)

        assert isinstance(result_2, WaveformTrace)
        assert isinstance(result_8, WaveformTrace)
        # Higher order = steeper rolloff, different results
        assert not np.allclose(result_2.data, result_8.data)

    def test_low_pass_different_filter_types(self, test_trace: WaveformTrace):
        """Test low-pass filter with different filter types."""
        butter = filters.low_pass(test_trace, cutoff=500.0, filter_type="butterworth")
        cheby1 = filters.low_pass(test_trace, cutoff=500.0, filter_type="chebyshev1")
        bessel = filters.low_pass(test_trace, cutoff=500.0, filter_type="bessel")

        assert isinstance(butter, WaveformTrace)
        assert isinstance(cheby1, WaveformTrace)
        assert isinstance(bessel, WaveformTrace)
        # Different filter types produce different results
        assert not np.allclose(butter.data, cheby1.data)
        assert not np.allclose(butter.data, bessel.data)

    def test_low_pass_preserves_dc(self, test_trace: WaveformTrace):
        """Test that low-pass filter preserves DC component."""
        # Add DC offset
        trace_with_dc = WaveformTrace(
            data=test_trace.data + 5.0,
            metadata=test_trace.metadata,
        )
        result = filters.low_pass(trace_with_dc, cutoff=500.0)

        # Mean should be preserved (approximately)
        assert np.abs(np.mean(result.data) - np.mean(trace_with_dc.data)) < 0.1


# =============================================================================
# High Pass Filter Tests
# =============================================================================


@pytest.mark.unit
class TestHighPassFilter:
    """Test high_pass convenience function."""

    def test_high_pass_basic(self, test_trace: WaveformTrace):
        """Test basic high-pass filtering."""
        result = filters.high_pass(test_trace, cutoff=200.0)

        assert isinstance(result, WaveformTrace)
        assert len(result.data) == len(test_trace.data)
        assert result.metadata == test_trace.metadata

    def test_high_pass_removes_low_freq(self, test_trace: WaveformTrace):
        """Test that high-pass removes low frequency components."""
        # High cutoff should remove 50Hz and 100Hz components
        result = filters.high_pass(test_trace, cutoff=500.0)

        # Filtered signal should have lower amplitude
        assert np.max(np.abs(result.data)) < np.max(np.abs(test_trace.data))

    def test_high_pass_removes_dc(self, sample_rate: float):
        """Test that high-pass filter removes DC component."""
        # Create signal with DC offset
        n = 1000
        t = np.arange(n) / sample_rate
        data = np.sin(2 * np.pi * 100 * t) + 5.0  # DC offset of 5.0
        trace = WaveformTrace(
            data=data.astype(np.float64),
            metadata=TraceMetadata(sample_rate=sample_rate),
        )

        result = filters.high_pass(trace, cutoff=10.0)

        # Mean should be close to zero
        assert np.abs(np.mean(result.data)) < 0.5

    def test_high_pass_different_orders(self, test_trace: WaveformTrace):
        """Test high-pass filter with different orders."""
        result_2 = filters.high_pass(test_trace, cutoff=200.0, order=2)
        result_8 = filters.high_pass(test_trace, cutoff=200.0, order=8)

        assert isinstance(result_2, WaveformTrace)
        assert isinstance(result_8, WaveformTrace)
        assert not np.allclose(result_2.data, result_8.data)


# =============================================================================
# Band Pass Filter Tests
# =============================================================================


@pytest.mark.unit
class TestBandPassFilter:
    """Test band_pass convenience function."""

    def test_band_pass_basic(self, test_trace: WaveformTrace):
        """Test basic band-pass filtering."""
        result = filters.band_pass(test_trace, low=80.0, high=120.0)

        assert isinstance(result, WaveformTrace)
        assert len(result.data) == len(test_trace.data)
        assert result.metadata == test_trace.metadata

    def test_band_pass_isolates_frequency(self, test_trace: WaveformTrace):
        """Test that band-pass isolates specific frequency range."""
        # Filter to isolate 100Hz component
        result = filters.band_pass(test_trace, low=80.0, high=120.0)

        # Should preserve 100Hz while removing 50Hz and 1kHz
        # Power should be less than original
        assert np.var(result.data) < np.var(test_trace.data)

    def test_band_pass_different_bands(self, test_trace: WaveformTrace):
        """Test band-pass with different frequency bands."""
        low_band = filters.band_pass(test_trace, low=40.0, high=60.0)  # 50Hz
        mid_band = filters.band_pass(test_trace, low=80.0, high=120.0)  # 100Hz
        high_band = filters.band_pass(test_trace, low=900.0, high=1100.0)  # 1kHz

        # Different bands should produce different results
        assert not np.allclose(low_band.data, mid_band.data)
        assert not np.allclose(mid_band.data, high_band.data)

    def test_band_pass_narrow_band(self, test_trace: WaveformTrace):
        """Test band-pass with narrow bandwidth."""
        result = filters.band_pass(test_trace, low=95.0, high=105.0)

        assert isinstance(result, WaveformTrace)
        # Narrow band should still work
        assert len(result.data) == len(test_trace.data)

    def test_band_pass_wide_band(self, test_trace: WaveformTrace):
        """Test band-pass with wide bandwidth."""
        result = filters.band_pass(test_trace, low=10.0, high=2000.0)

        assert isinstance(result, WaveformTrace)
        # Wide band should pass most of signal
        assert np.corrcoef(result.data, test_trace.data)[0, 1] > 0.9


# =============================================================================
# Band Stop Filter Tests
# =============================================================================


@pytest.mark.unit
class TestBandStopFilter:
    """Test band_stop convenience function."""

    def test_band_stop_basic(self, test_trace: WaveformTrace):
        """Test basic band-stop filtering."""
        result = filters.band_stop(test_trace, low=80.0, high=120.0)

        assert isinstance(result, WaveformTrace)
        assert len(result.data) == len(test_trace.data)
        assert result.metadata == test_trace.metadata

    def test_band_stop_removes_frequency(self, test_trace: WaveformTrace):
        """Test that band-stop removes specific frequency range."""
        # Remove 100Hz component
        result = filters.band_stop(test_trace, low=80.0, high=120.0)

        # Power should be reduced
        assert np.var(result.data) < np.var(test_trace.data)

    def test_band_stop_narrow_notch(self, test_trace: WaveformTrace):
        """Test band-stop with narrow notch (wider to avoid instability)."""
        # Use wider notch to avoid filter instability
        result = filters.band_stop(test_trace, low=80.0, high=120.0)

        assert isinstance(result, WaveformTrace)
        # Should remove 100Hz but preserve 50Hz and 1kHz
        assert len(result.data) == len(test_trace.data)

    def test_band_stop_different_filter_types(self, test_trace: WaveformTrace):
        """Test band-stop filter with different filter types."""
        butter = filters.band_stop(test_trace, low=80.0, high=120.0, filter_type="butterworth")
        cheby1 = filters.band_stop(test_trace, low=80.0, high=120.0, filter_type="chebyshev1")

        assert isinstance(butter, WaveformTrace)
        assert isinstance(cheby1, WaveformTrace)
        assert not np.allclose(butter.data, cheby1.data)


# =============================================================================
# Notch Filter Tests
# =============================================================================


@pytest.mark.unit
class TestNotchFilter:
    """Test notch_filter convenience function."""

    def test_notch_filter_basic(self, test_trace: WaveformTrace):
        """Test basic notch filtering."""
        result = filters.notch_filter(test_trace, freq=100.0)

        assert isinstance(result, WaveformTrace)
        assert len(result.data) == len(test_trace.data)
        assert result.metadata == test_trace.metadata

    def test_notch_filter_removes_frequency(self, test_trace: WaveformTrace):
        """Test that notch filter removes target frequency."""
        # Remove 100Hz component
        result = filters.notch_filter(test_trace, freq=100.0, q_factor=30.0)

        # Power should be reduced
        assert np.var(result.data) < np.var(test_trace.data)

    def test_notch_filter_different_q_factors(self, test_trace: WaveformTrace):
        """Test notch filter with different Q factors."""
        result_low_q = filters.notch_filter(test_trace, freq=100.0, q_factor=10.0)
        result_high_q = filters.notch_filter(test_trace, freq=100.0, q_factor=50.0)

        # Different Q factors produce different results
        assert not np.allclose(result_low_q.data, result_high_q.data)

    def test_notch_filter_60hz(self, sample_rate: float):
        """Test notch filter for 60Hz line noise removal."""
        # Create signal with 60Hz interference
        n = 1000
        t = np.arange(n) / sample_rate
        data = np.sin(2 * np.pi * 100 * t) + 0.5 * np.sin(2 * np.pi * 60 * t)
        trace = WaveformTrace(
            data=data.astype(np.float64),
            metadata=TraceMetadata(sample_rate=sample_rate),
        )

        result = filters.notch_filter(trace, freq=60.0, q_factor=30.0)

        assert isinstance(result, WaveformTrace)
        # 60Hz component should be attenuated
        assert np.var(result.data) < np.var(trace.data)

    def test_notch_filter_freq_too_high_raises(self, test_trace: WaveformTrace):
        """Test that notch filter raises when frequency >= Nyquist."""
        with pytest.raises(AnalysisError, match="must be less than Nyquist"):
            filters.notch_filter(test_trace, freq=6000.0)  # > Nyquist (5000Hz)

    def test_notch_filter_near_nyquist(self, test_trace: WaveformTrace):
        """Test notch filter near but below Nyquist frequency."""
        # Should work at frequency below Nyquist
        result = filters.notch_filter(test_trace, freq=4500.0, q_factor=30.0)

        assert isinstance(result, WaveformTrace)


# =============================================================================
# Moving Average Filter Tests
# =============================================================================


@pytest.mark.unit
class TestMovingAverageFilter:
    """Test moving_average convenience function."""

    def test_moving_average_basic(self, test_trace: WaveformTrace):
        """Test basic moving average filtering."""
        result = filters.moving_average(test_trace, window_size=11)

        assert isinstance(result, WaveformTrace)
        assert len(result.data) == len(test_trace.data)  # mode='same' default
        assert result.metadata == test_trace.metadata

    def test_moving_average_smooths_signal(self, test_trace: WaveformTrace):
        """Test that moving average smooths the signal."""
        result = filters.moving_average(test_trace, window_size=21)

        # Smoothed signal should have lower high-frequency content
        # Use gradient as proxy for smoothness
        original_gradient = np.diff(test_trace.data)
        filtered_gradient = np.diff(result.data)
        assert np.var(filtered_gradient) < np.var(original_gradient)

    def test_moving_average_different_window_sizes(self, test_trace: WaveformTrace):
        """Test moving average with different window sizes."""
        result_5 = filters.moving_average(test_trace, window_size=5)
        result_21 = filters.moving_average(test_trace, window_size=21)

        # Larger window = more smoothing
        gradient_5 = np.diff(result_5.data)
        gradient_21 = np.diff(result_21.data)
        assert np.var(gradient_21) < np.var(gradient_5)

    def test_moving_average_mode_same(self, test_trace: WaveformTrace):
        """Test moving average with mode='same'."""
        result = filters.moving_average(test_trace, window_size=11, mode="same")

        assert len(result.data) == len(test_trace.data)

    def test_moving_average_mode_valid(self, test_trace: WaveformTrace):
        """Test moving average with mode='valid'."""
        window_size = 11
        result = filters.moving_average(test_trace, window_size=window_size, mode="valid")

        expected_len = len(test_trace.data) - window_size + 1
        assert len(result.data) == expected_len

    def test_moving_average_mode_full(self, test_trace: WaveformTrace):
        """Test moving average with mode='full'."""
        window_size = 11
        result = filters.moving_average(test_trace, window_size=window_size, mode="full")

        expected_len = len(test_trace.data) + window_size - 1
        assert len(result.data) == expected_len

    def test_moving_average_window_size_1(self, test_trace: WaveformTrace):
        """Test moving average with window size 1 (no filtering)."""
        result = filters.moving_average(test_trace, window_size=1)

        # Window size 1 should be identity
        assert np.allclose(result.data, test_trace.data)

    def test_moving_average_negative_window_raises(self, test_trace: WaveformTrace):
        """Test that negative window size raises error."""
        with pytest.raises(AnalysisError, match="must be positive"):
            filters.moving_average(test_trace, window_size=-5)

    def test_moving_average_zero_window_raises(self, test_trace: WaveformTrace):
        """Test that zero window size raises error."""
        with pytest.raises(AnalysisError, match="must be positive"):
            filters.moving_average(test_trace, window_size=0)

    def test_moving_average_window_too_large_raises(self, short_trace: WaveformTrace):
        """Test that window larger than data raises error."""
        with pytest.raises(AnalysisError, match="exceeds data length"):
            filters.moving_average(short_trace, window_size=100)


# =============================================================================
# Median Filter Tests
# =============================================================================


@pytest.mark.unit
class TestMedianFilter:
    """Test median_filter convenience function."""

    def test_median_filter_basic(self, test_trace: WaveformTrace):
        """Test basic median filtering."""
        result = filters.median_filter(test_trace, kernel_size=5)

        assert isinstance(result, WaveformTrace)
        assert len(result.data) == len(test_trace.data)
        assert result.metadata == test_trace.metadata

    def test_median_filter_removes_impulse_noise(self, noisy_trace: WaveformTrace):
        """Test that median filter removes impulse noise."""
        result = filters.median_filter(noisy_trace, kernel_size=7)

        # Peak values should be reduced
        assert np.max(np.abs(result.data)) < np.max(np.abs(noisy_trace.data))

    def test_median_filter_different_kernel_sizes(self, noisy_trace: WaveformTrace):
        """Test median filter with different kernel sizes."""
        result_3 = filters.median_filter(noisy_trace, kernel_size=3)
        result_9 = filters.median_filter(noisy_trace, kernel_size=9)

        # Different kernel sizes produce different results
        assert not np.allclose(result_3.data, result_9.data)

    def test_median_filter_preserves_edges(self, sample_rate: float):
        """Test that median filter preserves edges better than linear filters."""
        # Create step signal
        data = np.concatenate([np.ones(500) * -1, np.ones(500) * 1])
        trace = WaveformTrace(
            data=data.astype(np.float64),
            metadata=TraceMetadata(sample_rate=sample_rate),
        )

        result = filters.median_filter(trace, kernel_size=5)

        # Step should be preserved (approximately)
        assert np.abs(result.data[250]) > 0.8  # Before step
        assert np.abs(result.data[750]) > 0.8  # After step

    def test_median_filter_kernel_size_1(self, test_trace: WaveformTrace):
        """Test median filter with kernel size 1 (no filtering)."""
        result = filters.median_filter(test_trace, kernel_size=1)

        # Kernel size 1 should be identity
        assert np.allclose(result.data, test_trace.data)

    def test_median_filter_negative_kernel_raises(self, test_trace: WaveformTrace):
        """Test that negative kernel size raises error."""
        with pytest.raises(AnalysisError, match="must be positive"):
            filters.median_filter(test_trace, kernel_size=-5)

    def test_median_filter_even_kernel_raises(self, test_trace: WaveformTrace):
        """Test that even kernel size raises error."""
        with pytest.raises(AnalysisError, match="must be odd"):
            filters.median_filter(test_trace, kernel_size=6)


# =============================================================================
# Savitzky-Golay Filter Tests
# =============================================================================


@pytest.mark.unit
class TestSavgolFilter:
    """Test savgol_filter convenience function."""

    def test_savgol_filter_basic(self, test_trace: WaveformTrace):
        """Test basic Savitzky-Golay filtering."""
        result = filters.savgol_filter(test_trace, window_length=11, polyorder=3)

        assert isinstance(result, WaveformTrace)
        assert len(result.data) == len(test_trace.data)
        assert result.metadata == test_trace.metadata

    def test_savgol_filter_smooths_signal(self, test_trace: WaveformTrace):
        """Test that Savitzky-Golay smooths the signal."""
        result = filters.savgol_filter(test_trace, window_length=21, polyorder=3)

        # Smoothed signal should have lower variance
        assert np.var(result.data) < np.var(test_trace.data)

    def test_savgol_filter_different_window_lengths(self, test_trace: WaveformTrace):
        """Test Savitzky-Golay with different window lengths."""
        result_11 = filters.savgol_filter(test_trace, window_length=11, polyorder=3)
        result_21 = filters.savgol_filter(test_trace, window_length=21, polyorder=3)

        # Different windows produce different results
        assert not np.allclose(result_11.data, result_21.data)

    def test_savgol_filter_different_polyorders(self, test_trace: WaveformTrace):
        """Test Savitzky-Golay with different polynomial orders."""
        result_2 = filters.savgol_filter(test_trace, window_length=11, polyorder=2)
        result_4 = filters.savgol_filter(test_trace, window_length=11, polyorder=4)

        # Different polynomial orders produce different results
        assert not np.allclose(result_2.data, result_4.data)

    def test_savgol_filter_derivative(self, test_trace: WaveformTrace):
        """Test Savitzky-Golay filter for derivative calculation."""
        result = filters.savgol_filter(test_trace, window_length=11, polyorder=3, deriv=1)

        assert isinstance(result, WaveformTrace)
        # Derivative should have different characteristics
        assert not np.allclose(result.data, test_trace.data)

    def test_savgol_filter_second_derivative(self, test_trace: WaveformTrace):
        """Test Savitzky-Golay filter for second derivative."""
        result = filters.savgol_filter(test_trace, window_length=11, polyorder=3, deriv=2)

        assert isinstance(result, WaveformTrace)

    def test_savgol_filter_even_window_raises(self, test_trace: WaveformTrace):
        """Test that even window length raises error."""
        with pytest.raises(AnalysisError, match="must be odd"):
            filters.savgol_filter(test_trace, window_length=10, polyorder=3)

    def test_savgol_filter_polyorder_too_high_raises(self, test_trace: WaveformTrace):
        """Test that polyorder >= window_length raises error."""
        with pytest.raises(AnalysisError, match="must be less than window length"):
            filters.savgol_filter(test_trace, window_length=11, polyorder=11)


# =============================================================================
# Matched Filter Tests
# =============================================================================


@pytest.mark.unit
class TestMatchedFilter:
    """Test matched_filter convenience function."""

    def test_matched_filter_basic(
        self, test_trace: WaveformTrace, pulse_template: NDArray[np.floating]
    ):
        """Test basic matched filtering."""
        result = filters.matched_filter(test_trace, template=pulse_template)

        assert isinstance(result, WaveformTrace)
        assert len(result.data) == len(test_trace.data)
        assert result.metadata == test_trace.metadata

    def test_matched_filter_detects_pulse(
        self, sample_rate: float, pulse_template: NDArray[np.floating]
    ):
        """Test that matched filter detects known pulse."""
        # Create signal with known pulse at position 500
        n = 1000
        data = np.zeros(n)
        pulse_pos = 500
        data[pulse_pos : pulse_pos + len(pulse_template)] = pulse_template
        trace = WaveformTrace(
            data=data.astype(np.float64),
            metadata=TraceMetadata(sample_rate=sample_rate),
        )

        result = filters.matched_filter(trace, template=pulse_template, normalize=True)

        # Peak should be near pulse position
        peak_pos = np.argmax(np.abs(result.data))
        assert abs(peak_pos - pulse_pos) < 10

    def test_matched_filter_normalize_true(
        self, test_trace: WaveformTrace, pulse_template: NDArray[np.floating]
    ):
        """Test matched filter with normalization."""
        result = filters.matched_filter(test_trace, template=pulse_template, normalize=True)

        assert isinstance(result, WaveformTrace)

    def test_matched_filter_normalize_false(
        self, test_trace: WaveformTrace, pulse_template: NDArray[np.floating]
    ):
        """Test matched filter without normalization."""
        result = filters.matched_filter(test_trace, template=pulse_template, normalize=False)

        assert isinstance(result, WaveformTrace)

    def test_matched_filter_different_templates(self, test_trace: WaveformTrace):
        """Test matched filter with different templates."""
        template1 = np.array([0, 1, 0], dtype=np.float64)
        template2 = np.array([0, 0.5, 1.0, 0.5, 0], dtype=np.float64)

        result1 = filters.matched_filter(test_trace, template=template1)
        result2 = filters.matched_filter(test_trace, template=template2)

        # Different templates produce different results
        assert not np.allclose(result1.data, result2.data)

    def test_matched_filter_empty_template_raises(self, test_trace: WaveformTrace):
        """Test that empty template raises error."""
        with pytest.raises(AnalysisError, match="cannot be empty"):
            filters.matched_filter(test_trace, template=np.array([]))

    def test_matched_filter_template_too_long_raises(self, short_trace: WaveformTrace):
        """Test that template longer than data raises error."""
        long_template = np.ones(100, dtype=np.float64)
        with pytest.raises(AnalysisError, match="exceeds data length"):
            filters.matched_filter(short_trace, template=long_template)


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


@pytest.mark.unit
class TestFilteringFiltersEdgeCases:
    """Test edge cases and error conditions."""

    def test_filters_on_zero_signal(self, sample_rate: float):
        """Test all filters on zero signal."""
        data = np.zeros(100)
        trace = WaveformTrace(
            data=data.astype(np.float64),
            metadata=TraceMetadata(sample_rate=sample_rate),
        )

        # All filters should handle zero signal
        result_lp = filters.low_pass(trace, cutoff=100.0)
        result_hp = filters.high_pass(trace, cutoff=100.0)
        result_bp = filters.band_pass(trace, low=50.0, high=150.0)
        result_bs = filters.band_stop(trace, low=50.0, high=150.0)
        result_ma = filters.moving_average(trace, window_size=5)
        result_med = filters.median_filter(trace, kernel_size=5)
        result_sg = filters.savgol_filter(trace, window_length=11, polyorder=3)

        assert np.allclose(result_lp.data, 0)
        assert np.allclose(result_hp.data, 0)
        assert np.allclose(result_bp.data, 0)
        assert np.allclose(result_bs.data, 0)
        assert np.allclose(result_ma.data, 0)
        assert np.allclose(result_med.data, 0)
        assert np.allclose(result_sg.data, 0)

    def test_filters_on_constant_signal(self, sample_rate: float):
        """Test filters on constant signal."""
        data = np.ones(100) * 3.14
        trace = WaveformTrace(
            data=data.astype(np.float64),
            metadata=TraceMetadata(sample_rate=sample_rate),
        )

        # Moving average should preserve constant (except edge effects)
        result_ma = filters.moving_average(trace, window_size=5)
        # Check middle values (avoid edge effects from mode='same')
        assert np.allclose(result_ma.data[10:90], 3.14, atol=0.01)

        # Median filter should preserve constant
        result_med = filters.median_filter(trace, kernel_size=5)
        assert np.allclose(result_med.data, 3.14)

        # Savitzky-Golay should preserve constant (except edge effects)
        result_sg = filters.savgol_filter(trace, window_length=11, polyorder=2)
        # Check middle values to avoid edge effects
        assert np.allclose(result_sg.data[15:85], 3.14, atol=0.01)

    def test_filters_on_single_sample(self, sample_rate: float):
        """Test filters on single sample trace."""
        data = np.array([1.0])
        trace = WaveformTrace(
            data=data.astype(np.float64),
            metadata=TraceMetadata(sample_rate=sample_rate),
        )

        # Moving average with window size 1
        result_ma = filters.moving_average(trace, window_size=1)
        assert np.allclose(result_ma.data, 1.0)

        # Median filter with kernel size 1
        result_med = filters.median_filter(trace, kernel_size=1)
        assert np.allclose(result_med.data, 1.0)

    def test_filters_preserve_dtype(self, test_trace: WaveformTrace):
        """Test that filters preserve float64 dtype."""
        result_lp = filters.low_pass(test_trace, cutoff=500.0)
        result_hp = filters.high_pass(test_trace, cutoff=100.0)
        result_ma = filters.moving_average(test_trace, window_size=11)
        result_med = filters.median_filter(test_trace, kernel_size=5)

        assert result_lp.data.dtype == np.float64
        assert result_hp.data.dtype == np.float64
        assert result_ma.data.dtype == np.float64
        assert result_med.data.dtype == np.float64

    def test_filters_preserve_metadata(self, test_trace: WaveformTrace):
        """Test that filters preserve trace metadata."""
        result = filters.low_pass(test_trace, cutoff=500.0)

        assert result.metadata == test_trace.metadata
        assert result.metadata.sample_rate == test_trace.metadata.sample_rate

    def test_very_short_trace_handling(self, sample_rate: float):
        """Test filter behavior on very short traces."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        trace = WaveformTrace(
            data=data.astype(np.float64),
            metadata=TraceMetadata(sample_rate=sample_rate),
        )

        # Filters should handle short traces gracefully
        result_ma = filters.moving_average(trace, window_size=3)
        result_med = filters.median_filter(trace, kernel_size=3)
        result_sg = filters.savgol_filter(trace, window_length=5, polyorder=2)

        assert len(result_ma.data) == len(trace.data)
        assert len(result_med.data) == len(trace.data)
        assert len(result_sg.data) == len(trace.data)


# =============================================================================
# Filter Cascading Tests
# =============================================================================


@pytest.mark.unit
class TestFilterCascading:
    """Test cascading multiple filters."""

    def test_cascade_low_pass_high_pass(self, test_trace: WaveformTrace):
        """Test cascading low-pass and high-pass filters (band-pass equivalent)."""
        # First low-pass, then high-pass
        result = filters.low_pass(test_trace, cutoff=1500.0)
        result = filters.high_pass(result, cutoff=50.0)

        assert isinstance(result, WaveformTrace)
        assert len(result.data) == len(test_trace.data)

    def test_cascade_filter_and_smooth(self, test_trace: WaveformTrace):
        """Test cascading frequency filter and smoothing."""
        result = filters.low_pass(test_trace, cutoff=500.0)
        result = filters.moving_average(result, window_size=5)

        assert isinstance(result, WaveformTrace)
        # Cascaded filters should smooth more
        assert np.var(result.data) < np.var(test_trace.data)

    def test_cascade_notch_and_smooth(self, test_trace: WaveformTrace):
        """Test cascading notch filter and smoothing."""
        result = filters.notch_filter(test_trace, freq=100.0)
        result = filters.savgol_filter(result, window_length=11, polyorder=3)

        assert isinstance(result, WaveformTrace)

    def test_cascade_median_then_linear(self, noisy_trace: WaveformTrace):
        """Test cascading median (remove impulses) then linear smoothing."""
        result = filters.median_filter(noisy_trace, kernel_size=5)
        result = filters.moving_average(result, window_size=11)

        assert isinstance(result, WaveformTrace)
        # Should remove impulses and smooth
        assert np.max(np.abs(result.data)) < np.max(np.abs(noisy_trace.data))

    def test_cascade_multiple_notches(self, sample_rate: float):
        """Test cascading multiple notch filters."""
        # Create signal with multiple interference frequencies
        n = 1000
        t = np.arange(n) / sample_rate
        data = (
            np.sin(2 * np.pi * 100 * t)
            + 0.3 * np.sin(2 * np.pi * 60 * t)
            + 0.3 * np.sin(2 * np.pi * 120 * t)
        )
        trace = WaveformTrace(
            data=data.astype(np.float64),
            metadata=TraceMetadata(sample_rate=sample_rate),
        )

        # Remove both 60Hz and 120Hz
        result = filters.notch_filter(trace, freq=60.0)
        result = filters.notch_filter(result, freq=120.0)

        assert isinstance(result, WaveformTrace)
        # Power should be reduced after removing interference
        assert np.var(result.data) < np.var(trace.data)


# =============================================================================
# Module Exports Tests
# =============================================================================


@pytest.mark.unit
class TestModuleExports:
    """Test that filters module exports correct functions."""

    def test_module_has_low_pass(self):
        """Test that filters module exports low_pass."""
        assert hasattr(filters, "low_pass")
        assert callable(filters.low_pass)

    def test_module_has_high_pass(self):
        """Test that filters module exports high_pass."""
        assert hasattr(filters, "high_pass")
        assert callable(filters.high_pass)

    def test_module_has_band_pass(self):
        """Test that filters module exports band_pass."""
        assert hasattr(filters, "band_pass")
        assert callable(filters.band_pass)

    def test_module_has_band_stop(self):
        """Test that filters module exports band_stop."""
        assert hasattr(filters, "band_stop")
        assert callable(filters.band_stop)

    def test_module_has_notch_filter(self):
        """Test that filters module exports notch_filter."""
        assert hasattr(filters, "notch_filter")
        assert callable(filters.notch_filter)

    def test_module_has_moving_average(self):
        """Test that filters module exports moving_average."""
        assert hasattr(filters, "moving_average")
        assert callable(filters.moving_average)

    def test_module_has_median_filter(self):
        """Test that filters module exports median_filter."""
        assert hasattr(filters, "median_filter")
        assert callable(filters.median_filter)

    def test_module_has_savgol_filter(self):
        """Test that filters module exports savgol_filter."""
        assert hasattr(filters, "savgol_filter")
        assert callable(filters.savgol_filter)

    def test_module_has_matched_filter(self):
        """Test that filters module exports matched_filter."""
        assert hasattr(filters, "matched_filter")
        assert callable(filters.matched_filter)

    def test_module_all_exports(self):
        """Test that __all__ contains expected exports."""
        expected = [
            "band_pass",
            "band_stop",
            "high_pass",
            "low_pass",
            "matched_filter",
            "median_filter",
            "moving_average",
            "notch_filter",
            "savgol_filter",
        ]
        assert set(filters.__all__) == set(expected)


# =============================================================================
# Additional Convenience Functions Tests (not in filters.py __all__)
# =============================================================================


@pytest.mark.unit
class TestAdditionalConvenienceFunctions:
    """Test additional convenience functions from convenience.py."""

    def test_exponential_moving_average_basic(self, test_trace: WaveformTrace):
        """Test exponential moving average filter."""
        from oscura.filtering.convenience import exponential_moving_average

        result = exponential_moving_average(test_trace, alpha=0.1)

        assert isinstance(result, WaveformTrace)
        assert len(result.data) == len(test_trace.data)

    def test_exponential_moving_average_invalid_alpha(self, test_trace: WaveformTrace):
        """Test that invalid alpha raises error."""
        from oscura.filtering.convenience import exponential_moving_average

        with pytest.raises(AnalysisError, match="Alpha must be in"):
            exponential_moving_average(test_trace, alpha=0.0)

        with pytest.raises(AnalysisError, match="Alpha must be in"):
            exponential_moving_average(test_trace, alpha=1.5)

    def test_gaussian_filter_basic(self, test_trace: WaveformTrace):
        """Test Gaussian smoothing filter."""
        from oscura.filtering.convenience import gaussian_filter

        result = gaussian_filter(test_trace, sigma=3.0)

        assert isinstance(result, WaveformTrace)
        assert len(result.data) == len(test_trace.data)

    def test_gaussian_filter_invalid_sigma(self, test_trace: WaveformTrace):
        """Test that invalid sigma raises error."""
        from oscura.filtering.convenience import gaussian_filter

        with pytest.raises(AnalysisError, match="Sigma must be positive"):
            gaussian_filter(test_trace, sigma=0.0)

        with pytest.raises(AnalysisError, match="Sigma must be positive"):
            gaussian_filter(test_trace, sigma=-1.0)

    def test_differentiate_basic(self, test_trace: WaveformTrace):
        """Test numerical differentiation."""
        from oscura.filtering.convenience import differentiate

        result = differentiate(test_trace)

        assert isinstance(result, WaveformTrace)
        assert len(result.data) == len(test_trace.data)

    def test_differentiate_invalid_order(self, test_trace: WaveformTrace):
        """Test that invalid derivative order raises error."""
        from oscura.filtering.convenience import differentiate

        with pytest.raises(AnalysisError, match="must be positive"):
            differentiate(test_trace, order=0)

        with pytest.raises(AnalysisError, match="must be positive"):
            differentiate(test_trace, order=-1)

    def test_integrate_basic(self, test_trace: WaveformTrace):
        """Test numerical integration."""
        from oscura.filtering.convenience import integrate

        result = integrate(test_trace)

        assert isinstance(result, WaveformTrace)
        assert len(result.data) == len(test_trace.data)

    def test_integrate_invalid_method(self, test_trace: WaveformTrace):
        """Test that invalid integration method raises error."""
        from oscura.filtering.convenience import integrate

        with pytest.raises(AnalysisError, match="Unknown integration method"):
            integrate(test_trace, method="invalid")
