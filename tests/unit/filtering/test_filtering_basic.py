"""Tests for Oscura filtering module.

Tests filter design, application, and introspection functionality.
"""

import numpy as np
import pytest

from oscura.core.exceptions import AnalysisError
from oscura.core.types import TraceMetadata, WaveformTrace
from oscura.filtering import (
    BandPassFilter,
    BandStopFilter,
    FilterIntrospection,
    HighPassFilter,
    LowPassFilter,
    band_pass,
    design_filter,
    design_filter_spec,
    high_pass,
    low_pass,
    matched_filter,
    median_filter,
    moving_average,
    notch_filter,
    savgol_filter,
)
from oscura.filtering.base import FIRFilter, IIRFilter

pytestmark = pytest.mark.unit


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_rate() -> float:
    """Standard sample rate for tests."""
    return 10_000.0  # 10 kHz


@pytest.fixture
def noisy_sine(sample_rate: float) -> WaveformTrace:
    """Generate a 100 Hz sine wave with high-frequency noise."""
    duration = 0.1  # 100 ms
    n_samples = int(sample_rate * duration)
    t = np.arange(n_samples) / sample_rate

    # 100 Hz signal with 1 kHz noise
    signal = np.sin(2 * np.pi * 100 * t)
    noise = 0.2 * np.sin(2 * np.pi * 1000 * t)

    return WaveformTrace(
        data=signal + noise,
        metadata=TraceMetadata(sample_rate=sample_rate),
    )


@pytest.fixture
def square_wave(sample_rate: float) -> WaveformTrace:
    """Generate a 100 Hz square wave."""
    duration = 0.05
    n_samples = int(sample_rate * duration)
    t = np.arange(n_samples) / sample_rate
    data = np.sign(np.sin(2 * np.pi * 100 * t))

    return WaveformTrace(
        data=data.astype(np.float64),
        metadata=TraceMetadata(sample_rate=sample_rate),
    )


# =============================================================================
# Filter Design Tests
# =============================================================================


class TestFilterDesign:
    """Tests for filter design functions."""

    def test_lowpass_filter_creation(self, sample_rate: float):
        """Test basic low-pass filter creation."""
        lpf = LowPassFilter(cutoff=500, sample_rate=sample_rate, order=4)

        assert lpf.order == 4  # 4th order Butterworth -> 2 SOS sections -> order 4
        assert lpf.sample_rate == sample_rate
        assert lpf.is_stable

    def test_highpass_filter_creation(self, sample_rate: float):
        """Test high-pass filter creation."""
        hpf = HighPassFilter(cutoff=500, sample_rate=sample_rate, order=4)

        assert hpf.is_stable
        assert hpf.order > 0

    def test_bandpass_filter_creation(self, sample_rate: float):
        """Test band-pass filter creation."""
        bpf = BandPassFilter(low=200, high=800, sample_rate=sample_rate, order=4)

        assert bpf.is_stable
        assert bpf.passband == (200, 800)

    def test_bandstop_filter_creation(self, sample_rate: float):
        """Test band-stop filter creation."""
        bsf = BandStopFilter(low=450, high=550, sample_rate=sample_rate, order=4)

        assert bsf.is_stable

    def test_design_filter_butterworth(self, sample_rate: float):
        """Test design_filter with Butterworth type."""
        filt = design_filter("butterworth", cutoff=1000, sample_rate=sample_rate, order=4)

        assert isinstance(filt, IIRFilter)
        assert filt.is_stable

    def test_design_filter_chebyshev(self, sample_rate: float):
        """Test design_filter with Chebyshev type."""
        filt = design_filter(
            "chebyshev1", cutoff=1000, sample_rate=sample_rate, order=4, ripple_db=0.5
        )

        assert filt.is_stable

    def test_design_filter_bessel(self, sample_rate: float):
        """Test design_filter with Bessel type."""
        filt = design_filter("bessel", cutoff=1000, sample_rate=sample_rate, order=4)

        assert filt.is_stable

    def test_design_filter_elliptic(self, sample_rate: float):
        """Test design_filter with Elliptic type."""
        filt = design_filter(
            "elliptic",
            cutoff=1000,
            sample_rate=sample_rate,
            order=4,
            ripple_db=0.5,
            stopband_atten_db=40,
        )

        assert filt.is_stable

    def test_design_filter_spec_auto_order(self, sample_rate: float):
        """Test automatic order calculation from specs."""
        filt = design_filter_spec(
            passband=500,
            stopband=1000,
            sample_rate=sample_rate,
            passband_ripple=1.0,
            stopband_atten=40.0,
        )

        assert filt.is_stable
        assert filt.order > 0

    def test_cutoff_above_nyquist_raises(self, sample_rate: float):
        """Test that cutoff above Nyquist raises error."""
        with pytest.raises(AnalysisError, match="Nyquist"):
            LowPassFilter(cutoff=6000, sample_rate=sample_rate)


# =============================================================================
# Filter Application Tests
# =============================================================================


class TestFilterApplication:
    """Tests for filter application."""

    def test_lowpass_removes_high_frequency(self, noisy_sine: WaveformTrace):
        """Test that low-pass filter removes high-frequency noise."""
        # Filter at 500 Hz (should pass 100 Hz, block 1 kHz)
        filtered = low_pass(noisy_sine, cutoff=300)

        # Check that high-frequency component is reduced
        original_fft = np.abs(np.fft.rfft(noisy_sine.data))
        filtered_fft = np.abs(np.fft.rfft(filtered.data))

        # Find 1 kHz bin (noise frequency)
        freq_resolution = noisy_sine.metadata.sample_rate / len(noisy_sine.data)
        noise_bin = int(1000 / freq_resolution)

        # Noise should be significantly reduced
        assert filtered_fft[noise_bin] < 0.1 * original_fft[noise_bin]

    def test_highpass_removes_dc(self, sample_rate: float):
        """Test that high-pass filter removes DC component."""
        # Signal with DC offset
        n = 1000
        t = np.arange(n) / sample_rate
        data = 5.0 + np.sin(2 * np.pi * 100 * t)  # DC = 5V

        trace = WaveformTrace(
            data=data,
            metadata=TraceMetadata(sample_rate=sample_rate),
        )

        filtered = high_pass(trace, cutoff=10)

        # DC should be removed
        assert abs(np.mean(filtered.data)) < 0.5

    def test_bandpass_isolates_frequency(self, sample_rate: float):
        """Test that band-pass isolates target frequency."""
        n = 2000
        t = np.arange(n) / sample_rate

        # Three frequencies: 100, 500, 2000 Hz
        data = (
            np.sin(2 * np.pi * 100 * t) + np.sin(2 * np.pi * 500 * t) + np.sin(2 * np.pi * 2000 * t)
        )

        trace = WaveformTrace(
            data=data,
            metadata=TraceMetadata(sample_rate=sample_rate),
        )

        # Band-pass around 500 Hz
        filtered = band_pass(trace, low=300, high=700)

        # Check FFT
        fft = np.abs(np.fft.rfft(filtered.data))
        freqs = np.fft.rfftfreq(n, 1 / sample_rate)

        # 500 Hz should be dominant
        peak_idx = np.argmax(fft[1:]) + 1  # Skip DC
        peak_freq = freqs[peak_idx]
        assert abs(peak_freq - 500) < 50

    def test_filter_preserves_length(self, noisy_sine: WaveformTrace):
        """Test that filtering preserves trace length."""
        filtered = low_pass(noisy_sine, cutoff=500)

        assert len(filtered.data) == len(noisy_sine.data)

    def test_filter_with_return_details(self, noisy_sine: WaveformTrace):
        """Test filter application with return_details=True."""
        lpf = LowPassFilter(
            cutoff=500,
            sample_rate=noisy_sine.metadata.sample_rate,
            order=4,
        )

        result = lpf.apply(noisy_sine, return_details=True)

        assert hasattr(result, "trace")
        assert hasattr(result, "transfer_function")
        assert hasattr(result, "impulse_response")
        assert result.transfer_function is not None


# =============================================================================
# Convenience Filter Tests
# =============================================================================


class TestConvenienceFilters:
    """Tests for convenience filtering functions."""

    def test_moving_average(self, noisy_sine: WaveformTrace):
        """Test moving average filter."""
        smoothed = moving_average(noisy_sine, window_size=11)

        # Should reduce noise variance
        assert np.std(smoothed.data) < np.std(noisy_sine.data)

    def test_median_filter_removes_spikes(self, sample_rate: float):
        """Test that median filter removes impulse noise."""
        n = 100
        data = np.zeros(n)
        data[50] = 100.0  # Spike

        trace = WaveformTrace(
            data=data,
            metadata=TraceMetadata(sample_rate=sample_rate),
        )

        filtered = median_filter(trace, kernel_size=5)

        # Spike should be removed
        assert np.max(filtered.data) < 10

    def test_savgol_filter_smoothing(self, noisy_sine: WaveformTrace):
        """Test Savitzky-Golay filter."""
        smoothed = savgol_filter(noisy_sine, window_length=11, polyorder=3)

        # Should reduce high-frequency content
        assert len(smoothed.data) == len(noisy_sine.data)

    def test_matched_filter_detection(self, sample_rate: float):
        """Test matched filter for pulse detection."""
        n = 200
        data = np.zeros(n)

        # Insert a pulse
        pulse = np.array([0, 0.5, 1.0, 0.5, 0])
        data[50:55] = pulse
        data[100:105] = pulse

        trace = WaveformTrace(
            data=data,
            metadata=TraceMetadata(sample_rate=sample_rate),
        )

        matched = matched_filter(trace, pulse)

        # Should have peaks at pulse locations
        peaks = np.where(matched.data > 0.8 * np.max(matched.data))[0]
        assert len(peaks) >= 2

    def test_notch_filter_removes_frequency(self, sample_rate: float):
        """Test notch filter removes specific frequency."""
        n = 1000
        t = np.arange(n) / sample_rate

        # Signal with 60 Hz interference
        data = np.sin(2 * np.pi * 100 * t) + 0.5 * np.sin(2 * np.pi * 60 * t)

        trace = WaveformTrace(
            data=data,
            metadata=TraceMetadata(sample_rate=sample_rate),
        )

        filtered = notch_filter(trace, freq=60, q_factor=30)

        # 60 Hz component should be reduced
        fft_before = np.abs(np.fft.rfft(trace.data))
        fft_after = np.abs(np.fft.rfft(filtered.data))

        freq_res = sample_rate / n
        bin_60hz = int(60 / freq_res)

        assert fft_after[bin_60hz] < 0.3 * fft_before[bin_60hz]


# =============================================================================
# Filter Introspection Tests
# =============================================================================


class TestFilterIntrospection:
    """Tests for filter introspection functionality."""

    def test_frequency_response(self, sample_rate: float):
        """Test frequency response calculation."""
        lpf = LowPassFilter(cutoff=1000, sample_rate=sample_rate, order=4)

        w, h = lpf.get_frequency_response(512)

        assert len(w) == 512
        assert len(h) == 512

        # Magnitude should be ~1 at DC
        assert abs(np.abs(h[0]) - 1.0) < 0.1

    def test_impulse_response(self, sample_rate: float):
        """Test impulse response calculation."""
        lpf = LowPassFilter(cutoff=1000, sample_rate=sample_rate, order=4)

        h = lpf.get_impulse_response(256)

        assert len(h) == 256
        # Impulse response should decay
        assert np.abs(h[-1]) < np.abs(h[0])

    def test_step_response(self, sample_rate: float):
        """Test step response calculation."""
        lpf = LowPassFilter(cutoff=1000, sample_rate=sample_rate, order=4)

        s = lpf.get_step_response(256)

        assert len(s) == 256
        # Step response should approach 1
        assert abs(s[-1] - 1.0) < 0.1

    def test_poles_and_zeros(self, sample_rate: float):
        """Test pole-zero extraction."""
        lpf = LowPassFilter(cutoff=1000, sample_rate=sample_rate, order=4)

        poles = lpf.poles

        # Should have poles and zeros
        assert len(poles) > 0

        # All poles should be inside unit circle (stable)
        assert np.all(np.abs(poles) < 1.0)

    def test_filter_introspection_class(self, sample_rate: float):
        """Test FilterIntrospection mixin."""
        lpf = LowPassFilter(cutoff=1000, sample_rate=sample_rate, order=4)
        introspect = FilterIntrospection(lpf)

        _freqs, mag = introspect.magnitude_response(db=True)
        _freqs2, phase = introspect.phase_response(degrees=True)

        assert len(mag) == 512
        assert len(phase) == 512

        # Magnitude should decrease with frequency (low-pass)
        assert mag[-1] < mag[0]


# =============================================================================
# FIR Filter Tests
# =============================================================================


class TestFIRFilter:
    """Tests for FIR filter functionality."""

    def test_fir_filter_creation(self, sample_rate: float):
        """Test FIR filter creation with coefficients."""
        coeffs = np.ones(11) / 11  # Simple moving average
        fir = FIRFilter(sample_rate=sample_rate, coeffs=coeffs)

        assert fir.order == 10
        assert fir.is_stable  # FIR always stable
        assert fir.is_linear_phase

    def test_fir_filter_application(self, noisy_sine: WaveformTrace):
        """Test FIR filter application."""
        coeffs = np.ones(11) / 11
        fir = FIRFilter(
            sample_rate=noisy_sine.metadata.sample_rate,
            coeffs=coeffs,
        )

        filtered = fir.apply(noisy_sine)

        # Should smooth the signal
        assert isinstance(filtered, WaveformTrace)
