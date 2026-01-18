"""Unit tests for filtering introspection module.

Tests filter introspection APIs, frequency response analysis,
visualization functions, and edge cases.
"""

import numpy as np
import pytest
from matplotlib.figure import Figure
from numpy.typing import NDArray
from scipy import signal

from oscura.filtering.base import FIRFilter, IIRFilter
from oscura.filtering.introspection import (
    FilterIntrospection,
    compare_filters,
    plot_bode,
    plot_group_delay,
    plot_impulse,
    plot_poles_zeros,
    plot_step,
)

pytestmark = pytest.mark.unit


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module", autouse=True)
def set_matplotlib_backend():
    """Set matplotlib backend to Agg for non-interactive testing."""
    import matplotlib

    matplotlib.use("Agg")
    yield


@pytest.fixture
def sample_rate() -> float:
    """Standard sample rate for tests."""
    return 10_000.0  # 10 kHz


@pytest.fixture
def butterworth_sos(sample_rate: float) -> NDArray[np.float64]:
    """Create Butterworth IIR filter SOS coefficients."""
    # 4th order Butterworth lowpass at 500 Hz
    sos = signal.butter(4, 500, fs=sample_rate, output="sos")
    return sos


@pytest.fixture
def butterworth_filter(sample_rate: float, butterworth_sos: NDArray[np.float64]) -> IIRFilter:
    """Create Butterworth IIR filter."""
    return IIRFilter(sample_rate=sample_rate, sos=butterworth_sos)


@pytest.fixture
def fir_coeffs() -> NDArray[np.float64]:
    """Create simple FIR filter coefficients (moving average)."""
    return np.ones(11, dtype=np.float64) / 11.0


@pytest.fixture
def fir_filter(sample_rate: float, fir_coeffs: NDArray[np.float64]) -> FIRFilter:
    """Create FIR filter."""
    return FIRFilter(sample_rate=sample_rate, coeffs=fir_coeffs)


@pytest.fixture
def linear_phase_fir() -> NDArray[np.float64]:
    """Create symmetric FIR coefficients for linear phase."""
    coeffs = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1], dtype=np.float64)
    return coeffs / np.sum(coeffs)


@pytest.fixture
def linear_phase_filter(sample_rate: float, linear_phase_fir: NDArray[np.float64]) -> FIRFilter:
    """Create linear phase FIR filter."""
    return FIRFilter(sample_rate=sample_rate, coeffs=linear_phase_fir)


# =============================================================================
# FilterIntrospection Tests
# =============================================================================


@pytest.mark.unit
class TestFilterIntrospection:
    """Test FilterIntrospection class."""

    def test_init_with_iir_filter(self, butterworth_filter: IIRFilter):
        """Test initialization with IIR filter."""
        introspect = FilterIntrospection(butterworth_filter)

        assert introspect.filter is butterworth_filter
        assert introspect._filter is butterworth_filter

    def test_init_with_fir_filter(self, fir_filter: FIRFilter):
        """Test initialization with FIR filter."""
        introspect = FilterIntrospection(fir_filter)

        assert introspect.filter is fir_filter

    def test_filter_property(self, butterworth_filter: IIRFilter):
        """Test filter property getter."""
        introspect = FilterIntrospection(butterworth_filter)

        assert introspect.filter is butterworth_filter

    def test_magnitude_response_default(self, butterworth_filter: IIRFilter):
        """Test magnitude response with default parameters."""
        introspect = FilterIntrospection(butterworth_filter)

        freqs, mag = introspect.magnitude_response()

        assert len(freqs) == 512
        assert len(mag) == 512
        assert freqs[0] == 0
        assert freqs[-1] == butterworth_filter.sample_rate / 2
        # Default is dB
        assert mag[0] < 10  # Should be around 0 dB at DC
        assert mag[-1] < -20  # Should be attenuated at Nyquist

    def test_magnitude_response_linear(self, butterworth_filter: IIRFilter):
        """Test magnitude response in linear scale."""
        introspect = FilterIntrospection(butterworth_filter)

        freqs, mag = introspect.magnitude_response(db=False)

        assert len(freqs) == 512
        assert len(mag) == 512
        # Linear scale: magnitude should be positive
        assert np.all(mag >= 0)
        assert mag[0] > 0.9  # Close to 1 at DC for lowpass

    def test_magnitude_response_custom_freqs(self, butterworth_filter: IIRFilter):
        """Test magnitude response with custom frequencies."""
        introspect = FilterIntrospection(butterworth_filter)
        custom_freqs = np.array([100, 200, 500, 1000], dtype=np.float64)

        freqs, mag = introspect.magnitude_response(custom_freqs)

        assert len(freqs) == len(custom_freqs)
        assert len(mag) == len(custom_freqs)
        assert np.array_equal(freqs, custom_freqs)

    def test_magnitude_response_no_sample_rate_raises(self, butterworth_sos: NDArray[np.float64]):
        """Test magnitude response raises when no sample rate and no freqs."""
        filt = IIRFilter(sample_rate=None, sos=butterworth_sos)
        introspect = FilterIntrospection(filt)

        with pytest.raises(ValueError, match="sample_rate"):
            introspect.magnitude_response()

    def test_magnitude_response_no_sample_rate_with_freqs(
        self, butterworth_sos: NDArray[np.float64]
    ):
        """Test magnitude response raises without sample rate even with custom freqs."""
        from oscura.core.exceptions import AnalysisError

        filt = IIRFilter(sample_rate=None, sos=butterworth_sos)
        introspect = FilterIntrospection(filt)
        custom_freqs = np.array([100, 200, 500], dtype=np.float64)

        # Even with custom freqs, need sample rate for Hz conversion
        with pytest.raises(AnalysisError):
            introspect.magnitude_response(custom_freqs)

    def test_magnitude_response_db_floor(self, butterworth_filter: IIRFilter):
        """Test magnitude response applies dB floor to avoid log(0)."""
        introspect = FilterIntrospection(butterworth_filter)

        freqs, mag = introspect.magnitude_response(db=True)

        # Should not have -inf values
        assert np.all(np.isfinite(mag))
        # Should apply floor at 1e-12
        assert np.all(mag >= 20 * np.log10(1e-12))

    def test_phase_response_default(self, butterworth_filter: IIRFilter):
        """Test phase response with default parameters."""
        introspect = FilterIntrospection(butterworth_filter)

        freqs, phase = introspect.phase_response()

        assert len(freqs) == 512
        assert len(phase) == 512
        # Default is unwrapped and degrees
        assert np.all(np.abs(phase) < 720)  # Should be reasonable

    def test_phase_response_radians(self, butterworth_filter: IIRFilter):
        """Test phase response in radians."""
        introspect = FilterIntrospection(butterworth_filter)

        freqs, phase = introspect.phase_response(degrees=False)

        assert len(freqs) == 512
        assert len(phase) == 512
        # Radians: should be within reasonable range
        assert np.all(np.abs(phase) <= 4 * np.pi)

    def test_phase_response_wrapped(self, butterworth_filter: IIRFilter):
        """Test phase response without unwrapping."""
        introspect = FilterIntrospection(butterworth_filter)

        freqs, phase = introspect.phase_response(unwrap=False, degrees=False)

        assert len(freqs) == 512
        assert len(phase) == 512
        # Wrapped: should be within -pi to pi
        assert np.all(phase >= -np.pi)
        assert np.all(phase <= np.pi)

    def test_phase_response_custom_freqs(self, butterworth_filter: IIRFilter):
        """Test phase response with custom frequencies."""
        introspect = FilterIntrospection(butterworth_filter)
        custom_freqs = np.array([100, 500, 1000], dtype=np.float64)

        freqs, phase = introspect.phase_response(custom_freqs)

        assert len(freqs) == len(custom_freqs)
        assert len(phase) == len(custom_freqs)

    def test_phase_response_no_sample_rate_raises(self, butterworth_sos: NDArray[np.float64]):
        """Test phase response raises when no sample rate and no freqs."""
        filt = IIRFilter(sample_rate=None, sos=butterworth_sos)
        introspect = FilterIntrospection(filt)

        with pytest.raises(ValueError, match="sample_rate"):
            introspect.phase_response()

    def test_phase_response_linear_phase_fir(self, linear_phase_filter: FIRFilter):
        """Test phase response for linear phase FIR."""
        introspect = FilterIntrospection(linear_phase_filter)

        freqs, phase = introspect.phase_response(unwrap=True, degrees=False)

        # Linear phase FIR should have linear phase vs frequency
        # (approximately)
        assert len(freqs) == 512
        assert len(phase) == 512

    def test_group_delay_hz(self, butterworth_filter: IIRFilter):
        """Test group delay in Hz and seconds."""
        introspect = FilterIntrospection(butterworth_filter)

        freqs, gd = introspect.group_delay_hz()

        assert len(freqs) > 0
        assert len(gd) > 0
        assert len(freqs) == len(gd)
        # Group delay should be non-negative (mostly)
        assert np.all(gd >= -1e-6)  # Allow small numerical errors
        # Should be in seconds
        assert np.all(gd < 1)  # Should be less than 1 second

    def test_group_delay_hz_no_sample_rate(self, butterworth_sos: NDArray[np.float64]):
        """Test group delay without sample rate (normalized)."""
        filt = IIRFilter(sample_rate=None, sos=butterworth_sos)
        introspect = FilterIntrospection(filt)

        freqs, gd = introspect.group_delay_hz()

        assert len(freqs) > 0
        assert len(gd) > 0
        # Without sample rate, returns normalized values

    def test_passband_ripple(self, butterworth_filter: IIRFilter):
        """Test passband ripple calculation."""
        introspect = FilterIntrospection(butterworth_filter)

        ripple = introspect.passband_ripple(passband_edge=400)

        assert isinstance(ripple, float)
        assert ripple >= 0
        # Butterworth has monotonic passband, low ripple
        assert ripple < 1.0  # Should be less than 1 dB

    def test_passband_ripple_no_sample_rate_raises(self, butterworth_sos: NDArray[np.float64]):
        """Test passband ripple raises without sample rate."""
        filt = IIRFilter(sample_rate=None, sos=butterworth_sos)
        introspect = FilterIntrospection(filt)

        with pytest.raises(ValueError, match="Sample rate"):
            introspect.passband_ripple(400)

    def test_stopband_attenuation(self, butterworth_filter: IIRFilter):
        """Test stopband attenuation calculation."""
        introspect = FilterIntrospection(butterworth_filter)

        attenuation = introspect.stopband_attenuation(stopband_edge=2000)

        assert isinstance(attenuation, float)
        assert attenuation > 0  # Positive value
        # Should have significant attenuation in stopband
        assert attenuation > 10  # At least 10 dB

    def test_stopband_attenuation_no_sample_rate_raises(self, butterworth_sos: NDArray[np.float64]):
        """Test stopband attenuation raises without sample rate."""
        filt = IIRFilter(sample_rate=None, sos=butterworth_sos)
        introspect = FilterIntrospection(filt)

        with pytest.raises(ValueError, match="Sample rate"):
            introspect.stopband_attenuation(2000)

    def test_cutoff_frequency_default_threshold(self, butterworth_filter: IIRFilter):
        """Test cutoff frequency with -3dB threshold."""
        introspect = FilterIntrospection(butterworth_filter)

        fc = introspect.cutoff_frequency()

        assert isinstance(fc, float)
        assert fc > 0
        # Should be around 500 Hz (the designed cutoff)
        assert 400 < fc < 600

    def test_cutoff_frequency_custom_threshold(self, butterworth_filter: IIRFilter):
        """Test cutoff frequency with custom threshold."""
        introspect = FilterIntrospection(butterworth_filter)

        fc_3db = introspect.cutoff_frequency(threshold_db=-3.0)
        fc_6db = introspect.cutoff_frequency(threshold_db=-6.0)

        # -6dB cutoff should be higher than -3dB (deeper attenuation at higher freq)
        assert fc_6db > fc_3db

    def test_cutoff_frequency_no_crossing(self, sample_rate: float):
        """Test cutoff frequency when threshold never crossed."""
        # Create an all-pass filter that never crosses threshold
        b = np.array([1.0])
        a = np.array([1.0])
        filt = IIRFilter(sample_rate=sample_rate, ba=(b, a))
        introspect = FilterIntrospection(filt)

        # No attenuation, so should never cross -3dB threshold
        fc = introspect.cutoff_frequency()

        # Should return Nyquist frequency
        assert fc == sample_rate / 2

    def test_cutoff_frequency_no_sample_rate_raises(self, butterworth_sos: NDArray[np.float64]):
        """Test cutoff frequency raises without sample rate."""
        filt = IIRFilter(sample_rate=None, sos=butterworth_sos)
        introspect = FilterIntrospection(filt)

        with pytest.raises(ValueError, match="Sample rate"):
            introspect.cutoff_frequency()


# =============================================================================
# Plotting Function Tests
# =============================================================================


@pytest.mark.unit
class TestPlotBode:
    """Test plot_bode function."""

    def test_plot_bode_basic(self, butterworth_filter: IIRFilter):
        """Test basic Bode plot generation."""
        fig = plot_bode(butterworth_filter)

        assert fig is not None
        assert isinstance(fig, Figure)

    def test_plot_bode_custom_figsize(self, butterworth_filter: IIRFilter):
        """Test Bode plot with custom figure size."""
        fig = plot_bode(butterworth_filter, figsize=(12, 10))

        assert fig is not None
        # Check figure size (allow some tolerance for rounding)
        size = fig.get_size_inches()
        assert abs(size[0] - 12) < 0.1
        assert abs(size[1] - 10) < 0.1

    def test_plot_bode_custom_freq_range(self, butterworth_filter: IIRFilter):
        """Test Bode plot with custom frequency range."""
        fig = plot_bode(butterworth_filter, freq_range=(10, 2000))

        assert fig is not None
        assert isinstance(fig, Figure)

    def test_plot_bode_custom_n_points(self, butterworth_filter: IIRFilter):
        """Test Bode plot with custom number of points."""
        fig = plot_bode(butterworth_filter, n_points=1024)

        assert fig is not None
        assert isinstance(fig, Figure)

    def test_plot_bode_custom_title(self, butterworth_filter: IIRFilter):
        """Test Bode plot with custom title."""
        fig = plot_bode(butterworth_filter, title="My Custom Filter")

        assert fig is not None
        assert isinstance(fig, Figure)

    def test_plot_bode_no_sample_rate_raises(self, butterworth_sos: NDArray[np.float64]):
        """Test Bode plot raises when filter has no sample rate."""
        filt = IIRFilter(sample_rate=None, sos=butterworth_sos)

        with pytest.raises(ValueError, match="sample rate"):
            plot_bode(filt)

    def test_plot_bode_default_freq_range(self, butterworth_filter: IIRFilter):
        """Test Bode plot with default frequency range."""
        fig = plot_bode(butterworth_filter, freq_range=None)

        assert fig is not None
        assert isinstance(fig, Figure)


@pytest.mark.unit
class TestPlotImpulse:
    """Test plot_impulse function."""

    def test_plot_impulse_basic(self, butterworth_filter: IIRFilter):
        """Test basic impulse response plot."""
        fig = plot_impulse(butterworth_filter)

        assert fig is not None
        assert isinstance(fig, Figure)

    def test_plot_impulse_custom_n_samples(self, butterworth_filter: IIRFilter):
        """Test impulse plot with custom number of samples."""
        fig = plot_impulse(butterworth_filter, n_samples=512)

        assert fig is not None
        assert isinstance(fig, Figure)

    def test_plot_impulse_custom_figsize(self, butterworth_filter: IIRFilter):
        """Test impulse plot with custom figure size."""
        fig = plot_impulse(butterworth_filter, figsize=(12, 6))

        assert fig is not None
        size = fig.get_size_inches()
        assert abs(size[0] - 12) < 0.1
        assert abs(size[1] - 6) < 0.1

    def test_plot_impulse_custom_title(self, butterworth_filter: IIRFilter):
        """Test impulse plot with custom title."""
        fig = plot_impulse(butterworth_filter, title="My Impulse Response")

        assert fig is not None
        assert isinstance(fig, Figure)

    def test_plot_impulse_with_sample_rate(self, butterworth_filter: IIRFilter):
        """Test impulse plot labels with sample rate (uses time in us)."""
        fig = plot_impulse(butterworth_filter)

        assert fig is not None
        assert isinstance(fig, Figure)

    def test_plot_impulse_no_sample_rate(self, butterworth_sos: NDArray[np.float64]):
        """Test impulse plot without sample rate (uses samples)."""
        filt = IIRFilter(sample_rate=None, sos=butterworth_sos)
        fig = plot_impulse(filt)

        assert fig is not None
        assert isinstance(fig, Figure)


@pytest.mark.unit
class TestPlotStep:
    """Test plot_step function."""

    def test_plot_step_basic(self, butterworth_filter: IIRFilter):
        """Test basic step response plot."""
        fig = plot_step(butterworth_filter)

        assert fig is not None
        assert isinstance(fig, Figure)

    def test_plot_step_custom_n_samples(self, butterworth_filter: IIRFilter):
        """Test step plot with custom number of samples."""
        fig = plot_step(butterworth_filter, n_samples=512)

        assert fig is not None
        assert isinstance(fig, Figure)

    def test_plot_step_custom_figsize(self, butterworth_filter: IIRFilter):
        """Test step plot with custom figure size."""
        fig = plot_step(butterworth_filter, figsize=(12, 6))

        assert fig is not None
        size = fig.get_size_inches()
        assert abs(size[0] - 12) < 0.1
        assert abs(size[1] - 6) < 0.1

    def test_plot_step_custom_title(self, butterworth_filter: IIRFilter):
        """Test step plot with custom title."""
        fig = plot_step(butterworth_filter, title="My Step Response")

        assert fig is not None
        assert isinstance(fig, Figure)

    def test_plot_step_with_sample_rate(self, butterworth_filter: IIRFilter):
        """Test step plot with sample rate (uses time axis)."""
        fig = plot_step(butterworth_filter)

        assert fig is not None
        assert isinstance(fig, Figure)

    def test_plot_step_no_sample_rate(self, butterworth_sos: NDArray[np.float64]):
        """Test step plot without sample rate (uses samples)."""
        filt = IIRFilter(sample_rate=None, sos=butterworth_sos)
        fig = plot_step(filt)

        assert fig is not None
        assert isinstance(fig, Figure)


@pytest.mark.unit
class TestPlotPolesZeros:
    """Test plot_poles_zeros function."""

    def test_plot_poles_zeros_basic(self, butterworth_filter: IIRFilter):
        """Test basic pole-zero plot."""
        fig = plot_poles_zeros(butterworth_filter)

        assert fig is not None
        assert isinstance(fig, Figure)

    def test_plot_poles_zeros_custom_figsize(self, butterworth_filter: IIRFilter):
        """Test pole-zero plot with custom figure size."""
        fig = plot_poles_zeros(butterworth_filter, figsize=(10, 10))

        assert fig is not None
        size = fig.get_size_inches()
        assert abs(size[0] - 10) < 0.1
        assert abs(size[1] - 10) < 0.1

    def test_plot_poles_zeros_custom_title(self, butterworth_filter: IIRFilter):
        """Test pole-zero plot with custom title."""
        fig = plot_poles_zeros(butterworth_filter, title="My Pole-Zero Plot")

        assert fig is not None
        assert isinstance(fig, Figure)

    def test_plot_poles_zeros_fir_raises(self, fir_filter: FIRFilter):
        """Test pole-zero plot raises for FIR filter."""
        with pytest.raises(ValueError, match="IIR"):
            plot_poles_zeros(fir_filter)

    def test_plot_poles_zeros_stable_filter(self, butterworth_filter: IIRFilter):
        """Test pole-zero plot for stable filter shows STABLE."""
        fig = plot_poles_zeros(butterworth_filter)

        assert fig is not None
        assert isinstance(fig, Figure)

    def test_plot_poles_zeros_unstable_filter(self, sample_rate: float):
        """Test pole-zero plot for unstable filter shows UNSTABLE."""
        # Create unstable filter
        b = np.array([1.0])
        a = np.array([1.0, -1.5])  # Pole at z=1.5 (unstable)
        filt = IIRFilter(sample_rate=sample_rate, ba=(b, a))

        fig = plot_poles_zeros(filt)

        assert fig is not None
        assert isinstance(fig, Figure)


@pytest.mark.unit
class TestPlotGroupDelay:
    """Test plot_group_delay function."""

    def test_plot_group_delay_basic(self, butterworth_filter: IIRFilter):
        """Test basic group delay plot."""
        fig = plot_group_delay(butterworth_filter)

        assert fig is not None
        assert isinstance(fig, Figure)

    def test_plot_group_delay_custom_figsize(self, butterworth_filter: IIRFilter):
        """Test group delay plot with custom figure size."""
        fig = plot_group_delay(butterworth_filter, figsize=(12, 6))

        assert fig is not None
        size = fig.get_size_inches()
        assert abs(size[0] - 12) < 0.1
        assert abs(size[1] - 6) < 0.1

    def test_plot_group_delay_custom_freq_range(self, butterworth_filter: IIRFilter):
        """Test group delay plot with custom frequency range."""
        fig = plot_group_delay(butterworth_filter, freq_range=(10, 2000))

        assert fig is not None
        assert isinstance(fig, Figure)

    def test_plot_group_delay_custom_n_points(self, butterworth_filter: IIRFilter):
        """Test group delay plot with custom number of points."""
        fig = plot_group_delay(butterworth_filter, n_points=1024)

        assert fig is not None
        assert isinstance(fig, Figure)

    def test_plot_group_delay_custom_title(self, butterworth_filter: IIRFilter):
        """Test group delay plot with custom title."""
        fig = plot_group_delay(butterworth_filter, title="My Group Delay")

        assert fig is not None
        assert isinstance(fig, Figure)

    def test_plot_group_delay_no_sample_rate_raises(self, butterworth_sos: NDArray[np.float64]):
        """Test group delay plot raises when filter has no sample rate."""
        filt = IIRFilter(sample_rate=None, sos=butterworth_sos)

        with pytest.raises(ValueError, match="sample rate"):
            plot_group_delay(filt)


@pytest.mark.unit
class TestCompareFilters:
    """Test compare_filters function."""

    def test_compare_filters_basic(self, butterworth_filter: IIRFilter, fir_filter: FIRFilter):
        """Test basic filter comparison."""
        filters = [butterworth_filter, fir_filter]
        fig = compare_filters(filters)

        assert fig is not None
        assert isinstance(fig, Figure)

    def test_compare_filters_with_labels(
        self, butterworth_filter: IIRFilter, fir_filter: FIRFilter
    ):
        """Test filter comparison with custom labels."""
        filters = [butterworth_filter, fir_filter]
        labels = ["Butterworth", "Moving Average"]
        fig = compare_filters(filters, labels)

        assert fig is not None
        assert isinstance(fig, Figure)

    def test_compare_filters_default_labels(
        self, butterworth_filter: IIRFilter, fir_filter: FIRFilter
    ):
        """Test filter comparison with default labels."""
        filters = [butterworth_filter, fir_filter]
        fig = compare_filters(filters, labels=None)

        assert fig is not None
        assert isinstance(fig, Figure)

    def test_compare_filters_custom_figsize(
        self, butterworth_filter: IIRFilter, fir_filter: FIRFilter
    ):
        """Test filter comparison with custom figure size."""
        filters = [butterworth_filter, fir_filter]
        fig = compare_filters(filters, figsize=(14, 12))

        assert fig is not None
        size = fig.get_size_inches()
        assert abs(size[0] - 14) < 0.1
        assert abs(size[1] - 12) < 0.1

    def test_compare_filters_custom_freq_range(
        self, butterworth_filter: IIRFilter, fir_filter: FIRFilter
    ):
        """Test filter comparison with custom frequency range."""
        filters = [butterworth_filter, fir_filter]
        fig = compare_filters(filters, freq_range=(10, 2000))

        assert fig is not None
        assert isinstance(fig, Figure)

    def test_compare_filters_custom_n_points(
        self, butterworth_filter: IIRFilter, fir_filter: FIRFilter
    ):
        """Test filter comparison with custom number of points."""
        filters = [butterworth_filter, fir_filter]
        fig = compare_filters(filters, n_points=1024)

        assert fig is not None
        assert isinstance(fig, Figure)

    def test_compare_filters_label_count_mismatch_raises(
        self, butterworth_filter: IIRFilter, fir_filter: FIRFilter
    ):
        """Test filter comparison raises when label count doesn't match filter count."""
        filters = [butterworth_filter, fir_filter]
        labels = ["Only One Label"]

        with pytest.raises(ValueError, match="Number of labels"):
            compare_filters(filters, labels)

    def test_compare_filters_no_sample_rate_raises(self, butterworth_sos: NDArray[np.float64]):
        """Test filter comparison raises when first filter has no sample rate."""
        filt1 = IIRFilter(sample_rate=None, sos=butterworth_sos)
        filt2 = IIRFilter(sample_rate=None, sos=butterworth_sos)

        with pytest.raises(ValueError, match="sample rate"):
            compare_filters([filt1, filt2])

    def test_compare_filters_single_filter(self, butterworth_filter: IIRFilter):
        """Test filter comparison with single filter."""
        filters = [butterworth_filter]
        fig = compare_filters(filters)

        assert fig is not None
        assert isinstance(fig, Figure)

    def test_compare_filters_many_filters(
        self,
        sample_rate: float,
        butterworth_sos: NDArray[np.float64],
        fir_coeffs: NDArray[np.float64],
    ):
        """Test filter comparison with many filters."""
        filt1 = IIRFilter(sample_rate=sample_rate, sos=butterworth_sos)
        filt2 = FIRFilter(sample_rate=sample_rate, coeffs=fir_coeffs)
        # Create another IIR filter with different cutoff
        sos2 = signal.butter(4, 1000, fs=sample_rate, output="sos")
        filt3 = IIRFilter(sample_rate=sample_rate, sos=sos2)

        filters = [filt1, filt2, filt3]
        labels = ["Low", "Medium", "High"]
        fig = compare_filters(filters, labels)

        assert fig is not None
        assert isinstance(fig, Figure)


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


@pytest.mark.unit
class TestFilteringIntrospectionEdgeCases:
    """Test edge cases and error handling."""

    def test_magnitude_response_zero_frequency(self, butterworth_filter: IIRFilter):
        """Test magnitude response at zero frequency."""
        introspect = FilterIntrospection(butterworth_filter)
        freqs = np.array([0.0], dtype=np.float64)

        freqs_out, mag = introspect.magnitude_response(freqs, db=False)

        assert len(mag) == 1
        assert mag[0] > 0
        # Should be close to 1 for lowpass at DC

    def test_magnitude_response_nyquist(self, butterworth_filter: IIRFilter):
        """Test magnitude response at Nyquist frequency."""
        introspect = FilterIntrospection(butterworth_filter)
        nyquist = butterworth_filter.sample_rate / 2
        freqs = np.array([nyquist], dtype=np.float64)

        freqs_out, mag = introspect.magnitude_response(freqs, db=True)

        assert len(mag) == 1
        # Should be attenuated at Nyquist for lowpass

    def test_phase_response_zero_frequency(self, butterworth_filter: IIRFilter):
        """Test phase response at zero frequency."""
        introspect = FilterIntrospection(butterworth_filter)
        freqs = np.array([0.0], dtype=np.float64)

        freqs_out, phase = introspect.phase_response(freqs, degrees=True)

        assert len(phase) == 1
        # Phase at DC should be close to zero for lowpass

    def test_cutoff_frequency_dc_only(self, sample_rate: float):
        """Test cutoff frequency for all-pass DC filter."""
        # Create a simple all-pass filter (b=[1], a=[1])
        b = np.array([1.0])
        a = np.array([1.0])
        filt = IIRFilter(sample_rate=sample_rate, ba=(b, a))
        introspect = FilterIntrospection(filt)

        # Should return Nyquist since never crosses threshold
        fc = introspect.cutoff_frequency()

        assert fc == sample_rate / 2

    def test_passband_ripple_zero_edge(self, butterworth_filter: IIRFilter):
        """Test passband ripple with very small edge frequency."""
        introspect = FilterIntrospection(butterworth_filter)

        ripple = introspect.passband_ripple(passband_edge=1.0)

        assert isinstance(ripple, float)
        assert ripple >= 0

    def test_stopband_attenuation_at_nyquist(self, butterworth_filter: IIRFilter):
        """Test stopband attenuation from near Nyquist to Nyquist."""
        introspect = FilterIntrospection(butterworth_filter)
        nyquist = butterworth_filter.sample_rate / 2

        attenuation = introspect.stopband_attenuation(stopband_edge=nyquist - 100)

        assert isinstance(attenuation, float)
        assert attenuation >= 0

    def test_filter_introspection_with_non_designed_filter(self, sample_rate: float):
        """Test introspection with filter that has no coefficients."""
        filt = IIRFilter(sample_rate=sample_rate)
        introspect = FilterIntrospection(filt)

        # Should still create introspection object
        assert introspect.filter is filt

    def test_plot_bode_very_high_order(self, sample_rate: float):
        """Test Bode plot with very high order filter."""
        # Create high order filter
        sos = signal.butter(16, 500, fs=sample_rate, output="sos")
        filt = IIRFilter(sample_rate=sample_rate, sos=sos)

        fig = plot_bode(filt)

        assert fig is not None
        assert isinstance(fig, Figure)

    def test_plot_impulse_very_long(self, butterworth_filter: IIRFilter):
        """Test impulse plot with very long response."""
        fig = plot_impulse(butterworth_filter, n_samples=10000)

        assert fig is not None
        assert isinstance(fig, Figure)

    def test_compare_filters_empty_list(self):
        """Test compare_filters with empty filter list."""
        # Should handle gracefully or raise appropriate error
        # Current implementation will likely fail with index error
        with pytest.raises((ValueError, IndexError)):
            compare_filters([])

    def test_group_delay_constant(self, linear_phase_filter: FIRFilter):
        """Test that linear phase FIR has approximately constant group delay."""
        introspect = FilterIntrospection(linear_phase_filter)

        freqs, gd = introspect.group_delay_hz()

        # Linear phase should have constant group delay
        # Allow some variation due to numerical effects
        assert np.std(gd) < 1e-3

    def test_magnitude_response_single_frequency(self, butterworth_filter: IIRFilter):
        """Test magnitude response at single frequency point."""
        introspect = FilterIntrospection(butterworth_filter)
        freqs = np.array([500.0], dtype=np.float64)  # At cutoff

        freqs_out, mag = introspect.magnitude_response(freqs, db=True)

        assert len(mag) == 1
        # At cutoff, should be around -3 dB
        assert -4 < mag[0] < -2

    def test_phase_response_single_frequency(self, butterworth_filter: IIRFilter):
        """Test phase response at single frequency point."""
        introspect = FilterIntrospection(butterworth_filter)
        freqs = np.array([500.0], dtype=np.float64)

        freqs_out, phase = introspect.phase_response(freqs, degrees=True)

        assert len(phase) == 1
        assert np.isfinite(phase[0])

    def test_plot_functions_return_figure(self, butterworth_filter: IIRFilter):
        """Test that all plot functions return Figure objects."""
        fig1 = plot_bode(butterworth_filter)
        fig2 = plot_impulse(butterworth_filter)
        fig3 = plot_step(butterworth_filter)
        fig4 = plot_poles_zeros(butterworth_filter)
        fig5 = plot_group_delay(butterworth_filter)

        assert all(isinstance(fig, Figure) for fig in [fig1, fig2, fig3, fig4, fig5])
