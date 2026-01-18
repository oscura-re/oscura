"""Unit tests for filtering base module.

Tests base filter classes and abstractions including Filter, IIRFilter, and FIRFilter.
"""

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy import signal

from oscura.core.exceptions import AnalysisError
from oscura.core.types import TraceMetadata, WaveformTrace
from oscura.filtering.base import FilterResult, FIRFilter, IIRFilter

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
    """Generate a test waveform trace."""
    n = 1000
    t = np.arange(n) / sample_rate
    # 100 Hz sine + 1 kHz noise
    data = np.sin(2 * np.pi * 100 * t) + 0.3 * np.sin(2 * np.pi * 1000 * t)
    return WaveformTrace(
        data=data.astype(np.float64),
        metadata=TraceMetadata(sample_rate=sample_rate),
    )


@pytest.fixture
def butterworth_sos(sample_rate: float) -> NDArray[np.float64]:
    """Create Butterworth filter SOS coefficients."""
    # 4th order Butterworth lowpass at 500 Hz
    sos = signal.butter(4, 500, fs=sample_rate, output="sos")
    return sos


@pytest.fixture
def butterworth_ba(sample_rate: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Create Butterworth filter B/A coefficients."""
    b, a = signal.butter(4, 500, fs=sample_rate, output="ba")
    return b, a


@pytest.fixture
def fir_coeffs() -> NDArray[np.float64]:
    """Create simple FIR filter coefficients (moving average)."""
    return np.ones(11, dtype=np.float64) / 11.0


@pytest.fixture
def linear_phase_fir() -> NDArray[np.float64]:
    """Create symmetric FIR coefficients for linear phase."""
    coeffs = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1], dtype=np.float64)
    return coeffs / np.sum(coeffs)


# =============================================================================
# FilterResult Tests
# =============================================================================


@pytest.mark.unit
class TestFilterResult:
    """Test FilterResult dataclass."""

    def test_filter_result_creation(self, test_trace: WaveformTrace):
        """Test creating a FilterResult."""
        result = FilterResult(trace=test_trace)

        assert result.trace is test_trace
        assert result.transfer_function is None
        assert result.impulse_response is None
        assert result.group_delay is None

    def test_filter_result_with_details(self, test_trace: WaveformTrace):
        """Test FilterResult with introspection data."""
        h = np.ones(512, dtype=np.complex128)
        impulse = np.ones(256, dtype=np.float64)
        gd = np.ones(512, dtype=np.float64)

        result = FilterResult(
            trace=test_trace,
            transfer_function=h,
            impulse_response=impulse,
            group_delay=gd,
        )

        assert result.transfer_function is not None
        assert len(result.transfer_function) == 512
        assert result.impulse_response is not None
        assert len(result.impulse_response) == 256
        assert result.group_delay is not None
        assert len(result.group_delay) == 512


# =============================================================================
# IIRFilter Tests
# =============================================================================


@pytest.mark.unit
class TestIIRFilter:
    """Test IIRFilter base class."""

    def test_iir_creation_with_sos(self, sample_rate: float, butterworth_sos: NDArray[np.float64]):
        """Test creating IIR filter with SOS coefficients."""
        filt = IIRFilter(sample_rate=sample_rate, sos=butterworth_sos)

        assert filt.sample_rate == sample_rate
        assert filt.sos is not None
        assert filt.sos.shape[1] == 6  # SOS format
        assert filt.is_stable
        assert filt.order > 0

    def test_iir_creation_with_ba(
        self, sample_rate: float, butterworth_ba: tuple[NDArray[np.float64], NDArray[np.float64]]
    ):
        """Test creating IIR filter with B/A coefficients."""
        b, a = butterworth_ba
        filt = IIRFilter(sample_rate=sample_rate, ba=(b, a))

        assert filt.ba is not None
        assert len(filt.ba[0]) > 0  # Numerator
        assert len(filt.ba[1]) > 0  # Denominator
        assert filt.is_stable

    def test_iir_creation_no_coeffs(self, sample_rate: float):
        """Test creating IIR filter without coefficients."""
        filt = IIRFilter(sample_rate=sample_rate)

        assert filt.sample_rate == sample_rate
        assert filt.sos is None
        assert filt.ba is None
        assert filt.order == 0
        assert filt.is_stable  # Not designed yet, considered stable

    def test_iir_sos_to_ba_conversion(
        self, sample_rate: float, butterworth_sos: NDArray[np.float64]
    ):
        """Test automatic SOS to B/A conversion."""
        filt = IIRFilter(sample_rate=sample_rate, sos=butterworth_sos)

        ba = filt.ba
        assert ba is not None
        b, a = ba
        assert len(b) > 0
        assert len(a) > 0
        # First coefficient of denominator should be 1
        assert np.isclose(a[0], 1.0)

    def test_iir_order_calculation_sos(
        self, sample_rate: float, butterworth_sos: NDArray[np.float64]
    ):
        """Test order calculation from SOS."""
        filt = IIRFilter(sample_rate=sample_rate, sos=butterworth_sos)

        # 4th order filter = 2 SOS sections
        assert filt.order == 2 * len(butterworth_sos)

    def test_iir_order_calculation_ba(
        self, sample_rate: float, butterworth_ba: tuple[NDArray[np.float64], NDArray[np.float64]]
    ):
        """Test order calculation from B/A."""
        b, a = butterworth_ba
        filt = IIRFilter(sample_rate=sample_rate, ba=(b, a))

        # Order = length of denominator - 1
        assert filt.order == len(a) - 1

    def test_iir_stability_check_stable(
        self, sample_rate: float, butterworth_sos: NDArray[np.float64]
    ):
        """Test stability check for stable filter."""
        filt = IIRFilter(sample_rate=sample_rate, sos=butterworth_sos)

        assert filt.is_stable
        poles = filt.poles
        assert np.all(np.abs(poles) < 1.0)

    def test_iir_stability_check_unstable(self, sample_rate: float):
        """Test stability check for unstable filter."""
        # Create unstable filter (pole outside unit circle)
        b = np.array([1.0])
        a = np.array([1.0, -1.5])  # Pole at z=1.5 (unstable)

        filt = IIRFilter(sample_rate=sample_rate, ba=(b, a))

        assert not filt.is_stable

    def test_iir_poles_extraction(self, sample_rate: float, butterworth_sos: NDArray[np.float64]):
        """Test extracting filter poles."""
        filt = IIRFilter(sample_rate=sample_rate, sos=butterworth_sos)

        poles = filt.poles
        assert len(poles) > 0
        assert poles.dtype == np.complex128
        # All poles should be inside unit circle for stable filter
        assert np.all(np.abs(poles) < 1.0)

    def test_iir_zeros_extraction(self, sample_rate: float, butterworth_sos: NDArray[np.float64]):
        """Test extracting filter zeros."""
        filt = IIRFilter(sample_rate=sample_rate, sos=butterworth_sos)

        zeros = filt.zeros
        assert len(zeros) > 0
        assert zeros.dtype == np.complex128

    def test_iir_apply_with_sos(
        self,
        sample_rate: float,
        butterworth_sos: NDArray[np.float64],
        test_trace: WaveformTrace,
    ):
        """Test applying IIR filter with SOS coefficients."""
        filt = IIRFilter(sample_rate=sample_rate, sos=butterworth_sos)
        result = filt.apply(test_trace)

        assert isinstance(result, WaveformTrace)
        assert len(result.data) == len(test_trace.data)
        assert result.metadata == test_trace.metadata

    def test_iir_apply_with_ba(
        self,
        sample_rate: float,
        butterworth_ba: tuple[NDArray[np.float64], NDArray[np.float64]],
        test_trace: WaveformTrace,
    ):
        """Test applying IIR filter with B/A coefficients."""
        b, a = butterworth_ba
        filt = IIRFilter(sample_rate=sample_rate, ba=(b, a))
        result = filt.apply(test_trace)

        assert isinstance(result, WaveformTrace)
        assert len(result.data) == len(test_trace.data)

    def test_iir_apply_no_coeffs_raises(self, sample_rate: float, test_trace: WaveformTrace):
        """Test that applying filter without coefficients raises error."""
        filt = IIRFilter(sample_rate=sample_rate)

        with pytest.raises(AnalysisError, match="not designed"):
            filt.apply(test_trace)

    def test_iir_apply_unstable_raises(self, sample_rate: float, test_trace: WaveformTrace):
        """Test that applying unstable filter raises error."""
        # Unstable filter
        b = np.array([1.0])
        a = np.array([1.0, -1.5])
        filt = IIRFilter(sample_rate=sample_rate, ba=(b, a))

        with pytest.raises(AnalysisError, match="unstable"):
            filt.apply(test_trace)

    def test_iir_apply_filtfilt(
        self,
        sample_rate: float,
        butterworth_sos: NDArray[np.float64],
        test_trace: WaveformTrace,
    ):
        """Test zero-phase filtering with filtfilt."""
        filt = IIRFilter(sample_rate=sample_rate, sos=butterworth_sos)
        result = filt.apply(test_trace, filtfilt=True)

        assert isinstance(result, WaveformTrace)
        assert len(result.data) == len(test_trace.data)

    def test_iir_apply_causal(
        self,
        sample_rate: float,
        butterworth_sos: NDArray[np.float64],
        test_trace: WaveformTrace,
    ):
        """Test causal filtering (no filtfilt)."""
        filt = IIRFilter(sample_rate=sample_rate, sos=butterworth_sos)
        result = filt.apply(test_trace, filtfilt=False)

        assert isinstance(result, WaveformTrace)
        assert len(result.data) == len(test_trace.data)

    def test_iir_apply_causal_ba(
        self,
        sample_rate: float,
        butterworth_ba: tuple[NDArray[np.float64], NDArray[np.float64]],
        test_trace: WaveformTrace,
    ):
        """Test causal filtering with B/A coefficients (no filtfilt)."""
        b, a = butterworth_ba
        filt = IIRFilter(sample_rate=sample_rate, ba=(b, a))
        result = filt.apply(test_trace, filtfilt=False)

        assert isinstance(result, WaveformTrace)
        assert len(result.data) == len(test_trace.data)

    def test_iir_apply_return_details(
        self,
        sample_rate: float,
        butterworth_sos: NDArray[np.float64],
        test_trace: WaveformTrace,
    ):
        """Test applying filter with return_details=True."""
        filt = IIRFilter(sample_rate=sample_rate, sos=butterworth_sos)
        result = filt.apply(test_trace, return_details=True)

        assert isinstance(result, FilterResult)
        assert result.trace is not None
        assert result.transfer_function is not None
        assert result.impulse_response is not None
        assert result.group_delay is not None

    def test_iir_frequency_response_sos(
        self, sample_rate: float, butterworth_sos: NDArray[np.float64]
    ):
        """Test frequency response calculation from SOS."""
        filt = IIRFilter(sample_rate=sample_rate, sos=butterworth_sos)

        w, h = filt.get_frequency_response(512)

        assert len(w) == 512
        assert len(h) == 512
        assert h.dtype == np.complex128
        # Magnitude at DC should be close to 1 for lowpass
        assert np.abs(h[0]) > 0.9

    def test_iir_frequency_response_ba(
        self, sample_rate: float, butterworth_ba: tuple[NDArray[np.float64], NDArray[np.float64]]
    ):
        """Test frequency response calculation from B/A."""
        b, a = butterworth_ba
        filt = IIRFilter(sample_rate=sample_rate, ba=(b, a))

        w, h = filt.get_frequency_response(256)

        assert len(w) == 256
        assert len(h) == 256

    def test_iir_frequency_response_not_designed_raises(self, sample_rate: float):
        """Test that frequency response raises when filter not designed."""
        filt = IIRFilter(sample_rate=sample_rate)

        with pytest.raises(AnalysisError, match="not designed"):
            filt.get_frequency_response()

    def test_iir_frequency_response_array_worN(
        self, sample_rate: float, butterworth_sos: NDArray[np.float64]
    ):
        """Test frequency response with array worN."""
        filt = IIRFilter(sample_rate=sample_rate, sos=butterworth_sos)

        freqs = np.linspace(0, np.pi, 100)
        w, h = filt.get_frequency_response(freqs)

        assert len(w) == len(freqs)
        assert len(h) == len(freqs)

    def test_iir_impulse_response_sos(
        self, sample_rate: float, butterworth_sos: NDArray[np.float64]
    ):
        """Test impulse response calculation from SOS."""
        filt = IIRFilter(sample_rate=sample_rate, sos=butterworth_sos)

        h = filt.get_impulse_response(256)

        assert len(h) == 256
        assert h.dtype == np.float64
        # First sample should be non-zero
        assert h[0] != 0

    def test_iir_impulse_response_ba(
        self, sample_rate: float, butterworth_ba: tuple[NDArray[np.float64], NDArray[np.float64]]
    ):
        """Test impulse response calculation from B/A."""
        b, a = butterworth_ba
        filt = IIRFilter(sample_rate=sample_rate, ba=(b, a))

        h = filt.get_impulse_response(128)

        assert len(h) == 128
        assert h.dtype == np.float64

    def test_iir_impulse_response_not_designed_raises(self, sample_rate: float):
        """Test that impulse response raises when filter not designed."""
        filt = IIRFilter(sample_rate=sample_rate)

        with pytest.raises(AnalysisError, match="not designed"):
            filt.get_impulse_response()

    def test_iir_step_response_sos(self, sample_rate: float, butterworth_sos: NDArray[np.float64]):
        """Test step response calculation from SOS."""
        filt = IIRFilter(sample_rate=sample_rate, sos=butterworth_sos)

        s = filt.get_step_response(256)

        assert len(s) == 256
        assert s.dtype == np.float64
        # Step response should settle to ~1 for unity gain filter
        assert np.abs(s[-1] - 1.0) < 0.1

    def test_iir_step_response_ba(
        self, sample_rate: float, butterworth_ba: tuple[NDArray[np.float64], NDArray[np.float64]]
    ):
        """Test step response calculation from B/A."""
        b, a = butterworth_ba
        filt = IIRFilter(sample_rate=sample_rate, ba=(b, a))

        s = filt.get_step_response(128)

        assert len(s) == 128

    def test_iir_step_response_not_designed_raises(self, sample_rate: float):
        """Test that step response raises when filter not designed."""
        filt = IIRFilter(sample_rate=sample_rate)

        with pytest.raises(AnalysisError, match="not designed"):
            filt.get_step_response()


# =============================================================================
# FIRFilter Tests
# =============================================================================


@pytest.mark.unit
class TestFIRFilter:
    """Test FIRFilter base class."""

    def test_fir_creation_with_coeffs(self, sample_rate: float, fir_coeffs: NDArray[np.float64]):
        """Test creating FIR filter with coefficients."""
        filt = FIRFilter(sample_rate=sample_rate, coeffs=fir_coeffs)

        assert filt.sample_rate == sample_rate
        assert filt.coeffs is not None
        assert len(filt.coeffs) == 11
        assert filt.is_stable  # FIR always stable
        assert filt.order == 10  # N-1

    def test_fir_creation_no_coeffs(self, sample_rate: float):
        """Test creating FIR filter without coefficients."""
        filt = FIRFilter(sample_rate=sample_rate)

        assert filt.sample_rate == sample_rate
        assert filt.coeffs is None
        assert filt.order == 0
        assert filt.is_stable

    def test_fir_coeffs_setter(self, sample_rate: float, fir_coeffs: NDArray[np.float64]):
        """Test setting FIR coefficients."""
        filt = FIRFilter(sample_rate=sample_rate)
        filt.coeffs = fir_coeffs

        assert filt.coeffs is not None
        assert np.array_equal(filt.coeffs, fir_coeffs)
        assert filt.order == 10

    def test_fir_order_calculation(self, sample_rate: float):
        """Test order calculation from coefficients."""
        coeffs = np.ones(21, dtype=np.float64)
        filt = FIRFilter(sample_rate=sample_rate, coeffs=coeffs)

        assert filt.order == 20  # length - 1

    def test_fir_is_always_stable(self, sample_rate: float):
        """Test that FIR filters are always stable."""
        # Even with arbitrary coefficients
        coeffs = np.array([100, -50, 75, -25], dtype=np.float64)
        filt = FIRFilter(sample_rate=sample_rate, coeffs=coeffs)

        assert filt.is_stable

    def test_fir_linear_phase_symmetric(
        self, sample_rate: float, linear_phase_fir: NDArray[np.float64]
    ):
        """Test linear phase detection for symmetric coefficients."""
        filt = FIRFilter(sample_rate=sample_rate, coeffs=linear_phase_fir)

        assert filt.is_linear_phase

    def test_fir_linear_phase_antisymmetric(self, sample_rate: float):
        """Test linear phase detection for antisymmetric coefficients."""
        coeffs = np.array([1, 2, 3, -3, -2, -1], dtype=np.float64)
        filt = FIRFilter(sample_rate=sample_rate, coeffs=coeffs)

        assert filt.is_linear_phase

    def test_fir_not_linear_phase(self, sample_rate: float):
        """Test linear phase detection for non-symmetric coefficients."""
        coeffs = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        filt = FIRFilter(sample_rate=sample_rate, coeffs=coeffs)

        assert not filt.is_linear_phase

    def test_fir_linear_phase_no_coeffs(self, sample_rate: float):
        """Test linear phase detection when no coefficients."""
        filt = FIRFilter(sample_rate=sample_rate)

        assert not filt.is_linear_phase

    def test_fir_apply_mode_same(
        self, sample_rate: float, fir_coeffs: NDArray[np.float64], test_trace: WaveformTrace
    ):
        """Test applying FIR filter with mode='same'."""
        filt = FIRFilter(sample_rate=sample_rate, coeffs=fir_coeffs)
        result = filt.apply(test_trace, mode="same")

        assert isinstance(result, WaveformTrace)
        assert len(result.data) == len(test_trace.data)  # Same length
        assert result.metadata == test_trace.metadata

    def test_fir_apply_mode_valid(
        self, sample_rate: float, fir_coeffs: NDArray[np.float64], test_trace: WaveformTrace
    ):
        """Test applying FIR filter with mode='valid'."""
        filt = FIRFilter(sample_rate=sample_rate, coeffs=fir_coeffs)
        result = filt.apply(test_trace, mode="valid")

        assert isinstance(result, WaveformTrace)
        # Valid mode: output length = N - M + 1
        expected_len = len(test_trace.data) - len(fir_coeffs) + 1
        assert len(result.data) == expected_len

    def test_fir_apply_mode_full(
        self, sample_rate: float, fir_coeffs: NDArray[np.float64], test_trace: WaveformTrace
    ):
        """Test applying FIR filter with mode='full'."""
        filt = FIRFilter(sample_rate=sample_rate, coeffs=fir_coeffs)
        result = filt.apply(test_trace, mode="full")

        assert isinstance(result, WaveformTrace)
        # Full mode: output length = N + M - 1
        expected_len = len(test_trace.data) + len(fir_coeffs) - 1
        assert len(result.data) == expected_len

    def test_fir_apply_no_coeffs_raises(self, sample_rate: float, test_trace: WaveformTrace):
        """Test that applying filter without coefficients raises error."""
        filt = FIRFilter(sample_rate=sample_rate)

        with pytest.raises(AnalysisError, match="not designed"):
            filt.apply(test_trace)

    def test_fir_apply_return_details(
        self, sample_rate: float, fir_coeffs: NDArray[np.float64], test_trace: WaveformTrace
    ):
        """Test applying filter with return_details=True."""
        filt = FIRFilter(sample_rate=sample_rate, coeffs=fir_coeffs)
        result = filt.apply(test_trace, return_details=True)

        assert isinstance(result, FilterResult)
        assert result.trace is not None
        assert result.transfer_function is not None
        assert result.impulse_response is not None
        assert result.group_delay is not None

    def test_fir_frequency_response(self, sample_rate: float, fir_coeffs: NDArray[np.float64]):
        """Test frequency response calculation."""
        filt = FIRFilter(sample_rate=sample_rate, coeffs=fir_coeffs)

        w, h = filt.get_frequency_response(512)

        assert len(w) == 512
        assert len(h) == 512
        assert h.dtype == np.complex128

    def test_fir_frequency_response_not_designed_raises(self, sample_rate: float):
        """Test that frequency response raises when filter not designed."""
        filt = FIRFilter(sample_rate=sample_rate)

        with pytest.raises(AnalysisError, match="not designed"):
            filt.get_frequency_response()

    def test_fir_impulse_response(self, sample_rate: float, fir_coeffs: NDArray[np.float64]):
        """Test impulse response (should match coefficients)."""
        filt = FIRFilter(sample_rate=sample_rate, coeffs=fir_coeffs)

        h = filt.get_impulse_response(256)

        assert len(h) == 256
        assert h.dtype == np.float64
        # First N samples should match coefficients
        assert np.allclose(h[: len(fir_coeffs)], fir_coeffs)
        # Rest should be zero
        assert np.allclose(h[len(fir_coeffs) :], 0)

    def test_fir_impulse_response_truncated(self, sample_rate: float):
        """Test impulse response when n_samples < coeffs length."""
        coeffs = np.ones(100, dtype=np.float64)
        filt = FIRFilter(sample_rate=sample_rate, coeffs=coeffs)

        h = filt.get_impulse_response(50)

        assert len(h) == 50
        # Should be first 50 coefficients
        assert np.allclose(h, coeffs[:50])

    def test_fir_impulse_response_not_designed_raises(self, sample_rate: float):
        """Test that impulse response raises when filter not designed."""
        filt = FIRFilter(sample_rate=sample_rate)

        with pytest.raises(AnalysisError, match="not designed"):
            filt.get_impulse_response()

    def test_fir_step_response(self, sample_rate: float, fir_coeffs: NDArray[np.float64]):
        """Test step response calculation."""
        filt = FIRFilter(sample_rate=sample_rate, coeffs=fir_coeffs)

        s = filt.get_step_response(256)

        assert len(s) == 256
        assert s.dtype == np.float64
        # Step response should be cumsum of impulse response
        # For normalized moving average, should settle to 1
        assert np.abs(s[-1] - 1.0) < 0.1

    def test_fir_step_response_not_designed_raises(self, sample_rate: float):
        """Test that step response raises when filter not designed."""
        filt = FIRFilter(sample_rate=sample_rate)

        with pytest.raises(AnalysisError, match="not designed"):
            filt.get_step_response()

    def test_fir_group_delay(self, sample_rate: float, linear_phase_fir: NDArray[np.float64]):
        """Test group delay calculation."""
        filt = FIRFilter(sample_rate=sample_rate, coeffs=linear_phase_fir)

        w, gd = filt.get_group_delay(512)

        assert len(w) == 512
        assert len(gd) == 512
        assert gd.dtype == np.float64
        # Linear phase FIR should have constant group delay
        assert np.std(gd) < 1.0  # Should be nearly constant

    def test_fir_group_delay_not_designed_raises(self, sample_rate: float):
        """Test that group delay raises when filter not designed."""
        filt = FIRFilter(sample_rate=sample_rate)

        with pytest.raises(AnalysisError, match="not designed"):
            filt.get_group_delay()


# =============================================================================
# Filter Base Class Tests
# =============================================================================


@pytest.mark.unit
class TestFilterBase:
    """Test Filter base class functionality."""

    def test_filter_sample_rate_property(self, sample_rate: float, fir_coeffs: NDArray[np.float64]):
        """Test sample_rate property getter."""
        filt = FIRFilter(sample_rate=sample_rate, coeffs=fir_coeffs)

        assert filt.sample_rate == sample_rate

    def test_filter_sample_rate_setter(self, sample_rate: float, fir_coeffs: NDArray[np.float64]):
        """Test sample_rate property setter."""
        filt = FIRFilter(sample_rate=sample_rate, coeffs=fir_coeffs)

        new_rate = 20_000.0
        filt.sample_rate = new_rate

        assert filt.sample_rate == new_rate

    def test_filter_sample_rate_setter_marks_redesign(
        self, sample_rate: float, fir_coeffs: NDArray[np.float64]
    ):
        """Test that changing sample_rate marks filter for redesign."""
        filt = FIRFilter(sample_rate=sample_rate, coeffs=fir_coeffs)
        assert filt._is_designed

        filt.sample_rate = 20_000.0

        assert not filt._is_designed

    def test_filter_sample_rate_setter_same_value(
        self, sample_rate: float, fir_coeffs: NDArray[np.float64]
    ):
        """Test that setting same sample_rate doesn't mark for redesign."""
        filt = FIRFilter(sample_rate=sample_rate, coeffs=fir_coeffs)
        assert filt._is_designed

        filt.sample_rate = sample_rate

        assert filt._is_designed  # Still designed

    def test_filter_transfer_function(
        self, sample_rate: float, butterworth_sos: NDArray[np.float64]
    ):
        """Test transfer function calculation."""
        filt = IIRFilter(sample_rate=sample_rate, sos=butterworth_sos)

        freqs = np.linspace(0, sample_rate / 2, 100)
        h = filt.get_transfer_function(freqs)

        assert len(h) == len(freqs)
        assert h.dtype == np.complex128

    def test_filter_transfer_function_default_freqs(
        self, sample_rate: float, butterworth_sos: NDArray[np.float64]
    ):
        """Test transfer function with default frequencies."""
        filt = IIRFilter(sample_rate=sample_rate, sos=butterworth_sos)

        h = filt.get_transfer_function()

        assert len(h) == 512  # Default

    def test_filter_transfer_function_no_sample_rate_raises(
        self, butterworth_sos: NDArray[np.float64]
    ):
        """Test that transfer function raises without sample_rate."""
        filt = IIRFilter(sample_rate=None, sos=butterworth_sos)

        with pytest.raises(AnalysisError, match="Sample rate must be set"):
            filt.get_transfer_function()

    def test_filter_group_delay_base(
        self, sample_rate: float, butterworth_sos: NDArray[np.float64]
    ):
        """Test group delay calculation from base class."""
        filt = IIRFilter(sample_rate=sample_rate, sos=butterworth_sos)

        w, gd = filt.get_group_delay(512)

        assert len(w) == 512
        assert len(gd) == 512
        assert gd.dtype == np.float64
        # Group delay should be non-negative
        assert np.all(gd >= -1)  # Allow small numerical errors

    def test_filter_group_delay_array_worN(
        self, sample_rate: float, butterworth_sos: NDArray[np.float64]
    ):
        """Test group delay with array worN."""
        filt = IIRFilter(sample_rate=sample_rate, sos=butterworth_sos)

        freqs = np.linspace(0, np.pi, 100)
        w, gd = filt.get_group_delay(freqs)

        assert len(w) == len(freqs)
        assert len(gd) == len(freqs)


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


@pytest.mark.unit
class TestFilteringBaseEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_coefficients(self, sample_rate: float):
        """Test filter with empty coefficients."""
        filt = FIRFilter(sample_rate=sample_rate, coeffs=np.array([]))

        assert filt.order == -1  # 0 - 1

    def test_single_coefficient(self, sample_rate: float):
        """Test filter with single coefficient."""
        filt = FIRFilter(sample_rate=sample_rate, coeffs=np.array([1.0]))

        assert filt.order == 0
        assert filt.is_stable

    def test_very_short_trace(self, sample_rate: float, fir_coeffs: NDArray[np.float64]):
        """Test filtering very short trace."""
        data = np.array([1.0, 2.0, 3.0])
        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        filt = FIRFilter(sample_rate=sample_rate, coeffs=fir_coeffs)
        result = filt.apply(trace, mode="same")

        # When signal is shorter than filter, mode='same' returns max length
        assert len(result.data) == max(len(data), len(fir_coeffs))

    def test_single_sample_trace(self, sample_rate: float, fir_coeffs: NDArray[np.float64]):
        """Test filtering single sample trace."""
        data = np.array([1.0])
        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        filt = FIRFilter(sample_rate=sample_rate, coeffs=fir_coeffs)
        result = filt.apply(trace, mode="same")

        # When signal is shorter than filter, mode='same' returns max length
        assert len(result.data) == max(len(data), len(fir_coeffs))

    def test_zero_signal(self, sample_rate: float, fir_coeffs: NDArray[np.float64]):
        """Test filtering all-zero signal."""
        data = np.zeros(100)
        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        filt = FIRFilter(sample_rate=sample_rate, coeffs=fir_coeffs)
        result = filt.apply(trace)

        assert np.allclose(result.data, 0)

    def test_constant_signal(self, sample_rate: float, fir_coeffs: NDArray[np.float64]):
        """Test filtering constant signal."""
        data = np.ones(100) * 5.0
        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        filt = FIRFilter(sample_rate=sample_rate, coeffs=fir_coeffs)
        result = filt.apply(trace)

        # Moving average of constant should be constant in the middle
        # (edge effects at start and end due to zero-padding)
        middle = result.data[20:80]  # Avoid edges
        assert np.allclose(middle, 5.0, atol=0.01)

    def test_nan_in_signal(self, sample_rate: float, fir_coeffs: NDArray[np.float64]):
        """Test filtering signal with NaN values."""
        data = np.ones(100)
        data[50] = np.nan
        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        filt = FIRFilter(sample_rate=sample_rate, coeffs=fir_coeffs)
        result = filt.apply(trace)

        # NaN should propagate
        assert np.any(np.isnan(result.data))

    def test_inf_in_signal(self, sample_rate: float, fir_coeffs: NDArray[np.float64]):
        """Test filtering signal with inf values."""
        data = np.ones(100)
        data[50] = np.inf
        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        filt = FIRFilter(sample_rate=sample_rate, coeffs=fir_coeffs)
        result = filt.apply(trace)

        # Inf should propagate or affect nearby samples
        assert np.any(np.isinf(result.data)) or np.any(np.abs(result.data) > 1e10)

    def test_very_large_values(self, sample_rate: float, fir_coeffs: NDArray[np.float64]):
        """Test filtering signal with very large values."""
        data = np.ones(100) * 1e15
        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        filt = FIRFilter(sample_rate=sample_rate, coeffs=fir_coeffs)
        result = filt.apply(trace)

        # Should handle large values
        assert np.all(np.isfinite(result.data))
        assert np.all(np.abs(result.data) > 1e14)

    def test_no_poles_or_zeros(self, sample_rate: float):
        """Test poles/zeros extraction when filter not designed."""
        filt = IIRFilter(sample_rate=sample_rate)

        poles = filt.poles
        zeros = filt.zeros

        assert len(poles) == 0
        assert len(zeros) == 0
