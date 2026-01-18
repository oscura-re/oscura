"""Unit tests for filter design module.

Tests filter design functions, convenience classes, and auto-design capabilities.


Coverage:
- design_filter() for all filter types (butterworth, chebyshev1, chebyshev2, bessel, elliptic)
- design_filter_spec() for auto-order computation
- Convenience filter classes (LowPassFilter, HighPassFilter, BandPassFilter, BandStopFilter)
- Filter type classes (ButterworthFilter, ChebyshevType1Filter, etc.)
- suggest_filter_type() and auto_design_filter()
- Edge cases and error handling
"""

import numpy as np
import pytest

from oscura.core.exceptions import AnalysisError
from oscura.core.types import TraceMetadata, WaveformTrace
from oscura.filtering.base import IIRFilter
from oscura.filtering.design import (
    BandPassFilter,
    BandStopFilter,
    BesselFilter,
    ButterworthFilter,
    ChebyshevType1Filter,
    ChebyshevType2Filter,
    EllipticFilter,
    HighPassFilter,
    LowPassFilter,
    auto_design_filter,
    design_filter,
    design_filter_spec,
    suggest_filter_type,
)

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
def high_sample_rate() -> float:
    """High sample rate for band-edge testing."""
    return 100_000.0  # 100 kHz


# =============================================================================
# design_filter() Tests
# =============================================================================


@pytest.mark.unit
class TestDesignFilter:
    """Test design_filter() function."""

    def test_butterworth_lowpass_sos(self, sample_rate: float):
        """Test Butterworth lowpass filter design with SOS output."""
        filt = design_filter(
            filter_type="butterworth",
            cutoff=1000.0,
            sample_rate=sample_rate,
            order=4,
            btype="lowpass",
            output="sos",
        )

        assert isinstance(filt, IIRFilter)
        assert filt.sos is not None
        assert filt.sos.shape[1] == 6  # SOS format
        assert filt.order == 4  # 4th order filter
        assert filt.is_stable

    def test_butterworth_lowpass_ba(self, sample_rate: float):
        """Test Butterworth lowpass filter design with B/A output."""
        filt = design_filter(
            filter_type="butterworth",
            cutoff=1000.0,
            sample_rate=sample_rate,
            order=4,
            btype="lowpass",
            output="ba",
        )

        assert isinstance(filt, IIRFilter)
        assert filt.ba is not None
        b, a = filt.ba
        assert len(b) > 0
        assert len(a) > 0
        assert filt.is_stable

    def test_chebyshev1_ba_output(self, sample_rate: float):
        """Test Chebyshev Type I filter with B/A output."""
        filt = design_filter(
            filter_type="chebyshev1",
            cutoff=1000.0,
            sample_rate=sample_rate,
            order=4,
            ripple_db=0.5,
            output="ba",
        )

        assert filt.ba is not None
        assert filt.is_stable

    def test_chebyshev2_ba_output(self, sample_rate: float):
        """Test Chebyshev Type II filter with B/A output."""
        filt = design_filter(
            filter_type="chebyshev2",
            cutoff=1000.0,
            sample_rate=sample_rate,
            order=4,
            stopband_atten_db=40.0,
            output="ba",
        )

        assert filt.ba is not None
        assert filt.is_stable

    def test_bessel_ba_output(self, sample_rate: float):
        """Test Bessel filter with B/A output."""
        filt = design_filter(
            filter_type="bessel",
            cutoff=1000.0,
            sample_rate=sample_rate,
            order=4,
            output="ba",
        )

        assert filt.ba is not None
        assert filt.is_stable

    def test_elliptic_ba_output(self, sample_rate: float):
        """Test Elliptic filter with B/A output."""
        filt = design_filter(
            filter_type="elliptic",
            cutoff=1000.0,
            sample_rate=sample_rate,
            order=4,
            ripple_db=0.5,
            stopband_atten_db=60.0,
            output="ba",
        )

        assert filt.ba is not None
        assert filt.is_stable

    def test_butterworth_highpass(self, sample_rate: float):
        """Test Butterworth highpass filter design."""
        filt = design_filter(
            filter_type="butterworth",
            cutoff=1000.0,
            sample_rate=sample_rate,
            order=4,
            btype="highpass",
        )

        assert filt.is_stable
        # Verify frequency response: attenuate low frequencies
        _w, h = filt.get_frequency_response(512)
        # Low frequency should be attenuated
        assert np.abs(h[10]) < 0.5
        # High frequency should pass
        assert np.abs(h[-50]) > 0.5

    def test_butterworth_bandpass(self, sample_rate: float):
        """Test Butterworth bandpass filter design."""
        filt = design_filter(
            filter_type="butterworth",
            cutoff=(500.0, 1500.0),
            sample_rate=sample_rate,
            order=4,
            btype="bandpass",
        )

        assert filt.is_stable
        # Verify frequency response: pass middle frequencies
        _w, h = filt.get_frequency_response(512)
        # Low frequency should be attenuated
        assert np.abs(h[10]) < 0.7
        # High frequency should be attenuated
        assert np.abs(h[-10]) < 0.7
        # Middle should pass
        assert np.abs(h[len(h) // 4]) > 0.1  # Less strict check

    def test_butterworth_bandstop(self, sample_rate: float):
        """Test Butterworth bandstop filter design."""
        filt = design_filter(
            filter_type="butterworth",
            cutoff=(900.0, 1100.0),
            sample_rate=sample_rate,
            order=4,
            btype="bandstop",
        )

        assert filt.is_stable

    def test_chebyshev1_design(self, sample_rate: float):
        """Test Chebyshev Type I filter design."""
        filt = design_filter(
            filter_type="chebyshev1",
            cutoff=1000.0,
            sample_rate=sample_rate,
            order=4,
            btype="lowpass",
            ripple_db=0.5,
        )

        assert filt.is_stable
        assert filt.order > 0

    def test_chebyshev2_design(self, sample_rate: float):
        """Test Chebyshev Type II filter design."""
        filt = design_filter(
            filter_type="chebyshev2",
            cutoff=1000.0,
            sample_rate=sample_rate,
            order=4,
            btype="lowpass",
            stopband_atten_db=40.0,
        )

        assert filt.is_stable
        assert filt.order > 0

    def test_bessel_design(self, sample_rate: float):
        """Test Bessel filter design."""
        filt = design_filter(
            filter_type="bessel",
            cutoff=1000.0,
            sample_rate=sample_rate,
            order=4,
            btype="lowpass",
        )

        assert filt.is_stable
        assert filt.order > 0

    def test_elliptic_design(self, sample_rate: float):
        """Test Elliptic filter design."""
        filt = design_filter(
            filter_type="elliptic",
            cutoff=1000.0,
            sample_rate=sample_rate,
            order=4,
            btype="lowpass",
            ripple_db=0.5,
            stopband_atten_db=60.0,
        )

        assert filt.is_stable
        assert filt.order > 0

    def test_analog_filter_design(self, sample_rate: float):
        """Test analog filter design (s-domain)."""
        filt = design_filter(
            filter_type="butterworth",
            cutoff=1000.0,  # Hz, not normalized
            sample_rate=sample_rate,
            order=4,
            btype="lowpass",
            analog=True,
        )

        # Analog filters use s-domain, stability check may differ
        # Just verify the filter was created
        assert filt.sos is not None or filt.ba is not None

    def test_invalid_filter_type_raises(self, sample_rate: float):
        """Test that invalid filter type raises error."""
        with pytest.raises(AnalysisError, match="Unknown filter type"):
            design_filter(
                filter_type="invalid",  # type: ignore[arg-type]
                cutoff=1000.0,
                sample_rate=sample_rate,
                order=4,
            )

    def test_cutoff_above_nyquist_raises(self, sample_rate: float):
        """Test that cutoff above Nyquist raises error."""
        with pytest.raises(AnalysisError, match="Normalized cutoff"):
            design_filter(
                filter_type="butterworth",
                cutoff=6000.0,  # Above Nyquist (5000 Hz)
                sample_rate=sample_rate,
                order=4,
            )

    def test_bandpass_cutoff_above_nyquist_raises(self, sample_rate: float):
        """Test that bandpass cutoff above Nyquist raises error."""
        with pytest.raises(AnalysisError, match="Normalized cutoff"):
            design_filter(
                filter_type="butterworth",
                cutoff=(1000.0, 6000.0),  # High edge above Nyquist
                sample_rate=sample_rate,
                order=4,
                btype="bandpass",
            )

    def test_zero_cutoff_raises(self, sample_rate: float):
        """Test that zero cutoff raises error."""
        with pytest.raises(AnalysisError, match="Normalized cutoff"):
            design_filter(
                filter_type="butterworth",
                cutoff=0.0,
                sample_rate=sample_rate,
                order=4,
            )

    def test_negative_cutoff_raises(self, sample_rate: float):
        """Test that negative cutoff raises error."""
        with pytest.raises(AnalysisError):
            design_filter(
                filter_type="butterworth",
                cutoff=-1000.0,
                sample_rate=sample_rate,
                order=4,
            )


# =============================================================================
# design_filter_spec() Tests
# =============================================================================


@pytest.mark.unit
class TestDesignFilterSpec:
    """Test design_filter_spec() function."""

    def test_lowpass_spec_design(self, sample_rate: float):
        """Test lowpass filter design from specifications."""
        filt = design_filter_spec(
            passband=1000.0,
            stopband=2000.0,
            sample_rate=sample_rate,
            passband_ripple=1.0,
            stopband_atten=40.0,
            filter_type="butterworth",
        )

        assert isinstance(filt, IIRFilter)
        assert filt.is_stable
        assert filt.order > 0

    def test_highpass_spec_design(self, sample_rate: float):
        """Test highpass filter design from specifications."""
        filt = design_filter_spec(
            passband=2000.0,
            stopband=1000.0,  # Stopband lower than passband = highpass
            sample_rate=sample_rate,
            passband_ripple=1.0,
            stopband_atten=40.0,
            filter_type="butterworth",
        )

        assert filt.is_stable

    def test_bandpass_spec_design(self, sample_rate: float):
        """Test bandpass filter design from specifications."""
        filt = design_filter_spec(
            passband=(1000.0, 2000.0),
            stopband=(500.0, 2500.0),  # Wider stopband
            sample_rate=sample_rate,
            passband_ripple=1.0,
            stopband_atten=40.0,
            filter_type="elliptic",
        )

        assert filt.is_stable

    def test_bandstop_spec_design(self, sample_rate: float):
        """Test bandstop filter design from specifications."""
        filt = design_filter_spec(
            passband=(500.0, 2500.0),  # Wider passband
            stopband=(1000.0, 2000.0),  # Narrower stopband
            sample_rate=sample_rate,
            passband_ripple=1.0,
            stopband_atten=40.0,
            filter_type="butterworth",
        )

        assert filt.is_stable

    def test_elliptic_spec_design(self, sample_rate: float):
        """Test elliptic filter with spec-based design."""
        filt = design_filter_spec(
            passband=1000.0,
            stopband=1500.0,
            sample_rate=sample_rate,
            passband_ripple=0.5,
            stopband_atten=60.0,
            filter_type="elliptic",
        )

        assert filt.is_stable
        # Elliptic should have sharper transition
        assert filt.order > 0

    def test_chebyshev1_spec_design(self, sample_rate: float):
        """Test Chebyshev Type I filter with spec-based design."""
        filt = design_filter_spec(
            passband=1000.0,
            stopband=2000.0,
            sample_rate=sample_rate,
            passband_ripple=1.0,
            stopband_atten=40.0,
            filter_type="chebyshev1",
        )

        assert filt.is_stable

    def test_chebyshev2_spec_design(self, sample_rate: float):
        """Test Chebyshev Type II filter with spec-based design."""
        filt = design_filter_spec(
            passband=1000.0,
            stopband=2000.0,
            sample_rate=sample_rate,
            passband_ripple=1.0,
            stopband_atten=40.0,
            filter_type="chebyshev2",
        )

        assert filt.is_stable

    def test_bessel_spec_design(self, sample_rate: float):
        """Test Bessel filter with spec-based design (uses Butterworth ord)."""
        filt = design_filter_spec(
            passband=1000.0,
            stopband=2000.0,
            sample_rate=sample_rate,
            passband_ripple=1.0,
            stopband_atten=40.0,
            filter_type="bessel",
        )

        assert filt.is_stable

    def test_spec_design_analog(self, sample_rate: float):
        """Test analog filter design from specifications."""
        filt = design_filter_spec(
            passband=1000.0,
            stopband=2000.0,
            sample_rate=sample_rate,
            passband_ripple=1.0,
            stopband_atten=40.0,
            filter_type="butterworth",
            analog=True,
        )

        # Analog filters use s-domain
        assert filt.sos is not None or filt.ba is not None

    def test_spec_design_tight_transition(self, high_sample_rate: float):
        """Test filter design with tight transition band."""
        filt = design_filter_spec(
            passband=10000.0,
            stopband=12000.0,  # Tight transition
            sample_rate=high_sample_rate,
            passband_ripple=0.5,
            stopband_atten=60.0,
            filter_type="elliptic",
        )

        assert filt.is_stable
        # Should produce higher order filter for tight transition
        assert filt.order > 4

    def test_spec_design_order_failure_raises(self, sample_rate: float):
        """Test that impossible specs raise error."""
        # Impossible specs: passband above Nyquist
        with pytest.raises(AnalysisError):
            design_filter_spec(
                passband=6000.0,  # Above Nyquist (5000 Hz)
                stopband=7000.0,  # Also above Nyquist
                sample_rate=sample_rate,
                passband_ripple=1.0,
                stopband_atten=40.0,
                filter_type="butterworth",
            )


# =============================================================================
# Convenience Filter Classes Tests
# =============================================================================


@pytest.mark.unit
class TestLowPassFilter:
    """Test LowPassFilter convenience class."""

    def test_lowpass_creation(self, sample_rate: float):
        """Test creating LowPassFilter."""
        lpf = LowPassFilter(cutoff=1000.0, sample_rate=sample_rate, order=4)

        assert lpf.cutoff == 1000.0
        assert lpf.sample_rate == sample_rate
        assert lpf.order == 4  # 4th order filter
        assert lpf.is_stable

    def test_lowpass_apply(self, sample_rate: float, test_trace: WaveformTrace):
        """Test applying LowPassFilter."""
        lpf = LowPassFilter(cutoff=500.0, sample_rate=sample_rate, order=4)
        result = lpf.apply(test_trace)

        assert isinstance(result, WaveformTrace)
        assert len(result.data) == len(test_trace.data)

    def test_lowpass_custom_filter_type(self, sample_rate: float):
        """Test LowPassFilter with custom filter type."""
        lpf = LowPassFilter(
            cutoff=1000.0,
            sample_rate=sample_rate,
            order=4,
            filter_type="chebyshev1",
            ripple_db=0.5,
        )

        assert lpf.is_stable

    def test_lowpass_elliptic(self, sample_rate: float):
        """Test LowPassFilter with elliptic type."""
        lpf = LowPassFilter(
            cutoff=1000.0,
            sample_rate=sample_rate,
            order=3,
            filter_type="elliptic",
            ripple_db=0.5,
            stopband_atten_db=60.0,
        )

        assert lpf.is_stable


@pytest.mark.unit
class TestHighPassFilter:
    """Test HighPassFilter convenience class."""

    def test_highpass_creation(self, sample_rate: float):
        """Test creating HighPassFilter."""
        hpf = HighPassFilter(cutoff=1000.0, sample_rate=sample_rate, order=4)

        assert hpf.cutoff == 1000.0
        assert hpf.sample_rate == sample_rate
        assert hpf.is_stable

    def test_highpass_apply(self, sample_rate: float, test_trace: WaveformTrace):
        """Test applying HighPassFilter."""
        hpf = HighPassFilter(cutoff=500.0, sample_rate=sample_rate, order=4)
        result = hpf.apply(test_trace)

        assert isinstance(result, WaveformTrace)
        assert len(result.data) == len(test_trace.data)

    def test_highpass_custom_filter_type(self, sample_rate: float):
        """Test HighPassFilter with custom filter type."""
        hpf = HighPassFilter(
            cutoff=1000.0,
            sample_rate=sample_rate,
            order=4,
            filter_type="bessel",
        )

        assert hpf.is_stable


@pytest.mark.unit
class TestBandPassFilter:
    """Test BandPassFilter convenience class."""

    def test_bandpass_creation(self, sample_rate: float):
        """Test creating BandPassFilter."""
        bpf = BandPassFilter(low=500.0, high=1500.0, sample_rate=sample_rate, order=4)

        assert bpf.passband == (500.0, 1500.0)
        assert bpf.sample_rate == sample_rate
        assert bpf.is_stable

    def test_bandpass_apply(self, sample_rate: float, test_trace: WaveformTrace):
        """Test applying BandPassFilter."""
        bpf = BandPassFilter(low=500.0, high=1500.0, sample_rate=sample_rate, order=4)
        result = bpf.apply(test_trace)

        assert isinstance(result, WaveformTrace)
        assert len(result.data) == len(test_trace.data)

    def test_bandpass_custom_filter_type(self, sample_rate: float):
        """Test BandPassFilter with custom filter type."""
        bpf = BandPassFilter(
            low=500.0,
            high=1500.0,
            sample_rate=sample_rate,
            order=4,
            filter_type="chebyshev2",
            stopband_atten_db=50.0,
        )

        assert bpf.is_stable


@pytest.mark.unit
class TestBandStopFilter:
    """Test BandStopFilter convenience class."""

    def test_bandstop_creation(self, sample_rate: float):
        """Test creating BandStopFilter."""
        bsf = BandStopFilter(low=900.0, high=1100.0, sample_rate=sample_rate, order=4)

        assert bsf.stopband == (900.0, 1100.0)
        assert bsf.sample_rate == sample_rate
        assert bsf.is_stable

    def test_bandstop_apply(self, sample_rate: float, test_trace: WaveformTrace):
        """Test applying BandStopFilter (notch filter)."""
        bsf = BandStopFilter(low=900.0, high=1100.0, sample_rate=sample_rate, order=4)
        result = bsf.apply(test_trace)

        assert isinstance(result, WaveformTrace)
        assert len(result.data) == len(test_trace.data)

    def test_bandstop_60hz_notch(self):
        """Test 60 Hz notch filter (common use case)."""
        bsf = BandStopFilter(low=58.0, high=62.0, sample_rate=1000.0, order=4)

        assert bsf.stopband == (58.0, 62.0)
        assert bsf.is_stable


# =============================================================================
# Filter Type Classes Tests
# =============================================================================


@pytest.mark.unit
class TestFilterTypeClasses:
    """Test filter type convenience classes."""

    def test_butterworth_filter_class(self, sample_rate: float):
        """Test ButterworthFilter class."""
        filt = ButterworthFilter(cutoff=1000.0, sample_rate=sample_rate, order=4)

        assert filt.is_stable
        assert filt.order > 0

    def test_butterworth_bandpass(self, sample_rate: float):
        """Test ButterworthFilter with bandpass."""
        filt = ButterworthFilter(
            cutoff=(500.0, 1500.0), sample_rate=sample_rate, order=4, btype="bandpass"
        )

        assert filt.is_stable

    def test_chebyshev_type1_filter_class(self, sample_rate: float):
        """Test ChebyshevType1Filter class."""
        filt = ChebyshevType1Filter(cutoff=1000.0, sample_rate=sample_rate, order=4, ripple_db=0.5)

        assert filt.is_stable

    def test_chebyshev_type2_filter_class(self, sample_rate: float):
        """Test ChebyshevType2Filter class."""
        filt = ChebyshevType2Filter(
            cutoff=1000.0, sample_rate=sample_rate, order=4, stopband_atten_db=40.0
        )

        assert filt.is_stable

    def test_bessel_filter_class(self, sample_rate: float):
        """Test BesselFilter class."""
        filt = BesselFilter(cutoff=1000.0, sample_rate=sample_rate, order=4)

        assert filt.is_stable

    def test_elliptic_filter_class(self, sample_rate: float):
        """Test EllipticFilter class."""
        filt = EllipticFilter(
            cutoff=1000.0,
            sample_rate=sample_rate,
            order=4,
            ripple_db=0.5,
            stopband_atten_db=60.0,
        )

        assert filt.is_stable


# =============================================================================
# suggest_filter_type() Tests
# =============================================================================


@pytest.mark.unit
class TestSuggestFilterType:
    """Test suggest_filter_type() function."""

    def test_suggest_elliptic_sharp_transition(self):
        """Test suggesting elliptic for sharp transition with ripple tolerance."""
        ftype = suggest_filter_type(
            transition_bandwidth=0.1,  # Sharp transition
            passband_ripple_db=0.5,  # Can tolerate ripple
            stopband_atten_db=60.0,
        )

        assert ftype == "elliptic"

    def test_suggest_chebyshev2_moderate_sharp(self):
        """Test suggesting Chebyshev2 for moderate sharpness with low ripple."""
        ftype = suggest_filter_type(
            transition_bandwidth=0.15,  # Moderate transition
            passband_ripple_db=0.05,  # Low passband ripple
            stopband_atten_db=50.0,
        )

        assert ftype == "chebyshev2"

    def test_suggest_chebyshev1_moderate_ripple(self):
        """Test suggesting Chebyshev1 for moderate sharpness with ripple tolerance."""
        ftype = suggest_filter_type(
            transition_bandwidth=0.25,
            passband_ripple_db=1.0,  # Can tolerate ripple
            stopband_atten_db=40.0,
        )

        assert ftype == "chebyshev1"

    def test_suggest_bessel_low_attenuation(self):
        """Test suggesting Bessel for low attenuation (phase linearity)."""
        ftype = suggest_filter_type(
            transition_bandwidth=0.4,
            passband_ripple_db=0.1,
            stopband_atten_db=30.0,  # Low attenuation
        )

        assert ftype == "bessel"

    def test_suggest_butterworth_default(self):
        """Test suggesting Butterworth as default for balanced requirements."""
        ftype = suggest_filter_type(
            transition_bandwidth=0.35,
            passband_ripple_db=0.1,
            stopband_atten_db=45.0,
        )

        assert ftype == "butterworth"

    def test_suggest_very_sharp_elliptic(self):
        """Test elliptic suggestion for very sharp transition."""
        ftype = suggest_filter_type(
            transition_bandwidth=0.05,  # Very sharp
            passband_ripple_db=1.0,
            stopband_atten_db=80.0,
        )

        assert ftype == "elliptic"


# =============================================================================
# auto_design_filter() Tests
# =============================================================================


@pytest.mark.unit
class TestAutoDesignFilter:
    """Test auto_design_filter() function."""

    def test_auto_design_with_suggestion(self, sample_rate: float):
        """Test automatic filter design with type suggestion."""
        filt, info = auto_design_filter(
            passband=1000.0,
            stopband=1500.0,
            sample_rate=sample_rate,
            passband_ripple_db=0.5,
            stopband_atten_db=60.0,
            suggest_type=True,
        )

        assert isinstance(filt, IIRFilter)
        assert filt.is_stable
        assert info["filter_type"] in [
            "butterworth",
            "chebyshev1",
            "chebyshev2",
            "bessel",
            "elliptic",
        ]
        assert info["order"] > 0
        assert info["cutoff"] == 1000.0
        assert info["transition_bandwidth"] == 500.0
        assert info["passband_ripple_db"] == 0.5
        assert info["stopband_atten_db"] == 60.0

    def test_auto_design_without_suggestion(self, sample_rate: float):
        """Test automatic filter design without type suggestion (defaults to Butterworth)."""
        filt, info = auto_design_filter(
            passband=1000.0,
            stopband=2000.0,
            sample_rate=sample_rate,
            suggest_type=False,
        )

        assert filt.is_stable
        assert info["filter_type"] == "butterworth"

    def test_auto_design_bandpass(self, sample_rate: float):
        """Test automatic bandpass filter design."""
        filt, info = auto_design_filter(
            passband=(1000.0, 2000.0),
            stopband=(500.0, 2500.0),
            sample_rate=sample_rate,
            passband_ripple_db=1.0,
            stopband_atten_db=40.0,
        )

        assert filt.is_stable
        assert info["order"] > 0
        # Transition bandwidth is average of both edges
        expected_trans = ((1000 - 500) + (2500 - 2000)) / 2.0
        assert info["transition_bandwidth"] == expected_trans

    def test_auto_design_bandstop(self, sample_rate: float):
        """Test automatic bandstop filter design."""
        filt, info = auto_design_filter(
            passband=(500.0, 2500.0),
            stopband=(1000.0, 2000.0),
            sample_rate=sample_rate,
        )

        assert filt.is_stable
        assert info["order"] > 0

    def test_auto_design_highpass(self, sample_rate: float):
        """Test automatic highpass filter design."""
        filt, info = auto_design_filter(
            passband=2000.0,
            stopband=1000.0,  # Stopband lower = highpass
            sample_rate=sample_rate,
        )

        assert filt.is_stable
        assert info["transition_bandwidth"] == 1000.0

    def test_auto_design_tight_specs(self, high_sample_rate: float):
        """Test automatic design with tight specifications."""
        filt, info = auto_design_filter(
            passband=10000.0,
            stopband=11000.0,  # Narrow transition
            sample_rate=high_sample_rate,
            passband_ripple_db=0.1,
            stopband_atten_db=80.0,  # High attenuation
            suggest_type=True,
        )

        assert filt.is_stable
        # Should suggest elliptic for tight specs
        assert info["filter_type"] == "elliptic"
        # Should have higher order
        assert info["order"] > 6

    def test_auto_design_relaxed_specs(self, sample_rate: float):
        """Test automatic design with relaxed specifications."""
        filt, info = auto_design_filter(
            passband=1000.0,
            stopband=3000.0,  # Wide transition
            sample_rate=sample_rate,
            passband_ripple_db=2.0,
            stopband_atten_db=30.0,  # Low attenuation
            suggest_type=True,
        )

        assert filt.is_stable
        # Relaxed specs - might suggest various filter types
        assert info["filter_type"] in ["butterworth", "bessel", "chebyshev1", "chebyshev2"]


# =============================================================================
# Filter Verification Tests
# =============================================================================


@pytest.mark.unit
class TestFilterVerification:
    """Test that designed filters meet specifications."""

    def test_lowpass_meets_spec(self, sample_rate: float):
        """Test that lowpass filter meets passband/stopband specifications."""
        passband = 1000.0
        stopband = 2000.0
        filt = design_filter_spec(
            passband=passband,
            stopband=stopband,
            sample_rate=sample_rate,
            passband_ripple=1.0,
            stopband_atten=40.0,
            filter_type="elliptic",
        )

        # Compute frequency response
        freqs = np.linspace(0, sample_rate / 2, 1024)
        h = filt.get_transfer_function(freqs)
        mag_db = 20 * np.log10(np.abs(h) + 1e-10)

        # Find passband and stopband indices
        passband_idx = np.where(freqs <= passband)[0]
        stopband_idx = np.where(freqs >= stopband)[0]

        # Check passband: should be within ripple spec
        passband_mag = mag_db[passband_idx]
        # Allow some tolerance for passband ripple
        assert np.all(passband_mag > -2.0)  # Should be close to 0 dB

        # Check stopband: should meet attenuation spec
        stopband_mag = mag_db[stopband_idx]
        # Should be attenuated (allowing some tolerance)
        assert np.mean(stopband_mag) < -30.0

    def test_highpass_meets_spec(self, sample_rate: float):
        """Test that highpass filter meets specifications."""
        filt = design_filter_spec(
            passband=2000.0,
            stopband=1000.0,
            sample_rate=sample_rate,
            passband_ripple=1.0,
            stopband_atten=40.0,
            filter_type="butterworth",
        )

        freqs = np.linspace(0, sample_rate / 2, 1024)
        h = filt.get_transfer_function(freqs)
        mag_db = 20 * np.log10(np.abs(h) + 1e-10)

        # Stopband should be attenuated
        stopband_idx = np.where(freqs <= 1000)[0]
        assert np.mean(mag_db[stopband_idx]) < -20.0

    def test_filter_stability(self, sample_rate: float):
        """Test that all designed filters are stable."""
        for filter_type in ["butterworth", "chebyshev1", "chebyshev2", "elliptic", "bessel"]:
            filt = design_filter(
                filter_type=filter_type,  # type: ignore[arg-type]
                cutoff=1000.0,
                sample_rate=sample_rate,
                order=4,
                ripple_db=1.0,
                stopband_atten_db=40.0,
            )
            assert filt.is_stable, f"{filter_type} filter is not stable"

    def test_filter_causality(self, sample_rate: float, test_trace: WaveformTrace):
        """Test that causal filtering doesn't look ahead."""
        filt = LowPassFilter(cutoff=500.0, sample_rate=sample_rate, order=4)

        # Apply causal filter
        result = filt.apply(test_trace, filtfilt=False)

        # Output should be delayed, not leading
        # First few samples should show filter transient
        assert not np.allclose(result.data[:10], test_trace.data[:10])


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


@pytest.mark.unit
class TestFilteringDesignEdgeCases:
    """Test edge cases and error handling."""

    def test_very_low_order(self, sample_rate: float):
        """Test filter with order 1."""
        filt = design_filter(
            filter_type="butterworth",
            cutoff=1000.0,
            sample_rate=sample_rate,
            order=1,
        )

        assert filt.is_stable

    def test_high_order_filter(self, sample_rate: float):
        """Test filter with high order."""
        filt = design_filter(
            filter_type="butterworth",
            cutoff=1000.0,
            sample_rate=sample_rate,
            order=10,
        )

        assert filt.is_stable
        assert filt.order == 10  # 10th order filter

    def test_cutoff_near_nyquist(self, sample_rate: float):
        """Test filter with cutoff very close to Nyquist."""
        filt = design_filter(
            filter_type="butterworth",
            cutoff=4900.0,  # Close to Nyquist (5000 Hz)
            sample_rate=sample_rate,
            order=2,
        )

        assert filt.is_stable

    def test_very_low_cutoff(self, sample_rate: float):
        """Test filter with very low cutoff frequency."""
        filt = design_filter(
            filter_type="butterworth",
            cutoff=10.0,  # Very low
            sample_rate=sample_rate,
            order=4,
        )

        assert filt.is_stable

    def test_bandpass_narrow(self, sample_rate: float):
        """Test narrow bandpass filter."""
        filt = design_filter(
            filter_type="butterworth",
            cutoff=(990.0, 1010.0),  # Very narrow (20 Hz)
            sample_rate=sample_rate,
            order=4,
            btype="bandpass",
        )

        assert filt.is_stable

    def test_bandpass_wide(self, sample_rate: float):
        """Test wide bandpass filter."""
        filt = design_filter(
            filter_type="butterworth",
            cutoff=(100.0, 4900.0),  # Very wide
            sample_rate=sample_rate,
            order=2,
            btype="bandpass",
        )

        assert filt.is_stable

    def test_zero_ripple_chebyshev(self, sample_rate: float):
        """Test Chebyshev filter with very small ripple."""
        filt = design_filter(
            filter_type="chebyshev1",
            cutoff=1000.0,
            sample_rate=sample_rate,
            order=4,
            ripple_db=0.01,  # Very small ripple
        )

        assert filt.is_stable

    def test_high_stopband_atten(self, sample_rate: float):
        """Test filter with very high stopband attenuation."""
        filt = design_filter(
            filter_type="elliptic",
            cutoff=1000.0,
            sample_rate=sample_rate,
            order=6,
            ripple_db=0.5,
            stopband_atten_db=100.0,  # Very high attenuation
        )

        assert filt.is_stable

    def test_bandpass_inverted_order_raises(self, sample_rate: float):
        """Test that bandpass with inverted cutoffs raises error."""
        with pytest.raises(AnalysisError):  # Wrapped scipy error
            design_filter(
                filter_type="butterworth",
                cutoff=(2000.0, 1000.0),  # Inverted!
                sample_rate=sample_rate,
                order=4,
                btype="bandpass",
            )

    def test_spec_with_equal_bands_raises(self, sample_rate: float):
        """Test that equal passband and stopband raises error."""
        with pytest.raises(AnalysisError):  # Wrapped scipy error
            design_filter_spec(
                passband=1000.0,
                stopband=1000.0,  # Same as passband
                sample_rate=sample_rate,
            )


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.unit
class TestFilterIntegration:
    """Test filter design integration with application."""

    def test_design_and_apply(self, sample_rate: float, test_trace: WaveformTrace):
        """Test complete workflow: design and apply filter."""
        # Design filter
        filt = design_filter(
            filter_type="butterworth",
            cutoff=500.0,
            sample_rate=sample_rate,
            order=4,
            btype="lowpass",
        )

        # Apply to trace
        result = filt.apply(test_trace)

        # Verify result
        assert len(result.data) == len(test_trace.data)
        # Signal should be smoother (lower variance)
        assert np.var(result.data) < np.var(test_trace.data)

    def test_spec_design_and_apply(self, sample_rate: float, test_trace: WaveformTrace):
        """Test spec-based design and application."""
        filt = design_filter_spec(
            passband=300.0,
            stopband=700.0,
            sample_rate=sample_rate,
            passband_ripple=1.0,
            stopband_atten=40.0,
        )

        result = filt.apply(test_trace)
        assert len(result.data) == len(test_trace.data)

    def test_auto_design_and_apply(self, sample_rate: float, test_trace: WaveformTrace):
        """Test auto design and application."""
        filt, info = auto_design_filter(
            passband=300.0,
            stopband=700.0,
            sample_rate=sample_rate,
        )

        result = filt.apply(test_trace)
        assert len(result.data) == len(test_trace.data)
        assert info["filter_type"] is not None

    def test_multiple_filter_types(self, sample_rate: float, test_trace: WaveformTrace):
        """Test applying different filter types to same signal."""
        results = {}
        for ftype in ["butterworth", "chebyshev1", "chebyshev2", "bessel", "elliptic"]:
            filt = design_filter(
                filter_type=ftype,  # type: ignore[arg-type]
                cutoff=500.0,
                sample_rate=sample_rate,
                order=4,
                ripple_db=1.0,
                stopband_atten_db=40.0,
            )
            result = filt.apply(test_trace)
            results[ftype] = result.data

        # All results should be different
        for ftype1, data1 in results.items():
            for ftype2, data2 in results.items():
                if ftype1 != ftype2:
                    # Different filters produce different outputs
                    assert not np.allclose(data1, data2)
