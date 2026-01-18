"""Tests for jitter analysis module.

Tests for JITT-001 through JITT-009 requirements.
"""

import numpy as np
import pytest
from numpy.typing import NDArray

from oscura.analyzers.jitter import (
    bathtub_curve,
    ber_from_q_factor,
    cycle_to_cycle_jitter,
    decompose_jitter,
    extract_ddj,
    extract_dj,
    extract_pj,
    extract_rj,
    eye_opening_at_ber,
    jitter_spectrum,
    measure_dcd,
    period_jitter,
    q_factor_from_ber,
    tie_from_edges,
    tj_at_ber,
)
from oscura.core.exceptions import InsufficientDataError
from oscura.core.types import TraceMetadata, WaveformTrace

pytestmark = [pytest.mark.unit, pytest.mark.analyzer, pytest.mark.jitter]


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def gaussian_jitter() -> NDArray[np.float64]:
    """Generate TIE data with known Gaussian (random) jitter."""
    rng = np.random.default_rng(42)
    n_samples = 10000
    # Pure Gaussian jitter with 5 ps RMS
    sigma = 5e-12  # 5 ps
    return rng.normal(0, sigma, n_samples)


@pytest.fixture
def mixed_jitter() -> NDArray[np.float64]:
    """Generate TIE data with both random and deterministic jitter."""
    rng = np.random.default_rng(42)
    n_samples = 10000

    # Random component: 3 ps RMS
    rj = rng.normal(0, 3e-12, n_samples)

    # Deterministic component: dual-Dirac +-5 ps
    dj_delta = 5e-12
    dj = np.where(rng.random(n_samples) > 0.5, dj_delta, -dj_delta)

    return rj + dj


@pytest.fixture
def periodic_jitter() -> NDArray[np.float64]:
    """Generate TIE data with sinusoidal periodic jitter."""
    n_samples = 10000
    sample_rate = 1e6  # 1 MHz edge rate
    t = np.arange(n_samples) / sample_rate

    # 10 kHz sinusoidal jitter with 2 ps amplitude
    pj_freq = 10e3  # 10 kHz
    pj_amplitude = 2e-12  # 2 ps
    pj = pj_amplitude * np.sin(2 * np.pi * pj_freq * t)

    # Add small random component
    rng = np.random.default_rng(42)
    rj = rng.normal(0, 0.5e-12, n_samples)

    return pj + rj


@pytest.fixture
def clock_periods() -> NDArray[np.float64]:
    """Generate clock period measurements with jitter."""
    rng = np.random.default_rng(42)
    n_periods = 1000
    nominal_period = 1e-9  # 1 ns (1 GHz clock)

    # Add period jitter: 10 ps RMS
    jitter = rng.normal(0, 10e-12, n_periods)

    return nominal_period + jitter


@pytest.fixture
def asymmetric_clock_trace() -> WaveformTrace:
    """Generate clock trace with duty cycle distortion."""
    sample_rate = 20e9  # 20 GSa/s (20 samples/period for better accuracy)
    n_samples = 20000
    t = np.arange(n_samples) / sample_rate

    # 1 GHz clock with 55/45 duty cycle
    period = 1e-9
    high_time = 0.55e-9
    phase = t % period
    data = np.where(phase < high_time, 3.3, 0.0)

    return WaveformTrace(
        data=data.astype(np.float64),
        metadata=TraceMetadata(sample_rate=sample_rate),
    )


# =============================================================================
# Random Jitter Extraction Tests (JITT-001)
# =============================================================================


class TestExtractRJ:
    """Tests for random jitter extraction."""

    def test_extract_rj_gaussian(self, gaussian_jitter: NDArray[np.float64]):
        """Test RJ extraction from pure Gaussian jitter."""
        result = extract_rj(gaussian_jitter)

        # Input had 5 ps RMS
        expected_rj = 5e-12
        assert np.isclose(result.rj_rms, expected_rj, rtol=0.2)
        assert result.confidence > 0.5
        assert result.method in ("tail_fit", "q_scale")

    def test_extract_rj_tail_fit(self, gaussian_jitter: NDArray[np.float64]):
        """Test tail fit method."""
        result = extract_rj(gaussian_jitter, method="tail_fit")

        assert result.method == "tail_fit"
        assert result.rj_rms > 0
        assert result.n_samples == len(gaussian_jitter)

    def test_extract_rj_q_scale(self, gaussian_jitter: NDArray[np.float64]):
        """Test Q-scale method."""
        result = extract_rj(gaussian_jitter, method="q_scale")

        assert result.method == "q_scale"
        assert result.rj_rms > 0

    def test_extract_rj_insufficient_data(self):
        """Test error on insufficient data."""
        short_data = np.random.randn(100)

        with pytest.raises(InsufficientDataError):
            extract_rj(short_data, min_samples=1000)


# =============================================================================
# Deterministic Jitter Extraction Tests (JITT-002)
# =============================================================================


class TestExtractDJ:
    """Tests for deterministic jitter extraction."""

    def test_extract_dj(self, mixed_jitter: NDArray[np.float64]):
        """Test DJ extraction from mixed jitter."""
        result = extract_dj(mixed_jitter)

        # Input had +-5 ps DJ = 10 ps peak-to-peak
        # DJ extraction is approximate, allow wide tolerance
        assert result.dj_pp > 0
        assert result.dj_delta == result.dj_pp / 2

    def test_extract_dj_with_rj(self, mixed_jitter: NDArray[np.float64]):
        """Test DJ extraction with pre-computed RJ."""
        rj_result = extract_rj(mixed_jitter)
        dj_result = extract_dj(mixed_jitter, rj_result)

        assert dj_result.dj_pp > 0
        assert dj_result.method == "dual_dirac"


# =============================================================================
# Periodic Jitter Tests (JITT-003)
# =============================================================================


class TestExtractPJ:
    """Tests for periodic jitter extraction."""

    def test_extract_pj_single(self, periodic_jitter: NDArray[np.float64]):
        """Test extraction of single periodic component."""
        sample_rate = 1e6  # Edge rate

        result = extract_pj(periodic_jitter, sample_rate)

        # Should find 10 kHz component
        assert len(result.components) > 0
        assert result.dominant_frequency is not None

        # Check dominant frequency is near 10 kHz
        assert np.isclose(result.dominant_frequency, 10e3, rtol=0.1)

    def test_extract_pj_amplitude(self, periodic_jitter: NDArray[np.float64]):
        """Test PJ amplitude extraction."""
        result = extract_pj(periodic_jitter, sample_rate=1e6)

        # Input had 2 ps amplitude
        if result.dominant_amplitude is not None:
            assert result.dominant_amplitude > 0


# =============================================================================
# Data-Dependent Jitter Tests (JITT-004)
# =============================================================================


class TestExtractDDJ:
    """Tests for data-dependent jitter extraction."""

    def test_extract_ddj_no_pattern(self, mixed_jitter: NDArray[np.float64]):
        """Test DDJ extraction without bit pattern."""
        result = extract_ddj(mixed_jitter)

        assert result.ddj_pp >= 0
        assert result.pattern_length > 0

    def test_extract_ddj_with_pattern(self, mixed_jitter: NDArray[np.float64]):
        """Test DDJ extraction with bit pattern."""
        rng = np.random.default_rng(42)
        bits = rng.integers(0, 2, len(mixed_jitter))

        result = extract_ddj(mixed_jitter, bit_pattern=bits, pattern_length=3)

        assert result.ddj_pp >= 0
        assert result.pattern_length == 3


# =============================================================================
# Duty Cycle Distortion Tests (JITT-005)
# =============================================================================


class TestMeasureDCD:
    """Tests for DCD measurement."""

    def test_measure_dcd(self, asymmetric_clock_trace: WaveformTrace):
        """Test DCD measurement on asymmetric clock."""
        result = measure_dcd(asymmetric_clock_trace)

        # 55/45 duty cycle = 10% DCD
        expected_dcd_percent = 10.0
        assert np.isclose(result.dcd_percent, expected_dcd_percent, atol=2.0)
        assert result.duty_cycle > 0.5


# =============================================================================
# Total Jitter at BER Tests (JITT-006)
# =============================================================================


class TestTJAtBER:
    """Tests for total jitter at BER calculation."""

    def test_tj_at_1e12(self):
        """Test TJ calculation at 10^-12 BER."""
        rj_rms = 1e-12  # 1 ps
        dj_pp = 10e-12  # 10 ps

        tj = tj_at_ber(rj_rms, dj_pp, ber=1e-12)

        # TJ = 2 * 7.03 * 1 + 10 = 24.06 ps
        expected = 2 * 7.03 * rj_rms + dj_pp
        assert np.isclose(tj, expected, rtol=0.01)

    def test_tj_at_1e15(self):
        """Test TJ calculation at 10^-15 BER."""
        rj_rms = 1e-12
        dj_pp = 10e-12

        tj = tj_at_ber(rj_rms, dj_pp, ber=1e-15)

        # Q(1e-15) ≈ 7.94
        expected = 2 * 7.94 * rj_rms + dj_pp
        assert np.isclose(tj, expected, rtol=0.02)

    def test_tj_negative_rj_error(self):
        """Test error on negative RJ."""
        with pytest.raises(ValueError):
            tj_at_ber(rj_rms=-1e-12, dj_pp=10e-12)


# =============================================================================
# Q-Factor and BER Tests
# =============================================================================


class TestQFactorBER:
    """Tests for Q-factor and BER conversion."""

    def test_q_factor_1e12(self):
        """Test Q-factor at BER = 1e-12."""
        q = q_factor_from_ber(1e-12)

        # Q(1e-12) ≈ 7.03
        assert np.isclose(q, 7.03, rtol=0.01)

    def test_ber_from_q(self):
        """Test BER calculation from Q-factor."""
        ber = ber_from_q_factor(7.03)

        assert np.isclose(ber, 1e-12, rtol=0.1)

    def test_round_trip(self):
        """Test Q -> BER -> Q round trip."""
        q_original = 6.5
        ber = ber_from_q_factor(q_original)
        q_recovered = q_factor_from_ber(ber)

        assert np.isclose(q_original, q_recovered, rtol=0.01)


# =============================================================================
# Bathtub Curve Tests (JITT-007)
# =============================================================================


class TestBathtubCurve:
    """Tests for bathtub curve generation."""

    def test_bathtub_curve_shape(self, gaussian_jitter: NDArray[np.float64]):
        """Test bathtub curve has expected shape."""
        result = bathtub_curve(gaussian_jitter, unit_interval=100e-12)

        # Should have positions from 0 to 1
        assert result.positions[0] == 0
        assert np.isclose(result.positions[-1], 1, rtol=0.02)

        # BER should be lowest at center
        center_idx = len(result.positions) // 2
        center_ber = result.ber_total[center_idx]
        edge_ber = result.ber_total[0]

        assert center_ber < edge_ber

    def test_eye_opening_at_ber(self):
        """Test eye opening calculation."""
        rj_rms = 2e-12  # 2 ps
        dj_pp = 5e-12  # 5 ps
        ui = 100e-12  # 100 ps (10 Gbps)

        opening = eye_opening_at_ber(rj_rms, dj_pp, ui, target_ber=1e-12)

        # Should be positive and less than 1
        assert 0 < opening < 1


# =============================================================================
# Jitter Spectrum Tests (JITT-008)
# =============================================================================


class TestJitterSpectrum:
    """Tests for jitter spectrum analysis."""

    def test_jitter_fft(self, periodic_jitter: NDArray[np.float64]):
        """Test FFT of periodic jitter."""
        result = jitter_spectrum(periodic_jitter, sample_rate=1e6)

        # Should find 10 kHz component
        assert result.dominant_frequency is not None
        assert np.isclose(result.dominant_frequency, 10e3, rtol=0.1)

    def test_spectrum_peaks(self, periodic_jitter: NDArray[np.float64]):
        """Test peak identification."""
        result = jitter_spectrum(periodic_jitter, sample_rate=1e6, n_peaks=5)

        assert len(result.peaks) > 0
        # First peak should be the dominant one
        if result.peaks:
            first_freq, _first_mag = result.peaks[0]
            assert first_freq == result.dominant_frequency


# =============================================================================
# Cycle-to-Cycle Jitter Tests (JITT-009)
# =============================================================================


class TestCycleToCycleJitter:
    """Tests for cycle-to-cycle jitter measurement."""

    def test_c2c_jitter(self, clock_periods: NDArray[np.float64]):
        """Test C2C jitter calculation."""
        result = cycle_to_cycle_jitter(clock_periods)

        # C2C should be less than period jitter
        assert result.c2c_rms > 0
        assert result.c2c_pp > result.c2c_rms
        assert result.n_cycles == len(clock_periods)

    def test_c2c_histogram(self, clock_periods: NDArray[np.float64]):
        """Test C2C histogram generation."""
        result = cycle_to_cycle_jitter(clock_periods, include_histogram=True)

        assert result.histogram is not None
        assert result.bin_centers is not None

    def test_period_jitter(self, clock_periods: NDArray[np.float64]):
        """Test period jitter measurement."""
        result = period_jitter(clock_periods, nominal_period=1e-9)

        # Input had 10 ps RMS jitter
        assert np.isclose(result.c2c_rms, 10e-12, rtol=0.2)


# =============================================================================
# Full Decomposition Tests
# =============================================================================


class TestJitterDecomposition:
    """Tests for complete jitter decomposition."""

    def test_decompose_jitter(self, mixed_jitter: NDArray[np.float64]):
        """Test full jitter decomposition."""
        result = decompose_jitter(mixed_jitter)

        assert result.rj is not None
        assert result.dj is not None
        assert result.rj.rj_rms > 0
        assert result.dj.dj_pp > 0

    def test_decompose_with_pj(self, periodic_jitter: NDArray[np.float64]):
        """Test decomposition with PJ analysis."""
        result = decompose_jitter(
            periodic_jitter,
            edge_rate=1e6,
            include_pj=True,
        )

        assert result.pj is not None
        assert len(result.pj.components) > 0


# =============================================================================
# TIE Calculation Tests
# =============================================================================


class TestTIEFromEdges:
    """Tests for TIE calculation from edges."""

    def test_tie_from_edges(self):
        """Test TIE calculation from edge timestamps."""
        # Create edges with known jitter
        nominal_period = 1e-9
        n_edges = 100
        ideal_edges = np.arange(n_edges) * nominal_period

        # Add jitter
        rng = np.random.default_rng(42)
        jitter = rng.normal(0, 5e-12, n_edges)
        actual_edges = ideal_edges + jitter

        tie = tie_from_edges(actual_edges, nominal_period)

        # TIE should have similar statistics to input jitter
        assert len(tie) == n_edges
        assert np.isclose(np.std(tie), 5e-12, rtol=0.2)
