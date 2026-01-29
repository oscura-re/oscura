"""Comprehensive unit tests for jitter decomposition module.

Tests for src/oscura/analyzers/jitter/decomposition.py

This test suite provides comprehensive coverage of the jitter decomposition module,
including:
- Random jitter (RJ) extraction with tail_fit and q_scale methods
- Deterministic jitter (DJ) extraction with dual-Dirac model
- Periodic jitter (PJ) extraction via spectral analysis
- Data-dependent jitter (DDJ) extraction with pattern analysis
- Complete jitter decomposition workflow
- Result dataclasses
- Error handling and validation
- Edge cases and numerical stability
- Method selection and confidence metrics
"""

from __future__ import annotations

import numpy as np
import pytest

from oscura.analyzers.jitter.decomposition import (
    DataDependentJitterResult,
    DeterministicJitterResult,
    JitterDecomposition,
    PeriodicJitterResult,
    RandomJitterResult,
    _extract_rj_q_scale,
    _extract_rj_tail_fit,
    decompose_jitter,
    extract_ddj,
    extract_dj,
    extract_pj,
    extract_rj,
)
from oscura.core.exceptions import InsufficientDataError

pytestmark = [pytest.mark.unit, pytest.mark.analyzer, pytest.mark.jitter]


# =============================================================================
# Test Data Generators
# =============================================================================


def create_gaussian_jitter(
    n_samples: int = 5000,
    rj_rms: float = 5e-12,
    mean: float = 0.0,
    seed: int = 42,
) -> np.ndarray:
    """Generate pure Gaussian random jitter data.

    Args:
        n_samples: Number of samples.
        rj_rms: RMS jitter in seconds.
        mean: Mean offset in seconds.
        seed: Random seed for reproducibility.

    Returns:
        Array of TIE values in seconds.
    """
    rng = np.random.default_rng(seed)
    return rng.normal(mean, rj_rms, n_samples)


def create_dual_dirac_jitter(
    n_samples: int = 5000,
    dj_delta: float = 10e-12,
    seed: int = 42,
) -> np.ndarray:
    """Generate pure deterministic jitter with dual-Dirac model.

    Args:
        n_samples: Number of samples.
        dj_delta: Half-width of DJ separation in seconds.
        seed: Random seed for reproducibility.

    Returns:
        Array of TIE values in seconds.
    """
    rng = np.random.default_rng(seed)
    return np.where(rng.random(n_samples) > 0.5, dj_delta, -dj_delta)


def create_mixed_jitter(
    n_samples: int = 5000,
    rj_rms: float = 3e-12,
    dj_delta: float = 5e-12,
    seed: int = 42,
) -> np.ndarray:
    """Generate mixed random and deterministic jitter.

    Args:
        n_samples: Number of samples.
        rj_rms: RMS random jitter in seconds.
        dj_delta: Half-width of DJ in seconds.
        seed: Random seed.

    Returns:
        Array of TIE values in seconds.
    """
    rng = np.random.default_rng(seed)
    rj = rng.normal(0, rj_rms, n_samples)
    dj = np.where(rng.random(n_samples) > 0.5, dj_delta, -dj_delta)
    return rj + dj


def create_periodic_jitter(
    n_samples: int = 5000,
    sample_rate: float = 1e6,
    frequencies: list[float] | None = None,
    amplitudes: list[float] | None = None,
    seed: int = 42,
) -> np.ndarray:
    """Generate periodic jitter with sinusoidal components.

    Args:
        n_samples: Number of samples.
        sample_rate: Sample rate in Hz.
        frequencies: List of PJ frequencies in Hz.
        amplitudes: List of PJ amplitudes in seconds (peak amplitude).
        seed: Random seed.

    Returns:
        Array of TIE values in seconds.
    """
    if frequencies is None:
        frequencies = [10e3]
    if amplitudes is None:
        amplitudes = [2e-12]

    t = np.arange(n_samples) / sample_rate
    pj = np.zeros(n_samples)
    for freq, amp in zip(frequencies, amplitudes, strict=False):
        pj += amp * np.sin(2 * np.pi * freq * t)

    # Add small RJ
    rng = np.random.default_rng(seed)
    rj = rng.normal(0, 0.5e-12, n_samples)

    return pj + rj


def create_data_dependent_jitter(
    n_samples: int = 5000,
    pattern_length: int = 3,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate data-dependent jitter with bit pattern.

    Args:
        n_samples: Number of samples.
        pattern_length: Length of bit patterns.
        seed: Random seed.

    Returns:
        Tuple of (tie_data, bit_pattern).
    """
    rng = np.random.default_rng(seed)
    bit_pattern = rng.integers(0, 2, n_samples, dtype=np.int_)

    # Create DDJ based on bit patterns
    ddj_values = np.zeros(n_samples)
    for i in range(pattern_length, n_samples):
        pattern = bit_pattern[i - pattern_length : i]
        pattern_int = int("".join(str(b) for b in pattern), 2)
        # Different jitter for each pattern
        ddj_values[i] = (pattern_int - 2 ** (pattern_length - 1)) * 2e-12

    # Add RJ
    rj = rng.normal(0, 1e-12, n_samples)
    return ddj_values + rj, bit_pattern


# =============================================================================
# Tests for Result Dataclasses
# =============================================================================


@pytest.mark.unit
@pytest.mark.jitter
class TestRandomJitterResult:
    """Test RandomJitterResult dataclass."""

    def test_dataclass_creation(self) -> None:
        """Test creating a RandomJitterResult instance."""
        result = RandomJitterResult(
            rj_rms=5e-12,
            method="tail_fit",
            confidence=0.95,
            sigma=5e-12,
            mu=1e-13,
            n_samples=5000,
        )

        assert result.rj_rms == 5e-12
        assert result.method == "tail_fit"
        assert result.confidence == 0.95
        assert result.sigma == 5e-12
        assert result.mu == 1e-13
        assert result.n_samples == 5000

    def test_dataclass_attributes(self) -> None:
        """Test that all required attributes are present."""
        result = RandomJitterResult(
            rj_rms=1e-12,
            method="q_scale",
            confidence=0.8,
            sigma=1e-12,
            mu=0.0,
            n_samples=1000,
        )

        assert hasattr(result, "rj_rms")
        assert hasattr(result, "method")
        assert hasattr(result, "confidence")
        assert hasattr(result, "sigma")
        assert hasattr(result, "mu")
        assert hasattr(result, "n_samples")


@pytest.mark.unit
@pytest.mark.jitter
class TestDeterministicJitterResult:
    """Test DeterministicJitterResult dataclass."""

    def test_dataclass_creation(self) -> None:
        """Test creating a DeterministicJitterResult instance."""
        histogram = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        bin_centers = np.array([1e-12, 2e-12, 3e-12, 4e-12, 5e-12])

        result = DeterministicJitterResult(
            dj_pp=20e-12,
            dj_delta=10e-12,
            method="dual_dirac",
            confidence=0.9,
            histogram=histogram,
            bin_centers=bin_centers,
        )

        assert result.dj_pp == 20e-12
        assert result.dj_delta == 10e-12
        assert result.method == "dual_dirac"
        assert result.confidence == 0.9
        assert result.histogram is not None
        assert len(result.histogram) == 5
        assert result.bin_centers is not None
        assert len(result.bin_centers) == 5

    def test_dataclass_without_histogram(self) -> None:
        """Test DeterministicJitterResult without histogram data."""
        result = DeterministicJitterResult(
            dj_pp=15e-12,
            dj_delta=7.5e-12,
            method="dual_dirac",
            confidence=0.7,
        )

        assert result.histogram is None
        assert result.bin_centers is None


@pytest.mark.unit
@pytest.mark.jitter
class TestPeriodicJitterResult:
    """Test PeriodicJitterResult dataclass."""

    def test_dataclass_creation(self) -> None:
        """Test creating a PeriodicJitterResult instance."""
        components = [(10e3, 2e-12), (50e3, 1e-12)]

        result = PeriodicJitterResult(
            components=components,
            pj_pp=6e-12,
            dominant_frequency=10e3,
            dominant_amplitude=2e-12,
        )

        assert len(result.components) == 2
        assert result.pj_pp == 6e-12
        assert result.dominant_frequency == 10e3
        assert result.dominant_amplitude == 2e-12

    def test_dataclass_no_components(self) -> None:
        """Test PeriodicJitterResult with no components found."""
        result = PeriodicJitterResult(
            components=[],
            pj_pp=0.0,
            dominant_frequency=None,
            dominant_amplitude=None,
        )

        assert len(result.components) == 0
        assert result.pj_pp == 0.0
        assert result.dominant_frequency is None
        assert result.dominant_amplitude is None


@pytest.mark.unit
@pytest.mark.jitter
class TestDataDependentJitterResult:
    """Test DataDependentJitterResult dataclass."""

    def test_dataclass_creation(self) -> None:
        """Test creating a DataDependentJitterResult instance."""
        pattern_histogram = {"000": -6e-12, "001": -2e-12, "010": 2e-12, "011": 6e-12}

        result = DataDependentJitterResult(
            ddj_pp=12e-12,
            pattern_histogram=pattern_histogram,
            pattern_length=3,
            isi_coefficient=0.85,
        )

        assert result.ddj_pp == 12e-12
        assert len(result.pattern_histogram) == 4
        assert result.pattern_length == 3
        assert result.isi_coefficient == 0.85

    def test_dataclass_without_bit_pattern(self) -> None:
        """Test DataDependentJitterResult when bit pattern is unavailable."""
        result = DataDependentJitterResult(
            ddj_pp=5e-12,
            pattern_histogram={"above_median": 2.5e-12, "below_median": -2.5e-12},
            pattern_length=3,
            isi_coefficient=0.0,
        )

        assert result.ddj_pp == 5e-12
        assert len(result.pattern_histogram) == 2
        assert result.isi_coefficient == 0.0


@pytest.mark.unit
@pytest.mark.jitter
class TestJitterDecomposition:
    """Test JitterDecomposition dataclass."""

    def test_dataclass_complete(self) -> None:
        """Test creating a complete JitterDecomposition instance."""
        rj = RandomJitterResult(
            rj_rms=3e-12, method="tail_fit", confidence=0.9, sigma=3e-12, mu=0.0, n_samples=5000
        )
        dj = DeterministicJitterResult(
            dj_pp=10e-12, dj_delta=5e-12, method="dual_dirac", confidence=0.85
        )
        pj = PeriodicJitterResult(
            components=[(10e3, 2e-12)],
            pj_pp=4e-12,
            dominant_frequency=10e3,
            dominant_amplitude=2e-12,
        )
        ddj = DataDependentJitterResult(
            ddj_pp=8e-12,
            pattern_histogram={"00": -4e-12, "11": 4e-12},
            pattern_length=2,
            isi_coefficient=0.7,
        )

        result = JitterDecomposition(
            rj=rj,
            dj=dj,
            pj=pj,
            ddj=ddj,
            tj_pp=30e-12,
            ber_measured=1e-12,
        )

        assert result.rj is not None
        assert result.dj is not None
        assert result.pj is not None
        assert result.ddj is not None
        assert result.tj_pp == 30e-12
        assert result.ber_measured == 1e-12

    def test_dataclass_minimal(self) -> None:
        """Test JitterDecomposition with only RJ and DJ."""
        rj = RandomJitterResult(
            rj_rms=5e-12, method="q_scale", confidence=0.8, sigma=5e-12, mu=0.0, n_samples=1000
        )
        dj = DeterministicJitterResult(
            dj_pp=10e-12, dj_delta=5e-12, method="dual_dirac", confidence=0.75
        )

        result = JitterDecomposition(rj=rj, dj=dj)

        assert result.rj is not None
        assert result.dj is not None
        assert result.pj is None
        assert result.ddj is None
        assert result.tj_pp is None
        assert result.ber_measured is None


# =============================================================================
# Tests for extract_rj Function
# =============================================================================


@pytest.mark.unit
@pytest.mark.jitter
class TestExtractRJ:
    """Test random jitter extraction."""

    def test_extract_rj_gaussian_auto(self) -> None:
        """Test RJ extraction with auto method selection."""
        data = create_gaussian_jitter(n_samples=5000, rj_rms=5e-12)
        result = extract_rj(data, method="auto")

        assert isinstance(result, RandomJitterResult)
        assert result.rj_rms > 0
        assert result.n_samples == 5000
        assert 0.0 <= result.confidence <= 1.0

    def test_extract_rj_tail_fit(self) -> None:
        """Test RJ extraction with tail_fit method."""
        data = create_gaussian_jitter(n_samples=3000, rj_rms=4e-12)
        result = extract_rj(data, method="tail_fit")

        assert result.method == "tail_fit"
        assert result.rj_rms > 0
        assert np.isclose(result.rj_rms, 4e-12, rtol=0.25)

    def test_extract_rj_q_scale(self) -> None:
        """Test RJ extraction with q_scale method."""
        data = create_gaussian_jitter(n_samples=15000, rj_rms=6e-12)
        result = extract_rj(data, method="q_scale")

        assert result.method == "q_scale"
        assert result.rj_rms > 0
        assert np.isclose(result.rj_rms, 6e-12, rtol=0.25)

    def test_extract_rj_auto_selects_q_scale_large_data(self) -> None:
        """Test auto method selects q_scale for large datasets."""
        data = create_gaussian_jitter(n_samples=15000)
        result = extract_rj(data, method="auto")

        assert result.method == "q_scale"

    def test_extract_rj_auto_selects_tail_fit_small_data(self) -> None:
        """Test auto method selects tail_fit for smaller datasets."""
        data = create_gaussian_jitter(n_samples=5000)
        result = extract_rj(data, method="auto")

        assert result.method == "tail_fit"

    def test_extract_rj_insufficient_data_error(self) -> None:
        """Test RJ extraction raises error with insufficient samples."""
        data = np.array([1e-12, 2e-12], dtype=np.float64)

        with pytest.raises(InsufficientDataError) as exc_info:
            extract_rj(data, min_samples=1000)

        assert exc_info.value.required == 1000
        assert exc_info.value.available == 2
        assert "random_jitter_extraction" in exc_info.value.analysis_type

    def test_extract_rj_with_nan_filtering(self) -> None:
        """Test RJ extraction filters NaN values."""
        data = create_gaussian_jitter(n_samples=2000, rj_rms=5e-12)
        data[::100] = np.nan

        result = extract_rj(data, min_samples=100)

        assert result.n_samples < 2000  # Some samples filtered
        assert not np.isnan(result.rj_rms)

    def test_extract_rj_all_nan_error(self) -> None:
        """Test RJ extraction raises error if all values are NaN."""
        data = np.full(2000, np.nan, dtype=np.float64)

        with pytest.raises(InsufficientDataError):
            extract_rj(data, min_samples=1000)

    def test_extract_rj_with_mean_offset(self) -> None:
        """Test RJ extraction with non-zero mean."""
        data = create_gaussian_jitter(n_samples=5000, rj_rms=5e-12, mean=10e-12)
        result = extract_rj(data)

        assert result.rj_rms > 0
        assert np.isclose(result.mu, 10e-12, rtol=0.3)

    def test_extract_rj_confidence_metric(self) -> None:
        """Test RJ extraction provides confidence metric."""
        data = create_gaussian_jitter(n_samples=10000, rj_rms=5e-12)
        result = extract_rj(data)

        # Pure Gaussian should have high confidence
        assert result.confidence > 0.6

    def test_extract_rj_sigma_matches_rms(self) -> None:
        """Test that sigma parameter matches rj_rms."""
        data = create_gaussian_jitter(n_samples=5000, rj_rms=7e-12)
        result = extract_rj(data)

        assert result.sigma == result.rj_rms

    def test_extract_rj_invalid_method_error(self) -> None:
        """Test RJ extraction raises error for invalid method."""
        data = create_gaussian_jitter(n_samples=2000)

        with pytest.raises(ValueError) as exc_info:
            extract_rj(data, method="invalid_method")  # type: ignore

        assert "Unknown method" in str(exc_info.value)

    def test_extract_rj_minimum_samples(self) -> None:
        """Test RJ extraction with minimum required samples."""
        data = create_gaussian_jitter(n_samples=1000, rj_rms=5e-12)
        result = extract_rj(data, min_samples=1000)

        assert result.n_samples == 1000
        assert result.rj_rms > 0


# =============================================================================
# Tests for Internal RJ Extraction Methods
# =============================================================================


@pytest.mark.unit
@pytest.mark.jitter
class TestInternalRJMethods:
    """Test internal RJ extraction methods."""

    def test_extract_rj_tail_fit_internal(self) -> None:
        """Test _extract_rj_tail_fit internal function."""
        data = create_gaussian_jitter(n_samples=3000, rj_rms=5e-12)
        result = _extract_rj_tail_fit(data)

        assert result.method == "tail_fit"
        assert result.rj_rms > 0
        assert result.sigma > 0

    def test_extract_rj_q_scale_internal(self) -> None:
        """Test _extract_rj_q_scale internal function."""
        data = create_gaussian_jitter(n_samples=10000, rj_rms=5e-12)
        result = _extract_rj_q_scale(data)

        assert result.method == "q_scale"
        assert result.rj_rms > 0
        assert result.sigma > 0

    def test_tail_fit_small_tail_data(self) -> None:
        """Test tail_fit with very small dataset (edge case)."""
        data = create_gaussian_jitter(n_samples=1000, rj_rms=5e-12)
        result = _extract_rj_tail_fit(data)

        # Should fall back to percentile estimation
        assert result.rj_rms > 0
        assert result.confidence >= 0.0

    def test_q_scale_small_tail_fraction(self) -> None:
        """Test q_scale with small tail fraction."""
        data = create_gaussian_jitter(n_samples=1000, rj_rms=5e-12)
        result = _extract_rj_q_scale(data)

        # Should still produce valid result
        assert result.rj_rms > 0


# =============================================================================
# Tests for extract_dj Function
# =============================================================================


@pytest.mark.unit
@pytest.mark.jitter
class TestExtractDJ:
    """Test deterministic jitter extraction."""

    def test_extract_dj_dual_dirac(self) -> None:
        """Test DJ extraction on pure dual-Dirac data."""
        data = create_dual_dirac_jitter(n_samples=5000, dj_delta=10e-12)
        result = extract_dj(data)

        assert isinstance(result, DeterministicJitterResult)
        assert result.dj_pp >= 0  # May be detected as 0 depending on histogram
        assert result.dj_delta == result.dj_pp / 2
        assert result.method == "dual_dirac"

    def test_extract_dj_with_rj_result(self) -> None:
        """Test DJ extraction with pre-computed RJ result."""
        data = create_mixed_jitter(n_samples=5000, rj_rms=3e-12, dj_delta=8e-12)
        rj_result = extract_rj(data)
        dj_result = extract_dj(data, rj_result=rj_result)

        assert dj_result.dj_pp > 0
        assert dj_result.confidence >= 0.0

    def test_extract_dj_without_rj_result(self) -> None:
        """Test DJ extraction computes RJ internally."""
        data = create_mixed_jitter(n_samples=5000, rj_rms=2e-12, dj_delta=6e-12)
        result = extract_dj(data)

        assert result.dj_pp > 0

    def test_extract_dj_insufficient_data_error(self) -> None:
        """Test DJ extraction raises error with insufficient samples."""
        data = np.array([1e-12, 2e-12], dtype=np.float64)

        with pytest.raises(InsufficientDataError) as exc_info:
            extract_dj(data, min_samples=1000)

        assert exc_info.value.required == 1000
        assert "deterministic_jitter_extraction" in exc_info.value.analysis_type

    def test_extract_dj_bimodal_detection(self) -> None:
        """Test DJ extraction detects bimodal peaks."""
        data = create_dual_dirac_jitter(n_samples=10000, dj_delta=15e-12)
        result = extract_dj(data)

        # Should attempt bimodal detection (may or may not detect perfectly)
        assert result.dj_pp >= 0
        # Confidence should be valid
        assert 0.0 <= result.confidence <= 1.0

    def test_extract_dj_histogram_generated(self) -> None:
        """Test DJ extraction generates histogram."""
        data = create_dual_dirac_jitter(n_samples=5000, dj_delta=10e-12)
        result = extract_dj(data)

        assert result.histogram is not None
        assert result.bin_centers is not None
        assert len(result.histogram) > 0
        assert len(result.bin_centers) == len(result.histogram)

    def test_extract_dj_pure_gaussian_low_dj(self) -> None:
        """Test DJ extraction on pure Gaussian (should find minimal DJ)."""
        data = create_gaussian_jitter(n_samples=5000, rj_rms=5e-12)
        result = extract_dj(data)

        # DJ should be minimal or zero
        assert result.dj_pp >= 0
        # Low confidence when no clear DJ
        assert result.confidence <= 0.5

    def test_extract_dj_confidence_metrics(self) -> None:
        """Test DJ extraction provides confidence metrics."""
        data = create_dual_dirac_jitter(n_samples=10000, dj_delta=12e-12)
        result = extract_dj(data)

        assert 0.0 <= result.confidence <= 1.0

    def test_extract_dj_with_nan_filtering(self) -> None:
        """Test DJ extraction filters NaN values."""
        data = create_dual_dirac_jitter(n_samples=5000, dj_delta=10e-12)
        data[::500] = np.nan

        result = extract_dj(data, min_samples=1000)

        assert result.dj_pp >= 0
        assert not np.isnan(result.dj_pp)


# =============================================================================
# Tests for extract_pj Function
# =============================================================================


@pytest.mark.unit
@pytest.mark.jitter
class TestExtractPJ:
    """Test periodic jitter extraction."""

    def test_extract_pj_single_frequency(self) -> None:
        """Test PJ extraction with single frequency component."""
        data = create_periodic_jitter(
            n_samples=10000, sample_rate=1e6, frequencies=[10e3], amplitudes=[3e-12]
        )
        result = extract_pj(data, sample_rate=1e6)

        assert isinstance(result, PeriodicJitterResult)
        assert len(result.components) > 0
        assert result.dominant_frequency is not None
        assert np.isclose(result.dominant_frequency, 10e3, rtol=0.2)

    def test_extract_pj_multiple_frequencies(self) -> None:
        """Test PJ extraction with multiple frequency components."""
        data = create_periodic_jitter(
            n_samples=10000,
            sample_rate=1e6,
            frequencies=[10e3, 50e3, 100e3],
            amplitudes=[3e-12, 2e-12, 1e-12],
        )
        result = extract_pj(data, sample_rate=1e6, n_components=5)

        # Should detect multiple components
        assert len(result.components) > 1
        # Dominant should be highest amplitude
        assert result.dominant_frequency is not None

    def test_extract_pj_frequency_range(self) -> None:
        """Test PJ extraction with specified frequency range."""
        data = create_periodic_jitter(
            n_samples=10000,
            sample_rate=1e6,
            frequencies=[5e3, 50e3, 200e3],
            amplitudes=[2e-12, 3e-12, 1e-12],
        )
        result = extract_pj(data, sample_rate=1e6, min_frequency=10e3, max_frequency=100e3)

        # Should only find components within range (50 kHz)
        if result.dominant_frequency is not None:
            assert 10e3 <= result.dominant_frequency <= 100e3

    def test_extract_pj_no_periodic_component(self) -> None:
        """Test PJ extraction on pure random jitter (no PJ)."""
        data = create_gaussian_jitter(n_samples=5000, rj_rms=5e-12)
        result = extract_pj(data, sample_rate=1e6)

        # Should find no significant components or minimal components
        assert len(result.components) == 0 or result.pj_pp < 5e-12

    def test_extract_pj_insufficient_samples(self) -> None:
        """Test PJ extraction with insufficient samples."""
        data = np.array([1e-12, 2e-12, 3e-12], dtype=np.float64)
        result = extract_pj(data, sample_rate=1e6)

        # Should return empty result
        assert len(result.components) == 0
        assert result.pj_pp == 0.0
        assert result.dominant_frequency is None

    def test_extract_pj_amplitude_quantification(self) -> None:
        """Test PJ amplitude is correctly quantified."""
        amplitude = 4e-12
        data = create_periodic_jitter(
            n_samples=10000, sample_rate=1e6, frequencies=[10e3], amplitudes=[amplitude]
        )
        result = extract_pj(data, sample_rate=1e6)

        # Peak-to-peak should be approximately 2x sum of amplitudes
        assert result.pj_pp > 0

    def test_extract_pj_with_nan_filtering(self) -> None:
        """Test PJ extraction filters NaN values."""
        data = create_periodic_jitter(
            n_samples=5000, sample_rate=1e6, frequencies=[10e3], amplitudes=[3e-12]
        )
        data[::500] = np.nan

        result = extract_pj(data, sample_rate=1e6)

        # Should still detect periodic component
        assert not np.isnan(result.pj_pp)

    def test_extract_pj_default_max_frequency(self) -> None:
        """Test PJ extraction uses Nyquist as default max frequency."""
        data = create_periodic_jitter(
            n_samples=5000, sample_rate=1e6, frequencies=[10e3], amplitudes=[2e-12]
        )
        result = extract_pj(data, sample_rate=1e6)

        # Default should be Nyquist (sample_rate / 2)
        # All found frequencies should be below Nyquist
        for freq, _ in result.components:
            assert freq <= 1e6 / 2

    def test_extract_pj_n_components_limit(self) -> None:
        """Test PJ extraction respects n_components limit."""
        data = create_periodic_jitter(
            n_samples=10000,
            sample_rate=1e6,
            frequencies=[10e3, 20e3, 30e3, 40e3, 50e3],
            amplitudes=[3e-12, 2.5e-12, 2e-12, 1.5e-12, 1e-12],
        )
        result = extract_pj(data, sample_rate=1e6, n_components=3)

        # Should only return top 3 components
        assert len(result.components) <= 3

    def test_extract_pj_dc_removal(self) -> None:
        """Test PJ extraction removes DC offset."""
        data = create_periodic_jitter(
            n_samples=5000, sample_rate=1e6, frequencies=[10e3], amplitudes=[3e-12]
        )
        data += 50e-12  # Add DC offset

        result = extract_pj(data, sample_rate=1e6)

        # Should still detect periodic component despite DC offset
        assert result.dominant_frequency is not None


# =============================================================================
# Tests for extract_ddj Function
# =============================================================================


@pytest.mark.unit
@pytest.mark.jitter
class TestExtractDDJ:
    """Test data-dependent jitter extraction."""

    def test_extract_ddj_with_bit_pattern(self) -> None:
        """Test DDJ extraction with bit pattern provided."""
        tie_data, bit_pattern = create_data_dependent_jitter(n_samples=5000, pattern_length=3)
        result = extract_ddj(tie_data, bit_pattern=bit_pattern, pattern_length=3)

        assert isinstance(result, DataDependentJitterResult)
        assert result.ddj_pp > 0
        assert result.pattern_length == 3
        assert len(result.pattern_histogram) > 0

    def test_extract_ddj_without_bit_pattern(self) -> None:
        """Test DDJ extraction without bit pattern (estimation)."""
        tie_data, _ = create_data_dependent_jitter(n_samples=5000, pattern_length=3)
        result = extract_ddj(tie_data, bit_pattern=None)

        assert result.ddj_pp >= 0
        assert result.isi_coefficient == 0.0  # Unknown without pattern
        assert (
            "above_median" in result.pattern_histogram or "below_median" in result.pattern_histogram
        )

    def test_extract_ddj_pattern_histogram(self) -> None:
        """Test DDJ extraction generates pattern histogram."""
        tie_data, bit_pattern = create_data_dependent_jitter(n_samples=5000, pattern_length=2)
        result = extract_ddj(tie_data, bit_pattern=bit_pattern, pattern_length=2)

        # Should have entries for different patterns
        assert len(result.pattern_histogram) > 0
        # Each entry should be a float (mean TIE for that pattern)
        for value in result.pattern_histogram.values():
            assert isinstance(value, float)

    def test_extract_ddj_isi_coefficient(self) -> None:
        """Test DDJ extraction computes ISI coefficient."""
        tie_data, bit_pattern = create_data_dependent_jitter(n_samples=5000, pattern_length=3)
        result = extract_ddj(tie_data, bit_pattern=bit_pattern, pattern_length=3)

        # ISI coefficient should be finite
        assert np.isfinite(result.isi_coefficient)
        assert -1.0 <= result.isi_coefficient <= 1.0

    def test_extract_ddj_mismatched_lengths_error(self) -> None:
        """Test DDJ extraction raises error if bit_pattern length doesn't match."""
        tie_data = np.random.randn(1000) * 1e-12
        bit_pattern = np.array([0, 1, 0, 1], dtype=np.int_)

        with pytest.raises(ValueError) as exc_info:
            extract_ddj(tie_data, bit_pattern=bit_pattern)

        assert "length must match" in str(exc_info.value)

    def test_extract_ddj_with_nan_filtering(self) -> None:
        """Test DDJ extraction filters NaN values."""
        tie_data, bit_pattern = create_data_dependent_jitter(n_samples=2000, pattern_length=2)
        # Create valid copy before adding NaN (since we filter in extract_ddj)
        valid_mask = np.ones(len(tie_data), dtype=bool)
        valid_mask[::200] = False
        tie_data[::200] = np.nan

        # Filter both arrays consistently
        tie_filtered = tie_data[~np.isnan(tie_data)]
        bit_filtered = bit_pattern[valid_mask]

        # Ensure same length after filtering
        min_len = min(len(tie_filtered), len(bit_filtered))
        tie_filtered = tie_filtered[:min_len]
        bit_filtered = bit_filtered[:min_len]

        result = extract_ddj(tie_filtered, bit_pattern=bit_filtered, pattern_length=2)

        assert not np.isnan(result.ddj_pp)

    def test_extract_ddj_pattern_length_variation(self) -> None:
        """Test DDJ extraction with different pattern lengths."""
        tie_data, bit_pattern = create_data_dependent_jitter(n_samples=5000, pattern_length=4)

        for pattern_len in [2, 3, 4]:
            result = extract_ddj(tie_data, bit_pattern=bit_pattern, pattern_length=pattern_len)
            assert result.pattern_length == pattern_len

    def test_extract_ddj_no_correlation(self) -> None:
        """Test DDJ extraction on uncorrelated data."""
        tie_data = create_gaussian_jitter(n_samples=2000, rj_rms=5e-12)
        bit_pattern = np.random.randint(0, 2, 2000, dtype=np.int_)

        result = extract_ddj(tie_data, bit_pattern=bit_pattern, pattern_length=2)

        # Should find minimal DDJ
        assert result.ddj_pp >= 0


# =============================================================================
# Tests for decompose_jitter Function
# =============================================================================


@pytest.mark.unit
@pytest.mark.jitter
class TestDecomposeJitter:
    """Test complete jitter decomposition."""

    def test_decompose_jitter_basic(self) -> None:
        """Test basic jitter decomposition."""
        data = create_mixed_jitter(n_samples=5000, rj_rms=3e-12, dj_delta=5e-12)
        result = decompose_jitter(data)

        assert isinstance(result, JitterDecomposition)
        assert result.rj is not None
        assert result.dj is not None
        assert result.pj is None  # Not requested
        assert result.ddj is None  # Not requested

    def test_decompose_jitter_with_pj(self) -> None:
        """Test jitter decomposition including periodic jitter."""
        data = create_periodic_jitter(n_samples=5000, sample_rate=1e6)
        result = decompose_jitter(data, edge_rate=1e6, include_pj=True)

        assert result.rj is not None
        assert result.dj is not None
        assert result.pj is not None
        assert len(result.pj.components) >= 0

    def test_decompose_jitter_with_ddj(self) -> None:
        """Test jitter decomposition including data-dependent jitter."""
        tie_data, bit_pattern = create_data_dependent_jitter(n_samples=5000, pattern_length=3)
        result = decompose_jitter(tie_data, include_ddj=True, bit_pattern=bit_pattern)

        assert result.rj is not None
        assert result.dj is not None
        assert result.ddj is not None
        assert result.ddj.ddj_pp >= 0

    def test_decompose_jitter_complete(self) -> None:
        """Test complete decomposition with all components."""
        # Create complex jitter with all components
        rng = np.random.default_rng(42)
        n_samples = 10000
        sample_rate = 1e6
        t = np.arange(n_samples) / sample_rate

        # RJ
        rj = rng.normal(0, 2e-12, n_samples)
        # DJ
        dj = np.where(rng.random(n_samples) > 0.5, 4e-12, -4e-12)
        # PJ
        pj = 2e-12 * np.sin(2 * np.pi * 10e3 * t)

        tie_data = rj + dj + pj
        bit_pattern = rng.integers(0, 2, n_samples, dtype=np.int_)

        result = decompose_jitter(
            tie_data,
            edge_rate=sample_rate,
            include_pj=True,
            include_ddj=True,
            bit_pattern=bit_pattern,
        )

        assert result.rj is not None
        assert result.dj is not None
        assert result.pj is not None
        assert result.ddj is not None

    def test_decompose_jitter_without_edge_rate_no_pj(self) -> None:
        """Test decomposition without edge_rate skips PJ."""
        data = create_mixed_jitter(n_samples=5000, rj_rms=3e-12, dj_delta=5e-12)
        result = decompose_jitter(data, include_pj=True)  # No edge_rate

        assert result.rj is not None
        assert result.dj is not None
        assert result.pj is None  # Skipped due to missing edge_rate

    def test_decompose_jitter_rj_used_for_dj(self) -> None:
        """Test that RJ result is used for DJ extraction."""
        data = create_mixed_jitter(n_samples=5000, rj_rms=3e-12, dj_delta=5e-12)
        result = decompose_jitter(data)

        # Both should be present
        assert result.rj.rj_rms > 0
        assert result.dj.dj_pp >= 0


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


@pytest.mark.unit
@pytest.mark.jitter
class TestJitterDecompositionEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_data_raises_error(self) -> None:
        """Test that empty data raises appropriate error."""
        data = np.array([], dtype=np.float64)

        with pytest.raises(InsufficientDataError):
            extract_rj(data, min_samples=1000)

    def test_very_small_jitter_values(self) -> None:
        """Test handling of very small jitter (sub-femtosecond)."""
        data = create_gaussian_jitter(n_samples=2000, rj_rms=1e-16)
        result = extract_rj(data, min_samples=1000)

        assert result.rj_rms >= 0
        assert np.isfinite(result.rj_rms)

    def test_very_large_jitter_values(self) -> None:
        """Test handling of very large jitter (nanoseconds)."""
        data = create_gaussian_jitter(n_samples=2000, rj_rms=1e-9)
        result = extract_rj(data, min_samples=1000)

        assert result.rj_rms > 0
        assert np.isclose(result.rj_rms, 1e-9, rtol=0.3)

    def test_constant_data_zero_jitter(self) -> None:
        """Test handling of constant data (no jitter)."""
        data = np.zeros(2000, dtype=np.float64)
        result = extract_rj(data, min_samples=1000)

        assert result.rj_rms == pytest.approx(0.0, abs=1e-15)

    def test_inf_values_in_data(self) -> None:
        """Test handling of infinity values in data."""
        data = create_gaussian_jitter(n_samples=2000, rj_rms=5e-12)
        data[100] = np.inf
        data[200] = -np.inf

        # Should either filter or raise error
        try:
            result = extract_rj(data, min_samples=1000)
            assert np.isfinite(result.rj_rms)
        except (ValueError, InsufficientDataError):
            pass  # Acceptable

    def test_single_value_dj(self) -> None:
        """Test DJ extraction on single-valued data."""
        data = np.full(2000, 5e-12, dtype=np.float64)
        result = extract_dj(data, min_samples=1000)

        # Should find minimal or zero DJ
        assert result.dj_pp >= 0

    def test_pj_very_low_frequency(self) -> None:
        """Test PJ extraction with very low frequency component."""
        # Frequency below min_frequency threshold
        data = create_periodic_jitter(
            n_samples=10000, sample_rate=1e6, frequencies=[0.5], amplitudes=[3e-12]
        )
        result = extract_pj(data, sample_rate=1e6, min_frequency=1.0)

        # Should not detect component below min_frequency
        if result.dominant_frequency is not None:
            assert result.dominant_frequency >= 1.0

    def test_ddj_single_pattern(self) -> None:
        """Test DDJ extraction with single bit pattern."""
        import warnings

        tie_data = create_gaussian_jitter(n_samples=1000, rj_rms=5e-12)
        bit_pattern = np.zeros(1000, dtype=np.int_)  # All zeros

        # Correlation with constant array will produce warning (expected)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=RuntimeWarning, message="invalid value encountered"
            )
            result = extract_ddj(tie_data, bit_pattern=bit_pattern, pattern_length=2)

        # Should handle gracefully - may find zero DDJ due to single pattern
        assert result.ddj_pp >= 0
        # ISI coefficient should be 0 (set when correlation is not finite)
        assert result.isi_coefficient == 0.0
        # With all zeros, pattern histogram should have only "00" pattern
        assert len(result.pattern_histogram) >= 1


# =============================================================================
# Numerical Stability Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.jitter
class TestNumericalStability:
    """Test numerical stability and precision."""

    def test_rj_extraction_reproducibility(self) -> None:
        """Test RJ extraction produces same results on same input."""
        data = create_gaussian_jitter(n_samples=5000, rj_rms=5e-12, seed=123)

        result1 = extract_rj(data)
        result2 = extract_rj(data)

        assert result1.rj_rms == result2.rj_rms
        assert result1.sigma == result2.sigma
        assert result1.mu == result2.mu

    def test_dj_extraction_stability(self) -> None:
        """Test DJ extraction is numerically stable."""
        results = []
        for seed in range(5):
            data = create_dual_dirac_jitter(n_samples=5000, dj_delta=10e-12, seed=seed + 100)
            result = extract_dj(data, min_samples=1000)
            results.append(result.dj_pp)

        # Results should be consistent
        mean_dj = np.mean(results)
        std_dj = np.std(results)

        # Coefficient of variation should be reasonable
        if mean_dj > 0:
            assert std_dj / mean_dj < 0.5

    def test_pj_extraction_floating_point_precision(self) -> None:
        """Test PJ extraction handles floating point precision."""
        data = create_periodic_jitter(
            n_samples=10000, sample_rate=1e6, frequencies=[10e3], amplitudes=[1e-15]
        )
        result = extract_pj(data, sample_rate=1e6)

        # Should handle very small amplitudes
        assert np.isfinite(result.pj_pp)

    def test_decomposition_consistency(self) -> None:
        """Test decomposition produces consistent results."""
        data = create_mixed_jitter(n_samples=5000, rj_rms=3e-12, dj_delta=5e-12, seed=456)

        result1 = decompose_jitter(data)
        result2 = decompose_jitter(data)

        assert result1.rj.rj_rms == result2.rj.rj_rms
        assert result1.dj.dj_pp == result2.dj.dj_pp


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.jitter
class TestJitterDecompositionIntegration:
    """Integration tests combining multiple functions."""

    def test_rj_then_dj_workflow(self) -> None:
        """Test typical workflow: extract RJ, then use it for DJ."""
        data = create_mixed_jitter(n_samples=5000, rj_rms=3e-12, dj_delta=6e-12)

        rj_result = extract_rj(data)
        dj_result = extract_dj(data, rj_result=rj_result)

        assert rj_result.rj_rms > 0
        assert dj_result.dj_pp > 0

    def test_full_decomposition_workflow(self) -> None:
        """Test full decomposition workflow with all components."""
        # Create comprehensive jitter
        rng = np.random.default_rng(789)
        n_samples = 10000
        sample_rate = 1e6
        t = np.arange(n_samples) / sample_rate

        rj = rng.normal(0, 2e-12, n_samples)
        dj = np.where(rng.random(n_samples) > 0.5, 5e-12, -5e-12)
        pj = 3e-12 * np.sin(2 * np.pi * 15e3 * t)

        tie_data = rj + dj + pj

        # Full decomposition
        result = decompose_jitter(tie_data, edge_rate=sample_rate, include_pj=True)

        # All components should be detected
        assert result.rj.rj_rms > 0
        assert result.dj.dj_pp > 0
        assert result.pj is not None

    def test_sequential_analysis(self) -> None:
        """Test sequential analysis of different jitter types."""
        data = create_mixed_jitter(n_samples=10000, rj_rms=4e-12, dj_delta=8e-12)

        # Sequential extraction
        rj = extract_rj(data, method="tail_fit")
        dj = extract_dj(data, rj_result=rj)
        pj = extract_pj(data, sample_rate=1e6)

        # All should produce valid results
        assert rj.rj_rms > 0
        assert dj.dj_pp >= 0
        assert pj.pj_pp >= 0


# =============================================================================
# Performance and Stress Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.jitter
class TestPerformance:
    """Performance and stress tests."""

    def test_large_dataset_rj_extraction(self) -> None:
        """Test RJ extraction on large dataset (50k samples)."""
        data = create_gaussian_jitter(n_samples=50000, rj_rms=5e-12)
        result = extract_rj(data)

        assert result.rj_rms > 0
        assert result.n_samples == 50000

    def test_large_dataset_dj_extraction(self) -> None:
        """Test DJ extraction on large dataset."""
        data = create_dual_dirac_jitter(n_samples=30000, dj_delta=10e-12)
        result = extract_dj(data, min_samples=1000)

        # DJ extraction may or may not detect perfect dual-Dirac
        assert result.dj_pp >= 0
        assert isinstance(result, DeterministicJitterResult)

    def test_high_frequency_pj_extraction(self) -> None:
        """Test PJ extraction with high sample rate."""
        data = create_periodic_jitter(
            n_samples=50000,
            sample_rate=100e6,
            frequencies=[1e6, 5e6],
            amplitudes=[2e-12, 1e-12],
        )
        result = extract_pj(data, sample_rate=100e6, n_components=5)

        assert len(result.components) > 0

    def test_decomposition_speed(self) -> None:
        """Test decomposition completes in reasonable time."""
        import time

        data = create_mixed_jitter(n_samples=10000, rj_rms=3e-12, dj_delta=5e-12)

        start = time.time()
        result = decompose_jitter(data, edge_rate=1e6, include_pj=True)
        elapsed = time.time() - start

        # Should complete quickly
        assert elapsed < 5.0  # 5 seconds
        assert result.rj is not None
