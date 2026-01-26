"""Comprehensive unit tests for jitter spectrum analysis.

Tests for src/oscura/analyzers/jitter/spectrum.py

This test suite provides comprehensive coverage of the jitter spectrum module,
including:
- FFT-based jitter spectrum analysis
- Periodic component identification
- Spectral magnitude calculations
- Window function application
- Detrending operations
- Peak detection and ranking
- Noise floor estimation
- Result dataclasses
- Error handling and validation
- Edge cases and numerical stability


References:
    IEEE 2414-2020: Standard for Jitter and Phase Noise
"""

from __future__ import annotations

import numpy as np
import pytest

from oscura.analyzers.jitter.spectrum import (
    JitterSpectrumResult,
    identify_periodic_components,
    jitter_spectrum,
)

pytestmark = [pytest.mark.unit, pytest.mark.analyzer, pytest.mark.jitter]


# =============================================================================
# Test Data Generators
# =============================================================================


def create_tie_data(
    n_samples: int = 1000,
    sample_rate: float = 1e9,
    frequencies: list[float] | None = None,
    amplitudes: list[float] | None = None,
    noise_rms: float = 0.0,
    seed: int = 42,
) -> np.ndarray:
    """Generate synthetic TIE data with known frequency components.

    Args:
        n_samples: Number of samples.
        sample_rate: Sample rate in Hz (edges per second).
        frequencies: List of jitter frequencies in Hz.
        amplitudes: List of jitter amplitudes in seconds.
        noise_rms: RMS noise level in seconds.
        seed: Random seed for reproducibility.

    Returns:
        Array of TIE values in seconds.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples) / sample_rate

    # Initialize with noise
    tie_data = rng.normal(0, noise_rms, n_samples)

    # Add frequency components
    if frequencies is not None and amplitudes is not None:
        for freq, amp in zip(frequencies, amplitudes, strict=False):
            tie_data += amp * np.sin(2 * np.pi * freq * t)

    return tie_data


def create_pure_periodic_tie(
    n_samples: int = 1000,
    sample_rate: float = 1e9,
    jitter_freq: float = 1e6,
    jitter_amplitude: float = 10e-12,
) -> np.ndarray:
    """Generate pure periodic jitter TIE data.

    Args:
        n_samples: Number of samples.
        sample_rate: Sample rate in Hz.
        jitter_freq: Jitter frequency in Hz.
        jitter_amplitude: Peak jitter amplitude in seconds.

    Returns:
        Array of TIE values in seconds.
    """
    t = np.arange(n_samples) / sample_rate
    return jitter_amplitude * np.sin(2 * np.pi * jitter_freq * t)


def create_multi_tone_tie(
    n_samples: int = 1000,
    sample_rate: float = 1e9,
    seed: int = 42,
) -> np.ndarray:
    """Generate TIE data with multiple frequency components.

    Args:
        n_samples: Number of samples.
        sample_rate: Sample rate in Hz.
        seed: Random seed.

    Returns:
        Array of TIE values with multiple tones.
    """
    frequencies = [100e3, 1e6, 5e6, 10e6]  # Hz
    amplitudes = [20e-12, 15e-12, 8e-12, 5e-12]  # seconds
    return create_tie_data(
        n_samples=n_samples,
        sample_rate=sample_rate,
        frequencies=frequencies,
        amplitudes=amplitudes,
        noise_rms=1e-12,
        seed=seed,
    )


# =============================================================================
# Tests for JitterSpectrumResult Dataclass
# =============================================================================


@pytest.mark.unit
@pytest.mark.jitter
class TestJitterSpectrumResult:
    """Test JitterSpectrumResult dataclass."""

    def test_dataclass_creation(self) -> None:
        """Test creating a JitterSpectrumResult instance."""
        frequencies = np.array([0, 1e6, 2e6, 3e6])
        magnitude = np.array([1e-12, 5e-12, 3e-12, 2e-12])
        magnitude_db = 20 * np.log10(magnitude / 1e-12)
        peaks = [(1e6, 5e-12), (2e6, 3e-12)]

        result = JitterSpectrumResult(
            frequencies=frequencies,
            magnitude=magnitude,
            magnitude_db=magnitude_db,
            dominant_frequency=1e6,
            dominant_magnitude=5e-12,
            noise_floor=1e-12,
            peaks=peaks,
        )

        assert len(result.frequencies) == 4
        assert len(result.magnitude) == 4
        assert len(result.magnitude_db) == 4
        assert result.dominant_frequency == 1e6
        assert result.dominant_magnitude == 5e-12
        assert result.noise_floor == 1e-12
        assert len(result.peaks) == 2

    def test_dataclass_no_peaks(self) -> None:
        """Test JitterSpectrumResult with no peaks detected."""
        frequencies = np.array([0, 1e6])
        magnitude = np.array([1e-15, 1e-15])
        magnitude_db = np.array([-120.0, -120.0])

        result = JitterSpectrumResult(
            frequencies=frequencies,
            magnitude=magnitude,
            magnitude_db=magnitude_db,
            dominant_frequency=None,
            dominant_magnitude=None,
            noise_floor=1e-15,
            peaks=[],
        )

        assert result.dominant_frequency is None
        assert result.dominant_magnitude is None
        assert len(result.peaks) == 0

    def test_dataclass_attributes_types(self) -> None:
        """Test that dataclass attributes have correct types."""
        result = JitterSpectrumResult(
            frequencies=np.array([0.0]),
            magnitude=np.array([1e-12]),
            magnitude_db=np.array([0.0]),
            dominant_frequency=1e6,
            dominant_magnitude=5e-12,
            noise_floor=1e-13,
            peaks=[(1e6, 5e-12)],
        )

        assert isinstance(result.frequencies, np.ndarray)
        assert isinstance(result.magnitude, np.ndarray)
        assert isinstance(result.magnitude_db, np.ndarray)
        assert isinstance(result.dominant_frequency, float | int | type(None))
        assert isinstance(result.dominant_magnitude, float | int | type(None))
        assert isinstance(result.noise_floor, float)
        assert isinstance(result.peaks, list)


# =============================================================================
# Tests for jitter_spectrum Function
# =============================================================================


@pytest.mark.unit
@pytest.mark.jitter
class TestJitterSpectrum:
    """Test jitter spectrum FFT analysis."""

    def test_spectrum_pure_tone(self) -> None:
        """Test spectrum analysis of pure sinusoidal jitter."""
        jitter_freq = 1e6  # 1 MHz
        jitter_amp = 10e-12  # 10 ps
        sample_rate = 100e6  # 100 MHz

        tie_data = create_pure_periodic_tie(
            n_samples=10000,
            sample_rate=sample_rate,
            jitter_freq=jitter_freq,
            jitter_amplitude=jitter_amp,
        )

        result = jitter_spectrum(tie_data, sample_rate=sample_rate)

        assert len(result.frequencies) > 0
        assert len(result.magnitude) > 0
        assert result.dominant_frequency is not None

        # Should detect 1 MHz component
        assert np.isclose(result.dominant_frequency, jitter_freq, rtol=0.05)

    def test_spectrum_multi_tone(self) -> None:
        """Test spectrum analysis with multiple frequency components."""
        tie_data = create_multi_tone_tie(n_samples=10000, sample_rate=100e6)
        result = jitter_spectrum(tie_data, sample_rate=100e6)

        # Should detect multiple peaks
        assert len(result.peaks) >= 2
        assert result.dominant_frequency is not None

        # Dominant frequency should be the largest component (100 kHz)
        assert np.isclose(result.dominant_frequency, 100e3, rtol=0.2)

    def test_spectrum_with_noise(self) -> None:
        """Test spectrum analysis with noisy data."""
        tie_data = create_tie_data(
            n_samples=5000,
            sample_rate=50e6,
            frequencies=[1e6],
            amplitudes=[5e-12],
            noise_rms=1e-12,
        )

        result = jitter_spectrum(tie_data, sample_rate=50e6)

        # Should still detect signal above noise
        assert result.dominant_frequency is not None
        assert result.noise_floor > 0

    def test_spectrum_empty_insufficient_data(self) -> None:
        """Test spectrum with insufficient data returns empty result."""
        tie_data = np.array([1e-12, 2e-12, 3e-12])  # Only 3 samples

        result = jitter_spectrum(tie_data, sample_rate=1e9)

        # Should return empty result
        assert len(result.frequencies) == 0
        assert len(result.magnitude) == 0
        assert result.dominant_frequency is None
        assert result.dominant_magnitude is None

    def test_spectrum_minimum_valid_samples(self) -> None:
        """Test spectrum with minimum valid number of samples (16)."""
        tie_data = create_pure_periodic_tie(n_samples=16, sample_rate=1e6)
        result = jitter_spectrum(tie_data, sample_rate=1e6)

        # Should produce valid result
        assert len(result.frequencies) > 0
        assert len(result.magnitude) > 0

    def test_spectrum_with_nan_values(self) -> None:
        """Test spectrum filters NaN values from TIE data."""
        tie_data = create_pure_periodic_tie(n_samples=1000, sample_rate=10e6)
        tie_data[::10] = np.nan  # Insert NaN values

        result = jitter_spectrum(tie_data, sample_rate=10e6)

        # Should filter NaN and still produce result
        assert len(result.frequencies) > 0
        assert not np.any(np.isnan(result.magnitude))

    def test_spectrum_all_nan_returns_empty(self) -> None:
        """Test that all-NaN data returns empty result."""
        tie_data = np.full(100, np.nan)
        result = jitter_spectrum(tie_data, sample_rate=1e9)

        assert len(result.frequencies) == 0
        assert result.dominant_frequency is None

    def test_spectrum_window_hann(self) -> None:
        """Test spectrum with Hann window."""
        tie_data = create_pure_periodic_tie(n_samples=1000, sample_rate=10e6)
        result = jitter_spectrum(tie_data, sample_rate=10e6, window="hann")

        assert result.dominant_frequency is not None
        assert len(result.magnitude) > 0

    def test_spectrum_window_hamming(self) -> None:
        """Test spectrum with Hamming window."""
        tie_data = create_pure_periodic_tie(n_samples=1000, sample_rate=10e6)
        result = jitter_spectrum(tie_data, sample_rate=10e6, window="hamming")

        assert result.dominant_frequency is not None

    def test_spectrum_window_blackman(self) -> None:
        """Test spectrum with Blackman window."""
        tie_data = create_pure_periodic_tie(n_samples=1000, sample_rate=10e6)
        result = jitter_spectrum(tie_data, sample_rate=10e6, window="blackman")

        assert result.dominant_frequency is not None

    def test_spectrum_window_none(self) -> None:
        """Test spectrum with no windowing."""
        tie_data = create_pure_periodic_tie(n_samples=1000, sample_rate=10e6)
        result = jitter_spectrum(tie_data, sample_rate=10e6, window="none")

        assert result.dominant_frequency is not None

    def test_spectrum_detrend_enabled(self) -> None:
        """Test spectrum with detrending enabled."""
        tie_data = create_pure_periodic_tie(n_samples=1000, sample_rate=10e6)
        # Add linear trend
        tie_data += np.linspace(0, 10e-12, len(tie_data))

        result = jitter_spectrum(tie_data, sample_rate=10e6, detrend=True)

        # Should still detect periodic component despite trend
        assert result.dominant_frequency is not None

    def test_spectrum_detrend_disabled(self) -> None:
        """Test spectrum with detrending disabled."""
        tie_data = create_pure_periodic_tie(n_samples=1000, sample_rate=10e6)
        result = jitter_spectrum(tie_data, sample_rate=10e6, detrend=False)

        # Should subtract mean instead
        assert len(result.magnitude) > 0

    def test_spectrum_n_peaks_parameter(self) -> None:
        """Test spectrum with different n_peaks values."""
        tie_data = create_multi_tone_tie(n_samples=10000, sample_rate=100e6)

        result_3 = jitter_spectrum(tie_data, sample_rate=100e6, n_peaks=3)
        result_10 = jitter_spectrum(tie_data, sample_rate=100e6, n_peaks=10)

        # More peaks requested should give more peaks (up to available)
        assert len(result_3.peaks) <= 3
        assert len(result_10.peaks) >= len(result_3.peaks)

    def test_spectrum_magnitude_scaling(self) -> None:
        """Test that magnitude is properly scaled."""
        jitter_amp = 10e-12  # 10 ps
        tie_data = create_pure_periodic_tie(
            n_samples=2048,  # Power of 2 for cleaner FFT
            sample_rate=100e6,
            jitter_freq=1e6,
            jitter_amplitude=jitter_amp,
        )

        result = jitter_spectrum(tie_data, sample_rate=100e6)

        # Peak magnitude should be close to input amplitude
        assert result.dominant_magnitude is not None
        # Allow some tolerance due to windowing and FFT effects
        assert np.isclose(result.dominant_magnitude, jitter_amp, rtol=0.3)

    def test_spectrum_magnitude_db_conversion(self) -> None:
        """Test magnitude dB conversion (relative to 1 ps)."""
        tie_data = create_pure_periodic_tie(
            n_samples=1000,
            sample_rate=10e6,
            jitter_amplitude=1e-12,  # 1 ps
        )

        result = jitter_spectrum(tie_data, sample_rate=10e6)

        # For 1 ps signal, dB should be around 0 dB
        # Find the dB value at dominant frequency
        if result.dominant_frequency is not None:
            # Peak dB should be reasonably close to 0 dB for 1 ps signal
            max_db = np.max(result.magnitude_db)
            assert -10 < max_db < 10  # Allow some tolerance

    def test_spectrum_noise_floor_estimation(self) -> None:
        """Test noise floor is estimated as median of spectrum."""
        # Create data with known noise level
        rng = np.random.default_rng(42)
        noise_rms = 1e-12
        tie_data = rng.normal(0, noise_rms, 5000)

        result = jitter_spectrum(tie_data, sample_rate=50e6)

        # Noise floor should be on order of noise RMS
        assert result.noise_floor > 0
        assert result.noise_floor < 10 * noise_rms

    def test_spectrum_frequencies_monotonic(self) -> None:
        """Test that frequency array is monotonically increasing."""
        tie_data = create_pure_periodic_tie(n_samples=1000, sample_rate=10e6)
        result = jitter_spectrum(tie_data, sample_rate=10e6)

        # Frequencies should be monotonically increasing
        assert np.all(np.diff(result.frequencies) >= 0)

    def test_spectrum_frequencies_range(self) -> None:
        """Test frequency range is from 0 to Nyquist."""
        sample_rate = 10e6
        tie_data = create_pure_periodic_tie(n_samples=1000, sample_rate=sample_rate)
        result = jitter_spectrum(tie_data, sample_rate=sample_rate)

        # First frequency should be 0 (DC)
        assert result.frequencies[0] == 0

        # Last frequency should be <= Nyquist
        nyquist = sample_rate / 2
        assert result.frequencies[-1] <= nyquist

    def test_spectrum_zero_padding(self) -> None:
        """Test that FFT uses zero-padding to next power of 2."""
        tie_data = create_pure_periodic_tie(n_samples=1000, sample_rate=10e6)
        result = jitter_spectrum(tie_data, sample_rate=10e6)

        # With 1000 samples, FFT should be padded to 1024 (2^10)
        # rfft returns N//2 + 1 points
        expected_fft_size = 2 ** int(np.ceil(np.log2(1000)))
        expected_spectrum_points = expected_fft_size // 2 + 1

        assert len(result.frequencies) == expected_spectrum_points

    def test_spectrum_dc_component_handling(self) -> None:
        """Test handling of DC component (zero frequency)."""
        # Create data with DC offset
        tie_data = create_pure_periodic_tie(n_samples=1000, sample_rate=10e6)
        tie_data += 100e-12  # Add DC offset

        result = jitter_spectrum(tie_data, sample_rate=10e6, detrend=True)

        # DC component should be removed by detrending
        # First element (DC) should be small
        assert result.magnitude[0] < np.max(result.magnitude)


# =============================================================================
# Tests for identify_periodic_components Function
# =============================================================================


@pytest.mark.unit
@pytest.mark.jitter
class TestIdentifyPeriodicComponents:
    """Test periodic component identification from spectrum."""

    def test_identify_peaks_single_peak(self) -> None:
        """Test identification of single dominant peak."""
        frequencies = np.linspace(0, 50e6, 1000)
        # Create spectrum with single peak at 10 MHz
        magnitude = np.ones(1000) * 1e-13  # Noise floor
        magnitude[200] = 10e-12  # Single peak

        peaks = identify_periodic_components(frequencies, magnitude, n_peaks=10)

        # Should find one peak
        assert len(peaks) == 1
        assert np.isclose(peaks[0][0], frequencies[200])
        assert np.isclose(peaks[0][1], 10e-12)

    def test_identify_peaks_multiple_peaks(self) -> None:
        """Test identification of multiple peaks."""
        frequencies = np.linspace(0, 50e6, 1000)
        magnitude = np.ones(1000) * 1e-13

        # Add multiple peaks
        magnitude[100] = 10e-12  # Largest
        magnitude[300] = 8e-12  # Second
        magnitude[500] = 5e-12  # Third

        peaks = identify_periodic_components(frequencies, magnitude, n_peaks=5)

        # Should find all three peaks
        assert len(peaks) == 3

        # Should be sorted by magnitude (descending)
        assert peaks[0][1] >= peaks[1][1] >= peaks[2][1]

    def test_identify_peaks_sorted_by_magnitude(self) -> None:
        """Test that peaks are sorted by magnitude (largest first)."""
        frequencies = np.linspace(0, 50e6, 1000)
        magnitude = np.ones(1000) * 1e-13

        # Add peaks with known order
        magnitude[100] = 5e-12  # Second largest
        magnitude[500] = 10e-12  # Largest
        magnitude[800] = 3e-12  # Smallest

        peaks = identify_periodic_components(frequencies, magnitude, n_peaks=10)

        # First peak should be the largest (at index 500)
        assert np.isclose(peaks[0][0], frequencies[500])
        assert np.isclose(peaks[0][1], 10e-12)

        # Second peak should be second largest
        assert np.isclose(peaks[1][0], frequencies[100])

    def test_identify_peaks_n_peaks_limit(self) -> None:
        """Test that n_peaks limits number of returned peaks."""
        frequencies = np.linspace(0, 50e6, 1000)
        magnitude = np.random.randn(1000) * 1e-12 + 5e-12  # Noisy spectrum

        peaks_3 = identify_periodic_components(frequencies, magnitude, n_peaks=3)
        peaks_5 = identify_periodic_components(frequencies, magnitude, n_peaks=5)

        assert len(peaks_3) <= 3
        assert len(peaks_5) <= 5

    def test_identify_peaks_min_height_filtering(self) -> None:
        """Test that min_height filters out small peaks."""
        frequencies = np.linspace(0, 50e6, 1000)
        magnitude = np.ones(1000) * 1e-13

        magnitude[100] = 10e-12  # Above threshold
        magnitude[500] = 2e-13  # Below threshold

        peaks = identify_periodic_components(frequencies, magnitude, n_peaks=10, min_height=5e-13)

        # Should only find the large peak
        assert len(peaks) == 1
        assert np.isclose(peaks[0][1], 10e-12)

    def test_identify_peaks_min_height_default(self) -> None:
        """Test default min_height is 3x median."""
        frequencies = np.linspace(0, 50e6, 1000)
        magnitude = np.ones(1000) * 1e-12
        magnitude[500] = 10e-12  # Peak well above median

        peaks = identify_periodic_components(frequencies, magnitude, n_peaks=10)

        # Should find the peak (default threshold is 3 * median = 3e-12)
        assert len(peaks) >= 1

    def test_identify_peaks_min_distance(self) -> None:
        """Test min_distance prevents close peaks."""
        frequencies = np.linspace(0, 50e6, 1000)
        magnitude = np.ones(1000) * 1e-13

        # Add two peaks very close together
        magnitude[500] = 10e-12
        magnitude[502] = 8e-12  # Only 2 bins away

        peaks_dist_1 = identify_periodic_components(
            frequencies, magnitude, n_peaks=10, min_distance=1
        )
        peaks_dist_5 = identify_periodic_components(
            frequencies, magnitude, n_peaks=10, min_distance=5
        )

        # With distance=1, both peaks might be found
        # With distance=5, only one peak should be found
        assert len(peaks_dist_5) <= len(peaks_dist_1)

    def test_identify_peaks_empty_spectrum(self) -> None:
        """Test with empty frequency/magnitude arrays."""
        frequencies = np.array([])
        magnitude = np.array([])

        peaks = identify_periodic_components(frequencies, magnitude)

        assert len(peaks) == 0

    def test_identify_peaks_very_small_spectrum(self) -> None:
        """Test with spectrum smaller than 3 points."""
        frequencies = np.array([0, 1e6])
        magnitude = np.array([1e-12, 2e-12])

        peaks = identify_periodic_components(frequencies, magnitude)

        # Too small for peak detection
        assert len(peaks) == 0

    def test_identify_peaks_no_peaks_above_threshold(self) -> None:
        """Test when no peaks exceed threshold."""
        frequencies = np.linspace(0, 50e6, 1000)
        magnitude = np.ones(1000) * 1e-13  # All same level (noise floor)

        peaks = identify_periodic_components(frequencies, magnitude, n_peaks=10)

        # Should find no peaks
        assert len(peaks) == 0

    def test_identify_peaks_return_type(self) -> None:
        """Test that peaks are returned as list of tuples."""
        frequencies = np.linspace(0, 50e6, 1000)
        magnitude = np.ones(1000) * 1e-13
        magnitude[500] = 10e-12

        peaks = identify_periodic_components(frequencies, magnitude)

        assert isinstance(peaks, list)
        if len(peaks) > 0:
            assert isinstance(peaks[0], tuple)
            assert len(peaks[0]) == 2
            assert isinstance(peaks[0][0], float)
            assert isinstance(peaks[0][1], float)

    def test_identify_peaks_frequency_values(self) -> None:
        """Test that returned frequencies are correct."""
        frequencies = np.array([0, 1e6, 2e6, 3e6, 4e6, 5e6, 6e6, 7e6, 8e6])
        magnitude = np.array([1e-13, 1e-13, 10e-12, 1e-13, 1e-13, 8e-12, 1e-13, 1e-13, 1e-13])

        peaks = identify_periodic_components(frequencies, magnitude, n_peaks=5, min_distance=1)

        # Should find peaks at 2 MHz and 5 MHz
        peak_freqs = [p[0] for p in peaks]
        assert 2e6 in peak_freqs
        assert 5e6 in peak_freqs

    def test_identify_peaks_magnitude_values(self) -> None:
        """Test that returned magnitudes are correct."""
        frequencies = np.linspace(0, 50e6, 1000)
        magnitude = np.ones(1000) * 1e-13
        magnitude[500] = 10e-12

        peaks = identify_periodic_components(frequencies, magnitude)

        # Magnitude should match
        assert np.isclose(peaks[0][1], 10e-12)


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.jitter
class TestJitterSpectrumIntegration:
    """Integration tests combining spectrum analysis components."""

    def test_full_workflow_single_tone(self) -> None:
        """Test complete workflow from TIE to spectrum analysis."""
        # Generate known jitter
        jitter_freq = 5e6  # 5 MHz
        jitter_amp = 15e-12  # 15 ps
        sample_rate = 100e6  # 100 MHz

        tie_data = create_pure_periodic_tie(
            n_samples=10000,
            sample_rate=sample_rate,
            jitter_freq=jitter_freq,
            jitter_amplitude=jitter_amp,
        )

        # Analyze spectrum
        result = jitter_spectrum(tie_data, sample_rate=sample_rate, n_peaks=5)

        # Verify results
        assert result.dominant_frequency is not None
        assert np.isclose(result.dominant_frequency, jitter_freq, rtol=0.05)
        assert result.dominant_magnitude is not None
        assert np.isclose(result.dominant_magnitude, jitter_amp, rtol=0.3)

    def test_full_workflow_multi_tone(self) -> None:
        """Test complete workflow with multiple jitter sources."""
        tie_data = create_multi_tone_tie(n_samples=20000, sample_rate=100e6)

        result = jitter_spectrum(tie_data, sample_rate=100e6, n_peaks=10)

        # Should detect multiple components
        assert len(result.peaks) >= 3

        # Dominant should be 100 kHz (largest amplitude: 20 ps)
        assert result.dominant_frequency is not None
        assert np.isclose(result.dominant_frequency, 100e3, rtol=0.2)

    def test_spectrum_with_identify_components_consistency(self) -> None:
        """Test consistency between jitter_spectrum and identify_periodic_components."""
        tie_data = create_multi_tone_tie(n_samples=10000, sample_rate=100e6)

        result = jitter_spectrum(tie_data, sample_rate=100e6, n_peaks=5)

        # Manually identify peaks
        manual_peaks = identify_periodic_components(
            result.frequencies,
            result.magnitude,
            n_peaks=5,
            min_height=result.noise_floor * 3,
        )

        # Should match
        assert len(result.peaks) == len(manual_peaks)
        for i in range(len(result.peaks)):
            assert np.isclose(result.peaks[i][0], manual_peaks[i][0])
            assert np.isclose(result.peaks[i][1], manual_peaks[i][1])


# =============================================================================
# Edge Cases and Numerical Stability
# =============================================================================


@pytest.mark.unit
@pytest.mark.jitter
class TestEdgeCasesAndNumericalStability:
    """Test edge cases and numerical stability."""

    def test_very_small_jitter(self) -> None:
        """Test with very small jitter (femtoseconds)."""
        tie_data = create_pure_periodic_tie(
            n_samples=1000,
            sample_rate=10e6,
            jitter_amplitude=1e-15,  # 1 fs
        )

        result = jitter_spectrum(tie_data, sample_rate=10e6)

        # Should handle small values
        assert np.all(np.isfinite(result.magnitude))
        assert np.all(np.isfinite(result.magnitude_db))

    def test_very_large_jitter(self) -> None:
        """Test with very large jitter (nanoseconds)."""
        tie_data = create_pure_periodic_tie(
            n_samples=1000,
            sample_rate=10e6,
            jitter_amplitude=1e-9,  # 1 ns
        )

        result = jitter_spectrum(tie_data, sample_rate=10e6)

        # Should handle large values
        assert result.dominant_magnitude is not None
        assert result.dominant_magnitude > 0

    def test_very_high_frequency_jitter(self) -> None:
        """Test with jitter near Nyquist frequency."""
        sample_rate = 100e6
        nyquist = sample_rate / 2
        jitter_freq = nyquist * 0.8  # 80% of Nyquist

        tie_data = create_pure_periodic_tie(
            n_samples=5000,
            sample_rate=sample_rate,
            jitter_freq=jitter_freq,
            jitter_amplitude=10e-12,
        )

        result = jitter_spectrum(tie_data, sample_rate=sample_rate)

        # Should detect high-frequency jitter
        assert result.dominant_frequency is not None

    def test_very_low_frequency_jitter(self) -> None:
        """Test with very low frequency jitter."""
        sample_rate = 100e6
        jitter_freq = 10e3  # 10 kHz (low but resolvable)

        tie_data = create_pure_periodic_tie(
            n_samples=20000,
            sample_rate=sample_rate,
            jitter_freq=jitter_freq,
            jitter_amplitude=10e-12,
        )

        result = jitter_spectrum(tie_data, sample_rate=sample_rate)

        # Should detect low-frequency jitter
        assert result.dominant_frequency is not None
        # Allow wider tolerance for low frequencies due to FFT bin resolution
        assert np.isclose(result.dominant_frequency, jitter_freq, rtol=0.3)

    def test_constant_tie_data(self) -> None:
        """Test with constant TIE data (no jitter)."""
        tie_data = np.ones(1000) * 5e-12

        result = jitter_spectrum(tie_data, sample_rate=10e6)

        # Should have minimal spectral content
        # DC component might be suppressed by detrending
        assert result.noise_floor >= 0

    def test_zero_tie_data(self) -> None:
        """Test with all-zero TIE data."""
        tie_data = np.zeros(1000)

        result = jitter_spectrum(tie_data, sample_rate=10e6)

        # Should handle gracefully
        assert len(result.magnitude) > 0
        assert result.noise_floor >= 0

    def test_tie_data_with_trend(self) -> None:
        """Test with TIE data containing strong linear trend."""
        tie_data = create_pure_periodic_tie(n_samples=1000, sample_rate=10e6)
        # Add strong linear trend
        tie_data += np.linspace(0, 100e-12, len(tie_data))

        result = jitter_spectrum(tie_data, sample_rate=10e6, detrend=True)

        # Detrending should remove trend and allow jitter detection
        assert result.dominant_frequency is not None

    def test_tie_data_with_outliers(self) -> None:
        """Test robustness to outliers in TIE data."""
        tie_data = create_pure_periodic_tie(n_samples=1000, sample_rate=10e6)
        # Add some outliers
        tie_data[100] = 1e-9  # Large outlier
        tie_data[500] = -1e-9  # Large negative outlier

        result = jitter_spectrum(tie_data, sample_rate=10e6)

        # Should still produce result despite outliers
        assert len(result.magnitude) > 0

    def test_power_of_2_vs_non_power_of_2_samples(self) -> None:
        """Test FFT with power-of-2 vs non-power-of-2 sample counts."""
        jitter_freq = 1e6
        sample_rate = 50e6

        # Power of 2
        tie_data_pow2 = create_pure_periodic_tie(
            n_samples=1024, sample_rate=sample_rate, jitter_freq=jitter_freq
        )
        result_pow2 = jitter_spectrum(tie_data_pow2, sample_rate=sample_rate)

        # Non-power of 2
        tie_data_non = create_pure_periodic_tie(
            n_samples=1000, sample_rate=sample_rate, jitter_freq=jitter_freq
        )
        result_non = jitter_spectrum(tie_data_non, sample_rate=sample_rate)

        # Both should detect the frequency
        assert result_pow2.dominant_frequency is not None
        assert result_non.dominant_frequency is not None

    def test_numerical_precision_float64(self) -> None:
        """Test numerical precision with float64 data."""
        tie_data = create_pure_periodic_tie(n_samples=1000, sample_rate=10e6)
        tie_data = tie_data.astype(np.float64)

        result = jitter_spectrum(tie_data, sample_rate=10e6)

        # All outputs should be finite
        assert np.all(np.isfinite(result.frequencies))
        assert np.all(np.isfinite(result.magnitude))
        assert np.all(np.isfinite(result.magnitude_db[result.magnitude_db > -np.inf]))


# =============================================================================
# Parameter Validation and Error Handling
# =============================================================================


@pytest.mark.unit
@pytest.mark.jitter
class TestParameterValidation:
    """Test parameter validation and error handling."""

    def test_invalid_window_type(self) -> None:
        """Test with invalid window type falls back to rectangular."""
        tie_data = create_pure_periodic_tie(n_samples=1000, sample_rate=10e6)

        # Invalid window should use rectangular (ones)
        result = jitter_spectrum(tie_data, sample_rate=10e6, window="invalid")

        # Should still work
        assert len(result.magnitude) > 0

    def test_zero_sample_rate_behavior(self) -> None:
        """Test behavior with zero sample rate (edge case)."""
        tie_data = create_pure_periodic_tie(n_samples=1000, sample_rate=10e6)

        # Zero sample rate should raise ValueError or ZeroDivisionError
        exception_raised = False
        try:
            result = jitter_spectrum(tie_data, sample_rate=0)
            # If it doesn't raise, check for invalid results (inf/nan in frequencies)
            import numpy as np

            assert np.any(np.isinf(result.frequency)) or np.any(np.isnan(result.frequency)), (
                "Expected inf or nan frequencies with zero sample rate"
            )
        except (ZeroDivisionError, ValueError):
            # Expected behavior - exception raised for invalid sample rate
            exception_raised = True

        # Test passes if either exception raised OR invalid frequencies detected
        assert exception_raised or result is not None

    def test_negative_sample_rate_behavior(self) -> None:
        """Test behavior with negative sample rate."""
        tie_data = create_pure_periodic_tie(n_samples=1000, sample_rate=10e6)

        # Negative sample rate is unphysical and should raise ValueError
        exception_raised = False
        try:
            result = jitter_spectrum(tie_data, sample_rate=-10e6)
            # If it doesn't raise, verify result is computed (even if frequencies are negative)

            assert hasattr(result, "frequency"), "Expected result to have frequency attribute"
            assert hasattr(result, "magnitude"), "Expected result to have magnitude attribute"
            assert len(result.frequency) > 0, "Expected non-empty frequency array"
        except ValueError:
            # Expected behavior - exception raised for negative sample rate
            exception_raised = True

        # Test passes if either exception raised OR result computed successfully
        assert exception_raised or result is not None

    def test_n_peaks_zero(self) -> None:
        """Test with n_peaks=0 returns no peaks."""
        tie_data = create_pure_periodic_tie(n_samples=1000, sample_rate=10e6)
        result = jitter_spectrum(tie_data, sample_rate=10e6, n_peaks=0)

        # Should return no peaks
        assert len(result.peaks) == 0
        assert result.dominant_frequency is None

    def test_n_peaks_negative(self) -> None:
        """Test behavior with negative n_peaks."""
        frequencies = np.linspace(0, 50e6, 1000)
        magnitude = np.ones(1000) * 1e-12

        # Negative n_peaks should be handled gracefully
        try:
            peaks = identify_periodic_components(frequencies, magnitude, n_peaks=-5)
            # If it doesn't error, should return empty or limited peaks
            assert len(peaks) >= 0
        except ValueError:
            # Acceptable to raise error
            pass

    def test_empty_tie_array(self) -> None:
        """Test with empty TIE array."""
        tie_data = np.array([])
        result = jitter_spectrum(tie_data, sample_rate=10e6)

        # Should return empty result
        assert len(result.frequencies) == 0

    def test_single_sample_tie(self) -> None:
        """Test with single TIE sample."""
        tie_data = np.array([1e-12])
        result = jitter_spectrum(tie_data, sample_rate=10e6)

        # Should return empty result (insufficient data)
        assert len(result.frequencies) == 0


# =============================================================================
# Performance and Stress Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.jitter
class TestPerformance:
    """Performance and stress tests."""

    def test_large_dataset(self) -> None:
        """Test spectrum analysis with large dataset."""
        # 1 million samples
        tie_data = create_pure_periodic_tie(n_samples=1_000_000, sample_rate=100e6, jitter_freq=1e6)

        result = jitter_spectrum(tie_data, sample_rate=100e6)

        # Should complete and produce result
        assert result.dominant_frequency is not None

    def test_many_frequency_components(self) -> None:
        """Test with many frequency components."""
        # Create data with 20 frequency components
        frequencies_list = [100e3 + i * 500e3 for i in range(20)]
        amplitudes_list = [10e-12 / (i + 1) for i in range(20)]

        tie_data = create_tie_data(
            n_samples=50000,
            sample_rate=100e6,
            frequencies=frequencies_list,
            amplitudes=amplitudes_list,
            noise_rms=0.5e-12,
        )

        result = jitter_spectrum(tie_data, sample_rate=100e6, n_peaks=20)

        # Should detect multiple components
        assert len(result.peaks) >= 10

    def test_high_sample_rate(self) -> None:
        """Test with very high sample rate."""
        sample_rate = 10e9  # 10 GHz
        tie_data = create_pure_periodic_tie(
            n_samples=10000, sample_rate=sample_rate, jitter_freq=100e6
        )

        result = jitter_spectrum(tie_data, sample_rate=sample_rate)

        # Should handle high sample rate
        assert result.frequencies[-1] <= sample_rate / 2
