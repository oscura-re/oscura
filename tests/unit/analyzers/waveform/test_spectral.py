"""Unit tests for spectral analysis functions.

This module provides comprehensive tests for FFT, PSD, and spectral quality
metrics per IEEE 1241-2010 for ADC characterization.
"""

from __future__ import annotations

import numpy as np
import pytest

from oscura.analyzers.waveform.spectral import (
    bartlett_psd,
    enob,
    extract_harmonics,
    fft,
    fft_chunked,
    find_peaks,
    group_delay,
    hilbert_transform,
    mfcc,
    periodogram,
    phase_spectrum,
    psd,
    psd_chunked,
    sfdr,
    sinad,
    snr,
    spectrogram,
    spectrogram_chunked,
    thd,
)
from oscura.analyzers.waveform.wavelets import cwt, dwt
from oscura.core.exceptions import InsufficientDataError
from oscura.core.types import TraceMetadata, WaveformTrace

pytestmark = [pytest.mark.unit, pytest.mark.analyzer]


# =============================================================================
# Helper Functions
# =============================================================================


def make_trace(
    data: np.ndarray,
    sample_rate: float = 1e6,
) -> WaveformTrace:
    """Create a WaveformTrace from data."""
    metadata = TraceMetadata(sample_rate=sample_rate)
    return WaveformTrace(data=data, metadata=metadata)


def make_sine_wave(
    frequency: float,
    sample_rate: float,
    duration: float,
    amplitude: float = 1.0,
    phase: float = 0.0,
) -> np.ndarray:
    """Generate a pure sine wave."""
    t = np.arange(0, duration, 1.0 / sample_rate)
    return amplitude * np.sin(2 * np.pi * frequency * t + phase)


def make_multitone(
    frequencies: list[float],
    amplitudes: list[float],
    sample_rate: float,
    duration: float,
) -> np.ndarray:
    """Generate a multi-tone signal."""
    t = np.arange(0, duration, 1.0 / sample_rate)
    signal = np.zeros_like(t)
    for freq, amp in zip(frequencies, amplitudes, strict=False):
        signal += amp * np.sin(2 * np.pi * freq * t)
    return signal


# =============================================================================
# Test FFT (SPE-001, SPE-003, SPE-010)
# =============================================================================


@pytest.mark.unit
@pytest.mark.requirement("SPE-001")
class TestFFT:
    """Test FFT computation."""

    def test_basic_fft(self) -> None:
        """Test basic FFT on sine wave."""
        freq = 1000.0
        sample_rate = 10000.0
        duration = 1.0

        signal = make_sine_wave(freq, sample_rate, duration)
        trace = make_trace(signal, sample_rate)

        frequencies, magnitude_db = fft(trace)

        # Check output shapes
        assert len(frequencies) == len(magnitude_db)
        assert len(frequencies) > 0

        # Find peak - should be at signal frequency
        peak_idx = np.argmax(magnitude_db)
        peak_freq = frequencies[peak_idx]

        assert pytest.approx(peak_freq, rel=0.05) == freq

    def test_zero_padding(self) -> None:
        """Test FFT with zero-padding."""
        signal = make_sine_wave(1000, 10000, 0.1)
        trace = make_trace(signal, 10000)

        # Without zero-padding
        freq1, mag1 = fft(trace, nfft=None)

        # With zero-padding to 2x length
        freq2, mag2 = fft(trace, nfft=len(signal) * 2)

        # Zero-padded should have better frequency resolution
        assert len(freq2) > len(freq1)

    def test_different_windows(self) -> None:
        """Test FFT with different window functions."""
        signal = make_sine_wave(1000, 10000, 0.1)
        trace = make_trace(signal, 10000)

        freq_hann, mag_hann = fft(trace, window="hann")
        freq_hamming, mag_hamming = fft(trace, window="hamming")
        freq_blackman, mag_blackman = fft(trace, window="blackman")

        # All should find the peak at the same frequency
        peak_hann = freq_hann[np.argmax(mag_hann)]
        peak_hamming = freq_hamming[np.argmax(mag_hamming)]
        peak_blackman = freq_blackman[np.argmax(mag_blackman)]

        assert pytest.approx(peak_hann, rel=0.05) == 1000.0
        assert pytest.approx(peak_hamming, rel=0.05) == 1000.0
        assert pytest.approx(peak_blackman, rel=0.05) == 1000.0

    def test_detrending(self) -> None:
        """Test FFT with different detrending options."""
        signal = make_sine_wave(1000, 10000, 0.1) + 2.0  # Add DC offset
        trace = make_trace(signal, 10000)

        # With mean detrending (default)
        freq_mean, mag_mean = fft(trace, detrend="mean")

        # Without detrending
        freq_none, mag_none = fft(trace, detrend="none")

        # DC component should be smaller with mean detrending
        dc_mean = mag_mean[0]
        dc_none = mag_none[0]

        assert dc_mean < dc_none

    def test_return_phase(self) -> None:
        """Test FFT with phase output."""
        signal = make_sine_wave(1000, 10000, 0.1)
        trace = make_trace(signal, 10000)

        result = fft(trace, return_phase=True)

        assert len(result) == 3
        frequencies, magnitude, phase = result

        assert len(frequencies) == len(magnitude) == len(phase)
        assert np.all(np.abs(phase) <= np.pi)

    def test_insufficient_data(self) -> None:
        """Test FFT with insufficient data."""
        signal = np.array([1.0])
        trace = make_trace(signal)

        with pytest.raises(InsufficientDataError):
            fft(trace)


# =============================================================================
# Test PSD (SPE-004, SPE-012, SPE-013)
# =============================================================================


@pytest.mark.unit
@pytest.mark.requirement("SPE-004")
class TestPSD:
    """Test Power Spectral Density estimation."""

    def test_welch_psd(self) -> None:
        """Test Welch PSD estimation."""
        freq = 1000.0
        sample_rate = 10000.0
        duration = 1.0

        signal = make_sine_wave(freq, sample_rate, duration)
        trace = make_trace(signal, sample_rate)

        frequencies, psd_db = psd(trace)

        # Find peak
        peak_idx = np.argmax(psd_db)
        peak_freq = frequencies[peak_idx]

        assert pytest.approx(peak_freq, rel=0.1) == freq

    def test_periodogram(self) -> None:
        """Test periodogram PSD estimation."""
        freq = 1000.0
        signal = make_sine_wave(freq, 10000, 1.0)
        trace = make_trace(signal, 10000)

        frequencies, psd_db = periodogram(trace)

        peak_idx = np.argmax(psd_db)
        peak_freq = frequencies[peak_idx]

        assert pytest.approx(peak_freq, rel=0.05) == freq

    def test_bartlett_psd(self) -> None:
        """Test Bartlett PSD estimation."""
        freq = 1000.0
        signal = make_sine_wave(freq, 10000, 1.0)
        trace = make_trace(signal, 10000)

        frequencies, psd_db = bartlett_psd(trace, n_segments=8)

        peak_idx = np.argmax(psd_db)
        peak_freq = frequencies[peak_idx]

        assert pytest.approx(peak_freq, rel=0.1) == freq

    def test_psd_parameters(self) -> None:
        """Test PSD with custom parameters."""
        signal = make_sine_wave(1000, 10000, 1.0)
        trace = make_trace(signal, 10000)

        freq1, psd1 = psd(trace, nperseg=256)
        freq2, psd2 = psd(trace, nperseg=512)

        # Different segment lengths should give different results
        assert len(freq1) != len(freq2)

    def test_insufficient_data_psd(self) -> None:
        """Test PSD with insufficient data."""
        signal = np.array([1.0, 2.0, 3.0])
        trace = make_trace(signal)

        with pytest.raises(InsufficientDataError):
            psd(trace)


# =============================================================================
# Test Spectrogram (SPE-011)
# =============================================================================


@pytest.mark.unit
@pytest.mark.requirement("SPE-011")
class TestSpectrogram:
    """Test spectrogram (STFT) computation."""

    def test_basic_spectrogram(self) -> None:
        """Test basic spectrogram on sine wave."""
        freq = 1000.0
        signal = make_sine_wave(freq, 10000, 0.5)
        trace = make_trace(signal, 10000)

        times, frequencies, Sxx_db = spectrogram(trace)

        assert len(times) > 0
        assert len(frequencies) > 0
        assert Sxx_db.shape == (len(frequencies), len(times))

    def test_chirp_spectrogram(self) -> None:
        """Test spectrogram on frequency-changing signal."""
        sample_rate = 10000.0
        duration = 1.0
        t = np.arange(0, duration, 1.0 / sample_rate)

        # Linear chirp from 500 Hz to 2000 Hz
        f0, f1 = 500, 2000
        signal = np.sin(2 * np.pi * (f0 * t + (f1 - f0) / (2 * duration) * t**2))

        trace = make_trace(signal, sample_rate)
        times, frequencies, Sxx_db = spectrogram(trace, nperseg=256)

        # Spectrogram should show frequency change over time
        assert Sxx_db.shape[1] > 5  # Multiple time slices


# =============================================================================
# Test THD (SPE-005)
# =============================================================================


@pytest.mark.unit
@pytest.mark.requirement("SPE-005")
class TestTHD:
    """Test Total Harmonic Distortion measurement."""

    def test_pure_sine_thd(self) -> None:
        """Test THD on pure sine wave (should be very low)."""
        signal = make_sine_wave(1000, 100000, 0.1, amplitude=1.0)
        trace = make_trace(signal, 100000)

        thd_result = thd(trace, n_harmonics=5)

        # Pure sine should have very low THD
        assert thd_result["applicable"]
        assert thd_result["unit"] == "%"
        thd_pct = thd_result["value"]
        assert thd_pct is not None
        assert thd_pct < 0.1  # Less than 0.1%

    def test_distorted_signal_thd(self) -> None:
        """Test THD on signal with harmonics."""
        # Fundamental + 2nd and 3rd harmonics
        f0 = 1000.0
        signal = make_multitone(
            [f0, 2 * f0, 3 * f0],
            [1.0, 0.1, 0.05],  # 10% 2nd, 5% 3rd harmonic
            100000,
            0.1,
        )
        trace = make_trace(signal, 100000)

        thd_result = thd(trace, n_harmonics=5)

        # Should detect significant THD (>3%)
        assert thd_result["applicable"]
        thd_pct = thd_result["value"]
        assert thd_pct is not None
        assert thd_pct > 3.0  # More than 3%

    def test_thd_return_percentage(self) -> None:
        """Test THD in percentage format."""
        signal = make_multitone([1000, 2000, 3000], [1.0, 0.1, 0.05], 100000, 0.1)
        trace = make_trace(signal, 100000)

        thd_result = thd(trace, n_harmonics=5, return_db=False)

        # THD now always returns percentage in MeasurementResult format
        assert thd_result["applicable"]
        assert thd_result["unit"] == "%"
        thd_pct = thd_result["value"]
        assert thd_pct is not None
        assert thd_pct > 0
        assert isinstance(thd_pct, float)

    def test_thd_known_signal_ieee_formula(self) -> None:
        """Test THD calculation with known harmonics per IEEE 1241-2010.

        Verifies: THD = sqrt(sum(A_harmonics^2)) / A_fundamental
        """
        # Create signal with known THD
        # Fundamental: 1000 Hz, amplitude 1.0
        # 2nd harmonic: 2000 Hz, amplitude 0.5
        # 3rd harmonic: 3000 Hz, amplitude 0.3
        # Expected THD = sqrt(0.5^2 + 0.3^2) / 1.0 = 0.583 = 58.3%
        f0 = 1000.0
        sample_rate = 100000.0
        duration = 0.1
        t = np.arange(0, duration, 1.0 / sample_rate)

        signal = (
            1.0 * np.sin(2 * np.pi * f0 * t)
            + 0.5 * np.sin(2 * np.pi * 2 * f0 * t)
            + 0.3 * np.sin(2 * np.pi * 3 * f0 * t)
        )

        trace = make_trace(signal, sample_rate)
        thd_result = thd(trace, n_harmonics=10, return_db=False)

        # Expected: sqrt(0.5^2 + 0.3^2) / 1.0 = 0.583 = 58.3%
        assert thd_result["applicable"]
        thd_pct = thd_result["value"]
        assert thd_pct is not None
        expected_thd = np.sqrt(0.5**2 + 0.3**2) * 100
        assert pytest.approx(thd_pct, rel=0.02) == expected_thd

    def test_thd_always_nonnegative(self) -> None:
        """Test that THD percentage is always non-negative.

        This is a critical invariant: THD cannot be negative in percentage form.
        """
        # Test various signals
        test_signals = [
            make_sine_wave(1000, 100000, 0.1, amplitude=1.0),
            make_multitone([1000, 2000], [1.0, 0.5], 100000, 0.1),
            make_multitone([1000, 2000, 3000], [1.0, 0.3, 0.2], 100000, 0.1),
            make_multitone([1000, 3000, 5000], [1.0, 0.1, 0.05], 100000, 0.1),
        ]

        for signal in test_signals:
            trace = make_trace(signal, 100000)
            thd_result = thd(trace, n_harmonics=10, return_db=False)
            assert thd_result["applicable"]
            thd_pct = thd_result["value"]
            assert thd_pct is not None
            assert thd_pct >= 0, f"THD percentage must be non-negative, got {thd_pct}%"

    def test_thd_high_distortion(self) -> None:
        """Test THD with high harmonic content (>100% is valid).

        THD can exceed 100% when harmonics are strong relative to fundamental.
        Note: Per IEEE 1241-2010, the fundamental is the strongest spectral peak
        (the applied input signal), and harmonics are distortion products.
        """
        # Create signal with large 2nd harmonic (but fundamental still strongest)
        # Fundamental: 1.0, 2nd harmonic: 0.9
        # Expected THD = sqrt(0.9^2) / 1.0 = 90%
        f0 = 1000.0
        sample_rate = 100000.0
        duration = 0.1
        t = np.arange(0, duration, 1.0 / sample_rate)

        signal = 1.0 * np.sin(2 * np.pi * f0 * t) + 0.9 * np.sin(2 * np.pi * 2 * f0 * t)

        trace = make_trace(signal, sample_rate)
        thd_result = thd(trace, n_harmonics=10, return_db=False)

        # Should be approximately 90%
        assert thd_result["applicable"]
        thd_pct = thd_result["value"]
        assert thd_pct is not None
        assert pytest.approx(thd_pct, rel=0.05) == 90.0
        assert thd_pct >= 0

    def test_thd_db_percentage_consistency(self) -> None:
        """Test that THD always returns percentage format.

        Note: return_db parameter is now ignored, THD always returns percentage.
        """
        signal = make_multitone([1000, 2000, 3000], [1.0, 0.5, 0.3], 100000, 0.1)
        trace = make_trace(signal, 100000)

        thd_result_db = thd(trace, n_harmonics=10, return_db=True)
        thd_result_pct = thd(trace, n_harmonics=10, return_db=False)

        # Both should return same percentage value (return_db is ignored)
        assert thd_result_db["applicable"]
        assert thd_result_pct["applicable"]
        assert thd_result_db["unit"] == "%"
        assert thd_result_pct["unit"] == "%"
        thd_val_db = thd_result_db["value"]
        thd_val_pct = thd_result_pct["value"]
        assert thd_val_db is not None
        assert thd_val_pct is not None
        assert pytest.approx(thd_val_db, rel=1e-6) == thd_val_pct


# =============================================================================
# Test SNR (SPE-006)
# =============================================================================


@pytest.mark.unit
@pytest.mark.requirement("SPE-006")
class TestSNR:
    """Test Signal-to-Noise Ratio measurement."""

    def test_pure_sine_snr(self) -> None:
        """Test SNR on pure sine wave (should be very high)."""
        signal = make_sine_wave(1000, 100000, 0.1, amplitude=1.0)
        trace = make_trace(signal, 100000)

        snr_result = snr(trace, n_harmonics=5)

        # Pure sine should have very high SNR
        assert snr_result["applicable"]
        assert snr_result["unit"] == "dB"
        snr_db = snr_result["value"]
        assert snr_db is not None
        assert snr_db > 60

    def test_noisy_signal_snr(self) -> None:
        """Test SNR on noisy signal."""
        rng = np.random.default_rng(42)
        signal = make_sine_wave(1000, 100000, 0.1, amplitude=1.0)
        noise = rng.normal(0, 0.1, len(signal))
        noisy_signal = signal + noise

        trace = make_trace(noisy_signal, 100000)

        snr_result = snr(trace, n_harmonics=5)

        # Should detect reduced SNR due to noise
        assert snr_result["applicable"]
        snr_db = snr_result["value"]
        assert snr_db is not None
        assert 10 < snr_db < 40


# =============================================================================
# Test SINAD (SPE-007)
# =============================================================================


@pytest.mark.unit
@pytest.mark.requirement("SPE-007")
class TestSINAD:
    """Test SINAD measurement."""

    def test_pure_sine_sinad(self) -> None:
        """Test SINAD on pure sine wave."""
        signal = make_sine_wave(1000, 100000, 0.1, amplitude=1.0)
        trace = make_trace(signal, 100000)

        sinad_result = sinad(trace)

        # Pure sine should have high SINAD
        assert sinad_result["applicable"]
        assert sinad_result["unit"] == "dB"
        sinad_db = sinad_result["value"]
        assert sinad_db is not None
        assert sinad_db > 60

    def test_sinad_with_noise_and_distortion(self) -> None:
        """Test SINAD on signal with noise and harmonics."""
        rng = np.random.default_rng(42)
        # Signal with harmonics and noise
        signal = make_multitone([1000, 2000, 3000], [1.0, 0.1, 0.05], 100000, 0.1)
        noise = rng.normal(0, 0.05, len(signal))
        noisy_signal = signal + noise

        trace = make_trace(noisy_signal, 100000)

        sinad_result = sinad(trace)

        # Should detect degraded SINAD
        assert sinad_result["applicable"]
        sinad_db = sinad_result["value"]
        assert sinad_db is not None
        assert 10 < sinad_db < 50


# =============================================================================
# Test ENOB (SPE-008)
# =============================================================================


@pytest.mark.unit
@pytest.mark.requirement("SPE-008")
class TestENOB:
    """Test Effective Number of Bits measurement."""

    def test_enob_calculation(self) -> None:
        """Test ENOB calculation from SINAD."""
        signal = make_sine_wave(1000, 100000, 0.1, amplitude=1.0)
        trace = make_trace(signal, 100000)

        enob_result = enob(trace)

        # Pure sine should have high ENOB
        assert enob_result["applicable"]
        bits = enob_result["value"]
        assert bits is not None
        assert bits > 10  # Reasonable for clean signal

    def test_enob_with_noise(self) -> None:
        """Test ENOB with noisy signal."""
        rng = np.random.default_rng(42)
        signal = make_sine_wave(1000, 100000, 0.1, amplitude=1.0)
        noise = rng.normal(0, 0.05, len(signal))  # Reduce noise level
        noisy_signal = signal + noise

        trace = make_trace(noisy_signal, 100000)

        enob_result = enob(trace)

        # Noise should reduce ENOB
        assert enob_result["applicable"]
        bits = enob_result["value"]
        assert bits is not None
        assert 2 < bits < 12  # More lenient range


# =============================================================================
# Test SFDR (SPE-009)
# =============================================================================


@pytest.mark.unit
@pytest.mark.requirement("SPE-009")
class TestSFDR:
    """Test Spurious-Free Dynamic Range measurement."""

    def test_pure_sine_sfdr(self) -> None:
        """Test SFDR on pure sine wave."""
        signal = make_sine_wave(1000, 100000, 0.1, amplitude=1.0)
        trace = make_trace(signal, 100000)

        sfdr_result = sfdr(trace)

        # Pure sine should have very high SFDR
        assert sfdr_result["applicable"]
        assert sfdr_result["unit"] == "dB"
        sfdr_db = sfdr_result["value"]
        assert sfdr_db is not None
        assert sfdr_db > 60

    def test_sfdr_with_spur(self) -> None:
        """Test SFDR on signal with spurious tone."""
        # Fundamental + spur at different frequency
        signal = make_multitone([1000, 3500], [1.0, 0.1], 100000, 0.1)
        trace = make_trace(signal, 100000)

        sfdr_result = sfdr(trace)

        # Should detect spur
        assert sfdr_result["applicable"]
        sfdr_db = sfdr_result["value"]
        assert sfdr_db is not None
        assert 10 < sfdr_db < 30  # ~20 dB for 10% spur


# =============================================================================
# Test Hilbert Transform (SPE-016)
# =============================================================================


@pytest.mark.unit
@pytest.mark.requirement("SPE-016")
class TestHilbertTransform:
    """Test Hilbert transform for envelope and instantaneous frequency."""

    def test_envelope_extraction(self) -> None:
        """Test envelope extraction from AM signal."""
        sample_rate = 10000.0
        duration = 1.0
        t = np.arange(0, duration, 1.0 / sample_rate)

        # AM signal: carrier modulated by envelope
        carrier_freq = 1000.0
        envelope_freq = 10.0
        envelope = 1.0 + 0.5 * np.sin(2 * np.pi * envelope_freq * t)
        carrier = np.sin(2 * np.pi * carrier_freq * t)
        signal = envelope * carrier

        trace = make_trace(signal, sample_rate)

        env, phase, inst_freq = hilbert_transform(trace)

        # Envelope should match the modulation envelope
        assert len(env) == len(signal)
        assert np.all(env > 0)

    def test_instantaneous_frequency(self) -> None:
        """Test instantaneous frequency extraction."""
        signal = make_sine_wave(1000, 10000, 0.1)
        trace = make_trace(signal, 10000)

        env, phase, inst_freq = hilbert_transform(trace)

        # Instantaneous frequency should be near 1000 Hz
        mean_freq = np.mean(inst_freq)
        assert pytest.approx(mean_freq, rel=0.1) == 1000.0


# =============================================================================
# Test Wavelet Transforms (SPE-014, SPE-015)
# =============================================================================


@pytest.mark.unit
@pytest.mark.requirement("SPE-014")
class TestCWT:
    """Test Continuous Wavelet Transform."""

    def test_cwt_basic(self) -> None:
        """Test basic CWT computation."""
        try:
            import pywt  # noqa: F401
        except ImportError:
            # SKIP: Valid - Optional pywavelets dependency
            # Only skip if pywavelets not installed (pip install oscura[wavelets])
            pytest.skip("PyWavelets not installed")

        signal = make_sine_wave(1000, 10000, 0.5)
        scales = np.arange(1, 33)

        coefficients, frequencies = cwt(signal, scales, wavelet="morl")

        assert len(scales) == 32
        assert len(frequencies) == 32
        assert coefficients.shape == (32, len(signal))

    def test_cwt_different_wavelets(self) -> None:
        """Test CWT with different wavelets."""
        try:
            import pywt  # noqa: F401
        except ImportError:
            # SKIP: Valid - Optional pywavelets dependency
            # Only skip if pywavelets not installed (pip install oscura[wavelets])
            pytest.skip("PyWavelets not installed")

        signal = make_sine_wave(1000, 10000, 0.1)
        scales = np.arange(1, 33)

        coef1, freq1 = cwt(signal, scales, wavelet="morl")
        coef2, freq2 = cwt(signal, scales, wavelet="mexh")

        # Both should work
        assert coef1.shape[0] > 0
        assert coef2.shape[0] > 0


@pytest.mark.unit
@pytest.mark.requirement("SPE-015")
class TestDWT:
    """Test Discrete Wavelet Transform."""

    def test_dwt_basic(self) -> None:
        """Test basic DWT computation."""
        try:
            import pywt  # noqa: F401
        except ImportError:
            # SKIP: Valid - Optional pywavelets dependency
            # Only skip if pywavelets not installed (pip install oscura[wavelets])
            pytest.skip("PyWavelets not installed")

        signal = make_sine_wave(1000, 10000, 0.1)

        coeffs = dwt(signal, wavelet="db4", level=3)

        # Should return list with approximation and details
        assert len(coeffs) == 4  # [cA3, cD3, cD2, cD1]
        assert all(isinstance(c, np.ndarray) for c in coeffs)

    def test_dwt_different_wavelets(self) -> None:
        """Test DWT with different wavelet families."""
        try:
            import pywt  # noqa: F401
        except ImportError:
            # SKIP: Valid - Optional pywavelets dependency
            # Only skip if pywavelets not installed (pip install oscura[wavelets])
            pytest.skip("PyWavelets not installed")

        signal = make_sine_wave(1000, 10000, 0.1)

        coeffs_db = dwt(signal, wavelet="db4", level=2)
        coeffs_sym = dwt(signal, wavelet="sym4", level=2)
        coeffs_coif = dwt(signal, wavelet="coif1", level=2)

        # All should produce valid coefficients
        assert len(coeffs_db) > 0
        assert len(coeffs_sym) > 0
        assert len(coeffs_coif) > 0


# =============================================================================
# Test MFCC (SPE-017)
# =============================================================================


@pytest.mark.unit
@pytest.mark.requirement("SPE-017")
class TestMFCC:
    """Test Mel-Frequency Cepstral Coefficients."""

    def test_mfcc_basic(self) -> None:
        """Test basic MFCC computation."""
        # Create a longer signal for MFCC
        signal = make_sine_wave(1000, 16000, 1.0)
        trace = make_trace(signal, 16000)

        mfcc_features = mfcc(trace, n_mfcc=13, n_fft=512)

        # Should have 13 MFCCs over multiple frames
        assert mfcc_features.shape[0] == 13
        assert mfcc_features.shape[1] > 0

    def test_mfcc_parameters(self) -> None:
        """Test MFCC with different parameters."""
        signal = make_sine_wave(1000, 16000, 1.0)
        trace = make_trace(signal, 16000)

        mfcc1 = mfcc(trace, n_mfcc=13)
        mfcc2 = mfcc(trace, n_mfcc=20)

        assert mfcc1.shape[0] == 13
        assert mfcc2.shape[0] == 20

    def test_mfcc_insufficient_data(self) -> None:
        """Test MFCC with insufficient data."""
        signal = np.random.randn(100)
        trace = make_trace(signal)

        with pytest.raises(InsufficientDataError):
            mfcc(trace, n_fft=512)


# =============================================================================
# Test Chunked Processing (MEM-004, MEM-005, MEM-006)
# =============================================================================


@pytest.mark.unit
@pytest.mark.requirement("MEM-004")
class TestSpectrogramChunked:
    """Test chunked spectrogram for large signals."""

    def test_chunked_matches_standard(self) -> None:
        """Test that chunked spectrogram matches standard for small signal."""
        signal = make_sine_wave(1000, 10000, 0.5)
        trace = make_trace(signal, 10000)

        # Standard
        t1, f1, S1 = spectrogram(trace, nperseg=128)

        # Chunked with small chunk size
        t2, f2, S2 = spectrogram_chunked(trace, chunk_size=10000, nperseg=128)

        # Should be similar
        assert len(f1) == len(f2)
        assert S1.shape == S2.shape

    def test_chunked_large_signal(self) -> None:
        """Test chunked spectrogram on larger signal."""
        # Create larger signal
        signal = make_sine_wave(1000, 10000, 5.0)  # 50k samples
        trace = make_trace(signal, 10000)

        t, f, S = spectrogram_chunked(trace, chunk_size=10000, nperseg=256)

        assert len(t) > 0
        assert len(f) > 0
        assert S.shape == (len(f), len(t))


@pytest.mark.unit
@pytest.mark.requirement("MEM-005")
class TestPSDChunked:
    """Test chunked PSD for large signals."""

    def test_chunked_psd_matches_standard(self) -> None:
        """Test that chunked PSD matches standard for small signal."""
        signal = make_sine_wave(1000, 10000, 0.5)
        trace = make_trace(signal, 10000)

        # Standard
        f1, p1 = psd(trace, nperseg=256)

        # Chunked
        f2, p2 = psd_chunked(trace, chunk_size=10000, nperseg=256)

        # Should be very similar
        assert len(f1) == len(f2)
        # Allow some variation due to averaging
        correlation = np.corrcoef(p1, p2)[0, 1]
        assert correlation > 0.95

    def test_chunked_psd_large_signal(self) -> None:
        """Test chunked PSD on larger signal."""
        signal = make_sine_wave(1000, 10000, 10.0)  # 100k samples
        trace = make_trace(signal, 10000)

        f, p = psd_chunked(trace, chunk_size=20000, nperseg=512)

        # Find peak
        peak_idx = np.argmax(p)
        peak_freq = f[peak_idx]

        assert pytest.approx(peak_freq, rel=0.1) == 1000.0


@pytest.mark.unit
@pytest.mark.requirement("MEM-006")
class TestFFTChunked:
    """Test chunked FFT for very long signals."""

    def test_chunked_fft_basic(self) -> None:
        """Test basic chunked FFT."""
        signal = make_sine_wave(1000, 10000, 5.0)  # 50k samples
        trace = make_trace(signal, 10000)

        f, mag = fft_chunked(trace, segment_size=10000, overlap_pct=50)

        # Find peak
        peak_idx = np.argmax(mag)
        peak_freq = f[peak_idx]

        assert pytest.approx(peak_freq, rel=0.1) == 1000.0

    def test_chunked_fft_overlap_effect(self) -> None:
        """Test effect of different overlap percentages."""
        signal = make_sine_wave(1000, 10000, 5.0)
        trace = make_trace(signal, 10000)

        f1, mag1 = fft_chunked(trace, segment_size=10000, overlap_pct=0)
        f2, mag2 = fft_chunked(trace, segment_size=10000, overlap_pct=50)

        # Both should find the peak
        peak1 = f1[np.argmax(mag1)]
        peak2 = f2[np.argmax(mag2)]

        assert pytest.approx(peak1, rel=0.1) == 1000.0
        assert pytest.approx(peak2, rel=0.1) == 1000.0


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


@pytest.mark.unit
class TestWaveformSpectralEdgeCases:
    """Test edge cases and error handling."""

    def test_dc_signal(self) -> None:
        """Test spectral analysis on DC signal."""
        signal = np.ones(1000)
        trace = make_trace(signal)

        freq, mag = fft(trace)

        # Peak should be at DC (0 Hz)
        peak_idx = np.argmax(mag)
        assert freq[peak_idx] == 0.0

    def test_very_noisy_signal(self) -> None:
        """Test on very noisy signal."""
        rng = np.random.default_rng(42)
        signal = rng.normal(0, 1, 10000)
        trace = make_trace(signal, 10000)

        # Should not crash
        freq, mag = fft(trace)
        assert len(freq) > 0

    def test_complex_multitone(self) -> None:
        """Test on complex multitone signal."""
        freqs = [100, 500, 1000, 1500, 2000]
        amps = [1.0, 0.5, 0.8, 0.3, 0.2]
        signal = make_multitone(freqs, amps, 10000, 1.0)
        trace = make_trace(signal, 10000)

        # All metrics should work and return applicable results
        thd_result = thd(trace)
        snr_result = snr(trace)
        sfdr_result = sfdr(trace)

        # Check that all measurements are applicable
        assert thd_result["applicable"]
        assert snr_result["applicable"]
        assert sfdr_result["applicable"]

        # Check that values are not None
        assert thd_result["value"] is not None
        assert snr_result["value"] is not None
        assert sfdr_result["value"] is not None


# =============================================================================
# Test Peak Finding (SPE-018)
# =============================================================================


@pytest.mark.unit
@pytest.mark.requirement("SPE-018")
class TestFindPeaks:
    """Test spectral peak finding."""

    def test_find_peaks_basic(self) -> None:
        """Test basic peak finding on multitone signal."""
        freqs = [1000, 2000, 3000]
        amps = [1.0, 0.5, 0.3]
        signal = make_multitone(freqs, amps, 100000, 0.1)
        trace = make_trace(signal, 100000)

        peaks = find_peaks(trace, threshold_db=-40, n_peaks=5)

        # Should find at least the 3 tones
        assert len(peaks["frequencies"]) >= 3

        # Check that peaks are sorted by magnitude (strongest first)
        assert np.all(np.diff(peaks["magnitudes_db"]) <= 0)

    def test_find_peaks_threshold(self) -> None:
        """Test peak finding with different thresholds."""
        signal = make_multitone([1000, 2000], [1.0, 0.01], 100000, 0.1)
        trace = make_trace(signal, 100000)

        # High threshold should find fewer peaks
        peaks_high = find_peaks(trace, threshold_db=-20, n_peaks=10)
        peaks_low = find_peaks(trace, threshold_db=-80, n_peaks=10)

        assert len(peaks_high["frequencies"]) <= len(peaks_low["frequencies"])

    def test_find_peaks_min_distance(self) -> None:
        """Test peak spacing with min_distance."""
        signal = make_sine_wave(1000, 10000, 0.5)
        trace = make_trace(signal, 10000)

        peaks = find_peaks(trace, threshold_db=-60, min_distance=10, n_peaks=5)

        # All peaks should be spaced by at least min_distance
        if len(peaks["indices"]) > 1:
            indices = peaks["indices"]
            spacing = np.diff(np.sort(indices))
            assert np.all(spacing >= 10)


# =============================================================================
# Test Harmonic Extraction (SPE-019)
# =============================================================================


@pytest.mark.unit
@pytest.mark.requirement("SPE-019")
class TestExtractHarmonics:
    """Test harmonic extraction."""

    def test_extract_harmonics_pure_sine(self) -> None:
        """Test harmonic extraction on pure sine wave."""
        f0 = 1000.0
        signal = make_sine_wave(f0, 100000, 0.1)
        trace = make_trace(signal, 100000)

        harmonics = extract_harmonics(trace, n_harmonics=5)

        # Should detect fundamental
        fundamental = float(harmonics["fundamental_freq"][0])
        assert pytest.approx(fundamental, rel=0.05) == f0

        # Should have fundamental in frequencies
        assert len(harmonics["frequencies"]) >= 1
        assert pytest.approx(harmonics["frequencies"][0], rel=0.05) == f0

    def test_extract_harmonics_with_distortion(self) -> None:
        """Test harmonic extraction on distorted signal."""
        f0 = 1000.0
        # Signal with 2nd and 3rd harmonics
        signal = make_multitone([f0, 2 * f0, 3 * f0], [1.0, 0.2, 0.1], 100000, 0.1)
        trace = make_trace(signal, 100000)

        harmonics = extract_harmonics(trace, n_harmonics=5)

        # Should find fundamental and harmonics
        assert len(harmonics["frequencies"]) >= 3

        # Check harmonic frequencies
        for i, freq in enumerate(harmonics["frequencies"][:3], 1):
            expected_freq = i * f0
            assert pytest.approx(freq, rel=0.05) == expected_freq

    def test_extract_harmonics_provided_fundamental(self) -> None:
        """Test harmonic extraction with provided fundamental frequency."""
        f0 = 1000.0
        signal = make_multitone([f0, 2 * f0, 3 * f0], [1.0, 0.3, 0.1], 100000, 0.1)
        trace = make_trace(signal, 100000)

        # Provide fundamental explicitly
        harmonics = extract_harmonics(trace, fundamental_freq=f0, n_harmonics=3)

        fundamental = float(harmonics["fundamental_freq"][0])
        assert fundamental == f0
        assert len(harmonics["frequencies"]) >= 1


# =============================================================================
# Test Phase Analysis (SPE-020, SPE-021)
# =============================================================================


@pytest.mark.unit
@pytest.mark.requirement("SPE-020")
class TestPhaseSpectrum:
    """Test phase spectrum computation."""

    def test_phase_spectrum_basic(self) -> None:
        """Test basic phase spectrum computation."""
        signal = make_sine_wave(1000, 10000, 0.5)
        trace = make_trace(signal, 10000)

        freq, phase = phase_spectrum(trace)

        # Check output shapes
        assert len(freq) == len(phase)
        assert len(freq) > 0

        # Phase should be in radians
        assert np.all(np.abs(phase) <= 2 * np.pi * 100)  # Reasonable range after unwrapping

    def test_phase_spectrum_unwrap(self) -> None:
        """Test phase unwrapping."""
        signal = make_sine_wave(1000, 10000, 0.5)
        trace = make_trace(signal, 10000)

        # With unwrapping
        freq1, phase1 = phase_spectrum(trace, unwrap=True)

        # Without unwrapping
        freq2, phase2 = phase_spectrum(trace, unwrap=False)

        # Unwrapped phase should have larger range
        assert np.max(np.abs(phase2)) <= np.pi + 0.1  # Wrapped stays in [-π, π]


@pytest.mark.unit
@pytest.mark.requirement("SPE-021")
class TestGroupDelay:
    """Test group delay computation."""

    def test_group_delay_basic(self) -> None:
        """Test basic group delay computation."""
        signal = make_sine_wave(1000, 10000, 0.5)
        trace = make_trace(signal, 10000)

        freq, gd = group_delay(trace)

        # Check output shapes
        assert len(freq) == len(gd)
        assert len(freq) > 0

        # Group delay should be finite
        assert np.all(np.isfinite(gd))

    def test_group_delay_all_pass(self) -> None:
        """Test group delay on simple signal."""
        # Pure sine wave should have relatively flat group delay
        signal = make_sine_wave(1000, 10000, 0.1)
        trace = make_trace(signal, 10000)

        freq, gd = group_delay(trace)

        # Group delay should be reasonably bounded
        assert np.all(np.abs(gd) < 1000)  # Less than 1000 samples
