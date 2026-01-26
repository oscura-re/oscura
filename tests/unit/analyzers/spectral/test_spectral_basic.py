"""Unit tests for spectral analysis functions.

Tests for FFT, PSD, and spectral quality metrics (THD, SNR, SINAD, ENOB).
"""

import numpy as np
import pytest

from oscura.analyzers.waveform.spectral import (
    bartlett_psd,
    enob,
    fft,
    hilbert_transform,
    periodogram,
    psd,
    sfdr,
    sinad,
    snr,
    spectrogram,
    thd,
)
from oscura.core.types import TraceMetadata, WaveformTrace

pytestmark = [pytest.mark.unit, pytest.mark.analyzer, pytest.mark.spectral]


@pytest.fixture
def pure_sine(sample_rate: float) -> WaveformTrace:
    """Generate pure 10 kHz sine wave."""
    duration = 0.01  # 10 ms
    t = np.arange(0, duration, 1 / sample_rate)
    data = np.sin(2 * np.pi * 10000 * t)  # 10 kHz

    return WaveformTrace(
        data=data,
        metadata=TraceMetadata(sample_rate=sample_rate),
    )


@pytest.fixture
def distorted_sine(sample_rate: float) -> WaveformTrace:
    """Generate sine wave with harmonics (distortion)."""
    duration = 0.01
    t = np.arange(0, duration, 1 / sample_rate)

    # Fundamental + harmonics
    fundamental = np.sin(2 * np.pi * 10000 * t)
    second_harmonic = 0.1 * np.sin(2 * np.pi * 20000 * t)
    third_harmonic = 0.05 * np.sin(2 * np.pi * 30000 * t)

    data = fundamental + second_harmonic + third_harmonic

    return WaveformTrace(
        data=data,
        metadata=TraceMetadata(sample_rate=sample_rate),
    )


@pytest.fixture
def noisy_sine(sample_rate: float) -> WaveformTrace:
    """Generate sine wave with noise."""
    duration = 0.01
    t = np.arange(0, duration, 1 / sample_rate)

    rng = np.random.default_rng(42)
    signal = np.sin(2 * np.pi * 10000 * t)
    noise = rng.normal(0, 0.1, len(t))

    data = signal + noise

    return WaveformTrace(
        data=data,
        metadata=TraceMetadata(sample_rate=sample_rate),
    )


class TestFFT:
    """Tests for FFT computation."""

    def test_fft_returns_frequency_and_magnitude(self, pure_sine: WaveformTrace) -> None:
        """FFT should return frequency axis and magnitude in dB."""
        freq, mag = fft(pure_sine)

        assert len(freq) == len(mag)
        assert freq[0] == 0  # DC at index 0
        assert freq[-1] <= pure_sine.metadata.sample_rate / 2

    def test_fft_peak_at_signal_frequency(self, pure_sine: WaveformTrace) -> None:
        """FFT peak should be at the signal frequency."""
        freq, mag = fft(pure_sine)

        peak_idx = np.argmax(mag[1:]) + 1  # Skip DC
        peak_freq = freq[peak_idx]

        # Should be close to 10 kHz
        assert abs(peak_freq - 10000) < 100  # Within 100 Hz

    def test_fft_with_phase(self, pure_sine: WaveformTrace) -> None:
        """FFT with return_phase should return 3 arrays."""
        freq, mag, phase = fft(pure_sine, return_phase=True)

        assert len(freq) == len(mag) == len(phase)
        assert phase.dtype in [np.float64, np.float32]

    def test_fft_windowing(self, pure_sine: WaveformTrace) -> None:
        """Different windows should produce different results."""
        _, mag_hann = fft(pure_sine, window="hann")
        _, mag_rect = fft(pure_sine, window="rectangular")

        # Results should differ
        assert not np.allclose(mag_hann, mag_rect)

    def test_fft_zero_padding(self, pure_sine: WaveformTrace) -> None:
        """Zero padding should increase frequency resolution."""
        freq1, _ = fft(pure_sine, nfft=None)
        freq2, _ = fft(pure_sine, nfft=len(pure_sine.data) * 2)

        # More points with zero padding
        assert len(freq2) > len(freq1)


class TestPSD:
    """Tests for Power Spectral Density estimation."""

    def test_psd_returns_frequency_and_power(self, pure_sine: WaveformTrace) -> None:
        """PSD should return frequency and power density."""
        freq, psd_db = psd(pure_sine)

        assert len(freq) == len(psd_db)
        assert np.all(np.isfinite(psd_db))

    def test_periodogram(self, pure_sine: WaveformTrace) -> None:
        """Periodogram should produce PSD estimate."""
        freq, psd_db = periodogram(pure_sine)

        assert len(freq) > 0
        assert np.all(np.isfinite(psd_db))

    def test_bartlett_psd(self, pure_sine: WaveformTrace) -> None:
        """Bartlett's method should reduce variance."""
        freq, _psd_db = bartlett_psd(pure_sine, n_segments=4)

        assert len(freq) > 0


class TestSpectrogram:
    """Tests for spectrogram computation."""

    def test_spectrogram_returns_3_arrays(self, pure_sine: WaveformTrace) -> None:
        """Spectrogram should return time, frequency, and magnitude."""
        times, freq, Sxx = spectrogram(pure_sine)

        assert len(times) > 0
        assert len(freq) > 0
        assert Sxx.shape == (len(freq), len(times))

    def test_spectrogram_magnitude_in_db(self, pure_sine: WaveformTrace) -> None:
        """Spectrogram magnitude should be in dB."""
        _, _, Sxx = spectrogram(pure_sine)

        # dB values can be negative
        assert np.min(Sxx) < 0 or np.max(Sxx) > 0


class TestTHD:
    """Tests for Total Harmonic Distortion."""

    def test_thd_pure_sine_is_low(self, pure_sine: WaveformTrace) -> None:
        """THD of pure sine should be very low."""
        thd_db = thd(pure_sine)

        # Pure sine should have THD < -40 dB
        assert thd_db < -40

    def test_thd_distorted_is_higher(
        self, pure_sine: WaveformTrace, distorted_sine: WaveformTrace
    ) -> None:
        """THD of distorted signal should be higher."""
        thd_pure = thd(pure_sine)
        thd_distorted = thd(distorted_sine)

        assert thd_distorted > thd_pure

    def test_thd_percentage(self, distorted_sine: WaveformTrace) -> None:
        """THD can be returned as percentage."""
        thd_pct = thd(distorted_sine, return_db=False)

        assert thd_pct > 0
        assert thd_pct < 100  # Should be reasonable


class TestSNR:
    """Tests for Signal-to-Noise Ratio."""

    def test_snr_pure_sine_is_high(self, pure_sine: WaveformTrace) -> None:
        """SNR of pure sine should be very high."""
        snr_db = snr(pure_sine)

        # Pure sine should have high SNR (limited by numerical precision)
        assert snr_db > 60

    def test_snr_noisy_is_lower(self, pure_sine: WaveformTrace, noisy_sine: WaveformTrace) -> None:
        """SNR of noisy signal should be lower."""
        snr_pure = snr(pure_sine)
        snr_noisy = snr(noisy_sine)

        assert snr_noisy < snr_pure


class TestSINAD:
    """Tests for Signal-to-Noise and Distortion."""

    def test_sinad_pure_sine_is_high(self, pure_sine: WaveformTrace) -> None:
        """SINAD of pure sine should be high."""
        sinad_db = sinad(pure_sine)

        assert sinad_db > 50

    def test_sinad_distorted_is_lower(
        self, pure_sine: WaveformTrace, distorted_sine: WaveformTrace
    ) -> None:
        """SINAD with distortion should be lower."""
        sinad_pure = sinad(pure_sine)
        sinad_distorted = sinad(distorted_sine)

        assert sinad_distorted < sinad_pure


class TestENOB:
    """Tests for Effective Number of Bits."""

    def test_enob_pure_sine(self, pure_sine: WaveformTrace) -> None:
        """ENOB should be positive for clean signal."""
        bits = enob(pure_sine)

        assert bits > 0
        assert np.isfinite(bits)

    def test_enob_formula(self, pure_sine: WaveformTrace) -> None:
        """ENOB should follow (SINAD - 1.76) / 6.02 formula."""
        sinad_db = sinad(pure_sine)
        enob_val = enob(pure_sine)

        expected = (sinad_db - 1.76) / 6.02
        assert abs(enob_val - expected) < 0.01


class TestSFDR:
    """Tests for Spurious-Free Dynamic Range."""

    def test_sfdr_pure_sine_is_high(self, pure_sine: WaveformTrace) -> None:
        """SFDR of pure sine should be high."""
        sfdr_db = sfdr(pure_sine)

        assert sfdr_db > 50

    def test_sfdr_with_spur(self, distorted_sine: WaveformTrace) -> None:
        """SFDR with harmonics should be lower."""
        sfdr_db = sfdr(distorted_sine)

        # Should detect the harmonics as spurs
        assert sfdr_db > 0


class TestHilbert:
    """Tests for Hilbert transform."""

    def test_hilbert_returns_envelope(self, pure_sine: WaveformTrace) -> None:
        """Hilbert should return envelope, phase, and inst freq."""
        envelope, phase, inst_freq = hilbert_transform(pure_sine)

        assert len(envelope) == len(pure_sine.data)
        assert len(phase) == len(pure_sine.data)
        assert len(inst_freq) == len(pure_sine.data)

    def test_hilbert_envelope_of_sine_is_constant(self, pure_sine: WaveformTrace) -> None:
        """Envelope of pure sine should be approximately constant."""
        envelope, _, _ = hilbert_transform(pure_sine)

        # Trim edges (edge effects)
        middle = envelope[100:-100]

        # Should be close to 1.0 for unit amplitude sine
        assert abs(np.mean(middle) - 1.0) < 0.1
        assert np.std(middle) < 0.1
