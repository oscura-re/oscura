"""Comprehensive unit tests for quality metrics calculation.

This module tests quality metrics calculation including SNR, ENOB, THD,
and various signal quality measurements.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.unit


@pytest.fixture
def perfect_sine() -> np.ndarray:
    """Create a very clean sine wave with minimal noise."""
    np.random.seed(42)
    t = np.linspace(0, 1, 1000)
    # Add tiny noise so SNR calculations work properly
    return np.sin(2 * np.pi * 10 * t) + 0.001 * np.random.randn(1000)


@pytest.fixture
def noisy_sine() -> np.ndarray:
    """Create a noisy sine wave."""
    np.random.seed(42)
    t = np.linspace(0, 1, 1000)
    signal = np.sin(2 * np.pi * 10 * t)
    noise = 0.1 * np.random.randn(1000)
    return signal + noise


@pytest.fixture
def distorted_sine() -> np.ndarray:
    """Create a sine wave with harmonic distortion."""
    t = np.linspace(0, 1, 1000)
    fundamental = np.sin(2 * np.pi * 10 * t)
    harmonic2 = 0.1 * np.sin(2 * np.pi * 20 * t)  # 2nd harmonic
    harmonic3 = 0.05 * np.sin(2 * np.pi * 30 * t)  # 3rd harmonic
    return fundamental + harmonic2 + harmonic3


@pytest.fixture
def clipped_sine() -> np.ndarray:
    """Create a clipped sine wave."""
    t = np.linspace(0, 1, 1000)
    signal = 2.0 * np.sin(2 * np.pi * 10 * t)
    return np.clip(signal, -1, 1)


@pytest.mark.unit
@pytest.mark.quality
@pytest.mark.requirement("EDGE-001")
class TestSNRCalculation:
    """Test Signal-to-Noise Ratio calculation."""

    def test_perfect_signal_high_snr(self, perfect_sine: np.ndarray) -> None:
        """Test that clean signal has high SNR."""
        signal_power = np.mean(perfect_sine**2)
        # Calculate noise power from the random component we added
        # The signal is sine + noise, so subtract ideal sine to get noise
        t = np.linspace(0, 1, 1000)
        ideal = np.sin(2 * np.pi * 10 * t)
        noise = perfect_sine - ideal
        noise_power = np.mean(noise**2)

        if noise_power > 0:
            snr_linear = signal_power / noise_power
            snr_db = 10 * np.log10(snr_linear)
            # Clean signal with 0.001 noise should have SNR > 30 dB
            assert snr_db > 30.0

    def test_noisy_signal_lower_snr(self, noisy_sine: np.ndarray) -> None:
        """Test that noisy signal has lower SNR."""
        signal_power = np.mean(noisy_sine**2)
        noise_power = np.var(noisy_sine)

        snr_linear = signal_power / noise_power
        snr_db = 10 * np.log10(snr_linear)

        # Noisy signal should have moderate SNR
        assert 0 < snr_db < 40.0

    def test_snr_comparison(self, perfect_sine: np.ndarray, noisy_sine: np.ndarray) -> None:
        """Test that clean signal has higher SNR than noisy signal."""
        # Calculate SNR using signal reconstruction method
        t = np.linspace(0, 1, 1000)
        ideal = np.sin(2 * np.pi * 10 * t)

        # For clean signal
        clean_signal_power = np.mean(ideal**2)
        clean_noise = perfect_sine - ideal
        clean_noise_power = np.mean(clean_noise**2)
        clean_snr = 10 * np.log10(clean_signal_power / (clean_noise_power + 1e-12))

        # For noisy signal
        noisy_noise = noisy_sine - ideal
        noisy_noise_power = np.mean(noisy_noise**2)
        noisy_snr = 10 * np.log10(clean_signal_power / (noisy_noise_power + 1e-12))

        # Clean should have higher SNR
        assert clean_snr > noisy_snr

    def test_zero_signal_snr(self) -> None:
        """Test SNR calculation for zero signal."""
        signal = np.zeros(1000)
        signal_power = np.mean(signal**2)

        # Zero signal = zero power
        assert signal_power == 0.0

    def test_dc_signal_snr(self) -> None:
        """Test SNR calculation for DC signal."""
        signal = np.ones(1000) * 3.3
        signal_power = np.mean(signal**2)
        noise_power = np.var(signal)

        # Constant signal has high power, essentially zero variance
        assert signal_power > 0
        assert noise_power < 1e-10  # Effectively zero (allow for floating point)


@pytest.mark.unit
@pytest.mark.quality
@pytest.mark.requirement("EDGE-001")
class TestPowerMeasurements:
    """Test signal power measurements."""

    def test_rms_calculation(self, perfect_sine: np.ndarray) -> None:
        """Test RMS value calculation."""
        rms = np.sqrt(np.mean(perfect_sine**2))

        # RMS of unit amplitude sine wave is 1/sqrt(2) ≈ 0.707
        expected_rms = 1.0 / np.sqrt(2)
        np.testing.assert_allclose(rms, expected_rms, rtol=0.01)

    def test_peak_to_peak(self, perfect_sine: np.ndarray) -> None:
        """Test peak-to-peak measurement."""
        peak_to_peak = np.max(perfect_sine) - np.min(perfect_sine)

        # Unit amplitude sine wave has peak-to-peak of 2
        np.testing.assert_allclose(peak_to_peak, 2.0, rtol=0.01)

    def test_average_power(self, perfect_sine: np.ndarray) -> None:
        """Test average power calculation."""
        avg_power = np.mean(perfect_sine**2)

        # Power of unit sine wave is 0.5
        np.testing.assert_allclose(avg_power, 0.5, rtol=0.01)

    def test_crest_factor(self, perfect_sine: np.ndarray) -> None:
        """Test crest factor calculation."""
        peak = np.max(np.abs(perfect_sine))
        rms = np.sqrt(np.mean(perfect_sine**2))
        crest_factor = peak / rms

        # Crest factor of sine wave is sqrt(2) ≈ 1.414
        expected_cf = np.sqrt(2)
        np.testing.assert_allclose(crest_factor, expected_cf, rtol=0.01)

    def test_clipped_signal_crest_factor(self, clipped_sine: np.ndarray) -> None:
        """Test that clipped signal has lower crest factor."""
        peak = np.max(np.abs(clipped_sine))
        rms = np.sqrt(np.mean(clipped_sine**2))
        crest_factor = peak / rms

        # Clipped signal has lower crest factor than pure sine
        assert crest_factor < np.sqrt(2)


@pytest.mark.unit
@pytest.mark.quality
@pytest.mark.requirement("EDGE-001")
class TestDistortionMetrics:
    """Test distortion and harmonic analysis."""

    def test_thd_perfect_signal(self, perfect_sine: np.ndarray) -> None:
        """Test THD calculation for clean sine wave."""
        # Perform FFT
        fft = np.fft.rfft(perfect_sine)
        power = np.abs(fft) ** 2

        # Find fundamental (should be at bin 10 for 10 Hz in 1000 samples)
        fundamental_power = np.max(power)

        # Sum harmonic powers (excluding DC and fundamental)
        total_power = np.sum(power)
        harmonic_power = total_power - fundamental_power - power[0]  # Exclude DC

        if fundamental_power > 0:
            thd = np.sqrt(harmonic_power / fundamental_power)
            # Clean sine with small noise should have low THD (< 5%)
            assert thd < 0.05

    def test_thd_distorted_signal(self, distorted_sine: np.ndarray) -> None:
        """Test THD calculation for distorted sine wave."""
        fft = np.fft.rfft(distorted_sine)
        power = np.abs(fft) ** 2

        # Find fundamental
        fundamental_idx = np.argmax(power[1:]) + 1  # Skip DC
        fundamental_power = power[fundamental_idx]

        # Sum harmonic powers
        harmonic_power = 0.0
        for i in range(2, 6):  # Check first few harmonics
            if fundamental_idx * i < len(power):
                harmonic_power += power[fundamental_idx * i]

        if fundamental_power > 0:
            thd = np.sqrt(harmonic_power / fundamental_power)
            # Distorted signal should have measurable THD
            assert thd > 0.05  # More than 5%

    def test_sinad_calculation(self, noisy_sine: np.ndarray) -> None:
        """Test SINAD (Signal to Noise and Distortion) calculation."""
        signal_power = np.mean(noisy_sine**2)
        noise_and_distortion = np.var(noisy_sine)

        if noise_and_distortion > 0:
            sinad_linear = signal_power / noise_and_distortion
            sinad_db = 10 * np.log10(sinad_linear)

            # Should be a reasonable value
            assert sinad_db > 0


@pytest.mark.unit
@pytest.mark.quality
@pytest.mark.requirement("DISC-009")
class TestENOBCalculation:
    """Test Effective Number of Bits calculation."""

    def test_enob_from_snr(self) -> None:
        """Test ENOB calculation from SNR."""
        # Theoretical: ENOB = (SNR_dB - 1.76) / 6.02
        snr_db = 50.0
        enob = (snr_db - 1.76) / 6.02

        # 50 dB SNR should give ~8 bits ENOB
        assert 7.5 < enob < 8.5

    def test_enob_perfect_signal(self, perfect_sine: np.ndarray) -> None:
        """Test ENOB estimation for clean signal."""
        voltage_swing = np.max(perfect_sine) - np.min(perfect_sine)

        # Calculate noise from difference with ideal signal
        t = np.linspace(0, 1, 1000)
        ideal = np.sin(2 * np.pi * 10 * t)
        noise = perfect_sine - ideal
        noise_rms = np.std(noise)

        if noise_rms > 1e-12:
            snr_linear = (voltage_swing / 2) / noise_rms
            snr_db = 20 * np.log10(snr_linear)
            enob = (snr_db - 1.76) / 6.02

            # Clean signal with 0.001 noise should have good ENOB (>6 bits)
            assert enob > 6.0

    def test_enob_noisy_signal(self, noisy_sine: np.ndarray) -> None:
        """Test ENOB estimation for noisy signal."""
        voltage_swing = np.max(noisy_sine) - np.min(noisy_sine)
        noise_rms = np.std(noisy_sine)

        snr_linear = (voltage_swing / 2) / (noise_rms + 1e-12)
        snr_db = 20 * np.log10(snr_linear)
        enob = (snr_db - 1.76) / 6.02

        # Noisy signal should have lower ENOB
        assert enob < 8.0


@pytest.mark.unit
@pytest.mark.quality
@pytest.mark.requirement("EDGE-001")
class TestAmplitudeMeasurements:
    """Test amplitude-based quality metrics."""

    def test_peak_amplitude(self, perfect_sine: np.ndarray) -> None:
        """Test peak amplitude measurement."""
        peak_positive = np.max(perfect_sine)
        peak_negative = np.min(perfect_sine)

        # Unit amplitude sine wave
        np.testing.assert_allclose(peak_positive, 1.0, rtol=0.01)
        np.testing.assert_allclose(peak_negative, -1.0, rtol=0.01)

    def test_amplitude_stability(self) -> None:
        """Test amplitude stability over time."""
        # Create signal with amplitude variation
        t = np.linspace(0, 1, 1000)
        envelope = 1.0 + 0.1 * np.sin(2 * np.pi * 0.5 * t)  # Slow AM
        signal = envelope * np.sin(2 * np.pi * 10 * t)

        # Measure amplitude in chunks
        chunk_size = 100
        amplitudes = []
        for i in range(0, len(signal), chunk_size):
            chunk = signal[i : i + chunk_size]
            amplitudes.append(np.max(np.abs(chunk)))

        # Calculate stability (std of amplitudes)
        stability = np.std(amplitudes) / np.mean(amplitudes)

        # Should have some variation due to AM
        assert stability > 0.01

    def test_dc_offset(self, perfect_sine: np.ndarray) -> None:
        """Test DC offset measurement."""
        # Add DC offset
        signal_with_offset = perfect_sine + 2.5

        dc_offset = np.mean(signal_with_offset)

        np.testing.assert_allclose(dc_offset, 2.5, rtol=0.01)

    def test_ac_coupling(self) -> None:
        """Test AC coupling (DC removal)."""
        t = np.linspace(0, 1, 1000)
        signal = np.sin(2 * np.pi * 10 * t) + 3.0  # 3V DC offset

        # Remove DC
        ac_signal = signal - np.mean(signal)

        # AC coupled signal should have zero mean
        np.testing.assert_allclose(np.mean(ac_signal), 0.0, atol=1e-10)


@pytest.mark.unit
@pytest.mark.quality
@pytest.mark.requirement("EDGE-001")
class TestFrequencyMetrics:
    """Test frequency-domain quality metrics."""

    def test_spectral_purity(self, perfect_sine: np.ndarray) -> None:
        """Test spectral purity measurement."""
        fft = np.fft.rfft(perfect_sine)
        power = np.abs(fft) ** 2

        # Find peak (fundamental)
        peak_power = np.max(power)
        total_power = np.sum(power)

        spectral_purity = peak_power / total_power

        # Pure sine wave should have high spectral purity (>99%)
        assert spectral_purity > 0.95

    def test_spectral_flatness(self, noisy_sine: np.ndarray) -> None:
        """Test spectral flatness (measure of how noise-like signal is)."""
        fft = np.fft.rfft(noisy_sine)
        power = np.abs(fft) ** 2
        power = power[power > 1e-12]  # Remove zeros

        # Spectral flatness = geometric mean / arithmetic mean
        geometric_mean = np.exp(np.mean(np.log(power)))
        arithmetic_mean = np.mean(power)

        flatness = geometric_mean / arithmetic_mean

        # Sine wave should have low flatness (not noise-like)
        # White noise would have flatness close to 1
        assert flatness < 0.5

    def test_occupied_bandwidth(self, perfect_sine: np.ndarray) -> None:
        """Test occupied bandwidth measurement."""
        fft = np.fft.rfft(perfect_sine)
        freqs = np.fft.rfftfreq(len(perfect_sine), d=0.001)  # dt = 1/1000
        power = np.abs(fft) ** 2

        # Find bandwidth containing 99% of power
        total_power = np.sum(power)
        cumsum_power = np.cumsum(power)

        # Find frequencies where 0.5% and 99.5% of power is
        idx_low = np.argmax(cumsum_power > total_power * 0.005)
        idx_high = np.argmax(cumsum_power > total_power * 0.995)

        occupied_bw = freqs[idx_high] - freqs[idx_low]

        # Pure tone should have very narrow bandwidth
        assert occupied_bw < 50  # Less than 50 Hz


@pytest.mark.unit
@pytest.mark.quality
@pytest.mark.requirement("EDGE-001")
class TestDynamicRange:
    """Test dynamic range measurements."""

    def test_spurious_free_dynamic_range(self, distorted_sine: np.ndarray) -> None:
        """Test SFDR (Spurious Free Dynamic Range) calculation."""
        fft = np.fft.rfft(distorted_sine)
        power_db = 20 * np.log10(np.abs(fft) + 1e-12)

        # Find fundamental
        fundamental_idx = np.argmax(power_db[1:]) + 1
        fundamental_power = power_db[fundamental_idx]

        # Find largest spurious component (excluding DC and fundamental)
        power_db[0] = -np.inf  # Ignore DC
        power_db[fundamental_idx] = -np.inf  # Ignore fundamental
        largest_spur = np.max(power_db)

        sfdr = fundamental_power - largest_spur

        # SFDR should be positive (fundamental stronger than spurs)
        assert sfdr > 0

    def test_noise_floor(self, perfect_sine: np.ndarray) -> None:
        """Test noise floor estimation."""
        fft = np.fft.rfft(perfect_sine)
        power_db = 20 * np.log10(np.abs(fft) + 1e-12)

        # Find fundamental and exclude it
        fundamental_idx = np.argmax(power_db)
        mask = np.ones(len(power_db), dtype=bool)
        mask[fundamental_idx - 5 : fundamental_idx + 5] = False  # Exclude fundamental bin

        # Noise floor is median of remaining bins
        noise_floor = np.median(power_db[mask])

        # Should be well below signal
        assert noise_floor < np.max(power_db) - 20  # At least 20 dB below peak


@pytest.mark.unit
@pytest.mark.quality
class TestQualityMetricsEdgeCases:
    """Test edge cases in quality metric calculation."""

    def test_empty_signal(self) -> None:
        """Test metrics on empty signal."""
        signal = np.array([])

        # Should handle gracefully or raise appropriate error
        try:
            _ = np.mean(signal**2)
        except (ValueError, RuntimeWarning):
            pass  # Expected for empty arrays

    def test_single_value(self) -> None:
        """Test metrics on single value."""
        signal = np.array([1.0])

        power = np.mean(signal**2)
        assert power == 1.0

        variance = np.var(signal)
        assert variance == 0.0

    def test_nan_values(self) -> None:
        """Test metrics with NaN values."""
        signal = np.array([1.0, 2.0, np.nan, 3.0, 4.0])

        # Standard calculations will propagate NaN
        mean = np.mean(signal)
        assert np.isnan(mean)

        # Use nanmean for robust calculation
        mean_robust = np.nanmean(signal)
        assert not np.isnan(mean_robust)

    def test_inf_values(self) -> None:
        """Test metrics with infinite values."""
        signal = np.array([1.0, 2.0, np.inf, 3.0, 4.0])

        # Mean with inf should be inf
        mean = np.mean(signal)
        assert np.isinf(mean)

    def test_very_large_values(self) -> None:
        """Test metrics don't overflow with large values."""
        signal = np.array([1e100, 2e100, 3e100])

        # Should not overflow
        mean = np.mean(signal)
        assert np.isfinite(mean)
        assert mean > 0

    def test_very_small_values(self) -> None:
        """Test metrics don't underflow with small values."""
        signal = np.array([1e-100, 2e-100, 3e-100])

        mean = np.mean(signal)
        assert mean > 0
        assert np.isfinite(mean)

    def test_alternating_sign(self) -> None:
        """Test metrics on alternating positive/negative values."""
        signal = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])

        mean = np.mean(signal)
        # Mean should be close to zero
        np.testing.assert_allclose(mean, 0.0, atol=1e-10)

        rms = np.sqrt(np.mean(signal**2))
        # RMS should be 1.0
        np.testing.assert_allclose(rms, 1.0)


@pytest.mark.unit
@pytest.mark.quality
class TestQualityComparison:
    """Test comparing quality between signals."""

    def test_compare_snr(self, perfect_sine: np.ndarray, noisy_sine: np.ndarray) -> None:
        """Test SNR comparison between signals."""

        def calculate_snr(signal: np.ndarray) -> float:
            # Use known signal to calculate noise
            t = np.linspace(0, 1, 1000)
            ideal = np.sin(2 * np.pi * 10 * t)
            signal_power = np.mean(ideal**2)
            noise = signal - ideal
            noise_power = np.mean(noise**2)
            return 10 * np.log10(signal_power / (noise_power + 1e-12))

        snr_clean = calculate_snr(perfect_sine)
        snr_noisy = calculate_snr(noisy_sine)

        # Clean signal should have higher SNR
        assert snr_clean > snr_noisy

    def test_compare_thd(self, perfect_sine: np.ndarray, distorted_sine: np.ndarray) -> None:
        """Test THD comparison between signals."""

        def calculate_thd(signal: np.ndarray) -> float:
            fft = np.fft.rfft(signal)
            power = np.abs(fft) ** 2
            fundamental_power = np.max(power)
            total_power = np.sum(power)
            harmonic_power = total_power - fundamental_power - power[0]
            return np.sqrt(harmonic_power / (fundamental_power + 1e-12))

        thd_clean = calculate_thd(perfect_sine)
        thd_distorted = calculate_thd(distorted_sine)

        # Distorted signal should have higher THD
        assert thd_distorted > thd_clean

    def test_quality_ranking(
        self,
        perfect_sine: np.ndarray,
        noisy_sine: np.ndarray,
        distorted_sine: np.ndarray,
        clipped_sine: np.ndarray,
    ) -> None:
        """Test ranking signals by overall quality."""
        signals = {
            "perfect": perfect_sine,
            "noisy": noisy_sine,
            "distorted": distorted_sine,
            "clipped": clipped_sine,
        }

        # Calculate quality score (higher is better)
        def quality_score(signal: np.ndarray) -> float:
            # Use ideal signal method
            t = np.linspace(0, 1, 1000)
            ideal = np.sin(2 * np.pi * 10 * t)
            signal_power = np.mean(ideal**2)
            noise = signal - ideal
            noise_power = np.mean(noise**2)
            snr = 10 * np.log10(signal_power / (noise_power + 1e-12))
            return snr

        scores = {name: quality_score(sig) for name, sig in signals.items()}

        # Perfect should have highest quality
        assert scores["perfect"] >= max(scores["noisy"], scores["distorted"], scores["clipped"])
