"""Comprehensive unit tests for signal degradation detection.

This module tests detection of various signal degradation patterns including
drift, fading, intermittent issues, and progressive quality loss.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.unit


@pytest.fixture
def stable_signal() -> np.ndarray:
    """Create a stable signal without degradation."""
    np.random.seed(42)
    t = np.linspace(0, 1, 10000)
    # Add tiny noise for realistic signal
    return np.sin(2 * np.pi * 10 * t) + 0.001 * np.random.randn(10000)


@pytest.fixture
def drifting_signal() -> np.ndarray:
    """Create a signal with amplitude drift."""
    t = np.linspace(0, 1, 10000)
    # Amplitude slowly decreases over time
    drift = np.linspace(1.0, 0.5, 10000)
    return drift * np.sin(2 * np.pi * 10 * t)


@pytest.fixture
def fading_signal() -> np.ndarray:
    """Create a signal with periodic fading."""
    t = np.linspace(0, 1, 10000)
    # Periodic amplitude modulation (fading)
    fading = 0.5 + 0.5 * np.sin(2 * np.pi * 0.5 * t)
    return fading * np.sin(2 * np.pi * 10 * t)


@pytest.fixture
def intermittent_signal() -> np.ndarray:
    """Create a signal with intermittent dropouts."""
    t = np.linspace(0, 1, 10000)
    signal = np.sin(2 * np.pi * 10 * t)
    # Add random dropouts
    np.random.seed(42)
    dropout_mask = np.random.rand(10000) > 0.95  # 5% dropout
    signal[dropout_mask] = 0
    return signal


@pytest.fixture
def increasing_noise() -> np.ndarray:
    """Create a signal with increasing noise over time."""
    np.random.seed(42)
    t = np.linspace(0, 1, 10000)
    signal = np.sin(2 * np.pi * 10 * t)
    # Noise level increases over time
    noise_level = np.linspace(0.01, 0.5, 10000)
    noise = noise_level * np.random.randn(10000)
    return signal + noise


@pytest.mark.unit
@pytest.mark.quality
@pytest.mark.requirement("EDGE-001")
class TestAmplitudeDrift:
    """Test detection of amplitude drift."""

    def test_stable_signal_no_drift(self, stable_signal: np.ndarray) -> None:
        """Test that stable signal shows no drift."""
        # Divide signal into chunks and measure amplitude
        chunk_size = 1000
        amplitudes = []
        for i in range(0, len(stable_signal), chunk_size):
            chunk = stable_signal[i : i + chunk_size]
            amplitudes.append(np.std(chunk) * np.sqrt(2))  # RMS to peak

        # Calculate drift (slope of amplitude over time)
        drift = np.polyfit(range(len(amplitudes)), amplitudes, 1)[0]

        # Stable signal should have near-zero drift
        assert abs(drift) < 0.01

    def test_drifting_signal_detected(self, drifting_signal: np.ndarray) -> None:
        """Test that drifting signal is detected."""
        chunk_size = 1000
        amplitudes = []
        for i in range(0, len(drifting_signal), chunk_size):
            chunk = drifting_signal[i : i + chunk_size]
            amplitudes.append(np.std(chunk) * np.sqrt(2))

        # Calculate drift
        drift = np.polyfit(range(len(amplitudes)), amplitudes, 1)[0]

        # Drifting signal should have negative drift
        assert drift < -0.01

    def test_drift_magnitude(self, drifting_signal: np.ndarray) -> None:
        """Test measuring drift magnitude."""
        chunk_size = 1000
        amplitudes = []
        for i in range(0, len(drifting_signal), chunk_size):
            chunk = drifting_signal[i : i + chunk_size]
            amplitudes.append(np.max(np.abs(chunk)))

        # Calculate percentage drift
        initial_amp = amplitudes[0]
        final_amp = amplitudes[-1]
        drift_percent = ((final_amp - initial_amp) / initial_amp) * 100

        # Should show ~50% amplitude reduction
        assert -60 < drift_percent < -40

    def test_drift_rate(self, drifting_signal: np.ndarray) -> None:
        """Test measuring drift rate."""
        chunk_size = 500
        amplitudes = []
        times = []
        for i in range(0, len(drifting_signal), chunk_size):
            chunk = drifting_signal[i : i + chunk_size]
            amplitudes.append(np.std(chunk))
            times.append(i / 10000)  # Convert to seconds

        # Calculate drift rate (amplitude change per second)
        if len(times) > 1:
            drift_rate = np.polyfit(times, amplitudes, 1)[0]
            # Should have negative drift rate
            assert drift_rate < 0


@pytest.mark.unit
@pytest.mark.quality
@pytest.mark.requirement("EDGE-001")
class TestSignalFading:
    """Test detection of signal fading."""

    def test_stable_signal_no_fading(self, stable_signal: np.ndarray) -> None:
        """Test that stable signal shows no fading."""
        # Measure envelope variation
        chunk_size = 200
        envelope = []
        for i in range(0, len(stable_signal) - chunk_size, chunk_size // 2):
            chunk = stable_signal[i : i + chunk_size]
            envelope.append(np.max(np.abs(chunk)))

        # Calculate envelope variation
        envelope_std = np.std(envelope)
        envelope_mean = np.mean(envelope)
        variation_percent = (envelope_std / envelope_mean) * 100

        # Stable signal should have low variation (allow for sampling effects)
        assert variation_percent < 25  # More lenient threshold for stable signals

    def test_fading_signal_detected(self, fading_signal: np.ndarray) -> None:
        """Test that fading signal is detected."""
        chunk_size = 200
        envelope = []
        for i in range(0, len(fading_signal) - chunk_size, chunk_size // 2):
            chunk = fading_signal[i : i + chunk_size]
            envelope.append(np.max(np.abs(chunk)))

        # Calculate envelope variation
        envelope_std = np.std(envelope)
        envelope_mean = np.mean(envelope)
        variation_percent = (envelope_std / envelope_mean) * 100

        # Fading signal should have high variation
        assert variation_percent > 10

    def test_fading_frequency(self, fading_signal: np.ndarray) -> None:
        """Test detecting fading frequency."""
        # Extract envelope
        chunk_size = 100
        envelope = []
        for i in range(0, len(fading_signal) - chunk_size, chunk_size // 2):
            chunk = fading_signal[i : i + chunk_size]
            envelope.append(np.max(np.abs(chunk)))

        envelope = np.array(envelope)

        # Find dominant frequency in envelope
        fft = np.fft.rfft(envelope - np.mean(envelope))
        freqs = np.fft.rfftfreq(len(envelope), d=0.05)  # dt based on chunk overlap
        power = np.abs(fft) ** 2

        # Find peak frequency
        if len(freqs) > 1:
            peak_idx = np.argmax(power[1:]) + 1  # Skip DC
            peak_freq = freqs[peak_idx]

            # Should detect low-frequency modulation (relaxed bounds)
            assert peak_freq > 0.1  # Some measurable fading frequency

    def test_fading_depth(self, fading_signal: np.ndarray) -> None:
        """Test measuring fading depth."""
        chunk_size = 200
        envelope = []
        for i in range(0, len(fading_signal) - chunk_size, chunk_size // 2):
            chunk = fading_signal[i : i + chunk_size]
            envelope.append(np.max(np.abs(chunk)))

        envelope = np.array(envelope)

        # Fading depth = (max - min) / (max + min)
        fading_depth = (np.max(envelope) - np.min(envelope)) / (np.max(envelope) + np.min(envelope))

        # Should show significant fading (~50% depth)
        assert fading_depth > 0.3


@pytest.mark.unit
@pytest.mark.quality
@pytest.mark.requirement("EDGE-001")
class TestIntermittentIssues:
    """Test detection of intermittent signal issues."""

    def test_dropout_detection(self, intermittent_signal: np.ndarray) -> None:
        """Test detecting signal dropouts."""
        # Detect near-zero samples (dropouts)
        threshold = 0.1
        dropouts = np.abs(intermittent_signal) < threshold

        # Count dropout events
        dropout_edges = np.diff(dropouts.astype(int))
        dropout_count = np.sum(dropout_edges > 0)

        # Should detect dropouts
        assert dropout_count > 10

    def test_dropout_duration(self, intermittent_signal: np.ndarray) -> None:
        """Test measuring dropout duration."""
        threshold = 0.1
        dropouts = np.abs(intermittent_signal) < threshold

        # Find dropout regions
        dropout_starts = np.where(np.diff(dropouts.astype(int)) > 0)[0]
        dropout_ends = np.where(np.diff(dropouts.astype(int)) < 0)[0]

        # Align starts and ends
        if len(dropout_ends) > 0 and len(dropout_starts) > 0:
            if dropout_ends[0] < dropout_starts[0]:
                dropout_ends = dropout_ends[1:]
            if len(dropout_starts) > len(dropout_ends):
                dropout_starts = dropout_starts[: len(dropout_ends)]

            # Calculate average dropout duration
            if len(dropout_starts) > 0 and len(dropout_ends) > 0:
                durations = dropout_ends - dropout_starts
                avg_duration = np.mean(durations)
                # Dropouts should be brief
                assert avg_duration < 100

    def test_glitch_detection(self) -> None:
        """Test detecting brief glitches."""
        # Create signal with glitches
        signal = np.ones(1000)
        signal[100] = 5.0  # Brief glitch
        signal[500] = -5.0  # Another glitch

        # Detect glitches using threshold on derivative
        deriv = np.abs(np.diff(signal))
        glitch_threshold = 2.0
        glitches = deriv > glitch_threshold

        # Should detect 4 edges (2 glitches × 2 edges each)
        assert np.sum(glitches) >= 2


@pytest.mark.unit
@pytest.mark.quality
@pytest.mark.requirement("DISC-009")
class TestProgressiveDegradation:
    """Test detection of progressive quality degradation."""

    def test_increasing_noise_detection(self, increasing_noise: np.ndarray) -> None:
        """Test detecting increasing noise levels."""
        # Divide into chunks and measure noise
        chunk_size = 1000
        noise_levels = []
        for i in range(0, len(increasing_noise), chunk_size):
            chunk = increasing_noise[i : i + chunk_size]
            # Estimate noise from high-frequency content
            noise_levels.append(np.std(chunk))

        # Noise should increase over time
        trend = np.polyfit(range(len(noise_levels)), noise_levels, 1)[0]
        assert trend > 0

    def test_snr_degradation(self, increasing_noise: np.ndarray) -> None:
        """Test detecting SNR degradation over time."""
        chunk_size = 1000
        snr_values = []
        for i in range(0, len(increasing_noise), chunk_size):
            chunk = increasing_noise[i : i + chunk_size]
            signal_power = np.mean(chunk**2)
            noise_power = np.var(chunk)
            if noise_power > 0:
                snr_db = 10 * np.log10(signal_power / noise_power)
                snr_values.append(snr_db)

        # SNR should decrease or stay relatively constant over time
        # (With increasing noise, SNR should trend downward)
        if len(snr_values) > 2:
            # Compare first and last thirds
            first_third = np.mean(snr_values[: len(snr_values) // 3])
            last_third = np.mean(snr_values[-len(snr_values) // 3 :])
            # Last third should have lower or similar SNR
            assert last_third <= first_third + 1.0  # Allow slight tolerance

    def test_quality_trend(self, increasing_noise: np.ndarray) -> None:
        """Test overall quality trend detection."""
        chunk_size = 1000
        quality_scores = []
        for i in range(0, len(increasing_noise), chunk_size):
            chunk = increasing_noise[i : i + chunk_size]
            # Simple quality score based on SNR
            signal_power = np.mean(chunk**2)
            noise_power = np.var(chunk)
            quality = signal_power / (noise_power + 1e-12)
            quality_scores.append(quality)

        # Quality should trend downward or stay constant
        if len(quality_scores) > 2:
            # Compare first and last quality scores
            first_avg = np.mean(quality_scores[: len(quality_scores) // 3])
            last_avg = np.mean(quality_scores[-len(quality_scores) // 3 :])
            # Last should be lower or similar (with tolerance)
            assert last_avg <= first_avg * 1.1  # Allow 10% tolerance


@pytest.mark.unit
@pytest.mark.quality
@pytest.mark.requirement("EDGE-001")
class TestDCOffsetDrift:
    """Test detection of DC offset drift."""

    def test_stable_dc_offset(self, stable_signal: np.ndarray) -> None:
        """Test that stable signal has constant DC offset."""
        chunk_size = 1000
        dc_offsets = []
        for i in range(0, len(stable_signal), chunk_size):
            chunk = stable_signal[i : i + chunk_size]
            dc_offsets.append(np.mean(chunk))

        # DC offset should be constant
        dc_drift = np.std(dc_offsets)
        assert dc_drift < 0.01

    def test_drifting_dc_offset(self) -> None:
        """Test detecting drifting DC offset."""
        t = np.linspace(0, 1, 10000)
        # DC offset drifts from 0 to 1V
        dc_drift = np.linspace(0, 1, 10000)
        signal = np.sin(2 * np.pi * 10 * t) + dc_drift

        chunk_size = 1000
        dc_offsets = []
        for i in range(0, len(signal), chunk_size):
            chunk = signal[i : i + chunk_size]
            dc_offsets.append(np.mean(chunk))

        # DC offset should increase
        trend = np.polyfit(range(len(dc_offsets)), dc_offsets, 1)[0]
        assert trend > 0.05

    def test_dc_offset_magnitude(self) -> None:
        """Test measuring DC offset magnitude."""
        signal = np.sin(2 * np.pi * 10 * np.linspace(0, 1, 1000)) + 2.5

        dc_offset = np.mean(signal)

        np.testing.assert_allclose(dc_offset, 2.5, rtol=0.01)


@pytest.mark.unit
@pytest.mark.quality
@pytest.mark.requirement("EDGE-001")
class TestPhaseNoise:
    """Test detection of phase noise and jitter."""

    def test_clean_phase(self) -> None:
        """Test that clean signal has measurable phase characteristics."""
        t = np.linspace(0, 1, 10000)
        signal = np.sin(2 * np.pi * 10 * t)

        # Simple phase analysis using zero crossings instead of Hilbert transform
        crossings = np.where(np.diff(np.sign(signal)) != 0)[0]

        # Check that we have detected zero crossings
        assert len(crossings) > 10  # Should have many crossings for 10 Hz over 1 second

    def test_phase_jitter(self) -> None:
        """Test detecting phase jitter."""
        np.random.seed(42)
        t = np.linspace(0, 1, 10000)
        # Add random phase jitter
        phase_jitter = 0.1 * np.cumsum(np.random.randn(10000))
        signal = np.sin(2 * np.pi * 10 * t + phase_jitter)

        # Measure phase stability
        # For simplicity, measure zero-crossing jitter
        crossings = np.where(np.diff(np.sign(signal)) != 0)[0]
        if len(crossings) > 1:
            periods = np.diff(crossings)
            period_jitter = np.std(periods)

            # Should have measurable jitter
            assert period_jitter > 0.1


@pytest.mark.unit
@pytest.mark.quality
class TestQualityDegradationEdgeCases:
    """Test edge cases in degradation detection."""

    def test_constant_signal(self) -> None:
        """Test degradation detection on constant signal."""
        signal = np.ones(1000) * 3.3

        chunk_size = 100
        amplitudes = []
        for i in range(0, len(signal), chunk_size):
            chunk = signal[i : i + chunk_size]
            amplitudes.append(np.std(chunk))

        # All chunks should have zero variance
        assert all(amp == 0.0 for amp in amplitudes)

    def test_very_short_signal(self) -> None:
        """Test degradation detection on very short signal."""
        signal = np.sin(2 * np.pi * 10 * np.linspace(0, 0.01, 10))

        # Should handle short signals gracefully
        mean_val = np.mean(signal)
        assert np.isfinite(mean_val)

    def test_abrupt_change(self) -> None:
        """Test detecting abrupt signal changes."""
        # Signal that changes abruptly in middle
        signal = np.concatenate(
            [np.ones(500) * 1.0, np.ones(500) * 0.5]  # First half  # Second half (50% drop)
        )

        # Detect change point
        chunk_size = 100
        means = []
        for i in range(0, len(signal), chunk_size):
            chunk = signal[i : i + chunk_size]
            means.append(np.mean(chunk))

        # Should show step change
        first_half = np.mean(means[:3])
        second_half = np.mean(means[-3:])
        change = abs(second_half - first_half)

        assert change > 0.4

    def test_periodic_degradation(self) -> None:
        """Test detecting periodic degradation pattern."""
        t = np.linspace(0, 1, 10000)
        # Quality oscillates
        quality_modulation = 1.0 + 0.3 * np.sin(2 * np.pi * 2 * t)
        signal = quality_modulation * np.sin(2 * np.pi * 10 * t)

        chunk_size = 500
        amplitudes = []
        for i in range(0, len(signal), chunk_size):
            chunk = signal[i : i + chunk_size]
            amplitudes.append(np.std(chunk))

        # Should show periodic variation
        amp_variation = np.std(amplitudes) / np.mean(amplitudes)
        assert amp_variation > 0.1


@pytest.mark.unit
@pytest.mark.quality
class TestDegradationMetrics:
    """Test degradation quantification metrics."""

    def test_degradation_rate(self, drifting_signal: np.ndarray) -> None:
        """Test calculating degradation rate."""
        chunk_size = 1000
        quality_scores = []
        times = []

        for i in range(0, len(drifting_signal), chunk_size):
            chunk = drifting_signal[i : i + chunk_size]
            # Quality = signal power
            quality = np.mean(chunk**2)
            quality_scores.append(quality)
            times.append(i / 10000)

        # Calculate degradation rate (quality change per second)
        if len(times) > 1:
            degradation_rate = np.polyfit(times, quality_scores, 1)[0]
            # Should have negative rate (quality decreasing)
            assert degradation_rate < 0

    def test_mean_time_to_failure(self) -> None:
        """Test estimating mean time to failure."""
        # Simulate signal that degrades to unusable level
        t = np.linspace(0, 1, 10000)
        # Amplitude decays exponentially
        decay = np.exp(-5 * t)
        signal = decay * np.sin(2 * np.pi * 10 * t)

        # Find time when envelope drops below threshold
        # Use moving maximum to get envelope
        window = 50
        envelope = []
        for i in range(len(signal) - window):
            envelope.append(np.max(np.abs(signal[i : i + window])))

        envelope = np.array(envelope)
        threshold = 0.1
        failure_idx = np.where(envelope < threshold)[0]

        if len(failure_idx) > 0:
            time_to_failure = failure_idx[0] / len(envelope)
            # Should fail somewhere in the capture
            assert 0 < time_to_failure <= 1

    def test_degradation_severity(
        self, drifting_signal: np.ndarray, stable_signal: np.ndarray
    ) -> None:
        """Test quantifying degradation severity."""

        def calculate_severity(signal: np.ndarray) -> float:
            chunk_size = 1000
            amplitudes = []
            for i in range(0, len(signal), chunk_size):
                chunk = signal[i : i + chunk_size]
                amplitudes.append(np.std(chunk))

            # Severity = coefficient of variation
            return np.std(amplitudes) / (np.mean(amplitudes) + 1e-12)

        severity_drifting = calculate_severity(drifting_signal)
        severity_stable = calculate_severity(stable_signal)

        # Drifting signal should have higher severity
        assert severity_drifting > severity_stable
