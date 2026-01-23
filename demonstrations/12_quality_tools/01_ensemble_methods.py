"""Ensemble Measurement Methods

Demonstrates using multiple measurement methods for the same parameter:
- Multiple measurement approaches for robustness
- Majority voting and weighted averaging
- Outlier rejection algorithms
- Confidence interval estimation

This demonstration shows:
1. How to measure the same parameter using different methods
2. How to combine results using ensemble techniques
3. How to detect and reject outliers
4. How to estimate measurement confidence
5. Practical applications in noisy environments
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

# Add demonstrations to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from typing import TYPE_CHECKING

from demonstrations.common import (
    BaseDemo,
    add_noise,
    generate_sine_wave,
    validate_approximately,
)

if TYPE_CHECKING:
    from oscura.core.types import WaveformTrace


class EnsembleMethodsDemo(BaseDemo):
    """Demonstrate ensemble measurement methods for robust analysis."""

    def __init__(self) -> None:
        """Initialize ensemble methods demonstration."""
        super().__init__(
            name="ensemble_methods",
            description="Use multiple measurement methods for robust parameter estimation",
            capabilities=[
                "oscura.ensemble.majority_voting",
                "oscura.ensemble.weighted_averaging",
                "oscura.ensemble.outlier_rejection",
                "oscura.ensemble.confidence_intervals",
            ],
            related_demos=[
                "02_basic_analysis/01_waveform_measurements.py",
                "12_quality_tools/02_quality_scoring.py",
            ],
        )

    def generate_test_data(self) -> dict[str, Any]:
        """Generate test signals with known parameters."""
        self.info("Creating test signals with varying noise levels...")

        # Clean signal - 1kHz sine, 1V amplitude
        clean = self._create_signal(freq=1000.0, amplitude=1.0, noise_level=0.0)
        self.info("  ✓ Clean signal (SNR=inf)")

        # Low noise signal
        low_noise = self._create_signal(freq=1000.0, amplitude=1.0, noise_level=0.01)
        self.info("  ✓ Low noise signal (SNR≈40dB)")

        # Medium noise signal
        med_noise = self._create_signal(freq=1000.0, amplitude=1.0, noise_level=0.05)
        self.info("  ✓ Medium noise signal (SNR≈26dB)")

        # High noise signal
        high_noise = self._create_signal(freq=1000.0, amplitude=1.0, noise_level=0.1)
        self.info("  ✓ High noise signal (SNR≈20dB)")

        return {
            "clean": clean,
            "low_noise": low_noise,
            "medium_noise": med_noise,
            "high_noise": high_noise,
            "true_frequency": 1000.0,
            "true_amplitude": 1.0,
        }

    def _create_signal(self, freq: float, amplitude: float, noise_level: float) -> WaveformTrace:
        """Create synthetic signal with noise."""
        sample_rate = 100_000.0
        duration = 0.1  # 100ms
        signal = generate_sine_wave(freq, amplitude, sample_rate, duration)

        if noise_level > 0:
            signal = add_noise(signal, noise_level)

        return signal

    def run_demonstration(self, data: dict[str, Any]) -> dict[str, Any]:
        """Demonstrate ensemble measurement methods."""
        results: dict[str, Any] = {}

        # Part 1: Multiple frequency measurement methods
        self.section("Part 1: Multiple Frequency Measurement Methods")
        self.info("Measuring frequency using different approaches...")

        signal = data["medium_noise"]
        freq_measurements = self._measure_frequency_multiple_methods(signal)

        self.info(f"\nFrequency measurements (true={data['true_frequency']:.1f} Hz):")
        for method, freq in freq_measurements.items():
            error = abs(freq - data["true_frequency"])
            self.info(f"  {method:20s}: {freq:8.2f} Hz (error: {error:.2f} Hz)")

        results["frequency_measurements"] = freq_measurements

        # Part 2: Ensemble combination methods
        self.section("Part 2: Ensemble Combination Methods")

        # Simple averaging
        avg_freq = np.mean(list(freq_measurements.values()))
        self.info(f"Simple average: {avg_freq:.2f} Hz")

        # Weighted averaging
        weights = self._calculate_weights(freq_measurements, data["true_frequency"])
        weighted_freq = self._weighted_average(freq_measurements, weights)
        self.info(f"Weighted average: {weighted_freq:.2f} Hz")
        self.info("\nWeights assigned:")
        for method, weight in weights.items():
            self.info(f"  {method:20s}: {weight:.3f}")

        # Median (robust to outliers)
        median_freq = np.median(list(freq_measurements.values()))
        self.info(f"\nMedian: {median_freq:.2f} Hz")

        results["average"] = avg_freq
        results["weighted_average"] = weighted_freq
        results["median"] = median_freq

        # Part 3: Outlier detection and rejection
        self.section("Part 3: Outlier Detection and Rejection")

        # Add an outlier measurement
        freq_with_outlier = freq_measurements.copy()
        freq_with_outlier["corrupted_method"] = 1500.0  # Bad measurement

        self.info("Added corrupted measurement: 1500.00 Hz")

        # Detect outliers using IQR method
        outliers = self._detect_outliers_iqr(freq_with_outlier)
        self.info(f"\nOutliers detected: {outliers}")

        # Reject outliers
        clean_measurements = {k: v for k, v in freq_with_outlier.items() if k not in outliers}
        robust_freq = np.median(list(clean_measurements.values()))
        self.info(f"Robust estimate (outliers removed): {robust_freq:.2f} Hz")

        results["outliers"] = outliers
        results["robust_estimate"] = robust_freq

        # Part 4: Confidence intervals
        self.section("Part 4: Confidence Interval Estimation")

        # Bootstrap confidence interval
        ci_lower, ci_upper = self._bootstrap_confidence_interval(
            list(freq_measurements.values()), confidence=0.95
        )
        self.info(f"95% Confidence Interval: [{ci_lower:.2f}, {ci_upper:.2f}] Hz")
        self.info(
            f"Interval width: {ci_upper - ci_lower:.2f} Hz ({100 * (ci_upper - ci_lower) / weighted_freq:.1f}%)"
        )

        # Standard error
        std_error = np.std(list(freq_measurements.values())) / np.sqrt(len(freq_measurements))
        self.info(f"Standard error: {std_error:.2f} Hz")

        results["confidence_interval"] = (ci_lower, ci_upper)
        results["standard_error"] = std_error

        # Part 5: Noise robustness comparison
        self.section("Part 5: Noise Robustness Comparison")

        self.info("Testing ensemble methods across noise levels...\n")
        noise_results = []

        for noise_name in ["clean", "low_noise", "medium_noise", "high_noise"]:
            signal = data[noise_name]
            measurements = self._measure_frequency_multiple_methods(signal)

            simple_avg = np.mean(list(measurements.values()))
            weights = self._calculate_weights(measurements, data["true_frequency"])
            weighted_avg = self._weighted_average(measurements, weights)
            median_val = np.median(list(measurements.values()))

            simple_err = abs(simple_avg - data["true_frequency"])
            weighted_err = abs(weighted_avg - data["true_frequency"])
            median_err = abs(median_val - data["true_frequency"])

            noise_results.append(
                {
                    "level": noise_name,
                    "simple": simple_err,
                    "weighted": weighted_err,
                    "median": median_err,
                }
            )

            self.info(f"{noise_name:15s}:")
            self.info(f"  Simple avg error:   {simple_err:6.2f} Hz")
            self.info(f"  Weighted avg error: {weighted_err:6.2f} Hz")
            self.info(f"  Median error:       {median_err:6.2f} Hz")
            self.info("")

        results["noise_robustness"] = noise_results

        # Part 6: Majority voting for binary decisions
        self.section("Part 6: Majority Voting for Binary Decisions")

        # Simulate multiple detectors voting on signal presence
        detections = {
            "energy_detector": True,
            "correlation_detector": True,
            "threshold_detector": False,  # False positive
            "matched_filter": True,
            "cyclostationary": True,
        }

        self.info("Detection votes:")
        for detector, vote in detections.items():
            self.info(f"  {detector:25s}: {'SIGNAL' if vote else 'NO SIGNAL'}")

        # Majority vote
        votes = list(detections.values())
        majority = sum(votes) > len(votes) / 2
        confidence = sum(votes) / len(votes)

        self.info(f"\nMajority decision: {'SIGNAL' if majority else 'NO SIGNAL'}")
        self.info(f"Confidence: {confidence:.1%} ({sum(votes)}/{len(votes)} votes)")

        results["majority_vote"] = majority
        results["vote_confidence"] = confidence

        return results

    def _measure_frequency_multiple_methods(self, signal: WaveformTrace) -> dict[str, float]:
        """Measure frequency using multiple methods."""
        data = signal.data
        sample_rate = signal.metadata.sample_rate

        # Method 1: Zero crossing
        zero_crossings = np.where(np.diff(np.sign(data)))[0]
        if len(zero_crossings) > 2:
            period = np.mean(np.diff(zero_crossings)) * 2 / sample_rate
            freq_zc = 1.0 / period if period > 0 else 0.0
        else:
            freq_zc = 0.0

        # Method 2: FFT peak
        fft = np.fft.rfft(data)
        freqs = np.fft.rfftfreq(len(data), 1 / sample_rate)
        peak_idx = np.argmax(np.abs(fft)[1:]) + 1  # Skip DC
        freq_fft = freqs[peak_idx]

        # Method 3: Autocorrelation
        autocorr = np.correlate(data - np.mean(data), data - np.mean(data), mode="full")
        autocorr = autocorr[len(autocorr) // 2 :]
        # Find first peak after zero lag
        peaks = []
        for i in range(1, len(autocorr) - 1):
            if autocorr[i] > autocorr[i - 1] and autocorr[i] > autocorr[i + 1]:
                peaks.append(i)
                if len(peaks) >= 2:
                    break
        if len(peaks) >= 1:
            period = peaks[0] / sample_rate
            freq_autocorr = 1.0 / period if period > 0 else 0.0
        else:
            freq_autocorr = freq_fft

        # Method 4: Peak-to-peak timing
        peaks_idx = []
        for i in range(1, len(data) - 1):
            if data[i] > data[i - 1] and data[i] > data[i + 1] and data[i] > 0:
                peaks_idx.append(i)
        if len(peaks_idx) > 2:
            period = np.mean(np.diff(peaks_idx)) / sample_rate
            freq_peak = 1.0 / period if period > 0 else 0.0
        else:
            freq_peak = freq_fft

        # Method 5: Hilbert transform instantaneous frequency
        analytic = np.fft.ifft(np.concatenate([fft, np.zeros(len(fft) - 2)]))  # Simplified Hilbert
        inst_phase = np.unwrap(np.angle(analytic))
        inst_freq = np.diff(inst_phase) * sample_rate / (2 * np.pi)
        freq_hilbert = np.median(inst_freq)

        return {
            "zero_crossing": freq_zc,
            "fft_peak": freq_fft,
            "autocorrelation": freq_autocorr,
            "peak_timing": freq_peak,
            "hilbert": freq_hilbert,
        }

    def _calculate_weights(
        self, measurements: dict[str, float], true_value: float
    ) -> dict[str, float]:
        """Calculate weights based on inverse error (simplified)."""
        # In real scenarios, weights would be based on method reliability
        # Here we use inverse distance from true value for demonstration
        errors = {k: abs(v - true_value) for k, v in measurements.items()}
        # Avoid division by zero
        errors = {k: max(v, 0.01) for k, v in errors.items()}
        inv_errors = {k: 1.0 / v for k, v in errors.items()}
        total = sum(inv_errors.values())
        weights = {k: v / total for k, v in inv_errors.items()}
        return weights

    def _weighted_average(self, measurements: dict[str, float], weights: dict[str, float]) -> float:
        """Calculate weighted average."""
        return sum(measurements[k] * weights[k] for k in measurements)

    def _detect_outliers_iqr(self, measurements: dict[str, float]) -> list[str]:
        """Detect outliers using Interquartile Range method."""
        values = np.array(list(measurements.values()))
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = [k for k, v in measurements.items() if v < lower_bound or v > upper_bound]
        return outliers

    def _bootstrap_confidence_interval(
        self, data: list[float], confidence: float = 0.95, n_bootstrap: int = 1000
    ) -> tuple[float, float]:
        """Calculate bootstrap confidence interval."""
        bootstrap_means = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_means.append(np.mean(sample))

        alpha = 1 - confidence
        lower = np.percentile(bootstrap_means, 100 * alpha / 2)
        upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
        return lower, upper

    def validate(self, results: dict[str, Any]) -> bool:
        """Validate demonstration results."""
        self.info("Validating ensemble methods...")
        all_valid = True

        # Check frequency measurements exist
        if "frequency_measurements" not in results:
            self.error("Missing frequency measurements")
            return False

        measurements = results["frequency_measurements"]
        if len(measurements) < 3:
            self.error("Insufficient measurement methods")
            all_valid = False

        # Check ensemble results
        for key in ["average", "weighted_average", "median"]:
            if key not in results:
                self.error(f"Missing {key} result")
                all_valid = False
            else:
                # Should be close to 1000 Hz
                if not validate_approximately(results[key], 1000.0, tolerance=100.0):
                    self.error(
                        f"{key} result out of range: {results[key]:.2f} Hz (expected ~1000 Hz)"
                    )
                    all_valid = False

        # Check outlier detection
        if "outliers" in results and "corrupted_method" not in results["outliers"]:
            self.warning("Outlier detection missed corrupted measurement")

        # Check confidence interval
        if "confidence_interval" in results:
            ci_lower, ci_upper = results["confidence_interval"]
            if ci_lower > ci_upper:
                self.error("Invalid confidence interval")
                all_valid = False

        # Check majority vote
        if "majority_vote" in results and results["majority_vote"] is not True:
            self.error("Majority vote incorrect (should be True)")
            all_valid = False

        if all_valid:
            self.success("All ensemble methods validated successfully")

        return all_valid


if __name__ == "__main__":
    demo = EnsembleMethodsDemo()
    success = demo.execute()
    exit(0 if success else 1)
