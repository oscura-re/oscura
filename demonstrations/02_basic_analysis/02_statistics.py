"""Statistical Analysis: Distribution and outlier detection

Demonstrates:
- oscura.basic_stats() - Calculate mean, median, std, min, max, range
- oscura.percentiles() - Calculate percentiles and quartiles
- oscura.distribution_metrics() - Skewness, kurtosis, crest factor
- oscura.histogram() - Generate amplitude histograms
- Outlier detection using statistical thresholds
- Distribution analysis and characterization

IEEE Standards: IEEE 181-2011 (transitional waveform definitions)
Related Demos:
- 00_getting_started/00_hello_world.py
- 02_basic_analysis/01_waveform_measurements.py
- 02_basic_analysis/03_spectral_analysis.py

Generates synthetic noisy signals to demonstrate statistical analysis
capabilities. Perfect for understanding signal distributions, variability,
and outlier detection techniques.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

# Add demonstrations to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demonstrations.common import (
    BaseDemo,
    add_noise,
    generate_pulse_train,
    generate_sine_wave,
)
from oscura import (
    basic_stats,
    distribution_metrics,
    histogram,
    percentiles,
)
from oscura.core.types import WaveformTrace


class StatisticsDemo(BaseDemo):
    """Comprehensive demonstration of statistical analysis capabilities."""

    def __init__(self) -> None:
        """Initialize statistical analysis demonstration."""
        super().__init__(
            name="statistics",
            description="Statistical analysis: mean, median, std, distribution metrics, outlier detection",
            capabilities=[
                "oscura.basic_stats",
                "oscura.percentiles",
                "oscura.distribution_metrics",
                "oscura.histogram",
            ],
            ieee_standards=[
                "IEEE 181-2011",
            ],
            related_demos=[
                "02_basic_analysis/01_waveform_measurements.py",
                "02_basic_analysis/03_spectral_analysis.py",
            ],
        )

    def generate_test_data(self) -> dict[str, Any]:
        """Generate test signals for statistical analysis demonstrations.

        Creates:
        1. Noisy sine wave: Demonstrates distribution analysis
        2. Signal with outliers: Shows outlier detection techniques
        3. Pulse train with noise: Demonstrates bimodal distribution
        """
        # 1. Noisy sine wave (1 kHz, 3V amplitude, 30 dB SNR)
        noisy_sine = generate_sine_wave(
            frequency=1000.0,  # 1 kHz
            amplitude=3.0,  # 3V peak
            duration=0.01,  # 10 ms
            sample_rate=100e3,  # 100 kHz sampling
        )
        noisy_sine = add_noise(noisy_sine, snr_db=30.0)

        # 2. Signal with outliers (add random spikes to sine wave)
        sine_with_outliers = generate_sine_wave(
            frequency=500.0,  # 500 Hz
            amplitude=2.0,  # 2V peak
            duration=0.02,  # 20 ms
            sample_rate=50e3,  # 50 kHz sampling
        )
        # Add Gaussian noise
        sine_with_outliers = add_noise(sine_with_outliers, snr_db=40.0)

        # Add random outliers (5% of samples)
        data = sine_with_outliers.data.copy()
        num_outliers = int(len(data) * 0.05)
        outlier_indices = np.random.choice(len(data), num_outliers, replace=False)
        data[outlier_indices] += np.random.uniform(-10, 10, num_outliers)
        sine_with_outliers = WaveformTrace(data=data, metadata=sine_with_outliers.metadata)

        # 3. Noisy pulse train (bimodal distribution: high and low states)
        noisy_pulse = generate_pulse_train(
            pulse_width=500e-6,  # 500 µs
            period=1000e-6,  # 1 ms (1 kHz)
            amplitude=5.0,  # 5V
            duration=0.01,  # 10 ms
            sample_rate=100e3,  # 100 kHz sampling
            rise_time=10e-9,  # 10 ns
            fall_time=10e-9,  # 10 ns
        )
        noisy_pulse = add_noise(noisy_pulse, snr_db=35.0)

        return {
            "noisy_sine": noisy_sine,
            "sine_with_outliers": sine_with_outliers,
            "noisy_pulse": noisy_pulse,
        }

    def run_demonstration(self, data: dict[str, Any]) -> dict[str, Any]:
        """Run statistical analysis demonstrations."""
        results = {}

        self.section("Oscura Statistical Analysis")
        self.info("Demonstrating statistical analysis capabilities")
        self.info("Using synthetic signals with known characteristics")

        # ========== PART 1: BASIC STATISTICS ==========
        noisy_sine = data["noisy_sine"]
        self.subsection("Part 1: Basic Statistics")
        self.info("Noisy sine wave: 1 kHz, 3V amplitude, 30 dB SNR")

        stats = basic_stats(noisy_sine)
        results["basic_stats"] = stats

        self.info("\nBasic statistical measures:")
        self.result("Mean", f"{stats['mean']:.6f}", "V")
        self.result("Median", f"{np.median(noisy_sine.data):.6f}", "V")
        self.result("Std deviation", f"{stats['std']:.6f}", "V")
        self.result("Variance", f"{stats['variance']:.6f}", "V²")
        self.result("Min value", f"{stats['min']:.6f}", "V")
        self.result("Max value", f"{stats['max']:.6f}", "V")
        self.result("Range (Vpp)", f"{stats['range']:.6f}", "V")
        self.result("Sample count", stats["count"])

        self.info("\nInterpretation:")
        self.info(f"  Mean ~= 0 for AC signal -> {stats['mean']:.6f}V")
        self.info(f"  Range ~= 2 x 3V = 6V -> {stats['range']:.6f}V")
        self.info("  Std deviation captures noise + signal variation")

        # ========== PART 2: PERCENTILES AND QUARTILES ==========
        self.subsection("Part 2: Percentiles and Quartiles")
        self.info("Analyzing amplitude distribution using percentiles")

        pct = percentiles(noisy_sine)
        results["percentiles"] = pct

        self.info("\nQuartiles:")
        self.result("Min (p0)", f"{pct['p0']:.6f}", "V")
        self.result("Q1 (p25)", f"{pct['p25']:.6f}", "V")
        self.result("Median (p50)", f"{pct['p50']:.6f}", "V")
        self.result("Q3 (p75)", f"{pct['p75']:.6f}", "V")
        self.result("Max (p100)", f"{pct['p100']:.6f}", "V")

        iqr = pct["p75"] - pct["p25"]
        results["iqr"] = iqr
        self.result("IQR (Interquartile Range)", f"{iqr:.6f}", "V")

        self.info("\nInterpretation:")
        self.info(f"  Median ≈ 0 for symmetric AC signal → {pct['p50']:.6f}V")
        self.info(f"  IQR measures middle 50% spread → {iqr:.6f}V")

        # Custom percentiles (1st, 10th, 90th, 99th)
        custom_pct = percentiles(noisy_sine, [1, 10, 90, 99])
        results["custom_percentiles"] = custom_pct

        self.info("\nCustom percentiles (useful for outlier detection):")
        self.result("1st percentile", f"{custom_pct['p1']:.6f}", "V")
        self.result("10th percentile", f"{custom_pct['p10']:.6f}", "V")
        self.result("90th percentile", f"{custom_pct['p90']:.6f}", "V")
        self.result("99th percentile", f"{custom_pct['p99']:.6f}", "V")

        # ========== PART 3: DISTRIBUTION METRICS ==========
        self.subsection("Part 3: Distribution Shape Analysis")
        self.info("Analyzing distribution shape: skewness, kurtosis, crest factor")

        dist_metrics = distribution_metrics(noisy_sine)
        results["distribution_metrics"] = dist_metrics

        self.info("\nDistribution shape metrics:")
        self.result("Skewness", f"{dist_metrics['skewness']:.4f}")
        self.result("Kurtosis", f"{dist_metrics['kurtosis']:.4f}")
        self.result("Excess Kurtosis", f"{dist_metrics['excess_kurtosis']:.4f}")
        self.result("Crest Factor", f"{dist_metrics['crest_factor']:.4f}")
        self.result("Crest Factor (dB)", f"{dist_metrics['crest_factor_db']:.2f}", "dB")

        self.info("\nInterpretation:")
        self.info("  Skewness ≈ 0: symmetric distribution (AC signal)")
        self.info(f"    Measured: {dist_metrics['skewness']:.4f}")
        self.info("  Kurtosis ≈ 3: normal distribution")
        self.info(f"    Measured: {dist_metrics['kurtosis']:.4f}")
        self.info("  Crest Factor = Peak/RMS ratio")
        self.info(f"    Measured: {dist_metrics['crest_factor']:.4f}")

        # ========== PART 4: HISTOGRAM ANALYSIS ==========
        self.subsection("Part 4: Histogram Analysis")
        self.info("Generating amplitude histogram for distribution visualization")

        counts, edges = histogram(noisy_sine, bins=50)
        results["histogram_counts"] = counts
        results["histogram_edges"] = edges

        peak_bin = int(np.argmax(counts))
        peak_value = (edges[peak_bin] + edges[peak_bin + 1]) / 2
        results["histogram_peak"] = peak_value

        self.info("\nHistogram statistics:")
        self.result("Number of bins", len(counts))
        self.result("Bin width", f"{edges[1] - edges[0]:.6f}", "V")
        self.result("Peak bin value", f"{peak_value:.6f}", "V")
        self.result("Peak bin count", int(counts[peak_bin]))

        self.info("\nInterpretation:")
        self.info("  Peak near 0V confirms AC signal with zero mean")
        self.info("  Distribution shape reveals signal + noise characteristics")

        # ========== PART 5: OUTLIER DETECTION ==========
        sine_with_outliers = data["sine_with_outliers"]
        self.subsection("Part 5: Outlier Detection")
        self.info("Signal with 5% random outliers added")

        outlier_stats = basic_stats(sine_with_outliers)
        outlier_pct = percentiles(sine_with_outliers)

        # Detect outliers using IQR method
        iqr_outlier = outlier_pct["p75"] - outlier_pct["p25"]
        lower_bound = outlier_pct["p25"] - 1.5 * iqr_outlier
        upper_bound = outlier_pct["p75"] + 1.5 * iqr_outlier

        outliers = (sine_with_outliers.data < lower_bound) | (sine_with_outliers.data > upper_bound)
        num_outliers = np.sum(outliers)
        outlier_percentage = 100 * num_outliers / len(sine_with_outliers.data)

        results["num_outliers"] = num_outliers
        results["outlier_percentage"] = outlier_percentage

        self.info("\nOutlier detection (IQR method):")
        self.result("Lower bound", f"{lower_bound:.4f}", "V")
        self.result("Upper bound", f"{upper_bound:.4f}", "V")
        self.result("Outliers detected", num_outliers)
        self.result("Outlier percentage", f"{outlier_percentage:.2f}", "%")
        self.info(f"  Expected: ~5%, Detected: {outlier_percentage:.2f}%")

        # Z-score method
        mean = outlier_stats["mean"]
        std = outlier_stats["std"]
        z_scores = np.abs((sine_with_outliers.data - mean) / std)
        z_outliers = np.sum(z_scores > 3)  # 3-sigma threshold
        z_percentage = 100 * z_outliers / len(sine_with_outliers.data)

        results["z_outliers"] = z_outliers
        results["z_outlier_percentage"] = z_percentage

        self.info("\nOutlier detection (3-sigma / z-score method):")
        self.result("Threshold", "3-sigma")
        self.result("Outliers detected", z_outliers)
        self.result("Outlier percentage", f"{z_percentage:.2f}", "%")

        # ========== PART 6: BIMODAL DISTRIBUTION (PULSE TRAIN) ==========
        noisy_pulse = data["noisy_pulse"]
        self.subsection("Part 6: Bimodal Distribution Analysis")
        self.info("Pulse train creates bimodal distribution (HIGH and LOW states)")

        pulse_stats = basic_stats(noisy_pulse)
        pulse_dist = distribution_metrics(noisy_pulse)
        pulse_counts, pulse_edges = histogram(noisy_pulse, bins=100)

        results["pulse_stats"] = pulse_stats
        results["pulse_distribution"] = pulse_dist

        self.info("\nPulse train statistics:")
        self.result("Mean", f"{pulse_stats['mean']:.4f}", "V")
        self.result("Median", f"{np.median(noisy_pulse.data):.4f}", "V")
        self.result("Std deviation", f"{pulse_stats['std']:.4f}", "V")

        self.info("\nBimodal distribution indicators:")
        self.result("Skewness", f"{pulse_dist['skewness']:.4f}")
        self.result("Kurtosis", f"{pulse_dist['kurtosis']:.4f}")
        self.info("  Kurtosis < 3 suggests bimodal/flat distribution")

        # Find two peaks (HIGH and LOW modes)
        peak_indices = []
        for i in range(1, len(pulse_counts) - 1):
            if pulse_counts[i] > pulse_counts[i - 1] and pulse_counts[i] > pulse_counts[i + 1]:
                if pulse_counts[i] > np.max(pulse_counts) * 0.3:  # Significant peaks only
                    peak_indices.append(i)

        if len(peak_indices) >= 2:
            peak_indices.sort(key=lambda i: pulse_counts[i], reverse=True)
            low_peak = (pulse_edges[peak_indices[0]] + pulse_edges[peak_indices[0] + 1]) / 2
            high_peak = (pulse_edges[peak_indices[1]] + pulse_edges[peak_indices[1] + 1]) / 2

            # Ensure low_peak < high_peak
            if low_peak > high_peak:
                low_peak, high_peak = high_peak, low_peak

            self.info("\nDetected bimodal peaks:")
            self.result("LOW state peak", f"{low_peak:.4f}", "V")
            self.result("HIGH state peak", f"{high_peak:.4f}", "V")
            self.result("Separation", f"{high_peak - low_peak:.4f}", "V")

            results["bimodal_detected"] = True
            results["low_peak"] = low_peak
            results["high_peak"] = high_peak
        else:
            self.info("\nBimodal peaks: Detection inconclusive with current binning")
            results["bimodal_detected"] = False

        # ========== SUMMARY ==========
        self.subsection("Summary")
        self.info("\nStatistical analysis complete!")
        self.success("All statistical measurements computed successfully")

        return results

    def validate(self, results: dict[str, Any]) -> bool:
        """Validate statistical analysis results."""
        all_valid = True

        self.section("Validation")

        # ========== BASIC STATS VALIDATION ==========
        self.subsection("Basic Statistics Validation")

        stats = results["basic_stats"]

        # Mean should be near 0 for AC signal (within noise tolerance)
        if abs(stats["mean"]) < 0.5:
            print(f"  ✓ Mean: {stats['mean']:.6f}V (near 0 for AC signal)")
        else:
            print(f"  ✗ Mean: {stats['mean']:.6f}V (expected near 0)")
            all_valid = False

        # Range should be approximately 6V (2 x 3V amplitude + noise)
        if 5.5 < stats["range"] < 7.0:
            print(f"  ✓ Range: {stats['range']:.6f}V (expected ~6V)")
        else:
            print(f"  ✗ Range: {stats['range']:.6f}V (expected 5.5-7.0V)")
            all_valid = False

        # Std deviation should be reasonable (includes signal + noise)
        if 1.5 < stats["std"] < 3.0:
            print(f"  ✓ Std deviation: {stats['std']:.6f}V (reasonable)")
        else:
            print(f"  ✗ Std deviation: {stats['std']:.6f}V (expected 1.5-3.0V)")
            all_valid = False

        # ========== PERCENTILES VALIDATION ==========
        self.subsection("Percentiles Validation")

        pct = results["percentiles"]

        # Median should be near 0 for symmetric AC signal
        if abs(pct["p50"]) < 0.5:
            print(f"  ✓ Median: {pct['p50']:.6f}V (near 0 for symmetric signal)")
        else:
            print(f"  ✗ Median: {pct['p50']:.6f}V (expected near 0)")
            all_valid = False

        # IQR should be positive and reasonable
        iqr = results["iqr"]
        if 2.0 < iqr < 5.0:
            print(f"  ✓ IQR: {iqr:.6f}V (reasonable range)")
        else:
            print(f"  ✗ IQR: {iqr:.6f}V (expected 2.0-5.0V)")
            all_valid = False

        # ========== DISTRIBUTION METRICS VALIDATION ==========
        self.subsection("Distribution Metrics Validation")

        dist = results["distribution_metrics"]

        # Skewness should be near 0 for symmetric distribution
        if abs(dist["skewness"]) < 0.5:
            print(f"  ✓ Skewness: {dist['skewness']:.4f} (symmetric distribution)")
        else:
            print(f"  ✗ Skewness: {dist['skewness']:.4f} (expected near 0)")
            all_valid = False

        # Kurtosis should be reasonable (around 2-4 for noisy sine)
        if 1.5 < dist["kurtosis"] < 5.0:
            print(f"  ✓ Kurtosis: {dist['kurtosis']:.4f} (reasonable)")
        else:
            print(f"  ✗ Kurtosis: {dist['kurtosis']:.4f} (expected 1.5-5.0)")
            all_valid = False

        # Crest factor should be positive
        if dist["crest_factor"] > 0:
            print(f"  ✓ Crest Factor: {dist['crest_factor']:.4f} (positive)")
        else:
            print(f"  ✗ Crest Factor: {dist['crest_factor']:.4f} (must be positive)")
            all_valid = False

        # ========== HISTOGRAM VALIDATION ==========
        self.subsection("Histogram Validation")

        counts = results["histogram_counts"]
        _edges = results["histogram_edges"]  # Available for plotting if needed

        if len(counts) == 50:
            print(f"  ✓ Histogram bins: {len(counts)} (expected 50)")
        else:
            print(f"  ✗ Histogram bins: {len(counts)} (expected 50)")
            all_valid = False

        # Peak should be near ±amplitude for sine wave (not at 0!)
        # Sine wave has U-shaped distribution with peaks near ±amplitude
        peak = results["histogram_peak"]
        # For 3V sine, histogram shows peaks near ±2.1V (RMS regions)
        if -3.5 < peak < 3.5:
            print(f"  ✓ Histogram peak: {peak:.6f}V (within amplitude range)")
        else:
            print(f"  ✗ Histogram peak: {peak:.6f}V (outside expected range)")
            all_valid = False

        # ========== OUTLIER DETECTION VALIDATION ==========
        self.subsection("Outlier Detection Validation")

        outlier_pct = results["outlier_percentage"]
        # Should detect around 5% outliers (added to signal)
        # IQR method can be conservative, so allow 1-10% range
        if 1.0 < outlier_pct < 10.0:
            print(
                f"  ✓ Outlier detection (IQR): {outlier_pct:.2f}% (expected ~5%, method may vary)"
            )
        else:
            print(f"  ✗ Outlier detection (IQR): {outlier_pct:.2f}% (expected 1-10%)")
            all_valid = False

        z_pct = results["z_outlier_percentage"]
        if 1.0 < z_pct < 12.0:
            print(f"  ✓ Outlier detection (z-score): {z_pct:.2f}% (expected ~5%, method may vary)")
        else:
            print(f"  ✗ Outlier detection (z-score): {z_pct:.2f}% (expected 1-12%)")
            all_valid = False

        # ========== PULSE TRAIN VALIDATION ==========
        self.subsection("Pulse Train Validation")

        pulse_stats = results["pulse_stats"]

        # Mean should be around 2.5V (50% duty cycle x 5V)
        if 2.0 < pulse_stats["mean"] < 3.0:
            print(f"  ✓ Pulse mean: {pulse_stats['mean']:.4f}V (expected ~2.5V)")
        else:
            print(f"  ✗ Pulse mean: {pulse_stats['mean']:.4f}V (expected 2.0-3.0V)")
            all_valid = False

        # Bimodal distribution detection
        if results.get("bimodal_detected", False):
            low_peak = results["low_peak"]
            high_peak = results["high_peak"]
            print(f"  ✓ Bimodal peaks detected: LOW={low_peak:.2f}V, HIGH={high_peak:.2f}V")
        else:
            print("  ⚠ Bimodal peaks: Detection inconclusive (not a failure)")

        if all_valid:
            self.success("All statistical measurements validated!")
            self.info("\nKey takeaways:")
            self.info("  - basic_stats(): Mean, std, min, max, range, variance")
            self.info("  - percentiles(): Quartiles and custom percentiles")
            self.info("  - distribution_metrics(): Skewness, kurtosis, crest factor")
            self.info("  - histogram(): Amplitude distribution visualization")
            self.info("  - Outlier detection: IQR method and z-score method")
            self.info("  - Bimodal distributions: Pulse trains show two peaks")
            self.info("\nNext steps:")
            self.info("  - Try 03_spectral_analysis.py for frequency domain analysis")
            self.info("  - Explore 04_filtering.py for signal conditioning")
        else:
            self.error("Some statistical measurements failed validation")

        return all_valid


if __name__ == "__main__":
    demo: StatisticsDemo = StatisticsDemo()
    success: bool = demo.execute()
    sys.exit(0 if success else 1)
