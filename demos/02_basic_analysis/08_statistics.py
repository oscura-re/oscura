#!/usr/bin/env python3
"""Statistical Analysis: Statistical signal characterization.

This demo demonstrates statistical analysis of signals:
- Mean, median, standard deviation
- Histogram analysis and binning
- Distribution fitting (normal, exponential)
- Correlation analysis
- Signal-to-noise ratio (SNR)
- Coefficient of variation
- Skewness and kurtosis

Related demos:
- 01_waveform_basics.py - Basic measurements
- 03_spectral_basics.py - Spectral analysis
- ../04_advanced_analysis/08_statistics_advanced.py - Advanced statistics

Usage:
    python demos/02_basic_analysis/08_statistics.py
    python demos/02_basic_analysis/08_statistics.py --verbose
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np

import oscura as osc
from demos.common import BaseDemo, ValidationSuite, print_info, print_result, print_table
from demos.common.base_demo import run_demo_main
from demos.common.data_generation import add_noise, generate_sine_wave, generate_square_wave
from demos.common.formatting import print_subheader


class StatisticsDemo(BaseDemo):
    """Statistical signal analysis demonstration."""

    name = "Statistical Signal Analysis"
    description = "Mean, histogram, distribution, correlation, and statistical characterization"
    category = "basic_analysis"

    capabilities = [
        "oscura.mean",
        "oscura.median",
        "oscura.std",
        "oscura.histogram",
        "oscura.correlation",
        "oscura.snr",
        "oscura.skewness",
        "oscura.kurtosis",
    ]

    related_demos = [
        "01_waveform_basics.py",
        "03_spectral_basics.py",
        "../04_advanced_analysis/08_statistics_advanced.py",
    ]

    def __init__(self, **kwargs):
        """Initialize demo."""
        super().__init__(**kwargs)
        self.clean_signal = None
        self.noisy_signal = None
        self.square_signal = None

    def generate_data(self) -> None:
        """Generate test signals for statistical analysis."""
        print_info("Generating statistical test signals...")

        # 1. Clean sine wave
        self.clean_signal = generate_sine_wave(
            frequency=1000.0,  # 1 kHz
            amplitude=2.0,  # 2V peak
            duration=0.01,  # 10 ms
            sample_rate=100e3,  # 100 kHz
        )

        # 2. Noisy sine wave
        self.noisy_signal = add_noise(self.clean_signal, snr_db=20.0)

        # 3. Square wave (different distribution)
        self.square_signal = generate_square_wave(
            frequency=1000.0,  # 1 kHz
            amplitude=3.0,  # 3V
            duration=0.01,  # 10 ms
            sample_rate=100e3,  # 100 kHz
            duty_cycle=0.5,
        )

        print_result("Clean sine wave", "1 kHz, 2V peak")
        print_result("Noisy sine wave", "1 kHz, 2V peak, 20 dB SNR")
        print_result("Square wave", "1 kHz, 3V, 50% duty")

    def run_analysis(self) -> None:
        """Execute statistical analysis."""
        # ========== PART 1: BASIC STATISTICS ==========
        print_subheader("Part 1: Basic Statistical Measures")
        print_info("Computing fundamental statistics")

        # Clean signal statistics
        clean_mean = osc.mean(self.clean_signal)
        clean_median = osc.median(self.clean_signal)
        clean_std = osc.std(self.clean_signal)
        clean_min = osc.min(self.clean_signal)
        clean_max = osc.max(self.clean_signal)

        self.results["clean_mean"] = clean_mean
        self.results["clean_median"] = clean_median
        self.results["clean_std"] = clean_std
        self.results["clean_min"] = clean_min
        self.results["clean_max"] = clean_max

        print_result("Clean - Mean", f"{clean_mean:.6f} V")
        print_result("Clean - Median", f"{clean_median:.6f} V")
        print_result("Clean - Std Dev", f"{clean_std:.4f} V")
        print_result("Clean - Min", f"{clean_min:.4f} V")
        print_result("Clean - Max", f"{clean_max:.4f} V")

        # Noisy signal statistics
        noisy_mean = osc.mean(self.noisy_signal)
        noisy_median = osc.median(self.noisy_signal)
        noisy_std = osc.std(self.noisy_signal)

        self.results["noisy_mean"] = noisy_mean
        self.results["noisy_median"] = noisy_median
        self.results["noisy_std"] = noisy_std

        print_result("Noisy - Mean", f"{noisy_mean:.6f} V")
        print_result("Noisy - Median", f"{noisy_median:.6f} V")
        print_result("Noisy - Std Dev", f"{noisy_std:.4f} V")

        # Coefficient of variation (CV = std/mean)
        cv_clean = abs(clean_std / clean_mean) if clean_mean != 0 else 0
        cv_noisy = abs(noisy_std / noisy_mean) if noisy_mean != 0 else 0

        self.results["cv_clean"] = cv_clean
        self.results["cv_noisy"] = cv_noisy

        print_result("Clean - CV", f"{cv_clean:.4f}")
        print_result("Noisy - CV", f"{cv_noisy:.4f}")

        # ========== PART 2: HISTOGRAM ANALYSIS ==========
        print_subheader("Part 2: Histogram Analysis")
        print_info("Analyzing signal value distributions")

        # Compute histograms
        hist_clean = osc.histogram(self.clean_signal, bins=50)
        hist_noisy = osc.histogram(self.noisy_signal, bins=50)
        hist_square = osc.histogram(self.square_signal, bins=50)

        self.results["hist_clean_bins"] = len(hist_clean["counts"])
        self.results["hist_clean_peak"] = np.max(hist_clean["counts"])

        print_result("Clean histogram bins", len(hist_clean["counts"]))
        print_result("Clean histogram peak", np.max(hist_clean["counts"]))

        # Find most common value (mode approximation)
        mode_idx_clean = np.argmax(hist_clean["counts"])
        mode_value_clean = hist_clean["edges"][mode_idx_clean]

        mode_idx_square = np.argmax(hist_square["counts"])
        mode_value_square = hist_square["edges"][mode_idx_square]

        self.results["mode_clean"] = mode_value_clean
        self.results["mode_square"] = mode_value_square

        print_result("Clean mode value", f"{mode_value_clean:.4f} V")
        print_result("Square mode value", f"{mode_value_square:.4f} V")

        # ========== PART 3: DISTRIBUTION SHAPE ==========
        print_subheader("Part 3: Distribution Shape Analysis")
        print_info("Computing skewness and kurtosis")

        # Skewness (measure of asymmetry)
        skew_clean = osc.skewness(self.clean_signal)
        skew_square = osc.skewness(self.square_signal)

        self.results["skew_clean"] = skew_clean
        self.results["skew_square"] = skew_square

        print_result("Clean skewness", f"{skew_clean:.4f}")
        print_result("Square skewness", f"{skew_square:.4f}")
        print_info("  (0 = symmetric, >0 = right-skewed, <0 = left-skewed)")

        # Kurtosis (measure of tail heaviness)
        kurt_clean = osc.kurtosis(self.clean_signal)
        kurt_square = osc.kurtosis(self.square_signal)

        self.results["kurt_clean"] = kurt_clean
        self.results["kurt_square"] = kurt_square

        print_result("Clean kurtosis", f"{kurt_clean:.4f}")
        print_result("Square kurtosis", f"{kurt_square:.4f}")
        print_info("  (3 = normal, >3 = heavy tails, <3 = light tails)")

        # ========== PART 4: SIGNAL-TO-NOISE RATIO ==========
        print_subheader("Part 4: Signal-to-Noise Ratio (SNR)")
        print_info("Measuring signal quality")

        # Compute SNR for noisy signal
        snr_measured = osc.snr(self.noisy_signal)
        self.results["snr_measured"] = snr_measured

        print_result("Measured SNR", f"{snr_measured:.2f} dB")
        print_info("  Expected SNR: 20 dB (from noise generation)")

        # Estimate noise power
        noise_estimate = self.noisy_signal.data - self.clean_signal.data
        noise_power = np.mean(noise_estimate**2)
        signal_power = np.mean(self.clean_signal.data**2)
        snr_from_comparison = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else 0

        self.results["snr_from_comparison"] = snr_from_comparison

        print_result("SNR from comparison", f"{snr_from_comparison:.2f} dB")

        # ========== PART 5: CORRELATION ANALYSIS ==========
        print_subheader("Part 5: Correlation Analysis")
        print_info("Measuring signal correlation")

        # Correlation between clean and noisy signal
        corr_clean_noisy = osc.correlation(self.clean_signal, self.noisy_signal)
        self.results["corr_clean_noisy"] = corr_clean_noisy

        print_result("Clean vs Noisy correlation", f"{corr_clean_noisy:.6f}")
        print_info("  (1.0 = perfect correlation, 0.0 = uncorrelated)")

        # Auto-correlation of clean signal (should be periodic)
        autocorr_clean = osc.autocorrelation(self.clean_signal)
        self.results["autocorr_peak"] = np.max(autocorr_clean["values"])

        print_result("Auto-correlation peak", f"{self.results['autocorr_peak']:.6f}")

        # Correlation between clean and square (different waveforms)
        corr_sine_square = osc.correlation(self.clean_signal, self.square_signal)
        self.results["corr_sine_square"] = corr_sine_square

        print_result("Sine vs Square correlation", f"{corr_sine_square:.6f}")

        # ========== PART 6: STATISTICAL COMPARISON TABLE ==========
        print_subheader("Part 6: Statistical Comparison")

        headers = ["Statistic", "Clean Sine", "Noisy Sine", "Square Wave"]
        rows = [
            ["Mean", f"{clean_mean:.6f}", f"{noisy_mean:.6f}", f"{osc.mean(self.square_signal):.6f}"],
            ["Median", f"{clean_median:.6f}", f"{noisy_median:.6f}", f"{osc.median(self.square_signal):.6f}"],
            ["Std Dev", f"{clean_std:.4f}", f"{noisy_std:.4f}", f"{osc.std(self.square_signal):.4f}"],
            ["CV", f"{cv_clean:.4f}", f"{cv_noisy:.4f}", "-"],
            ["Skewness", f"{skew_clean:.4f}", "-", f"{skew_square:.4f}"],
            ["Kurtosis", f"{kurt_clean:.4f}", "-", f"{kurt_square:.4f}"],
        ]

        print_table(headers, rows)

        # ========== PART 7: PERCENTILES ==========
        print_subheader("Part 7: Percentile Analysis")
        print_info("Computing percentile values")

        # Calculate percentiles for noisy signal
        p5 = osc.percentile(self.noisy_signal, 5)
        p25 = osc.percentile(self.noisy_signal, 25)
        p50 = osc.percentile(self.noisy_signal, 50)  # Median
        p75 = osc.percentile(self.noisy_signal, 75)
        p95 = osc.percentile(self.noisy_signal, 95)

        self.results["p5"] = p5
        self.results["p25"] = p25
        self.results["p50"] = p50
        self.results["p75"] = p75
        self.results["p95"] = p95

        print_result("5th percentile", f"{p5:.4f} V")
        print_result("25th percentile (Q1)", f"{p25:.4f} V")
        print_result("50th percentile (median)", f"{p50:.4f} V")
        print_result("75th percentile (Q3)", f"{p75:.4f} V")
        print_result("95th percentile", f"{p95:.4f} V")

        # Interquartile range (IQR)
        iqr = p75 - p25
        self.results["iqr"] = iqr
        print_result("Interquartile range (IQR)", f"{iqr:.4f} V")

        # ========== MEASUREMENT INTERPRETATION ==========
        print_subheader("Statistical Analysis Interpretation")

        print_info("\n[Basic Statistics]")
        print_info(f"  Clean signal has near-zero mean ({clean_mean:.6f}V) - AC signal")
        print_info(f"  Noise increases std dev: {clean_std:.4f}V → {noisy_std:.4f}V")
        print_info(f"  Coefficient of variation increases with noise")

        print_info("\n[Distribution Shape]")
        print_info(f"  Sine wave: symmetric (skewness ≈ {skew_clean:.4f})")
        print_info(f"  Square wave: bimodal distribution (two peaks at ±3V)")
        print_info(f"  Kurtosis indicates tail behavior relative to normal distribution")

        print_info("\n[Signal Quality]")
        print_info(f"  SNR: {snr_measured:.2f}dB (target: 20dB)")
        print_info(f"  High correlation ({corr_clean_noisy:.6f}) despite noise")
        print_info(f"  Low sine-square correlation ({corr_sine_square:.6f}) - different waveforms")

        print_info("\n[Percentiles]")
        print_info(f"  IQR: {iqr:.4f}V captures central 50% of data")
        print_info(f"  5%-95% range: {p95-p5:.4f}V captures 90% of values")

    def validate_results(self, suite: ValidationSuite) -> None:
        """Validate statistical analysis results."""
        # Basic statistics - clean signal should have near-zero mean
        suite.check_range("Clean mean", abs(self.results["clean_mean"]), 0.0, 0.01)
        suite.check_range("Clean std", self.results["clean_std"], 1.3, 1.5)

        # Noisy signal should have higher std dev
        suite.check_range("Noisy std", self.results["noisy_std"], 1.4, 1.7)

        # Histogram validation
        suite.check_equal("Histogram bins", self.results["hist_clean_bins"], 50)

        # Skewness - sine should be nearly symmetric
        suite.check_range("Clean skewness", abs(self.results["skew_clean"]), 0.0, 0.5)

        # SNR validation
        suite.check_range("Measured SNR", self.results["snr_measured"], 15, 25)
        suite.check_range("SNR from comparison", self.results["snr_from_comparison"], 15, 25)

        # Correlation validation
        suite.check_range("Clean-noisy correlation", self.results["corr_clean_noisy"], 0.8, 1.0)
        suite.check_range("Sine-square correlation", abs(self.results["corr_sine_square"]), 0.0, 0.5)

        # Percentile validation
        suite.check_range("IQR", self.results["iqr"], 1.0, 3.0)


if __name__ == "__main__":
    sys.exit(run_demo_main(StatisticsDemo))
