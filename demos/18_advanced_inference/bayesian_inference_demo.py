#!/usr/bin/env python3
"""Bayesian Inference for Signal Analysis Demonstration.

This demo showcases Oscura's Bayesian inference capabilities for
signal characterization with full uncertainty quantification.

**Features Demonstrated**:
- Prior distribution specification
- Posterior calculation with credible intervals
- Baud rate inference from edge timing
- Symbol count estimation from amplitude histogram
- Sequential Bayesian updating
- Confidence-based decision making
- Multiple distribution families

**Bayesian Approach**:
P(parameter | data) ~ P(data | parameter) * P(parameter)
Posterior ~ Likelihood * Prior

**Advantages over Point Estimates**:
- Full uncertainty quantification
- Natural handling of limited data
- Prior knowledge incorporation
- Confidence intervals, not just point values
- Sequential updating capability

**Use Cases**:
- Unknown baud rate detection
- Signal level inference
- Protocol parameter estimation
- Quality assessment with uncertainty

Usage:
    python bayesian_inference_demo.py
    python bayesian_inference_demo.py --verbose

Author: Oscura Development Team
Date: 2026-01-16
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demos.common import BaseDemo, ValidationSuite, print_info, print_result
from demos.common.base_demo import run_demo_main
from demos.common.formatting import GREEN, RED, RESET, YELLOW, print_subheader

# Oscura imports
from oscura.inference.bayesian import (
    BayesianInference,
    Posterior,
    Prior,
    SequentialBayesian,
)


class BayesianInferenceDemo(BaseDemo):
    """Bayesian Inference for Signal Analysis Demonstration.

    This demo generates signals with known parameters and uses Bayesian
    inference to recover those parameters with uncertainty quantification.
    """

    name = "Bayesian Inference Demo"
    description = "Demonstrates Bayesian inference for signal characterization"
    category = "advanced_inference"

    def __init__(self, **kwargs):
        """Initialize demo."""
        super().__init__(**kwargs)

        # True parameters (to be inferred)
        self.true_baud_rate = 115200  # UART baud rate
        self.true_symbol_count = 4  # PAM-4 signal
        self.true_duty_cycle = 0.45  # Slightly asymmetric

        # Data storage
        self.edge_times = None
        self.amplitude_histogram = None
        self.duty_samples = None

    def _generate_uart_edges(self, n_bytes: int = 100) -> np.ndarray:
        """Generate UART edge timestamps with timing jitter.

        Args:
            n_bytes: Number of bytes to simulate.

        Returns:
            Array of edge timestamps.
        """
        bit_period = 1 / self.true_baud_rate
        jitter_rms = bit_period * 0.02  # 2% jitter

        edges = []
        current_time = 0.0

        for _ in range(n_bytes):
            # Start bit
            edges.append(current_time)
            current_time += bit_period + np.random.randn() * jitter_rms

            # 8 data bits (random pattern)
            prev_bit = 0
            for _ in range(8):
                bit = np.random.randint(0, 2)
                if bit != prev_bit:
                    edges.append(current_time)
                current_time += bit_period + np.random.randn() * jitter_rms
                prev_bit = bit

            # Stop bit
            edges.append(current_time)
            current_time += bit_period + np.random.randn() * jitter_rms

            # Inter-byte gap
            current_time += bit_period * np.random.uniform(1, 5)

        return np.array(edges)

    def _generate_pam4_samples(self, n_samples: int = 10000) -> np.ndarray:
        """Generate PAM-4 signal amplitude samples.

        Args:
            n_samples: Number of samples.

        Returns:
            Array of amplitude values.
        """
        symbols = np.array([0.0, 0.33, 0.67, 1.0])  # PAM-4 levels
        noise_std = 0.03

        # Random symbol selection
        symbol_idx = np.random.randint(0, len(symbols), n_samples)
        amplitudes = symbols[symbol_idx] + np.random.randn(n_samples) * noise_std

        return amplitudes

    def _generate_duty_cycle_samples(self, n_cycles: int = 500) -> tuple[np.ndarray, np.ndarray]:
        """Generate high and low times for duty cycle estimation.

        Args:
            n_cycles: Number of cycles.

        Returns:
            Tuple of (high_times, low_times).
        """
        period = 1e-6  # 1 MHz clock
        jitter_rms = period * 0.01

        high_times = self.true_duty_cycle * period + np.random.randn(n_cycles) * jitter_rms
        low_times = (1 - self.true_duty_cycle) * period + np.random.randn(n_cycles) * jitter_rms

        return high_times, low_times

    def generate_data(self) -> None:
        """Generate test data for Bayesian inference.

        Loads from file if available (--data-file override or default NPZ),
        otherwise generates synthetic UART edges, PAM-4 amplitudes, and duty cycle samples.
        """
        # Try loading data from file
        file_to_load = None

        # 1. Check CLI override
        if self.data_file and self.data_file.exists():
            file_to_load = self.data_file
            print_info(f"Loading Bayesian inference data from CLI override: {self.data_file}")
        # 2. Check default generated data
        elif default_file := self.find_default_data_file("bayesian_inference.npz"):
            file_to_load = default_file
            print_info(f"Loading Bayesian inference data from default file: {default_file.name}")

        # Load from file if found
        if file_to_load:
            try:
                data = np.load(file_to_load)
                self.edge_times = data["edge_times"]
                self.amplitude_histogram = data["amplitude_histogram"]
                high_times = data["high_times"]
                low_times = data["low_times"]
                self.duty_samples = (high_times, low_times)

                # Load parameters if available
                if "true_baud_rate" in data:
                    self.true_baud_rate = int(data["true_baud_rate"])
                if "true_symbol_count" in data:
                    self.true_symbol_count = int(data["true_symbol_count"])
                if "true_duty_cycle" in data:
                    self.true_duty_cycle = float(data["true_duty_cycle"])

                print_result("Data loaded from file", file_to_load.name)
                print_result("UART edges", len(self.edge_times))
                print_result("Amplitude histogram bins", len(self.amplitude_histogram))
                print_result("Duty cycle samples", len(high_times))
                print_info(f"  True baud rate: {self.true_baud_rate} bps")
                print_info(f"  True symbol count: {self.true_symbol_count}")
                print_info(f"  True duty cycle: {self.true_duty_cycle * 100:.1f}%")
                return
            except Exception as e:
                print_info(f"Failed to load data from file: {e}, falling back to synthetic")

        # Generate synthetic data
        print_info("Generating test data with known parameters...")

        # UART edges
        print_info(f"  True baud rate: {self.true_baud_rate} bps")
        self.edge_times = self._generate_uart_edges(n_bytes=100)
        print_result("UART edges generated", len(self.edge_times))

        # PAM-4 amplitudes
        print_info(f"  True symbol count: {self.true_symbol_count}")
        amplitudes = self._generate_pam4_samples(n_samples=10000)
        self.amplitude_histogram, _ = np.histogram(amplitudes, bins=50)
        print_result("Amplitude samples", len(amplitudes))

        # Duty cycle samples
        print_info(f"  True duty cycle: {self.true_duty_cycle * 100:.1f}%")
        high_times, low_times = self._generate_duty_cycle_samples(n_cycles=500)
        self.duty_samples = (high_times, low_times)
        print_result("Duty cycle samples", len(high_times))

    def run_analysis(self) -> None:
        """Perform Bayesian inference on test data."""
        inference = BayesianInference()

        # ===== Baud Rate Inference =====
        print_subheader("Baud Rate Inference")

        posterior = inference.infer_baud_rate(self.edge_times)

        print_result("Prior", "Log-uniform [100, 10M]")
        print_result("Estimated baud rate", f"{posterior.mean:.0f} bps")
        print_result("Standard deviation", f"{posterior.std:.0f} bps")
        print_result("95% CI lower", f"{posterior.ci_lower:.0f} bps")
        print_result("95% CI upper", f"{posterior.ci_upper:.0f} bps")
        print_result("Confidence", f"{posterior.confidence:.2%}")

        self.results["baud_mean"] = posterior.mean
        self.results["baud_std"] = posterior.std
        self.results["baud_confidence"] = posterior.confidence

        # Check accuracy
        error_pct = abs(posterior.mean - self.true_baud_rate) / self.true_baud_rate * 100
        if error_pct < 5:
            print_info(f"  {GREEN}Within 5% of true value ({error_pct:.1f}%){RESET}")
        elif error_pct < 10:
            print_info(f"  {YELLOW}Within 10% of true value ({error_pct:.1f}%){RESET}")
        else:
            print_info(f"  {RED}Error: {error_pct:.1f}%{RESET}")

        self.results["baud_error_pct"] = error_pct

        # Check if true value is within CI
        in_ci = posterior.ci_lower <= self.true_baud_rate <= posterior.ci_upper
        if in_ci:
            print_info(f"  {GREEN}True value within 95% CI{RESET}")
        else:
            print_info(f"  {RED}True value outside 95% CI{RESET}")

        # ===== Symbol Count Inference =====
        print_subheader("Symbol Count Inference")

        symbol_posterior = inference.infer_symbol_count(self.amplitude_histogram)

        print_result("Estimated symbols", f"{int(symbol_posterior.mean)}")
        print_result("Confidence", f"{symbol_posterior.confidence:.2%}")

        self.results["symbol_count"] = int(symbol_posterior.mean)
        self.results["symbol_confidence"] = symbol_posterior.confidence

        if int(symbol_posterior.mean) == self.true_symbol_count:
            print_info(f"  {GREEN}Correctly identified as PAM-{self.true_symbol_count}{RESET}")
        else:
            print_info(
                f"  {RED}Estimated {int(symbol_posterior.mean)}, true is {self.true_symbol_count}{RESET}"
            )

        # ===== Sequential Bayesian Updating =====
        print_subheader("Sequential Bayesian Updating")

        # Start with weak prior
        prior = Prior("normal", {"mean": 100000, "std": 50000})
        sequential = SequentialBayesian("baud_rate", prior)

        # Process edges in batches
        edge_diffs = np.diff(self.edge_times)
        batch_size = 50
        confidences = []

        print_info("Sequential updating with edge observations:")
        for i in range(0, len(edge_diffs), batch_size):
            batch = edge_diffs[i : i + batch_size]

            # Update with batch
            for obs in batch:
                if obs > 0:
                    observed_freq = 1.0 / obs

                    # Create likelihood function for this observation
                    def likelihood(baud_rate: float, obs_freq: float = observed_freq) -> float:
                        # Likelihood of observing this frequency given a baud rate
                        # Model as Gaussian around expected frequency
                        expected = baud_rate
                        sigma = baud_rate * 0.05  # 5% uncertainty
                        return float(np.exp(-0.5 * ((obs_freq - expected) / sigma) ** 2))

                    sequential.update(likelihood)

            confidence = sequential.get_confidence()
            if isinstance(sequential.current_posterior, Posterior):
                mean = sequential.current_posterior.mean
            else:
                mean = sequential.current_posterior.params.get("mean", 0.0)
            confidences.append(confidence)

            if (i // batch_size + 1) % 3 == 0 or i == 0:
                print_info(
                    f"  After {i + len(batch)} edges: mean={mean:.0f}, confidence={confidence:.2%}"
                )

        final_confidence = confidences[-1]
        print_result("Final confidence", f"{final_confidence:.2%}")

        self.results["sequential_final_confidence"] = final_confidence

        # ===== Prior Sensitivity Analysis =====
        print_subheader("Prior Sensitivity Analysis")

        priors = {
            "Uninformative (uniform)": Prior("uniform", {"low": 1000, "high": 1000000}),
            "Log-uniform (default)": Prior("log_uniform", {"low": 100, "high": 10000000}),
            "Informative (normal)": Prior("normal", {"mean": 115200, "std": 10000}),
            "Wrong prior (normal)": Prior("normal", {"mean": 9600, "std": 1000}),
        }

        print_info("Same data, different priors:")
        for prior_name, prior in priors.items():
            result = inference.infer_baud_rate(self.edge_times, prior=prior)
            error = abs(result.mean - self.true_baud_rate) / self.true_baud_rate * 100

            status = f"{GREEN}" if error < 5 else (f"{YELLOW}" if error < 20 else f"{RED}")
            print_info(f"  {prior_name:30s}: {result.mean:>8.0f} bps ({status}{error:.1f}%{RESET})")

        # ===== Confidence Interpretation =====
        print_subheader("Confidence Interpretation")

        print_info("Confidence scores map to signal quality:")
        confidence_levels = [
            (0.95, "Very High", "Strong evidence, proceed with high confidence"),
            (0.80, "High", "Good evidence, proceed with normal confidence"),
            (0.60, "Medium", "Moderate evidence, verify with additional data"),
            (0.40, "Low", "Weak evidence, more data needed"),
            (0.0, "Very Low", "Insufficient evidence, analysis unreliable"),
        ]

        for threshold, level, description in confidence_levels:
            if posterior.confidence >= threshold:
                print_info(f"  Current: {level} - {description}")
                break

        # Summary
        print_subheader("Summary")
        print_info("Parameter            True Value    Estimated    Error")
        print_info("-" * 55)
        print_info(
            f"Baud rate (bps)      {self.true_baud_rate:>10}    {posterior.mean:>10.0f}    {self.results['baud_error_pct']:.1f}%"
        )
        print_info(
            f"Symbol count         {self.true_symbol_count:>10}    {self.results['symbol_count']:>10}    "
            + ("0%" if self.results["symbol_count"] == self.true_symbol_count else "error")
        )

    def validate_results(self, suite: ValidationSuite) -> None:
        """Validate Bayesian inference results."""
        # Check baud rate inference
        baud_error = self.results.get("baud_error_pct", 100)
        suite.check_less(
            "Baud rate error",
            baud_error,
            20,  # Within 20%
            category="baud_rate",
        )

        # Check confidence
        confidence = self.results.get("baud_confidence", 0)
        suite.check_greater(
            "Baud rate confidence",
            confidence,
            0.5,  # At least 50% confident
            category="confidence",
        )

        # Check symbol count (algorithm may estimate 3-4 for noisy signals)
        suite.check_range(
            "Symbol count",
            self.results.get("symbol_count", 0),
            self.true_symbol_count - 1,  # Allow one below
            self.true_symbol_count,  # Up to true value
            category="symbols",
        )

        # Check sequential updating improved confidence
        final_conf = self.results.get("sequential_final_confidence", 0)
        suite.check_greater(
            "Sequential final confidence",
            final_conf,
            0.6,
            category="sequential",
        )


if __name__ == "__main__":
    sys.exit(run_demo_main(BayesianInferenceDemo))
