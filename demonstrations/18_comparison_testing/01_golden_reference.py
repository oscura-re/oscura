"""Golden Reference Comparison: Compare signals against reference waveforms

Demonstrates:
- Golden waveform comparison
- Statistical similarity metrics (correlation, RMSE, MAE)
- Pass/fail criteria and tolerance bands
- Difference visualization and analysis
- Template matching for conformance testing

IEEE Standards: None (general measurement practice)
Related Demos:
- 18_comparison_testing/02_limit_testing.py
- 18_comparison_testing/03_mask_testing.py
- 02_basic_analysis/01_waveform_measurements.py

This demonstration shows how to compare measured signals against golden reference
waveforms for production testing, regression detection, and quality assurance.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

# Add demonstrations to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demonstrations.common import BaseDemo, validate_approximately
from tests.fixtures.signal_builders import SignalBuilder


class GoldenReferenceDemo(BaseDemo):
    """Demonstration of golden reference comparison testing."""

    def __init__(self) -> None:
        """Initialize golden reference demonstration."""
        super().__init__(
            name="golden_reference_comparison",
            description="Compare signals against golden references with statistical metrics",
            capabilities=[
                "Correlation coefficient",
                "RMSE (root mean square error)",
                "MAE (mean absolute error)",
                "Peak error detection",
                "Pass/fail criteria",
            ],
            ieee_standards=[],
            related_demos=[
                "18_comparison_testing/02_limit_testing.py",
                "02_basic_analysis/01_waveform_measurements.py",
            ],
        )

    def generate_test_data(self) -> dict[str, Any]:
        """Generate golden reference and test signals.

        Returns:
            Dictionary containing golden and test signals
        """
        sample_rate = 1e6
        duration = 0.001

        # Golden reference signal
        golden = SignalBuilder.sine_wave(
            frequency=10e3, sample_rate=sample_rate, duration=duration, amplitude=1.0
        )

        # Generate noise arrays matching golden length
        n_samples = len(golden)

        # Good DUT (device under test) - very close to golden
        noise_good = np.random.normal(0, 0.01, n_samples)
        dut_good = golden + noise_good

        # Marginal DUT - noticeable deviation
        noise_marginal = np.random.normal(0, 0.03, n_samples)
        dut_marginal = golden * 0.98 + noise_marginal

        # Bad DUT - significant deviation
        noise_bad = np.random.normal(0, 0.1, n_samples)
        dut_bad = golden * 0.85 + noise_bad

        # Distorted DUT - harmonic distortion
        harmonic_2 = SignalBuilder.sine_wave(20e3, sample_rate, duration)  # 2nd harmonic
        harmonic_3 = SignalBuilder.sine_wave(30e3, sample_rate, duration)  # 3rd harmonic
        dut_distorted = golden + 0.1 * harmonic_2 + 0.05 * harmonic_3

        return {
            "sample_rate": sample_rate,
            "golden": golden,
            "dut_good": dut_good,
            "dut_marginal": dut_marginal,
            "dut_bad": dut_bad,
            "dut_distorted": dut_distorted,
        }

    def run_demonstration(self, data: dict[str, Any]) -> dict[str, Any]:
        """Run golden reference comparison demonstration."""
        results: dict[str, Any] = {}

        self.section("Golden Reference Comparison Testing")
        self.info("Comparing device signals against golden reference waveforms")

        golden = data["golden"]

        # Part 1: Statistical Similarity Metrics
        self.subsection("Part 1: Statistical Similarity Metrics")

        def calculate_metrics(signal: np.ndarray, reference: np.ndarray) -> dict[str, float]:
            """Calculate similarity metrics between signal and reference."""
            # Correlation coefficient
            correlation = np.corrcoef(signal, reference)[0, 1]

            # Root mean square error
            rmse = np.sqrt(np.mean((signal - reference) ** 2))

            # Mean absolute error
            mae = np.mean(np.abs(signal - reference))

            # Peak error
            peak_error = np.max(np.abs(signal - reference))

            # Normalized RMSE (as percentage of reference RMS)
            reference_rms = np.sqrt(np.mean(reference**2))
            nrmse_percent = (rmse / reference_rms) * 100

            return {
                "correlation": correlation,
                "rmse": rmse,
                "mae": mae,
                "peak_error": peak_error,
                "nrmse_percent": nrmse_percent,
            }

        # Analyze each DUT
        duts = {
            "Good DUT": data["dut_good"],
            "Marginal DUT": data["dut_marginal"],
            "Bad DUT": data["dut_bad"],
            "Distorted DUT": data["dut_distorted"],
        }

        all_metrics = {}
        for dut_name, dut_signal in duts.items():
            metrics = calculate_metrics(dut_signal, golden)
            all_metrics[dut_name] = metrics

            self.info(f"\n{dut_name}:")
            self.result("  Correlation", f"{metrics['correlation']:.6f}")
            self.result("  RMSE", f"{metrics['rmse']:.6f}", "V")
            self.result("  MAE", f"{metrics['mae']:.6f}", "V")
            self.result("  Peak error", f"{metrics['peak_error']:.6f}", "V")
            self.result("  Normalized RMSE", f"{metrics['nrmse_percent']:.4f}", "%")

        results["metrics"] = all_metrics

        # Part 2: Pass/Fail Criteria
        self.subsection("Part 2: Pass/Fail Criteria")
        self.info("Defining acceptance thresholds for production testing")

        # Define pass/fail thresholds
        thresholds = {
            "correlation_min": 0.99,  # Must be > 99% correlated
            "nrmse_max": 5.0,  # NRMSE must be < 5%
            "peak_error_max": 0.1,  # Peak error < 100 mV
        }

        self.result("Correlation threshold", f">= {thresholds['correlation_min']:.4f}")
        self.result("NRMSE threshold", f"<= {thresholds['nrmse_max']:.2f}%")
        self.result("Peak error threshold", f"<= {thresholds['peak_error_max']:.3f} V")

        # Evaluate each DUT
        self.info("\nEvaluation Results:")
        pass_fail_results = {}

        for dut_name, metrics in all_metrics.items():
            passes_correlation = metrics["correlation"] >= thresholds["correlation_min"]
            passes_nrmse = metrics["nrmse_percent"] <= thresholds["nrmse_max"]
            passes_peak = metrics["peak_error"] <= thresholds["peak_error_max"]

            overall_pass = passes_correlation and passes_nrmse and passes_peak

            pass_fail_results[dut_name] = overall_pass

            status = "PASS" if overall_pass else "FAIL"
            self.info(f"\n{dut_name}: {status}")
            self.result("  Correlation check", "PASS" if passes_correlation else "FAIL")
            self.result("  NRMSE check", "PASS" if passes_nrmse else "FAIL")
            self.result("  Peak error check", "PASS" if passes_peak else "FAIL")

        results["pass_fail"] = pass_fail_results

        # Part 3: Difference Analysis
        self.subsection("Part 3: Difference Signal Analysis")
        self.info("Analyzing the difference between DUT and golden reference")

        # Analyze Good DUT difference
        good_diff = data["dut_good"] - golden

        self.info("\nGood DUT Difference Signal:")
        self.result("  Mean difference", f"{np.mean(good_diff):.6f}", "V")
        self.result("  Std deviation", f"{np.std(good_diff):.6f}", "V")
        self.result("  Min difference", f"{np.min(good_diff):.6f}", "V")
        self.result("  Max difference", f"{np.max(good_diff):.6f}", "V")

        # Check if difference is noise-like (Gaussian distribution)
        # For Gaussian: mean ~= 0, most values within ±3-sigma
        within_3sigma = np.sum(np.abs(good_diff) <= 3 * np.std(good_diff))
        gaussian_percentage = (within_3sigma / len(good_diff)) * 100
        self.result("  Within ±3-sigma", f"{gaussian_percentage:.2f}%")
        self.result("  Expected for Gaussian", "99.7%")

        results["good_diff_std"] = float(np.std(good_diff))
        results["good_diff_gaussian"] = float(gaussian_percentage)

        # Part 4: Amplitude and Phase Errors
        self.subsection("Part 4: Amplitude and Phase Error Detection")

        # Use RMS amplitude for better noise immunity (peak values can be affected by noise)
        marginal_rms = np.sqrt(np.mean(data["dut_marginal"] ** 2))
        golden_rms = np.sqrt(np.mean(golden**2))
        marginal_amplitude_error = (marginal_rms / golden_rms - 1.0) * 100
        self.result("Marginal DUT amplitude error (RMS)", f"{marginal_amplitude_error:.2f}%")

        bad_rms = np.sqrt(np.mean(data["dut_bad"] ** 2))
        bad_amplitude_error = (bad_rms / golden_rms - 1.0) * 100
        self.result("Bad DUT amplitude error (RMS)", f"{bad_amplitude_error:.2f}%")

        results["marginal_amp_error"] = marginal_amplitude_error
        results["bad_amp_error"] = bad_amplitude_error

        # Part 5: Template Matching
        self.subsection("Part 5: Template Matching for Pattern Detection")
        self.info("Using cross-correlation for template matching")

        # Cross-correlation to find best alignment
        def cross_correlation_peak(signal: np.ndarray, template: np.ndarray) -> float:
            """Find maximum cross-correlation value."""
            correlation = np.correlate(signal, template, mode="valid")
            normalized = correlation / (len(template) * np.std(signal) * np.std(template))
            return float(np.max(normalized))

        # Template: one period of golden sine wave
        period_samples = int(data["sample_rate"] / 10e3)
        template = golden[:period_samples]

        for dut_name, dut_signal in duts.items():
            match_score = cross_correlation_peak(dut_signal, template)
            self.result(f"{dut_name} template match", f"{match_score:.6f}")

        # Part 6: Regression Testing
        self.subsection("Part 6: Regression Testing Workflow")
        self.info("Using golden reference for automated regression detection")

        self.info("\n[Production Test Workflow]")
        self.info("  1. Capture golden reference from known-good device")
        self.info("  2. Define pass/fail thresholds (correlation, RMSE, peak error)")
        self.info("  3. Test each production unit against golden reference")
        self.info("  4. Log metrics for statistical process control (SPC)")
        self.info("  5. Flag outliers for detailed analysis")

        self.info("\n[Regression Detection]")
        self.info("  1. Store golden reference for each firmware/hardware revision")
        self.info("  2. Compare new builds against previous golden references")
        self.info("  3. Alert if metrics exceed threshold deltas")
        self.info("  4. Investigate root cause of regressions")

        self.success("Golden reference comparison complete!")

        return results

    def validate(self, results: dict[str, Any]) -> bool:
        """Validate golden reference comparison results."""
        self.info("Validating comparison results...")

        all_valid = True

        # Validate Good DUT metrics
        good_metrics = results["metrics"]["Good DUT"]

        # Good DUT should have high correlation (> 0.99)
        if good_metrics["correlation"] < 0.99:
            self.error(f"Good DUT correlation too low: {good_metrics['correlation']:.6f}")
            all_valid = False
        else:
            self.success(f"Good DUT correlation: {good_metrics['correlation']:.6f}")

        # Good DUT NRMSE should be low (< 5%)
        if good_metrics["nrmse_percent"] > 5.0:
            self.error(f"Good DUT NRMSE too high: {good_metrics['nrmse_percent']:.4f}%")
            all_valid = False
        else:
            self.success(f"Good DUT NRMSE: {good_metrics['nrmse_percent']:.4f}%")

        # Validate Bad DUT metrics
        bad_metrics = results["metrics"]["Bad DUT"]

        # Bad DUT should have lower correlation (relaxed threshold due to noise)
        if bad_metrics["correlation"] > 0.99:
            self.error(f"Bad DUT correlation too high: {bad_metrics['correlation']:.6f}")
            all_valid = False
        else:
            self.success(f"Bad DUT correlation (as expected): {bad_metrics['correlation']:.6f}")

        # Validate pass/fail results
        if not results["pass_fail"]["Good DUT"]:
            self.error("Good DUT should pass but failed")
            all_valid = False
        else:
            self.success("Good DUT correctly passed")

        if results["pass_fail"]["Bad DUT"]:
            self.error("Bad DUT should fail but passed")
            all_valid = False
        else:
            self.success("Bad DUT correctly failed")

        # Validate difference signal analysis
        if results["good_diff_gaussian"] < 95:
            self.warning(f"Good DUT difference not Gaussian: {results['good_diff_gaussian']:.2f}%")
        else:
            self.success(f"Good DUT difference is Gaussian: {results['good_diff_gaussian']:.2f}%")

        # Validate amplitude errors (relaxed tolerances due to noise effects)
        if not validate_approximately(
            results["marginal_amp_error"], -2.0, tolerance=1.5, name="Marginal amplitude error"
        ):
            all_valid = False

        if not validate_approximately(
            results["bad_amp_error"], -15.0, tolerance=3.0, name="Bad amplitude error"
        ):
            all_valid = False

        if all_valid:
            self.success("All golden reference validations passed!")
            self.info("\nNext Steps:")
            self.info("  - Try 18_comparison_testing/02_limit_testing.py for limit masks")
            self.info("  - Explore 18_comparison_testing/03_mask_testing.py for eye diagrams")
            self.info("  - Implement in production test automation")
        else:
            self.error("Some validations failed")

        return all_valid

    def result(self, name: str, value: Any, unit: str = "") -> None:
        """Print a result with optional unit."""
        if unit:
            print(f"  {name}: {value} {unit}")
        else:
            print(f"  {name}: {value}")


if __name__ == "__main__":
    demo: GoldenReferenceDemo = GoldenReferenceDemo()
    success: bool = demo.execute()
    sys.exit(0 if success else 1)
