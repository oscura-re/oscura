"""Regression Testing: Detect changes across software/hardware revisions

Demonstrates:
- Regression test suite creation
- Baseline capture and comparison
- Automated regression detection
- Trend analysis across revisions

IEEE Standards: None (software engineering practice)
Related Demos:
- 18_comparison_testing/01_golden_reference.py
- 18_comparison_testing/02_limit_testing.py

This demonstration shows how to implement automated regression testing
for detecting unintended changes across firmware or hardware revisions.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

# Add demonstrations to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demonstrations.common import BaseDemo
from tests.fixtures.signal_builders import SignalBuilder


class RegressionTestingDemo(BaseDemo):
    """Demonstration of regression testing workflow."""

    def __init__(self) -> None:
        """Initialize regression testing demonstration."""
        super().__init__(
            name="regression_testing",
            description="Automated regression detection across revisions",
            capabilities=[
                "Baseline creation",
                "Revision comparison",
                "Trend analysis",
                "Automated detection",
            ],
            ieee_standards=[],
            related_demos=[
                "18_comparison_testing/01_golden_reference.py",
            ],
        )

    def generate_test_data(self) -> dict[str, Any]:
        """Generate baseline and revision signals."""
        sample_rate = 1e6
        duration = 0.001

        # Baseline (v1.0)
        baseline = SignalBuilder.sine_wave(
            frequency=10e3, sample_rate=sample_rate, duration=duration, amplitude=1.0
        )

        # Generate noise arrays matching baseline length
        n_samples = len(baseline)

        # v1.1 - no change
        noise_v1_1 = np.random.normal(0, 0.005, n_samples)
        v1_1 = baseline + noise_v1_1

        # v1.2 - small amplitude regression
        noise_v1_2 = np.random.normal(0, 0.005, n_samples)
        v1_2 = baseline * 0.98 + noise_v1_2

        # v2.0 - significant change (intentional redesign)
        noise_v2_0 = np.random.normal(0, 0.01, n_samples)
        v2_0 = baseline * 1.2 + noise_v2_0

        return {
            "baseline": baseline,
            "v1_1": v1_1,
            "v1_2": v1_2,
            "v2_0": v2_0,
        }

    def run_demonstration(self, data: dict[str, Any]) -> dict[str, Any]:
        """Run regression testing demonstration."""
        results: dict[str, Any] = {}

        self.section("Regression Testing Demonstration")
        self.info("Detecting changes across firmware/hardware revisions")

        baseline = data["baseline"]

        # Part 1: Baseline Capture
        self.subsection("Part 1: Baseline Capture (v1.0)")
        baseline_amplitude = np.max(baseline)
        baseline_rms = np.sqrt(np.mean(baseline**2))

        self.result("Baseline amplitude", f"{baseline_amplitude:.6f}", "V")
        self.result("Baseline RMS", f"{baseline_rms:.6f}", "V")

        results["baseline_amplitude"] = float(baseline_amplitude)
        results["baseline_rms"] = float(baseline_rms)

        # Part 2: Revision Comparison
        self.subsection("Part 2: Revision Comparison")

        revisions = {
            "v1.1": data["v1_1"],
            "v1.2": data["v1_2"],
            "v2.0": data["v2_0"],
        }

        # Regression threshold: ±5% change
        regression_threshold = 0.05

        regression_results = {}
        for version, signal in revisions.items():
            amplitude = np.max(signal)
            rms = np.sqrt(np.mean(signal**2))

            amp_change = (amplitude - baseline_amplitude) / baseline_amplitude
            rms_change = (rms - baseline_rms) / baseline_rms

            is_regression = (
                abs(amp_change) > regression_threshold or abs(rms_change) > regression_threshold
            )

            regression_results[version] = {
                "amplitude": float(amplitude),
                "rms": float(rms),
                "amp_change_percent": float(amp_change * 100),
                "rms_change_percent": float(rms_change * 100),
                "is_regression": is_regression,
            }

            self.info(f"\n{version}:")
            self.result("  Amplitude change", f"{amp_change * 100:+.2f}%")
            self.result("  RMS change", f"{rms_change * 100:+.2f}%")
            self.result("  Regression detected", "YES" if is_regression else "NO")

        results["regression_results"] = regression_results

        self.success("Regression testing demonstration complete!")

        return results

    def validate(self, results: dict[str, Any]) -> bool:
        """Validate regression testing results."""
        self.info("Validating regression detection...")

        all_valid = True

        # v1.1 should not be a regression
        if results["regression_results"]["v1.1"]["is_regression"]:
            self.error("v1.1 incorrectly flagged as regression")
            all_valid = False
        else:
            self.success("v1.1 correctly not flagged")

        # v1.2 should not be flagged (within 5%)
        if results["regression_results"]["v1.2"]["is_regression"]:
            self.warning("v1.2 flagged as regression (marginal case)")

        # v2.0 should be flagged
        if not results["regression_results"]["v2.0"]["is_regression"]:
            self.error("v2.0 should be flagged as regression")
            all_valid = False
        else:
            self.success("v2.0 correctly flagged as regression")

        if all_valid:
            self.success("All regression testing validations passed!")
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
    demo: RegressionTestingDemo = RegressionTestingDemo()
    success: bool = demo.execute()
    sys.exit(0 if success else 1)
