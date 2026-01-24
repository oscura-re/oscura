"""Limit Testing: Test signals against limit masks and tolerance bands

Demonstrates:
- Limit mask testing (upper/lower bounds)
- Tolerance band checking
- Margin analysis (how close to limits)
- Compliance reporting
- Multi-level limit testing (warning/alarm/critical)

IEEE Standards: None (general test practice)
Related Demos:
- 18_comparison_testing/01_golden_reference.py
- 18_comparison_testing/03_mask_testing.py
- 02_basic_analysis/01_waveform_measurements.py

This demonstration shows how to test signals against predefined limit masks
for compliance testing, margin analysis, and production qualification.
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


class LimitTestingDemo(BaseDemo):
    """Demonstration of limit mask and tolerance band testing."""

    def __init__(self) -> None:
        """Initialize limit testing demonstration."""
        super().__init__(
            name="limit_testing",
            description="Test signals against limit masks with margin analysis",
            capabilities=[
                "Upper/lower limit masks",
                "Tolerance band checking",
                "Margin calculation",
                "Violation detection",
                "Multi-level limits",
            ],
            ieee_standards=[],
            related_demos=[
                "18_comparison_testing/01_golden_reference.py",
                "18_comparison_testing/03_mask_testing.py",
            ],
        )

    def generate_test_data(self) -> dict[str, Any]:
        """Generate test signals with various limit scenarios."""
        sample_rate = 1e6
        duration = 0.001

        # Nominal signal (within limits)
        nominal = SignalBuilder.sine_wave(
            frequency=10e3, sample_rate=sample_rate, duration=duration, amplitude=0.8
        )

        # Marginal signal (close to limits)
        marginal = SignalBuilder.sine_wave(
            frequency=10e3, sample_rate=sample_rate, duration=duration, amplitude=0.95
        )

        # Violating signal (exceeds limits)
        violating = SignalBuilder.sine_wave(
            frequency=10e3, sample_rate=sample_rate, duration=duration, amplitude=1.2
        )

        # Transient violation (brief excursion)
        transient = SignalBuilder.sine_wave(
            frequency=10e3, sample_rate=sample_rate, duration=duration, amplitude=0.8
        )
        # Add a glitch that violates limits
        glitch_start = len(transient) // 2
        glitch_end = glitch_start + 10
        transient[glitch_start:glitch_end] = 1.5

        return {
            "sample_rate": sample_rate,
            "nominal": nominal,
            "marginal": marginal,
            "violating": violating,
            "transient": transient,
        }

    def run_demonstration(self, data: dict[str, Any]) -> dict[str, Any]:
        """Run limit testing demonstration."""
        results: dict[str, Any] = {}

        self.section("Limit Mask Testing Demonstration")
        self.info("Testing signals against predefined limit masks")

        # Part 1: Simple Upper/Lower Limits
        self.subsection("Part 1: Simple Upper/Lower Limit Testing")

        # Define absolute limits
        upper_limit = 1.0
        lower_limit = -1.0

        self.result("Upper limit", f"{upper_limit:.2f}", "V")
        self.result("Lower limit", f"{lower_limit:.2f}", "V")

        def check_limits(signal: np.ndarray, upper: float, lower: float) -> dict[str, Any]:
            """Check signal against limits and return statistics."""
            violations_upper = signal > upper
            violations_lower = signal < lower

            num_violations_upper = np.sum(violations_upper)
            num_violations_lower = np.sum(violations_lower)
            total_violations = num_violations_upper + num_violations_lower

            if total_violations > 0:
                max_excursion_upper = (
                    np.max(signal[violations_upper] - upper) if num_violations_upper > 0 else 0
                )
                max_excursion_lower = (
                    np.max(lower - signal[violations_lower]) if num_violations_lower > 0 else 0
                )
            else:
                max_excursion_upper = 0
                max_excursion_lower = 0

            # Calculate margin (closest approach to limit)
            margin_upper = upper - np.max(signal)
            margin_lower = np.min(signal) - lower
            worst_margin = min(margin_upper, margin_lower)

            return {
                "violations_upper": int(num_violations_upper),
                "violations_lower": int(num_violations_lower),
                "total_violations": int(total_violations),
                "max_excursion_upper": float(max_excursion_upper),
                "max_excursion_lower": float(max_excursion_lower),
                "margin_upper": float(margin_upper),
                "margin_lower": float(margin_lower),
                "worst_margin": float(worst_margin),
                "pass": total_violations == 0,
            }

        # Test each signal
        signals = {
            "Nominal": data["nominal"],
            "Marginal": data["marginal"],
            "Violating": data["violating"],
            "Transient": data["transient"],
        }

        limit_results = {}
        for name, signal in signals.items():
            result = check_limits(signal, upper_limit, lower_limit)
            limit_results[name] = result

            self.info(f"\n{name} Signal:")
            self.result("  Status", "PASS" if result["pass"] else "FAIL")
            self.result("  Upper violations", result["violations_upper"])
            self.result("  Lower violations", result["violations_lower"])
            self.result("  Worst margin", f"{result['worst_margin'] * 1000:.2f}", "mV")

            if not result["pass"]:
                self.result(
                    "  Max upper excursion", f"{result['max_excursion_upper'] * 1000:.2f}", "mV"
                )
                self.result(
                    "  Max lower excursion", f"{result['max_excursion_lower'] * 1000:.2f}", "mV"
                )

        results["limit_results"] = limit_results

        # Part 2: Tolerance Band Testing
        self.subsection("Part 2: Tolerance Band Testing")
        self.info("Testing signal against ideal waveform ± tolerance")

        # Ideal reference
        ideal = SignalBuilder.sine_wave(
            frequency=10e3, sample_rate=data["sample_rate"], duration=0.001, amplitude=1.0
        )

        # Tolerance band (±10%)
        tolerance_percent = 10
        upper_band = ideal * (1 + tolerance_percent / 100)
        lower_band = ideal * (1 - tolerance_percent / 100)

        self.result("Tolerance", f"±{tolerance_percent}%")

        def check_tolerance_band(
            signal: np.ndarray, upper_band: np.ndarray, lower_band: np.ndarray
        ) -> dict[str, Any]:
            """Check if signal stays within tolerance band."""
            violations = (signal > upper_band) | (signal < lower_band)
            num_violations = np.sum(violations)
            violation_percent = (num_violations / len(signal)) * 100

            return {
                "num_violations": int(num_violations),
                "violation_percent": float(violation_percent),
                "pass": num_violations == 0,
            }

        # Test marginal signal against tolerance band
        band_result = check_tolerance_band(data["marginal"], upper_band, lower_band)

        self.result("Violations", band_result["num_violations"])
        self.result("Violation percentage", f"{band_result['violation_percent']:.4f}%")
        self.result("Status", "PASS" if band_result["pass"] else "FAIL")

        results["tolerance_band_result"] = band_result

        # Part 3: Margin Analysis
        self.subsection("Part 3: Margin Analysis")
        self.info("Quantifying how close signals are to limits")

        # Define margin categories
        self.info("\nMargin Categories:")
        self.info("  Excellent: margin > 30%")
        self.info("  Good:      margin 15-30%")
        self.info("  Fair:      margin 5-15%")
        self.info("  Poor:      margin < 5%")
        self.info("  Fail:      margin < 0 (violation)")

        def categorize_margin(margin: float) -> str:
            """Categorize margin health."""
            margin_percent = (margin / upper_limit) * 100
            if margin < 0:
                return "FAIL"
            elif margin_percent < 5:
                return "Poor"
            elif margin_percent < 15:
                return "Fair"
            elif margin_percent < 30:
                return "Good"
            else:
                return "Excellent"

        margin_analysis = {}
        for name, result in limit_results.items():
            category = categorize_margin(result["worst_margin"])
            margin_percent = (result["worst_margin"] / upper_limit) * 100
            margin_analysis[name] = {
                "margin": result["worst_margin"],
                "margin_percent": margin_percent,
                "category": category,
            }

            self.result(f"{name}", f"{margin_percent:+.2f}% ({category})")

        results["margin_analysis"] = margin_analysis

        # Part 4: Multi-Level Limits
        self.subsection("Part 4: Multi-Level Limit Testing")
        self.info("Warning/Alarm/Critical threshold levels")

        # Define multi-level limits
        critical_upper = 1.0  # Hard limit
        alarm_upper = 0.9  # 90% of limit
        warning_upper = 0.8  # 80% of limit

        self.result("Critical limit", f"{critical_upper:.2f}", "V")
        self.result("Alarm limit", f"{alarm_upper:.2f}", "V")
        self.result("Warning limit", f"{warning_upper:.2f}", "V")

        def check_multi_level(signal: np.ndarray) -> str:
            """Check signal against multi-level limits."""
            max_val = np.max(signal)

            if max_val > critical_upper:
                return "CRITICAL"
            elif max_val > alarm_upper:
                return "ALARM"
            elif max_val > warning_upper:
                return "WARNING"
            else:
                return "NORMAL"

        multi_level_results = {}
        for name, signal in signals.items():
            level = check_multi_level(signal)
            multi_level_results[name] = level
            self.result(f"{name}", level)

        results["multi_level_results"] = multi_level_results

        # Part 5: Compliance Reporting
        self.subsection("Part 5: Compliance Report Summary")

        total_tests = len(signals)
        passed_tests = sum(1 for r in limit_results.values() if r["pass"])
        failed_tests = total_tests - passed_tests
        pass_rate = (passed_tests / total_tests) * 100

        self.result("Total tests", total_tests)
        self.result("Passed", passed_tests)
        self.result("Failed", failed_tests)
        self.result("Pass rate", f"{pass_rate:.1f}%")

        results["compliance_summary"] = {
            "total": total_tests,
            "passed": passed_tests,
            "failed": failed_tests,
            "pass_rate": pass_rate,
        }

        self.success("Limit testing complete!")
        self.info("\nKey Capabilities:")
        self.info("  - Absolute limit testing (upper/lower bounds)")
        self.info("  - Tolerance band checking (reference ± tolerance)")
        self.info("  - Margin analysis (quantify proximity to limits)")
        self.info("  - Multi-level limits (warning/alarm/critical)")
        self.info("  - Compliance reporting (pass rate, violations)")

        return results

    def validate(self, results: dict[str, Any]) -> bool:
        """Validate limit testing results."""
        self.info("Validating limit test results...")

        all_valid = True

        # Nominal signal should pass
        if not results["limit_results"]["Nominal"]["pass"]:
            self.error("Nominal signal should pass")
            all_valid = False
        else:
            self.success("Nominal signal passed as expected")

        # Violating signal should fail
        if results["limit_results"]["Violating"]["pass"]:
            self.error("Violating signal should fail")
            all_valid = False
        else:
            self.success("Violating signal failed as expected")

        # Transient signal should fail (has glitch)
        if results["limit_results"]["Transient"]["pass"]:
            self.error("Transient signal should fail")
            all_valid = False
        else:
            self.success("Transient signal failed as expected")

        # Marginal signal should have poor margin
        if results["margin_analysis"]["Marginal"]["category"] not in ["Poor", "Fair"]:
            self.warning(
                f"Marginal signal margin: {results['margin_analysis']['Marginal']['category']}"
            )

        # Verify multi-level results
        if results["multi_level_results"]["Violating"] != "CRITICAL":
            self.error(
                f"Violating signal should be CRITICAL, got {results['multi_level_results']['Violating']}"
            )
            all_valid = False
        else:
            self.success("Violating signal correctly flagged as CRITICAL")

        if all_valid:
            self.success("All limit testing validations passed!")
            self.info("\nNext Steps:")
            self.info("  - Apply to production test automation")
            self.info("  - Try 18_comparison_testing/03_mask_testing.py for eye diagram masks")
            self.info("  - Implement statistical process control (SPC)")
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
    demo: LimitTestingDemo = LimitTestingDemo()
    success: bool = demo.execute()
    sys.exit(0 if success else 1)
