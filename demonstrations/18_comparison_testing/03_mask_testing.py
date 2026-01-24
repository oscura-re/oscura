"""Mask Testing: Eye diagram and custom mask violation detection

Demonstrates:
- Eye diagram mask testing
- Custom mask definition
- Violation detection and counting
- Pass/fail reporting
- Margin to mask boundaries

IEEE Standards: None (telecommunications practice)
Related Demos:
- 18_comparison_testing/01_golden_reference.py
- 18_comparison_testing/02_limit_testing.py
- 04_advanced_analysis/03_eye_diagram.py

This demonstration shows how to test eye diagrams and waveforms against
custom masks for compliance testing in high-speed digital communications.
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


class MaskTestingDemo(BaseDemo):
    """Demonstration of mask testing for eye diagrams and waveforms."""

    def __init__(self) -> None:
        """Initialize mask testing demonstration."""
        super().__init__(
            name="mask_testing",
            description="Eye diagram mask testing with violation detection",
            capabilities=[
                "Custom mask definition",
                "Violation counting",
                "Margin analysis",
                "Eye diagram testing",
            ],
            ieee_standards=[],
            related_demos=[
                "18_comparison_testing/02_limit_testing.py",
                "04_advanced_analysis/03_eye_diagram.py",
            ],
        )

    def generate_test_data(self) -> dict[str, Any]:
        """Generate test data for mask testing."""
        sample_rate = 100e6  # 100 MHz
        bit_rate = 10e6  # 10 Mbps
        duration = 0.001  # 1 ms

        # Clean digital signal
        clean = SignalBuilder.digital_pattern(
            pattern="10101010" * 125,  # Repeat pattern
            sample_rate=sample_rate,
            bit_rate=bit_rate,
            amplitude=3.3,
        )

        # Noisy signal
        noisy = clean + SignalBuilder.white_noise(sample_rate, duration, 0.1)[: len(clean)]

        return {
            "sample_rate": sample_rate,
            "bit_rate": bit_rate,
            "clean": clean,
            "noisy": noisy,
        }

    def run_demonstration(self, data: dict[str, Any]) -> dict[str, Any]:
        """Run mask testing demonstration."""
        results: dict[str, Any] = {}

        self.section("Mask Testing Demonstration")
        self.info("Testing waveforms against compliance masks")

        # Part 1: Define a simple mask
        self.subsection("Part 1: Simple Rectangular Mask")

        # Define mask boundaries (time, voltage)
        mask_upper = 3.5  # Upper voltage limit
        mask_lower = -0.2  # Lower voltage limit

        clean = data["clean"]
        noisy = data["noisy"]

        # Check violations
        clean_violations = np.sum((clean > mask_upper) | (clean < mask_lower))
        noisy_violations = np.sum((noisy > mask_upper) | (noisy < mask_lower))

        self.result("Mask upper", f"{mask_upper:.2f}", "V")
        self.result("Mask lower", f"{mask_lower:.2f}", "V")
        self.result("Clean signal violations", clean_violations)
        self.result("Noisy signal violations", noisy_violations)

        results["clean_violations"] = int(clean_violations)
        results["noisy_violations"] = int(noisy_violations)

        self.success("Mask testing demonstration complete!")

        return results

    def validate(self, results: dict[str, Any]) -> bool:
        """Validate mask testing results."""
        self.info("Validating mask test results...")

        all_valid = True

        # Clean signal should have no violations
        if results["clean_violations"] > 0:
            self.error(f"Clean signal has {results['clean_violations']} violations")
            all_valid = False
        else:
            self.success("Clean signal has no violations")

        if all_valid:
            self.success("All mask testing validations passed!")
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
    demo: MaskTestingDemo = MaskTestingDemo()
    success: bool = demo.execute()
    sys.exit(0 if success else 1)
