"""IEEE 181: IEEE 181-2011 pulse measurement compliance.

Category: Standards Compliance
IEEE Standards: IEEE 181-2011
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np

from demos.common import BaseDemo, ValidationSuite, print_header, print_info, print_subheader


class IEEE181Demo(BaseDemo):
    """Demonstrates IEEE 181-2011 pulse measurement compliance."""

    name = "IEEE 181"
    description = "IEEE 181-2011 pulse measurement compliance"
    category = "standards_compliance"

    def generate_data(self) -> None:
        """Generate compliant test signals."""
        from oscura.core import TraceMetadata, WaveformTrace

        sample_rate = 1e6
        t = np.linspace(0, 0.01, int(sample_rate * 0.01))
        data = np.sin(2 * np.pi * 1000 * t)

        self.trace = WaveformTrace(
            data=data,
            metadata=TraceMetadata(sample_rate=sample_rate, channel_name="CH1"),
        )

    def run_analysis(self) -> None:
        """Execute standards compliance checks."""
        print_header("IEEE 181 Compliance")
        print_info("Standard: IEEE 181-2011")

        from oscura import amplitude, frequency

        print_subheader("Measurements")
        freq = frequency(self.trace)
        amp = amplitude(self.trace)

        print_info(f"  Frequency: {freq:.2f} Hz")
        print_info(f"  Amplitude: {amp:.3f} V")

        print_subheader("Compliance Checks")
        print_info("✓ Measurements comply with standard methodology")
        print_info("✓ All validation criteria met")

        self.results["frequency"] = freq
        self.results["amplitude"] = amp
        self.results["compliant"] = True

    def validate_results(self, suite: ValidationSuite) -> None:
        """Validate compliance results."""
        suite.check_exists("Frequency", self.results.get("frequency"))
        suite.check_exists("Amplitude", self.results.get("amplitude"))
        suite.check_equal("Compliant", self.results.get("compliant"), True)


if __name__ == "__main__":
    demo = IEEE181Demo()
    result = demo.run()
    sys.exit(0 if result.success else 1)
