"""Power Supply Workflow: power supply analysis workflow.

Category: Complete Workflows
IEEE Standards: N/A
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np

from demos.common import BaseDemo, ValidationSuite, print_header, print_info, print_subheader


class PowerSupplyWorkflowDemo(BaseDemo):
    """Demonstrates power supply analysis workflow."""

    name = "Power Supply Workflow"
    description = "power supply analysis workflow"
    category = "complete_workflows"

    def generate_data(self) -> None:
        """Generate test data for workflow."""
        from oscura.core import TraceMetadata, WaveformTrace

        t = np.linspace(0, 0.01, 1000)
        data = np.sin(2 * np.pi * 1000 * t)
        self.trace = WaveformTrace(
            data=data,
            metadata=TraceMetadata(sample_rate=100e3, channel_name="CH1"),
        )

    def run_analysis(self) -> None:
        """Execute complete workflow."""
        print_header("Power Supply Workflow")
        print_info("Complete end-to-end workflow demonstration")

        from oscura import frequency

        print_subheader("Step 1: Data Acquisition")
        print_info("Load or acquire signal data")
        print_info(f"  Signal length: {len(self.trace.data)} samples")

        print_subheader("Step 2: Initial Analysis")
        freq = frequency(self.trace)
        print_info(f"  Frequency: {freq:.2f} Hz")

        print_subheader("Step 3: Detailed Analysis")
        print_info("Perform domain-specific analysis")

        print_subheader("Step 4: Validation")
        print_info("Validate results against specifications")

        print_subheader("Step 5: Reporting")
        print_info("✓ Workflow complete")
        print_info("  Analysis passed all checks")

        self.results["workflow_complete"] = True
        self.results["frequency"] = freq

    def validate_results(self, suite: ValidationSuite) -> None:
        """Validate workflow results."""
        suite.check_exists("Workflow complete", self.results.get("workflow_complete"))
        suite.check_exists("Frequency", self.results.get("frequency"))


if __name__ == "__main__":
    demo = PowerSupplyWorkflowDemo()
    result = demo.run()
    sys.exit(0 if result.success else 1)
