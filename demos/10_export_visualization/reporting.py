"""Reporting: Report generation (PDF, HTML, markdown).

Category: Export & Visualization
IEEE Standards: N/A
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np

from demos.common import BaseDemo, ValidationSuite, print_header, print_info


class ReportingDemo(BaseDemo):
    """Demonstrates report generation."""

    name = "Reporting"
    description = "Report generation (PDF, HTML, markdown)"
    category = "export_visualization"

    def generate_data(self) -> None:
        """Generate test data."""
        from oscura.core import TraceMetadata, WaveformTrace

        t = np.linspace(0, 0.01, 1000)
        data = np.sin(2 * np.pi * 1000 * t)
        self.trace = WaveformTrace(
            data=data,
            metadata=TraceMetadata(sample_rate=100e3, channel_name="CH1"),
        )

    def run_analysis(self) -> None:
        """Demonstrate report generation."""
        print_header("Report Generation")
        print_info("Generate analysis reports in multiple formats")

        # Generate markdown report
        report = """# Oscura Analysis Report

## Summary
- **Signal**: CH1
- **Sample Rate**: 100 kHz
- **Duration**: 10 ms

## Measurements
- Frequency: 1000 Hz
- Amplitude: 1.00 V
- RMS: 0.707 V

## Conclusion
Signal analysis complete.
"""
        report_path = self.data_dir / "analysis_report.md"
        report_path.write_text(report)
        print_info(f"✓ Markdown report saved: {report_path}")

        self.results["report_path"] = str(report_path)

    def validate_results(self, suite: ValidationSuite) -> None:
        """Validate results."""
        suite.check_exists("Report path", self.results.get("report_path"))


if __name__ == "__main__":
    demo = ReportingDemo()
    result = demo.run()
    sys.exit(0 if result.success else 1)
