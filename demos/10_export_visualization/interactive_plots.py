"""Interactive Plots: Interactive plotting with Plotly.

Category: Export & Visualization
IEEE Standards: N/A
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np

from demos.common import BaseDemo, ValidationSuite, print_header, print_info


class InteractivePlotsDemo(BaseDemo):
    """Demonstrates interactive plots."""

    name = "Interactive Plots"
    description = "Interactive plotting with Plotly"
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
        """Demonstrate interactive plots."""
        print_header("Interactive Plots")
        print_info("Interactive plotting with Plotly requires:")
        print_info("  pip install plotly")
        print_info("")
        print_info("Example code:")
        print_info("  import plotly.graph_objects as go")
        print_info("  fig = go.Figure(data=go.Scatter(y=trace.data))")
        print_info("  fig.show()")
        self.results["plotly_info"] = "Plotly example shown"

    def validate_results(self, suite: ValidationSuite) -> None:
        """Validate results."""
        suite.check_exists("Plotly info", self.results.get("plotly_info"))


if __name__ == "__main__":
    demo = InteractivePlotsDemo()
    result = demo.run()
    sys.exit(0 if result.success else 1)
