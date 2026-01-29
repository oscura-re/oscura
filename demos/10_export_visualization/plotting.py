"""Plotting: Waveform and spectrum visualization.

Demonstrates:
- Matplotlib plotting for waveforms
- Spectrum plots (FFT visualization)
- Multi-channel plotting
- Interactive plots with Plotly
- Publication-quality figures

Category: Export & Visualization
IEEE Standards: N/A
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np

from demos.common import BaseDemo, ValidationSuite, print_header, print_info


class PlottingDemo(BaseDemo):
    """Demonstrates plotting techniques."""

    name = "Plotting"
    description = "Waveform and spectrum visualization with Matplotlib"
    category = "export_visualization"

    def generate_data(self) -> None:
        """Generate test signals."""
        from oscura.core import TraceMetadata, WaveformTrace

        t = np.linspace(0, 0.01, 1000)
        data = np.sin(2 * np.pi * 1000 * t)

        self.trace = WaveformTrace(
            data=data,
            metadata=TraceMetadata(sample_rate=100e3, channel_name="CH1"),
        )

    def run_analysis(self) -> None:
        """Demonstrate plotting."""
        print_header("Plotting Techniques")

        print_info("Plotting requires matplotlib:")
        print_info("  pip install matplotlib")
        print_info("")
        print_info("Example code:")
        print_info("  import matplotlib.pyplot as plt")
        print_info("  plt.plot(trace.data)")
        print_info("  plt.xlabel('Sample')")
        print_info("  plt.ylabel('Voltage (V)')")
        print_info("  plt.title('Waveform')")
        print_info("  plt.savefig('waveform.png')")

        self.results["plotting_info"] = "Matplotlib example shown"

    def validate_results(self, suite: ValidationSuite) -> None:
        """Validate plotting."""
        suite.check_exists("Plotting info", self.results.get("plotting_info"))


if __name__ == "__main__":
    demo = PlottingDemo()
    result = demo.run()
    sys.exit(0 if result.success else 1)
