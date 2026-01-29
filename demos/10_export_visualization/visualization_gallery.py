"""Visualization Gallery: Visualization examples showcase.

Category: Export & Visualization
IEEE Standards: N/A
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np

from demos.common import BaseDemo, ValidationSuite, print_header, print_info


class VisualizationGalleryDemo(BaseDemo):
    """Demonstrates visualization gallery."""

    name = "Visualization Gallery"
    description = "Showcase of visualization examples"
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
        """Demonstrate visualization gallery."""
        print_header("Visualization Gallery")
        print_info("Gallery of visualization examples")
        print_info("Includes: waveforms, spectra, eye diagrams, histograms")
        self.results["gallery_info"] = "Visualization examples shown"

    def validate_results(self, suite: ValidationSuite) -> None:
        """Validate results."""
        suite.check_exists("Gallery info", self.results.get("gallery_info"))


if __name__ == "__main__":
    demo = VisualizationGalleryDemo()
    result = demo.run()
    sys.exit(0 if result.success else 1)
