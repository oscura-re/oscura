"""Supported Formats: Explore all 21+ file formats Oscura loads.

Demonstrates:
- oscura.get_supported_formats() - List all supported formats
- Format categories (oscilloscopes, logic analyzers, automotive, scientific)
- Format detection and loader mapping
- Example usage patterns for each major category

IEEE Standards: N/A (informational reference)

Related Demos:
- 00_getting_started/00_hello_world.py
- 01_data_loading/ (various format-specific demos)
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add demos to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from demos.common import BaseDemo, ValidationSuite, run_demo_main
from demos.common.formatting import print_info, print_result, print_subheader


class SupportedFormatsDemo(BaseDemo):
    """Comprehensive showcase of all supported file formats."""

    name = "Supported Formats"
    description = "Complete guide to all 21+ file formats Oscura supports"
    category = "getting_started"
    capabilities = [
        "oscura.get_supported_formats()",
        "oscura.load() with auto-detection",
    ]
    related_demos = [
        "00_getting_started/00_hello_world.py",
        "01_data_loading/",
    ]

    def generate_data(self) -> None:
        """No test data needed - this is an informational demo."""
        pass

    def run_analysis(self) -> None:
        """Run the supported formats demonstration."""
        import oscura

        print_info("Oscura supports 21+ file formats from test & measurement equipment")
        print_info("All formats are automatically detected - just use oscura.load(filename)")

        # Get all supported formats
        supported_formats = oscura.get_supported_formats()

        print_subheader("Quick Reference")
        print_result("Total supported formats", len(supported_formats))
        print_result("Auto-detection", "Enabled")
        print_result("Multi-channel loading", "Supported")
        print_result("Lazy loading", "Available for large files")

        # Display all formats
        print_subheader("All Supported File Extensions")
        format_list = sorted(supported_formats)
        formats_str = ", ".join(format_list)
        print_info(f"Formats: {formats_str}")

        # Categorize formats
        categories = {
            "Oscilloscopes": [".wfm", ".isf", ".bin", ".csv"],
            "Logic Analyzers": [".sr", ".vcd", ".logicdata"],
            "Automotive": [".blf", ".asc", ".mf4"],
            "Scientific": [".tdms", ".h5", ".hdf5", ".npz", ".wav"],
            "Network": [".pcap", ".pcapng"],
            "RF": [".s2p", ".snp"],
        }

        print_subheader("Formats by Category")
        for category, formats in categories.items():
            available = [fmt for fmt in formats if fmt in supported_formats]
            if available:
                print_result(category, ", ".join(available))

        # Show usage examples
        print_subheader("Usage Examples")

        print_info("\nOscilloscope Format (.wfm):")
        print_info("  trace = oscura.load('capture.wfm')")
        print_info("  print(f'Sample rate: {trace.metadata.sample_rate} Hz')")

        print_info("\nLogic Analyzer Format (.vcd):")
        print_info("  digital = oscura.load('logic.vcd')")
        print_info("  print(f'Digital trace: {len(digital.data)} samples')")

        print_info("\nAutomotive CAN (.blf):")
        print_info("  can_data = oscura.load('vehicle.blf')")
        print_info("  print(f'CAN frames: {len(can_data)}')")

        print_info("\nScientific Format (.tdms):")
        print_info("  data = oscura.load('experiment.tdms')")

        # Store results for validation
        self.results["format_count"] = len(supported_formats)
        self.results["has_wfm"] = ".wfm" in supported_formats
        self.results["has_vcd"] = ".vcd" in supported_formats
        self.results["has_blf"] = ".blf" in supported_formats

    def validate_results(self, suite: ValidationSuite) -> None:
        """Validate the results."""
        suite.check_greater_equal("Format count", self.results["format_count"], 15)
        suite.check_true("Has .wfm support", self.results["has_wfm"])
        suite.check_true("Has .vcd support", self.results["has_vcd"])
        suite.check_true("Has .blf support", self.results["has_blf"])

        if suite.all_passed():
            print_info("\nYou now understand Oscura's format support!")
            print_info("Next steps:")
            print_info("  - Explore 01_data_loading/ for format-specific examples")
            print_info("  - Try loading your own captured data files")


if __name__ == "__main__":
    sys.exit(run_demo_main(SupportedFormatsDemo))
