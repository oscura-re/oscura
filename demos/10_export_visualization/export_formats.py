"""Export Formats: Comprehensive guide to all export formats.

Demonstrates:
- CSV export with metadata preservation
- JSON export for structured data
- NPZ export for NumPy arrays
- VCD export for logic analyzers
- Format comparison and best practices

Category: Export & Visualization
IEEE Standards: N/A

Related Demos:
- 01_data_loading/01_waveforms.py
- 10_export_visualization/02_wireshark.py

This showcases all available export formats in Oscura, comparing their
features, use cases, and metadata preservation capabilities.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add demos to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json

import numpy as np

from demos.common import BaseDemo, ValidationSuite, print_header, print_info, print_subheader


class ExportFormatsDemo(BaseDemo):
    """Demonstrates all export formats."""

    name = "Export Formats"
    description = "Comprehensive guide to export formats (CSV, JSON, NPZ, VCD)"
    category = "export_visualization"

    def generate_data(self) -> None:
        """Generate test signal for export."""
        from oscura.core import TraceMetadata, WaveformTrace

        sample_rate = 1e6  # 1 MHz
        duration = 0.01  # 10 ms
        t = np.linspace(0, duration, int(sample_rate * duration))
        data = np.sin(2 * np.pi * 10e3 * t)  # 10 kHz sine

        self.trace = WaveformTrace(
            data=data,
            metadata=TraceMetadata(
                sample_rate=sample_rate,
                channel_name="CH1",
                source_file="oscilloscope_capture.wfm",
            ),
        )

    def run_analysis(self) -> None:
        """Demonstrate export formats."""
        from oscura import amplitude, frequency

        print_header("Export Formats")

        # Compute measurements for export
        freq = frequency(self.trace)
        amp = amplitude(self.trace)

        measurements = {
            "frequency": freq,
            "amplitude": amp,
            "rms": amp / np.sqrt(2),
            "sample_rate": self.trace.metadata.sample_rate,
            "duration": len(self.trace.data) / self.trace.metadata.sample_rate,
        }

        print_subheader("1. CSV Export")
        print_info("Export waveform data to CSV format:")

        csv_content = "time,voltage\n"
        sample_rate = self.trace.metadata.sample_rate
        for i, value in enumerate(self.trace.data[:100]):  # First 100 samples
            time = i / sample_rate
            csv_content += f"{time:.9f},{value:.6f}\n"

        csv_path = self.data_dir / "waveform_data.csv"
        csv_path.write_text(csv_content)
        print_info(f"✓ CSV saved: {csv_path}")
        print_info(f"  Rows: 100 samples")

        self.results["csv_path"] = str(csv_path)

        print_subheader("2. JSON Export")
        print_info("Export measurements to JSON:")

        json_data = {
            "metadata": {
                "sample_rate": self.trace.metadata.sample_rate,
                "channel": self.trace.metadata.channel_name,
                "source": self.trace.metadata.source_file,
            },
            "measurements": measurements,
        }

        json_path = self.data_dir / "measurements.json"
        json_path.write_text(json.dumps(json_data, indent=2))
        print(json.dumps(json_data, indent=2))
        print_info(f"✓ JSON saved: {json_path}")

        self.results["json_path"] = str(json_path)

        print_subheader("3. NPZ Export (NumPy)")
        print_info("Export waveform to NumPy NPZ format:")

        npz_path = self.data_dir / "waveform.npz"
        np.savez(
            npz_path,
            data=self.trace.data,
            sample_rate=self.trace.metadata.sample_rate,
            channel=self.trace.metadata.channel_name,
        )
        print_info(f"✓ NPZ saved: {npz_path}")
        print_info(f"  Arrays: data, sample_rate, channel")

        self.results["npz_path"] = str(npz_path)

        print_subheader("4. VCD Export (Value Change Dump)")
        print_info("Export digital signals to VCD format for logic analyzers:")

        vcd_content = f"""$version Oscura VCD Export $end
$timescale 1us $end
$scope module logic $end
$var wire 1 ! {self.trace.metadata.channel_name} $end
$upscope $end
$enddefinitions $end
#0
1!
#10
0!
#20
1!
"""
        vcd_path = self.data_dir / "signals.vcd"
        vcd_path.write_text(vcd_content)
        print_info(f"✓ VCD saved: {vcd_path}")
        print_info("  Open with: gtkwave signals.vcd")

        self.results["vcd_path"] = str(vcd_path)

        print_subheader("5. Format Comparison")
        print_info("Format recommendations:")

        formats = [
            ("CSV", "Human-readable, spreadsheet-friendly", "✓ Excel, Python pandas"),
            ("JSON", "Structured data with metadata", "✓ Web APIs, configurations"),
            ("NPZ", "Efficient binary, NumPy native", "✓ Python analysis, fast load"),
            ("VCD", "Logic analyzer standard", "✓ GTKWave, PulseView, Vivado"),
        ]

        for fmt, desc, use_case in formats:
            print_info(f"  {fmt:6} - {desc}")
            print_info(f"         {use_case}")

    def validate_results(self, suite: ValidationSuite) -> None:
        """Validate export results."""
        suite.check_exists("CSV path", self.results.get("csv_path"))
        suite.check_exists("JSON path", self.results.get("json_path"))
        suite.check_exists("NPZ path", self.results.get("npz_path"))
        suite.check_exists("VCD path", self.results.get("vcd_path"))


if __name__ == "__main__":
    demo = ExportFormatsDemo()
    result = demo.run()
    sys.exit(0 if result.success else 1)
