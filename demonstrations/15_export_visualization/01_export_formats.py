"""Export Formats: Comprehensive Guide to All Export Formats

Demonstrates:
- CSV export with metadata preservation
- JSON export for structured data
- HDF5 export for large datasets
- NPZ export for NumPy arrays
- MATLAB export for .mat files
- Format comparison and best practices
- Metadata preservation across formats

This demonstration showcases all available export formats in Oscura,
comparing their features, use cases, and metadata preservation capabilities.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

# Add demonstrations to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demonstrations.common import BaseDemo
from oscura.core.types import TraceMetadata, WaveformTrace
from oscura.exporters import exporters


class ExportFormatsDemo(BaseDemo):
    """Demonstrate all export formats and their capabilities."""

    def __init__(self) -> None:
        """Initialize export formats demonstration."""
        super().__init__(
            name="export_formats",
            description="Comprehensive guide to all export formats (CSV, JSON, HDF5, NPZ, MATLAB)",
            capabilities=[
                "oscura.exporters.csv",
                "oscura.exporters.json",
                "oscura.exporters.hdf5",
                "oscura.exporters.npz",
                "oscura.exporters.mat",
                "oscura.exporters.metadata_preservation",
            ],
            related_demos=[
                "15_export_visualization/02_wavedrom_timing.py",
                "15_export_visualization/04_report_generation.py",
            ],
        )

    def generate_test_data(self) -> dict[str, Any]:
        """Generate test signals for export.

        Returns:
            Dictionary containing:
            - waveform: WaveformTrace to export
            - measurements: Measurement results dictionary
            - multi_channel: Multiple traces
        """
        # Generate waveform
        sample_rate = 1e6  # 1 MHz
        duration = 0.01  # 10 ms
        frequency = 10e3  # 10 kHz
        num_samples = int(duration * sample_rate)

        t = np.arange(num_samples) / sample_rate
        data = np.sin(2 * np.pi * frequency * t)

        metadata = TraceMetadata(
            sample_rate=sample_rate,
            vertical_scale=0.1,  # 100 mV/div
            vertical_offset=0.0,
            channel_name="CH1",
            source_file="oscilloscope_capture.wfm",
        )

        waveform = WaveformTrace(data=data, metadata=metadata)

        # Generate measurements
        measurements = {
            "rms": float(np.sqrt(np.mean(data**2))),
            "peak": float(np.max(np.abs(data))),
            "frequency": frequency,
            "sample_rate": sample_rate,
            "duration": duration,
        }

        # Generate multi-channel data
        ch2_data = np.cos(2 * np.pi * frequency * t)
        ch2_metadata = TraceMetadata(
            sample_rate=sample_rate,
            channel_name="CH2",
            source_file="oscilloscope_capture.wfm",
        )
        ch2_trace = WaveformTrace(data=ch2_data, metadata=ch2_metadata)

        multi_channel = {"CH1": waveform, "CH2": ch2_trace}

        return {
            "waveform": waveform,
            "measurements": measurements,
            "multi_channel": multi_channel,
            "sample_rate": sample_rate,
        }

    def run_demonstration(self, data: dict[str, Any]) -> dict[str, Any]:
        """Run the export formats demonstration."""
        results: dict[str, Any] = {}
        output_dir = self.get_output_dir()

        self.section("Export Formats Demonstration")
        self.info("Comprehensive guide to exporting data in various formats")

        waveform = data["waveform"]
        measurements = data["measurements"]
        multi_channel = data["multi_channel"]

        # Part 1: CSV Export
        self.subsection("Part 1: CSV Export")
        self.info("Export to CSV format (human-readable, spreadsheet-compatible).")

        csv_path = output_dir / "waveform.csv"
        exporters.csv(waveform, csv_path, include_time=True, precision=9)

        csv_size = csv_path.stat().st_size
        self.result("CSV file created", str(csv_path))
        self.result("File size", f"{csv_size / 1024:.2f}", "KB")

        # Show CSV excerpt
        with open(csv_path) as f:
            lines = f.readlines()[:5]
            self.info("\nCSV excerpt:")
            for line in lines:
                self.info(f"  {line.rstrip()}")

        results["csv_path"] = str(csv_path)
        results["csv_size"] = csv_size

        # Part 2: JSON Export
        self.subsection("Part 2: JSON Export")
        self.info("Export to JSON format (structured data, metadata-rich).")

        json_path = output_dir / "measurements.json"
        exporters.json(measurements, json_path, pretty=True)

        json_size = json_path.stat().st_size
        self.result("JSON file created", str(json_path))
        self.result("File size", f"{json_size / 1024:.2f}", "KB")

        # Show JSON excerpt
        with open(json_path) as f:
            content = f.read()
            self.info("\nJSON content:")
            for line in content.split("\n")[:10]:
                self.info(f"  {line}")

        results["json_path"] = str(json_path)
        results["json_size"] = json_size

        # Part 3: HDF5 Export
        self.subsection("Part 3: HDF5 Export")
        self.info("Export to HDF5 format (efficient, metadata-rich, multi-dimensional).")

        try:
            hdf5_path = output_dir / "waveform.h5"
            exporters.hdf5(waveform, hdf5_path, compression="gzip", compression_opts=4)

            hdf5_size = hdf5_path.stat().st_size
            self.result("HDF5 file created", str(hdf5_path))
            self.result("File size", f"{hdf5_size / 1024:.2f}", "KB")
            self.result("Compression", "gzip (level 4)")

            # Multi-channel HDF5
            hdf5_multi_path = output_dir / "multi_channel.h5"
            exporters.hdf5(multi_channel, hdf5_multi_path)
            self.result("Multi-channel HDF5", str(hdf5_multi_path))

            results["hdf5_path"] = str(hdf5_path)
            results["hdf5_size"] = hdf5_size

        except ImportError:
            self.warning("h5py not installed - skipping HDF5 export")
            self.info("  Install with: pip install h5py")
            results["hdf5_path"] = None

        # Part 4: NPZ Export
        self.subsection("Part 4: NPZ Export")
        self.info("Export to NPZ format (NumPy compressed arrays).")

        npz_path = output_dir / "waveform.npz"
        exporters.npz(waveform, npz_path, compressed=True)

        npz_size = npz_path.stat().st_size
        self.result("NPZ file created", str(npz_path))
        self.result("File size", f"{npz_size / 1024:.2f}", "KB")
        self.result("Compression", "enabled")

        results["npz_path"] = str(npz_path)
        results["npz_size"] = npz_size

        # Part 5: MATLAB Export
        self.subsection("Part 5: MATLAB Export")
        self.info("Export to MATLAB .mat format (MATLAB/Octave compatible).")

        try:
            mat_path = output_dir / "waveform.mat"
            exporters.mat(waveform, mat_path)

            mat_size = mat_path.stat().st_size
            self.result("MAT file created", str(mat_path))
            self.result("File size", f"{mat_size / 1024:.2f}", "KB")
            self.result("MATLAB version", "5.0 (compatible)")

            results["mat_path"] = str(mat_path)
            results["mat_size"] = mat_size

        except ImportError:
            self.warning("scipy not installed - skipping MATLAB export")
            self.info("  Install with: pip install scipy")
            results["mat_path"] = None

        # Part 6: Format Comparison
        self.subsection("Part 6: Format Comparison")
        self.info("Compare file sizes and characteristics across formats.")

        comparison = []

        # CSV
        comparison.append(
            {
                "format": "CSV",
                "size_kb": csv_size / 1024,
                "human_readable": "Yes",
                "metadata": "Comments",
                "compression": "No",
                "use_case": "Spreadsheets, simple analysis",
            }
        )

        # JSON
        comparison.append(
            {
                "format": "JSON",
                "size_kb": json_size / 1024,
                "human_readable": "Yes",
                "metadata": "Full",
                "compression": "No",
                "use_case": "Web APIs, structured data",
            }
        )

        # HDF5
        if results.get("hdf5_path"):
            comparison.append(
                {
                    "format": "HDF5",
                    "size_kb": hdf5_size / 1024,
                    "human_readable": "No",
                    "metadata": "Full",
                    "compression": "Yes",
                    "use_case": "Large datasets, multi-channel",
                }
            )

        # NPZ
        comparison.append(
            {
                "format": "NPZ",
                "size_kb": npz_size / 1024,
                "human_readable": "No",
                "metadata": "Attributes",
                "compression": "Yes",
                "use_case": "NumPy workflows, Python",
            }
        )

        # MATLAB
        if results.get("mat_path"):
            comparison.append(
                {
                    "format": "MATLAB",
                    "size_kb": mat_size / 1024,
                    "human_readable": "No",
                    "metadata": "Partial",
                    "compression": "No",
                    "use_case": "MATLAB/Octave integration",
                }
            )

        self.info("\nFormat comparison table:")
        header = f"  {'Format':<10} {'Size (KB)':<12} {'Human':<8} {'Metadata':<12}"
        header += f" {'Compress':<10} Use Case"
        self.info(header)
        self.info("  " + "-" * 100)
        for fmt in comparison:
            self.info(
                f"  {fmt['format']:<10} {fmt['size_kb']:<12.2f} "
                f"{fmt['human_readable']:<8} {fmt['metadata']:<12} "
                f"{fmt['compression']:<10} {fmt['use_case']}"
            )

        results["comparison"] = comparison

        # Part 7: Best Practices
        self.subsection("Part 7: Best Practices and Recommendations")
        self.info("Format selection guidelines based on use case.")

        best_practices = {
            "Small datasets (<100 KB)": "CSV or JSON - easy to inspect and share",
            "Large datasets (>10 MB)": "HDF5 or NPZ - efficient storage with compression",
            "Multi-channel data": "HDF5 - hierarchical structure, metadata support",
            "Web/API integration": "JSON - standard format, wide compatibility",
            "MATLAB workflows": "MATLAB .mat - native integration",
            "Long-term archival": "HDF5 - self-describing, metadata-rich",
            "Quick inspection": "CSV - open in any text editor",
        }

        self.info("\nRecommendations:")
        for use_case, recommendation in best_practices.items():
            self.info(f"  • {use_case}:")
            self.info(f"    → {recommendation}")

        results["best_practices"] = best_practices

        self.success("Export formats demonstration complete!")

        return results

    def validate(self, results: dict[str, Any]) -> bool:
        """Validate the demonstration results."""
        self.info("Validating export formats...")

        # Validate CSV export
        if "csv_path" not in results:
            self.error("Missing CSV export")
            return False

        csv_path = Path(results["csv_path"])
        if not csv_path.exists():
            self.error(f"CSV file not found: {csv_path}")
            return False

        # Validate JSON export
        if "json_path" not in results:
            self.error("Missing JSON export")
            return False

        json_path = Path(results["json_path"])
        if not json_path.exists():
            self.error(f"JSON file not found: {json_path}")
            return False

        # Validate NPZ export
        if "npz_path" not in results:
            self.error("Missing NPZ export")
            return False

        npz_path = Path(results["npz_path"])
        if not npz_path.exists():
            self.error(f"NPZ file not found: {npz_path}")
            return False

        # Validate comparison table
        if "comparison" not in results:
            self.error("Missing format comparison")
            return False

        if len(results["comparison"]) < 3:
            self.error("Insufficient formats in comparison")
            return False

        self.success("All export format validations passed!")
        self.info("\nKey Takeaways:")
        self.info("  - CSV: Human-readable, spreadsheet-compatible")
        self.info("  - JSON: Structured data, web APIs, full metadata")
        self.info("  - HDF5: Large datasets, compression, multi-channel")
        self.info("  - NPZ: NumPy workflows, compressed arrays")
        self.info("  - MATLAB: Native MATLAB/Octave integration")
        self.info("\nNext steps:")
        self.info("  - Try 15_export_visualization/04_report_generation.py for PDF/HTML reports")

        return True


if __name__ == "__main__":
    demo: ExportFormatsDemo = ExportFormatsDemo()
    success: bool = demo.execute()
    sys.exit(0 if success else 1)
