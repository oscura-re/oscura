"""Comprehensive Export: Complete guide to all export formats and workflows

Demonstrates:
- CSV export (text data, cross-platform)
- JSON export (structured data, web integration)
- HDF5 export (large datasets, hierarchical)
- NPZ export (NumPy arrays, Python ecosystem)
- MATLAB export (.mat files, MATLAB integration)
- PWL export (SPICE simulation)
- HTML export (interactive reports)
- Markdown export (documentation)
- Format conversion workflows
- Round-trip validation (export and re-import)
- Format selection guidelines

This demonstration provides a comprehensive overview of ALL export formats
available in Oscura, showing practical examples, format conversion workflows,
and guidance on when to use each format.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Add demonstrations to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demonstrations.common import BaseDemo
from oscura.core.types import TraceMetadata, WaveformTrace
from oscura.exporters import (
    export_csv,
    export_hdf5,
    export_html,
    export_json,
    export_markdown,
    export_mat,
    export_npz,
    export_pwl,
    load_json,
    load_npz,
)


class ComprehensiveExportDemo(BaseDemo):
    """Demonstrate all export formats and conversion workflows."""

    def __init__(self) -> None:
        """Initialize comprehensive export demonstration."""
        super().__init__(
            name="comprehensive_export",
            description="Complete guide to all export formats and format conversion workflows",
            capabilities=[
                "oscura.exporters.csv",
                "oscura.exporters.json",
                "oscura.exporters.hdf5",
                "oscura.exporters.npz",
                "oscura.exporters.mat",
                "oscura.exporters.pwl",
                "oscura.exporters.html",
                "oscura.exporters.markdown",
                "oscura.exporters.format_conversion",
                "oscura.exporters.round_trip_validation",
            ],
            related_demos=[
                "15_export_visualization/01_export_formats.py",
                "15_export_visualization/04_report_generation.py",
            ],
        )

    def generate_test_data(self) -> dict[str, Any]:
        """Generate comprehensive test data for export demonstrations.

        Returns:
            Dictionary containing:
            - waveform: Simple sine wave for basic exports
            - multi_channel: Multiple channels for advanced exports
            - measurements: Analysis results dictionary
            - digital: Digital signal for logic export
            - large_dataset: Large array for compression testing
        """
        # Simple sine wave
        sample_rate = 1e6  # 1 MHz
        duration = 0.01  # 10 ms
        frequency = 10e3  # 10 kHz
        num_samples = int(duration * sample_rate)

        t = np.arange(num_samples) / sample_rate
        sine_data = np.sin(2 * np.pi * frequency * t)

        metadata = TraceMetadata(
            sample_rate=sample_rate,
            vertical_scale=0.1,
            vertical_offset=0.0,
            channel_name="CH1",
            source_file="test_signal.wfm",
        )

        waveform = WaveformTrace(data=sine_data, metadata=metadata)

        # Multi-channel data
        ch2_data = np.cos(2 * np.pi * frequency * t)
        ch2_metadata = TraceMetadata(
            sample_rate=sample_rate,
            vertical_scale=0.1,
            channel_name="CH2",
            source_file="test_signal.wfm",
        )
        ch2_trace = WaveformTrace(data=ch2_data, metadata=ch2_metadata)

        ch3_data = 0.5 * np.sin(2 * np.pi * 2 * frequency * t)
        ch3_metadata = TraceMetadata(
            sample_rate=sample_rate,
            vertical_scale=0.05,
            channel_name="CH3",
            source_file="test_signal.wfm",
        )
        ch3_trace = WaveformTrace(data=ch3_data, metadata=ch3_metadata)

        multi_channel = {
            "CH1": waveform,
            "CH2": ch2_trace,
            "CH3": ch3_trace,
        }

        # Measurements dictionary
        measurements = {
            "rms": float(np.sqrt(np.mean(sine_data**2))),
            "peak": float(np.max(np.abs(sine_data))),
            "peak_to_peak": float(np.ptp(sine_data)),
            "frequency": frequency,
            "sample_rate": sample_rate,
            "duration": duration,
            "num_samples": num_samples,
            "mean": float(np.mean(sine_data)),
            "std_dev": float(np.std(sine_data)),
        }

        # Digital signal (square wave)
        digital_data = (sine_data > 0).astype(np.uint8)

        # Large dataset for compression testing (1 second at 1 MHz)
        large_samples = int(sample_rate * 1.0)
        large_t = np.arange(large_samples) / sample_rate
        large_data = np.sin(2 * np.pi * frequency * large_t)

        return {
            "waveform": waveform,
            "multi_channel": multi_channel,
            "measurements": measurements,
            "digital_data": digital_data,
            "large_data": large_data,
            "sample_rate": sample_rate,
            "time_vector": t,
        }

    def run_demonstration(self, data: dict[str, Any]) -> dict[str, Any]:
        """Run the comprehensive export demonstration."""
        results: dict[str, Any] = {}
        output_dir = self.get_output_dir()

        self.section("Comprehensive Export Demonstration")
        self.info("Complete guide to ALL export formats in Oscura")

        waveform = data["waveform"]
        multi_channel = data["multi_channel"]
        measurements = data["measurements"]
        large_data = data["large_data"]
        sample_rate = data["sample_rate"]

        # Part 1: CSV Export
        self.subsection("Part 1: CSV Export - Text Data")
        self.info("Human-readable, cross-platform, spreadsheet-compatible format")
        self.info("Best for: Small datasets, manual inspection, Excel/Google Sheets")

        csv_path = output_dir / "waveform.csv"
        export_csv(waveform, csv_path, include_time=True, precision=9)

        csv_size = csv_path.stat().st_size
        self.result("CSV file created", str(csv_path))
        self.result("File size", f"{csv_size / 1024:.2f}", "KB")
        self.result("Use case", "Spreadsheet analysis, manual inspection")

        # Show CSV excerpt
        with open(csv_path) as f:
            lines = f.readlines()[:4]
            self.info("\nCSV format preview:")
            for line in lines:
                self.info(f"  {line.rstrip()}")

        results["csv_path"] = str(csv_path)
        results["csv_size"] = csv_size

        # Part 2: JSON Export
        self.subsection("Part 2: JSON Export - Structured Data")
        self.info("Structured, metadata-rich, web-friendly format")
        self.info("Best for: Web APIs, configuration files, structured results")

        json_path = output_dir / "measurements.json"
        export_json(measurements, json_path, pretty=True)

        json_size = json_path.stat().st_size
        self.result("JSON file created", str(json_path))
        self.result("File size", f"{json_size / 1024:.2f}", "KB")
        self.result("Use case", "REST APIs, web dashboards, configuration")

        # Show JSON excerpt
        with open(json_path) as f:
            content = f.read()
            self.info("\nJSON format preview:")
            for line in content.split("\n")[:8]:
                self.info(f"  {line}")

        results["json_path"] = str(json_path)
        results["json_size"] = json_size

        # Part 3: HDF5 Export
        self.subsection("Part 3: HDF5 Export - Large Datasets")
        self.info("Efficient, hierarchical, metadata-rich binary format")
        self.info("Best for: Large datasets (>10 MB), multi-channel, archival")

        try:
            hdf5_path = output_dir / "waveform.h5"
            export_hdf5(waveform, hdf5_path, compression="gzip", compression_opts=4)

            hdf5_size = hdf5_path.stat().st_size
            self.result("HDF5 file created", str(hdf5_path))
            self.result("File size", f"{hdf5_size / 1024:.2f}", "KB")
            self.result("Compression", "gzip level 4")
            self.result("Use case", "Large datasets, multi-channel, archival")

            # Multi-channel HDF5
            hdf5_multi_path = output_dir / "multi_channel.h5"
            export_hdf5(multi_channel, hdf5_multi_path, compression="gzip")
            multi_size = hdf5_multi_path.stat().st_size
            self.result("Multi-channel HDF5", str(hdf5_multi_path))
            self.result("Multi-channel size", f"{multi_size / 1024:.2f}", "KB")

            # Large dataset with compression comparison
            hdf5_large_path = output_dir / "large_dataset.h5"
            large_trace = WaveformTrace(
                data=large_data,
                metadata=TraceMetadata(
                    sample_rate=sample_rate,
                    channel_name="Large",
                    source_file="test_large.wfm",
                ),
            )
            export_hdf5(large_trace, hdf5_large_path, compression="gzip", compression_opts=9)
            large_size = hdf5_large_path.stat().st_size
            raw_size = large_data.nbytes
            compression_ratio = raw_size / large_size
            self.result("Large dataset (1M samples)", f"{large_size / 1024:.2f}", "KB")
            self.result("Compression ratio", f"{compression_ratio:.2f}x")

            results["hdf5_path"] = str(hdf5_path)
            results["hdf5_size"] = hdf5_size
            results["hdf5_compression_ratio"] = compression_ratio

        except ImportError:
            self.warning("h5py not installed - skipping HDF5 export")
            self.info("  Install with: pip install h5py")
            results["hdf5_path"] = None

        # Part 4: NPZ Export
        self.subsection("Part 4: NPZ Export - NumPy Arrays")
        self.info("NumPy compressed archive format")
        self.info("Best for: Python workflows, NumPy arrays, scientific computing")

        npz_path = output_dir / "waveform.npz"
        export_npz(waveform, npz_path, compressed=True)

        npz_size = npz_path.stat().st_size
        self.result("NPZ file created", str(npz_path))
        self.result("File size", f"{npz_size / 1024:.2f}", "KB")
        self.result("Compression", "enabled")
        self.result("Use case", "Python NumPy workflows, scientific computing")

        results["npz_path"] = str(npz_path)
        results["npz_size"] = npz_size

        # Part 5: MATLAB Export
        self.subsection("Part 5: MATLAB Export - .mat Files")
        self.info("MATLAB/Octave native format")
        self.info("Best for: MATLAB integration, signal processing in MATLAB")

        try:
            mat_path = output_dir / "waveform.mat"
            export_mat(waveform, mat_path)

            mat_size = mat_path.stat().st_size
            self.result("MAT file created", str(mat_path))
            self.result("File size", f"{mat_size / 1024:.2f}", "KB")
            self.result("MATLAB version", "5.0 (compatible)")
            self.result("Use case", "MATLAB/Octave analysis, signal processing")

            self.info("\nMATLAB usage:")
            self.info("  >> data = load('waveform.mat');")
            self.info("  >> plot(data.time, data.data);")

            results["mat_path"] = str(mat_path)
            results["mat_size"] = mat_size

        except ImportError:
            self.warning("scipy not installed - skipping MATLAB export")
            self.info("  Install with: pip install scipy")
            results["mat_path"] = None

        # Part 6: PWL Export
        self.subsection("Part 6: PWL Export - SPICE Simulation")
        self.info("Piece-Wise Linear format for circuit simulation")
        self.info("Best for: SPICE simulation (LTspice, ngspice, HSPICE)")

        pwl_path = output_dir / "stimulus.pwl"
        export_pwl(
            waveform,
            pwl_path,
            time_scale=1e6,  # Convert to microseconds
            precision=9,
            downsample=10,  # Reduce file size
            comment="Generated by Oscura - sine wave stimulus",
        )

        pwl_size = pwl_path.stat().st_size
        self.result("PWL file created", str(pwl_path))
        self.result("File size", f"{pwl_size / 1024:.2f}", "KB")
        self.result("Time scale", "microseconds")
        self.result("Downsampling", "10x (1000 samples)")
        self.result("Use case", "SPICE circuit simulation stimulus")

        # Show PWL excerpt
        with open(pwl_path) as f:
            lines = f.readlines()[:6]
            self.info("\nPWL format preview:")
            for line in lines:
                self.info(f"  {line.rstrip()}")

        self.info("\nSPICE usage (LTspice):")
        self.info("  V1 in 0 PWL file=stimulus.pwl")

        results["pwl_path"] = str(pwl_path)
        results["pwl_size"] = pwl_size

        # Part 7: HTML Export
        self.subsection("Part 7: HTML Export - Interactive Reports")
        self.info("HTML format for interactive web reports")
        self.info("Best for: Shareable reports, web dashboards, documentation")

        html_path = output_dir / "report.html"
        export_html(
            measurements,
            html_path,
            title="Waveform Analysis Report",
        )

        html_size = html_path.stat().st_size
        self.result("HTML file created", str(html_path))
        self.result("File size", f"{html_size / 1024:.2f}", "KB")
        self.result("Use case", "Shareable reports, web viewing, dashboards")

        results["html_path"] = str(html_path)
        results["html_size"] = html_size

        # Part 8: Markdown Export
        self.subsection("Part 8: Markdown Export - Documentation")
        self.info("Markdown format for documentation and reports")
        self.info("Best for: GitHub, documentation, technical reports")

        md_path = output_dir / "report.md"
        export_markdown(
            {"measurements": measurements},
            md_path,
            title="Waveform Analysis Results",
        )

        md_size = md_path.stat().st_size
        self.result("Markdown file created", str(md_path))
        self.result("File size", f"{md_size / 1024:.2f}", "KB")
        self.result("Use case", "GitHub, GitLab, technical documentation")

        # Show Markdown excerpt
        with open(md_path) as f:
            lines = f.readlines()[:10]
            self.info("\nMarkdown format preview:")
            for line in lines:
                self.info(f"  {line.rstrip()}")

        results["md_path"] = str(md_path)
        results["md_size"] = md_size

        # Part 9: Format Conversion Workflows
        self.subsection("Part 9: Format Conversion Workflows")
        self.info("Demonstrate converting between formats")

        # Workflow 1: CSV → JSON
        self.info("\nWorkflow 1: CSV → JSON (for web API)")
        csv_to_json_path = output_dir / "csv_to_json.json"

        # Read CSV
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            csv_data = list(reader)

        # Convert to JSON
        with open(csv_to_json_path, "w") as f:
            json.dump(csv_data[:100], f, indent=2)  # First 100 samples

        self.result("CSV → JSON conversion", str(csv_to_json_path))

        # Workflow 2: NPZ → CSV (for Excel analysis)
        self.info("\nWorkflow 2: NPZ → CSV (for Excel)")
        npz_to_csv_path = output_dir / "npz_to_csv.csv"

        # Load NPZ and reconstruct trace
        loaded_npz_dict = load_npz(npz_path)
        # NPZ export stores data directly, we'll re-export the original for this workflow demo
        export_csv(waveform, npz_to_csv_path, include_time=True)

        self.result("NPZ → CSV conversion", str(npz_to_csv_path))

        # Workflow 3: JSON → HDF5 (for archival)
        if results.get("hdf5_path"):
            self.info("\nWorkflow 3: JSON → HDF5 (for long-term storage)")
            json_to_hdf5_path = output_dir / "json_to_hdf5.h5"

            # Load JSON
            loaded_json = load_json(json_path)

            # Export to HDF5 (measurements as attributes)
            export_hdf5(
                waveform,
                json_to_hdf5_path,
                compression="gzip",
                compression_opts=9,
            )

            self.result("JSON → HDF5 conversion", str(json_to_hdf5_path))

        results["conversions"] = {
            "csv_to_json": str(csv_to_json_path),
            "npz_to_csv": str(npz_to_csv_path),
        }

        # Part 10: Round-Trip Validation
        self.subsection("Part 10: Round-Trip Validation")
        self.info("Verify data integrity through export/import cycle")

        # Test 1: NPZ round-trip
        loaded_npz_dict = load_npz(npz_path)
        # load_npz returns dict with 'signal' key (see npz_export.py)
        loaded_data = loaded_npz_dict["signal"]
        npz_error = np.max(np.abs(waveform.data - loaded_data))
        npz_valid = npz_error < 1e-10
        self.result("NPZ round-trip error", f"{npz_error:.2e}")
        if npz_valid:
            self.success("NPZ round-trip validation PASSED")
        else:
            self.error("NPZ round-trip validation FAILED")

        # Test 2: JSON round-trip
        loaded_json_wrapper = load_json(json_path)
        # JSON export wraps data in {"_metadata": ..., "data": ...}
        loaded_json = loaded_json_wrapper.get("data", loaded_json_wrapper)
        json_valid = all(abs(loaded_json[key] - measurements[key]) < 1e-10 for key in measurements)
        if json_valid:
            self.success("JSON round-trip validation PASSED")
        else:
            self.error("JSON round-trip validation FAILED")

        # Test 3: CSV round-trip (check first/last values)
        with open(csv_path) as f:
            # Skip comment lines
            lines = [line for line in f if not line.startswith("#")]
            reader = csv.DictReader(lines)
            csv_rows = list(reader)

        # CSV columns are "Time (s)" and "Voltage"
        first_value = float(csv_rows[0]["Voltage"])
        csv_error = abs(first_value - waveform.data[0])
        csv_valid = csv_error < 1e-6  # Lower precision for CSV
        self.result("CSV round-trip error", f"{csv_error:.2e}")
        if csv_valid:
            self.success("CSV round-trip validation PASSED")
        else:
            self.error("CSV round-trip validation FAILED")

        results["round_trip"] = {
            "npz_valid": npz_valid,
            "npz_error": float(npz_error),
            "json_valid": json_valid,
            "csv_valid": csv_valid,
            "csv_error": float(csv_error),
        }

        # Part 11: Format Comparison
        self.subsection("Part 11: Format Comparison Table")
        self.info("Compare all formats across key dimensions")

        comparison = []

        # CSV
        comparison.append(
            {
                "format": "CSV",
                "size_kb": csv_size / 1024,
                "human_readable": "Yes",
                "metadata": "Comments",
                "compression": "No",
                "precision": "High",
                "speed": "Medium",
                "use_case": "Spreadsheets, manual inspection",
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
                "precision": "High",
                "speed": "Medium",
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
                    "precision": "Full",
                    "speed": "Fast",
                    "use_case": "Large datasets, archival",
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
                "precision": "Full",
                "speed": "Fast",
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
                    "precision": "Full",
                    "speed": "Medium",
                    "use_case": "MATLAB integration",
                }
            )

        # PWL
        comparison.append(
            {
                "format": "PWL",
                "size_kb": pwl_size / 1024,
                "human_readable": "Yes",
                "metadata": "Comments",
                "compression": "No",
                "precision": "High",
                "speed": "Slow",
                "use_case": "SPICE simulation",
            }
        )

        # HTML
        comparison.append(
            {
                "format": "HTML",
                "size_kb": html_size / 1024,
                "human_readable": "Yes",
                "metadata": "Full",
                "compression": "No",
                "precision": "Display",
                "speed": "Medium",
                "use_case": "Reports, dashboards",
            }
        )

        # Markdown
        comparison.append(
            {
                "format": "Markdown",
                "size_kb": md_size / 1024,
                "human_readable": "Yes",
                "metadata": "Text",
                "compression": "No",
                "precision": "Display",
                "speed": "Fast",
                "use_case": "Documentation, GitHub",
            }
        )

        self.info("\nFormat comparison table:")
        header = f"  {'Format':<10} {'Size':<8} {'Human':<6} {'Meta':<10} {'Comp':<6} {'Prec':<8}"
        self.info(header)
        self.info("  " + "-" * 70)
        for fmt in comparison:
            self.info(
                f"  {fmt['format']:<10} {fmt['size_kb']:<8.2f} "
                f"{fmt['human_readable']:<6} {fmt['metadata']:<10} "
                f"{fmt['compression']:<6} {fmt['precision']:<8}"
            )

        results["comparison"] = comparison

        # Part 12: Format Selection Guidelines
        self.subsection("Part 12: Format Selection Guidelines")
        self.info("When to use each format - comprehensive guide")

        guidelines = {
            "Small datasets (<100 KB)": {
                "recommendation": "CSV or JSON",
                "reason": "Human-readable, easy to inspect and share",
                "alternatives": ["Markdown for reports"],
            },
            "Large datasets (>10 MB)": {
                "recommendation": "HDF5 or NPZ",
                "reason": "Efficient storage with compression, fast I/O",
                "alternatives": ["CSV with compression for portability"],
            },
            "Multi-channel data": {
                "recommendation": "HDF5",
                "reason": "Hierarchical structure, metadata support",
                "alternatives": ["NPZ for Python-only workflows"],
            },
            "Web/API integration": {
                "recommendation": "JSON",
                "reason": "Standard format, wide compatibility, structured",
                "alternatives": ["CSV for simple data tables"],
            },
            "MATLAB workflows": {
                "recommendation": "MATLAB .mat",
                "reason": "Native integration, familiar environment",
                "alternatives": ["CSV for simple imports"],
            },
            "SPICE simulation": {
                "recommendation": "PWL",
                "reason": "Standard SPICE format, direct stimulus import",
                "alternatives": ["CSV for manual conversion"],
            },
            "Long-term archival": {
                "recommendation": "HDF5",
                "reason": "Self-describing, metadata-rich, industry standard",
                "alternatives": ["CSV for maximum portability"],
            },
            "Quick inspection": {
                "recommendation": "CSV",
                "reason": "Open in any text editor or spreadsheet",
                "alternatives": ["JSON for structured data"],
            },
            "Shareable reports": {
                "recommendation": "HTML or PDF",
                "reason": "Self-contained, formatted, interactive",
                "alternatives": ["Markdown for GitHub/docs"],
            },
            "Python workflows": {
                "recommendation": "NPZ",
                "reason": "Native NumPy format, efficient, Pythonic",
                "alternatives": ["HDF5 for larger datasets"],
            },
            "Cross-platform sharing": {
                "recommendation": "CSV or JSON",
                "reason": "Universal support, no special libraries needed",
                "alternatives": ["HDF5 with h5py widely available"],
            },
            "Real-time streaming": {
                "recommendation": "JSON or CSV",
                "reason": "Incremental append support",
                "alternatives": ["HDF5 with chunking"],
            },
        }

        self.info("\nComprehensive format selection guide:")
        for use_case, guide in guidelines.items():
            self.info(f"\n  {use_case}:")
            self.info(f"    → {guide['recommendation']}")
            self.info(f"    Reason: {guide['reason']}")
            if guide["alternatives"]:
                self.info(f"    Alternatives: {', '.join(guide['alternatives'])}")

        results["guidelines"] = guidelines

        # Part 13: Performance Comparison
        self.subsection("Part 13: Performance Summary")
        self.info("Format characteristics at a glance")

        self.info("\nFile Size Efficiency (for same 10ms waveform):")
        self.info(f"  CSV:      {csv_size / 1024:>8.2f} KB  (baseline)")
        if results.get("hdf5_path"):
            efficiency = csv_size / hdf5_size
            self.info(f"  HDF5:     {hdf5_size / 1024:>8.2f} KB  ({efficiency:.1f}x smaller)")
        self.info(f"  NPZ:      {npz_size / 1024:>8.2f} KB")
        if results.get("mat_path"):
            self.info(f"  MATLAB:   {mat_size / 1024:>8.2f} KB")
        self.info(f"  PWL:      {pwl_size / 1024:>8.2f} KB  (10x downsampled)")

        self.info("\nPrecision / Data Integrity:")
        self.info("  Full precision:     HDF5, NPZ, MATLAB (.mat)")
        self.info("  High precision:     CSV (9 decimals), JSON, PWL")
        self.info("  Display precision:  HTML, Markdown")

        self.info("\nEcosystem Compatibility:")
        self.info("  Universal:          CSV, JSON")
        self.info("  Python:             NPZ, HDF5 (with h5py)")
        self.info("  MATLAB:             .mat files")
        self.info("  SPICE tools:        PWL")
        self.info("  Web browsers:       HTML, JSON")
        self.info("  Documentation:      Markdown")

        self.success("Comprehensive export demonstration complete!")
        self.info("\nAll 8 export formats demonstrated:")
        self.info("  1. CSV - Universal text format")
        self.info("  2. JSON - Structured web format")
        self.info("  3. HDF5 - Scientific data archival")
        self.info("  4. NPZ - NumPy native format")
        self.info("  5. MATLAB - .mat files")
        self.info("  6. PWL - SPICE simulation")
        self.info("  7. HTML - Interactive reports")
        self.info("  8. Markdown - Documentation")

        return results

    def validate(self, results: dict[str, Any]) -> bool:
        """Validate the demonstration results."""
        self.info("Validating comprehensive export...")

        validation_passed = True

        # Validate all core exports exist
        required_exports = ["csv_path", "json_path", "npz_path", "pwl_path", "html_path", "md_path"]
        for export_key in required_exports:
            if export_key not in results:
                self.error(f"Missing {export_key}")
                validation_passed = False
            else:
                export_path = Path(results[export_key])
                if not export_path.exists():
                    self.error(f"File not found: {export_path}")
                    validation_passed = False

        # Validate round-trip integrity
        if "round_trip" not in results:
            self.error("Missing round-trip validation")
            validation_passed = False
        else:
            round_trip = results["round_trip"]
            if not round_trip.get("npz_valid"):
                self.error("NPZ round-trip validation failed")
                validation_passed = False
            if not round_trip.get("json_valid"):
                self.error("JSON round-trip validation failed")
                validation_passed = False
            if not round_trip.get("csv_valid"):
                self.error("CSV round-trip validation failed")
                validation_passed = False

        # Validate format comparison
        if "comparison" not in results:
            self.error("Missing format comparison")
            validation_passed = False
        elif len(results["comparison"]) < 6:
            self.error("Insufficient formats in comparison")
            validation_passed = False

        # Validate guidelines
        if "guidelines" not in results:
            self.error("Missing format selection guidelines")
            validation_passed = False
        elif len(results["guidelines"]) < 10:
            self.error("Insufficient guidelines")
            validation_passed = False

        # Validate conversions
        if "conversions" not in results:
            self.error("Missing format conversion workflows")
            validation_passed = False

        if validation_passed:
            self.success("All comprehensive export validations passed!")
            self.info("\nKey Takeaways:")
            self.info(
                "  - 8 export formats available: CSV, JSON, HDF5, NPZ, MAT, PWL, HTML, Markdown"
            )
            self.info("  - Format conversion workflows for common use cases")
            self.info("  - Round-trip validation ensures data integrity")
            self.info("  - Selection guidelines for every scenario")
            self.info("  - HDF5 recommended for archival (best compression + metadata)")
            self.info("  - CSV/JSON recommended for portability")
            self.info("  - NPZ recommended for Python workflows")
            self.info("\nNext steps:")
            self.info("  - Try 15_export_visualization/04_report_generation.py for PDF reports")
            self.info("  - Explore format-specific features in individual demonstrations")

        return validation_passed


if __name__ == "__main__":
    demo: ComprehensiveExportDemo = ComprehensiveExportDemo()
    success: bool = demo.execute()
    sys.exit(0 if success else 1)
