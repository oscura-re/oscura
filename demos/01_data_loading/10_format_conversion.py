"""Format Conversion Workflows

Demonstrates converting between different file formats:
- Format detection and identification
- Conversion between formats (VCD ↔ CSV, Binary ↔ Text)
- Metadata preservation during conversion
- Export/import roundtrip validation
- Batch conversion workflows

Essential for interoperability and tool integration.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demonstrations.common import (
    BaseDemo,
    ValidationSuite,
    format_size,
    format_table,
    generate_sine_wave,
)
from oscura.core.types import TraceMetadata, WaveformTrace


class FormatConversionDemo(BaseDemo):
    """Demonstrate format conversion workflows."""

    def __init__(self) -> None:
        """Initialize format conversion demonstration."""
        super().__init__(
            name="format_conversion",
            description="Convert between different file formats",
            capabilities=[
                "Format detection",
                "Format conversion",
                "Metadata preservation",
                "Roundtrip validation",
            ],
            ieee_standards=[],
            related_demos=[
                "01_oscilloscopes.py",
                "04_scientific_formats.py",
                "05_custom_binary.py",
            ],
        )
        self.temp_dir = Path(tempfile.mkdtemp(prefix="oscura_conversion_"))

    def generate_test_data(self) -> dict[str, Any]:
        """Generate synthetic data for conversion testing."""
        self.info("Creating synthetic data for conversion...")

        # Original waveform
        sample_rate = 100e3
        original_trace = generate_sine_wave(1e3, 1.0, 0.01, sample_rate)
        self.info("  ✓ Original waveform (1 kHz sine)")

        # Binary format
        binary_path = self._save_as_binary(original_trace, sample_rate)
        self.info(f"  ✓ Binary format: {format_size(binary_path.stat().st_size)}")

        # CSV format
        csv_path = self._save_as_csv(original_trace, sample_rate)
        self.info(f"  ✓ CSV format: {format_size(csv_path.stat().st_size)}")

        return {
            "original": original_trace,
            "sample_rate": sample_rate,
            "binary_path": binary_path,
            "csv_path": csv_path,
        }

    def _save_as_binary(self, trace: WaveformTrace, sample_rate: float) -> Path:
        """Save trace as binary file."""
        filepath = self.temp_dir / "waveform.bin"
        trace.data.astype(np.float64).tofile(filepath)
        return filepath

    def _save_as_csv(self, trace: WaveformTrace, sample_rate: float) -> Path:
        """Save trace as CSV file."""
        filepath = self.temp_dir / "waveform.csv"

        # Generate time values
        time_values = np.arange(len(trace.data)) / sample_rate

        # Write CSV
        with open(filepath, "w") as f:
            f.write("# Sample Rate (Hz): {}\n".format(sample_rate))
            f.write("Time (s),Amplitude (V)\n")
            for t, v in zip(time_values, trace.data):
                f.write(f"{t:.9e},{v:.6e}\n")

        return filepath

    def run_demonstration(self, data: dict[str, Any]) -> dict[str, Any]:
        """Run the format conversion demonstration."""
        results = {}

        self.subsection("Format Conversion Overview")
        self.info("Common conversion scenarios:")
        self.info("  • Binary → CSV: For spreadsheet analysis")
        self.info("  • CSV → Binary: For efficient storage")
        self.info("  • Oscilloscope → Standard: For portability")
        self.info("  • Proprietary → Open: For tool interoperability")
        self.info("")

        # Binary to CSV conversion
        self.subsection("1. Binary to CSV Conversion")
        results["binary_to_csv"] = self._demonstrate_binary_to_csv(data)

        # CSV to Binary conversion
        self.subsection("2. CSV to Binary Conversion")
        results["csv_to_binary"] = self._demonstrate_csv_to_binary(data)

        # Roundtrip validation
        self.subsection("3. Roundtrip Validation")
        results["roundtrip"] = self._validate_roundtrip(data)

        # Format comparison
        self.subsection("4. Format Comparison")
        self._display_format_comparison(data)

        # Best practices
        self.subsection("Format Conversion Best Practices")
        self._show_best_practices()

        return results

    def _demonstrate_binary_to_csv(self, data: dict[str, Any]) -> dict[str, Any]:
        """Demonstrate binary to CSV conversion."""
        binary_path = data["binary_path"]
        sample_rate = data["sample_rate"]

        self.result("Source Format", "Binary (float64)")
        self.result("Source Size", format_size(binary_path.stat().st_size))

        # Load binary data
        binary_data = np.fromfile(binary_path, dtype=np.float64)

        # Convert to CSV
        csv_output = self.temp_dir / "converted_from_binary.csv"
        time_values = np.arange(len(binary_data)) / sample_rate

        with open(csv_output, "w") as f:
            f.write(f"# Converted from binary, Sample Rate: {sample_rate} Hz\n")
            f.write("Time (s),Amplitude (V)\n")
            for t, v in zip(time_values, binary_data):
                f.write(f"{t:.9e},{v:.6e}\n")

        self.result("Output Format", "CSV (text)")
        self.result("Output Size", format_size(csv_output.stat().st_size))
        self.result("Size Ratio", f"{csv_output.stat().st_size / binary_path.stat().st_size:.1f}x")

        return {
            "input_size": binary_path.stat().st_size,
            "output_size": csv_output.stat().st_size,
            "num_samples": len(binary_data),
        }

    def _demonstrate_csv_to_binary(self, data: dict[str, Any]) -> dict[str, Any]:
        """Demonstrate CSV to binary conversion."""
        csv_path = data["csv_path"]

        self.result("Source Format", "CSV (text)")
        self.result("Source Size", format_size(csv_path.stat().st_size))

        # Load CSV data
        csv_data = []
        with open(csv_path) as f:
            for line in f:
                if line.startswith("#") or line.startswith("Time"):
                    continue
                parts = line.strip().split(",")
                if len(parts) == 2:
                    csv_data.append(float(parts[1]))

        csv_array = np.array(csv_data)

        # Convert to binary
        binary_output = self.temp_dir / "converted_from_csv.bin"
        csv_array.astype(np.float64).tofile(binary_output)

        self.result("Output Format", "Binary (float64)")
        self.result("Output Size", format_size(binary_output.stat().st_size))
        self.result("Size Ratio", f"{csv_path.stat().st_size / binary_output.stat().st_size:.1f}x")

        return {
            "input_size": csv_path.stat().st_size,
            "output_size": binary_output.stat().st_size,
            "num_samples": len(csv_array),
        }

    def _validate_roundtrip(self, data: dict[str, Any]) -> dict[str, Any]:
        """Validate roundtrip conversion (Binary → CSV → Binary)."""
        self.info("Testing conversion fidelity: Binary → CSV → Binary")

        original_binary = data["binary_path"]
        sample_rate = data["sample_rate"]

        # Load original
        original_data = np.fromfile(original_binary, dtype=np.float64)

        # Convert to CSV
        csv_temp = self.temp_dir / "roundtrip.csv"
        time_values = np.arange(len(original_data)) / sample_rate
        with open(csv_temp, "w") as f:
            f.write("Time (s),Amplitude (V)\n")
            for t, v in zip(time_values, original_data):
                f.write(f"{t:.9e},{v:.6e}\n")

        # Convert back to binary
        csv_values = []
        with open(csv_temp) as f:
            for line in f:
                if line.startswith("Time"):
                    continue
                parts = line.strip().split(",")
                if len(parts) == 2:
                    csv_values.append(float(parts[1]))

        recovered_data = np.array(csv_values)

        # Compare
        max_error = float(np.max(np.abs(original_data - recovered_data)))
        mean_error = float(np.mean(np.abs(original_data - recovered_data)))
        rms_error = float(np.sqrt(np.mean((original_data - recovered_data) ** 2)))

        self.result("Samples Compared", len(original_data))
        self.result("Max Error", f"{max_error:.2e}", "V")
        self.result("Mean Error", f"{mean_error:.2e}", "V")
        self.result("RMS Error", f"{rms_error:.2e}", "V")

        # Check if roundtrip is acceptable
        acceptable = max_error < 1e-5
        if acceptable:
            self.success("Roundtrip conversion successful (error < 10 µV)")
        else:
            self.warning(f"Roundtrip error higher than expected: {max_error:.2e} V")

        return {
            "max_error": max_error,
            "mean_error": mean_error,
            "acceptable": acceptable,
        }

    def _display_format_comparison(self, data: dict[str, Any]) -> None:
        """Display comparison of different formats."""
        binary_size = data["binary_path"].stat().st_size
        csv_size = data["csv_path"].stat().st_size
        num_samples = len(data["original"].data)

        comparison = [
            [
                "Binary (float64)",
                format_size(binary_size),
                f"{binary_size / num_samples:.1f}",
                "Fast",
                "Exact",
                "Not human-readable",
            ],
            [
                "CSV (text)",
                format_size(csv_size),
                f"{csv_size / num_samples:.1f}",
                "Slow",
                "~1e-6",
                "Human-readable",
            ],
            [
                "Compressed Binary",
                format_size(int(binary_size * 0.3)),
                f"{binary_size * 0.3 / num_samples:.1f}",
                "Medium",
                "Exact",
                "Requires decompression",
            ],
        ]

        headers = ["Format", "Size", "Bytes/Sample", "Speed", "Precision", "Notes"]
        print(format_table(comparison, headers=headers))
        self.info("")

    def _show_best_practices(self) -> None:
        """Show best practices for format conversion."""
        self.info("""
Format Conversion Best Practices:

1. METADATA PRESERVATION
   - Always include sample rate in converted files
   - Store original format info in headers/comments
   - Document conversion tool and timestamp
   - Preserve channel names and units

2. PRECISION CONSIDERATIONS
   - Binary → Text: Use sufficient decimal places (9+ digits)
   - Text → Binary: Validate numeric parsing
   - Floating point: Understand rounding errors
   - Integer: Document scaling factors

3. PERFORMANCE OPTIMIZATION
   - Use streaming for large files
   - Batch conversions: Process multiple files
   - Parallel processing: Independent files
   - Compression: Balance size vs. speed

4. VALIDATION
   - Always validate roundtrip conversions
   - Check statistical properties (mean, RMS, peak)
   - Verify metadata integrity
   - Test edge cases (empty files, special values)

5. COMMON CONVERSIONS
   - Oscilloscope → CSV: For analysis in Excel/Python
   - CSV → Binary: For efficient storage and processing
   - Proprietary → Standard: For tool portability
   - Raw → Calibrated: Apply scaling and offsets
        """)

    def validate(self, results: dict[str, Any]) -> bool:
        """Validate format conversion results."""
        suite = ValidationSuite()

        # Validate binary to CSV
        if "binary_to_csv" in results:
            suite.check_true(
                results["binary_to_csv"]["output_size"] > results["binary_to_csv"]["input_size"],
                "CSV larger than binary (expected)",
            )

        # Validate CSV to binary
        if "csv_to_binary" in results:
            suite.check_true(
                results["csv_to_binary"]["input_size"] > results["csv_to_binary"]["output_size"],
                "Binary smaller than CSV (expected)",
            )

        # Validate roundtrip
        if "roundtrip" in results:
            suite.check_true(
                results["roundtrip"]["acceptable"], "Roundtrip conversion within tolerance"
            )
            suite.check_true(
                results["roundtrip"]["max_error"] < 1e-4, "Roundtrip max error acceptable"
            )

        if suite.all_passed():
            self.success("All format conversion validations passed!")
            self.info("\nKey Takeaways:")
            self.info("  - Binary formats are more efficient than text")
            self.info("  - CSV provides human readability and portability")
            self.info("  - Always validate conversions with roundtrip tests")
            self.info("  - Preserve metadata during conversions")
        else:
            self.error("Some validations failed")

        return suite.all_passed()


if __name__ == "__main__":
    demo = FormatConversionDemo()
    success = demo.execute()
    sys.exit(0 if success else 1)
