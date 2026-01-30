"""CLI Usage: Command-line interface automation patterns.

Demonstrates:
- Oscura CLI command patterns
- Batch file processing from command line
- Progress bars and logging configuration
- argparse integration for custom tools
- Practical CLI automation examples

Category: Integration
IEEE Standards: N/A

Related Demos:
- 01_data_loading/01_waveforms.py
- 02_basic_analysis/01_measurements.py

This demonstrates how to integrate Oscura into command-line workflows,
build CLI tools, and process files in batch mode with progress tracking.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add demos to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse

import numpy as np

from demos.common import BaseDemo, ValidationSuite, print_header, print_info, print_subheader


class CLIUsageDemo(BaseDemo):
    """Demonstrates CLI integration patterns."""

    name = "CLI Usage"
    description = "Command-line interface automation patterns"
    category = "integration"

    def generate_data(self) -> None:
        """Generate test files for batch processing."""
        from oscura.core import TraceMetadata, WaveformTrace

        # Create multiple test signals
        self.test_files = []
        for freq in [1000, 5000, 10000]:
            t = np.linspace(0, 0.01, 1000)
            data = np.sin(2 * np.pi * freq * t)

            trace = WaveformTrace(
                data=data,
                metadata=TraceMetadata(
                    sample_rate=100e3,
                    channel_name=f"CH_{freq}Hz",
                ),
            )

            # Save to file
            filepath = self.data_dir / f"signal_{freq}hz.npz"
            np.savez(filepath, data=trace.data, sample_rate=trace.metadata.sample_rate)
            self.test_files.append(filepath)

    def run_analysis(self) -> None:
        """Demonstrate CLI integration patterns."""
        print_header("CLI Usage Patterns")

        print_subheader("1. Basic CLI Tool with argparse")
        print_info("Example: Custom signal analyzer CLI tool")

        def create_analyzer_cli() -> argparse.ArgumentParser:
            """Create CLI parser for signal analyzer."""
            parser = argparse.ArgumentParser(
                description="Oscura Signal Analyzer",
                formatter_class=argparse.RawDescriptionHelpFormatter,
                epilog="""
Examples:
  %(prog)s input.wfm -o json
  %(prog)s *.wfm --batch --parallel 4
  %(prog)s signal.wfm --measurements rise_time fall_time
                """,
            )

            parser.add_argument("input", help="Input file or glob pattern")
            parser.add_argument(
                "-o",
                "--output",
                choices=["json", "csv", "table"],
                default="table",
                help="Output format (default: table)",
            )
            parser.add_argument(
                "-m",
                "--measurements",
                nargs="+",
                help="Specific measurements to run",
            )
            parser.add_argument(
                "--batch",
                action="store_true",
                help="Process multiple files",
            )
            parser.add_argument(
                "-v",
                "--verbose",
                action="count",
                default=0,
                help="Increase verbosity (-v, -vv, -vvv)",
            )

            return parser

        parser = create_analyzer_cli()
        print_info("CLI parser created")

        # Show sample argument parsing
        test_args = ["input.wfm", "-o", "json", "-v"]
        args = parser.parse_args(test_args)
        print_info(f"Example parsed args: {args}")

        print_subheader("2. Batch Processing")
        print_info("Process multiple files with progress tracking:")

        from oscura import frequency

        def batch_analyze(files: list[Path], verbose: bool = False) -> list[dict]:
            """Batch analyze multiple files."""
            results = []

            for i, filepath in enumerate(files, 1):
                if verbose:
                    print_info(f"  [{i}/{len(files)}] Processing {filepath.name}")

                try:
                    # Load and analyze
                    data_dict = np.load(str(filepath))
                    from oscura.core import TraceMetadata, WaveformTrace

                    trace = WaveformTrace(
                        data=data_dict["data"],
                        metadata=TraceMetadata(
                            sample_rate=float(data_dict["sample_rate"]),
                        ),
                    )

                    freq = frequency(trace)

                    results.append(
                        {
                            "file": filepath.name,
                            "frequency": freq,
                            "status": "success",
                        }
                    )

                except Exception as e:
                    results.append({"file": filepath.name, "status": "error", "error": str(e)})

            return results

        print_info(f"Processing {len(self.test_files)} files...")
        batch_results = batch_analyze(self.test_files, verbose=True)

        print_info("Batch results:")
        for result in batch_results:
            if result["status"] == "success":
                print_info(f"  ✓ {result['file']}: {result['frequency']:.1f} Hz")
            else:
                print_info(f"  ✗ {result['file']}: {result.get('error', 'Unknown error')}")

        self.results["batch_results"] = batch_results

        print_subheader("3. Progress Bar Implementation")
        print_info("Simple progress bar for long operations:")

        def process_with_progress(items: list, process_fn):
            """Process items with progress indication."""
            results = []
            total = len(items)

            for i, item in enumerate(items, 1):
                # Simple progress bar
                percent = i / total * 100
                bar_length = 40
                filled = int(bar_length * i / total)
                bar = "=" * filled + "-" * (bar_length - filled)
                print(f"\r  [{bar}] {percent:.1f}% ({i}/{total})", end="", flush=True)

                results.append(process_fn(item))

            print()  # New line after progress
            return results

        print_info("Processing with progress bar:")
        test_items = list(range(10))
        _ = process_with_progress(test_items, lambda x: x * 2)
        print_info("✓ Complete")

        print_subheader("4. Logging Configuration")
        print_info("Configure logging for CLI tools:")

        import logging

        def setup_cli_logging(verbosity: int) -> None:
            """Setup logging based on verbosity level."""
            level = [logging.WARNING, logging.INFO, logging.DEBUG][min(verbosity, 2)]

            logging.basicConfig(
                level=level,
                format="%(asctime)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )

        # Demonstrate different log levels
        for verbosity in range(3):
            setup_cli_logging(verbosity)
            level_name = ["WARNING", "INFO", "DEBUG"][verbosity]
            print_info(f"  Verbosity {verbosity} = {level_name}")

        print_subheader("5. Output Format Examples")
        print_info("Support multiple output formats:")

        measurements = {
            "frequency": 1000.0,
            "amplitude": 2.0,
            "rms": 1.414,
        }

        # JSON format
        import json

        json_output = json.dumps(measurements, indent=2)
        print_info("JSON format:")
        print(json_output)

        # CSV format
        csv_output = "measurement,value\n"
        for key, value in measurements.items():
            csv_output += f"{key},{value}\n"
        print_info("CSV format:")
        print(csv_output)

        # Table format (simple)
        print_info("Table format:")
        for key, value in measurements.items():
            print(f"  {key:15} {value:10.3f}")

    def validate_results(self, suite: ValidationSuite) -> None:
        """Validate CLI usage results."""
        suite.check_exists("Batch results", self.results.get("batch_results"))
        batch_results = self.results.get("batch_results", [])
        suite.check_equal("Number of processed files", len(batch_results), len(self.test_files))


if __name__ == "__main__":
    demo = CLIUsageDemo()
    result = demo.run()
    sys.exit(0 if result.success else 1)
