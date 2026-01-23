"""CLI Usage: Command-line interface integration patterns

Demonstrates:
- oscura CLI commands (characterize, decode, batch)
- argparse integration for custom tools
- Batch file processing from CLI
- Progress bars and logging configuration
- Practical CLI tool examples

IEEE Standards: N/A
Related Demos:
- 01_data_loading/01_loading_waveforms.py
- 02_basic_analysis/01_waveform_measurements.py
- 03_protocol_decoding/01_uart.py

This demonstrates how to integrate Oscura into command-line workflows,
build CLI tools, and process files in batch mode with progress tracking.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add demonstrations to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demonstrations.common import BaseDemo, generate_sine_wave


class CLIUsageDemo(BaseDemo):
    """Demonstrates CLI integration patterns."""

    def __init__(self):
        """Initialize CLI usage demonstration."""
        super().__init__(
            name="cli_usage",
            description="Command-line interface integration patterns",
            capabilities=[
                "oscura.cli.main",
                "oscura.cli.batch",
                "oscura.cli.characterize",
                "oscura.cli.decode",
            ],
        )

    def generate_test_data(self) -> dict:
        """Generate test signals for CLI processing."""
        # Create multiple test signals
        signals = []
        for freq in [1000, 5000, 10000]:
            trace = generate_sine_wave(
                frequency=float(freq),
                amplitude=1.0,
                duration=0.01,
                sample_rate=100e3,
            )
            signals.append((f"{freq}hz", trace))

        return {"signals": signals}

    def run_demonstration(self, data: dict) -> dict:
        """Demonstrate CLI integration patterns."""
        from io import StringIO

        import numpy as np

        from oscura.cli.main import format_output
        from oscura.exporters import export_npz

        signals = data["signals"]
        output_dir = self.get_output_dir()

        self.section("1. CLI Output Formats")
        self.info("Oscura CLI supports JSON, CSV, HTML, and table output formats")

        # Create sample measurements
        measurements = {
            "frequency": 1000.0,
            "amplitude": 2.0,
            "rise_time": 2.5e-9,
            "fall_time": 2.8e-9,
            "thd": -45.2,
        }

        self.subsection("Table Format (default)")
        table_output = format_output(measurements, "table")
        print(table_output)

        self.subsection("JSON Format")
        json_output = format_output(measurements, "json")
        print(json_output[:200] + "...")  # Truncate for demo

        self.subsection("CSV Format")
        csv_output = format_output(measurements, "csv")
        print(csv_output)

        self.section("2. Building CLI Tools with argparse")
        self.info("Example: Custom signal analyzer CLI tool")

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
                choices=["json", "csv", "table", "html"],
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
                "--parallel",
                type=int,
                default=1,
                help="Number of parallel workers",
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

        # Show help
        self.info("CLI Help:")
        help_buffer = StringIO()
        parser.print_help(help_buffer)
        help_text = help_buffer.getvalue()
        for line in help_text.split("\n")[:20]:  # Show first 20 lines
            print(f"    {line}")

        # Simulate argument parsing
        self.subsection("Parsing Arguments")
        test_args = ["input.wfm", "-o", "json", "-v"]
        args = parser.parse_args(test_args)
        self.info(f"Parsed args: {args}")

        self.section("3. Batch Processing")
        self.info("Save test signals to files for batch processing")

        # Save test signals
        test_files = []
        for name, trace in signals:
            filepath = output_dir / f"test_{name}.npz"
            export_npz(trace, str(filepath))
            test_files.append(filepath)
            self.info(f"  Saved {filepath.name}")

        self.subsection("Batch Analysis Function")

        def batch_analyze(files: list[Path], parallel: int = 1, verbose: int = 0) -> list[dict]:
            """Batch analyze multiple files.

            Args:
                files: List of file paths to analyze
                parallel: Number of parallel workers
                verbose: Verbosity level

            Returns:
                List of analysis results
            """
            from oscura.analyzers.waveform.measurements import amplitude, frequency
            from oscura.loaders import load

            results = []

            for i, filepath in enumerate(files, 1):
                if verbose:
                    print(f"  [{i}/{len(files)}] Processing {filepath.name}")

                try:
                    # Load and analyze
                    trace = load(str(filepath))
                    freq = frequency(trace)
                    amp = amplitude(trace)

                    results.append(
                        {
                            "file": filepath.name,
                            "frequency": freq,
                            "amplitude": amp,
                            "status": "success",
                        }
                    )

                except Exception as e:
                    results.append({"file": filepath.name, "status": "error", "error": str(e)})

            return results

        # Run batch analysis
        self.info("Running batch analysis on test files...")
        batch_results = batch_analyze(test_files, verbose=1)

        self.subsection("Batch Results")
        for result in batch_results:
            if result["status"] == "success":
                self.result(
                    result["file"],
                    f"{result['frequency']:.1f} Hz, {result['amplitude']:.3f} V",
                )
            else:
                self.warning(f"{result['file']}: {result.get('error', 'Unknown error')}")

        self.section("4. Progress Bars and Logging")
        self.info("Configure logging for CLI tools")

        # Configure logging
        def setup_cli_logging(verbosity: int) -> None:
            """Setup logging based on verbosity level.

            Args:
                verbosity: 0=WARNING, 1=INFO, 2=DEBUG
            """
            level = [logging.WARNING, logging.INFO, logging.DEBUG][min(verbosity, 2)]

            logging.basicConfig(
                level=level,
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )

        # Demonstrate different log levels
        self.subsection("Logging Levels")
        for verbosity in range(3):
            setup_cli_logging(verbosity)
            _logger = logging.getLogger("oscura.demo")  # For demonstration
            level_name = ["WARNING", "INFO", "DEBUG"][verbosity]
            self.info(f"Verbosity {verbosity} = {level_name}")

        # Progress tracking (without tqdm for simplicity)
        self.subsection("Progress Tracking")

        def process_with_progress(items: list, process_fn):
            """Process items with progress indication.

            Args:
                items: Items to process
                process_fn: Function to apply to each item

            Returns:
                List of results
            """
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

        # Demonstrate progress
        self.info("Processing with progress bar:")
        test_items = list(range(10))
        _ = process_with_progress(test_items, lambda x: x * 2)

        self.section("5. Error Handling in CLI")
        self.info("Proper error handling with exit codes")

        def cli_main() -> int:
            """Example CLI main function with error handling.

            Returns:
                Exit code (0=success, 1=error)
            """
            try:
                # Simulate CLI operations
                print("  Initializing...")
                print("  Processing...")
                print("  Complete!")
                return 0

            except FileNotFoundError as e:
                print(f"  ERROR: File not found: {e}", file=sys.stderr)
                return 1

            except ValueError as e:
                print(f"  ERROR: Invalid value: {e}", file=sys.stderr)
                return 1

            except Exception as e:
                print(f"  ERROR: Unexpected error: {e}", file=sys.stderr)
                return 1

        exit_code = cli_main()
        self.info(f"CLI exit code: {exit_code}")

        self.section("6. Practical CLI Tool Example")
        self.info("Complete example: Signal Statistics Tool")

        def signal_stats_cli(filepath: str) -> dict:
            """Simple signal statistics CLI tool.

            Args:
                filepath: Path to signal file

            Returns:
                Dictionary of statistics
            """
            from oscura.analyzers.waveform.spectral import fft
            from oscura.loaders import load

            # Load signal
            trace = load(filepath)

            # Basic statistics
            stats = {
                "samples": len(trace.data),
                "min": float(np.min(trace.data)),
                "max": float(np.max(trace.data)),
                "mean": float(np.mean(trace.data)),
                "std": float(np.std(trace.data)),
            }

            # Add spectral stats
            freqs, mags = fft(trace)
            if len(mags) > 0:
                peak_idx = int(np.argmax(mags))
                stats["peak_frequency"] = float(freqs[peak_idx])

            return stats

        # Run on first test file
        self.subsection("Running Signal Stats Tool")
        stats = signal_stats_cli(str(test_files[0]))

        for key, value in stats.items():
            self.result(key, f"{value:.4g}")

        self.success("CLI integration patterns demonstrated!")

        return {
            "measurements": measurements,
            "batch_results": batch_results,
            "test_files": [str(f) for f in test_files],
            "stats": stats,
        }

    def validate(self, results: dict) -> bool:
        """Validate CLI integration results."""
        self.info("Validating CLI results...")

        # Check measurements exist
        if "measurements" not in results or not isinstance(results["measurements"], dict):
            self.error("Missing or invalid 'measurements' in results")
            return False
        self.success("measurements exists")

        # Check batch results
        if "batch_results" not in results or not isinstance(results["batch_results"], list):
            self.error("Missing or invalid 'batch_results' in results")
            return False
        self.success("batch_results exists")

        batch_results = results["batch_results"]
        if len(batch_results) != 3:
            self.error(f"Expected 3 batch results, got {len(batch_results)}")
            return False

        # Check all batch jobs succeeded
        for result in batch_results:
            if result["status"] != "success":
                self.error(f"Batch job failed: {result.get('error', 'Unknown')}")
                return False

        # Check test files were created
        if "test_files" not in results or not isinstance(results["test_files"], list):
            self.error("Missing or invalid 'test_files' in results")
            return False
        self.success("test_files exists")

        # Check stats
        if "stats" not in results or not isinstance(results["stats"], dict):
            self.error("Missing or invalid 'stats' in results")
            return False
        self.success("stats exists")

        self.success("All CLI integration tests passed!")
        self.info("\nNext steps:")
        self.info("  - Try 02_jupyter_notebooks.py for interactive analysis")
        self.info("  - See 03_llm_integration.py for AI-friendly outputs")
        self.info("  - Explore oscura CLI: oscura --help")

        return True


if __name__ == "__main__":
    demo = CLIUsageDemo()
    success = demo.execute()
    sys.exit(0 if success else 1)
