"""API Usage: Programmatic API usage patterns.

Demonstrates:
- Pythonic API design patterns
- Context managers for resource management
- Fluent interface patterns
- Error handling and exceptions
- Type hints and static typing

Category: Integration
IEEE Standards: N/A

Related Demos:
- 00_getting_started/01_core_types.py
- 02_basic_analysis/01_measurements.py

This demonstrates best practices for using Oscura as a library in
Python applications, including proper resource management, error
handling, and type-safe APIs.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add demos to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np

from demos.common import BaseDemo, ValidationSuite, print_header, print_info, print_subheader


class APIUsageDemo(BaseDemo):
    """Demonstrates programmatic API usage patterns."""

    name = "API Usage"
    description = "Programmatic API usage patterns"
    category = "integration"

    def generate_data(self) -> None:
        """Generate test data for API examples."""
        from oscura.core import TraceMetadata, WaveformTrace

        t = np.linspace(0, 0.01, 1000)
        data = np.sin(2 * np.pi * 1000 * t)

        self.trace = WaveformTrace(
            data=data,
            metadata=TraceMetadata(
                sample_rate=100e3,
                channel_name="CH1",
            ),
        )

    def run_analysis(self) -> None:
        """Demonstrate API usage patterns."""
        print_header("Programmatic API Usage")

        print_subheader("1. Basic API Usage")
        print_info("Simple, direct API usage:")

        from oscura import amplitude, frequency

        # Direct function calls
        freq = frequency(self.trace)
        amp = amplitude(self.trace)

        print_info(f"Frequency: {freq:.2f} Hz")
        print_info(f"Amplitude: {amp:.3f} V")

        self.results["frequency"] = freq
        self.results["amplitude"] = amp

        print_subheader("2. Context Manager Pattern")
        print_info("Resource management with context managers:")

        class TraceFile:
            """Context manager for trace file operations."""

            def __init__(self, filepath: Path):
                self.filepath = filepath
                self.trace = None

            def __enter__(self):
                """Load trace on context entry."""
                print_info(f"  Opening {self.filepath}")
                # Would load actual file here
                self.trace = self.trace  # Placeholder
                return self.trace

            def __exit__(self, exc_type, exc_val, exc_tb):
                """Cleanup on context exit."""
                print_info(f"  Closing {self.filepath}")
                self.trace = None
                return False

        # Usage example (commented since file doesn't exist)
        # with TraceFile(Path("signal.wfm")) as trace:
        #     freq = frequency(trace)

        print_info("✓ Context manager pattern demonstrated")

        print_subheader("3. Fluent Interface Pattern")
        print_info("Chainable API for readability:")

        class AnalysisPipeline:
            """Fluent interface for analysis pipeline."""

            def __init__(self, trace):
                self.trace = trace
                self.results = {}

            def measure_frequency(self):
                """Measure frequency."""
                from oscura import frequency

                self.results["frequency"] = frequency(self.trace)
                return self  # Enable chaining

            def measure_amplitude(self):
                """Measure amplitude."""
                from oscura import amplitude

                self.results["amplitude"] = amplitude(self.trace)
                return self  # Enable chaining

            def measure_rms(self):
                """Measure RMS."""
                from oscura import rms

                self.results["rms"] = rms(self.trace)
                return self  # Enable chaining

            def get_results(self):
                """Get all results."""
                return self.results

        # Fluent usage
        results = (
            AnalysisPipeline(self.trace)
            .measure_frequency()
            .measure_amplitude()
            .measure_rms()
            .get_results()
        )

        print_info("Fluent API results:")
        for key, value in results.items():
            print_info(f"  {key}: {value:.3f}")

        print_subheader("4. Error Handling")
        print_info("Robust error handling:")

        from oscura.core import WaveformTrace

        def safe_analyze(trace: WaveformTrace | None) -> dict:
            """Analyze trace with error handling."""
            if trace is None:
                raise ValueError("Trace cannot be None")

            if len(trace.data) == 0:
                raise ValueError("Trace data is empty")

            try:
                freq = frequency(trace)
                return {"frequency": freq, "status": "success"}
            except Exception as e:
                return {"status": "error", "error": str(e)}

        result = safe_analyze(self.trace)
        print_info(f"Safe analysis result: {result}")

        print_subheader("5. Type Hints and Type Safety")
        print_info("Using type hints for better IDE support:")

        from typing import TypedDict

        class MeasurementResult(TypedDict):
            """Type-safe measurement result."""

            frequency: float
            amplitude: float
            rms: float

        def typed_measure(trace: WaveformTrace) -> MeasurementResult:
            """Type-safe measurement function."""
            from oscura import amplitude, frequency, rms

            return {
                "frequency": frequency(trace),
                "amplitude": amplitude(trace),
                "rms": rms(trace),
            }

        typed_result = typed_measure(self.trace)
        print_info(f"Type-safe result: {typed_result}")

        print_subheader("6. Batch Processing API")
        print_info("Process multiple traces efficiently:")

        def batch_measure(traces: list[WaveformTrace]) -> list[dict]:
            """Batch measure multiple traces."""
            results = []
            for i, trace in enumerate(traces):
                try:
                    freq = frequency(trace)
                    results.append({"index": i, "frequency": freq, "success": True})
                except Exception as e:
                    results.append({"index": i, "success": False, "error": str(e)})
            return results

        # Example with single trace (would normally be multiple)
        batch_results = batch_measure([self.trace])
        print_info(f"Batch results: {batch_results}")

        print_subheader("7. Callback Pattern")
        print_info("Progress callbacks for long operations:")

        def analyze_with_callback(trace: WaveformTrace, progress_callback=None):
            """Analyze with progress callback."""
            if progress_callback:
                progress_callback(0, "Starting analysis...")

            freq = frequency(trace)

            if progress_callback:
                progress_callback(50, "Frequency measured")

            amp = amplitude(trace)

            if progress_callback:
                progress_callback(100, "Analysis complete")

            return {"frequency": freq, "amplitude": amp}

        def my_progress(percent: int, message: str):
            print_info(f"  [{percent}%] {message}")

        result = analyze_with_callback(self.trace, progress_callback=my_progress)
        print_info(f"✓ Callback result: {result}")

    def validate_results(self, suite: ValidationSuite) -> None:
        """Validate API usage results."""
        suite.check_exists("Frequency", self.results.get("frequency"))
        suite.check_exists("Amplitude", self.results.get("amplitude"))
        suite.check_type("Frequency", self.results.get("frequency"), float)
        suite.check_type("Amplitude", self.results.get("amplitude"), float)


if __name__ == "__main__":
    demo = APIUsageDemo()
    result = demo.run()
    sys.exit(0 if result.success else 1)
