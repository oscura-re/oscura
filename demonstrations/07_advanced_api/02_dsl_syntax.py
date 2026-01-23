"""DSL Syntax: Domain-Specific Language for Signal Processing

Demonstrates:
- Fluent API patterns for readable analysis
- Method chaining for signal transformations
- Query-like syntax for signal operations
- Expressive signal processing expressions

IEEE Standards: N/A
Related Demos:
- 07_advanced_api/01_pipeline_api.py
- 07_advanced_api/03_operators.py

Domain-Specific Languages make signal processing code more readable and
expressive, enabling query-like analysis workflows.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Add demonstrations to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demonstrations.common import BaseDemo, generate_sine_wave, validate_approximately
from oscura import (
    SignalBuilder,
    amplitude,
    frequency,
    high_pass,
    low_pass,
    rms,
    thd,
)


class FluentSignalAnalysis:
    """Fluent API wrapper for signal analysis."""

    def __init__(self, trace):
        """Initialize fluent analysis."""
        self.trace = trace
        self.filters_applied = []
        self.measurements = {}

    def filter_high_pass(self, cutoff: float):
        """Apply high-pass filter (fluent)."""
        self.trace = high_pass(self.trace, cutoff=cutoff)
        self.filters_applied.append(f"high_pass({cutoff} Hz)")
        return self

    def filter_low_pass(self, cutoff: float):
        """Apply low-pass filter (fluent)."""
        self.trace = low_pass(self.trace, cutoff=cutoff)
        self.filters_applied.append(f"low_pass({cutoff} Hz)")
        return self

    def measure_amplitude(self):
        """Measure amplitude (fluent)."""
        self.measurements["amplitude"] = amplitude(self.trace)
        return self

    def measure_rms(self):
        """Measure RMS (fluent)."""
        self.measurements["rms"] = rms(self.trace)
        return self

    def measure_frequency(self):
        """Measure frequency (fluent)."""
        self.measurements["frequency"] = frequency(self.trace)
        return self

    def measure_thd(self, fundamental: float | None = None):
        """Measure THD (fluent).

        Args:
            fundamental: Ignored (kept for API compatibility). THD auto-detects fundamental.
        """
        # THD automatically finds fundamental frequency from FFT
        self.measurements["thd"] = thd(self.trace)
        return self

    def get_trace(self):
        """Get the processed trace."""
        return self.trace

    def get_measurements(self):
        """Get all measurements."""
        return self.measurements

    def get_summary(self):
        """Get processing summary."""
        return {
            "filters": self.filters_applied,
            "measurements": self.measurements,
        }


class DSLSyntaxDemo(BaseDemo):
    """Demonstrate domain-specific language patterns for signal processing."""

    def __init__(self):
        """Initialize DSL syntax demonstration."""
        super().__init__(
            name="dsl_syntax",
            description="Domain-Specific Language for readable signal processing",
            capabilities=[
                "oscura.fluent_api",
                "oscura.method_chaining",
                "oscura.SignalBuilder",
            ],
        )

    def generate_test_data(self) -> dict:
        """Generate test signals."""
        # Noisy signal with DC offset
        noisy = generate_sine_wave(
            frequency=1000.0,
            amplitude=1.0,
            duration=0.01,
            sample_rate=100e3,
            offset=0.2,
        )
        noise = np.random.normal(0, 0.05, len(noisy.data))
        noisy.data = noisy.data + noise

        return {"signal": noisy}

    def run_demonstration(self, data: dict) -> dict:
        """Run DSL syntax demonstration."""
        signal = data["signal"]

        self.section("DSL Syntax: Expressive Signal Processing")

        # ===================================================================
        # Part 1: Fluent API for Analysis
        # ===================================================================
        self.subsection("1. Fluent API Pattern")
        self.info("Chain operations with readable syntax")

        # Fluent analysis
        analysis = (
            FluentSignalAnalysis(signal)
            .filter_high_pass(100.0)
            .filter_low_pass(5000.0)
            .measure_amplitude()
            .measure_rms()
            .measure_frequency()
            .measure_thd(1000.0)
        )

        summary = analysis.get_summary()

        self.info("Processing chain:")
        for i, filt in enumerate(summary["filters"], 1):
            self.info(f"  {i}. {filt}")

        self.info("\nMeasurements:")
        for name, value in summary["measurements"].items():
            if name == "thd":
                self.result(f"  {name.upper()}", f"{value:.2f}", "dB")
            elif name == "frequency":
                self.result(f"  {name.title()}", f"{value:.2f}", "Hz")
            else:
                self.result(f"  {name.title()}", f"{value:.4f}", "V")

        self.success("Fluent API enables readable analysis chains")

        # ===================================================================
        # Part 2: Signal Builder DSL
        # ===================================================================
        self.subsection("2. Signal Builder DSL")
        self.info("Declarative signal construction")

        # Build complex signal with fluent syntax
        # Use stronger harmonics to ensure THD is measurable above noise floor
        test_signal = (
            SignalBuilder(sample_rate=100e3, duration=0.01)
            .add_sine(frequency=1000, amplitude=1.0)
            .add_sine(frequency=2000, amplitude=0.5)  # 2nd harmonic (stronger)
            .add_sine(frequency=3000, amplitude=0.3)  # 3rd harmonic (stronger)
            .add_noise(snr_db=40)
            .build()
        )

        self.info("Signal construction:")
        self.info("  - 1000 Hz fundamental (1.0V)")
        self.info("  - 2000 Hz 2nd harmonic (0.5V)")
        self.info("  - 3000 Hz 3rd harmonic (0.3V)")
        self.info("  - Noise (SNR = 40 dB)")

        # Analyze the built signal
        built_freq = frequency(test_signal)
        built_thd = thd(test_signal)  # Auto-detects fundamental from FFT

        self.result("Detected frequency", f"{built_freq:.2f}", "Hz")
        self.result("THD", f"{built_thd:.2f}", "dB")
        self.success("Builder DSL simplifies signal generation")

        # ===================================================================
        # Part 3: Query-Like Signal Selection
        # ===================================================================
        self.subsection("3. Query-Like Signal Operations")
        self.info("Express intent clearly with query-style syntax")

        # Simulate query-like interface
        class SignalQuery:
            """Query-like interface for signal analysis."""

            def __init__(self, trace):
                self.trace = trace

            def where_frequency_above(self, cutoff: float):
                """Keep frequencies above cutoff."""
                self.trace = high_pass(self.trace, cutoff=cutoff)
                return self

            def where_frequency_below(self, cutoff: float):
                """Keep frequencies below cutoff."""
                self.trace = low_pass(self.trace, cutoff=cutoff)
                return self

            def select_amplitude(self):
                """Select amplitude measurement."""
                return amplitude(self.trace)

            def select_rms(self):
                """Select RMS measurement."""
                return rms(self.trace)

        # Use query syntax
        result_amp = (
            SignalQuery(signal)
            .where_frequency_above(100.0)  # Remove DC
            .where_frequency_below(5000.0)  # Remove HF noise
            .select_amplitude()
        )

        result_rms = (
            SignalQuery(signal)
            .where_frequency_above(100.0)
            .where_frequency_below(5000.0)
            .select_rms()
        )

        self.info("Query results:")
        self.result("  Amplitude", f"{result_amp:.4f}", "V")
        self.result("  RMS", f"{result_rms:.4f}", "V")
        self.success("Query syntax reads like natural language")

        # ===================================================================
        # Part 4: Expressive Measurement DSL
        # ===================================================================
        self.subsection("4. Expressive Measurement Patterns")
        self.info("Readable measurement expressions")

        # Define measurement expressions
        class MeasurementDSL:
            """Expressive measurement interface."""

            def __init__(self, trace):
                self.trace = trace

            def the(self, measurement_func):
                """Express 'the X of signal'."""
                return measurement_func(self.trace)

        # Use expressive syntax
        signal_dsl = MeasurementDSL(signal)

        peak_to_peak = signal_dsl.the(amplitude)
        root_mean_square = signal_dsl.the(rms)
        dominant_frequency = signal_dsl.the(frequency)

        self.info("Measurements using DSL:")
        self.info(f"  the(amplitude) = {peak_to_peak:.4f} V")
        self.info(f"  the(rms) = {root_mean_square:.4f} V")
        self.info(f"  the(frequency) = {dominant_frequency:.2f} Hz")

        self.success("DSL makes code self-documenting")

        # ===================================================================
        # Part 5: Comparison - Traditional vs DSL
        # ===================================================================
        self.subsection("5. Code Readability Comparison")

        self.info("Traditional approach:")
        self.info("  temp1 = high_pass(signal, cutoff=100.0)")
        self.info("  temp2 = low_pass(temp1, cutoff=5000.0)")
        self.info("  amp = amplitude(temp2)")
        self.info("  r = rms(temp2)")

        self.info("\nDSL approach:")
        self.info("  analysis = (FluentSignalAnalysis(signal)")
        self.info("      .filter_high_pass(100.0)")
        self.info("      .filter_low_pass(5000.0)")
        self.info("      .measure_amplitude()")
        self.info("      .measure_rms())")

        self.success("DSL eliminates temporary variables")
        self.success("DSL expresses intent more clearly")

        return {
            "fluent_amplitude": summary["measurements"]["amplitude"],
            "fluent_rms": summary["measurements"]["rms"],
            "built_frequency": built_freq,
            "built_thd": built_thd,
            "query_amplitude": result_amp,
            "query_rms": result_rms,
        }

    def validate(self, results: dict) -> bool:
        """Validate DSL results."""
        self.info("Validating DSL operations...")

        # Fluent and query should produce same results
        if not validate_approximately(
            results["fluent_amplitude"],
            results["query_amplitude"],
            tolerance=0.001,
            name="Fluent vs Query amplitude",
        ):
            return False

        if not validate_approximately(
            results["fluent_rms"],
            results["query_rms"],
            tolerance=0.001,
            name="Fluent vs Query RMS",
        ):
            return False

        # Built signal should have correct frequency
        if not validate_approximately(
            results["built_frequency"],
            1000.0,
            tolerance=0.01,
            name="Built signal frequency",
        ):
            return False

        # THD should be reasonable (harmonics present)
        # THD in dB: more negative = better (less distortion)
        # We want THD worse than -10 dB (i.e., greater than -10 dB, like -5 dB)
        # But we also don't want it TOO bad (should be better than 0 dB)
        if results["built_thd"] < -10.0:  # THD better than -10 dB (too clean)
            print(f"  ✗ THD too good: {results['built_thd']:.2f} dB (harmonics not strong enough)")
            return False
        if results["built_thd"] > 5.0:  # THD worse than 5 dB (too distorted)
            print(f"  ✗ THD too bad: {results['built_thd']:.2f} dB (excessive distortion)")
            return False
        print(f"  ✓ THD: {results['built_thd']:.2f} dB (harmonics detected)")

        self.success("All DSL operations validated!")
        self.info("\nKey takeaways:")
        self.info("  - Fluent APIs enable method chaining")
        self.info("  - Query-like syntax improves readability")
        self.info("  - SignalBuilder provides declarative construction")
        self.info("  - DSL makes code self-documenting")

        return True


if __name__ == "__main__":
    demo = DSLSyntaxDemo()
    success = demo.execute()
    sys.exit(0 if success else 1)
