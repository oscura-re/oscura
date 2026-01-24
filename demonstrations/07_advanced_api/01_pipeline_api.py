"""Pipeline API: Multi-stage Processing with pipe() and compose()

Demonstrates:
- oscura.pipe() - Left-to-right function composition
- oscura.compose() - Right-to-left function composition
- Function chaining for multi-stage processing
- Error handling in processing chains

IEEE Standards: N/A
Related Demos:
- 07_advanced_api/02_dsl_syntax.py
- 07_advanced_api/04_composition.py

Functional composition enables elegant multi-stage processing workflows,
chaining operations like filter → analyze → measure with clean syntax.
"""

from __future__ import annotations

import sys
import time
from functools import partial
from pathlib import Path

import numpy as np

# Add demonstrations to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demonstrations.common import BaseDemo, generate_sine_wave, validate_approximately
from oscura import (
    amplitude,
    compose,
    high_pass,
    low_pass,
    pipe,
    rms,
    thd,
)


class PipelineAPIDemo(BaseDemo):
    """Demonstrate pipeline composition with pipe() and compose()."""

    def __init__(self):
        """Initialize pipeline API demonstration."""
        super().__init__(
            name="pipeline_api",
            description="Multi-stage processing with pipe() and compose()",
            capabilities=[
                "oscura.pipe",
                "oscura.compose",
                "functional_composition",
            ],
        )

    def generate_test_data(self) -> dict:
        """Generate test signals for pipeline processing."""
        # Signal with DC offset and noise (needs filtering)
        noisy_signal = generate_sine_wave(
            frequency=1000.0,
            amplitude=1.0,
            duration=0.01,
            sample_rate=100e3,
            offset=0.5,  # DC offset to be removed
        )

        # Add noise
        noise = np.random.normal(0, 0.05, len(noisy_signal.data))
        noisy_signal.data = noisy_signal.data + noise

        # Clean reference signal
        clean_signal = generate_sine_wave(
            frequency=1000.0,
            amplitude=1.0,
            duration=0.01,
            sample_rate=100e3,
        )

        return {
            "noisy_signal": noisy_signal,
            "clean_signal": clean_signal,
        }

    def run_demonstration(self, data: dict) -> dict:
        """Run pipeline API demonstration."""
        noisy = data["noisy_signal"]
        _clean = data["clean_signal"]  # For reference

        self.section("Pipeline API: Multi-Stage Processing")

        # ===================================================================
        # Part 1: Basic pipe() Composition (Left-to-Right)
        # ===================================================================
        self.subsection("1. pipe(): Left-to-Right Composition")
        self.info("Apply operations sequentially: trace → high_pass → low_pass → rms")

        # Functional pipeline using pipe()
        start_time = time.time()
        result_rms = pipe(
            noisy,
            lambda t: high_pass(t, cutoff=100.0),
            lambda t: low_pass(t, cutoff=5000.0),
            rms,
        )
        pipe_time = time.time() - start_time

        self.info("Pipeline: noisy → high_pass(100Hz) → low_pass(5kHz) → rms()")
        self.result("Filtered RMS", f"{result_rms:.4f}", "V")
        self.result("Execution time", f"{pipe_time * 1000:.2f}", "ms")

        # Compare with unfiltered
        raw_rms = rms(noisy)
        self.result("Raw RMS (unfiltered)", f"{raw_rms:.4f}", "V")
        self.result("Improvement", f"{(raw_rms - result_rms) / raw_rms * 100:.1f}", "%")

        self.success("pipe() enables readable left-to-right composition")

        # ===================================================================
        # Part 2: compose() Function (Right-to-Left)
        # ===================================================================
        self.subsection("2. compose(): Right-to-Left Composition")
        self.info("Create reusable processing functions")

        # Create composed functions
        hp_filter = partial(high_pass, cutoff=100.0)
        lp_filter = partial(low_pass, cutoff=5000.0)

        # Compose filters (right-to-left: applies hp_filter first, then lp_filter)
        clean_signal_func = compose(lp_filter, hp_filter)

        # Apply composed function
        start_time = time.time()
        filtered = clean_signal_func(noisy)
        composed_rms = rms(filtered)
        compose_time = time.time() - start_time

        self.info("Composed function: compose(lp_filter, hp_filter)")
        self.result("Composed RMS", f"{composed_rms:.4f}", "V")
        self.result("Execution time", f"{compose_time * 1000:.2f}", "ms")

        # Verify pipe() and compose() produce same result
        self.result("Match with pipe()", f"{abs(composed_rms - result_rms) < 1e-10}", "")
        self.success("compose() creates reusable processing chains")

        # ===================================================================
        # Part 3: Complex Multi-Stage Pipeline
        # ===================================================================
        self.subsection("3. Complex Multi-Stage Analysis")
        self.info("Chain filtering and multiple measurements")

        # Multi-stage analysis with pipe()
        start_time = time.time()

        filtered_trace = pipe(
            noisy,
            lambda t: high_pass(t, cutoff=100.0),
            lambda t: low_pass(t, cutoff=5000.0),
        )

        # Perform multiple measurements on filtered trace
        measurements = {
            "amplitude": amplitude(filtered_trace),
            "rms": rms(filtered_trace),
            "thd": thd(filtered_trace),  # Auto-detects fundamental from FFT
        }

        analysis_time = time.time() - start_time

        self.info("Multi-stage analysis results:")
        self.result("  - Amplitude", f"{measurements['amplitude']:.4f}", "V")
        self.result("  - RMS", f"{measurements['rms']:.4f}", "V")
        self.result("  - THD", f"{measurements['thd']:.2f}", "dB")
        self.result("Total processing time", f"{analysis_time * 1000:.2f}", "ms")

        self.success("Pipelines enable complex multi-measurement workflows")

        # ===================================================================
        # Part 4: Reusable Processing Functions
        # ===================================================================
        self.subsection("4. Building Reusable Processing Functions")
        self.info("Create library of standard processing chains")

        # Define reusable processing chains
        def create_signal_cleaner(hp_cutoff: float, lp_cutoff: float):
            """Factory for signal cleaning functions."""
            return compose(
                partial(low_pass, cutoff=lp_cutoff),
                partial(high_pass, cutoff=hp_cutoff),
            )

        # Create specialized cleaners
        audio_cleaner = create_signal_cleaner(hp_cutoff=20.0, lp_cutoff=20000.0)
        power_cleaner = create_signal_cleaner(hp_cutoff=50.0, lp_cutoff=5000.0)

        # Use reusable cleaners
        audio_cleaned = audio_cleaner(noisy)
        power_cleaned = power_cleaner(noisy)

        self.info("Reusable processing functions:")
        self.result("  Audio cleaner RMS", f"{rms(audio_cleaned):.4f}", "V")
        self.result("  Power cleaner RMS", f"{rms(power_cleaned):.4f}", "V")

        self.success("Factories enable parameterized reusable pipelines")

        # ===================================================================
        # Part 5: Performance Comparison
        # ===================================================================
        self.subsection("5. Performance: Functional vs Sequential")
        self.info("Compare functional composition with sequential code")

        # Sequential processing (traditional)
        start_time = time.time()
        temp1 = high_pass(noisy, cutoff=100.0)
        temp2 = low_pass(temp1, cutoff=5000.0)
        seq_result = rms(temp2)
        sequential_time = time.time() - start_time

        # Functional processing
        start_time = time.time()
        func_result = pipe(
            noisy,
            lambda t: high_pass(t, cutoff=100.0),
            lambda t: low_pass(t, cutoff=5000.0),
            rms,
        )
        functional_time = time.time() - start_time

        self.result("Sequential time", f"{sequential_time * 1000:.2f}", "ms")
        self.result("Functional time", f"{functional_time * 1000:.2f}", "ms")
        self.result("Results match", f"{abs(seq_result - func_result) < 1e-10}", "")

        self.success("Functional composition provides cleaner syntax with similar performance")

        # ===================================================================
        # Part 6: Error Handling in Pipelines
        # ===================================================================
        self.subsection("6. Error Handling in Processing Chains")
        self.info("Graceful error handling with try/except")

        def safe_measurement(measure_func, trace):
            """Wrapper for safe measurement."""
            try:
                return measure_func(trace)
            except Exception as e:
                return f"Error: {str(e)[:50]}"

        # Create safe pipeline
        safe_results = {
            "amplitude": safe_measurement(amplitude, filtered_trace),
            "rms": safe_measurement(rms, filtered_trace),
        }

        self.info("Safe measurements:")
        for name, value in safe_results.items():
            if isinstance(value, str):
                self.warning(f"  {name}: {value}")
            else:
                self.result(f"  {name}", f"{value:.4f}", "V")

        self.success("Error handling ensures robust pipelines")

        # ===================================================================
        # Part 7: Practical Example - Complete Analysis Pipeline
        # ===================================================================
        self.subsection("7. Complete Analysis Pipeline")
        self.info("Real-world signal processing workflow")

        # Define complete pipeline
        def complete_analysis(signal):
            """Complete signal analysis pipeline."""
            # Filter
            cleaned = pipe(
                signal,
                partial(high_pass, cutoff=100.0),
                partial(low_pass, cutoff=5000.0),
            )

            # Measure
            return {
                "filtered_amplitude": amplitude(cleaned),
                "filtered_rms": rms(cleaned),
                "filtered_thd": thd(cleaned),  # Auto-detects fundamental from FFT
            }

        # Execute complete pipeline
        start_time = time.time()
        complete_results = complete_analysis(noisy)
        complete_time = time.time() - start_time

        self.info("Complete analysis results:")
        for name, value in complete_results.items():
            if "thd" in name:
                self.result(f"  {name}", f"{value:.2f}", "dB")
            else:
                self.result(f"  {name}", f"{value:.4f}", "V")
        self.result("Total time", f"{complete_time * 1000:.2f}", "ms")

        self.success("Functional pipelines enable elegant, maintainable workflows")

        return {
            "pipe_rms": result_rms,
            "composed_rms": composed_rms,
            "analysis_amplitude": measurements["amplitude"],
            "analysis_rms": measurements["rms"],
            "power_cleaner_rms": rms(power_cleaned),
            "complete_results": complete_results,
        }

    def validate(self, results: dict) -> bool:
        """Validate pipeline results."""
        self.info("Validating pipeline processing...")

        # pipe() and compose() should match
        if not validate_approximately(
            results["pipe_rms"],
            results["composed_rms"],
            tolerance=1e-9,
            name="pipe vs compose",
        ):
            return False

        # Analysis should be consistent
        if not validate_approximately(
            results["analysis_rms"],
            results["pipe_rms"],
            tolerance=0.01,
            name="Analysis consistency",
        ):
            return False

        # Complete analysis should have valid results
        complete = results["complete_results"]
        if not (0.5 < complete["filtered_amplitude"] < 3.0):
            print(f"  ✗ Complete analysis amplitude out of range: {complete['filtered_amplitude']}")
            return False
        print(f"  ✓ Complete analysis amplitude: {complete['filtered_amplitude']:.4f} V")

        self.success("All pipeline operations validated!")
        self.info("\nKey takeaways:")
        self.info("  - pipe() applies operations left-to-right (intuitive)")
        self.info("  - compose() creates reusable processing functions")
        self.info("  - Functional composition eliminates temporary variables")
        self.info("  - Factories enable parameterized pipelines")

        return True


if __name__ == "__main__":
    demo = PipelineAPIDemo()
    success = demo.execute()
    sys.exit(0 if success else 1)
