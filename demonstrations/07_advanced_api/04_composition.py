"""Composition: Higher-Order Functions and Decorators

Demonstrates:
- oscura.compose() - Function composition
- oscura.curry() - Partial application
- Higher-order functions for signal processing
- Decorator patterns for measurements

IEEE Standards: N/A
Related Demos:
- 07_advanced_api/01_pipeline_api.py
- 07_advanced_api/02_dsl_syntax.py

Function composition and currying enable powerful abstraction patterns,
creating reusable and composable signal processing components.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable
from dataclasses import replace
from functools import partial, wraps
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
    rms,
)


def timer_decorator(func: Callable) -> Callable:
    """Decorator to time function execution."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        return result, elapsed

    return wrapper


def cached_measurement(func: Callable) -> Callable:
    """Decorator to cache measurement results."""
    cache = {}

    @wraps(func)
    def wrapper(trace):
        # Use trace id as cache key
        trace_id = id(trace)
        if trace_id not in cache:
            cache[trace_id] = func(trace)
        return cache[trace_id]

    wrapper.cache = cache
    wrapper.cache_hits = 0
    wrapper.cache_misses = 0
    return wrapper


def validated_measurement(min_val: float, max_val: float) -> Callable:
    """Decorator factory for validating measurement results."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            if not (min_val <= result <= max_val):
                raise ValueError(f"Measurement {result} outside valid range [{min_val}, {max_val}]")
            return result

        return wrapper

    return decorator


class CompositionDemo(BaseDemo):
    """Demonstrate function composition and higher-order patterns."""

    def __init__(self):
        """Initialize composition demonstration."""
        super().__init__(
            name="composition",
            description="Higher-order functions and decorator patterns",
            capabilities=[
                "oscura.compose",
                "oscura.curry",
                "decorators",
                "higher_order_functions",
            ],
        )

    def generate_test_data(self) -> dict:
        """Generate test signals."""
        # Clean signal
        signal = generate_sine_wave(
            frequency=1000.0,
            amplitude=1.0,
            duration=0.01,
            sample_rate=100e3,
            offset=0.2,
        )

        # Add noise
        noise = np.random.normal(0, 0.05, len(signal.data))
        signal.data = signal.data + noise

        return {"signal": signal}

    def run_demonstration(self, data: dict) -> dict:
        """Run composition demonstration."""
        signal = data["signal"]

        self.section("Composition: Higher-Order Functions")

        # ===================================================================
        # Part 1: Function Composition
        # ===================================================================
        self.subsection("1. Function Composition")
        self.info("Combine functions into new functions")

        # Create composed functions
        def remove_dc(t):
            return high_pass(t, cutoff=100.0)

        def remove_noise(t):
            return low_pass(t, cutoff=5000.0)

        # Compose: remove_dc followed by remove_noise
        clean_signal = compose(remove_noise, remove_dc)

        # Apply composed function
        cleaned = clean_signal(signal)
        result_rms = rms(cleaned)

        self.info("Composition: remove_noise ∘ remove_dc")
        self.result("  Original RMS", f"{rms(signal):.4f}", "V")
        self.result("  Cleaned RMS", f"{result_rms:.4f}", "V")
        self.success("Composition creates reusable processing chains")

        # ===================================================================
        # Part 2: Currying and Partial Application
        # ===================================================================
        self.subsection("2. Currying and Partial Application")
        self.info("Create specialized functions from general ones")

        # Create specialized filter functions using partial application
        high_pass_100 = partial(high_pass, cutoff=100.0)
        high_pass_1k = partial(high_pass, cutoff=1000.0)
        low_pass_5k = partial(low_pass, cutoff=5000.0)

        # Use curried functions
        result1 = high_pass_100(signal)
        result2 = high_pass_1k(signal)
        result3 = low_pass_5k(signal)

        self.info("Curried filters:")
        self.result("  high_pass_100(signal)", f"{rms(result1):.4f}", "V")
        self.result("  high_pass_1k(signal)", f"{rms(result2):.4f}", "V")
        self.result("  low_pass_5k(signal)", f"{rms(result3):.4f}", "V")

        # Compose curried functions
        bandpass = compose(low_pass_5k, high_pass_100)
        bp_result = bandpass(signal)

        self.result("  bandpass(signal)", f"{rms(bp_result):.4f}", "V")
        self.success("Currying enables flexible function specialization")

        # ===================================================================
        # Part 3: Higher-Order Functions
        # ===================================================================
        self.subsection("3. Higher-Order Functions")
        self.info("Functions that operate on functions")

        def apply_filter_chain(signal, filters: list):
            """Higher-order function: apply sequence of filters."""
            result = signal
            for filt in filters:
                result = filt(result)
            return result

        # Define filter chain
        filters = [high_pass_100, low_pass_5k]

        # Apply using higher-order function
        filtered = apply_filter_chain(signal, filters)

        self.info("Filter chain: [high_pass_100, low_pass_5k]")
        self.result("  Result RMS", f"{rms(filtered):.4f}", "V")

        # Create measurement pipeline
        def measure_all(signal, measurements: list):
            """Higher-order function: apply multiple measurements."""
            return {name: func(signal) for name, func in measurements}

        # Define measurements
        measurements = [
            ("amplitude", amplitude),
            ("rms", rms),
        ]

        # Apply measurements
        results = measure_all(filtered, measurements)

        self.info("\nMeasurement results:")
        for name, value in results.items():
            self.result(f"  {name}", f"{value:.4f}", "V")

        self.success("Higher-order functions enable flexible processing")

        # ===================================================================
        # Part 4: Decorator Patterns
        # ===================================================================
        self.subsection("4. Decorator Patterns")
        self.info("Enhance functions with decorators")

        # Create decorated measurement
        @timer_decorator
        def measure_amplitude_timed(trace):
            return amplitude(trace)

        @timer_decorator
        def measure_rms_timed(trace):
            return rms(trace)

        # Use decorated functions
        amp_result, amp_time = measure_amplitude_timed(signal)
        rms_result, rms_time = measure_rms_timed(signal)

        self.info("Timed measurements:")
        self.result("  amplitude", f"{amp_result:.4f} V (took {amp_time * 1e6:.1f} µs)", "")
        self.result("  rms", f"{rms_result:.4f} V (took {rms_time * 1e6:.1f} µs)", "")

        self.success("Decorators add functionality without modifying functions")

        # ===================================================================
        # Part 5: Caching Decorator
        # ===================================================================
        self.subsection("5. Caching for Performance")
        self.info("Cache expensive computations")

        # Create cached measurement
        @cached_measurement
        def expensive_amplitude(trace):
            """Simulate expensive computation."""
            time.sleep(0.001)  # Simulate work
            return amplitude(trace)

        # First call (cache miss)
        start = time.time()
        result1 = expensive_amplitude(signal)
        time1 = time.time() - start

        # Second call (cache hit)
        start = time.time()
        result2 = expensive_amplitude(signal)
        time2 = time.time() - start

        self.info("Cached measurement performance:")
        self.result("  First call (cache miss)", f"{time1 * 1000:.2f}", "ms")
        self.result("  Second call (cache hit)", f"{time2 * 1000:.2f}", "ms")
        self.result("  Speedup", f"{time1 / time2:.1f}", "x")
        self.result("  Result unchanged", result1 == result2, "")

        self.success("Caching dramatically improves performance for repeated calls")

        # ===================================================================
        # Part 6: Validation Decorator
        # ===================================================================
        self.subsection("6. Validation Decorators")
        self.info("Automatic result validation")

        # Create validated measurement
        @validated_measurement(0.0, 10.0)
        def measure_amplitude_validated(trace):
            return amplitude(trace)

        try:
            valid_result = measure_amplitude_validated(signal)
            self.result("  Valid measurement", f"{valid_result:.4f}", "V")
            self.success("Validation passed")
        except ValueError as e:
            self.error(f"Validation failed: {e}")

        # Test with invalid data
        invalid_signal = replace(signal, data=signal.data * 100)  # Unrealistic amplitude

        try:
            invalid_result = measure_amplitude_validated(invalid_signal)
            self.error(f"Should have failed validation: {invalid_result}")
        except ValueError:
            self.success("Invalid data correctly rejected")

        # ===================================================================
        # Part 7: Practical Example - Measurement Suite
        # ===================================================================
        self.subsection("7. Practical Example: Measurement Suite")
        self.info("Combine patterns for robust measurement system")

        def create_measurement_suite(measurements: list) -> Callable:
            """Create a composable measurement suite."""

            @timer_decorator
            def suite(trace):
                results = {}
                for name, func in measurements:
                    try:
                        results[name] = func(trace)
                    except Exception as e:
                        results[name] = f"Error: {e}"
                return results

            return suite

        # Create suite
        suite = create_measurement_suite(
            [
                ("amplitude", amplitude),
                ("rms", rms),
            ]
        )

        # Execute suite
        suite_results, suite_time = suite(filtered)

        self.info("Measurement suite results:")
        for name, value in suite_results.items():
            if isinstance(value, str):
                self.warning(f"  {name}: {value}")
            else:
                self.result(f"  {name}", f"{value:.4f}", "V")
        self.result("Total execution time", f"{suite_time * 1000:.2f}", "ms")

        self.success("Composition patterns enable robust, maintainable systems")

        return {
            "composed_rms": result_rms,
            "curried_rms": rms(bp_result),
            "filtered_amplitude": results["amplitude"],
            "filtered_rms": results["rms"],
            "cached_result": result1,
            "validated_result": valid_result,
            "suite_amplitude": suite_results["amplitude"],
        }

    def validate(self, results: dict) -> bool:
        """Validate composition results."""
        self.info("Validating composition patterns...")

        # Composed and curried should produce similar results
        if not validate_approximately(
            results["composed_rms"],
            results["curried_rms"],
            tolerance=0.05,
            name="Composed vs Curried",
        ):
            return False

        # Higher-order function results should be consistent
        if not validate_approximately(
            results["filtered_amplitude"],
            results["suite_amplitude"],
            tolerance=0.01,
            name="HOF consistency",
        ):
            return False

        # Cached result should be valid
        if results["cached_result"] <= 0:
            print(f"  ✗ Cached result invalid: {results['cached_result']}")
            return False
        print(f"  ✓ Cached result valid: {results['cached_result']:.4f}")

        # Validated result should be in range
        if not (0.0 <= results["validated_result"] <= 10.0):
            print(f"  ✗ Validated result out of range: {results['validated_result']}")
            return False
        print(f"  ✓ Validated result in range: {results['validated_result']:.4f}")

        self.success("All composition patterns validated!")
        self.info("\nKey takeaways:")
        self.info("  - Composition creates reusable processing chains")
        self.info("  - Currying enables function specialization")
        self.info("  - Higher-order functions provide flexibility")
        self.info("  - Decorators add functionality transparently")

        return True


if __name__ == "__main__":
    demo = CompositionDemo()
    success = demo.execute()
    sys.exit(0 if success else 1)
