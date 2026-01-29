"""Operator Overloading: Natural Mathematical Expressions

Demonstrates:
- oscura.add(), subtract(), multiply(), divide() - Arithmetic operations
- Signal comparison using built-in operations
- Natural mathematical expressions for signal math
- Operator-based signal transformations

IEEE Standards: N/A
Related Demos:
- 07_advanced_api/02_dsl_syntax.py
- 02_basic_analysis/06_math_operations.py

Operator overloading enables natural mathematical expressions for signal
processing, making code more intuitive and readable.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

# Add demonstrations to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demonstrations.common import BaseDemo, generate_sine_wave, validate_approximately
from oscura import (
    add,
    amplitude,
    divide,
    frequency,
    multiply,
    rms,
    subtract,
)


class SignalWithOperators:
    """Wrapper class that supports operator overloading for signals."""

    def __init__(self, trace):
        """Initialize signal wrapper."""
        self.trace = trace

    def __hash__(self):
        """Return hash based on trace data."""
        return hash(id(self.trace))

    def __add__(self, other):
        """Add two signals using + operator."""
        if isinstance(other, SignalWithOperators):
            return SignalWithOperators(add(self.trace, other.trace))
        else:
            # Scalar addition
            result = replace(self.trace, data=self.trace.data + other)
            return SignalWithOperators(result)

    def __sub__(self, other):
        """Subtract signals using - operator."""
        if isinstance(other, SignalWithOperators):
            return SignalWithOperators(subtract(self.trace, other.trace))
        else:
            # Scalar subtraction
            result = replace(self.trace, data=self.trace.data - other)
            return SignalWithOperators(result)

    def __mul__(self, other):
        """Multiply signals using * operator."""
        if isinstance(other, SignalWithOperators):
            return SignalWithOperators(multiply(self.trace, other.trace))
        else:
            # Scalar multiplication
            result = replace(self.trace, data=self.trace.data * other)
            return SignalWithOperators(result)

    def __truediv__(self, other):
        """Divide signals using / operator."""
        if isinstance(other, SignalWithOperators):
            return SignalWithOperators(divide(self.trace, other.trace))
        else:
            # Scalar division
            result = replace(self.trace, data=self.trace.data / other)
            return SignalWithOperators(result)

    def __neg__(self):
        """Negate signal using - operator."""
        result = replace(self.trace, data=-self.trace.data)
        return SignalWithOperators(result)

    def __abs__(self):
        """Absolute value using abs()."""
        result = replace(self.trace, data=np.abs(self.trace.data))
        return SignalWithOperators(result)

    # Comparison operators
    def __eq__(self, other):
        """Equality comparison."""
        if isinstance(other, SignalWithOperators):
            return np.allclose(self.trace.data, other.trace.data, rtol=1e-5)
        return False

    def __gt__(self, value):
        """Greater than threshold (returns boolean array)."""
        return self.trace.data > value

    def __lt__(self, value):
        """Less than threshold (returns boolean array)."""
        return self.trace.data < value

    # Convenience methods
    def amplitude(self):
        """Get amplitude."""
        return amplitude(self.trace)

    def rms(self):
        """Get RMS value."""
        return rms(self.trace)

    def frequency(self):
        """Get frequency."""
        return frequency(self.trace)


class OperatorsDemo(BaseDemo):
    """Demonstrate operator overloading for natural signal math."""

    def __init__(self):
        """Initialize operators demonstration."""
        super().__init__(
            name="operators",
            description="Natural mathematical expressions with operator overloading",
            capabilities=[
                "oscura.add",
                "oscura.subtract",
                "oscura.multiply",
                "oscura.divide",
                "operator_overloading",
            ],
        )

    def generate_test_data(self) -> dict:
        """Generate test signals."""
        # Create test signals
        signal1 = generate_sine_wave(
            frequency=1000.0, amplitude=1.0, duration=0.01, sample_rate=100e3
        )

        signal2 = generate_sine_wave(
            frequency=2000.0, amplitude=0.5, duration=0.01, sample_rate=100e3
        )

        signal3 = generate_sine_wave(
            frequency=3000.0, amplitude=0.3, duration=0.01, sample_rate=100e3
        )

        return {
            "signal1": signal1,
            "signal2": signal2,
            "signal3": signal3,
        }

    def run_demonstration(self, data: dict) -> dict:
        """Run operators demonstration."""
        s1 = SignalWithOperators(data["signal1"])
        s2 = SignalWithOperators(data["signal2"])
        s3 = SignalWithOperators(data["signal3"])

        self.section("Operators: Natural Mathematical Expressions")

        # ===================================================================
        # Part 1: Basic Arithmetic Operators
        # ===================================================================
        self.subsection("1. Basic Arithmetic with + - * /")
        self.info("Natural mathematical expressions for signals")

        # Addition
        sum_signal = s1 + s2
        self.info("Expression: signal1 + signal2")
        self.result("  Result amplitude", f"{sum_signal.amplitude():.4f}", "V")

        # Subtraction
        diff_signal = s1 - s2
        self.info("\nExpression: signal1 - signal2")
        self.result("  Result amplitude", f"{diff_signal.amplitude():.4f}", "V")

        # Multiplication
        prod_signal = s1 * s2
        self.info("\nExpression: signal1 * signal2")
        self.result("  Result RMS", f"{prod_signal.rms():.4f}", "V")

        # Division
        quot_signal = s1 / s2
        self.info("\nExpression: signal1 / signal2")
        self.result("  Result RMS", f"{quot_signal.rms():.4f}", "V")

        self.success("Arithmetic operators work naturally with signals")

        # ===================================================================
        # Part 2: Scalar Operations
        # ===================================================================
        self.subsection("2. Scalar Operations")
        self.info("Combine signals with scalar values")

        # Scale signal
        scaled = s1 * 2.0
        self.info("Expression: signal * 2.0")
        self.result("  Original amplitude", f"{s1.amplitude():.4f}", "V")
        self.result("  Scaled amplitude", f"{scaled.amplitude():.4f}", "V")
        self.result("  Scale factor", f"{scaled.amplitude() / s1.amplitude():.2f}", "x")

        # Add DC offset
        offset_signal = s1 + 0.5
        self.info("\nExpression: signal + 0.5")
        original_mean = np.mean(s1.trace.data)
        offset_mean = np.mean(offset_signal.trace.data)
        self.result("  Original mean", f"{original_mean:.4f}", "V")
        self.result("  Offset mean", f"{offset_mean:.4f}", "V")
        self.result("  Offset added", f"{offset_mean - original_mean:.4f}", "V")

        self.success("Scalar operations work intuitively")

        # ===================================================================
        # Part 3: Complex Expressions
        # ===================================================================
        self.subsection("3. Complex Mathematical Expressions")
        self.info("Combine multiple operations in one expression")

        # Complex expression: (s1 + s2) * 0.5 - s3
        complex_result = (s1 + s2) * 0.5 - s3

        self.info("Expression: (signal1 + signal2) * 0.5 - signal3")
        self.result("  Result amplitude", f"{complex_result.amplitude():.4f}", "V")
        self.result("  Result RMS", f"{complex_result.rms():.4f}", "V")

        # Weighted sum: 0.5*s1 + 0.3*s2 + 0.2*s3
        weighted = s1 * 0.5 + s2 * 0.3 + s3 * 0.2

        self.info("\nExpression: 0.5*s1 + 0.3*s2 + 0.2*s3")
        self.result("  Weighted amplitude", f"{weighted.amplitude():.4f}", "V")
        self.result("  Weighted RMS", f"{weighted.rms():.4f}", "V")

        self.success("Complex expressions enable sophisticated signal processing")

        # ===================================================================
        # Part 4: Unary Operators
        # ===================================================================
        self.subsection("4. Unary Operators (Negation, Absolute Value)")
        self.info("Single-argument operations")

        # Negation
        negated = -s1
        self.info("Expression: -signal")
        self.result("  Original RMS", f"{s1.rms():.4f}", "V")
        self.result("  Negated RMS", f"{negated.rms():.4f}", "V")
        self.success("Negation preserves magnitude")

        # Absolute value
        abs_signal = abs(s1)
        self.info("\nExpression: abs(signal)")
        self.result("  Original min value", f"{np.min(s1.trace.data):.4f}", "V")
        self.result("  Absolute min value", f"{np.min(abs_signal.trace.data):.4f}", "V")
        self.success("Absolute value rectifies signal")

        # ===================================================================
        # Part 5: Comparison Operators
        # ===================================================================
        self.subsection("5. Comparison and Thresholding")
        self.info("Boolean operations for signal analysis")

        # Threshold detection
        threshold = 0.5
        above_threshold = s1 > threshold
        below_threshold = s1 < -threshold

        count_above = np.sum(above_threshold)
        count_below = np.sum(below_threshold)
        total_samples = len(s1.trace.data)

        self.info(f"Threshold: ±{threshold}V")
        self.result(
            "  Samples above +0.5V", count_above, f"({count_above / total_samples * 100:.1f}%)"
        )
        self.result(
            "  Samples below -0.5V", count_below, f"({count_below / total_samples * 100:.1f}%)"
        )

        # Signal equality
        equal_signals = s1 == s1
        different_signals = s1 == s2

        self.info("\nSignal comparison:")
        self.result("  signal1 == signal1", equal_signals, "")
        self.result("  signal1 == signal2", different_signals, "")

        self.success("Comparison operators enable threshold analysis")

        # ===================================================================
        # Part 6: Practical Application - Differential Signaling
        # ===================================================================
        self.subsection("6. Practical Application: Differential Signaling")
        self.info("Use operators for differential signal processing")

        # Simulate differential pair
        positive = s1
        negative = -s1 * 0.9 + SignalWithOperators(
            generate_sine_wave(
                frequency=100.0,
                amplitude=0.05,
                duration=0.01,
                sample_rate=100e3,  # Common-mode noise
            )
        )

        # Differential signal
        differential = positive - negative
        common_mode = (positive + negative) * 0.5

        self.info("Differential signaling:")
        self.result("  Positive amplitude", f"{positive.amplitude():.4f}", "V")
        self.result("  Negative amplitude", f"{negative.amplitude():.4f}", "V")
        self.result("  Differential amplitude", f"{differential.amplitude():.4f}", "V")
        self.result("  Common-mode amplitude", f"{common_mode.amplitude():.4f}", "V")

        cmrr = 20 * np.log10(differential.amplitude() / (common_mode.amplitude() + 1e-10))
        self.result("  CMRR", f"{cmrr:.1f}", "dB")

        self.success("Operators simplify differential signal analysis")

        return {
            "sum_amplitude": sum_signal.amplitude(),
            "diff_amplitude": diff_signal.amplitude(),
            "scaled_amplitude": scaled.amplitude(),
            "weighted_amplitude": weighted.amplitude(),
            "negated_rms": negated.rms(),
            "differential_amplitude": differential.amplitude(),
            "common_mode_amplitude": common_mode.amplitude(),
        }

    def validate(self, results: dict) -> bool:
        """Validate operator results."""
        self.info("Validating operator operations...")

        # Negation should preserve RMS
        s1_rms = rms(
            generate_sine_wave(frequency=1000.0, amplitude=1.0, duration=0.01, sample_rate=100e3)
        )
        if not validate_approximately(
            results["negated_rms"], s1_rms, tolerance=0.01, name="Negation preserves RMS"
        ):
            return False

        # Scaling should work correctly
        expected_scaled = (
            amplitude(
                generate_sine_wave(
                    frequency=1000.0, amplitude=1.0, duration=0.01, sample_rate=100e3
                )
            )
            * 2.0
        )
        if not validate_approximately(
            results["scaled_amplitude"],
            expected_scaled,
            tolerance=0.01,
            name="Scaling factor",
        ):
            return False

        # Differential should suppress common mode
        cmrr = 20 * np.log10(
            results["differential_amplitude"] / (results["common_mode_amplitude"] + 1e-10)
        )
        if cmrr < 20.0:  # Should have at least 20 dB CMRR
            print(f"  ✗ CMRR too low: {cmrr:.1f} dB (expected > 20 dB)")
            return False
        print(f"  ✓ CMRR: {cmrr:.1f} dB (good common-mode rejection)")

        self.success("All operator operations validated!")
        self.info("\nKey takeaways:")
        self.info("  - Operators enable natural mathematical expressions")
        self.info("  - Scalar operations work intuitively")
        self.info("  - Complex expressions improve code readability")
        self.info("  - Comparison operators enable threshold analysis")

        return True


if __name__ == "__main__":
    demo = OperatorsDemo()
    success = demo.execute()
    sys.exit(0 if success else 1)
