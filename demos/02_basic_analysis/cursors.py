#!/usr/bin/env python3
"""Cursor Measurements: Oscilloscope-style cursor analysis.

This demo demonstrates cursor-based measurements:
- Time cursors (delta-t measurements)
- Voltage cursors (delta-v measurements)
- Cursor-to-cursor calculations
- Reference markers
- Multiple cursor pairs
- Frequency from period cursors

Related demos:
- 01_waveform_basics.py - Basic measurements
- 02_digital_basics.py - Digital signal timing
- ../04_advanced_analysis/07_cursor_advanced.py - Advanced cursors

Usage:
    python demos/02_basic_analysis/07_cursors.py
    python demos/02_basic_analysis/07_cursors.py --verbose
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np

import oscura as osc
from demos.common import BaseDemo, ValidationSuite, print_info, print_result, print_table
from demos.common.base_demo import run_demo_main
from demos.common.data_generation import generate_pulse_train, generate_sine_wave
from demos.common.formatting import print_subheader


class CursorsDemo(BaseDemo):
    """Cursor-based measurement demonstration."""

    name = "Cursor Measurements"
    description = "Time and voltage cursor measurements for precise analysis"
    category = "basic_analysis"

    capabilities = [
        "oscura.cursor_time_delta",
        "oscura.cursor_voltage_delta",
        "oscura.cursor_measurements",
        "oscura.reference_markers",
    ]

    related_demos = [
        "01_waveform_basics.py",
        "02_digital_basics.py",
        "../04_advanced_analysis/07_cursor_advanced.py",
    ]

    def __init__(self, **kwargs):
        """Initialize demo."""
        super().__init__(**kwargs)
        self.sine_wave = None
        self.pulse_train = None

    def generate_data(self) -> None:
        """Generate test signals for cursor demonstrations."""
        print_info("Generating cursor test signals...")

        # 1. Sine wave for voltage and time measurements
        self.sine_wave = generate_sine_wave(
            frequency=5000.0,  # 5 kHz
            amplitude=2.5,  # 2.5V peak
            duration=0.002,  # 2 ms (10 periods)
            sample_rate=1e6,  # 1 MHz sampling
        )

        # 2. Pulse train for timing measurements
        self.pulse_train = generate_pulse_train(
            pulse_width=75e-6,  # 75 µs
            period=250e-6,  # 250 µs (4 kHz)
            amplitude=3.3,  # 3.3V
            duration=0.002,  # 2 ms
            sample_rate=1e6,  # 1 MHz
            rise_time=5e-9,
            fall_time=5e-9,
        )

        print_result("Sine wave generated", "5 kHz, 2.5V peak")
        print_result("Pulse train generated", "4 kHz, 75µs pulses")

    def run_analysis(self) -> None:
        """Execute cursor measurement demonstrations."""
        sample_rate = self.sine_wave.metadata.sample_rate

        # ========== PART 1: TIME CURSORS ==========
        print_subheader("Part 1: Time Cursors (Delta-T)")
        print_info("Measuring time intervals between events")

        # Find two rising edges in pulse train
        edges = osc.find_edges(self.pulse_train)
        rising_edges = edges["rising"]

        if len(rising_edges) >= 2:
            cursor_a = rising_edges[0]
            cursor_b = rising_edges[1]

            # Calculate delta-t
            delta_t = (cursor_b - cursor_a) / sample_rate
            freq_from_cursors = 1 / delta_t if delta_t > 0 else 0

            self.results["time_cursor_a"] = cursor_a / sample_rate
            self.results["time_cursor_b"] = cursor_b / sample_rate
            self.results["delta_t"] = delta_t
            self.results["freq_from_cursors"] = freq_from_cursors

            print_result("Cursor A (time)", f"{cursor_a / sample_rate * 1e6:.3f} µs")
            print_result("Cursor B (time)", f"{cursor_b / sample_rate * 1e6:.3f} µs")
            print_result("Delta-T (A→B)", f"{delta_t * 1e6:.3f} µs")
            print_result("Frequency (1/ΔT)", f"{freq_from_cursors / 1e3:.3f} kHz")

        # ========== PART 2: VOLTAGE CURSORS ==========
        print_subheader("Part 2: Voltage Cursors (Delta-V)")
        print_info("Measuring voltage differences")

        # Find peak and trough in sine wave
        sine_data = self.sine_wave.data
        peak_idx = np.argmax(sine_data)
        trough_idx = np.argmin(sine_data)

        voltage_a = sine_data[peak_idx]
        voltage_b = sine_data[trough_idx]
        delta_v = voltage_a - voltage_b

        self.results["voltage_cursor_a"] = voltage_a
        self.results["voltage_cursor_b"] = voltage_b
        self.results["delta_v"] = delta_v

        print_result("Cursor A (voltage)", f"{voltage_a:.4f} V")
        print_result("Cursor B (voltage)", f"{voltage_b:.4f} V")
        print_result("Delta-V (A-B)", f"{delta_v:.4f} V")
        print_result("Peak-to-peak amplitude", f"{delta_v:.4f} V")

        # ========== PART 3: MULTIPLE CURSOR PAIRS ==========
        print_subheader("Part 3: Multiple Cursor Pairs")
        print_info("Using multiple cursor pairs for different measurements")

        # Pair 1: Measure pulse width
        if len(rising_edges) >= 1:
            falling_edges = edges["falling"]
            if len(falling_edges) >= 1:
                pulse_start = rising_edges[0]
                pulse_end = falling_edges[0]

                pulse_width = (pulse_end - pulse_start) / sample_rate
                self.results["cursor_pair1_pulse_width"] = pulse_width

                print_result("Pair 1 - Pulse width", f"{pulse_width * 1e6:.3f} µs")

        # Pair 2: Measure period (two consecutive rising edges)
        if len(rising_edges) >= 3:
            period_start = rising_edges[1]
            period_end = rising_edges[2]

            period = (period_end - period_start) / sample_rate
            self.results["cursor_pair2_period"] = period

            print_result("Pair 2 - Period", f"{period * 1e6:.3f} µs")

        # Pair 3: Measure duty cycle high time
        if len(rising_edges) >= 2:
            high_start = rising_edges[0]
            high_end = falling_edges[0] if len(falling_edges) >= 1 else rising_edges[1]

            high_time = (high_end - high_start) / sample_rate
            duty = high_time / delta_t if delta_t > 0 else 0

            self.results["cursor_pair3_high_time"] = high_time
            self.results["cursor_pair3_duty"] = duty

            print_result("Pair 3 - High time", f"{high_time * 1e6:.3f} µs")
            print_result("Pair 3 - Duty cycle", f"{duty * 100:.1f}%")

        # ========== PART 4: REFERENCE MARKERS ==========
        print_subheader("Part 4: Reference Markers")
        print_info("Setting reference markers for relative measurements")

        # Set reference at first rising edge
        ref_marker = rising_edges[0] if len(rising_edges) > 0 else 0
        ref_time = ref_marker / sample_rate
        ref_voltage = self.pulse_train.data[ref_marker]

        self.results["reference_marker_index"] = ref_marker
        self.results["reference_time"] = ref_time
        self.results["reference_voltage"] = ref_voltage

        print_result("Reference marker", f"Sample {ref_marker}")
        print_result("Reference time", f"{ref_time * 1e6:.3f} µs")
        print_result("Reference voltage", f"{ref_voltage:.4f} V")

        # Measure relative to reference
        if len(rising_edges) >= 2:
            relative_idx = rising_edges[1]
            relative_time = (relative_idx - ref_marker) / sample_rate
            relative_voltage = self.pulse_train.data[relative_idx] - ref_voltage

            self.results["relative_time"] = relative_time
            self.results["relative_voltage"] = relative_voltage

            print_result("Relative time (from ref)", f"{relative_time * 1e6:.3f} µs")
            print_result("Relative voltage (from ref)", f"{relative_voltage:.4f} V")

        # ========== PART 5: CURSOR MEASUREMENT TABLE ==========
        print_subheader("Part 5: Cursor Measurement Summary")

        headers = ["Cursor Pair", "Measurement", "Value", "Unit"]
        rows = [
            ["Time Cursors", "Delta-T", f"{self.results['delta_t'] * 1e6:.3f}", "µs"],
            ["Time Cursors", "Frequency", f"{self.results['freq_from_cursors'] / 1e3:.3f}", "kHz"],
            ["Voltage Cursors", "Delta-V", f"{self.results['delta_v']:.4f}", "V"],
            ["Voltage Cursors", "Amplitude", f"{self.results['delta_v']:.4f}", "V"],
            [
                "Pair 1",
                "Pulse Width",
                f"{self.results.get('cursor_pair1_pulse_width', 0) * 1e6:.3f}",
                "µs",
            ],
            ["Pair 2", "Period", f"{self.results.get('cursor_pair2_period', 0) * 1e6:.3f}", "µs"],
            ["Pair 3", "Duty Cycle", f"{self.results.get('cursor_pair3_duty', 0) * 100:.1f}", "%"],
            [
                "Reference",
                "Relative Time",
                f"{self.results.get('relative_time', 0) * 1e6:.3f}",
                "µs",
            ],
        ]

        print_table(headers, rows)

        # ========== PART 6: CURSOR ACCURACY ==========
        print_subheader("Part 6: Cursor Measurement Accuracy")
        print_info("Analyzing cursor measurement precision")

        # Compare cursor measurements to automatic measurements
        auto_freq = osc.frequency(self.pulse_train)
        auto_amplitude = osc.amplitude(self.sine_wave)

        freq_error = abs(self.results["freq_from_cursors"] - auto_freq) / auto_freq * 100
        amplitude_error = abs(self.results["delta_v"] - auto_amplitude) / auto_amplitude * 100

        self.results["cursor_freq_error"] = freq_error
        self.results["cursor_amplitude_error"] = amplitude_error

        print_result("Auto frequency", f"{auto_freq / 1e3:.3f} kHz")
        print_result("Cursor frequency", f"{self.results['freq_from_cursors'] / 1e3:.3f} kHz")
        print_result("Frequency error", f"{freq_error:.2f}%")

        print_result("Auto amplitude", f"{auto_amplitude:.4f} V")
        print_result("Cursor amplitude", f"{self.results['delta_v']:.4f} V")
        print_result("Amplitude error", f"{amplitude_error:.2f}%")

        # ========== MEASUREMENT INTERPRETATION ==========
        print_subheader("Cursor Measurement Interpretation")

        print_info("\n[Time Cursors]")
        print_info(f"  Delta-T: {self.results['delta_t'] * 1e6:.3f}µs between rising edges")
        print_info(f"  Frequency: {self.results['freq_from_cursors'] / 1e3:.3f}kHz (from 1/ΔT)")
        print_info(f"  Accuracy: {100 - freq_error:.2f}% vs automatic measurement")

        print_info("\n[Voltage Cursors]")
        print_info(f"  Delta-V: {self.results['delta_v']:.4f}V peak-to-peak")
        print_info(f"  Peak: {voltage_a:.4f}V, Trough: {voltage_b:.4f}V")
        print_info(f"  Accuracy: {100 - amplitude_error:.2f}% vs automatic measurement")

        print_info("\n[Use Cases]")
        print_info("  - Precise timing measurements between specific events")
        print_info("  - Voltage level measurements at specific points")
        print_info("  - Manual verification of automatic measurements")
        print_info("  - Relative measurements from reference point")

    def validate_results(self, suite: ValidationSuite) -> None:
        """Validate cursor measurement results."""
        # Time cursor validation
        suite.check_range("Delta-T", self.results["delta_t"], 240e-6, 260e-6)
        suite.check_range("Frequency from cursors", self.results["freq_from_cursors"], 3800, 4200)

        # Voltage cursor validation
        suite.check_range("Delta-V", self.results["delta_v"], 4.9, 5.1)

        # Multiple cursor pairs
        if "cursor_pair1_pulse_width" in self.results:
            suite.check_range("Pulse width", self.results["cursor_pair1_pulse_width"], 70e-6, 80e-6)

        if "cursor_pair2_period" in self.results:
            suite.check_range("Period", self.results["cursor_pair2_period"], 240e-6, 260e-6)

        if "cursor_pair3_duty" in self.results:
            suite.check_range("Duty cycle", self.results["cursor_pair3_duty"], 0.25, 0.35)

        # Accuracy validation
        suite.check_range("Frequency error", self.results["cursor_freq_error"], 0, 5)
        suite.check_range("Amplitude error", self.results["cursor_amplitude_error"], 0, 5)


if __name__ == "__main__":
    sys.exit(run_demo_main(CursorsDemo))
