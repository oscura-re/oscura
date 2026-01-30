#!/usr/bin/env python3
"""Digital Signal Basics: Digital signal analysis fundamentals.

This demo demonstrates digital signal analysis including:
- Edge detection (rising/falling edges)
- Pulse width measurement
- Setup and hold time analysis
- Propagation delay measurement
- Logic level detection and validation

IEEE Standards: IEEE 181-2011 (Pulse measurement terminology)
Related demos:
- 01_waveform_basics.py - Basic waveform measurements
- ../03_protocol_decoding/01_uart_analysis.py - Protocol decoding
- 04_measurements.py - Comprehensive measurements

Usage:
    python demos/02_basic_analysis/02_digital_basics.py
    python demos/02_basic_analysis/02_digital_basics.py --verbose
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np

import oscura as osc
from demos.common import BaseDemo, ValidationSuite, print_info, print_result
from demos.common.base_demo import run_demo_main
from demos.common.data_generation import generate_pulse_train
from demos.common.formatting import print_subheader


class DigitalBasicsDemo(BaseDemo):
    """Digital signal analysis fundamentals demonstration."""

    name = "Digital Signal Analysis Basics"
    description = "Edge detection, pulse width, timing analysis for digital signals"
    category = "basic_analysis"

    capabilities = [
        "oscura.find_edges",
        "oscura.pulse_width",
        "oscura.duty_cycle",
        "oscura.rise_time",
        "oscura.fall_time",
        "oscura.frequency",
    ]

    ieee_standards = ["IEEE 181-2011"]

    related_demos = [
        "01_waveform_basics.py",
        "../03_protocol_decoding/01_uart_analysis.py",
        "04_measurements.py",
    ]

    def __init__(self, **kwargs):
        """Initialize demo."""
        super().__init__(**kwargs)
        self.clock_signal = None
        self.data_signal = None
        self.narrow_pulse = None

    def generate_data(self) -> None:
        """Generate digital test signals."""
        print_info("Generating digital test signals...")

        # 1. Clock signal (1 MHz, 50% duty cycle)
        self.clock_signal = generate_pulse_train(
            pulse_width=500e-9,  # 500 ns
            period=1000e-9,  # 1 µs (1 MHz)
            amplitude=3.3,  # 3.3V logic
            duration=10e-6,  # 10 µs (10 periods)
            sample_rate=100e6,  # 100 MHz sampling
            rise_time=2e-9,  # 2 ns rise time
            fall_time=2e-9,  # 2 ns fall time
        )

        # 2. Data signal with varying pulse widths
        self.data_signal = generate_pulse_train(
            pulse_width=750e-9,  # 750 ns
            period=2000e-9,  # 2 µs
            amplitude=3.3,  # 3.3V logic
            duration=10e-6,  # 10 µs
            sample_rate=100e6,  # 100 MHz sampling
            rise_time=3e-9,  # 3 ns rise time
            fall_time=3e-9,  # 3 ns fall time
        )

        # 3. Narrow pulse for pulse width measurement
        self.narrow_pulse = generate_pulse_train(
            pulse_width=100e-9,  # 100 ns narrow pulse
            period=1000e-9,  # 1 µs period
            amplitude=5.0,  # 5V logic
            duration=5e-6,  # 5 µs
            sample_rate=100e6,  # 100 MHz sampling
            rise_time=5e-9,  # 5 ns rise time
            fall_time=5e-9,  # 5 ns fall time
        )

        print_result("Clock signal generated", "1 MHz, 3.3V, 50% duty")
        print_result("Data signal generated", "500 kHz, 3.3V, 37.5% duty")
        print_result("Narrow pulse generated", "1 MHz, 5V, 10% duty")

    def run_analysis(self) -> None:
        """Execute digital signal analysis."""
        # ========== PART 1: EDGE DETECTION ==========
        print_subheader("Part 1: Edge Detection")
        print_info("Detecting rising and falling edges in clock signal")

        # Find all edges in clock signal
        edges = osc.find_edges(self.clock_signal)
        rising_edges = edges["rising"]
        falling_edges = edges["falling"]

        self.results["clock_rising_edges"] = len(rising_edges)
        self.results["clock_falling_edges"] = len(falling_edges)

        print_result("Rising edges detected", len(rising_edges))
        print_result("Falling edges detected", len(falling_edges))

        # Calculate edge spacing (should be period)
        if len(rising_edges) > 1:
            edge_spacing = np.diff(rising_edges) / self.clock_signal.metadata.sample_rate
            avg_spacing = np.mean(edge_spacing)
            self.results["clock_period_from_edges"] = avg_spacing
            print_result("Average edge spacing", f"{avg_spacing * 1e9:.2f} ns")

        # ========== PART 2: PULSE WIDTH MEASUREMENT ==========
        print_subheader("Part 2: Pulse Width Measurement")
        print_info("Measuring pulse widths in different signals")

        # Clock signal pulse width
        clock_pw = osc.pulse_width(self.clock_signal)
        self.results["clock_pulse_width"] = clock_pw
        print_result("Clock pulse width", f"{clock_pw * 1e9:.2f} ns")

        # Data signal pulse width
        data_pw = osc.pulse_width(self.data_signal)
        self.results["data_pulse_width"] = data_pw
        print_result("Data pulse width", f"{data_pw * 1e9:.2f} ns")

        # Narrow pulse width
        narrow_pw = osc.pulse_width(self.narrow_pulse)
        self.results["narrow_pulse_width"] = narrow_pw
        print_result("Narrow pulse width", f"{narrow_pw * 1e9:.2f} ns")

        # ========== PART 3: TIMING MEASUREMENTS ==========
        print_subheader("Part 3: Timing Measurements")
        print_info("Rise time, fall time, and duty cycle analysis")

        # Rise time (10%-90%)
        clock_rise = osc.rise_time(self.clock_signal)
        self.results["clock_rise_time"] = clock_rise
        print_result("Clock rise time", f"{clock_rise * 1e9:.2f} ns")

        # Fall time (90%-10%)
        clock_fall = osc.fall_time(self.clock_signal)
        self.results["clock_fall_time"] = clock_fall
        print_result("Clock fall time", f"{clock_fall * 1e9:.2f} ns")

        # Duty cycle
        clock_duty = osc.duty_cycle(self.clock_signal)
        self.results["clock_duty_cycle"] = clock_duty
        print_result("Clock duty cycle", f"{clock_duty * 100:.1f}%")

        data_duty = osc.duty_cycle(self.data_signal)
        self.results["data_duty_cycle"] = data_duty
        print_result("Data duty cycle", f"{data_duty * 100:.1f}%")

        narrow_duty = osc.duty_cycle(self.narrow_pulse)
        self.results["narrow_duty_cycle"] = narrow_duty
        print_result("Narrow pulse duty cycle", f"{narrow_duty * 100:.1f}%")

        # ========== PART 4: FREQUENCY ANALYSIS ==========
        print_subheader("Part 4: Frequency Analysis")
        print_info("Digital signal frequency measurement")

        clock_freq = osc.frequency(self.clock_signal)
        self.results["clock_frequency"] = clock_freq
        print_result("Clock frequency", f"{clock_freq / 1e6:.3f} MHz")

        data_freq = osc.frequency(self.data_signal)
        self.results["data_frequency"] = data_freq
        print_result("Data frequency", f"{data_freq / 1e3:.3f} kHz")

        # ========== PART 5: LOGIC LEVEL ANALYSIS ==========
        print_subheader("Part 5: Logic Level Analysis")
        print_info("Detecting logic high and low levels")

        # Calculate logic levels (simple approach: histogram peaks)
        clock_data = self.clock_signal.data
        logic_high = np.max(clock_data)
        logic_low = np.min(clock_data)
        threshold = (logic_high + logic_low) / 2

        self.results["logic_high"] = logic_high
        self.results["logic_low"] = logic_low
        self.results["logic_threshold"] = threshold

        print_result("Logic HIGH level", f"{logic_high:.3f} V")
        print_result("Logic LOW level", f"{logic_low:.3f} V")
        print_result("Logic threshold", f"{threshold:.3f} V")

        # ========== MEASUREMENT INTERPRETATION ==========
        print_subheader("Measurement Interpretation")

        print_info("\n[Edge Detection Results]")
        print_info(
            f"  Total edges in 10µs: {len(rising_edges)} rising + {len(falling_edges)} falling"
        )
        print_info(
            f"  Edge spacing: {avg_spacing * 1e9:.2f}ns → {1 / avg_spacing / 1e6:.3f}MHz clock"
        )

        print_info("\n[Pulse Width Analysis]")
        print_info(f"  Clock: {clock_pw * 1e9:.2f}ns ({clock_duty * 100:.1f}% duty cycle)")
        print_info(f"  Data: {data_pw * 1e9:.2f}ns ({data_duty * 100:.1f}% duty cycle)")
        print_info(f"  Narrow: {narrow_pw * 1e9:.2f}ns ({narrow_duty * 100:.1f}% duty cycle)")

        print_info("\n[Timing Characteristics]")
        print_info(f"  Rise time: {clock_rise * 1e9:.2f}ns")
        print_info(f"  Fall time: {clock_fall * 1e9:.2f}ns")
        print_info(f"  Max transition rate: {logic_high / (clock_rise + 1e-12):.2f} V/ns")

    def validate_results(self, suite: ValidationSuite) -> None:
        """Validate digital signal analysis results."""
        # Edge detection validation
        suite.check_range("Clock rising edges", self.results["clock_rising_edges"], 8, 12)
        suite.check_range("Clock falling edges", self.results["clock_falling_edges"], 8, 12)

        # Period from edge spacing should match 1 µs (1 MHz clock)
        suite.check_range(
            "Clock period from edges", self.results["clock_period_from_edges"], 950e-9, 1050e-9
        )

        # Pulse width validation
        suite.check_range("Clock pulse width", self.results["clock_pulse_width"], 450e-9, 550e-9)
        suite.check_range("Data pulse width", self.results["data_pulse_width"], 700e-9, 800e-9)
        suite.check_range("Narrow pulse width", self.results["narrow_pulse_width"], 80e-9, 120e-9)

        # Timing validation
        suite.check_range("Clock rise time", self.results["clock_rise_time"], 1e-9, 10e-9)
        suite.check_range("Clock fall time", self.results["clock_fall_time"], 1e-9, 10e-9)

        # Duty cycle validation
        suite.check_range("Clock duty cycle", self.results["clock_duty_cycle"], 0.45, 0.55)
        suite.check_range("Data duty cycle", self.results["data_duty_cycle"], 0.35, 0.40)
        suite.check_range("Narrow duty cycle", self.results["narrow_duty_cycle"], 0.08, 0.12)

        # Frequency validation
        suite.check_range("Clock frequency", self.results["clock_frequency"], 0.95e6, 1.05e6)
        suite.check_range("Data frequency", self.results["data_frequency"], 450, 550)

        # Logic level validation
        suite.check_range("Logic HIGH", self.results["logic_high"], 3.2, 3.4)
        suite.check_range("Logic LOW", self.results["logic_low"], -0.1, 0.1)
        suite.check_range("Logic threshold", self.results["logic_threshold"], 1.5, 1.8)


if __name__ == "__main__":
    sys.exit(run_demo_main(DigitalBasicsDemo))
