#!/usr/bin/env python3
"""Comprehensive Measurements: Complete IEEE 181 measurement suite.

This demo demonstrates comprehensive waveform measurements following IEEE 181-2011:
- All pulse measurements (width, period, frequency)
- Amplitude measurements (peak, RMS, average, min, max)
- Timing measurements (rise/fall time, propagation delay)
- Overshoot and undershoot detection
- Statistical waveform characterization
- Measurement report generation

IEEE Standards: IEEE 181-2011 (Pulse measurement and analysis)
Related demos:
- 01_waveform_basics.py - Basic waveform measurements
- 02_digital_basics.py - Digital signal analysis
- 08_statistics.py - Statistical analysis

Usage:
    python demos/02_basic_analysis/04_measurements.py
    python demos/02_basic_analysis/04_measurements.py --verbose
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import ClassVar

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


import oscura as osc
from demos.common import BaseDemo, ValidationSuite, print_info, print_result, print_table
from demos.common.base_demo import run_demo_main
from demos.common.data_generation import generate_pulse_train
from demos.common.formatting import print_subheader


class ComprehensiveMeasurementsDemo(BaseDemo):
    """Comprehensive waveform measurement suite demonstration."""

    name = "Comprehensive Measurement Suite"
    description = "Complete IEEE 181-2011 compliant measurement suite"
    category = "basic_analysis"

    capabilities: ClassVar[list[str]] = [
        "oscura.amplitude",
        "oscura.frequency",
        "oscura.period",
        "oscura.pulse_width",
        "oscura.duty_cycle",
        "oscura.rise_time",
        "oscura.fall_time",
        "oscura.overshoot",
        "oscura.undershoot",
        "oscura.rms",
        "oscura.mean",
        "oscura.peak_to_peak",
        "oscura.min",
        "oscura.max",
    ]

    ieee_standards: ClassVar[list[str]] = ["IEEE 181-2011"]

    related_demos: ClassVar[list[str]] = [
        "01_waveform_basics.py",
        "02_digital_basics.py",
        "08_statistics.py",
    ]

    def __init__(self, **kwargs):
        """Initialize demo."""
        super().__init__(**kwargs)
        self.test_signal = None

    def generate_data(self) -> None:
        """Generate comprehensive test signal."""
        print_info("Generating comprehensive test signal...")

        # Generate pulse train with realistic characteristics
        self.test_signal = generate_pulse_train(
            pulse_width=250e-6,  # 250 µs
            period=1000e-6,  # 1 ms (1 kHz)
            amplitude=5.0,  # 5V
            duration=0.01,  # 10 ms (10 periods)
            sample_rate=10e6,  # 10 MHz sampling for high resolution
            rise_time=5e-9,  # 5 ns rise time
            fall_time=5e-9,  # 5 ns fall time
        )

        print_result("Test signal generated", "1 kHz pulse train, 5V, 25% duty cycle")
        print_result("Sample rate", f"{self.test_signal.metadata.sample_rate / 1e6:.1f} MHz")
        print_result(
            "Duration",
            f"{len(self.test_signal.data) / self.test_signal.metadata.sample_rate * 1e3:.1f} ms",
        )

    def run_analysis(self) -> None:
        """Execute comprehensive measurement suite."""
        measurements: ClassVar[dict[str, str]] = {}

        # ========== PART 1: AMPLITUDE MEASUREMENTS ==========
        print_subheader("Part 1: Amplitude Measurements (IEEE 181 Section 4)")
        print_info("Measuring voltage characteristics")

        measurements["peak_to_peak"] = osc.amplitude(self.test_signal)
        measurements["max"] = osc.max(self.test_signal)
        measurements["min"] = osc.min(self.test_signal)
        measurements["mean"] = osc.mean(self.test_signal)
        measurements["rms"] = osc.rms(self.test_signal)

        print_result("Peak-to-peak amplitude", f"{measurements['peak_to_peak']:.4f} V")
        print_result("Maximum value", f"{measurements['max']:.4f} V")
        print_result("Minimum value", f"{measurements['min']:.4f} V")
        print_result("Mean (DC offset)", f"{measurements['mean']:.4f} V")
        print_result("RMS voltage", f"{measurements['rms']:.4f} V")

        # ========== PART 2: TIMING MEASUREMENTS ==========
        print_subheader("Part 2: Timing Measurements (IEEE 181 Section 5)")
        print_info("Measuring temporal characteristics")

        measurements["period"] = osc.period(self.test_signal)
        measurements["frequency"] = osc.frequency(self.test_signal)
        measurements["pulse_width"] = osc.pulse_width(self.test_signal)
        measurements["duty_cycle"] = osc.duty_cycle(self.test_signal)

        print_result("Period", f"{measurements['period'] * 1e3:.6f} ms")
        print_result("Frequency", f"{measurements['frequency']:.3f} Hz")
        print_result("Pulse width", f"{measurements['pulse_width'] * 1e6:.3f} µs")
        print_result("Duty cycle", f"{measurements['duty_cycle'] * 100:.2f}%")

        # ========== PART 3: EDGE MEASUREMENTS ==========
        print_subheader("Part 3: Edge Measurements (IEEE 181 Section 6)")
        print_info("Measuring edge transition characteristics")

        measurements["rise_time"] = osc.rise_time(self.test_signal)
        measurements["fall_time"] = osc.fall_time(self.test_signal)

        print_result("Rise time (10%-90%)", f"{measurements['rise_time'] * 1e9:.3f} ns")
        print_result("Fall time (90%-10%)", f"{measurements['fall_time'] * 1e9:.3f} ns")

        # Calculate slew rate
        slew_rate_rise = (
            measurements["peak_to_peak"] / measurements["rise_time"]
            if measurements["rise_time"] > 0
            else 0
        )
        slew_rate_fall = (
            measurements["peak_to_peak"] / measurements["fall_time"]
            if measurements["fall_time"] > 0
            else 0
        )

        measurements["slew_rate_rise"] = slew_rate_rise
        measurements["slew_rate_fall"] = slew_rate_fall

        print_result("Slew rate (rising)", f"{slew_rate_rise / 1e9:.3f} V/ns")
        print_result("Slew rate (falling)", f"{slew_rate_fall / 1e9:.3f} V/ns")

        # ========== PART 4: ABERRATION MEASUREMENTS ==========
        print_subheader("Part 4: Aberration Measurements (IEEE 181 Section 7)")
        print_info("Detecting signal distortions")

        measurements["overshoot"] = osc.overshoot(self.test_signal)
        measurements["undershoot"] = osc.undershoot(self.test_signal)

        print_result("Overshoot", f"{measurements['overshoot']:.4f} V")
        print_result("Undershoot", f"{measurements['undershoot']:.4f} V")

        # Calculate overshoot percentage
        overshoot_pct = abs(measurements["overshoot"]) / measurements["peak_to_peak"] * 100
        undershoot_pct = abs(measurements["undershoot"]) / measurements["peak_to_peak"] * 100

        measurements["overshoot_percent"] = overshoot_pct
        measurements["undershoot_percent"] = undershoot_pct

        print_result("Overshoot percentage", f"{overshoot_pct:.2f}%")
        print_result("Undershoot percentage", f"{undershoot_pct:.2f}%")

        # ========== PART 5: DERIVED MEASUREMENTS ==========
        print_subheader("Part 5: Derived Measurements")
        print_info("Computing derived parameters")

        # Form factor (RMS / Average)
        form_factor = measurements["rms"] / measurements["mean"] if measurements["mean"] > 0 else 0
        measurements["form_factor"] = form_factor
        print_result("Form factor (RMS/Mean)", f"{form_factor:.4f}")

        # Crest factor (Peak / RMS)
        crest_factor = measurements["max"] / measurements["rms"] if measurements["rms"] > 0 else 0
        measurements["crest_factor"] = crest_factor
        print_result("Crest factor (Peak/RMS)", f"{crest_factor:.4f}")

        # Average power (for resistive load)
        avg_power = measurements["rms"] ** 2  # Assuming 1Ω load
        measurements["avg_power"] = avg_power
        print_result("Average power (1Ω load)", f"{avg_power:.4f} W")

        # ========== PART 6: MEASUREMENT SUMMARY ==========
        print_subheader("Part 6: Comprehensive Measurement Report")

        # Create measurement table
        headers: ClassVar[list[str]] = ["Measurement", "Value", "Unit", "IEEE 181 Section"]
        rows: ClassVar[list[str]] = [
            ["Peak-to-peak", f"{measurements['peak_to_peak']:.4f}", "V", "4.1"],
            ["Maximum", f"{measurements['max']:.4f}", "V", "4.2"],
            ["Minimum", f"{measurements['min']:.4f}", "V", "4.3"],
            ["Mean", f"{measurements['mean']:.4f}", "V", "4.4"],
            ["RMS", f"{measurements['rms']:.4f}", "V", "4.5"],
            ["Period", f"{measurements['period'] * 1e3:.6f}", "ms", "5.1"],
            ["Frequency", f"{measurements['frequency']:.3f}", "Hz", "5.2"],
            ["Pulse Width", f"{measurements['pulse_width'] * 1e6:.3f}", "µs", "5.3"],
            ["Duty Cycle", f"{measurements['duty_cycle'] * 100:.2f}", "%", "5.4"],
            ["Rise Time", f"{measurements['rise_time'] * 1e9:.3f}", "ns", "6.1"],
            ["Fall Time", f"{measurements['fall_time'] * 1e9:.3f}", "ns", "6.2"],
            ["Overshoot", f"{measurements['overshoot']:.4f}", "V", "7.1"],
            ["Undershoot", f"{measurements['undershoot']:.4f}", "V", "7.2"],
        ]

        print_table(headers, rows)

        # Store all measurements in results
        self.results.update(measurements)

        # ========== MEASUREMENT INTERPRETATION ==========
        print_subheader("Measurement Interpretation")

        print_info("\n[Signal Characterization]")
        print_info(f"  Signal type: Pulse train at {measurements['frequency']:.1f}Hz")
        print_info(f"  Voltage swing: {measurements['min']:.3f}V to {measurements['max']:.3f}V")
        print_info(
            f"  Duty cycle: {measurements['duty_cycle'] * 100:.1f}% → pulse is {measurements['pulse_width'] * 1e6:.1f}µs of {measurements['period'] * 1e3:.3f}ms"
        )

        print_info("\n[Edge Performance]")
        print_info(
            f"  Rise time: {measurements['rise_time'] * 1e9:.3f}ns → slew rate {slew_rate_rise / 1e9:.3f}V/ns"
        )
        print_info(
            f"  Fall time: {measurements['fall_time'] * 1e9:.3f}ns → slew rate {slew_rate_fall / 1e9:.3f}V/ns"
        )
        print_info(
            f"  Edge symmetry: {'Symmetric' if abs(measurements['rise_time'] - measurements['fall_time']) < 1e-9 else 'Asymmetric'}"
        )

        print_info("\n[Signal Quality]")
        print_info(
            f"  Overshoot: {overshoot_pct:.2f}% ({'Acceptable' if overshoot_pct < 10 else 'High'})"
        )
        print_info(
            f"  Undershoot: {undershoot_pct:.2f}% ({'Acceptable' if undershoot_pct < 10 else 'High'})"
        )
        print_info(f"  Form factor: {form_factor:.4f}")
        print_info(f"  Crest factor: {crest_factor:.4f}")

    def validate_results(self, suite: ValidationSuite) -> None:
        """Validate measurement results."""
        # Amplitude measurements
        suite.check_range("Peak-to-peak", self.results["peak_to_peak"], 4.9, 5.1)
        suite.check_range("Maximum", self.results["max"], 4.9, 5.1)
        suite.check_range("Minimum", self.results["min"], -0.1, 0.1)
        suite.check_range("Mean", self.results["mean"], 1.2, 1.3)  # 25% duty cycle x 5V
        suite.check_range("RMS", self.results["rms"], 2.0, 2.6)

        # Timing measurements
        suite.check_range("Period", self.results["period"], 0.99e-3, 1.01e-3)
        suite.check_range("Frequency", self.results["frequency"], 990, 1010)
        suite.check_range("Pulse width", self.results["pulse_width"], 240e-6, 260e-6)
        suite.check_range("Duty cycle", self.results["duty_cycle"], 0.24, 0.26)

        # Edge measurements (limited by sampling and rise/fall time)
        suite.check_range("Rise time", self.results["rise_time"], 1e-9, 100e-9)
        suite.check_range("Fall time", self.results["fall_time"], 1e-9, 100e-9)

        # Slew rate validation
        suite.check_range("Slew rate rise", self.results["slew_rate_rise"], 1e8, 1e12)
        suite.check_range("Slew rate fall", self.results["slew_rate_fall"], 1e8, 1e12)

        # Aberration measurements
        suite.check_range("Overshoot %", self.results["overshoot_percent"], 0, 20)
        suite.check_range("Undershoot %", self.results["undershoot_percent"], 0, 20)

        # Derived measurements
        suite.check_range("Form factor", self.results["form_factor"], 1.0, 3.0)
        suite.check_range("Crest factor", self.results["crest_factor"], 1.5, 3.0)


if __name__ == "__main__":
    sys.exit(run_demo_main(ComprehensiveMeasurementsDemo))
