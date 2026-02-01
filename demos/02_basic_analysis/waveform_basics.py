#!/usr/bin/env python3
"""Waveform Basics: Fundamental waveform parameter measurements.

This demo demonstrates core waveform measurements including:
- Amplitude measurements (peak-to-peak, RMS, average)
- Frequency and period measurement
- Rise/fall time analysis
- Duty cycle measurement
- Overshoot/undershoot detection

IEEE Standards: IEEE 181-2011 (Pulse measurement terminology)
Related demos:
- 02_digital_basics.py - Digital signal analysis
- 04_measurements.py - Comprehensive measurement suite
- ../00_getting_started/01_hello_world.py - Basic introduction

Usage:
    python demos/02_basic_analysis/01_waveform_basics.py
    python demos/02_basic_analysis/01_waveform_basics.py --verbose
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import ClassVar

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


import oscura as osc
from demos.common import BaseDemo, ValidationSuite, print_info, print_result
from demos.common.base_demo import run_demo_main
from demos.common.data_generation import (
    generate_pulse_train,
    generate_sine_wave,
    generate_square_wave,
)
from demos.common.formatting import print_subheader


class WaveformBasicsDemo(BaseDemo):
    """Fundamental waveform measurement demonstrations."""

    name = "Waveform Measurements Basics"
    description = "Core waveform measurements: amplitude, frequency, timing, duty cycle"
    category = "basic_analysis"

    capabilities: ClassVar[list[str]] = [
        "oscura.amplitude",
        "oscura.frequency",
        "oscura.period",
        "oscura.rms",
        "oscura.mean",
        "oscura.rise_time",
        "oscura.fall_time",
        "oscura.duty_cycle",
        "oscura.overshoot",
        "oscura.undershoot",
    ]

    ieee_standards: ClassVar[list[str]] = ["IEEE 181-2011"]

    related_demos: ClassVar[list[str]] = [
        "../00_getting_started/01_hello_world.py",
        "02_digital_basics.py",
        "04_measurements.py",
    ]

    def __init__(self, **kwargs):
        """Initialize demo."""
        super().__init__(**kwargs)
        self.pulse_train = None
        self.sine_wave = None
        self.square_wave = None

    def generate_data(self) -> None:
        """Generate test waveforms for measurement demonstrations."""
        print_info("Generating test signals...")

        # 1. Pulse train with realistic rise/fall times (1 kHz, 50% duty cycle)
        self.pulse_train = generate_pulse_train(
            pulse_width=500e-6,  # 500 µs
            period=1000e-6,  # 1 ms (1 kHz)
            amplitude=5.0,  # 5V
            duration=0.005,  # 5 ms (5 periods)
            sample_rate=1e6,  # 1 MHz sampling
            rise_time=10e-9,  # 10 ns rise time
            fall_time=10e-9,  # 10 ns fall time
        )

        # 2. Sine wave for RMS measurements (1 kHz, 3V amplitude)
        self.sine_wave = generate_sine_wave(
            frequency=1000.0,  # 1 kHz
            amplitude=3.0,  # 3V peak
            duration=0.005,  # 5 ms
            sample_rate=1e6,  # 1 MHz sampling
        )

        # 3. Square wave for overshoot demonstration (2 kHz, 50% duty cycle)
        self.square_wave = generate_square_wave(
            frequency=2000.0,  # 2 kHz
            amplitude=3.0,  # 3V
            duration=0.005,  # 5 ms
            sample_rate=1e6,  # 1 MHz sampling
            duty_cycle=0.5,  # 50% duty cycle
        )

        print_result("Pulse train generated", "1 kHz, 5V, 50% duty")
        print_result("Sine wave generated", "1 kHz, 3V peak")
        print_result("Square wave generated", "2 kHz, 3V, 50% duty")

    def run_analysis(self) -> None:
        """Execute waveform measurements."""
        # ========== PART 1: PULSE TRAIN MEASUREMENTS ==========
        print_subheader("Part 1: Pulse Train Measurements")
        print_info("Pulse train: 1 kHz, 50% duty cycle, 5V amplitude, 10ns rise/fall")

        # Amplitude (peak-to-peak)
        vpp = osc.amplitude(self.pulse_train)
        self.results["pulse_amplitude"] = vpp
        print_result("Amplitude (Vpp)", f"{vpp:.4f} V")

        # Period and frequency
        t_period = osc.period(self.pulse_train)
        self.results["pulse_period"] = t_period
        print_result("Period", f"{t_period * 1e3:.6f} ms")

        freq = osc.frequency(self.pulse_train)
        self.results["pulse_frequency"] = freq
        print_result("Frequency", f"{freq:.2f} Hz")

        # Rise time (10-90% voltage)
        t_rise = osc.rise_time(self.pulse_train)
        self.results["pulse_rise_time"] = t_rise
        print_result("Rise time (10%-90%)", f"{t_rise * 1e9:.2f} ns")

        # Fall time (90-10% voltage)
        t_fall = osc.fall_time(self.pulse_train)
        self.results["pulse_fall_time"] = t_fall
        print_result("Fall time (90%-10%)", f"{t_fall * 1e9:.2f} ns")

        # Duty cycle
        duty = osc.duty_cycle(self.pulse_train)
        self.results["pulse_duty_cycle"] = duty
        print_result("Duty cycle", f"{duty * 100:.1f}%")

        # Mean (DC offset)
        mean_val = osc.mean(self.pulse_train)
        self.results["pulse_mean"] = mean_val
        print_result("DC offset (mean)", f"{mean_val:.4f} V")

        # ========== PART 2: SINE WAVE MEASUREMENTS ==========
        print_subheader("Part 2: Sine Wave Measurements (RMS and Amplitude)")
        print_info("Sine wave: 1 kHz, 3V peak amplitude")

        # Amplitude
        sine_vpp = osc.amplitude(self.sine_wave)
        self.results["sine_amplitude"] = sine_vpp
        print_result("Amplitude (Vpp)", f"{sine_vpp:.4f} V")

        # RMS voltage
        vrms = osc.rms(self.sine_wave)
        self.results["sine_rms"] = vrms
        print_result("RMS voltage", f"{vrms:.4f} V")
        print_info(f"Expected RMS for 3V peak sine: 3/√2 ≈ 2.121V (measured: {vrms:.4f}V)")

        # Mean (should be near 0 for AC sine wave)
        sine_mean = osc.mean(self.sine_wave)
        self.results["sine_mean"] = sine_mean
        print_result("Mean (DC offset)", f"{sine_mean:.6e} V")

        # Frequency
        sine_freq = osc.frequency(self.sine_wave)
        self.results["sine_frequency"] = sine_freq
        print_result("Frequency", f"{sine_freq:.2f} Hz")

        # ========== PART 3: SQUARE WAVE WITH OVERSHOOT ==========
        print_subheader("Part 3: Square Wave - Overshoot and Undershoot")
        print_info("Square wave: 2 kHz, 3V amplitude, 50% duty cycle")

        # Overshoot (positive transient)
        over = osc.overshoot(self.square_wave)
        self.results["square_overshoot"] = over
        print_result("Overshoot", f"{over:.4f} V")

        # Undershoot (negative transient)
        under = osc.undershoot(self.square_wave)
        self.results["square_undershoot"] = under
        print_result("Undershoot", f"{under:.4f} V")

        # Amplitude
        square_vpp = osc.amplitude(self.square_wave)
        self.results["square_amplitude"] = square_vpp
        print_result("Amplitude (Vpp)", f"{square_vpp:.4f} V")

        # Frequency
        square_freq = osc.frequency(self.square_wave)
        self.results["square_frequency"] = square_freq
        print_result("Frequency", f"{square_freq:.2f} Hz")

        # Duty cycle
        square_duty = osc.duty_cycle(self.square_wave)
        self.results["square_duty_cycle"] = square_duty
        print_result("Duty cycle", f"{square_duty * 100:.1f}%")

        # ========== MEASUREMENT INTERPRETATION ==========
        print_subheader("Measurement Interpretation")

        print_info("\n[Pulse Train Analysis]")
        print_info(
            f"  Period = 1/Frequency → {1 / freq * 1e3:.6f}ms matches {t_period * 1e3:.6f}ms"
        )
        print_info(f"  Duty Cycle = Pulse Width / Period → {duty * 100:.1f}% (target: 50%)")
        print_info(f"  Mean Voltage = Duty Cycle x Amplitude → {mean_val:.4f}V (target: 2.5V)")

        print_info("\n[Sine Wave Analysis]")
        print_info(f"  Amplitude (Vpp) = 2 x Peak → {sine_vpp:.4f}V (target: 6V)")
        print_info(f"  RMS = Peak / √2 → {vrms:.4f}V (target: 2.121V)")
        print_info(f"  Mean ≈ 0 for AC signal → {sine_mean:.6e}V")

        print_info("\n[Square Wave Analysis]")
        print_info(f"  Overshoot: {over:.4f}V, Undershoot: {under:.4f}V")
        print_info(f"  Frequency: {square_freq:.2f}Hz (target: 2000Hz)")

    def validate_results(self, suite: ValidationSuite) -> None:
        """Validate measurement results."""
        # Pulse train validation
        suite.check_range("Pulse amplitude", self.results["pulse_amplitude"], 4.9, 5.1)
        suite.check_range("Pulse period", self.results["pulse_period"], 0.98e-3, 1.02e-3)
        suite.check_range("Pulse frequency", self.results["pulse_frequency"], 980, 1020)
        suite.check_range("Pulse duty cycle", self.results["pulse_duty_cycle"], 0.48, 0.52)
        suite.check_range("Pulse mean", self.results["pulse_mean"], 2.4, 2.6)

        # Rise/fall time validation (limited by sampling resolution)
        suite.check_range("Rise time", self.results["pulse_rise_time"], 500e-9, 1000e-9)
        suite.check_range("Fall time", self.results["pulse_fall_time"], 500e-9, 1000e-9)

        # Sine wave validation
        suite.check_range("Sine amplitude", self.results["sine_amplitude"], 5.9, 6.1)
        suite.check_range("Sine RMS", self.results["sine_rms"], 2.1, 2.15)
        suite.check_range("Sine mean", abs(self.results["sine_mean"]), 0.0, 1e-14)
        suite.check_range("Sine frequency", self.results["sine_frequency"], 980, 1020)

        # Square wave validation
        suite.check_range("Square amplitude", self.results["square_amplitude"], 5.9, 6.1)
        suite.check_range("Square frequency", self.results["square_frequency"], 1980, 2020)
        suite.check_range("Square duty cycle", self.results["square_duty_cycle"], 0.48, 0.52)

        # Overshoot/undershoot (typical range for square wave transitions)
        suite.check_range("Square overshoot", abs(self.results["square_overshoot"]), 0.5, 1.5)
        suite.check_range("Square undershoot", abs(self.results["square_undershoot"]), 0.5, 1.5)


if __name__ == "__main__":
    sys.exit(run_demo_main(WaveformBasicsDemo))
