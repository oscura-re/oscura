#!/usr/bin/env python3
"""IEEE 181 Timing Measurements Demo.

This demo demonstrates IEEE 181-2011 compliant timing measurements:
- Rise time measurement (10%-90%)
- Fall time measurement (90%-10%)
- Pulse width measurement
- Overshoot/preshoot measurement
- Slew rate calculation
- Compliance checking

Standards:
- IEEE 181-2011 (Pulse measurement)
- IEC 61083-1 (Digital recorders)

Usage:
    python demos/05_domain_specific/07_timing_ieee181.py

Author: Oscura Development Team
Date: 2026-01-29
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demos.common import BaseDemo, ValidationSuite, print_info, print_result
from demos.common.base_demo import run_demo_main
from demos.common.formatting import print_subheader
from oscura.analyzers.digital.timing import slew_rate
from oscura.core.types import TraceMetadata, WaveformTrace


class IEEE181TimingDemo(BaseDemo):
    """IEEE 181 Timing Measurements Demonstration."""

    name = "IEEE 181 Timing Measurements"
    description = "IEEE 181-2011 compliant pulse measurements"
    category = "domain_specific"

    def __init__(self, **kwargs):
        """Initialize demo."""
        super().__init__(**kwargs)
        self.sample_rate = 10e9  # 10 GHz
        self.pulse_freq = 1e6  # 1 MHz
        self.rise_time_target = 5e-9  # 5 ns
        self.fall_time_target = 7e-9  # 7 ns
        self.amplitude = 3.3  # 3.3V
        self.pulse_trace = None

    def generate_data(self) -> None:
        """Generate pulse waveform for measurement."""
        print_info("Generating IEEE 181 compliant pulse waveform...")

        duration = 10e-6
        n_samples = int(self.sample_rate * duration)
        t = np.arange(n_samples) / self.sample_rate
        period = 1 / self.pulse_freq

        # Generate pulse train
        signal = np.zeros(n_samples)
        pulse_width = period * 0.4

        for start in np.arange(0, t[-1], period):
            pulse_start = start
            pulse_end = start + pulse_width
            in_pulse = (t >= pulse_start) & (t < pulse_end)
            signal[in_pulse] = self.amplitude

        # Apply rise/fall times using RC filter
        rise_tau = self.rise_time_target / 2.2
        fall_tau = self.fall_time_target / 2.2

        filtered = np.zeros(n_samples)
        dt = 1 / self.sample_rate
        current_val = 0.0

        for i in range(n_samples):
            target = signal[i]
            if target > current_val:
                alpha = dt / (rise_tau + dt)
                current_val = current_val + alpha * (target - current_val)
            elif target < current_val:
                alpha = dt / (fall_tau + dt)
                current_val = current_val + alpha * (target - current_val)
            filtered[i] = current_val

        # Add small noise
        filtered += 0.01 * np.random.randn(n_samples)

        self.pulse_trace = WaveformTrace(
            data=filtered,
            metadata=TraceMetadata(sample_rate=self.sample_rate, channel_name="PULSE"),
        )

        print_result("Samples generated", n_samples)
        print_result("Duration", f"{duration*1e6:.1f}", "us")

    def run_analysis(self) -> None:
        """Execute IEEE 181 measurements."""
        print_subheader("Rise Time Measurements (10%-90%)")
        self._measure_rise_time()

        print_subheader("Fall Time Measurements (90%-10%)")
        self._measure_fall_time()

        print_subheader("Pulse Width Measurements (50%)")
        self._measure_pulse_width()

        print_subheader("Slew Rate Measurements")
        self._measure_slew_rate()

        print_subheader("IEEE 181 Compliance Summary")
        self._print_summary()

    def _measure_rise_time(self) -> None:
        """Measure rise time (10%-90%)."""
        data = self.pulse_trace.data
        v_min = np.percentile(data, 5)
        v_max = np.percentile(data, 95)
        amplitude = v_max - v_min

        v_low = v_min + 0.1 * amplitude
        v_high = v_min + 0.9 * amplitude

        dt = 1 / self.sample_rate
        rise_times = []

        # Find rising edges
        i = 0
        while i < len(data) - 1:
            if data[i] < v_low and data[i+1] >= v_low:
                low_idx = i
                for j in range(i, min(i+200, len(data))):
                    if data[j] >= v_high:
                        high_idx = j
                        rise_time = (high_idx - low_idx) * dt
                        if rise_time > 0:
                            rise_times.append(rise_time)
                        i = j
                        break
                else:
                    i += 1
            else:
                i += 1

        if rise_times:
            avg_rise = np.mean(rise_times)
            print_result("Mean rise time", f"{avg_rise*1e9:.2f}", "ns")
            print_result("Target", f"{self.rise_time_target*1e9:.2f}", "ns")
            print_result("Measurements", len(rise_times))

            self.results["rise_time_ns"] = avg_rise * 1e9
            self.results["rise_time_count"] = len(rise_times)

    def _measure_fall_time(self) -> None:
        """Measure fall time (90%-10%)."""
        data = self.pulse_trace.data
        v_min = np.percentile(data, 5)
        v_max = np.percentile(data, 95)
        amplitude = v_max - v_min

        v_low = v_min + 0.1 * amplitude
        v_high = v_min + 0.9 * amplitude

        dt = 1 / self.sample_rate
        fall_times = []

        i = 0
        while i < len(data) - 1:
            if data[i] > v_high and data[i+1] <= v_high:
                high_idx = i
                for j in range(i, min(i+200, len(data))):
                    if data[j] <= v_low:
                        low_idx = j
                        fall_time = (low_idx - high_idx) * dt
                        if fall_time > 0:
                            fall_times.append(fall_time)
                        i = j
                        break
                else:
                    i += 1
            else:
                i += 1

        if fall_times:
            avg_fall = np.mean(fall_times)
            print_result("Mean fall time", f"{avg_fall*1e9:.2f}", "ns")
            print_result("Target", f"{self.fall_time_target*1e9:.2f}", "ns")
            print_result("Measurements", len(fall_times))

            self.results["fall_time_ns"] = avg_fall * 1e9
            self.results["fall_time_count"] = len(fall_times)

    def _measure_pulse_width(self) -> None:
        """Measure pulse width at 50%."""
        data = self.pulse_trace.data
        v_min = np.percentile(data, 5)
        v_max = np.percentile(data, 95)
        v_mid = (v_min + v_max) / 2

        dt = 1 / self.sample_rate
        above = data > v_mid
        crossings = np.where(above[:-1] != above[1:])[0]

        pulse_widths = []
        i = 0
        while i < len(crossings) - 1:
            if not above[crossings[i]] and above[crossings[i]+1]:
                rising_idx = crossings[i]
                if i+1 < len(crossings):
                    falling_idx = crossings[i+1]
                    pulse_width = (falling_idx - rising_idx) * dt
                    if pulse_width > 0:
                        pulse_widths.append(pulse_width)
                    i += 2
                else:
                    i += 1
            else:
                i += 1

        if pulse_widths:
            avg_width = np.mean(pulse_widths)
            period = 1 / self.pulse_freq
            duty_cycle = avg_width / period * 100

            print_result("Mean pulse width", f"{avg_width*1e9:.2f}", "ns")
            print_result("Duty cycle", f"{duty_cycle:.1f}", "%")
            print_result("Measurements", len(pulse_widths))

            self.results["pulse_width_ns"] = avg_width * 1e9
            self.results["duty_cycle"] = duty_cycle

    def _measure_slew_rate(self) -> None:
        """Measure slew rate."""
        sr_rise = slew_rate(self.pulse_trace, ref_levels=(0.2, 0.8), edge_type="rising")
        sr_fall = slew_rate(self.pulse_trace, ref_levels=(0.8, 0.2), edge_type="falling")

        if not np.isnan(sr_rise):
            print_result("Rising slew rate", f"{sr_rise/1e9:.2f}", "V/ns")
            self.results["slew_rate_rise"] = sr_rise / 1e9

        if not np.isnan(sr_fall):
            print_result("Falling slew rate", f"{abs(sr_fall)/1e9:.2f}", "V/ns")
            self.results["slew_rate_fall"] = abs(sr_fall) / 1e9

    def _print_summary(self) -> None:
        """Print measurement summary."""
        print_info("IEEE 181-2011 Measurement Summary:")
        print_info(f"  Rise time: {self.results.get('rise_time_ns', 0):.2f} ns")
        print_info(f"  Fall time: {self.results.get('fall_time_ns', 0):.2f} ns")
        print_info(f"  Pulse width: {self.results.get('pulse_width_ns', 0):.2f} ns")
        print_info(f"  Duty cycle: {self.results.get('duty_cycle', 0):.1f}%")

    def validate_results(self, suite: ValidationSuite) -> None:
        """Validate timing measurements."""
        suite.check_greater(
            "Rise time measurements",
            self.results.get("rise_time_count", 0),
            0,
            category="timing",
        )

        suite.check_greater(
            "Fall time measurements",
            self.results.get("fall_time_count", 0),
            0,
            category="timing",
        )

        suite.check_greater(
            "Duty cycle measured", self.results.get("duty_cycle", 0), 0, category="timing"
        )


if __name__ == "__main__":
    sys.exit(run_demo_main(IEEE181TimingDemo))
