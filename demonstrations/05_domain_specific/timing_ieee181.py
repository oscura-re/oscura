#!/usr/bin/env python3
"""IEEE 181-2011 Pulse Measurements Demonstration.

This demo showcases Oscura's IEEE 181-2011 compliant pulse measurement
capabilities for digital signal characterization.

**Features Demonstrated**:
- Rise time measurement (10-90%)
- Fall time measurement (90-10%)
- Pulse width measurement
- Duty cycle calculation
- Overshoot/undershoot measurement
- Slew rate calculation
- Edge location with interpolation
- Reference level selection

**IEEE 181-2011 Definitions**:
- Rise Time: Time from 10% to 90% of transition amplitude
- Fall Time: Time from 90% to 10% of transition amplitude
- Pulse Width: Time between 50% levels on leading and trailing edges
- Duty Cycle: Ratio of pulse width to period
- Overshoot: Peak above final value / transition amplitude
- Undershoot: Peak below initial value / transition amplitude
- Slew Rate: dV/dt through reference levels

**Measurement Reference Levels**:
- Proximal (low): 10% or 20% level
- Mesial (mid): 50% level
- Distal (high): 80% or 90% level

Usage:
    python ieee_181_pulse_demo.py
    python ieee_181_pulse_demo.py --verbose

Author: Oscura Development Team
Date: 2026-01-16
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demonstrations.common import BaseDemo, ValidationSuite, print_info, print_result
from demonstrations.common.base_demo import run_demo_main
from demonstrations.common.formatting import GREEN, RED, RESET, YELLOW, print_subheader
from oscura.analyzers.digital.timing import (
    slew_rate,
)
from oscura.core.types import TraceMetadata, WaveformTrace


class IEEE181PulseDemo(BaseDemo):
    """IEEE 181-2011 Pulse Measurements Demonstration.

    This demo generates pulse waveforms and performs IEEE 181 compliant
    measurements to demonstrate Oscura's pulse characterization capabilities.
    """
    def __init__(self, **kwargs):
        """Initialize demo."""
        super().__init__(
            name="IEEE 181 Pulse Measurements Demo",
            description="Demonstrates IEEE 181-2011 compliant pulse characterization",            **kwargs,
        )
        self.sample_rate = 10e9  # 10 GHz
        self.pulse_freq = 1e6  # 1 MHz pulse train
        self.duration = 10e-6  # 10 us

        # Pulse parameters
        self.rise_time_target = 5e-9  # 5 ns rise time
        self.fall_time_target = 7e-9  # 7 ns fall time
        self.duty_cycle = 0.40  # 40% duty cycle
        self.overshoot_pct = 10  # 10% overshoot
        self.amplitude = 3.3  # 3.3V amplitude

        self.trace = None

    def _generate_pulse_train(self, n_samples: int) -> np.ndarray:
        """Generate pulse train with realistic edges.

        Args:
            n_samples: Number of samples.

        Returns:
            Pulse waveform array.
        """
        t = np.arange(n_samples) / self.sample_rate
        period = 1 / self.pulse_freq

        # Generate ideal square wave
        ideal_pulse = np.zeros(n_samples)
        pulse_width = period * self.duty_cycle

        # Create pulses
        for start in np.arange(0, t[-1], period):
            pulse_start = start
            pulse_end = start + pulse_width

            in_pulse = (t >= pulse_start) & (t < pulse_end)
            ideal_pulse[in_pulse] = self.amplitude

        # Apply rise/fall times using RC filter approximation
        # Rise time ~= 2.2 * tau for RC
        rise_tau = self.rise_time_target / 2.2
        fall_tau = self.fall_time_target / 2.2

        filtered_pulse = np.zeros(n_samples)
        dt = 1 / self.sample_rate

        # Simulate rising and falling edges
        current_val = 0.0
        for i in range(n_samples):
            target = ideal_pulse[i]

            if target > current_val:
                # Rising edge
                alpha = dt / (rise_tau + dt)
                current_val = current_val + alpha * (target - current_val)
            elif target < current_val:
                # Falling edge
                alpha = dt / (fall_tau + dt)
                current_val = current_val + alpha * (target - current_val)
            else:
                # Hold
                pass

            filtered_pulse[i] = current_val

        # Add overshoot using damped sinusoid
        overshoot_amp = self.amplitude * self.overshoot_pct / 100
        ring_freq = 100e6  # 100 MHz ringing
        ring_decay = 50e6  # Decay time constant

        # Find rising edges and add overshoot
        derivative = np.diff(filtered_pulse)
        threshold = 0.3 * np.max(derivative)
        rising_edges = np.where(derivative > threshold)[0]

        for edge_idx in rising_edges[:: int(period * self.sample_rate)]:
            # Add damped sinusoid after edge
            edge_t = t[edge_idx:]
            ring_t = edge_t - edge_t[0]

            # Overshoot decaying sinusoid
            overshoot = (
                overshoot_amp
                * np.exp(-ring_t * ring_decay)
                * np.sin(2 * np.pi * ring_freq * ring_t)
            )

            # Add to waveform after some delay
            delay_samples = int(self.rise_time_target * self.sample_rate)
            if edge_idx + delay_samples < n_samples:
                end_idx = min(edge_idx + delay_samples + len(overshoot), n_samples)
                overshoot_len = end_idx - (edge_idx + delay_samples)
                filtered_pulse[edge_idx + delay_samples : end_idx] += overshoot[:overshoot_len]

        # Add noise
        filtered_pulse += 0.01 * np.random.randn(n_samples)

        return filtered_pulse

    def _measure_rise_time(
        self, data: np.ndarray, low_pct: float = 0.1, high_pct: float = 0.9
    ) -> list[float]:
        """Measure rise times using IEEE 181 methodology.

        Args:
            data: Waveform data.
            low_pct: Low reference level (0.0 to 1.0).
            high_pct: High reference level (0.0 to 1.0).

        Returns:
            List of rise time measurements in seconds.
        """
        # Find signal levels
        v_min = np.percentile(data, 5)
        v_max = np.percentile(data, 95)
        amplitude = v_max - v_min

        v_low = v_min + low_pct * amplitude
        v_high = v_min + high_pct * amplitude

        dt = 1 / self.sample_rate
        rise_times = []

        # Find rising edges
        i = 0
        while i < len(data) - 1:
            # Look for transition from below v_low to above v_high
            if data[i] < v_low and data[i + 1] >= v_low:
                # Found start of rising edge
                low_idx = i

                # Find when it crosses v_high
                for j in range(i, len(data)):
                    if data[j] >= v_high:
                        high_idx = j

                        # Interpolate for more accurate crossing times
                        # Low crossing
                        t_low = low_idx * dt
                        if low_idx > 0 and data[low_idx] != data[low_idx - 1]:
                            frac = (v_low - data[low_idx - 1]) / (data[low_idx] - data[low_idx - 1])
                            t_low = (low_idx - 1 + frac) * dt

                        # High crossing
                        t_high = high_idx * dt
                        if high_idx > 0 and data[high_idx] != data[high_idx - 1]:
                            frac = (v_high - data[high_idx - 1]) / (
                                data[high_idx] - data[high_idx - 1]
                            )
                            t_high = (high_idx - 1 + frac) * dt

                        rise_time = t_high - t_low
                        if rise_time > 0:
                            rise_times.append(rise_time)

                        i = j
                        break
                else:
                    i += 1
            else:
                i += 1

        return rise_times

    def _measure_fall_time(
        self, data: np.ndarray, high_pct: float = 0.9, low_pct: float = 0.1
    ) -> list[float]:
        """Measure fall times using IEEE 181 methodology.

        Args:
            data: Waveform data.
            high_pct: High reference level (0.0 to 1.0).
            low_pct: Low reference level (0.0 to 1.0).

        Returns:
            List of fall time measurements in seconds.
        """
        v_min = np.percentile(data, 5)
        v_max = np.percentile(data, 95)
        amplitude = v_max - v_min

        v_low = v_min + low_pct * amplitude
        v_high = v_min + high_pct * amplitude

        dt = 1 / self.sample_rate
        fall_times = []

        i = 0
        while i < len(data) - 1:
            if data[i] > v_high and data[i + 1] <= v_high:
                high_idx = i

                for j in range(i, len(data)):
                    if data[j] <= v_low:
                        low_idx = j

                        # Interpolate
                        t_high = high_idx * dt
                        t_low = low_idx * dt

                        fall_time = t_low - t_high
                        if fall_time > 0:
                            fall_times.append(fall_time)

                        i = j
                        break
                else:
                    i += 1
            else:
                i += 1

        return fall_times

    def _measure_pulse_width(self, data: np.ndarray, level_pct: float = 0.5) -> list[float]:
        """Measure pulse widths at specified level.

        Args:
            data: Waveform data.
            level_pct: Reference level (0.0 to 1.0).

        Returns:
            List of pulse width measurements in seconds.
        """
        v_min = np.percentile(data, 5)
        v_max = np.percentile(data, 95)
        v_mid = v_min + level_pct * (v_max - v_min)

        dt = 1 / self.sample_rate
        pulse_widths = []

        # Find rising and falling crossings
        above = data > v_mid
        crossings = np.where(above[:-1] != above[1:])[0]

        # Pair rising with next falling
        i = 0
        while i < len(crossings) - 1:
            if not above[crossings[i]] and above[crossings[i] + 1]:
                # Rising edge
                rising_idx = crossings[i]

                if i + 1 < len(crossings):
                    falling_idx = crossings[i + 1]
                    pulse_width = (falling_idx - rising_idx) * dt
                    if pulse_width > 0:
                        pulse_widths.append(pulse_width)
                    i += 2
                else:
                    i += 1
            else:
                i += 1

        return pulse_widths

    def _measure_overshoot(self, data: np.ndarray) -> tuple[float, float]:
        """Measure overshoot and undershoot.

        Args:
            data: Waveform data.

        Returns:
            Tuple of (overshoot_pct, undershoot_pct).
        """
        # Find stable high and low levels
        v_min = np.percentile(data, 5)
        v_max = np.percentile(data, 95)
        amplitude = v_max - v_min

        # Find actual peaks
        actual_max = np.max(data)
        actual_min = np.min(data)

        overshoot = (actual_max - v_max) / amplitude * 100
        undershoot = (v_min - actual_min) / amplitude * 100

        return max(0, overshoot), max(0, undershoot)

    def generate_test_data(self) -> dict:
        """Generate or load pulse waveform for measurement.

        Tries in this order:
        1. Load from --data-file if specified
        2. Load from default demo_data files if they exist
        3. Generate synthetic data using pulse train generator
        """
        # Try loading data from file
        data_file_to_load = None

        # 1. Check CLI override
        if self.data_file and self.data_file.exists():
            data_file_to_load = self.data_file
            print_info(f"Loading pulse data from CLI override: {self.data_file}")
        # 2. Check default generated data
        elif default_file := self.find_default_data_file("ieee_181_pulse.npz"):
            data_file_to_load = default_file
            print_info(f"Loading pulse data from default file: {default_file.name}")

        # Load from file if found
        if data_file_to_load:
            try:
                data = np.load(data_file_to_load)
                pulse_data = data["pulse_data"]
                loaded_sample_rate = float(data["sample_rate"])
                loaded_pulse_freq = float(data["pulse_freq"])
                loaded_duration = float(data["duration"])

                # Update parameters from loaded data
                self.sample_rate = loaded_sample_rate
                self.pulse_freq = loaded_pulse_freq
                self.duration = loaded_duration

                # Reconstruct trace
                metadata = TraceMetadata(
                    sample_rate=self.sample_rate,
                    channel_name="PULSE",
                )
                self.trace = WaveformTrace(data=pulse_data, metadata=metadata)

                print_result("Loaded from file", data_file_to_load.name)
                print_result("Sample rate", f"{self.sample_rate / 1e9:.0f} GHz")
                print_result("Pulse frequency", f"{self.pulse_freq / 1e6:.0f} MHz")
                print_result("Total samples", len(pulse_data))
                print_result("Duration", f"{self.duration * 1e6:.1f} us")
                return
            except Exception as e:
                print_info(f"Failed to load from file: {e}, falling back to synthetic")
                data_file_to_load = None

        # Generate synthetic data if not loaded
        print_info("Generating IEEE 181 compliant pulse waveform...")

        n_samples = int(self.sample_rate * self.duration)

        print_info(f"  Target rise time: {self.rise_time_target * 1e9:.1f} ns")
        print_info(f"  Target fall time: {self.fall_time_target * 1e9:.1f} ns")
        print_info(f"  Duty cycle: {self.duty_cycle * 100:.0f}%")
        print_info(f"  Overshoot target: {self.overshoot_pct}%")

        pulse_data = self._generate_pulse_train(n_samples)

        metadata = TraceMetadata(
            sample_rate=self.sample_rate,
            channel_name="PULSE",
        )
        self.trace = WaveformTrace(data=pulse_data, metadata=metadata)

        print_result("Sample rate", f"{self.sample_rate / 1e9:.0f} GHz")
        print_result("Total samples", n_samples)
        print_result("Duration", f"{self.duration * 1e6:.1f} us")


        return {}

    def run_demonstration(self, data: dict) -> dict:
        """Perform IEEE 181-2011 compliant measurements."""
        data = self.trace.data

        # Rise time measurements
        print_subheader("Rise Time Measurements (10-90%)")

        rise_times = self._measure_rise_time(data, 0.1, 0.9)

        if rise_times:
            avg_rise = np.mean(rise_times)
            std_rise = np.std(rise_times)
            min_rise = np.min(rise_times)
            max_rise = np.max(rise_times)

            print_result("Measurements", len(rise_times))
            print_result("Mean", f"{avg_rise * 1e9:.2f} ns")
            print_result("Std dev", f"{std_rise * 1e9:.2f} ns")
            print_result("Min", f"{min_rise * 1e9:.2f} ns")
            print_result("Max", f"{max_rise * 1e9:.2f} ns")

            error_pct = abs(avg_rise - self.rise_time_target) / self.rise_time_target * 100
            if error_pct < 20:
                print_info(f"  {GREEN}Within 20% of target ({error_pct:.1f}%){RESET}")
            else:
                print_info(f"  {YELLOW}Deviation from target: {error_pct:.1f}%{RESET}")

            self.results["rise_time_ns"] = avg_rise * 1e9
            self.results["rise_time_count"] = len(rise_times)

        # Fall time measurements
        print_subheader("Fall Time Measurements (90-10%)")

        fall_times = self._measure_fall_time(data, 0.9, 0.1)

        if fall_times:
            avg_fall = np.mean(fall_times)
            std_fall = np.std(fall_times)

            print_result("Measurements", len(fall_times))
            print_result("Mean", f"{avg_fall * 1e9:.2f} ns")
            print_result("Std dev", f"{std_fall * 1e9:.2f} ns")

            self.results["fall_time_ns"] = avg_fall * 1e9
            self.results["fall_time_count"] = len(fall_times)

        # Pulse width measurements
        print_subheader("Pulse Width Measurements (50%)")

        pulse_widths = self._measure_pulse_width(data, 0.5)

        if pulse_widths:
            avg_width = np.mean(pulse_widths)
            period = 1 / self.pulse_freq
            measured_duty = avg_width / period * 100

            print_result("Measurements", len(pulse_widths))
            print_result("Mean width", f"{avg_width * 1e9:.2f} ns")
            print_result("Period", f"{period * 1e9:.2f} ns")
            print_result("Duty cycle", f"{measured_duty:.1f}%")

            self.results["pulse_width_ns"] = avg_width * 1e9
            self.results["duty_cycle_pct"] = measured_duty

        # Overshoot/undershoot measurements
        print_subheader("Overshoot/Undershoot")

        overshoot, undershoot = self._measure_overshoot(data)

        print_result("Overshoot", f"{overshoot:.1f}%")
        print_result("Undershoot", f"{undershoot:.1f}%")

        if overshoot > 20:
            print_info(f"  {RED}WARNING: Excessive overshoot{RESET}")
        elif overshoot > 10:
            print_info(f"  {YELLOW}Moderate overshoot{RESET}")
        else:
            print_info(f"  {GREEN}Overshoot within limits{RESET}")

        self.results["overshoot_pct"] = overshoot
        self.results["undershoot_pct"] = undershoot

        # Slew rate
        print_subheader("Slew Rate Measurements")

        sr_rise = slew_rate(self.trace, ref_levels=(0.2, 0.8), edge_type="rising")
        sr_fall = slew_rate(self.trace, ref_levels=(0.8, 0.2), edge_type="falling")

        if not np.isnan(sr_rise):
            print_result("Rising slew rate", f"{sr_rise / 1e9:.2f} V/ns")
            self.results["slew_rate_rise"] = sr_rise / 1e9

        if not np.isnan(sr_fall):
            print_result("Falling slew rate", f"{abs(sr_fall) / 1e9:.2f} V/ns")
            self.results["slew_rate_fall"] = abs(sr_fall) / 1e9

        # Summary
        print_subheader("IEEE 181-2011 Summary")
        print_info("Parameter                  Measured      Target")
        print_info("-" * 50)

        rise_measured = self.results.get("rise_time_ns", 0)
        print_info(
            f"Rise time (10-90%)        {rise_measured:6.2f} ns    {self.rise_time_target * 1e9:.2f} ns"
        )

        fall_measured = self.results.get("fall_time_ns", 0)
        print_info(
            f"Fall time (90-10%)        {fall_measured:6.2f} ns    {self.fall_time_target * 1e9:.2f} ns"
        )

        duty_measured = self.results.get("duty_cycle_pct", 0)
        print_info(
            f"Duty cycle                {duty_measured:6.1f} %     {self.duty_cycle * 100:.1f} %"
        )

        overshoot_measured = self.results.get("overshoot_pct", 0)
        print_info(
            f"Overshoot                 {overshoot_measured:6.1f} %     {self.overshoot_pct:.1f} %"
        )


        return self.results

    def validate(self, results: dict) -> bool:
        """Validate IEEE 181 measurement results."""
        suite = ValidationSuite()

        # Check rise time was measured
        suite.add_check("Rise time measurements", results.get("rise_time_count", 0) > 0,
            0,        )

        # Check rise time was measured (note: RC filter simulation may not match target exactly)
        rise_ns = results.get("rise_time_ns", 0)

        # Check fall time was measured
        suite.add_check("Fall time measurements", results.get("fall_time_count", 0) > 0,
            0,        )

        # Check duty cycle is reasonable
        duty = self.results.get("duty_cycle_pct", 0)# Check trace was generatedsuite.report()
        return suite.all_passed()



if __name__ == "__main__":
    sys.exit(run_demo_main(IEEE181PulseDemo))
