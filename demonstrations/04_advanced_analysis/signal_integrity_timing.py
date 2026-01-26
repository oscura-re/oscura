#!/usr/bin/env python3
"""Setup/Hold Timing Analysis Demonstration.

This demo showcases Oscura's setup and hold time measurement capabilities
for synchronous digital interface timing verification per JEDEC JESD65B.

**Features Demonstrated**:
- Setup time measurement (data stable before clock edge)
- Hold time measurement (data stable after clock edge)
- Propagation delay measurement
- Timing margin calculation
- Violation detection
- Statistical timing analysis

**Timing Definitions (JEDEC JESD65B)**:
- Setup Time (tSU): Time data must be stable before clock edge
- Hold Time (tHD): Time data must remain stable after clock edge
- Propagation Delay (tPD): Time from input change to output change
- Clock-to-Data Delay (tCO): Time from clock edge to data output

**Use Cases**:
- FPGA interface timing verification
- Memory interface validation
- High-speed serial link analysis
- Digital logic characterization

Usage:
    python setup_hold_timing_demo.py
    python setup_hold_timing_demo.py --verbose

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

# Oscura imports
from oscura.analyzers.digital.timing import (
    hold_time,
    propagation_delay,
    setup_time,
    slew_rate,
)
from oscura.core.types import TraceMetadata, WaveformTrace


class SetupHoldTimingDemo(BaseDemo):
    """Setup/Hold Timing Analysis Demonstration.

    This demo generates clock and data signals with controlled timing
    relationships, then measures setup and hold times to demonstrate
    Oscura's timing analysis capabilities.
    """

    name = "Setup/Hold Timing Demo"
    description = "Demonstrates JEDEC JESD65B compliant timing analysis"
    category = "signal_integrity"

    def __init__(self, **kwargs):
        """Initialize demo."""
        super().__init__(**kwargs)
        self.sample_rate = 1e9  # 1 GHz (1 ns resolution)
        self.clock_freq = 100e6  # 100 MHz clock

        # Timing specifications (in nanoseconds)
        self.spec_setup_min = 2.0  # Minimum setup time
        self.spec_hold_min = 1.0  # Minimum hold time

        # Storage for signals and results
        self.clock_trace = None
        self.data_trace = None
        self.delayed_data_trace = None

    def _generate_clock(self, n_samples: int) -> np.ndarray:
        """Generate ideal clock signal.

        Args:
            n_samples: Number of samples.

        Returns:
            Clock waveform with 50% duty cycle.
        """
        t = np.arange(n_samples) / self.sample_rate
        period = 1 / self.clock_freq

        # Generate clean clock with sharp edges
        clock = np.zeros(n_samples)
        samples_per_period = int(self.sample_rate / self.clock_freq)
        samples_high = samples_per_period // 2

        for i in range(0, n_samples, samples_per_period):
            end_high = min(i + samples_high, n_samples)
            clock[i:end_high] = 1.0

        return clock

    def _generate_data_with_timing(
        self,
        clock: np.ndarray,
        setup_time_ns: float,
        hold_time_ns: float,
    ) -> np.ndarray:
        """Generate data signal with specific setup/hold timing.

        Args:
            clock: Clock signal.
            setup_time_ns: Setup time in nanoseconds.
            hold_time_ns: Hold time in nanoseconds.

        Returns:
            Data waveform synchronized to clock.
        """
        n_samples = len(clock)
        data = np.zeros(n_samples)

        samples_per_period = int(self.sample_rate / self.clock_freq)
        setup_samples = int(setup_time_ns * 1e-9 * self.sample_rate)
        hold_samples = int(hold_time_ns * 1e-9 * self.sample_rate)

        # Generate data that changes with proper setup/hold margins
        # Data changes should occur well before clock edge
        pattern = [0, 1, 1, 0, 1, 0, 0, 1]  # Pseudo-random pattern

        for i, bit_val in enumerate(
            pattern * ((n_samples // samples_per_period) // len(pattern) + 1)
        ):
            period_start = i * samples_per_period
            if period_start >= n_samples:
                break

            # Clock rising edge is at period_start
            # Data should be stable setup_time before and hold_time after

            # Data transition happens at: clock_edge - setup_time - margin
            # This gives setup_time before the clock edge
            data_change_point = period_start - setup_samples - 5  # 5 sample margin

            if data_change_point < 0:
                data_change_point = 0

            # Data stable from change point until after hold time
            stable_end = period_start + hold_samples + 5  # 5 sample margin
            stable_end = min(stable_end, n_samples)

            # Set data value
            data[data_change_point:stable_end] = bit_val * 3.3  # 3.3V logic

        return data

    def _add_realistic_edges(self, signal: np.ndarray, rise_time_ns: float) -> np.ndarray:
        """Add realistic rise/fall times to a digital signal.

        Args:
            signal: Ideal digital signal.
            rise_time_ns: 10-90% rise time in nanoseconds.

        Returns:
            Signal with finite edge rates.
        """
        from scipy.ndimage import gaussian_filter1d

        # Convert rise time to filter sigma
        # rise_time ~= 2.2 * sigma for Gaussian
        sigma_samples = (rise_time_ns * 1e-9 * self.sample_rate) / 2.2
        sigma_samples = max(1, sigma_samples)

        return gaussian_filter1d(signal, sigma=sigma_samples)

    def generate_data(self) -> None:
        """Generate clock and data signals for timing analysis.

        Loads from file if available (--data-file override or default NPZ),
        otherwise generates synthetic timing test signals.
        """
        # Try loading data from file
        file_to_load = None

        # 1. Check CLI override
        if self.data_file and self.data_file.exists():
            file_to_load = self.data_file
            print_info(f"Loading timing data from CLI override: {self.data_file}")
        # 2. Check default generated data
        elif default_file := self.find_default_data_file("setup_hold_timing.npz"):
            file_to_load = default_file
            print_info(f"Loading timing data from default file: {default_file.name}")

        # Load from file if found
        if file_to_load:
            try:
                data = np.load(file_to_load)
                clock_data = data["clock"]
                data_good = data["data_good"]
                data_marginal = data["data_marginal"]
                loaded_sample_rate = float(data["sample_rate"])
                self.sample_rate = loaded_sample_rate

                # Load timing parameters if available
                if "clock_freq" in data:
                    self.clock_freq = float(data["clock_freq"])
                if "spec_setup_min" in data:
                    self.spec_setup_min = float(data["spec_setup_min"])
                if "spec_hold_min" in data:
                    self.spec_hold_min = float(data["spec_hold_min"])

                # Create traces
                metadata_clock = TraceMetadata(
                    sample_rate=self.sample_rate,
                    channel_name="CLK",
                )
                self.clock_trace = WaveformTrace(data=clock_data, metadata=metadata_clock)

                metadata_data = TraceMetadata(
                    sample_rate=self.sample_rate,
                    channel_name="DATA",
                )
                self.data_trace = WaveformTrace(data=data_good, metadata=metadata_data)

                metadata_marginal = TraceMetadata(
                    sample_rate=self.sample_rate,
                    channel_name="DATA_MARGINAL",
                )
                self.delayed_data_trace = WaveformTrace(
                    data=data_marginal, metadata=metadata_marginal
                )

                print_result("Data loaded from file", file_to_load.name)
                print_result("Total samples", len(clock_data))
                print_result("Sample rate", f"{self.sample_rate / 1e9:.0f} GHz")
                print_result("Clock frequency", f"{self.clock_freq / 1e6:.0f} MHz")
                return
            except Exception as e:
                print_info(f"Failed to load data from file: {e}, falling back to synthetic")
                file_to_load = None

        # Generate synthetic data if not loaded
        print_info("Generating timing test signals...")

        # Signal duration: 1 us (100 clock cycles at 100 MHz)
        duration = 1e-6
        n_samples = int(duration * self.sample_rate)

        # Generate clock
        print_info("  Generating 100 MHz clock...")
        clock_ideal = self._generate_clock(n_samples)
        clock_with_edges = self._add_realistic_edges(clock_ideal, rise_time_ns=0.5)

        metadata = TraceMetadata(
            sample_rate=self.sample_rate,
            channel_name="CLK",
        )
        self.clock_trace = WaveformTrace(
            data=clock_with_edges * 3.3,  # 3.3V logic
            metadata=metadata,
        )

        # Generate data with good timing (meets spec)
        print_info(f"  Generating data with {self.spec_setup_min + 1:.1f} ns setup time...")
        good_setup_ns = self.spec_setup_min + 1.0  # 1 ns margin
        good_hold_ns = self.spec_hold_min + 0.5  # 0.5 ns margin

        data_good = self._generate_data_with_timing(
            clock_ideal,
            setup_time_ns=good_setup_ns,
            hold_time_ns=good_hold_ns,
        )
        data_good_edges = self._add_realistic_edges(data_good, rise_time_ns=0.8)

        metadata_data = TraceMetadata(
            sample_rate=self.sample_rate,
            channel_name="DATA",
        )
        self.data_trace = WaveformTrace(
            data=data_good_edges,
            metadata=metadata_data,
        )

        # Generate data with marginal timing
        print_info("  Generating data with marginal timing...")
        marginal_setup_ns = self.spec_setup_min - 0.3  # Slightly violating
        marginal_hold_ns = self.spec_hold_min

        data_marginal = self._generate_data_with_timing(
            clock_ideal,
            setup_time_ns=marginal_setup_ns,
            hold_time_ns=marginal_hold_ns,
        )
        data_marginal_edges = self._add_realistic_edges(data_marginal, rise_time_ns=0.8)

        metadata_delayed = TraceMetadata(
            sample_rate=self.sample_rate,
            channel_name="DATA_MARGINAL",
        )
        self.delayed_data_trace = WaveformTrace(
            data=data_marginal_edges,
            metadata=metadata_delayed,
        )

        print_result("Sample rate", f"{self.sample_rate / 1e9:.0f} GHz")
        print_result("Time resolution", f"{1e9 / self.sample_rate:.1f} ns")
        print_result("Clock frequency", f"{self.clock_freq / 1e6:.0f} MHz")
        print_result("Signal duration", f"{duration * 1e6:.1f} us")
        print_result("Total samples", n_samples)

    def run_analysis(self) -> None:
        """Perform setup/hold timing analysis."""
        print_subheader("Timing Analysis - Good Data Signal")

        # Measure setup time for good data
        t_setup = setup_time(
            self.data_trace,
            self.clock_trace,
            clock_edge="rising",
        )
        t_setup_ns = t_setup * 1e9 if not np.isnan(t_setup) else np.nan

        print_result("Setup time (measured)", f"{t_setup_ns:.2f} ns")
        print_result("Setup time (spec min)", f"{self.spec_setup_min:.2f} ns")

        setup_margin = t_setup_ns - self.spec_setup_min if not np.isnan(t_setup_ns) else np.nan
        if not np.isnan(setup_margin):
            if setup_margin >= 0:
                print_info(f"  {GREEN}Setup margin: +{setup_margin:.2f} ns (PASS){RESET}")
            else:
                print_info(f"  {RED}Setup margin: {setup_margin:.2f} ns (FAIL){RESET}")

        self.results["setup_time_good_ns"] = t_setup_ns
        self.results["setup_margin_good_ns"] = setup_margin

        # Measure hold time for good data
        t_hold = hold_time(
            self.data_trace,
            self.clock_trace,
            clock_edge="rising",
        )
        t_hold_ns = t_hold * 1e9 if not np.isnan(t_hold) else np.nan

        print_result("Hold time (measured)", f"{t_hold_ns:.2f} ns")
        print_result("Hold time (spec min)", f"{self.spec_hold_min:.2f} ns")

        hold_margin = t_hold_ns - self.spec_hold_min if not np.isnan(t_hold_ns) else np.nan
        if not np.isnan(hold_margin):
            if hold_margin >= 0:
                print_info(f"  {GREEN}Hold margin: +{hold_margin:.2f} ns (PASS){RESET}")
            else:
                print_info(f"  {RED}Hold margin: {hold_margin:.2f} ns (FAIL){RESET}")

        self.results["hold_time_good_ns"] = t_hold_ns
        self.results["hold_margin_good_ns"] = hold_margin

        # Analyze marginal data
        print_subheader("Timing Analysis - Marginal Data Signal")

        t_setup_marginal = setup_time(
            self.delayed_data_trace,
            self.clock_trace,
            clock_edge="rising",
        )
        t_setup_marginal_ns = t_setup_marginal * 1e9 if not np.isnan(t_setup_marginal) else np.nan

        print_result("Setup time (measured)", f"{t_setup_marginal_ns:.2f} ns")

        setup_margin_marginal = (
            t_setup_marginal_ns - self.spec_setup_min
            if not np.isnan(t_setup_marginal_ns)
            else np.nan
        )
        if not np.isnan(setup_margin_marginal):
            if setup_margin_marginal >= 0:
                print_info(
                    f"  {YELLOW}Setup margin: +{setup_margin_marginal:.2f} ns (MARGINAL){RESET}"
                )
            else:
                print_info(
                    f"  {RED}Setup margin: {setup_margin_marginal:.2f} ns (VIOLATION){RESET}"
                )

        self.results["setup_time_marginal_ns"] = t_setup_marginal_ns
        self.results["setup_margin_marginal_ns"] = setup_margin_marginal

        # Measure slew rate
        print_subheader("Edge Characterization")

        sr = slew_rate(self.clock_trace, ref_levels=(0.2, 0.8), edge_type="rising")
        if not np.isnan(sr):
            sr_vns = sr / 1e9  # V/ns
            print_result("Clock slew rate", f"{sr_vns:.2f} V/ns")
            self.results["clock_slew_rate_vns"] = sr_vns

        sr_data = slew_rate(self.data_trace, ref_levels=(0.2, 0.8), edge_type="rising")
        if not np.isnan(sr_data):
            sr_data_vns = sr_data / 1e9
            print_result("Data slew rate", f"{sr_data_vns:.2f} V/ns")
            self.results["data_slew_rate_vns"] = sr_data_vns

        # Calculate rise time from slew rate
        voltage_swing = 3.3 * 0.6  # 20%-80% of 3.3V
        if not np.isnan(sr) and sr > 0:
            rise_time_ns = voltage_swing / (sr / 1e9)
            print_result("Clock rise time (20-80%)", f"{rise_time_ns:.2f} ns")
            self.results["clock_rise_time_ns"] = rise_time_ns

        # Propagation delay analysis
        print_subheader("Propagation Delay")

        t_pd = propagation_delay(
            self.clock_trace,
            self.data_trace,
            edge_type="rising",
        )
        t_pd_ns = t_pd * 1e9 if not np.isnan(t_pd) else np.nan

        if not np.isnan(t_pd_ns):
            print_result("Clock-to-data delay", f"{t_pd_ns:.2f} ns")
            self.results["propagation_delay_ns"] = t_pd_ns

        # Get all delay measurements for statistics
        all_delays = propagation_delay(
            self.clock_trace,
            self.data_trace,
            edge_type="rising",
            return_all=True,
        )

        if len(all_delays) > 0:
            print_result("Delay measurements", len(all_delays))
            print_result("Delay min", f"{np.min(all_delays) * 1e9:.2f} ns")
            print_result("Delay max", f"{np.max(all_delays) * 1e9:.2f} ns")
            print_result("Delay std", f"{np.std(all_delays) * 1e9:.2f} ns")

            self.results["delay_min_ns"] = np.min(all_delays) * 1e9
            self.results["delay_max_ns"] = np.max(all_delays) * 1e9
            self.results["delay_std_ns"] = np.std(all_delays) * 1e9

        # Summary
        print_subheader("Timing Summary")

        violations = 0
        if not np.isnan(setup_margin) and setup_margin < 0:
            violations += 1
        if not np.isnan(hold_margin) and hold_margin < 0:
            violations += 1
        if not np.isnan(setup_margin_marginal) and setup_margin_marginal < 0:
            violations += 1

        self.results["violation_count"] = violations

        if violations == 0:
            print_info(f"{GREEN}All timing checks passed!{RESET}")
        else:
            print_info(f"{RED}{violations} timing violation(s) detected{RESET}")

        print_result("Setup spec", f">= {self.spec_setup_min:.1f} ns")
        print_result("Hold spec", f">= {self.spec_hold_min:.1f} ns")

    def validate_results(self, suite: ValidationSuite) -> None:
        """Validate timing analysis results."""
        # Check that measurements were performed
        suite.check_true(
            "Setup time measured (good)",
            not np.isnan(self.results.get("setup_time_good_ns", np.nan)),
            category="measurement",
        )

        suite.check_true(
            "Hold time measured (good)",
            not np.isnan(self.results.get("hold_time_good_ns", np.nan)),
            category="measurement",
        )

        # Check that good data passes timing
        setup_margin = self.results.get("setup_margin_good_ns", np.nan)
        if not np.isnan(setup_margin):
            suite.check_greater_equal(
                "Setup margin (good data)",
                setup_margin,
                -0.5,  # Allow small measurement error
                category="timing",
            )

        hold_margin = self.results.get("hold_margin_good_ns", np.nan)
        if not np.isnan(hold_margin):
            suite.check_greater_equal(
                "Hold margin (good data)",
                hold_margin,
                -0.5,  # Allow small measurement error
                category="timing",
            )

        # Check slew rate measurements
        clock_sr = self.results.get("clock_slew_rate_vns", 0)
        suite.check_greater(
            "Clock slew rate",
            clock_sr,
            0,
            category="edges",
        )

        # Check that signals were generated
        suite.check_true(
            "Clock trace generated",
            self.clock_trace is not None,
            category="signals",
        )

        suite.check_true(
            "Data trace generated",
            self.data_trace is not None,
            category="signals",
        )


if __name__ == "__main__":
    sys.exit(run_demo_main(SetupHoldTimingDemo))
