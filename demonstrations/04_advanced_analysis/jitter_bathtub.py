#!/usr/bin/env python3
"""Bathtub Curve Jitter Analysis Demonstration.

This demo showcases Oscura's bathtub curve generation and jitter
analysis capabilities for high-speed serial link characterization.

**Features Demonstrated**:
- Time Interval Error (TIE) measurement
- Bathtub curve generation
- BER extrapolation (1e-12)
- Eye opening at target BER
- Random jitter (RJ) estimation
- Deterministic jitter (DJ) estimation
- Total jitter at BER
- Dual-Dirac model fitting

**Bathtub Curve**:
The bathtub curve shows BER vs sampling position within a unit
interval. It's called "bathtub" because:
- Low BER at eye center (bottom of tub)
- High BER at eye edges (sides of tub)

**Jitter Components**:
- Random Jitter (RJ): Gaussian, unbounded
- Deterministic Jitter (DJ): Bounded, non-Gaussian
  - Periodic Jitter (PJ): Related to switching frequency
  - Data-Dependent Jitter (DDJ): Pattern-dependent
  - Duty Cycle Distortion (DCD): Asymmetric high/low times

**Key Formulas**:
- TJ @ BER = 2 * Q(BER) * RJ_rms + DJ_pp
- Q(1e-12) = 7.03

Usage:
    python bathtub_curve_demo.py
    python bathtub_curve_demo.py --verbose

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
from oscura.analyzers.jitter.ber import (
    bathtub_curve,
    q_factor_from_ber,
    tj_at_ber,
)
from oscura.analyzers.jitter.decomposition import decompose_jitter
from oscura.analyzers.jitter.measurements import (
    measure_dcd,
)
from oscura.core.types import TraceMetadata, WaveformTrace


class BathtubCurveDemo(BaseDemo):
    """Bathtub Curve Jitter Analysis Demonstration.

    This demo generates clock signals with jitter components and performs
    comprehensive jitter analysis including bathtub curve generation.
    """

    def __init__(self, **kwargs):
        """Initialize demo."""
        super().__init__(
            name="Bathtub Curve Jitter Demo",
            description="Demonstrates bathtub curve and jitter analysis for serial links",
            **kwargs,
        )
        self.sample_rate = 10e9  # 10 GHz
        self.clock_freq = 1e9  # 1 GHz clock

        # Jitter parameters
        self.rj_rms_ps = 2.0  # 2 ps RMS random jitter
        self.pj_pp_ps = 5.0  # 5 ps periodic jitter
        self.pj_freq = 10e6  # 10 MHz periodic jitter frequency
        self.dcd_ps = 3.0  # 3 ps duty cycle distortion

        self.trace = None
        self.tie_data = None

    def _generate_jittered_clock(self, n_samples: int) -> np.ndarray:
        """Generate clock signal with jitter components.

        Args:
            n_samples: Number of samples.

        Returns:
            Clock waveform.
        """
        dt = 1 / self.sample_rate
        t = np.arange(n_samples) * dt

        clock_period = 1 / self.clock_freq
        n_cycles = int(t[-1] / clock_period) + 1

        # Generate ideal edge times
        ideal_edges = np.arange(n_cycles) * clock_period

        # Add jitter to each edge
        rj = self.rj_rms_ps * 1e-12 * np.random.randn(n_cycles)
        pj = self.pj_pp_ps * 1e-12 / 2 * np.sin(2 * np.pi * self.pj_freq * ideal_edges)

        # DCD: alternating positive/negative offset
        dcd = np.zeros(n_cycles)
        dcd[::2] = self.dcd_ps * 1e-12 / 2
        dcd[1::2] = -self.dcd_ps * 1e-12 / 2

        jittered_edges = ideal_edges + rj + pj + dcd

        # Store TIE (Time Interval Error)
        self.tie_data = jittered_edges - ideal_edges

        # Generate waveform from edges
        clock = np.zeros(n_samples)
        state = 0

        for edge_time in jittered_edges:
            edge_idx = int(edge_time * self.sample_rate)
            if 0 <= edge_idx < n_samples:
                # Toggle state
                state = 1 - state
                clock[edge_idx:] = state

        # Add realistic rise/fall times using convolution
        rise_samples = int(0.1e-9 * self.sample_rate)  # 100 ps rise time
        if rise_samples > 1:
            kernel = np.ones(rise_samples) / rise_samples
            clock = np.convolve(clock, kernel, mode="same")

        # Add noise
        clock += 0.02 * np.random.randn(n_samples)

        return clock * 3.3  # 3.3V amplitude

    def generate_test_data(self) -> dict:
        """Generate jittered clock signal.

        Generates synthetic jittered clock with RJ, PJ, and DCD.
        """
        print_info("Generating jittered clock signal...")

        duration = 10e-6  # 10 us (10,000 cycles at 1 GHz)
        n_samples = int(self.sample_rate * duration)

        print_info(f"  Clock frequency: {self.clock_freq / 1e9:.1f} GHz")
        print_info(f"  RJ (RMS): {self.rj_rms_ps:.1f} ps")
        print_info(f"  PJ (pk-pk): {self.pj_pp_ps:.1f} ps @ {self.pj_freq / 1e6:.0f} MHz")
        print_info(f"  DCD: {self.dcd_ps:.1f} ps")

        clock_data = self._generate_jittered_clock(n_samples)

        metadata = TraceMetadata(
            sample_rate=self.sample_rate,
            channel_name="CLK",
        )
        self.trace = WaveformTrace(data=clock_data, metadata=metadata)

        print_result("Sample rate", f"{self.sample_rate / 1e9:.1f} GHz")
        print_result("Duration", f"{duration * 1e6:.1f} us")
        print_result("Total samples", n_samples)
        print_result("TIE points", len(self.tie_data))

        return {}

    def run_demonstration(self, data: dict) -> dict:
        """Perform jitter analysis and generate bathtub curve."""
        print_subheader("TIE Analysis")

        # Basic TIE statistics
        tie_ps = self.tie_data * 1e12  # Convert to picoseconds

        print_result("TIE samples", len(tie_ps))
        print_result("TIE mean", f"{np.mean(tie_ps):.2f} ps")
        print_result("TIE std (RMS)", f"{np.std(tie_ps):.2f} ps")
        print_result("TIE pk-pk", f"{np.ptp(tie_ps):.2f} ps")

        self.results["tie_rms_ps"] = np.std(tie_ps)
        self.results["tie_pp_ps"] = np.ptp(tie_ps)

        # Jitter decomposition
        print_subheader("Jitter Decomposition")

        # Use jitter decomposition
        decomp = decompose_jitter(self.tie_data)

        rj_measured = decomp.rj_rms * 1e12
        dj_measured = decomp.dj_pp * 1e12
        tj_measured = decomp.tj_pp * 1e12

        print_result("Random Jitter (RJ) RMS", f"{rj_measured:.2f} ps")
        print_result("Deterministic Jitter (DJ) pk-pk", f"{dj_measured:.2f} ps")
        print_result("Total Jitter (TJ) pk-pk", f"{tj_measured:.2f} ps")

        self.results["rj_rms_ps"] = rj_measured
        self.results["dj_pp_ps"] = dj_measured
        self.results["tj_pp_ps"] = tj_measured

        # Compare with input parameters
        print_info("Comparison with input parameters:")
        print_info(f"  RJ: measured={rj_measured:.2f} ps, input={self.rj_rms_ps:.1f} ps")
        print_info(
            f"  DJ: measured={dj_measured:.2f} ps, input={self.pj_pp_ps + self.dcd_ps:.1f} ps (PJ+DCD)"
        )

        # Bathtub curve
        print_subheader("Bathtub Curve")

        unit_interval = 1 / self.clock_freq
        target_ber = 1e-12

        bathtub = bathtub_curve(
            self.tie_data,
            unit_interval=unit_interval,
            target_ber=target_ber,
        )

        print_result("Unit interval", f"{unit_interval * 1e12:.2f} ps")
        print_result("Target BER", f"{target_ber:.0e}")
        print_result("Eye opening @ BER", f"{bathtub.eye_opening:.4f} UI")
        print_result("Eye opening (time)", f"{bathtub.eye_opening * unit_interval * 1e12:.2f} ps")

        self.results["eye_opening_ui"] = bathtub.eye_opening
        self.results["eye_opening_ps"] = bathtub.eye_opening * unit_interval * 1e12

        # Eye opening evaluation
        if bathtub.eye_opening > 0.4:
            rating = f"{GREEN}Excellent{RESET}"
        elif bathtub.eye_opening > 0.3:
            rating = f"{GREEN}Good{RESET}"
        elif bathtub.eye_opening > 0.2:
            rating = f"{YELLOW}Marginal{RESET}"
        else:
            rating = f"{RED}Poor{RESET}"
        print_info(f"Eye quality: {rating}")

        # Total jitter at BER
        print_subheader("Total Jitter at BER")

        q_factor = q_factor_from_ber(target_ber)
        print_result("Q-factor for 1e-12", f"{q_factor:.3f}")

        tj_ber = tj_at_ber(
            rj_rms=decomp.rj_rms,
            dj_pp=decomp.dj_pp,
            ber=target_ber,
        )

        print_result("TJ @ 1e-12 BER", f"{tj_ber * 1e12:.2f} ps")
        print_result(
            "TJ @ 1e-12 BER (UI)",
            f"{tj_ber / unit_interval:.4f} UI",
        )

        self.results["tj_at_ber_ps"] = tj_ber * 1e12

        # Formula verification
        tj_formula = 2 * q_factor * decomp.rj_rms + decomp.dj_pp
        print_info(
            f"Formula: TJ = 2*Q*RJ + DJ = 2*{q_factor:.3f}*{decomp.rj_rms * 1e12:.2f}ps + {decomp.dj_pp * 1e12:.2f}ps"
        )
        print_info(f"         TJ = {tj_formula * 1e12:.2f} ps")

        # Bathtub curve shape summary
        print_subheader("Bathtub Curve Data")

        print_info("Position (UI)  BER_left     BER_right    BER_total")
        print_info("-" * 55)

        for i in range(0, len(bathtub.positions), max(1, len(bathtub.positions) // 10)):
            pos = bathtub.positions[i]
            ber_l = bathtub.ber_left[i]
            ber_r = bathtub.ber_right[i]
            ber_t = bathtub.ber_total[i]

            print_info(f"  {pos:6.3f}       {ber_l:10.2e}   {ber_r:10.2e}   {ber_t:10.2e}")

        # DCD analysis
        print_subheader("Duty Cycle Distortion")

        dcd_result = measure_dcd(self.trace)

        print_result("DCD", f"{dcd_result.dcd_seconds * 1e12:.2f} ps")
        print_result("DCD %", f"{dcd_result.dcd_percent:.2f}%")
        print_result("Duty cycle", f"{dcd_result.duty_cycle * 100:.2f}%")
        print_result("Mean high time", f"{dcd_result.mean_high_time * 1e12:.2f} ps")
        print_result("Mean low time", f"{dcd_result.mean_low_time * 1e12:.2f} ps")

        self.results["dcd_measured_ps"] = dcd_result.dcd_seconds * 1e12
        self.results["duty_cycle_pct"] = dcd_result.duty_cycle * 100

        # Summary
        print_subheader("Jitter Summary")
        print_info("Component             Measured      Injected")
        print_info("-" * 45)
        print_info(f"RJ (RMS)              {rj_measured:6.2f} ps    {self.rj_rms_ps:.2f} ps")
        print_info(
            f"DJ (pk-pk)            {dj_measured:6.2f} ps    {self.pj_pp_ps + self.dcd_ps:.2f} ps"
        )
        print_info(f"TJ @ 1e-12 BER        {tj_ber * 1e12:6.2f} ps")
        print_info(f"Eye Opening           {bathtub.eye_opening:6.4f} UI")

        return self.results

    def validate(self, results: dict) -> bool:
        """Validate jitter analysis results."""
        suite = ValidationSuite()

        # Check TIE was calculated
        tie_rms = results.get("tie_rms_ps", 0)
        suite.add_check("TIE RMS calculated", tie_rms > 0, f"Got {tie_rms:.2f} ps")

        # Check jitter decomposition
        rj_rms = results.get("rj_rms_ps", 0)
        suite.add_check("RJ RMS calculated", rj_rms > 0, f"Got {rj_rms:.2f} ps")

        # Check eye opening is reasonable
        eye = results.get("eye_opening_ui", 0)
        suite.add_check("Eye opening calculated", 0 < eye < 1, f"Got {eye:.4f} UI")

        # Check TJ at BER
        tj_at_ber = results.get("tj_at_ber_ps", 0)
        suite.add_check("TJ at BER calculated", tj_at_ber > 0, f"Got {tj_at_ber:.2f} ps")

        # Check trace was generated
        suite.add_check(
            "Trace generated",
            self.trace is not None and len(self.trace.data) > 0,
            f"Got {len(self.trace.data) if self.trace is not None else 0} samples",
        )

        # Check TIE data
        suite.add_check(
            "TIE data generated",
            self.tie_data is not None and len(self.tie_data) > 0,
            f"Got {len(self.tie_data) if self.tie_data is not None else 0} TIE points",
        )

        suite.report()
        return suite.all_passed()


if __name__ == "__main__":
    sys.exit(run_demo_main(BathtubCurveDemo))
