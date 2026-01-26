#!/usr/bin/env python3
"""Data-Dependent Jitter (DDJ) and Duty Cycle Distortion (DCD) Demonstration.

This demo showcases Oscura's DDJ and DCD analysis capabilities for
high-speed serial data characterization per IEEE 2414-2020.

**Features Demonstrated**:
- Data-Dependent Jitter (DDJ) extraction
- Duty Cycle Distortion (DCD) measurement
- Pattern-dependent jitter analysis
- ISI (Inter-Symbol Interference) correlation
- Pattern histogram generation
- Jitter component separation

**DDJ Sources**:
- Inter-Symbol Interference (ISI)
- Channel frequency response
- Transmitter pre-emphasis mismatch
- Receiver equalization mismatch

**DCD Sources**:
- Asymmetric rise/fall times
- Threshold voltage offset
- Driver duty cycle error
- Temperature variations

**Key Formulas**:
- DDJ_pp = max(TIE_pattern) - min(TIE_pattern)
- DCD = |mean_high_time - mean_low_time|
- DCD% = DCD / period * 100

Usage:
    python ddj_dcd_demo.py
    python ddj_dcd_demo.py --verbose

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
from oscura.analyzers.jitter import (
    extract_ddj,
    extract_dj,
    extract_rj,
    measure_dcd,
    tie_from_edges,
)
from oscura.core.types import TraceMetadata, WaveformTrace


class DDJDCDDemo(BaseDemo):
    """DDJ and DCD Jitter Analysis Demonstration.

    This demo generates signals with known DDJ and DCD characteristics
    and uses Oscura to extract and quantify these jitter components.
    """

    name = "DDJ/DCD Jitter Demo"
    description = "Demonstrates data-dependent jitter and duty cycle distortion analysis"
    category = "jitter_analysis"

    def __init__(self, **kwargs):
        """Initialize demo."""
        super().__init__(**kwargs)
        self.sample_rate = 10e9  # 10 GHz
        self.data_rate = 1e9  # 1 Gbps

        # Injected jitter parameters
        self.dcd_ps = 15.0  # 15 ps DCD (asymmetric duty cycle)
        self.ddj_ps = 10.0  # 10 ps DDJ (pattern-dependent)
        self.rj_rms_ps = 3.0  # 3 ps RMS random jitter

        self.trace = None
        self.tie_data = None
        self.bit_pattern = None

    def _generate_prbs_with_ddj(self, n_bits: int = 5000) -> tuple[np.ndarray, np.ndarray]:
        """Generate PRBS pattern with DDJ and DCD.

        Args:
            n_bits: Number of bits to generate.

        Returns:
            Tuple of (waveform, bit_pattern).
        """
        bit_period = 1 / self.data_rate
        samples_per_bit = int(self.sample_rate / self.data_rate)

        # Generate PRBS-7 pattern
        prbs = np.zeros(n_bits, dtype=int)
        state = 0x7F  # Initial state
        for i in range(n_bits):
            bit = (state >> 6) ^ (state >> 5) & 1
            prbs[i] = state & 1
            state = ((state << 1) | bit) & 0x7F

        # Generate edge times with jitter
        edge_times = []
        current_time = 0.0
        prev_bit = 0
        run_length = 0

        for _i, bit in enumerate(prbs):
            if bit != prev_bit:
                # Add jitter to edge
                rj = self.rj_rms_ps * 1e-12 * np.random.randn()

                # DDJ: depends on run length (ISI)
                # Longer runs have more ISI
                ddj_factor = min(run_length, 5) / 5.0
                ddj = self.ddj_ps * 1e-12 * ddj_factor * (1 if bit == 1 else -1)

                # DCD: rising edges early, falling edges late (or vice versa)
                dcd = self.dcd_ps * 1e-12 / 2 * (1 if bit == 1 else -1)

                edge_time = current_time + rj + ddj + dcd
                edge_times.append((edge_time, bit, run_length))
                run_length = 0
            else:
                run_length += 1

            current_time += bit_period
            prev_bit = bit

        # Generate waveform from edges
        n_samples = int(n_bits * samples_per_bit)
        waveform = np.zeros(n_samples)
        state = 0

        for edge_time, bit, _ in edge_times:
            edge_idx = int(edge_time * self.sample_rate)
            if 0 <= edge_idx < n_samples:
                state = bit
                waveform[edge_idx:] = state

        # Add realistic rise/fall times
        rise_samples = int(0.1e-9 * self.sample_rate)  # 100 ps
        if rise_samples > 1:
            kernel = np.ones(rise_samples) / rise_samples
            waveform = np.convolve(waveform, kernel, mode="same")

        # Add noise
        waveform += 0.02 * np.random.randn(n_samples)

        # Scale to voltage
        waveform = waveform * 0.8 + 0.1  # 0.1V to 0.9V

        return waveform, prbs

    def generate_data(self) -> None:
        """Generate or load test signal with DDJ and DCD.

        Tries in this order:
        1. Load from --data-file if specified
        2. Load from default demo_data files if they exist
        3. Generate synthetic data with PRBS pattern
        """
        # Try loading data from file
        data_file_to_load = None

        # 1. Check CLI override
        if self.data_file and self.data_file.exists():
            data_file_to_load = self.data_file
            print_info(f"Loading DDJ/DCD data from CLI override: {self.data_file}")
        # 2. Check default generated data
        elif default_file := self.find_default_data_file("ddj_dcd_jitter.npz"):
            data_file_to_load = default_file
            print_info(f"Loading DDJ/DCD data from default file: {default_file.name}")

        # Load from file if found
        if data_file_to_load:
            try:
                data = np.load(data_file_to_load)
                waveform = data["waveform"]
                self.bit_pattern = data["bit_pattern"]
                self.sample_rate = float(data["sample_rate"])
                self.data_rate = float(data["data_rate"])

                # Load injected jitter parameters if available
                if "dcd_ps" in data:
                    self.dcd_ps = float(data["dcd_ps"])
                if "ddj_ps" in data:
                    self.ddj_ps = float(data["ddj_ps"])
                if "rj_rms_ps" in data:
                    self.rj_rms_ps = float(data["rj_rms_ps"])

                print_info(f"  Data rate: {self.data_rate / 1e9:.1f} Gbps")
                print_info(f"  Injected DCD: {self.dcd_ps:.1f} ps")
                print_info(f"  Injected DDJ: {self.ddj_ps:.1f} ps")
                print_info(f"  Injected RJ: {self.rj_rms_ps:.1f} ps RMS")

                metadata = TraceMetadata(
                    sample_rate=self.sample_rate,
                    channel_name="DATA",
                )
                self.trace = WaveformTrace(data=waveform, metadata=metadata)

                print_result("Sample rate", f"{self.sample_rate / 1e9:.1f} GHz")
                print_result("Total samples", len(waveform))
                print_result("Bit pattern length", len(self.bit_pattern))
                return
            except Exception as e:
                print_info(f"  {YELLOW}Failed to load: {e}{RESET}")
                print_info("  Falling back to synthetic generation...")

        # Fallback: Generate synthetic data
        print_info("Generating PRBS signal with injected DDJ and DCD...")

        print_info(f"  Data rate: {self.data_rate / 1e9:.1f} Gbps")
        print_info(f"  Injected DCD: {self.dcd_ps:.1f} ps")
        print_info(f"  Injected DDJ: {self.ddj_ps:.1f} ps")
        print_info(f"  Injected RJ: {self.rj_rms_ps:.1f} ps RMS")

        waveform, self.bit_pattern = self._generate_prbs_with_ddj(n_bits=5000)

        metadata = TraceMetadata(
            sample_rate=self.sample_rate,
            channel_name="DATA",
        )
        self.trace = WaveformTrace(data=waveform, metadata=metadata)

        print_result("Sample rate", f"{self.sample_rate / 1e9:.1f} GHz")
        print_result("Total samples", len(waveform))
        print_result("Bit pattern length", len(self.bit_pattern))

    def run_analysis(self) -> None:
        """Perform DDJ and DCD analysis."""
        # ===== DCD Measurement =====
        print_subheader("Duty Cycle Distortion Analysis")

        dcd_result = measure_dcd(self.trace)

        print_result("DCD (time)", f"{dcd_result.dcd_seconds * 1e12:.2f} ps")
        print_result("DCD (%)", f"{dcd_result.dcd_percent:.2f}%")
        print_result("Duty cycle", f"{dcd_result.duty_cycle * 100:.2f}%")
        print_result("Mean high time", f"{dcd_result.mean_high_time * 1e12:.2f} ps")
        print_result("Mean low time", f"{dcd_result.mean_low_time * 1e12:.2f} ps")
        print_result("Period", f"{dcd_result.period * 1e12:.2f} ps")

        self.results["dcd_ps"] = dcd_result.dcd_seconds * 1e12
        self.results["dcd_percent"] = dcd_result.dcd_percent
        self.results["duty_cycle"] = dcd_result.duty_cycle

        # Evaluate DCD
        if dcd_result.dcd_percent < 2:
            rating = f"{GREEN}Excellent{RESET}"
        elif dcd_result.dcd_percent < 5:
            rating = f"{GREEN}Good{RESET}"
        elif dcd_result.dcd_percent < 10:
            rating = f"{YELLOW}Marginal{RESET}"
        else:
            rating = f"{RED}Poor{RESET}"
        print_info(f"DCD quality: {rating}")

        # Compare with injected value
        dcd_error = abs(dcd_result.dcd_seconds * 1e12 - self.dcd_ps)
        if dcd_error < 5:
            print_info(f"  {GREEN}Measured DCD within 5 ps of injected{RESET}")
        else:
            print_info(f"  {YELLOW}DCD measurement error: {dcd_error:.1f} ps{RESET}")

        # ===== Extract TIE for DDJ Analysis =====
        print_subheader("Time Interval Error Extraction")

        # Get edge timestamps
        data = self.trace.data
        threshold = (np.max(data) + np.min(data)) / 2

        # Find rising edges
        above = data >= threshold
        rising_indices = np.where(~above[:-1] & above[1:])[0]
        rising_edges = rising_indices / self.sample_rate

        if len(rising_edges) > 10:
            # Calculate TIE
            self.tie_data = tie_from_edges(rising_edges)

            print_result("TIE samples", len(self.tie_data))
            print_result("TIE mean", f"{np.mean(self.tie_data) * 1e12:.2f} ps")
            print_result("TIE std", f"{np.std(self.tie_data) * 1e12:.2f} ps")
            print_result("TIE pk-pk", f"{np.ptp(self.tie_data) * 1e12:.2f} ps")

            self.results["tie_rms_ps"] = np.std(self.tie_data) * 1e12
            self.results["tie_pp_ps"] = np.ptp(self.tie_data) * 1e12

        # ===== DDJ Extraction =====
        print_subheader("Data-Dependent Jitter Analysis")

        if self.tie_data is not None and len(self.tie_data) > 100:
            # Extract DDJ
            ddj_result = extract_ddj(self.tie_data, pattern_length=3)

            print_result("DDJ (pk-pk)", f"{ddj_result.ddj_pp * 1e12:.2f} ps")
            print_result("ISI coefficient", f"{ddj_result.isi_coefficient:.3f}")
            print_result("Pattern length", ddj_result.pattern_length)

            self.results["ddj_pp_ps"] = ddj_result.ddj_pp * 1e12
            self.results["isi_coefficient"] = ddj_result.isi_coefficient

            # Show pattern histogram
            print_info("Pattern-dependent mean TIE:")
            if ddj_result.pattern_histogram:
                sorted_patterns = sorted(ddj_result.pattern_histogram.items())
                for pattern, mean_tie in sorted_patterns[:8]:  # Show first 8
                    tie_ps = mean_tie * 1e12
                    print_info(f"  Pattern '{pattern}': {tie_ps:+.2f} ps")

            # Compare with injected DDJ
            ddj_error = abs(ddj_result.ddj_pp * 1e12 - self.ddj_ps)
            if ddj_error < 5:
                print_info(f"  {GREEN}Measured DDJ within 5 ps of injected{RESET}")
            else:
                print_info(f"  {YELLOW}DDJ measurement difference: {ddj_error:.1f} ps{RESET}")

        # ===== Jitter Component Separation =====
        print_subheader("Jitter Component Separation")

        if self.tie_data is not None and len(self.tie_data) >= 1000:
            # Extract RJ
            rj_result = extract_rj(self.tie_data, min_samples=1000)
            print_result("RJ (RMS)", f"{rj_result.rj_rms * 1e12:.2f} ps")
            print_result("RJ confidence", f"{rj_result.confidence:.2%}")

            self.results["rj_rms_ps"] = rj_result.rj_rms * 1e12

            # Extract DJ
            dj_result = extract_dj(self.tie_data, rj_result, min_samples=1000)
            print_result("DJ (pk-pk)", f"{dj_result.dj_pp * 1e12:.2f} ps")
            print_result("DJ confidence", f"{dj_result.confidence:.2%}")

            self.results["dj_pp_ps"] = dj_result.dj_pp * 1e12

            # Compare RJ with injected
            rj_error = abs(rj_result.rj_rms * 1e12 - self.rj_rms_ps)
            if rj_error < 2:
                print_info(f"  {GREEN}RJ within 2 ps of injected{RESET}")
            else:
                print_info(f"  {YELLOW}RJ difference: {rj_error:.1f} ps{RESET}")

        # ===== Summary =====
        print_subheader("Jitter Budget Summary")

        print_info("Component          Injected    Measured")
        print_info("-" * 45)
        print_info(
            f"DCD (ps)           {self.dcd_ps:>8.2f}    {self.results.get('dcd_ps', 0):>8.2f}"
        )
        print_info(
            f"DDJ pk-pk (ps)     {self.ddj_ps:>8.2f}    {self.results.get('ddj_pp_ps', 0):>8.2f}"
        )
        print_info(
            f"RJ RMS (ps)        {self.rj_rms_ps:>8.2f}    {self.results.get('rj_rms_ps', 0):>8.2f}"
        )

        # Total DJ budget check
        total_dj = self.results.get("dcd_ps", 0) + self.results.get("ddj_pp_ps", 0)
        print_info(f"Total DJ (estimated): {total_dj:.2f} ps")

    def validate_results(self, suite: ValidationSuite) -> None:
        """Validate DDJ/DCD demo results."""
        # Check DCD was measured
        suite.check_greater(
            "DCD measured",
            self.results.get("dcd_ps", 0),
            0,
            category="dcd",
        )

        # Check duty cycle is reasonable
        duty = self.results.get("duty_cycle", 0)
        suite.check_range(
            "Duty cycle",
            duty,
            0.3,  # At least 30%
            0.7,  # At most 70%
            category="dcd",
        )

        # Check DDJ was extracted
        suite.check_greater(
            "DDJ measured",
            self.results.get("ddj_pp_ps", 0),
            0,
            category="ddj",
        )

        # Check RJ was extracted
        suite.check_greater(
            "RJ measured",
            self.results.get("rj_rms_ps", 0),
            0,
            category="rj",
        )

        # Check trace was generated
        suite.check_true(
            "Trace generated",
            self.trace is not None,
            category="signals",
        )

        suite.check_true(
            "TIE data generated",
            self.tie_data is not None,
            category="signals",
        )


if __name__ == "__main__":
    sys.exit(run_demo_main(DDJDCDDemo))
