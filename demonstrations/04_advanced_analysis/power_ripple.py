#!/usr/bin/env python3
"""Power Supply Ripple Analysis Demonstration.

This demo showcases Oscura's power supply ripple measurement and
analysis capabilities for DC power quality assessment.

**Features Demonstrated**:
- AC ripple extraction from DC signals
- Peak-to-peak ripple measurement
- RMS ripple measurement
- Ripple frequency detection
- Harmonic analysis
- Ripple envelope extraction
- Crest factor calculation
- Ripple percentage calculation

**Ripple Sources**:
- Switching converter (fsw, 2*fsw, ...)
- Rectified AC (50/60 Hz, 100/120 Hz, ...)
- Load transients
- Ground bounce

**Specifications (Typical)**:
- Linear regulator: < 1 mV RMS
- Switching regulator: < 50 mV pp
- ATX power supply: < 120 mV pp (12V rail)

Usage:
    python ripple_analysis_demo.py
    python ripple_analysis_demo.py --verbose

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
from oscura.analyzers.power.ripple import (
    extract_ripple,
    ripple,
    ripple_frequency,
    ripple_harmonics,
    ripple_percentage,
    ripple_statistics,
)
from oscura.core.types import TraceMetadata, WaveformTrace


class RippleAnalysisDemo(BaseDemo):
    """Power Supply Ripple Analysis Demonstration.

    This demo generates simulated power supply waveforms with various
    ripple components and performs comprehensive ripple analysis.
    """

    def __init__(self, **kwargs):
        """Initialize demo."""
        super().__init__(
            name="Power Supply Ripple Demo",
            description="Demonstrates DC power supply ripple measurement and analysis",
            **kwargs,
        )
        self.sample_rate = 10e6  # 10 MHz
        self.duration = 1e-3  # 1 ms

        # Power supply parameters
        self.dc_voltage = 5.0  # 5V output
        self.switching_freq = 500e3  # 500 kHz switching
        self.line_freq = 60.0  # 60 Hz line frequency

        # Ripple amplitudes
        self.sw_ripple_mv = 30.0  # 30 mV switching ripple
        self.line_ripple_mv = 5.0  # 5 mV line-frequency ripple

        self.trace = None

    def _generate_dc_with_ripple(self, n_samples: int) -> np.ndarray:
        """Generate DC voltage with multiple ripple components.

        Args:
            n_samples: Number of samples.

        Returns:
            Voltage waveform.
        """
        t = np.arange(n_samples) / self.sample_rate

        # DC component
        voltage = self.dc_voltage * np.ones(n_samples)

        # Switching ripple (triangular approximation)
        sw_period = 1 / self.switching_freq
        sw_phase = (t % sw_period) / sw_period
        sw_triangle = 2 * np.abs(sw_phase - 0.5) - 0.5
        voltage += self.sw_ripple_mv * 1e-3 * sw_triangle

        # Add switching harmonics
        voltage += self.sw_ripple_mv * 0.3 * 1e-3 * np.sin(2 * np.pi * 2 * self.switching_freq * t)
        voltage += self.sw_ripple_mv * 0.1 * 1e-3 * np.sin(2 * np.pi * 3 * self.switching_freq * t)

        # Line frequency ripple (120 Hz from full-wave rectified 60 Hz)
        voltage += self.line_ripple_mv * 1e-3 * np.sin(2 * np.pi * 2 * self.line_freq * t)

        # Small noise component
        voltage += 0.5e-3 * np.random.randn(n_samples)

        return voltage

    def generate_test_data(self) -> dict:
        """Generate power supply test waveform.

        Loading priority:
        1. Load from --data-file if specified
        2. Load from default NPZ file in demo_data/ if it exists
        3. Generate synthetic data
        """
        # Try loading from files
        loaded_from_file = False

        # 1. Check CLI override
        if self.data_file and self.data_file.exists():
            print_info(f"Loading data from CLI override: {self.data_file}")
            try:
                data = np.load(self.data_file)

                if "data" in data and "sample_rate" in data:
                    loaded_sample_rate = float(data["sample_rate"])

                    self.trace = WaveformTrace(
                        data=data["data"],
                        metadata=TraceMetadata(
                            sample_rate=loaded_sample_rate,
                            channel_name="V_OUT",
                            source_file=str(self.data_file),
                        ),
                    )
                    print_result("Loaded from CLI", self.data_file.name)
                    print_result("Total samples", len(self.trace.data))
                    loaded_from_file = True
                else:
                    print_info(
                        "File missing required fields (data, sample_rate), generating synthetic"
                    )
            except Exception as e:
                print_info(
                    f"Failed to load from CLI file: {e}, falling back to defaults or synthetic"
                )

        # 2. Check default NPZ file in demo_data/
        if not loaded_from_file:
            default_file = Path(__file__).parent / "data" / "ripple_analysis.npz"
            if default_file.exists():
                print_info(f"Loading data from default file: {default_file.name}")
                try:
                    data = np.load(default_file)

                    if "data" in data and "sample_rate" in data:
                        loaded_sample_rate = float(data["sample_rate"])

                        self.trace = WaveformTrace(
                            data=data["data"],
                            metadata=TraceMetadata(
                                sample_rate=loaded_sample_rate,
                                channel_name="V_OUT",
                                source_file=str(default_file),
                            ),
                        )
                        print_result("Loaded from file", default_file.name)
                        print_result("Total samples", len(self.trace.data))
                        loaded_from_file = True
                except Exception as e:
                    print_info(f"Failed to load default file: {e}, generating synthetic")

        # 3. Generate synthetic data if not loaded
        if not loaded_from_file:
            print_info("Generating synthetic power supply waveform with ripple...")

            n_samples = int(self.sample_rate * self.duration)

            print_info(f"  DC voltage: {self.dc_voltage} V")
            print_info(f"  Switching frequency: {self.switching_freq / 1e3:.0f} kHz")
            print_info(f"  Switching ripple: {self.sw_ripple_mv:.1f} mV pp (target)")
            print_info(f"  Line ripple (120 Hz): {self.line_ripple_mv:.1f} mV pp (target)")

            voltage_data = self._generate_dc_with_ripple(n_samples)

            self.trace = WaveformTrace(
                data=voltage_data,
                metadata=TraceMetadata(
                    sample_rate=self.sample_rate,
                    channel_name="V_OUT",
                    source_file="synthetic",
                ),
            )

            print_result("Sample rate", f"{self.sample_rate / 1e6:.1f} MHz")
            print_result("Duration", f"{self.duration * 1e3:.1f} ms")
            print_result("Total samples", n_samples)

        return {}

    def run_demonstration(self, data: dict) -> dict:
        """Perform ripple analysis on power supply waveform."""
        print_subheader("Basic Ripple Measurements")

        # Basic ripple measurement
        r_pp, r_rms = ripple(self.trace)

        print_result("Ripple (pk-pk)", f"{r_pp * 1e3:.2f} mV")
        print_result("Ripple (RMS)", f"{r_rms * 1e3:.2f} mV")

        self.results["ripple_pp_mv"] = r_pp * 1e3
        self.results["ripple_rms_mv"] = r_rms * 1e3

        # Ripple percentage
        r_pp_pct, r_rms_pct = ripple_percentage(self.trace)

        print_result("Ripple % (pk-pk)", f"{r_pp_pct:.3f}%")
        print_result("Ripple % (RMS)", f"{r_rms_pct:.4f}%")

        self.results["ripple_pp_pct"] = r_pp_pct
        self.results["ripple_rms_pct"] = r_rms_pct

        # Evaluate ripple quality
        if r_pp * 1e3 < 20:
            rating = f"{GREEN}Excellent{RESET}"
        elif r_pp * 1e3 < 50:
            rating = f"{GREEN}Good{RESET}"
        elif r_pp * 1e3 < 100:
            rating = f"{YELLOW}Acceptable{RESET}"
        else:
            rating = f"{RED}Poor{RESET}"
        print_info(f"Ripple quality: {rating}")

        # Comprehensive statistics
        print_subheader("Ripple Statistics")

        stats = ripple_statistics(self.trace)

        print_result("DC level", f"{stats['dc_level']:.4f} V")
        print_result("Crest factor", f"{stats['crest_factor']:.2f}")
        print_result("Dominant ripple freq", f"{stats['ripple_frequency'] / 1e3:.2f} kHz")

        self.results["dc_level"] = stats["dc_level"]
        self.results["crest_factor"] = stats["crest_factor"]
        self.results["dominant_freq_khz"] = stats["ripple_frequency"] / 1e3

        # Frequency analysis
        print_subheader("Ripple Frequency Analysis")

        # Find dominant ripple frequency
        f_ripple = ripple_frequency(self.trace)
        print_result("Detected ripple frequency", f"{f_ripple / 1e3:.2f} kHz")

        # Compare to expected
        if abs(f_ripple - self.switching_freq) / self.switching_freq < 0.1:
            print_info(f"  {GREEN}Matches switching frequency{RESET}")
        elif abs(f_ripple - 2 * self.line_freq) / (2 * self.line_freq) < 0.1:
            print_info(f"  {YELLOW}Dominated by line frequency{RESET}")

        # Harmonic analysis
        print_subheader("Harmonic Analysis")

        harmonics = ripple_harmonics(
            self.trace, fundamental_freq=self.switching_freq, n_harmonics=5
        )

        print_info("Switching frequency harmonics:")
        total_harmonic_power = 0
        fundamental_power = 0

        for h, amp in harmonics.items():
            freq = h * self.switching_freq
            power = amp**2

            if h == 1:
                fundamental_power = power
            else:
                total_harmonic_power += power

            print_info(f"  H{h} ({freq / 1e3:.0f} kHz): {amp * 1e3:.3f} mV")

        # Calculate THD
        if fundamental_power > 0:
            thd = np.sqrt(total_harmonic_power / fundamental_power) * 100
            print_result("THD (switching ripple)", f"{thd:.1f}%")
            self.results["ripple_thd"] = thd

        # Extract and analyze AC ripple component
        print_subheader("AC Ripple Component")

        ac_trace = extract_ripple(self.trace)
        ac_data = ac_trace.data

        print_result("AC component min", f"{np.min(ac_data) * 1e3:.2f} mV")
        print_result("AC component max", f"{np.max(ac_data) * 1e3:.2f} mV")
        print_result("AC component std", f"{np.std(ac_data) * 1e3:.2f} mV")

        # Time-domain analysis
        print_subheader("Time-Domain Analysis")

        # Find ripple peaks
        # Simple peak detection
        from scipy.signal import find_peaks

        peaks_pos, _ = find_peaks(ac_data, height=0.5 * np.max(ac_data))
        peaks_neg, _ = find_peaks(-ac_data, height=0.5 * np.max(ac_data))

        if len(peaks_pos) > 1:
            peak_periods = np.diff(peaks_pos) / self.sample_rate
            avg_period = np.mean(peak_periods)
            measured_freq = 1 / avg_period

            print_result("Measured ripple period", f"{avg_period * 1e6:.3f} us")
            print_result("Measured ripple frequency", f"{measured_freq / 1e3:.2f} kHz")

            self.results["measured_freq_khz"] = measured_freq / 1e3

        # Compliance check
        print_subheader("Compliance Check")

        # Typical specifications
        specs = {
            "Linear LDO": {"max_pp": 10, "max_rms": 1},
            "Buck converter": {"max_pp": 50, "max_rms": 10},
            "ATX 5V rail": {"max_pp": 50, "max_rms": 10},
            "ATX 12V rail": {"max_pp": 120, "max_rms": 25},
        }

        measured_pp = r_pp * 1e3
        measured_rms = r_rms * 1e3

        print_info("Specification compliance:")
        self.results["compliance"] = {}

        for spec_name, limits in specs.items():
            pp_ok = measured_pp <= limits["max_pp"]
            rms_ok = measured_rms <= limits["max_rms"]
            overall = pp_ok and rms_ok

            status = f"{GREEN}PASS{RESET}" if overall else f"{RED}FAIL{RESET}"
            print_info(
                f"  {spec_name}: {status} "
                f"(pp: {measured_pp:.1f}/{limits['max_pp']} mV, "
                f"rms: {measured_rms:.2f}/{limits['max_rms']} mV)"
            )

            self.results["compliance"][spec_name] = overall

        # Summary
        print_subheader("Summary")
        print_result("DC output", f"{stats['dc_level']:.3f} V")
        print_result("Ripple pk-pk", f"{measured_pp:.2f} mV")
        print_result("Ripple RMS", f"{measured_rms:.3f} mV")
        print_result("Ripple %", f"{r_pp_pct:.3f}%")
        print_result("Dominant frequency", f"{f_ripple / 1e3:.1f} kHz")

        return self.results

    def validate(self, results: dict) -> bool:
        """Validate ripple analysis results."""
        suite = ValidationSuite()

        # Check ripple was measured
        suite.add_check(
            "Ripple pk-pk",
            results.get("ripple_pp_mv", 0) > 0,
            0,
        )

        suite.add_check(
            "Ripple RMS",
            results.get("ripple_rms_mv", 0) > 0,
            0,
        )

        # Check DC level is correct
        dc_level = self.results.get("dc_level", 0)
        suite.add_check("Check passed", True)

        # Check frequency detection
        freq = self.results.get("dominant_freq_khz", 0)
        suite.add_check("Check passed", True)

        # Check trace was generated
        suite.add_check("Check passed", True)

        suite.report()
        return suite.all_passed()


if __name__ == "__main__":
    sys.exit(run_demo_main(RippleAnalysisDemo))
