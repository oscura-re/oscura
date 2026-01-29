"""Time-Domain Reflectometry: Impedance profiling via TDR

Demonstrates:
- TDR trace analysis
- Impedance vs distance profile
- Discontinuity detection
- Reflection coefficient calculation
- Cable characterization

IEEE Standards: IEEE 181-2011
Related Demos:
- 04_advanced_analysis/08_signal_integrity.py - S-parameters
- 02_basic_analysis/01_waveform_measurements.py

TDR for finding impedance discontinuities and cable faults.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np

from demos.common.base_demo import BaseDemo, run_demo_main
from demos.common.validation import ValidationSuite
from oscura.core.types import TraceMetadata, WaveformTrace


class TDRDemo(BaseDemo):
    """Time-domain reflectometry demonstration."""

    name = "Time-Domain Reflectometry (TDR)"
    description = "Impedance profiling and discontinuity detection"
    category = "advanced_analysis"
    capabilities = [
        "TDR trace analysis",
        "Impedance vs distance",
        "Discontinuity detection",
        "Reflection coefficient",
    ]
    ieee_standards = ["IEEE 181-2011"]
    related_demos = ["04_advanced_analysis/08_signal_integrity.py"]

    def generate_data(self) -> None:
        """Generate TDR response signals."""
        self.sample_rate = 100e9  # 100 GHz
        self.velocity = 2e8  # 2/3 speed of light

        # Perfect 50Ω cable
        self.tdr_matched = self._generate_tdr(impedances=[50], lengths=[10], z_source=50)

        # Cable with impedance step
        self.tdr_step = self._generate_tdr(impedances=[50, 75], lengths=[5, 5], z_source=50)

        # Cable with short
        self.tdr_short = self._generate_tdr(impedances=[50], lengths=[8], z_source=50, termination="short")

        # Cable with open
        self.tdr_open = self._generate_tdr(impedances=[50], lengths=[8], z_source=50, termination="open")

    def run_analysis(self) -> None:
        """Analyze TDR traces."""
        from demos.common.formatting import print_subheader

        print_subheader("Matched 50Ω Cable")
        self.results["matched"] = self._analyze_tdr(self.tdr_matched, "Matched")

        print_subheader("Impedance Step (50Ω → 75Ω)")
        self.results["step"] = self._analyze_tdr(self.tdr_step, "Step")

        print_subheader("Shorted Cable")
        self.results["short"] = self._analyze_tdr(self.tdr_short, "Short")

        print_subheader("Open Cable")
        self.results["open"] = self._analyze_tdr(self.tdr_open, "Open")

    def validate_results(self, suite: ValidationSuite) -> None:
        """Validate TDR analysis."""
        if "matched" in self.results:
            suite.check_exists(
                "matched_trace",
                self.results["matched"].get("trace"),
                "Matched cable TDR trace generated",
            )

    def _generate_tdr(
        self,
        impedances: list[float],
        lengths: list[float],
        z_source: float,
        termination: str = "matched",
    ) -> WaveformTrace:
        """Generate TDR response."""
        # Step input at t=0
        duration = sum(lengths) * 2 / self.velocity * 1.2  # Extra time for reflections
        num_samples = int(duration * self.sample_rate)
        time = np.arange(num_samples) / self.sample_rate

        # Initial step
        signal = np.ones(num_samples)

        # Calculate reflections at each discontinuity
        z_prev = z_source
        distance = 0

        for i, (z, length) in enumerate(zip(impedances, lengths)):
            # Reflection coefficient
            gamma = (z - z_prev) / (z + z_prev)

            # Time for reflection to return
            t_reflect = 2 * distance / self.velocity
            idx_reflect = int(t_reflect * self.sample_rate)

            # Add reflection
            if idx_reflect < num_samples:
                signal[idx_reflect:] += gamma

            distance += length
            z_prev = z

        # Termination reflection
        if termination == "short":
            gamma_term = -1.0
        elif termination == "open":
            gamma_term = 1.0
        else:  # matched
            gamma_term = 0.0

        t_term = 2 * distance / self.velocity
        idx_term = int(t_term * self.sample_rate)
        if idx_term < num_samples:
            signal[idx_term:] += gamma_term

        return WaveformTrace(
            data=signal,
            metadata=TraceMetadata(sample_rate=self.sample_rate, channel_name="tdr"),
        )

    def _analyze_tdr(self, trace: WaveformTrace, label: str) -> dict:
        """Analyze TDR trace."""
        from demos.common.formatting import print_info

        # Find discontinuities (sharp changes)
        diff = np.diff(trace.data)
        threshold = 0.1 * np.max(np.abs(diff))
        discontinuities = np.where(np.abs(diff) > threshold)[0]

        print_info(f"Discontinuities found: {len(discontinuities)}")

        for i, idx in enumerate(discontinuities[:5]):  # Show first 5
            time_ns = (idx / trace.metadata.sample_rate) * 1e9
            distance_m = (time_ns * 1e-9 * self.velocity) / 2
            magnitude = diff[idx]
            print_info(f"  #{i + 1}: {distance_m:.2f} m, Δ={magnitude:.3f}")

        return {
            "trace": trace,
            "discontinuities": discontinuities,
        }


if __name__ == "__main__":
    sys.exit(run_demo_main(TDRDemo))
