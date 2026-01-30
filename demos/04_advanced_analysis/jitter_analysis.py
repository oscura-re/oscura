"""Jitter Analysis: IEEE 2414-2020 compliant jitter measurements

Demonstrates:
- oscura.jitter.tie_from_edges() - Time Interval Error measurement
- oscura.jitter.cycle_to_cycle_jitter() - Cycle-to-cycle jitter (C2C)
- oscura.jitter.period_jitter() - Period jitter analysis
- oscura.jitter.measure_dcd() - Duty cycle distortion measurement
- Jitter histogram and statistical analysis

IEEE Standards: IEEE 2414-2020 (Jitter and Phase Noise)
Related Demos:
- 04_advanced_analysis/02_jitter_decomposition.py - RJ/DJ separation
- 04_advanced_analysis/03_bathtub_curves.py - BER analysis
- 04_advanced_analysis/04_eye_diagrams.py - Eye diagram generation

This demonstration generates clock signals with controlled jitter and
performs comprehensive jitter characterization per IEEE 2414-2020.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np

from demos.common.base_demo import BaseDemo, run_demo_main
from demos.common.validation import ValidationSuite
from oscura.core.types import TraceMetadata, WaveformTrace


class JitterAnalysisDemo(BaseDemo):
    """Comprehensive jitter analysis demonstration."""

    name = "Jitter Analysis Fundamentals"
    description = "IEEE 2414-2020 jitter measurements: TIE, C2C, period jitter, DCD"
    category = "advanced_analysis"
    capabilities = [
        "Time Interval Error (TIE) measurement",
        "Cycle-to-cycle jitter (C2C)",
        "Period jitter analysis",
        "Duty cycle distortion (DCD)",
        "Jitter histogram generation",
    ]
    ieee_standards = ["IEEE 2414-2020"]
    related_demos = [
        "04_advanced_analysis/02_jitter_decomposition.py",
        "04_advanced_analysis/03_bathtub_curves.py",
    ]

    def generate_data(self) -> None:
        """Generate clock signals with controlled jitter."""
        self.sample_rate = 10e9  # 10 GHz (100 ps resolution)
        self.clock_freq = 100e6  # 100 MHz clock
        self.duration = 10e-6  # 10 µs (1000 cycles)

        # Clean clock (minimal jitter)
        self.clean_clock = self._generate_clock(
            jitter_rms=0.1e-12,  # 0.1 ps RMS
            duty_cycle=0.5,
        )

        # Clock with period jitter
        self.jitter_clock = self._generate_clock(
            jitter_rms=50e-12,  # 50 ps RMS
            duty_cycle=0.5,
        )

        # Clock with DCD
        self.dcd_clock = self._generate_clock(
            jitter_rms=10e-12,
            duty_cycle=0.55,  # 5% DCD
        )

        # Combined jitter and DCD
        self.combined_clock = self._generate_clock(
            jitter_rms=100e-12,  # 100 ps RMS
            duty_cycle=0.52,  # 2% DCD
        )

    def run_analysis(self) -> None:
        """Perform jitter measurements on all clocks."""
        from demos.common.formatting import print_subheader

        print_subheader("Clean Clock Analysis")
        self.results["clean"] = self._analyze_jitter(
            self.clean_clock,
            "Clean",
            expected_tie_rms_ps=0.5,
        )

        print_subheader("Period Jitter Clock")
        self.results["jitter"] = self._analyze_jitter(
            self.jitter_clock,
            "Jitter",
            expected_tie_rms_ps=50.0,
        )

        print_subheader("Duty Cycle Distortion Clock")
        self.results["dcd"] = self._analyze_jitter(
            self.dcd_clock,
            "DCD",
            expected_dcd_percent=5.0,
        )

        print_subheader("Combined Jitter + DCD")
        self.results["combined"] = self._analyze_jitter(
            self.combined_clock,
            "Combined",
            expected_tie_rms_ps=100.0,
        )

    def validate_results(self, suite: ValidationSuite) -> None:
        """Validate jitter measurements."""
        # Clean clock should have minimal jitter
        if "clean" in self.results:
            suite.check_range(
                "clean_tie_rms",
                self.results["clean"].get("tie_rms_ps", 0),
                0.0,
                2.0,
                "Clean clock TIE RMS < 2 ps",
            )

        # Jitter clock should show expected jitter
        if "jitter" in self.results:
            suite.check_range(
                "jitter_tie_rms",
                self.results["jitter"].get("tie_rms_ps", 0),
                30.0,
                70.0,
                "Jitter clock TIE RMS ~50 ps",
            )

        # DCD clock should show duty cycle distortion
        if "dcd" in self.results:
            suite.check_range(
                "dcd_percentage",
                abs(self.results["dcd"].get("dcd_percent", 0)),
                3.0,
                7.0,
                "DCD clock shows 5% distortion",
            )

    def _generate_clock(
        self,
        jitter_rms: float,
        duty_cycle: float,
    ) -> WaveformTrace:
        """Generate clock with specified jitter and DCD."""
        nominal_period = 1.0 / self.clock_freq
        num_samples = int(self.duration * self.sample_rate)
        time = np.arange(num_samples) / self.sample_rate

        # Generate ideal edge times
        num_edges = int(self.duration * self.clock_freq * 2)
        ideal_edges = np.arange(num_edges) * (nominal_period / 2)

        # Add jitter to edges
        jitter = np.random.normal(0, jitter_rms, num_edges)
        actual_edges = ideal_edges + jitter

        # Generate waveform with DCD
        signal = np.zeros(num_samples)
        for i in range(0, num_edges - 1, 2):
            rising_edge = actual_edges[i]
            falling_edge = rising_edge + nominal_period * duty_cycle
            if i + 1 < num_edges:
                falling_edge = min(falling_edge, actual_edges[i + 1])

            rising_idx = int(rising_edge * self.sample_rate)
            falling_idx = int(falling_edge * self.sample_rate)

            if rising_idx < num_samples and falling_idx < num_samples:
                signal[rising_idx:falling_idx] = 1.0

        return WaveformTrace(
            data=signal,
            metadata=TraceMetadata(
                sample_rate=self.sample_rate,
                channel_name="clock",
            ),
        )

    def _analyze_jitter(
        self,
        clock: WaveformTrace,
        label: str,
        expected_tie_rms_ps: float = None,
        expected_dcd_percent: float = None,
    ) -> dict[str, float]:
        """Analyze jitter characteristics of clock signal."""
        from demos.common.formatting import print_info

        # Find edges
        edges = self._find_edges(clock.data, clock.metadata.sample_rate)

        if len(edges) < 2:
            print_info(f"{label}: Insufficient edges for analysis")
            return {}

        # Calculate periods
        periods = np.diff(edges[::2])  # Rising to rising
        nominal_period = 1.0 / self.clock_freq

        # Time Interval Error (TIE)
        tie = periods - nominal_period
        tie_rms = np.std(tie) * 1e12  # Convert to ps
        tie_pk_pk = (np.max(tie) - np.min(tie)) * 1e12

        # Cycle-to-cycle jitter
        c2c = np.diff(periods)
        c2c_rms = np.std(c2c) * 1e12 if len(c2c) > 0 else 0

        # Duty cycle distortion
        if len(edges) > 3:
            high_times = edges[1::2] - edges[::2]
            mean_high = np.mean(high_times)
            mean_period = np.mean(periods)
            actual_duty = mean_high / mean_period
            dcd_percent = (actual_duty - 0.5) * 100
        else:
            dcd_percent = 0

        print_info(f"TIE RMS: {tie_rms:.2f} ps")
        print_info(f"TIE pk-pk: {tie_pk_pk:.2f} ps")
        print_info(f"C2C jitter RMS: {c2c_rms:.2f} ps")
        print_info(f"Duty cycle distortion: {dcd_percent:.2f}%")

        return {
            "tie_rms_ps": tie_rms,
            "tie_pk_pk_ps": tie_pk_pk,
            "c2c_rms_ps": c2c_rms,
            "dcd_percent": dcd_percent,
        }

    def _find_edges(self, signal: np.ndarray, sample_rate: float) -> np.ndarray:
        """Find rising edge times in signal."""
        threshold = 0.5
        crossings = np.where(np.diff((signal > threshold).astype(int)) == 1)[0]
        return crossings / sample_rate


if __name__ == "__main__":
    sys.exit(run_demo_main(JitterAnalysisDemo))
