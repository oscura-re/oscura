"""Jitter Decomposition: RJ/DJ separation using Dual-Dirac model

Demonstrates:
- Random jitter (RJ) extraction
- Deterministic jitter (DJ) extraction
- Periodic jitter (PJ) detection
- Dual-Dirac statistical model
- Total jitter (TJ) calculation

IEEE Standards: IEEE 2414-2020 (Jitter decomposition methods)
Related Demos:
- 04_advanced_analysis/01_jitter_analysis.py - Basic jitter measurements
- 04_advanced_analysis/03_bathtub_curves.py - BER analysis
- 04_advanced_analysis/04_eye_diagrams.py - Eye diagrams

Separates random and deterministic jitter components for root cause
analysis and compliance testing per IEEE 2414-2020.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np

from demos.common.base_demo import BaseDemo, run_demo_main
from demos.common.validation import ValidationSuite
from oscura.core.types import TraceMetadata, WaveformTrace


class JitterDecompositionDemo(BaseDemo):
    """RJ/DJ jitter decomposition demonstration."""

    name = "Jitter Decomposition (RJ/DJ)"
    description = "Separate random and deterministic jitter using Dual-Dirac model"
    category = "advanced_analysis"
    capabilities = [
        "Random jitter (RJ) extraction",
        "Deterministic jitter (DJ) extraction",
        "Periodic jitter (PJ) detection",
        "Dual-Dirac model fitting",
        "Total jitter extrapolation",
    ]
    ieee_standards = ["IEEE 2414-2020"]
    related_demos = [
        "04_advanced_analysis/01_jitter_analysis.py",
        "04_advanced_analysis/03_bathtub_curves.py",
    ]

    def generate_data(self) -> None:
        """Generate clocks with RJ, DJ, and mixed jitter."""
        self.sample_rate = 10e9
        self.clock_freq = 100e6
        self.duration = 10e-6

        # Pure random jitter
        self.rj_clock = self._generate_clock(rj_rms=50e-12, dj_pk=0, pj_amp=0)

        # Pure deterministic jitter
        self.dj_clock = self._generate_clock(rj_rms=5e-12, dj_pk=80e-12, pj_amp=0)

        # Periodic jitter component
        self.pj_clock = self._generate_clock(rj_rms=10e-12, dj_pk=0, pj_amp=60e-12, pj_freq=1e6)

        # Mixed RJ + DJ
        self.mixed_clock = self._generate_clock(
            rj_rms=40e-12,
            dj_pk=60e-12,
            pj_amp=30e-12,
            pj_freq=5e5,
        )

    def run_analysis(self) -> None:
        """Decompose jitter into RJ and DJ components."""
        from demos.common.formatting import print_subheader

        print_subheader("Pure Random Jitter")
        self.results["rj"] = self._decompose_jitter(self.rj_clock, "RJ Only")

        print_subheader("Pure Deterministic Jitter")
        self.results["dj"] = self._decompose_jitter(self.dj_clock, "DJ Only")

        print_subheader("Periodic Jitter")
        self.results["pj"] = self._decompose_jitter(self.pj_clock, "PJ")

        print_subheader("Mixed RJ + DJ")
        self.results["mixed"] = self._decompose_jitter(self.mixed_clock, "Mixed")

    def validate_results(self, suite: ValidationSuite) -> None:
        """Validate jitter decomposition."""
        if "rj" in self.results:
            suite.check_range(
                "rj_dominant",
                self.results["rj"].get("rj_rms_ps", 0),
                40.0,
                60.0,
                "RJ clock: RJ component dominant",
            )

        if "dj" in self.results:
            suite.check_range(
                "dj_dominant",
                self.results["dj"].get("dj_pk_ps", 0),
                60.0,
                100.0,
                "DJ clock: DJ component dominant",
            )

    def _generate_clock(
        self,
        rj_rms: float,
        dj_pk: float,
        pj_amp: float,
        pj_freq: float = 1e6,
    ) -> WaveformTrace:
        """Generate clock with RJ, DJ, and PJ components."""
        nominal_period = 1.0 / self.clock_freq
        num_edges = int(self.duration * self.clock_freq * 2)
        ideal_edges = np.arange(num_edges) * (nominal_period / 2)

        # Random jitter component (Gaussian)
        rj = np.random.normal(0, rj_rms, num_edges)

        # Deterministic jitter component (bounded)
        dj = np.random.uniform(-dj_pk / 2, dj_pk / 2, num_edges)

        # Periodic jitter component
        pj = pj_amp * np.sin(2 * np.pi * pj_freq * ideal_edges)

        # Total jitter
        actual_edges = ideal_edges + rj + dj + pj

        # Generate waveform
        num_samples = int(self.duration * self.sample_rate)
        signal = np.zeros(num_samples)

        for i in range(0, num_edges - 1, 2):
            rising_idx = int(actual_edges[i] * self.sample_rate)
            falling_idx = int(actual_edges[i + 1] * self.sample_rate)
            if 0 <= rising_idx < num_samples and 0 <= falling_idx < num_samples:
                signal[rising_idx:falling_idx] = 1.0

        return WaveformTrace(
            data=signal,
            metadata=TraceMetadata(sample_rate=self.sample_rate, channel_name="clock"),
        )

    def _decompose_jitter(self, clock: WaveformTrace, label: str) -> dict[str, float]:
        """Decompose jitter into RJ and DJ using Dual-Dirac model."""
        from demos.common.formatting import print_info

        edges = self._find_edges(clock.data, clock.metadata.sample_rate)
        if len(edges) < 10:
            return {}

        periods = np.diff(edges[::2])
        tie = periods - (1.0 / self.clock_freq)

        # Dual-Dirac model: fit tails to Gaussian
        tie_sorted = np.sort(tie)
        n = len(tie_sorted)

        # Use outer 27% for RJ estimation (±3σ region)
        lower_idx = int(0.135 * n)
        upper_idx = int(0.865 * n)

        # Fit Gaussian to tails
        tail_data = np.concatenate([tie_sorted[:lower_idx], tie_sorted[upper_idx:]])
        if len(tail_data) > 5:
            rj_rms = np.std(tail_data)
        else:
            rj_rms = np.std(tie)

        # DJ is the bounded component
        dj_pk = np.max(tie) - np.min(tie) - 6 * rj_rms

        # Total jitter at BER 1e-12 (7.04σ)
        tj_1e12 = 7.04 * rj_rms + abs(dj_pk)

        rj_rms_ps = rj_rms * 1e12
        dj_pk_ps = max(0, dj_pk * 1e12)
        tj_1e12_ps = tj_1e12 * 1e12

        print_info(f"RJ (RMS): {rj_rms_ps:.2f} ps")
        print_info(f"DJ (pk-pk): {dj_pk_ps:.2f} ps")
        print_info(f"TJ @ BER 1e-12: {tj_1e12_ps:.2f} ps")

        return {
            "rj_rms_ps": rj_rms_ps,
            "dj_pk_ps": dj_pk_ps,
            "tj_1e12_ps": tj_1e12_ps,
        }

    def _find_edges(self, signal: np.ndarray, sample_rate: float) -> np.ndarray:
        """Find rising edge times."""
        crossings = np.where(np.diff((signal > 0.5).astype(int)) == 1)[0]
        return crossings / sample_rate


if __name__ == "__main__":
    sys.exit(run_demo_main(JitterDecompositionDemo))
