"""Bathtub Curves: BER vs sample point analysis

Demonstrates:
- Bathtub curve generation from jitter distribution
- Bit error rate (BER) calculation
- Eye opening measurement
- Timing margin analysis at various BER levels
- Extrapolation to low BER (1e-12, 1e-15)

IEEE Standards: IEEE 2414-2020
Related Demos:
- 04_advanced_analysis/01_jitter_analysis.py
- 04_advanced_analysis/02_jitter_decomposition.py
- 04_advanced_analysis/04_eye_diagrams.py

Generates bathtub curves for BER analysis and timing margin assessment.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from typing import TYPE_CHECKING, ClassVar

import numpy as np
from scipy import special

from demos.common.base_demo import BaseDemo, run_demo_main
from oscura.core.types import TraceMetadata, WaveformTrace

if TYPE_CHECKING:
    from demos.common.validation import ValidationSuite


class BathtubCurveDemo(BaseDemo):
    """Bathtub curve and BER analysis demonstration."""

    name = "Bathtub Curves and BER Analysis"
    description = "Generate bathtub curves for timing margin and BER analysis"
    category = "advanced_analysis"
    capabilities: ClassVar[list[str]] = [
        "Bathtub curve generation",
        "BER vs sample point",
        "Eye opening calculation",
        "Timing margin at BER threshold",
        "Low BER extrapolation (1e-12, 1e-15)",
    ]
    ieee_standards: ClassVar[list[str]] = ["IEEE 2414-2020"]
    related_demos: ClassVar[list[str]] = [
        "04_advanced_analysis/01_jitter_analysis.py",
        "04_advanced_analysis/02_jitter_decomposition.py",
    ]

    def generate_data(self) -> None:
        """Generate clock with known jitter for bathtub curve."""
        self.sample_rate = 10e9
        self.clock_freq = 100e6
        self.duration = 100e-6  # 100 µs for good statistics

        # Clock with moderate jitter
        self.clock = self._generate_jittered_clock(
            rj_rms=40e-12,  # 40 ps RMS
            dj_pk=60e-12,  # 60 ps pk-pk
        )

    def run_analysis(self) -> None:
        """Generate bathtub curves."""
        from demos.common.formatting import print_info, print_subheader

        print_subheader("Bathtub Curve Generation")

        # Extract jitter distribution
        edges = self._find_edges(self.clock.data, self.sample_rate)
        tie = self._calculate_tie(edges, self.clock_freq)

        # Generate bathtub curve
        sample_points = np.linspace(-0.5, 0.5, 1000)  # Normalized to UI
        ber_left, ber_right = self._generate_bathtub(tie, sample_points)

        # Find eye opening at various BER levels
        eye_opening_1e6 = self._find_eye_opening(sample_points, ber_left, ber_right, 1e-6)
        eye_opening_1e12 = self._find_eye_opening(sample_points, ber_left, ber_right, 1e-12)

        print_info(f"Eye opening @ BER 1e-6: {eye_opening_1e6 * 100:.1f}% UI")
        print_info(f"Eye opening @ BER 1e-12: {eye_opening_1e12 * 100:.1f}% UI")

        self.results["bathtub"] = {
            "sample_points": sample_points,
            "ber_left": ber_left,
            "ber_right": ber_right,
            "eye_opening_1e6": eye_opening_1e6,
            "eye_opening_1e12": eye_opening_1e12,
        }

    def validate_results(self, suite: ValidationSuite) -> None:
        """Validate bathtub curve results."""
        if "bathtub" in self.results:
            eye_1e12 = self.results["bathtub"].get("eye_opening_1e12", 0)
            suite.check_range(
                "eye_opening",
                eye_1e12 * 100,
                10.0,
                80.0,
                "Eye opening @ 1e-12 BER is reasonable",
            )

    def _generate_jittered_clock(self, rj_rms: float, dj_pk: float) -> WaveformTrace:
        """Generate clock with RJ and DJ."""
        nominal_period = 1.0 / self.clock_freq
        num_edges = int(self.duration * self.clock_freq * 2)
        ideal_edges = np.arange(num_edges) * (nominal_period / 2)

        rj = np.random.normal(0, rj_rms, num_edges)
        dj = np.random.uniform(-dj_pk / 2, dj_pk / 2, num_edges)
        actual_edges = ideal_edges + rj + dj

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

    def _find_edges(self, signal: np.ndarray, sample_rate: float) -> np.ndarray:
        """Find rising edges."""
        crossings = np.where(np.diff((signal > 0.5).astype(int)) == 1)[0]
        return crossings / sample_rate

    def _calculate_tie(self, edges: np.ndarray, clock_freq: float) -> np.ndarray:
        """Calculate TIE from edges."""
        periods = np.diff(edges[::2])
        nominal_period = 1.0 / clock_freq
        return periods - nominal_period

    def _generate_bathtub(
        self,
        tie: np.ndarray,
        sample_points: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate bathtub curves from TIE distribution."""
        unit_interval = 1.0 / self.clock_freq

        # Normalize TIE to UI
        tie_ui = tie / unit_interval

        # Calculate CDF for left and right edges
        tie_mean = np.mean(tie_ui)
        tie_std = np.std(tie_ui)

        # Left edge: probability of early arrival
        ber_left = 0.5 * special.erfc((sample_points - tie_mean) / (tie_std * np.sqrt(2)))

        # Right edge: probability of late arrival
        ber_right = 0.5 * special.erfc((-sample_points - tie_mean) / (tie_std * np.sqrt(2)))

        return ber_left, ber_right

    def _find_eye_opening(
        self,
        sample_points: np.ndarray,
        ber_left: np.ndarray,
        ber_right: np.ndarray,
        ber_threshold: float,
    ) -> float:
        """Find eye opening at specified BER threshold."""
        # Find points where BER < threshold
        valid_left = sample_points[ber_left < ber_threshold]
        valid_right = sample_points[ber_right < ber_threshold]

        if len(valid_left) == 0 or len(valid_right) == 0:
            return 0.0

        left_edge = np.max(valid_left)
        right_edge = np.min(valid_right)

        return max(0.0, right_edge - left_edge)


if __name__ == "__main__":
    sys.exit(run_demo_main(BathtubCurveDemo))
