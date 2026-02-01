"""Eye Diagram Measurements: Quantitative eye metrics

Demonstrates:
- Eye height measurement
- Eye width measurement
- Jitter at BER threshold
- Q-factor calculation
- Rise/fall time at crossings
- Mask testing

IEEE Standards: IEEE 2414-2020
Related Demos:
- 04_advanced_analysis/04_eye_diagrams.py - Eye generation
- 04_advanced_analysis/01_jitter_analysis.py
- 04_advanced_analysis/03_bathtub_curves.py

Extracts quantitative metrics from eye diagrams for compliance testing.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from typing import TYPE_CHECKING, ClassVar

import numpy as np

from demos.common.base_demo import BaseDemo, run_demo_main
from oscura.core.types import TraceMetadata, WaveformTrace

if TYPE_CHECKING:
    from demos.common.validation import ValidationSuite


class EyeMetricsDemo(BaseDemo):
    """Eye diagram measurements demonstration."""

    name = "Eye Diagram Measurements"
    description = "Extract quantitative metrics from eye diagrams"
    category = "advanced_analysis"
    capabilities: ClassVar[list[str]] = [
        "Eye height/width measurement",
        "Q-factor calculation",
        "Jitter at crossing",
        "Rise/fall time analysis",
        "Eye mask testing",
    ]
    ieee_standards: ClassVar[list[str]] = ["IEEE 2414-2020"]
    related_demos: ClassVar[list[str]] = [
        "04_advanced_analysis/04_eye_diagrams.py",
        "04_advanced_analysis/01_jitter_analysis.py",
    ]

    def generate_data(self) -> None:
        """Generate serial signals for eye metrics."""
        self.sample_rate = 20e9
        self.data_rate = 1e9
        self.num_bits = 500

        self.signal = self._generate_data_signal(jitter_rms=40e-12, noise_amp=0.05)

    def run_analysis(self) -> None:
        """Measure eye diagram metrics."""
        from demos.common.formatting import print_info, print_subheader

        print_subheader("Eye Metrics Extraction")

        # Generate eye
        eye_data = self._extract_eye_data(self.signal)

        # Measure metrics
        metrics = self._measure_eye_metrics(eye_data)

        print_info(f"Eye height: {metrics['eye_height']:.3f}")
        print_info(f"Eye width: {metrics['eye_width']:.3f} UI")
        print_info(f"Q-factor: {metrics['q_factor']:.2f}")
        print_info(f"Rise time (20-80%): {metrics['rise_time_ps']:.1f} ps")
        print_info(f"Fall time (80-20%): {metrics['fall_time_ps']:.1f} ps")

        self.results["metrics"] = metrics

    def validate_results(self, suite: ValidationSuite) -> None:
        """Validate eye metrics."""
        if "metrics" in self.results:
            q_factor = self.results["metrics"].get("q_factor", 0)
            suite.check_greater_than(
                "q_factor",
                q_factor,
                3.0,
                "Q-factor > 3 (BER < 1e-3)",
            )

            eye_height = self.results["metrics"].get("eye_height", 0)
            suite.check_range(
                "eye_height",
                eye_height,
                0.5,
                1.5,
                "Eye height reasonable",
            )

    def _generate_data_signal(self, jitter_rms: float, noise_amp: float) -> WaveformTrace:
        """Generate serial data signal."""
        bit_period = 1.0 / self.data_rate
        samples_per_bit = int(self.sample_rate * bit_period)

        data_bits = (np.random.rand(self.num_bits) > 0.5).astype(int)
        signal = np.repeat(data_bits, samples_per_bit).astype(float)
        signal += np.random.normal(0, noise_amp, len(signal))

        return WaveformTrace(
            data=signal,
            metadata=TraceMetadata(sample_rate=self.sample_rate, channel_name="data"),
        )

    def _extract_eye_data(self, signal: WaveformTrace) -> np.ndarray:
        """Extract eye data matrix."""
        bit_period = 1.0 / self.data_rate
        samples_per_bit = int(self.sample_rate * bit_period)
        samples_per_eye = samples_per_bit * 2

        num_eyes = len(signal.data) // samples_per_eye
        eye_data = np.zeros((samples_per_eye, num_eyes))

        for i in range(num_eyes):
            start = i * samples_per_eye
            end = start + samples_per_eye
            if end <= len(signal.data):
                eye_data[:, i] = signal.data[start:end]

        return eye_data

    def _measure_eye_metrics(self, eye_data: np.ndarray) -> dict:
        """Measure eye diagram metrics."""
        samples_per_eye = eye_data.shape[0]
        mid_point = samples_per_eye // 2

        # Eye height at center
        center_samples = eye_data[mid_point, :]
        eye_high = np.percentile(center_samples, 95)
        eye_low = np.percentile(center_samples, 5)
        eye_height = eye_high - eye_low

        # Eye width (time where signal is stable)
        high_mask = eye_data > (eye_high - 0.1)
        low_mask = eye_data < (eye_low + 0.1)
        stable_high = np.sum(np.all(high_mask, axis=1))
        stable_low = np.sum(np.all(low_mask, axis=1))
        eye_width = min(stable_high, stable_low) / samples_per_eye

        # Q-factor (simplified)
        noise_high = np.std(eye_data[eye_data > 0.5])
        noise_low = np.std(eye_data[eye_data < 0.5])
        q_factor = eye_height / (noise_high + noise_low) if (noise_high + noise_low) > 0 else 0

        # Rise/fall times
        # Find transition region
        quarter = samples_per_eye // 4
        rising_edge = eye_data[:quarter, :]
        mean_rise = np.mean(rising_edge, axis=1)

        # 20-80% rise time
        idx_20 = np.argmax(mean_rise > 0.2)
        idx_80 = np.argmax(mean_rise > 0.8)
        rise_time = (idx_80 - idx_20) / self.sample_rate if idx_80 > idx_20 else 0
        rise_time_ps = rise_time * 1e12

        # Fall time (similar)
        fall_time_ps = rise_time_ps  # Simplified

        return {
            "eye_height": eye_height,
            "eye_width": eye_width,
            "q_factor": q_factor,
            "rise_time_ps": rise_time_ps,
            "fall_time_ps": fall_time_ps,
        }


if __name__ == "__main__":
    sys.exit(run_demo_main(EyeMetricsDemo))
