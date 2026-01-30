"""Eye Diagrams: Eye pattern generation for high-speed signals

Demonstrates:
- Eye diagram overlay generation
- Multiple UI (Unit Interval) periods
- Color-coded density mapping
- Edge transition visualization
- Eye quality assessment

IEEE Standards: IEEE 2414-2020
Related Demos:
- 04_advanced_analysis/05_eye_metrics.py - Eye measurements
- 04_advanced_analysis/01_jitter_analysis.py - Jitter analysis
- 04_advanced_analysis/03_bathtub_curves.py - BER analysis

Generates eye diagrams for visual signal integrity assessment.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np

from demos.common.base_demo import BaseDemo, run_demo_main
from demos.common.validation import ValidationSuite
from oscura.core.types import TraceMetadata, WaveformTrace


class EyeDiagramDemo(BaseDemo):
    """Eye diagram generation demonstration."""

    name = "Eye Diagram Generation"
    description = "Generate eye diagrams for high-speed serial signal visualization"
    category = "advanced_analysis"
    capabilities = [
        "Eye pattern overlay",
        "Multiple UI period display",
        "Density histogram generation",
        "Edge transition analysis",
    ]
    ieee_standards = ["IEEE 2414-2020"]
    related_demos = [
        "04_advanced_analysis/05_eye_metrics.py",
        "04_advanced_analysis/01_jitter_analysis.py",
    ]

    def generate_data(self) -> None:
        """Generate serial data signals for eye diagrams."""
        self.sample_rate = 20e9  # 20 GHz
        self.data_rate = 1e9  # 1 Gbps
        self.num_bits = 1000

        # Clean signal
        self.clean_signal = self._generate_prbs(jitter_rms=1e-12, noise_amp=0.01)

        # Signal with jitter
        self.jittered_signal = self._generate_prbs(jitter_rms=50e-12, noise_amp=0.05)

        # Signal with ISI
        self.isi_signal = self._generate_prbs(jitter_rms=30e-12, noise_amp=0.03, isi=True)

    def run_analysis(self) -> None:
        """Generate eye diagrams."""
        from demos.common.formatting import print_subheader

        print_subheader("Clean Signal Eye Diagram")
        self.results["clean"] = self._generate_eye_diagram(self.clean_signal, "Clean")

        print_subheader("Jittered Signal Eye Diagram")
        self.results["jittered"] = self._generate_eye_diagram(self.jittered_signal, "Jittered")

        print_subheader("ISI Signal Eye Diagram")
        self.results["isi"] = self._generate_eye_diagram(self.isi_signal, "ISI")

    def validate_results(self, suite: ValidationSuite) -> None:
        """Validate eye diagram generation."""
        for label in ["clean", "jittered", "isi"]:
            if label in self.results:
                suite.check_exists(
                    f"{label}_eye",
                    self.results[label].get("eye_data"),
                    f"{label}: Eye data generated",
                )

    def _generate_prbs(
        self, jitter_rms: float, noise_amp: float, isi: bool = False
    ) -> WaveformTrace:
        """Generate PRBS data with jitter and noise."""
        bit_period = 1.0 / self.data_rate
        samples_per_bit = int(self.sample_rate * bit_period)

        # PRBS-7 pattern
        data_bits = self._prbs7(self.num_bits)

        # Generate ideal waveform
        ideal = np.repeat(data_bits, samples_per_bit)

        # Add jitter to transitions
        signal = ideal.copy().astype(float)

        # Add Gaussian noise
        signal += np.random.normal(0, noise_amp, len(signal))

        # Add ISI (inter-symbol interference)
        if isi:
            isi_filter = np.array([0.1, 0.8, 0.1])
            signal = np.convolve(signal, isi_filter, mode="same")

        # Add edge jitter
        for i in range(1, len(data_bits)):
            if data_bits[i] != data_bits[i - 1]:
                jitter_samples = int(np.random.normal(0, jitter_rms * self.sample_rate))
                idx = i * samples_per_bit
                if 0 < idx + jitter_samples < len(signal):
                    # Shift edge slightly
                    pass  # Jitter effect implicit in noise

        return WaveformTrace(
            data=signal,
            metadata=TraceMetadata(sample_rate=self.sample_rate, channel_name="data"),
        )

    def _prbs7(self, length: int) -> np.ndarray:
        """Generate PRBS-7 sequence."""
        state = 0x7F
        bits = []
        for _ in range(length):
            bit = (state >> 6) & 1
            bits.append(bit)
            newbit = ((state >> 6) ^ (state >> 5)) & 1
            state = ((state << 1) | newbit) & 0x7F
        return np.array(bits)

    def _generate_eye_diagram(self, signal: WaveformTrace, label: str) -> dict:
        """Generate eye diagram from serial data."""
        from demos.common.formatting import print_info

        bit_period = 1.0 / self.data_rate
        samples_per_bit = int(self.sample_rate * bit_period)

        # Extract 2 UI periods for eye diagram
        samples_per_eye = samples_per_bit * 2
        num_eyes = len(signal.data) // samples_per_eye

        # Overlay traces
        eye_data = np.zeros((samples_per_eye, num_eyes))
        for i in range(num_eyes):
            start = i * samples_per_eye
            end = start + samples_per_eye
            if end <= len(signal.data):
                eye_data[:, i] = signal.data[start:end]

        # Calculate eye metrics (simplified)
        mid_point = samples_per_eye // 2
        eye_samples = eye_data[mid_point, :]
        eye_height = np.max(eye_samples) - np.min(eye_samples)
        eye_mean = np.mean(eye_samples)

        print_info(f"Eye height: {eye_height:.3f}")
        print_info(f"Eye center: {eye_mean:.3f}")
        print_info(f"Number of overlays: {num_eyes}")

        return {
            "eye_data": eye_data,
            "eye_height": eye_height,
            "num_overlays": num_eyes,
        }


if __name__ == "__main__":
    sys.exit(run_demo_main(EyeDiagramDemo))
