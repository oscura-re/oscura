"""Cross-Correlation Analysis: Signal correlation and time delay

Demonstrates:
- Cross-correlation calculation
- Time delay estimation
- Phase relationship analysis
- Signal similarity measurement
- Propagation delay measurement

IEEE Standards: IEEE 181-2011
Related Demos:
- 02_basic_analysis/01_waveform_measurements.py
- 04_advanced_analysis/11_statistics_advanced.py

Cross-correlation for finding time delays and signal relationships.
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


class CorrelationDemo(BaseDemo):
    """Cross-correlation analysis demonstration."""

    name = "Cross-Correlation Analysis"
    description = "Signal correlation and propagation delay measurement"
    category = "advanced_analysis"
    capabilities: ClassVar[list[str]] = [
        "Cross-correlation calculation",
        "Time delay estimation",
        "Phase relationship",
        "Signal similarity",
    ]
    ieee_standards: ClassVar[list[str]] = ["IEEE 181-2011"]
    related_demos: ClassVar[list[str]] = ["02_basic_analysis/01_waveform_measurements.py"]

    def generate_data(self) -> None:
        """Generate signal pairs for correlation."""
        self.sample_rate = 10e6
        duration = 0.001
        t = np.arange(0, duration, 1 / self.sample_rate)

        # Reference signal
        ref = np.sin(2 * np.pi * 1000 * t)

        # Delayed signal
        delay_samples = 50
        delayed = np.roll(ref, delay_samples)

        # Noisy correlated signal
        noisy = ref + 0.2 * np.random.randn(len(ref))

        # Uncorrelated signal
        uncorrelated = np.random.randn(len(ref))

        self.ref_signal = WaveformTrace(
            data=ref,
            metadata=TraceMetadata(sample_rate=self.sample_rate, channel_name="reference"),
        )
        self.delayed_signal = WaveformTrace(
            data=delayed,
            metadata=TraceMetadata(sample_rate=self.sample_rate, channel_name="delayed"),
        )
        self.noisy_signal = WaveformTrace(
            data=noisy,
            metadata=TraceMetadata(sample_rate=self.sample_rate, channel_name="noisy"),
        )
        self.uncorrelated_signal = WaveformTrace(
            data=uncorrelated,
            metadata=TraceMetadata(sample_rate=self.sample_rate, channel_name="uncorrelated"),
        )

        self.expected_delay_us = delay_samples / self.sample_rate * 1e6

    def run_analysis(self) -> None:
        """Perform correlation analysis."""
        from demos.common.formatting import print_subheader

        print_subheader("Delayed Signal Correlation")
        self.results["delayed"] = self._correlate_signals(
            self.ref_signal,
            self.delayed_signal,
            "Delayed",
        )

        print_subheader("Noisy Signal Correlation")
        self.results["noisy"] = self._correlate_signals(
            self.ref_signal,
            self.noisy_signal,
            "Noisy",
        )

        print_subheader("Uncorrelated Signal")
        self.results["uncorrelated"] = self._correlate_signals(
            self.ref_signal,
            self.uncorrelated_signal,
            "Uncorrelated",
        )

    def validate_results(self, suite: ValidationSuite) -> None:
        """Validate correlation results."""
        if "delayed" in self.results:
            delay_us = self.results["delayed"]["delay_us"]
            suite.check_approximately(
                "delay_estimation",
                delay_us,
                self.expected_delay_us,
                1.0,
                "Delay estimated accurately",
            )

    def _correlate_signals(
        self,
        sig1: WaveformTrace,
        sig2: WaveformTrace,
        label: str,
    ) -> dict:
        """Correlate two signals."""
        from demos.common.formatting import print_info

        # Normalize signals
        s1 = (sig1.data - np.mean(sig1.data)) / np.std(sig1.data)
        s2 = (sig2.data - np.mean(sig2.data)) / np.std(sig2.data)

        # Cross-correlation
        correlation = np.correlate(s1, s2, mode="same")
        correlation = correlation / len(s1)

        # Find peak
        peak_idx = np.argmax(np.abs(correlation))
        peak_value = correlation[peak_idx]

        # Calculate delay
        center_idx = len(correlation) // 2
        delay_samples = peak_idx - center_idx
        delay_us = delay_samples / sig1.metadata.sample_rate * 1e6

        print_info(f"Peak correlation: {peak_value:.3f}")
        print_info(f"Delay: {delay_us:.2f} µs")

        return {
            "correlation": correlation,
            "peak_value": peak_value,
            "delay_us": delay_us,
        }


if __name__ == "__main__":
    sys.exit(run_demo_main(CorrelationDemo))
