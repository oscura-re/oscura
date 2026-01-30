"""Power Analysis: DC/AC power and harmonics per IEEE 1459-2010

Demonstrates:
- DC power calculation
- AC power (RMS)
- Power factor measurement
- Harmonic power analysis
- IEEE 1459-2010 compliance

IEEE Standards: IEEE 1459-2010 (Power definitions)
Related Demos:
- 04_advanced_analysis/07_efficiency.py - Efficiency analysis
- 02_basic_analysis/02_spectral_analysis.py - Frequency domain

DC and AC power measurements for power supply characterization.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np

from demos.common.base_demo import BaseDemo, run_demo_main
from demos.common.validation import ValidationSuite
from oscura.core.types import TraceMetadata, WaveformTrace


class PowerAnalysisDemo(BaseDemo):
    """Power analysis demonstration."""

    name = "Power Analysis (DC/AC/Harmonics)"
    description = "IEEE 1459-2010 compliant power measurements"
    category = "advanced_analysis"
    capabilities = [
        "DC power calculation",
        "AC RMS power",
        "Power factor",
        "Harmonic analysis",
        "THD measurement",
    ]
    ieee_standards = ["IEEE 1459-2010"]
    related_demos = ["04_advanced_analysis/07_efficiency.py"]

    def generate_data(self) -> None:
        """Generate voltage/current waveforms."""
        self.sample_rate = 1e6
        duration = 0.1
        t = np.arange(0, duration, 1 / self.sample_rate)

        # DC power supply
        self.v_dc = WaveformTrace(
            data=np.full_like(t, 5.0),
            metadata=TraceMetadata(sample_rate=self.sample_rate, channel_name="v_dc"),
        )
        self.i_dc = WaveformTrace(
            data=np.full_like(t, 2.0),
            metadata=TraceMetadata(sample_rate=self.sample_rate, channel_name="i_dc"),
        )

        # AC power (60 Hz)
        self.v_ac = WaveformTrace(
            data=170 * np.sin(2 * np.pi * 60 * t),
            metadata=TraceMetadata(sample_rate=self.sample_rate, channel_name="v_ac"),
        )
        self.i_ac = WaveformTrace(
            data=10 * np.sin(2 * np.pi * 60 * t - np.pi / 4),  # 45° phase shift
            metadata=TraceMetadata(sample_rate=self.sample_rate, channel_name="i_ac"),
        )

        # AC with harmonics
        v_fundamental = 170 * np.sin(2 * np.pi * 60 * t)
        v_3rd = 34 * np.sin(2 * np.pi * 180 * t)  # 3rd harmonic
        v_5th = 17 * np.sin(2 * np.pi * 300 * t)  # 5th harmonic
        self.v_harmonics = WaveformTrace(
            data=v_fundamental + v_3rd + v_5th,
            metadata=TraceMetadata(sample_rate=self.sample_rate, channel_name="v_harmonics"),
        )
        self.i_harmonics = WaveformTrace(
            data=10 * np.sin(2 * np.pi * 60 * t),
            metadata=TraceMetadata(sample_rate=self.sample_rate, channel_name="i_harmonics"),
        )

    def run_analysis(self) -> None:
        """Perform power analysis."""
        from demos.common.formatting import print_subheader

        print_subheader("DC Power")
        self.results["dc"] = self._analyze_dc_power(self.v_dc, self.i_dc)

        print_subheader("AC Power (60 Hz)")
        self.results["ac"] = self._analyze_ac_power(self.v_ac, self.i_ac)

        print_subheader("AC with Harmonics")
        self.results["harmonics"] = self._analyze_ac_power(self.v_harmonics, self.i_harmonics)

    def validate_results(self, suite: ValidationSuite) -> None:
        """Validate power measurements."""
        if "dc" in self.results:
            suite.check_approximately(
                "dc_power",
                self.results["dc"]["power"],
                10.0,
                0.1,
                "DC power = 5V * 2A = 10W",
            )

        if "ac" in self.results:
            suite.check_range(
                "power_factor",
                self.results["ac"]["power_factor"],
                0.6,
                0.8,
                "Power factor ~0.707 (45° phase)",
            )

    def _analyze_dc_power(self, voltage: WaveformTrace, current: WaveformTrace) -> dict:
        """Analyze DC power."""
        from demos.common.formatting import print_info

        v_avg = np.mean(voltage.data)
        i_avg = np.mean(current.data)
        power = v_avg * i_avg

        print_info(f"Voltage: {v_avg:.3f} V")
        print_info(f"Current: {i_avg:.3f} A")
        print_info(f"Power: {power:.3f} W")

        return {"voltage": v_avg, "current": i_avg, "power": power}

    def _analyze_ac_power(self, voltage: WaveformTrace, current: WaveformTrace) -> dict:
        """Analyze AC power."""
        from demos.common.formatting import print_info

        v_rms = np.sqrt(np.mean(voltage.data**2))
        i_rms = np.sqrt(np.mean(current.data**2))
        apparent_power = v_rms * i_rms

        real_power = np.mean(voltage.data * current.data)
        power_factor = real_power / apparent_power if apparent_power > 0 else 0

        # THD
        v_fft = np.fft.rfft(voltage.data)
        v_fundamental = abs(v_fft[int(60 * len(voltage.data) / voltage.metadata.sample_rate)])
        v_harmonics = np.sum(abs(v_fft) ** 2) - v_fundamental**2
        thd = np.sqrt(v_harmonics) / v_fundamental if v_fundamental > 0 else 0

        print_info(f"V_RMS: {v_rms:.1f} V")
        print_info(f"I_RMS: {i_rms:.2f} A")
        print_info(f"Apparent power: {apparent_power:.1f} VA")
        print_info(f"Real power: {real_power:.1f} W")
        print_info(f"Power factor: {power_factor:.3f}")
        print_info(f"THD: {thd * 100:.1f}%")

        return {
            "v_rms": v_rms,
            "i_rms": i_rms,
            "apparent_power": apparent_power,
            "real_power": real_power,
            "power_factor": power_factor,
            "thd": thd,
        }


if __name__ == "__main__":
    sys.exit(run_demo_main(PowerAnalysisDemo))
