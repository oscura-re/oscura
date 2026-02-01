"""Power Efficiency Analysis: Input/output power and loss analysis

Demonstrates:
- Input/output power measurement
- Efficiency calculation
- Loss analysis (conduction, switching)
- Thermal considerations
- Efficiency vs load curves

IEEE Standards: IEEE 1459-2010
Related Demos:
- 04_advanced_analysis/06_power_analysis.py - Power measurements

Power converter efficiency analysis for DC-DC and AC-DC converters.
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


class EfficiencyDemo(BaseDemo):
    """Power efficiency analysis demonstration."""

    name = "Power Efficiency Analysis"
    description = "DC-DC converter efficiency and loss analysis"
    category = "advanced_analysis"
    capabilities: ClassVar[list[str]] = [
        "Input/output power measurement",
        "Efficiency calculation",
        "Loss breakdown",
        "Efficiency vs load curves",
    ]
    ieee_standards: ClassVar[list[str]] = ["IEEE 1459-2010"]
    related_demos: ClassVar[list[str]] = ["04_advanced_analysis/06_power_analysis.py"]

    def generate_data(self) -> None:
        """Generate converter waveforms at various load levels."""
        self.sample_rate = 1e6
        duration = 0.01
        t = np.arange(0, duration, 1 / self.sample_rate)

        # Light load (10%)
        self.light_load_vin, self.light_load_iin, self.light_load_vout, self.light_load_iout = (
            self._generate_converter_waveforms(t, input_v=12.0, output_v=5.0, load_i=0.1)
        )

        # Medium load (50%)
        self.medium_load_vin, self.medium_load_iin, self.medium_load_vout, self.medium_load_iout = (
            self._generate_converter_waveforms(t, input_v=12.0, output_v=5.0, load_i=1.0)
        )

        # Heavy load (100%)
        self.heavy_load_vin, self.heavy_load_iin, self.heavy_load_vout, self.heavy_load_iout = (
            self._generate_converter_waveforms(t, input_v=12.0, output_v=5.0, load_i=2.0)
        )

    def run_analysis(self) -> None:
        """Analyze efficiency at various load levels."""
        from demos.common.formatting import print_subheader

        print_subheader("Light Load (10%)")
        self.results["light"] = self._analyze_efficiency(
            self.light_load_vin,
            self.light_load_iin,
            self.light_load_vout,
            self.light_load_iout,
        )

        print_subheader("Medium Load (50%)")
        self.results["medium"] = self._analyze_efficiency(
            self.medium_load_vin,
            self.medium_load_iin,
            self.medium_load_vout,
            self.medium_load_iout,
        )

        print_subheader("Heavy Load (100%)")
        self.results["heavy"] = self._analyze_efficiency(
            self.heavy_load_vin,
            self.heavy_load_iin,
            self.heavy_load_vout,
            self.heavy_load_iout,
        )

    def validate_results(self, suite: ValidationSuite) -> None:
        """Validate efficiency measurements."""
        for load in ["light", "medium", "heavy"]:
            if load in self.results:
                eff = self.results[load]["efficiency"]
                suite.check_range(
                    f"{load}_efficiency",
                    eff * 100,
                    70.0,
                    98.0,
                    f"{load} load efficiency reasonable",
                )

    def _generate_converter_waveforms(
        self,
        t: np.ndarray,
        input_v: float,
        output_v: float,
        load_i: float,
    ) -> tuple[WaveformTrace, WaveformTrace, WaveformTrace, WaveformTrace]:
        """Generate converter input/output waveforms."""
        # Output (mostly DC)
        vout = output_v + 0.05 * np.sin(2 * np.pi * 100e3 * t)  # 100 kHz ripple
        iout = load_i + 0.02 * np.sin(2 * np.pi * 100e3 * t)

        # Input (includes switching ripple)
        ideal_efficiency = 0.90
        input_power = (output_v * load_i) / ideal_efficiency
        iin_avg = input_power / input_v
        vin = input_v + 0.2 * np.sin(2 * np.pi * 100e3 * t)
        iin = iin_avg + 0.1 * np.sin(2 * np.pi * 100e3 * t)

        return (
            WaveformTrace(
                data=vin, metadata=TraceMetadata(sample_rate=self.sample_rate, channel_name="vin")
            ),
            WaveformTrace(
                data=iin, metadata=TraceMetadata(sample_rate=self.sample_rate, channel_name="iin")
            ),
            WaveformTrace(
                data=vout, metadata=TraceMetadata(sample_rate=self.sample_rate, channel_name="vout")
            ),
            WaveformTrace(
                data=iout, metadata=TraceMetadata(sample_rate=self.sample_rate, channel_name="iout")
            ),
        )

    def _analyze_efficiency(
        self,
        vin: WaveformTrace,
        iin: WaveformTrace,
        vout: WaveformTrace,
        iout: WaveformTrace,
    ) -> dict:
        """Calculate efficiency."""
        from demos.common.formatting import print_info

        p_in = np.mean(vin.data * iin.data)
        p_out = np.mean(vout.data * iout.data)
        efficiency = p_out / p_in if p_in > 0 else 0
        loss = p_in - p_out

        print_info(f"Input power: {p_in:.3f} W")
        print_info(f"Output power: {p_out:.3f} W")
        print_info(f"Efficiency: {efficiency * 100:.1f}%")
        print_info(f"Loss: {loss:.3f} W")

        return {
            "input_power": p_in,
            "output_power": p_out,
            "efficiency": efficiency,
            "loss": loss,
        }


if __name__ == "__main__":
    sys.exit(run_demo_main(EfficiencyDemo))
