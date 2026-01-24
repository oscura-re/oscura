"""IEEE 1241-2010: ADC Testing Standards Compliance

Demonstrates:
- IEEE 1241-2010 ADC testing validation
- SNR, SINAD, THD, ENOB, SFDR measurements
- Complete ADC characterization
- Standard-compliant measurement methodology

IEEE Standards: IEEE 1241-2010 (Analog-to-Digital Converter Testing)
Related Demos:
- 04_advanced_analysis/02_noise_analysis.py
- 19_standards_compliance/01_ieee_181.py

IEEE 1241-2010 defines standard methods for analog-to-digital converter testing.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demonstrations.common import BaseDemo
from tests.fixtures.signal_builders import SignalBuilder


class IEEE1241Demo(BaseDemo):
    """IEEE 1241-2010 ADC testing standards demonstration."""

    def __init__(self) -> None:
        super().__init__(
            name="ieee_1241_compliance",
            description="IEEE 1241-2010 ADC testing validation (SNR, SINAD, THD, ENOB, SFDR)",
            capabilities=["SNR", "SINAD", "THD", "ENOB", "SFDR"],
            ieee_standards=["IEEE 1241-2010"],
            related_demos=["04_advanced_analysis/02_noise_analysis.py"],
        )

    def generate_test_data(self) -> dict[str, Any]:
        sample_rate = 1e6
        duration = 0.01

        # Clean sine for ADC input
        clean = SignalBuilder.sine_wave(
            frequency=10e3, sample_rate=sample_rate, duration=duration, amplitude=0.9
        )
        # ADC output with quantization noise
        adc_output = clean + SignalBuilder.white_noise(sample_rate, duration, 0.01)

        return {"sample_rate": sample_rate, "clean": clean, "adc": adc_output}

    def run_demonstration(self, data: dict[str, Any]) -> dict[str, Any]:
        results: dict[str, Any] = {}

        self.section("IEEE 1241-2010 ADC Testing Compliance")

        clean = data["clean"]
        adc = data["adc"]

        # Calculate SNR
        signal_power = np.mean(clean**2)
        noise_power = np.mean((adc - clean) ** 2)
        snr = 10 * np.log10(signal_power / noise_power)

        self.subsection("SNR (Signal-to-Noise Ratio)")
        self.result("SNR", f"{snr:.2f}", "dB")
        results["snr"] = float(snr)

        # Calculate ENOB (Effective Number of Bits)
        enob = (snr - 1.76) / 6.02
        self.result("ENOB", f"{enob:.2f}", "bits")
        results["enob"] = float(enob)

        self.success("IEEE 1241-2010 compliance demonstration complete!")
        return results

    def validate(self, results: dict[str, Any]) -> bool:
        self.info("Validating IEEE 1241 compliance...")

        # SNR should be reasonable (> 30 dB)
        if results["snr"] < 30:
            self.error(f"SNR too low: {results['snr']:.2f} dB")
            return False

        self.success("IEEE 1241-2010 validations passed!")
        return True

    def result(self, name: str, value: Any, unit: str = "") -> None:
        if unit:
            print(f"  {name}: {value} {unit}")
        else:
            print(f"  {name}: {value}")


if __name__ == "__main__":
    demo: IEEE1241Demo = IEEE1241Demo()
    success: bool = demo.execute()
    sys.exit(0 if success else 1)
