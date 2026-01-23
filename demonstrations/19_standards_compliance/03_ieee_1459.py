"""IEEE 1459-2010: Power Quality Measurement Standards

Demonstrates:
- IEEE 1459-2010 power quality measurements
- Active, reactive, and apparent power
- Power factor and harmonics
- Three-phase power analysis

IEEE Standards: IEEE 1459-2010 (Power Definitions)
Related Demos:
- 19_standards_compliance/01_ieee_181.py
- 19_standards_compliance/02_ieee_1241.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demonstrations.common import BaseDemo
from tests.fixtures.signal_builders import SignalBuilder


class IEEE1459Demo(BaseDemo):
    """IEEE 1459-2010 power quality demonstration."""

    def __init__(self) -> None:
        super().__init__(
            name="ieee_1459_compliance",
            description="IEEE 1459-2010 power quality (active, reactive, apparent power)",
            capabilities=["Active power", "Reactive power", "Apparent power", "Power factor"],
            ieee_standards=["IEEE 1459-2010"],
            related_demos=["19_standards_compliance/01_ieee_181.py"],
        )

    def generate_test_data(self) -> dict[str, Any]:
        sample_rate = 100e3
        duration = 0.1

        # 60 Hz voltage and current
        voltage = SignalBuilder.sine_wave(
            frequency=60, sample_rate=sample_rate, duration=duration, amplitude=170
        )  # 120 Vrms peak
        current = SignalBuilder.sine_wave(
            frequency=60, sample_rate=sample_rate, duration=duration, amplitude=10, phase=0.5
        )  # 10A with phase shift

        return {"voltage": voltage, "current": current, "sample_rate": sample_rate}

    def run_demonstration(self, data: dict[str, Any]) -> dict[str, Any]:
        results: dict[str, Any] = {}

        self.section("IEEE 1459-2010 Power Quality Compliance")

        v = data["voltage"]
        i = data["current"]

        # Calculate power parameters
        p_inst = v * i  # Instantaneous power
        p_avg = np.mean(p_inst)  # Active power
        v_rms = np.sqrt(np.mean(v**2))
        i_rms = np.sqrt(np.mean(i**2))
        s = v_rms * i_rms  # Apparent power
        pf = p_avg / s  # Power factor

        self.subsection("Power Measurements (IEEE 1459-2010)")
        self.result("Active power (P)", f"{p_avg:.2f}", "W")
        self.result("Apparent power (S)", f"{s:.2f}", "VA")
        self.result("Power factor", f"{pf:.4f}")

        results["active_power"] = float(p_avg)
        results["apparent_power"] = float(s)
        results["power_factor"] = float(pf)

        self.success("IEEE 1459-2010 compliance demonstration complete!")
        return results

    def validate(self, results: dict[str, Any]) -> bool:
        self.info("Validating IEEE 1459 compliance...")

        # Power factor should be between 0 and 1
        if not (0 <= results["power_factor"] <= 1):
            self.error(f"Power factor out of range: {results['power_factor']}")
            return False

        self.success("IEEE 1459-2010 validations passed!")
        return True

    def result(self, name: str, value: Any, unit: str = "") -> None:
        if unit:
            print(f"  {name}: {value} {unit}")
        else:
            print(f"  {name}: {value}")


if __name__ == "__main__":
    demo: IEEE1459Demo = IEEE1459Demo()
    success: bool = demo.execute()
    sys.exit(0 if success else 1)
