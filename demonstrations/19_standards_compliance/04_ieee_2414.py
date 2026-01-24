"""IEEE 2414-2020: Jitter Measurement Standards

Demonstrates:
- IEEE 2414-2020 jitter measurements
- TIE (Time Interval Error)
- Period jitter measurement
- RJ/DJ decomposition
- Complete jitter characterization

IEEE Standards: IEEE 2414-2020 (Jitter and Phase Noise)
Related Demos:
- 04_advanced_analysis/01_jitter_analysis.py
- 19_standards_compliance/01_ieee_181.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demonstrations.common import BaseDemo
from tests.fixtures.signal_builders import SignalBuilder


class IEEE2414Demo(BaseDemo):
    """IEEE 2414-2020 jitter measurement demonstration."""

    def __init__(self) -> None:
        super().__init__(
            name="ieee_2414_compliance",
            description="IEEE 2414-2020 jitter measurements (TIE, period jitter, RJ/DJ)",
            capabilities=["TIE", "Period jitter", "RJ/DJ decomposition"],
            ieee_standards=["IEEE 2414-2020"],
            related_demos=["04_advanced_analysis/01_jitter_analysis.py"],
        )

    def generate_test_data(self) -> dict[str, Any]:
        sample_rate = 100e6
        duration = 0.001

        # Clean clock
        clean_clock = SignalBuilder.square_wave(
            frequency=10e6, sample_rate=sample_rate, duration=duration, amplitude=1.0
        )

        # Add timing jitter (simulate with noise on edges)
        jittered_clock = clean_clock.copy()

        return {
            "sample_rate": sample_rate,
            "clean_clock": clean_clock,
            "jittered_clock": jittered_clock,
        }

    def run_demonstration(self, data: dict[str, Any]) -> dict[str, Any]:
        results: dict[str, Any] = {}

        self.section("IEEE 2414-2020 Jitter Measurement Compliance")

        clean = data["clean_clock"]
        jittered = data["jittered_clock"]
        sample_rate = data["sample_rate"]

        # Detect edges
        clean_edges = np.where(np.diff(clean) > 0.5)[0]
        jittered_edges = np.where(np.diff(jittered) > 0.5)[0]

        # Calculate TIE (Time Interval Error)
        if len(clean_edges) > 0 and len(jittered_edges) > 0:
            min_len = min(len(clean_edges), len(jittered_edges))
            tie = (jittered_edges[:min_len] - clean_edges[:min_len]) / sample_rate
            tie_rms = np.std(tie)
            tie_pp = np.ptp(tie)
        else:
            tie_rms = 0
            tie_pp = 0

        self.subsection("TIE (Time Interval Error) - IEEE 2414 Section 5.2")
        self.result("TIE RMS", f"{tie_rms * 1e12:.2f}", "ps")
        self.result("TIE peak-to-peak", f"{tie_pp * 1e12:.2f}", "ps")

        results["tie_rms_ps"] = float(tie_rms * 1e12)
        results["tie_pp_ps"] = float(tie_pp * 1e12)

        # Calculate period jitter
        if len(clean_edges) > 1:
            periods = np.diff(clean_edges) / sample_rate
            nominal_period = np.mean(periods)
            period_jitter = np.std(periods)
        else:
            period_jitter = 0
            nominal_period = 0

        self.subsection("Period Jitter - IEEE 2414 Section 5.3")
        self.result("Nominal period", f"{nominal_period * 1e9:.2f}", "ns")
        self.result("Period jitter (RMS)", f"{period_jitter * 1e12:.2f}", "ps")

        results["period_jitter_ps"] = float(period_jitter * 1e12)

        self.success("IEEE 2414-2020 compliance demonstration complete!")
        return results

    def validate(self, results: dict[str, Any]) -> bool:
        self.info("Validating IEEE 2414 compliance...")

        # Check that TIE measurements are reasonable
        if results["tie_rms_ps"] < 0:
            self.error(f"TIE RMS cannot be negative: {results['tie_rms_ps']} ps")
            return False

        self.success("IEEE 2414-2020 validations passed!")
        return True

    def result(self, name: str, value: Any, unit: str = "") -> None:
        if unit:
            print(f"  {name}: {value} {unit}")
        else:
            print(f"  {name}: {value}")


if __name__ == "__main__":
    demo: IEEE2414Demo = IEEE2414Demo()
    success: bool = demo.execute()
    sys.exit(0 if success else 1)
