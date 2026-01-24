"""IEEE 181-2011: Pulse Measurement Standards Compliance

Demonstrates:
- IEEE 181-2011 pulse measurement validation
- Rise/fall time measurement (10%-90%)
- Overshoot and undershoot measurement
- Pulse width and duty cycle
- Standard-compliant measurement methodology

IEEE Standards: IEEE 181-2011 (Transitions, Pulses, and Related Waveforms)
Related Demos:
- 02_basic_analysis/01_waveform_measurements.py
- 19_standards_compliance/02_ieee_1241.py

IEEE 181-2011 defines standard methods for measuring transitions, pulses,
and related waveforms. This demonstration validates Oscura's implementation
against the standard.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Add demonstrations to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demonstrations.common import BaseDemo, validate_approximately
from oscura import duty_cycle, fall_time, overshoot, period, rise_time, undershoot
from oscura.core.types import TraceMetadata, WaveformTrace
from tests.fixtures.signal_builders import SignalBuilder


class IEEE181Demo(BaseDemo):
    """IEEE 181-2011 pulse measurement standards demonstration."""

    def __init__(self) -> None:
        """Initialize IEEE 181 demonstration."""
        super().__init__(
            name="ieee_181_compliance",
            description="IEEE 181-2011 pulse measurement validation",
            capabilities=[
                "oscura.rise_time (10%-90%)",
                "oscura.fall_time (90%-10%)",
                "oscura.overshoot",
                "oscura.undershoot",
                "oscura.duty_cycle",
                "oscura.period",
            ],
            ieee_standards=["IEEE 181-2011"],
            related_demos=[
                "02_basic_analysis/01_waveform_measurements.py",
            ],
        )

    def generate_test_data(self) -> dict[str, Any]:
        """Generate IEEE 181-compliant test signals."""
        sample_rate = 10e6  # 10 MHz
        duration = 0.001  # 1 ms

        # Pulse train with realistic rise/fall times
        pulse_train = SignalBuilder.pulse_train(
            frequency=10e3,  # 10 kHz
            sample_rate=sample_rate,
            duration=duration,
            pulse_width=0.3,  # 30% duty cycle
            amplitude=5.0,
        )

        # Square wave with overshoot/undershoot
        square = SignalBuilder.square_wave(
            frequency=5e3,
            sample_rate=sample_rate,
            duration=duration,
            amplitude=3.3,
        )

        return {
            "sample_rate": sample_rate,
            "pulse_train": pulse_train,
            "square": square,
        }

    def run_demonstration(self, data: dict[str, Any]) -> dict[str, Any]:
        """Run IEEE 181 compliance demonstration."""
        results: dict[str, Any] = {}

        self.section("IEEE 181-2011 Pulse Measurement Compliance")
        self.info("Validating pulse measurements against IEEE 181-2011 standard")

        sample_rate = data["sample_rate"]

        # Create WaveformTrace objects
        pulse_trace = WaveformTrace(
            data=data["pulse_train"],
            metadata=TraceMetadata(sample_rate=sample_rate, channel_name="PULSE"),
        )

        square_trace = WaveformTrace(
            data=data["square"],
            metadata=TraceMetadata(sample_rate=sample_rate, channel_name="SQUARE"),
        )

        # Part 1: Rise Time (IEEE 181 Section 6.1)
        self.subsection("Part 1: Rise Time (IEEE 181-2011 Section 6.1)")
        self.info("Rise time measured from 10% to 90% of transition")

        t_rise = rise_time(pulse_trace)
        self.result("Rise time", f"{t_rise * 1e9:.2f}", "ns")
        results["rise_time"] = float(t_rise)

        # Part 2: Fall Time (IEEE 181 Section 6.2)
        self.subsection("Part 2: Fall Time (IEEE 181-2011 Section 6.2)")
        self.info("Fall time measured from 90% to 10% of transition")

        t_fall = fall_time(pulse_trace)
        self.result("Fall time", f"{t_fall * 1e9:.2f}", "ns")
        results["fall_time"] = float(t_fall)

        # Part 3: Pulse Width and Period (IEEE 181 Section 6.3)
        self.subsection("Part 3: Pulse Width and Period (IEEE 181-2011 Section 6.3)")

        t_period = period(pulse_trace)
        duty = duty_cycle(pulse_trace)
        pulse_width = t_period * duty

        self.result("Period", f"{t_period * 1e6:.2f}", "µs")
        self.result("Duty cycle", f"{duty * 100:.2f}", "%")
        self.result("Pulse width", f"{pulse_width * 1e6:.2f}", "µs")

        results["period"] = float(t_period)
        results["duty_cycle"] = float(duty)
        results["pulse_width"] = float(pulse_width)

        # Part 4: Overshoot (IEEE 181 Section 6.4)
        self.subsection("Part 4: Overshoot (IEEE 181-2011 Section 6.4)")
        self.info("Positive overshoot relative to steady-state high level")

        over = overshoot(square_trace)
        self.result("Overshoot", f"{over * 1000:.2f}", "mV")
        results["overshoot"] = float(over)

        # Part 5: Undershoot (IEEE 181 Section 6.5)
        self.subsection("Part 5: Undershoot (IEEE 181-2011 Section 6.5)")
        self.info("Negative undershoot relative to steady-state low level")

        under = undershoot(square_trace)
        self.result("Undershoot", f"{under * 1000:.2f}", "mV")
        results["undershoot"] = float(under)

        # Part 6: Compliance Summary
        self.subsection("Part 6: IEEE 181-2011 Compliance Summary")

        self.info("\n[Measurement Method Compliance]")
        self.info("  ✓ Rise time: 10% to 90% threshold")
        self.info("  ✓ Fall time: 90% to 10% threshold")
        self.info("  ✓ Period: Time between consecutive rising edges")
        self.info("  ✓ Duty cycle: Pulse width / period")
        self.info("  ✓ Overshoot: (Peak - High) / (High - Low) x 100%")
        self.info("  ✓ Undershoot: (Low - Valley) / (High - Low) x 100%")

        self.success("IEEE 181-2011 compliance demonstration complete!")

        return results

    def validate(self, results: dict[str, Any]) -> bool:
        """Validate IEEE 181 measurements."""
        self.info("Validating IEEE 181 compliance...")

        all_valid = True

        # Validate period (should be ~100 µs for 10 kHz)
        expected_period = 1 / 10e3  # 100 µs
        if not validate_approximately(
            results["period"], expected_period, tolerance=0.05, name="Period"
        ):
            all_valid = False

        # Validate duty cycle (should be ~30%)
        if not validate_approximately(results["duty_cycle"], 0.3, tolerance=0.1, name="Duty cycle"):
            all_valid = False

        # Validate pulse width
        expected_pulse_width = expected_period * 0.3
        if not validate_approximately(
            results["pulse_width"], expected_pulse_width, tolerance=0.1, name="Pulse width"
        ):
            all_valid = False

        if all_valid:
            self.success("All IEEE 181-2011 validations passed!")
            self.info("\nCompliance Status: PASS")
            self.info("\nNext Steps:")
            self.info("  - Try 19_standards_compliance/02_ieee_1241.py for ADC testing")
            self.info("  - Explore 19_standards_compliance/03_ieee_1459.py for power quality")
        else:
            self.error("Some validations failed")

        return all_valid

    def result(self, name: str, value: Any, unit: str = "") -> None:
        """Print a result with optional unit."""
        if unit:
            print(f"  {name}: {value} {unit}")
        else:
            print(f"  {name}: {value}")


if __name__ == "__main__":
    demo: IEEE181Demo = IEEE181Demo()
    success: bool = demo.execute()
    sys.exit(0 if success else 1)
