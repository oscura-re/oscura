"""Custom Measurements: Extend Oscura with user-defined measurements

Demonstrates:
- oscura.register_measurement() - Register a custom measurement
- oscura.get_measurement_registry() - Access the measurement registry
- oscura.list_measurements() - List available measurements
- Creating custom measurement functions
- Using custom measurements like built-in measurements

IEEE Standards: N/A
Related Demos:
- 08_extensibility/01_custom_analyzer.py
- 02_basic_analysis/01_waveform_measurements.py

Custom measurements allow you to extend Oscura with domain-specific or
application-specific measurements. Register once, use everywhere.

This is a P0 CRITICAL feature - demonstrates extensibility to users.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

# Add demonstrations to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import oscura as osc
from demonstrations.common import BaseDemo, generate_sine_wave, validate_approximately

if TYPE_CHECKING:
    from oscura.core.types import WaveformTrace


class CustomMeasurementDemo(BaseDemo):
    """Demonstrates custom measurement registration and usage."""

    def __init__(self) -> None:
        """Initialize custom measurement demonstration."""
        super().__init__(
            name="custom_measurement",
            description="Create and register custom measurements",
            capabilities=[
                "oscura.register_measurement",
                "oscura.get_measurement_registry",
                "oscura.list_measurements",
                "oscura.MeasurementDefinition",
            ],
            related_demos=[
                "08_extensibility/01_custom_analyzer.py",
                "02_basic_analysis/01_waveform_measurements.py",
            ],
        )

    def generate_test_data(self) -> dict[str, WaveformTrace]:
        """Generate test signals for measurement demonstrations.

        Returns:
            Dictionary with test traces:
            - 'sine_1khz': 1 kHz sine wave at 1V amplitude
            - 'sine_2khz': 2 kHz sine wave at 0.5V amplitude
        """
        # 1 kHz sine wave - typical signal
        sine_1khz = generate_sine_wave(
            frequency=1000.0,  # 1 kHz
            amplitude=1.0,  # 1V peak
            duration=0.01,  # 10ms
            sample_rate=100e3,  # 100 kHz sampling
        )

        # 2 kHz sine wave with lower amplitude - for multi-signal analysis
        sine_2khz = generate_sine_wave(
            frequency=2000.0,  # 2 kHz
            amplitude=0.5,  # 0.5V peak
            duration=0.01,  # 10ms
            sample_rate=100e3,  # 100 kHz sampling
        )

        return {
            "sine_1khz": sine_1khz,
            "sine_2khz": sine_2khz,
        }

    def run_demonstration(
        self, data: dict[str, WaveformTrace]
    ) -> dict[str, float | int | bool | str]:
        """Run custom measurement demonstration."""
        sine_1khz = data["sine_1khz"]
        sine_2khz = data["sine_2khz"]
        results: dict[str, float | int | bool | str] = {}

        # ===== Section 1: Understanding the Measurement Registry =====
        self.section("Part 1: Measurement Registry")
        self.subsection("Initial State")

        # Get registry and list built-in measurements
        registry = osc.get_measurement_registry()
        self.info("Getting the measurement registry: osc.get_measurement_registry()")

        all_measurements = osc.list_measurements()
        self.result("Total built-in measurements", len(all_measurements))
        self.info(f"Examples: {', '.join(all_measurements[:5])}")

        # Show measurements by category
        amplitude_measurements = osc.list_measurements(category="amplitude")
        self.result("Amplitude measurements available", len(amplitude_measurements))
        self.info(f"Examples: {', '.join(amplitude_measurements)}")

        results["built_in_count"] = len(all_measurements)

        # ===== Section 2: Creating Custom Measurements =====
        self.section("Part 2: Creating Custom Measurements")
        self.subsection("Measurement 1: Crest Factor")
        self.info("Crest Factor = Peak Value / RMS Value")
        self.info(
            "Useful for: Signal quality analysis, clipping detection, power distribution analysis"
        )

        def calculate_crest_factor(trace: WaveformTrace) -> float:
            """Calculate crest factor (peak / RMS).

            Crest factor indicates the relationship between peak and
            average (RMS) values. High crest factor indicates high peaks
            relative to RMS, suggesting non-sinusoidal waveforms or clipping.

            Args:
                trace: WaveformTrace to analyze

            Returns:
                Crest factor (dimensionless ratio)
            """
            import numpy as np

            peak = float(np.abs(trace.data).max())
            rms = float(np.sqrt(np.mean(trace.data**2)))

            # Avoid division by zero
            if rms == 0:
                return 0.0

            return peak / rms

        # Register the custom measurement
        self.info("Registering measurement: osc.register_measurement(...)")
        osc.register_measurement(
            name="crest_factor",
            func=calculate_crest_factor,
            units="ratio",
            category="amplitude",
            description="Peak value / RMS value - measures signal peakiness",
            tags=["quality", "clipping", "power"],
        )
        self.result("Registered", "crest_factor", "✓")
        results["crest_factor_registered"] = True

        self.subsection("Measurement 2: Form Factor")
        self.info("Form Factor = RMS Value / Mean Absolute Value")
        self.info(
            "Useful for: Waveform shape analysis, signal characterization, AC vs DC identification"
        )

        def calculate_form_factor(trace: WaveformTrace) -> float:
            """Calculate form factor (RMS / MAV).

            Form factor characterizes waveform shape. For a pure sine wave,
            form factor is π/(2√2) ≈ 1.11. Non-sinusoidal waveforms have
            different values, enabling shape identification.

            Args:
                trace: WaveformTrace to analyze

            Returns:
                Form factor (dimensionless ratio)
            """
            import numpy as np

            rms = float(np.sqrt(np.mean(trace.data**2)))
            mav = float(np.mean(np.abs(trace.data)))

            # Avoid division by zero
            if mav == 0:
                return 0.0

            return rms / mav

        osc.register_measurement(
            name="form_factor",
            func=calculate_form_factor,
            units="ratio",
            category="amplitude",
            description="RMS value / Mean absolute value - characterizes waveform shape",
            tags=["shape", "characterization", "quality"],
        )
        self.result("Registered", "form_factor", "✓")
        results["form_factor_registered"] = True

        self.subsection("Measurement 3: Peak-to-RMS Ratio")

        def calculate_peak_to_rms(trace: WaveformTrace) -> float:
            """Calculate peak-to-RMS ratio.

            Related to crest factor but specifically uses peak magnitude
            without normalization. Useful for signal strength analysis.

            Args:
                trace: WaveformTrace to analyze

            Returns:
                Peak-to-RMS ratio
            """
            import numpy as np

            peak = float(np.abs(trace.data).max())
            rms = float(np.sqrt(np.mean(trace.data**2)))

            if rms == 0:
                return 0.0

            return peak / rms

        osc.register_measurement(
            name="peak_to_rms",
            func=calculate_peak_to_rms,
            units="ratio",
            category="amplitude",
            description="Peak magnitude / RMS - signal strength indicator",
            tags=["strength", "amplitude"],
        )
        self.result("Registered", "peak_to_rms", "✓")

        # ===== Section 3: Using Custom Measurements =====
        self.section("Part 3: Using Custom Measurements")
        self.subsection("Measuring the 1 kHz Sine Wave")

        # Calculate crest factor
        cf_1khz = registry.get("crest_factor")(sine_1khz)
        self.result("Crest factor (1 kHz sine)", f"{cf_1khz:.4f}", "(ratio)")

        # Expected crest factor for sine wave: sqrt(2) ≈ 1.414
        import math

        expected_cf = float(math.sqrt(2))
        validation = validate_approximately(cf_1khz, expected_cf, tolerance=0.1)
        self.info(
            f"Expected ~{expected_cf:.3f} (for ideal sine), "
            f"Validation: {'✓' if validation else '✗'}"
        )
        results["crest_factor_1khz"] = cf_1khz

        # Calculate form factor
        ff_1khz = registry.get("form_factor")(sine_1khz)
        self.result("Form factor (1 kHz sine)", f"{ff_1khz:.4f}", "(ratio)")

        # Expected form factor for sine: π/(2√2) ≈ 1.110
        expected_ff = float(math.pi / (2 * math.sqrt(2)))
        validation = validate_approximately(ff_1khz, expected_ff, tolerance=0.1)
        self.info(
            f"Expected ~{expected_ff:.3f} (for ideal sine), "
            f"Validation: {'✓' if validation else '✗'}"
        )
        results["form_factor_1khz"] = ff_1khz

        # Calculate peak-to-RMS
        ptrms_1khz = registry.get("peak_to_rms")(sine_1khz)
        self.result("Peak-to-RMS (1 kHz sine)", f"{ptrms_1khz:.4f}", "(ratio)")
        results["peak_to_rms_1khz"] = ptrms_1khz

        self.subsection("Comparing Multiple Signals")
        self.info("Comparing 1 kHz vs 2 kHz sine waves")

        # Calculate metrics for both signals
        cf_2khz = registry.get("crest_factor")(sine_2khz)
        ff_2khz = registry.get("form_factor")(sine_2khz)
        ptrms_2khz = registry.get("peak_to_rms")(sine_2khz)

        self.result("Crest factor (2 kHz)", f"{cf_2khz:.4f}", "ratio")
        self.result("Form factor (2 kHz)", f"{ff_2khz:.4f}", "ratio")
        self.result("Peak-to-RMS (2 kHz)", f"{ptrms_2khz:.4f}", "ratio")

        self.info("Both signals are sine waves, so metrics should be similar:")
        cf_diff = abs(cf_1khz - cf_2khz)
        ff_diff = abs(ff_1khz - ff_2khz)
        self.result("Crest factor difference", f"{cf_diff:.4f}", "(should be small)")
        self.result("Form factor difference", f"{ff_diff:.4f}", "(should be small)")

        results["cf_diff"] = cf_diff
        results["ff_diff"] = ff_diff

        # ===== Section 4: Registry Inspection =====
        self.section("Part 4: Registry Inspection")

        # List all measurements (should now include our custom ones)
        all_now = osc.list_measurements()
        self.result(
            "Total measurements after registration",
            len(all_now),
            f"(+{len(all_now) - len(all_measurements)} custom)",
        )

        # List amplitude category measurements
        amplitude_now = osc.list_measurements(category="amplitude")
        self.info(f"Amplitude category now has: {len(amplitude_now)} measurements")

        # List by tags
        quality_tagged = osc.list_measurements(tags=["quality"])
        self.result("Measurements tagged 'quality'", len(quality_tagged))
        self.info(f"They are: {', '.join(quality_tagged)}")

        # Get metadata for custom measurements
        self.subsection("Custom Measurement Metadata")
        cf_metadata = registry.get_metadata("crest_factor")
        self.info("crest_factor:")
        self.result("  Units", cf_metadata.get("units", "N/A"))
        self.result("  Category", cf_metadata.get("category", "N/A"))
        description = cf_metadata.get("description", "")
        if isinstance(description, str) and len(description) > 50:
            self.result("  Description", description[:50] + "...")
        else:
            self.result("  Description", str(description))
        tags = cf_metadata.get("tags", [])
        if isinstance(tags, list):
            self.result("  Tags", ", ".join(str(t) for t in tags))
        else:
            self.result("  Tags", str(tags))

        results["amplitude_category_count"] = len(amplitude_now)
        results["total_measurements_final"] = len(all_now)

        # ===== Section 5: Practical Example =====
        self.section("Part 5: Practical Example")
        self.subsection("Signal Quality Analysis")
        self.info("Using custom measurements for signal quality analysis")

        # For a clean sine wave, we expect specific ratios
        self.info("Quality metrics for clean sine waves:")
        self.result("  Expected crest factor", "~1.414", "(√2)")
        self.result("  Expected form factor", "~1.110", "(π/2√2)")

        self.info("Actual measurements:")
        self.result("  1 kHz crest factor", f"{cf_1khz:.4f}")
        self.result("  1 kHz form factor", f"{ff_1khz:.4f}")

        quality_check = cf_1khz > 1.3 and cf_1khz < 1.5
        self.result(
            "Signal quality", "✓ Clean sine wave" if quality_check else "✗ Possible distortion"
        )

        results["quality_check_passed"] = quality_check

        return results

    def validate(self, results: dict[str, float | int | bool | str]) -> bool:
        """Validate demonstration results.

        Args:
            results: Results dictionary from run_demonstration

        Returns:
            True if all validations passed
        """
        validations = [
            ("Crest factor registered", "crest_factor_registered", True),
            ("Form factor registered", "form_factor_registered", True),
            ("Custom measurements registered", "total_measurements_final", lambda x: x >= 3),
            ("1 kHz crest factor reasonable", "crest_factor_1khz", lambda x: 1.3 < x < 1.5),
            ("1 kHz form factor reasonable", "form_factor_1khz", lambda x: 1.0 < x < 1.2),
            ("Measurements are comparable", "cf_diff", lambda x: x < 0.05),
            (
                "Final count includes customs",
                "total_measurements_final",
                lambda x: x >= results.get("built_in_count", 0),
            ),
            ("Signal quality check passed", "quality_check_passed", True),
        ]

        all_passed = True
        for name, key, check in validations:
            if key not in results:
                self.error(f"Missing result key: {key}")
                all_passed = False
                continue

            value = results[key]

            if isinstance(check, bool):
                passed = value is True
            elif callable(check):
                try:
                    passed = check(value)
                except Exception as e:
                    self.error(f"Validation {name} raised exception: {e}")
                    passed = False
            else:
                passed = value == check

            if passed:
                self.result(f"✓ {name}", "PASS")
            else:
                self.result(f"✗ {name}", "FAIL")
                all_passed = False

        return all_passed


if __name__ == "__main__":
    demo = CustomMeasurementDemo()
    success = demo.execute()
    exit(0 if success else 1)
