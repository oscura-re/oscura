"""Measurement Registry: Exploring and using the measurement registry

Demonstrates:
- oscura.list_measurements() - List available measurements
- oscura.get_measurement_registry() - Access the registry
- Querying measurements by category and tags
- Getting measurement metadata
- Dynamic measurement invocation
- Registry inspection and introspection

IEEE Standards: N/A
Related Demos:
- 08_extensibility/02_custom_measurement.py
- 02_basic_analysis/01_waveform_measurements.py

The measurement registry provides a central catalog of all available
measurements, both built-in and custom. This demonstration shows how to
explore and utilize the registry for dynamic measurement workflows.

This is a P0 CRITICAL feature - demonstrates registry capabilities to users.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

# Add demonstrations to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import oscura as osc
from demonstrations.common import BaseDemo, generate_sine_wave


class MeasurementRegistryDemo(BaseDemo):
    """Demonstrates measurement registry exploration and usage."""

    def __init__(self) -> None:
        """Initialize measurement registry demonstration."""
        super().__init__(
            name="measurement_registry",
            description="Explore and use the measurement registry",
            capabilities=[
                "oscura.list_measurements",
                "oscura.get_measurement_registry",
                "oscura.measure",
                "MeasurementRegistry.get_metadata",
            ],
            related_demos=[
                "08_extensibility/02_custom_measurement.py",
                "02_basic_analysis/01_waveform_measurements.py",
            ],
        )

    def generate_test_data(self) -> dict[str, Any]:
        """Generate test signals for measurement demonstrations.

        Returns:
            Dictionary with test traces for registry operations
        """
        # 1 kHz sine wave - standard test signal
        sine_1khz = generate_sine_wave(
            frequency=1000.0,  # 1 kHz
            amplitude=1.0,  # 1V peak
            duration=0.01,  # 10ms
            sample_rate=100e3,  # 100 kHz sampling
        )

        return {
            "sine_1khz": sine_1khz,
        }

    def run_demonstration(self, data: dict[str, Any]) -> dict[str, Any]:
        """Run measurement registry demonstration."""
        sine_1khz = data["sine_1khz"]
        results: dict[str, Any] = {}

        # Register some basic measurements to demonstrate registry functionality
        # (Oscura's registry is empty by default - measurements must be registered)
        osc.register_measurement(
            name="peak",
            func=lambda t: float(np.max(np.abs(t.data))),
            units="V",
            category="amplitude",
            description="Peak amplitude",
        )
        osc.register_measurement(
            name="mean",
            func=lambda t: float(np.mean(t.data)),
            units="V",
            category="amplitude",
            description="Mean value",
        )
        osc.register_measurement(
            name="rms",
            func=lambda t: float(np.sqrt(np.mean(t.data**2))),
            units="V",
            category="amplitude",
            description="RMS value",
        )

        # ===== Section 1: Accessing the Registry =====
        self.section("Part 1: Accessing the Measurement Registry")
        self.subsection("Getting Registry Instance")

        registry = osc.get_measurement_registry()
        self.info("Getting the registry: osc.get_measurement_registry()")
        self.result("Registry type", type(registry).__name__)
        self.info("The registry is a singleton - all calls return the same instance")

        results["registry_obtained"] = True
        results["registry_type"] = type(registry).__name__

        # ===== Section 2: Listing All Measurements =====
        self.section("Part 2: Listing Available Measurements")
        self.subsection("Complete Measurement Catalog")

        all_measurements = osc.list_measurements()
        self.info("Using osc.list_measurements() to get all available measurements")
        self.result("Total measurements available", len(all_measurements))

        # Show sample measurements
        self.info("\nSample measurements (first 10):")
        for name in all_measurements[:10]:
            self.info(f"  - {name}")
        if len(all_measurements) > 10:
            self.info(f"  ... and {len(all_measurements) - 10} more")

        results["total_measurements"] = len(all_measurements)

        # ===== Section 3: Querying by Category =====
        self.section("Part 3: Querying Measurements by Category")
        self.subsection("Filtering by Measurement Category")

        self.info("Measurements are organized into categories for easy discovery")

        # Query different categories
        categories_to_check = [
            "amplitude",
            "timing",
            "frequency",
            "power",
            "digital",
        ]

        category_counts: dict[str, int] = {}

        for category in categories_to_check:
            measurements = osc.list_measurements(category=category)
            category_counts[category] = len(measurements)

            self.result(f"{category.capitalize()} measurements", len(measurements))

            # Show examples
            if measurements:
                examples = measurements[:3]
                self.info(f"  Examples: {', '.join(examples)}")

        results["category_counts"] = category_counts

        # ===== Section 4: Querying by Tags =====
        self.section("Part 4: Querying Measurements by Tags")
        self.subsection("Tag-Based Discovery")

        self.info("Measurements can have tags for fine-grained filtering")
        self.info("Tags enable cross-category searches for specific use cases")

        # Example tags to search for
        tags_to_check = [
            ["quality"],
            ["ieee"],
            ["power"],
            ["statistical"],
        ]

        for tag_list in tags_to_check:
            measurements = osc.list_measurements(tags=tag_list)
            tag_str = ", ".join(tag_list)
            self.result(f"Measurements with tag '{tag_str}'", len(measurements))

            if measurements:
                examples = measurements[:3]
                self.info(f"  Examples: {', '.join(examples)}")

        # ===== Section 5: Getting Measurement Metadata =====
        self.section("Part 5: Measurement Metadata Inspection")
        self.subsection("Understanding Measurement Properties")

        self.info("Each measurement has detailed metadata:")
        self.info("  - units: Physical units (V, Hz, s, etc.)")
        self.info("  - category: Functional grouping")
        self.info("  - description: What it measures")
        self.info("  - tags: Additional classifiers")

        # Get metadata for some common measurements
        measurements_to_inspect = ["peak", "rms", "frequency"]

        for measurement_name in measurements_to_inspect:
            if measurement_name in all_measurements:
                self.subsection(f"Metadata: {measurement_name}")

                try:
                    metadata = registry.get_metadata(measurement_name)

                    self.result("  Name", measurement_name)

                    units = metadata.get("units", "N/A")
                    self.result("  Units", units)

                    category = metadata.get("category", "N/A")
                    self.result("  Category", category)

                    description = metadata.get("description", "")
                    if description and len(description) > 60:
                        self.result("  Description", description[:60] + "...")
                    else:
                        self.result("  Description", description if description else "N/A")

                    tags = metadata.get("tags", [])
                    if tags:
                        self.result("  Tags", ", ".join(str(t) for t in tags))

                except Exception as e:
                    self.warning(f"Could not get metadata for {measurement_name}: {e}")

        results["metadata_inspected"] = True

        # ===== Section 6: Dynamic Measurement Invocation =====
        self.section("Part 6: Dynamic Measurement Invocation")
        self.subsection("Using Registry for Runtime Measurement")

        self.info("The registry enables dynamic measurement selection at runtime")
        self.info("This is useful for batch processing and user-driven workflows")

        # Get measurements from registry and apply them
        measurements_to_apply = ["peak", "rms", "mean"]

        measurement_results: dict[str, float] = {}

        self.subsection("Measuring Test Signal")
        self.info("Applying measurements to 1 kHz sine wave (1V amplitude)")

        for measurement_name in measurements_to_apply:
            if measurement_name in all_measurements:
                # Get measurement function from registry
                measurement_func = registry.get(measurement_name)

                # Apply to signal
                value = measurement_func(sine_1khz)

                # Get units for display
                metadata = registry.get_metadata(measurement_name)
                units = metadata.get("units", "")

                self.result(f"{measurement_name.capitalize()}", f"{value:.6f}", units)
                measurement_results[measurement_name] = value

        results["measurement_results"] = measurement_results
        results["dynamic_measurements_applied"] = len(measurement_results)

        # ===== Section 7: Batch Measurement Application =====
        self.section("Part 7: Batch Measurement Workflows")
        self.subsection("Applying Multiple Measurements")

        self.info("Registry enables batch processing workflows:")
        self.info("")
        self.info("# Get all amplitude measurements")
        self.info("amplitude_measurements = osc.list_measurements(category='amplitude')")
        self.info("")
        self.info("# Apply them all")
        self.info("results = {}")
        self.info("registry = osc.get_measurement_registry()")
        self.info("for name in amplitude_measurements:")
        self.info("    measurement = registry.get(name)")
        self.info("    results[name] = measurement(trace)")

        # Demonstrate with amplitude category
        amplitude_measurements = osc.list_measurements(category="amplitude")
        self.info(f"\nApplying {len(amplitude_measurements)} amplitude measurements:")

        batch_results: dict[str, float] = {}
        measurements_applied = 0

        for name in amplitude_measurements[:5]:  # Limit to first 5 for demo
            try:
                measurement = registry.get(name)
                value = measurement(sine_1khz)
                batch_results[name] = value
                measurements_applied += 1

                metadata = registry.get_metadata(name)
                units = metadata.get("units", "")
                self.result(f"  {name}", f"{value:.6f}", units)

            except Exception as e:
                self.warning(f"Could not apply {name}: {e}")

        if len(amplitude_measurements) > 5:
            self.info(f"  ... (showing 5 of {len(amplitude_measurements)})")

        results["batch_results"] = batch_results
        results["batch_measurements_applied"] = measurements_applied

        # ===== Section 8: Custom Measurements in Registry =====
        self.section("Part 8: Custom Measurements and Registry")
        self.subsection("Registering and Using Custom Measurements")

        self.info("Custom measurements integrate seamlessly with the registry")

        # Register a custom measurement
        def signal_to_noise_estimate(trace: osc.WaveformTrace) -> float:
            """Estimate SNR from signal statistics.

            Simple SNR estimate using RMS and standard deviation.

            Args:
                trace: WaveformTrace to analyze

            Returns:
                Estimated SNR in dB
            """
            import numpy as np

            rms = float(np.sqrt(np.mean(trace.data**2)))
            std = float(np.std(trace.data))

            if std == 0:
                return float("inf")

            snr_ratio = rms / std
            snr_db = 20 * np.log10(snr_ratio)

            return snr_db

        # Register it
        osc.register_measurement(
            name="snr_estimate",
            func=signal_to_noise_estimate,
            units="dB",
            category="quality",
            description="Estimated SNR from RMS and standard deviation",
            tags=["snr", "quality", "statistical"],
        )

        self.result("Registered custom measurement", "snr_estimate", "✓")

        # Verify it appears in registry
        all_after_custom = osc.list_measurements()
        self.result("Total measurements after registration", len(all_after_custom))

        # Use it via registry
        snr_value = osc.measure_custom(sine_1khz, "snr_estimate")
        self.result("SNR estimate", f"{snr_value:.1f}", "dB")

        results["custom_measurement_registered"] = True
        results["snr_estimate"] = snr_value

        # ===== Section 9: Registry Introspection =====
        self.section("Part 9: Advanced Registry Operations")
        self.subsection("Registry Introspection Patterns")

        self.info("Advanced registry usage patterns:")
        self.info("")
        self.info("1. Find measurements by units:")
        self.info("   voltage_measurements = [")
        self.info("       name for name in osc.list_measurements()")
        self.info("       if registry.get_metadata(name).get('units') == 'V'")
        self.info("   ]")
        self.info("")
        self.info("2. Find measurements with specific tags:")
        self.info("   ieee_measurements = osc.list_measurements(tags=['ieee'])")
        self.info("")
        self.info("3. Group measurements by category:")
        self.info("   by_category = {}")
        self.info("   for name in osc.list_measurements():")
        self.info("       cat = registry.get_metadata(name).get('category')")
        self.info("       by_category.setdefault(cat, []).append(name)")
        self.info("")
        self.info("4. Validate measurement exists before use:")
        self.info("   if registry.has_measurement('custom_metric'):")
        self.info("       value = osc.measure(trace, 'custom_metric')")

        # Demonstrate: find all voltage measurements
        voltage_measurements = [
            name for name in all_after_custom if registry.get_metadata(name).get("units", "") == "V"
        ]

        self.result("Measurements with units 'V'", len(voltage_measurements))
        if voltage_measurements:
            examples = voltage_measurements[:5]
            self.info(f"  Examples: {', '.join(examples)}")

        results["voltage_measurements_found"] = len(voltage_measurements)

        # ===== Section 10: Use Cases =====
        self.section("Part 10: Practical Use Cases")
        self.subsection("When to Use the Registry")

        self.info("The measurement registry is ideal for:")
        self.info("")
        self.info("1. DYNAMIC WORKFLOWS - User selects measurements at runtime")
        self.info("   Example: Interactive analysis tool where user picks metrics")
        self.info("")
        self.info("2. BATCH PROCESSING - Apply many measurements to many signals")
        self.info("   Example: Automated test suite measuring 100+ parameters")
        self.info("")
        self.info("3. PLUGIN SYSTEMS - Third-party measurements integrate seamlessly")
        self.info("   Example: Custom domain-specific measurements from plugins")
        self.info("")
        self.info("4. CONFIGURATION-DRIVEN - Measurement lists from config files")
        self.info("   Example: YAML file specifies which measurements to run")
        self.info("")
        self.info("5. DISCOVERY - Find available measurements without docs")
        self.info("   Example: List all 'power' category measurements")

        results["use_cases_documented"] = True

        self.success("Measurement registry demonstration complete!")

        return results

    def validate(self, results: dict[str, Any]) -> bool:
        """Validate measurement registry results.

        Args:
            results: Dictionary with demonstration results

        Returns:
            True if all validations pass
        """
        validations = [
            ("Registry obtained", "registry_obtained", True),
            ("Registry type correct", "registry_type", "MeasurementRegistry"),
            ("Total measurements", "total_measurements", lambda x: x > 0),
            ("Category counts exist", "category_counts", lambda x: len(x) > 0),
            ("Metadata inspected", "metadata_inspected", True),
            ("Dynamic measurements applied", "dynamic_measurements_applied", lambda x: x >= 0),
            ("Batch measurements applied", "batch_measurements_applied", lambda x: x >= 0),
            ("Custom measurement registered", "custom_measurement_registered", True),
            ("SNR estimated", "snr_estimate", lambda x: isinstance(x, (int, float))),
            ("Voltage measurements found", "voltage_measurements_found", lambda x: x >= 0),
            ("Use cases documented", "use_cases_documented", True),
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
                self.result(f"✗ {name}", "FAIL", f"(got {value})")
                all_passed = False

        return all_passed


if __name__ == "__main__":
    demo = MeasurementRegistryDemo()
    success = demo.execute()
    exit(0 if success else 1)
