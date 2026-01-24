"""Plugin Development: Complete plugin lifecycle from creation to deployment

Demonstrates:
- Plugin class structure and registration
- Creating custom loaders, analyzers, and decoders
- Plugin metadata and documentation
- Plugin discovery and loading
- Testing and validation

IEEE Standards: N/A
Related Demos:
- 08_extensibility/01_plugin_basics.py
- 08_extensibility/06_plugin_templates.py

This demonstrates the full lifecycle of plugin development, showing how to
create, register, and test custom plugins that extend Oscura's functionality.

This is a P0 CRITICAL feature - demonstrates plugin development to users.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

# Add demonstrations to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np

import oscura as osc
from demonstrations.common import BaseDemo
from oscura.analyzers.protocols.base import ChannelDef, OptionDef, ProtocolDecoder
from oscura.core.types import DigitalTrace, ProtocolPacket, TraceMetadata

if TYPE_CHECKING:
    from collections.abc import Iterator

    from numpy.typing import NDArray

    from oscura.core.types import WaveformTrace


class PluginDevelopmentDemo(BaseDemo):
    """Demonstrates complete plugin development lifecycle."""

    def __init__(self) -> None:
        """Initialize plugin development demonstration."""
        super().__init__(
            name="plugin_development",
            description="Complete plugin lifecycle from development to deployment",
            capabilities=[
                "oscura.ProtocolDecoder",
                "oscura.register_measurement",
                "oscura.load_plugin",
                "Plugin development patterns",
            ],
            related_demos=[
                "08_extensibility/01_plugin_basics.py",
                "08_extensibility/06_plugin_templates.py",
            ],
        )

    def generate_test_data(self) -> dict[str, Any]:
        """Generate test data for plugin demonstrations.

        Returns:
            Dictionary with test signals for plugin validation
        """
        # Generate a simple digital signal for decoder testing
        sample_rate = 1e6  # 1 MHz
        duration = 0.001  # 1ms
        num_samples = int(sample_rate * duration)

        # Create a simple digital pulse pattern
        data = np.zeros(num_samples, dtype=bool)
        # Add some pulses
        data[100:200] = True
        data[300:350] = True
        data[500:600] = True

        trace_metadata = TraceMetadata(
            sample_rate=sample_rate,
            channel_name="plugin_test",
        )
        digital_trace = DigitalTrace(
            data=data,
            metadata=trace_metadata,
        )

        # Generate analog signal for loader testing
        t = np.linspace(0, duration, num_samples)
        analog_signal = np.sin(2 * np.pi * 1000 * t)  # 1 kHz sine

        return {
            "digital_trace": digital_trace,
            "analog_data": analog_signal,
            "sample_rate": sample_rate,
            "duration": duration,
        }

    def run_demonstration(self, data: dict[str, Any]) -> dict[str, Any]:
        """Run plugin development demonstration."""
        results: dict[str, Any] = {}

        # ===== Section 1: Understanding Plugin Structure =====
        self.section("Part 1: Plugin Architecture")
        self.subsection("Core Plugin Components")

        self.info("Every Oscura plugin consists of:")
        self.info("  1. Plugin Class - Inherits from appropriate base")
        self.info("     - ProtocolDecoder for protocol decoders")
        self.info("     - Measurement functions for custom measurements")
        self.info("     - Custom classes for analyzers/loaders")
        self.info("")
        self.info("  2. Metadata - Describes plugin capabilities")
        self.info("     - name: Unique identifier")
        self.info("     - version: Semantic version")
        self.info("     - author: Plugin author")
        self.info("     - description: Functionality description")
        self.info("")
        self.info("  3. Entry Point - Registration mechanism")
        self.info("     - Allows plugin discovery")
        self.info("     - Enables dynamic loading")
        self.info("")
        self.info("  4. Tests - Validation suite")
        self.info("     - Unit tests for functionality")
        self.info("     - Integration tests for workflows")

        results["architecture_documented"] = True

        # ===== Section 2: Example Plugin - Custom Decoder =====
        self.section("Part 2: Example - Custom Protocol Decoder")
        self.subsection("Creating a Simple Pulse Width Decoder")

        self.info("We'll create a decoder that detects pulse widths")
        self.info("This is useful for custom protocols or PWM analysis")

        # Define the custom decoder - inherits from ProtocolDecoder (not AsyncDecoder)
        # since it doesn't need baudrate-based timing
        class PulseWidthDecoder(ProtocolDecoder):
            """Custom decoder that measures pulse widths.

            This decoder identifies digital pulses and reports their widths.
            Useful for PWM analysis or custom pulse-width encoded protocols.
            """

            # Decoder identification
            id = "pulse_width"
            name = "Pulse Width"
            longname = "Pulse Width Measurement Decoder"
            desc = "Decodes pulses by measuring their widths"
            license = "MIT"

            # Channel definitions
            channels = [ChannelDef(id="data", name="Data", desc="Digital signal to analyze")]  # noqa: RUF012

            # Options
            options = [  # noqa: RUF012
                OptionDef(
                    id="min_width",
                    name="Minimum Width",
                    desc="Minimum pulse width in seconds",
                    default=1e-6,
                ),
                OptionDef(
                    id="idle_state",
                    name="Idle State",
                    desc="Idle state of signal (0 or 1)",
                    default=0,
                    values=[0, 1],
                ),
            ]

            def decode(
                self,
                trace: DigitalTrace,
                **channels: NDArray[np.bool_],
            ) -> Iterator[ProtocolPacket]:
                """Decode pulse widths from digital trace.

                Args:
                    trace: Digital trace to analyze
                    **channels: Additional channels (unused)

                Yields:
                    ProtocolPacket for each detected pulse
                """
                min_width = self.get_option("min_width")
                idle_state = self.get_option("idle_state")

                data = trace.data
                sample_rate = trace.metadata.sample_rate

                # Find transitions
                if idle_state == 0:
                    # Rising edges mark pulse start
                    rising = np.where(~data[:-1] & data[1:])[0]
                    falling = np.where(data[:-1] & ~data[1:])[0]
                else:
                    # Falling edges mark pulse start
                    rising = np.where(data[:-1] & ~data[1:])[0]
                    falling = np.where(~data[:-1] & data[1:])[0]

                # Match rising with falling edges
                for start_idx in rising:
                    # Find next falling edge
                    next_falling = falling[falling > start_idx]
                    if len(next_falling) == 0:
                        continue

                    end_idx = next_falling[0]
                    width_samples = end_idx - start_idx
                    width_seconds = width_samples / sample_rate

                    # Filter by minimum width
                    if width_seconds < min_width:
                        continue

                    # Create packet with pulse width info
                    timestamp = start_idx / sample_rate
                    pulse_data = np.packbits(data[start_idx : end_idx + 1]).tobytes()

                    yield ProtocolPacket(
                        timestamp=timestamp,
                        protocol=self.id,
                        data=pulse_data,
                        annotations={
                            "width_seconds": width_seconds,
                            "width_samples": int(width_samples),
                            "start_index": int(start_idx),
                            "end_index": int(end_idx),
                        },
                        errors=[],
                    )

        self.info("Decoder class defined")
        self.result("Decoder ID", PulseWidthDecoder.id)
        self.result("Decoder name", PulseWidthDecoder.name)
        self.result("Required channels", len(PulseWidthDecoder.channels))
        self.result("Configurable options", len(PulseWidthDecoder.options))

        results["decoder_defined"] = True
        results["decoder_id"] = PulseWidthDecoder.id

        # ===== Section 3: Testing the Plugin =====
        self.section("Part 3: Plugin Testing")
        self.subsection("Validating Decoder Functionality")

        digital_trace = data["digital_trace"]

        # Instantiate decoder with options
        decoder = PulseWidthDecoder(min_width=50e-6)  # 50 microseconds minimum
        self.info(f"Created decoder instance with min_width={decoder.get_option('min_width')}")

        # Decode the test signal
        packets = list(decoder.decode(digital_trace))
        self.result("Pulses detected", len(packets))

        results["packets_decoded"] = len(packets)

        if packets:
            self.subsection("Pulse Details")
            for i, packet in enumerate(packets[:3], 1):  # Show first 3
                self.info(f"Pulse {i}:")
                self.result("  Timestamp", f"{packet.timestamp * 1e6:.1f} us")
                width_us = packet.annotations.get("width_seconds", 0) * 1e6
                self.result("  Width", f"{width_us:.1f} us")
                self.result("  Samples", packet.annotations.get("width_samples", 0))

            if len(packets) > 3:
                self.info(f"  ... and {len(packets) - 3} more pulses")

            results["first_pulse_width"] = packets[0].annotations.get("width_seconds", 0)

        # ===== Section 4: Custom Measurement Plugin =====
        self.section("Part 4: Example - Custom Measurement")
        self.subsection("Creating a Custom Measurement Function")

        self.info("Custom measurements extend analysis capabilities")
        self.info("They integrate seamlessly with the measurement registry")

        def calculate_pulse_duty_cycle(trace: WaveformTrace) -> float:
            """Calculate duty cycle of a digital signal.

            Duty cycle = (time high) / (total time)

            Args:
                trace: WaveformTrace (should be digital/boolean)

            Returns:
                Duty cycle as a ratio (0.0 to 1.0)
            """
            data = (
                trace.data.astype(bool) if not np.issubdtype(trace.data.dtype, bool) else trace.data
            )
            high_count = np.sum(data)
            total_count = len(data)

            if total_count == 0:
                return 0.0

            return float(high_count / total_count)

        self.info("Measurement function defined: calculate_pulse_duty_cycle")

        # Register the measurement
        osc.register_measurement(
            name="pulse_duty_cycle",
            func=calculate_pulse_duty_cycle,
            units="ratio",
            category="digital",
            description="Duty cycle of digital signal (time high / total time)",
            tags=["pwm", "digital", "timing"],
        )

        self.result("Registered", "pulse_duty_cycle")
        results["measurement_registered"] = True

        # Test the custom measurement
        self.subsection("Testing Custom Measurement")

        # Convert digital trace to WaveformTrace for measurement
        waveform = osc.WaveformTrace(
            data=digital_trace.data.astype(float),
            metadata=digital_trace.metadata,
        )

        duty_cycle = osc.measure_custom(waveform, "pulse_duty_cycle")
        self.result("Duty cycle", f"{duty_cycle:.3f}", "(ratio)")
        self.result("Duty cycle percentage", f"{duty_cycle * 100:.1f}%")

        results["duty_cycle"] = duty_cycle

        # ===== Section 5: Plugin Documentation =====
        self.section("Part 5: Plugin Documentation Best Practices")
        self.subsection("Essential Documentation")

        self.info("Good plugin documentation includes:")
        self.info("")
        self.info("1. Docstring - What the plugin does")
        self.info('   """Decodes XYZ protocol from digital traces."""')
        self.info("")
        self.info("2. Parameters - All options and defaults")
        self.info("   options = [")
        self.info('       OptionDef("baudrate", "Baud Rate", default=9600),')
        self.info("   ]")
        self.info("")
        self.info("3. Examples - Usage patterns")
        self.info("   Example:")
        self.info("       >>> decoder = MyDecoder(baudrate=115200)")
        self.info("       >>> packets = list(decoder.decode(trace))")
        self.info("")
        self.info("4. Error Handling - Expected exceptions")
        self.info("   Raises:")
        self.info("       ValueError: If baudrate is invalid")
        self.info("")
        self.info("5. References - Related standards/protocols")
        self.info("   References:")
        self.info("       - Protocol spec: https://...")
        self.info("       - IEEE Standard XXX")

        results["documentation_guidelines"] = True

        # ===== Section 6: Plugin Registration Patterns =====
        self.section("Part 6: Plugin Registration Patterns")
        self.subsection("Entry Point Registration")

        self.info("Plugins use Python entry points for discovery:")
        self.info("")
        self.info("In pyproject.toml:")
        self.info("[project.entry-points.'oscura.decoders']")
        self.info('pulse_width = "my_plugin.decoders:PulseWidthDecoder"')
        self.info("")
        self.info("[project.entry-points.'oscura.measurements']")
        self.info('duty_cycle = "my_plugin.measurements:calculate_duty_cycle"')
        self.info("")
        self.info("This enables:")
        self.info("  - Automatic discovery via osc.list_plugins()")
        self.info("  - Dynamic loading via osc.load_plugin()")
        self.info("  - Lazy initialization (loaded only when needed)")

        results["registration_documented"] = True

        # ===== Section 7: Plugin Testing Strategy =====
        self.section("Part 7: Plugin Testing Strategy")
        self.subsection("Recommended Test Structure")

        self.info("Comprehensive testing ensures reliability:")
        self.info("")
        self.info("1. Unit Tests - Test individual methods")
        self.info("   - Test option validation")
        self.info("   - Test edge cases")
        self.info("   - Test error handling")
        self.info("")
        self.info("2. Integration Tests - Test with real data")
        self.info("   - Test with known good signals")
        self.info("   - Test with corrupt data")
        self.info("   - Test with edge conditions")
        self.info("")
        self.info("3. Performance Tests - Ensure efficiency")
        self.info("   - Test with large datasets")
        self.info("   - Measure decode time")
        self.info("   - Check memory usage")
        self.info("")
        self.info("4. Validation Tests - Verify correctness")
        self.info("   - Compare against reference decoder")
        self.info("   - Validate output format")
        self.info("   - Check packet completeness")

        self.subsection("Example Test Pattern")
        self.info("def test_pulse_width_decoder():")
        self.info("    # Arrange")
        self.info("    decoder = PulseWidthDecoder(min_width=1e-6)")
        self.info("    trace = generate_test_signal()")
        self.info("")
        self.info("    # Act")
        self.info("    packets = list(decoder.decode(trace))")
        self.info("")
        self.info("    # Assert")
        self.info("    assert len(packets) > 0")
        self.info("    assert all(p.protocol == 'pulse_width' for p in packets)")
        self.info("    assert all('width_seconds' in p.annotations for p in packets)")

        results["testing_strategy_documented"] = True

        # ===== Section 8: Plugin Packaging =====
        self.section("Part 8: Plugin Packaging and Distribution")
        self.subsection("Packaging Best Practices")

        self.info("To distribute your plugin:")
        self.info("")
        self.info("1. Project Structure:")
        self.info("   oscura-plugin-myfeature/")
        self.info("   |-- pyproject.toml")
        self.info("   |-- README.md")
        self.info("   |-- src/")
        self.info("   |   +-- oscura_plugin_myfeature/")
        self.info("   |       |-- __init__.py")
        self.info("   |       +-- decoder.py")
        self.info("   +-- tests/")
        self.info("       +-- test_decoder.py")
        self.info("")
        self.info("2. Dependencies:")
        self.info("   [project]")
        self.info('   dependencies = ["oscura>=0.5.0"]')
        self.info("")
        self.info("3. Entry Points:")
        self.info("   [project.entry-points.'oscura.decoders']")
        self.info('   myfeature = "oscura_plugin_myfeature:MyDecoder"')
        self.info("")
        self.info("4. Installation:")
        self.info("   pip install oscura-plugin-myfeature")
        self.info("")
        self.info("5. Verification:")
        self.info("   >>> import oscura as osc")
        self.info("   >>> 'myfeature' in osc.list_plugins()['oscura.decoders']")
        self.info("   True")

        results["packaging_documented"] = True

        self.success("Plugin development demonstration complete!")

        return results

    def validate(self, results: dict[str, Any]) -> bool:
        """Validate plugin development results.

        Args:
            results: Dictionary with demonstration results

        Returns:
            True if all validations pass
        """
        validations = [
            ("Architecture documented", "architecture_documented", True),
            ("Decoder defined", "decoder_defined", True),
            ("Decoder has ID", "decoder_id", lambda x: x == "pulse_width"),
            ("Packets decoded", "packets_decoded", lambda x: x > 0),
            ("Custom measurement registered", "measurement_registered", True),
            ("Duty cycle calculated", "duty_cycle", lambda x: 0.0 <= x <= 1.0),
            ("Documentation guidelines provided", "documentation_guidelines", True),
            ("Registration patterns documented", "registration_documented", True),
            ("Testing strategy documented", "testing_strategy_documented", True),
            ("Packaging documented", "packaging_documented", True),
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
                self.result(f"PASS: {name}", "OK")
            else:
                self.result(f"FAIL: {name}", f"(got {value})")
                all_passed = False

        return all_passed


if __name__ == "__main__":
    demo = PluginDevelopmentDemo()
    success = demo.execute()
    exit(0 if success else 1)
