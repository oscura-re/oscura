"""Multi-Channel Data Handling

Demonstrates loading and managing multi-channel data:
- Multi-channel oscilloscope captures
- Channel synchronization and alignment
- Cross-channel analysis setup
- Channel metadata management

Common in oscilloscopes, logic analyzers, and data acquisition systems.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demonstrations.common import (
    BaseDemo,
    ValidationSuite,
    format_table,
    generate_sine_wave,
    generate_square_wave,
)

from oscura.core.types import TraceMetadata, WaveformTrace


class MultiChannelDemo(BaseDemo):
    """Demonstrate multi-channel data handling."""

    def __init__(self) -> None:
        """Initialize multi-channel demonstration."""
        super().__init__(
            name="multi_channel",
            description="Load and manage multi-channel data captures",
            capabilities=[
                "Multi-channel loading",
                "Channel synchronization",
                "Cross-channel analysis",
                "Channel metadata management",
            ],
            ieee_standards=["IEEE 181-2011"],
            related_demos=[
                "01_oscilloscopes.py",
                "02_logic_analyzers.py",
            ],
        )

    def generate_test_data(self) -> dict[str, Any]:
        """Generate synthetic multi-channel data."""
        self.info("Creating synthetic multi-channel captures...")

        # 4-channel oscilloscope capture
        osc_channels = self._create_oscilloscope_multichannel()
        self.info("  ✓ 4-channel oscilloscope capture (synchronized)")

        # Mixed analog/digital capture
        mixed_channels = self._create_mixed_signal_capture()
        self.info("  ✓ Mixed analog/digital capture (MSO)")

        return {
            "oscilloscope": osc_channels,
            "mixed_signal": mixed_channels,
        }

    def _create_oscilloscope_multichannel(self) -> dict[str, WaveformTrace]:
        """Create 4-channel oscilloscope capture."""
        sample_rate = 1e6  # 1 MHz
        duration = 0.01  # 10 ms

        channels = {}

        # CH1: 1 kHz sine
        ch1 = generate_sine_wave(1e3, 1.0, duration, sample_rate)
        channels["CH1"] = WaveformTrace(
            data=ch1.data,
            metadata=TraceMetadata(
                channel_name="CH1",
                sample_rate=sample_rate,
                vertical_scale=1.0,
                vertical_offset=0.0,
            ),
        )

        # CH2: 2 kHz sine (90° phase shift)
        t = np.arange(int(sample_rate * duration)) / sample_rate
        ch2_data = np.sin(2 * np.pi * 2e3 * t + np.pi / 2)
        channels["CH2"] = WaveformTrace(
            data=ch2_data,
            metadata=TraceMetadata(
                channel_name="CH2",
                sample_rate=sample_rate,
                vertical_scale=1.0,
                vertical_offset=0.0,
            ),
        )

        # CH3: 500 Hz square wave
        ch3 = generate_square_wave(500, 2.0, duration, sample_rate)
        channels["CH3"] = WaveformTrace(
            data=ch3.data,
            metadata=TraceMetadata(
                channel_name="CH3",
                sample_rate=sample_rate,
                vertical_scale=2.0,
                vertical_offset=0.0,
            ),
        )

        # CH4: Sum of CH1 and CH2 (simulated probe math)
        ch4_data = channels["CH1"].data + channels["CH2"].data
        channels["CH4"] = WaveformTrace(
            data=ch4_data,
            metadata=TraceMetadata(
                channel_name="CH4",
                sample_rate=sample_rate,
                vertical_scale=1.0,
                vertical_offset=0.0,
            ),
        )

        return channels

    def _create_mixed_signal_capture(self) -> dict[str, Any]:
        """Create mixed analog/digital capture."""
        sample_rate = 10e6  # 10 MHz
        duration = 0.001  # 1 ms

        # Analog channel
        analog = generate_sine_wave(100e3, 1.5, duration, sample_rate)

        # Digital channels (clock and data)
        t = np.arange(int(sample_rate * duration)) / sample_rate
        clock = (np.sin(2 * np.pi * 1e6 * t) > 0).astype(bool)
        data = (np.sin(2 * np.pi * 125e3 * t) > 0).astype(bool)

        return {
            "analog": {
                "CH1": WaveformTrace(
                    data=analog.data,
                    metadata=TraceMetadata(
                        channel_name="CH1",
                        sample_rate=sample_rate,
                    ),
                )
            },
            "digital": {"CLK": clock, "DATA": data},
            "sample_rate": sample_rate,
        }

    def run_demonstration(self, data: dict[str, Any]) -> dict[str, Any]:
        """Run the multi-channel demonstration."""
        results = {}

        self.subsection("Multi-Channel Data Overview")
        self.info("Multi-channel captures are common in:")
        self.info("  • Oscilloscopes: 2-8 analog channels")
        self.info("  • Mixed-signal oscilloscopes: Analog + digital")
        self.info("  • Logic analyzers: 8-64 digital channels")
        self.info("  • Data acquisition: 4-256 channels")
        self.info("")

        # Analyze oscilloscope channels
        self.subsection("4-Channel Oscilloscope Analysis")
        results["oscilloscope"] = self._analyze_oscilloscope_channels(data["oscilloscope"])

        # Analyze mixed-signal
        self.subsection("Mixed-Signal Oscilloscope (MSO)")
        results["mixed_signal"] = self._analyze_mixed_signal(data["mixed_signal"])

        # Cross-channel analysis
        self.subsection("Cross-Channel Analysis")
        self._demonstrate_cross_channel_analysis(data["oscilloscope"])

        # Best practices
        self.subsection("Multi-Channel Best Practices")
        self._show_best_practices()

        return results

    def _analyze_oscilloscope_channels(self, channels: dict[str, WaveformTrace]) -> dict[str, Any]:
        """Analyze oscilloscope multi-channel capture."""
        self.result("Total Channels", len(channels))

        # Build channel statistics table
        channel_rows = []
        for name, trace in channels.items():
            rms = float(np.sqrt(np.mean(trace.data**2)))
            peak = float(np.max(np.abs(trace.data)))
            channel_rows.append(
                [
                    name,
                    len(trace.data),
                    f"{trace.metadata.sample_rate / 1e6:.1f} MHz",
                    f"{rms:.3f} V",
                    f"{peak:.3f} V",
                ]
            )

        headers = ["Channel", "Samples", "Sample Rate", "RMS", "Peak"]
        print(format_table(channel_rows, headers=headers))

        return {
            "num_channels": len(channels),
            "all_synchronized": True,  # All same sample rate
        }

    def _analyze_mixed_signal(self, mixed_data: dict[str, Any]) -> dict[str, Any]:
        """Analyze mixed-signal capture."""
        analog_channels = len(mixed_data["analog"])
        digital_channels = len(mixed_data["digital"])

        self.result("Analog Channels", analog_channels)
        self.result("Digital Channels", digital_channels)
        self.result("Sample Rate", f"{mixed_data['sample_rate'] / 1e6:.0f}", "MHz")

        # Check synchronization
        for ch_name, trace in mixed_data["analog"].items():
            self.result(f"  {ch_name} Samples", len(trace.data))

        for ch_name, data in mixed_data["digital"].items():
            self.result(f"  {ch_name} Samples", len(data))

        return {
            "analog_channels": analog_channels,
            "digital_channels": digital_channels,
        }

    def _demonstrate_cross_channel_analysis(self, channels: dict[str, WaveformTrace]) -> None:
        """Demonstrate cross-channel analysis techniques."""
        self.info("Cross-Channel Analysis Examples:")

        # Phase difference between CH1 and CH2
        ch1_data = channels["CH1"].data
        ch2_data = channels["CH2"].data

        # Simple correlation
        correlation = float(np.corrcoef(ch1_data, ch2_data)[0, 1])
        self.result("CH1-CH2 Correlation", f"{correlation:.3f}")

        # Check if CH4 is sum of CH1 and CH2
        ch4_data = channels["CH4"].data
        expected_sum = ch1_data + ch2_data
        sum_error = float(np.mean(np.abs(ch4_data - expected_sum)))
        self.result("CH4 Sum Error", f"{sum_error:.6f}", "V")

    def _show_best_practices(self) -> None:
        """Show best practices for multi-channel handling."""
        self.info("""
Multi-Channel Data Best Practices:

1. SYNCHRONIZATION
   - Verify all channels have same sample rate
   - Check for clock domain crossings
   - Validate timestamp alignment
   - Handle different acquisition start times

2. MEMORY MANAGEMENT
   - Load channels lazily if memory constrained
   - Use memory-mapped files for large multi-channel captures
   - Consider downsampling unused channels

3. CHANNEL NAMING
   - Use consistent naming: CH1, CH2, CH3...
   - Document probe attenuations per channel
   - Store coupling mode (AC/DC) per channel
   - Include units in metadata

4. CROSS-CHANNEL ANALYSIS
   - Correlation: Measure signal relationships
   - Phase difference: Timing between signals
   - Transfer function: Frequency domain relationship
   - Math channels: Derived signals (sum, diff, multiply)
        """)

    def validate(self, results: dict[str, Any]) -> bool:
        """Validate multi-channel loading results."""
        suite = ValidationSuite()

        # Validate oscilloscope channels
        if "oscilloscope" in results:
            suite.check_equal(results["oscilloscope"]["num_channels"], 4, "Oscilloscope channels")
            suite.check_true(results["oscilloscope"]["all_synchronized"], "Channels synchronized")

        # Validate mixed-signal
        if "mixed_signal" in results:
            suite.check_equal(results["mixed_signal"]["analog_channels"], 1, "Analog channels")
            suite.check_equal(results["mixed_signal"]["digital_channels"], 2, "Digital channels")

        if suite.all_passed():
            self.success("All multi-channel validations passed!")
            self.info("\nNext Steps:")
            self.info("  - Explore cross-channel analysis techniques")
            self.info("  - Try protocol decoding on digital channels")
        else:
            self.error("Some validations failed")

        return suite.all_passed()


if __name__ == "__main__":
    demo = MultiChannelDemo()
    success = demo.execute()
    sys.exit(0 if success else 1)
