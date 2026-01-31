"""Logic Analyzer File Format Loading

Demonstrates loading and handling logic analyzer capture formats:
- Sigrok .sr files (PulseView/sigrok-cli captures)
- VCD (Value Change Dump) files from simulators
- Digital waveform extraction and timing analysis
- Multi-channel digital data handling

IEEE Standards: IEEE 1364-2005 (Verilog VCD format)
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
)

from oscura.core.types import DigitalTrace, TraceMetadata


class LogicAnalyzerLoadingDemo(BaseDemo):
    """Demonstrate loading logic analyzer capture formats."""

    def __init__(self) -> None:
        """Initialize logic analyzer loading demonstration."""
        super().__init__(
            name="logic_analyzer_loading",
            description="Load and analyze logic analyzer capture formats",
            capabilities=[
                "oscura.loaders.load_sigrok",
                "oscura.loaders.load_vcd",
                "DigitalTrace data extraction",
                "Multi-channel digital waveforms",
                "Timing relationship analysis",
            ],
            ieee_standards=["IEEE 1364-2005"],
            related_demos=[
                "01_oscilloscopes.py",
                "07_multi_channel.py",
            ],
        )

    def generate_test_data(self) -> dict[str, Any]:
        """Generate synthetic logic analyzer captures."""
        self.info("Creating synthetic logic analyzer captures...")

        # Sigrok-style multi-channel digital capture
        sigrok_data = self._create_sigrok_synthetic()
        self.info("  ✓ Sigrok capture (8 channels, 1 MHz sampling)")

        # VCD from HDL simulator
        vcd_data = self._create_vcd_synthetic()
        self.info("  ✓ VCD simulator output (4 signals, 100 ns timescale)")

        # Saleae Logic-style high-speed capture
        saleae_data = self._create_saleae_synthetic()
        self.info("  ✓ Saleae Logic capture (16 channels, 24 MHz)")

        return {
            "sigrok": sigrok_data,
            "vcd": vcd_data,
            "saleae": saleae_data,
        }

    def _create_sigrok_synthetic(self) -> dict[str, DigitalTrace]:
        """Create synthetic Sigrok multi-channel capture."""
        sample_rate = 1e6  # 1 MHz
        duration = 0.001  # 1 ms
        num_samples = int(sample_rate * duration)
        time_array = np.arange(num_samples) / sample_rate

        channels = {}

        # Clock signal (100 kHz)
        clock_freq = 100e3
        clock_data = (np.sin(2 * np.pi * clock_freq * time_array) > 0).astype(np.uint8)
        channels["CLK"] = DigitalTrace(
            data=clock_data.astype(bool),
            metadata=TraceMetadata(
                channel_name="CLK",
                sample_rate=sample_rate,
            ),
        )

        # Data signal (serial data)
        data_freq = 10e3
        data_signal = (np.sin(2 * np.pi * data_freq * time_array) > 0).astype(np.uint8)
        channels["DATA"] = DigitalTrace(
            data=data_signal.astype(bool),
            metadata=TraceMetadata(
                channel_name="DATA",
                sample_rate=sample_rate,
            ),
        )

        # Chip select (active during middle of capture)
        cs_data = np.ones(num_samples, dtype=np.uint8)
        cs_data[200:800] = 0  # Active low for middle section
        channels["CS"] = DigitalTrace(
            data=cs_data.astype(bool),
            metadata=TraceMetadata(
                channel_name="CS",
                sample_rate=sample_rate,
            ),
        )

        return channels

    def _create_vcd_synthetic(self) -> dict[str, Any]:
        """Create synthetic VCD data structure."""
        return {
            "timescale": "1ns",
            "signals": {
                "clk": {
                    "type": "wire",
                    "width": 1,
                    "transitions": [(0, 0), (10, 1), (20, 0), (30, 1)],
                },
                "rst": {"type": "wire", "width": 1, "transitions": [(0, 1), (50, 0)]},
                "data": {
                    "type": "wire",
                    "width": 8,
                    "transitions": [(0, 0x00), (100, 0xFF), (200, 0xAA)],
                },
                "valid": {"type": "wire", "width": 1, "transitions": [(0, 0), (100, 1), (300, 0)]},
            },
            "duration_ns": 1000,
        }

    def _create_saleae_synthetic(self) -> dict[str, DigitalTrace]:
        """Create synthetic Saleae Logic high-speed capture."""
        sample_rate = 24e6  # 24 MHz
        duration = 0.0001  # 100 μs
        num_samples = int(sample_rate * duration)
        time_array = np.arange(num_samples) / sample_rate

        channels = {}

        # SCLK - 1 MHz clock
        sclk_freq = 1e6
        sclk_data = (np.sin(2 * np.pi * sclk_freq * time_array) > 0).astype(np.uint8)
        channels["SCLK"] = DigitalTrace(
            data=sclk_data.astype(bool),
            metadata=TraceMetadata(
                channel_name="SCLK",
                sample_rate=sample_rate,
            ),
        )

        # MOSI - Data output from master
        mosi_freq = 125e3
        mosi_data = (np.sin(2 * np.pi * mosi_freq * time_array + 0.5) > 0).astype(np.uint8)
        channels["MOSI"] = DigitalTrace(
            data=mosi_data.astype(bool),
            metadata=TraceMetadata(
                channel_name="MOSI",
                sample_rate=sample_rate,
            ),
        )

        return channels

    def run_demonstration(self, data: dict[str, Any]) -> dict[str, Any]:
        """Execute the logic analyzer loading demonstration."""
        results: dict[str, Any] = {}

        # Section 1: Sigrok Multi-Channel Analysis
        self.subsection("Sigrok Multi-Channel Digital Capture")
        self.info("Sigrok/PulseView captures store digital channels in ZIP archives")

        sigrok_channels = data["sigrok"]
        self.info(f"Loaded {len(sigrok_channels)} channels:")

        channel_info_rows = []
        for name, trace in sigrok_channels.items():
            num_transitions = np.sum(np.diff(trace.data.astype(int)) != 0)
            channel_info_rows.append(
                [
                    name,
                    str(len(trace.data)),
                    str(num_transitions),
                    f"{trace.metadata.sample_rate / 1e6:.1f} MHz",
                    f"{len(trace.data) / trace.metadata.sample_rate * 1e3:.2f} ms",
                ]
            )

        headers = ["Channel", "Samples", "Transitions", "Sample Rate", "Duration"]
        print(format_table(channel_info_rows, headers=headers))

        # Analyze timing relationship between CLK and DATA
        clk_trace = sigrok_channels["CLK"]
        data_trace = sigrok_channels["DATA"]

        # Find clock edges
        clk_rising = np.where(np.diff(clk_trace.data.astype(int)) > 0)[0]
        self.info(f"Clock rising edges detected: {len(clk_rising)}")

        # Check data stability at clock edges
        data_at_edges = data_trace.data[clk_rising]
        self.info(
            f"Data samples at clock edges: {np.sum(data_at_edges)} HIGH, "
            f"{len(data_at_edges) - np.sum(data_at_edges)} LOW"
        )

        results["clk_rising_edges"] = len(clk_rising)
        results["sigrok_channels"] = len(sigrok_channels)

        # Section 2: VCD File Analysis
        self.subsection("VCD (Value Change Dump) Format")
        self.info("VCD files store signal transitions from HDL simulators")

        vcd_data = data["vcd"]
        self.info(f"Timescale: {vcd_data['timescale']}")
        self.info(f"Total duration: {vcd_data['duration_ns']} ns")
        self.info(f"Signals: {len(vcd_data['signals'])}")

        signal_info_rows = []
        for sig_name, sig_data in vcd_data["signals"].items():
            signal_info_rows.append(
                [
                    sig_name,
                    sig_data["type"],
                    f"{sig_data['width']} bits",
                    str(len(sig_data["transitions"])),
                ]
            )

        headers = ["Signal", "Type", "Width", "Transitions"]
        print(format_table(signal_info_rows, headers=headers))

        results["vcd_signals"] = len(vcd_data["signals"])

        # Section 3: Saleae Logic High-Speed Capture
        self.subsection("Saleae Logic High-Speed Digital Capture")
        self.info("Saleae Logic analyzers support up to 500 MHz digital sampling")

        saleae_channels = data["saleae"]
        self.info(f"Loaded {len(saleae_channels)} channels")

        saleae_info_rows = []
        for name, trace in saleae_channels.items():
            num_transitions = np.sum(np.diff(trace.data.astype(int)) != 0)
            duty_cycle = np.mean(trace.data) * 100
            saleae_info_rows.append(
                [
                    name,
                    str(len(trace.data)),
                    str(num_transitions),
                    f"{duty_cycle:.1f}%",
                    f"{trace.metadata.sample_rate / 1e6:.0f} MHz",
                ]
            )

        headers = ["Channel", "Samples", "Transitions", "Duty Cycle", "Sample Rate"]
        print(format_table(saleae_info_rows, headers=headers))

        results["saleae_channels"] = len(saleae_channels)
        results["saleae_sample_rate"] = float(saleae_channels["SCLK"].metadata.sample_rate)

        # Section 4: Format Comparison
        self.subsection("Logic Analyzer Format Comparison")

        comparison_rows = [
            [
                "Sigrok .sr",
                "1-32 digital",
                "Up to 200 MHz",
                "ZIP archive, metadata",
                "General purpose LA",
            ],
            [
                "VCD",
                "Unlimited",
                "Timescale-based",
                "HDL simulator output",
                "Verification, simulation",
            ],
            [
                "Saleae Binary",
                "8-16 digital",
                "Up to 500 MHz",
                "High-speed, compression",
                "High-speed protocols",
            ],
        ]

        headers = ["Format", "Channels", "Max Sample Rate", "Features", "Use Case"]
        print(format_table(comparison_rows, headers=headers))

        self.success("Logic analyzer format loading complete!")
        return results

    def validate(self, results: dict[str, Any]) -> bool:
        """Validate logic analyzer loading results."""
        suite = ValidationSuite()

        # Validate Sigrok channel count
        suite.check_equal(results["sigrok_channels"], 3, "Sigrok channels")

        # Validate VCD signals
        suite.check_equal(results["vcd_signals"], 4, "VCD signals")

        # Validate Saleae sample rate
        suite.check_approximately(
            results["saleae_sample_rate"], 24e6, tolerance=0.01, name="Saleae sample rate"
        )

        # Validate clock edges exist
        suite.check_true(results["clk_rising_edges"] > 50, "Clock rising edges detected")

        if suite.all_passed():
            self.success("All logic analyzer validations passed!")
            self.info("\nNext Steps:")
            self.info("  - Explore protocol decoding demos for digital signal analysis")
            self.info("  - Try timing analysis demos for setup/hold time checks")
        else:
            self.error("Some validations failed")

        return suite.all_passed()


if __name__ == "__main__":
    demo = LogicAnalyzerLoadingDemo()
    success = demo.execute()
    sys.exit(0 if success else 1)
