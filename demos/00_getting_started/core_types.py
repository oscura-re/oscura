"""Core Types: Understanding Oscura's Data Structures.

Demonstrates Oscura's fundamental data structures:
- TraceMetadata - Timing and calibration information
- WaveformTrace - Analog waveform signals
- DigitalTrace - Digital/logic signals
- ProtocolPacket - Decoded protocol data

IEEE Standards: IEEE 1241-2010 (ADC Terminology)

Related Demos:
- 00_getting_started/00_hello_world.py
- 02_basic_analysis/01_waveform_basics.py
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# Add demos to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from demos.common import BaseDemo, ValidationSuite, run_demo_main
from demos.common.formatting import print_info, print_result, print_subheader
from oscura.core.types import (
    CalibrationInfo,
    DigitalTrace,
    ProtocolPacket,
    TraceMetadata,
    WaveformTrace,
)


class CoreTypesDemo(BaseDemo):
    """Demonstrate Oscura's core data types and their properties."""

    name = "Core Types"
    description = "Learn Oscura's fundamental data structures: traces, metadata, packets"
    category = "getting_started"
    capabilities = [
        "oscura.TraceMetadata",
        "oscura.WaveformTrace",
        "oscura.DigitalTrace",
        "oscura.ProtocolPacket",
        "oscura.CalibrationInfo",
    ]
    ieee_standards = ["IEEE 1241-2010"]
    related_demos = [
        "00_getting_started/00_hello_world.py",
        "02_basic_analysis/01_waveform_basics.py",
    ]

    def generate_data(self) -> None:
        """Generate test data for all core types."""
        # Generate waveform data: 10 kHz sine wave at 1V amplitude, 1 MHz sampling
        duration = 0.001  # 1 ms
        sample_rate = 1e6  # 1 MHz
        frequency = 10e3  # 10 kHz
        num_samples = int(duration * sample_rate)
        t = np.arange(num_samples) / sample_rate
        self.waveform_data = 1.0 * np.sin(2 * np.pi * frequency * t)

        # Generate digital data: 50% duty cycle clock at 100 kHz
        digital_frequency = 100e3  # 100 kHz
        phase = (t * digital_frequency) % 1.0
        self.digital_data = phase < 0.5  # Boolean array

        self.sample_rate = sample_rate
        self.duration = duration
        self.frequency = frequency

    def run_analysis(self) -> None:
        """Run the core types demonstration."""
        print_info("Understanding the fundamental data structures for signal analysis")

        # === Part 1: TraceMetadata ===
        print_subheader("1. TraceMetadata: Timing and Calibration")
        print_info("TraceMetadata stores timing information and instrument settings")

        metadata = TraceMetadata(
            sample_rate=self.sample_rate,
            channel_name="CH1",
            time_offset=0.0,
            trigger_time=datetime.now(),
        )

        print_result("Sample rate", metadata.sample_rate, "Hz")
        print_result("Channel name", metadata.channel_name)
        print_result("Time offset", metadata.time_offset, "s")

        # === Part 2: WaveformTrace ===
        print_subheader("2. WaveformTrace: Analog Signals")
        print_info("WaveformTrace combines analog data with metadata")

        waveform = WaveformTrace(
            data=self.waveform_data,
            metadata=metadata,
        )

        print_result("Number of samples", len(waveform.data))
        print_result("Duration", len(waveform.data) / waveform.metadata.sample_rate, "s")
        print_result("Peak value", np.max(waveform.data), "V")
        print_result("Min value", np.min(waveform.data), "V")

        # === Part 3: DigitalTrace ===
        print_subheader("3. DigitalTrace: Logic Signals")
        print_info("DigitalTrace represents digital (high/low) signals")

        digital = DigitalTrace(
            data=self.digital_data,
            metadata=TraceMetadata(sample_rate=self.sample_rate, channel_name="CLK"),
        )

        print_result("Number of samples", len(digital.data))
        print_result("Logic HIGH count", np.sum(digital.data))
        print_result("Logic LOW count", np.sum(~digital.data))
        duty_cycle = np.mean(digital.data) * 100
        print_result("Duty cycle", duty_cycle, "%")

        # === Part 4: CalibrationInfo ===
        print_subheader("4. CalibrationInfo: Instrument Configuration")
        print_info("CalibrationInfo stores probe and instrument settings")

        calibration = CalibrationInfo(
            probe_attenuation=10.0,
            vertical_scale=1.0,
            vertical_offset=0.0,
            bandwidth_limit=None,
        )

        print_result("Probe attenuation", calibration.probe_attenuation, "x")
        print_result("Vertical scale", calibration.vertical_scale, "V/div")
        print_result("Vertical offset", calibration.vertical_offset, "V")

        # === Part 5: ProtocolPacket ===
        print_subheader("5. ProtocolPacket: Decoded Data")
        print_info("ProtocolPacket represents decoded protocol frames")

        packet = ProtocolPacket(
            start_time=0.0,
            end_time=1.0e-6,
            protocol="UART",
            data={"byte": 0x42, "parity": "even"},
        )

        print_result("Start time", packet.start_time, "s")
        print_result("End time", packet.end_time, "s")
        print_result("Protocol", packet.protocol)
        print_result("Data", str(packet.data))

        # Store results for validation
        self.results["waveform_samples"] = len(waveform.data)
        self.results["digital_duty_cycle"] = duty_cycle
        self.results["probe_attenuation"] = calibration.probe_attenuation
        self.results["packet_protocol"] = packet.protocol

    def validate_results(self, suite: ValidationSuite) -> None:
        """Validate the results."""
        suite.check_equal("Waveform samples", self.results["waveform_samples"], 1000)
        suite.check_close("Digital duty cycle", self.results["digital_duty_cycle"], 50.0, rtol=0.01)
        suite.check_equal("Probe attenuation", self.results["probe_attenuation"], 10.0)
        suite.check_equal("Packet protocol", self.results["packet_protocol"], "UART")

        if suite.all_passed():
            print_info("\nYou now understand Oscura's core data types!")
            print_info("Next steps:")
            print_info("  - Try 02_supported_formats.py to learn about file I/O")
            print_info("  - Explore 02_basic_analysis/ for signal processing")


if __name__ == "__main__":
    sys.exit(run_demo_main(CoreTypesDemo))
