"""Core Types: Understanding Oscura's Data Structures

Demonstrates:
- TraceMetadata - Timing and calibration information
- WaveformTrace - Analog waveform signals
- DigitalTrace - Digital/logic signals
- ProtocolPacket - Decoded protocol data
- Creating, accessing, and converting between types

IEEE Standards: IEEE 1241-2010 (ADC Terminology)
Related Demos:
- 00_getting_started/00_hello_world.py
- 02_basic_analysis/01_waveform_measurements.py

This demonstration teaches the fundamental data structures used throughout
Oscura. You'll learn how to create traces, access their properties, and
understand the relationship between different data types.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# Add demonstrations to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demonstrations.common import BaseDemo, validate_approximately, validate_exists
from oscura.core.types import (
    CalibrationInfo,
    DigitalTrace,
    ProtocolPacket,
    TraceMetadata,
    WaveformTrace,
)


class CoreTypesDemo(BaseDemo):
    """Demonstrate Oscura's core data types and their properties."""

    def __init__(self) -> None:
        """Initialize core types demonstration."""
        super().__init__(
            name="core_types",
            description="Learn Oscura's fundamental data structures: traces, metadata, packets",
            capabilities=[
                "oscura.TraceMetadata",
                "oscura.WaveformTrace",
                "oscura.DigitalTrace",
                "oscura.ProtocolPacket",
                "oscura.CalibrationInfo",
            ],
            ieee_standards=["IEEE 1241-2010"],
            related_demos=[
                "00_getting_started/00_hello_world.py",
                "02_basic_analysis/01_waveform_measurements.py",
            ],
        )

    def generate_test_data(self) -> dict[str, Any]:
        """Generate test data for all core types.

        Returns:
            Dictionary containing:
            - waveform_data: Sine wave signal
            - digital_data: Clock signal
            - calibration_info: Instrument metadata
        """
        # Generate waveform data: 10 kHz sine wave at 1V amplitude, 1 MHz sampling
        duration = 0.001  # 1 ms
        sample_rate = 1e6  # 1 MHz
        frequency = 10e3  # 10 kHz
        num_samples = int(duration * sample_rate)
        t = np.arange(num_samples) / sample_rate
        waveform_data = 1.0 * np.sin(2 * np.pi * frequency * t)

        # Generate digital data: 50% duty cycle clock at 100 kHz
        digital_frequency = 100e3  # 100 kHz
        phase = (t * digital_frequency) % 1.0
        digital_data = phase < 0.5  # Boolean array

        return {
            "waveform_data": waveform_data,
            "digital_data": digital_data,
            "sample_rate": sample_rate,
            "duration": duration,
            "frequency": frequency,
        }

    def run_demonstration(self, data: dict[str, Any]) -> dict[str, Any]:
        """Run the core types demonstration."""
        results: dict[str, Any] = {}

        self.section("Oscura Core Types Demonstration")
        self.info("Understanding the fundamental data structures for signal analysis")

        # Part 1: TraceMetadata
        self.subsection("Part 1: TraceMetadata")
        self.info("TraceMetadata stores timing and acquisition information.")

        metadata = TraceMetadata(
            sample_rate=data["sample_rate"],
            vertical_scale=0.1,  # 100 mV per division
            vertical_offset=0.0,  # No vertical offset
            channel_name="CH1",
            source_file="oscilloscope_capture_001.vcd",
        )

        self.result("Sample rate", f"{metadata.sample_rate:.2e}", "Hz")
        self.result("Time base (1/sample_rate)", f"{metadata.time_base:.2e}", "s/sample")
        self.result("Vertical scale", metadata.vertical_scale, "V/div")
        self.result("Channel name", metadata.channel_name)
        self.result("Source file", metadata.source_file)

        results["metadata"] = metadata

        # Part 2: WaveformTrace - Analog signals
        self.subsection("Part 2: WaveformTrace - Analog Signals")
        self.info("WaveformTrace stores sampled analog voltage data as a numpy array.")

        waveform = WaveformTrace(
            data=data["waveform_data"],
            metadata=metadata,
        )

        self.result("Number of samples", len(waveform))
        self.result("Data type", waveform.data.dtype)
        self.result("Signal duration", f"{waveform.duration:.6f}", "seconds")
        self.result("Min value", f"{np.min(waveform.data):.4f}", "V")
        self.result("Max value", f"{np.max(waveform.data):.4f}", "V")
        self.result("Mean value", f"{np.mean(waveform.data):.6f}", "V")

        # Demonstrate time_vector property
        time_vec = waveform.time_vector
        self.result("Time vector length", len(time_vec))
        self.result("First sample time", f"{time_vec[0]:.9f}", "s")
        self.result("Last sample time", f"{time_vec[-1]:.9f}", "s")
        self.result("Sample 0 value", f"{waveform.data[0]:.6f}", "V")
        self.result("Sample 500 value", f"{waveform.data[500]:.6f}", "V")

        results["waveform"] = waveform
        results["waveform_min"] = float(np.min(waveform.data))
        results["waveform_max"] = float(np.max(waveform.data))

        # Part 3: DigitalTrace - Logic signals
        self.subsection("Part 3: DigitalTrace - Digital/Logic Signals")
        self.info("DigitalTrace stores sampled digital (boolean) data.")

        digital_metadata = TraceMetadata(
            sample_rate=data["sample_rate"],
            channel_name="CLK",
            source_file="oscilloscope_capture_001.vcd",
        )

        digital = DigitalTrace(
            data=data["digital_data"],
            metadata=digital_metadata,
        )

        self.result("Number of samples", len(digital))
        self.result("Data type", digital.data.dtype)
        self.result("Signal duration", f"{digital.duration:.6f}", "seconds")
        self.result("High samples", np.sum(digital.data))
        self.result("Low samples", np.sum(~digital.data))
        self.result("High percentage", f"{100 * np.mean(digital.data):.1f}", "%")

        results["digital"] = digital
        results["digital_high_pct"] = float(100 * np.mean(digital.data))

        # Part 4: Extracting edges from digital signals
        self.subsection("Part 4: Edge Detection from Digital Signals")
        self.info("Extracting rising and falling edges from digital traces.")

        # Compute edges
        transitions = np.diff(digital.data.astype(int))
        rising_indices = np.where(transitions > 0)[0]
        falling_indices = np.where(transitions < 0)[0]

        rising_times = digital.time_vector[rising_indices]
        falling_times = digital.time_vector[falling_indices]

        self.result("Number of rising edges", len(rising_times))
        self.result("Number of falling edges", len(falling_times))

        if len(rising_times) > 1:
            period = rising_times[1] - rising_times[0]
            frequency = 1.0 / period
            self.result("Measured period", f"{period:.9f}", "s")
            self.result("Measured frequency", f"{frequency:.2e}", "Hz")
            results["digital_frequency"] = frequency
        else:
            results["digital_frequency"] = None

        # Part 5: ProtocolPacket - Decoded data
        self.subsection("Part 5: ProtocolPacket - Decoded Protocol Data")
        self.info("ProtocolPacket stores decoded protocol information.")

        packet1 = ProtocolPacket(
            timestamp=0.0001,
            protocol="UART",
            data=b"Hello",
            annotations={"baud_rate": 115200, "bits": 8, "parity": "none"},
        )

        self.result("Packet timestamp", f"{packet1.timestamp:.9f}", "s")
        self.result("Protocol", packet1.protocol)
        self.result("Data (hex)", packet1.data.hex())
        self.result("Data (ascii)", packet1.data.decode())
        self.result("Packet length", len(packet1), "bytes")
        self.result("Baud rate", packet1.annotations.get("baud_rate"), "bps")
        self.result("Has errors", packet1.has_errors)

        # Create a packet with errors
        self.info("\nPacket with errors:")
        packet2 = ProtocolPacket(
            timestamp=0.0002,
            protocol="UART",
            data=b"Error",
            annotations={"baud_rate": 9600},
            errors=["Parity error", "Frame error"],
        )

        self.result("Packet 2 timestamp", f"{packet2.timestamp:.9f}", "s")
        self.result("Packet 2 errors", len(packet2.errors))
        self.result("Has errors", packet2.has_errors)

        results["packet"] = packet1
        results["packet_length"] = len(packet1)

        # Part 6: CalibrationInfo - Instrument traceability
        self.subsection("Part 6: CalibrationInfo - Instrument Calibration")
        self.info("CalibrationInfo stores instrument traceability information.")

        cal_info = CalibrationInfo(
            instrument="Tektronix DPO7254C",
            serial_number="C012345",
            calibration_date=datetime(2024, 12, 15),
            calibration_due_date=datetime(2025, 12, 15),
            firmware_version="1.2.3",
            probe_attenuation=10.0,
            coupling="DC",
            vertical_resolution=8,
        )

        self.result("Instrument", cal_info.instrument)
        self.result("Serial number", cal_info.serial_number)
        self.result("Calibration date", cal_info.calibration_date)
        self.result("Calibration due", cal_info.calibration_due_date)
        self.result("Probe attenuation", f"{cal_info.probe_attenuation}x")
        self.result("Coupling", cal_info.coupling)
        self.result("Vertical resolution", cal_info.vertical_resolution, "bits")
        self.result("Traceability summary", cal_info.traceability_summary)

        results["calibration_info"] = cal_info

        # Part 7: Integration - TraceMetadata with CalibrationInfo
        self.subsection("Part 7: Integration - Adding Calibration to Metadata")
        self.info("TraceMetadata can include CalibrationInfo for complete traceability.")

        metadata_with_cal = TraceMetadata(
            sample_rate=1e9,  # 1 GSa/s
            vertical_scale=0.05,  # 50 mV/div
            channel_name="CH1",
            calibration_info=cal_info,
        )

        self.result("Sample rate", f"{metadata_with_cal.sample_rate:.2e}", "Hz")
        self.result("Calibration info present", metadata_with_cal.calibration_info is not None)
        if metadata_with_cal.calibration_info is not None:
            self.result("Instrument", metadata_with_cal.calibration_info.instrument)

        results["metadata_with_cal"] = metadata_with_cal

        self.success("Core types demonstration complete!")

        return results

    def validate(self, results: dict[str, Any]) -> bool:
        """Validate the demonstration results."""
        self.info("Validating core types...")

        # Validate waveform properties
        if not validate_exists(results.get("waveform"), "WaveformTrace"):
            return False

        if not validate_approximately(
            results["waveform_max"], 1.0, tolerance=0.01, name="Waveform max"
        ):
            return False

        if not validate_approximately(
            results["waveform_min"], -1.0, tolerance=0.01, name="Waveform min"
        ):
            return False

        # Validate digital signal properties
        if not validate_exists(results.get("digital"), "DigitalTrace"):
            return False

        if not validate_approximately(
            results["digital_high_pct"], 50.0, tolerance=2.0, name="Digital duty cycle"
        ):
            return False

        # Validate digital frequency if computed
        if results.get("digital_frequency") is not None:
            if not validate_approximately(
                results["digital_frequency"], 100e3, tolerance=0.05, name="Digital frequency"
            ):
                return False

        # Validate packet properties
        if not validate_exists(results.get("packet"), "ProtocolPacket"):
            return False

        if results["packet_length"] != 5:
            self.error(f"Expected packet length 5, got {results['packet_length']}")
            return False

        # Validate calibration info
        if not validate_exists(results.get("calibration_info"), "CalibrationInfo"):
            return False

        cal_obj = results.get("calibration_info")
        if isinstance(cal_obj, CalibrationInfo):
            if cal_obj.instrument != "Tektronix DPO7254C":
                self.error(f"Expected Tektronix DPO7254C, got {cal_obj.instrument}")
                return False

            if cal_obj.probe_attenuation != 10.0:
                self.error(f"Expected probe attenuation 10.0, got {cal_obj.probe_attenuation}")
                return False
        else:
            self.error("calibration_info is not a CalibrationInfo instance")
            return False

        # Validate metadata with calibration
        if not validate_exists(results.get("metadata_with_cal"), "TraceMetadata"):
            return False

        meta_obj = results.get("metadata_with_cal")
        if isinstance(meta_obj, TraceMetadata):
            if meta_obj.calibration_info is None:
                self.error("Expected metadata_with_cal to have calibration_info")
                return False
        else:
            self.error("metadata_with_cal is not a TraceMetadata instance")
            return False

        self.success("All core type validations passed!")
        self.info("\nKey Takeaways:")
        self.info("  - TraceMetadata: Timing and acquisition info (sample_rate, channel_name)")
        self.info("  - WaveformTrace: Analog signals with numpy array data + metadata")
        self.info("  - DigitalTrace: Digital/logic signals with boolean numpy array")
        self.info("  - ProtocolPacket: Decoded protocol data (timestamp, protocol, data)")
        self.info("  - CalibrationInfo: Instrument traceability and calibration history")
        self.info("  - All traces have time_vector and duration properties")
        self.info("\nNext steps:")
        self.info("  - Try 02_basic_analysis/01_waveform_measurements.py for signal measurements")
        self.info("  - Explore protocol decoding in 03_protocol_decoding/")
        self.info("  - Learn about analyzers in 02_basic_analysis/")

        return True


if __name__ == "__main__":
    demo: CoreTypesDemo = CoreTypesDemo()
    success: bool = demo.execute()
    sys.exit(0 if success else 1)
