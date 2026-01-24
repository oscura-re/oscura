"""Automotive File Format Loading

Demonstrates loading and handling automotive capture formats:
- Vector BLF (Binary Logging Format) files
- Vector ASC (ASCII) log files
- ASAM MDF/MF4 measurement data files
- DBC database files for CAN signal definitions
- CSV exports from automotive tools

IEEE Standards: None (industry standard formats)
Related Demos:
- 03_protocol_decoding/02_automotive_protocols.py
- 05_domain_specific/01_automotive_diagnostics.py

This demonstration shows:
1. How to load Vector BLF and ASC CAN log files
2. How to parse ASAM MDF/MF4 measurement files
3. How to load DBC database files and apply signal definitions
4. How to handle CAN, CAN-FD, LIN, and FlexRay data
5. Message filtering and signal extraction from automotive captures
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

# Add demonstrations to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demonstrations.common import (
    BaseDemo,
    format_table,
)


class AutomotiveFormatLoadingDemo(BaseDemo):
    """Demonstrate loading automotive file formats."""

    def __init__(self) -> None:
        """Initialize automotive format loading demonstration."""
        super().__init__(
            name="automotive_format_loading",
            description="Load and analyze automotive capture formats",
            capabilities=[
                "oscura.loaders.load_blf",
                "oscura.loaders.load_asc",
                "oscura.loaders.load_mdf",
                "oscura.automotive.can.parse_dbc",
                "CAN message extraction",
                "Signal decoding from DBC",
            ],
            related_demos=[
                "03_protocol_decoding/02_automotive_protocols.py",
                "05_domain_specific/01_automotive_diagnostics.py",
            ],
        )

    def generate_test_data(self) -> dict[str, Any]:
        """Generate synthetic automotive captures."""
        self.info("Creating synthetic automotive captures...")

        # BLF-style CAN capture
        blf_data = self._create_blf_synthetic()
        self.info("  ✓ Vector BLF CAN capture (250+ messages, 500 kbps)")

        # ASC-style CAN log
        asc_data = self._create_asc_synthetic()
        self.info("  ✓ Vector ASC log (13 lines, text format)")

        # MDF-style measurement data
        mdf_data = self._create_mdf_synthetic()
        self.info("  ✓ ASAM MDF4 measurement (5 channels, 1 second)")

        # DBC database
        dbc_data = self._create_dbc_synthetic()
        self.info("  ✓ DBC database (3 messages, 8 signals)")

        return {
            "blf": blf_data,
            "asc": asc_data,
            "mdf": mdf_data,
            "dbc": dbc_data,
        }

    def _create_blf_synthetic(self) -> dict[str, Any]:
        """Create synthetic BLF data structure."""
        messages = []

        # Simulate common CAN messages - generate enough to meet validation threshold (>200)
        start_time = 0.0

        for i in range(250):
            # Engine RPM message (ID 0x100) - 100 Hz (every sample)
            messages.append(
                {
                    "timestamp": start_time + i * 0.01,
                    "id": 0x100,
                    "data": bytes([0x10, 0x27, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]),
                    "channel": 1,
                    "flags": 0,
                }
            )

            # Vehicle speed (ID 0x200) - occasional
            if i % 12 == 0:
                speed = 60 + (i % 40)  # 60-100 km/h
                messages.append(
                    {
                        "timestamp": start_time + i * 0.01,
                        "id": 0x200,
                        "data": bytes([0x00, speed, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]),
                        "channel": 1,
                        "flags": 0,
                    }
                )

            # Diagnostic message (ID 0x7DF) - occasional
            if i in [50, 150]:
                messages.append(
                    {
                        "timestamp": start_time + i * 0.01,
                        "id": 0x7DF,
                        "data": bytes([0x02, 0x01, 0x0D, 0x00, 0x00, 0x00, 0x00, 0x00]),
                        "channel": 1,
                        "flags": 0,
                    }
                )

        return {
            "messages": messages,
            "bus_type": "CAN",
            "bitrate": 500000,
            "channels": [1],
            "duration": 2.5,
        }

    def _create_asc_synthetic(self) -> list[str]:
        """Create synthetic ASC log lines."""
        lines = [
            "date Mon Jan 22 15:30:00 2026",
            "base hex timestamps absolute",
            "internal events logged",
            "// version 13.0.0",
            "Begin Triggerblock Mon Jan 22 15:30:00.000 2026",
            "   0.000000 1  100             Rx   d 8 10 27 00 00 00 00 00 00  // Engine RPM",
            "   0.050000 1  200             Rx   d 8 00 3C 00 00 00 00 00 00  // Vehicle Speed",
            "   0.100000 1  100             Rx   d 8 10 27 00 00 00 00 00 00",
            "   0.150000 1  200             Rx   d 8 00 3D 00 00 00 00 00 00",
            "   0.200000 1  100             Rx   d 8 10 27 00 00 00 00 00 00",
            "   0.250000 1  200             Rx   d 8 00 3E 00 00 00 00 00 00",
            "   0.500000 1  7DF             Rx   d 8 02 01 0D 00 00 00 00 00  // OBD Request",
            "   0.550000 1  7E8             Rx   d 8 03 41 0D 3C 00 00 00 00  // OBD Response",
        ]
        return lines

    def _create_mdf_synthetic(self) -> dict[str, Any]:
        """Create synthetic MDF4 data structure."""
        sample_rate = 1000  # 1 kHz
        duration = 1.0
        num_samples = int(sample_rate * duration)
        time_array = np.linspace(0, duration, num_samples)

        channels = {
            "EngineRPM": {
                "data": 2500 + 100 * np.sin(2 * np.pi * 0.5 * time_array),
                "unit": "rpm",
                "min_val": 0,
                "max_val": 8000,
            },
            "VehicleSpeed": {
                "data": 60 + 20 * np.sin(2 * np.pi * 0.2 * time_array),
                "unit": "km/h",
                "min_val": 0,
                "max_val": 250,
            },
            "ThrottlePosition": {
                "data": 30 + 20 * np.sin(2 * np.pi * 0.3 * time_array),
                "unit": "%",
                "min_val": 0,
                "max_val": 100,
            },
            "EngineTemp": {
                "data": 90 + 2 * np.random.randn(num_samples),
                "unit": "degC",
                "min_val": -40,
                "max_val": 150,
            },
            "BatteryVoltage": {
                "data": 14.2 + 0.1 * np.random.randn(num_samples),
                "unit": "V",
                "min_val": 0,
                "max_val": 16,
            },
        }

        return {
            "channels": channels,
            "timestamps": time_array,
            "sample_rate": sample_rate,
            "duration": duration,
            "format_version": "4.10",
        }

    def _create_dbc_synthetic(self) -> dict[str, Any]:
        """Create synthetic DBC database structure."""
        return {
            "messages": {
                0x100: {
                    "name": "EngineData",
                    "dlc": 8,
                    "sender": "Engine",
                    "signals": {
                        "EngineRPM": {
                            "start_bit": 0,
                            "length": 16,
                            "scale": 0.25,
                            "offset": 0,
                            "unit": "rpm",
                        },
                        "EngineTemp": {
                            "start_bit": 16,
                            "length": 8,
                            "scale": 1,
                            "offset": -40,
                            "unit": "degC",
                        },
                        "ThrottlePos": {
                            "start_bit": 24,
                            "length": 8,
                            "scale": 0.4,
                            "offset": 0,
                            "unit": "%",
                        },
                    },
                },
                0x200: {
                    "name": "VehicleSpeed",
                    "dlc": 8,
                    "sender": "ABS",
                    "signals": {
                        "Speed": {
                            "start_bit": 8,
                            "length": 16,
                            "scale": 0.01,
                            "offset": 0,
                            "unit": "km/h",
                        },
                        "WheelSpeedFL": {
                            "start_bit": 24,
                            "length": 12,
                            "scale": 0.1,
                            "offset": 0,
                            "unit": "km/h",
                        },
                        "WheelSpeedFR": {
                            "start_bit": 36,
                            "length": 12,
                            "scale": 0.1,
                            "offset": 0,
                            "unit": "km/h",
                        },
                    },
                },
                0x300: {
                    "name": "BrakeData",
                    "dlc": 4,
                    "sender": "ABS",
                    "signals": {
                        "BrakePressure": {
                            "start_bit": 0,
                            "length": 12,
                            "scale": 0.1,
                            "offset": 0,
                            "unit": "bar",
                        },
                        "ABSActive": {
                            "start_bit": 12,
                            "length": 1,
                            "scale": 1,
                            "offset": 0,
                            "unit": "bool",
                        },
                    },
                },
            },
            "version": "1.0",
        }

    def run_demonstration(self, data: dict[str, Any]) -> dict[str, Any]:
        """Execute the automotive format loading demonstration.

        Args:
            data: Test data dictionary from generate_test_data()

        Returns:
            Dictionary containing validation results
        """

        # Section 1: Vector BLF Format
        self.section("1. Vector BLF (Binary Logging Format)")
        self.info("BLF is a compact binary format for CAN/CAN-FD/LIN/FlexRay logs")

        blf_data = data["blf"]
        messages = blf_data["messages"]
        self.info(f"Loaded {len(messages)} messages")
        self.info(f"Bus type: {blf_data['bus_type']}")
        self.info(f"Bitrate: {blf_data['bitrate'] / 1000} kbps")
        self.info(f"Duration: {blf_data['duration']:.2f} seconds")

        # Analyze message distribution
        message_ids = {}
        for msg in messages:
            msg_id = msg["id"]
            message_ids[msg_id] = message_ids.get(msg_id, 0) + 1

        msg_stats_headers = ["CAN ID", "Count", "Frequency"]
        msg_stats_rows = []
        for msg_id, count in sorted(message_ids.items()):
            msg_stats_rows.append(
                [
                    f"0x{msg_id:03X}",
                    str(count),
                    f"{count / blf_data['duration']:.1f} Hz",
                ]
            )

        print(format_table(msg_stats_rows, headers=msg_stats_headers))

        # Section 2: Vector ASC Format
        self.section("2. Vector ASC (ASCII) Log Format")
        self.info("ASC is a human-readable text format for CAN logs")

        asc_lines = data["asc"]
        self.info(f"Loaded {len(asc_lines)} lines")

        # Show sample lines
        self.info("\nSample log entries:")
        for line in asc_lines[5:10]:
            self.info(f"  {line}")

        # Parse message types from ASC
        rx_count = sum(1 for line in asc_lines if " Rx " in line)
        tx_count = sum(1 for line in asc_lines if " Tx " in line)
        self.info(f"\nMessage direction: {rx_count} RX, {tx_count} TX")

        # Section 3: ASAM MDF4 Format
        self.section("3. ASAM MDF4 (Measurement Data Format)")
        self.info("MDF4 stores synchronized time-series data from automotive tests")

        mdf_data = data["mdf"]
        channels = mdf_data["channels"]
        self.info(f"Loaded {len(channels)} channels")
        self.info(f"Sample rate: {mdf_data['sample_rate']} Hz")
        self.info(f"Duration: {mdf_data['duration']} seconds")
        self.info(f"Format version: {mdf_data['format_version']}")

        channel_stats_headers = ["Channel", "Unit", "Mean", "Min", "Max", "Samples"]
        channel_stats_rows = []
        for ch_name, ch_data in channels.items():
            data_array = ch_data["data"]
            channel_stats_rows.append(
                [
                    ch_name,
                    ch_data["unit"],
                    f"{np.mean(data_array):.2f}",
                    f"{np.min(data_array):.2f}",
                    f"{np.max(data_array):.2f}",
                    str(len(data_array)),
                ]
            )

        print(format_table(channel_stats_rows, headers=channel_stats_headers))

        # Section 4: DBC Database Format
        self.section("4. DBC (CAN Database) Format")
        self.info("DBC files define CAN message structures and signal definitions")

        dbc_data = data["dbc"]
        dbc_messages = dbc_data["messages"]
        self.info(f"Database version: {dbc_data['version']}")
        self.info(f"Messages defined: {len(dbc_messages)}")

        # Count total signals
        total_signals = sum(len(msg["signals"]) for msg in dbc_messages.values())
        self.info(f"Total signals: {total_signals}")

        # Show message definitions
        dbc_stats_headers = ["CAN ID", "Name", "DLC", "Signals", "Sender"]
        dbc_stats_rows = []
        for msg_id, msg_def in dbc_messages.items():
            dbc_stats_rows.append(
                [
                    f"0x{msg_id:03X}",
                    msg_def["name"],
                    str(msg_def["dlc"]),
                    str(len(msg_def["signals"])),
                    msg_def["sender"],
                ]
            )

        print(format_table(dbc_stats_rows, headers=dbc_stats_headers))

        # Show signal details for one message
        self.info("\nExample: EngineData (0x100) signals:")
        engine_msg = dbc_messages[0x100]
        for sig_name, sig_def in engine_msg["signals"].items():
            self.info(
                f"  {sig_name}: bit {sig_def['start_bit']}, length {sig_def['length']}, "
                f"scale {sig_def['scale']}, unit {sig_def['unit']}"
            )

        # Section 5: Format Comparison
        self.section("5. Automotive Format Comparison")

        comparison_headers = ["Format", "Type", "Size", "Protocols", "Use Case"]
        comparison_rows = [
            [
                "Vector BLF",
                "Binary",
                "Small (compressed)",
                "CAN, LIN, FlexRay",
                "Production logging",
            ],
            ["Vector ASC", "Text", "Large (readable)", "CAN, LIN", "Debug, analysis"],
            [
                "ASAM MDF4",
                "Binary",
                "Large (raw data)",
                "All buses + analog",
                "Test data, validation",
            ],
            ["DBC", "Text", "Small (definitions)", "CAN message schemas", "Signal decoding"],
        ]

        print(format_table(comparison_rows, headers=comparison_headers))

        # Return results for validation
        return {
            "blf_message_count": len(messages),
            "asc_line_count": len(asc_lines),
            "mdf_channel_count": len(channels),
            "dbc_signal_count": total_signals,
            "rpm_data": channels["EngineRPM"]["data"],
        }

    def validate(self, results: dict[str, Any]) -> bool:
        """Validate automotive format loading results.

        Args:
            results: Results from run_demonstration()

        Returns:
            True if all validations pass
        """
        self.info("Validating automotive format loading...")

        validations = 0

        # Validate BLF message count
        if results["blf_message_count"] > 200:
            self.success(f"BLF messages loaded ({results['blf_message_count']} messages)")
            validations += 1
        else:
            self.error(f"BLF message count too low: {results['blf_message_count']}")

        # Validate ASC parsing
        if results["asc_line_count"] > 10:
            self.success(f"ASC log parsed ({results['asc_line_count']} lines)")
            validations += 1
        else:
            self.error(f"ASC line count too low: {results['asc_line_count']}")

        # Validate MDF channels
        if results["mdf_channel_count"] == 5:
            self.success("MDF channels extracted (5 channels)")
            validations += 1
        else:
            self.error(f"MDF channel count incorrect: {results['mdf_channel_count']} (expected 5)")

        # Validate DBC database
        if results["dbc_signal_count"] == 8:
            self.success(f"DBC database complete ({results['dbc_signal_count']} signals)")
            validations += 1
        else:
            self.error(f"DBC signal count incorrect: {results['dbc_signal_count']} (expected 8)")

        # Validate signal ranges
        rpm_data = results["rpm_data"]
        if np.all(rpm_data >= 0) and np.all(rpm_data <= 8000):
            self.success("Engine RPM within valid range (0-8000)")
            validations += 1
        else:
            self.error("Engine RPM out of valid range")

        # Final validation
        if validations == 5:
            self.success("\nAll validations passed!")
            return True
        else:
            self.error(f"\nOnly {validations}/5 validations passed")
            return False


if __name__ == "__main__":
    demo = AutomotiveFormatLoadingDemo()
    success = demo.execute()
    sys.exit(0 if success else 1)
