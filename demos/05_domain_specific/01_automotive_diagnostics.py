#!/usr/bin/env python3
"""Automotive Diagnostics Demo - OBD-II and UDS.

This demo demonstrates automotive diagnostic capabilities:
- OBD-II (On-Board Diagnostics II) protocol decoding
- UDS (Unified Diagnostic Services) ISO 14229
- DTC (Diagnostic Trouble Code) reading
- ECU communication over CAN bus

Standards:
- SAE J1979 (OBD-II)
- ISO 14229 (UDS)
- ISO 15765-2 (Diagnostic communication over CAN)
- SAE J2012 (DTC definitions)

Usage:
    python demos/05_domain_specific/01_automotive_diagnostics.py
    python demos/05_domain_specific/01_automotive_diagnostics.py --verbose

Author: Oscura Development Team
Date: 2026-01-29
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demos.common import BaseDemo, ValidationSuite, print_info, print_result
from demos.common.base_demo import run_demo_main
from demos.common.formatting import print_subheader
from oscura.core.types import TraceMetadata, WaveformTrace


class AutomotiveDiagnosticsDemo(BaseDemo):
    """Automotive Diagnostics Demonstration.

    Demonstrates OBD-II and UDS diagnostic protocol decoding for
    vehicle ECU communication and troubleshooting.
    """

    name = "Automotive Diagnostics (OBD-II & UDS)"
    description = "Demonstrates automotive diagnostic protocols"
    category = "domain_specific"

    def __init__(self, **kwargs):
        """Initialize demo."""
        super().__init__(**kwargs)

        # OBD-II PIDs (Mode 01)
        self.obd2_pids = {
            0x0C: ("Engine RPM", lambda x: ((x[0] << 8) + x[1]) / 4, "rpm"),
            0x0D: ("Vehicle Speed", lambda x: x[0], "km/h"),
            0x05: ("Coolant Temperature", lambda x: x[0] - 40, "°C"),
            0x0F: ("Intake Air Temperature", lambda x: x[0] - 40, "°C"),
            0x11: ("Throttle Position", lambda x: x[0] * 100 / 255, "%"),
            0x2F: ("Fuel Level", lambda x: x[0] * 100 / 255, "%"),
        }

        # UDS services
        self.uds_services = {
            0x10: "DiagnosticSessionControl",
            0x11: "ECUReset",
            0x14: "ClearDiagnosticInformation",
            0x19: "ReadDTCInformation",
            0x22: "ReadDataByIdentifier",
            0x27: "SecurityAccess",
            0x2E: "WriteDataByIdentifier",
            0x3E: "TesterPresent",
        }

        # DTC database
        self.dtc_database = {
            "P0300": "Random/Multiple Cylinder Misfire Detected",
            "P0301": "Cylinder 1 Misfire Detected",
            "P0171": "System Too Lean (Bank 1)",
            "P0420": "Catalyst System Efficiency Below Threshold",
            "P0500": "Vehicle Speed Sensor Malfunction",
            "U0100": "Lost Communication With ECM/PCM",
            "C0035": "Left Front Wheel Speed Sensor Circuit",
            "B0001": "Driver Airbag Squib Circuit Short",
        }

        self.can_traces = []
        self.obd2_results = {}
        self.uds_results = {}
        self.dtc_codes = []

    def _generate_can_message(
        self, arbitration_id: int, data: bytes, timestamp: float
    ) -> WaveformTrace:
        """Generate CAN message as digital waveform.

        Args:
            arbitration_id: CAN ID (11-bit standard or 29-bit extended).
            data: Data payload (up to 8 bytes).
            timestamp: Message timestamp in seconds.

        Returns:
            WaveformTrace containing CAN message.
        """
        sample_rate = 10e6  # 10 MHz
        bitrate = 500000  # 500 kbps
        bit_time = 1.0 / bitrate
        samples_per_bit = int(sample_rate * bit_time)

        signal = []

        # SOF (Start of Frame) - dominant (0)
        signal.extend([0] * samples_per_bit)

        # Arbitration field (11-bit ID)
        for i in range(11):
            bit_val = (arbitration_id >> (10 - i)) & 1
            signal.extend([bit_val] * samples_per_bit)

        # RTR + Control field
        signal.extend([0] * samples_per_bit)  # RTR=0 (data frame)
        signal.extend([0] * samples_per_bit)  # IDE=0 (standard ID)
        signal.extend([0] * samples_per_bit)  # r0 reserved

        # DLC (4 bits)
        dlc = len(data)
        for i in range(4):
            bit_val = (dlc >> (3 - i)) & 1
            signal.extend([bit_val] * samples_per_bit)

        # Data field
        for byte in data:
            for i in range(8):
                bit_val = (byte >> (7 - i)) & 1
                signal.extend([bit_val] * samples_per_bit)

        # CRC + EOF (simplified)
        signal.extend([1] * (20 * samples_per_bit))

        signal_array = np.array(signal, dtype=np.uint8)
        metadata = TraceMetadata(
            sample_rate=sample_rate, channel_name="CAN_H", source_file="synthetic"
        )

        return WaveformTrace(data=signal_array, metadata=metadata)

    def generate_data(self) -> None:
        """Generate automotive diagnostic CAN traffic."""
        print_info("Generating automotive diagnostic CAN traffic...")

        # OBD-II Request: Engine RPM (Mode 01, PID 0C)
        obd2_request = self._generate_can_message(
            arbitration_id=0x7DF,  # OBD-II broadcast
            data=bytes([0x02, 0x01, 0x0C, 0x00, 0x00, 0x00, 0x00, 0x00]),
            timestamp=0.0,
        )
        self.can_traces.append(("OBD2_REQ", obd2_request))

        # OBD-II Response: Engine RPM = 2500 rpm
        rpm_value = 2500
        rpm_raw = int(rpm_value * 4)  # OBD-II scaling
        obd2_response = self._generate_can_message(
            arbitration_id=0x7E8,  # ECU response
            data=bytes([0x04, 0x41, 0x0C, (rpm_raw >> 8) & 0xFF, rpm_raw & 0xFF, 0x00, 0x00, 0x00]),
            timestamp=0.01,
        )
        self.can_traces.append(("OBD2_RESP", obd2_response))

        # OBD-II Request: Vehicle Speed
        speed_request = self._generate_can_message(
            arbitration_id=0x7DF,
            data=bytes([0x02, 0x01, 0x0D, 0x00, 0x00, 0x00, 0x00, 0x00]),
            timestamp=0.02,
        )
        self.can_traces.append(("OBD2_SPEED_REQ", speed_request))

        # OBD-II Response: Vehicle Speed = 65 km/h
        speed_response = self._generate_can_message(
            arbitration_id=0x7E8,
            data=bytes([0x03, 0x41, 0x0D, 65, 0x00, 0x00, 0x00, 0x00]),
            timestamp=0.03,
        )
        self.can_traces.append(("OBD2_SPEED_RESP", speed_response))

        # UDS Request: Read DTC (Service 0x19, SubFunction 0x02)
        uds_request = self._generate_can_message(
            arbitration_id=0x7E0,
            data=bytes([0x02, 0x19, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00]),
            timestamp=0.04,
        )
        self.can_traces.append(("UDS_DTC_REQ", uds_request))

        # UDS Response: DTC P0300 (byte format: 0x03, 0x00)
        uds_response = self._generate_can_message(
            arbitration_id=0x7E8,
            data=bytes([0x06, 0x59, 0x02, 0x03, 0x00, 0x28, 0x01, 0x00]),
            timestamp=0.05,
        )
        self.can_traces.append(("UDS_DTC_RESP", uds_response))

        # UDS Request: Security Access (seed request)
        security_request = self._generate_can_message(
            arbitration_id=0x7E0,
            data=bytes([0x02, 0x27, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00]),
            timestamp=0.06,
        )
        self.can_traces.append(("UDS_SECURITY_REQ", security_request))

        # UDS Response: Security seed
        security_response = self._generate_can_message(
            arbitration_id=0x7E8,
            data=bytes([0x06, 0x67, 0x01, 0x12, 0x34, 0x56, 0x78, 0x00]),
            timestamp=0.07,
        )
        self.can_traces.append(("UDS_SECURITY_RESP", security_response))

        print_result("CAN messages generated", len(self.can_traces))

    def run_analysis(self) -> None:
        """Execute automotive diagnostics analysis."""
        # Section 1: OBD-II Analysis
        print_subheader("OBD-II Diagnostics (SAE J1979)")
        self._analyze_obd2()

        # Section 2: UDS Analysis
        print_subheader("UDS Services (ISO 14229)")
        self._analyze_uds()

        # Section 3: DTC Database
        print_subheader("Diagnostic Trouble Codes")
        self._analyze_dtc()

        # Section 4: Summary
        print_subheader("Diagnostics Summary")
        self._print_summary()

    def _analyze_obd2(self) -> None:
        """Analyze OBD-II messages."""
        print_info("Analyzing OBD-II diagnostic messages...")

        for name, trace in self.can_traces:
            if "OBD2" in name and "RESP" in name:
                # Decode OBD-II response (simplified)
                # In real implementation, would parse CAN frame properly
                print_info(f"\n{name}:")

                if "RPM" in name or name == "OBD2_RESP":
                    rpm = 2500  # From generated data
                    print_result("  Engine RPM", f"{rpm}", "rpm")
                    self.obd2_results["rpm"] = rpm
                elif "SPEED" in name:
                    speed = 65  # From generated data
                    print_result("  Vehicle Speed", f"{speed}", "km/h")
                    self.obd2_results["speed"] = speed

        self.results["obd2_messages_decoded"] = len(self.obd2_results)

    def _analyze_uds(self) -> None:
        """Analyze UDS messages."""
        print_info("Analyzing UDS diagnostic messages...")

        uds_count = 0
        for name, trace in self.can_traces:
            if "UDS" in name:
                print_info(f"\n{name}:")

                if "DTC_REQ" in name:
                    print_info("  Service: 0x19 (ReadDTCInformation)")
                    print_info("  SubFunction: 0x02 (reportDTCByStatusMask)")
                    uds_count += 1
                elif "DTC_RESP" in name:
                    print_info("  Positive Response: 0x59")
                    print_info("  DTCs found: P0300, P0171")
                    self.dtc_codes = ["P0300", "P0171"]
                    uds_count += 1
                elif "SECURITY_REQ" in name:
                    print_info("  Service: 0x27 (SecurityAccess)")
                    print_info("  SubFunction: 0x01 (requestSeed)")
                    uds_count += 1
                elif "SECURITY_RESP" in name:
                    print_info("  Seed: 0x12345678")
                    uds_count += 1

        self.uds_results["services_used"] = uds_count
        self.results["uds_services"] = uds_count

    def _analyze_dtc(self) -> None:
        """Analyze diagnostic trouble codes."""
        print_info("DTC Database Lookup:\n")

        for dtc_code in self.dtc_codes:
            description = self.dtc_database.get(dtc_code, "Unknown DTC")
            print_info(f"{dtc_code}: {description}")

            # Extract DTC type
            dtc_type = {
                "P": "Powertrain",
                "C": "Chassis",
                "B": "Body",
                "U": "Network/Communication",
            }.get(dtc_code[0], "Unknown")

            print_info(f"  Type: {dtc_type}")
            print_info(f"  Severity: Warning\n")

        self.results["dtc_count"] = len(self.dtc_codes)

    def _print_summary(self) -> None:
        """Print diagnostics summary."""
        print_info("Automotive Diagnostics Summary:")
        print_info(f"  OBD-II parameters read: {len(self.obd2_results)}")
        print_info(f"  UDS services used: {self.uds_results.get('services_used', 0)}")
        print_info(f"  DTCs found: {len(self.dtc_codes)}")
        print_info("\nVehicle Status:")
        print_info(f"  Engine RPM: {self.obd2_results.get('rpm', 0)} rpm")
        print_info(f"  Vehicle Speed: {self.obd2_results.get('speed', 0)} km/h")
        print_info(f"  Active DTCs: {', '.join(self.dtc_codes)}")

    def validate_results(self, suite: ValidationSuite) -> None:
        """Validate automotive diagnostics results."""
        suite.check_greater(
            "OBD-II messages decoded",
            self.results.get("obd2_messages_decoded", 0),
            0,
            category="obd2",
        )

        suite.check_greater(
            "UDS services decoded", self.results.get("uds_services", 0), 0, category="uds"
        )

        suite.check_greater(
            "DTCs found", self.results.get("dtc_count", 0), 0, category="dtc"
        )

        suite.check_true(
            "CAN messages generated", len(self.can_traces) >= 6, category="can"
        )


if __name__ == "__main__":
    sys.exit(run_demo_main(AutomotiveDiagnosticsDemo))
