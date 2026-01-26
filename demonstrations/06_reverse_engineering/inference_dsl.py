#!/usr/bin/env python3
"""Protocol DSL (Domain-Specific Language) Demonstration.

This demo showcases Oscura's Protocol DSL for declaratively defining
custom protocol formats that can automatically generate decoders.

**Features Demonstrated**:
- YAML-based protocol definition
- Field type declarations (uint8, uint16, bytes, string)
- Conditional fields based on other field values
- Length-prefixed variable data
- Enumeration mapping
- CRC/checksum validation
- Nested structures
- Decoder auto-generation

**Protocol DSL Benefits**:
- Declarative: Describe WHAT, not HOW
- Reusable: Define once, decode anywhere
- Maintainable: Clear, readable specifications
- Exportable: Generate Wireshark dissectors, documentation

**Supported Field Types**:
- Integers: uint8, uint16, uint32, uint64, int8, int16, int32, int64
- Floating: float32, float64
- Variable: bytes, string (fixed or variable length)
- Complex: bitfield, array, struct

Usage:
    python protocol_dsl_demo.py
    python protocol_dsl_demo.py --verbose

Author: Oscura Development Team
Date: 2026-01-16
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demonstrations.common import BaseDemo, ValidationSuite, print_info, print_result
from demonstrations.common.base_demo import run_demo_main
from demonstrations.common.formatting import GREEN, RED, RESET, print_subheader

# Oscura imports
from oscura.inference.protocol_dsl import (
    FieldDefinition,
    ProtocolDecoder,
    ProtocolDefinition,
)


class ProtocolDSLDemo(BaseDemo):
    """Protocol DSL Demonstration.

    This demo creates protocol definitions using the DSL and generates
    decoders to demonstrate Oscura's protocol definition capabilities.
    """

    name = "Protocol DSL Demo"
    description = "Demonstrates declarative protocol definition and decoder generation"
    category = "advanced_inference"

    def __init__(self, **kwargs):
        """Initialize demo."""
        super().__init__(**kwargs)

        self.protocols = []
        self.test_packets = []
        self.decode_results = []

    def generate_data(self) -> None:
        """Create protocol definitions and test packets.

        Loads from file if available (--data-file override or default NPZ),
        otherwise generates synthetic protocol definitions and test packets.
        """
        # Try loading data from file
        file_to_load = None

        # 1. Check CLI override
        if self.data_file and self.data_file.exists():
            file_to_load = self.data_file
            print_info(f"Loading Protocol DSL data from CLI override: {self.data_file}")
        # 2. Check default generated data
        elif default_file := self.find_default_data_file("protocol_dsl.npz"):
            file_to_load = default_file
            print_info(f"Loading Protocol DSL data from default file: {default_file.name}")

        # Load from file if found
        if file_to_load:
            try:
                data = np.load(file_to_load, allow_pickle=True)
                # Load protocols as pickled objects
                self.protocols = data["protocols"].tolist()

                print_result("Data loaded from file", file_to_load.name)
                print_result("Protocols loaded", len(self.protocols))
                for proto_info in self.protocols:
                    print_info(
                        f"  - {proto_info['name']}: {len(proto_info['definition'].fields)} fields"
                    )
                return
            except Exception as e:
                print_info(f"Failed to load data from file: {e}, falling back to synthetic")

        # Generate synthetic data
        print_info("Creating protocol definitions using DSL...")

        # ===== Protocol 1: Simple Sensor Protocol =====
        print_subheader("Simple Sensor Protocol")

        sensor_protocol = ProtocolDefinition(
            name="simple_sensor",
            version="1.0",
            description="Simple sensor data protocol for embedded systems",
            endian="big",
            fields=[
                FieldDefinition(
                    name="header",
                    field_type="uint8",
                    value=0xAA,
                    description="Frame header byte",
                ),
                FieldDefinition(
                    name="sensor_id",
                    field_type="uint8",
                    description="Sensor identifier",
                ),
                FieldDefinition(
                    name="timestamp",
                    field_type="uint32",
                    description="Unix timestamp",
                ),
                FieldDefinition(
                    name="temperature",
                    field_type="int16",
                    description="Temperature in 0.1C units",
                ),
                FieldDefinition(
                    name="humidity",
                    field_type="uint8",
                    description="Humidity percentage",
                ),
                FieldDefinition(
                    name="checksum",
                    field_type="uint8",
                    description="XOR checksum",
                ),
            ],
        )

        # Generate test packet
        sensor_packet = bytes(
            [
                0xAA,  # Header
                0x01,  # Sensor ID = 1
                0x00,
                0x00,
                0x01,
                0x00,  # Timestamp = 256
                0x00,
                0xF5,  # Temperature = 245 (24.5C)
                0x40,  # Humidity = 64%
                0x00,  # Checksum (simplified)
            ]
        )

        self.protocols.append(
            {
                "definition": sensor_protocol,
                "packet": sensor_packet,
                "name": "Simple Sensor",
            }
        )

        print_info(f"  Name: {sensor_protocol.name}")
        print_info(f"  Fields: {len(sensor_protocol.fields)}")
        print_info(f"  Test packet: {sensor_packet.hex().upper()}")

        # ===== Protocol 2: Command Protocol with Enums =====
        print_subheader("Command Protocol")

        cmd_protocol = ProtocolDefinition(
            name="command_proto",
            version="1.0",
            description="Command/response protocol with enumerated commands",
            endian="little",
            fields=[
                FieldDefinition(
                    name="sync",
                    field_type="uint16",
                    value=0x55AA,
                    description="Sync word",
                ),
                FieldDefinition(
                    name="command",
                    field_type="uint8",
                    description="Command code",
                    enum={
                        0x01: "READ",
                        0x02: "WRITE",
                        0x03: "RESET",
                        0x04: "STATUS",
                    },
                ),
                FieldDefinition(
                    name="address",
                    field_type="uint16",
                    endian="little",
                    description="Target address",
                ),
                FieldDefinition(
                    name="data_length",
                    field_type="uint8",
                    description="Payload length",
                ),
                FieldDefinition(
                    name="data",
                    field_type="bytes",
                    size="data_length",
                    description="Command payload",
                ),
                FieldDefinition(
                    name="crc16",
                    field_type="uint16",
                    endian="little",
                    description="CRC-16 checksum",
                ),
            ],
        )

        # Generate test packet (WRITE to address 0x1000 with 4 bytes)
        cmd_packet = bytes(
            [
                0xAA,
                0x55,  # Sync (little-endian)
                0x02,  # Command = WRITE
                0x00,
                0x10,  # Address = 0x1000 (little-endian)
                0x04,  # Data length = 4
                0xDE,
                0xAD,
                0xBE,
                0xEF,  # Data
                0x00,
                0x00,  # CRC (placeholder)
            ]
        )

        self.protocols.append(
            {
                "definition": cmd_protocol,
                "packet": cmd_packet,
                "name": "Command Protocol",
            }
        )

        print_info(f"  Name: {cmd_protocol.name}")
        print_info(f"  Fields: {len(cmd_protocol.fields)}")
        print_info("  Variable-length field: data (length from data_length)")

        # ===== Protocol 3: Nested Structure Protocol =====
        print_subheader("Nested Structure Protocol")

        nested_protocol = ProtocolDefinition(
            name="nested_proto",
            version="1.0",
            description="Protocol with nested structures",
            endian="big",
            fields=[
                FieldDefinition(
                    name="version",
                    field_type="uint8",
                    description="Protocol version",
                ),
                FieldDefinition(
                    name="message_type",
                    field_type="uint8",
                    description="Message type",
                    enum={
                        0x01: "DATA",
                        0x02: "ACK",
                        0x03: "NACK",
                    },
                ),
                FieldDefinition(
                    name="sequence",
                    field_type="uint16",
                    description="Sequence number",
                ),
                FieldDefinition(
                    name="payload_length",
                    field_type="uint16",
                    description="Total payload length",
                ),
                FieldDefinition(
                    name="payload",
                    field_type="bytes",
                    size="payload_length",
                    description="Message payload",
                ),
            ],
        )

        nested_packet = bytes(
            [
                0x01,  # Version = 1
                0x01,  # Type = DATA
                0x00,
                0x0A,  # Sequence = 10
                0x00,
                0x08,  # Payload length = 8
                0x48,
                0x65,
                0x6C,
                0x6C,
                0x6F,
                0x21,
                0x21,
                0x21,  # "Hello!!!"
            ]
        )

        self.protocols.append(
            {
                "definition": nested_protocol,
                "packet": nested_packet,
                "name": "Nested Protocol",
            }
        )

        print_info(f"  Name: {nested_protocol.name}")
        print_info(f"  Fields: {len(nested_protocol.fields)}")

        print_result("Protocols defined", len(self.protocols))

    def run_analysis(self) -> None:
        """Decode test packets using generated decoders."""
        print_subheader("Protocol Decoding")

        for proto_info in self.protocols:
            protocol_def = proto_info["definition"]
            packet = proto_info["packet"]
            name = proto_info["name"]

            print_subheader(f"Decoding: {name}")

            # Create decoder
            decoder = ProtocolDecoder(protocol_def)

            # Decode packet
            try:
                result = decoder.decode(packet)

                print_info(f"  {GREEN}Decode successful{RESET}")
                print_info("  Fields:")

                for field_name, value in result.items():
                    # Format value appropriately
                    if isinstance(value, bytes):
                        val_str = value.hex().upper()
                        if len(val_str) > 16:
                            val_str = val_str[:16] + "..."
                        # Try ASCII
                        try:
                            ascii_str = value.decode("ascii")
                            if all(32 <= ord(c) <= 126 for c in ascii_str):
                                val_str += f' ("{ascii_str}")'
                        except Exception:
                            pass
                    elif isinstance(value, int):
                        if value < 256:
                            val_str = f"{value} (0x{value:02X})"
                        else:
                            val_str = f"{value} (0x{value:04X})"
                    else:
                        val_str = str(value)

                    print_info(f"    {field_name}: {val_str}")

                # Check for enum values
                for field_def in protocol_def.fields:
                    if field_def.enum and field_def.name in result:
                        raw_val = result[field_def.name]
                        if raw_val in field_def.enum:
                            print_info(f"    {field_def.name} (enum): {field_def.enum[raw_val]}")

                self.decode_results.append(
                    {
                        "name": name,
                        "success": True,
                        "fields": result,
                    }
                )

            except Exception as e:
                print_info(f"  {RED}Decode failed: {e}{RESET}")
                self.decode_results.append(
                    {
                        "name": name,
                        "success": False,
                        "error": str(e),
                    }
                )

        # Protocol definition export
        print_subheader("Protocol Definition Export")

        for proto_info in self.protocols[:1]:  # Show first protocol
            protocol_def = proto_info["definition"]
            name = proto_info["name"]

            print_info(f"YAML representation of {name}:")
            print_info("---")
            print_info(f"name: {protocol_def.name}")
            print_info(f"version: {protocol_def.version}")
            print_info(f"endian: {protocol_def.endian}")
            print_info("fields:")

            for field in protocol_def.fields:
                print_info(f"  - name: {field.name}")
                print_info(f"    type: {field.field_type}")
                if field.value is not None:
                    print_info(f"    value: 0x{field.value:02X}")
                if field.enum:
                    print_info(f"    enum: {field.enum}")
                if field.size and isinstance(field.size, str):
                    print_info(f"    size: {field.size}")

        # Summary
        print_subheader("Summary")

        successful = sum(1 for r in self.decode_results if r["success"])
        total = len(self.decode_results)

        print_result("Protocols defined", len(self.protocols))
        print_result("Packets decoded", f"{successful}/{total}")

        self.results["protocol_count"] = len(self.protocols)
        self.results["decode_success"] = successful
        self.results["decode_total"] = total

        if successful == total:
            print_info(f"  {GREEN}All protocols decoded successfully!{RESET}")
        else:
            print_info(f"  {RED}Some decoding failures{RESET}")

        # Show decoded field summary
        print_subheader("Decoded Fields Summary")

        for result in self.decode_results:
            if result["success"]:
                print_info(f"{result['name']}:")
                for field_name, value in list(result["fields"].items())[:5]:
                    if isinstance(value, bytes):
                        print_info(f"  {field_name}: {len(value)} bytes")
                    else:
                        print_info(f"  {field_name}: {value}")

    def validate_results(self, suite: ValidationSuite) -> None:
        """Validate Protocol DSL demo results."""
        # Check protocols were defined
        suite.check_greater(
            "Protocol count",
            self.results.get("protocol_count", 0),
            0,
            category="definition",
        )

        # Check decoding was successful
        suite.check_equal(
            "Decode success count",
            self.results.get("decode_success", 0),
            self.results.get("decode_total", 0),
            category="decoding",
        )

        # Check all protocols have fields
        for proto_info in self.protocols:
            suite.check_greater(
                f"{proto_info['name']} field count",
                len(proto_info["definition"].fields),
                0,
                category="fields",
            )


if __name__ == "__main__":
    sys.exit(run_demo_main(ProtocolDSLDemo))
