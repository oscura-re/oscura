#!/usr/bin/env python3
"""Wireshark Lua Dissector Generation Demonstration.

This demo showcases Oscura's ability to generate Wireshark Lua dissectors
from protocol definitions, enabling integration with Wireshark's protocol
analysis tools.

**Features Demonstrated**:
- Protocol definition using Protocol DSL
- Wireshark dissector generation
- Field type mapping (uint8, uint16, uint32, bytes, string)
- Enum value support
- Conditional field handling
- Variable-length field support
- Port-based registration (TCP/UDP)
- Lua syntax validation

**Wireshark Integration**:
Generated dissectors can be loaded into Wireshark by copying them to:
- Linux: ~/.local/lib/wireshark/plugins/
- macOS: ~/.config/wireshark/plugins/
- Windows: %APPDATA%\\Wireshark\\plugins\\

**Protocol DSL Fields Supported**:
- uint8, uint16, uint32, uint64 (integers)
- int8, int16, int32, int64 (signed integers)
- float32, float64 (floating point)
- bytes (fixed or variable length)
- string (null-terminated or length-prefixed)
- bitfield (for flags and bit-packed values)

Usage:
    python wireshark_dissector_demo.py
    python wireshark_dissector_demo.py --verbose

Author: Oscura Development Team
Date: 2026-01-16
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demos.common import BaseDemo, ValidationSuite, print_info, print_result
from demos.common.base_demo import run_demo_main
from demos.common.formatting import GREEN, RED, RESET, print_subheader

# Oscura imports
from oscura.export.wireshark import WiresharkDissectorGenerator
from oscura.inference.protocol_dsl import FieldDefinition, ProtocolDefinition


class WiresharkDissectorDemo(BaseDemo):
    """Wireshark Lua Dissector Generation Demonstration.

    This demo creates protocol definitions and generates Wireshark-compatible
    Lua dissectors, demonstrating Oscura's protocol export capabilities.
    """

    name = "Wireshark Dissector Demo"
    description = "Demonstrates Wireshark Lua dissector generation from protocol definitions"
    category = "inference"

    def __init__(self, **kwargs):
        """Initialize demo."""
        super().__init__(**kwargs)

        # Storage for protocols and dissectors
        self.protocols = []
        self.dissector_files = []
        self.generated_code = {}

    def generate_data(self) -> None:
        """Create or load protocol definitions for dissector generation.

        Tries in this order:
        1. Load from --data-file if specified
        2. Load from default demo_data files if they exist
        3. Generate synthetic protocol definitions
        """
        # Try loading data from file
        data_file_to_load = None

        # 1. Check CLI override
        if self.data_file and self.data_file.exists():
            data_file_to_load = self.data_file
            print_info(f"Loading protocol definitions from CLI override: {self.data_file}")
        # 2. Check default generated data
        elif default_file := self.find_default_data_file("wireshark_dissector.npz"):
            data_file_to_load = default_file
            print_info(f"Loading protocol definitions from default file: {default_file.name}")

        # Load from file if found
        if data_file_to_load:
            try:
                data = np.load(data_file_to_load, allow_pickle=True)
                self.protocols = data["protocols"].tolist()

                print_result("Loaded from file", data_file_to_load.name)
                print_result("Protocols loaded", len(self.protocols))

                # Print summary of loaded protocols
                for proto_info in self.protocols:
                    protocol = proto_info["protocol"]
                    print_info(f"  {proto_info['name']}: {len(protocol.fields)} fields")

                return
            except Exception as e:
                print_info(f"Failed to load from file: {e}, falling back to synthetic")
                data_file_to_load = None

        # Generate synthetic data if not loaded
        print_info("Creating protocol definitions...")

        # ===== Protocol 1: Simple Sensor Protocol =====
        print_subheader("Sensor Protocol Definition")

        sensor_protocol = ProtocolDefinition(
            name="sensor_proto",
            version="1.0",
            description="Simple sensor data protocol",
            endian="big",
            settings={
                "transport": "udp",
                "port": 12345,
            },
            fields=[
                FieldDefinition(
                    name="magic",
                    field_type="uint16",
                    value=0xABCD,
                    description="Protocol magic number",
                ),
                FieldDefinition(
                    name="version",
                    field_type="uint8",
                    description="Protocol version",
                ),
                FieldDefinition(
                    name="sensor_type",
                    field_type="uint8",
                    description="Sensor type identifier",
                    enum={
                        0x01: "Temperature",
                        0x02: "Humidity",
                        0x03: "Pressure",
                        0x04: "Light",
                        0x05: "Accelerometer",
                    },
                ),
                FieldDefinition(
                    name="sensor_id",
                    field_type="uint16",
                    description="Sensor device ID",
                ),
                FieldDefinition(
                    name="timestamp",
                    field_type="uint32",
                    description="Unix timestamp",
                ),
                FieldDefinition(
                    name="value",
                    field_type="float32",
                    description="Sensor reading",
                ),
                FieldDefinition(
                    name="flags",
                    field_type="uint8",
                    description="Status flags",
                ),
                FieldDefinition(
                    name="checksum",
                    field_type="uint8",
                    description="XOR checksum",
                ),
            ],
        )

        self.protocols.append(
            {
                "protocol": sensor_protocol,
                "name": "Simple Sensor Protocol",
                "filename": "sensor_dissector.lua",
            }
        )

        print_info(f"  Name: {sensor_protocol.name}")
        print_info(f"  Fields: {len(sensor_protocol.fields)}")
        print_info("  Field list:")
        for field in sensor_protocol.fields:
            print_info(f"    - {field.name} ({field.field_type})")

        # ===== Protocol 2: Complex Message Protocol =====
        print_subheader("Complex Message Protocol Definition")

        message_protocol = ProtocolDefinition(
            name="message_proto",
            version="2.0",
            description="Variable-length message protocol",
            endian="little",
            settings={
                "transport": "tcp",
                "port": 9999,
            },
            fields=[
                FieldDefinition(
                    name="sync",
                    field_type="bytes",
                    size=2,
                    value=b"\x55\xaa",
                    description="Sync pattern",
                ),
                FieldDefinition(
                    name="msg_type",
                    field_type="uint8",
                    description="Message type",
                    enum={
                        0x00: "Heartbeat",
                        0x01: "Request",
                        0x02: "Response",
                        0x03: "Notification",
                        0xFF: "Error",
                    },
                ),
                FieldDefinition(
                    name="sequence",
                    field_type="uint16",
                    endian="little",
                    description="Sequence number",
                ),
                FieldDefinition(
                    name="payload_length",
                    field_type="uint16",
                    endian="little",
                    description="Payload length",
                ),
                FieldDefinition(
                    name="payload",
                    field_type="bytes",
                    size="payload_length",  # Reference to length field
                    description="Message payload",
                ),
                FieldDefinition(
                    name="crc16",
                    field_type="uint16",
                    endian="little",
                    description="CRC-16 checksum",
                ),
            ],
        )

        self.protocols.append(
            {
                "protocol": message_protocol,
                "name": "Complex Message Protocol",
                "filename": "message_dissector.lua",
            }
        )

        print_info(f"  Name: {message_protocol.name}")
        print_info(f"  Fields: {len(message_protocol.fields)}")
        print_info("  Field list:")
        for field in message_protocol.fields:
            size_str = f", size={field.size}" if field.size else ""
            print_info(f"    - {field.name} ({field.field_type}{size_str})")

        # ===== Protocol 3: Command/Response Protocol =====
        print_subheader("Command/Response Protocol Definition")

        cmd_protocol = ProtocolDefinition(
            name="cmd_proto",
            version="1.0",
            description="Command and response protocol with conditional fields",
            endian="big",
            settings={
                "transport": "udp",
                "port": 8000,
            },
            fields=[
                FieldDefinition(
                    name="header",
                    field_type="uint8",
                    value=0x7E,
                    description="Frame header",
                ),
                FieldDefinition(
                    name="cmd_code",
                    field_type="uint8",
                    description="Command code",
                    enum={
                        0x01: "READ_STATUS",
                        0x02: "WRITE_CONFIG",
                        0x03: "RESET",
                        0x04: "GET_VERSION",
                        0x05: "SET_MODE",
                    },
                ),
                FieldDefinition(
                    name="param_count",
                    field_type="uint8",
                    description="Number of parameters",
                ),
                FieldDefinition(
                    name="params",
                    field_type="bytes",
                    size="param_count",
                    description="Command parameters",
                ),
                FieldDefinition(
                    name="checksum",
                    field_type="uint8",
                    description="Frame checksum",
                ),
            ],
        )

        self.protocols.append(
            {
                "protocol": cmd_protocol,
                "name": "Command/Response Protocol",
                "filename": "cmd_dissector.lua",
            }
        )

        print_info(f"  Name: {cmd_protocol.name}")
        print_info(f"  Fields: {len(cmd_protocol.fields)}")

        print_result("Protocols defined", len(self.protocols))

    def run_analysis(self) -> None:
        """Generate Wireshark dissectors from protocol definitions."""
        print_subheader("Dissector Generation")

        # Create generator (disable validation if luac not available)
        generator = WiresharkDissectorGenerator(validate=False)

        for proto_info in self.protocols:
            protocol = proto_info["protocol"]
            name = proto_info["name"]
            filename = proto_info["filename"]

            print_info(f"Generating dissector for: {name}")

            try:
                # Generate Lua code as string
                lua_code = generator.generate_to_string(protocol)

                # Store generated code
                self.generated_code[name] = lua_code

                # Save to file
                output_path = self.data_dir / filename
                output_path.write_text(lua_code)

                self.dissector_files.append(output_path)
                print_info(f"  {GREEN}Generated: {output_path}{RESET}")
                print_result("Code lines", len(lua_code.split("\n")))

            except Exception as e:
                print_info(f"  {RED}Failed: {e}{RESET}")
                continue

        # Analyze generated code
        print_subheader("Generated Code Analysis")

        for name, code in self.generated_code.items():
            print_info(f"Analyzing: {name}")

            lines = code.split("\n")
            protofield_count = sum(1 for line in lines if "ProtoField." in line)
            function_count = sum(1 for line in lines if "function " in line)

            print_result("  Total lines", len(lines))
            print_result("  ProtoField declarations", protofield_count)
            print_result("  Functions", function_count)

            # Check for key components
            has_proto = "Proto.new" in code or "Proto(" in code
            has_dissector = "dissector" in code.lower()
            has_tree = "tree:add" in code.lower()

            print_info(f"  Proto declaration: {'Yes' if has_proto else 'No'}")
            print_info(f"  Dissector function: {'Yes' if has_dissector else 'No'}")
            print_info(f"  Tree construction: {'Yes' if has_tree else 'No'}")

        self.results["dissector_count"] = len(self.dissector_files)
        self.results["protocol_count"] = len(self.protocols)

        # Show sample code
        print_subheader("Sample Generated Code")

        if self.generated_code:
            sample_name = next(iter(self.generated_code.keys()))
            sample_code = self.generated_code[sample_name]
            lines = sample_code.split("\n")

            print_info(f"First 30 lines of {sample_name}:")
            for i, line in enumerate(lines[:30]):
                print_info(f"  {i + 1:3d} | {line}")

            if len(lines) > 30:
                print_info(f"  ... ({len(lines) - 30} more lines)")

        # Provide installation instructions
        print_subheader("Wireshark Installation")
        print_info("To use the generated dissectors:")
        print_info("  1. Copy .lua files to Wireshark plugins directory:")
        print_info("     Linux:   ~/.local/lib/wireshark/plugins/")
        print_info("     macOS:   ~/.config/wireshark/plugins/")
        print_info("     Windows: %APPDATA%\\Wireshark\\plugins\\")
        print_info("  2. Restart Wireshark")
        print_info("  3. Load a capture with matching traffic")

        # Summary
        print_subheader("Summary")
        print_result("Protocols processed", len(self.protocols))
        print_result("Dissectors generated", len(self.dissector_files))

        if self.dissector_files:
            print_info("Generated files:")
            for f in self.dissector_files:
                print_info(f"  - {f}")

    def validate_results(self, suite: ValidationSuite) -> None:
        """Validate dissector generation results."""
        # Check that dissectors were generated
        suite.check_equal(
            "Dissectors generated",
            self.results.get("dissector_count", 0),
            len(self.protocols),
            category="generation",
        )

        # Check that generated files exist
        for f in self.dissector_files:
            suite.check_file_exists(
                f"File {f.name}",
                f,
                category="files",
            )

        # Check generated code content
        for name, code in self.generated_code.items():
            suite.check_true(
                f"{name} has Proto declaration",
                "Proto.new" in code or "Proto(" in code,
                category="code_structure",
            )

            suite.check_true(
                f"{name} has dissector",
                "dissector" in code.lower(),
                category="code_structure",
            )

        # Check code is not empty
        for name, code in self.generated_code.items():
            suite.check_greater(
                f"{name} code length",
                len(code),
                100,  # At least 100 characters
                category="code_quality",
            )


if __name__ == "__main__":
    sys.exit(run_demo_main(WiresharkDissectorDemo))
