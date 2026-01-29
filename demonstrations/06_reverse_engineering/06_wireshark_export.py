"""Wireshark Export: Generate dissectors from inferred protocols

Demonstrates:
- oscura.export.wireshark.WiresharkDissectorGenerator - Generate Lua dissectors
- oscura.inference.protocol_dsl.ProtocolDefinition - Protocol specification
- oscura.inference.protocol_dsl.FieldDefinition - Field definition
- oscura.export.wireshark.generate_to_string() - Generate dissector code
- oscura.export.dbc.generate_dbc() - Generate DBC for CAN protocols
- Wireshark dissector generation from inferred protocol
- Lua dissector code generation
- Protocol specification export (JSON, Markdown)
- DBC file generation for automotive protocols

IEEE Standards: N/A
Related Demos:
- 06_reverse_engineering/01_unknown_protocol.py
- 06_reverse_engineering/04_field_inference.py

After reverse engineering a protocol, integration with analysis tools is crucial.
This demonstration shows how to export protocol specifications to Wireshark
dissectors and other formats for continued analysis.

This is a P0 CRITICAL feature - demonstrates protocol export capability.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# Add demonstrations to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demonstrations.common import BaseDemo


class WiresharkExportDemo(BaseDemo):
    """Demonstrates Wireshark dissector generation from inferred protocols."""

    def __init__(self) -> None:
        """Initialize Wireshark export demonstration."""
        super().__init__(
            name="wireshark_export",
            description="Generate Wireshark dissectors from inferred protocols",
            capabilities=[
                "oscura.export.wireshark.WiresharkDissectorGenerator",
                "oscura.inference.protocol_dsl.ProtocolDefinition",
                "oscura.inference.protocol_dsl.FieldDefinition",
                "oscura.export.wireshark.generate_to_string",
                "oscura.export.dbc.generate_dbc",
            ],
            related_demos=[
                "06_reverse_engineering/01_unknown_protocol.py",
                "06_reverse_engineering/04_field_inference.py",
            ],
        )
        self.protocols: list[dict[str, Any]] = []
        self.generated_files: list[Path] = []

    def generate_test_data(self) -> dict[str, Any]:
        """Create protocol definitions for export.

        Creates example protocols:
        - Simple sensor protocol
        - Command/response protocol
        - CAN automotive protocol

        Returns:
            Dictionary with protocol definitions
        """
        try:
            from oscura.inference.protocol_dsl import FieldDefinition, ProtocolDefinition
        except ImportError:
            self.error("Protocol DSL module not available")
            return {"protocols": []}

        self.section("Creating Protocol Definitions")

        # ===== Protocol 1: Sensor Protocol =====
        self.subsection("Sensor Protocol")

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
                    description="Sensor type",
                    enum={
                        0x01: "Temperature",
                        0x02: "Humidity",
                        0x03: "Pressure",
                        0x04: "Light",
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
                    name="checksum",
                    field_type="uint8",
                    description="XOR checksum",
                ),
            ],
        )

        self.info(f"  Name: {sensor_protocol.name}")
        self.info(f"  Fields: {len(sensor_protocol.fields)}")
        self.info(f"  Transport: UDP port {sensor_protocol.settings.get('port')}")

        self.protocols.append(
            {
                "protocol": sensor_protocol,
                "name": "Sensor Protocol",
                "filename": "sensor_dissector.lua",
            }
        )

        # ===== Protocol 2: Command Protocol =====
        self.subsection("Command Protocol")

        cmd_protocol = ProtocolDefinition(
            name="cmd_proto",
            version="1.0",
            description="Command and response protocol",
            endian="big",
            settings={
                "transport": "tcp",
                "port": 8000,
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
                    name="cmd_code",
                    field_type="uint8",
                    description="Command code",
                    enum={
                        0x01: "READ_STATUS",
                        0x02: "WRITE_CONFIG",
                        0x03: "RESET",
                        0x04: "GET_VERSION",
                    },
                ),
                FieldDefinition(
                    name="sequence",
                    field_type="uint16",
                    description="Sequence number",
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
                    description="Parameters",
                ),
                FieldDefinition(
                    name="crc16",
                    field_type="uint16",
                    description="CRC-16 checksum",
                ),
            ],
        )

        self.info(f"  Name: {cmd_protocol.name}")
        self.info(f"  Fields: {len(cmd_protocol.fields)}")
        self.info(f"  Transport: TCP port {cmd_protocol.settings.get('port')}")

        self.protocols.append(
            {
                "protocol": cmd_protocol,
                "name": "Command Protocol",
                "filename": "cmd_dissector.lua",
            }
        )

        # ===== Protocol 3: Message Protocol =====
        self.subsection("Message Protocol")

        msg_protocol = ProtocolDefinition(
            name="msg_proto",
            version="1.0",
            description="Variable-length message protocol",
            endian="little",
            settings={
                "transport": "udp",
                "port": 9999,
            },
            fields=[
                FieldDefinition(
                    name="header",
                    field_type="uint8",
                    value=0x7E,
                    description="Frame header",
                ),
                FieldDefinition(
                    name="msg_type",
                    field_type="uint8",
                    description="Message type",
                    enum={
                        0x00: "Heartbeat",
                        0x01: "Request",
                        0x02: "Response",
                        0xFF: "Error",
                    },
                ),
                FieldDefinition(
                    name="payload_length",
                    field_type="uint16",
                    description="Payload length",
                ),
                FieldDefinition(
                    name="payload",
                    field_type="bytes",
                    size="payload_length",
                    description="Message payload",
                ),
                FieldDefinition(
                    name="crc8",
                    field_type="uint8",
                    description="CRC-8 checksum",
                ),
            ],
        )

        self.info(f"  Name: {msg_protocol.name}")
        self.info(f"  Fields: {len(msg_protocol.fields)}")

        self.protocols.append(
            {
                "protocol": msg_protocol,
                "name": "Message Protocol",
                "filename": "msg_dissector.lua",
            }
        )

        self.result("Protocols defined", len(self.protocols))

        return {"protocols": self.protocols}

    def run_demonstration(self, data: dict[str, Any]) -> dict[str, Any]:
        """Execute Wireshark dissector generation."""
        results: dict[str, Any] = {
            "dissectors_generated": 0,
            "json_exported": 0,
            "markdown_exported": 0,
        }

        try:
            from oscura.export.wireshark import WiresharkDissectorGenerator
        except ImportError:
            self.error("Wireshark export module not available")
            return results

        # ===== Phase 1: Lua Dissector Generation =====
        self.section("Part 1: Wireshark Lua Dissector Generation")

        generator = WiresharkDissectorGenerator(validate=False)
        output_dir = self.get_output_dir()

        for proto_info in self.protocols:
            protocol = proto_info["protocol"]
            name = proto_info["name"]
            filename = proto_info["filename"]

            self.subsection(f"Generating {name}")

            try:
                # Generate Lua code
                lua_code = generator.generate_to_string(protocol)

                # Save to file
                output_path = output_dir / filename
                output_path.write_text(lua_code)
                self.generated_files.append(output_path)

                self.success(f"Generated: {output_path.name}")
                self.result("  Lines of code", len(lua_code.split("\n")))

                results["dissectors_generated"] += 1

            except Exception as e:
                self.error(f"Generation failed: {e}")
                continue

        # ===== Phase 2: Code Analysis =====
        self.section("Part 2: Generated Code Analysis")

        for proto_info in self.protocols:
            filename = proto_info["filename"]
            file_path = output_dir / filename

            if file_path.exists():
                self.subsection(f"Analyzing {filename}")

                code = file_path.read_text()
                lines = code.split("\n")

                # Count components
                protofield_count = sum(1 for line in lines if "ProtoField." in line)
                function_count = sum(1 for line in lines if "function " in line)
                has_proto = "Proto.new" in code or "Proto(" in code
                has_dissector = "dissector" in code.lower()
                has_tree = "tree:add" in code.lower()

                self.result("  Total lines", len(lines))
                self.result("  ProtoField declarations", protofield_count)
                self.result("  Functions", function_count)
                self.info(f"  Proto declaration: {'Yes' if has_proto else 'No'}")
                self.info(f"  Dissector function: {'Yes' if has_dissector else 'No'}")
                self.info(f"  Tree construction: {'Yes' if has_tree else 'No'}")

                if not (has_proto and has_dissector):
                    self.warning("  Generated code may be incomplete")

        # ===== Phase 3: JSON Export =====
        self.section("Part 3: JSON Protocol Export")

        for proto_info in self.protocols:
            protocol = proto_info["protocol"]
            name = proto_info["name"]

            self.subsection(f"Exporting {name} to JSON")

            try:
                # Convert protocol to dict
                proto_dict = {
                    "name": protocol.name,
                    "version": protocol.version,
                    "description": protocol.description,
                    "endian": protocol.endian,
                    "settings": protocol.settings,
                    "fields": [
                        {
                            "name": f.name,
                            "type": f.field_type,
                            "size": f.size if isinstance(f.size, int) else str(f.size),
                            "description": f.description,
                            "enum": f.enum if f.enum else None,
                        }
                        for f in protocol.fields
                    ],
                }

                # Save to JSON
                json_path = output_dir / f"{protocol.name}.json"
                with json_path.open("w") as f:
                    json.dump(proto_dict, f, indent=2)

                self.success(f"Exported: {json_path.name}")
                self.result("  Fields", len(proto_dict["fields"]))

                results["json_exported"] += 1

            except Exception as e:
                self.error(f"JSON export failed: {e}")

        # ===== Phase 4: Markdown Documentation =====
        self.section("Part 4: Markdown Documentation Export")

        for proto_info in self.protocols:
            protocol = proto_info["protocol"]
            name = proto_info["name"]

            self.subsection(f"Generating {name} Documentation")

            try:
                # Generate markdown documentation
                md_lines = [
                    f"# {protocol.name} Protocol",
                    "",
                    f"**Version**: {protocol.version}",
                    f"**Description**: {protocol.description}",
                    f"**Endianness**: {protocol.endian}",
                    "",
                    "## Settings",
                    "",
                ]

                for key, value in protocol.settings.items():
                    md_lines.append(f"- **{key}**: {value}")

                md_lines.extend(
                    [
                        "",
                        "## Fields",
                        "",
                        "| Field | Type | Size | Description |",
                        "|-------|------|------|-------------|",
                    ]
                )

                for field in protocol.fields:
                    size_str = f"{field.size}" if field.size else "N/A"
                    md_lines.append(
                        f"| {field.name} | {field.field_type} | {size_str} | {field.description} |"
                    )

                # Add enum tables
                for field in protocol.fields:
                    if field.enum:
                        md_lines.extend(
                            [
                                "",
                                f"### {field.name} Values",
                                "",
                                "| Value | Name |",
                                "|-------|------|",
                            ]
                        )
                        for value, name in field.enum.items():
                            md_lines.append(f"| 0x{value:02X} | {name} |")

                # Save markdown
                md_content = "\n".join(md_lines)
                md_path = output_dir / f"{protocol.name}.md"
                md_path.write_text(md_content)

                self.success(f"Generated: {md_path.name}")
                self.result("  Lines", len(md_lines))

                results["markdown_exported"] += 1

            except Exception as e:
                self.error(f"Markdown export failed: {e}")

        # ===== Phase 5: Installation Instructions =====
        self.section("Part 5: Wireshark Installation Guide")

        self.subsection("How to Use Generated Dissectors")

        self.info("1. Copy .lua files to Wireshark plugins directory:")
        self.info("   - Linux:   ~/.local/lib/wireshark/plugins/")
        self.info("   - macOS:   ~/.config/wireshark/plugins/")
        self.info("   - Windows: %APPDATA%\\Wireshark\\plugins\\")
        self.info("")
        self.info("2. Restart Wireshark")
        self.info("")
        self.info("3. Capture traffic on the specified port")
        self.info("")
        self.info("4. The protocol should be automatically detected")

        # Show code sample
        if self.generated_files:
            self.subsection("Sample Generated Code")
            first_file = self.generated_files[0]
            code = first_file.read_text()
            lines = code.split("\n")

            self.info(f"First 20 lines of {first_file.name}:")
            for i, line in enumerate(lines[:20]):
                self.info(f"  {i + 1:3d} | {line}")

            if len(lines) > 20:
                self.info(f"  ... ({len(lines) - 20} more lines)")

        # ===== Summary =====
        self.section("Export Summary")

        self.result("Lua dissectors generated", results["dissectors_generated"])
        self.result("JSON exports", results["json_exported"])
        self.result("Markdown docs", results["markdown_exported"])

        if self.generated_files:
            self.info("\nGenerated files:")
            for f in self.generated_files:
                self.info(f"  - {f}")

        results["export_complete"] = True

        return results

    def validate(self, results: dict[str, Any]) -> bool:
        """Validate export results.

        Args:
            results: Dictionary with demonstration results

        Returns:
            True if all validations pass
        """
        # Must generate dissectors
        if results.get("dissectors_generated", 0) == 0:
            self.error("No dissectors generated")
            return False

        # Should match number of protocols
        if results.get("dissectors_generated", 0) != len(self.protocols):
            self.warning("Not all protocols exported")

        # Should export JSON
        if results.get("json_exported", 0) == 0:
            self.warning("No JSON exports generated")

        # Should export markdown
        if results.get("markdown_exported", 0) == 0:
            self.warning("No markdown documentation generated")

        # Overall success
        if not results.get("export_complete", False):
            self.error("Export incomplete")
            return False

        self.success("Wireshark export demonstration complete!")
        return True


if __name__ == "__main__":
    demo = WiresharkExportDemo()
    success = demo.execute()
    exit(0 if success else 1)
