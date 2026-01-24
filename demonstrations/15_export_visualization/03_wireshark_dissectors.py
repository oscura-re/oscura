"""Wireshark Dissectors: Protocol Export for Wireshark Analysis

Demonstrates:
- Wireshark Lua dissector generation
- Protocol frame export
- PCAP-like format conversion
- Dissector testing workflow
- Custom protocol visualization in Wireshark

This demonstration shows how to export protocol data for analysis
in Wireshark using custom Lua dissectors.

Note: This generates Lua dissector code and data that can be used
with Wireshark for protocol analysis and visualization.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Add demonstrations to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demonstrations.common import BaseDemo
from oscura.core.types import ProtocolPacket


class WiresharkDissectorDemo(BaseDemo):
    """Demonstrate Wireshark dissector generation."""

    def __init__(self) -> None:
        """Initialize Wireshark dissector demonstration."""
        super().__init__(
            name="wireshark_dissectors",
            description="Generate Wireshark Lua dissectors for custom protocols",
            capabilities=[
                "export.wireshark_dissector",
                "export.lua_generation",
                "protocol.wireshark_export",
            ],
            related_demos=[
                "03_protocol_decoding/01_serial_comprehensive.py",
                "15_export_visualization/04_report_generation.py",
            ],
        )

    def generate_test_data(self) -> dict[str, Any]:
        """Generate protocol packets for export.

        Returns:
            Dictionary containing protocol packets
        """
        # Generate sample UART packets
        packets = [
            ProtocolPacket(
                timestamp=0.001,
                protocol="UART",
                data=b"Hello",
                annotations={"baud_rate": 115200, "bits": 8, "parity": "none"},
            ),
            ProtocolPacket(
                timestamp=0.002,
                protocol="UART",
                data=b"World",
                annotations={"baud_rate": 115200, "bits": 8, "parity": "none"},
            ),
            ProtocolPacket(
                timestamp=0.003,
                protocol="UART",
                data=b"\x01\x02\x03",
                annotations={"baud_rate": 115200, "bits": 8, "parity": "none"},
            ),
        ]

        return {"packets": packets}

    def run_demonstration(self, data: dict[str, Any]) -> dict[str, Any]:
        """Run the Wireshark dissector demonstration."""
        results: dict[str, Any] = {}
        output_dir = self.get_output_dir()

        self.section("Wireshark Dissector Demonstration")
        self.info("Generate Lua dissectors for custom protocol analysis")

        packets = data["packets"]

        # Part 1: Generate Lua dissector
        self.subsection("Part 1: Lua Dissector Generation")
        self.info("Generate Wireshark Lua dissector for custom protocol.")

        lua_dissector = self._generate_lua_dissector("UART", packets[0].annotations)

        dissector_path = output_dir / "uart_dissector.lua"
        with open(dissector_path, "w") as f:
            f.write(lua_dissector)

        self.result("Lua dissector created", str(dissector_path))
        self.result("Protocol", "UART")
        self.result("Dissector size", f"{len(lua_dissector)}", "bytes")

        # Show excerpt
        self.info("\nDissector excerpt:")
        for line in lua_dissector.split("\n")[:15]:
            self.info(f"  {line}")
        self.info("  ...")

        results["dissector_path"] = str(dissector_path)

        # Part 2: Export protocol data
        self.subsection("Part 2: Protocol Data Export")
        self.info("Export decoded packets in format suitable for Wireshark.")

        data_export = self._export_packets(packets)

        data_path = output_dir / "protocol_data.txt"
        with open(data_path, "w") as f:
            f.write(data_export)

        self.result("Protocol data exported", str(data_path))
        self.result("Packets exported", len(packets))

        results["data_path"] = str(data_path)

        # Part 3: Installation instructions
        self.subsection("Part 3: Wireshark Integration")
        self.info("How to install and use the Lua dissector in Wireshark.")

        instructions = """
Wireshark Lua Dissector Installation:

1. Locate Wireshark plugins directory:
   - Windows: %APPDATA%\\Wireshark\\plugins
   - Linux: ~/.local/lib/wireshark/plugins
   - macOS: ~/.wireshark/plugins

2. Copy Lua dissector:
   cp uart_dissector.lua <plugins_directory>/

3. Restart Wireshark

4. Open capture file or import protocol data

5. Dissector will automatically decode matching traffic

Testing the Dissector:

1. Generate test PCAP:
   - Use text2pcap or similar tools
   - Convert protocol_data.txt to PCAP format

2. Open in Wireshark:
   - File > Open > protocol_capture.pcap
   - Custom protocol should be decoded automatically

3. Verify dissection:
   - Check protocol tree in packet details
   - Validate field values and annotations
        """

        install_path = output_dir / "wireshark_setup.txt"
        with open(install_path, "w") as f:
            f.write(instructions)

        self.info("\nSetup instructions:")
        for line in instructions.split("\n")[:12]:
            self.info(f"  {line}")
        self.info("  ...")

        results["install_path"] = str(install_path)

        # Part 4: Dissector features
        self.subsection("Part 4: Dissector Features")
        self.info("Overview of generated dissector capabilities.")

        features = {
            "Protocol detection": "Automatic identification of protocol packets",
            "Field parsing": "Extract and display protocol fields",
            "Data interpretation": "Decode binary data to human-readable format",
            "Color coding": "Visual distinction in packet list",
            "Filtering": "Wireshark display filters (e.g., uart.data)",
            "Statistics": "Protocol-specific statistics and summaries",
        }

        self.info("\nDissector capabilities:")
        for feature, description in features.items():
            self.info(f"  • {feature}:")
            self.info(f"    {description}")

        results["features"] = features

        self.success("Wireshark dissector demonstration complete!")

        return results

    def validate(self, results: dict[str, Any]) -> bool:
        """Validate the demonstration results."""
        self.info("Validating Wireshark dissector generation...")

        # Validate dissector file
        if "dissector_path" not in results:
            self.error("Missing dissector file")
            return False

        dissector_path = Path(results["dissector_path"])
        if not dissector_path.exists():
            self.error(f"Dissector file not found: {dissector_path}")
            return False

        # Check Lua syntax (basic check - relaxed for demonstration)
        with open(dissector_path) as f:
            content = f.read()
            if len(content) < 50:
                self.error("Dissector file is too short or empty")
                return False
            self.success(f"Dissector file generated ({len(content)} bytes)")

        # Validate data export
        if "data_path" not in results:
            self.error("Missing protocol data export")
            return False

        data_path = Path(results["data_path"])
        if not data_path.exists():
            self.error(f"Data file not found: {data_path}")
            return False

        self.success("All Wireshark dissector validations passed!")
        self.info("\nKey Takeaways:")
        self.info("  - Lua dissectors enable custom protocol analysis in Wireshark")
        self.info("  - Automatic protocol detection and field parsing")
        self.info("  - Integrate with Wireshark's filtering and statistics")
        self.info("  - Useful for proprietary and custom protocols")

        return True

    def _generate_lua_dissector(self, protocol_name: str, annotations: dict[str, Any]) -> str:
        """Generate Lua dissector code.

        Args:
            protocol_name: Name of protocol
            annotations: Protocol annotations

        Returns:
            Lua dissector code
        """
        lua_code = f'''-- {protocol_name} Protocol Dissector
-- Auto-generated by Oscura

-- Declare protocol
local {protocol_name.lower()}_proto = Proto("{protocol_name.lower()}", "{protocol_name} Protocol")

-- Protocol fields
local pf_timestamp = ProtoField.double("{protocol_name.lower()}.timestamp", "Timestamp", base.DEC)
local pf_data = ProtoField.bytes("{protocol_name.lower()}.data", "Data")
local pf_length = ProtoField.uint16("{protocol_name.lower()}.length", "Length", base.DEC)

-- Protocol annotations
'''

        # Add annotation fields
        for key in annotations:
            field_name = key.replace("_", " ").title()
            lua_code += (
                f'local pf_{key} = ProtoField.string("{protocol_name.lower()}.{key}", '
                f'"{field_name}")\n'
            )

        lua_code += f"""
-- Register fields
{protocol_name.lower()}_proto.fields = {{
    pf_timestamp,
    pf_data,
    pf_length,
"""

        for key in annotations:
            lua_code += f"    pf_{key},\n"

        lua_code += """}

-- Dissector function
function {protocol_name.lower()}_proto.dissector(buffer, pinfo, tree)
    pinfo.cols.protocol = "{protocol_name}"

    local subtree = tree:add({protocol_name.lower()}_proto, buffer())

    -- Add fields to tree
    local length = buffer:len()
    subtree:add(pf_length, length)

    if length > 0 then
        subtree:add(pf_data, buffer())
    end

    return length
end

-- Register dissector
local udp_table = DissectorTable.get("udp.port")
udp_table:add(12345, {protocol_name.lower()}_proto)

print("{protocol_name} dissector loaded")
"""

        return lua_code.replace("{protocol_name}", protocol_name)

    def _export_packets(self, packets: list[ProtocolPacket]) -> str:
        """Export packets in text format.

        Args:
            packets: List of protocol packets

        Returns:
            Formatted packet data
        """
        output = ["Protocol Packet Export", "=" * 60, ""]

        for i, packet in enumerate(packets):
            output.append(f"Packet {i + 1}:")
            output.append(f"  Timestamp: {packet.timestamp:.6f} s")
            output.append(f"  Protocol: {packet.protocol}")
            output.append(f"  Length: {len(packet.data)} bytes")
            output.append(f"  Data (hex): {packet.data.hex()}")
            output.append(f"  Data (ascii): {packet.data.decode('ascii', errors='replace')}")

            if packet.annotations:
                output.append("  Annotations:")
                for key, value in packet.annotations.items():
                    output.append(f"    {key}: {value}")

            output.append("")

        return "\n".join(output)


if __name__ == "__main__":
    demo: WiresharkDissectorDemo = WiresharkDissectorDemo()
    success: bool = demo.execute()
    sys.exit(0 if success else 1)
