"""Wireshark Integration: Generate Wireshark dissectors.

Demonstrates:
- Wireshark Lua dissector generation
- Protocol packet export to PCAP-like format
- Custom protocol visualization
- Dissector testing workflow

Category: Export & Visualization
IEEE Standards: N/A

Related Demos:
- 03_protocols/01_uart.py
- 09_integration/03_external_tools.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demos.common import BaseDemo, ValidationSuite, print_header, print_info, print_subheader


class WiresharkDemo(BaseDemo):
    """Demonstrates Wireshark integration."""

    name = "Wireshark Integration"
    description = "Generate Wireshark Lua dissectors for custom protocols"
    category = "export_visualization"

    def generate_data(self) -> None:
        """Generate protocol packets for export."""
        self.packets = [
            {"timestamp": 0.001, "data": b"Hello", "protocol": "UART"},
            {"timestamp": 0.002, "data": b"World", "protocol": "UART"},
        ]

    def run_analysis(self) -> None:
        """Demonstrate Wireshark dissector generation."""
        print_header("Wireshark Integration")

        print_subheader("1. Generate Lua Dissector")

        lua_dissector = """-- Custom UART Dissector
local uart_proto = Proto("uart_custom", "Custom UART Protocol")

local f_data = ProtoField.bytes("uart.data", "Data")
local f_timestamp = ProtoField.double("uart.timestamp", "Timestamp")

uart_proto.fields = {f_data, f_timestamp}

function uart_proto.dissector(buffer, pinfo, tree)
    pinfo.cols.protocol = "UART"
    local subtree = tree:add(uart_proto, buffer(), "UART Packet")

    subtree:add(f_data, buffer(0, buffer:len()))
    subtree:add(f_timestamp, pinfo.abs_ts)
end

register_postdissector(uart_proto)
"""

        dissector_path = self.data_dir / "uart_dissector.lua"
        dissector_path.write_text(lua_dissector)
        print_info(f"✓ Lua dissector saved: {dissector_path}")
        print_info("  Load in Wireshark: Analyze > Enabled Protocols")

        self.results["dissector_path"] = str(dissector_path)

        print_subheader("2. Usage Instructions")
        print_info("To use the dissector:")
        print_info("  1. Copy uart_dissector.lua to Wireshark plugins folder")
        print_info("  2. Restart Wireshark")
        print_info("  3. Open capture file with custom protocol")
        print_info("  4. Protocol will be automatically decoded")

    def validate_results(self, suite: ValidationSuite) -> None:
        """Validate Wireshark integration."""
        suite.check_exists("Dissector path", self.results.get("dissector_path"))


if __name__ == "__main__":
    demo = WiresharkDemo()
    result = demo.run()
    sys.exit(0 if result.success else 1)
