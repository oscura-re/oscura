"""External Tools Integration: Tool integration (GDB, IDA, Wireshark, etc.).

Demonstrates:
- Wireshark dissector generation and export
- GDB integration for firmware debugging
- IDA Pro / Ghidra integration patterns
- Logic analyzer tool integration (Saleae, PulseView)
- Data exchange formats (PCAP, VCD, CSV)

Category: Integration
IEEE Standards: N/A

Related Demos:
- 06_reverse_engineering/06_wireshark_export.py
- 10_export_visualization/02_wireshark.py

This demonstrates how to integrate Oscura with external reverse engineering
and analysis tools for comprehensive hardware/firmware analysis workflows.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add demos to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np

from demos.common import BaseDemo, ValidationSuite, print_header, print_info, print_subheader


class ExternalToolsDemo(BaseDemo):
    """Demonstrates integration with external analysis tools."""

    name = "External Tools Integration"
    description = "Integrate with GDB, IDA, Wireshark, and other tools"
    category = "integration"

    def generate_data(self) -> None:
        """Generate test data for tool integration."""
        from oscura.core import TraceMetadata, WaveformTrace

        # Generate sample UART-like signal for export
        sample_rate = 115200 * 10  # 10x baud rate
        duration = 0.01
        num_samples = int(sample_rate * duration)

        # Simulate UART idle state with data bits
        data = np.ones(num_samples) * 3.3  # Idle high

        self.uart_trace = WaveformTrace(
            data=data,
            metadata=TraceMetadata(
                sample_rate=sample_rate,
                channel_name="UART_TX",
            ),
        )

    def run_analysis(self) -> None:
        """Demonstrate external tool integration."""
        print_header("External Tools Integration")

        print_subheader("1. Wireshark Integration")
        print_info("Generate Wireshark dissector for custom protocol:")

        # Example Wireshark dissector (Lua)
        dissector_lua = """
-- Custom Protocol Dissector
local custom_proto = Proto("custom", "Custom Protocol")

local f_sync = ProtoField.uint16("custom.sync", "Sync", base.HEX)
local f_type = ProtoField.uint8("custom.type", "Type", base.HEX)
local f_length = ProtoField.uint8("custom.length", "Length", base.DEC)
local f_payload = ProtoField.bytes("custom.payload", "Payload")

custom_proto.fields = {f_sync, f_type, f_length, f_payload}

function custom_proto.dissector(buffer, pinfo, tree)
    pinfo.cols.protocol = "CUSTOM"
    local subtree = tree:add(custom_proto, buffer(), "Custom Protocol")

    subtree:add(f_sync, buffer(0,2))
    subtree:add(f_type, buffer(2,1))
    subtree:add(f_length, buffer(3,1))
    local length = buffer(3,1):uint()
    subtree:add(f_payload, buffer(4,length))
end

DissectorTable.get("udp.port"):add(12345, custom_proto)
"""

        dissector_path = self.data_dir / "custom_dissector.lua"
        dissector_path.write_text(dissector_lua)
        print_info(f"✓ Dissector saved: {dissector_path}")
        print_info("  Load in Wireshark: Analyze > Enabled Protocols > Add Lua script")

        self.results["dissector_path"] = str(dissector_path)

        print_subheader("2. VCD Export for Logic Analyzers")
        print_info("Export to VCD format for PulseView/GTKWave:")

        vcd_content = """$version Oscura VCD Export $end
$timescale 1us $end
$scope module logic $end
$var wire 1 ! UART_TX $end
$upscope $end
$enddefinitions $end
#0
1!
#10
0!
#20
1!
"""
        vcd_path = self.data_dir / "export.vcd"
        vcd_path.write_text(vcd_content)
        print_info(f"✓ VCD saved: {vcd_path}")
        print_info("  Open with: pulseview export.vcd")

        self.results["vcd_path"] = str(vcd_path)

        print_subheader("3. GDB Integration Pattern")
        print_info("Use Oscura to correlate signal traces with firmware execution:")

        gdb_script = """
# GDB script to capture and correlate with Oscura
import subprocess
import json

def trace_function(function_name):
    '''Trace function entry/exit with timing.'''

    class FunctionTracer(gdb.Breakpoint):
        def __init__(self, location):
            super().__init__(location)
            self.silent = True
            self.timestamps = []

        def stop(self):
            import time
            self.timestamps.append({
                'time': time.time(),
                'function': function_name,
                'event': 'entry'
            })
            return False  # Don't stop execution

    return FunctionTracer(function_name)

# Example usage:
# tracer = trace_function('uart_send')
# Continue execution, then export timestamps for Oscura correlation
"""
        gdb_script_path = self.data_dir / "gdb_trace.py"
        gdb_script_path.write_text(gdb_script)
        print_info(f"✓ GDB script saved: {gdb_script_path}")
        print_info("  Use: gdb -x gdb_trace.py firmware.elf")

        self.results["gdb_script_path"] = str(gdb_script_path)

        print_subheader("4. IDA Pro / Ghidra Integration")
        print_info("Export protocol structure for reverse engineering:")

        ida_script = """
# IDA Python script to import Oscura protocol analysis

import idaapi
import idc

def import_oscura_protocol(json_path):
    '''Import protocol structure from Oscura analysis.'''
    import json

    with open(json_path) as f:
        protocol = json.load(f)

    # Create structures based on protocol fields
    sid = idc.add_struc(-1, protocol['name'], 0)

    for field in protocol['fields']:
        idc.add_struc_member(
            sid,
            field['name'],
            field['offset'],
            idc.FF_BYTE | idc.FF_DATA,
            -1,
            field['size']
        )

    print(f"Created structure: {protocol['name']}")

# Example usage:
# import_oscura_protocol('protocol_analysis.json')
"""
        ida_script_path = self.data_dir / "ida_import.py"
        ida_script_path.write_text(ida_script)
        print_info(f"✓ IDA script saved: {ida_script_path}")
        print_info("  Use: File > Script file > ida_import.py")

        self.results["ida_script_path"] = str(ida_script_path)

        print_subheader("5. PCAP Export for Wireshark")
        print_info("Convert protocol data to PCAP format:")

        # Simple PCAP file header (not full implementation)
        print_info("PCAP format:")
        print_info("  - Global header (24 bytes)")
        print_info("  - Packet header (16 bytes per packet)")
        print_info("  - Packet data (variable)")
        print_info("  See: https://wiki.wireshark.org/Development/LibpcapFileFormat")

        print_subheader("6. CSV Export for Spreadsheet Analysis")
        print_info("Export measurements to CSV:")

        csv_content = """timestamp,channel,value,measurement
0.000000,UART_TX,3.30,idle
0.000010,UART_TX,0.00,start_bit
0.000020,UART_TX,3.30,data_bit
0.000030,UART_TX,3.30,data_bit
"""
        csv_path = self.data_dir / "measurements.csv"
        csv_path.write_text(csv_content)
        print_info(f"✓ CSV saved: {csv_path}")

        self.results["csv_path"] = str(csv_path)

    def validate_results(self, suite: ValidationSuite) -> None:
        """Validate external tool integration results."""
        suite.check_exists("Dissector path", self.results.get("dissector_path"))
        suite.check_exists("VCD path", self.results.get("vcd_path"))
        suite.check_exists("GDB script path", self.results.get("gdb_script_path"))
        suite.check_exists("IDA script path", self.results.get("ida_script_path"))
        suite.check_exists("CSV path", self.results.get("csv_path"))


if __name__ == "__main__":
    demo = ExternalToolsDemo()
    result = demo.run()
    sys.exit(0 if result.success else 1)
