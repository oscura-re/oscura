#!/usr/bin/env python3
"""Demonstration of Wireshark Lua dissector generation.

This example shows how to generate a functional Wireshark dissector from
a ProtocolSpec object obtained from reverse engineering workflows.

The generated dissector can be loaded into Wireshark for interactive
protocol analysis and debugging.
"""

from pathlib import Path

from oscura.export.wireshark_dissector import DissectorConfig, WiresharkDissectorGenerator
from oscura.workflows.reverse_engineering import FieldSpec, ProtocolSpec


def main() -> None:
    """Generate example Wireshark dissector."""
    # Create a protocol spec (normally from reverse engineering workflow)
    spec = ProtocolSpec(
        name="ExampleProtocol",
        baud_rate=115200,
        frame_format="8N1",
        sync_pattern="aa55",
        frame_length=16,
        fields=[
            # Header
            FieldSpec(
                name="sync",
                offset=0,
                size=2,
                field_type="bytes",
            ),
            FieldSpec(
                name="version",
                offset=2,
                size=1,
                field_type="uint8",
            ),
            FieldSpec(
                name="msg_type",
                offset=3,
                size=1,
                field_type="uint8",
            ),
            FieldSpec(
                name="length",
                offset=4,
                size=2,
                field_type="uint16",
            ),
            # Payload
            FieldSpec(
                name="data",
                offset=6,
                size=8,
                field_type="bytes",
            ),
            # Footer
            FieldSpec(
                name="crc",
                offset=14,
                size=2,
                field_type="uint16",
            ),
        ],
        checksum_type="crc16",
        checksum_position=-1,  # Last field
        confidence=0.95,
    )

    # Add enum values to msg_type field
    spec.fields[3].enum = {  # type: ignore[attr-defined]
        0x01: "DATA",
        0x02: "ACK",
        0x03: "NACK",
        0x04: "KEEPALIVE",
    }

    # Configure dissector generation
    config = DissectorConfig(
        protocol_name="ExampleProtocol",
        port=5000,  # Register on UDP/TCP port 5000
        include_crc_validation=True,
        generate_test_pcap=True,
        wireshark_version="3.0+",
    )

    # Generate dissector
    generator = WiresharkDissectorGenerator(config)

    # Sample messages for test PCAP
    sample_messages = [
        # Sync + Version + Type + Length + Data (8 bytes) + CRC
        b"\xaa\x55\x01\x01\x00\x08hello123\x12\x34",
        b"\xaa\x55\x01\x02\x00\x08world456\x56\x78",
        b"\xaa\x55\x01\x04\x00\x08keepaliv\x9a\xbc",
    ]

    # Generate dissector and test PCAP
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    lua_path, pcap_path = generator.generate(
        spec,
        sample_messages=sample_messages,
        output_path=output_dir / "example_protocol.lua",
    )

    print("\n✅ Wireshark dissector generated successfully!")
    print("\n📄 Generated files:")
    print(f"   - Lua dissector: {lua_path}")
    print(f"   - Test PCAP:     {pcap_path}")

    print("\n📋 Installation instructions:")
    print(f"   1. Copy {lua_path.name} to Wireshark plugins directory:")
    print("      - Linux:   ~/.local/lib/wireshark/plugins/")
    print("      - macOS:   ~/.config/wireshark/plugins/")
    print("      - Windows: %APPDATA%\\Wireshark\\plugins\\")
    print("   2. Restart Wireshark")
    print(f"   3. Open {pcap_path.name} in Wireshark")
    print("   4. Protocol should be automatically decoded as 'ExampleProtocol'")

    print("\n🔍 Testing the dissector:")
    print("   - UDP packets on port 5000 will be decoded")
    print("   - CRC validation will show 'Valid' or 'Invalid'")
    print("   - Message types will show as enum values (DATA, ACK, etc.)")

    # Show snippet of generated Lua code
    print("\n📝 Generated Lua code preview:")
    lua_code = lua_path.read_text()
    lines = lua_code.split("\n")[:30]
    for line in lines:
        print(f"   {line}")
    print(f"   ... ({len(lua_code.splitlines())} total lines)")


if __name__ == "__main__":
    main()
