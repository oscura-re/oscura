#!/usr/bin/env python3
"""Generate minimal synthetic PCAP files for testing.

Creates basic PCAP files with valid headers and minimal packet data.
Used for integration tests when real network captures aren't available.
"""

import struct
from pathlib import Path


def write_pcap_header(f):
    """Write PCAP global header."""
    # PCAP magic number (little-endian)
    f.write(struct.pack("<I", 0xA1B2C3D4))
    # Version 2.4
    f.write(struct.pack("<HH", 2, 4))
    # Timezone offset (0)
    f.write(struct.pack("<i", 0))
    # Timestamp accuracy (0)
    f.write(struct.pack("<I", 0))
    # Snapshot length (65535)
    f.write(struct.pack("<I", 65535))
    # Data link type (Ethernet = 1)
    f.write(struct.pack("<I", 1))


def write_packet(f, timestamp, data):
    """Write a PCAP packet record."""
    # Packet header
    # Timestamp seconds
    f.write(struct.pack("<I", timestamp))
    # Timestamp microseconds
    f.write(struct.pack("<I", 0))
    # Captured length
    f.write(struct.pack("<I", len(data)))
    # Original length
    f.write(struct.pack("<I", len(data)))
    # Packet data
    f.write(data)


def create_http_pcap(output_path: Path):
    """Create minimal HTTP PCAP for testing."""
    with open(output_path, "wb") as f:
        write_pcap_header(f)

        # Ethernet + IP + TCP + HTTP GET request
        eth_header = bytes.fromhex("ffffffffffff000000000000")  # MAC addresses
        eth_header += bytes.fromhex("0800")  # EtherType: IPv4

        ip_header = bytes.fromhex("4500")  # Version 4, IHL 5, TOS 0
        ip_header += bytes.fromhex("0050")  # Total length
        ip_header += bytes.fromhex("0000")  # Identification
        ip_header += bytes.fromhex("4000")  # Flags + fragment offset
        ip_header += bytes.fromhex("40")  # TTL
        ip_header += bytes.fromhex("06")  # Protocol: TCP
        ip_header += bytes.fromhex("0000")  # Header checksum
        ip_header += bytes.fromhex("c0a80101")  # Source IP: 192.168.1.1
        ip_header += bytes.fromhex("c0a80102")  # Dest IP: 192.168.1.2

        tcp_header = bytes.fromhex("c350")  # Source port: 50000
        tcp_header += bytes.fromhex("0050")  # Dest port: 80 (HTTP)
        tcp_header += bytes.fromhex("00000000")  # Sequence number
        tcp_header += bytes.fromhex("00000000")  # ACK number
        tcp_header += bytes.fromhex("5002")  # Data offset + flags (SYN)
        tcp_header += bytes.fromhex("7210")  # Window size
        tcp_header += bytes.fromhex("0000")  # Checksum
        tcp_header += bytes.fromhex("0000")  # Urgent pointer

        http_data = b"GET / HTTP/1.1\r\nHost: example.com\r\n\r\n"

        packet_data = eth_header + ip_header + tcp_header + http_data
        write_packet(f, 1000000, packet_data)

        # HTTP response packet
        http_response = b"HTTP/1.1 200 OK\r\nContent-Length: 13\r\n\r\nHello, World!"
        response_packet = eth_header + ip_header + tcp_header + http_response
        write_packet(f, 1000001, response_packet)


def create_modbus_pcap(output_path: Path):
    """Create minimal Modbus TCP PCAP for testing."""
    with open(output_path, "wb") as f:
        write_pcap_header(f)

        # Ethernet + IP + TCP + Modbus
        eth_header = bytes.fromhex("ffffffffffff000000000000")
        eth_header += bytes.fromhex("0800")

        ip_header = bytes.fromhex("45000040000040004006")
        ip_header += bytes.fromhex("0000c0a80101c0a80102")

        tcp_header = bytes.fromhex("c3500502")  # Ports: 50000 -> 502 (Modbus)
        tcp_header += bytes.fromhex("0000000000000000")
        tcp_header += bytes.fromhex("5002721000000000")

        # Modbus TCP request (Read Holding Registers)
        modbus_data = bytes.fromhex("000100000006")  # MBAP Header
        modbus_data += bytes.fromhex("01")  # Unit ID
        modbus_data += bytes.fromhex("03")  # Function code: Read Holding Registers
        modbus_data += bytes.fromhex("00000001")  # Start address: 0, quantity: 1

        packet_data = eth_header + ip_header + tcp_header + modbus_data
        write_packet(f, 2000000, packet_data)


def create_ftp_pcap(output_path: Path):
    """Create minimal FTP PCAP for testing."""
    with open(output_path, "wb") as f:
        write_pcap_header(f)

        # Ethernet + IP + TCP + FTP
        eth_header = bytes.fromhex("ffffffffffff000000000000")
        eth_header += bytes.fromhex("0800")

        ip_header = bytes.fromhex("45000040000040004006")
        ip_header += bytes.fromhex("0000c0a80101c0a80102")

        tcp_header = bytes.fromhex("c3500021")  # Ports: 50000 -> 21 (FTP)
        tcp_header += bytes.fromhex("0000000000000000")
        tcp_header += bytes.fromhex("5002721000000000")

        ftp_data = b"220 FTP Server Ready\r\n"

        packet_data = eth_header + ip_header + tcp_header + ftp_data
        write_packet(f, 3000000, packet_data)


def create_ssh_pcap(output_path: Path):
    """Create minimal SSH PCAP for testing."""
    with open(output_path, "wb") as f:
        write_pcap_header(f)

        # Ethernet + IP + TCP + SSH
        eth_header = bytes.fromhex("ffffffffffff000000000000")
        eth_header += bytes.fromhex("0800")

        ip_header = bytes.fromhex("45000040000040004006")
        ip_header += bytes.fromhex("0000c0a80101c0a80102")

        tcp_header = bytes.fromhex("c3500016")  # Ports: 50000 -> 22 (SSH)
        tcp_header += bytes.fromhex("0000000000000000")
        tcp_header += bytes.fromhex("5002721000000000")

        ssh_data = b"SSH-2.0-OpenSSH_8.0\r\n"

        packet_data = eth_header + ip_header + tcp_header + ssh_data
        write_packet(f, 4000000, packet_data)


def main():
    """Generate all test PCAP files."""
    base_dir = Path(__file__).parent.parent / "test_data" / "formats" / "pcap"

    pcaps = [
        (base_dir / "tcp" / "http" / "http.pcap", create_http_pcap),
        (base_dir / "industrial" / "modbus_tcp" / "modbus.pcap", create_modbus_pcap),
        (base_dir / "tcp" / "ftp" / "ftp.pcap", create_ftp_pcap),
        (base_dir / "tcp" / "ssh" / "ssh.pcap", create_ssh_pcap),
    ]

    print("Generating synthetic PCAP files for testing...")
    for pcap_path, generator_func in pcaps:
        pcap_path.parent.mkdir(parents=True, exist_ok=True)
        generator_func(pcap_path)
        print(f"✓ Created {pcap_path.relative_to(base_dir.parent.parent)}")

    print("\n✓ All PCAP files generated successfully")
    print("  These are minimal synthetic PCAPs for integration testing")
    print("  They contain valid PCAP headers and basic packet structures")


if __name__ == "__main__":
    main()
