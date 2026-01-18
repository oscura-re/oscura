#!/usr/bin/env python3
"""Generate synthetic UDP PCAP file for testing."""

import struct
import time
from pathlib import Path


def write_pcap_header(f):
    """Write PCAP file header."""
    magic = 0xA1B2C3D4  # PCAP magic number
    version_major = 2
    version_minor = 4
    thiszone = 0
    sigfigs = 0
    snaplen = 65535
    network = 1  # Ethernet

    f.write(
        struct.pack(
            "<IHHIIII", magic, version_major, version_minor, thiszone, sigfigs, snaplen, network
        )
    )


def write_pcap_packet(f, timestamp, packet_data):
    """Write a single packet to PCAP."""
    ts_sec = int(timestamp)
    ts_usec = int((timestamp - ts_sec) * 1000000)
    incl_len = len(packet_data)
    orig_len = len(packet_data)

    f.write(struct.pack("<IIII", ts_sec, ts_usec, incl_len, orig_len))
    f.write(packet_data)


def create_ethernet_header(src_mac, dst_mac, eth_type=0x0800):
    """Create Ethernet header."""
    return dst_mac + src_mac + struct.pack(">H", eth_type)


def create_ip_header(src_ip, dst_ip, protocol=17, payload_len=0):
    """Create IPv4 header."""
    version_ihl = 0x45  # Version 4, header length 5 (20 bytes)
    tos = 0
    total_len = 20 + payload_len
    identification = 0
    flags_offset = 0
    ttl = 64
    checksum = 0  # Simplified - not calculating real checksum

    # Pack IP addresses
    src_ip_bytes = bytes(map(int, src_ip.split(".")))
    dst_ip_bytes = bytes(map(int, dst_ip.split(".")))

    header = struct.pack(
        ">BBHHHBBH",
        version_ihl,
        tos,
        total_len,
        identification,
        flags_offset,
        ttl,
        protocol,
        checksum,
    )
    header += src_ip_bytes + dst_ip_bytes

    return header


def create_udp_header(src_port, dst_port, payload_len):
    """Create UDP header."""
    length = 8 + payload_len
    checksum = 0  # Simplified

    return struct.pack(">HHHH", src_port, dst_port, length, checksum)


def create_udp_packet(src_ip, dst_ip, src_port, dst_port, payload):
    """Create complete UDP packet."""
    # MAC addresses (dummy)
    src_mac = b"\x00\x11\x22\x33\x44\x55"
    dst_mac = b"\x00\xaa\xbb\xcc\xdd\xee"

    # Build packet layers
    eth_header = create_ethernet_header(src_mac, dst_mac)
    udp_header = create_udp_header(src_port, dst_port, len(payload))
    ip_header = create_ip_header(src_ip, dst_ip, 17, 8 + len(payload))

    return eth_header + ip_header + udp_header + payload


def generate_test_pcap(output_path: Path, num_packets: int = 100):
    """Generate synthetic UDP PCAP with various patterns."""
    print(f"Generating {num_packets} UDP packets...")

    with open(output_path, "wb") as f:
        write_pcap_header(f)

        base_time = time.time()

        # Generate different types of UDP traffic
        for i in range(num_packets):
            timestamp = base_time + i * 0.01  # 10ms intervals

            # Vary the traffic patterns
            if i % 10 == 0:
                # DNS-like pattern (port 53)
                payload = b"\x00\x01"  # Transaction ID
                payload += b"\x01\x00"  # Flags
                payload += b"\x00\x01\x00\x00\x00\x00\x00\x00"  # Counts
                payload += b"\x03www\x06google\x03com\x00"  # Query
                payload += b"\x00\x01\x00\x01"  # Type A, Class IN

                packet = create_udp_packet("192.168.1.100", "8.8.8.8", 12345 + i, 53, payload)

            elif i % 7 == 0:
                # NTP-like pattern (port 123)
                payload = b"\x23" + b"\x00" * 47  # NTP packet

                packet = create_udp_packet("192.168.1.100", "129.6.15.28", 12345 + i, 123, payload)

            elif i % 5 == 0:
                # Custom binary protocol
                payload = b"\xaa\xbb"  # Header magic
                payload += struct.pack(">H", i)  # Sequence number
                payload += struct.pack(">I", len(b"Hello World"))  # Length
                payload += b"Hello World"  # Data
                payload += b"\xcc\xdd"  # Footer

                packet = create_udp_packet("192.168.1.100", "192.168.1.200", 5000, 5001, payload)

            else:
                # Generic data
                payload = f"Packet {i}: Test data with some content".encode()
                packet = create_udp_packet(
                    "192.168.1.100", "192.168.1.200", 10000 + i, 20000 + i, payload
                )

            write_pcap_packet(f, timestamp, packet)

    print(f"✓ Generated PCAP: {output_path}")
    print(f"  • Packets: {num_packets}")
    print(f"  • Size: {output_path.stat().st_size:,} bytes")


if __name__ == "__main__":
    output = Path("test_data/synthetic/udp_test.pcap")
    output.parent.mkdir(parents=True, exist_ok=True)
    generate_test_pcap(output, num_packets=200)
