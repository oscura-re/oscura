#!/usr/bin/env python3
"""Generate comprehensive set of PCAP files for integration tests."""

import struct
import time
from pathlib import Path


def write_pcap_header(f):
    """Write PCAP file header."""
    magic = 0xA1B2C3D4
    version_major, version_minor = 2, 4
    thiszone, sigfigs, snaplen, network = 0, 0, 65535, 1
    f.write(
        struct.pack(
            "<IHHIIII", magic, version_major, version_minor, thiszone, sigfigs, snaplen, network
        )
    )


def write_pcap_packet(f, timestamp, packet_data):
    """Write packet to PCAP."""
    ts_sec = int(timestamp)
    ts_usec = int((timestamp - ts_sec) * 1000000)
    f.write(struct.pack("<IIII", ts_sec, ts_usec, len(packet_data), len(packet_data)))
    f.write(packet_data)


def create_udp_packet(src_ip, dst_ip, src_port, dst_port, payload):
    """Create complete UDP packet."""
    src_mac, dst_mac = b"\x00\x11\x22\x33\x44\x55", b"\x00\xaa\xbb\xcc\xdd\xee"

    # Ethernet header
    eth = dst_mac + src_mac + struct.pack(">H", 0x0800)

    # IP header
    src_ip_bytes = bytes(map(int, src_ip.split(".")))
    dst_ip_bytes = bytes(map(int, dst_ip.split(".")))
    ip_len = 20 + 8 + len(payload)
    ip = struct.pack(">BBHHHBBH", 0x45, 0, ip_len, 0, 0, 64, 17, 0)
    ip += src_ip_bytes + dst_ip_bytes

    # UDP header
    udp = struct.pack(">HHHH", src_port, dst_port, 8 + len(payload), 0)

    return eth + ip + udp + payload


def generate_stream_reassembly_pcap(output: Path):
    """Generate PCAP for stream reassembly testing."""
    print(f"Generating stream reassembly PCAP...")

    with open(output, "wb") as f:
        write_pcap_header(f)
        base_time = time.time()

        # Stream 1: Fragmented message across multiple packets
        stream1_data = (
            b"This is a long message that will be split across multiple UDP packets for reassembly testing. "
            * 10
        )
        chunk_size = 200

        for i in range(0, len(stream1_data), chunk_size):
            chunk = stream1_data[i : i + chunk_size]
            seq_num = i // chunk_size
            payload = struct.pack(">HH", 1, seq_num) + chunk  # Stream ID, Sequence num
            packet = create_udp_packet("192.168.1.100", "192.168.1.200", 5000, 5001, payload)
            write_pcap_packet(f, base_time + seq_num * 0.001, packet)

        # Stream 2: Interleaved packets
        stream2_data = b"Stream 2 data for testing interleaved reassembly. " * 5
        for i in range(0, len(stream2_data), chunk_size):
            chunk = stream2_data[i : i + chunk_size]
            seq_num = i // chunk_size
            payload = struct.pack(">HH", 2, seq_num) + chunk
            packet = create_udp_packet("192.168.1.100", "192.168.1.200", 5002, 5003, payload)
            write_pcap_packet(f, base_time + seq_num * 0.001 + 0.0005, packet)

    size_kb = output.stat().st_size / 1024
    print(f"  ✓ {output.name} ({size_kb:.1f} KB)")


def generate_entropy_pcap(output: Path):
    """Generate PCAP for entropy analysis testing."""
    print(f"Generating entropy analysis PCAP...")

    with open(output, "wb") as f:
        write_pcap_header(f)
        base_time = time.time()

        # Low entropy packets (repeated pattern)
        for i in range(50):
            payload = b"\x00\x11\x22\x33" * 25  # Repeating pattern
            packet = create_udp_packet("192.168.1.100", "192.168.1.200", 6000, 6001, payload)
            write_pcap_packet(f, base_time + i * 0.01, packet)

        # High entropy packets (random-like data)
        for i in range(50, 100):
            # Pseudo-random data using simple PRNG
            seed = i * 12345
            payload = bytes([(seed * j) % 256 for j in range(100)])
            packet = create_udp_packet("192.168.1.100", "192.168.1.200", 6000, 6001, payload)
            write_pcap_packet(f, base_time + i * 0.01, packet)

    size_kb = output.stat().st_size / 1024
    print(f"  ✓ {output.name} ({size_kb:.1f} KB)")


def generate_segmented_pcap(output: Path):
    """Generate PCAP with segment markers for testing."""
    print(f"Generating segmented UDP PCAP...")

    with open(output, "wb") as f:
        write_pcap_header(f)
        base_time = time.time()

        # Create 5 segments, each with distinct patterns
        for segment in range(5):
            segment_marker = struct.pack(">HH", 0xFFFF, segment)  # Segment marker

            for pkt in range(20):
                payload = segment_marker
                payload += f"Segment {segment}, Packet {pkt}".encode().ljust(80, b"\x00")
                packet = create_udp_packet(
                    "192.168.1.100", "192.168.1.200", 7000 + segment, 7100 + segment, payload
                )
                write_pcap_packet(f, base_time + segment * 0.2 + pkt * 0.01, packet)

    size_kb = output.stat().st_size / 1024
    print(f"  ✓ {output.name} ({size_kb:.1f} KB)")


def generate_workflow_pcap(output: Path):
    """Generate PCAP for integration workflow testing."""
    print(f"Generating integration workflow PCAP...")

    with open(output, "wb") as f:
        write_pcap_header(f)
        base_time = time.time()

        # Simulate a protocol with request/response pairs
        for i in range(100):
            # Request
            request = struct.pack(">HHI", 0x1234, i, 0x00000001)  # Magic, seq, type=request
            request += f"GET /data/{i}".encode().ljust(50, b"\x00")
            packet = create_udp_packet("192.168.1.100", "192.168.1.200", 8000, 8001, request)
            write_pcap_packet(f, base_time + i * 0.02, packet)

            # Response
            response = struct.pack(">HHI", 0x1234, i, 0x00000002)  # Magic, seq, type=response
            response += f"OK: Data for request {i}".encode().ljust(50, b"\x00")
            packet = create_udp_packet("192.168.1.200", "192.168.1.100", 8001, 8000, response)
            write_pcap_packet(f, base_time + i * 0.02 + 0.005, packet)

    size_kb = output.stat().st_size / 1024
    print(f"  ✓ {output.name} ({size_kb:.1f} KB)")


def main():
    """Generate all test PCAP files."""
    output_dir = Path("test_data/real_captures/protocols")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Generating Synthetic PCAP Test Files")
    print("=" * 60)
    print()

    generate_stream_reassembly_pcap(output_dir / "udp_stream_reassembly.pcap")
    generate_entropy_pcap(output_dir / "udp_entropy_analysis.pcap")
    generate_segmented_pcap(output_dir / "udp_segments.pcap")
    generate_workflow_pcap(output_dir / "integration_workflow.pcap")

    print()
    print("=" * 60)
    print("PCAP Generation Complete!")
    print("=" * 60)
    total_size = sum(f.stat().st_size for f in output_dir.glob("*.pcap"))
    print(f"Total size: {total_size / 1024:.1f} KB")
    print(f"Files: {len(list(output_dir.glob('*.pcap')))}")


if __name__ == "__main__":
    main()
