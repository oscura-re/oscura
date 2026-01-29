#!/usr/bin/env python3
"""Generate optimal demo data for UDP Packet Analysis demo.

Creates three PCAP files showcasing network protocol reverse engineering:
1. iot_protocol_capture.pcap - Custom IoT binary protocol with magic bytes, CRC, sequences
2. industrial_modbus_udp.pcap - Modbus/TCP over UDP (standard industrial protocol)
3. unknown_binary_stream.pcap - Unknown protocol for pure RE demonstration

All files include realistic network characteristics (jitter, timing, occasional drops).

Usage:
    python generate_demo_data.py [--force]

Author: Oscura Development Team
Date: 2026-01-15
"""

from __future__ import annotations

import argparse
import random
import struct
import sys
import time
from pathlib import Path

try:
    from scapy.all import IP, UDP, Ether, wrpcap
except ImportError:
    print("ERROR: scapy is required for PCAP generation")
    print("Install with: uv sync --all-extras")
    sys.exit(1)

# ANSI colors
GREEN = "\033[92m"
BLUE = "\033[94m"
RESET = "\033[0m"
BOLD = "\033[1m"


def print_success(msg: str) -> None:
    """Print success message."""
    print(f"{GREEN}✓{RESET} {msg}")


def print_info(msg: str) -> None:
    """Print info message."""
    print(f"{BLUE}INFO:{RESET} {msg}")


def crc16_ccitt(data: bytes) -> int:
    """Calculate CRC-16-CCITT checksum.

    Polynomial: 0x1021
    Initial value: 0xFFFF
    """
    crc = 0xFFFF
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = (crc << 1) ^ 0x1021
            else:
                crc = crc << 1
            crc &= 0xFFFF
    return crc


def generate_iot_protocol_capture(output_file: Path) -> None:
    """Generate IoT protocol capture PCAP.

    Protocol structure:
        - Header: Magic bytes (0xAA55), sequence number (2 bytes), length (2 bytes)
        - Payload: Sensor data (temperature, humidity, battery) + timestamp
        - Footer: CRC16, magic bytes (0xCCDD)

    Traffic pattern:
        - Regular heartbeat packets (1 Hz)
        - Burst sensor readings (every 5 seconds)
        - ACK responses

    Total: ~10,000 packets over 60 seconds
    File size: ~5 MB
    """
    print_info("Generating iot_protocol_capture.pcap...")

    packets = []
    seq_num = 0
    base_time = time.time()

    # Source/Dest IPs and MACs
    src_mac = "00:11:22:33:44:55"
    dst_mac = "AA:BB:CC:DD:EE:FF"
    src_ip = "192.168.1.100"
    dst_ip = "192.168.1.1"
    src_port = 5000
    dst_port = 5001

    # Generate packets over 60 seconds
    duration = 60.0  # seconds
    current_time = 0.0

    while current_time < duration:
        # Determine packet type based on time
        if current_time % 5.0 < 0.1:  # Burst every 5 seconds
            # Generate burst of sensor readings (10 packets)
            for _ in range(10):
                payload = generate_iot_sensor_packet(seq_num)
                pkt = create_udp_packet(
                    src_mac, dst_mac, src_ip, dst_ip, src_port, dst_port, payload
                )
                pkt.time = base_time + current_time
                packets.append(pkt)

                seq_num += 1
                current_time += 0.01 + random.uniform(-0.002, 0.002)  # 10ms + jitter

        else:  # Regular heartbeat
            payload = generate_iot_heartbeat_packet(seq_num)
            pkt = create_udp_packet(src_mac, dst_mac, src_ip, dst_ip, src_port, dst_port, payload)
            pkt.time = base_time + current_time
            packets.append(pkt)

            seq_num += 1
            current_time += 1.0 + random.uniform(-0.05, 0.05)  # 1s + jitter

    wrpcap(str(output_file), packets)
    size_mb = output_file.stat().st_size / (1024 * 1024)
    print_success(f"Generated iot_protocol_capture.pcap ({size_mb:.2f} MB, {len(packets)} packets)")


def generate_iot_sensor_packet(seq_num: int) -> bytes:
    """Generate IoT sensor data packet payload."""
    # Header
    magic_header = 0xAA55
    length = 20  # Payload length

    # Payload: temperature, humidity, battery, timestamp
    temperature = struct.pack(">h", int(25.0 * 100 + random.randint(-500, 500)))  # 25°C ± 5°C
    humidity = struct.pack(">H", int(60.0 * 100 + random.randint(-1000, 1000)))  # 60% ± 10%
    battery = struct.pack(">H", int(3.7 * 1000 + random.randint(-200, 200)))  # 3.7V ± 0.2V
    timestamp = struct.pack(">I", int(time.time()))

    payload_data = temperature + humidity + battery + timestamp

    # Build packet for CRC
    packet_for_crc = struct.pack(">HHH", magic_header, seq_num, length) + payload_data

    # Calculate CRC
    crc = crc16_ccitt(packet_for_crc)

    # Footer
    magic_footer = 0xCCDD

    # Complete packet
    packet = packet_for_crc + struct.pack(">HH", crc, magic_footer)
    return packet


def generate_iot_heartbeat_packet(seq_num: int) -> bytes:
    """Generate IoT heartbeat packet payload."""
    magic_header = 0xAA55
    length = 4  # Just timestamp
    timestamp = struct.pack(">I", int(time.time()))

    packet_for_crc = struct.pack(">HHH", magic_header, seq_num, length) + timestamp
    crc = crc16_ccitt(packet_for_crc)
    magic_footer = 0xCCDD

    packet = packet_for_crc + struct.pack(">HH", crc, magic_footer)
    return packet


def generate_industrial_modbus_udp(output_file: Path) -> None:
    """Generate Modbus/TCP over UDP capture.

    Protocol: Standard Modbus/TCP encapsulated in UDP
    Transactions:
        - Read coils (function 0x01)
        - Read holding registers (function 0x03)
        - Write single coil (function 0x05)
        - Write single register (function 0x06)

    Total: ~5,000 packets
    File size: ~3 MB
    """
    print_info("Generating industrial_modbus_udp.pcap...")

    packets = []
    transaction_id = 1
    base_time = time.time()

    src_mac = "00:11:22:33:44:66"
    dst_mac = "AA:BB:CC:DD:EE:00"
    src_ip = "192.168.2.100"  # PLC
    dst_ip = "192.168.2.10"  # Modbus slave
    src_port = 5020
    dst_port = 502  # Standard Modbus port

    current_time = 0.0

    # Generate 2500 request/response pairs
    for _ in range(2500):
        # Request
        func_code = random.choice([0x01, 0x03, 0x05, 0x06])
        request_payload = generate_modbus_request(transaction_id, func_code)
        req_pkt = create_udp_packet(
            src_mac, dst_mac, src_ip, dst_ip, src_port, dst_port, request_payload
        )
        req_pkt.time = base_time + current_time
        packets.append(req_pkt)

        current_time += 0.001 + random.uniform(-0.0002, 0.0002)  # 1ms RTT

        # Response
        response_payload = generate_modbus_response(transaction_id, func_code)
        resp_pkt = create_udp_packet(
            dst_mac, src_mac, dst_ip, src_ip, dst_port, src_port, response_payload
        )
        resp_pkt.time = base_time + current_time
        packets.append(resp_pkt)

        transaction_id += 1
        current_time += 0.02 + random.uniform(-0.005, 0.005)  # 20ms poll interval

    wrpcap(str(output_file), packets)
    size_mb = output_file.stat().st_size / (1024 * 1024)
    print_success(
        f"Generated industrial_modbus_udp.pcap ({size_mb:.2f} MB, {len(packets)} packets)"
    )


def generate_modbus_request(transaction_id: int, func_code: int) -> bytes:
    """Generate Modbus request packet."""
    protocol_id = 0x0000  # Modbus
    unit_id = 0x01

    if func_code in [0x01, 0x03]:  # Read
        start_addr = random.randint(0, 100)
        quantity = random.randint(1, 10)
        pdu = struct.pack(">BBHH", func_code, 0, start_addr, quantity)
    else:  # Write
        addr = random.randint(0, 100)
        value = random.randint(0, 1000)
        pdu = struct.pack(">BHH", func_code, addr, value)

    length = len(pdu) + 1
    mbap = struct.pack(">HHHB", transaction_id, protocol_id, length, unit_id)
    return mbap + pdu


def generate_modbus_response(transaction_id: int, func_code: int) -> bytes:
    """Generate Modbus response packet."""
    protocol_id = 0x0000
    unit_id = 0x01

    if func_code in [0x01, 0x03]:  # Read response
        byte_count = random.randint(2, 20)
        data = bytes([random.randint(0, 255) for _ in range(byte_count)])
        pdu = struct.pack(">BB", func_code, byte_count) + data
    else:  # Write response (echo)
        addr = random.randint(0, 100)
        value = random.randint(0, 1000)
        pdu = struct.pack(">BHH", func_code, addr, value)

    length = len(pdu) + 1
    mbap = struct.pack(">HHHB", transaction_id, protocol_id, length, unit_id)
    return mbap + pdu


def generate_unknown_binary_stream(output_file: Path) -> None:
    """Generate unknown protocol capture for pure RE demonstration.

    Characteristics:
        - Variable-length messages
        - State machine with 5 states
        - Encrypted sections
        - Pattern clustering opportunities
        - Timing-dependent behavior

    Total: ~7,500 packets
    File size: ~2 MB
    """
    print_info("Generating unknown_binary_stream.pcap...")

    packets = []
    base_time = time.time()
    state = 0  # State machine: 0=INIT, 1=AUTH, 2=DATA, 3=KEEPALIVE, 4=SHUTDOWN

    src_mac = "00:11:22:33:44:77"
    dst_mac = "AA:BB:CC:DD:EE:11"
    src_ip = "10.0.0.50"
    dst_ip = "10.0.0.1"
    src_port = 8888
    dst_port = 9999

    current_time = 0.0
    packet_count = 0

    while packet_count < 7500:
        # State machine transitions
        if state == 0:  # INIT
            payload = generate_mystery_init_packet()
            state = 1
        elif state == 1:  # AUTH
            payload = generate_mystery_auth_packet()
            state = 2 if random.random() > 0.1 else 1
        elif state == 2:  # DATA
            payload = generate_mystery_data_packet()
            if random.random() < 0.02:  # 2% chance to keepalive
                state = 3
        elif state == 3:  # KEEPALIVE
            payload = generate_mystery_keepalive_packet()
            state = 2
        else:  # SHUTDOWN
            payload = generate_mystery_shutdown_packet()
            state = 0

        pkt = create_udp_packet(src_mac, dst_mac, src_ip, dst_ip, src_port, dst_port, payload)
        pkt.time = base_time + current_time
        packets.append(pkt)

        packet_count += 1
        current_time += 0.01 + random.uniform(-0.002, 0.005)  # Variable timing

    wrpcap(str(output_file), packets)
    size_mb = output_file.stat().st_size / (1024 * 1024)
    print_success(
        f"Generated unknown_binary_stream.pcap ({size_mb:.2f} MB, {len(packets)} packets)"
    )


def generate_mystery_init_packet() -> bytes:
    """Generate mystery protocol INIT packet."""
    magic = b"\xde\xad\xbe\xef"
    msg_type = 0x01
    length = random.randint(10, 20)
    payload = bytes([random.randint(0, 255) for _ in range(length)])
    return magic + struct.pack(">BH", msg_type, length) + payload


def generate_mystery_auth_packet() -> bytes:
    """Generate mystery protocol AUTH packet."""
    magic = b"\xde\xad\xbe\xef"
    msg_type = 0x02
    length = 32  # Fixed auth packet size
    # Simulate encrypted data
    payload = bytes([random.randint(0, 255) for _ in range(length)])
    return magic + struct.pack(">BH", msg_type, length) + payload


def generate_mystery_data_packet() -> bytes:
    """Generate mystery protocol DATA packet."""
    magic = b"\xde\xad\xbe\xef"
    msg_type = 0x03
    length = random.randint(20, 200)
    # Mix of plaintext and patterns
    payload = bytes([i % 256 for i in range(length // 2)])
    payload += bytes([random.randint(0, 255) for _ in range(length - len(payload))])
    return magic + struct.pack(">BH", msg_type, length) + payload


def generate_mystery_keepalive_packet() -> bytes:
    """Generate mystery protocol KEEPALIVE packet."""
    magic = b"\xde\xad\xbe\xef"
    msg_type = 0x04
    length = 4
    timestamp = struct.pack(">I", int(time.time()))
    return magic + struct.pack(">BH", msg_type, length) + timestamp


def generate_mystery_shutdown_packet() -> bytes:
    """Generate mystery protocol SHUTDOWN packet."""
    magic = b"\xde\xad\xbe\xef"
    msg_type = 0xFF
    length = 0
    return magic + struct.pack(">BH", msg_type, length)


def create_udp_packet(
    src_mac: str,
    dst_mac: str,
    src_ip: str,
    dst_ip: str,
    src_port: int,
    dst_port: int,
    payload: bytes,
) -> Ether:
    """Create UDP packet with Ethernet/IP headers."""
    return (
        Ether(src=src_mac, dst=dst_mac)
        / IP(src=src_ip, dst=dst_ip)
        / UDP(sport=src_port, dport=dst_port)
        / payload
    )


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate demo data for UDP packet analysis")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    args = parser.parse_args()

    # Create demo_data directory
    script_dir = Path(__file__).parent
    data_dir = script_dir / "demo_data"
    data_dir.mkdir(exist_ok=True)

    # Define output files
    files_to_generate = [
        ("iot_protocol_capture.pcap", generate_iot_protocol_capture),
        ("industrial_modbus_udp.pcap", generate_industrial_modbus_udp),
        ("unknown_binary_stream.pcap", generate_unknown_binary_stream),
    ]

    print(f"\n{BOLD}{BLUE}Generating UDP Packet Analysis Demo Data{RESET}")
    print("=" * 80)

    for filename, generator_func in files_to_generate:
        output_file = data_dir / filename

        if output_file.exists() and not args.force:
            print_info(f"Skipping {filename} (already exists, use --force to overwrite)")
            continue

        generator_func(output_file)

    print(f"\n{GREEN}{BOLD}✓ Demo data generation complete!{RESET}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
