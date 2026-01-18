#!/usr/bin/env python3
"""Generate Demo Data for Automotive Protocol Demonstrations.

This script generates realistic automotive protocol data files for use with
the comprehensive automotive demo. All data is synthetic but representative
of real vehicle bus traffic.

Generated Files:
    - can_bus_normal_traffic.mf4 (~10 MB): CAN 2.0B with engine/transmission/body
    - can_fd_high_speed.mf4 (~8 MB): CAN-FD high-throughput data
    - lin_body_control.wfm (~2 MB): LIN 2.0 @ 19.2 kbps
    - obd2_diagnostic_session.pcap (~1 MB): OBD-II diagnostic sequence
    - uds_security_sequence.mf4 (~3 MB): UDS security access + memory read
    - demo_signals.dbc (~5 KB): Already provided

Requirements:
    - asammdf: pip install asammdf
    - cantools: pip install cantools
    - python-can: pip install python-can
    - scapy: pip install scapy

Usage:
    python generate_demo_data.py
    python generate_demo_data.py --output-dir /path/to/output
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Check for optional dependencies
try:
    import asammdf  # noqa: F401
    from asammdf import MDF, Signal

    HAS_ASAMMDF = True
except ImportError:
    HAS_ASAMMDF = False

try:
    import cantools  # noqa: F401

    HAS_CANTOOLS = True
except ImportError:
    HAS_CANTOOLS = False

try:
    from scapy.all import IP, UDP, Ether, wrpcap
    from scapy.packet import Raw

    HAS_SCAPY = True
except ImportError:
    HAS_SCAPY = False


def generate_can_bus_normal_traffic(output_path: Path) -> None:
    """Generate CAN 2.0B normal traffic MF4 file.

    Generates 60 seconds of realistic CAN traffic with engine, transmission,
    and body control messages at appropriate update rates.

    Args:
        output_path: Path for output .mf4 file
    """
    if not HAS_ASAMMDF:
        print("⚠ Skipping CAN MF4 generation: asammdf not installed")
        return

    print(f"Generating CAN bus normal traffic: {output_path}")

    # Duration and sample rates
    duration = 60.0  # seconds
    sample_rate_100hz = 100  # 10 ms period
    sample_rate_50hz = 50  # 20 ms period
    sample_rate_20hz = 20  # 50 ms period
    sample_rate_10hz = 10  # 100 ms period
    _sample_rate_5hz = 5  # 200 ms period
    _sample_rate_2hz = 2  # 500 ms period
    sample_rate_1hz = 1  # 1000 ms period

    signals = []

    # Message 0x280: Engine Status (100 Hz)
    t_100hz = np.arange(0, duration, 1 / sample_rate_100hz)
    engine_rpm = 800 + 1200 * (1 + np.sin(2 * np.pi * 0.05 * t_100hz)) / 2  # 800-2000 RPM
    engine_load = 20 + 60 * (1 + np.sin(2 * np.pi * 0.03 * t_100hz)) / 2  # 20-80%
    coolant_temp = 80 + 10 * np.sin(2 * np.pi * 0.01 * t_100hz)  # 70-90°C
    throttle_position = 10 + 70 * (1 + np.sin(2 * np.pi * 0.04 * t_100hz)) / 2  # 10-80%

    signals.extend(
        [
            Signal(
                samples=engine_rpm,
                timestamps=t_100hz,
                name="Engine_RPM",
                unit="rpm",
                comment="Engine speed",
            ),
            Signal(
                samples=engine_load,
                timestamps=t_100hz,
                name="Engine_Load",
                unit="%",
                comment="Engine load",
            ),
            Signal(
                samples=coolant_temp,
                timestamps=t_100hz,
                name="Coolant_Temp",
                unit="degC",
                comment="Coolant temperature",
            ),
            Signal(
                samples=throttle_position,
                timestamps=t_100hz,
                name="Throttle_Position",
                unit="%",
                comment="Throttle position",
            ),
        ]
    )

    # Message 0x300: Vehicle Speed (50 Hz)
    t_50hz = np.arange(0, duration, 1 / sample_rate_50hz)
    vehicle_speed = 50 + 30 * (1 + np.sin(2 * np.pi * 0.02 * t_50hz)) / 2  # 50-80 km/h

    signals.append(
        Signal(
            samples=vehicle_speed,
            timestamps=t_50hz,
            name="Vehicle_Speed",
            unit="km/h",
            comment="Vehicle speed",
        )
    )

    # Message 0x400: Transmission (20 Hz)
    t_20hz = np.arange(0, duration, 1 / sample_rate_20hz)
    current_gear = np.clip(np.floor(t_20hz / 10), 1, 6).astype(int)  # Shift through gears
    transmission_temp = 70 + 5 * np.sin(2 * np.pi * 0.005 * t_20hz)  # 65-75°C

    signals.extend(
        [
            Signal(
                samples=current_gear,
                timestamps=t_20hz,
                name="Current_Gear",
                unit="",
                comment="Current gear position",
            ),
            Signal(
                samples=transmission_temp,
                timestamps=t_20hz,
                name="Transmission_Temp",
                unit="degC",
                comment="Transmission temperature",
            ),
        ]
    )

    # Message 0x100: Brake Status (100 Hz)
    t_100hz_2 = np.arange(0, duration, 1 / sample_rate_100hz)
    brake_pressure = 5 + 50 * np.abs(np.sin(2 * np.pi * 0.1 * t_100hz_2))  # 5-55 bar

    signals.append(
        Signal(
            samples=brake_pressure,
            timestamps=t_100hz_2,
            name="Brake_Pressure",
            unit="bar",
            comment="Brake pressure",
        )
    )

    # Message 0x200: Steering (100 Hz)
    steering_angle = 300 * np.sin(2 * np.pi * 0.05 * t_100hz)  # -300 to +300 degrees

    signals.append(
        Signal(
            samples=steering_angle,
            timestamps=t_100hz,
            name="Steering_Angle",
            unit="deg",
            comment="Steering wheel angle",
        )
    )

    # Message 0x140: Body Control (10 Hz)
    t_10hz = np.arange(0, duration, 1 / sample_rate_10hz)
    headlights_on = (t_10hz > 30).astype(int)  # Turn on at 30s

    signals.append(
        Signal(
            samples=headlights_on,
            timestamps=t_10hz,
            name="Headlights_On",
            unit="",
            comment="Headlight status",
        )
    )

    # Message 0x1C0: Battery (1 Hz)
    t_1hz = np.arange(0, duration, 1 / sample_rate_1hz)
    battery_voltage = 14.2 + 0.3 * np.sin(2 * np.pi * 0.01 * t_1hz)  # 13.9-14.5V
    battery_current = 20 + 10 * np.sin(2 * np.pi * 0.02 * t_1hz)  # 10-30A

    signals.extend(
        [
            Signal(
                samples=battery_voltage,
                timestamps=t_1hz,
                name="Battery_Voltage",
                unit="V",
                comment="Battery voltage",
            ),
            Signal(
                samples=battery_current,
                timestamps=t_1hz,
                name="Battery_Current",
                unit="A",
                comment="Battery current",
            ),
        ]
    )

    # Create MDF file
    mdf = MDF(version="4.10")
    mdf.append(signals)
    mdf.save(output_path, overwrite=True)

    print(f"  ✓ Generated {output_path.stat().st_size / 1024 / 1024:.1f} MB")


def generate_can_fd_high_speed(output_path: Path) -> None:
    """Generate CAN-FD high-throughput MF4 file.

    Args:
        output_path: Path for output .mf4 file
    """
    if not HAS_ASAMMDF:
        print("⚠ Skipping CAN-FD generation: asammdf not installed")
        return

    print(f"Generating CAN-FD high-speed traffic: {output_path}")

    # CAN-FD supports 64-byte payloads at higher data rates
    duration = 30.0
    sample_rate = 200  # 5 ms period

    t = np.arange(0, duration, 1 / sample_rate)

    signals = []

    # Camera data stream (high throughput)
    camera_frame_id = np.arange(len(t)) % 65536
    camera_timestamp = t * 1000  # milliseconds

    signals.extend(
        [
            Signal(
                samples=camera_frame_id,
                timestamps=t,
                name="Camera_Frame_ID",
                unit="",
                comment="Camera frame sequence number",
            ),
            Signal(
                samples=camera_timestamp,
                timestamps=t,
                name="Camera_Timestamp",
                unit="ms",
                comment="Camera timestamp",
            ),
        ]
    )

    # RADAR data (high bandwidth)
    radar_targets = 5 + 10 * np.random.random(len(t))  # 5-15 targets
    radar_range = 50 + 100 * np.random.random(len(t))  # 50-150 m

    signals.extend(
        [
            Signal(
                samples=radar_targets,
                timestamps=t,
                name="RADAR_Targets",
                unit="",
                comment="Number of detected targets",
            ),
            Signal(
                samples=radar_range,
                timestamps=t,
                name="RADAR_Range",
                unit="m",
                comment="Closest target range",
            ),
        ]
    )

    # LIDAR point cloud metadata
    lidar_points = 50000 + 10000 * np.random.random(len(t))  # 50k-60k points
    lidar_scan_rate = 10 + 2 * np.sin(2 * np.pi * 0.05 * t)  # 8-12 Hz

    signals.extend(
        [
            Signal(
                samples=lidar_points,
                timestamps=t,
                name="LIDAR_Points",
                unit="",
                comment="Point cloud size",
            ),
            Signal(
                samples=lidar_scan_rate,
                timestamps=t,
                name="LIDAR_Scan_Rate",
                unit="Hz",
                comment="Scan rate",
            ),
        ]
    )

    # Create MDF file
    mdf = MDF(version="4.10")
    mdf.append(signals)
    mdf.save(output_path, overwrite=True)

    print(f"  ✓ Generated {output_path.stat().st_size / 1024 / 1024:.1f} MB")


def generate_obd2_diagnostic_session(output_path: Path) -> None:
    """Generate OBD-II diagnostic session PCAP file.

    Args:
        output_path: Path for output .pcap file
    """
    if not HAS_SCAPY:
        print("⚠ Skipping OBD-II PCAP generation: scapy not installed")
        return

    print(f"Generating OBD-II diagnostic session: {output_path}")

    packets = []

    # Helper to create CAN-over-UDP packet
    def create_can_packet(can_id: int, data: bytes, timestamp: float) -> Any:
        """Create CAN message encapsulated in UDP."""
        # Simple encapsulation: 4 bytes CAN ID + data
        payload = struct.pack("<I", can_id) + data
        pkt = (
            Ether(dst="ff:ff:ff:ff:ff:ff", src="00:11:22:33:44:55")
            / IP(src="192.168.1.100", dst="192.168.1.255")
            / UDP(sport=11898, dport=11898)
            / Raw(load=payload)
        )
        pkt.time = timestamp
        return pkt

    timestamp = 1.0

    # Mode 01: Request current data
    # PID 0x0C: Engine RPM
    req = create_can_packet(
        0x7DF, bytes([0x02, 0x01, 0x0C, 0x00, 0x00, 0x00, 0x00, 0x00]), timestamp
    )
    packets.append(req)
    timestamp += 0.05

    resp = create_can_packet(
        0x7E8, bytes([0x04, 0x41, 0x0C, 0x1F, 0x40, 0x00, 0x00, 0x00]), timestamp
    )
    packets.append(resp)
    timestamp += 0.5

    # PID 0x0D: Vehicle Speed
    req = create_can_packet(
        0x7DF, bytes([0x02, 0x01, 0x0D, 0x00, 0x00, 0x00, 0x00, 0x00]), timestamp
    )
    packets.append(req)
    timestamp += 0.05

    resp = create_can_packet(
        0x7E8, bytes([0x03, 0x41, 0x0D, 0x50, 0x00, 0x00, 0x00, 0x00]), timestamp
    )
    packets.append(resp)
    timestamp += 0.5

    # PID 0x05: Coolant Temperature
    req = create_can_packet(
        0x7DF, bytes([0x02, 0x01, 0x05, 0x00, 0x00, 0x00, 0x00, 0x00]), timestamp
    )
    packets.append(req)
    timestamp += 0.05

    resp = create_can_packet(
        0x7E8, bytes([0x03, 0x41, 0x05, 0x78, 0x00, 0x00, 0x00, 0x00]), timestamp
    )
    packets.append(resp)
    timestamp += 1.0

    # Mode 03: Request DTCs
    req = create_can_packet(
        0x7DF, bytes([0x01, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]), timestamp
    )
    packets.append(req)
    timestamp += 0.05

    resp = create_can_packet(
        0x7E8, bytes([0x06, 0x43, 0x02, 0x04, 0x20, 0x01, 0x71, 0x00]), timestamp
    )  # P0420, P0171
    packets.append(resp)
    timestamp += 1.0

    # Mode 09: Request VIN (PID 0x02)
    req = create_can_packet(
        0x7DF, bytes([0x02, 0x09, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00]), timestamp
    )
    packets.append(req)
    timestamp += 0.05

    # Multi-frame response (simplified - first frame only)
    resp = create_can_packet(
        0x7E8, bytes([0x10, 0x14, 0x49, 0x02, 0x01, 0x57, 0x56, 0x57]), timestamp
    )  # WVW...
    packets.append(resp)

    # Write PCAP
    wrpcap(str(output_path), packets)

    print(f"  ✓ Generated {output_path.stat().st_size / 1024:.1f} KB ({len(packets)} packets)")


def generate_uds_security_sequence(output_path: Path) -> None:
    """Generate UDS security access sequence MF4 file.

    Args:
        output_path: Path for output .mf4 file
    """
    if not HAS_ASAMMDF:
        print("⚠ Skipping UDS sequence generation: asammdf not installed")
        return

    print(f"Generating UDS security sequence: {output_path}")

    # UDS sequence timestamps
    timestamps = [1.0, 1.05, 2.0, 2.05, 2.5, 2.55, 3.0, 3.05, 4.0, 4.05]

    # Service IDs (simplified representation)
    service_ids = [
        0x10,
        0x50,  # Diagnostic Session Control
        0x27,
        0x67,  # Security Access - Request Seed
        0x27,
        0x67,  # Security Access - Send Key
        0x22,
        0x62,  # Read Data By Identifier
        0x11,
        0x51,  # ECU Reset
    ]

    # Sub-functions
    sub_functions = [0x02, 0x02, 0x01, 0x01, 0x02, 0x02, 0xF1, 0xF1, 0x01, 0x01]

    # Create signals
    signals = [
        Signal(
            samples=np.array(service_ids),
            timestamps=np.array(timestamps),
            name="UDS_Service_ID",
            unit="",
            comment="UDS service identifier",
        ),
        Signal(
            samples=np.array(sub_functions),
            timestamps=np.array(timestamps),
            name="UDS_Sub_Function",
            unit="",
            comment="UDS sub-function",
        ),
    ]

    # Create MDF file
    mdf = MDF(version="4.10")
    mdf.append(signals)
    mdf.save(output_path, overwrite=True)

    print(f"  ✓ Generated {output_path.stat().st_size / 1024:.1f} KB")


def generate_lin_body_control(output_path: Path) -> None:
    """Generate LIN body control waveform.

    Note: LIN is a single-wire protocol typically captured as analog waveform.
    This generates a simplified representation. For full LIN decoding,
    use Oscura's UART decoder with LIN-specific parameters.

    Args:
        output_path: Path for output .wfm file
    """
    print(f"⚠ LIN waveform generation not implemented: {output_path}")
    print("  Note: LIN requires analog waveform capture.")
    print("  Use Oscura's UART decoder on real LIN captures with:")
    print("    - Baud rate: 19200 bps")
    print("    - Break detection: enabled")
    print("    - Enhanced checksum: enabled")


def main() -> int:
    """Generate all automotive demo data files.

    Returns:
        Exit code (0 for success)
    """
    parser = argparse.ArgumentParser(
        description="Generate automotive protocol demo data files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "demo_data",
        help="Output directory for generated files",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing data files",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Automotive Demo Data Generator")
    print("=" * 80)
    print()
    print(f"Output directory: {output_dir}")
    print()

    # Check dependencies
    missing_deps = []
    if not HAS_ASAMMDF:
        missing_deps.append("asammdf")
    if not HAS_CANTOOLS:
        missing_deps.append("cantools")
    if not HAS_SCAPY:
        missing_deps.append("scapy")

    if missing_deps:
        print("⚠ Missing optional dependencies:")
        for dep in missing_deps:
            print(f"    - {dep}")
        print()
        print("Install with: uv sync --all-extras")
        print()

    # Generate files
    print("Generating demo data files...")
    print()

    generate_can_bus_normal_traffic(output_dir / "can_bus_normal_traffic.mf4")
    generate_can_fd_high_speed(output_dir / "can_fd_high_speed.mf4")
    generate_obd2_diagnostic_session(output_dir / "obd2_diagnostic_session.pcap")
    generate_uds_security_sequence(output_dir / "uds_security_sequence.mf4")
    generate_lin_body_control(output_dir / "lin_body_control.wfm")

    print()
    print("=" * 80)
    print("Demo Data Generation Complete!")
    print("=" * 80)
    print()
    print(f"Files generated in: {output_dir}")
    print()
    print("Note: demo_signals.dbc is already provided in the demo directory")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
