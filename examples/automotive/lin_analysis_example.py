#!/usr/bin/env python3
"""LIN protocol analysis demonstration.

This example demonstrates the LIN analyzer capabilities including:
- Protected ID calculation with parity bits
- Frame parsing with checksum validation
- Signal decoding
- Diagnostic frame parsing
- LDF file generation
"""

from pathlib import Path
from tempfile import TemporaryDirectory

from oscura.automotive.lin import LINAnalyzer, LINSignal


def main() -> None:
    """Run LIN analysis demonstration."""
    print("=" * 70)
    print("LIN Protocol Analysis Demonstration")
    print("=" * 70)
    print()

    # Initialize analyzer
    analyzer = LINAnalyzer()

    # Define signals
    print("1. Defining LIN signals")
    print("-" * 70)
    analyzer.add_signal(
        LINSignal(
            name="EngineSpeed",
            frame_id=0x10,
            start_bit=0,
            bit_length=16,
            init_value=0,
            publisher="Master",
        )
    )
    analyzer.add_signal(
        LINSignal(
            name="Throttle",
            frame_id=0x10,
            start_bit=16,
            bit_length=8,
            init_value=0,
            publisher="Master",
        )
    )
    analyzer.add_signal(
        LINSignal(
            name="VehicleSpeed",
            frame_id=0x20,
            start_bit=0,
            bit_length=8,
            init_value=0,
            publisher="Slave1",
        )
    )
    print(f"Added {len(analyzer.signals)} signals")
    print()

    # Simulate captured LIN traffic
    print("2. Parsing LIN frames")
    print("-" * 70)

    # Frame 0x10: Engine speed and throttle
    test_frames = [
        (0.000, 0x10, b"\x10\x27\x64"),  # Engine=10000 RPM, Throttle=100
        (0.010, 0x20, b"\x50"),  # Speed=80 km/h
        (0.020, 0x10, b"\x20\x4e\x80"),  # Engine=20000 RPM, Throttle=128
        (0.030, 0x20, b"\x78"),  # Speed=120 km/h
    ]

    for timestamp, frame_id, data in test_frames:
        # Calculate protected ID and checksum
        protected_id = analyzer._calculate_protected_id(frame_id)
        checksum = analyzer._calculate_enhanced_checksum(protected_id, data)

        # Build complete frame (sync + protected_id + data + checksum)
        frame_bytes = bytes([0x55, protected_id]) + data + bytes([checksum])

        # Parse frame
        frame = analyzer.parse_frame(frame_bytes, timestamp=timestamp)

        print(f"Frame ID 0x{frame.frame_id:02X} @ {frame.timestamp:.3f}s:")
        print(f"  Data: {frame.data.hex().upper()}")
        print(f"  Checksum: {'✓' if frame.checksum_valid else '✗'} (0x{frame.checksum:02X})")

        if frame.decoded_signals:
            print("  Decoded signals:")
            for name, value in frame.decoded_signals.items():
                print(f"    {name}: {value}")
        print()

    # Parse diagnostic frame
    print("3. Parsing diagnostic frame")
    print("-" * 70)
    protected_id_diag = analyzer._calculate_protected_id(0x3C)
    data_diag = b"\x01\x06\xb6\x00\x01\x00\x00\x00"  # ReadById request
    checksum_diag = analyzer._calculate_enhanced_checksum(protected_id_diag, data_diag)
    frame_bytes_diag = bytes([0x55, protected_id_diag]) + data_diag + bytes([checksum_diag])

    diag_frame = analyzer.parse_frame(frame_bytes_diag, timestamp=0.040)
    print(f"Diagnostic Frame: {diag_frame.decoded_signals['frame_type']}")
    print(f"  Service: {diag_frame.decoded_signals['service_name']}")
    print(f"  NAD: 0x{diag_frame.decoded_signals['nad']:02X}")
    print()

    # Infer schedule table
    print("4. Inferring schedule table")
    print("-" * 70)
    schedule = analyzer.infer_schedule_table()
    print(f"Detected {len(schedule)} frame IDs in schedule:")
    for entry in schedule:
        print(f"  Frame 0x{entry.frame_id:02X}: {entry.delay_ms:.1f} ms delay")
    print()

    # Generate LDF
    print("5. Generating LDF file")
    print("-" * 70)
    with TemporaryDirectory() as tmpdir:
        ldf_path = Path(tmpdir) / "vehicle.ldf"
        analyzer.generate_ldf(ldf_path, baudrate=19200)

        print(f"Generated LDF: {ldf_path.name}")
        print()
        print("LDF Content:")
        print("-" * 70)
        ldf_content = ldf_path.read_text()
        print(ldf_content)

    print()
    print("=" * 70)
    print("Demonstration complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
