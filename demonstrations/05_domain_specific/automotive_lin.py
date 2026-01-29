#!/usr/bin/env python3
"""LIN Protocol Decoding Demonstration.

This demo showcases Oscura's LIN (Local Interconnect Network) automotive
protocol decoding capabilities for LIN 1.x and 2.x frames.

**Features Demonstrated**:
- LIN 1.x and 2.x frame decoding
- Break field detection (>= 13 bit times)
- Sync byte validation (0x55)
- Protected Identifier (PID) extraction
- Parity validation
- Classic and enhanced checksum computation
- Data byte extraction
- Error detection (framing, parity, checksum)

**LIN Frame Structure**:
- Break Field: Dominant for >= 13 bit times
- Sync Byte: 0x55 (10101010 pattern for clock sync)
- Protected ID: 6-bit ID + 2-bit parity
- Data: 0-8 bytes (application dependent)
- Checksum: 8-bit (classic or enhanced)

**LIN Checksum Types**:
- Classic (LIN 1.x): Sum of data bytes only
- Enhanced (LIN 2.x): Sum of PID + data bytes

**Common Baud Rates**:
- 9600 bps (LIN 1.x)
- 19200 bps (LIN 2.0)
- 20000 bps (LIN 2.1/2.2)

Usage:
    python lin_demo.py
    python lin_demo.py --verbose

Author: Oscura Development Team
Date: 2026-01-16
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demonstrations.common import BaseDemo, ValidationSuite, print_info, print_result
from demonstrations.common.base_demo import run_demo_main
from demonstrations.common.formatting import GREEN, RED, RESET, print_subheader

# Oscura imports
from oscura.analyzers.protocols.lin import (
    decode_lin,
)
from oscura.core.types import DigitalTrace, TraceMetadata


class LINDemo(BaseDemo):
    """LIN Protocol Decoding Demonstration.

    This demo generates LIN bus signals with various frame types,
    then decodes them to demonstrate Oscura's LIN analysis capabilities.
    """

    def __init__(self, **kwargs):
        """Initialize demo."""
        super().__init__(
            name="LIN Protocol Demo",
            description="Demonstrates LIN 1.x/2.x automotive protocol decoding",
            **kwargs,
        )
        self.sample_rate = 1e6  # 1 MHz sampling
        self.baudrate = 19200  # 19200 bps (LIN 2.0)

        # Storage for signals and results
        self.bus_signal = None
        self.packets = []

    def _compute_parity(self, frame_id: int) -> int:
        """Compute LIN 2.x protected identifier parity.

        Args:
            frame_id: 6-bit frame identifier.

        Returns:
            2-bit parity value.
        """
        id0 = (frame_id >> 0) & 1
        id1 = (frame_id >> 1) & 1
        id2 = (frame_id >> 2) & 1
        id3 = (frame_id >> 3) & 1
        id4 = (frame_id >> 4) & 1
        id5 = (frame_id >> 5) & 1

        # P0 = ID0 ^ ID1 ^ ID2 ^ ID4
        p0 = id0 ^ id1 ^ id2 ^ id4

        # P1 = !(ID1 ^ ID3 ^ ID4 ^ ID5)
        p1 = (id1 ^ id3 ^ id4 ^ id5) ^ 1

        return (p1 << 1) | p0

    def _compute_checksum(self, frame_id: int, data: bytes, version: str = "2.x") -> int:
        """Compute LIN checksum.

        Args:
            frame_id: Frame identifier.
            data: Data bytes.
            version: LIN version ("1.x" or "2.x").

        Returns:
            Checksum byte.
        """
        if version == "1.x":
            # Classic checksum: sum of data bytes only
            checksum = sum(data)
        else:
            # Enhanced checksum: PID + data bytes
            pid = frame_id | (self._compute_parity(frame_id) << 6)
            checksum = pid + sum(data)

        # Handle carries
        while checksum > 0xFF:
            checksum = (checksum & 0xFF) + (checksum >> 8)

        return (~checksum) & 0xFF

    def _encode_uart_byte(self, byte_val: int) -> list[int]:
        """Encode a byte as UART frame (1 start, 8 data, 1 stop).

        Args:
            byte_val: Byte to encode.

        Returns:
            List of bit values (0 = dominant, 1 = recessive).
        """
        bits = [0]  # Start bit (dominant)

        # Data bits (LSB first)
        for i in range(8):
            bits.append((byte_val >> i) & 1)

        bits.append(1)  # Stop bit (recessive)

        return bits

    def generate_test_data(self) -> dict:
        """Generate or load LIN test signals.

        Tries in this order:
        1. Load from --data-file if specified
        2. Load from default demo_data files if they exist
        3. Generate synthetic data using LIN encoder methods

        Creates LIN signals demonstrating:
        1. Frame with ID 0x10 (temperature sensor)
        2. Frame with ID 0x21 (motor control)
        3. Frame with ID 0x3C (diagnostic request)
        """
        # Try loading data from file
        data_file_to_load = None

        # 1. Check CLI override
        if self.data_file and self.data_file.exists():
            data_file_to_load = self.data_file
            print_info(f"Loading LIN data from CLI override: {self.data_file}")
        # 2. Check default generated data
        elif default_file := self.find_default_data_file("lin_bus.npz"):
            data_file_to_load = default_file
            print_info(f"Loading LIN data from default file: {default_file.name}")

        # Load from file if found
        if data_file_to_load:
            try:
                data = np.load(data_file_to_load)
                self.bus_signal = data["bus_signal"]
                loaded_sample_rate = float(data["sample_rate"])
                loaded_baudrate = float(data["baudrate"])

                # Update parameters from loaded data
                self.sample_rate = loaded_sample_rate
                self.baudrate = loaded_baudrate

                print_result("Loaded from file", data_file_to_load.name)
                print_result("Sample rate", f"{self.sample_rate / 1e6:.1f} MHz")
                print_result("Baud rate", f"{self.baudrate} bps")
                print_result("Total samples", len(self.bus_signal))
                return
            except Exception as e:
                print_info(f"Failed to load from file: {e}, falling back to synthetic")
                data_file_to_load = None

        # Generate synthetic data if not loaded
        print_info("Generating LIN bus test signals...")

        samples_per_bit = int(self.sample_rate / self.baudrate)

        # Build signal as list of bit values
        bus_bits = []

        def add_bits(bits: list[int]) -> None:
            """Expand bits to samples."""
            for bit in bits:
                bus_bits.extend([bit] * samples_per_bit)

        # Add initial idle (recessive = high)
        add_bits([1] * 20)

        # ===== Frame 1: Temperature Sensor (ID 0x10) =====
        print_info("  Adding temperature sensor frame (ID 0x10)")

        frame_id_1 = 0x10
        data_1 = bytes([0x23, 0x01, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF])  # 8 bytes
        checksum_1 = self._compute_checksum(frame_id_1, data_1, "2.x")
        pid_1 = frame_id_1 | (self._compute_parity(frame_id_1) << 6)

        # Break field (dominant for >= 13 bits)
        add_bits([0] * 14)  # Break (14 bit times dominant)
        add_bits([1] * 2)  # Break delimiter (recessive)

        # Sync byte (0x55)
        add_bits(self._encode_uart_byte(0x55))

        # Protected ID
        add_bits(self._encode_uart_byte(pid_1))

        # Data bytes
        for byte_val in data_1:
            add_bits(self._encode_uart_byte(byte_val))

        # Checksum
        add_bits(self._encode_uart_byte(checksum_1))

        # Inter-frame space
        add_bits([1] * 10)

        # ===== Frame 2: Motor Control (ID 0x21) =====
        print_info("  Adding motor control frame (ID 0x21)")

        frame_id_2 = 0x21
        data_2 = bytes([0xFF, 0x00, 0x80, 0x40])  # 4 bytes
        checksum_2 = self._compute_checksum(frame_id_2, data_2, "2.x")
        pid_2 = frame_id_2 | (self._compute_parity(frame_id_2) << 6)

        # Break field
        add_bits([0] * 14)
        add_bits([1] * 2)

        # Sync byte
        add_bits(self._encode_uart_byte(0x55))

        # Protected ID
        add_bits(self._encode_uart_byte(pid_2))

        # Data bytes (extend to 8 for consistency)
        for i in range(8):
            if i < len(data_2):
                add_bits(self._encode_uart_byte(data_2[i]))
            else:
                add_bits(self._encode_uart_byte(0x00))

        # Checksum
        add_bits(self._encode_uart_byte(checksum_2))

        # Inter-frame space
        add_bits([1] * 10)

        # ===== Frame 3: Diagnostic Request (ID 0x3C) =====
        print_info("  Adding diagnostic request frame (ID 0x3C)")

        frame_id_3 = 0x3C
        data_3 = bytes([0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08])
        checksum_3 = self._compute_checksum(frame_id_3, data_3, "2.x")
        pid_3 = frame_id_3 | (self._compute_parity(frame_id_3) << 6)

        # Break field
        add_bits([0] * 14)
        add_bits([1] * 2)

        # Sync byte
        add_bits(self._encode_uart_byte(0x55))

        # Protected ID
        add_bits(self._encode_uart_byte(pid_3))

        # Data bytes
        for byte_val in data_3:
            add_bits(self._encode_uart_byte(byte_val))

        # Checksum
        add_bits(self._encode_uart_byte(checksum_3))

        # Final idle
        add_bits([1] * 20)

        # Convert to numpy array
        self.bus_signal = np.array(bus_bits, dtype=bool)

        print_result("Total samples", len(self.bus_signal))
        print_result("Baud rate", f"{self.baudrate} bps")
        print_result("Sample rate", f"{self.sample_rate / 1e6:.1f} MHz")
        print_result("Samples per bit", samples_per_bit)

        return {}

    def run_demonstration(self, data: dict) -> dict:
        """Decode LIN signals and analyze frames."""
        print_subheader("LIN Decoding")

        # Create digital trace
        metadata = TraceMetadata(sample_rate=self.sample_rate)
        trace = DigitalTrace(data=self.bus_signal, metadata=metadata)

        # Decode using LIN decoder
        self.packets = decode_lin(
            data=trace,
            sample_rate=self.sample_rate,
            baudrate=self.baudrate,
            version="2.x",
        )

        print_result("Decoded frames", len(self.packets))

        # Analyze each frame
        print_subheader("Frame Analysis")

        self.results["frame_count"] = len(self.packets)
        self.results["frame_ids"] = []
        self.results["error_count"] = 0
        self.results["total_data_bytes"] = 0

        for i, pkt in enumerate(self.packets):
            frame_id = pkt.annotations.get("frame_id", 0)
            pid = pkt.annotations.get("pid", 0)
            data_len = pkt.annotations.get("data_length", len(pkt.data))
            checksum = pkt.annotations.get("checksum", 0)
            version = pkt.annotations.get("version", "2.x")
            timestamp = pkt.timestamp * 1e3  # Convert to ms

            self.results["frame_ids"].append(frame_id)
            self.results["total_data_bytes"] += len(pkt.data)

            print_info(f"  Frame #{i + 1}: ID 0x{frame_id:02X} @ {timestamp:.3f} ms")
            print_info(f"    PID: 0x{pid:02X}")
            print_info(f"    Data length: {data_len} bytes")
            print_info(f"    Data: {pkt.data.hex().upper()}")
            print_info(f"    Checksum: 0x{checksum:02X}")
            print_info(f"    Version: {version}")

            # Check for errors
            if pkt.errors:
                self.results["error_count"] += len(pkt.errors)
                for error in pkt.errors:
                    print_info(f"    {RED}Error: {error}{RESET}")
            else:
                print_info(f"    {GREEN}Valid frame{RESET}")

            # Identify frame purpose based on ID
            if frame_id == 0x10:
                print_info("    Purpose: Temperature sensor data")
            elif frame_id == 0x21:
                print_info("    Purpose: Motor control command")
            elif frame_id == 0x3C:
                print_info("    Purpose: Diagnostic request (master)")
            elif frame_id == 0x3D:
                print_info("    Purpose: Diagnostic response (slave)")
            elif frame_id == 0x3E or frame_id == 0x3F:
                print_info("    Purpose: Reserved (user defined)")

        # Summary statistics
        print_subheader("Summary")
        print_result("Total frames", self.results["frame_count"])
        print_result("Total data bytes", self.results["total_data_bytes"])
        print_result("Errors detected", self.results["error_count"])

        # Show unique frame IDs
        unique_ids = set(self.results["frame_ids"])
        print_info("Frame IDs found:")
        for fid in sorted(unique_ids):
            count = self.results["frame_ids"].count(fid)
            print_info(f"  - 0x{fid:02X}: {count} frame(s)")

        # Timing analysis
        if len(self.packets) >= 2:
            print_subheader("Timing Analysis")

            frame_times = [pkt.timestamp for pkt in self.packets]
            inter_frame_gaps = np.diff(frame_times)

            print_result(
                "Average inter-frame gap",
                f"{np.mean(inter_frame_gaps) * 1e3:.3f} ms",
            )
            print_result("Min inter-frame gap", f"{np.min(inter_frame_gaps) * 1e3:.3f} ms")
            print_result("Max inter-frame gap", f"{np.max(inter_frame_gaps) * 1e3:.3f} ms")

            self.results["avg_gap_ms"] = np.mean(inter_frame_gaps) * 1e3

        return self.results

    def validate(self, results: dict) -> bool:
        """Validate LIN decoding results."""
        suite = ValidationSuite()

        # Check total frames
        frame_count = results.get("frame_count", 0)
        suite.add_check("Total frames decoded", frame_count > 0, f"Got {frame_count} frames")

        # Check for expected frame IDs
        frame_ids = results.get("frame_ids", [])
        suite.add_check(
            "Frame ID 0x10 found", 0x10 in frame_ids, f"IDs: {[hex(id) for id in frame_ids]}"
        )
        suite.add_check(
            "Frame ID 0x21 found", 0x21 in frame_ids, f"IDs: {[hex(id) for id in frame_ids]}"
        )
        suite.add_check(
            "Frame ID 0x3C found", 0x3C in frame_ids, f"IDs: {[hex(id) for id in frame_ids]}"
        )

        # Check data integrity
        total_data_bytes = results.get("total_data_bytes", 0)
        suite.add_check("Total data bytes", total_data_bytes > 0, f"Got {total_data_bytes} bytes")

        # Verify signal integrity
        suite.add_check(
            "Bus signal generated",
            self.bus_signal is not None and len(self.bus_signal) > 0,
            f"Got {len(self.bus_signal) if self.bus_signal is not None else 0} samples",
        )

        suite.report()
        return suite.all_passed()


if __name__ == "__main__":
    sys.exit(run_demo_main(LINDemo))
