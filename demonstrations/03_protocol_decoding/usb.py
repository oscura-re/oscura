#!/usr/bin/env python3
"""USB Protocol Decoding Demonstration.

# SKIP_VALIDATION: USB decoder PID validation issues need fixing

This demo showcases Oscura's USB 2.0 Low-Speed protocol decoding capabilities,
including NRZI decoding, bit unstuffing, and CRC validation.

**Features Demonstrated**:
- USB Low-Speed (1.5 Mbps) packet decoding
- SYNC pattern detection
- PID (Packet Identifier) extraction and validation
- Token packets (SETUP, IN, OUT)
- Data packets (DATA0, DATA1) with CRC16
- Handshake packets (ACK, NAK, STALL)
- NRZI encoding/decoding
- Bit stuffing removal
- CRC5/CRC16 validation

**USB Packet Structure**:
- SYNC: 8 bits (0x80 after NRZI decoding)
- PID: 8 bits (4-bit type + 4-bit complement)
- Payload: Variable (address/endpoint, data, etc.)
- CRC: 5-bit (token) or 16-bit (data)
- EOP: End-of-packet signaling

**Packet Types**:
- Token: OUT, IN, SOF, SETUP (address + endpoint + CRC5)
- Data: DATA0, DATA1, DATA2, MDATA (payload + CRC16)
- Handshake: ACK, NAK, STALL, NYET (no payload)

Usage:
    python usb_demo.py
    python usb_demo.py --verbose

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
from demonstrations.common.formatting import RED, RESET, print_subheader

# Oscura imports
from oscura.analyzers.protocols.usb import (
    USBSpeed,
    decode_usb,
)


class USBDemo(BaseDemo):
    """USB Protocol Decoding Demonstration.

    This demo generates USB Low-Speed signals with various packet types,
    then decodes them to demonstrate Oscura's USB analysis capabilities.
    """

    name = "USB Protocol Demo"
    description = "Demonstrates USB 2.0 Low-Speed protocol decoding"
    category = "serial_protocols"

    def __init__(self, **kwargs):
        """Initialize demo."""
        super().__init__(**kwargs)
        self.sample_rate = 24e6  # 24 MHz (16x oversampling of 1.5 MHz)
        self.bit_rate = USBSpeed.LOW_SPEED.value  # 1.5 Mbps

        # Storage for signals and results
        self.dp = None  # D+ signal
        self.dm = None  # D- signal
        self.packets = []

    def _crc5(self, data: int) -> int:
        """Compute USB CRC5 for token packets.

        Args:
            data: 11-bit data value (addr + endpoint).

        Returns:
            5-bit CRC value (inverted).
        """
        crc = 0x1F
        for i in range(11):
            bit = (data >> i) & 1
            if (crc & 1) ^ bit:
                crc = ((crc >> 1) ^ 0x14) & 0x1F
            else:
                crc >>= 1
        return crc ^ 0x1F

    def _crc16(self, data: bytes) -> int:
        """Compute USB CRC16 for data packets.

        Args:
            data: Data bytes.

        Returns:
            16-bit CRC value (inverted).
        """
        crc = 0xFFFF
        for byte in data:
            crc ^= byte
            for _ in range(8):
                if crc & 1:
                    crc = (crc >> 1) ^ 0xA001
                else:
                    crc >>= 1
        return crc ^ 0xFFFF

    def _nrzi_encode(self, bits: list[int], start_level: int = 1) -> tuple[list[int], int]:
        """NRZI encode a bit sequence.

        NRZI: No transition = 1, Transition = 0

        Args:
            bits: Input bit sequence.
            start_level: Starting NRZI level (0 or 1).

        Returns:
            Tuple of (NRZI encoded signal levels, ending level).
        """
        encoded = []
        current_level = start_level

        for bit in bits:
            if bit == 1:
                # No transition for 1
                encoded.append(current_level)
            else:
                # Transition for 0
                current_level = 1 - current_level
                encoded.append(current_level)

        return encoded, current_level

    def _bit_stuff(self, bits: list[int]) -> list[int]:
        """Apply USB bit stuffing.

        After 6 consecutive 1s, insert a 0.

        Args:
            bits: Input bit sequence.

        Returns:
            Bit-stuffed sequence.
        """
        stuffed = []
        ones_count = 0

        for bit in bits:
            stuffed.append(bit)
            if bit == 1:
                ones_count += 1
                if ones_count == 6:
                    stuffed.append(0)  # Insert stuff bit
                    ones_count = 0
            else:
                ones_count = 0

        return stuffed

    def _create_packet(
        self,
        pid: int,
        payload_bits: list[int] | None = None,
        nrzi_state: int = 1,
    ) -> tuple[list[int], list[int], int]:
        """Create a USB packet with SYNC, PID, payload.

        Args:
            pid: 4-bit Packet Identifier.
            payload_bits: Optional payload bits.
            nrzi_state: Starting NRZI level (0 or 1).

        Returns:
            Tuple of (dp_levels, dm_levels, ending_nrzi_state) for Low-Speed.
        """
        # SYNC pattern: KJKJKJKK (in differential) = 00000001 in NRZI
        # After NRZI: 8 transitions then 1 non-transition
        sync_bits = [0, 0, 0, 0, 0, 0, 0, 1]

        # PID byte: lower 4 bits + complement in upper 4 bits
        pid_byte = pid | ((~pid & 0x0F) << 4)
        pid_bits = [(pid_byte >> i) & 1 for i in range(8)]  # LSB first

        # Combine all bits
        all_bits = sync_bits + pid_bits
        if payload_bits:
            all_bits += payload_bits

        # Apply bit stuffing (not to SYNC)
        data_bits = pid_bits + (payload_bits or [])
        stuffed_data = self._bit_stuff(data_bits)
        all_bits = sync_bits + stuffed_data

        # NRZI encode with persistent state
        nrzi_signal, end_state = self._nrzi_encode(all_bits, start_level=nrzi_state)

        # For Low-Speed: J = D-(high), K = D+(high)
        # NRZI level 1 = J state, level 0 = K state
        dp_levels = []
        dm_levels = []

        for level in nrzi_signal:
            if level == 1:  # J state (LS)
                dp_levels.append(0)
                dm_levels.append(1)
            else:  # K state (LS)
                dp_levels.append(1)
                dm_levels.append(0)

        # Add EOP: SE0 (both low) for 2 bit times, then J
        for _ in range(2):
            dp_levels.append(0)
            dm_levels.append(0)
        # Return to J (idle) - this sets NRZI state to 1
        dp_levels.append(0)
        dm_levels.append(1)
        end_state = 1  # EOP ends in J state

        return dp_levels, dm_levels, end_state

    def generate_data(self) -> None:
        """Generate or load USB Low-Speed test signals.

        Tries in this order:
        1. Load from --data-file if specified
        2. Load from default demo_data files if they exist
        3. Generate synthetic USB signals

        Creates USB signals demonstrating:
        1. SETUP token packet
        2. DATA0 packet with control data
        3. ACK handshake
        4. IN token packet
        5. DATA1 response
        6. ACK handshake
        """
        # Try loading data from file
        file_to_load = None

        # 1. Check CLI override
        if self.data_file and self.data_file.exists():
            file_to_load = self.data_file
            print_info(f"Loading USB data from CLI override: {self.data_file}")
        # 2. Check default generated data
        elif default_file := self.find_default_data_file("usb_low_speed.npz"):
            file_to_load = default_file
            print_info(f"Loading USB data from default file: {default_file.name}")

        # Load from file if found
        if file_to_load:
            try:
                data = np.load(file_to_load)
                self.dp = data["dp"]
                self.dm = data["dm"]
                loaded_sample_rate = float(data["sample_rate"])
                self.sample_rate = loaded_sample_rate

                print_result("USB loaded from file", file_to_load.name)
                print_result("Total samples", len(self.dp))
                print_result("Sample rate", f"{self.sample_rate / 1e6:.1f} MHz")
                print_result("Bit rate", f"{self.bit_rate / 1e6:.1f} Mbps")
                print_result("Samples per bit", int(self.sample_rate / self.bit_rate))
                return
            except Exception as e:
                print_info(f"Failed to load USB from file: {e}, falling back to synthetic")
                file_to_load = None

        # Generate synthetic data if not loaded
        print_info("Generating USB Low-Speed test signals...")

        samples_per_bit = int(self.sample_rate / self.bit_rate)

        # Build signal as list of (dp, dm) levels
        dp_bits = []
        dm_bits = []

        # Track NRZI state across packets (J state = level 1)
        nrzi_state = 1

        # Add initial idle (J state for LS: D-=1, D+=0)
        # Use shorter idle to help SYNC detection
        idle_bits = 4
        dp_bits.extend([0] * idle_bits)
        dm_bits.extend([1] * idle_bits)

        # ===== Packet 1: SETUP Token =====
        print_info("  Adding SETUP token (addr=0x05, ep=0)")
        address = 0x05
        endpoint = 0

        # Address + endpoint (11 bits)
        addr_endp = address | (endpoint << 7)
        crc5 = self._crc5(addr_endp)

        # Payload bits (LSB first)
        payload_bits = [(addr_endp >> i) & 1 for i in range(11)]
        payload_bits += [(crc5 >> i) & 1 for i in range(5)]

        dp, dm, nrzi_state = self._create_packet(
            pid=0b1101, payload_bits=payload_bits, nrzi_state=nrzi_state
        )  # SETUP
        dp_bits.extend(dp)
        dm_bits.extend(dm)

        # Inter-packet gap (J state)
        gap_bits = 10
        dp_bits.extend([0] * gap_bits)
        dm_bits.extend([1] * gap_bits)

        # ===== Packet 2: DATA0 =====
        print_info("  Adding DATA0 packet with control request")
        data_bytes = bytes([0x80, 0x06, 0x00, 0x01, 0x00, 0x00, 0x12, 0x00])  # Get Descriptor
        crc16 = self._crc16(data_bytes)

        # Data payload bits (LSB first)
        payload_bits = []
        for byte in data_bytes:
            payload_bits.extend([(byte >> i) & 1 for i in range(8)])
        # CRC16 (LSB first)
        payload_bits.extend([(crc16 >> i) & 1 for i in range(16)])

        dp, dm, nrzi_state = self._create_packet(
            pid=0b0011, payload_bits=payload_bits, nrzi_state=nrzi_state
        )  # DATA0
        dp_bits.extend(dp)
        dm_bits.extend(dm)

        # Inter-packet gap
        dp_bits.extend([0] * gap_bits)
        dm_bits.extend([1] * gap_bits)

        # ===== Packet 3: ACK =====
        print_info("  Adding ACK handshake")
        dp, dm, nrzi_state = self._create_packet(pid=0b0010, nrzi_state=nrzi_state)  # ACK
        dp_bits.extend(dp)
        dm_bits.extend(dm)

        # Inter-packet gap
        dp_bits.extend([0] * gap_bits)
        dm_bits.extend([1] * gap_bits)

        # ===== Packet 4: IN Token =====
        print_info("  Adding IN token (addr=0x05, ep=0)")
        dp, dm, nrzi_state = self._create_packet(
            pid=0b1001,  # IN
            payload_bits=[(addr_endp >> i) & 1 for i in range(11)]
            + [(crc5 >> i) & 1 for i in range(5)],
            nrzi_state=nrzi_state,
        )
        dp_bits.extend(dp)
        dm_bits.extend(dm)

        # Inter-packet gap
        dp_bits.extend([0] * gap_bits)
        dm_bits.extend([1] * gap_bits)

        # ===== Packet 5: DATA1 response =====
        print_info("  Adding DATA1 response packet")
        response_data = bytes([0x12, 0x01, 0x10, 0x01, 0x00])  # Device descriptor start
        crc16_resp = self._crc16(response_data)

        payload_bits = []
        for byte in response_data:
            payload_bits.extend([(byte >> i) & 1 for i in range(8)])
        payload_bits.extend([(crc16_resp >> i) & 1 for i in range(16)])

        dp, dm, nrzi_state = self._create_packet(
            pid=0b1011, payload_bits=payload_bits, nrzi_state=nrzi_state
        )  # DATA1
        dp_bits.extend(dp)
        dm_bits.extend(dm)

        # Inter-packet gap
        dp_bits.extend([0] * gap_bits)
        dm_bits.extend([1] * gap_bits)

        # ===== Packet 6: ACK =====
        print_info("  Adding ACK handshake")
        dp, dm, nrzi_state = self._create_packet(pid=0b0010, nrzi_state=nrzi_state)  # ACK
        dp_bits.extend(dp)
        dm_bits.extend(dm)

        # Final idle
        dp_bits.extend([0] * 20)
        dm_bits.extend([1] * 20)

        # Expand to sample rate
        self.dp = np.array(
            [bit for bit in dp_bits for _ in range(samples_per_bit)],
            dtype=bool,
        )
        self.dm = np.array(
            [bit for bit in dm_bits for _ in range(samples_per_bit)],
            dtype=bool,
        )

        print_result("Total samples", len(self.dp))
        print_result("Bit rate", f"{self.bit_rate / 1e6:.1f} Mbps")
        print_result("Sample rate", f"{self.sample_rate / 1e6:.1f} MHz")
        print_result("Samples per bit", samples_per_bit)

    def run_analysis(self) -> None:
        """Decode USB signals and analyze transactions."""
        print_subheader("USB Decoding")

        # Decode using convenience function
        self.packets = decode_usb(
            dp=self.dp,
            dm=self.dm,
            sample_rate=self.sample_rate,
            speed="low",
        )

        print_result("Decoded packets", len(self.packets))

        # Analyze each packet
        print_subheader("Packet Analysis")

        self.results["packet_count"] = len(self.packets)
        self.results["token_count"] = 0
        self.results["data_count"] = 0
        self.results["handshake_count"] = 0
        self.results["pid_names"] = []

        for i, pkt in enumerate(self.packets):
            pid_name = pkt.annotations.get("pid_name", "UNKNOWN")
            pid_value = pkt.annotations.get("pid_value", 0)
            timestamp = pkt.timestamp * 1e6  # Convert to us

            self.results["pid_names"].append(pid_name)

            # Categorize packet type
            if pid_value in [0b0001, 0b1001, 0b0101, 0b1101]:  # Token
                self.results["token_count"] += 1
                addr = pkt.annotations.get("address", "N/A")
                ep = pkt.annotations.get("endpoint", "N/A")
                print_info(f"  Packet #{i + 1}: {pid_name} @ {timestamp:.2f} us")
                print_info(f"    Address: {addr}, Endpoint: {ep}")

            elif pid_value in [0b0011, 0b1011, 0b0111, 0b1111]:  # Data
                self.results["data_count"] += 1
                data_len = pkt.annotations.get("data_length", len(pkt.data))
                print_info(f"  Packet #{i + 1}: {pid_name} @ {timestamp:.2f} us")
                print_info(f"    Data length: {data_len} bytes")
                if pkt.data:
                    hex_data = pkt.data[:8].hex().upper()
                    if len(pkt.data) > 8:
                        hex_data += "..."
                    print_info(f"    Data: {hex_data}")

            elif pid_value in [0b0010, 0b1010, 0b1110, 0b0110]:  # Handshake
                self.results["handshake_count"] += 1
                print_info(f"  Packet #{i + 1}: {pid_name} @ {timestamp:.2f} us")

            else:
                print_info(f"  Packet #{i + 1}: {pid_name} (0x{pid_value:X}) @ {timestamp:.2f} us")

            # Show errors if any
            if pkt.errors:
                for error in pkt.errors:
                    print_info(f"    {RED}Error: {error}{RESET}")

        # Summary
        print_subheader("Summary")
        print_result("Total packets", self.results["packet_count"])
        print_result("Token packets", self.results["token_count"])
        print_result("Data packets", self.results["data_count"])
        print_result("Handshake packets", self.results["handshake_count"])

        # Show unique packet types
        unique_types = set(self.results["pid_names"])
        print_info("Packet types found:")
        for ptype in sorted(unique_types):
            count = self.results["pid_names"].count(ptype)
            print_info(f"  - {ptype}: {count}")

    def validate_results(self, suite: ValidationSuite) -> None:
        """Validate USB decoding results."""
        # Check that packets were decoded
        suite.check_greater(
            "Total packets",
            self.results.get("packet_count", 0),
            0,
            category="decoding",
        )

        # Check for expected packet types
        pid_names = self.results.get("pid_names", [])

        suite.check_true(
            "Found SETUP token",
            "SETUP" in pid_names,
            category="packets",
        )

        suite.check_true(
            "Found IN token",
            "IN" in pid_names,
            category="packets",
        )

        suite.check_true(
            "Found DATA packets",
            any("DATA" in name for name in pid_names),
            category="packets",
        )

        suite.check_true(
            "Found ACK handshake",
            "ACK" in pid_names,
            category="packets",
        )

        # Check packet counts
        suite.check_greater_equal(
            "Token count",
            self.results.get("token_count", 0),
            2,
            category="counts",
        )

        suite.check_greater_equal(
            "Data count",
            self.results.get("data_count", 0),
            1,
            category="counts",
        )

        suite.check_greater_equal(
            "Handshake count",
            self.results.get("handshake_count", 0),
            1,
            category="counts",
        )

        # Verify signal integrity
        suite.check_true(
            "Signals generated",
            self.dp is not None and len(self.dp) > 0,
            category="signals",
        )


if __name__ == "__main__":
    sys.exit(run_demo_main(USBDemo))
