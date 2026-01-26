#!/usr/bin/env python3
"""1-Wire Protocol Decoding Demonstration.

This demo showcases Oscura's 1-Wire protocol decoding capabilities
for temperature sensors (DS18B20), iButtons, and other 1-Wire devices.

**Features Demonstrated**:
- 1-Wire bus signal decoding
- Reset/presence pulse detection
- ROM command decoding (Read ROM, Match ROM, Search ROM)
- Function command decoding
- CRC-8 validation
- Temperature data conversion (DS18B20)
- Overdrive mode support

**1-Wire Timing (Standard Speed)**:
- Reset pulse: Master pulls low for >= 480 us
- Presence pulse: Slave pulls low for 60-240 us
- Write 0: Master holds low for 60-120 us
- Write 1: Master holds low for 1-15 us
- Read: Master samples at ~15 us

**Common ROM Commands**:
- 0x33: Read ROM (read 64-bit unique ID)
- 0x55: Match ROM (address specific device)
- 0xF0: Search ROM (discover devices)
- 0xCC: Skip ROM (address all devices)

Usage:
    python onewire_demo.py
    python onewire_demo.py --verbose

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
from oscura.analyzers.protocols.onewire import (
    decode_onewire,
)


class OneWireDemo(BaseDemo):
    """1-Wire Protocol Decoding Demonstration.

    This demo generates 1-Wire bus signals demonstrating typical
    DS18B20 temperature sensor communication.
    """

    def __init__(self, **kwargs):
        """Initialize demo."""
        super().__init__(
            name="1-Wire Protocol Demo",
            description="Demonstrates 1-Wire protocol decoding for temperature sensors",
            **kwargs,
        )
        self.sample_rate = 1e6  # 1 MHz sampling
        self.bus_signal = None
        self.packets = []

    def _crc8_onewire(self, data: bytes) -> int:
        """Calculate 1-Wire CRC-8.

        Uses polynomial x^8 + x^5 + x^4 + 1 (0x8C reflected).

        Args:
            data: Data bytes.

        Returns:
            CRC-8 value.
        """
        crc = 0
        for byte in data:
            crc ^= byte
            for _ in range(8):
                if crc & 1:
                    crc = (crc >> 1) ^ 0x8C
                else:
                    crc >>= 1
        return crc

    def _generate_reset_presence(self, bits: list[int], samples_per_us: float) -> None:
        """Generate reset and presence pulse.

        Args:
            bits: Bit list to append to.
            samples_per_us: Samples per microsecond.
        """
        # Reset pulse: Low for 480 us
        bits.extend([0] * int(480 * samples_per_us))

        # Master release (high)
        bits.extend([1] * int(15 * samples_per_us))

        # Presence pulse from slave: Low for 120 us
        bits.extend([0] * int(120 * samples_per_us))

        # Release
        bits.extend([1] * int(300 * samples_per_us))

    def _generate_write_bit(self, bits: list[int], bit_val: int, samples_per_us: float) -> None:
        """Generate write bit timing.

        Args:
            bits: Bit list to append to.
            bit_val: Bit value (0 or 1).
            samples_per_us: Samples per microsecond.
        """
        if bit_val == 0:
            # Write 0: Low for 60 us, then high
            bits.extend([0] * int(60 * samples_per_us))
            bits.extend([1] * int(10 * samples_per_us))
        else:
            # Write 1: Low for 6 us, then high
            bits.extend([0] * int(6 * samples_per_us))
            bits.extend([1] * int(64 * samples_per_us))

    def _generate_read_bit(self, bits: list[int], bit_val: int, samples_per_us: float) -> None:
        """Generate read bit timing.

        Args:
            bits: Bit list to append to.
            bit_val: Bit value from slave.
            samples_per_us: Samples per microsecond.
        """
        # Master initiates with low pulse
        bits.extend([0] * int(2 * samples_per_us))

        # Data from slave
        if bit_val == 0:
            bits.extend([0] * int(58 * samples_per_us))
        else:
            bits.extend([1] * int(58 * samples_per_us))

        # Recovery time
        bits.extend([1] * int(10 * samples_per_us))

    def _generate_write_byte(self, bits: list[int], byte_val: int, samples_per_us: float) -> None:
        """Generate write byte (LSB first).

        Args:
            bits: Bit list to append to.
            byte_val: Byte to write.
            samples_per_us: Samples per microsecond.
        """
        for i in range(8):
            bit = (byte_val >> i) & 1
            self._generate_write_bit(bits, bit, samples_per_us)

    def _generate_read_byte(self, bits: list[int], byte_val: int, samples_per_us: float) -> None:
        """Generate read byte (LSB first).

        Args:
            bits: Bit list to append to.
            byte_val: Byte value from slave.
            samples_per_us: Samples per microsecond.
        """
        for i in range(8):
            bit = (byte_val >> i) & 1
            self._generate_read_bit(bits, bit, samples_per_us)

    def generate_test_data(self) -> dict:
        """Generate 1-Wire test signals.

        Simulates DS18B20 temperature sensor communication:
        1. Reset + Presence
        2. Skip ROM (0xCC)
        3. Convert T (0x44)
        4. (wait for conversion)
        5. Reset + Presence
        6. Skip ROM (0xCC)
        7. Read Scratchpad (0xBE)
        8. Read 9 bytes of scratchpad
        """
        print_info("Generating 1-Wire test signals...")

        samples_per_us = self.sample_rate / 1e6
        bits = []

        # Initial idle
        bits.extend([1] * int(100 * samples_per_us))

        # ===== Transaction 1: Start temperature conversion =====
        print_info("  Transaction 1: Start temperature conversion")

        # Reset + Presence
        self._generate_reset_presence(bits, samples_per_us)

        # Skip ROM command (0xCC)
        print_info("    Sending Skip ROM (0xCC)")
        self._generate_write_byte(bits, 0xCC, samples_per_us)

        # Convert T command (0x44)
        print_info("    Sending Convert T (0x44)")
        self._generate_write_byte(bits, 0x44, samples_per_us)

        # Conversion time (simplified - just a gap)
        bits.extend([1] * int(1000 * samples_per_us))  # 1 ms gap

        # ===== Transaction 2: Read temperature =====
        print_info("  Transaction 2: Read temperature data")

        # Reset + Presence
        self._generate_reset_presence(bits, samples_per_us)

        # Skip ROM command (0xCC)
        print_info("    Sending Skip ROM (0xCC)")
        self._generate_write_byte(bits, 0xCC, samples_per_us)

        # Read Scratchpad command (0xBE)
        print_info("    Sending Read Scratchpad (0xBE)")
        self._generate_write_byte(bits, 0xBE, samples_per_us)

        # Read 9 bytes of scratchpad data
        # Byte 0: Temperature LSB
        # Byte 1: Temperature MSB
        # Byte 2: TH Register
        # Byte 3: TL Register
        # Byte 4: Configuration
        # Byte 5-7: Reserved
        # Byte 8: CRC

        # Temperature = 25.0625 C = 0x0191 (25 * 16 + 1)
        scratchpad = bytes(
            [
                0x91,
                0x01,  # Temperature LSB, MSB (25.0625 C)
                0x4B,
                0x46,  # TH, TL
                0x7F,  # Config (12-bit)
                0xFF,
                0xFF,
                0xFF,  # Reserved
            ]
        )
        crc = self._crc8_onewire(scratchpad)
        scratchpad = scratchpad + bytes([crc])

        print_info(f"    Reading scratchpad: {scratchpad.hex().upper()}")
        temp_raw = scratchpad[0] | (scratchpad[1] << 8)
        temp_c = temp_raw / 16.0
        print_info(f"    Temperature: {temp_c:.4f} C")

        for byte_val in scratchpad:
            self._generate_read_byte(bits, byte_val, samples_per_us)

        # Final idle
        bits.extend([1] * int(100 * samples_per_us))

        # Convert to numpy array
        self.bus_signal = np.array(bits, dtype=bool)

        print_result("Total samples", len(self.bus_signal))
        print_result("Duration", f"{len(self.bus_signal) / self.sample_rate * 1e3:.2f} ms")

        return {}

    def run_demonstration(self, data: dict) -> dict:
        """Decode 1-Wire signals and analyze transactions."""
        print_subheader("1-Wire Decoding")

        # Decode using convenience function
        self.packets = decode_onewire(
            data=self.bus_signal,
            sample_rate=self.sample_rate,
        )

        print_result("Decoded packets", len(self.packets))

        # Analyze packets (transaction-level)
        print_subheader("Packet Analysis")

        self.results["packet_count"] = len(self.packets)
        self.results["reset_count"] = 0
        self.results["command_bytes"] = []
        self.results["data_bytes"] = []

        for i, pkt in enumerate(self.packets):
            timestamp = pkt.timestamp * 1e3  # ms

            # The decoder emits transaction-level packets
            # Each transaction starts with a reset pulse
            rom_cmd = pkt.annotations.get("rom_command")
            rom_cmd_code = pkt.annotations.get("rom_command_code")
            byte_count = pkt.annotations.get("byte_count", 0)

            print_info(f"  Transaction #{i + 1} @ {timestamp:.3f} ms")

            # Each transaction implies a reset pulse
            if rom_cmd:
                self.results["reset_count"] += 1
                print_info("    Reset detected (transaction start)")

            # Parse transaction bytes
            if pkt.data:
                decoded_bytes = list(pkt.data)

                # First byte is the ROM command if present
                if rom_cmd_code is not None and len(decoded_bytes) > 0:
                    self.results["command_bytes"].append(rom_cmd_code)
                    print_info(f"    ROM Command: 0x{rom_cmd_code:02X} ({rom_cmd})")

                    # Rest are either function commands or data
                    if len(decoded_bytes) > 1:
                        # Second byte is typically function command
                        func_byte = decoded_bytes[1]
                        self.results["command_bytes"].append(func_byte)

                        func_names = {
                            0x44: "Convert T",
                            0x4E: "Write Scratchpad",
                            0xBE: "Read Scratchpad",
                            0x48: "Copy Scratchpad",
                            0xB8: "Recall E2",
                            0xB4: "Read Power Supply",
                        }
                        func_name = func_names.get(func_byte, "Unknown")
                        print_info(f"    Function: 0x{func_byte:02X} ({func_name})")

                        # Remaining bytes are data (if reading scratchpad)
                        if len(decoded_bytes) > 2:
                            data_bytes = decoded_bytes[2:]
                            self.results["data_bytes"].extend(data_bytes)
                            print_info(f"    Data: {len(data_bytes)} bytes")
                            for j, byte_val in enumerate(data_bytes):
                                print_info(f"      Byte {j}: 0x{byte_val:02X}")
                else:
                    # No ROM command - these might be data bytes
                    self.results["data_bytes"].extend(decoded_bytes)
                    print_info(f"    Data: {len(decoded_bytes)} bytes")

            # Show errors
            if pkt.errors:
                for error in pkt.errors:
                    print_info(f"    {RED}Error: {error}{RESET}")

        # Temperature interpretation
        print_subheader("Temperature Reading")

        data_bytes = self.results["data_bytes"]
        if len(data_bytes) >= 2:
            temp_lsb = data_bytes[0]
            temp_msb = data_bytes[1]
            temp_raw = temp_lsb | (temp_msb << 8)

            # Sign extend if negative
            if temp_raw & 0x8000:
                temp_raw = temp_raw - 0x10000

            temp_c = temp_raw / 16.0
            temp_f = temp_c * 9 / 5 + 32

            print_result("Temperature (raw)", f"0x{temp_raw & 0xFFFF:04X}")
            print_result("Temperature (C)", f"{temp_c:.4f}")
            print_result("Temperature (F)", f"{temp_f:.4f}")

            self.results["temperature_c"] = temp_c

            # Validate CRC if we have all 9 bytes
            if len(data_bytes) >= 9:
                scratchpad = bytes(data_bytes[:8])
                expected_crc = self._crc8_onewire(scratchpad)
                actual_crc = data_bytes[8]

                if expected_crc == actual_crc:
                    print_info(f"  {GREEN}CRC valid: 0x{actual_crc:02X}{RESET}")
                    self.results["crc_valid"] = True
                else:
                    print_info(
                        f"  {RED}CRC mismatch: expected 0x{expected_crc:02X}, "
                        f"got 0x{actual_crc:02X}{RESET}"
                    )
                    self.results["crc_valid"] = False

        # Summary
        print_subheader("Summary")
        print_result("Reset pulses", self.results["reset_count"])
        print_result("Commands sent", len(self.results["command_bytes"]))
        print_result("Bytes read", len(self.results["data_bytes"]))

        return self.results

    def validate(self, results: dict) -> bool:
        """Validate 1-Wire decoding results."""
        suite = ValidationSuite()

        # Check packets were decoded
        packet_count = results.get("packet_count", 0)
        suite.add_check("Packets decoded", packet_count > 0, f"Got {packet_count} packets")

        # Check for reset pulses
        reset_count = results.get("reset_count", 0)
        suite.add_check("Reset pulses detected", reset_count > 0, f"Got {reset_count} resets")

        # Check for command bytes
        command_bytes = len(results.get("command_bytes", []))
        suite.add_check("Command bytes decoded", command_bytes > 0, f"Got {command_bytes} commands")

        # Check signal was generated
        suite.add_check(
            "Signal generated",
            self.bus_signal is not None and len(self.bus_signal) > 0,
            f"Got {len(self.bus_signal) if self.bus_signal is not None else 0} samples",
        )

        suite.report()
        return suite.all_passed()


if __name__ == "__main__":
    sys.exit(run_demo_main(OneWireDemo))
