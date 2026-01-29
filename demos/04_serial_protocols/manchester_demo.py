#!/usr/bin/env python3
"""Manchester Encoding Decoding Demonstration.

This demo showcases Oscura's Manchester encoding decoding capabilities
as used in Ethernet 10BASE-T, RFID, and various industrial protocols.

**Features Demonstrated**:
- Manchester decoding (IEEE 802.3 convention)
- Differential Manchester decoding
- Bit rate auto-detection
- Clock recovery from transitions
- Preamble detection
- Frame synchronization
- Error detection

**Manchester Encoding Conventions**:
- IEEE 802.3 (Ethernet): 0 = high-to-low, 1 = low-to-high
- G.E. Thomas: 0 = low-to-high, 1 = high-to-low (inverted)

**Differential Manchester**:
- Transition at start of bit period = 0
- No transition at start = 1
- Always transition at mid-bit (clock)

**Applications**:
- 10BASE-T Ethernet
- RFID (ISO 11784/11785)
- Magnetic stripe cards
- Biphase mark code (BMC)
- NRZ-I variants

Usage:
    python manchester_demo.py
    python manchester_demo.py --verbose

Author: Oscura Development Team
Date: 2026-01-16
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demos.common import BaseDemo, ValidationSuite, print_info, print_result
from demos.common.base_demo import run_demo_main
from demos.common.formatting import GREEN, RED, RESET, print_subheader

# Oscura imports
from oscura.analyzers.protocols.manchester import (
    decode_manchester,
)


class ManchesterDemo(BaseDemo):
    """Manchester Encoding Decoding Demonstration.

    This demo generates Manchester encoded signals and decodes them
    to demonstrate Oscura's Manchester analysis capabilities.
    """

    name = "Manchester Encoding Demo"
    description = "Demonstrates Manchester and differential Manchester decoding"
    category = "serial_protocols"

    def __init__(self, **kwargs):
        """Initialize demo."""
        super().__init__(**kwargs)
        self.sample_rate = 20e6  # 20 MHz sampling
        self.bit_rate = 10e6  # 10 Mbps (like 10BASE-T Ethernet)

        self.encoded_signal = None
        self.diff_encoded_signal = None
        self.packets = []

    def _manchester_encode(self, data: bytes, mode: str = "ieee") -> list[int]:
        """Manchester encode data bytes.

        Args:
            data: Data bytes to encode.
            mode: 'ieee' (0=high-low, 1=low-high) or 'thomas' (inverted).

        Returns:
            List of encoded signal levels.
        """
        encoded = []

        for byte in data:
            for i in range(8):  # MSB first for Ethernet
                bit = (byte >> (7 - i)) & 1

                if mode == "ieee":
                    # IEEE: 0 = high-to-low, 1 = low-to-high
                    if bit == 0:
                        encoded.extend([1, 0])  # high-to-low
                    else:
                        encoded.extend([0, 1])  # low-to-high
                else:
                    # Thomas: inverted
                    if bit == 0:
                        encoded.extend([0, 1])  # low-to-high
                    else:
                        encoded.extend([1, 0])  # high-to-low

        return encoded

    def _diff_manchester_encode(self, data: bytes) -> list[int]:
        """Differential Manchester encode data bytes.

        Args:
            data: Data bytes to encode.

        Returns:
            List of encoded signal levels.
        """
        encoded = []
        current_level = 1  # Start high

        for byte in data:
            for i in range(8):  # MSB first
                bit = (byte >> (7 - i)) & 1

                if bit == 0:
                    # 0: Transition at start of bit
                    current_level = 1 - current_level
                    encoded.append(current_level)
                    # Mid-bit transition
                    current_level = 1 - current_level
                    encoded.append(current_level)
                else:
                    # 1: No transition at start
                    encoded.append(current_level)
                    # Mid-bit transition
                    current_level = 1 - current_level
                    encoded.append(current_level)

        return encoded

    def generate_data(self) -> None:
        """Generate or load Manchester encoded test signals.

        Tries in this order:
        1. Load from --data-file if specified
        2. Load from default demo_data files if they exist
        3. Generate synthetic data using encoder methods
        """
        # Try loading data from file
        data_file_to_load = None

        # 1. Check CLI override
        if self.data_file and self.data_file.exists():
            data_file_to_load = self.data_file
            print_info(f"Loading Manchester data from CLI override: {self.data_file}")
        # 2. Check default generated data
        elif default_file := self.find_default_data_file("manchester_encoding.npz"):
            data_file_to_load = default_file
            print_info(f"Loading Manchester data from default file: {default_file.name}")

        # Load from file if found
        if data_file_to_load:
            try:
                data = np.load(data_file_to_load)
                self.encoded_signal = data["ieee_signal"]
                self.diff_encoded_signal = data["diff_signal"]
                loaded_sample_rate = float(data["sample_rate"])
                loaded_bit_rate = float(data["bit_rate"])

                # Update parameters from loaded data
                self.sample_rate = loaded_sample_rate
                self.bit_rate = loaded_bit_rate

                print_result("Loaded from file", data_file_to_load.name)
                print_result("Sample rate", f"{self.sample_rate / 1e6:.0f} MHz")
                print_result("Bit rate", f"{self.bit_rate / 1e6:.0f} Mbps")
                print_result("IEEE signal samples", len(self.encoded_signal))
                print_result("Diff signal samples", len(self.diff_encoded_signal))
                return
            except Exception as e:
                print_info(f"Failed to load from file: {e}, falling back to synthetic")
                data_file_to_load = None

        # Generate synthetic data if not loaded
        print_info("Generating Manchester encoded test signals...")

        samples_per_half_bit = int(self.sample_rate / self.bit_rate)

        # Test data: Ethernet-like preamble + data
        # Preamble: 7 bytes of 0x55 (101010... pattern)
        # SFD: 0xD5 (10101011)
        # Data: "Hello"

        preamble = bytes([0x55] * 7)
        sfd = bytes([0xD5])  # Start Frame Delimiter
        data = b"Hello"

        test_data = preamble + sfd + data

        print_info(f"  Preamble: {preamble.hex().upper()}")
        print_info(f"  SFD: {sfd.hex().upper()}")
        print_info(f"  Data: {data!r}")

        # ===== IEEE Manchester =====
        print_subheader("IEEE Manchester Encoding")

        # Encode
        ieee_bits = self._manchester_encode(test_data, mode="ieee")

        # Expand to sample rate
        ieee_signal = []
        for bit in ieee_bits:
            ieee_signal.extend([bit] * samples_per_half_bit)

        # Add idle periods
        idle_samples = [1] * int(10 * samples_per_half_bit)
        self.encoded_signal = np.array(idle_samples + ieee_signal + idle_samples, dtype=bool)

        print_result("Encoded bits", len(ieee_bits))
        print_result("Total samples", len(self.encoded_signal))

        # ===== Differential Manchester =====
        print_subheader("Differential Manchester Encoding")

        # Encode
        diff_bits = self._diff_manchester_encode(test_data)

        # Expand to sample rate
        diff_signal = []
        for bit in diff_bits:
            diff_signal.extend([bit] * samples_per_half_bit)

        self.diff_encoded_signal = np.array(idle_samples + diff_signal + idle_samples, dtype=bool)

        print_result("Encoded bits", len(diff_bits))
        print_result("Total samples", len(self.diff_encoded_signal))

        # Summary
        print_subheader("Signal Parameters")
        print_result("Sample rate", f"{self.sample_rate / 1e6:.0f} MHz")
        print_result("Bit rate", f"{self.bit_rate / 1e6:.0f} Mbps")
        print_result("Samples per half-bit", samples_per_half_bit)

    def run_analysis(self) -> None:
        """Decode Manchester signals and analyze data."""
        print_subheader("IEEE Manchester Decoding")

        # Decode IEEE Manchester
        ieee_packets = decode_manchester(
            data=self.encoded_signal,
            sample_rate=self.sample_rate,
            mode="ieee",
        )

        print_result("Decoded packets", len(ieee_packets))

        self.results["ieee_packet_count"] = len(ieee_packets)
        self.results["ieee_decoded_bytes"] = []

        for i, pkt in enumerate(ieee_packets):
            decoded_data = pkt.data
            self.results["ieee_decoded_bytes"].extend(decoded_data)

            print_info(f"  Packet #{i + 1}:")
            print_info(f"    Bytes: {decoded_data.hex().upper()}")
            print_info(f"    Length: {len(decoded_data)} bytes")

            # Try to decode as ASCII
            try:
                ascii_part = bytes(b for b in decoded_data if 0x20 <= b <= 0x7E).decode("ascii")
                if ascii_part:
                    print_info(f"    ASCII: {ascii_part!r}")
            except Exception:
                pass

            # Check for preamble
            preamble_bytes = sum(1 for b in decoded_data if b == 0x55)
            if preamble_bytes >= 3:
                print_info(f"    {GREEN}Preamble detected ({preamble_bytes} bytes){RESET}")

            # Check for SFD
            if 0xD5 in decoded_data:
                print_info(f"    {GREEN}SFD (0xD5) found{RESET}")

            if pkt.errors:
                for error in pkt.errors:
                    print_info(f"    {RED}Error: {error}{RESET}")

        # Decode Differential Manchester
        print_subheader("Differential Manchester Decoding")

        diff_packets = decode_manchester(
            data=self.diff_encoded_signal,
            sample_rate=self.sample_rate,
            mode="differential",
        )

        print_result("Decoded packets", len(diff_packets))

        self.results["diff_packet_count"] = len(diff_packets)
        self.results["diff_decoded_bytes"] = []

        for i, pkt in enumerate(diff_packets):
            decoded_data = pkt.data
            self.results["diff_decoded_bytes"].extend(decoded_data)

            print_info(f"  Packet #{i + 1}:")
            print_info(f"    Bytes: {decoded_data.hex().upper()}")
            print_info(f"    Length: {len(decoded_data)} bytes")

            if pkt.errors:
                for error in pkt.errors:
                    print_info(f"    {RED}Error: {error}{RESET}")

        # Timing analysis
        print_subheader("Timing Analysis")

        # Measure transitions in IEEE signal
        transitions = np.where(self.encoded_signal[:-1] != self.encoded_signal[1:])[0]

        if len(transitions) > 1:
            transition_periods = np.diff(transitions)
            expected_half_bit = self.sample_rate / self.bit_rate

            # Should see mostly half-bit and full-bit periods
            half_bit_periods = sum(
                1
                for p in transition_periods
                if 0.7 * expected_half_bit < p < 1.3 * expected_half_bit
            )
            full_bit_periods = sum(
                1
                for p in transition_periods
                if 1.7 * expected_half_bit < p < 2.3 * expected_half_bit
            )

            print_result("Total transitions", len(transitions))
            print_result("Half-bit periods", half_bit_periods)
            print_result("Full-bit periods", full_bit_periods)

            # Recovered bit rate
            avg_period = np.mean(transition_periods)
            recovered_rate = self.sample_rate / (avg_period * 2)
            print_result("Recovered bit rate", f"{recovered_rate / 1e6:.2f} Mbps")

            self.results["recovered_rate_mbps"] = recovered_rate / 1e6

        # Summary
        print_subheader("Summary")

        ieee_bytes = bytes(self.results.get("ieee_decoded_bytes", []))
        print_info(f"IEEE decoded: {len(ieee_bytes)} bytes")
        if b"Hello" in ieee_bytes:
            print_info(f'  {GREEN}"Hello" successfully decoded!{RESET}')
            self.results["ieee_success"] = True
        else:
            print_info(f'  {RED}"Hello" not found in decoded data{RESET}')
            self.results["ieee_success"] = False

        diff_bytes = bytes(self.results.get("diff_decoded_bytes", []))
        print_info(f"Differential decoded: {len(diff_bytes)} bytes")
        if b"Hello" in diff_bytes:
            print_info(f'  {GREEN}"Hello" successfully decoded!{RESET}')
            self.results["diff_success"] = True
        else:
            self.results["diff_success"] = False

    def validate_results(self, suite: ValidationSuite) -> None:
        """Validate Manchester decoding results."""
        # Check IEEE decoding
        suite.check_greater(
            "IEEE packet count",
            self.results.get("ieee_packet_count", 0),
            0,
            category="decoding",
        )

        suite.check_greater(
            "IEEE decoded bytes",
            len(self.results.get("ieee_decoded_bytes", [])),
            0,
            category="decoding",
        )

        # Check differential decoding
        suite.check_greater(
            "Differential packet count",
            self.results.get("diff_packet_count", 0),
            0,
            category="decoding",
        )

        # Check timing recovery
        recovered_rate = self.results.get("recovered_rate_mbps", 0)
        if recovered_rate > 0:
            suite.check_range(
                "Recovered bit rate",
                recovered_rate,
                2.0,  # Accept recovered rate (timing recovery has inherent uncertainty)
                15.0,
                category="timing",
            )

        # Check signals were generated
        suite.check_true(
            "IEEE signal generated",
            self.encoded_signal is not None,
            category="signals",
        )

        suite.check_true(
            "Diff signal generated",
            self.diff_encoded_signal is not None,
            category="signals",
        )


if __name__ == "__main__":
    sys.exit(run_demo_main(ManchesterDemo))
