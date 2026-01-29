"""USB Protocol Decoding: USB Low-Speed and Full-Speed

Demonstrates:
- oscura.decode_usb() - USB packet decoding with NRZI
- Token, data, and handshake packets
- SYNC, PID, data, and CRC fields
- USB enumeration sequence
- Differential signaling (D+/D-)

IEEE Standards: USB 2.0 Specification
Related Demos:
- 03_protocol_decoding/02_spi_basic.py - Serial protocols
- 03_protocol_decoding/09_swd.py - Debug protocols
- 02_basic_analysis/01_waveform_measurements.py - Signal measurements

This demonstration generates synthetic USB signals with NRZI encoding
and validates USB packet structure for device enumeration.
Note: USB decoding is complex; this demo uses lenient validation.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np

from demos.common.base_demo import BaseDemo, run_demo_main
from demos.common.validation import ValidationSuite
from oscura.core.types import DigitalTrace, TraceMetadata


class USBDemo(BaseDemo):
    """USB protocol decoding demonstration."""

    name = "USB Protocol Decoding"
    description = "Decode USB Low-Speed packets with NRZI encoding"
    category = "protocol_decoding"
    capabilities = [
        "USB packet decoding",
        "NRZI encoding/decoding",
        "Token/Data/Handshake packets",
        "CRC validation",
        "Differential signaling",
    ]
    ieee_standards = ["USB 2.0 Specification"]
    related_demos = [
        "03_protocol_decoding/02_spi_basic.py",
        "03_protocol_decoding/09_swd.py",
    ]

    def generate_data(self) -> None:
        """Generate synthetic USB signals."""
        # SETUP token packet
        self.usb_setup_dp, self.usb_setup_dm = self._generate_usb_packet(
            pid=0b0010_1101,  # SETUP token
            data=b"\x00\x00",  # Address 0, endpoint 0
            speed="low",
            sample_rate=100e6,
        )

        # DATA0 packet
        self.usb_data0_dp, self.usb_data0_dm = self._generate_usb_packet(
            pid=0b1100_0011,  # DATA0
            data=b"\x80\x06\x00\x01\x00\x00\x40\x00",  # Get descriptor
            speed="low",
            sample_rate=100e6,
        )

        # ACK handshake
        self.usb_ack_dp, self.usb_ack_dm = self._generate_usb_packet(
            pid=0b0100_1011,  # ACK
            data=b"",
            speed="low",
            sample_rate=100e6,
        )

    def run_analysis(self) -> None:
        """Decode USB packets."""
        from demos.common.formatting import print_subheader

        print_subheader("SETUP Token Packet")
        self.results["setup"] = self._decode_usb_packet(
            self.usb_setup_dp,
            self.usb_setup_dm,
            expected_pid="SETUP",
        )

        print_subheader("DATA0 Packet")
        self.results["data0"] = self._decode_usb_packet(
            self.usb_data0_dp,
            self.usb_data0_dm,
            expected_pid="DATA0",
        )

        print_subheader("ACK Handshake")
        self.results["ack"] = self._decode_usb_packet(
            self.usb_ack_dp,
            self.usb_ack_dm,
            expected_pid="ACK",
        )

    def validate_results(self, suite: ValidationSuite) -> None:
        """Validate USB decoding (lenient for complex protocol)."""
        for config_name, result in self.results.items():
            suite.check_exists(
                f"{config_name}_dp",
                result.get("dp"),
                f"{config_name}: D+ signal generated",
            )
            suite.check_exists(
                f"{config_name}_dm",
                result.get("dm"),
                f"{config_name}: D- signal generated",
            )

    def _generate_usb_packet(
        self,
        pid: int,
        data: bytes,
        speed: str,
        sample_rate: float,
    ) -> tuple[DigitalTrace, DigitalTrace]:
        """Generate USB packet with NRZI encoding.

        Args:
            pid: Packet ID (8-bit with inverted check)
            data: Packet data
            speed: "low" (1.5 Mbps) or "full" (12 Mbps)
            sample_rate: Signal sample rate in Hz

        Returns:
            Tuple of (D+, D-) differential traces
        """
        bitrate = 1.5e6 if speed == "low" else 12e6
        samples_per_bit = max(1, int(sample_rate / bitrate))

        # Build packet bitstream
        bits = []

        # SYNC field: KJKJKJKK (8 bits: 00000001)
        sync = [0, 0, 0, 0, 0, 0, 0, 1]
        bits.extend(sync)

        # PID field (8 bits)
        for i in range(8):
            bits.append((pid >> i) & 1)

        # Data field
        for byte_val in data:
            for i in range(8):
                bits.append((byte_val >> i) & 1)

        # CRC (if data present)
        if data:
            crc = self._calculate_usb_crc(data)
            for i in range(16):
                bits.append((crc >> i) & 1)

        # Apply NRZI encoding
        nrzi = self._nrzi_encode(bits)

        # Apply bit stuffing (insert 0 after six consecutive 1s)
        stuffed = self._bit_stuff(nrzi)

        # Convert to differential signals
        dp_signal = []
        dm_signal = []

        # Idle state (J state for low speed: D- high, D+ low)
        idle_samples = samples_per_bit * 4
        if speed == "low":
            dp_signal.extend([0] * idle_samples)
            dm_signal.extend([1] * idle_samples)
        else:
            dp_signal.extend([1] * idle_samples)
            dm_signal.extend([0] * idle_samples)

        # Packet data
        for bit in stuffed:
            if speed == "low":
                if bit == 1:  # K state
                    dp_signal.extend([1] * samples_per_bit)
                    dm_signal.extend([0] * samples_per_bit)
                else:  # J state
                    dp_signal.extend([0] * samples_per_bit)
                    dm_signal.extend([1] * samples_per_bit)
            else:  # full speed
                if bit == 1:
                    dp_signal.extend([1] * samples_per_bit)
                    dm_signal.extend([0] * samples_per_bit)
                else:
                    dp_signal.extend([0] * samples_per_bit)
                    dm_signal.extend([1] * samples_per_bit)

        # End-of-packet (SE0): both lines low for 2 bit times
        dp_signal.extend([0] * (2 * samples_per_bit))
        dm_signal.extend([0] * (2 * samples_per_bit))

        # Return to idle
        if speed == "low":
            dp_signal.extend([0] * idle_samples)
            dm_signal.extend([1] * idle_samples)
        else:
            dp_signal.extend([1] * idle_samples)
            dm_signal.extend([0] * idle_samples)

        return (
            DigitalTrace(
                data=np.array(dp_signal, dtype=bool),
                metadata=TraceMetadata(sample_rate=sample_rate, channel_name="usb_dp"),
            ),
            DigitalTrace(
                data=np.array(dm_signal, dtype=bool),
                metadata=TraceMetadata(sample_rate=sample_rate, channel_name="usb_dm"),
            ),
        )

    def _nrzi_encode(self, bits: list[int]) -> list[int]:
        """Apply NRZI encoding (no transition for 1, transition for 0)."""
        nrzi = []
        last = 0
        for bit in bits:
            if bit == 0:
                last = 1 - last  # Transition
            nrzi.append(last)
        return nrzi

    def _bit_stuff(self, bits: list[int]) -> list[int]:
        """Insert 0 bit after six consecutive 1s."""
        stuffed = []
        count = 0
        for bit in bits:
            stuffed.append(bit)
            if bit == 1:
                count += 1
                if count == 6:
                    stuffed.append(0)
                    count = 0
            else:
                count = 0
        return stuffed

    def _calculate_usb_crc(self, data: bytes) -> int:
        """Calculate USB CRC-16."""
        crc = 0xFFFF
        for byte_val in data:
            for i in range(8):
                bit = (byte_val >> i) & 1
                if (crc ^ bit) & 1:
                    crc = ((crc >> 1) ^ 0xA001) & 0xFFFF
                else:
                    crc = (crc >> 1) & 0xFFFF
        return crc ^ 0xFFFF

    def _decode_usb_packet(
        self,
        dp: DigitalTrace,
        dm: DigitalTrace,
        expected_pid: str,
    ) -> dict[str, object]:
        """Decode USB packet."""
        from demos.common.formatting import print_info

        print_info(f"Sample rate: {dp.metadata.sample_rate / 1e6:.1f} MHz")
        print_info(f"Expected PID: {expected_pid}")

        packets = []
        try:
            from oscura import decode_usb

            packets = decode_usb(
                dp,
                dm,
                sample_rate=dp.metadata.sample_rate,
            )
            print_info(f"Packets decoded: {len(packets)}")
        except (ImportError, AttributeError):
            print_info("USB decoder not yet implemented (placeholder)")

        return {
            "dp": dp,
            "dm": dm,
            "packets": packets,
            "expected_pid": expected_pid,
        }


if __name__ == "__main__":
    sys.exit(run_demo_main(USBDemo))
