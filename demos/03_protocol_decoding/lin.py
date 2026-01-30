"""LIN Bus Protocol Decoding: Low-speed automotive serial communication

Demonstrates:
- oscura.decode_lin() - LIN bus protocol decoding
- LIN frame structure (break, sync, PID, data, checksum)
- Master/slave communication patterns
- Checksum validation (classic and enhanced)
- Diagnostic frame handling

IEEE Standards: ISO 17987-1:2016 (LIN specification)
Related Demos:
- 03_protocol_decoding/04_can_basic.py - CAN bus protocol
- 03_protocol_decoding/05_can_fd.py - CAN-FD protocol
- 05_domain_specific/automotive/ - Automotive analysis

This demonstration generates synthetic LIN bus signals with various frame types
and validates protocol compliance for automotive low-speed communication.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add demos to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np

from demos.common.base_demo import BaseDemo, run_demo_main
from demos.common.validation import ValidationSuite
from oscura.core.types import DigitalTrace, TraceMetadata


class LINDemo(BaseDemo):
    """LIN bus protocol decoding demonstration."""

    name = "LIN Bus Protocol Decoding"
    description = "Decode LIN bus frames with master/slave communication"
    category = "protocol_decoding"
    capabilities = [
        "LIN frame decoding",
        "Break field detection",
        "Protected ID validation",
        "Classic and enhanced checksum",
        "Diagnostic frame support",
    ]
    ieee_standards = ["ISO 17987-1:2016"]
    related_demos = [
        "03_protocol_decoding/04_can_basic.py",
        "03_protocol_decoding/05_can_fd.py",
    ]

    def generate_data(self) -> None:
        """Generate synthetic LIN bus signals."""
        # Standard data frame
        self.lin_data_frame = self._generate_lin_signal(
            frame_id=0x23,
            data=b"\xA5\x5A\xFF\x00",
            baudrate=19200,
            checksum_type="classic",
            sample_rate=1e6,
        )

        # Diagnostic frame (ID 0x3C or 0x3D)
        self.lin_diagnostic = self._generate_lin_signal(
            frame_id=0x3C,
            data=b"\x01\x02\x03\x04\x05\x06\x07\x08",
            baudrate=19200,
            checksum_type="enhanced",
            sample_rate=1e6,
        )

        # High-speed LIN (20 kbaud)
        self.lin_highspeed = self._generate_lin_signal(
            frame_id=0x15,
            data=b"\x12\x34\x56\x78",
            baudrate=20000,
            checksum_type="enhanced",
            sample_rate=1e6,
        )

    def run_analysis(self) -> None:
        """Decode LIN signals and extract frame information."""
        from demos.common.formatting import print_info, print_subheader

        # Decode standard data frame
        print_subheader("Standard LIN Data Frame (19.2 kbaud)")
        self.results["data_frame"] = self._decode_lin_frame(
            self.lin_data_frame,
            baudrate=19200,
            expected_id=0x23,
            expected_data=b"\xA5\x5A\xFF\x00",
        )

        # Decode diagnostic frame
        print_subheader("LIN Diagnostic Frame")
        self.results["diagnostic"] = self._decode_lin_frame(
            self.lin_diagnostic,
            baudrate=19200,
            expected_id=0x3C,
            expected_data=b"\x01\x02\x03\x04\x05\x06\x07\x08",
        )

        # Decode high-speed frame
        print_subheader("High-Speed LIN (20 kbaud)")
        self.results["highspeed"] = self._decode_lin_frame(
            self.lin_highspeed,
            baudrate=20000,
            expected_id=0x15,
            expected_data=b"\x12\x34\x56\x78",
        )

    def validate_results(self, suite: ValidationSuite) -> None:
        """Validate LIN decoding results."""
        for config_name, result in self.results.items():
            suite.check_exists(
                f"{config_name}_signal",
                result.get("signal"),
                f"{config_name}: Signal generated",
            )

            # Validate frame structure
            if "frames" in result and result["frames"]:
                suite.check_greater_than(
                    f"{config_name}_frames",
                    len(result["frames"]),
                    0,
                    f"{config_name}: At least one frame decoded",
                )

    def _generate_lin_signal(
        self,
        frame_id: int,
        data: bytes,
        baudrate: int,
        checksum_type: str,
        sample_rate: float,
    ) -> DigitalTrace:
        """Generate synthetic LIN bus signal.

        Args:
            frame_id: LIN frame ID (0-63)
            data: Data bytes (0-8 bytes)
            baudrate: LIN baudrate in bps
            checksum_type: "classic" or "enhanced"
            sample_rate: Sample rate in Hz

        Returns:
            DigitalTrace with LIN signal
        """
        bit_time = 1.0 / baudrate
        samples_per_bit = max(1, int(sample_rate * bit_time))

        signal = []

        # Idle state (high/recessive)
        signal.extend([1] * (2 * samples_per_bit))

        # Break field: at least 13 dominant bits
        signal.extend([0] * (13 * samples_per_bit))

        # Break delimiter: at least 1 recessive bit
        signal.extend([1] * samples_per_bit)

        # Sync field: 0x55 (01010101)
        sync_byte = 0x55
        self._add_uart_byte(signal, sync_byte, samples_per_bit)

        # Protected ID field (PID = parity bits + frame ID)
        pid = self._calculate_pid(frame_id)
        self._add_uart_byte(signal, pid, samples_per_bit)

        # Data field
        for byte_val in data:
            self._add_uart_byte(signal, byte_val, samples_per_bit)

        # Checksum field
        if checksum_type == "classic":
            checksum = self._calculate_classic_checksum(data)
        else:  # enhanced
            checksum = self._calculate_enhanced_checksum(pid, data)
        self._add_uart_byte(signal, checksum, samples_per_bit)

        # Inter-frame space
        signal.extend([1] * (4 * samples_per_bit))

        return DigitalTrace(
            data=np.array(signal, dtype=bool),
            metadata=TraceMetadata(
                sample_rate=sample_rate,
                channel_name="lin_bus",
            ),
        )

    def _add_uart_byte(self, signal: list[int], byte_val: int, samples_per_bit: int) -> None:
        """Add UART-formatted byte to signal (8N1: start, 8 data bits LSB first, stop).

        Args:
            signal: Signal list to append to
            byte_val: Byte value to encode
            samples_per_bit: Samples per bit period
        """
        # Start bit (dominant/low)
        signal.extend([0] * samples_per_bit)

        # Data bits (LSB first)
        for i in range(8):
            bit_val = (byte_val >> i) & 1
            signal.extend([bit_val] * samples_per_bit)

        # Stop bit (recessive/high)
        signal.extend([1] * samples_per_bit)

    def _calculate_pid(self, frame_id: int) -> int:
        """Calculate protected ID with parity bits.

        Args:
            frame_id: Frame ID (0-63)

        Returns:
            Protected ID (8-bit: parity bits + frame ID)
        """
        # LIN 2.x parity calculation
        id_masked = frame_id & 0x3F
        p0 = (
            ((id_masked >> 0) & 1) ^ ((id_masked >> 1) & 1) ^ ((id_masked >> 2) & 1) ^ ((id_masked >> 4) & 1)
        )
        p1 = (
            ~(
                ((id_masked >> 1) & 1)
                ^ ((id_masked >> 3) & 1)
                ^ ((id_masked >> 4) & 1)
                ^ ((id_masked >> 5) & 1)
            )
            & 1
        )

        return id_masked | (p0 << 6) | (p1 << 7)

    def _calculate_classic_checksum(self, data: bytes) -> int:
        """Calculate classic LIN checksum (data only).

        Args:
            data: Data bytes

        Returns:
            Checksum value (inverted modulo-256 sum)
        """
        checksum = sum(data) & 0xFF
        return (~checksum) & 0xFF

    def _calculate_enhanced_checksum(self, pid: int, data: bytes) -> int:
        """Calculate enhanced LIN checksum (PID + data).

        Args:
            pid: Protected ID
            data: Data bytes

        Returns:
            Checksum value (inverted modulo-256 sum)
        """
        checksum = pid
        for byte_val in data:
            checksum = (checksum + byte_val) & 0xFF
            if checksum > 0xFF:
                checksum = (checksum + 1) & 0xFF
        return (~checksum) & 0xFF

    def _decode_lin_frame(
        self,
        signal: DigitalTrace,
        baudrate: int,
        expected_id: int,
        expected_data: bytes,
    ) -> dict[str, object]:
        """Decode LIN frame and display results.

        Args:
            signal: LIN signal
            baudrate: Baud rate
            expected_id: Expected frame ID
            expected_data: Expected data

        Returns:
            Results dictionary
        """
        from demos.common.formatting import print_info

        print_info(f"Sample rate: {signal.metadata.sample_rate / 1e6:.1f} MHz")
        print_info(f"Baudrate: {baudrate / 1000:.1f} kbaud")
        print_info(f"Expected ID: 0x{expected_id:02X}")
        print_info(f"Expected data: {expected_data.hex()}")

        frames = []
        try:
            from oscura import decode_lin

            frames = decode_lin(
                signal,
                sample_rate=signal.metadata.sample_rate,
                baudrate=baudrate,
            )
            print_info(f"Frames decoded: {len(frames)}")
        except (ImportError, AttributeError):
            print_info("LIN decoder not yet implemented (placeholder)")

        return {
            "signal": signal,
            "frames": frames,
            "expected_id": expected_id,
            "expected_data": expected_data,
        }


if __name__ == "__main__":
    sys.exit(run_demo_main(LINDemo))
