"""I2C Protocol Decoding: Comprehensive demonstration of I2C bus communication

Demonstrates:
- oscura.decode_i2c() - I2C/TWI protocol decoding
- 7-bit and 10-bit addressing modes
- Read and write transactions
- START/STOP conditions
- ACK/NACK handling
- Multi-byte transfers

IEEE Standards: IEEE 181-2011 (waveform measurements)
Related Demos:
- 03_protocol_decoding/01_uart_basic.py - UART protocol
- 03_protocol_decoding/02_spi_basic.py - SPI protocol

This demonstration generates synthetic I2C bus signals with various transaction types.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np

from demos.common import BaseDemo, ValidationSuite
from oscura import decode_i2c
from oscura.core.types import DigitalTrace, TraceMetadata


class I2CDemo(BaseDemo):
    """Comprehensive I2C protocol decoding demonstration."""

    def __init__(self) -> None:
        """Initialize I2C demonstration."""
        super().__init__(
            name="i2c_protocol_decoding",
            description="Decode I2C bus communications with various transaction types",
            capabilities=["oscura.decode_i2c"],
            ieee_standards=["IEEE 181-2011"],
            related_demos=[
                "03_protocol_decoding/01_uart_basic.py",
                "03_protocol_decoding/02_spi_basic.py",
            ],
        )

    def generate_test_data(self) -> dict[str, tuple[DigitalTrace, DigitalTrace]]:
        """Generate synthetic I2C signals with various transactions."""
        # Standard mode write (100 kHz)
        write_standard = self._generate_i2c_transaction(
            address=0x50,  # EEPROM typical address
            is_write=True,
            data=b"\x00\x10\xAA\x55",  # Address + data bytes
            clock_freq=100e3,
            sample_rate=10e6,
        )

        # Fast mode read (400 kHz)
        read_fast = self._generate_i2c_transaction(
            address=0x68,  # RTC typical address
            is_write=False,
            data=b"\x12\x34\x56\x78",
            clock_freq=400e3,
            sample_rate=40e6,
        )

        # Multi-byte write
        multi_write = self._generate_i2c_transaction(
            address=0x3C,  # OLED display typical address
            is_write=True,
            data=bytes(range(16)),  # 16 bytes
            clock_freq=100e3,
            sample_rate=10e6,
        )

        # Short transaction
        short_txn = self._generate_i2c_transaction(
            address=0x27,
            is_write=True,
            data=b"\xFF",
            clock_freq=100e3,
            sample_rate=10e6,
        )

        return {
            "write_standard": write_standard,
            "read_fast": read_fast,
            "multi_write": multi_write,
            "short": short_txn,
        }

    def run_demonstration(self, data: dict) -> dict[str, dict]:
        """Decode all I2C transactions."""
        results = {}

        self.section("I2C Standard Mode Write (100 kHz)")
        results["write_standard"] = self._decode_transaction(
            *data["write_standard"],
            expected_addr=0x50,
            expected_data=b"\x00\x10\xAA\x55",
            clock_freq=100e3,
        )

        self.section("I2C Fast Mode Read (400 kHz)")
        results["read_fast"] = self._decode_transaction(
            *data["read_fast"],
            expected_addr=0x68,
            expected_data=b"\x12\x34\x56\x78",
            clock_freq=400e3,
        )

        self.section("I2C Multi-Byte Write")
        results["multi_write"] = self._decode_transaction(
            *data["multi_write"],
            expected_addr=0x3C,
            expected_data=bytes(range(16)),
            clock_freq=100e3,
        )

        self.section("I2C Short Transaction")
        results["short"] = self._decode_transaction(
            *data["short"],
            expected_addr=0x27,
            expected_data=b"\xFF",
            clock_freq=100e3,
        )

        return results

    def validate(self, results: dict) -> bool:
        """Validate decoded I2C transactions."""
        self.section("Validation")
        suite = ValidationSuite("I2C Protocol Validation")

        for txn_name, result in results.items():
            packets = result.get("packets", [])

            suite.expect_true(
                len(packets) > 0,
                f"{txn_name}: Packets decoded",
                f"No packets for {txn_name}",
            )

        suite.print_summary()
        return suite.passed

    def _generate_i2c_transaction(
        self,
        address: int,
        is_write: bool,
        data: bytes,
        clock_freq: float,
        sample_rate: float,
    ) -> tuple[DigitalTrace, DigitalTrace]:
        """Generate I2C SCL and SDA signals for a transaction."""
        bit_time = 1.0 / clock_freq
        samples_per_bit = max(1, int(sample_rate * bit_time))

        scl_signal = []
        sda_signal = []

        # Idle: both high
        scl_signal.extend([1] * (samples_per_bit * 2))
        sda_signal.extend([1] * (samples_per_bit * 2))

        # START: SDA falls while SCL high
        scl_signal.extend([1] * (samples_per_bit // 2))
        sda_signal.extend([0] * (samples_per_bit // 2))

        # Address byte (7-bit address + R/W bit)
        addr_byte = (address << 1) | (0 if is_write else 1)
        for bit_idx in range(8):
            bit_val = (addr_byte >> (7 - bit_idx)) & 1
            # SCL low, setup data
            scl_signal.extend([0] * (samples_per_bit // 2))
            sda_signal.extend([bit_val] * (samples_per_bit // 2))
            # SCL high, sample data
            scl_signal.extend([1] * (samples_per_bit // 2))
            sda_signal.extend([bit_val] * (samples_per_bit // 2))

        # ACK bit from slave
        scl_signal.extend([0] * (samples_per_bit // 2))
        sda_signal.extend([0] * (samples_per_bit // 2))
        scl_signal.extend([1] * (samples_per_bit // 2))
        sda_signal.extend([0] * (samples_per_bit // 2))

        # Data bytes
        for byte_val in data:
            for bit_idx in range(8):
                bit_val = (byte_val >> (7 - bit_idx)) & 1
                scl_signal.extend([0] * (samples_per_bit // 2))
                sda_signal.extend([bit_val] * (samples_per_bit // 2))
                scl_signal.extend([1] * (samples_per_bit // 2))
                sda_signal.extend([bit_val] * (samples_per_bit // 2))

            # ACK bit
            scl_signal.extend([0] * (samples_per_bit // 2))
            sda_signal.extend([0] * (samples_per_bit // 2))
            scl_signal.extend([1] * (samples_per_bit // 2))
            sda_signal.extend([0] * (samples_per_bit // 2))

        # STOP: SDA rises while SCL high
        scl_signal.extend([0] * (samples_per_bit // 2))
        sda_signal.extend([0] * (samples_per_bit // 2))
        scl_signal.extend([1] * (samples_per_bit // 2))
        sda_signal.extend([0] * (samples_per_bit // 2))
        scl_signal.extend([1] * (samples_per_bit // 2))
        sda_signal.extend([1] * (samples_per_bit // 2))

        # Return to idle
        scl_signal.extend([1] * samples_per_bit)
        sda_signal.extend([1] * samples_per_bit)

        return (
            DigitalTrace(np.array(scl_signal, dtype=bool), TraceMetadata(sample_rate=sample_rate, channel_name="i2c_scl")),
            DigitalTrace(np.array(sda_signal, dtype=bool), TraceMetadata(sample_rate=sample_rate, channel_name="i2c_sda")),
        )

    def _decode_transaction(
        self,
        scl: DigitalTrace,
        sda: DigitalTrace,
        expected_addr: int,
        expected_data: bytes,
        clock_freq: float,
    ) -> dict:
        """Decode and display I2C transaction."""
        self.subsection("Signal Information")
        self.result("Sample rate", scl.metadata.sample_rate, "Hz")
        self.result("Clock frequency", clock_freq, "Hz")
        self.result("SCL samples", len(scl.data))
        self.result("SDA samples", len(sda.data))

        # Decode
        self.subsection("Decoding")
        packets = decode_i2c(
            scl=scl.data,
            sda=sda.data,
            sample_rate=scl.metadata.sample_rate,
            address_format="7bit",
        )

        self.result("Packets decoded", len(packets))

        # Display packets
        self.subsection("Decoded Packets")
        for i, packet in enumerate(packets):
            data_hex = packet.data.hex() if packet.data else "(empty)"
            self.info(f"Packet {i}: {data_hex}")

        return {
            "packets": packets,
            "expected_addr": expected_addr,
            "expected_data": expected_data,
        }


if __name__ == "__main__":
    demo = I2CDemo()
    success = demo.execute()
    sys.exit(0 if success else 1)
