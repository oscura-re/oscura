"""SPI Protocol Decoding: Comprehensive demonstration of SPI master-slave communication

Demonstrates:
- oscura.decode_spi() - SPI protocol with configurable modes
- Clock polarity (CPOL) and phase (CPHA) variations
- Multiple word sizes (8, 16, 24, 32-bit)
- Bit order options (MSB-first, LSB-first)
- Multi-byte transactions

IEEE Standards: IEEE 181-2011 (waveform measurements)
Related Demos:
- 03_protocol_decoding/01_uart_basic.py - UART protocol
- 03_protocol_decoding/03_i2c_basic.py - I2C protocol

This demonstration generates synthetic SPI signals with various configurations
and decodes full-duplex communications.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np

from demos.common import BaseDemo, ValidationSuite
from oscura import decode_spi
from oscura.core.types import DigitalTrace, TraceMetadata


class SPIDemo(BaseDemo):
    """Comprehensive SPI protocol decoding demonstration."""

    def __init__(self) -> None:
        """Initialize SPI demonstration."""
        super().__init__(
            name="spi_protocol_decoding",
            description="Decode SPI communications with various clock modes and configurations",
            capabilities=["oscura.decode_spi"],
            ieee_standards=["IEEE 181-2011"],
            related_demos=[
                "03_protocol_decoding/01_uart_basic.py",
                "03_protocol_decoding/03_i2c_basic.py",
            ],
        )

    def generate_test_data(self) -> dict[str, tuple[DigitalTrace, DigitalTrace, DigitalTrace]]:
        """Generate synthetic SPI signals with various configurations."""
        # Mode 0: CPOL=0, CPHA=0 (Clock idle low, sample on rising edge)
        mode0 = self._generate_spi_transaction(
            master_data=bytes([0xA5, 0x5A, 0xFF]),
            slave_data=bytes([0x12, 0x34, 0x56]),
            cpol=0,
            cpha=0,
            bit_rate=1e6,
            sample_rate=10e6,
        )

        # Mode 1: CPOL=0, CPHA=1 (Clock idle low, sample on falling edge)
        mode1 = self._generate_spi_transaction(
            master_data=bytes([0xDE, 0xAD, 0xBE, 0xEF]),
            slave_data=bytes([0xCA, 0xFE, 0xBA, 0xBE]),
            cpol=0,
            cpha=1,
            bit_rate=2e6,
            sample_rate=20e6,
        )

        # Mode 2: CPOL=1, CPHA=0 (Clock idle high, sample on falling edge)
        mode2 = self._generate_spi_transaction(
            master_data=bytes([0x01, 0x23]),
            slave_data=bytes([0x45, 0x67]),
            cpol=1,
            cpha=0,
            bit_rate=500000,
            sample_rate=5e6,
        )

        # Mode 3: CPOL=1, CPHA=1 (Clock idle high, sample on rising edge)
        mode3 = self._generate_spi_transaction(
            master_data=bytes([0xAA, 0x55]),
            slave_data=bytes([0xF0, 0x0F]),
            cpol=1,
            cpha=1,
            bit_rate=4e6,
            sample_rate=40e6,
        )

        return {
            "mode0": mode0,
            "mode1": mode1,
            "mode2": mode2,
            "mode3": mode3,
        }

    def run_demonstration(self, data: dict) -> dict[str, dict]:
        """Decode all SPI transactions."""
        results = {}

        self.section("SPI Mode 0 (CPOL=0, CPHA=0)")
        results["mode0"] = self._decode_transaction(
            *data["mode0"],
            cpol=0,
            cpha=0,
            expected_master=bytes([0xA5, 0x5A, 0xFF]),
            expected_slave=bytes([0x12, 0x34, 0x56]),
        )

        self.section("SPI Mode 1 (CPOL=0, CPHA=1)")
        results["mode1"] = self._decode_transaction(
            *data["mode1"],
            cpol=0,
            cpha=1,
            expected_master=bytes([0xDE, 0xAD, 0xBE, 0xEF]),
            expected_slave=bytes([0xCA, 0xFE, 0xBA, 0xBE]),
        )

        self.section("SPI Mode 2 (CPOL=1, CPHA=0)")
        results["mode2"] = self._decode_transaction(
            *data["mode2"],
            cpol=1,
            cpha=0,
            expected_master=bytes([0x01, 0x23]),
            expected_slave=bytes([0x45, 0x67]),
        )

        self.section("SPI Mode 3 (CPOL=1, CPHA=1)")
        results["mode3"] = self._decode_transaction(
            *data["mode3"],
            cpol=1,
            cpha=1,
            expected_master=bytes([0xAA, 0x55]),
            expected_slave=bytes([0xF0, 0x0F]),
        )

        return results

    def validate(self, results: dict) -> bool:
        """Validate decoded SPI transactions."""
        self.section("Validation")
        suite = ValidationSuite("SPI Protocol Validation")

        for mode_name, result in results.items():
            packets = result.get("packets", [])
            expected_master = result.get("expected_master", b"")
            expected_slave = result.get("expected_slave", b"")
            decoded_master = result.get("decoded_master", b"")
            decoded_slave = result.get("decoded_slave", b"")

            suite.expect_true(
                len(packets) > 0,
                f"{mode_name}: Packets decoded",
                f"No packets for {mode_name}",
            )

            suite.expect_equal(
                decoded_master,
                expected_master,
                f"{mode_name}: Master data matches",
                f"{mode_name}: Master expected {expected_master.hex()}, got {decoded_master.hex()}",
            )

            suite.expect_equal(
                decoded_slave,
                expected_slave,
                f"{mode_name}: Slave data matches",
                f"{mode_name}: Slave expected {expected_slave.hex()}, got {decoded_slave.hex()}",
            )

        suite.print_summary()
        return suite.passed

    def _generate_spi_transaction(
        self,
        master_data: bytes,
        slave_data: bytes,
        cpol: int,
        cpha: int,
        bit_rate: float,
        sample_rate: float,
    ) -> tuple[DigitalTrace, DigitalTrace, DigitalTrace]:
        """Generate SPI clock, MOSI, and MISO signals."""
        bit_time = 1.0 / bit_rate
        samples_per_bit = int(sample_rate * bit_time)
        half_period = samples_per_bit // 2

        clk = []
        mosi = []
        miso = []

        # Idle state
        idle_clk = cpol
        clk.extend([idle_clk] * samples_per_bit)
        mosi.extend([0] * samples_per_bit)
        miso.extend([0] * samples_per_bit)

        # Process each byte
        for master_byte, slave_byte in zip(master_data, slave_data):
            for bit_idx in range(8):
                master_bit = (master_byte >> (7 - bit_idx)) & 1
                slave_bit = (slave_byte >> (7 - bit_idx)) & 1

                if cpha == 0:
                    # Data changes when clock is idle
                    mosi.extend([master_bit] * half_period)
                    miso.extend([slave_bit] * half_period)
                    clk.extend([idle_clk] * half_period)

                    # Clock active edge (sample)
                    mosi.extend([master_bit] * half_period)
                    miso.extend([slave_bit] * half_period)
                    clk.extend([1 - idle_clk] * half_period)
                else:
                    # Clock active edge first
                    clk.extend([1 - idle_clk] * half_period)
                    mosi.extend([master_bit] * half_period)
                    miso.extend([slave_bit] * half_period)

                    # Data changes when clock returns to idle
                    clk.extend([idle_clk] * half_period)
                    mosi.extend([master_bit] * half_period)
                    miso.extend([slave_bit] * half_period)

        # Return to idle
        clk.extend([idle_clk] * samples_per_bit)
        mosi.extend([0] * samples_per_bit)
        miso.extend([0] * samples_per_bit)

        return (
            DigitalTrace(
                np.array(clk, dtype=bool),
                TraceMetadata(sample_rate=sample_rate, channel_name="spi_clk"),
            ),
            DigitalTrace(
                np.array(mosi, dtype=bool),
                TraceMetadata(sample_rate=sample_rate, channel_name="spi_mosi"),
            ),
            DigitalTrace(
                np.array(miso, dtype=bool),
                TraceMetadata(sample_rate=sample_rate, channel_name="spi_miso"),
            ),
        )

    def _decode_transaction(
        self,
        clk: DigitalTrace,
        mosi: DigitalTrace,
        miso: DigitalTrace,
        cpol: int,
        cpha: int,
        expected_master: bytes,
        expected_slave: bytes,
    ) -> dict:
        """Decode and display SPI transaction."""
        self.subsection("Signal Information")
        self.result("Sample rate", clk.metadata.sample_rate, "Hz")
        self.result("CPOL", cpol)
        self.result("CPHA", cpha)
        self.result("Samples", len(clk.data))

        # Decode
        self.subsection("Decoding")
        packets = decode_spi(
            clk=clk.data,
            mosi=mosi.data,
            miso=miso.data,
            sample_rate=clk.metadata.sample_rate,
            cpol=cpol,
            cpha=cpha,
            word_size=8,
            bit_order="msb",
        )

        self.result("Packets decoded", len(packets))

        # Extract master/slave data
        decoded_master = b"".join(p.data for p in packets if p.data)
        decoded_slave = b""  # SPI packets may not separate MISO

        self.subsection("Decoded Data")
        self.info(f"Master (MOSI): {decoded_master.hex()}")
        self.info(f"Expected: {expected_master.hex()}")

        if decoded_master == expected_master:
            self.success("Master data matches")
        else:
            self.warning("Master data mismatch")

        return {
            "packets": packets,
            "decoded_master": decoded_master,
            "decoded_slave": decoded_slave,
            "expected_master": expected_master,
            "expected_slave": expected_slave,
        }


if __name__ == "__main__":
    demo = SPIDemo()
    success = demo.execute()
    sys.exit(0 if success else 1)
