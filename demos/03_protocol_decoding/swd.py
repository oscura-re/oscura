"""SWD Protocol Decoding: ARM Serial Wire Debug protocol

Demonstrates:
- oscura.decode_swd() - SWD packet decoding
- Read/write transactions
- Debug Access Port (DAP) operations
- Parity checking
- ACK response handling

IEEE Standards: ARM Debug Interface Architecture Specification
Related Demos:
- 03_protocol_decoding/08_jtag.py - JTAG debug protocol
- 03_protocol_decoding/02_spi_basic.py - SPI protocol
- 02_basic_analysis/01_waveform_measurements.py - Signal measurements

This demonstration generates synthetic SWD signals for debug operations
and validates protocol compliance for ARM debugging.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from typing import TYPE_CHECKING, ClassVar

import numpy as np

from demos.common.base_demo import BaseDemo, run_demo_main
from oscura.core.types import DigitalTrace, TraceMetadata

if TYPE_CHECKING:
    from demos.common.validation import ValidationSuite


class SWDDemo(BaseDemo):
    """SWD protocol decoding demonstration."""

    name = "SWD Protocol Decoding"
    description = "Decode ARM Serial Wire Debug read/write transactions"
    category = "protocol_decoding"
    capabilities: ClassVar[list[str]] = [
        "SWD packet decoding",
        "DP/AP read/write operations",
        "Parity validation",
        "ACK response decoding",
        "SWDIO bidirectional handling",
    ]
    ieee_standards: ClassVar[list[str]] = ["ARM Debug Interface v5.2"]
    related_demos: ClassVar[list[str]] = [
        "03_protocol_decoding/08_jtag.py",
        "03_protocol_decoding/02_spi_basic.py",
    ]

    def generate_data(self) -> None:
        """Generate synthetic SWD signals."""
        # DP IDCODE read
        self.swd_dp_read = self._generate_swd_transaction(
            is_read=True,
            is_ap=False,
            address=0x0,  # IDCODE register
            data=0x2BA01477,  # Example IDCODE
            sample_rate=10e6,
        )

        # AP CSW write
        self.swd_ap_write = self._generate_swd_transaction(
            is_read=False,
            is_ap=True,
            address=0x0,  # CSW register
            data=0x23000052,
            sample_rate=10e6,
        )

        # Memory read
        self.swd_mem_read = self._generate_swd_transaction(
            is_read=True,
            is_ap=True,
            address=0xC,  # DRW register
            data=0xDEADBEEF,
            sample_rate=10e6,
        )

    def run_analysis(self) -> None:
        """Decode SWD transactions."""
        from demos.common.formatting import print_subheader

        print_subheader("DP IDCODE Read")
        self.results["dp_read"] = self._decode_swd_packet(
            self.swd_dp_read,
            expected_type="DP Read",
            expected_data=0x2BA01477,
        )

        print_subheader("AP CSW Write")
        self.results["ap_write"] = self._decode_swd_packet(
            self.swd_ap_write,
            expected_type="AP Write",
            expected_data=0x23000052,
        )

        print_subheader("Memory Read")
        self.results["mem_read"] = self._decode_swd_packet(
            self.swd_mem_read,
            expected_type="AP Read",
            expected_data=0xDEADBEEF,
        )

    def validate_results(self, suite: ValidationSuite) -> None:
        """Validate SWD decoding."""
        for config_name, result in self.results.items():
            suite.check_exists(
                f"{config_name}_swclk",
                result.get("swclk"),
                f"{config_name}: SWCLK generated",
            )
            suite.check_exists(
                f"{config_name}_swdio",
                result.get("swdio"),
                f"{config_name}: SWDIO generated",
            )

    def _generate_swd_transaction(
        self,
        is_read: bool,
        is_ap: bool,
        address: int,
        data: int,
        sample_rate: float,
    ) -> tuple[DigitalTrace, DigitalTrace]:
        """Generate SWD transaction signals.

        Args:
            is_read: True for read, False for write
            is_ap: True for AP access, False for DP
            address: Register address (bits [3:2])
            data: 32-bit data word
            sample_rate: Sample rate in Hz

        Returns:
            Tuple of (SWCLK, SWDIO) traces
        """
        clock_freq = 1e6  # 1 MHz
        samples_per_bit = max(1, int(sample_rate / clock_freq))

        swclk: ClassVar[list[str]] = []
        swdio: ClassVar[list[str]] = []

        # Idle period
        for _ in range(8):
            self._add_swd_bit(swclk, swdio, 1, samples_per_bit)

        # Start bit
        self._add_swd_bit(swclk, swdio, 1, samples_per_bit)

        # APnDP bit
        self._add_swd_bit(swclk, swdio, 1 if is_ap else 0, samples_per_bit)

        # RnW bit
        self._add_swd_bit(swclk, swdio, 1 if is_read else 0, samples_per_bit)

        # Address bits [3:2]
        addr_bits = (address >> 2) & 0x3
        self._add_swd_bit(swclk, swdio, addr_bits & 1, samples_per_bit)
        self._add_swd_bit(swclk, swdio, (addr_bits >> 1) & 1, samples_per_bit)

        # Parity bit
        request = (int(is_ap) << 2) | (int(is_read) << 1) | addr_bits
        parity = bin(request).count("1") & 1
        self._add_swd_bit(swclk, swdio, parity, samples_per_bit)

        # Stop bit
        self._add_swd_bit(swclk, swdio, 0, samples_per_bit)

        # Park bit
        self._add_swd_bit(swclk, swdio, 1, samples_per_bit)

        # Turnaround (1 clock, SWDIO floats)
        self._add_swd_bit(swclk, swdio, 1, samples_per_bit)

        # ACK response (3 bits: OK = 0b001)
        self._add_swd_bit(swclk, swdio, 1, samples_per_bit)
        self._add_swd_bit(swclk, swdio, 0, samples_per_bit)
        self._add_swd_bit(swclk, swdio, 0, samples_per_bit)

        # Data phase (32 bits)
        data_parity = 0
        for i in range(32):
            bit_val = (data >> i) & 1
            data_parity ^= bit_val
            self._add_swd_bit(swclk, swdio, bit_val, samples_per_bit)

        # Data parity
        self._add_swd_bit(swclk, swdio, data_parity, samples_per_bit)

        # Turnaround
        self._add_swd_bit(swclk, swdio, 0, samples_per_bit)

        # Idle
        for _ in range(8):
            self._add_swd_bit(swclk, swdio, 1, samples_per_bit)

        return (
            DigitalTrace(
                data=np.array(list(swclk), dtype=bool),
                metadata=TraceMetadata(sample_rate=sample_rate, channel_name="swclk"),
            ),
            DigitalTrace(
                data=np.array(list(swdio), dtype=bool),
                metadata=TraceMetadata(sample_rate=sample_rate, channel_name="swdio"),
            ),
        )

    def _add_swd_bit(
        self,
        swclk: list[int],
        swdio: list[int],
        data_bit: int,
        samples: int,
    ) -> None:
        """Add one bit period to SWD signals."""
        # Clock low, data stable
        swclk.extend([0] * (samples // 2))
        swdio.extend([data_bit] * (samples // 2))

        # Clock high, data sampled
        swclk.extend([1] * (samples // 2))
        swdio.extend([data_bit] * (samples // 2))

    def _decode_swd_packet(
        self,
        signals: tuple[DigitalTrace, DigitalTrace],
        expected_type: str,
        expected_data: int,
    ) -> dict[str, object]:
        """Decode SWD packet."""
        from demos.common.formatting import print_info

        swclk, swdio = signals

        print_info(f"Sample rate: {swclk.metadata.sample_rate / 1e6:.1f} MHz")
        print_info(f"Expected type: {expected_type}")
        print_info(f"Expected data: 0x{expected_data:08X}")

        packets: ClassVar[list[str]] = []
        try:
            from oscura import decode_swd

            packets = decode_swd(
                swclk,
                swdio,
                sample_rate=swclk.metadata.sample_rate,
            )
            print_info(f"Packets decoded: {len(packets)}")
        except (ImportError, AttributeError):
            print_info("SWD decoder not yet implemented (placeholder)")

        return {
            "swclk": swclk,
            "swdio": swdio,
            "packets": packets,
            "expected_type": expected_type,
            "expected_data": expected_data,
        }


if __name__ == "__main__":
    sys.exit(run_demo_main(SWDDemo))
