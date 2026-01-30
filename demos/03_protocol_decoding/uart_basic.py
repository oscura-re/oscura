"""UART Protocol Decoding: Comprehensive demonstration of UART serial communication

Demonstrates:
- oscura.decode_uart() - UART/serial communication with configurable parameters
- Multiple baud rates (9600, 115200, 230400)
- Different data bit configurations (7, 8)
- Parity options (none, even, odd)
- Stop bit variations (1, 2)
- Error handling and validation

IEEE Standards: IEEE 181-2011 (waveform measurements)
Related Demos:
- 03_protocol_decoding/02_spi_basic.py - SPI protocol decoding
- 03_protocol_decoding/03_i2c_basic.py - I2C protocol decoding
- 02_basic_analysis/01_waveform_measurements.py - Signal measurements

This demonstration generates synthetic UART signals with various configurations
and decodes them to validate protocol compliance.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add demos to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np

from demos.common import BaseDemo, ValidationSuite
from oscura import decode_uart
from oscura.core.types import DigitalTrace, TraceMetadata


class UARTDemo(BaseDemo):
    """Comprehensive UART protocol decoding demonstration."""

    def __init__(self) -> None:
        """Initialize UART demonstration."""
        super().__init__(
            name="uart_protocol_decoding",
            description="Decode UART serial communications with various configurations",
            capabilities=[
                "oscura.decode_uart",
            ],
            ieee_standards=["IEEE 181-2011"],
            related_demos=[
                "03_protocol_decoding/02_spi_basic.py",
                "03_protocol_decoding/03_i2c_basic.py",
                "02_basic_analysis/01_waveform_measurements.py",
            ],
        )

    def generate_test_data(self) -> dict[str, DigitalTrace]:
        """Generate synthetic UART signals with various configurations.

        Returns:
            Dictionary with UART test signals
        """
        # Standard configuration: 9600 baud, 8N1
        uart_standard = self._generate_uart_signal(
            message=b"Hello UART!",
            baudrate=9600,
            data_bits=8,
            parity="none",
            stop_bits=1,
            sample_rate=1e6,
        )

        # High-speed: 115200 baud, 8N1
        uart_high_speed = self._generate_uart_signal(
            message=b"FastData",
            baudrate=115200,
            data_bits=8,
            parity="none",
            stop_bits=1,
            sample_rate=10e6,
        )

        # With even parity: 9600 baud, 8E1
        uart_even_parity = self._generate_uart_signal(
            message=b"Parity",
            baudrate=9600,
            data_bits=8,
            parity="even",
            stop_bits=1,
            sample_rate=1e6,
        )

        # 7-bit ASCII with 2 stop bits: 9600 baud, 7N2
        uart_7bit = self._generate_uart_signal(
            message=b"ASCII",
            baudrate=9600,
            data_bits=7,
            parity="none",
            stop_bits=2,
            sample_rate=1e6,
        )

        return {
            "standard": uart_standard,
            "high_speed": uart_high_speed,
            "even_parity": uart_even_parity,
            "7bit_2stop": uart_7bit,
        }

    def run_demonstration(self, data: dict[str, DigitalTrace]) -> dict[str, dict[str, object]]:
        """Decode all UART signals and display results.

        Args:
            data: Generated UART signals

        Returns:
            Dictionary of decoded results
        """
        results = {}

        # Standard UART (8N1)
        self.section("Standard UART (9600 8N1)")
        results["standard"] = self._decode_and_display(
            data["standard"],
            baudrate=9600,
            data_bits=8,
            parity="none",
            stop_bits=1,
            expected=b"Hello UART!",
        )

        # High-speed UART
        self.section("High-Speed UART (115200 8N1)")
        results["high_speed"] = self._decode_and_display(
            data["high_speed"],
            baudrate=115200,
            data_bits=8,
            parity="none",
            stop_bits=1,
            expected=b"FastData",
        )

        # Even parity UART
        self.section("UART with Even Parity (9600 8E1)")
        results["even_parity"] = self._decode_and_display(
            data["even_parity"],
            baudrate=9600,
            data_bits=8,
            parity="even",
            stop_bits=1,
            expected=b"Parity",
        )

        # 7-bit UART
        self.section("7-bit UART with 2 Stop Bits (9600 7N2)")
        results["7bit_2stop"] = self._decode_and_display(
            data["7bit_2stop"],
            baudrate=9600,
            data_bits=7,
            parity="none",
            stop_bits=2,
            expected=b"ASCII",
        )

        return results

    def validate(self, results: dict[str, dict[str, object]]) -> bool:
        """Validate decoded UART packets using ValidationSuite.

        Args:
            results: Decoded UART results

        Returns:
            True if all validations pass
        """
        self.section("Validation")

        suite = ValidationSuite("UART Protocol Validation")

        # Validate each configuration
        for config_name, result in results.items():
            packets = result.get("packets", [])
            expected = result.get("expected", b"")
            decoded = result.get("decoded", b"")

            suite.expect_true(
                len(packets) > 0,
                f"{config_name}: Packets decoded",
                f"No packets decoded for {config_name}",
            )

            suite.expect_equal(
                decoded,
                expected,
                f"{config_name}: Data matches",
                f"{config_name}: Expected {expected!r}, got {decoded!r}",
            )

        # Display results
        suite.print_summary()
        return suite.passed

    def _generate_uart_signal(
        self,
        message: bytes,
        baudrate: int,
        data_bits: int,
        parity: str,
        stop_bits: int,
        sample_rate: float,
    ) -> DigitalTrace:
        """Generate synthetic UART signal.

        Args:
            message: Message bytes to send
            baudrate: UART baud rate
            data_bits: Data bits per frame (7 or 8)
            parity: Parity mode ("none", "even", "odd")
            stop_bits: Number of stop bits (1 or 2)
            sample_rate: Sample rate in Hz

        Returns:
            DigitalTrace with UART signal
        """
        bit_time = 1.0 / baudrate
        samples_per_bit = int(sample_rate * bit_time)

        signal = []

        for byte_val in message:
            # Idle (mark) - high
            signal.extend([1] * (samples_per_bit * 2))

            # Start bit - low
            signal.extend([0] * samples_per_bit)

            # Data bits (LSB first)
            data_mask = (1 << data_bits) - 1
            data = byte_val & data_mask
            for i in range(data_bits):
                bit = (data >> i) & 1
                signal.extend([bit] * samples_per_bit)

            # Parity bit (if enabled)
            if parity != "none":
                ones_count = bin(data).count("1")
                if parity == "even":
                    parity_bit = ones_count % 2
                else:  # odd
                    parity_bit = (ones_count + 1) % 2
                signal.extend([parity_bit] * samples_per_bit)

            # Stop bits - high
            signal.extend([1] * (samples_per_bit * stop_bits))

        signal_array = np.array(signal, dtype=bool)

        metadata = TraceMetadata(
            sample_rate=sample_rate,
            channel_name="uart_tx",
        )

        return DigitalTrace(data=signal_array, metadata=metadata)

    def _decode_and_display(
        self,
        signal: DigitalTrace,
        baudrate: int,
        data_bits: int,
        parity: str,
        stop_bits: int,
        expected: bytes,
    ) -> dict[str, object]:
        """Decode UART signal and display results.

        Args:
            signal: UART signal to decode
            baudrate: Expected baud rate
            data_bits: Expected data bits
            parity: Expected parity
            stop_bits: Expected stop bits
            expected: Expected decoded message

        Returns:
            Results dictionary
        """
        self.subsection("Signal Information")
        self.result("Sample rate", signal.metadata.sample_rate, "Hz")
        self.result("Baudrate", baudrate, "bps")
        self.result("Configuration", f"{data_bits}{parity[0].upper()}{stop_bits}")
        self.result("Number of samples", len(signal.data))
        duration = len(signal.data) / signal.metadata.sample_rate
        self.result("Duration", duration, "s")

        # Decode UART
        self.subsection("Decoding")
        packets = decode_uart(
            signal,
            sample_rate=signal.metadata.sample_rate,
            baudrate=baudrate,
            data_bits=data_bits,
            parity=parity,
            stop_bits=stop_bits,
        )

        self.result("Packets decoded", len(packets))

        # Display decoded data
        decoded_bytes = b"".join(p.data for p in packets if p.data)

        self.subsection("Decoded Data")
        if decoded_bytes:
            display_str = decoded_bytes.decode("ascii", errors="replace")
            self.info(f"Message: {display_str!r}")
            self.info(f"Hex: {decoded_bytes.hex()}")
            self.info(f"Length: {len(decoded_bytes)} bytes")
        else:
            self.warning("No data decoded")

        # Validation
        if decoded_bytes == expected:
            self.success(f"Decoded correctly: {expected!r}")
        elif decoded_bytes:
            self.warning(f"Data mismatch: got {decoded_bytes!r}, expected {expected!r}")
        else:
            self.error("No data decoded")

        return {
            "packets": packets,
            "decoded": decoded_bytes,
            "expected": expected,
            "packet_count": len(packets),
        }


if __name__ == "__main__":
    demo: UARTDemo = UARTDemo()
    success: bool = demo.execute()
    sys.exit(0 if success else 1)
