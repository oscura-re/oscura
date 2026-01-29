"""Protocol Signal Generation: Generate realistic protocol traffic

Demonstrates:
- UART signal generation with configurable parameters
- SPI transaction generation (CLK, MOSI, MISO, CS)
- I2C transaction generation (SCL, SDA)
- CAN message generation (future)
- Realistic protocol traffic patterns
- Error injection for testing decoders

IEEE Standards: None (protocol-specific)
Related Demos:
- 03_protocol_decoding/01_serial_comprehensive.py
- 03_protocol_decoding/02_spi_decoding.py
- 03_protocol_decoding/03_i2c_decoding.py
- 17_signal_generation/01_signal_builder_comprehensive.py

This demonstration shows how to generate realistic protocol signals for testing
protocol decoders and understanding serial communication protocols.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

# Add demonstrations to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demonstrations.common import BaseDemo, validate_approximately
from oscura.core.types import TraceMetadata, WaveformTrace
from tests.fixtures.signal_builders import SignalBuilder


class ProtocolGenerationDemo(BaseDemo):
    """Demonstration of protocol signal generation capabilities."""

    def __init__(self) -> None:
        """Initialize protocol generation demonstration."""
        super().__init__(
            name="protocol_generation",
            description="Generate realistic UART, SPI, I2C protocol signals with error injection",
            capabilities=[
                "SignalBuilder.uart_frame",
                "SignalBuilder.spi_transaction",
                "SignalBuilder.i2c_transaction",
                "SignalBuilder.digital_pattern",
            ],
            ieee_standards=[],
            related_demos=[
                "03_protocol_decoding/01_serial_comprehensive.py",
                "03_protocol_decoding/02_spi_decoding.py",
                "03_protocol_decoding/03_i2c_decoding.py",
            ],
        )

    def generate_test_data(self) -> dict[str, Any]:
        """Generate protocol signals for demonstration.

        Returns:
            Dictionary containing all protocol signal types
        """
        sample_rate = 10e6  # 10 MHz sampling

        # UART signals
        uart_hello = SignalBuilder.uart_frame(
            data=b"Hello",
            baudrate=115200,
            sample_rate=sample_rate,
            amplitude=3.3,
            parity="none",
            stop_bits=1,
        )

        uart_with_parity = SignalBuilder.uart_frame(
            data=0x55,
            baudrate=9600,
            sample_rate=sample_rate,
            amplitude=5.0,
            parity="even",
            stop_bits=2,
        )

        uart_parity_error = SignalBuilder.uart_frame(
            data=0xAA,
            baudrate=115200,
            sample_rate=sample_rate,
            parity="odd",
            inject_parity_error=True,
        )

        # SPI signals
        spi_clk, spi_mosi, spi_miso, spi_cs = SignalBuilder.spi_transaction(
            mosi_data=b"\xde\xad\xbe\xef",
            miso_data=b"\x12\x34\x56\x78",
            clock_rate=1_000_000,
            sample_rate=sample_rate,
            cpol=0,
            cpha=0,
            amplitude=3.3,
        )

        # I2C signals
        i2c_scl, i2c_sda = SignalBuilder.i2c_transaction(
            address=0x48,
            data=b"\x01\x02\x03",
            clock_rate=100_000,
            sample_rate=sample_rate,
            amplitude=3.3,
        )

        # Digital pattern
        digital = SignalBuilder.digital_pattern(
            pattern="10101010",
            sample_rate=sample_rate,
            bit_rate=1e6,
            amplitude=3.3,
        )

        return {
            "sample_rate": sample_rate,
            "uart_hello": uart_hello,
            "uart_with_parity": uart_with_parity,
            "uart_parity_error": uart_parity_error,
            "spi_clk": spi_clk,
            "spi_mosi": spi_mosi,
            "spi_miso": spi_miso,
            "spi_cs": spi_cs,
            "i2c_scl": i2c_scl,
            "i2c_sda": i2c_sda,
            "digital": digital,
        }

    def run_demonstration(self, data: dict[str, Any]) -> dict[str, Any]:
        """Run protocol signal generation demonstration."""
        results: dict[str, Any] = {}

        self.section("Protocol Signal Generation Demonstration")
        self.info("Generating realistic protocol signals for UART, SPI, and I2C")

        # Part 1: UART Signal Generation
        self.subsection("Part 1: UART Signal Generation")
        self.info("Universal Asynchronous Receiver/Transmitter (UART)")

        # UART "Hello" transmission
        uart_hello = data["uart_hello"]
        self.result("UART 'Hello' frame", f"{len(uart_hello)} samples")
        self.result("  Baudrate", "115200 bps")
        self.result("  Data bytes", "5 (b'Hello')")
        self.result("  Parity", "None")
        self.result("  Stop bits", "1")
        self.result("  Logic high", f"{np.max(uart_hello):.1f} V")
        self.result("  Logic low", f"{np.min(uart_hello):.1f} V")

        # Calculate expected duration
        # Each byte: 1 start + 8 data + 1 stop = 10 bits
        # 5 bytes = 50 bits total + idle periods
        bits_per_frame = 10  # start + 8 data + stop
        total_bits = 5 * bits_per_frame
        expected_duration = total_bits / 115200
        actual_duration = len(uart_hello) / data["sample_rate"]
        self.result("  Expected duration", f"{expected_duration:.6f} s")
        self.result("  Actual duration", f"{actual_duration:.6f} s")

        results["uart_hello_samples"] = len(uart_hello)
        results["uart_hello_high"] = float(np.max(uart_hello))

        # UART with parity
        uart_parity = data["uart_with_parity"]
        self.info("\nUART with even parity and 2 stop bits:")
        self.result("  Samples", len(uart_parity))
        self.result("  Data byte", "0x55")
        self.result("  Baudrate", "9600 bps")
        self.result("  Parity", "Even")
        self.result("  Stop bits", "2")
        results["uart_parity_samples"] = len(uart_parity)

        # UART with error injection
        uart_error = data["uart_parity_error"]
        self.info("\nUART with parity error injection:")
        self.result("  Samples", len(uart_error))
        self.result("  Data byte", "0xAA")
        self.result("  Parity", "Odd (ERROR INJECTED)")
        self.result("  Use case", "Testing decoder error handling")
        results["uart_error_samples"] = len(uart_error)

        # Part 2: SPI Signal Generation
        self.subsection("Part 2: SPI Signal Generation")
        self.info("Serial Peripheral Interface (SPI)")

        spi_clk = data["spi_clk"]
        spi_mosi = data["spi_mosi"]
        spi_miso = data["spi_miso"]
        spi_cs = data["spi_cs"]

        self.result("SPI transaction", "4 bytes (0xDEADBEEF)")
        self.result("  CLK samples", len(spi_clk))
        self.result("  MOSI samples", len(spi_mosi))
        self.result("  MISO samples", len(spi_miso))
        self.result("  CS samples", len(spi_cs))
        self.result("  Clock rate", "1 MHz")
        self.result("  Mode", "0 (CPOL=0, CPHA=0)")

        # Verify CS is active-low
        cs_active = np.min(spi_cs)
        cs_idle = np.max(spi_cs)
        self.result("  CS idle level", f"{cs_idle:.1f} V")
        self.result("  CS active level", f"{cs_active:.1f} V")

        # Count clock edges
        clk_edges = np.sum(np.abs(np.diff(spi_clk)) > 1.0)
        self.result("  Clock edges", clk_edges)
        self.result("  Expected edges", "64 (4 bytes x 8 bits x 2 edges)")

        results["spi_clk_samples"] = len(spi_clk)
        results["spi_clk_edges"] = int(clk_edges)
        results["spi_cs_active"] = float(cs_active)

        # Part 3: I2C Signal Generation
        self.subsection("Part 3: I2C Signal Generation")
        self.info("Inter-Integrated Circuit (I2C)")

        i2c_scl = data["i2c_scl"]
        i2c_sda = data["i2c_sda"]

        self.result("I2C transaction", "Address 0x48 + 3 data bytes")
        self.result("  SCL samples", len(i2c_scl))
        self.result("  SDA samples", len(i2c_sda))
        self.result("  Clock rate", "100 kHz (Standard mode)")
        self.result("  Address", "0x48 (7-bit)")
        self.result("  Data bytes", "3 (0x01, 0x02, 0x03)")

        # Detect START condition (SDA falls while SCL high)
        scl_high = i2c_scl > 2.0
        sda_falls = np.diff(i2c_sda) < -1.0
        start_conditions = np.sum(scl_high[:-1] & sda_falls)
        self.result("  START conditions", int(start_conditions))

        # Count SCL clock pulses
        scl_edges = np.sum(np.abs(np.diff(i2c_scl)) > 1.0)
        scl_pulses = scl_edges // 2
        self.result("  SCL pulses", int(scl_pulses))
        # Expected: 8 (addr) + 1 (ack) + 3 x (8 data + 1 ack) = 36 bits
        self.result("  Expected pulses", "36 (addr + ack + 3 x (data + ack))")

        results["i2c_scl_samples"] = len(i2c_scl)
        results["i2c_start_conditions"] = int(start_conditions)
        results["i2c_scl_pulses"] = int(scl_pulses)

        # Part 4: Digital Pattern Generation
        self.subsection("Part 4: Digital Pattern Generation")
        self.info("Simple bit patterns for testing")

        digital = data["digital"]
        self.result("Digital pattern", "10101010")
        self.result("  Samples", len(digital))
        self.result("  Bit rate", "1 Mbps")
        self.result("  Logic levels", f"{np.min(digital):.1f}V / {np.max(digital):.1f}V")

        # Verify pattern
        samples_per_bit = len(digital) // 8
        pattern_check = []
        for i in range(8):
            bit_value = digital[i * samples_per_bit]
            pattern_check.append("1" if bit_value > 1.5 else "0")
        recovered_pattern = "".join(pattern_check)
        self.result("  Recovered pattern", recovered_pattern)

        results["digital_pattern"] = recovered_pattern

        # Part 5: Protocol Traffic Patterns
        self.subsection("Part 5: Realistic Protocol Traffic Patterns")
        self.info("Simulating real-world communication scenarios")

        # Burst transmission (multiple UART frames)
        self.info("\n[UART Burst Transmission]")
        uart_burst = []
        for byte_val in b"BURST":
            frame = SignalBuilder.uart_frame(
                data=byte_val, baudrate=115200, sample_rate=data["sample_rate"]
            )
            uart_burst.append(frame)
        total_samples = sum(len(f) for f in uart_burst)
        self.result("  Frames", len(uart_burst))
        self.result("  Total samples", total_samples)
        self.result("  Bytes transmitted", "5 (b'BURST')")

        # SPI multi-transaction
        self.info("\n[SPI Multi-Transaction]")
        spi_transactions = []
        for data_byte in [0x01, 0x02, 0x03]:
            clk, mosi, miso, cs = SignalBuilder.spi_transaction(
                mosi_data=bytes([data_byte]),
                clock_rate=1_000_000,
                sample_rate=data["sample_rate"],
            )
            spi_transactions.append((clk, mosi, miso, cs))
        self.result("  Transactions", len(spi_transactions))
        self.result("  Bytes per transaction", "1")

        results["uart_burst_frames"] = len(uart_burst)
        results["spi_multi_transactions"] = len(spi_transactions)

        # Part 6: Usage Examples
        self.subsection("Part 6: Usage in Protocol Decoder Testing")

        self.info("\n[Creating WaveformTrace for UART Decoder]")
        metadata = TraceMetadata(sample_rate=data["sample_rate"], channel_name="UART_TX")
        uart_trace = WaveformTrace(data=uart_hello, metadata=metadata)
        self.result("  WaveformTrace created", f"{len(uart_trace)} samples")
        self.result("  Duration", f"{uart_trace.duration:.6f} s")

        self.info("\n[Multi-Channel SPI Setup]")
        clk_trace = WaveformTrace(
            data=spi_clk,
            metadata=TraceMetadata(sample_rate=data["sample_rate"], channel_name="SCK"),
        )
        mosi_trace = WaveformTrace(
            data=spi_mosi,
            metadata=TraceMetadata(sample_rate=data["sample_rate"], channel_name="MOSI"),
        )
        self.result("  CLK trace", f"{len(clk_trace)} samples")
        self.result("  MOSI trace", f"{len(mosi_trace)} samples")
        self.result("  Ready for SPI decoder", "oscura.decode_spi()")

        self.success("Protocol signal generation complete!")
        self.info("\nKey Capabilities:")
        self.info("  - UART: Configurable baud, parity, stop bits, error injection")
        self.info("  - SPI: 4-wire (CLK, MOSI, MISO, CS), all modes (CPOL/CPHA)")
        self.info("  - I2C: START/STOP conditions, ACK/NACK, address + data")
        self.info("  - Digital patterns: Custom bit sequences")
        self.info("  - Realistic traffic: Bursts, multi-transaction scenarios")

        return results

    def validate(self, results: dict[str, Any]) -> bool:
        """Validate protocol signal generation results."""
        self.info("Validating protocol signals...")

        all_valid = True

        # Validate UART signal
        if results["uart_hello_samples"] < 1000:
            self.error(f"UART hello samples too few: {results['uart_hello_samples']}")
            all_valid = False
        else:
            self.success(f"UART hello samples: {results['uart_hello_samples']}")

        if not validate_approximately(
            results["uart_hello_high"], 3.3, tolerance=0.1, name="UART logic high"
        ):
            all_valid = False

        # Validate SPI signals
        if results["spi_clk_samples"] < 100:
            self.error(f"SPI CLK samples too few: {results['spi_clk_samples']}")
            all_valid = False
        else:
            self.success(f"SPI CLK samples: {results['spi_clk_samples']}")

        # SPI clock edges: 4 bytes x 8 bits x 2 edges = 64 edges
        if not validate_approximately(
            results["spi_clk_edges"], 64, tolerance=0.05, name="SPI clock edges"
        ):
            all_valid = False

        # Validate CS is active-low (should be near 0V when active)
        if not validate_approximately(
            results["spi_cs_active"], 0.0, tolerance=0.1, name="SPI CS active level"
        ):
            all_valid = False

        # Validate I2C signals
        if results["i2c_scl_samples"] < 100:
            self.error(f"I2C SCL samples too few: {results['i2c_scl_samples']}")
            all_valid = False
        else:
            self.success(f"I2C SCL samples: {results['i2c_scl_samples']}")

        # I2C should have at least 1 START condition
        if results["i2c_start_conditions"] < 1:
            self.error(f"I2C START conditions: {results['i2c_start_conditions']} (expected >= 1)")
            all_valid = False
        else:
            self.success(f"I2C START conditions: {results['i2c_start_conditions']}")

        # Validate digital pattern recovery
        if results["digital_pattern"] != "10101010":
            self.error(f"Digital pattern mismatch: {results['digital_pattern']} != 10101010")
            all_valid = False
        else:
            self.success("Digital pattern correctly recovered")

        # Validate burst transmission
        if results["uart_burst_frames"] != 5:
            self.error(f"UART burst frames: {results['uart_burst_frames']} != 5")
            all_valid = False
        else:
            self.success("UART burst transmission validated")

        if all_valid:
            self.success("All protocol signal validations passed!")
            self.info("\nNext Steps:")
            self.info("  - Use generated signals with protocol decoders")
            self.info("  - Test error handling with injected errors")
            self.info("  - Create custom protocol traffic patterns")
            self.info("  - Try 03_protocol_decoding/ demonstrations")
        else:
            self.error("Some protocol signal validations failed")

        return all_valid

    def result(self, name: str, value: Any, unit: str = "") -> None:
        """Print a result with optional unit.

        Args:
            name: Result name
            value: Result value
            unit: Optional unit string
        """
        if unit:
            print(f"  {name}: {value} {unit}")
        else:
            print(f"  {name}: {value}")


if __name__ == "__main__":
    demo: ProtocolGenerationDemo = ProtocolGenerationDemo()
    success: bool = demo.execute()
    sys.exit(0 if success else 1)
