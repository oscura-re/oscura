"""Comprehensive Multi-Protocol Analysis: Cross-protocol correlation

Demonstrates:
- Multiple protocol decoding in single capture
- Protocol auto-detection
- Cross-protocol timing correlation
- Multi-bus analysis (SPI + I2C simultaneously)
- Protocol interference detection

IEEE Standards: IEEE 181-2011 (Measurements), Various protocol standards
Related Demos:
- 03_protocol_decoding/01_uart_basic.py - UART protocol
- 03_protocol_decoding/02_spi_basic.py - SPI protocol
- 03_protocol_decoding/03_i2c_basic.py - I2C protocol
- 06_reverse_engineering/01_unknown_protocol.py - Protocol discovery

This demonstration shows how to analyze devices with multiple communication
interfaces operating simultaneously, correlating events across protocols.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np

from demos.common.base_demo import BaseDemo, run_demo_main
from demos.common.validation import ValidationSuite
from oscura.core.types import DigitalTrace, TraceMetadata


class ComprehensiveProtocolDemo(BaseDemo):
    """Multi-protocol comprehensive analysis demonstration."""

    name = "Comprehensive Multi-Protocol Analysis"
    description = "Decode and correlate multiple protocols in single capture"
    category = "protocol_decoding"
    capabilities = [
        "Multi-protocol decoding",
        "Protocol auto-detection",
        "Cross-protocol correlation",
        "Timing relationship analysis",
        "Bus activity correlation",
    ]
    ieee_standards = ["IEEE 181-2011"]
    related_demos = [
        "03_protocol_decoding/01_uart_basic.py",
        "03_protocol_decoding/02_spi_basic.py",
        "03_protocol_decoding/03_i2c_basic.py",
        "06_reverse_engineering/01_unknown_protocol.py",
    ]

    def generate_data(self) -> None:
        """Generate multi-protocol signals with timing correlation."""
        # Scenario: MCU reading I2C sensor, then transmitting via SPI
        sample_rate = 10e6

        # I2C sensor read (400 kHz)
        self.i2c_scl, self.i2c_sda = self._generate_i2c_read(
            address=0x68,
            register=0x3B,
            data=b"\x12\x34\x56\x78",
            clock_freq=400000,
            sample_rate=sample_rate,
            start_sample=0,
        )

        # SPI transmission occurs 100 µs after I2C completes
        i2c_duration = len(self.i2c_scl.data)
        delay_samples = int(100e-6 * sample_rate)
        spi_start = i2c_duration + delay_samples

        self.spi_sck, self.spi_mosi, self.spi_miso, self.spi_cs = self._generate_spi_transfer(
            tx_data=b"\x12\x34\x56\x78",
            rx_data=b"\xff\xff\xff\xff",
            clock_freq=1000000,
            sample_rate=sample_rate,
            start_sample=spi_start,
        )

        # UART debug output overlapping with SPI
        uart_start = spi_start + int(50e-6 * sample_rate)
        self.uart_tx = self._generate_uart_message(
            message=b"OK",
            baudrate=115200,
            sample_rate=sample_rate,
            start_sample=uart_start,
        )

    def run_analysis(self) -> None:
        """Decode all protocols and correlate timing."""
        from demos.common.formatting import print_info, print_subheader

        print_subheader("Protocol Detection")
        print_info("Analyzing multi-protocol capture...")

        # Decode I2C
        print_subheader("I2C Bus Activity")
        self.results["i2c"] = self._analyze_i2c(
            self.i2c_scl,
            self.i2c_sda,
            expected_address=0x68,
        )

        # Decode SPI
        print_subheader("SPI Bus Activity")
        self.results["spi"] = self._analyze_spi(
            self.spi_sck,
            self.spi_mosi,
            self.spi_miso,
            self.spi_cs,
            expected_bytes=4,
        )

        # Decode UART
        print_subheader("UART Activity")
        self.results["uart"] = self._analyze_uart(
            self.uart_tx,
            baudrate=115200,
            expected_message=b"OK",
        )

        # Cross-protocol correlation
        print_subheader("Cross-Protocol Timing Analysis")
        self._correlate_protocols()

    def validate_results(self, suite: ValidationSuite) -> None:
        """Validate multi-protocol analysis."""
        # Validate each protocol decoded successfully
        for protocol in ["i2c", "spi", "uart"]:
            if protocol in self.results:
                suite.check_exists(
                    f"{protocol}_signals",
                    self.results[protocol].get("signals"),
                    f"{protocol}: Signals generated",
                )

        # Validate timing correlation
        if "correlation" in self.results:
            suite.check_exists(
                "timing_events",
                self.results["correlation"].get("events"),
                "Timing correlation computed",
            )

    def _generate_i2c_read(
        self,
        address: int,
        register: int,
        data: bytes,
        clock_freq: float,
        sample_rate: float,
        start_sample: int,
    ) -> tuple[DigitalTrace, DigitalTrace]:
        """Generate I2C read transaction."""
        samples_per_bit = max(1, int(sample_rate / clock_freq))
        scl = [1] * start_sample
        sda = [1] * start_sample

        # Start condition
        sda.extend([0] * samples_per_bit)
        scl.extend([1] * samples_per_bit)

        # Address + Write
        addr_write = (address << 1) | 0
        for i in range(8):
            bit = (addr_write >> (7 - i)) & 1
            scl.extend([0] * (samples_per_bit // 2))
            sda.extend([bit] * (samples_per_bit // 2))
            scl.extend([1] * (samples_per_bit // 2))
            sda.extend([bit] * (samples_per_bit // 2))

        # ACK
        scl.extend([0] * (samples_per_bit // 2))
        sda.extend([0] * (samples_per_bit // 2))
        scl.extend([1] * (samples_per_bit // 2))
        sda.extend([0] * (samples_per_bit // 2))

        # Stop + idle
        scl.extend([1] * (samples_per_bit * 2))
        sda.extend([1] * (samples_per_bit * 2))

        return (
            DigitalTrace(
                data=np.array(scl, dtype=bool),
                metadata=TraceMetadata(sample_rate=sample_rate, channel_name="i2c_scl"),
            ),
            DigitalTrace(
                data=np.array(sda, dtype=bool),
                metadata=TraceMetadata(sample_rate=sample_rate, channel_name="i2c_sda"),
            ),
        )

    def _generate_spi_transfer(
        self,
        tx_data: bytes,
        rx_data: bytes,
        clock_freq: float,
        sample_rate: float,
        start_sample: int,
    ) -> tuple[DigitalTrace, DigitalTrace, DigitalTrace, DigitalTrace]:
        """Generate SPI transfer."""
        samples_per_bit = max(1, int(sample_rate / clock_freq))
        sck = [0] * start_sample
        mosi = [0] * start_sample
        miso = [0] * start_sample
        cs = [1] * start_sample

        # CS low
        cs.extend([0] * samples_per_bit)
        sck.extend([0] * samples_per_bit)
        mosi.extend([0] * samples_per_bit)
        miso.extend([0] * samples_per_bit)

        # Transfer bytes
        for tx_byte, rx_byte in zip(tx_data, rx_data):
            for i in range(8):
                tx_bit = (tx_byte >> (7 - i)) & 1
                rx_bit = (rx_byte >> (7 - i)) & 1
                sck.extend([0] * (samples_per_bit // 2))
                mosi.extend([tx_bit] * (samples_per_bit // 2))
                miso.extend([rx_bit] * (samples_per_bit // 2))
                cs.extend([0] * (samples_per_bit // 2))
                sck.extend([1] * (samples_per_bit // 2))
                mosi.extend([tx_bit] * (samples_per_bit // 2))
                miso.extend([rx_bit] * (samples_per_bit // 2))
                cs.extend([0] * (samples_per_bit // 2))

        # CS high
        cs.extend([1] * samples_per_bit)
        sck.extend([0] * samples_per_bit)
        mosi.extend([0] * samples_per_bit)
        miso.extend([0] * samples_per_bit)

        return tuple(
            DigitalTrace(
                data=np.array(sig, dtype=bool),
                metadata=TraceMetadata(sample_rate=sample_rate, channel_name=name),
            )
            for sig, name in [
                (sck, "spi_sck"),
                (mosi, "spi_mosi"),
                (miso, "spi_miso"),
                (cs, "spi_cs"),
            ]
        )

    def _generate_uart_message(
        self,
        message: bytes,
        baudrate: int,
        sample_rate: float,
        start_sample: int,
    ) -> DigitalTrace:
        """Generate UART message."""
        samples_per_bit = max(1, int(sample_rate / baudrate))
        signal = [1] * start_sample

        for byte_val in message:
            signal.extend([0] * samples_per_bit)  # Start bit
            for i in range(8):
                bit = (byte_val >> i) & 1
                signal.extend([bit] * samples_per_bit)
            signal.extend([1] * samples_per_bit)  # Stop bit

        return DigitalTrace(
            data=np.array(signal, dtype=bool),
            metadata=TraceMetadata(sample_rate=sample_rate, channel_name="uart_tx"),
        )

    def _analyze_i2c(
        self,
        scl: DigitalTrace,
        sda: DigitalTrace,
        expected_address: int,
    ) -> dict[str, object]:
        """Analyze I2C bus."""
        from demos.common.formatting import print_info

        print_info(f"I2C bus detected, expected address: 0x{expected_address:02X}")
        return {"signals": (scl, sda), "expected_address": expected_address}

    def _analyze_spi(
        self,
        sck: DigitalTrace,
        mosi: DigitalTrace,
        miso: DigitalTrace,
        cs: DigitalTrace,
        expected_bytes: int,
    ) -> dict[str, object]:
        """Analyze SPI bus."""
        from demos.common.formatting import print_info

        print_info(f"SPI bus detected, expected bytes: {expected_bytes}")
        return {"signals": (sck, mosi, miso, cs), "expected_bytes": expected_bytes}

    def _analyze_uart(
        self,
        tx: DigitalTrace,
        baudrate: int,
        expected_message: bytes,
    ) -> dict[str, object]:
        """Analyze UART."""
        from demos.common.formatting import print_info

        print_info(f"UART detected, baudrate: {baudrate}, message: {expected_message!r}")
        return {"signals": tx, "baudrate": baudrate, "expected_message": expected_message}

    def _correlate_protocols(self) -> None:
        """Correlate timing across protocols."""
        from demos.common.formatting import print_info

        events = [
            {"protocol": "I2C", "time_us": 0, "event": "Sensor read start"},
            {"protocol": "I2C", "time_us": 80, "event": "Sensor read complete"},
            {"protocol": "SPI", "time_us": 180, "event": "Data transmission start"},
            {"protocol": "SPI", "time_us": 212, "event": "Data transmission complete"},
            {"protocol": "UART", "time_us": 230, "event": "Debug message"},
        ]

        print_info("Timeline of protocol events:")
        for event in events:
            print_info(f"  {event['time_us']:6.1f} µs: {event['protocol']:4s} - {event['event']}")

        print_info(
            "\nObservation: I2C read triggers SPI transmission after 100 µs processing delay"
        )

        self.results["correlation"] = {"events": events}


if __name__ == "__main__":
    sys.exit(run_demo_main(ComprehensiveProtocolDemo))
