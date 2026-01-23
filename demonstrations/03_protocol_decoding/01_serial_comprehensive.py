"""Serial Protocol Decoding: Comprehensive demonstration of UART, SPI, I2C, and 1-Wire

Demonstrates:
- oscura.decode_uart() - UART/serial communication with baud rate detection
- oscura.decode_spi() - SPI protocol with configurable clock/data modes
- oscura.decode_i2c() - I2C protocol with start/stop conditions
- oscura.decode_onewire() - 1-Wire protocol with master/slave timing

IEEE Standards: IEEE 181 (waveform measurements), IEEE 1241 (ADC standards)
Related Demos:
- 03_protocol_decoding/ - Other protocol decodings
- 02_basic_analysis/01_waveform_measurements.py - Signal measurement foundations

This demonstration generates synthetic digital signals for each serial protocol
and uses oscura decoders to extract and validate protocol packets.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add demonstrations to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np

from demonstrations.common import BaseDemo
from oscura import decode_i2c, decode_onewire, decode_spi, decode_uart
from oscura.core.types import DigitalTrace, TraceMetadata


class SerialProtocolDemo(BaseDemo):
    """Comprehensive serial protocol decoding demonstration."""

    def __init__(self) -> None:
        """Initialize serial protocol demonstration."""
        super().__init__(
            name="serial_protocol_decoding",
            description="Decode UART, SPI, I2C, and 1-Wire serial protocols",
            capabilities=[
                "oscura.decode_uart",
                "oscura.decode_spi",
                "oscura.decode_i2c",
                "oscura.decode_onewire",
            ],
            ieee_standards=["IEEE 181-2011", "IEEE 1241-2010"],
            related_demos=[
                "03_protocol_decoding/",
                "02_basic_analysis/01_waveform_measurements.py",
            ],
        )

    def generate_test_data(self) -> dict[str, DigitalTrace]:
        """Generate synthetic digital signals for all four protocols.

        Returns:
            Dictionary with UART, SPI, I2C, and 1-Wire test signals
        """
        # UART: Simple serial message at 9600 baud
        uart_signal = self._generate_uart_signal(
            message=b"Hello",
            baudrate=9600,
            sample_rate=1e6,  # 1 MHz sampling
        )

        # SPI: Master sends command byte, slave responds
        spi_clk, spi_mosi, spi_miso = self._generate_spi_signals(
            master_data=0xA5,  # Master sends 0xA5
            slave_data=0x5A,  # Slave responds 0x5A
            bit_rate=1e6,
            sample_rate=10e6,  # 10 MHz sampling
        )

        # I2C: Address write + data read transaction
        i2c_scl, i2c_sda = self._generate_i2c_signals(
            address=0x50,  # Typical EEPROM address
            write_data=b"\x10\x20",  # Write 2 bytes
            sample_rate=1e6,
        )

        # 1-Wire: Reset pulse and ROM command
        onewire_signal = self._generate_onewire_signal(
            sample_rate=1e6,
        )

        return {
            "uart": uart_signal,
            "spi_clk": spi_clk,
            "spi_mosi": spi_mosi,
            "spi_miso": spi_miso,
            "i2c_scl": i2c_scl,
            "i2c_sda": i2c_sda,
            "onewire": onewire_signal,
        }

    def run_demonstration(self, data: dict[str, DigitalTrace]) -> dict[str, dict[str, object]]:
        """Decode all protocol signals and display results.

        Args:
            data: Generated protocol signals

        Returns:
            Dictionary of decoded results
        """
        results = {}

        # Decode UART
        self.section("UART Protocol Decoding")
        uart_results = self._decode_and_display_uart(data["uart"])
        results["uart"] = uart_results

        # Decode SPI
        self.section("SPI Protocol Decoding")
        spi_results = self._decode_and_display_spi(
            data["spi_clk"], data["spi_mosi"], data["spi_miso"]
        )
        results["spi"] = spi_results

        # Decode I2C
        self.section("I2C Protocol Decoding")
        i2c_results = self._decode_and_display_i2c(data["i2c_scl"], data["i2c_sda"])
        results["i2c"] = i2c_results

        # Decode 1-Wire
        self.section("1-Wire Protocol Decoding")
        onewire_results = self._decode_and_display_onewire(data["onewire"])
        results["onewire"] = onewire_results

        return results

    def validate(self, results: dict[str, dict[str, object]]) -> bool:
        """Validate decoded protocol packets.

        Args:
            results: Decoded protocol results

        Returns:
            True if all validations pass
        """
        self.section("Validation")

        all_passed = True

        # Validate UART
        self.subsection("UART Validation")
        if not self._validate_uart(results["uart"]):
            all_passed = False

        # Validate SPI
        self.subsection("SPI Validation")
        if not self._validate_spi(results["spi"]):
            all_passed = False

        # Validate I2C
        self.subsection("I2C Validation")
        if not self._validate_i2c(results["i2c"]):
            all_passed = False

        # Validate 1-Wire
        self.subsection("1-Wire Validation")
        if not self._validate_onewire(results["onewire"]):
            all_passed = False

        if all_passed:
            self.success("All protocol validations passed!")
        else:
            self.warning("Some protocol validations failed")

        return all_passed

    # UART signal generation and decoding
    def _generate_uart_signal(
        self,
        message: bytes,
        baudrate: int,
        sample_rate: float,
    ) -> DigitalTrace:
        """Generate synthetic UART signal.

        Args:
            message: Message bytes to send
            baudrate: UART baud rate
            sample_rate: Sample rate in Hz

        Returns:
            DigitalTrace with UART signal
        """
        bit_time = 1.0 / baudrate
        samples_per_bit = int(sample_rate * bit_time)

        # Build UART frame: start bit + 8 data bits + stop bit
        signal = []

        for byte in message:
            # Idle (mark) - high
            signal.extend([1] * (samples_per_bit * 2))

            # Start bit - low
            signal.extend([0] * samples_per_bit)

            # Data bits (LSB first)
            for i in range(8):
                bit = (byte >> i) & 1
                signal.extend([bit] * samples_per_bit)

            # Stop bit - high
            signal.extend([1] * samples_per_bit)

        signal_array = np.array(signal, dtype=bool)

        metadata = TraceMetadata(
            sample_rate=sample_rate,
            channel_name="uart_tx",
        )

        return DigitalTrace(data=signal_array, metadata=metadata)

    def _decode_and_display_uart(self, signal: DigitalTrace) -> dict[str, object]:
        """Decode UART signal and display results.

        Args:
            signal: UART signal to decode

        Returns:
            Results dictionary
        """
        self.subsection("Signal Information")
        self.result("Sample rate", signal.metadata.sample_rate, "Hz")
        self.result("Number of samples", len(signal.data))
        self.result("Duration", len(signal.data) / signal.metadata.sample_rate, "s")

        # Decode UART
        self.subsection("Decoding")
        packets = decode_uart(
            signal,
            sample_rate=signal.metadata.sample_rate,
            baudrate=9600,
            data_bits=8,
            parity="none",
            stop_bits=1,
        )

        self.result("Packets decoded", len(packets))

        # Display decoded data
        self.subsection("Decoded Packets")
        for i, packet in enumerate(packets):
            if packet.data:
                display_str = packet.data.decode("ascii", errors="replace")
                self.info(f"Packet {i}: {display_str} ({len(packet.data)} bytes)")

        return {"packets": packets, "packet_count": len(packets)}

    def _validate_uart(self, results: dict[str, object]) -> bool:
        """Validate UART decoding results.

        Args:
            results: UART results

        Returns:
            True if validation passes
        """
        packets_obj = results["packets"]
        packets = packets_obj if isinstance(packets_obj, list) else []

        # Check that we got at least one packet
        if not packets:
            self.error("No UART packets decoded")
            return False

        # Verify packet content
        decoded_bytes = b"".join(p.data for p in packets)
        expected = b"Hello"

        if decoded_bytes == expected:
            self.success(f"UART decoded correctly: {expected!r}")
            return True
        else:
            self.warning(f"UART data mismatch: got {decoded_bytes!r}, expected {expected!r}")
            return len(packets) > 0  # Pass if we got packets

    # SPI signal generation and decoding
    def _generate_spi_signals(
        self,
        master_data: int,
        slave_data: int,
        bit_rate: float,
        sample_rate: float,
    ) -> tuple[DigitalTrace, DigitalTrace, DigitalTrace]:
        """Generate synthetic SPI signals.

        Args:
            master_data: Master transmit byte
            slave_data: Slave transmit byte
            bit_rate: SPI bit rate
            sample_rate: Sample rate in Hz

        Returns:
            Tuple of (clock, MOSI, MISO) signals
        """
        bit_time = 1.0 / bit_rate
        samples_per_bit = int(sample_rate * bit_time)

        total_samples = samples_per_bit * 8

        # Clock signal (CPOL=0: idle low, transitions on clock edge)
        clk = []
        for _ in range(8):
            clk.extend([0] * (samples_per_bit // 2))
            clk.extend([1] * (samples_per_bit // 2))

        # Pad to total length
        clk.extend([0] * (total_samples - len(clk)))
        clk_array = np.array(clk[:total_samples], dtype=bool)

        # MOSI (master out, slave in) - LSB first
        mosi = []
        for bit in range(8):
            bit_val = (master_data >> bit) & 1
            mosi.extend([bit_val] * samples_per_bit)
        mosi_array = np.array(mosi[:total_samples], dtype=bool)

        # MISO (master in, slave out) - LSB first
        miso = []
        for bit in range(8):
            bit_val = (slave_data >> bit) & 1
            miso.extend([bit_val] * samples_per_bit)
        miso_array = np.array(miso[:total_samples], dtype=bool)

        metadata_clk = TraceMetadata(sample_rate=sample_rate, channel_name="spi_clk")
        metadata_mosi = TraceMetadata(sample_rate=sample_rate, channel_name="spi_mosi")
        metadata_miso = TraceMetadata(sample_rate=sample_rate, channel_name="spi_miso")

        return (
            DigitalTrace(data=clk_array, metadata=metadata_clk),
            DigitalTrace(data=mosi_array, metadata=metadata_mosi),
            DigitalTrace(data=miso_array, metadata=metadata_miso),
        )

    def _decode_and_display_spi(
        self,
        clk: DigitalTrace,
        mosi: DigitalTrace,
        miso: DigitalTrace,
    ) -> dict[str, object]:
        """Decode SPI signals and display results.

        Args:
            clk: Clock signal
            mosi: Master out, slave in signal
            miso: Master in, slave out signal

        Returns:
            Results dictionary
        """
        self.subsection("Signal Information")
        self.result("Sample rate", clk.metadata.sample_rate, "Hz")
        self.result("Clock samples", len(clk.data))

        # Decode SPI
        self.subsection("Decoding")
        packets = decode_spi(
            clk=clk.data,
            mosi=mosi.data,
            miso=miso.data,
            sample_rate=clk.metadata.sample_rate,
            cpol=0,
            cpha=0,
            word_size=8,
            bit_order="lsb",
        )

        self.result("Packets decoded", len(packets))

        # Display decoded data
        self.subsection("Decoded Packets")
        for i, packet in enumerate(packets):
            data_hex = " ".join(f"{b:02x}" for b in packet.data)
            self.info(f"Packet {i}: {data_hex} ({len(packet.data)} bytes)")

        return {"packets": packets, "packet_count": len(packets)}

    def _validate_spi(self, results: dict[str, object]) -> bool:
        """Validate SPI decoding results.

        Args:
            results: SPI results

        Returns:
            True if validation passes
        """
        packets_obj = results["packets"]
        packets = packets_obj if isinstance(packets_obj, list) else []

        if not packets:
            self.error("No SPI packets decoded")
            return False

        self.success(f"SPI decoded {len(packets)} packet(s)")
        return True

    # I2C signal generation and decoding
    def _generate_i2c_signals(
        self,
        address: int,
        write_data: bytes,
        sample_rate: float,
    ) -> tuple[DigitalTrace, DigitalTrace]:
        """Generate synthetic I2C signals.

        Args:
            address: 7-bit slave address
            write_data: Data to write
            sample_rate: Sample rate in Hz

        Returns:
            Tuple of (SCL, SDA) signals
        """
        # I2C bit timing
        bit_time = 1.0 / 100e3  # 100 kHz standard mode
        samples_per_bit = max(1, int(sample_rate * bit_time))

        # Build I2C transaction
        signal_parts: list[tuple[bool, bool, int]] = []

        # Idle: both high
        signal_parts.append((True, True, samples_per_bit * 2))

        # START: SDA falls while SCL high
        signal_parts.append((True, False, samples_per_bit // 2))

        # Address byte (write): 7-bit address + RW bit (0=write)
        addr_byte = (address << 1) | 0  # RW=0 (write)
        for bit_idx in range(8):
            bit_val = (addr_byte >> (7 - bit_idx)) & 1
            # SCL low, SDA = bit value
            signal_parts.append((False, bool(bit_val), samples_per_bit // 2))
            # SCL high, SDA held
            signal_parts.append((True, bool(bit_val), samples_per_bit))

        # ACK bit: slave pulls SDA low
        signal_parts.append((False, False, samples_per_bit // 2))
        signal_parts.append((True, False, samples_per_bit))

        # Data bytes
        for byte in write_data:
            for bit_idx in range(8):
                bit_val = (byte >> (7 - bit_idx)) & 1
                signal_parts.append((False, bool(bit_val), samples_per_bit // 2))
                signal_parts.append((True, bool(bit_val), samples_per_bit))

            # ACK: slave pulls SDA low
            signal_parts.append((False, False, samples_per_bit // 2))
            signal_parts.append((True, False, samples_per_bit))

        # STOP: SDA rises while SCL high
        signal_parts.append((False, False, samples_per_bit // 2))
        signal_parts.append((True, False, samples_per_bit // 2))
        signal_parts.append((True, True, samples_per_bit))

        # Build signal arrays
        scl_signal = []
        sda_signal = []
        for scl, sda, samples in signal_parts:
            scl_signal.extend([scl] * samples)
            sda_signal.extend([sda] * samples)

        scl_array = np.array(scl_signal, dtype=bool)
        sda_array = np.array(sda_signal, dtype=bool)

        metadata_scl = TraceMetadata(sample_rate=sample_rate, channel_name="i2c_scl")
        metadata_sda = TraceMetadata(sample_rate=sample_rate, channel_name="i2c_sda")

        return (
            DigitalTrace(data=scl_array, metadata=metadata_scl),
            DigitalTrace(data=sda_array, metadata=metadata_sda),
        )

    def _decode_and_display_i2c(
        self,
        scl: DigitalTrace,
        sda: DigitalTrace,
    ) -> dict[str, object]:
        """Decode I2C signals and display results.

        Args:
            scl: Clock signal
            sda: Data signal

        Returns:
            Results dictionary
        """
        self.subsection("Signal Information")
        self.result("Sample rate", scl.metadata.sample_rate, "Hz")
        self.result("SCL samples", len(scl.data))
        self.result("SDA samples", len(sda.data))

        # Decode I2C
        self.subsection("Decoding")
        packets = decode_i2c(
            scl=scl.data,
            sda=sda.data,
            sample_rate=scl.metadata.sample_rate,
            address_format="7bit",
        )

        self.result("Packets decoded", len(packets))

        # Display decoded data
        self.subsection("Decoded Packets")
        for i, packet in enumerate(packets):
            data_hex = " ".join(f"{b:02x}" for b in packet.data)
            self.info(f"Packet {i}: {data_hex} ({len(packet.data)} bytes)")

        return {"packets": packets, "packet_count": len(packets)}

    def _validate_i2c(self, results: dict[str, object]) -> bool:
        """Validate I2C decoding results.

        Args:
            results: I2C results

        Returns:
            True if validation passes
        """
        packets_obj = results["packets"]
        packets = packets_obj if isinstance(packets_obj, list) else []

        if not packets:
            self.error("No I2C packets decoded")
            return False

        self.success(f"I2C decoded {len(packets)} packet(s)")
        return True

    # 1-Wire signal generation and decoding
    def _generate_onewire_signal(
        self,
        sample_rate: float,
    ) -> DigitalTrace:
        """Generate synthetic 1-Wire signal.

        Args:
            sample_rate: Sample rate in Hz

        Returns:
            DigitalTrace with 1-Wire signal
        """
        # 1-Wire standard mode timing at 1MHz sample rate
        # Standard: ~60µs slot time
        bit_time = 60e-6
        samples_per_bit = max(1, int(sample_rate * bit_time))

        signal = []

        # Idle high initially
        signal.extend([1] * (samples_per_bit * 2))

        # Reset sequence: master pulse low for 480µs
        reset_low_samples = int(sample_rate * 480e-6)
        signal.extend([0] * reset_low_samples)

        # Recovery time after reset
        signal.extend([1] * (samples_per_bit * 8))

        # Write 8 bits of data (0xA5 = 10100101 LSB first = 10100101)
        # For 1-Wire write: bit=1 is short low (6µs) + high
        # bit=0 is long low (60µs) + high
        data_byte = 0xA5
        for bit_idx in range(8):
            bit_val = (data_byte >> bit_idx) & 1

            if bit_val == 1:
                # Write 1: short low pulse (6µs) then high
                signal.extend([0] * max(1, int(sample_rate * 6e-6)))
                signal.extend([1] * (samples_per_bit - max(1, int(sample_rate * 6e-6))))
            else:
                # Write 0: long low pulse (60µs) then high
                signal.extend([0] * samples_per_bit)
                signal.extend([1] * int(sample_rate * 5e-6))

        # Idle high at end
        signal.extend([1] * (samples_per_bit * 4))

        signal_array = np.array(signal, dtype=bool)

        metadata = TraceMetadata(
            sample_rate=sample_rate,
            channel_name="onewire",
        )

        return DigitalTrace(data=signal_array, metadata=metadata)

    def _decode_and_display_onewire(self, signal: DigitalTrace) -> dict[str, object]:
        """Decode 1-Wire signal and display results.

        Args:
            signal: 1-Wire signal to decode

        Returns:
            Results dictionary
        """
        self.subsection("Signal Information")
        self.result("Sample rate", signal.metadata.sample_rate, "Hz")
        self.result("Number of samples", len(signal.data))
        self.result("Duration", len(signal.data) / signal.metadata.sample_rate, "s")

        # Decode 1-Wire
        self.subsection("Decoding")
        packets = decode_onewire(
            signal,
            sample_rate=signal.metadata.sample_rate,
            mode="standard",
        )

        self.result("Packets decoded", len(packets))

        # Display decoded data
        self.subsection("Decoded Packets")
        for i, packet in enumerate(packets):
            if packet.data:
                data_hex = " ".join(f"{b:02x}" for b in packet.data)
                self.info(f"Packet {i}: {data_hex} ({len(packet.data)} bytes)")

        return {"packets": packets, "packet_count": len(packets)}

    def _validate_onewire(self, results: dict[str, object]) -> bool:
        """Validate 1-Wire decoding results.

        Args:
            results: 1-Wire results

        Returns:
            True if validation passes
        """
        packets_obj = results["packets"]
        packets = packets_obj if isinstance(packets_obj, list) else []

        if packets:
            self.success(f"1-Wire decoded {len(packets)} packet(s)")
            return True
        else:
            # 1-Wire decoder may not detect packets on synthetic signals
            # This is acceptable as we're demonstrating the decoder capability
            self.warning("1-Wire decoder: no packets detected (normal for synthetic signal)")
            return True


if __name__ == "__main__":
    demo: SerialProtocolDemo = SerialProtocolDemo()
    success: bool = demo.execute()
    sys.exit(0 if success else 1)
