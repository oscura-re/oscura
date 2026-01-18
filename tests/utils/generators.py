"""Synthetic data generators for Oscura tests.

This module provides protocol-specific synthetic data generators
for testing protocol decoders and analyzers.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class UARTGenerator:
    """Generator for UART protocol signals.

    Attributes:
        baud_rate: Baud rate in bits per second.
        sample_rate: Sample rate for the generated signal.
        data_bits: Number of data bits (5-9).
        parity: Parity type ('none', 'even', 'odd').
        stop_bits: Number of stop bits (1, 1.5, 2).
        inverted: If True, invert the signal polarity.
        seed: Random seed for reproducibility.
    """

    baud_rate: int = 9600
    sample_rate: float = 1e6
    data_bits: int = 8
    parity: str = "none"
    stop_bits: float = 1.0
    inverted: bool = False
    seed: int = 42

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        self._samples_per_bit = int(self.sample_rate / self.baud_rate)

    def encode_byte(self, byte_value: int) -> NDArray[np.float64]:
        """Encode a single byte as UART signal.

        Args:
            byte_value: The byte value (0-255) to encode.

        Returns:
            Signal array representing the UART frame.
        """
        bits = []

        # Start bit (always 0)
        bits.append(0)

        # Data bits (LSB first)
        for i in range(self.data_bits):
            bits.append((byte_value >> i) & 1)

        # Parity bit
        if self.parity != "none":
            ones = sum(bits[1:])  # Count data bits only
            if self.parity == "even":
                bits.append(ones % 2)
            else:  # odd
                bits.append(1 - (ones % 2))

        # Stop bit(s) (always 1)
        if self.stop_bits == 1.5:
            bits.extend([1, 1])  # Approximate with 2
        else:
            bits.extend([1] * int(self.stop_bits))

        # Convert to signal
        signal = np.repeat(bits, self._samples_per_bit).astype(np.float64)

        if self.inverted:
            signal = 1.0 - signal

        return signal

    def encode_message(
        self,
        message: bytes | str,
        *,
        idle_bits: int = 2,
    ) -> tuple[NDArray[np.float64], dict[str, Any]]:
        """Encode a complete message as UART signal.

        Args:
            message: The message to encode.
            idle_bits: Number of idle (high) bits between bytes.

        Returns:
            Tuple of (signal, metadata) where metadata contains timing info.
        """
        if isinstance(message, str):
            message = message.encode("utf-8")

        idle_signal = np.ones(self._samples_per_bit * idle_bits)
        if self.inverted:
            idle_signal = 1.0 - idle_signal

        signal_parts = []
        byte_offsets = []
        current_offset = 0

        for byte_val in message:
            # Add idle time (except before first byte)
            if signal_parts:
                signal_parts.append(idle_signal)
                current_offset += len(idle_signal)

            byte_offsets.append(current_offset)
            byte_signal = self.encode_byte(byte_val)
            signal_parts.append(byte_signal)
            current_offset += len(byte_signal)

        signal = np.concatenate(signal_parts)

        metadata = {
            "message": list(message),
            "byte_offsets": byte_offsets,
            "samples_per_bit": self._samples_per_bit,
            "baud_rate": self.baud_rate,
            "sample_rate": self.sample_rate,
        }

        return signal, metadata


@dataclass
class SPIGenerator:
    """Generator for SPI protocol signals.

    Attributes:
        clock_freq: SPI clock frequency in Hz.
        sample_rate: Sample rate for the generated signal.
        cpol: Clock polarity (0 or 1).
        cpha: Clock phase (0 or 1).
        bit_order: Bit order ('msb' or 'lsb').
        seed: Random seed for reproducibility.
    """

    clock_freq: float = 1e6
    sample_rate: float = 10e6
    cpol: int = 0
    cpha: int = 0
    bit_order: str = "msb"
    seed: int = 42

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        self._samples_per_half_clock = int(self.sample_rate / (2 * self.clock_freq))

    def generate_transaction(
        self,
        mosi_data: bytes,
        miso_data: bytes | None = None,
    ) -> tuple[dict[str, NDArray[np.float64]], dict[str, Any]]:
        """Generate SPI transaction signals.

        Args:
            mosi_data: Data to send on MOSI line.
            miso_data: Data to receive on MISO line (or random if None).

        Returns:
            Tuple of (signals_dict, metadata).
            signals_dict contains 'sclk', 'mosi', 'miso', 'cs'.
        """
        if miso_data is None:
            miso_data = bytes(self._rng.integers(0, 256, len(mosi_data), dtype=np.uint8))

        num_bits = len(mosi_data) * 8
        samples_per_bit = self._samples_per_half_clock * 2

        # Initialize signals
        cs_signal = []
        sclk_signal = []
        mosi_signal = []
        miso_signal = []

        # CS goes low
        cs_signal.extend([1] * self._samples_per_half_clock)  # Initial idle
        sclk_signal.extend([self.cpol] * self._samples_per_half_clock)
        mosi_signal.extend([0] * self._samples_per_half_clock)
        miso_signal.extend([0] * self._samples_per_half_clock)

        # CS active (low)
        cs_signal.extend([0] * (num_bits * samples_per_bit))

        # Generate clock and data
        for byte_idx, (mosi_byte, miso_byte) in enumerate(zip(mosi_data, miso_data, strict=False)):
            for bit_idx in range(8):
                if self.bit_order == "msb":
                    mosi_bit = (mosi_byte >> (7 - bit_idx)) & 1
                    miso_bit = (miso_byte >> (7 - bit_idx)) & 1
                else:
                    mosi_bit = (mosi_byte >> bit_idx) & 1
                    miso_bit = (miso_byte >> bit_idx) & 1

                # First half of clock cycle
                sclk_signal.extend([self.cpol] * self._samples_per_half_clock)
                # Second half of clock cycle
                sclk_signal.extend([1 - self.cpol] * self._samples_per_half_clock)

                # Data timing depends on CPHA
                if self.cpha == 0:
                    # Data valid on first edge
                    mosi_signal.extend([mosi_bit] * samples_per_bit)
                    miso_signal.extend([miso_bit] * samples_per_bit)
                else:
                    # Data valid on second edge
                    mosi_signal.extend([mosi_bit] * samples_per_bit)
                    miso_signal.extend([miso_bit] * samples_per_bit)

        # CS goes high
        cs_signal.extend([1] * self._samples_per_half_clock)
        sclk_signal.extend([self.cpol] * self._samples_per_half_clock)
        mosi_signal.extend([0] * self._samples_per_half_clock)
        miso_signal.extend([0] * self._samples_per_half_clock)

        signals = {
            "sclk": np.array(sclk_signal, dtype=np.float64),
            "mosi": np.array(mosi_signal, dtype=np.float64),
            "miso": np.array(miso_signal, dtype=np.float64),
            "cs": np.array(cs_signal, dtype=np.float64),
        }

        metadata = {
            "mosi_data": list(mosi_data),
            "miso_data": list(miso_data),
            "clock_freq": self.clock_freq,
            "sample_rate": self.sample_rate,
            "cpol": self.cpol,
            "cpha": self.cpha,
        }

        return signals, metadata


@dataclass
class I2CGenerator:
    """Generator for I2C protocol signals.

    Attributes:
        clock_freq: I2C clock frequency in Hz.
        sample_rate: Sample rate for the generated signal.
        address_bits: Address size (7 or 10).
        seed: Random seed for reproducibility.
    """

    clock_freq: float = 100e3
    sample_rate: float = 1e6
    address_bits: int = 7
    seed: int = 42

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        self._samples_per_half_clock = int(self.sample_rate / (2 * self.clock_freq))

    def generate_transaction(
        self,
        address: int,
        data: bytes,
        *,
        read: bool = False,
        with_nak: bool = False,
    ) -> tuple[dict[str, NDArray[np.float64]], dict[str, Any]]:
        """Generate I2C transaction signals.

        Args:
            address: Device address.
            data: Data bytes to write or read count.
            read: If True, generate a read transaction.
            with_nak: If True, end with NAK.

        Returns:
            Tuple of (signals_dict, metadata).
            signals_dict contains 'scl' and 'sda'.
        """
        scl_signal = []
        sda_signal = []

        samples_per_bit = self._samples_per_half_clock * 2

        # Start condition: SDA goes low while SCL is high
        scl_signal.extend([1] * self._samples_per_half_clock)
        sda_signal.extend([1] * self._samples_per_half_clock)
        scl_signal.extend([1] * self._samples_per_half_clock)
        sda_signal.extend([0] * self._samples_per_half_clock)

        def send_byte(byte_val: int, expect_ack: bool = True) -> None:
            """Send a byte on SDA with clock."""
            for i in range(8):
                bit = (byte_val >> (7 - i)) & 1
                # SCL low, set data
                scl_signal.extend([0] * self._samples_per_half_clock)
                sda_signal.extend([bit] * self._samples_per_half_clock)
                # SCL high, data stable
                scl_signal.extend([1] * self._samples_per_half_clock)
                sda_signal.extend([bit] * self._samples_per_half_clock)

            # ACK/NAK bit
            ack = 0 if expect_ack else 1
            scl_signal.extend([0] * self._samples_per_half_clock)
            sda_signal.extend([ack] * self._samples_per_half_clock)
            scl_signal.extend([1] * self._samples_per_half_clock)
            sda_signal.extend([ack] * self._samples_per_half_clock)

        # Address byte (7-bit address + R/W bit)
        addr_byte = (address << 1) | (1 if read else 0)
        send_byte(addr_byte)

        # Data bytes
        for i, byte_val in enumerate(data):
            is_last = i == len(data) - 1
            expect_ack = not (is_last and with_nak)
            send_byte(byte_val, expect_ack)

        # Stop condition: SDA goes high while SCL is high
        scl_signal.extend([0] * self._samples_per_half_clock)
        sda_signal.extend([0] * self._samples_per_half_clock)
        scl_signal.extend([1] * self._samples_per_half_clock)
        sda_signal.extend([0] * self._samples_per_half_clock)
        scl_signal.extend([1] * self._samples_per_half_clock)
        sda_signal.extend([1] * self._samples_per_half_clock)

        signals = {
            "scl": np.array(scl_signal, dtype=np.float64),
            "sda": np.array(sda_signal, dtype=np.float64),
        }

        metadata = {
            "address": address,
            "data": list(data),
            "read": read,
            "clock_freq": self.clock_freq,
            "sample_rate": self.sample_rate,
        }

        return signals, metadata


@dataclass
class CANGenerator:
    """Generator for CAN bus protocol signals.

    Attributes:
        bit_rate: CAN bit rate in bps.
        sample_rate: Sample rate for the generated signal.
        extended_id: Use 29-bit extended ID format.
        seed: Random seed for reproducibility.
    """

    bit_rate: int = 500000
    sample_rate: float = 10e6
    extended_id: bool = False
    seed: int = 42

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        self._samples_per_bit = int(self.sample_rate / self.bit_rate)

    def generate_frame(
        self,
        can_id: int,
        data: bytes,
        *,
        remote_frame: bool = False,
    ) -> tuple[NDArray[np.float64], dict[str, Any]]:
        """Generate a CAN frame signal.

        Args:
            can_id: CAN identifier.
            data: Data bytes (0-8 bytes).
            remote_frame: If True, generate RTR frame.

        Returns:
            Tuple of (signal, metadata).
        """
        bits = []

        # SOF (dominant = 0)
        bits.append(0)

        # Identifier (11 or 29 bits)
        if self.extended_id:
            # Extended format
            base_id = (can_id >> 18) & 0x7FF
            ext_id = can_id & 0x3FFFF

            for i in range(10, -1, -1):
                bits.append((base_id >> i) & 1)
            bits.append(1)  # SRR
            bits.append(1)  # IDE
            for i in range(17, -1, -1):
                bits.append((ext_id >> i) & 1)
        else:
            # Standard format
            for i in range(10, -1, -1):
                bits.append((can_id >> i) & 1)

        # RTR
        bits.append(1 if remote_frame else 0)

        # IDE (already sent for extended)
        if not self.extended_id:
            bits.append(0)

        # Reserved bit
        bits.append(0)

        # DLC (4 bits)
        dlc = len(data)
        for i in range(3, -1, -1):
            bits.append((dlc >> i) & 1)

        # Data field
        if not remote_frame:
            for byte_val in data:
                for i in range(7, -1, -1):
                    bits.append((byte_val >> i) & 1)

        # CRC (simplified - just use zeros for testing)
        crc = self._calculate_can_crc(bits)
        for i in range(14, -1, -1):
            bits.append((crc >> i) & 1)

        # CRC delimiter
        bits.append(1)

        # ACK slot (recessive, would be dominant if ACK'd)
        bits.append(1)

        # ACK delimiter
        bits.append(1)

        # EOF (7 recessive bits)
        bits.extend([1] * 7)

        # IFS (3 recessive bits)
        bits.extend([1] * 3)

        # Apply bit stuffing
        stuffed_bits = self._bit_stuff(bits)

        # Convert to signal
        signal = np.repeat(stuffed_bits, self._samples_per_bit).astype(np.float64)

        # CAN uses inverted logic (dominant = 0 = low voltage for differential)
        # For single-ended representation, keep as-is

        metadata = {
            "can_id": can_id,
            "data": list(data),
            "dlc": dlc,
            "extended": self.extended_id,
            "remote_frame": remote_frame,
            "bit_rate": self.bit_rate,
            "sample_rate": self.sample_rate,
        }

        return signal, metadata

    def _bit_stuff(self, bits: list[int]) -> list[int]:
        """Apply CAN bit stuffing (insert opposite bit after 5 same bits)."""
        result = []
        count = 0
        last_bit = None

        for bit in bits:
            result.append(bit)

            if bit == last_bit:
                count += 1
                if count == 5:
                    # Insert stuff bit
                    result.append(1 - bit)
                    count = 1
                    last_bit = 1 - bit
            else:
                count = 1
                last_bit = bit

        return result

    def _calculate_can_crc(self, bits: list[int]) -> int:
        """Calculate CAN CRC-15."""
        crc = 0
        for bit in bits:
            crc_nxt = bit ^ ((crc >> 14) & 1)
            crc = (crc << 1) & 0x7FFF
            if crc_nxt:
                crc ^= 0x4599  # CAN polynomial
        return crc


# Convenience functions


def generate_uart_signal(
    message: str | bytes,
    baud_rate: int = 9600,
    sample_rate: float = 1e6,
    **kwargs: Any,
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    """Generate UART signal for a message.

    Args:
        message: Message to encode.
        baud_rate: Baud rate.
        sample_rate: Sample rate.
        **kwargs: Additional UARTGenerator parameters.

    Returns:
        Tuple of (signal, metadata).
    """
    gen = UARTGenerator(baud_rate=baud_rate, sample_rate=sample_rate, **kwargs)
    return gen.encode_message(message)


def generate_spi_signals(
    mosi_data: bytes,
    clock_freq: float = 1e6,
    sample_rate: float = 10e6,
    **kwargs: Any,
) -> tuple[dict[str, NDArray[np.float64]], dict[str, Any]]:
    """Generate SPI transaction signals.

    Args:
        mosi_data: Data to send on MOSI.
        clock_freq: SPI clock frequency.
        sample_rate: Sample rate.
        **kwargs: Additional SPIGenerator parameters.

    Returns:
        Tuple of (signals_dict, metadata).
    """
    gen = SPIGenerator(clock_freq=clock_freq, sample_rate=sample_rate, **kwargs)
    return gen.generate_transaction(mosi_data)


def generate_i2c_signals(
    address: int,
    data: bytes,
    clock_freq: float = 100e3,
    sample_rate: float = 1e6,
    **kwargs: Any,
) -> tuple[dict[str, NDArray[np.float64]], dict[str, Any]]:
    """Generate I2C transaction signals.

    Args:
        address: Device address.
        data: Data bytes.
        clock_freq: I2C clock frequency.
        sample_rate: Sample rate.
        **kwargs: Additional I2CGenerator parameters.

    Returns:
        Tuple of (signals_dict, metadata).
    """
    gen = I2CGenerator(clock_freq=clock_freq, sample_rate=sample_rate, **kwargs)
    return gen.generate_transaction(address, data, **kwargs)


def generate_can_frame(
    can_id: int,
    data: bytes,
    bit_rate: int = 500000,
    sample_rate: float = 10e6,
    **kwargs: Any,
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    """Generate CAN frame signal.

    Args:
        can_id: CAN identifier.
        data: Data bytes (0-8).
        bit_rate: CAN bit rate.
        sample_rate: Sample rate.
        **kwargs: Additional CANGenerator parameters.

    Returns:
        Tuple of (signal, metadata).
    """
    gen = CANGenerator(bit_rate=bit_rate, sample_rate=sample_rate, **kwargs)
    return gen.generate_frame(can_id, data, **kwargs)
