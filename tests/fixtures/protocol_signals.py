"""Protocol signal generation for testing protocol decoders.

This module provides functions to generate realistic protocol signals
for UART, SPI, I2C, and CAN testing with configurable parameters and error injection.
"""

from __future__ import annotations

from typing import Literal

import numpy as np


def generate_uart_signal(
    data: bytes,
    baudrate: int = 115200,
    sample_rate: float = 10e6,
    amplitude: float = 3.3,
    data_bits: int = 8,
    parity: Literal["none", "odd", "even", "mark", "space"] = "none",
    stop_bits: float = 1,
    idle_level: int = 1,
    inject_parity_error: bool = False,
    inject_framing_error: bool = False,
) -> np.ndarray:
    """Generate UART signal with configurable parameters.

    Args:
        data: Bytes to transmit
        baudrate: UART baud rate
        sample_rate: Sampling rate in Hz
        amplitude: Logic high level in volts
        data_bits: Number of data bits (5-9)
        parity: Parity mode
        stop_bits: Number of stop bits (1, 1.5, 2)
        idle_level: Idle line level (0 or 1)
        inject_parity_error: Force incorrect parity bit
        inject_framing_error: Force incorrect stop bit

    Returns:
        Generated UART signal with idle before and after
    """
    samples_per_bit = int(sample_rate / baudrate)
    signal = []
    idle_value = amplitude if idle_level else 0.0

    # Initial idle
    signal.extend([idle_value] * (samples_per_bit * 2))

    # Generate frame for each byte
    for data_byte in data:
        frame = []

        # Start bit (opposite of idle)
        frame.append(0 if idle_level else 1)

        # Data bits (LSB first)
        for i in range(data_bits):
            frame.append((data_byte >> i) & 1)

        # Parity bit
        if parity != "none":
            ones_count = sum((data_byte >> i) & 1 for i in range(data_bits))

            if parity == "odd":
                parity_bit = (ones_count + 1) % 2
            elif parity == "even":
                parity_bit = ones_count % 2
            elif parity == "mark":
                parity_bit = 1
            else:  # space
                parity_bit = 0

            if inject_parity_error:
                parity_bit = 1 - parity_bit

            frame.append(parity_bit)

        # Stop bits
        stop_bit_value = idle_level
        if inject_framing_error:
            stop_bit_value = 1 - idle_level

        if stop_bits == 1:
            frame.append(stop_bit_value)
        elif stop_bits == 1.5:
            frame.append(stop_bit_value)  # Will use 1.5x samples
        elif stop_bits == 2:
            frame.extend([stop_bit_value, stop_bit_value])

        # Convert bits to voltage levels
        for bit_idx, bit in enumerate(frame):
            level = amplitude if bit else 0.0

            # Handle 1.5 stop bits
            if stop_bits == 1.5 and bit_idx == len(frame) - 1:
                signal.extend([level] * int(samples_per_bit * 1.5))
            else:
                signal.extend([level] * samples_per_bit)

    # Final idle
    signal.extend([idle_value] * (samples_per_bit * 2))

    return np.array(signal)


def generate_spi_signals(
    mosi_data: bytes,
    miso_data: bytes | None = None,
    clock_rate: int = 1_000_000,
    sample_rate: float = 10e6,
    cpol: int = 0,
    cpha: int = 0,
    amplitude: float = 3.3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate SPI transaction signals.

    Args:
        mosi_data: Data to transmit on MOSI
        miso_data: Data to transmit on MISO
        clock_rate: SPI clock rate in Hz
        sample_rate: Sampling rate in Hz
        cpol: Clock polarity (0 or 1)
        cpha: Clock phase (0 or 1)
        amplitude: Logic high level in volts

    Returns:
        Tuple of (CLK, MOSI, MISO, CS) signals
    """
    if miso_data is None:
        miso_data = mosi_data

    samples_per_bit = int(sample_rate / clock_rate / 2)
    total_bits = len(mosi_data) * 8
    total_samples = samples_per_bit * 2 * total_bits + samples_per_bit * 4

    # Initialize arrays
    clk = np.full(total_samples, amplitude if cpol else 0.0)
    mosi = np.zeros(total_samples)
    miso = np.zeros(total_samples)
    cs = np.full(total_samples, amplitude)  # Active low

    idx = samples_per_bit
    cs[idx:] = 0.0  # Activate CS

    for byte_idx in range(len(mosi_data)):
        mosi_byte = mosi_data[byte_idx]
        miso_byte = miso_data[byte_idx] if byte_idx < len(miso_data) else 0

        for bit_idx in range(8):
            # Extract bit (MSB first)
            mosi_bit = (mosi_byte >> (7 - bit_idx)) & 1
            miso_bit = (miso_byte >> (7 - bit_idx)) & 1

            # Set data based on CPHA
            if cpha == 0:
                mosi[idx : idx + samples_per_bit * 2] = amplitude if mosi_bit else 0.0
                miso[idx : idx + samples_per_bit * 2] = amplitude if miso_bit else 0.0

            # Generate clock pulse
            if cpol == 0:
                clk[idx + samples_per_bit : idx + samples_per_bit * 2] = amplitude
            else:
                clk[idx + samples_per_bit : idx + samples_per_bit * 2] = 0.0

            if cpha == 1:
                mosi[idx + samples_per_bit : idx + samples_per_bit * 2] = (
                    amplitude if mosi_bit else 0.0
                )
                miso[idx + samples_per_bit : idx + samples_per_bit * 2] = (
                    amplitude if miso_bit else 0.0
                )

            idx += samples_per_bit * 2

    cs[idx:] = amplitude

    return clk, mosi, miso, cs


def generate_i2c_signals(
    address: int,
    data: bytes,
    clock_rate: int = 100_000,
    sample_rate: float = 10e6,
    amplitude: float = 3.3,
    inject_nack: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate I2C transaction signals.

    Args:
        address: 7-bit I2C address
        data: Data bytes to transmit
        clock_rate: I2C clock rate in Hz
        sample_rate: Sampling rate in Hz
        amplitude: Logic high level in volts
        inject_nack: Force NACK instead of ACK

    Returns:
        Tuple of (SCL, SDA) signals
    """
    samples_per_bit = int(sample_rate / clock_rate)
    half_bit = samples_per_bit // 2

    # Calculate total samples
    total_bits = 1 + 8 + 1 + len(data) * 9 + 1
    total_samples = samples_per_bit * total_bits + samples_per_bit * 2

    # Initialize with idle (both high)
    scl = np.full(total_samples, amplitude)
    sda = np.full(total_samples, amplitude)

    idx = samples_per_bit

    # START condition: SDA falls while SCL high
    sda[idx : idx + half_bit] = 0.0
    idx += half_bit

    # Address + R/W bit (LSB = 0 for write)
    addr_byte = (address << 1) | 0

    for bit_idx in range(8):
        bit = (addr_byte >> (7 - bit_idx)) & 1

        # Clock low
        scl[idx : idx + half_bit] = 0.0
        sda[idx : idx + samples_per_bit] = amplitude if bit else 0.0

        idx += half_bit

        # Clock high
        scl[idx : idx + half_bit] = amplitude
        idx += half_bit

    # ACK/NACK bit
    scl[idx : idx + half_bit] = 0.0
    if inject_nack:
        sda[idx : idx + samples_per_bit] = amplitude  # NACK
    else:
        sda[idx : idx + samples_per_bit] = 0.0  # ACK
    idx += half_bit
    scl[idx : idx + half_bit] = amplitude
    idx += half_bit

    # Data bytes
    for byte_val in data:
        for bit_idx in range(8):
            bit = (byte_val >> (7 - bit_idx)) & 1

            # Clock low
            scl[idx : idx + half_bit] = 0.0
            sda[idx : idx + samples_per_bit] = amplitude if bit else 0.0
            idx += half_bit

            # Clock high
            scl[idx : idx + half_bit] = amplitude
            idx += half_bit

        # ACK bit
        scl[idx : idx + half_bit] = 0.0
        sda[idx : idx + samples_per_bit] = 0.0  # ACK
        idx += half_bit
        scl[idx : idx + half_bit] = amplitude
        idx += half_bit

    # STOP condition: SDA rises while SCL high
    scl[idx : idx + half_bit] = 0.0
    sda[idx : idx + half_bit] = 0.0
    idx += half_bit
    scl[idx:] = amplitude
    sda[idx:] = amplitude

    return scl[:idx], sda[:idx]


def generate_can_signal(
    arbitration_id: int,
    data: bytes,
    bitrate: int = 500_000,
    sample_rate: float = 10e6,
    amplitude: float = 3.3,
    inject_crc_error: bool = False,
) -> np.ndarray:
    """Generate simplified CAN signal for testing.

    Note: This generates a simplified CAN frame without bit stuffing
    for basic decoder testing.

    Args:
        arbitration_id: CAN arbitration ID (11-bit)
        data: Data bytes (0-8 bytes)
        bitrate: CAN bit rate in Hz
        sample_rate: Sampling rate in Hz
        amplitude: Logic high level in volts
        inject_crc_error: Force incorrect CRC

    Returns:
        Generated CAN signal
    """
    samples_per_bit = int(sample_rate / bitrate)
    frame = []

    # SOF (Start of Frame) - dominant (0)
    frame.append(0)

    # Arbitration ID (11 bits, MSB first)
    for i in range(10, -1, -1):
        frame.append((arbitration_id >> i) & 1)

    # RTR bit (0 for data frame)
    frame.append(0)

    # IDE bit (0 for standard frame)
    frame.append(0)

    # Reserved bit
    frame.append(0)

    # DLC (4 bits, data length code)
    dlc = len(data)
    for i in range(3, -1, -1):
        frame.append((dlc >> i) & 1)

    # Data bytes
    for byte_val in data:
        for i in range(7, -1, -1):
            frame.append((byte_val >> i) & 1)

    # CRC (15 bits) - simplified, not real CRC
    crc = 0x5555 if not inject_crc_error else 0xAAAA
    for i in range(14, -1, -1):
        frame.append((crc >> i) & 1)

    # CRC delimiter (recessive)
    frame.append(1)

    # ACK slot (dominant from receiver)
    frame.append(0)

    # ACK delimiter (recessive)
    frame.append(1)

    # EOF (7 recessive bits)
    frame.extend([1] * 7)

    # Convert to voltage levels
    signal = []
    for bit in frame:
        level = amplitude if bit else 0.0
        signal.extend([level] * samples_per_bit)

    # Add idle
    signal.extend([amplitude] * (samples_per_bit * 3))

    return np.array(signal)
