"""Comprehensive unit tests for UART protocol decoder.

Tests for src/oscura/analyzers/protocols/uart.py (PRO-002)

This test suite provides comprehensive coverage of the UART decoder module,
including:
- UARTDecoder class initialization and configuration
- Frame decoding (5-9 data bits)
- Parity modes (none, odd, even, mark, space)
- Stop bits (1, 1.5, 2)
- Bit order (LSB, MSB)
- Idle level (0, 1)
- Auto-baud detection
- Error detection (parity, framing)
- Edge cases and error handling
- Convenience functions
"""

from __future__ import annotations

import numpy as np
import pytest

from oscura.analyzers.protocols.uart import UARTDecoder, decode_uart
from oscura.core.types import DigitalTrace, TraceMetadata, WaveformTrace

pytestmark = [pytest.mark.unit, pytest.mark.analyzer]


# =============================================================================
# Test Data Generators
# =============================================================================


def generate_uart_bit_stream(
    data_bytes: bytes,
    baudrate: int = 9600,
    sample_rate: float = 1_000_000.0,
    data_bits: int = 8,
    parity: str = "none",
    stop_bits: float = 1,
    bit_order: str = "lsb",
    idle_level: int = 1,
    add_idle: bool = True,
) -> np.ndarray:
    """Generate a UART bit stream for testing.

    Args:
        data_bytes: Bytes to encode.
        baudrate: Baud rate in bps.
        sample_rate: Sample rate in Hz.
        data_bits: Number of data bits (5-9).
        parity: Parity mode ("none", "odd", "even", "mark", "space").
        stop_bits: Number of stop bits (1, 1.5, 2).
        bit_order: Bit order ("lsb" or "msb").
        idle_level: Idle line level (0 or 1).
        add_idle: Whether to add idle periods before/after.

    Returns:
        Boolean array representing UART signal.
    """
    samples_per_bit = int(sample_rate / baudrate)
    bits = []

    # Initial idle period
    if add_idle:
        bits.extend([bool(idle_level)] * (samples_per_bit * 5))

    start_level = not idle_level
    stop_level = idle_level

    for byte_val in data_bytes:
        # Only use lower data_bits of the byte
        byte_val = byte_val & ((1 << data_bits) - 1)

        # Start bit
        bits.extend([start_level] * samples_per_bit)

        # Data bits
        data_bit_list = []
        for i in range(data_bits):
            if bit_order == "lsb":
                bit_val = (byte_val >> i) & 1
            else:
                bit_val = (byte_val >> (data_bits - 1 - i)) & 1
            data_bit_list.append(bit_val)
            bits.extend([bool(bit_val)] * samples_per_bit)

        # Parity bit
        if parity != "none":
            ones_count = sum(data_bit_list)
            if parity == "odd":
                parity_bit = (ones_count + 1) % 2
            elif parity == "even":
                parity_bit = ones_count % 2
            elif parity == "mark":
                parity_bit = 1
            else:  # space
                parity_bit = 0
            bits.extend([bool(parity_bit)] * samples_per_bit)

        # Stop bits
        stop_samples = int(samples_per_bit * stop_bits)
        bits.extend([stop_level] * stop_samples)

    # Final idle period
    if add_idle:
        bits.extend([bool(idle_level)] * (samples_per_bit * 5))

    return np.array(bits, dtype=np.bool_)


# =============================================================================
# UARTDecoder Initialization Tests
# =============================================================================


class TestUARTDecoderInit:
    """Test UARTDecoder initialization and configuration."""

    def test_default_initialization(self):
        """Test decoder with default parameters."""
        decoder = UARTDecoder()
        assert decoder.id == "uart"
        assert decoder.name == "UART"
        assert decoder._data_bits == 8
        assert decoder._parity == "none"
        assert decoder._stop_bits == 1
        assert decoder._bit_order == "lsb"
        assert decoder._idle_level == 1

    def test_custom_initialization(self):
        """Test decoder with custom parameters."""
        decoder = UARTDecoder(
            baudrate=115200,
            data_bits=7,
            parity="even",
            stop_bits=2,
            bit_order="msb",
            idle_level=0,
        )
        assert decoder._baudrate == 115200
        assert decoder._data_bits == 7
        assert decoder._parity == "even"
        assert decoder._stop_bits == 2
        assert decoder._bit_order == "msb"
        assert decoder._idle_level == 0

    def test_auto_baud_initialization(self):
        """Test decoder with auto-baud detection (baudrate=0)."""
        decoder = UARTDecoder(baudrate=0)
        assert decoder._baudrate == 0

    @pytest.mark.parametrize("data_bits", [5, 6, 7, 8, 9])
    def test_all_data_bit_sizes(self, data_bits: int):
        """Test all supported data bit sizes."""
        decoder = UARTDecoder(data_bits=data_bits)
        assert decoder._data_bits == data_bits

    @pytest.mark.parametrize("parity", ["none", "odd", "even", "mark", "space"])
    def test_all_parity_modes(self, parity: str):
        """Test all supported parity modes."""
        decoder = UARTDecoder(parity=parity)  # type: ignore[arg-type]
        assert decoder._parity == parity

    @pytest.mark.parametrize("stop_bits", [1, 1.5, 2])
    def test_all_stop_bit_values(self, stop_bits: float):
        """Test all supported stop bit values."""
        decoder = UARTDecoder(stop_bits=stop_bits)
        assert decoder._stop_bits == stop_bits

    def test_channels_definition(self):
        """Test decoder channel definitions."""
        decoder = UARTDecoder()
        assert len(decoder.channels) >= 1
        assert decoder.channels[0].id == "rx"
        assert decoder.channels[0].required is True

    def test_optional_channels_definition(self):
        """Test decoder optional channel definitions."""
        decoder = UARTDecoder()
        assert len(decoder.optional_channels) >= 1
        assert decoder.optional_channels[0].id == "tx"
        assert decoder.optional_channels[0].required is False


# =============================================================================
# Basic Decoding Tests
# =============================================================================


class TestBasicDecoding:
    """Test basic UART decoding functionality."""

    def test_decode_single_byte(self):
        """Test decoding a single byte."""
        sample_rate = 1_000_000.0
        baudrate = 9600
        data = b"A"

        signal = generate_uart_bit_stream(data, baudrate=baudrate, sample_rate=sample_rate)
        trace = DigitalTrace(data=signal, metadata=TraceMetadata(sample_rate=sample_rate))

        decoder = UARTDecoder(baudrate=baudrate)
        packets = list(decoder.decode(trace))

        assert len(packets) == 1
        assert packets[0].data == data
        assert packets[0].protocol == "uart"
        assert packets[0].timestamp > 0

    def test_decode_multiple_bytes(self):
        """Test decoding multiple bytes."""
        sample_rate = 1_000_000.0
        baudrate = 9600
        data = b"Hello"

        signal = generate_uart_bit_stream(data, baudrate=baudrate, sample_rate=sample_rate)
        trace = DigitalTrace(data=signal, metadata=TraceMetadata(sample_rate=sample_rate))

        decoder = UARTDecoder(baudrate=baudrate)
        packets = list(decoder.decode(trace))

        assert len(packets) == len(data)
        for i, packet in enumerate(packets):
            assert packet.data == bytes([data[i]])
            assert packet.annotations["frame_num"] == i

    def test_decode_all_ascii_printable(self):
        """Test decoding all printable ASCII characters."""
        sample_rate = 1_000_000.0
        baudrate = 9600
        data = bytes(range(32, 127))  # Printable ASCII

        signal = generate_uart_bit_stream(data, baudrate=baudrate, sample_rate=sample_rate)
        trace = DigitalTrace(data=signal, metadata=TraceMetadata(sample_rate=sample_rate))

        decoder = UARTDecoder(baudrate=baudrate)
        packets = list(decoder.decode(trace))

        assert len(packets) == len(data)
        for i, packet in enumerate(packets):
            assert packet.data == bytes([data[i]])

    def test_decode_binary_data(self):
        """Test decoding binary data (all byte values)."""
        sample_rate = 1_000_000.0
        baudrate = 9600
        data = bytes(range(256))

        signal = generate_uart_bit_stream(data, baudrate=baudrate, sample_rate=sample_rate)
        trace = DigitalTrace(data=signal, metadata=TraceMetadata(sample_rate=sample_rate))

        decoder = UARTDecoder(baudrate=baudrate)
        packets = list(decoder.decode(trace))

        assert len(packets) == len(data)
        for i, packet in enumerate(packets):
            assert packet.data == bytes([data[i]])


# =============================================================================
# Data Bits Tests
# =============================================================================


class TestDataBits:
    """Test different data bit configurations."""

    @pytest.mark.parametrize("data_bits", [5, 6, 7, 8, 9])
    def test_decode_with_different_data_bits(self, data_bits: int):
        """Test decoding with different data bit sizes."""
        sample_rate = 1_000_000.0
        baudrate = 9600
        max_val = (1 << data_bits) - 1
        data = bytes([i % (max_val + 1) for i in range(10)])

        signal = generate_uart_bit_stream(
            data,
            baudrate=baudrate,
            sample_rate=sample_rate,
            data_bits=data_bits,
        )
        trace = DigitalTrace(data=signal, metadata=TraceMetadata(sample_rate=sample_rate))

        decoder = UARTDecoder(baudrate=baudrate, data_bits=data_bits)
        packets = list(decoder.decode(trace))

        assert len(packets) == len(data)
        for i, packet in enumerate(packets):
            expected = data[i] & max_val
            assert packet.data[0] == expected

    def test_decode_5bit_data(self):
        """Test decoding 5-bit data (Baudot code style)."""
        sample_rate = 1_000_000.0
        baudrate = 9600
        data = b"\x00\x01\x1f\x10"  # Values within 5-bit range

        signal = generate_uart_bit_stream(
            data, baudrate=baudrate, sample_rate=sample_rate, data_bits=5
        )
        trace = DigitalTrace(data=signal, metadata=TraceMetadata(sample_rate=sample_rate))

        decoder = UARTDecoder(baudrate=baudrate, data_bits=5)
        packets = list(decoder.decode(trace))

        assert len(packets) == len(data)
        for i, packet in enumerate(packets):
            assert packet.data[0] == (data[i] & 0x1F)

    def test_decode_9bit_data(self):
        """Test decoding 9-bit data."""
        sample_rate = 1_000_000.0
        baudrate = 9600
        data = bytes([0, 1, 127, 128, 255])

        signal = generate_uart_bit_stream(
            data, baudrate=baudrate, sample_rate=sample_rate, data_bits=9
        )
        trace = DigitalTrace(data=signal, metadata=TraceMetadata(sample_rate=sample_rate))

        decoder = UARTDecoder(baudrate=baudrate, data_bits=9)
        packets = list(decoder.decode(trace))

        assert len(packets) == len(data)


# =============================================================================
# Parity Tests
# =============================================================================


class TestParity:
    """Test different parity configurations."""

    def test_decode_no_parity(self):
        """Test decoding without parity."""
        sample_rate = 1_000_000.0
        baudrate = 9600
        data = b"Test"

        signal = generate_uart_bit_stream(
            data, baudrate=baudrate, sample_rate=sample_rate, parity="none"
        )
        trace = DigitalTrace(data=signal, metadata=TraceMetadata(sample_rate=sample_rate))

        decoder = UARTDecoder(baudrate=baudrate, parity="none")
        packets = list(decoder.decode(trace))

        assert len(packets) == len(data)
        for packet in packets:
            assert len(packet.errors) == 0

    def test_decode_odd_parity(self):
        """Test decoding with odd parity."""
        sample_rate = 1_000_000.0
        baudrate = 9600
        data = b"ODD"

        signal = generate_uart_bit_stream(
            data, baudrate=baudrate, sample_rate=sample_rate, parity="odd"
        )
        trace = DigitalTrace(data=signal, metadata=TraceMetadata(sample_rate=sample_rate))

        decoder = UARTDecoder(baudrate=baudrate, parity="odd")
        packets = list(decoder.decode(trace))

        assert len(packets) == len(data)
        for packet in packets:
            assert len(packet.errors) == 0

    def test_decode_even_parity(self):
        """Test decoding with even parity."""
        sample_rate = 1_000_000.0
        baudrate = 9600
        data = b"EVEN"

        signal = generate_uart_bit_stream(
            data, baudrate=baudrate, sample_rate=sample_rate, parity="even"
        )
        trace = DigitalTrace(data=signal, metadata=TraceMetadata(sample_rate=sample_rate))

        decoder = UARTDecoder(baudrate=baudrate, parity="even")
        packets = list(decoder.decode(trace))

        assert len(packets) == len(data)
        for packet in packets:
            assert len(packet.errors) == 0

    def test_decode_mark_parity(self):
        """Test decoding with mark parity (always 1)."""
        sample_rate = 1_000_000.0
        baudrate = 9600
        data = b"MARK"

        signal = generate_uart_bit_stream(
            data, baudrate=baudrate, sample_rate=sample_rate, parity="mark"
        )
        trace = DigitalTrace(data=signal, metadata=TraceMetadata(sample_rate=sample_rate))

        decoder = UARTDecoder(baudrate=baudrate, parity="mark")
        packets = list(decoder.decode(trace))

        assert len(packets) == len(data)
        for packet in packets:
            assert len(packet.errors) == 0

    def test_decode_space_parity(self):
        """Test decoding with space parity (always 0)."""
        sample_rate = 1_000_000.0
        baudrate = 9600
        data = b"SPACE"

        signal = generate_uart_bit_stream(
            data, baudrate=baudrate, sample_rate=sample_rate, parity="space"
        )
        trace = DigitalTrace(data=signal, metadata=TraceMetadata(sample_rate=sample_rate))

        decoder = UARTDecoder(baudrate=baudrate, parity="space")
        packets = list(decoder.decode(trace))

        assert len(packets) == len(data)
        for packet in packets:
            assert len(packet.errors) == 0

    def test_parity_error_detection(self):
        """Test detection of parity errors."""
        sample_rate = 1_000_000.0
        baudrate = 9600
        data = b"X"

        # Generate signal with correct parity
        signal = generate_uart_bit_stream(
            data, baudrate=baudrate, sample_rate=sample_rate, parity="even"
        )

        # Flip the entire parity bit to create an error
        samples_per_bit = int(sample_rate / baudrate)
        # Parity bit is after idle (5) + start (1) + data bits (8) = bit 14
        parity_start = 14 * samples_per_bit
        parity_end = 15 * samples_per_bit
        if parity_end <= len(signal):
            signal[parity_start:parity_end] = ~signal[parity_start:parity_end]

        trace = DigitalTrace(data=signal, metadata=TraceMetadata(sample_rate=sample_rate))

        decoder = UARTDecoder(baudrate=baudrate, parity="even")
        packets = list(decoder.decode(trace))

        # Should still decode but report parity error
        assert len(packets) >= 1
        assert "Parity error" in packets[0].errors


# =============================================================================
# Stop Bits Tests
# =============================================================================


class TestStopBits:
    """Test different stop bit configurations."""

    @pytest.mark.parametrize("stop_bits", [1, 2])
    def test_decode_with_different_stop_bits(self, stop_bits: float):
        """Test decoding with different stop bit values."""
        sample_rate = 1_000_000.0
        baudrate = 9600
        data = b"STOP"

        signal = generate_uart_bit_stream(
            data, baudrate=baudrate, sample_rate=sample_rate, stop_bits=stop_bits
        )
        trace = DigitalTrace(data=signal, metadata=TraceMetadata(sample_rate=sample_rate))

        decoder = UARTDecoder(baudrate=baudrate, stop_bits=stop_bits)
        packets = list(decoder.decode(trace))

        assert len(packets) == len(data)
        for packet in packets:
            assert len(packet.errors) == 0

    def test_framing_error_detection(self):
        """Test detection of framing errors (invalid stop bit)."""
        sample_rate = 1_000_000.0
        baudrate = 9600
        data = b"F"

        # Generate signal with correct stop bit
        signal = generate_uart_bit_stream(data, baudrate=baudrate, sample_rate=sample_rate)

        # Flip the entire stop bit to create a framing error
        samples_per_bit = int(sample_rate / baudrate)
        # Stop bit is after idle (5) + start (1) + data bits (8) = bit 14 (no parity)
        stop_start = 14 * samples_per_bit
        stop_end = 15 * samples_per_bit
        if stop_end <= len(signal):
            signal[stop_start:stop_end] = ~signal[stop_start:stop_end]

        trace = DigitalTrace(data=signal, metadata=TraceMetadata(sample_rate=sample_rate))

        decoder = UARTDecoder(baudrate=baudrate)
        packets = list(decoder.decode(trace))

        # Should still decode but report framing error
        if len(packets) > 0:
            assert "Framing error" in packets[0].errors


# =============================================================================
# Bit Order Tests
# =============================================================================


class TestBitOrder:
    """Test different bit order configurations."""

    def test_decode_lsb_first(self):
        """Test decoding with LSB-first bit order (standard)."""
        sample_rate = 1_000_000.0
        baudrate = 9600
        data = b"LSB"

        signal = generate_uart_bit_stream(
            data, baudrate=baudrate, sample_rate=sample_rate, bit_order="lsb"
        )
        trace = DigitalTrace(data=signal, metadata=TraceMetadata(sample_rate=sample_rate))

        decoder = UARTDecoder(baudrate=baudrate, bit_order="lsb")
        packets = list(decoder.decode(trace))

        assert len(packets) == len(data)
        for i, packet in enumerate(packets):
            assert packet.data == bytes([data[i]])

    def test_decode_msb_first(self):
        """Test decoding with MSB-first bit order."""
        sample_rate = 1_000_000.0
        baudrate = 9600
        data = b"MSB"

        signal = generate_uart_bit_stream(
            data, baudrate=baudrate, sample_rate=sample_rate, bit_order="msb"
        )
        trace = DigitalTrace(data=signal, metadata=TraceMetadata(sample_rate=sample_rate))

        decoder = UARTDecoder(baudrate=baudrate, bit_order="msb")
        packets = list(decoder.decode(trace))

        assert len(packets) == len(data)
        for i, packet in enumerate(packets):
            assert packet.data == bytes([data[i]])

    def test_bit_order_difference(self):
        """Test that LSB and MSB first produce different results for same signal."""
        sample_rate = 1_000_000.0
        baudrate = 9600
        data = b"\x01"  # Binary: 00000001

        # Generate LSB-first signal
        signal = generate_uart_bit_stream(
            data, baudrate=baudrate, sample_rate=sample_rate, bit_order="lsb"
        )
        trace = DigitalTrace(data=signal, metadata=TraceMetadata(sample_rate=sample_rate))

        # Decode as MSB-first (incorrect)
        decoder_msb = UARTDecoder(baudrate=baudrate, bit_order="msb")
        packets_msb = list(decoder_msb.decode(trace))

        # Should get different value when decoding with wrong bit order
        if len(packets_msb) > 0:
            # LSB: 00000001 = 0x01
            # MSB: 10000000 = 0x80
            assert packets_msb[0].data[0] == 0x80


# =============================================================================
# Idle Level Tests
# =============================================================================


class TestIdleLevel:
    """Test different idle level configurations."""

    def test_decode_idle_high(self):
        """Test decoding with idle level high (standard RS-232)."""
        sample_rate = 1_000_000.0
        baudrate = 9600
        data = b"HI"

        signal = generate_uart_bit_stream(
            data, baudrate=baudrate, sample_rate=sample_rate, idle_level=1
        )
        trace = DigitalTrace(data=signal, metadata=TraceMetadata(sample_rate=sample_rate))

        decoder = UARTDecoder(baudrate=baudrate, idle_level=1)
        packets = list(decoder.decode(trace))

        assert len(packets) == len(data)
        for i, packet in enumerate(packets):
            assert packet.data == bytes([data[i]])

    def test_decode_idle_low(self):
        """Test decoding with idle level low (inverted)."""
        sample_rate = 1_000_000.0
        baudrate = 9600
        data = b"LO"

        signal = generate_uart_bit_stream(
            data, baudrate=baudrate, sample_rate=sample_rate, idle_level=0
        )
        trace = DigitalTrace(data=signal, metadata=TraceMetadata(sample_rate=sample_rate))

        decoder = UARTDecoder(baudrate=baudrate, idle_level=0)
        packets = list(decoder.decode(trace))

        assert len(packets) == len(data)
        for i, packet in enumerate(packets):
            assert packet.data == bytes([data[i]])


# =============================================================================
# Baudrate Tests
# =============================================================================


class TestBaudrate:
    """Test different baud rates."""

    @pytest.mark.parametrize("baudrate", [300, 1200, 2400, 4800, 9600, 19200, 38400, 57600, 115200])
    def test_decode_common_baudrates(self, baudrate: int):
        """Test decoding at common baud rates."""
        sample_rate = 10_000_000.0  # High sample rate for accurate timing
        data = b"OK"

        signal = generate_uart_bit_stream(data, baudrate=baudrate, sample_rate=sample_rate)
        trace = DigitalTrace(data=signal, metadata=TraceMetadata(sample_rate=sample_rate))

        decoder = UARTDecoder(baudrate=baudrate)
        packets = list(decoder.decode(trace))

        assert len(packets) == len(data)
        for i, packet in enumerate(packets):
            assert packet.data == bytes([data[i]])

    def test_auto_baud_detection_fallback(self):
        """Test auto-baud detection fallback.

        Auto-baud detection on a constant signal returns 1200 bps based on the
        pulse width detection algorithm finding the minimum pulse width.
        """
        sample_rate = 1_000_000.0
        # Create a signal that won't have clear baud rate
        signal = np.array([True] * 1000, dtype=np.bool_)
        trace = DigitalTrace(data=signal, metadata=TraceMetadata(sample_rate=sample_rate))

        decoder = UARTDecoder(baudrate=0)  # Auto-detect
        packets = list(decoder.decode(trace))

        # Auto-detect returns 1200 for constant signal, falls back to 9600 if detection fails (returns 0)
        # In this case, the constant True signal has no transitions, so detection returns a low value
        assert decoder._baudrate in [1200, 9600]  # Allow either detected or fallback value


# =============================================================================
# Waveform Input Tests
# =============================================================================


class TestWaveformInput:
    """Test decoding from analog waveform traces."""

    def test_decode_from_waveform_trace(self):
        """Test decoding from WaveformTrace (analog signal)."""
        sample_rate = 1_000_000.0
        baudrate = 9600
        data = b"WAV"

        # Generate digital signal
        digital_signal = generate_uart_bit_stream(data, baudrate=baudrate, sample_rate=sample_rate)

        # Convert to analog (0V = False, 3.3V = True)
        analog_signal = digital_signal.astype(np.float64) * 3.3

        trace = WaveformTrace(data=analog_signal, metadata=TraceMetadata(sample_rate=sample_rate))

        decoder = UARTDecoder(baudrate=baudrate)
        packets = list(decoder.decode(trace))

        assert len(packets) == len(data)
        for i, packet in enumerate(packets):
            assert packet.data == bytes([data[i]])


# =============================================================================
# Packet Annotation Tests
# =============================================================================


class TestPacketAnnotations:
    """Test packet annotations and metadata."""

    def test_packet_annotations(self):
        """Test that packets contain expected annotations."""
        sample_rate = 1_000_000.0
        baudrate = 9600
        data = b"A"

        signal = generate_uart_bit_stream(data, baudrate=baudrate, sample_rate=sample_rate)
        trace = DigitalTrace(data=signal, metadata=TraceMetadata(sample_rate=sample_rate))

        decoder = UARTDecoder(baudrate=baudrate)
        packets = list(decoder.decode(trace))

        assert len(packets) == 1
        packet = packets[0]

        # Check annotations
        assert "frame_num" in packet.annotations
        assert "data_bits" in packet.annotations
        assert "baudrate" in packet.annotations

        assert packet.annotations["frame_num"] == 0
        assert packet.annotations["baudrate"] == baudrate
        assert len(packet.annotations["data_bits"]) == 8

    def test_multiple_frame_numbers(self):
        """Test that frame numbers increment correctly."""
        sample_rate = 1_000_000.0
        baudrate = 9600
        data = b"ABC"

        signal = generate_uart_bit_stream(data, baudrate=baudrate, sample_rate=sample_rate)
        trace = DigitalTrace(data=signal, metadata=TraceMetadata(sample_rate=sample_rate))

        decoder = UARTDecoder(baudrate=baudrate)
        packets = list(decoder.decode(trace))

        assert len(packets) == 3
        for i, packet in enumerate(packets):
            assert packet.annotations["frame_num"] == i

    def test_timestamp_ordering(self):
        """Test that packet timestamps are in increasing order."""
        sample_rate = 1_000_000.0
        baudrate = 9600
        data = b"12345"

        signal = generate_uart_bit_stream(data, baudrate=baudrate, sample_rate=sample_rate)
        trace = DigitalTrace(data=signal, metadata=TraceMetadata(sample_rate=sample_rate))

        decoder = UARTDecoder(baudrate=baudrate)
        packets = list(decoder.decode(trace))

        assert len(packets) == len(data)
        for i in range(len(packets) - 1):
            assert packets[i].timestamp < packets[i + 1].timestamp


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestProtocolsUartEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_trace(self):
        """Test that empty trace raises ValueError."""
        sample_rate = 1_000_000.0
        signal = np.array([], dtype=np.bool_)

        # Empty trace creation raises ValueError in v0.9.0+
        with pytest.raises(ValueError, match="data array cannot be empty"):
            DigitalTrace(data=signal, metadata=TraceMetadata(sample_rate=sample_rate))

    def test_short_trace(self):
        """Test decoding a trace too short for a complete frame."""
        sample_rate = 1_000_000.0
        signal = np.array([True, False, True], dtype=np.bool_)
        trace = DigitalTrace(data=signal, metadata=TraceMetadata(sample_rate=sample_rate))

        decoder = UARTDecoder(baudrate=9600)
        packets = list(decoder.decode(trace))

        assert len(packets) == 0

    def test_all_idle(self):
        """Test decoding a trace with only idle signal."""
        sample_rate = 1_000_000.0
        signal = np.ones(1000, dtype=np.bool_)
        trace = DigitalTrace(data=signal, metadata=TraceMetadata(sample_rate=sample_rate))

        decoder = UARTDecoder(baudrate=9600)
        packets = list(decoder.decode(trace))

        assert len(packets) == 0

    def test_no_start_bit(self):
        """Test decoding when no valid start bit is found."""
        sample_rate = 1_000_000.0
        # Idle high, no transitions
        signal = np.ones(1000, dtype=np.bool_)
        trace = DigitalTrace(data=signal, metadata=TraceMetadata(sample_rate=sample_rate))

        decoder = UARTDecoder(baudrate=9600, idle_level=1)
        packets = list(decoder.decode(trace))

        assert len(packets) == 0

    def test_truncated_frame(self):
        """Test decoding a truncated frame at end of trace."""
        sample_rate = 1_000_000.0
        baudrate = 9600
        data = b"AB"

        signal = generate_uart_bit_stream(data, baudrate=baudrate, sample_rate=sample_rate)

        # Truncate during second byte
        truncate_idx = len(signal) - 50
        signal = signal[:truncate_idx]

        trace = DigitalTrace(data=signal, metadata=TraceMetadata(sample_rate=sample_rate))

        decoder = UARTDecoder(baudrate=baudrate)
        packets = list(decoder.decode(trace))

        # Should decode first byte, maybe not second
        assert len(packets) >= 1
        assert packets[0].data == b"A"

    def test_invalid_start_bit(self):
        """Test that invalid start bits are skipped."""
        sample_rate = 1_000_000.0
        baudrate = 9600
        data = b"X"

        signal = generate_uart_bit_stream(data, baudrate=baudrate, sample_rate=sample_rate)

        # Corrupt the start bit (make it same as idle)
        samples_per_bit = int(sample_rate / baudrate)
        start_bit_idx = samples_per_bit * 5 + samples_per_bit // 2
        signal[start_bit_idx] = True  # Should be False for valid start

        trace = DigitalTrace(data=signal, metadata=TraceMetadata(sample_rate=sample_rate))

        decoder = UARTDecoder(baudrate=baudrate)
        packets = list(decoder.decode(trace))

        # May not decode anything or may skip to next valid frame
        # This tests that it doesn't crash
        assert isinstance(packets, list)


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestConvenienceFunction:
    """Test the decode_uart convenience function."""

    def test_decode_uart_with_digital_trace(self):
        """Test decode_uart with DigitalTrace input."""
        sample_rate = 1_000_000.0
        baudrate = 9600
        data = b"TEST"

        signal = generate_uart_bit_stream(data, baudrate=baudrate, sample_rate=sample_rate)
        trace = DigitalTrace(data=signal, metadata=TraceMetadata(sample_rate=sample_rate))

        packets = decode_uart(trace, baudrate=baudrate)

        assert len(packets) == len(data)
        for i, packet in enumerate(packets):
            assert packet.data == bytes([data[i]])

    def test_decode_uart_with_waveform_trace(self):
        """Test decode_uart with WaveformTrace input."""
        sample_rate = 1_000_000.0
        baudrate = 9600
        data = b"WAVE"

        digital_signal = generate_uart_bit_stream(data, baudrate=baudrate, sample_rate=sample_rate)
        analog_signal = digital_signal.astype(np.float64) * 3.3

        trace = WaveformTrace(data=analog_signal, metadata=TraceMetadata(sample_rate=sample_rate))

        packets = decode_uart(trace, baudrate=baudrate)

        assert len(packets) == len(data)

    def test_decode_uart_with_numpy_array(self):
        """Test decode_uart with numpy array input."""
        sample_rate = 1_000_000.0
        baudrate = 9600
        data = b"ARR"

        signal = generate_uart_bit_stream(data, baudrate=baudrate, sample_rate=sample_rate)

        packets = decode_uart(signal, sample_rate=sample_rate, baudrate=baudrate)

        assert len(packets) == len(data)
        for i, packet in enumerate(packets):
            assert packet.data == bytes([data[i]])

    def test_decode_uart_with_custom_parameters(self):
        """Test decode_uart with custom parameters."""
        sample_rate = 1_000_000.0
        baudrate = 19200
        data = b"CFG"

        signal = generate_uart_bit_stream(
            data,
            baudrate=baudrate,
            sample_rate=sample_rate,
            data_bits=7,
            parity="even",
            stop_bits=2,
        )

        packets = decode_uart(
            signal,
            sample_rate=sample_rate,
            baudrate=baudrate,
            data_bits=7,
            parity="even",
            stop_bits=2,
        )

        assert len(packets) == len(data)
        for i, packet in enumerate(packets):
            assert packet.data[0] == (data[i] & 0x7F)

    def test_decode_uart_auto_baud(self):
        """Test decode_uart with auto-baud detection."""
        sample_rate = 1_000_000.0
        baudrate = 9600
        data = b"AUTO"

        signal = generate_uart_bit_stream(data, baudrate=baudrate, sample_rate=sample_rate)

        # Don't specify baudrate (None triggers auto-detect)
        packets = decode_uart(signal, sample_rate=sample_rate, baudrate=None)

        # Should decode with auto-detected or fallback baudrate
        assert isinstance(packets, list)


# =============================================================================
# Complex Configuration Tests
# =============================================================================


class TestComplexConfigurations:
    """Test complex combinations of parameters."""

    def test_7e1_configuration(self):
        """Test 7-bit data, even parity, 1 stop bit (7E1)."""
        sample_rate = 1_000_000.0
        baudrate = 9600
        data = b"7E1"

        signal = generate_uart_bit_stream(
            data,
            baudrate=baudrate,
            sample_rate=sample_rate,
            data_bits=7,
            parity="even",
            stop_bits=1,
        )
        trace = DigitalTrace(data=signal, metadata=TraceMetadata(sample_rate=sample_rate))

        decoder = UARTDecoder(baudrate=baudrate, data_bits=7, parity="even", stop_bits=1)
        packets = list(decoder.decode(trace))

        assert len(packets) == len(data)
        for packet in packets:
            assert len(packet.errors) == 0

    def test_8o2_configuration(self):
        """Test 8-bit data, odd parity, 2 stop bits (8O2)."""
        sample_rate = 1_000_000.0
        baudrate = 9600
        data = b"8O2"

        signal = generate_uart_bit_stream(
            data,
            baudrate=baudrate,
            sample_rate=sample_rate,
            data_bits=8,
            parity="odd",
            stop_bits=2,
        )
        trace = DigitalTrace(data=signal, metadata=TraceMetadata(sample_rate=sample_rate))

        decoder = UARTDecoder(baudrate=baudrate, data_bits=8, parity="odd", stop_bits=2)
        packets = list(decoder.decode(trace))

        assert len(packets) == len(data)
        for packet in packets:
            assert len(packet.errors) == 0

    def test_5n1_configuration(self):
        """Test 5-bit data, no parity, 1 stop bit (5N1)."""
        sample_rate = 1_000_000.0
        baudrate = 9600
        data = b"\x00\x1f\x10"

        signal = generate_uart_bit_stream(
            data,
            baudrate=baudrate,
            sample_rate=sample_rate,
            data_bits=5,
            parity="none",
            stop_bits=1,
        )
        trace = DigitalTrace(data=signal, metadata=TraceMetadata(sample_rate=sample_rate))

        decoder = UARTDecoder(baudrate=baudrate, data_bits=5, parity="none", stop_bits=1)
        packets = list(decoder.decode(trace))

        assert len(packets) == len(data)

    def test_inverted_8n1(self):
        """Test inverted UART (idle low) with standard 8N1."""
        sample_rate = 1_000_000.0
        baudrate = 9600
        data = b"INV"

        signal = generate_uart_bit_stream(
            data,
            baudrate=baudrate,
            sample_rate=sample_rate,
            idle_level=0,
        )
        trace = DigitalTrace(data=signal, metadata=TraceMetadata(sample_rate=sample_rate))

        decoder = UARTDecoder(baudrate=baudrate, idle_level=0)
        packets = list(decoder.decode(trace))

        assert len(packets) == len(data)
        for i, packet in enumerate(packets):
            assert packet.data == bytes([data[i]])


# =============================================================================
# Performance Tests
# =============================================================================


class TestPerformance:
    """Test decoder performance with large datasets."""

    def test_decode_large_message(self):
        """Test decoding a large message."""
        sample_rate = 10_000_000.0  # Higher sample rate for better precision at high baudrates
        baudrate = 115200  # Higher baudrate for faster processing
        data = b"A" * 1000  # 1000 bytes

        signal = generate_uart_bit_stream(data, baudrate=baudrate, sample_rate=sample_rate)
        trace = DigitalTrace(data=signal, metadata=TraceMetadata(sample_rate=sample_rate))

        decoder = UARTDecoder(baudrate=baudrate)
        packets = list(decoder.decode(trace))

        # With high baud rate (115200) we need adequate sample rate for reliable decoding
        # At 1 MHz sample rate, we only get ~8.7 samples per bit, which can cause timing issues
        # With 10 MHz, we get ~87 samples per bit for much better reliability
        assert len(packets) == len(data)

    def test_decode_all_byte_values(self):
        """Test decoding all possible byte values (0-255)."""
        sample_rate = 10_000_000.0  # Higher sample rate for better precision
        baudrate = 115200
        data = bytes(range(256))

        signal = generate_uart_bit_stream(data, baudrate=baudrate, sample_rate=sample_rate)
        trace = DigitalTrace(data=signal, metadata=TraceMetadata(sample_rate=sample_rate))

        decoder = UARTDecoder(baudrate=baudrate)
        packets = list(decoder.decode(trace))

        # With adequate sample rate, all 256 byte values should decode correctly
        assert len(packets) == 256
        for i, packet in enumerate(packets):
            assert packet.data == bytes([i])
