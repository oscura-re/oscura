"""Comprehensive unit tests for SPI protocol decoder.

Tests for src/oscura/analyzers/protocols/spi.py (PRO-003)

This test suite provides comprehensive coverage of the SPI decoder module,
including:
- SPIDecoder class initialization and configuration
- SPI mode decoding (CPOL/CPHA combinations)
- Word size variations (4, 8, 16, 24, 32 bits)
- Bit order (MSB-first, LSB-first)
- MOSI/MISO data decoding
- Chip select (CS) handling
- Edge cases and error handling
- Convenience functions
"""

from __future__ import annotations

import numpy as np
import pytest

from oscura.analyzers.protocols.spi import SPIDecoder, decode_spi
from oscura.core.types import DigitalTrace, TraceMetadata

pytestmark = [pytest.mark.unit, pytest.mark.analyzer]


# =============================================================================
# Test Data Generators
# =============================================================================


def generate_spi_signals(
    data_bytes: bytes,
    cpol: int = 0,
    cpha: int = 0,
    word_size: int = 8,
    bit_order: str = "msb",
    samples_per_bit: int = 10,
    include_miso: bool = True,
    miso_data: bytes | None = None,
    include_cs: bool = False,
    cs_polarity: int = 0,
) -> dict[str, np.ndarray]:
    """Generate SPI signals for testing.

    Args:
        data_bytes: Data to transmit on MOSI.
        cpol: Clock polarity (0=idle low, 1=idle high).
        cpha: Clock phase (0=sample on first edge, 1=sample on second edge).
        word_size: Bits per word.
        bit_order: Bit order ("msb" or "lsb").
        samples_per_bit: Samples per clock cycle.
        include_miso: Whether to include MISO signal.
        miso_data: Data to transmit on MISO (if None, mirrors MOSI).
        include_cs: Whether to include CS signal.
        cs_polarity: CS active level (0=active low, 1=active high).

    Returns:
        Dictionary with "clk", "mosi", and optionally "miso" and "cs" arrays.
    """
    # Convert bytes to bits
    # For word_size > 8, group bytes into words (big-endian)
    # For word_size <= 8, process each byte separately
    mosi_bits = []
    bytes_per_word = max(1, (word_size + 7) // 8)

    # Process data in word-sized chunks
    for word_idx in range(0, len(data_bytes), bytes_per_word):
        # Get bytes for this word (pad if needed)
        word_bytes = data_bytes[word_idx : word_idx + bytes_per_word]
        if len(word_bytes) < bytes_per_word:
            word_bytes += b"\x00" * (bytes_per_word - len(word_bytes))

        # Convert bytes to word value (big-endian)
        word_value = int.from_bytes(word_bytes, "big")

        # Extract bits from word value
        for i in range(word_size):
            if bit_order == "msb":
                bit = (word_value >> (word_size - 1 - i)) & 1
            else:
                bit = (word_value >> i) & 1
            mosi_bits.append(bool(bit))

    # Generate MISO bits if needed
    if include_miso:
        if miso_data is None:
            miso_data = data_bytes
        miso_bits = []

        # Process MISO data the same way
        for word_idx in range(0, len(miso_data), bytes_per_word):
            word_bytes = miso_data[word_idx : word_idx + bytes_per_word]
            if len(word_bytes) < bytes_per_word:
                word_bytes += b"\x00" * (bytes_per_word - len(word_bytes))

            word_value = int.from_bytes(word_bytes, "big")

            for i in range(word_size):
                if bit_order == "msb":
                    bit = (word_value >> (word_size - 1 - i)) & 1
                else:
                    bit = (word_value >> i) & 1
                miso_bits.append(bool(bit))
    else:
        miso_bits = []

    # Calculate total samples
    n_bits = len(mosi_bits)
    idle_samples = samples_per_bit * 2
    total_samples = idle_samples + (n_bits * samples_per_bit) + idle_samples

    # Initialize signals with idle states
    clk = np.full(total_samples, cpol == 1, dtype=bool)
    mosi = np.zeros(total_samples, dtype=bool)
    if include_miso:
        miso = np.zeros(total_samples, dtype=bool)
    if include_cs:
        # Inactive state
        cs = np.full(total_samples, cs_polarity == 0, dtype=bool)

    # Generate clock and data signals for each bit
    for bit_idx in range(n_bits):
        start_sample = idle_samples + bit_idx * samples_per_bit

        # For CPHA=0: data valid before first edge, sampled on first edge
        # For CPHA=1: data valid after first edge, sampled on second edge

        # Set data for entire bit period (will be stable during sample edge)
        mosi[start_sample : start_sample + samples_per_bit] = mosi_bits[bit_idx]
        if include_miso and bit_idx < len(miso_bits):
            miso[start_sample : start_sample + samples_per_bit] = miso_bits[bit_idx]

        # Generate clock edges
        # For CPOL=0: idle low, active high
        # For CPOL=1: idle high, active low
        if cpol == 0:
            # Rising edge at 1/4, falling edge at 3/4
            clk[start_sample + samples_per_bit // 4 : start_sample + 3 * samples_per_bit // 4] = (
                True
            )
        else:
            # Falling edge at 1/4, rising edge at 3/4
            clk[start_sample + samples_per_bit // 4 : start_sample + 3 * samples_per_bit // 4] = (
                False
            )

        # CS active during transmission
        if include_cs:
            cs_active = cs_polarity == 1
            cs[start_sample : start_sample + samples_per_bit] = cs_active

    result = {"clk": clk, "mosi": mosi}
    if include_miso:
        result["miso"] = miso
    if include_cs:
        result["cs"] = cs

    return result


def create_clock_edges(
    n_edges: int,
    cpol: int = 0,
    cpha: int = 0,
    samples_per_bit: int = 10,
) -> np.ndarray:
    """Create a clock signal with specified number of edges.

    Args:
        n_edges: Number of sampling edges to generate.
        cpol: Clock polarity (0=idle low, 1=idle high).
        cpha: Clock phase (determines which edge is sampling edge).
        samples_per_bit: Samples per bit period.

    Returns:
        Clock signal as boolean array.
    """
    total_samples = n_edges * samples_per_bit + samples_per_bit
    clk = np.full(total_samples, cpol == 1, dtype=bool)

    for i in range(n_edges):
        start = i * samples_per_bit + samples_per_bit // 4
        end = start + samples_per_bit // 2

        if cpol == 0:
            clk[start:end] = True
        else:
            clk[start:end] = False

    return clk


# =============================================================================
# SPIDecoder Tests
# =============================================================================


class TestSPIDecoderInitialization:
    """Test SPIDecoder initialization and configuration."""

    def test_default_initialization(self):
        """Test decoder with default parameters."""
        decoder = SPIDecoder()

        assert decoder._cpol == 0
        assert decoder._cpha == 0
        assert decoder._word_size == 8
        assert decoder._bit_order == "msb"
        assert decoder._cs_polarity == 0

    def test_mode_0_initialization(self):
        """Test SPI Mode 0 (CPOL=0, CPHA=0)."""
        decoder = SPIDecoder(cpol=0, cpha=0)

        assert decoder._cpol == 0
        assert decoder._cpha == 0

    def test_mode_1_initialization(self):
        """Test SPI Mode 1 (CPOL=0, CPHA=1)."""
        decoder = SPIDecoder(cpol=0, cpha=1)

        assert decoder._cpol == 0
        assert decoder._cpha == 1

    def test_mode_2_initialization(self):
        """Test SPI Mode 2 (CPOL=1, CPHA=0)."""
        decoder = SPIDecoder(cpol=1, cpha=0)

        assert decoder._cpol == 1
        assert decoder._cpha == 0

    def test_mode_3_initialization(self):
        """Test SPI Mode 3 (CPOL=1, CPHA=1)."""
        decoder = SPIDecoder(cpol=1, cpha=1)

        assert decoder._cpol == 1
        assert decoder._cpha == 1

    def test_word_size_variations(self):
        """Test various word sizes."""
        for word_size in [4, 8, 16, 24, 32]:
            decoder = SPIDecoder(word_size=word_size)
            assert decoder._word_size == word_size

    def test_bit_order_msb(self):
        """Test MSB-first bit order."""
        decoder = SPIDecoder(bit_order="msb")
        assert decoder._bit_order == "msb"

    def test_bit_order_lsb(self):
        """Test LSB-first bit order."""
        decoder = SPIDecoder(bit_order="lsb")
        assert decoder._bit_order == "lsb"

    def test_cs_polarity_active_low(self):
        """Test CS active low."""
        decoder = SPIDecoder(cs_polarity=0)
        assert decoder._cs_polarity == 0

    def test_cs_polarity_active_high(self):
        """Test CS active high."""
        decoder = SPIDecoder(cs_polarity=1)
        assert decoder._cs_polarity == 1

    def test_class_attributes(self):
        """Test class-level attributes."""
        assert SPIDecoder.id == "spi"
        assert SPIDecoder.name == "SPI"
        assert SPIDecoder.longname == "Serial Peripheral Interface"
        assert SPIDecoder.desc == "SPI bus protocol decoder"

    def test_channel_definitions(self):
        """Test channel definitions."""
        decoder = SPIDecoder()

        # Required channels
        assert len(decoder.channels) == 2
        assert decoder.channels[0].id == "clk"
        assert decoder.channels[0].required is True
        assert decoder.channels[1].id == "mosi"
        assert decoder.channels[1].required is True

        # Optional channels
        assert len(decoder.optional_channels) == 2
        assert decoder.optional_channels[0].id == "miso"
        assert decoder.optional_channels[0].required is False
        assert decoder.optional_channels[1].id == "cs"
        assert decoder.optional_channels[1].required is False

    def test_option_definitions(self):
        """Test option definitions."""
        decoder = SPIDecoder()

        assert len(decoder.options) == 5

        # Check option IDs
        option_ids = [opt.id for opt in decoder.options]
        assert "cpol" in option_ids
        assert "cpha" in option_ids
        assert "word_size" in option_ids
        assert "bit_order" in option_ids
        assert "cs_polarity" in option_ids


class TestSPIDecoderMode0:
    """Test SPI Mode 0 (CPOL=0, CPHA=0) - most common mode."""

    def test_single_byte_decode(self):
        """Test decoding a single byte in Mode 0."""
        decoder = SPIDecoder(cpol=0, cpha=0, word_size=8)
        signals = generate_spi_signals(b"\xa5", cpol=0, cpha=0)

        packets = list(decoder.decode(**signals, sample_rate=1000.0))

        assert len(packets) == 1
        assert packets[0].protocol == "spi"
        assert packets[0].data == b"\xa5"
        assert packets[0].annotations["mosi_value"] == 0xA5
        assert packets[0].annotations["word_size"] == 8
        assert packets[0].annotations["mode"] == 0

    def test_multiple_bytes_decode(self):
        """Test decoding multiple bytes in Mode 0."""
        decoder = SPIDecoder(cpol=0, cpha=0, word_size=8)
        data = b"\x12\x34\x56\x78"
        signals = generate_spi_signals(data, cpol=0, cpha=0)

        packets = list(decoder.decode(**signals, sample_rate=1000.0))

        assert len(packets) == 4
        for i, expected_byte in enumerate(data):
            assert packets[i].data == bytes([expected_byte])
            assert packets[i].annotations["mosi_value"] == expected_byte
            assert packets[i].annotations["word_num"] == i

    def test_miso_decode(self):
        """Test MISO data decoding in Mode 0."""
        decoder = SPIDecoder(cpol=0, cpha=0, word_size=8)
        mosi_data = b"\xaa"
        miso_data = b"\x55"
        signals = generate_spi_signals(mosi_data, cpol=0, cpha=0, miso_data=miso_data)

        packets = list(decoder.decode(**signals, sample_rate=1000.0))

        assert len(packets) == 1
        assert packets[0].annotations["mosi_value"] == 0xAA
        assert packets[0].annotations["miso_value"] == 0x55

    def test_with_digital_trace(self):
        """Test decoding with DigitalTrace input."""
        decoder = SPIDecoder(cpol=0, cpha=0, word_size=8)
        signals = generate_spi_signals(b"\xff", cpol=0, cpha=0)

        metadata = TraceMetadata(sample_rate=1000.0)
        trace = DigitalTrace(data=signals["clk"], metadata=metadata)

        packets = list(
            decoder.decode(
                trace=trace,
                mosi=signals["mosi"],
                miso=signals["miso"],
            )
        )

        assert len(packets) == 1
        assert packets[0].annotations["mosi_value"] == 0xFF


class TestSPIDecoderMode1:
    """Test SPI Mode 1 (CPOL=0, CPHA=1)."""

    def test_single_byte_decode(self):
        """Test decoding a single byte in Mode 1."""
        decoder = SPIDecoder(cpol=0, cpha=1, word_size=8)
        signals = generate_spi_signals(b"\x5a", cpol=0, cpha=1)

        packets = list(decoder.decode(**signals, sample_rate=1000.0))

        assert len(packets) == 1
        assert packets[0].annotations["mosi_value"] == 0x5A
        assert packets[0].annotations["mode"] == 1


class TestSPIDecoderMode2:
    """Test SPI Mode 2 (CPOL=1, CPHA=0)."""

    def test_single_byte_decode(self):
        """Test decoding a single byte in Mode 2."""
        decoder = SPIDecoder(cpol=1, cpha=0, word_size=8)
        signals = generate_spi_signals(b"\xc3", cpol=1, cpha=0)

        packets = list(decoder.decode(**signals, sample_rate=1000.0))

        assert len(packets) == 1
        assert packets[0].annotations["mosi_value"] == 0xC3
        assert packets[0].annotations["mode"] == 2


class TestSPIDecoderMode3:
    """Test SPI Mode 3 (CPOL=1, CPHA=1)."""

    def test_single_byte_decode(self):
        """Test decoding a single byte in Mode 3."""
        decoder = SPIDecoder(cpol=1, cpha=1, word_size=8)
        signals = generate_spi_signals(b"\x3c", cpol=1, cpha=1)

        packets = list(decoder.decode(**signals, sample_rate=1000.0))

        assert len(packets) == 1
        assert packets[0].annotations["mosi_value"] == 0x3C
        assert packets[0].annotations["mode"] == 3


class TestSPIDecoderWordSizes:
    """Test various word sizes."""

    def test_4bit_words(self):
        """Test 4-bit word size."""
        decoder = SPIDecoder(cpol=0, cpha=0, word_size=4)
        signals = generate_spi_signals(b"\x0f", cpol=0, cpha=0, word_size=4)

        packets = list(decoder.decode(**signals, sample_rate=1000.0))

        assert len(packets) == 1
        assert packets[0].annotations["mosi_value"] == 0x0F
        assert packets[0].annotations["word_size"] == 4

    def test_8bit_words(self):
        """Test 8-bit word size (standard)."""
        decoder = SPIDecoder(cpol=0, cpha=0, word_size=8)
        signals = generate_spi_signals(b"\xff", cpol=0, cpha=0, word_size=8)

        packets = list(decoder.decode(**signals, sample_rate=1000.0))

        assert len(packets) == 1
        assert packets[0].annotations["mosi_value"] == 0xFF
        assert packets[0].annotations["word_size"] == 8

    def test_16bit_words(self):
        """Test 16-bit word size."""
        decoder = SPIDecoder(cpol=0, cpha=0, word_size=16)
        # For 16-bit word, we need 2 bytes per word
        signals = generate_spi_signals(b"\x12\x34", cpol=0, cpha=0, word_size=16)

        packets = list(decoder.decode(**signals, sample_rate=1000.0))

        assert len(packets) == 1
        assert packets[0].annotations["mosi_value"] == 0x1234
        assert packets[0].annotations["word_size"] == 16

    def test_24bit_words(self):
        """Test 24-bit word size."""
        decoder = SPIDecoder(cpol=0, cpha=0, word_size=24)
        signals = generate_spi_signals(b"\xab\xcd\xef", cpol=0, cpha=0, word_size=24)

        packets = list(decoder.decode(**signals, sample_rate=1000.0))

        assert len(packets) == 1
        assert packets[0].annotations["mosi_value"] == 0xABCDEF
        assert packets[0].annotations["word_size"] == 24

    def test_32bit_words(self):
        """Test 32-bit word size."""
        decoder = SPIDecoder(cpol=0, cpha=0, word_size=32)
        signals = generate_spi_signals(b"\xde\xad\xbe\xef", cpol=0, cpha=0, word_size=32)

        packets = list(decoder.decode(**signals, sample_rate=1000.0))

        assert len(packets) == 1
        assert packets[0].annotations["mosi_value"] == 0xDEADBEEF
        assert packets[0].annotations["word_size"] == 32


class TestSPIDecoderBitOrder:
    """Test MSB-first and LSB-first bit orders."""

    def test_msb_first(self):
        """Test MSB-first bit order."""
        decoder = SPIDecoder(cpol=0, cpha=0, word_size=8, bit_order="msb")
        signals = generate_spi_signals(b"\xa5", cpol=0, cpha=0, bit_order="msb")

        packets = list(decoder.decode(**signals, sample_rate=1000.0))

        assert len(packets) == 1
        assert packets[0].annotations["mosi_value"] == 0xA5

    def test_lsb_first(self):
        """Test LSB-first bit order."""
        decoder = SPIDecoder(cpol=0, cpha=0, word_size=8, bit_order="lsb")
        signals = generate_spi_signals(b"\xa5", cpol=0, cpha=0, bit_order="lsb")

        packets = list(decoder.decode(**signals, sample_rate=1000.0))

        assert len(packets) == 1
        # 0xA5 = 10100101 in MSB, becomes 10100101 in LSB which is still 0xA5
        assert packets[0].annotations["mosi_value"] == 0xA5

    def test_lsb_asymmetric_pattern(self):
        """Test LSB-first with asymmetric bit pattern."""
        decoder = SPIDecoder(cpol=0, cpha=0, word_size=8, bit_order="lsb")
        # In LSB-first mode, bit 0 is transmitted first
        # 0x03 = 00000011 binary: bit 0 = 1, bit 1 = 1, bits 2-7 = 0
        # Bits on wire: [1,1,0,0,0,0,0,0] (bit 0 first)
        # When decoded as LSB-first: first bit → bit 0, second → bit 1, etc.
        # Result: 0x03 (value is preserved)
        signals = generate_spi_signals(b"\x03", cpol=0, cpha=0, bit_order="lsb")

        packets = list(decoder.decode(**signals, sample_rate=1000.0))

        assert len(packets) == 1
        # LSB-first mode preserves the value when both TX and RX use same mode
        assert packets[0].annotations["mosi_value"] == 0x03


class TestSPIDecoderChipSelect:
    """Test chip select (CS) handling."""

    def test_cs_active_low(self):
        """Test CS active low (standard)."""
        decoder = SPIDecoder(cpol=0, cpha=0, cs_polarity=0)
        signals = generate_spi_signals(b"\xaa", cpol=0, cpha=0, include_cs=True, cs_polarity=0)

        packets = list(decoder.decode(**signals, sample_rate=1000.0))

        assert len(packets) == 1
        assert packets[0].annotations["mosi_value"] == 0xAA

    def test_cs_active_high(self):
        """Test CS active high."""
        decoder = SPIDecoder(cpol=0, cpha=0, cs_polarity=1)
        signals = generate_spi_signals(b"\x55", cpol=0, cpha=0, include_cs=True, cs_polarity=1)

        packets = list(decoder.decode(**signals, sample_rate=1000.0))

        assert len(packets) == 1
        assert packets[0].annotations["mosi_value"] == 0x55

    def test_cs_deasserted_blocks_decode(self):
        """Test that deasserted CS blocks decoding."""
        decoder = SPIDecoder(cpol=0, cpha=0, cs_polarity=0)

        # Create signals with CS deasserted (high for active-low)
        signals = generate_spi_signals(b"\xff", cpol=0, cpha=0, include_cs=True)
        signals["cs"][:] = True  # CS high = deasserted for active-low

        packets = list(decoder.decode(**signals, sample_rate=1000.0))

        # No packets should be decoded when CS is deasserted
        assert len(packets) == 0


class TestSPIDecoderBitsToValue:
    """Test the _bits_to_value internal method."""

    def test_bits_to_value_msb(self):
        """Test MSB-first bit to value conversion."""
        decoder = SPIDecoder(bit_order="msb")

        # 10100101 = 0xA5
        bits = [1, 0, 1, 0, 0, 1, 0, 1]
        value = decoder._bits_to_value(bits)

        assert value == 0xA5

    def test_bits_to_value_lsb(self):
        """Test LSB-first bit to value conversion."""
        decoder = SPIDecoder(bit_order="lsb")

        # LSB: bit 0 is LSB
        bits = [1, 0, 1, 0, 0, 1, 0, 1]  # 0xA5 in LSB order
        value = decoder._bits_to_value(bits)

        assert value == 0xA5

    def test_bits_to_value_empty(self):
        """Test empty bit list."""
        decoder = SPIDecoder()

        value = decoder._bits_to_value([])

        assert value == 0

    def test_bits_to_value_single_bit(self):
        """Test single bit conversion."""
        decoder = SPIDecoder()

        assert decoder._bits_to_value([0]) == 0
        assert decoder._bits_to_value([1]) == 1

    def test_bits_to_value_16bit(self):
        """Test 16-bit value conversion."""
        decoder = SPIDecoder(bit_order="msb")

        # 0x1234 = 0001001000110100
        bits = [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0]
        value = decoder._bits_to_value(bits)

        assert value == 0x1234


class TestSPIDecoderEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_clk_signal(self):
        """Test with empty clock signal."""
        decoder = SPIDecoder()

        clk = np.array([], dtype=bool)
        mosi = np.array([], dtype=bool)

        packets = list(decoder.decode(clk=clk, mosi=mosi, sample_rate=1000.0))

        assert len(packets) == 0

    def test_no_clock_edges(self):
        """Test with constant clock (no edges)."""
        decoder = SPIDecoder(cpol=0, cpha=0)

        clk = np.zeros(100, dtype=bool)
        mosi = np.ones(100, dtype=bool)

        packets = list(decoder.decode(clk=clk, mosi=mosi, sample_rate=1000.0))

        assert len(packets) == 0

    def test_none_clk(self):
        """Test with None clock signal."""
        decoder = SPIDecoder()

        packets = list(decoder.decode(clk=None, mosi=None, sample_rate=1000.0))

        assert len(packets) == 0

    def test_none_mosi(self):
        """Test with None MOSI signal."""
        decoder = SPIDecoder()

        clk = np.array([False, True, False, True], dtype=bool)

        packets = list(decoder.decode(clk=clk, mosi=None, sample_rate=1000.0))

        assert len(packets) == 0

    def test_mismatched_signal_lengths(self):
        """Test with mismatched signal lengths."""
        decoder = SPIDecoder(cpol=0, cpha=0)

        clk = np.array([False, True] * 50, dtype=bool)
        mosi = np.array([True, False] * 25, dtype=bool)  # Half length

        # Should truncate to shortest length
        packets = list(decoder.decode(clk=clk, mosi=mosi, sample_rate=1000.0))

        # Should still decode based on available samples
        assert isinstance(packets, list)

    def test_partial_word(self):
        """Test with incomplete word (not enough bits)."""
        decoder = SPIDecoder(cpol=0, cpha=0, word_size=8)

        # Only 4 clock edges (4 bits) when we need 8
        clk = create_clock_edges(4, cpol=0, cpha=0)
        mosi = np.ones(len(clk), dtype=bool)

        packets = list(decoder.decode(clk=clk, mosi=mosi, sample_rate=1000.0))

        # Should not yield incomplete word
        assert len(packets) == 0

    def test_exact_word_boundary(self):
        """Test with exactly one complete word."""
        decoder = SPIDecoder(cpol=0, cpha=0, word_size=8)

        clk = create_clock_edges(8, cpol=0, cpha=0)
        mosi = np.ones(len(clk), dtype=bool)

        packets = list(decoder.decode(clk=clk, mosi=mosi, sample_rate=1000.0))

        assert len(packets) == 1
        assert packets[0].annotations["mosi_value"] == 0xFF

    def test_miso_without_data(self):
        """Test MISO with None value."""
        decoder = SPIDecoder(cpol=0, cpha=0)
        signals = generate_spi_signals(b"\xaa", cpol=0, cpha=0, include_miso=False)

        packets = list(decoder.decode(**signals, sample_rate=1000.0))

        assert len(packets) == 1
        assert "miso_value" not in packets[0].annotations
        assert "miso_bits" not in packets[0].annotations


class TestSPIDecoderTiming:
    """Test timing and sample rate calculations."""

    def test_timestamp_calculation(self):
        """Test packet timestamp calculation."""
        decoder = SPIDecoder(cpol=0, cpha=0, word_size=8)
        signals = generate_spi_signals(b"\x42", cpol=0, cpha=0, samples_per_bit=10)

        sample_rate = 1000.0
        packets = list(decoder.decode(**signals, sample_rate=sample_rate))

        assert len(packets) == 1
        assert packets[0].timestamp >= 0.0
        assert isinstance(packets[0].timestamp, float)

    def test_high_sample_rate(self):
        """Test with high sample rate."""
        decoder = SPIDecoder(cpol=0, cpha=0, word_size=8)
        signals = generate_spi_signals(b"\x99", cpol=0, cpha=0)

        sample_rate = 1e9  # 1 GHz
        packets = list(decoder.decode(**signals, sample_rate=sample_rate))

        assert len(packets) == 1
        assert packets[0].timestamp < 1.0  # Should be in nanoseconds range

    def test_multiple_packets_timing(self):
        """Test timing for multiple packets."""
        decoder = SPIDecoder(cpol=0, cpha=0, word_size=8)
        signals = generate_spi_signals(b"\x11\x22", cpol=0, cpha=0)

        sample_rate = 1000.0
        packets = list(decoder.decode(**signals, sample_rate=sample_rate))

        assert len(packets) == 2
        # Second packet should have later timestamp
        assert packets[1].timestamp > packets[0].timestamp


class TestSPIDecoderAnnotations:
    """Test packet annotations."""

    def test_basic_annotations(self):
        """Test basic annotation fields."""
        decoder = SPIDecoder(cpol=0, cpha=0, word_size=8)
        signals = generate_spi_signals(b"\xab", cpol=0, cpha=0)

        packets = list(decoder.decode(**signals, sample_rate=1000.0))

        assert len(packets) == 1
        annotations = packets[0].annotations

        assert "word_num" in annotations
        assert "mosi_bits" in annotations
        assert "mosi_value" in annotations
        assert "word_size" in annotations
        assert "mode" in annotations

        assert annotations["word_num"] == 0
        assert annotations["mosi_value"] == 0xAB
        assert annotations["word_size"] == 8

    def test_mosi_bits_annotation(self):
        """Test MOSI bits annotation."""
        decoder = SPIDecoder(cpol=0, cpha=0, word_size=8)
        signals = generate_spi_signals(b"\xa5", cpol=0, cpha=0)

        packets = list(decoder.decode(**signals, sample_rate=1000.0))

        mosi_bits = packets[0].annotations["mosi_bits"]

        assert len(mosi_bits) == 8
        # 0xA5 = 10100101
        assert mosi_bits == [1, 0, 1, 0, 0, 1, 0, 1]

    def test_mode_annotation(self):
        """Test SPI mode annotation."""
        test_cases = [
            (0, 0, 0),  # Mode 0
            (0, 1, 1),  # Mode 1
            (1, 0, 2),  # Mode 2
            (1, 1, 3),  # Mode 3
        ]

        for cpol, cpha, expected_mode in test_cases:
            decoder = SPIDecoder(cpol=cpol, cpha=cpha)
            signals = generate_spi_signals(b"\xff", cpol=cpol, cpha=cpha)

            packets = list(decoder.decode(**signals, sample_rate=1000.0))

            assert packets[0].annotations["mode"] == expected_mode

    def test_word_num_increments(self):
        """Test word_num increments for multiple words."""
        decoder = SPIDecoder(cpol=0, cpha=0, word_size=8)
        signals = generate_spi_signals(b"\x11\x22\x33", cpol=0, cpha=0)

        packets = list(decoder.decode(**signals, sample_rate=1000.0))

        assert len(packets) == 3
        for i, packet in enumerate(packets):
            assert packet.annotations["word_num"] == i


class TestSPIDecoderDataFormats:
    """Test data format conversions."""

    def test_packet_data_field(self):
        """Test packet data field contains bytes."""
        decoder = SPIDecoder(cpol=0, cpha=0, word_size=8)
        signals = generate_spi_signals(b"\x42", cpol=0, cpha=0)

        packets = list(decoder.decode(**signals, sample_rate=1000.0))

        assert len(packets) == 1
        assert isinstance(packets[0].data, bytes)
        assert packets[0].data == b"\x42"

    def test_16bit_data_encoding(self):
        """Test 16-bit word encoding as bytes."""
        decoder = SPIDecoder(cpol=0, cpha=0, word_size=16)
        signals = generate_spi_signals(b"\x12\x34", cpol=0, cpha=0, word_size=16)

        packets = list(decoder.decode(**signals, sample_rate=1000.0))

        # With word_size=16 and 2 bytes input, we get 1 packet (2 bytes = 1 word)
        assert len(packets) == 1
        # 16-bit word encoded as 2 bytes
        assert len(packets[0].data) == 2
        assert packets[0].data == b"\x12\x34"

    def test_32bit_data_encoding(self):
        """Test 32-bit word encoding as bytes."""
        decoder = SPIDecoder(cpol=0, cpha=0, word_size=32)
        signals = generate_spi_signals(b"\xaa\xbb\xcc\xdd", cpol=0, cpha=0, word_size=32)

        packets = list(decoder.decode(**signals, sample_rate=1000.0))

        assert len(packets) == 1
        assert len(packets[0].data) == 4
        assert packets[0].data == b"\xaa\xbb\xcc\xdd"


class TestSPIDecoderProtocolField:
    """Test protocol identification field."""

    def test_protocol_field(self):
        """Test that protocol field is set correctly."""
        decoder = SPIDecoder()
        signals = generate_spi_signals(b"\xff", cpol=0, cpha=0)

        packets = list(decoder.decode(**signals, sample_rate=1000.0))

        assert len(packets) == 1
        assert packets[0].protocol == "spi"


class TestSPIDecoderErrors:
    """Test error handling and error list."""

    def test_no_errors_on_valid_decode(self):
        """Test that errors list is empty for valid decoding."""
        decoder = SPIDecoder(cpol=0, cpha=0)
        signals = generate_spi_signals(b"\xaa", cpol=0, cpha=0)

        packets = list(decoder.decode(**signals, sample_rate=1000.0))

        assert len(packets) == 1
        assert packets[0].errors == []


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestDecodeSPIFunction:
    """Test the decode_spi convenience function."""

    def test_basic_decode(self):
        """Test basic usage of decode_spi."""
        signals = generate_spi_signals(b"\x42", cpol=0, cpha=0)

        packets = decode_spi(
            clk=signals["clk"],
            mosi=signals["mosi"],
            sample_rate=1000.0,
        )

        assert len(packets) == 1
        assert packets[0].annotations["mosi_value"] == 0x42

    def test_with_all_parameters(self):
        """Test decode_spi with all parameters."""
        signals = generate_spi_signals(
            b"\xaa", cpol=1, cpha=1, word_size=8, include_miso=True, include_cs=True
        )

        packets = decode_spi(
            clk=signals["clk"],
            mosi=signals["mosi"],
            miso=signals["miso"],
            cs=signals["cs"],
            sample_rate=10e6,
            cpol=1,
            cpha=1,
            word_size=8,
            bit_order="msb",
        )

        assert len(packets) == 1
        assert packets[0].annotations["mosi_value"] == 0xAA
        assert packets[0].annotations["mode"] == 3

    def test_returns_list(self):
        """Test that decode_spi returns a list."""
        signals = generate_spi_signals(b"\xff", cpol=0, cpha=0)

        result = decode_spi(clk=signals["clk"], mosi=signals["mosi"])

        assert isinstance(result, list)

    def test_multiple_words(self):
        """Test decode_spi with multiple words."""
        signals = generate_spi_signals(b"\x11\x22\x33", cpol=0, cpha=0)

        packets = decode_spi(
            clk=signals["clk"],
            mosi=signals["mosi"],
            sample_rate=1000.0,
        )

        assert len(packets) == 3
        assert packets[0].annotations["mosi_value"] == 0x11
        assert packets[1].annotations["mosi_value"] == 0x22
        assert packets[2].annotations["mosi_value"] == 0x33

    def test_16bit_words_convenience(self):
        """Test decode_spi with 16-bit words."""
        signals = generate_spi_signals(b"\xde\xad", cpol=0, cpha=0, word_size=16)

        packets = decode_spi(
            clk=signals["clk"],
            mosi=signals["mosi"],
            sample_rate=1000.0,
            word_size=16,
        )

        assert len(packets) == 1
        assert packets[0].annotations["mosi_value"] == 0xDEAD


class TestModuleExports:
    """Test module-level exports."""

    def test_all_export(self):
        """Test __all__ contains expected exports."""
        from oscura.analyzers.protocols.spi import __all__

        assert "SPIDecoder" in __all__
        assert "decode_spi" in __all__
        assert len(__all__) == 2


# =============================================================================
# Integration Tests
# =============================================================================


class TestSPIDecoderIntegration:
    """Integration tests for realistic SPI scenarios."""

    def test_spi_flash_read_command(self):
        """Test decoding SPI flash read command sequence."""
        # Typical SPI flash read: CMD (0x03) + ADDR (0x000000) + DATA
        decoder = SPIDecoder(cpol=0, cpha=0, word_size=8)

        # Command byte
        cmd_signals = generate_spi_signals(b"\x03", cpol=0, cpha=0)
        packets = list(decoder.decode(**cmd_signals, sample_rate=1e6))

        assert len(packets) == 1
        assert packets[0].annotations["mosi_value"] == 0x03

    def test_spi_adc_read(self):
        """Test ADC read with MISO data."""
        decoder = SPIDecoder(cpol=0, cpha=0, word_size=16)

        # ADC returns 12-bit value in 16-bit word
        mosi_data = b"\x00\x00"  # Dummy clocks
        miso_data = b"\x0a\xbc"  # ADC reading 0x0ABC
        signals = generate_spi_signals(mosi_data, cpol=0, cpha=0, word_size=16, miso_data=miso_data)

        packets = list(decoder.decode(**signals, sample_rate=1e6))

        assert len(packets) == 1
        assert packets[0].annotations["miso_value"] == 0x0ABC

    def test_spi_display_transaction(self):
        """Test SPI display controller transaction."""
        decoder = SPIDecoder(cpol=0, cpha=0, word_size=8)

        # Display command sequence: CMD + DATA bytes
        data = b"\x2a\x00\x00\x00\xef"  # Set column address
        signals = generate_spi_signals(data, cpol=0, cpha=0)

        packets = list(decoder.decode(**signals, sample_rate=10e6))

        assert len(packets) == 5
        expected_bytes = [0x2A, 0x00, 0x00, 0x00, 0xEF]
        for i, expected in enumerate(expected_bytes):
            assert packets[i].annotations["mosi_value"] == expected

    def test_continuous_streaming(self):
        """Test continuous SPI streaming without CS."""
        decoder = SPIDecoder(cpol=0, cpha=0, word_size=8)

        # Stream of sequential bytes
        data = bytes(range(16))
        signals = generate_spi_signals(data, cpol=0, cpha=0)

        packets = list(decoder.decode(**signals, sample_rate=1e6))

        assert len(packets) == 16
        for i in range(16):
            assert packets[i].annotations["mosi_value"] == i
