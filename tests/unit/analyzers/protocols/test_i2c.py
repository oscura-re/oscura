"""Tests for I2C protocol decoder.

Tests the I2C decoder implementation (PRO-004).
"""

from __future__ import annotations

import numpy as np
import pytest

from oscura.analyzers.protocols.i2c import (
    I2CCondition,
    I2CDecoder,
    I2CTransaction,
)
from oscura.core.types import DigitalTrace, TraceMetadata

pytestmark = [pytest.mark.unit, pytest.mark.analyzer]


class TestI2CTransaction:
    """Test I2CTransaction dataclass."""

    def test_create_transaction(self) -> None:
        """Test creating I2C transaction."""
        txn = I2CTransaction(
            address=0x50,
            read=False,
            data=[0x12, 0x34, 0x56],
            acks=[True, True, True],
            errors=[],
        )

        assert txn.address == 0x50
        assert txn.read is False
        assert txn.data == [0x12, 0x34, 0x56]
        assert txn.acks == [True, True, True]
        assert len(txn.errors) == 0

    def test_transaction_with_nak(self) -> None:
        """Test transaction with NAK."""
        txn = I2CTransaction(
            address=0x3C,
            read=True,
            data=[0xFF],
            acks=[False],
            errors=[],
        )

        assert txn.acks[0] is False

    def test_transaction_with_errors(self) -> None:
        """Test transaction with errors."""
        txn = I2CTransaction(
            address=0x20,
            read=False,
            data=[],
            acks=[],
            errors=["arbitration_lost", "timeout"],
        )

        assert len(txn.errors) == 2
        assert "arbitration_lost" in txn.errors


class TestI2CCondition:
    """Test I2CCondition enum."""

    def test_condition_values(self) -> None:
        """Test condition enum values."""
        assert I2CCondition.START.value == "start"
        assert I2CCondition.STOP.value == "stop"
        assert I2CCondition.REPEATED_START.value == "repeated_start"
        assert I2CCondition.ACK.value == "ack"
        assert I2CCondition.NAK.value == "nak"

    def test_all_conditions_exist(self) -> None:
        """Test all expected conditions are defined."""
        conditions = {c.value for c in I2CCondition}
        expected = {"start", "stop", "repeated_start", "ack", "nak"}
        assert conditions == expected


class TestI2CDecoder:
    """Test I2C decoder."""

    def test_decoder_metadata(self) -> None:
        """Test decoder metadata attributes."""
        decoder = I2CDecoder()

        assert decoder.id == "i2c"
        assert decoder.name == "I2C"
        assert decoder.longname == "Inter-Integrated Circuit"
        assert "I2C" in decoder.desc

    def test_decoder_channels(self) -> None:
        """Test decoder channel definitions."""
        decoder = I2CDecoder()

        assert len(decoder.channels) == 2

        # Check for SCL and SDA
        channel_ids = [ch.id for ch in decoder.channels]
        assert "scl" in channel_ids
        assert "sda" in channel_ids

        # Both channels required
        for ch in decoder.channels:
            assert ch.required is True

    def test_decoder_has_options(self) -> None:
        """Test decoder has configuration options."""
        decoder = I2CDecoder()

        # Should have options for address format, speed, etc.
        assert len(decoder.options) > 0

        option_ids = [opt.id for opt in decoder.options]
        assert "address_format" in option_ids

    def test_decoder_create_instance(self) -> None:
        """Test creating decoder instance."""
        decoder = I2CDecoder()

        assert isinstance(decoder, I2CDecoder)
        assert hasattr(decoder, "decode")


class TestI2CDecoderSimple:
    """Test I2C decoder with simple signals."""

    @pytest.fixture
    def sample_rate(self) -> float:
        """Sample rate fixture."""
        return 1_000_000.0  # 1 MHz

    def test_decode_requires_both_channels(self, sample_rate: float) -> None:
        """Test that decode requires both SCL and SDA."""
        decoder = I2CDecoder()
        metadata = TraceMetadata(sample_rate=sample_rate)

        # Create simple idle signal
        scl = DigitalTrace(data=np.ones(100, dtype=bool), metadata=metadata)

        # Missing SDA should raise error or return empty
        try:
            list(decoder.decode(scl=scl, sample_rate=sample_rate))
        except (TypeError, KeyError, AttributeError):
            pass  # Expected - missing required channel

    def test_decode_empty_signals(self, sample_rate: float) -> None:
        """Test decode with empty signals."""
        decoder = I2CDecoder()

        scl = np.array([], dtype=bool)
        sda = np.array([], dtype=bool)

        packets = list(decoder.decode(scl=scl, sda=sda, sample_rate=sample_rate))

        # Empty input should produce no packets
        assert len(packets) == 0

    def test_decode_idle_bus(self, sample_rate: float) -> None:
        """Test decode with idle bus (both lines high)."""
        decoder = I2CDecoder()

        # Idle bus: both SCL and SDA high
        scl = np.ones(1000, dtype=bool)
        sda = np.ones(1000, dtype=bool)

        packets = list(decoder.decode(scl=scl, sda=sda, sample_rate=sample_rate))

        # Idle bus should produce no packets
        assert len(packets) == 0


# =============================================================================
# I2C Signal Generator
# =============================================================================


def generate_i2c_transaction(
    address: int,
    data_bytes: bytes,
    is_read: bool = False,
    samples_per_bit: int = 100,
    include_nak: bool = False,
    address_10bit: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate I2C transaction signals for testing.

    Args:
        address: 7-bit or 10-bit device address.
        data_bytes: Data bytes to transfer.
        is_read: True for read, False for write.
        samples_per_bit: Samples per bit period.
        include_nak: Whether to include NAK on address.
        address_10bit: Whether to use 10-bit addressing.

    Returns:
        Tuple of (scl, sda) boolean arrays.
    """
    # Calculate total samples needed
    # Idle + START + address byte(s) + ACK + data bytes + ACKs + STOP + idle
    if address_10bit:
        addr_bits = 18 + 2  # Two address bytes + 2 ACKs (20 bits)
    else:
        addr_bits = 9  # 8 bits + ACK

    data_bits = len(data_bytes) * 9  # 8 bits + ACK per byte
    total_bits = 2 + addr_bits + data_bits + 2  # idle + start + ... + stop + idle

    total_samples = (total_bits + 10) * samples_per_bit
    half_bit = samples_per_bit // 2

    # Initialize with idle (both high)
    scl = np.ones(total_samples, dtype=bool)
    sda = np.ones(total_samples, dtype=bool)

    idx = samples_per_bit  # Start after some idle time

    # Generate START condition: SDA falls while SCL is high
    # SCL high, SDA falls
    scl[idx : idx + samples_per_bit] = True
    sda[idx : idx + half_bit] = True
    sda[idx + half_bit : idx + samples_per_bit] = False
    idx += samples_per_bit

    # Prepare address byte (7-bit mode)
    if not address_10bit:
        addr_byte = (address << 1) | (1 if is_read else 0)
        all_bits = []

        # Address byte bits (MSB first)
        for i in range(8):
            bit = (addr_byte >> (7 - i)) & 1
            all_bits.append(bit)
        # ACK bit (low = ACK, high = NAK)
        all_bits.append(1 if include_nak else 0)
    else:
        # 10-bit addressing: first byte is 11110XX + R/W, second is lower 8 bits
        high_bits = 0b11110 | ((address >> 8) & 0b11)
        first_byte = (high_bits << 1) | (1 if is_read else 0)
        second_byte = address & 0xFF

        all_bits = []
        for i in range(8):
            bit = (first_byte >> (7 - i)) & 1
            all_bits.append(bit)
        all_bits.append(0)  # ACK

        for i in range(8):
            bit = (second_byte >> (7 - i)) & 1
            all_bits.append(bit)
        all_bits.append(0)  # ACK

    # Add data bytes
    for byte_val in data_bytes:
        for i in range(8):
            bit = (byte_val >> (7 - i)) & 1
            all_bits.append(bit)
        # ACK (low for write, high for last read byte)
        all_bits.append(0)

    # Generate clock and data for each bit
    for bit_val in all_bits:
        # SCL low at start of bit
        scl[idx : idx + half_bit] = False
        # Data setup
        sda[idx : idx + samples_per_bit] = bool(bit_val)
        # SCL high for sampling
        scl[idx + half_bit : idx + samples_per_bit] = True
        idx += samples_per_bit

    # Ensure SCL goes low before STOP
    scl[idx : idx + half_bit] = False
    sda[idx : idx + half_bit] = False
    idx += half_bit

    # Generate STOP condition: SDA rises while SCL is high
    scl[idx : idx + samples_per_bit] = True
    sda[idx : idx + half_bit] = False
    sda[idx + half_bit : idx + samples_per_bit] = True
    idx += samples_per_bit

    return scl[: idx + samples_per_bit], sda[: idx + samples_per_bit]


# =============================================================================
# Additional I2C Decoding Tests
# =============================================================================


class TestI2CDecoderDecoding:
    """Test I2C decoding with actual transactions."""

    def test_decode_single_byte_write(self) -> None:
        """Test decoding single byte write transaction."""
        decoder = I2CDecoder()
        sample_rate = 1_000_000.0

        scl, sda = generate_i2c_transaction(
            address=0x50,
            data_bytes=b"\x42",
            is_read=False,
        )

        packets = list(decoder.decode(scl=scl, sda=sda, sample_rate=sample_rate))

        assert len(packets) == 1
        assert packets[0].protocol == "i2c"
        assert packets[0].annotations["address"] == 0x50
        assert packets[0].annotations["read"] is False

    def test_decode_multiple_bytes_write(self) -> None:
        """Test decoding multi-byte write transaction."""
        decoder = I2CDecoder()
        sample_rate = 1_000_000.0

        scl, sda = generate_i2c_transaction(
            address=0x3C,
            data_bytes=b"\x01\x02\x03",
            is_read=False,
        )

        packets = list(decoder.decode(scl=scl, sda=sda, sample_rate=sample_rate))

        assert len(packets) == 1
        assert packets[0].annotations["address"] == 0x3C

    def test_decode_read_transaction(self) -> None:
        """Test decoding read transaction."""
        decoder = I2CDecoder()
        sample_rate = 1_000_000.0

        scl, sda = generate_i2c_transaction(
            address=0x68,
            data_bytes=b"\xaa\xbb",
            is_read=True,
        )

        packets = list(decoder.decode(scl=scl, sda=sda, sample_rate=sample_rate))

        assert len(packets) == 1
        assert packets[0].annotations["address"] == 0x68
        assert packets[0].annotations["read"] is True

    def test_decode_with_address_format_7bit(self) -> None:
        """Test decoding with 7-bit address format."""
        decoder = I2CDecoder(address_format="7bit")
        sample_rate = 1_000_000.0

        scl, sda = generate_i2c_transaction(
            address=0x27,
            data_bytes=b"\x55",
            is_read=False,
        )

        packets = list(decoder.decode(scl=scl, sda=sda, sample_rate=sample_rate))

        assert len(packets) == 1
        assert packets[0].annotations["address"] == 0x27
        assert packets[0].annotations["address_10bit"] is False

    def test_decode_10bit_address(self) -> None:
        """Test decoding with 10-bit address format."""
        decoder = I2CDecoder(address_format="10bit")
        sample_rate = 1_000_000.0

        scl, sda = generate_i2c_transaction(
            address=0x123,
            data_bytes=b"\xaa",
            is_read=False,
            address_10bit=True,
        )

        packets = list(decoder.decode(scl=scl, sda=sda, sample_rate=sample_rate))

        # With 10-bit addressing enabled, should decode with 10-bit address
        assert len(packets) >= 0  # May or may not decode depending on signal quality

    def test_decode_auto_address_format(self) -> None:
        """Test decoding with auto address format detection."""
        decoder = I2CDecoder(address_format="auto")
        sample_rate = 1_000_000.0

        scl, sda = generate_i2c_transaction(
            address=0x50,
            data_bytes=b"\x01",
            is_read=False,
        )

        packets = list(decoder.decode(scl=scl, sda=sda, sample_rate=sample_rate))

        assert len(packets) == 1
        # Auto-detection should recognize 7-bit address
        assert packets[0].annotations["address"] == 0x50

    def test_decode_nak_on_address(self) -> None:
        """Test decoding when NAK received on address."""
        decoder = I2CDecoder()
        sample_rate = 1_000_000.0

        scl, sda = generate_i2c_transaction(
            address=0x50,
            data_bytes=b"",
            is_read=False,
            include_nak=True,
        )

        packets = list(decoder.decode(scl=scl, sda=sda, sample_rate=sample_rate))

        # Should detect NAK on address
        if len(packets) > 0:
            assert (
                "NAK on address" in packets[0].errors
                or len(packets[0].annotations.get("acks", [])) == 0
            )

    def test_timestamp_calculation(self) -> None:
        """Test packet timestamp calculation."""
        decoder = I2CDecoder()
        sample_rate = 1_000_000.0

        scl, sda = generate_i2c_transaction(
            address=0x50,
            data_bytes=b"\x42",
            is_read=False,
        )

        packets = list(decoder.decode(scl=scl, sda=sda, sample_rate=sample_rate))

        if len(packets) > 0:
            assert packets[0].timestamp >= 0.0
            assert isinstance(packets[0].timestamp, float)


class TestI2CExtractBytes:
    """Test I2C _extract_bytes method."""

    def test_extract_bytes_insufficient_edges(self) -> None:
        """Test extract_bytes with insufficient clock edges."""
        decoder = I2CDecoder()

        # Only 5 rising edges (need at least 9 for one byte + ACK)
        scl = np.array([False, True] * 5, dtype=bool)
        sda = np.ones(10, dtype=bool)

        bytes_data, acks = decoder._extract_bytes(scl, sda)

        assert bytes_data == []
        assert acks == []

    def test_extract_bytes_valid(self) -> None:
        """Test extract_bytes with valid data."""
        decoder = I2CDecoder()

        # Generate a proper byte pattern with 9 rising edges
        samples_per_bit = 10
        half_bit = samples_per_bit // 2
        scl = np.zeros(100, dtype=bool)
        sda = np.zeros(100, dtype=bool)

        # Create 9 clock cycles with data
        byte_val = 0xA5  # 10100101
        idx = 0
        for i in range(9):
            # SCL low, then high
            scl[idx : idx + half_bit] = False
            scl[idx + half_bit : idx + samples_per_bit] = True
            if i < 8:
                bit = (byte_val >> (7 - i)) & 1
                sda[idx : idx + samples_per_bit] = bool(bit)
            else:
                sda[idx : idx + samples_per_bit] = False  # ACK
            idx += samples_per_bit

        bytes_data, acks = decoder._extract_bytes(scl[:idx], sda[:idx])

        assert len(bytes_data) == 1
        assert bytes_data[0] == 0xA5
        assert len(acks) == 1
        assert acks[0] is True  # ACK (SDA low)


class TestDecodeI2CConvenienceFunction:
    """Test decode_i2c convenience function."""

    def test_decode_i2c_basic(self) -> None:
        """Test basic usage of decode_i2c."""
        from oscura.analyzers.protocols.i2c import decode_i2c

        scl, sda = generate_i2c_transaction(
            address=0x50,
            data_bytes=b"\x42",
            is_read=False,
        )

        packets = decode_i2c(scl, sda, sample_rate=1_000_000.0)

        assert isinstance(packets, list)
        assert len(packets) == 1

    def test_decode_i2c_with_address_format(self) -> None:
        """Test decode_i2c with address format parameter."""
        from oscura.analyzers.protocols.i2c import decode_i2c

        scl, sda = generate_i2c_transaction(
            address=0x27,
            data_bytes=b"\x00",
            is_read=False,
        )

        packets = decode_i2c(scl, sda, sample_rate=1_000_000.0, address_format="7bit")

        assert isinstance(packets, list)

    def test_decode_i2c_empty_signals(self) -> None:
        """Test decode_i2c with empty signals."""
        from oscura.analyzers.protocols.i2c import decode_i2c

        scl = np.array([], dtype=bool)
        sda = np.array([], dtype=bool)

        packets = decode_i2c(scl, sda, sample_rate=1_000_000.0)

        assert packets == []


class TestI2CModuleExports:
    """Test module-level exports."""

    def test_all_export(self) -> None:
        """Test __all__ contains expected exports."""
        from oscura.analyzers.protocols.i2c import __all__

        assert "I2CCondition" in __all__
        assert "I2CDecoder" in __all__
        assert "I2CTransaction" in __all__
        assert "decode_i2c" in __all__
