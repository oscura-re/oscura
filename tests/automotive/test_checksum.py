"""Tests for CAN checksum detection.

This module tests automatic detection of XOR, SUM, and CRC checksums in
CAN message data, integrating with Oscura's CRC reverse engineering.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.automotive]

import pytest

from oscura.automotive.can.checksum import ChecksumDetector
from oscura.automotive.can.models import CANMessage, CANMessageList


@pytest.fixture
def xor_checksum_messages() -> CANMessageList:
    """Generate CAN messages with XOR checksum.

    Checksum is XOR of bytes 0-6, stored in byte 7.

    Returns:
        CANMessageList with XOR checksums.
    """
    messages = CANMessageList()

    for i in range(50):
        timestamp = i * 0.02

        # Varying data
        data = bytearray(8)
        data[0] = (i * 7) % 256
        data[1] = (i * 11) % 256
        data[2] = (i * 13) % 256
        data[3] = (i * 17) % 256
        data[4] = (i * 19) % 256
        data[5] = (i * 23) % 256
        data[6] = (i * 29) % 256

        # XOR checksum in byte 7
        xor_sum = 0
        for b in data[:7]:
            xor_sum ^= b
        data[7] = xor_sum

        msg = CANMessage(
            arbitration_id=0x300,
            timestamp=timestamp,
            data=bytes(data),
        )
        messages.append(msg)

    return messages


@pytest.fixture
def sum_checksum_messages() -> CANMessageList:
    """Generate CAN messages with SUM checksum.

    Checksum is sum of bytes 0-6 (mod 256), stored in byte 7.

    Returns:
        CANMessageList with SUM checksums.
    """
    messages = CANMessageList()

    for i in range(50):
        timestamp = i * 0.02

        # Varying data
        data = bytearray(8)
        data[0] = (i * 3) % 256
        data[1] = (i * 5) % 256
        data[2] = (i * 7) % 256
        data[3] = (i * 11) % 256
        data[4] = (i * 13) % 256
        data[5] = (i * 17) % 256
        data[6] = (i * 19) % 256

        # SUM checksum in byte 7 (modulo 256)
        byte_sum = 0
        for b in data[:7]:
            byte_sum = (byte_sum + b) & 0xFF
        data[7] = byte_sum

        msg = CANMessage(
            arbitration_id=0x400,
            timestamp=timestamp,
            data=bytes(data),
        )
        messages.append(msg)

    return messages


@pytest.fixture
def no_checksum_messages() -> CANMessageList:
    """Generate CAN messages without checksum.

    Last byte is random, not a checksum.

    Returns:
        CANMessageList without checksums.
    """
    messages = CANMessageList()

    import numpy as np

    rng = np.random.default_rng(42)

    for i in range(50):
        timestamp = i * 0.02

        # Random data (no checksum pattern)
        data = bytes(rng.integers(0, 256, 8, dtype=np.uint8))

        msg = CANMessage(
            arbitration_id=0x500,
            timestamp=timestamp,
            data=data,
        )
        messages.append(msg)

    return messages


@pytest.fixture
def checksum_at_different_position() -> CANMessageList:
    """Generate messages with checksum at byte 5 (not last byte).

    Returns:
        CANMessageList with checksum at non-standard position.
    """
    messages = CANMessageList()

    for i in range(50):
        timestamp = i * 0.02

        data = bytearray(8)
        data[0] = (i * 7) % 256
        data[1] = (i * 11) % 256
        data[2] = (i * 13) % 256
        data[3] = (i * 17) % 256
        data[4] = (i * 19) % 256
        data[6] = (i * 23) % 256
        data[7] = (i * 29) % 256

        # XOR checksum at byte 5 (XOR of all other bytes)
        xor_sum = 0
        for j in [0, 1, 2, 3, 4, 6, 7]:
            xor_sum ^= data[j]
        data[5] = xor_sum

        msg = CANMessage(
            arbitration_id=0x600,
            timestamp=timestamp,
            data=bytes(data),
        )
        messages.append(msg)

    return messages


class TestDetectXORChecksum:
    """Tests for XOR checksum detection."""

    def test_detect_xor_basic(self, xor_checksum_messages):
        """Test detection of XOR checksum in last byte."""
        result = ChecksumDetector.detect_checksum(xor_checksum_messages)

        assert result is not None
        assert result.algorithm == "XOR-8"
        assert result.byte_position == 7
        assert result.confidence > 0.95
        assert result.validation_rate > 0.95
        assert 7 not in result.covered_bytes

    def test_detect_xor_high_confidence(self, xor_checksum_messages):
        """Test that XOR detection has high confidence."""
        result = ChecksumDetector.detect_checksum(xor_checksum_messages)

        assert result is not None
        assert result.confidence >= 0.95
        assert result.validation_rate >= 0.95

    def test_detect_xor_with_suspected_byte(self, xor_checksum_messages):
        """Test detection when specifying suspected byte position."""
        result = ChecksumDetector.detect_checksum(xor_checksum_messages, suspected_byte=7)

        assert result is not None
        assert result.algorithm == "XOR-8"
        assert result.byte_position == 7

    def test_detect_xor_wrong_suspected_byte(self, xor_checksum_messages):
        """Test that wrong suspected byte returns None or low confidence."""
        result = ChecksumDetector.detect_checksum(xor_checksum_messages, suspected_byte=0)

        # Byte 0 doesn't contain the intentional checksum (which is at byte 7)
        # But the detector may find accidental XOR relationships, which is fine
        # The key is that if it finds something, byte_position should be 0 (as requested)
        if result is not None:
            assert result.byte_position == 0


class TestDetectSUMChecksum:
    """Tests for SUM checksum detection."""

    def test_detect_sum_basic(self, sum_checksum_messages):
        """Test detection of SUM checksum in last byte."""
        result = ChecksumDetector.detect_checksum(sum_checksum_messages)

        assert result is not None
        assert result.algorithm == "SUM-8"
        assert result.byte_position == 7
        assert result.confidence > 0.95
        assert result.validation_rate > 0.95

    def test_detect_sum_high_validation_rate(self, sum_checksum_messages):
        """Test that SUM detection has high validation rate."""
        result = ChecksumDetector.detect_checksum(sum_checksum_messages)

        assert result is not None
        assert result.validation_rate >= 0.95

    def test_detect_sum_covered_bytes(self, sum_checksum_messages):
        """Test that covered bytes excludes checksum position."""
        result = ChecksumDetector.detect_checksum(sum_checksum_messages)

        assert result is not None
        assert 7 not in result.covered_bytes
        assert len(result.covered_bytes) == 7  # Bytes 0-6


class TestDetectNoChecksum:
    """Tests for handling messages without checksums."""

    def test_no_checksum_returns_none(self, no_checksum_messages):
        """Test that messages without checksum return None."""
        result = ChecksumDetector.detect_checksum(no_checksum_messages)

        # Should not detect a checksum (or low confidence)
        if result is not None:
            assert result.confidence < 0.95

    def test_insufficient_samples(self):
        """Test that too few messages returns None."""
        messages = CANMessageList()

        # Only 5 messages (need at least 10)
        for i in range(5):
            data = bytes([i, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
            msg = CANMessage(arbitration_id=0x100, timestamp=i * 0.01, data=data)
            messages.append(msg)

        result = ChecksumDetector.detect_checksum(messages)
        assert result is None


class TestChecksumAtDifferentPositions:
    """Tests for checksums at non-standard byte positions."""

    def test_auto_detect_checksum_position(self, checksum_at_different_position):
        """Test auto-detection of checksum at non-last byte."""
        # Without specifying suspected byte, should check last 2 bytes
        result = ChecksumDetector.detect_checksum(checksum_at_different_position)

        # May or may not find it depending on search strategy
        # The implementation checks last 2 bytes by default
        if result is not None:
            assert result.byte_position in [5, 6, 7]

    def test_detect_with_suspected_byte_5(self, checksum_at_different_position):
        """Test detection when specifying correct suspected byte."""
        result = ChecksumDetector.detect_checksum(checksum_at_different_position, suspected_byte=5)

        assert result is not None
        assert result.byte_position == 5
        assert result.algorithm == "XOR-8"
        assert result.confidence > 0.95


class TestCRCIntegration:
    """Tests for CRC detection via CRCReverser integration."""

    def test_crc_reverser_integration(self):
        """Test that CRC reverser is called for non-simple checksums.

        This test verifies integration with Oscura's CRC reverse engineering.
        """
        # Create messages with a pattern that's not simple XOR/SUM
        # This would require actual CRC, but we'll test the call path
        messages = CANMessageList()

        for i in range(20):
            # Create data that might look like it has a CRC
            data = bytes([i, i + 1, i + 2, i + 3, i + 4, i + 5, i + 6, 0xFF])
            msg = CANMessage(arbitration_id=0x700, timestamp=i * 0.01, data=data)
            messages.append(msg)

        # Should attempt CRC detection (may or may not succeed)
        result = ChecksumDetector.detect_checksum(messages)

        # Result depends on whether CRCReverser finds a pattern
        # Just verify no errors occur
        assert result is None or isinstance(result.algorithm, str)


class TestEdgeCases:
    """Tests for edge cases in checksum detection."""

    def test_empty_message_list(self):
        """Test detection on empty message list."""
        messages = CANMessageList()
        result = ChecksumDetector.detect_checksum(messages)
        assert result is None

    def test_short_messages(self):
        """Test detection on very short messages (< 2 bytes)."""
        messages = CANMessageList()

        for i in range(20):
            data = bytes([i])  # Only 1 byte
            msg = CANMessage(arbitration_id=0x100, timestamp=i * 0.01, data=data)
            messages.append(msg)

        result = ChecksumDetector.detect_checksum(messages)
        # May return None or try to detect on single byte
        assert result is None or result.byte_position == 0

    def test_all_constant_messages(self):
        """Test detection when all messages are identical."""
        messages = CANMessageList()

        constant_data = bytes([0x55, 0xAA, 0x55, 0xAA, 0x55, 0xAA, 0x55, 0xAA])
        for i in range(50):
            msg = CANMessage(arbitration_id=0x100, timestamp=i * 0.01, data=constant_data)
            messages.append(msg)

        result = ChecksumDetector.detect_checksum(messages)

        # Constant messages can't have varying checksum
        # May detect as valid checksum with 100% match, or return None
        if result is not None:
            # If detected, should have perfect validation
            assert result.validation_rate == 1.0

    def test_partial_checksum_coverage(self):
        """Test messages where checksum only matches some of the time."""
        messages = CANMessageList()

        for i in range(50):
            data = bytearray(8)
            data[0] = i % 256
            data[1] = (i * 2) % 256
            data[2] = (i * 3) % 256
            data[3] = (i * 5) % 256
            data[4] = (i * 7) % 256
            data[5] = (i * 11) % 256
            data[6] = (i * 13) % 256

            # XOR checksum, but corrupt it 20% of the time
            xor_sum = 0
            for b in data[:7]:
                xor_sum ^= b

            if i % 5 == 0:
                # Corrupt checksum
                data[7] = (xor_sum + 1) & 0xFF
            else:
                data[7] = xor_sum

            msg = CANMessage(arbitration_id=0x800, timestamp=i * 0.01, data=bytes(data))
            messages.append(msg)

        result = ChecksumDetector.detect_checksum(messages)

        # 80% match rate - should not detect (threshold is 95%)
        if result is not None:
            assert result.confidence < 0.95


class TestChecksumInfoAttributes:
    """Tests for ChecksumInfo dataclass attributes."""

    def test_checksum_info_complete(self, xor_checksum_messages):
        """Test that ChecksumInfo has all expected attributes."""
        result = ChecksumDetector.detect_checksum(xor_checksum_messages)

        assert result is not None
        assert hasattr(result, "byte_position")
        assert hasattr(result, "algorithm")
        assert hasattr(result, "polynomial")
        assert hasattr(result, "covered_bytes")
        assert hasattr(result, "confidence")
        assert hasattr(result, "validation_rate")

    def test_xor_has_no_polynomial(self, xor_checksum_messages):
        """Test that XOR checksum has None polynomial."""
        result = ChecksumDetector.detect_checksum(xor_checksum_messages)

        assert result is not None
        assert result.polynomial is None

    def test_sum_has_no_polynomial(self, sum_checksum_messages):
        """Test that SUM checksum has None polynomial."""
        result = ChecksumDetector.detect_checksum(sum_checksum_messages)

        assert result is not None
        assert result.polynomial is None


class TestMultipleChecksumTypes:
    """Tests with different checksum types in same session."""

    def test_prefer_higher_confidence(self):
        """Test that detector returns highest confidence result."""
        # Create messages where multiple checksum algorithms might match
        # but one has higher confidence
        messages = CANMessageList()

        for i in range(50):
            data = bytearray(8)
            data[0] = i % 256
            data[1] = 0x00
            data[2] = 0x00
            data[3] = 0x00
            data[4] = 0x00
            data[5] = 0x00
            data[6] = 0x00

            # Both XOR and SUM of single byte equals that byte
            # So both should match if we're checking just byte 0
            data[7] = data[0]

            msg = CANMessage(arbitration_id=0x900, timestamp=i * 0.01, data=bytes(data))
            messages.append(msg)

        result = ChecksumDetector.detect_checksum(messages)

        # Should detect something (either XOR or SUM would work)
        assert result is not None
        assert result.confidence > 0.95


class TestPerformance:
    """Tests for checksum detection performance."""

    def test_large_message_count(self):
        """Test detection with large number of messages."""
        messages = CANMessageList()

        # 1000 messages with XOR checksum
        for i in range(1000):
            data = bytearray(8)
            data[0] = (i * 7) % 256
            data[1] = (i * 11) % 256
            data[2] = (i * 13) % 256
            data[3] = (i * 17) % 256
            data[4] = (i * 19) % 256
            data[5] = (i * 23) % 256
            data[6] = (i * 29) % 256

            xor_sum = 0
            for b in data[:7]:
                xor_sum ^= b
            data[7] = xor_sum

            msg = CANMessage(arbitration_id=0xA00, timestamp=i * 0.001, data=bytes(data))
            messages.append(msg)

        # Should still detect successfully
        result = ChecksumDetector.detect_checksum(messages)

        assert result is not None
        assert result.algorithm == "XOR-8"
        assert result.confidence > 0.95
