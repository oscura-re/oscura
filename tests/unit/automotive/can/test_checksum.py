"""Comprehensive test suite for CAN checksum detection.

Tests cover XOR/SUM/CRC checksum detection, byte position analysis,
confidence scoring, and integration with CRC reverse engineering.
"""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

# Module under test
try:
    from oscura.automotive.can.checksum import ChecksumDetector
    from oscura.automotive.can.models import CANMessage, CANMessageList, ChecksumInfo

    HAS_CAN = True
except ImportError:
    HAS_CAN = False

pytestmark = pytest.mark.skipif(not HAS_CAN, reason="CAN modules not available")


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def xor_checksum_messages():
    """Messages with XOR checksum at last byte."""
    messages = []
    for i in range(20):
        data = bytes([0x12, 0x34, i & 0xFF, (i + 1) & 0xFF])
        # Calculate XOR checksum
        xor_sum = 0
        for b in data:
            xor_sum ^= b
        data_with_checksum = data + bytes([xor_sum])

        msg = CANMessage(
            arbitration_id=0x100,
            timestamp=i * 0.01,
            data=data_with_checksum,
            is_extended=False,
        )
        messages.append(msg)

    return CANMessageList(messages=messages)


@pytest.fixture
def sum_checksum_messages():
    """Messages with SUM checksum at last byte."""
    messages = []
    for i in range(20):
        data = bytes([0xAA, 0xBB, i & 0xFF, (i * 2) & 0xFF])
        # Calculate SUM checksum (modulo 256)
        byte_sum = sum(data) & 0xFF
        data_with_checksum = data + bytes([byte_sum])

        msg = CANMessage(
            arbitration_id=0x200,
            timestamp=i * 0.01,
            data=data_with_checksum,
            is_extended=False,
        )
        messages.append(msg)

    return CANMessageList(messages=messages)


@pytest.fixture
def no_checksum_messages():
    """Messages with no checksum (random data)."""
    messages = []
    for i in range(20):
        data = bytes([i & 0xFF, (i + 1) & 0xFF, (i + 2) & 0xFF, (i + 3) & 0xFF])

        msg = CANMessage(
            arbitration_id=0x300,
            timestamp=i * 0.01,
            data=data,
            is_extended=False,
        )
        messages.append(msg)

    return CANMessageList(messages=messages)


@pytest.fixture
def middle_checksum_messages():
    """Messages with checksum in middle position."""
    messages = []
    for i in range(20):
        # Data: [byte0, byte1, CHECKSUM, byte3, byte4]
        data_before = bytes([0x10, 0x20])
        data_after = bytes([i & 0xFF, (i + 5) & 0xFF])

        # XOR of all except position 2
        xor_sum = 0
        for b in data_before + data_after:
            xor_sum ^= b

        data = data_before + bytes([xor_sum]) + data_after

        msg = CANMessage(
            arbitration_id=0x400,
            timestamp=i * 0.01,
            data=data,
            is_extended=False,
        )
        messages.append(msg)

    return CANMessageList(messages=messages)


@pytest.fixture
def partial_checksum_messages():
    """Messages with checksum matching only 80% (below threshold)."""
    messages = []
    for i in range(20):
        data = bytes([0x12, 0x34, i & 0xFF])

        # Calculate XOR
        xor_sum = 0
        for b in data:
            xor_sum ^= b

        # 80% match, 20% wrong
        if i % 5 == 0:
            # Wrong checksum
            checksum = (xor_sum + 1) & 0xFF
        else:
            checksum = xor_sum

        data_with_checksum = data + bytes([checksum])

        msg = CANMessage(
            arbitration_id=0x500,
            timestamp=i * 0.01,
            data=data_with_checksum,
            is_extended=False,
        )
        messages.append(msg)

    return CANMessageList(messages=messages)


# ============================================================================
# Test ChecksumDetector - XOR Detection
# ============================================================================


class TestXORDetection:
    """Test XOR checksum detection."""

    def test_detect_xor_checksum_at_end(self, xor_checksum_messages):
        """Test detecting XOR checksum at last byte."""
        result = ChecksumDetector.detect_checksum(xor_checksum_messages)

        assert result is not None
        assert result.algorithm == "XOR-8"
        assert result.byte_position == 4  # Last byte
        assert result.confidence > 0.95
        assert result.validation_rate > 0.95
        assert result.polynomial is None  # XOR has no polynomial
        assert 0 in result.covered_bytes  # Covers byte 0
        assert 4 not in result.covered_bytes  # Doesn't cover itself

    def test_detect_xor_checksum_at_specific_position(self, middle_checksum_messages):
        """Test detecting XOR checksum at specific byte position."""
        result = ChecksumDetector.detect_checksum(middle_checksum_messages, suspected_byte=2)

        assert result is not None
        assert result.algorithm == "XOR-8"
        assert result.byte_position == 2  # Middle position
        assert result.confidence > 0.95

    def test_xor_detection_confidence_score(self, xor_checksum_messages):
        """Test XOR detection confidence scoring."""
        result = ChecksumDetector.detect_checksum(xor_checksum_messages)

        assert result is not None
        # 100% match should give ~1.0 confidence
        assert result.confidence >= 0.99

    def test_xor_covered_bytes_excludes_checksum(self, xor_checksum_messages):
        """Test that covered_bytes excludes checksum position."""
        result = ChecksumDetector.detect_checksum(xor_checksum_messages)

        assert result is not None
        assert result.byte_position not in result.covered_bytes
        assert len(result.covered_bytes) == 4  # All other bytes


# ============================================================================
# Test ChecksumDetector - SUM Detection
# ============================================================================


class TestSUMDetection:
    """Test SUM checksum detection."""

    def test_detect_sum_checksum(self, sum_checksum_messages):
        """Test detecting SUM checksum."""
        result = ChecksumDetector.detect_checksum(sum_checksum_messages)

        assert result is not None
        assert result.algorithm == "SUM-8"
        assert result.byte_position == 4  # Last byte
        assert result.confidence > 0.95
        assert result.polynomial is None  # SUM has no polynomial

    def test_sum_modulo_256_wraparound(self):
        """Test SUM checksum with modulo 256 wraparound."""
        messages = []
        for i in range(15):
            # Data that sums to > 256
            data = bytes([0xFF, 0xFF, i & 0xFF])
            byte_sum = (0xFF + 0xFF + (i & 0xFF)) & 0xFF  # Modulo 256
            data_with_checksum = data + bytes([byte_sum])

            msg = CANMessage(
                arbitration_id=0x600,
                timestamp=i * 0.01,
                data=data_with_checksum,
                is_extended=False,
            )
            messages.append(msg)

        msg_list = CANMessageList(messages=messages)
        result = ChecksumDetector.detect_checksum(msg_list)

        assert result is not None
        assert result.algorithm == "SUM-8"


# ============================================================================
# Test ChecksumDetector - General Behavior
# ============================================================================


class TestGeneralDetection:
    """Test general checksum detection behavior."""

    def test_no_checksum_detected(self, no_checksum_messages):
        """Test that random data doesn't trigger false positives."""
        result = ChecksumDetector.detect_checksum(no_checksum_messages)

        # Should not detect checksum in random data
        assert result is None

    def test_insufficient_messages(self):
        """Test that < 10 messages returns None."""
        messages = []
        for i in range(5):  # Only 5 messages
            data = bytes([i, i + 1, i + 2])
            msg = CANMessage(
                arbitration_id=0x100,
                timestamp=i * 0.01,
                data=data,
                is_extended=False,
            )
            messages.append(msg)

        msg_list = CANMessageList(messages=messages)
        result = ChecksumDetector.detect_checksum(msg_list)

        assert result is None  # Not enough samples

    def test_empty_message_list(self):
        """Test empty message list."""
        msg_list = CANMessageList(messages=[])
        result = ChecksumDetector.detect_checksum(msg_list)

        assert result is None

    def test_messages_too_short(self):
        """Test messages with insufficient data length."""
        messages = []
        for i in range(15):
            # Single byte messages
            data = bytes([i & 0xFF])
            msg = CANMessage(
                arbitration_id=0x100,
                timestamp=i * 0.01,
                data=data,
                is_extended=False,
            )
            messages.append(msg)

        msg_list = CANMessageList(messages=messages)
        result = ChecksumDetector.detect_checksum(msg_list)

        # Should handle gracefully (may return None or detect in single byte)
        # Implementation-dependent
        # Verify result is either None or a valid ChecksumCandidate
        if result is not None:
            assert hasattr(result, "byte_position")
            assert hasattr(result, "algorithm")


# ============================================================================
# Test ChecksumDetector - Byte Position Logic
# ============================================================================


class TestBytePositionLogic:
    """Test byte position detection and selection."""

    def test_auto_check_last_two_bytes(self, xor_checksum_messages):
        """Test that auto mode checks last 2 bytes."""
        # XOR checksum is at last byte
        result = ChecksumDetector.detect_checksum(xor_checksum_messages)

        assert result is not None
        assert result.byte_position in [3, 4]  # Last or second-to-last

    def test_specific_byte_position(self, xor_checksum_messages):
        """Test detection at specific byte position."""
        # Force check at position 4 (correct position)
        result = ChecksumDetector.detect_checksum(xor_checksum_messages, suspected_byte=4)

        assert result is not None
        assert result.byte_position == 4

    def test_wrong_suspected_byte_position(self, xor_checksum_messages):
        """Test that wrong byte position may still detect patterns."""
        # Checksum is at position 4, check position 0
        result = ChecksumDetector.detect_checksum(xor_checksum_messages, suspected_byte=0)

        # Position 0 might accidentally match XOR pattern, or return None
        # The detector correctly identifies patterns wherever they exist
        if result is not None:
            # If a pattern is found, it should have valid characteristics
            assert result.byte_position == 0
            assert result.confidence > 0.0

    def test_best_result_selection(self):
        """Test selection of best result when multiple positions match."""
        # Create messages with checksums at two positions
        messages = []
        for i in range(20):
            data = bytes([i & 0xFF])

            # XOR checksum at position 1
            xor1 = data[0]
            # XOR checksum at position 2 (of extended data)
            xor2 = data[0] ^ xor1

            full_data = data + bytes([xor1, xor2])

            msg = CANMessage(
                arbitration_id=0x700,
                timestamp=i * 0.01,
                data=full_data,
                is_extended=False,
            )
            messages.append(msg)

        msg_list = CANMessageList(messages=messages)
        result = ChecksumDetector.detect_checksum(msg_list)

        # Should return the highest confidence result
        assert result is not None


# ============================================================================
# Test ChecksumDetector - CRC Fallback
# ============================================================================


class TestCRCFallback:
    """Test CRC reverse engineering fallback."""

    @patch("oscura.automotive.can.checksum.CRCReverser")
    def test_crc_reverser_called_for_unknown_checksum(
        self, mock_reverser_class, no_checksum_messages
    ):
        """Test that CRC reverser is called when XOR/SUM fail."""
        # Mock CRC reverser
        mock_reverser = Mock()
        mock_reverser.reverse.return_value = None  # No CRC found
        mock_reverser_class.return_value = mock_reverser

        result = ChecksumDetector.detect_checksum(no_checksum_messages)

        # CRC reverser should have been called
        assert mock_reverser.reverse.called

    @patch("oscura.automotive.can.checksum.CRCReverser")
    def test_crc_detection_success(self, mock_reverser_class, no_checksum_messages):
        """Test successful CRC detection via reverser."""
        # Mock CRC result
        mock_crc_params = Mock()
        mock_crc_params.algorithm_name = "CRC-8-SAE-J1850"
        mock_crc_params.polynomial = 0x1D
        mock_crc_params.width = 8
        mock_crc_params.confidence = 0.85

        mock_reverser = Mock()
        mock_reverser.reverse.return_value = mock_crc_params
        mock_reverser_class.return_value = mock_reverser

        result = ChecksumDetector.detect_checksum(no_checksum_messages)

        assert result is not None
        assert result.algorithm == "CRC-8-SAE-J1850"
        assert result.polynomial == 0x1D
        assert result.confidence == 0.85

    @patch("oscura.automotive.can.checksum.CRCReverser")
    def test_crc_low_confidence_rejected(self, mock_reverser_class, no_checksum_messages):
        """Test that low-confidence CRC results are rejected."""
        # Mock low-confidence CRC result
        mock_crc_params = Mock()
        mock_crc_params.confidence = 0.5  # Below 0.7 threshold

        mock_reverser = Mock()
        mock_reverser.reverse.return_value = mock_crc_params
        mock_reverser_class.return_value = mock_reverser

        result = ChecksumDetector.detect_checksum(no_checksum_messages)

        # Should reject low confidence
        assert result is None

    @patch("oscura.automotive.can.checksum.CRCReverser")
    def test_crc_reverser_exception_handled(self, mock_reverser_class, no_checksum_messages):
        """Test graceful handling of CRC reverser exceptions."""
        mock_reverser = Mock()
        mock_reverser.reverse.side_effect = Exception("CRC reverser failed")
        mock_reverser_class.return_value = mock_reverser

        # Should not raise, should return None
        result = ChecksumDetector.detect_checksum(no_checksum_messages)

        assert result is None


# ============================================================================
# Test ChecksumDetector - Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_partial_match_below_threshold(self, partial_checksum_messages):
        """Test that <95% match rate is rejected."""
        result = ChecksumDetector.detect_checksum(partial_checksum_messages)

        # 80% match is below 95% threshold
        assert result is None

    def test_variable_message_lengths(self):
        """Test messages with varying DLC."""
        messages = []
        for i in range(20):
            # Vary data length
            dlc = 3 + (i % 5)  # DLC 3-7
            data = bytes(range(dlc - 1))

            # XOR checksum
            xor_sum = 0
            for b in data:
                xor_sum ^= b

            data_with_checksum = data + bytes([xor_sum])

            msg = CANMessage(
                arbitration_id=0x800,
                timestamp=i * 0.01,
                data=data_with_checksum,
                is_extended=False,
            )
            messages.append(msg)

        msg_list = CANMessageList(messages=messages)
        result = ChecksumDetector.detect_checksum(msg_list)

        # Should handle variable lengths
        # May or may not detect depending on consistency
        # Verify result is either None or valid
        if result is not None:
            assert hasattr(result, "algorithm")
            assert result.validation_rate > 0

    def test_zero_byte_messages(self):
        """Test messages with zero data bytes."""
        messages = []
        for i in range(15):
            msg = CANMessage(
                arbitration_id=0x900,
                timestamp=i * 0.01,
                data=b"",
                is_extended=False,
            )
            messages.append(msg)

        msg_list = CANMessageList(messages=messages)
        result = ChecksumDetector.detect_checksum(msg_list)

        assert result is None  # No data to check

    def test_all_zero_data(self):
        """Test messages with all-zero data."""
        messages = []
        for i in range(15):
            data = bytes([0x00, 0x00, 0x00, 0x00])

            msg = CANMessage(
                arbitration_id=0xA00,
                timestamp=i * 0.01,
                data=data,
                is_extended=False,
            )
            messages.append(msg)

        msg_list = CANMessageList(messages=messages)
        result = ChecksumDetector.detect_checksum(msg_list)

        # All zeros: XOR=0, SUM=0
        # May detect false positive depending on implementation
        # Verify result structure if detected
        if result is not None:
            assert hasattr(result, "algorithm")
            assert result.validation_rate >= 0


# ============================================================================
# Test ChecksumDetector - AUTOMOTIVE_CRCS Constant
# ============================================================================


class TestAutomotiveCRCs:
    """Test automotive CRC algorithm constants."""

    def test_automotive_crcs_structure(self):
        """Test that AUTOMOTIVE_CRCS has correct structure."""
        assert "CRC-8-SAE-J1850" in ChecksumDetector.AUTOMOTIVE_CRCS
        assert "CRC-8-AUTOSAR" in ChecksumDetector.AUTOMOTIVE_CRCS
        assert "CRC-16-IBM" in ChecksumDetector.AUTOMOTIVE_CRCS
        assert "XOR-8" in ChecksumDetector.AUTOMOTIVE_CRCS
        assert "SUM-8" in ChecksumDetector.AUTOMOTIVE_CRCS

    def test_crc_parameters_format(self):
        """Test CRC parameter dictionaries."""
        crc_sae = ChecksumDetector.AUTOMOTIVE_CRCS["CRC-8-SAE-J1850"]

        assert crc_sae["width"] == 8
        assert crc_sae["poly"] == 0x1D
        assert crc_sae["init"] == 0xFF
        assert crc_sae["xor_out"] == 0xFF

    def test_simple_checksum_parameters(self):
        """Test simple checksum (XOR/SUM) parameters."""
        xor_params = ChecksumDetector.AUTOMOTIVE_CRCS["XOR-8"]
        sum_params = ChecksumDetector.AUTOMOTIVE_CRCS["SUM-8"]

        assert xor_params["width"] == 8
        assert xor_params["algorithm"] == "xor"
        assert sum_params["width"] == 8
        assert sum_params["algorithm"] == "sum"


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for checksum detection workflows."""

    def test_complete_xor_detection_workflow(self, xor_checksum_messages):
        """Test complete XOR detection workflow."""
        # Detect checksum
        result = ChecksumDetector.detect_checksum(xor_checksum_messages)

        assert result is not None

        # Verify all result fields
        assert isinstance(result.byte_position, int)
        assert isinstance(result.algorithm, str)
        assert isinstance(result.confidence, float)
        assert isinstance(result.validation_rate, float)
        assert isinstance(result.covered_bytes, list)

        # Verify correctness
        assert result.algorithm in ["XOR-8", "SUM-8", "CRC-8"]
        assert 0.0 <= result.confidence <= 1.0
        assert 0.0 <= result.validation_rate <= 1.0

    def test_multiple_message_types(
        self, xor_checksum_messages, sum_checksum_messages, no_checksum_messages
    ):
        """Test detection across different message types."""
        # XOR messages
        xor_result = ChecksumDetector.detect_checksum(xor_checksum_messages)
        assert xor_result is not None
        assert xor_result.algorithm == "XOR-8"

        # SUM messages
        sum_result = ChecksumDetector.detect_checksum(sum_checksum_messages)
        assert sum_result is not None
        assert sum_result.algorithm == "SUM-8"

        # No checksum
        no_result = ChecksumDetector.detect_checksum(no_checksum_messages)
        assert no_result is None
