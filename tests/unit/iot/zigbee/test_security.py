"""Tests for Zigbee security module.

Tests security header parsing and frame encryption detection.
"""

from __future__ import annotations

from oscura.iot.zigbee.security import (
    get_security_level_name,
    is_frame_encrypted,
    parse_security_header,
)


class TestSecurityHeaderParsing:
    """Test security header parsing."""

    def test_parse_basic_security_header(self) -> None:
        """Test parsing basic security header."""
        data = bytes(
            [
                0x05,  # Security control (level 5)
                0x01,
                0x00,
                0x00,
                0x00,  # Frame counter
            ]
        )

        result = parse_security_header(data)

        assert result["security_level"] == 5
        assert result["frame_counter"] == 1
        assert result["extended_nonce"] is False
        assert result["header_length"] == 5

    def test_parse_security_header_with_extended_nonce(self) -> None:
        """Test parsing security header with extended nonce."""
        data = bytes(
            [
                0x25,  # Security control (level 5, extended nonce)
                0x01,
                0x00,
                0x00,
                0x00,  # Frame counter
                0x45,
                0x23,
                0xA1,
                0x40,
                0x20,
                0xA2,
                0x13,
                0x00,  # Source address (little-endian)
            ]
        )

        result = parse_security_header(data)

        assert result["extended_nonce"] is True
        # Bytes [0x45, 0x23, 0xA1, 0x40, 0x20, 0xA2, 0x13, 0x00] in little-endian = 0x0013A22040A12345
        assert result["source_address"] == 0x0013A22040A12345
        assert result["header_length"] == 13

    def test_parse_security_header_with_key_sequence(self) -> None:
        """Test parsing security header with key sequence number."""
        data = bytes(
            [
                0x0D,  # Security control (level 5, key identifier = 01)
                0x01,
                0x00,
                0x00,
                0x00,  # Frame counter
                0x05,  # Key sequence number
            ]
        )

        result = parse_security_header(data)

        assert result["key_identifier"] == 1
        assert result["key_sequence_number"] == 5
        assert result["header_length"] == 6

    def test_parse_security_header_insufficient_data(self) -> None:
        """Test parsing security header with insufficient data."""
        data = bytes([0x05, 0x01])  # Too short

        result = parse_security_header(data)

        assert "error" in result

    def test_parse_security_header_missing_extended_nonce(self) -> None:
        """Test parsing security header missing extended nonce data."""
        data = bytes(
            [
                0x25,  # Extended nonce flag set
                0x01,
                0x00,
                0x00,
                0x00,  # Frame counter
                # Missing 8 bytes of source address
            ]
        )

        result = parse_security_header(data)

        assert "error" in result

    def test_parse_security_header_missing_key_sequence(self) -> None:
        """Test parsing security header missing key sequence."""
        data = bytes(
            [
                0x0D,  # Key identifier = 01
                0x01,
                0x00,
                0x00,
                0x00,  # Frame counter
                # Missing key sequence number
            ]
        )

        result = parse_security_header(data)

        assert "error" in result


class TestSecurityLevels:
    """Test security level utilities."""

    def test_security_level_names(self) -> None:
        """Test security level name lookup."""
        assert get_security_level_name(0) == "None"
        assert get_security_level_name(1) == "MIC-32"
        assert get_security_level_name(4) == "ENC"
        assert get_security_level_name(5) == "ENC-MIC-32"
        assert get_security_level_name(6) == "ENC-MIC-64"
        assert get_security_level_name(7) == "ENC-MIC-128"

    def test_security_level_unknown(self) -> None:
        """Test unknown security level."""
        result = get_security_level_name(99)
        assert "Unknown" in result


class TestFrameEncryption:
    """Test frame encryption detection."""

    def test_is_frame_encrypted_true(self) -> None:
        """Test detecting encrypted frame."""
        frame_control = 0x0208  # Security bit set (bit 9)

        assert is_frame_encrypted(frame_control) is True

    def test_is_frame_encrypted_false(self) -> None:
        """Test detecting unencrypted frame."""
        frame_control = 0x0008  # Security bit not set

        assert is_frame_encrypted(frame_control) is False

    def test_is_frame_encrypted_various_controls(self) -> None:
        """Test encryption detection with various frame controls."""
        assert is_frame_encrypted(0x0200) is True  # Only security bit
        assert is_frame_encrypted(0x0000) is False  # No bits
        assert is_frame_encrypted(0xFFFF) is True  # All bits (includes security)
        assert is_frame_encrypted(0x01FF) is False  # All bits except security


class TestSecurityEdgeCases:
    """Test security edge cases."""

    def test_parse_empty_data(self) -> None:
        """Test parsing empty security data."""
        data = bytes([])

        result = parse_security_header(data)

        assert "error" in result

    def test_parse_single_byte(self) -> None:
        """Test parsing single byte."""
        data = bytes([0x05])

        result = parse_security_header(data)

        assert "error" in result

    def test_parse_frame_counter_zero(self) -> None:
        """Test parsing with frame counter zero."""
        data = bytes(
            [
                0x05,  # Security control
                0x00,
                0x00,
                0x00,
                0x00,  # Frame counter = 0
            ]
        )

        result = parse_security_header(data)

        assert result["frame_counter"] == 0
        assert "error" not in result

    def test_parse_frame_counter_max(self) -> None:
        """Test parsing with maximum frame counter."""
        data = bytes(
            [
                0x05,  # Security control
                0xFF,
                0xFF,
                0xFF,
                0xFF,  # Frame counter = max
            ]
        )

        result = parse_security_header(data)

        assert result["frame_counter"] == 0xFFFFFFFF

    def test_parse_all_security_levels(self) -> None:
        """Test parsing all security levels."""
        for level in range(8):
            data = bytes(
                [
                    level,  # Security control with level
                    0x01,
                    0x00,
                    0x00,
                    0x00,  # Frame counter
                ]
            )

            result = parse_security_header(data)

            assert result["security_level"] == level
            assert "error" not in result
