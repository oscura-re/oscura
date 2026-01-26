"""Tests for FlexRay CRC calculations.

Tests cover:
- Header CRC-11 calculation
- Frame CRC-24 calculation
- CRC verification functions
- Edge cases (empty payload, maximum payload)
"""

from oscura.automotive.flexray.crc import (
    calculate_frame_crc,
    calculate_header_crc,
    verify_frame_crc,
    verify_header_crc,
)


class TestHeaderCRC:
    """Tests for FlexRay header CRC-11 calculation."""

    def test_header_crc_basic(self) -> None:
        """Test basic header CRC calculation."""
        crc = calculate_header_crc(
            reserved=0,
            payload_preamble=0,
            null_frame=0,
            sync_frame=0,
            startup_frame=0,
            frame_id=100,
            payload_length=5,
        )

        # CRC should be 11 bits
        assert 0 <= crc <= 0x7FF

    def test_header_crc_different_frame_ids(self) -> None:
        """Test header CRC changes with different frame IDs."""
        crc1 = calculate_header_crc(0, 0, 0, 0, 0, 100, 5)
        crc2 = calculate_header_crc(0, 0, 0, 0, 0, 200, 5)

        assert crc1 != crc2

    def test_header_crc_different_payload_lengths(self) -> None:
        """Test header CRC changes with different payload lengths."""
        crc1 = calculate_header_crc(0, 0, 0, 0, 0, 100, 5)
        crc2 = calculate_header_crc(0, 0, 0, 0, 0, 100, 10)

        assert crc1 != crc2

    def test_header_crc_with_flags(self) -> None:
        """Test header CRC with different flag combinations."""
        # Sync frame
        crc_sync = calculate_header_crc(0, 0, 0, 1, 0, 100, 5)

        # Startup frame
        crc_startup = calculate_header_crc(0, 0, 0, 0, 1, 100, 5)

        # Null frame
        crc_null = calculate_header_crc(0, 0, 1, 0, 0, 100, 5)

        # All should be different
        assert crc_sync != crc_startup
        assert crc_sync != crc_null
        assert crc_startup != crc_null

    def test_header_crc_verify_valid(self) -> None:
        """Test verifying valid header CRC."""
        crc = calculate_header_crc(0, 0, 0, 0, 0, 100, 5)
        valid = verify_header_crc(0, 0, 0, 0, 0, 100, 5, crc)

        assert valid is True

    def test_header_crc_verify_invalid(self) -> None:
        """Test verifying invalid header CRC."""
        crc = calculate_header_crc(0, 0, 0, 0, 0, 100, 5)
        valid = verify_header_crc(0, 0, 0, 0, 0, 100, 5, crc ^ 0xFF)  # Corrupt CRC

        assert valid is False

    def test_header_crc_zero_frame_id(self) -> None:
        """Test header CRC with minimum valid parameters."""
        # Note: Frame ID 0 is technically invalid in FlexRay,
        # but CRC should still calculate
        crc = calculate_header_crc(0, 0, 0, 0, 0, 0, 0)
        assert 0 <= crc <= 0x7FF

    def test_header_crc_max_frame_id(self) -> None:
        """Test header CRC with maximum frame ID."""
        crc = calculate_header_crc(0, 0, 0, 0, 0, 2047, 127)
        assert 0 <= crc <= 0x7FF

    def test_header_crc_deterministic(self) -> None:
        """Test that header CRC is deterministic."""
        crc1 = calculate_header_crc(0, 0, 0, 0, 0, 100, 5)
        crc2 = calculate_header_crc(0, 0, 0, 0, 0, 100, 5)

        assert crc1 == crc2


class TestFrameCRC:
    """Tests for FlexRay frame CRC-24 calculation."""

    def test_frame_crc_basic(self) -> None:
        """Test basic frame CRC calculation."""
        header = bytes([0x00, 0x64, 0x00, 0x00, 0x00])  # Simple header
        payload = bytes([0x01, 0x02, 0x03, 0x04, 0x05])

        crc = calculate_frame_crc(header, payload)

        # CRC should be 24 bits
        assert 0 <= crc <= 0xFFFFFF

    def test_frame_crc_empty_payload(self) -> None:
        """Test frame CRC with empty payload."""
        header = bytes([0x00, 0x64, 0x00, 0x00, 0x00])
        payload = b""

        crc = calculate_frame_crc(header, payload)
        assert 0 <= crc <= 0xFFFFFF

    def test_frame_crc_different_payloads(self) -> None:
        """Test frame CRC changes with different payloads."""
        header = bytes([0x00, 0x64, 0x00, 0x00, 0x00])

        crc1 = calculate_frame_crc(header, b"\x01\x02\x03")
        crc2 = calculate_frame_crc(header, b"\x04\x05\x06")

        assert crc1 != crc2

    def test_frame_crc_different_headers(self) -> None:
        """Test frame CRC changes with different headers."""
        payload = bytes([0x01, 0x02, 0x03])

        header1 = bytes([0x00, 0x64, 0x00, 0x00, 0x00])
        header2 = bytes([0x00, 0xC8, 0x00, 0x00, 0x00])

        crc1 = calculate_frame_crc(header1, payload)
        crc2 = calculate_frame_crc(header2, payload)

        assert crc1 != crc2

    def test_frame_crc_verify_valid(self) -> None:
        """Test verifying valid frame CRC."""
        header = bytes([0x00, 0x64, 0x00, 0x00, 0x00])
        payload = bytes([0x01, 0x02, 0x03])

        crc = calculate_frame_crc(header, payload)
        valid = verify_frame_crc(header, payload, crc)

        assert valid is True

    def test_frame_crc_verify_invalid(self) -> None:
        """Test verifying invalid frame CRC."""
        header = bytes([0x00, 0x64, 0x00, 0x00, 0x00])
        payload = bytes([0x01, 0x02, 0x03])

        crc = calculate_frame_crc(header, payload)
        valid = verify_frame_crc(header, payload, crc ^ 0xFFFF)  # Corrupt CRC

        assert valid is False

    def test_frame_crc_maximum_payload(self) -> None:
        """Test frame CRC with maximum payload length."""
        header = bytes([0x00, 0x64, 0x00, 0x00, 0x00])
        payload = bytes(range(254))  # Maximum 254 bytes

        crc = calculate_frame_crc(header, payload)
        assert 0 <= crc <= 0xFFFFFF

        # Verify
        valid = verify_frame_crc(header, payload, crc)
        assert valid is True

    def test_frame_crc_deterministic(self) -> None:
        """Test that frame CRC is deterministic."""
        header = bytes([0x00, 0x64, 0x00, 0x00, 0x00])
        payload = bytes([0x01, 0x02, 0x03])

        crc1 = calculate_frame_crc(header, payload)
        crc2 = calculate_frame_crc(header, payload)

        assert crc1 == crc2

    def test_frame_crc_single_bit_change(self) -> None:
        """Test that single bit change in data changes CRC."""
        header = bytes([0x00, 0x64, 0x00, 0x00, 0x00])
        payload1 = bytes([0x00, 0x00, 0x00])
        payload2 = bytes([0x01, 0x00, 0x00])  # Single bit different

        crc1 = calculate_frame_crc(header, payload1)
        crc2 = calculate_frame_crc(header, payload2)

        assert crc1 != crc2


class TestCRCIntegration:
    """Integration tests for CRC functions."""

    def test_complete_frame_crc_chain(self) -> None:
        """Test complete CRC calculation and verification chain."""
        # Calculate header CRC
        header_crc = calculate_header_crc(0, 0, 0, 0, 0, 100, 5)

        # Build header bytes
        header_int = (
            (0 << 39)
            | (0 << 38)
            | (0 << 37)
            | (0 << 36)
            | (0 << 35)
            | (100 << 24)
            | (5 << 17)
            | (header_crc << 6)
            | (0 << 0)
        )
        header_bytes = header_int.to_bytes(5, "big")

        # Verify header CRC
        assert verify_header_crc(0, 0, 0, 0, 0, 100, 5, header_crc)

        # Create payload
        payload = bytes(range(10))

        # Calculate frame CRC
        frame_crc = calculate_frame_crc(header_bytes, payload)

        # Verify frame CRC
        assert verify_frame_crc(header_bytes, payload, frame_crc)

    def test_crc_detects_header_corruption(self) -> None:
        """Test that CRC detects header corruption."""
        header = bytes([0x00, 0x64, 0x00, 0x00, 0x00])
        payload = b""

        # Calculate CRC for valid header
        crc = calculate_frame_crc(header, payload)

        # Corrupt header
        corrupted_header = bytes([0xFF, 0x64, 0x00, 0x00, 0x00])

        # CRC should not verify
        assert not verify_frame_crc(corrupted_header, payload, crc)

    def test_crc_detects_payload_corruption(self) -> None:
        """Test that CRC detects payload corruption."""
        header = bytes([0x00, 0x64, 0x00, 0x00, 0x00])
        payload = bytes([0x01, 0x02, 0x03])

        # Calculate CRC for valid payload
        crc = calculate_frame_crc(header, payload)

        # Corrupt payload
        corrupted_payload = bytes([0xFF, 0x02, 0x03])

        # CRC should not verify
        assert not verify_frame_crc(header, corrupted_payload, crc)
