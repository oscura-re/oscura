"""Tests for CRC polynomial reverse engineering."""

from __future__ import annotations

import pytest

from oscura.inference.crc_reverse import (
    STANDARD_CRCS,
    CRCParameters,
    CRCReverser,
    verify_crc,
)

pytestmark = [pytest.mark.unit]


class TestCRCReverser:
    """Test CRC polynomial reverse engineering."""

    def test_crc16_ccitt_recovery(self):
        """Test CRC-16-CCITT (0x1021) recovery."""
        # Generate test data with known CRC-16-CCITT
        reverser = CRCReverser()

        # Known CRC-16-CCITT results (poly=0x1021, init=0xFFFF, xor_out=0x0000)
        messages = [
            (b"Hello", b"\x1a\x3d"),  # Precomputed
            (b"World", b"\x7f\x16"),
            (b"Test1", b"\xbc\x0e"),
            (b"Test2", b"\xfc\x67"),
            (b"12345", b"\x29\xb1"),
        ]

        # Actually compute these correctly
        messages_computed = []
        for data in [b"Hello", b"World", b"Test1", b"Test2", b"12345"]:
            crc = reverser._calculate_crc(
                data=data,
                poly=0x1021,
                width=16,
                init=0xFFFF,
                xor_out=0x0000,
                refin=False,
                refout=False,
            )
            messages_computed.append((data, crc.to_bytes(2, "big")))

        params = reverser.reverse(messages_computed)

        assert params is not None
        assert params.polynomial == 0x1021
        assert params.width == 16
        assert params.init == 0xFFFF
        assert params.xor_out == 0x0000
        assert params.reflect_in is False
        assert params.reflect_out is False
        assert params.test_pass_rate == 1.0
        assert params.algorithm_name == "CRC-16-CCITT"

    def test_crc16_ibm_recovery(self):
        """Test CRC-16-IBM (0x8005) recovery with reflection."""
        reverser = CRCReverser()

        # Generate messages with CRC-16-IBM (reflected)
        messages = []
        for data in [b"Test", b"Data", b"CRC", b"Check", b"Valid"]:
            crc = reverser._calculate_crc(
                data=data,
                poly=0x8005,
                width=16,
                init=0x0000,
                xor_out=0x0000,
                refin=True,
                refout=True,
            )
            messages.append((data, crc.to_bytes(2, "big")))

        params = reverser.reverse(messages)

        assert params is not None
        assert params.polynomial == 0x8005
        assert params.width == 16
        assert params.init == 0x0000
        assert params.xor_out == 0x0000
        assert params.reflect_in is True
        assert params.reflect_out is True
        assert params.test_pass_rate == 1.0
        assert params.algorithm_name == "CRC-16-IBM"

    def test_crc32_recovery(self):
        """Test CRC-32 recovery."""
        reverser = CRCReverser()

        # Generate messages with CRC-32 (Ethernet)
        messages = []
        for data in [b"abc", b"test", b"data", b"hello", b"world"]:
            crc = reverser._calculate_crc(
                data=data,
                poly=0x04C11DB7,
                width=32,
                init=0xFFFFFFFF,
                xor_out=0xFFFFFFFF,
                refin=True,
                refout=True,
            )
            messages.append((data, crc.to_bytes(4, "big")))

        params = reverser.reverse(messages)

        assert params is not None
        assert params.polynomial == 0x04C11DB7
        assert params.width == 32
        assert params.init == 0xFFFFFFFF
        assert params.xor_out == 0xFFFFFFFF
        assert params.reflect_in is True
        assert params.reflect_out is True
        assert params.test_pass_rate == 1.0
        assert params.algorithm_name == "CRC-32"

    def test_crc8_recovery(self):
        """Test CRC-8 recovery."""
        reverser = CRCReverser()

        # Generate messages with CRC-8
        messages = []
        for data in [b"a", b"b", b"c", b"d", b"test"]:
            crc = reverser._calculate_crc(
                data=data,
                poly=0x07,
                width=8,
                init=0x00,
                xor_out=0x00,
                refin=False,
                refout=False,
            )
            messages.append((data, crc.to_bytes(1, "big")))

        params = reverser.reverse(messages)

        assert params is not None
        assert params.polynomial == 0x07
        assert params.width == 8
        assert params.init == 0x00
        assert params.xor_out == 0x00
        assert params.reflect_in is False
        assert params.reflect_out is False
        assert params.test_pass_rate == 1.0

    def test_auto_width_detection(self):
        """Test automatic CRC width detection."""
        reverser = CRCReverser()

        # 8-bit CRC
        messages_8 = [(b"test", b"\x00")]
        width = reverser._detect_width(messages_8)
        assert width == 8

        # 16-bit CRC
        messages_16 = [(b"test", b"\x00\x00")]
        width = reverser._detect_width(messages_16)
        assert width == 16

        # 32-bit CRC
        messages_32 = [(b"test", b"\x00\x00\x00\x00")]
        width = reverser._detect_width(messages_32)
        assert width == 32

    def test_insufficient_messages(self):
        """Test that insufficient messages raises error."""
        reverser = CRCReverser()

        # Only 3 messages (need at least 4)
        messages = [
            (b"test1", b"\x00\x00"),
            (b"test2", b"\x00\x00"),
            (b"test3", b"\x00\x00"),
        ]

        with pytest.raises(ValueError, match="at least 4 message pairs"):
            reverser.reverse(messages)

    def test_verify_crc(self):
        """Test CRC verification function."""
        params = CRCParameters(
            polynomial=0x1021,
            width=16,
            init=0xFFFF,
            xor_out=0x0000,
            reflect_in=False,
            reflect_out=False,
            confidence=1.0,
        )

        reverser = CRCReverser()
        correct_crc = reverser._calculate_crc(b"Hello", 0x1021, 16, 0xFFFF, 0x0000, False, False)

        # Correct CRC should verify
        assert verify_crc(b"Hello", correct_crc.to_bytes(2, "big"), params)

        # Incorrect CRC should not verify
        assert not verify_crc(b"Hello", b"\x00\x00", params)

    def test_xor_differential(self):
        """Test XOR differential calculation."""
        reverser = CRCReverser()

        result = reverser._xor_bytes(b"\x01\x02\x03", b"\x04\x05\x06")
        assert result == b"\x05\x07\x05"

        # Different lengths
        result = reverser._xor_bytes(b"\x01\x02", b"\x04\x05\x06")
        assert result == b"\x05\x07\x06"

    def test_bit_reflection(self):
        """Test bit reflection functions."""
        reverser = CRCReverser()

        # Reflect byte: 0b10101010 -> 0b01010101
        assert reverser._reflect_byte(0b10101010) == 0b01010101
        assert reverser._reflect_byte(0b11110000) == 0b00001111

        # Reflect 16-bit value
        assert reverser._reflect(0x0001, 16) == 0x8000
        assert reverser._reflect(0xFF00, 16) == 0x00FF

    def test_standard_algorithm_identification(self):
        """Test identification of standard CRC algorithms."""
        reverser = CRCReverser()

        # CRC-16-CCITT
        name = reverser._identify_standard(0x1021, 16, 0xFFFF, 0x0000, False, False)
        assert name == "CRC-16-CCITT"

        # CRC-32
        name = reverser._identify_standard(0x04C11DB7, 32, 0xFFFFFFFF, 0xFFFFFFFF, True, True)
        assert name == "CRC-32"

        # Unknown algorithm
        name = reverser._identify_standard(0x1234, 16, 0x0000, 0x0000, False, False)
        assert name is None

    def test_crc_parameters_repr(self):
        """Test CRCParameters string representation."""
        params = CRCParameters(
            polynomial=0x1021,
            width=16,
            init=0xFFFF,
            xor_out=0x0000,
            reflect_in=False,
            reflect_out=False,
            confidence=0.95,
        )

        repr_str = repr(params)
        assert "0x1021" in repr_str
        assert "width=16" in repr_str
        assert "confidence=0.95" in repr_str

    def test_verbose_mode(self):
        """Test verbose output during reversal."""
        reverser = CRCReverser(verbose=True)

        messages = []
        for data in [b"test1", b"test2", b"test3", b"test4", b"test5"]:
            crc = reverser._calculate_crc(
                data=data,
                poly=0x1021,
                width=16,
                init=0x0000,
                xor_out=0x0000,
                refin=False,
                refout=False,
            )
            messages.append((data, crc.to_bytes(2, "big")))

        # Should not raise, just print to stdout
        params = reverser.reverse(messages)
        assert params is not None

    def test_crc16_xmodem_recovery(self):
        """Test CRC-16-XMODEM recovery (different init than CCITT)."""
        reverser = CRCReverser()

        messages = []
        for data in [b"XMODEM", b"TEST", b"DATA", b"SEND", b"FILE"]:
            crc = reverser._calculate_crc(
                data=data,
                poly=0x1021,
                width=16,
                init=0x0000,
                xor_out=0x0000,
                refin=False,
                refout=False,
            )
            messages.append((data, crc.to_bytes(2, "big")))

        params = reverser.reverse(messages)

        assert params is not None
        assert params.polynomial == 0x1021
        assert params.width == 16
        assert params.init == 0x0000
        assert params.xor_out == 0x0000
        assert params.algorithm_name == "CRC-16-XMODEM"

    def test_crc16_modbus_recovery(self):
        """Test CRC-16-MODBUS recovery."""
        reverser = CRCReverser()

        messages = []
        for data in [b"\x01\x03", b"\x02\x04", b"\x03\x05", b"\x04\x06", b"\x05\x07"]:
            crc = reverser._calculate_crc(
                data=data,
                poly=0x8005,
                width=16,
                init=0xFFFF,
                xor_out=0x0000,
                refin=True,
                refout=True,
            )
            messages.append((data, crc.to_bytes(2, "big")))

        params = reverser.reverse(messages)

        assert params is not None
        assert params.polynomial == 0x8005
        assert params.width == 16
        assert params.init == 0xFFFF
        assert params.xor_out == 0x0000
        assert params.reflect_in is True
        assert params.reflect_out is True
        assert params.algorithm_name == "CRC-16-MODBUS"

    def test_varying_message_lengths(self):
        """Test with messages of varying lengths."""
        reverser = CRCReverser()

        messages = []
        test_data = [b"a", b"ab", b"abc", b"abcd", b"abcdefgh"]
        for data in test_data:
            crc = reverser._calculate_crc(
                data=data,
                poly=0x1021,
                width=16,
                init=0xFFFF,
                xor_out=0x0000,
                refin=False,
                refout=False,
            )
            messages.append((data, crc.to_bytes(2, "big")))

        params = reverser.reverse(messages)

        assert params is not None
        assert params.polynomial == 0x1021
        assert params.test_pass_rate == 1.0


class TestStandardCRCs:
    """Test that all standard CRC definitions are valid."""

    def test_all_standard_crcs_defined(self):
        """Test that STANDARD_CRCS dictionary is properly defined."""
        assert len(STANDARD_CRCS) > 0

        required_keys = {"width", "poly", "init", "xor_out", "refin", "refout"}
        for params in STANDARD_CRCS.values():
            assert set(params.keys()) == required_keys
            assert isinstance(params["width"], int)
            assert isinstance(params["poly"], int)
            assert isinstance(params["init"], int)
            assert isinstance(params["xor_out"], int)
            assert isinstance(params["refin"], bool)
            assert isinstance(params["refout"], bool)

    def test_standard_crc_widths(self):
        """Test that standard CRCs have valid widths."""
        valid_widths = {8, 16, 32, 64}
        for params in STANDARD_CRCS.values():
            assert params["width"] in valid_widths


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_message(self):
        """Test with empty message data."""
        reverser = CRCReverser()

        messages = []
        for _ in range(5):
            crc = reverser._calculate_crc(b"", 0x1021, 16, 0xFFFF, 0x0000, False, False)
            messages.append((b"", crc.to_bytes(2, "big")))

        # Should handle gracefully (might not find polynomial with all-empty data)
        params = reverser.reverse(messages)
        # We don't assert params is not None because empty data is degenerate

    def test_single_byte_messages(self):
        """Test with single-byte messages."""
        reverser = CRCReverser()

        messages = []
        for byte_val in [0x01, 0x02, 0x03, 0x04, 0x05]:
            data = bytes([byte_val])
            crc = reverser._calculate_crc(data, 0x07, 8, 0x00, 0x00, False, False)
            messages.append((data, crc.to_bytes(1, "big")))

        params = reverser.reverse(messages)
        assert params is not None
        assert params.width == 8

    def test_confidence_scoring(self):
        """Test that confidence scoring works correctly."""
        reverser = CRCReverser()

        # Create messages with correct CRCs
        messages = []
        for data in [b"test1", b"test2", b"test3", b"test4", b"test5"]:
            crc = reverser._calculate_crc(data, 0x1021, 16, 0xFFFF, 0x0000, False, False)
            messages.append((data, crc.to_bytes(2, "big")))

        params = reverser.reverse(messages)

        # All messages should validate
        assert params is not None
        assert params.confidence == 1.0
        assert params.test_pass_rate == 1.0
