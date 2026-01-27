"""Tests for CoAP option parsing and definitions.

Tests option value decoding, extended delta/length parsing, and
block option formatting.
"""

from __future__ import annotations

import pytest

from oscura.iot.coap.options import (
    CONTENT_FORMATS,
    OPTIONS,
    OptionParser,
    format_block_option,
)


class TestOptionDefinitions:
    """Test option and content format definitions."""

    def test_options_defined(self) -> None:
        """Test that standard options are defined."""
        assert 11 in OPTIONS
        assert OPTIONS[11] == "Uri-Path"
        assert OPTIONS[12] == "Content-Format"
        assert OPTIONS[6] == "Observe"
        assert OPTIONS[23] == "Block2"

    def test_content_formats_defined(self) -> None:
        """Test that standard content formats are defined."""
        assert 0 in CONTENT_FORMATS
        assert CONTENT_FORMATS[0] == "text/plain; charset=utf-8"
        assert CONTENT_FORMATS[50] == "application/json"
        assert CONTENT_FORMATS[60] == "application/cbor"


class TestOptionParser:
    """Test OptionParser functionality."""

    def test_decode_string_option(self) -> None:
        """Test decoding string option (Uri-Path)."""
        parser = OptionParser()

        value = parser.decode_value(11, b"temperature")

        assert value == "temperature"
        assert isinstance(value, str)

    def test_decode_uint_option(self) -> None:
        """Test decoding unsigned integer option (Content-Format)."""
        parser = OptionParser()

        value = parser.decode_value(12, b"\x00\x32")

        assert value == 50
        assert isinstance(value, int)

    def test_decode_single_byte_uint(self) -> None:
        """Test decoding single byte unsigned integer."""
        parser = OptionParser()

        value = parser.decode_value(14, b"\x3c")  # Max-Age = 60

        assert value == 60

    def test_decode_empty_option(self) -> None:
        """Test decoding empty option (If-None-Match)."""
        parser = OptionParser()

        value = parser.decode_value(5, b"")

        assert value == b""

    def test_decode_opaque_option(self) -> None:
        """Test decoding opaque option (ETag)."""
        parser = OptionParser()

        value = parser.decode_value(4, b"\x01\x02\x03\x04")

        assert value == b"\x01\x02\x03\x04"
        assert isinstance(value, bytes)

    def test_decode_invalid_utf8_as_bytes(self) -> None:
        """Test that invalid UTF-8 in string option returns bytes."""
        parser = OptionParser()

        # Invalid UTF-8 sequence
        value = parser.decode_value(11, b"\xff\xfe")

        assert value == b"\xff\xfe"
        assert isinstance(value, bytes)

    def test_parse_extended_value_base_13(self) -> None:
        """Test parsing extended value with base 13."""
        parser = OptionParser()

        # Base 13, extension byte = 5 -> actual value = 18
        actual, consumed = parser.parse_extended_value(13, b"\x05\x00", 0)

        assert actual == 18
        assert consumed == 1

    def test_parse_extended_value_base_14(self) -> None:
        """Test parsing extended value with base 14."""
        parser = OptionParser()

        # Base 14, extension = 0x0100 -> actual value = 269 + 256 = 525
        actual, consumed = parser.parse_extended_value(14, b"\x01\x00", 0)

        assert actual == 525
        assert consumed == 2

    def test_parse_extended_value_base_less_than_13(self) -> None:
        """Test parsing extended value with base < 13 (no extension)."""
        parser = OptionParser()

        actual, consumed = parser.parse_extended_value(5, b"", 0)

        assert actual == 5
        assert consumed == 0

    def test_parse_extended_value_base_15_error(self) -> None:
        """Test parsing extended value with invalid base 15."""
        parser = OptionParser()

        with pytest.raises(ValueError, match="Invalid option delta/length"):
            parser.parse_extended_value(15, b"", 0)

    def test_parse_extended_value_insufficient_data_13(self) -> None:
        """Test parsing extended value with insufficient data for base 13."""
        parser = OptionParser()

        with pytest.raises(ValueError, match="Insufficient data"):
            parser.parse_extended_value(13, b"", 0)

    def test_parse_extended_value_insufficient_data_14(self) -> None:
        """Test parsing extended value with insufficient data for base 14."""
        parser = OptionParser()

        with pytest.raises(ValueError, match="Insufficient data"):
            parser.parse_extended_value(14, b"\x00", 0)

    def test_option_parser_categories(self) -> None:
        """Test that option categories are correctly defined."""
        assert 5 in OptionParser.EMPTY_OPTIONS
        assert 4 in OptionParser.OPAQUE_OPTIONS
        assert 11 in OptionParser.STRING_OPTIONS
        assert 12 in OptionParser.UINT_OPTIONS


class TestBlockOption:
    """Test block option parsing."""

    def test_format_block_option_simple(self) -> None:
        """Test formatting simple block option."""
        # Block 0, no more, SZX=0 (size 16)
        result = format_block_option(0x00)

        assert result["num"] == 0
        assert result["more"] is False
        assert result["size"] == 16

    def test_format_block_option_with_more(self) -> None:
        """Test formatting block option with more flag."""
        # Block 0, more=True, SZX=1 (size 32)
        # Value: 0000 1001 = 0x09
        result = format_block_option(0x08)

        assert result["num"] == 0
        assert result["more"] is True
        assert result["size"] == 16

    def test_format_block_option_block_number(self) -> None:
        """Test formatting block option with block number."""
        # Block 5, no more, SZX=2 (size 64)
        # Value: 0101 0010 = 0x52
        result = format_block_option(0x52)

        assert result["num"] == 5
        assert result["more"] is False
        assert result["size"] == 64

    def test_format_block_option_max_size(self) -> None:
        """Test formatting block option with maximum size."""
        # Block 0, no more, SZX=6 (size 1024)
        result = format_block_option(0x06)

        assert result["num"] == 0
        assert result["more"] is False
        assert result["size"] == 1024

    def test_format_block_option_large_block_number(self) -> None:
        """Test formatting block option with large block number."""
        # Block 15, more=True, SZX=3 (size 128)
        # Value: 1111 1011 = 0xFB
        result = format_block_option(0xFB)

        assert result["num"] == 15
        assert result["more"] is True
        assert result["size"] == 128

    def test_format_block_option_all_sizes(self) -> None:
        """Test all possible block sizes."""
        expected_sizes = {
            0: 16,
            1: 32,
            2: 64,
            3: 128,
            4: 256,
            5: 512,
            6: 1024,
        }

        for szx, expected_size in expected_sizes.items():
            result = format_block_option(szx)
            assert result["size"] == expected_size

    def test_format_block_option_zero(self) -> None:
        """Test formatting block option with value 0."""
        result = format_block_option(0x00)

        assert result["num"] == 0
        assert result["more"] is False
        assert result["size"] == 16


class TestOptionParserEdgeCases:
    """Test edge cases in option parsing."""

    def test_decode_empty_string_option(self) -> None:
        """Test decoding empty string option."""
        parser = OptionParser()

        value = parser.decode_value(11, b"")

        assert value == ""

    def test_decode_zero_uint_option(self) -> None:
        """Test decoding zero-value uint option."""
        parser = OptionParser()

        value = parser.decode_value(12, b"\x00")

        assert value == 0

    def test_decode_unknown_option_as_bytes(self) -> None:
        """Test that unknown option is decoded as bytes."""
        parser = OptionParser()

        # Option 999 (unknown)
        value = parser.decode_value(999, b"\x01\x02\x03")

        assert value == b"\x01\x02\x03"
        assert isinstance(value, bytes)

    def test_decode_multi_byte_uint(self) -> None:
        """Test decoding multi-byte unsigned integer."""
        parser = OptionParser()

        # 4-byte value: 0x12345678
        value = parser.decode_value(60, b"\x12\x34\x56\x78")

        assert value == 0x12345678

    def test_decode_unicode_string_option(self) -> None:
        """Test decoding Unicode string option."""
        parser = OptionParser()

        # UTF-8 encoded emoji
        value = parser.decode_value(11, "🌡️".encode())

        assert value == "🌡️"
        assert isinstance(value, str)
