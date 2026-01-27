"""CoAP option definitions and parsing helpers.

Provides option number to name mappings, content format definitions,
and helper functions for decoding CoAP options.

References:
    RFC 7252 Section 5.10: Option Definitions
    RFC 7252 Section 12.3: CoAP Content-Formats Registry
"""

from __future__ import annotations

from typing import Any, ClassVar

# CoAP option number to name mapping (RFC 7252)
OPTIONS: dict[int, str] = {
    1: "If-Match",
    3: "Uri-Host",
    4: "ETag",
    5: "If-None-Match",
    6: "Observe",  # RFC 7641
    7: "Uri-Port",
    8: "Location-Path",
    11: "Uri-Path",
    12: "Content-Format",
    14: "Max-Age",
    15: "Uri-Query",
    17: "Accept",
    20: "Location-Query",
    23: "Block2",  # RFC 7959
    27: "Block1",  # RFC 7959
    28: "Size2",  # RFC 7959
    35: "Proxy-Uri",
    39: "Proxy-Scheme",
    60: "Size1",  # RFC 7959
}

# CoAP content format codes (RFC 7252)
CONTENT_FORMATS: dict[int, str] = {
    0: "text/plain; charset=utf-8",
    40: "application/link-format",
    41: "application/xml",
    42: "application/octet-stream",
    47: "application/exi",
    50: "application/json",
    60: "application/cbor",
    # Additional formats from IANA registry
    100: "application/senml+json",
    101: "application/sensml+json",
    110: "application/senml+cbor",
    111: "application/sensml+cbor",
    112: "application/senml-exi",
    113: "application/sensml-exi",
    256: "application/coap-group+json",
    10000: "application/pkcs7-mime; smime-type=server-generated-key",
    10001: "application/pkcs7-mime; smime-type=certs-only",
    10002: "application/pkcs8",
    10003: "application/csrattrs",
    10004: "application/pkcs10",
    10005: "application/pkix-cert",
}


class OptionParser:
    """Helper class for parsing CoAP options.

    Provides methods for decoding option values based on their option number
    and handling CoAP option delta encoding.

    Example:
        >>> parser = OptionParser()
        >>> value = parser.decode_value(11, b"temperature")
        >>> print(value)
        'temperature'
    """

    # Empty format options (no value)
    EMPTY_OPTIONS: ClassVar[set[int]] = {5}  # If-None-Match

    # Opaque options (bytes)
    OPAQUE_OPTIONS: ClassVar[set[int]] = {1, 4}  # If-Match, ETag

    # String options (UTF-8)
    STRING_OPTIONS: ClassVar[set[int]] = {3, 8, 11, 15, 20, 35, 39}

    # Unsigned integer options
    UINT_OPTIONS: ClassVar[set[int]] = {6, 7, 12, 14, 17, 23, 27, 28, 60}

    @staticmethod
    def decode_value(option_num: int, value: bytes) -> str | int | bytes:
        """Decode option value based on option number.

        Args:
            option_num: CoAP option number.
            value: Raw option value bytes.

        Returns:
            Decoded value as string, int, or bytes depending on option type.

        Example:
            >>> parser = OptionParser()
            >>> parser.decode_value(11, b"sensor")
            'sensor'
            >>> parser.decode_value(12, b"\\x00\\x32")
            50
        """
        if option_num in OptionParser.EMPTY_OPTIONS:
            return b""

        if not value:
            # Empty value - return appropriate type
            if option_num in OptionParser.STRING_OPTIONS:
                return ""
            if option_num in OptionParser.UINT_OPTIONS:
                return 0
            return b""

        if option_num in OptionParser.STRING_OPTIONS:
            try:
                return value.decode("utf-8")
            except UnicodeDecodeError:
                return value

        if option_num in OptionParser.UINT_OPTIONS:
            return int.from_bytes(value, "big")

        # Default to opaque (bytes)
        return value

    @staticmethod
    def parse_extended_value(base: int, data: bytes, offset: int) -> tuple[int, int]:
        """Parse extended delta or length value.

        CoAP uses extended encoding for delta/length values >= 13:
        - 13: value is (base value from next byte) + 13
        - 14: value is (base value from next 2 bytes) + 269
        - 15: reserved (payload marker or error)

        Args:
            base: Base value (0-15) from option header nibble.
            data: Complete message data.
            offset: Current offset in data.

        Returns:
            Tuple of (actual_value, bytes_consumed).

        Raises:
            ValueError: If extended format is invalid or data insufficient.

        Example:
            >>> parser = OptionParser()
            >>> value, consumed = parser.parse_extended_value(13, b"\\x05\\x00", 0)
            >>> value, consumed
            (18, 1)
        """
        if base < 13:
            return base, 0

        if base == 13:
            if len(data) < offset + 1:
                raise ValueError("Insufficient data for extended option delta/length")
            return data[offset] + 13, 1

        if base == 14:
            if len(data) < offset + 2:
                raise ValueError("Insufficient data for extended option delta/length")
            return int.from_bytes(data[offset : offset + 2], "big") + 269, 2

        # base == 15
        raise ValueError("Invalid option delta/length value (15)")


def format_block_option(value: int) -> dict[str, Any]:
    """Parse Block1/Block2 option value.

    Block options encode: NUM (block number), M (more flag), SZX (size exponent).

    Format (variable length, 1-3 bytes):
        0 1 2 3 4 5 6 7
       +-+-+-+-+-+-+-+-+
       |  NUM  |M| SZX |
       +-+-+-+-+-+-+-+-+

    Args:
        value: Block option value as unsigned integer.

    Returns:
        Dictionary with 'num' (block number), 'more' (more flag), and 'size' (block size).

    Example:
        >>> result = format_block_option(0x08)  # Block 0, no more, size 16
        >>> result['num'], result['more'], result['size']
        (0, False, 16)
    """
    num = value >> 4
    more = bool((value >> 3) & 0x01)
    szx = value & 0x07
    size = 2 ** (szx + 4)  # Size = 2^(SZX+4), range: 16-1024 bytes

    return {
        "num": num,
        "more": more,
        "size": size,
    }


__all__ = [
    "CONTENT_FORMATS",
    "OPTIONS",
    "OptionParser",
    "format_block_option",
]
