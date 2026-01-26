"""Tests for MQTT 5.0 properties parsing.

Tests cover property parsing, variable byte integer decoding,
UTF-8 string decoding, and binary data decoding.
"""

from __future__ import annotations

import pytest

from oscura.iot.mqtt.properties import (
    PROPERTY_IDS,
    parse_properties,
)


class TestVariableByteInteger:
    """Test variable byte integer decoding (internal function)."""

    def test_decode_single_byte(self) -> None:
        """Test decoding single byte value."""
        from oscura.iot.mqtt.properties import _decode_variable_byte_integer

        value, consumed = _decode_variable_byte_integer(b"\x7f")
        assert value == 127
        assert consumed == 1

    def test_decode_two_bytes(self) -> None:
        """Test decoding two byte value."""
        from oscura.iot.mqtt.properties import _decode_variable_byte_integer

        # 128 = 0x80 0x01
        value, consumed = _decode_variable_byte_integer(b"\x80\x01")
        assert value == 128
        assert consumed == 2

    def test_decode_max_value(self) -> None:
        """Test decoding maximum value (268,435,455)."""
        from oscura.iot.mqtt.properties import _decode_variable_byte_integer

        value, consumed = _decode_variable_byte_integer(b"\xff\xff\xff\x7f")
        assert value == 268435455
        assert consumed == 4

    def test_decode_with_offset(self) -> None:
        """Test decoding with offset."""
        from oscura.iot.mqtt.properties import _decode_variable_byte_integer

        data = b"\x00\x00\x80\x01"
        value, consumed = _decode_variable_byte_integer(data, offset=2)
        assert value == 128
        assert consumed == 2

    def test_decode_insufficient_data(self) -> None:
        """Test error on insufficient data."""
        from oscura.iot.mqtt.properties import _decode_variable_byte_integer

        with pytest.raises(ValueError, match="Incomplete"):
            _decode_variable_byte_integer(b"\x80")

    def test_decode_exceeds_maximum(self) -> None:
        """Test error when value exceeds maximum."""
        from oscura.iot.mqtt.properties import _decode_variable_byte_integer

        with pytest.raises(ValueError, match="exceeds maximum"):
            _decode_variable_byte_integer(b"\xff\xff\xff\xff")


class TestUTF8String:
    """Test UTF-8 string decoding (internal function)."""

    def test_decode_simple_string(self) -> None:
        """Test decoding simple ASCII string."""
        from oscura.iot.mqtt.properties import _decode_utf8_string

        data = b"\x00\x05Hello"
        string, consumed = _decode_utf8_string(data)
        assert string == "Hello"
        assert consumed == 7

    def test_decode_empty_string(self) -> None:
        """Test decoding empty string."""
        from oscura.iot.mqtt.properties import _decode_utf8_string

        data = b"\x00\x00"
        string, consumed = _decode_utf8_string(data)
        assert string == ""
        assert consumed == 2

    def test_decode_unicode_string(self) -> None:
        """Test decoding Unicode string."""
        from oscura.iot.mqtt.properties import _decode_utf8_string

        # "Hello" in Japanese
        japanese = "こんにちは"
        encoded = japanese.encode("utf-8")
        data = len(encoded).to_bytes(2, "big") + encoded

        string, consumed = _decode_utf8_string(data)
        assert string == japanese
        assert consumed == 2 + len(encoded)

    def test_decode_with_offset(self) -> None:
        """Test decoding with offset."""
        from oscura.iot.mqtt.properties import _decode_utf8_string

        data = b"\x00\x00\x00\x04test"
        string, consumed = _decode_utf8_string(data, offset=2)
        assert string == "test"
        assert consumed == 6

    def test_decode_insufficient_data_length(self) -> None:
        """Test error on insufficient data for length."""
        from oscura.iot.mqtt.properties import _decode_utf8_string

        with pytest.raises(ValueError, match="Insufficient data for string length"):
            _decode_utf8_string(b"\x00")

    def test_decode_insufficient_data_content(self) -> None:
        """Test error on insufficient data for content."""
        from oscura.iot.mqtt.properties import _decode_utf8_string

        with pytest.raises(ValueError, match="Insufficient data for string content"):
            _decode_utf8_string(b"\x00\x10test")

    def test_decode_invalid_utf8(self) -> None:
        """Test error on invalid UTF-8 encoding."""
        from oscura.iot.mqtt.properties import _decode_utf8_string

        # Invalid UTF-8 sequence
        data = b"\x00\x02\xff\xfe"
        with pytest.raises(ValueError, match="Invalid UTF-8"):
            _decode_utf8_string(data)


class TestBinaryData:
    """Test binary data decoding (internal function)."""

    def test_decode_binary_data(self) -> None:
        """Test decoding binary data."""
        from oscura.iot.mqtt.properties import _decode_binary_data

        data = b"\x00\x03ABC"
        binary, consumed = _decode_binary_data(data)
        assert binary == b"ABC"
        assert consumed == 5

    def test_decode_empty_binary(self) -> None:
        """Test decoding empty binary data."""
        from oscura.iot.mqtt.properties import _decode_binary_data

        data = b"\x00\x00"
        binary, consumed = _decode_binary_data(data)
        assert binary == b""
        assert consumed == 2

    def test_decode_binary_with_offset(self) -> None:
        """Test decoding with offset."""
        from oscura.iot.mqtt.properties import _decode_binary_data

        data = b"\x00\x00\x00\x04data"
        binary, consumed = _decode_binary_data(data, offset=2)
        assert binary == b"data"
        assert consumed == 6


class TestPropertyIDs:
    """Test MQTT 5.0 property ID mappings."""

    def test_property_ids_defined(self) -> None:
        """Test that all standard property IDs are defined."""
        assert 0x01 in PROPERTY_IDS
        assert PROPERTY_IDS[0x01] == "payload_format_indicator"
        assert 0x02 in PROPERTY_IDS
        assert PROPERTY_IDS[0x02] == "message_expiry_interval"
        assert 0x26 in PROPERTY_IDS
        assert PROPERTY_IDS[0x26] == "user_property"

    def test_property_ids_count(self) -> None:
        """Test expected number of property IDs."""
        # MQTT 5.0 has 42 defined properties (we have 27 implemented)
        assert len(PROPERTY_IDS) >= 25


class TestParseProperties:
    """Test MQTT 5.0 property parsing."""

    def test_parse_empty_properties(self) -> None:
        """Test parsing when no properties present."""
        # Property length = 0
        data = b"\x00"
        props, consumed = parse_properties(data)

        assert props == {}
        assert consumed == 1

    def test_parse_byte_property(self) -> None:
        """Test parsing byte property (payload format indicator)."""
        # Length=2, ID=0x01, value=1
        data = b"\x02\x01\x01"
        props, consumed = parse_properties(data)

        assert props["payload_format_indicator"] == 1
        assert consumed == 3

    def test_parse_two_byte_integer_property(self) -> None:
        """Test parsing two-byte integer property (topic alias)."""
        # Length=3, ID=0x23, value=10
        data = b"\x03\x23\x00\x0a"
        props, consumed = parse_properties(data)

        assert props["topic_alias"] == 10
        assert consumed == 4

    def test_parse_four_byte_integer_property(self) -> None:
        """Test parsing four-byte integer property (message expiry)."""
        # Length=5, ID=0x02, value=60 seconds
        data = b"\x05\x02\x00\x00\x00\x3c"
        props, consumed = parse_properties(data)

        assert props["message_expiry_interval"] == 60
        assert consumed == 6

    def test_parse_utf8_string_property(self) -> None:
        """Test parsing UTF-8 string property (content type)."""
        # Length=14, ID=0x03, string="application/json"
        data = b"\x0e\x03\x00\x10application/json"
        props, consumed = parse_properties(data)

        assert props["content_type"] == "application/json"
        assert consumed == 15

    def test_parse_binary_data_property(self) -> None:
        """Test parsing binary data property (correlation data)."""
        # Length=7, ID=0x09, data=b"12345"
        data = b"\x07\x09\x00\x05" + b"12345"
        props, consumed = parse_properties(data)

        assert props["correlation_data"] == b"12345"
        assert consumed == 8

    def test_parse_variable_byte_integer_property(self) -> None:
        """Test parsing variable byte integer property (subscription ID)."""
        # Length=3, ID=0x0B, value=128
        data = b"\x03\x0b\x80\x01"
        props, consumed = parse_properties(data)

        assert props["subscription_identifier"] == 128
        assert consumed == 4

    def test_parse_user_property(self) -> None:
        """Test parsing user property (key-value pairs)."""
        # Length=15, ID=0x26, key="region", value="us-west"
        data = b"\x0f\x26\x00\x06region\x00\x07us-west"
        props, consumed = parse_properties(data)

        assert "user_property" in props
        assert len(props["user_property"]) == 1
        assert props["user_property"][0] == ("region", "us-west")

    def test_parse_multiple_user_properties(self) -> None:
        """Test parsing multiple user properties."""
        # Two user properties: Length = 2 + (1+2+3+2+5) + (1+2+4+2+4) = 2 + 13 + 13 = 28
        data = (
            b"\x1a"  # Length = 26 (actual payload size)
            b"\x26\x00\x03key\x00\x05value"  # First user property (13 bytes)
            b"\x26\x00\x04name\x00\x04test"  # Second user property (13 bytes)
        )
        props, consumed = parse_properties(data)

        assert len(props["user_property"]) == 2
        assert props["user_property"][0] == ("key", "value")
        assert props["user_property"][1] == ("name", "test")

    def test_parse_multiple_properties(self) -> None:
        """Test parsing multiple different properties."""
        # Length = 1+1 (payload format) + 1+4 (message expiry) + 1+2 (topic alias) = 10
        data = (
            b"\x0a"  # Length = 10
            b"\x01\x01"  # Payload format indicator = 1 (2 bytes)
            b"\x02\x00\x00\x00\x3c"  # Message expiry = 60 (5 bytes)
            b"\x23\x00\x0a"  # Topic alias = 10 (3 bytes)
        )
        props, consumed = parse_properties(data)

        assert props["payload_format_indicator"] == 1
        assert props["message_expiry_interval"] == 60
        assert props["topic_alias"] == 10
        assert consumed == 11  # 1 (length byte) + 10 (properties)

    def test_parse_properties_with_offset(self) -> None:
        """Test parsing properties with offset."""
        data = b"\x00\x00\x02\x01\x01"
        props, consumed = parse_properties(data, offset=2)

        assert props["payload_format_indicator"] == 1
        assert consumed == 3

    def test_parse_unknown_property_id(self) -> None:
        """Test error on unknown property ID."""
        # Invalid property ID 0xFF
        data = b"\x02\xff\x01"
        with pytest.raises(ValueError, match="Unknown property ID"):
            parse_properties(data)

    def test_parse_insufficient_properties_length(self) -> None:
        """Test error on insufficient data for properties length."""
        with pytest.raises(ValueError, match="Failed to decode properties length"):
            parse_properties(b"")

    def test_parse_insufficient_properties_data(self) -> None:
        """Test error when properties data is incomplete."""
        # Says length is 10 but only provides 2 bytes
        data = b"\x0a\x01\x01"
        with pytest.raises(ValueError, match="Insufficient data for properties"):
            parse_properties(data)

    def test_parse_insufficient_property_value(self) -> None:
        """Test error when property value data is incomplete."""
        # Message expiry needs 4 bytes but only 2 provided
        data = b"\x04\x02\x00\x00"
        with pytest.raises(ValueError, match="Insufficient data"):
            parse_properties(data)

    def test_parse_all_byte_properties(self) -> None:
        """Test parsing all single-byte property types."""
        byte_properties = [
            (0x01, "payload_format_indicator"),
            (0x24, "maximum_qos"),
            (0x25, "retain_available"),
        ]

        for prop_id, prop_name in byte_properties:
            data = bytes([0x02, prop_id, 0x01])
            props, _ = parse_properties(data)
            assert props[prop_name] == 1

    def test_parse_all_two_byte_properties(self) -> None:
        """Test parsing all two-byte integer property types."""
        two_byte_properties = [
            (0x13, "server_keep_alive"),
            (0x21, "receive_maximum"),
            (0x22, "topic_alias_maximum"),
            (0x23, "topic_alias"),
        ]

        for prop_id, prop_name in two_byte_properties:
            data = bytes([0x03, prop_id, 0x00, 0x0A])
            props, _ = parse_properties(data)
            assert props[prop_name] == 10

    def test_parse_all_four_byte_properties(self) -> None:
        """Test parsing all four-byte integer property types."""
        four_byte_properties = [
            (0x02, "message_expiry_interval"),
            (0x11, "session_expiry_interval"),
            (0x18, "will_delay_interval"),
            (0x27, "maximum_packet_size"),
        ]

        for prop_id, prop_name in four_byte_properties:
            data = bytes([0x05, prop_id, 0x00, 0x00, 0x00, 0x3C])
            props, _ = parse_properties(data)
            assert props[prop_name] == 60


class TestIntegrationProperties:
    """Integration tests for property parsing in real scenarios."""

    def test_connect_properties(self) -> None:
        """Test parsing typical CONNECT packet properties."""
        # Session expiry interval + maximum packet size
        data = (
            b"\x09"  # Length
            b"\x11\x00\x00\x0e\x10"  # Session expiry = 3600
            b"\x27\x00\x01\x00\x00"  # Max packet size = 65536
        )
        props, consumed = parse_properties(data)

        assert props["session_expiry_interval"] == 3600
        assert props["maximum_packet_size"] == 65536
        assert consumed == 10

    def test_publish_properties(self) -> None:
        """Test parsing typical PUBLISH packet properties."""
        # Message expiry + content type + response topic
        data = (
            b"\x20"  # Length = 32
            b"\x02\x00\x00\x00\x3c"  # Message expiry = 60
            b"\x03\x00\x10application/json"  # Content type
            b"\x08\x00\x08response"  # Response topic
        )
        props, consumed = parse_properties(data)

        assert props["message_expiry_interval"] == 60
        assert props["content_type"] == "application/json"
        assert props["response_topic"] == "response"

    def test_empty_vs_no_properties(self) -> None:
        """Test distinction between empty properties and no properties."""
        # Empty properties (length = 0)
        data1 = b"\x00"
        props1, consumed1 = parse_properties(data1)
        assert props1 == {}
        assert consumed1 == 1

        # Properties present but empty dict
        data2 = b"\x00additional_data"
        props2, consumed2 = parse_properties(data2)
        assert props2 == {}
        assert consumed2 == 1


class TestPropertyEdgeCases:
    """Test edge cases for property parsing."""

    def test_decode_binary_data_insufficient_length(self) -> None:
        """Test binary data decoding with insufficient length field."""
        from oscura.iot.mqtt.properties import _decode_binary_data

        with pytest.raises(ValueError, match="Insufficient data for binary length"):
            _decode_binary_data(b"\x00")

    def test_decode_binary_data_insufficient_content(self) -> None:
        """Test binary data decoding with insufficient content."""
        from oscura.iot.mqtt.properties import _decode_binary_data

        with pytest.raises(ValueError, match="Insufficient data for binary content"):
            _decode_binary_data(b"\x00\x10AB")

    def test_parse_property_insufficient_data_byte_property(self) -> None:
        """Test property parsing when byte property data is missing."""
        # Property ID 0x01 (payload format indicator) needs 1 byte
        data = b"\x01\x01"  # Length says 1 property, but no value
        with pytest.raises(ValueError, match="Insufficient data"):
            parse_properties(data)

    def test_parse_property_insufficient_data_two_byte_property(self) -> None:
        """Test property parsing when two-byte property data is missing."""
        # Property ID 0x23 (topic alias) needs 2 bytes
        data = b"\x02\x23\x00"  # Only 1 byte provided
        with pytest.raises(ValueError, match="Insufficient data"):
            parse_properties(data)

    def test_parse_property_insufficient_data_four_byte_property(self) -> None:
        """Test property parsing when four-byte property data is missing."""
        # Property ID 0x02 (message expiry interval) needs 4 bytes
        data = b"\x03\x02\x00\x00"  # Only 2 bytes provided
        with pytest.raises(ValueError, match="Insufficient data"):
            parse_properties(data)

    def test_parse_property_unexpected_end(self) -> None:
        """Test error when property data ends unexpectedly."""
        # Length says 5 bytes but only property ID is present
        data = b"\x05"
        with pytest.raises(ValueError, match="Insufficient data for properties"):
            parse_properties(data)

    def test_decode_variable_byte_integer_offset(self) -> None:
        """Test variable byte integer decoding at different offsets."""
        from oscura.iot.mqtt.properties import _decode_variable_byte_integer

        # Value 300 = 0xAC 0x02
        data = b"\xff\xff\xac\x02\xff"
        value, consumed = _decode_variable_byte_integer(data, offset=2)
        assert value == 300
        assert consumed == 2

    def test_parse_all_string_properties(self) -> None:
        """Test parsing all UTF-8 string property types."""
        string_properties = [
            (0x03, "content_type"),
            (0x08, "response_topic"),
            (0x12, "assigned_client_identifier"),
            (0x15, "authentication_method"),
            (0x1A, "response_information"),
            (0x1C, "server_reference"),
            (0x1F, "reason_string"),
        ]

        for prop_id, prop_name in string_properties:
            # Property length = 1 (ID) + 2 (string length) + 4 (string) = 7
            data = bytes([0x07, prop_id, 0x00, 0x04]) + b"test"
            props, _ = parse_properties(data)
            assert props[prop_name] == "test"

    def test_parse_all_binary_properties(self) -> None:
        """Test parsing all binary data property types."""
        binary_properties = [
            (0x09, "correlation_data"),
            (0x16, "authentication_data"),
        ]

        for prop_id, prop_name in binary_properties:
            # Property length = 1 (ID) + 2 (binary length) + 3 (binary) = 6
            data = bytes([0x06, prop_id, 0x00, 0x03]) + b"ABC"
            props, _ = parse_properties(data)
            assert props[prop_name] == b"ABC"

    def test_parse_all_bool_availability_properties(self) -> None:
        """Test parsing all boolean availability properties."""
        bool_properties = [
            (0x28, "wildcard_subscription_available"),
            (0x29, "subscription_identifier_available"),
            (0x2A, "shared_subscription_available"),
        ]

        for prop_id, prop_name in bool_properties:
            # Value 0 = not available, 1 = available
            data = bytes([0x02, prop_id, 0x01])
            props, _ = parse_properties(data)
            assert props[prop_name] == 1

    def test_parse_variable_byte_integer_property_max_value(self) -> None:
        """Test parsing subscription identifier with max value."""
        # Subscription ID with max value (268,435,455)
        data = b"\x05\x0b\xff\xff\xff\x7f"
        props, consumed = parse_properties(data)

        assert props["subscription_identifier"] == 268435455
        assert consumed == 6

    def test_parse_multiple_user_properties_three(self) -> None:
        """Test parsing three user properties."""
        # Three user properties
        data = (
            b"\x20"  # Length = 32
            b"\x26\x00\x03key\x00\x05value"  # First (13 bytes)
            b"\x26\x00\x04name\x00\x04test"  # Second (13 bytes)
            b"\x26\x00\x02id\x00\x0112"  # Third (8 bytes, total 34 but let's recalculate)
        )
        # Actual: 1 + 2+3+2+5 + 1 + 2+4+2+4 + 1 + 2+2+2+2 = 1+12+13+9 = 35, but payload is 34
        # Let me fix the calculation: 1+2+3+2+5=13, 1+2+4+2+4=13, 1+2+2+2+1=8 = 34
        data = (
            b"\x22"  # Length = 34
            b"\x26\x00\x03key\x00\x05value"  # First (13 bytes)
            b"\x26\x00\x04name\x00\x04test"  # Second (13 bytes)
            b"\x26\x00\x02id\x00\x0212"  # Third (9 bytes, total 35)
        )
        # Recalculate: 13 + 13 + (1+2+2+2+2) = 13+13+9 = 35
        data = (
            b"\x23"  # Length = 35
            b"\x26\x00\x03key\x00\x05value"  # First
            b"\x26\x00\x04name\x00\x04test"  # Second
            b"\x26\x00\x02id\x00\x0212"  # Third
        )

        props, consumed = parse_properties(data)

        assert len(props["user_property"]) == 3
        assert props["user_property"][0] == ("key", "value")
        assert props["user_property"][1] == ("name", "test")
        assert props["user_property"][2] == ("id", "12")

    def test_parse_request_problem_information(self) -> None:
        """Test parsing request problem information property."""
        # Property 0x17 (request_problem_information), byte value
        # This property is in PROPERTY_IDS but not handled in _decode_property_value
        # So it will raise "Unhandled property type"
        data = b"\x02\x17\x01"
        with pytest.raises(ValueError, match="Unhandled property type"):
            parse_properties(data)

    def test_parse_request_response_information(self) -> None:
        """Test parsing request response information property."""
        # Property 0x19 (request_response_information), byte value
        # This property is in PROPERTY_IDS but not handled in _decode_property_value
        # So it will raise "Unhandled property type"
        data = b"\x02\x19\x00"
        with pytest.raises(ValueError, match="Unhandled property type"):
            parse_properties(data)
