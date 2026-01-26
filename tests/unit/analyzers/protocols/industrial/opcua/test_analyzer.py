"""Tests for OPC UA protocol analyzer."""

from __future__ import annotations

from pathlib import Path

import pytest

from oscura.analyzers.protocols.industrial.opcua import OPCUAAnalyzer


class TestOPCUAAnalyzer:
    """Test suite for OPC UA protocol analyzer."""

    def test_analyzer_initialization(self) -> None:
        """Test analyzer initializes correctly."""
        analyzer = OPCUAAnalyzer()
        assert len(analyzer.messages) == 0
        assert len(analyzer.nodes) == 0
        assert analyzer.security_mode == "None"

    def test_parse_hello_message(self) -> None:
        """Test parsing OPC UA Hello message."""
        analyzer = OPCUAAnalyzer()

        # Construct Hello message
        # HEL + F + MessageSize + Protocol Version + Buffers + Endpoint URL
        hello = bytearray()
        hello.extend(b"HEL")  # Message type
        hello.extend(b"F")  # Chunk type (Final)

        # Build payload first to calculate size
        payload = bytearray()
        payload.extend((0).to_bytes(4, "little"))  # Protocol version 0
        payload.extend((65536).to_bytes(4, "little"))  # Receive buffer size
        payload.extend((65536).to_bytes(4, "little"))  # Send buffer size
        payload.extend((0).to_bytes(4, "little"))  # Max message size (0 = no limit)
        payload.extend((0).to_bytes(4, "little"))  # Max chunk count (0 = no limit)

        # Endpoint URL (length-prefixed string)
        url = b"opc.tcp://localhost:4840"
        payload.extend(len(url).to_bytes(4, "little"))
        payload.extend(url)

        # Write message size (header + payload)
        hello.extend((8 + len(payload)).to_bytes(4, "little"))
        hello.extend(payload)

        msg = analyzer.parse_message(bytes(hello), timestamp=1.0)

        assert msg.timestamp == 1.0
        assert msg.message_type == "HEL"
        assert msg.is_final is True
        assert msg.chunk_type == "F"
        assert msg.decoded_service["protocol_version"] == 0
        assert msg.decoded_service["receive_buffer_size"] == 65536
        assert msg.decoded_service["send_buffer_size"] == 65536
        assert msg.decoded_service["endpoint_url"] == "opc.tcp://localhost:4840"

    def test_parse_acknowledge_message(self) -> None:
        """Test parsing OPC UA Acknowledge message."""
        analyzer = OPCUAAnalyzer()

        # Construct Acknowledge message
        ack = bytearray()
        ack.extend(b"ACK")  # Message type
        ack.extend(b"F")  # Chunk type
        ack.extend((28).to_bytes(4, "little"))  # Message size

        # Acknowledge payload
        ack.extend((0).to_bytes(4, "little"))  # Protocol version
        ack.extend((65536).to_bytes(4, "little"))  # Receive buffer size
        ack.extend((65536).to_bytes(4, "little"))  # Send buffer size
        ack.extend((0).to_bytes(4, "little"))  # Max message size
        ack.extend((0).to_bytes(4, "little"))  # Max chunk count

        msg = analyzer.parse_message(bytes(ack), timestamp=2.0)

        assert msg.timestamp == 2.0
        assert msg.message_type == "ACK"
        assert msg.is_final is True
        assert msg.decoded_service["protocol_version"] == 0
        assert msg.decoded_service["receive_buffer_size"] == 65536

    def test_parse_open_secure_channel(self) -> None:
        """Test parsing Open Secure Channel message."""
        analyzer = OPCUAAnalyzer()

        # Construct OpenSecureChannel message
        opn = bytearray()
        opn.extend(b"OPN")  # Message type
        opn.extend(b"F")  # Chunk type

        # Build payload first
        payload = bytearray()
        payload.extend((1).to_bytes(4, "little"))  # Secure channel ID

        # Security policy URI
        policy_uri = b"http://opcfoundation.org/UA/SecurityPolicy#None"
        payload.extend(len(policy_uri).to_bytes(4, "little"))
        payload.extend(policy_uri)

        # Write message size
        opn.extend((8 + len(payload)).to_bytes(4, "little"))
        opn.extend(payload)

        msg = analyzer.parse_message(bytes(opn))

        assert msg.message_type == "OPN"
        assert msg.decoded_service["secure_channel_id"] == 1
        assert (
            msg.decoded_service["security_policy_uri"]
            == "http://opcfoundation.org/UA/SecurityPolicy#None"
        )

    def test_parse_message_chunk(self) -> None:
        """Test parsing MSG chunk with service data."""
        analyzer = OPCUAAnalyzer()

        # Construct MSG chunk
        msg = bytearray()
        msg.extend(b"MSG")  # Message type
        msg.extend(b"F")  # Chunk type (Final)
        msg.extend((32).to_bytes(4, "little"))  # Message size

        # MSG chunk payload
        msg.extend((1).to_bytes(4, "little"))  # Secure channel ID
        msg.extend((1).to_bytes(4, "little"))  # Security token ID
        msg.extend((1).to_bytes(4, "little"))  # Sequence number
        msg.extend((1).to_bytes(4, "little"))  # Request ID

        # Service payload (simplified - would contain service type NodeId + data)
        msg.extend(b"\x01\x00\xa5\x01")  # FourByte NodeId encoding for ReadRequest (421)
        msg.extend(b"\x00" * 4)  # Dummy service payload

        parsed = analyzer.parse_message(bytes(msg), timestamp=3.0)

        assert parsed.timestamp == 3.0
        assert parsed.message_type == "MSG"
        assert parsed.is_final is True
        assert parsed.decoded_service["secure_channel_id"] == 1
        assert parsed.decoded_service["security_token_id"] == 1
        assert parsed.decoded_service["sequence_number"] == 1
        assert parsed.decoded_service["request_id"] == 1
        assert parsed.decoded_service["service_id"] == 421

    def test_parse_error_message(self) -> None:
        """Test parsing Error message."""
        analyzer = OPCUAAnalyzer()

        # Construct Error message
        err = bytearray()
        err.extend(b"ERR")  # Message type
        err.extend(b"F")  # Chunk type

        # Build payload
        payload = bytearray()
        payload.extend((0x80010000).to_bytes(4, "little"))  # Error code (Bad_Unexpected)

        # Reason string
        reason = b"Invalid message"
        payload.extend(len(reason).to_bytes(4, "little"))
        payload.extend(reason)

        # Write message size
        err.extend((8 + len(payload)).to_bytes(4, "little"))
        err.extend(payload)

        msg = analyzer.parse_message(bytes(err))

        assert msg.message_type == "ERR"
        assert msg.decoded_service["error_code"] == 0x80010000
        assert msg.decoded_service["reason"] == "Invalid message"

    def test_parse_message_too_short(self) -> None:
        """Test parsing message with insufficient data."""
        analyzer = OPCUAAnalyzer()

        # Message shorter than minimum 8 bytes
        short_msg = b"HEL"

        with pytest.raises(ValueError, match="too short"):
            analyzer.parse_message(short_msg)

    def test_parse_invalid_message_type(self) -> None:
        """Test parsing message with invalid type."""
        analyzer = OPCUAAnalyzer()

        # Invalid message type "XYZ"
        invalid = bytearray()
        invalid.extend(b"XYZ")
        invalid.extend(b"F")
        invalid.extend((8).to_bytes(4, "little"))

        with pytest.raises(ValueError, match="Unknown OPC UA message type"):
            analyzer.parse_message(bytes(invalid))

    def test_parse_invalid_chunk_type(self) -> None:
        """Test parsing message with invalid chunk type."""
        analyzer = OPCUAAnalyzer()

        # Valid message type but invalid chunk type 'X'
        invalid = bytearray()
        invalid.extend(b"HEL")
        invalid.extend(b"X")  # Invalid chunk type
        invalid.extend((8).to_bytes(4, "little"))

        with pytest.raises(ValueError, match="Invalid chunk type"):
            analyzer.parse_message(bytes(invalid))

    def test_parse_message_size_mismatch(self) -> None:
        """Test parsing message with size mismatch."""
        analyzer = OPCUAAnalyzer()

        # Message with incorrect size field
        invalid = bytearray()
        invalid.extend(b"HEL")
        invalid.extend(b"F")
        invalid.extend((100).to_bytes(4, "little"))  # Claims 100 bytes
        # But only 8 bytes provided

        with pytest.raises(ValueError, match="Message size mismatch"):
            analyzer.parse_message(bytes(invalid))

    def test_parse_hello_with_null_url(self) -> None:
        """Test parsing Hello message with null endpoint URL."""
        analyzer = OPCUAAnalyzer()

        hello = bytearray()
        hello.extend(b"HEL")
        hello.extend(b"F")

        # Build payload
        payload = bytearray()
        payload.extend((0).to_bytes(4, "little"))  # Protocol version
        payload.extend((65536).to_bytes(4, "little"))  # Receive buffer size
        payload.extend((65536).to_bytes(4, "little"))  # Send buffer size
        payload.extend((0).to_bytes(4, "little"))  # Max message size
        payload.extend((0).to_bytes(4, "little"))  # Max chunk count

        # Null string (length = -1)
        payload.extend((0xFFFFFFFF).to_bytes(4, "little"))

        # Write message size
        hello.extend((8 + len(payload)).to_bytes(4, "little"))
        hello.extend(payload)

        msg = analyzer.parse_message(bytes(hello))

        assert msg.message_type == "HEL"
        assert msg.decoded_service["endpoint_url"] is None

    def test_chunk_types(self) -> None:
        """Test different chunk types (Final, Continue, Abort)."""
        analyzer = OPCUAAnalyzer()

        for chunk_type in ["F", "C", "A"]:
            msg = bytearray()
            msg.extend(b"MSG")
            msg.extend(chunk_type.encode("ascii"))
            msg.extend((24).to_bytes(4, "little"))
            msg.extend((1).to_bytes(4, "little"))  # Secure channel ID
            msg.extend((1).to_bytes(4, "little"))  # Security token ID
            msg.extend((1).to_bytes(4, "little"))  # Sequence number
            msg.extend((1).to_bytes(4, "little"))  # Request ID

            parsed = analyzer.parse_message(bytes(msg))
            assert parsed.chunk_type == chunk_type
            assert parsed.is_final == (chunk_type == "F")

    def test_service_id_mapping(self) -> None:
        """Test service ID to name mapping."""
        analyzer = OPCUAAnalyzer()

        # Test known service IDs
        assert analyzer.SERVICE_IDS[421] == "ReadRequest"
        assert analyzer.SERVICE_IDS[424] == "ReadResponse"
        assert analyzer.SERVICE_IDS[673] == "WriteRequest"
        assert analyzer.SERVICE_IDS[527] == "BrowseRequest"
        assert analyzer.SERVICE_IDS[826] == "PublishRequest"

    def test_message_type_constants(self) -> None:
        """Test message type constant mappings."""
        analyzer = OPCUAAnalyzer()

        assert analyzer.MESSAGE_TYPES[0x48454C] == "HEL"
        assert analyzer.MESSAGE_TYPES[0x41434B] == "ACK"
        assert analyzer.MESSAGE_TYPES[0x4F504E] == "OPN"
        assert analyzer.MESSAGE_TYPES[0x434C4F] == "CLO"
        assert analyzer.MESSAGE_TYPES[0x4D5347] == "MSG"
        assert analyzer.MESSAGE_TYPES[0x455252] == "ERR"

    def test_export_address_space_empty(self, tmp_path: Path) -> None:
        """Test exporting empty address space."""
        analyzer = OPCUAAnalyzer()
        output_file = tmp_path / "opcua_nodes.json"

        analyzer.export_address_space(output_file)

        assert output_file.exists()

        import json

        with output_file.open() as f:
            data = json.load(f)

        assert data["nodes"] == []
        assert data["message_count"] == 0
        assert data["security_mode"] == "None"

    def test_multiple_messages(self) -> None:
        """Test parsing multiple messages in sequence."""
        analyzer = OPCUAAnalyzer()

        # Parse Hello
        hello = bytearray()
        hello.extend(b"HEL")
        hello.extend(b"F")

        hello_payload = bytearray()
        hello_payload.extend((0).to_bytes(4, "little"))
        hello_payload.extend((65536).to_bytes(4, "little"))
        hello_payload.extend((65536).to_bytes(4, "little"))
        hello_payload.extend((0).to_bytes(4, "little"))
        hello_payload.extend((0).to_bytes(4, "little"))
        hello_payload.extend((0xFFFFFFFF).to_bytes(4, "little"))  # Null URL

        hello.extend((8 + len(hello_payload)).to_bytes(4, "little"))
        hello.extend(hello_payload)

        analyzer.parse_message(bytes(hello), timestamp=1.0)

        # Parse Acknowledge
        ack = bytearray()
        ack.extend(b"ACK")
        ack.extend(b"F")
        ack.extend((28).to_bytes(4, "little"))
        ack.extend((0).to_bytes(4, "little"))
        ack.extend((65536).to_bytes(4, "little"))
        ack.extend((65536).to_bytes(4, "little"))
        ack.extend((0).to_bytes(4, "little"))
        ack.extend((0).to_bytes(4, "little"))

        analyzer.parse_message(bytes(ack), timestamp=2.0)

        assert len(analyzer.messages) == 2
        assert analyzer.messages[0].message_type == "HEL"
        assert analyzer.messages[0].timestamp == 1.0
        assert analyzer.messages[1].message_type == "ACK"
        assert analyzer.messages[1].timestamp == 2.0

    def test_hello_payload_too_short(self) -> None:
        """Test Hello message with insufficient payload."""
        analyzer = OPCUAAnalyzer()

        hello = bytearray()
        hello.extend(b"HEL")
        hello.extend(b"F")
        hello.extend((16).to_bytes(4, "little"))  # Claims 16 bytes total
        hello.extend((0).to_bytes(4, "little"))  # Only 4 bytes of payload (need 20)
        hello.extend((0).to_bytes(4, "little"))

        with pytest.raises(ValueError, match="Hello message too short"):
            analyzer.parse_message(bytes(hello))

    def test_message_chunk_too_short(self) -> None:
        """Test MSG chunk with insufficient payload."""
        analyzer = OPCUAAnalyzer()

        msg = bytearray()
        msg.extend(b"MSG")
        msg.extend(b"F")
        msg.extend((16).to_bytes(4, "little"))  # Claims 16 bytes
        msg.extend((1).to_bytes(4, "little"))  # Only 8 bytes payload (need 16)
        msg.extend((1).to_bytes(4, "little"))

        with pytest.raises(ValueError, match="MSG chunk too short"):
            analyzer.parse_message(bytes(msg))

    def test_invalid_message_size_too_small(self) -> None:
        """Test message with invalid size (less than minimum)."""
        analyzer = OPCUAAnalyzer()

        invalid = bytearray()
        invalid.extend(b"HEL")
        invalid.extend(b"F")
        invalid.extend((4).to_bytes(4, "little"))  # Invalid size < 8

        with pytest.raises(ValueError, match="Invalid message size"):
            analyzer.parse_message(bytes(invalid))

    def test_service_name_assignment(self) -> None:
        """Test that service names are correctly assigned to messages."""
        analyzer = OPCUAAnalyzer()

        msg = bytearray()
        msg.extend(b"MSG")
        msg.extend(b"F")
        msg.extend((32).to_bytes(4, "little"))
        msg.extend((1).to_bytes(4, "little"))
        msg.extend((1).to_bytes(4, "little"))
        msg.extend((1).to_bytes(4, "little"))
        msg.extend((1).to_bytes(4, "little"))
        msg.extend(b"\x01\x00\xa5\x01")  # ReadRequest (421)
        msg.extend(b"\x00" * 4)

        parsed = analyzer.parse_message(bytes(msg))

        assert parsed.service_id == 421
        assert parsed.service_name == "ReadRequest"

    def test_close_secure_channel_message(self) -> None:
        """Test parsing Close Secure Channel message."""
        analyzer = OPCUAAnalyzer()

        # CLO message (similar structure to OPN, simplified)
        clo = bytearray()
        clo.extend(b"CLO")
        clo.extend(b"F")

        payload = bytearray()
        payload.extend((1).to_bytes(4, "little"))  # Secure channel ID

        clo.extend((8 + len(payload)).to_bytes(4, "little"))
        clo.extend(payload)

        msg = analyzer.parse_message(bytes(clo))

        assert msg.message_type == "CLO"
        # CLO uses same parser as OPN for now (simplified)
        assert msg.decoded_service["secure_channel_id"] == 1
