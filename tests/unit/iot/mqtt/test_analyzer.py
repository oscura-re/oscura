"""Tests for MQTT protocol analyzer.

Tests cover MQTT 3.1.1 and 5.0 packet parsing, session tracking,
and topic hierarchy generation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from oscura.iot.mqtt import MQTTAnalyzer


class TestMQTTAnalyzer:
    """Test MQTT analyzer initialization and basic operations."""

    def test_init(self) -> None:
        """Test analyzer initialization."""
        analyzer = MQTTAnalyzer()
        assert len(analyzer.packets) == 0
        assert len(analyzer.sessions) == 0
        assert len(analyzer.topics) == 0

    def test_packet_types_defined(self) -> None:
        """Test packet type mapping is complete."""
        analyzer = MQTTAnalyzer()
        assert len(analyzer.PACKET_TYPES) == 15
        assert analyzer.PACKET_TYPES[1] == "CONNECT"
        assert analyzer.PACKET_TYPES[3] == "PUBLISH"
        assert analyzer.PACKET_TYPES[15] == "AUTH"


class TestFixedHeaderParsing:
    """Test MQTT fixed header parsing."""

    def test_parse_fixed_header_connect(self) -> None:
        """Test parsing CONNECT fixed header."""
        analyzer = MQTTAnalyzer()
        # CONNECT packet: type=1, flags=0, length=16
        data = b"\x10\x10"
        ptype, flags, length = analyzer._parse_fixed_header(data)
        assert ptype == 1
        assert flags == 0
        assert length == 16

    def test_parse_fixed_header_publish_qos0(self) -> None:
        """Test parsing PUBLISH fixed header with QoS 0."""
        analyzer = MQTTAnalyzer()
        # PUBLISH: type=3, flags=0 (QoS 0, no DUP, no RETAIN)
        data = b"\x30\x0a"
        ptype, flags, length = analyzer._parse_fixed_header(data)
        assert ptype == 3
        assert flags == 0
        assert length == 10

    def test_parse_fixed_header_publish_qos1_retain(self) -> None:
        """Test parsing PUBLISH fixed header with QoS 1 and RETAIN."""
        analyzer = MQTTAnalyzer()
        # PUBLISH: type=3, flags=0x03 (QoS 1, RETAIN)
        data = b"\x33\x0c"
        ptype, flags, length = analyzer._parse_fixed_header(data)
        assert ptype == 3
        assert flags == 0x03
        assert length == 12

    def test_parse_fixed_header_variable_length(self) -> None:
        """Test parsing variable length encoding (>127 bytes)."""
        analyzer = MQTTAnalyzer()
        # Length = 128 (0x80 0x01 in variable byte encoding)
        data = b"\x30\x80\x01"
        ptype, flags, length = analyzer._parse_fixed_header(data)
        assert ptype == 3
        assert length == 128

    def test_parse_fixed_header_max_length(self) -> None:
        """Test parsing maximum variable length (268,435,455)."""
        analyzer = MQTTAnalyzer()
        # Maximum length: 0xFF 0xFF 0xFF 0x7F
        data = b"\x30\xff\xff\xff\x7f"
        ptype, flags, length = analyzer._parse_fixed_header(data)
        assert ptype == 3
        assert length == 268435455

    def test_parse_fixed_header_insufficient_data(self) -> None:
        """Test error on insufficient data."""
        analyzer = MQTTAnalyzer()
        with pytest.raises(ValueError, match="Insufficient data"):
            analyzer._parse_fixed_header(b"\x10")

    def test_parse_fixed_header_incomplete_length(self) -> None:
        """Test error on incomplete variable length."""
        analyzer = MQTTAnalyzer()
        # Length byte with continuation bit but no next byte
        with pytest.raises(ValueError, match="Incomplete remaining length"):
            analyzer._parse_fixed_header(b"\x30\x80")

    def test_parse_fixed_header_exceeds_maximum(self) -> None:
        """Test error when length exceeds maximum."""
        analyzer = MQTTAnalyzer()
        # Length with 5 bytes (invalid)
        with pytest.raises(ValueError, match="exceeds maximum"):
            analyzer._parse_fixed_header(b"\x30\xff\xff\xff\xff\x7f")


class TestConnectParsing:
    """Test CONNECT packet parsing."""

    def test_parse_connect_mqtt311_minimal(self) -> None:
        """Test parsing minimal MQTT 3.1.1 CONNECT packet."""
        analyzer = MQTTAnalyzer()
        # Protocol: MQTT, level: 4, flags: 0x02 (clean session), keepalive: 60, client ID: "test"
        data = b"\x00\x04MQTT\x04\x02\x00\x3c\x00\x04test"
        result = analyzer._parse_connect(data)

        assert result["protocol_name"] == "MQTT"
        assert result["protocol_version"] == "3.1.1"
        assert result["keep_alive"] == 60
        assert result["client_id"] == "test"
        assert result["flags"]["clean_session"] is True
        assert result["username"] is None
        assert result["password"] is None

    def test_parse_connect_with_username_password(self) -> None:
        """Test parsing CONNECT with username and password."""
        analyzer = MQTTAnalyzer()
        # flags: 0xC2 (clean session, username, password)
        data = (
            b"\x00\x04MQTT\x04\xc2\x00\x3c"
            b"\x00\x04test"  # client ID
            b"\x00\x05admin"  # username
            b"\x00\x06secret"  # password
        )
        result = analyzer._parse_connect(data)

        assert result["client_id"] == "test"
        assert result["username"] == "admin"
        assert result["password"] == b"secret"

    def test_parse_connect_with_will(self) -> None:
        """Test parsing CONNECT with Last Will and Testament."""
        analyzer = MQTTAnalyzer()
        # flags: 0x06 (clean session, will flag)
        data = (
            b"\x00\x04MQTT\x04\x06\x00\x3c"
            b"\x00\x04test"  # client ID
            b"\x00\x0astatus/lwt"  # will topic
            b"\x00\x07offline"  # will message
        )
        result = analyzer._parse_connect(data)

        assert result["will_topic"] == "status/lwt"
        assert result["will_message"] == b"offline"
        assert result["flags"]["will_flag"] is True

    def test_parse_connect_mqtt_31(self) -> None:
        """Test parsing MQTT 3.1 CONNECT packet."""
        analyzer = MQTTAnalyzer()
        # Protocol level 3 = MQTT 3.1
        data = b"\x00\x06MQIsdp\x03\x02\x00\x3c\x00\x04test"
        result = analyzer._parse_connect(data)

        assert result["protocol_name"] == "MQIsdp"
        assert result["protocol_version"] == "3.1"

    def test_parse_connect_empty_client_id(self) -> None:
        """Test parsing CONNECT with empty client ID."""
        analyzer = MQTTAnalyzer()
        # Empty client ID with clean session
        data = b"\x00\x04MQTT\x04\x02\x00\x3c\x00\x00"
        result = analyzer._parse_connect(data)

        assert result["client_id"] == ""

    def test_parse_connect_insufficient_data(self) -> None:
        """Test error on insufficient data."""
        analyzer = MQTTAnalyzer()
        with pytest.raises(ValueError, match="Insufficient data"):
            analyzer._parse_connect(b"\x00\x04MQTT")


class TestPublishParsing:
    """Test PUBLISH packet parsing."""

    def test_parse_publish_qos0(self) -> None:
        """Test parsing PUBLISH with QoS 0."""
        analyzer = MQTTAnalyzer()
        # Topic: "test", no packet ID (QoS 0), payload: "hello"
        data = b"\x00\x04testhello"
        result = analyzer._parse_publish(data, flags=0x00)

        assert result["topic"] == "test"
        assert result["qos"] == 0
        assert result["payload"] == b"hello"
        assert result["packet_id"] is None
        assert result["flags"]["dup"] is False
        assert result["flags"]["retain"] is False

    def test_parse_publish_qos1_with_packet_id(self) -> None:
        """Test parsing PUBLISH with QoS 1 and packet ID."""
        analyzer = MQTTAnalyzer()
        # QoS 1 requires packet ID
        data = b"\x00\x04test\x00\x01hello"
        result = analyzer._parse_publish(data, flags=0x02)  # QoS 1

        assert result["topic"] == "test"
        assert result["qos"] == 1
        assert result["packet_id"] == 1
        assert result["payload"] == b"hello"

    def test_parse_publish_with_dup_retain(self) -> None:
        """Test parsing PUBLISH with DUP and RETAIN flags."""
        analyzer = MQTTAnalyzer()
        data = b"\x00\x04test\x00\x01data"
        # flags = 0x0B: DUP (bit 3), QoS 1 (bits 1-2), RETAIN (bit 0)
        result = analyzer._parse_publish(data, flags=0x0B)

        assert result["flags"]["dup"] is True
        assert result["flags"]["retain"] is True
        assert result["qos"] == 1

    def test_parse_publish_qos2(self) -> None:
        """Test parsing PUBLISH with QoS 2."""
        analyzer = MQTTAnalyzer()
        data = b"\x00\x06sensor\x00\x0a22.5"
        result = analyzer._parse_publish(data, flags=0x04)  # QoS 2

        assert result["topic"] == "sensor"
        assert result["qos"] == 2
        assert result["packet_id"] == 10

    def test_parse_publish_empty_payload(self) -> None:
        """Test parsing PUBLISH with empty payload."""
        analyzer = MQTTAnalyzer()
        data = b"\x00\x05empty"
        result = analyzer._parse_publish(data, flags=0x00)

        assert result["topic"] == "empty"
        assert result["payload"] == b""

    def test_parse_publish_multilevel_topic(self) -> None:
        """Test parsing PUBLISH with multi-level topic."""
        analyzer = MQTTAnalyzer()
        # Topic length: 24 bytes = "home/sensor/temperature"
        data = b"\x00\x17home/sensor/temperature25.3"
        result = analyzer._parse_publish(data, flags=0x00)

        assert result["topic"] == "home/sensor/temperature"
        assert result["payload"] == b"25.3"

    def test_parse_publish_insufficient_data_topic(self) -> None:
        """Test error on insufficient data for topic."""
        analyzer = MQTTAnalyzer()
        with pytest.raises(ValueError, match="Insufficient data for topic"):
            analyzer._parse_publish(b"\x00\x10test", flags=0x00)

    def test_parse_publish_insufficient_data_packet_id(self) -> None:
        """Test error on missing packet ID for QoS > 0."""
        analyzer = MQTTAnalyzer()
        with pytest.raises(ValueError, match="Insufficient data for packet ID"):
            analyzer._parse_publish(b"\x00\x04test", flags=0x02)


class TestSubscribeParsing:
    """Test SUBSCRIBE packet parsing."""

    def test_parse_subscribe_single_topic(self) -> None:
        """Test parsing SUBSCRIBE with single topic."""
        analyzer = MQTTAnalyzer()
        # Packet ID: 1, topic: "test", QoS: 1
        data = b"\x00\x01\x00\x04test\x01"
        result = analyzer._parse_subscribe(data)

        assert result["packet_id"] == 1
        assert len(result["topics"]) == 1
        assert result["topics"][0] == ("test", 1)

    def test_parse_subscribe_multiple_topics(self) -> None:
        """Test parsing SUBSCRIBE with multiple topics."""
        analyzer = MQTTAnalyzer()
        data = b"\x00\x02\x00\x06sensor\x00\x00\x06status\x01"
        result = analyzer._parse_subscribe(data)

        assert result["packet_id"] == 2
        assert len(result["topics"]) == 2
        assert result["topics"][0] == ("sensor", 0)
        assert result["topics"][1] == ("status", 1)

    def test_parse_subscribe_wildcard_topics(self) -> None:
        """Test parsing SUBSCRIBE with wildcard topics."""
        analyzer = MQTTAnalyzer()
        # Multi-level wildcard (#) and single-level (+)
        # Format: packet_id(2) + topic_len(2) + topic + qos(1) + topic_len(2) + topic + qos(1)
        data = b"\x00\x03\x00\x06home/#\x02\x00\x08sensor/+\x01"
        result = analyzer._parse_subscribe(data)

        assert len(result["topics"]) == 2
        assert result["topics"][0] == ("home/#", 2)
        assert result["topics"][1] == ("sensor/+", 1)

    def test_parse_subscribe_insufficient_data(self) -> None:
        """Test error on insufficient data."""
        analyzer = MQTTAnalyzer()
        with pytest.raises(ValueError, match="Incomplete topic filter"):
            analyzer._parse_subscribe(b"\x00\x01\x00\x10test")


class TestAckParsing:
    """Test acknowledgment packet parsing (PUBACK, PUBREC, PUBREL, PUBCOMP)."""

    def test_parse_puback_minimal(self) -> None:
        """Test parsing PUBACK with just packet ID."""
        analyzer = MQTTAnalyzer()
        data = b"\x00\x01"
        result = analyzer._parse_ack(data, "PUBACK")

        assert result["packet_id"] == 1
        assert result["reason_code"] is None

    def test_parse_puback_with_reason_code(self) -> None:
        """Test parsing PUBACK with reason code (MQTT 5.0)."""
        analyzer = MQTTAnalyzer()
        data = b"\x00\x01\x00"  # Packet ID 1, reason code 0 (success)
        result = analyzer._parse_ack(data, "PUBACK")

        assert result["packet_id"] == 1
        assert result["reason_code"] == 0

    def test_parse_ack_all_types(self) -> None:
        """Test parsing all acknowledgment types."""
        analyzer = MQTTAnalyzer()
        data = b"\x00\x05"

        for packet_type in ["PUBACK", "PUBREC", "PUBREL", "PUBCOMP", "UNSUBACK"]:
            result = analyzer._parse_ack(data, packet_type)
            assert result["packet_id"] == 5


class TestSubackParsing:
    """Test SUBACK packet parsing."""

    def test_parse_suback_single_topic(self) -> None:
        """Test parsing SUBACK for single subscription."""
        analyzer = MQTTAnalyzer()
        # Packet ID 1, return code 0x01 (maximum QoS 1)
        data = b"\x00\x01\x01"
        result = analyzer._parse_suback(data)

        assert result["packet_id"] == 1
        assert result["return_codes"] == [1]

    def test_parse_suback_multiple_topics(self) -> None:
        """Test parsing SUBACK for multiple subscriptions."""
        analyzer = MQTTAnalyzer()
        # Different return codes
        data = b"\x00\x02\x00\x01\x02\x80"
        result = analyzer._parse_suback(data)

        assert result["packet_id"] == 2
        assert result["return_codes"] == [0, 1, 2, 0x80]

    def test_parse_suback_failure(self) -> None:
        """Test parsing SUBACK with failure code."""
        analyzer = MQTTAnalyzer()
        # 0x80 = subscription failure
        data = b"\x00\x03\x80"
        result = analyzer._parse_suback(data)

        assert result["return_codes"] == [0x80]


class TestUnsubscribeParsing:
    """Test UNSUBSCRIBE packet parsing."""

    def test_parse_unsubscribe_single_topic(self) -> None:
        """Test parsing UNSUBSCRIBE with single topic."""
        analyzer = MQTTAnalyzer()
        data = b"\x00\x01\x00\x04test"
        result = analyzer._parse_unsubscribe(data)

        assert result["packet_id"] == 1
        assert result["topics"] == ["test"]

    def test_parse_unsubscribe_multiple_topics(self) -> None:
        """Test parsing UNSUBSCRIBE with multiple topics."""
        analyzer = MQTTAnalyzer()
        data = b"\x00\x02\x00\x06sensor\x00\x06status"
        result = analyzer._parse_unsubscribe(data)

        assert result["packet_id"] == 2
        assert result["topics"] == ["sensor", "status"]


class TestConnackParsing:
    """Test CONNACK packet parsing."""

    def test_parse_connack_success(self) -> None:
        """Test parsing successful CONNACK."""
        analyzer = MQTTAnalyzer()
        # Session present=0, return code=0 (accepted)
        data = b"\x00\x00"
        result = analyzer._parse_connack(data)

        assert result["flags"]["session_present"] is False
        assert result["return_code"] == 0

    def test_parse_connack_session_present(self) -> None:
        """Test parsing CONNACK with session present."""
        analyzer = MQTTAnalyzer()
        data = b"\x01\x00"
        result = analyzer._parse_connack(data)

        assert result["flags"]["session_present"] is True

    def test_parse_connack_refused(self) -> None:
        """Test parsing CONNACK with connection refused."""
        analyzer = MQTTAnalyzer()
        # Return code 5 = not authorized
        data = b"\x00\x05"
        result = analyzer._parse_connack(data)

        assert result["return_code"] == 5


class TestDisconnectParsing:
    """Test DISCONNECT packet parsing."""

    def test_parse_disconnect_mqtt311(self) -> None:
        """Test parsing MQTT 3.1.1 DISCONNECT (empty)."""
        analyzer = MQTTAnalyzer()
        result = analyzer._parse_disconnect(b"")
        assert result == {}

    def test_parse_disconnect_mqtt50_with_reason(self) -> None:
        """Test parsing MQTT 5.0 DISCONNECT with reason code."""
        analyzer = MQTTAnalyzer()
        data = b"\x00"  # Reason code 0 (normal disconnection)
        result = analyzer._parse_disconnect(data)

        assert result["reason_code"] == 0


class TestFullPacketParsing:
    """Test complete packet parsing through parse_packet()."""

    def test_parse_connect_packet_full(self) -> None:
        """Test parsing complete CONNECT packet."""
        analyzer = MQTTAnalyzer()
        # Fixed header + variable header
        data = b"\x10\x10\x00\x04MQTT\x04\x02\x00\x3c\x00\x04test"
        packet = analyzer.parse_packet(data, timestamp=1.0)

        assert packet.timestamp == 1.0
        assert packet.packet_type == "CONNECT"
        assert packet.protocol_version == "3.1.1"
        assert len(analyzer.packets) == 1
        assert "test" in analyzer.sessions

    def test_parse_publish_packet_full(self) -> None:
        """Test parsing complete PUBLISH packet."""
        analyzer = MQTTAnalyzer()
        # PUBLISH QoS 0, topic "test", payload "data"
        data = b"\x30\x0a\x00\x04testdata"
        packet = analyzer.parse_packet(data, timestamp=2.0)

        assert packet.packet_type == "PUBLISH"
        assert packet.topic == "test"
        assert packet.payload == b"data"
        assert packet.qos == 0
        assert "test" in analyzer.topics

    def test_parse_subscribe_packet_full(self) -> None:
        """Test parsing complete SUBSCRIBE packet."""
        analyzer = MQTTAnalyzer()
        # SUBSCRIBE packet
        data = b"\x82\x0b\x00\x01\x00\x06sensor\x01"
        packet = analyzer.parse_packet(data)

        assert packet.packet_type == "SUBSCRIBE"
        assert "sensor" in analyzer.topics

    def test_parse_pingreq_packet(self) -> None:
        """Test parsing PINGREQ packet."""
        analyzer = MQTTAnalyzer()
        data = b"\xc0\x00"
        packet = analyzer.parse_packet(data)

        assert packet.packet_type == "PINGREQ"
        assert packet.payload == b""

    def test_parse_pingresp_packet(self) -> None:
        """Test parsing PINGRESP packet."""
        analyzer = MQTTAnalyzer()
        data = b"\xd0\x00"
        packet = analyzer.parse_packet(data)

        assert packet.packet_type == "PINGRESP"

    def test_parse_unknown_packet_type(self) -> None:
        """Test error on unknown packet type."""
        analyzer = MQTTAnalyzer()
        # Packet type 0 is reserved
        data = b"\x00\x00"
        with pytest.raises(ValueError, match="Unknown packet type"):
            analyzer.parse_packet(data)

    def test_parse_insufficient_data(self) -> None:
        """Test error on insufficient packet data."""
        analyzer = MQTTAnalyzer()
        with pytest.raises(ValueError, match="Insufficient data"):
            analyzer.parse_packet(b"\x10")

    def test_parse_incomplete_packet(self) -> None:
        """Test error when packet is incomplete."""
        analyzer = MQTTAnalyzer()
        # Says length is 16 but only provides 5 bytes
        data = b"\x10\x10\x00\x04M"
        with pytest.raises(ValueError, match="Incomplete packet data"):
            analyzer.parse_packet(data)


class TestSessionTracking:
    """Test MQTT session tracking."""

    def test_track_session_from_connect(self) -> None:
        """Test session creation from CONNECT packet."""
        analyzer = MQTTAnalyzer()
        connect_data = {
            "client_id": "test_client",
            "username": "admin",
            "protocol_version": "3.1.1",
            "keep_alive": 60,
            "flags": {"clean_session": True},
            "will_topic": "status/lwt",
            "will_message": b"offline",
        }

        analyzer._track_session(connect_data)

        assert "test_client" in analyzer.sessions
        session = analyzer.sessions["test_client"]
        assert session.client_id == "test_client"
        assert session.username == "admin"
        assert session.keep_alive == 60
        assert session.will_topic == "status/lwt"

    def test_track_multiple_sessions(self) -> None:
        """Test tracking multiple client sessions."""
        analyzer = MQTTAnalyzer()

        for i in range(3):
            data = {
                "client_id": f"client_{i}",
                "protocol_version": "3.1.1",
                "keep_alive": 60,
                "flags": {"clean_session": True},
            }
            analyzer._track_session(data)

        assert len(analyzer.sessions) == 3
        assert "client_0" in analyzer.sessions
        assert "client_2" in analyzer.sessions


class TestTopicHierarchy:
    """Test topic hierarchy generation."""

    def test_get_topic_hierarchy_single_level(self) -> None:
        """Test hierarchy with single-level topics."""
        analyzer = MQTTAnalyzer()
        analyzer.topics = {"sensor", "status", "control"}

        tree = analyzer.get_topic_hierarchy()

        assert "sensor" in tree
        assert "status" in tree
        assert "control" in tree

    def test_get_topic_hierarchy_multilevel(self) -> None:
        """Test hierarchy with multi-level topics."""
        analyzer = MQTTAnalyzer()
        analyzer.topics = {
            "home/sensor/temperature",
            "home/sensor/humidity",
            "home/control/light",
        }

        tree = analyzer.get_topic_hierarchy()

        assert "home" in tree
        assert "sensor" in tree["home"]
        assert "control" in tree["home"]
        assert "temperature" in tree["home"]["sensor"]
        assert "humidity" in tree["home"]["sensor"]
        assert "light" in tree["home"]["control"]

    def test_get_topic_hierarchy_empty(self) -> None:
        """Test hierarchy with no topics."""
        analyzer = MQTTAnalyzer()
        tree = analyzer.get_topic_hierarchy()
        assert tree == {}

    def test_get_topic_hierarchy_mixed_depth(self) -> None:
        """Test hierarchy with topics of varying depth."""
        analyzer = MQTTAnalyzer()
        analyzer.topics = {
            "test",
            "home/sensor",
            "office/floor1/room3/temperature",
        }

        tree = analyzer.get_topic_hierarchy()

        assert "test" in tree
        assert "home" in tree
        assert "office" in tree
        assert "floor1" in tree["office"]


class TestTopologyExport:
    """Test topology export functionality."""

    def test_export_topology_basic(self, tmp_path: Path) -> None:
        """Test exporting topology to JSON file."""
        analyzer = MQTTAnalyzer()
        analyzer.topics = {"test/topic", "sensor/data"}

        connect_data = {
            "client_id": "test",
            "username": "user",
            "protocol_version": "3.1.1",
            "keep_alive": 60,
            "flags": {"clean_session": True},
        }
        analyzer._track_session(connect_data)

        output_file = tmp_path / "topology.json"
        analyzer.export_topology(output_file)

        assert output_file.exists()

        with output_file.open() as f:
            data = json.load(f)

        assert "topic_hierarchy" in data
        assert "topics" in data
        assert "sessions" in data
        assert "packet_count" in data

        assert "test/topic" in data["topics"]
        assert "test" in data["sessions"]
        assert data["sessions"]["test"]["username"] == "user"

    def test_export_topology_with_hierarchy(self, tmp_path: Path) -> None:
        """Test exporting topology with hierarchical structure."""
        analyzer = MQTTAnalyzer()
        analyzer.topics = {"home/sensor/temp", "home/sensor/humidity"}

        output_file = tmp_path / "hierarchy.json"
        analyzer.export_topology(output_file)

        with output_file.open() as f:
            data = json.load(f)

        hierarchy = data["topic_hierarchy"]
        assert "home" in hierarchy
        assert "sensor" in hierarchy["home"]
        assert "temp" in hierarchy["home"]["sensor"]
        assert "humidity" in hierarchy["home"]["sensor"]

    def test_export_topology_empty(self, tmp_path: Path) -> None:
        """Test exporting empty topology."""
        analyzer = MQTTAnalyzer()

        output_file = tmp_path / "empty.json"
        analyzer.export_topology(output_file)

        with output_file.open() as f:
            data = json.load(f)

        assert data["topics"] == []
        assert data["sessions"] == {}
        assert data["packet_count"] == 0


class TestIntegration:
    """Integration tests for complete MQTT analysis workflows."""

    def test_mqtt_conversation_flow(self) -> None:
        """Test analyzing complete MQTT conversation."""
        analyzer = MQTTAnalyzer()

        # Client connects
        connect_pkt = b"\x10\x10\x00\x04MQTT\x04\x02\x00\x3c\x00\x04test"
        packet1 = analyzer.parse_packet(connect_pkt, timestamp=0.0)
        assert packet1.packet_type == "CONNECT"

        # Server acknowledges
        connack_pkt = b"\x20\x02\x00\x00"
        packet2 = analyzer.parse_packet(connack_pkt, timestamp=0.1)
        assert packet2.packet_type == "CONNACK"

        # Client subscribes
        subscribe_pkt = b"\x82\x0b\x00\x01\x00\x06sensor\x01"
        packet3 = analyzer.parse_packet(subscribe_pkt, timestamp=0.2)
        assert packet3.packet_type == "SUBSCRIBE"

        # Server acknowledges subscription
        suback_pkt = b"\x90\x03\x00\x01\x01"
        packet4 = analyzer.parse_packet(suback_pkt, timestamp=0.3)
        assert packet4.packet_type == "SUBACK"

        # Client publishes
        publish_pkt = b"\x30\x0c\x00\x06sensor22.5"
        packet5 = analyzer.parse_packet(publish_pkt, timestamp=1.0)
        assert packet5.packet_type == "PUBLISH"
        assert packet5.topic == "sensor"

        # Verify state
        assert len(analyzer.packets) == 5
        assert "test" in analyzer.sessions
        assert "sensor" in analyzer.topics

    def test_multiple_clients_and_topics(self) -> None:
        """Test analyzing traffic from multiple clients."""
        analyzer = MQTTAnalyzer()

        # Client 1 connects
        # Remaining length = 2+"MQTT"(4)+1+1+2+2+"test"(4) = 16
        connect1 = b"\x10\x10\x00\x04MQTT\x04\x02\x00\x3c\x00\x04test"
        analyzer.parse_packet(connect1)

        # Client 2 connects with different ID
        connect2 = b"\x10\x10\x00\x04MQTT\x04\x02\x00\x3c\x00\x04demo"
        analyzer.parse_packet(connect2)

        # Client 1 publishes to topic1
        pub1 = b"\x30\x0c\x00\x06topic1data1"
        analyzer.parse_packet(pub1)

        # Client 2 publishes to topic2
        pub2 = b"\x30\x0c\x00\x06topic2data2"
        analyzer.parse_packet(pub2)

        assert len(analyzer.sessions) == 2
        assert "test" in analyzer.sessions
        assert "demo" in analyzer.sessions
        assert len(analyzer.topics) == 2
        assert "topic1" in analyzer.topics
        assert "topic2" in analyzer.topics

    def test_qos_levels_handling(self) -> None:
        """Test handling of different QoS levels."""
        analyzer = MQTTAnalyzer()

        # QoS 0 publish
        pub_qos0 = b"\x30\x0c\x00\x06topic0data"
        pkt0 = analyzer.parse_packet(pub_qos0)
        assert pkt0.qos == 0
        assert pkt0.packet_id is None

        # QoS 1 publish
        pub_qos1 = b"\x32\x0e\x00\x06topic1\x00\x01data"
        pkt1 = analyzer.parse_packet(pub_qos1)
        assert pkt1.qos == 1
        assert pkt1.packet_id == 1

        # QoS 2 publish
        pub_qos2 = b"\x34\x0e\x00\x06topic2\x00\x02data"
        pkt2 = analyzer.parse_packet(pub_qos2)
        assert pkt2.qos == 2
        assert pkt2.packet_id == 2


class TestEdgeCasesAndUncoveredPaths:
    """Test edge cases and previously uncovered code paths."""

    def test_parse_auth_packet_empty_data(self) -> None:
        """Test parsing AUTH packet with no reason code."""
        analyzer = MQTTAnalyzer()
        with pytest.raises(ValueError, match="Insufficient data for AUTH"):
            analyzer._parse_auth(b"")

    def test_parse_auth_packet_with_properties(self) -> None:
        """Test parsing AUTH packet with MQTT 5.0 properties."""
        analyzer = MQTTAnalyzer()
        # Reason code 0x00, then property length 0x02, property 0x01 (payload format), value 0x01
        data = b"\x00\x02\x01\x01"
        result = analyzer._parse_auth(data)

        assert result["protocol_version"] == "5.0"
        assert result["reason_code"] == 0
        assert "payload_format_indicator" in result["properties"]

    def test_parse_auth_packet_full(self) -> None:
        """Test parsing complete AUTH packet."""
        analyzer = MQTTAnalyzer()
        # AUTH packet: type=15, flags=0, length=1 (just reason code)
        data = b"\xf0\x01\x00"
        packet = analyzer.parse_packet(data)

        assert packet.packet_type == "AUTH"
        assert packet.protocol_version == "5.0"

    def test_parse_mqtt_string_insufficient_length(self) -> None:
        """Test _parse_mqtt_string with incomplete length field."""
        analyzer = MQTTAnalyzer()
        with pytest.raises(ValueError, match="Incomplete string length field"):
            analyzer._parse_mqtt_string(b"\x00", 0)

    def test_parse_mqtt_string_insufficient_data(self) -> None:
        """Test _parse_mqtt_string with incomplete string data."""
        analyzer = MQTTAnalyzer()
        # Says length is 10 but only has 2 bytes
        with pytest.raises(ValueError, match="Incomplete string data"):
            analyzer._parse_mqtt_string(b"\x00\x0aHi", 0)

    def test_parse_mqtt_binary_insufficient_length(self) -> None:
        """Test _parse_mqtt_binary with incomplete length field."""
        analyzer = MQTTAnalyzer()
        with pytest.raises(ValueError, match="Incomplete binary length field"):
            analyzer._parse_mqtt_binary(b"\x00", 0)

    def test_parse_mqtt_binary_insufficient_data(self) -> None:
        """Test _parse_mqtt_binary with incomplete binary data."""
        analyzer = MQTTAnalyzer()
        # Says length is 5 but only has 2 bytes
        with pytest.raises(ValueError, match="Incomplete binary data"):
            analyzer._parse_mqtt_binary(b"\x00\x05AB", 0)

    def test_parse_connect_insufficient_data_username(self) -> None:
        """Test CONNECT parsing with incomplete username data."""
        analyzer = MQTTAnalyzer()
        # flags: 0x82 (clean session, username flag, but username missing)
        data = b"\x00\x04MQTT\x04\x82\x00\x3c\x00\x04test"
        with pytest.raises(ValueError):
            analyzer._parse_connect(data)

    def test_parse_connect_insufficient_data_password(self) -> None:
        """Test CONNECT parsing with incomplete password data."""
        analyzer = MQTTAnalyzer()
        # flags: 0xC2 (clean session, username, password, but password missing)
        data = b"\x00\x04MQTT\x04\xc2\x00\x3c\x00\x04test\x00\x05admin"
        with pytest.raises(ValueError):
            analyzer._parse_connect(data)

    def test_parse_connect_with_will_qos_retain(self) -> None:
        """Test parsing CONNECT with Will QoS and retain flags."""
        analyzer = MQTTAnalyzer()
        # flags: 0x2E (clean session, will flag, will QoS=1, will retain)
        data = (
            b"\x00\x04MQTT\x04\x2e\x00\x3c"
            b"\x00\x04test"  # client ID
            b"\x00\x06status"  # will topic
            b"\x00\x04gone"  # will message
        )
        result = analyzer._parse_connect(data)

        assert result["flags"]["will_flag"] is True
        assert result["flags"]["will_qos"] == 1
        assert result["flags"]["will_retain"] is True
        assert result["will_topic"] == "status"
        assert result["will_message"] == b"gone"

    def test_parse_subscribe_missing_options(self) -> None:
        """Test SUBSCRIBE parsing with missing subscription options."""
        analyzer = MQTTAnalyzer()
        # Packet ID + topic filter without subscription options byte
        data = b"\x00\x01\x00\x04test"
        with pytest.raises(ValueError, match="Missing subscription options"):
            analyzer._parse_subscribe(data)

    def test_parse_subscribe_incomplete_topic_length(self) -> None:
        """Test SUBSCRIBE parsing with incomplete topic length."""
        analyzer = MQTTAnalyzer()
        # Packet ID + incomplete topic filter length
        data = b"\x00\x01\x00"
        with pytest.raises(ValueError, match="Incomplete topic filter"):
            analyzer._parse_subscribe(data)

    def test_parse_unsubscribe_incomplete_topic(self) -> None:
        """Test UNSUBSCRIBE parsing with incomplete topic."""
        analyzer = MQTTAnalyzer()
        # Packet ID + incomplete topic filter
        data = b"\x00\x01\x00\x10test"
        with pytest.raises(ValueError, match="Incomplete topic filter"):
            analyzer._parse_unsubscribe(data)

    def test_parse_suback_insufficient_data(self) -> None:
        """Test SUBACK parsing with insufficient data."""
        analyzer = MQTTAnalyzer()
        with pytest.raises(ValueError, match="Insufficient data for SUBACK"):
            analyzer._parse_suback(b"\x00")

    def test_parse_connack_with_mqtt50_properties(self) -> None:
        """Test CONNACK parsing with MQTT 5.0 properties."""
        analyzer = MQTTAnalyzer()
        # Session present=1, return code=0, property length=2, property 0x01, value 0x01
        data = b"\x01\x00\x02\x01\x01"
        result = analyzer._parse_connack(data)

        assert result["flags"]["session_present"] is True
        assert result["return_code"] == 0
        assert "payload_format_indicator" in result["properties"]

    def test_parse_disconnect_mqtt50_with_properties(self) -> None:
        """Test DISCONNECT parsing with MQTT 5.0 properties."""
        analyzer = MQTTAnalyzer()
        # Reason code 0x00, property length=2, property 0x01, value 0x01
        data = b"\x00\x02\x01\x01"
        result = analyzer._parse_disconnect(data)

        assert result["reason_code"] == 0
        assert "payload_format_indicator" in result["properties"]

    def test_parse_ack_with_mqtt50_properties(self) -> None:
        """Test ACK parsing with MQTT 5.0 reason code and properties."""
        analyzer = MQTTAnalyzer()
        # Packet ID, reason code, property length, property
        data = b"\x00\x01\x00\x02\x01\x01"
        result = analyzer._parse_ack(data, "PUBACK")

        assert result["packet_id"] == 1
        assert result["reason_code"] == 0
        assert "payload_format_indicator" in result["properties"]

    def test_parse_ack_insufficient_data(self) -> None:
        """Test ACK parsing with insufficient data for packet ID."""
        analyzer = MQTTAnalyzer()
        with pytest.raises(ValueError, match="Insufficient data"):
            analyzer._parse_ack(b"\x00", "PUBACK")

    def test_calculate_header_size_multibyte_length(self) -> None:
        """Test _calculate_header_size with multi-byte remaining length."""
        analyzer = MQTTAnalyzer()
        # Fixed header with 3-byte remaining length (0x80 0x80 0x01 = 16384)
        data = b"\x30\x80\x80\x01rest_of_packet"
        header_size = analyzer._calculate_header_size(data)
        assert header_size == 4  # 1 (fixed) + 3 (length bytes)

    def test_parse_connect_mqtt50_with_properties(self) -> None:
        """Test CONNECT parsing with MQTT 5.0 properties."""
        analyzer = MQTTAnalyzer()
        # Protocol level 5 = MQTT 5.0
        # Protocol + level + flags + keep alive + property length + properties + client ID
        data = (
            b"\x00\x04MQTT\x05\x02\x00\x3c"  # MQTT 5.0
            b"\x02\x01\x01"  # Properties: length=2, property 0x01, value 0x01
            b"\x00\x04test"  # Client ID
        )
        result = analyzer._parse_connect(data)

        assert result["protocol_version"] == "5.0"
        assert result["client_id"] == "test"
        assert "payload_format_indicator" in result["properties"]

    def test_parse_connect_mqtt50_with_will_properties(self) -> None:
        """Test CONNECT parsing with MQTT 5.0 will properties."""
        analyzer = MQTTAnalyzer()
        # Protocol level 5, will flag set
        data = (
            b"\x00\x04MQTT\x05\x06\x00\x3c"  # MQTT 5.0, will flag
            b"\x00"  # Connect properties length = 0
            b"\x00\x04test"  # Client ID
            b"\x02\x01\x01"  # Will properties: length=2, property 0x01, value 0x01
            b"\x00\x06status"  # Will topic
            b"\x00\x07offline"  # Will message
        )
        result = analyzer._parse_connect(data)

        assert result["protocol_version"] == "5.0"
        assert result["will_topic"] == "status"
        assert result["will_message"] == b"offline"

    def test_parse_packet_with_default_timestamp(self) -> None:
        """Test packet parsing with default timestamp."""
        analyzer = MQTTAnalyzer()
        data = b"\x10\x10\x00\x04MQTT\x04\x02\x00\x3c\x00\x04test"
        packet = analyzer.parse_packet(data)  # No timestamp provided
        assert packet.timestamp == 0.0

    def test_parse_publish_insufficient_topic_length_data(self) -> None:
        """Test PUBLISH parsing with incomplete topic length."""
        analyzer = MQTTAnalyzer()
        with pytest.raises(ValueError, match="Insufficient data for topic name length"):
            analyzer._parse_publish(b"\x00", flags=0x00)

    def test_parse_publish_qos0_empty_topic(self) -> None:
        """Test PUBLISH parsing with empty topic (valid)."""
        analyzer = MQTTAnalyzer()
        # Topic length = 0, payload = "data"
        data = b"\x00\x00data"
        result = analyzer._parse_publish(data, flags=0x00)

        assert result["topic"] == ""
        assert result["payload"] == b"data"

    def test_topic_hierarchy_with_trailing_slash(self) -> None:
        """Test topic hierarchy with topics ending in slash."""
        analyzer = MQTTAnalyzer()
        analyzer.topics = {"home/", "home/sensor/"}

        tree = analyzer.get_topic_hierarchy()

        assert "home" in tree
        assert "" in tree["home"]  # Empty string from trailing slash

    def test_session_tracking_missing_username(self) -> None:
        """Test session tracking without username."""
        analyzer = MQTTAnalyzer()
        connect_data = {
            "client_id": "test",
            "protocol_version": "3.1.1",
            "keep_alive": 60,
            "flags": {"clean_session": False},
        }

        analyzer._track_session(connect_data)

        session = analyzer.sessions["test"]
        assert session.username is None
        assert session.clean_session is False

    def test_export_topology_complex_session_data(self, tmp_path: Path) -> None:
        """Test exporting topology with complete session data."""
        from oscura.iot.mqtt import MQTTPacket

        analyzer = MQTTAnalyzer()
        analyzer.topics = {"home/sensor/temp"}

        connect_data = {
            "client_id": "sensor01",
            "username": "admin",
            "protocol_version": "5.0",
            "keep_alive": 120,
            "flags": {"clean_session": False},
            "will_topic": "status/lwt",
            "will_message": b"offline",
        }
        analyzer._track_session(connect_data)

        # Simulate some packets
        pkt1 = MQTTPacket(
            timestamp=0.0,
            packet_type="CONNECT",
            protocol_version="5.0",
            flags={},
        )
        pkt2 = MQTTPacket(
            timestamp=0.1,
            packet_type="CONNACK",
            protocol_version="5.0",
            flags={},
        )
        analyzer.packets.append(pkt1)
        analyzer.packets.append(pkt2)

        output_file = tmp_path / "complex_topology.json"
        analyzer.export_topology(output_file)

        with output_file.open() as f:
            data = json.load(f)

        session = data["sessions"]["sensor01"]
        assert session["username"] == "admin"
        assert session["protocol_version"] == "5.0"
        assert session["keep_alive"] == 120
        assert session["clean_session"] is False
        assert session["will_topic"] == "status/lwt"

    def test_parse_packet_types_all_defined(self) -> None:
        """Test that all MQTT packet types can be parsed."""
        analyzer = MQTTAnalyzer()

        # Test all defined packet types (1-15)
        packet_types_to_test = [
            (1, b"\x10\x10\x00\x04MQTT\x04\x02\x00\x3c\x00\x04test"),  # CONNECT
            (2, b"\x20\x02\x00\x00"),  # CONNACK
            (3, b"\x30\x0a\x00\x04testdata"),  # PUBLISH
            (4, b"\x40\x02\x00\x01"),  # PUBACK
            (5, b"\x50\x02\x00\x01"),  # PUBREC
            (6, b"\x62\x02\x00\x01"),  # PUBREL (must have QoS 1 flag set)
            (7, b"\x70\x02\x00\x01"),  # PUBCOMP
            (8, b"\x82\x09\x00\x01\x00\x04test\x01"),  # SUBSCRIBE (need QoS byte)
            (9, b"\x90\x03\x00\x01\x00"),  # SUBACK
            (10, b"\xa2\x08\x00\x01\x00\x04test"),  # UNSUBSCRIBE
            (11, b"\xb0\x03\x00\x01\x00"),  # UNSUBACK (need return code byte)
            (12, b"\xc0\x00"),  # PINGREQ
            (13, b"\xd0\x00"),  # PINGRESP
            (14, b"\xe0\x00"),  # DISCONNECT
            (15, b"\xf0\x01\x00"),  # AUTH
        ]

        for ptype, data in packet_types_to_test:
            packet = analyzer.parse_packet(data)
            assert packet.packet_type == analyzer.PACKET_TYPES[ptype]
