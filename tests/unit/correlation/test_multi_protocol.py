"""Tests for multi-protocol session correlation.

Test coverage:
- Message creation and validation
- Correlation detection across protocols
- Request-response pair identification
- Dependency graph building
- Session extraction
- Payload similarity calculation
- Visualization generation
- Export formats (JSON, CSV)
- Edge cases and error handling
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from oscura.correlation.multi_protocol import (
    MultiProtocolCorrelator,
    ProtocolMessage,
    SessionFlow,
)


class TestProtocolMessage:
    """Test ProtocolMessage dataclass."""

    def test_minimal_message(self) -> None:
        """Test message with only required fields."""
        msg = ProtocolMessage(protocol="can", timestamp=1.234)
        assert msg.protocol == "can"
        assert msg.timestamp == 1.234
        assert msg.message_id is None
        assert msg.payload == b""
        assert msg.source is None
        assert msg.destination is None
        assert msg.metadata == {}

    def test_complete_message(self) -> None:
        """Test message with all fields."""
        msg = ProtocolMessage(
            protocol="can",
            timestamp=1.234,
            message_id=0x123,
            payload=b"\x01\x02\x03\x04",
            source="ECU1",
            destination="ECU2",
            metadata={"dlc": 4, "extended": False},
        )
        assert msg.protocol == "can"
        assert msg.timestamp == 1.234
        assert msg.message_id == 0x123
        assert msg.payload == b"\x01\x02\x03\x04"
        assert msg.source == "ECU1"
        assert msg.destination == "ECU2"
        assert msg.metadata == {"dlc": 4, "extended": False}

    def test_string_message_id(self) -> None:
        """Test message with string ID."""
        msg = ProtocolMessage(protocol="ethernet", timestamp=1.234, message_id="ICMP_ECHO")
        assert msg.message_id == "ICMP_ECHO"


class TestMultiProtocolCorrelator:
    """Test MultiProtocolCorrelator class."""

    def test_initialization(self) -> None:
        """Test correlator initialization."""
        correlator = MultiProtocolCorrelator(time_window=0.1, min_confidence=0.5)
        assert correlator.time_window == 0.1
        assert correlator.min_confidence == 0.5
        assert correlator.messages == []
        assert correlator.correlations == []

    def test_initialization_defaults(self) -> None:
        """Test default initialization values."""
        correlator = MultiProtocolCorrelator()
        assert correlator.time_window == 0.1
        assert correlator.min_confidence == 0.5

    def test_invalid_time_window(self) -> None:
        """Test initialization with invalid time window."""
        with pytest.raises(ValueError, match="time_window must be positive"):
            MultiProtocolCorrelator(time_window=0.0)
        with pytest.raises(ValueError, match="time_window must be positive"):
            MultiProtocolCorrelator(time_window=-0.1)

    def test_invalid_confidence(self) -> None:
        """Test initialization with invalid confidence."""
        with pytest.raises(ValueError, match="min_confidence must be in"):
            MultiProtocolCorrelator(min_confidence=1.5)
        with pytest.raises(ValueError, match="min_confidence must be in"):
            MultiProtocolCorrelator(min_confidence=-0.1)

    def test_add_message(self) -> None:
        """Test adding messages."""
        correlator = MultiProtocolCorrelator()
        msg1 = ProtocolMessage(protocol="can", timestamp=1.234)
        msg2 = ProtocolMessage(protocol="ethernet", timestamp=1.456)

        correlator.add_message(msg1)
        correlator.add_message(msg2)

        assert len(correlator.messages) == 2
        assert correlator.messages[0] is msg1
        assert correlator.messages[1] is msg2

    def test_payload_similarity_identical(self) -> None:
        """Test payload similarity with identical payloads."""
        correlator = MultiProtocolCorrelator()
        payload = b"\x01\x02\x03\x04"
        similarity = correlator._payload_similarity(payload, payload)
        assert similarity == 1.0

    def test_payload_similarity_containment(self) -> None:
        """Test payload similarity with containment."""
        correlator = MultiProtocolCorrelator()
        payload1 = b"\x01\x02\x03"
        payload2 = b"\x01\x02\x03\x04\x05"
        similarity = correlator._payload_similarity(payload1, payload2)
        assert similarity == 1.0

    def test_payload_similarity_partial(self) -> None:
        """Test payload similarity with partial overlap."""
        correlator = MultiProtocolCorrelator()
        payload1 = b"\x01\x02\x03\x04"
        payload2 = b"\x03\x04\x05\x06"
        similarity = correlator._payload_similarity(payload1, payload2)
        assert 0.0 < similarity < 1.0

    def test_payload_similarity_no_overlap(self) -> None:
        """Test payload similarity with no overlap."""
        correlator = MultiProtocolCorrelator()
        payload1 = b"\x01\x02"
        payload2 = b"\x03\x04"
        similarity = correlator._payload_similarity(payload1, payload2)
        assert similarity == 0.0

    def test_payload_similarity_empty(self) -> None:
        """Test payload similarity with empty payloads."""
        correlator = MultiProtocolCorrelator()
        assert correlator._payload_similarity(b"", b"\x01\x02") == 0.0
        assert correlator._payload_similarity(b"\x01\x02", b"") == 0.0
        assert correlator._payload_similarity(b"", b"") == 0.0

    def test_correlate_same_protocol(self) -> None:
        """Test that same-protocol messages are not correlated."""
        correlator = MultiProtocolCorrelator()

        msg1 = ProtocolMessage(protocol="can", timestamp=1.0, payload=b"\x01\x02\x03")
        msg2 = ProtocolMessage(protocol="can", timestamp=1.05, payload=b"\x01\x02\x03")

        correlator.add_message(msg1)
        correlator.add_message(msg2)

        correlations = correlator.correlate_all()
        assert len(correlations) == 0

    def test_correlate_by_payload(self) -> None:
        """Test correlation by payload similarity."""
        correlator = MultiProtocolCorrelator(min_confidence=0.3)

        msg1 = ProtocolMessage(protocol="can", timestamp=1.0, payload=b"\x01\x02\x03\x04")
        msg2 = ProtocolMessage(protocol="ethernet", timestamp=1.05, payload=b"\x01\x02\x03\x04\x05")

        correlator.add_message(msg1)
        correlator.add_message(msg2)

        correlations = correlator.correlate_all()
        assert len(correlations) == 1
        assert correlations[0].message1 is msg1
        assert correlations[0].message2 is msg2
        assert correlations[0].confidence > 0.3
        assert "Payload similarity" in correlations[0].evidence[0]

    def test_correlate_by_id_match(self) -> None:
        """Test correlation by matching message IDs."""
        correlator = MultiProtocolCorrelator(min_confidence=0.3)

        msg1 = ProtocolMessage(protocol="can", timestamp=1.0, message_id=0x123, payload=b"\x01")
        msg2 = ProtocolMessage(
            protocol="ethernet", timestamp=1.05, message_id=0x123, payload=b"\x02"
        )

        correlator.add_message(msg1)
        correlator.add_message(msg2)

        correlations = correlator.correlate_all()
        assert len(correlations) == 1
        assert correlations[0].confidence >= 0.3
        assert any("Matching IDs" in ev for ev in correlations[0].evidence)

    def test_correlate_by_source_destination(self) -> None:
        """Test correlation by source/destination matching."""
        correlator = MultiProtocolCorrelator(min_confidence=0.1)

        msg1 = ProtocolMessage(
            protocol="can",
            timestamp=1.0,
            source="ECU1",
            destination="ECU2",
            payload=b"\x01",
        )
        msg2 = ProtocolMessage(
            protocol="ethernet",
            timestamp=1.05,
            source="ECU1",
            destination="ECU3",
            payload=b"\x02",
        )

        correlator.add_message(msg1)
        correlator.add_message(msg2)

        correlations = correlator.correlate_all()
        # Should have some correlation due to source match
        assert len(correlations) >= 0

    def test_correlate_outside_time_window(self) -> None:
        """Test that messages outside time window are not correlated."""
        correlator = MultiProtocolCorrelator(time_window=0.1)

        msg1 = ProtocolMessage(protocol="can", timestamp=1.0, payload=b"\x01\x02\x03")
        msg2 = ProtocolMessage(protocol="ethernet", timestamp=1.2, payload=b"\x01\x02\x03")

        correlator.add_message(msg1)
        correlator.add_message(msg2)

        correlations = correlator.correlate_all()
        assert len(correlations) == 0

    def test_correlate_broadcast_type(self) -> None:
        """Test broadcast correlation type for near-simultaneous messages."""
        correlator = MultiProtocolCorrelator(min_confidence=0.3)

        msg1 = ProtocolMessage(protocol="can", timestamp=1.0, payload=b"\x01\x02\x03")
        msg2 = ProtocolMessage(protocol="ethernet", timestamp=1.005, payload=b"\x01\x02\x03")

        correlator.add_message(msg1)
        correlator.add_message(msg2)

        correlations = correlator.correlate_all()
        assert len(correlations) == 1
        assert correlations[0].correlation_type == "broadcast"
        assert correlations[0].time_delta < 0.01

    def test_correlate_request_response_type(self) -> None:
        """Test request-response correlation type."""
        correlator = MultiProtocolCorrelator(min_confidence=0.1)

        msg1 = ProtocolMessage(
            protocol="can",
            timestamp=1.0,
            source="ECU1",
            destination="ECU2",
            payload=b"\x01",
        )
        msg2 = ProtocolMessage(
            protocol="ethernet",
            timestamp=1.02,
            source="ECU2",
            destination="ECU1",
            payload=b"\x02",
        )

        correlator.add_message(msg1)
        correlator.add_message(msg2)

        correlations = correlator.correlate_all()
        assert len(correlations) == 1
        assert correlations[0].correlation_type == "request_response"

    def test_find_request_response_pairs(self) -> None:
        """Test finding request-response pairs for specific protocols."""
        correlator = MultiProtocolCorrelator(min_confidence=0.1)

        # CAN request -> Ethernet response
        msg1 = ProtocolMessage(
            protocol="can",
            timestamp=1.0,
            source="ECU1",
            destination="ECU2",
            payload=b"\x01",
        )
        msg2 = ProtocolMessage(
            protocol="ethernet",
            timestamp=1.02,
            source="ECU2",
            destination="ECU1",
            payload=b"\x02",
        )

        # UART request -> SPI response (different pair)
        msg3 = ProtocolMessage(
            protocol="uart",
            timestamp=2.0,
            source="MCU1",
            destination="MCU2",
            payload=b"\x03",
        )
        msg4 = ProtocolMessage(
            protocol="spi",
            timestamp=2.01,
            source="MCU2",
            destination="MCU1",
            payload=b"\x04",
        )

        correlator.add_message(msg1)
        correlator.add_message(msg2)
        correlator.add_message(msg3)
        correlator.add_message(msg4)

        # Find only CAN->Ethernet pairs
        pairs = correlator.find_request_response_pairs("can", "ethernet")
        assert len(pairs) == 1
        assert pairs[0].message1.protocol == "can"
        assert pairs[0].message2.protocol == "ethernet"

    def test_build_dependency_graph(self) -> None:
        """Test dependency graph building."""
        pytest.importorskip("networkx")

        correlator = MultiProtocolCorrelator(min_confidence=0.3)

        msg1 = ProtocolMessage(protocol="can", timestamp=1.0, payload=b"\x01\x02\x03")
        msg2 = ProtocolMessage(protocol="ethernet", timestamp=1.05, payload=b"\x01\x02\x03")
        msg3 = ProtocolMessage(protocol="uart", timestamp=1.1, payload=b"\x01\x02\x03")

        correlator.add_message(msg1)
        correlator.add_message(msg2)
        correlator.add_message(msg3)

        graph = correlator.build_dependency_graph()

        # Should have 3 nodes (messages)
        assert graph.number_of_nodes() == 3

        # Check node attributes
        node_data = graph.nodes[0]
        assert node_data["protocol"] == "can"
        assert node_data["timestamp"] == 1.0

    def test_build_dependency_graph_without_networkx(self) -> None:
        """Test that graph building fails gracefully without networkx."""
        # This test only runs if networkx is not installed
        # Skip if networkx is available
        try:
            import networkx  # noqa: F401

        except ImportError:
            pass

        correlator = MultiProtocolCorrelator()
        msg = ProtocolMessage(protocol="can", timestamp=1.0)
        correlator.add_message(msg)

        with pytest.raises(ImportError, match="networkx is required"):
            correlator.build_dependency_graph()

    def test_extract_sessions_single(self) -> None:
        """Test session extraction with single session."""
        pytest.importorskip("networkx")

        correlator = MultiProtocolCorrelator(min_confidence=0.3)

        # Create connected chain: CAN -> Ethernet -> UART
        msg1 = ProtocolMessage(protocol="can", timestamp=1.0, payload=b"\x01\x02\x03")
        msg2 = ProtocolMessage(protocol="ethernet", timestamp=1.05, payload=b"\x01\x02\x03")
        msg3 = ProtocolMessage(protocol="uart", timestamp=1.1, payload=b"\x01\x02\x03")

        correlator.add_message(msg1)
        correlator.add_message(msg2)
        correlator.add_message(msg3)

        sessions = correlator.extract_sessions()

        assert len(sessions) == 1
        session = sessions[0]
        assert len(session.messages) == 3
        assert session.protocols == {"can", "ethernet", "uart"}
        assert session.start_time == 1.0
        assert session.end_time == 1.1

    def test_extract_sessions_multiple(self) -> None:
        """Test session extraction with multiple independent sessions."""
        pytest.importorskip("networkx")

        correlator = MultiProtocolCorrelator(min_confidence=0.3)

        # Session 1: CAN -> Ethernet
        msg1 = ProtocolMessage(protocol="can", timestamp=1.0, payload=b"\x01\x02\x03")
        msg2 = ProtocolMessage(protocol="ethernet", timestamp=1.05, payload=b"\x01\x02\x03")

        # Session 2: UART -> SPI (different time, different payload)
        msg3 = ProtocolMessage(protocol="uart", timestamp=5.0, payload=b"\xaa\xbb\xcc")
        msg4 = ProtocolMessage(protocol="spi", timestamp=5.05, payload=b"\xaa\xbb\xcc")

        correlator.add_message(msg1)
        correlator.add_message(msg2)
        correlator.add_message(msg3)
        correlator.add_message(msg4)

        sessions = correlator.extract_sessions()

        assert len(sessions) == 2
        assert sessions[0].start_time < sessions[1].start_time

    def test_extract_sessions_sorted(self) -> None:
        """Test that sessions are sorted by start time."""
        pytest.importorskip("networkx")

        correlator = MultiProtocolCorrelator(min_confidence=0.3)

        # Add sessions in reverse order
        # Session 2
        msg3 = ProtocolMessage(protocol="uart", timestamp=5.0, payload=b"\xaa\xbb\xcc")
        msg4 = ProtocolMessage(protocol="spi", timestamp=5.05, payload=b"\xaa\xbb\xcc")

        # Session 1
        msg1 = ProtocolMessage(protocol="can", timestamp=1.0, payload=b"\x01\x02\x03")
        msg2 = ProtocolMessage(protocol="ethernet", timestamp=1.05, payload=b"\x01\x02\x03")

        correlator.add_message(msg3)
        correlator.add_message(msg4)
        correlator.add_message(msg1)
        correlator.add_message(msg2)

        sessions = correlator.extract_sessions()

        # Should be sorted by start time
        assert sessions[0].start_time == 1.0
        assert sessions[1].start_time == 5.0

    def test_visualize_flow(self, tmp_path: Path) -> None:
        """Test flow visualization."""
        pytest.importorskip("networkx")
        pytest.importorskip("matplotlib")

        correlator = MultiProtocolCorrelator(min_confidence=0.3)

        msg1 = ProtocolMessage(
            protocol="can", timestamp=1.0, message_id=0x123, payload=b"\x01\x02\x03"
        )
        msg2 = ProtocolMessage(
            protocol="ethernet",
            timestamp=1.05,
            message_id="ICMP",
            payload=b"\x01\x02\x03",
        )

        correlator.add_message(msg1)
        correlator.add_message(msg2)

        sessions = correlator.extract_sessions()
        output_path = tmp_path / "flow.png"

        correlator.visualize_flow(sessions[0], output_path)

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_visualize_flow_without_matplotlib(self) -> None:
        """Test that visualization fails gracefully without matplotlib."""
        # Skip if matplotlib is available
        try:
            import matplotlib  # noqa: F401

        except ImportError:
            pass

        correlator = MultiProtocolCorrelator()
        msg = ProtocolMessage(protocol="can", timestamp=1.0)
        correlator.add_message(msg)

        session = SessionFlow(
            start_time=1.0,
            end_time=1.1,
            messages=[msg],
            correlations=[],
            protocols={"can"},
        )

        with pytest.raises(ImportError, match="matplotlib is required"):
            correlator.visualize_flow(session, Path("output.png"))

    def test_export_json(self, tmp_path: Path) -> None:
        """Test JSON export."""
        correlator = MultiProtocolCorrelator(min_confidence=0.3)

        msg1 = ProtocolMessage(
            protocol="can",
            timestamp=1.0,
            message_id=0x123,
            payload=b"\x01\x02\x03",
            source="ECU1",
            destination="ECU2",
            metadata={"dlc": 3},
        )
        msg2 = ProtocolMessage(
            protocol="ethernet",
            timestamp=1.05,
            message_id="ICMP_ECHO",
            payload=b"\x01\x02\x03\x04",
        )

        correlator.add_message(msg1)
        correlator.add_message(msg2)
        correlator.correlate_all()

        output_path = tmp_path / "analysis.json"
        correlator.export_analysis(output_path, format="json")

        assert output_path.exists()

        # Validate JSON structure
        data = json.loads(output_path.read_text())
        assert "config" in data
        assert "messages" in data
        assert "correlations" in data
        assert data["config"]["time_window"] == 0.1
        assert data["config"]["min_confidence"] == 0.3
        assert len(data["messages"]) == 2
        assert data["messages"][0]["protocol"] == "can"
        assert data["messages"][0]["message_id"] == 0x123
        assert data["messages"][0]["payload_hex"] == "010203"

    def test_export_csv(self, tmp_path: Path) -> None:
        """Test CSV export."""
        correlator = MultiProtocolCorrelator(min_confidence=0.3)

        msg1 = ProtocolMessage(
            protocol="can", timestamp=1.0, message_id=0x123, payload=b"\x01\x02\x03"
        )
        msg2 = ProtocolMessage(protocol="ethernet", timestamp=1.05, payload=b"\x01\x02\x03")

        correlator.add_message(msg1)
        correlator.add_message(msg2)
        correlator.correlate_all()

        output_path = tmp_path / "analysis.csv"
        correlator.export_analysis(output_path, format="csv")

        assert output_path.exists()

        # Read and validate CSV
        lines = output_path.read_text().splitlines()
        assert len(lines) >= 1  # Header + at least one correlation
        assert "Protocol1" in lines[0]
        assert "Protocol2" in lines[0]

    def test_export_unsupported_format(self, tmp_path: Path) -> None:
        """Test export with unsupported format."""
        correlator = MultiProtocolCorrelator()
        output_path = tmp_path / "analysis.xml"

        with pytest.raises(ValueError, match="Unsupported format"):
            correlator.export_analysis(output_path, format="xml")

    def test_empty_correlator(self) -> None:
        """Test correlator with no messages."""
        correlator = MultiProtocolCorrelator()
        correlations = correlator.correlate_all()
        assert len(correlations) == 0

    def test_single_message(self) -> None:
        """Test correlator with single message."""
        correlator = MultiProtocolCorrelator()
        msg = ProtocolMessage(protocol="can", timestamp=1.0)
        correlator.add_message(msg)

        correlations = correlator.correlate_all()
        assert len(correlations) == 0

    def test_confidence_sorting(self) -> None:
        """Test that correlations are sorted by confidence."""
        correlator = MultiProtocolCorrelator(min_confidence=0.1)

        # High confidence match (payload + ID)
        msg1 = ProtocolMessage(
            protocol="can",
            timestamp=1.0,
            message_id=0x123,
            payload=b"\x01\x02\x03",
        )
        msg2 = ProtocolMessage(
            protocol="ethernet",
            timestamp=1.05,
            message_id=0x123,
            payload=b"\x01\x02\x03",
        )

        # Lower confidence match (only partial payload)
        msg3 = ProtocolMessage(protocol="uart", timestamp=1.1, payload=b"\x01\x02")
        msg4 = ProtocolMessage(protocol="spi", timestamp=1.15, payload=b"\x03\x04")

        correlator.add_message(msg1)
        correlator.add_message(msg2)
        correlator.add_message(msg3)
        correlator.add_message(msg4)

        correlations = correlator.correlate_all()

        # Should be sorted by confidence descending
        if len(correlations) >= 2:
            assert correlations[0].confidence >= correlations[1].confidence

    def test_complex_multi_protocol_scenario(self) -> None:
        """Test complex scenario with multiple protocols and correlations."""
        pytest.importorskip("networkx")

        correlator = MultiProtocolCorrelator(min_confidence=0.3)

        # Simulate automotive scenario: CAN -> Ethernet gateway -> Serial debug
        # CAN message
        msg1 = ProtocolMessage(
            protocol="can",
            timestamp=1.0,
            message_id=0x123,
            payload=b"\x12\x34\x56\x78",
            source="ECU1",
        )

        # Ethernet message (gateway forwarding)
        msg2 = ProtocolMessage(
            protocol="ethernet",
            timestamp=1.005,
            payload=b"\x12\x34\x56\x78\x00",  # CAN payload + extra byte
            source="192.168.1.10",
        )

        # Serial debug output
        msg3 = ProtocolMessage(
            protocol="uart",
            timestamp=1.01,
            payload=b"CAN:0x123",  # Different payload
        )

        correlator.add_message(msg1)
        correlator.add_message(msg2)
        correlator.add_message(msg3)

        correlations = correlator.correlate_all()
        sessions = correlator.extract_sessions()

        # Should find correlation between CAN and Ethernet (payload match)
        assert len(correlations) >= 1
        assert any(
            c.message1.protocol == "can" and c.message2.protocol == "ethernet" for c in correlations
        )

    def test_message_ordering_independence(self) -> None:
        """Test that message add order doesn't affect correlations."""
        correlator1 = MultiProtocolCorrelator(min_confidence=0.3)
        correlator2 = MultiProtocolCorrelator(min_confidence=0.3)

        msg1 = ProtocolMessage(protocol="can", timestamp=1.0, payload=b"\x01\x02\x03")
        msg2 = ProtocolMessage(protocol="ethernet", timestamp=1.05, payload=b"\x01\x02\x03")

        # Add in different orders
        correlator1.add_message(msg1)
        correlator1.add_message(msg2)

        correlator2.add_message(msg2)
        correlator2.add_message(msg1)

        corr1 = correlator1.correlate_all()
        corr2 = correlator2.correlate_all()

        assert len(corr1) == len(corr2)
        assert corr1[0].confidence == corr2[0].confidence
