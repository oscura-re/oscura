"""Tests for DBC generator functionality.

This module tests DBC file generation from DiscoveryDocument and round-trip
validation (generate → parse → should match).
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.automotive]

from pathlib import Path

import pytest

from oscura.automotive.can.discovery import (
    DiscoveryDocument,
    MessageDiscovery,
    SignalDiscovery,
)
from oscura.automotive.can.models import CANMessage
from oscura.automotive.dbc.generator import DBCGenerator

# Skip tests if cantools not installed
pytest.importorskip("cantools")


@pytest.fixture
def simple_discovery_doc() -> DiscoveryDocument:
    """Create simple discovery document with one message and signals.

    Returns:
        DiscoveryDocument with synthetic discovered signals.
    """
    doc = DiscoveryDocument()

    # Create signal discoveries
    rpm_signal = SignalDiscovery(
        name="Engine_RPM",
        start_bit=16,
        length=16,
        byte_order="little_endian",
        value_type="unsigned",
        scale=0.25,
        offset=0.0,
        unit="rpm",
        min_value=0.0,
        max_value=16383.75,
        confidence=0.95,
        evidence=["Counter pattern", "Monotonic increase"],
    )

    temp_signal = SignalDiscovery(
        name="Engine_Temp",
        start_bit=32,
        length=8,
        byte_order="little_endian",
        value_type="unsigned",
        scale=1.0,
        offset=-40.0,
        unit="°C",
        min_value=-40.0,
        max_value=215.0,
        confidence=0.90,
        evidence=["Temperature range pattern"],
    )

    # Create message discovery
    msg = MessageDiscovery(
        id=0x280,
        name="Engine_Data",
        length=8,
        cycle_time_ms=10.0,
        confidence=0.95,
        signals=[rpm_signal, temp_signal],
    )

    doc.add_message(msg)
    return doc


@pytest.fixture
def multi_message_discovery() -> DiscoveryDocument:
    """Create discovery document with multiple messages.

    Returns:
        DiscoveryDocument with multiple messages.
    """
    doc = DiscoveryDocument()

    # Message 1: Engine data
    rpm_signal = SignalDiscovery(
        name="RPM",
        start_bit=0,
        length=16,
        byte_order="big_endian",
        value_type="unsigned",
        scale=0.25,
        offset=0.0,
        unit="rpm",
        confidence=0.95,
        evidence=[],
    )

    msg1 = MessageDiscovery(
        id=0x100,
        name="Engine",
        length=8,
        cycle_time_ms=10.0,
        confidence=0.95,
        signals=[rpm_signal],
    )

    # Message 2: Speed data
    speed_signal = SignalDiscovery(
        name="Vehicle_Speed",
        start_bit=0,
        length=16,
        byte_order="little_endian",
        value_type="unsigned",
        scale=0.01,
        offset=0.0,
        unit="km/h",
        confidence=0.90,
        evidence=[],
    )

    msg2 = MessageDiscovery(
        id=0x200,
        name="Speed",
        length=8,
        cycle_time_ms=20.0,
        confidence=0.90,
        signals=[speed_signal],
    )

    doc.add_message(msg1)
    doc.add_message(msg2)
    return doc


@pytest.fixture
def low_confidence_signals() -> DiscoveryDocument:
    """Create discovery with signals of varying confidence.

    Returns:
        DiscoveryDocument with low and high confidence signals.
    """
    doc = DiscoveryDocument()

    # High confidence signal
    high_conf = SignalDiscovery(
        name="High_Confidence",
        start_bit=0,
        length=8,
        byte_order="little_endian",
        value_type="unsigned",
        scale=1.0,
        offset=0.0,
        unit="",
        confidence=0.95,
        evidence=[],
    )

    # Low confidence signal
    low_conf = SignalDiscovery(
        name="Low_Confidence",
        start_bit=8,
        length=8,
        byte_order="little_endian",
        value_type="unsigned",
        scale=1.0,
        offset=0.0,
        unit="",
        confidence=0.5,
        evidence=[],
    )

    msg = MessageDiscovery(
        id=0x300,
        name="Mixed_Confidence",
        length=8,
        cycle_time_ms=10.0,
        confidence=0.8,
        signals=[high_conf, low_conf],
    )

    doc.add_message(msg)
    return doc


class TestDBCGeneration:
    """Tests for DBC file generation."""

    def test_generate_basic_dbc(self, tmp_path: Path, simple_discovery_doc: DiscoveryDocument):
        """Test generating basic DBC file."""
        output_path = tmp_path / "generated.dbc"

        DBCGenerator.generate(simple_discovery_doc, output_path)

        assert output_path.exists()
        content = output_path.read_text(encoding="latin-1")

        # Check DBC structure
        assert 'VERSION ""' in content
        assert "NS_ :" in content
        assert "BS_:" in content
        assert "BU_:" in content
        assert "BO_ 640 Engine_Data: 8" in content  # 0x280 = 640
        assert "SG_ Engine_RPM" in content
        assert "SG_ Engine_Temp" in content

    def test_generate_with_multiple_messages(
        self, tmp_path: Path, multi_message_discovery: DiscoveryDocument
    ):
        """Test generating DBC with multiple messages."""
        output_path = tmp_path / "multi.dbc"

        DBCGenerator.generate(multi_message_discovery, output_path)

        content = output_path.read_text(encoding="latin-1")

        # Check both messages present
        assert "BO_ 256 Engine: 8" in content  # 0x100 = 256
        assert "BO_ 512 Speed: 8" in content  # 0x200 = 512
        assert "SG_ RPM" in content
        assert "SG_ Vehicle_Speed" in content

    def test_generate_with_min_confidence_filter(
        self, tmp_path: Path, low_confidence_signals: DiscoveryDocument
    ):
        """Test filtering signals by confidence threshold."""
        output_path = tmp_path / "filtered.dbc"

        # Generate with 0.8 threshold - should exclude Low_Confidence signal
        DBCGenerator.generate(low_confidence_signals, output_path, min_confidence=0.8)

        content = output_path.read_text(encoding="latin-1")

        assert "SG_ High_Confidence" in content
        assert "SG_ Low_Confidence" not in content

    def test_generate_includes_all_with_zero_threshold(
        self, tmp_path: Path, low_confidence_signals: DiscoveryDocument
    ):
        """Test that min_confidence=0.0 includes all signals."""
        output_path = tmp_path / "all.dbc"

        DBCGenerator.generate(low_confidence_signals, output_path, min_confidence=0.0)

        content = output_path.read_text(encoding="latin-1")

        assert "SG_ High_Confidence" in content
        assert "SG_ Low_Confidence" in content

    def test_generate_with_comments(self, tmp_path: Path, simple_discovery_doc: DiscoveryDocument):
        """Test generating DBC with evidence comments."""
        output_path = tmp_path / "with_comments.dbc"

        DBCGenerator.generate(simple_discovery_doc, output_path, include_comments=True)

        content = output_path.read_text(encoding="latin-1")

        # Check for evidence in comments
        assert "CM_ SG_" in content
        assert "Counter pattern" in content or "Monotonic increase" in content

    def test_generate_without_comments(
        self, tmp_path: Path, simple_discovery_doc: DiscoveryDocument
    ):
        """Test generating DBC without comments."""
        output_path = tmp_path / "no_comments.dbc"

        DBCGenerator.generate(simple_discovery_doc, output_path, include_comments=False)

        content = output_path.read_text(encoding="latin-1")

        # Should not contain comments
        assert "CM_ SG_" not in content

    def test_generate_signal_format(self, tmp_path: Path, simple_discovery_doc: DiscoveryDocument):
        """Test that generated signal format is correct."""
        output_path = tmp_path / "format_test.dbc"

        DBCGenerator.generate(simple_discovery_doc, output_path)

        content = output_path.read_text(encoding="latin-1")

        # Check signal format: SG_ Name : StartBit|Length@ByteOrder ValueType (Scale,Offset) [Min|Max] "Unit"
        assert "SG_ Engine_RPM : 16|16@1+" in content  # Little-endian, unsigned
        assert "(0.25,0" in content  # Scale and offset (accept both 0 and 0.0)
        assert "[0" in content and "16383.75]" in content  # Min/max
        assert '"rpm"' in content  # Unit

    def test_generate_byte_order_encoding(self, tmp_path: Path):
        """Test correct encoding of big-endian vs little-endian."""
        doc = DiscoveryDocument()

        big_endian = SignalDiscovery(
            name="Big_Endian",
            start_bit=0,
            length=16,
            byte_order="big_endian",
            value_type="unsigned",
            scale=1.0,
            offset=0.0,
            unit="",
            confidence=1.0,
            evidence=[],
        )

        little_endian = SignalDiscovery(
            name="Little_Endian",
            start_bit=16,
            length=16,
            byte_order="little_endian",
            value_type="unsigned",
            scale=1.0,
            offset=0.0,
            unit="",
            confidence=1.0,
            evidence=[],
        )

        msg = MessageDiscovery(
            id=0x400,
            name="Endian_Test",
            length=8,
            cycle_time_ms=10.0,
            confidence=1.0,
            signals=[big_endian, little_endian],
        )
        doc.add_message(msg)

        output_path = tmp_path / "endian.dbc"
        DBCGenerator.generate(doc, output_path)

        content = output_path.read_text(encoding="latin-1")

        # Big-endian should have @0, little-endian @1
        assert "SG_ Big_Endian : 0|16@0+" in content
        assert "SG_ Little_Endian : 16|16@1+" in content

    def test_generate_signed_vs_unsigned(self, tmp_path: Path):
        """Test correct encoding of signed vs unsigned signals."""
        doc = DiscoveryDocument()

        unsigned = SignalDiscovery(
            name="Unsigned",
            start_bit=0,
            length=16,
            byte_order="little_endian",
            value_type="unsigned",
            scale=1.0,
            offset=0.0,
            unit="",
            confidence=1.0,
            evidence=[],
        )

        signed = SignalDiscovery(
            name="Signed",
            start_bit=16,
            length=16,
            byte_order="little_endian",
            value_type="signed",
            scale=1.0,
            offset=0.0,
            unit="",
            confidence=1.0,
            evidence=[],
        )

        msg = MessageDiscovery(
            id=0x500,
            name="Sign_Test",
            length=8,
            cycle_time_ms=10.0,
            confidence=1.0,
            signals=[unsigned, signed],
        )
        doc.add_message(msg)

        output_path = tmp_path / "sign.dbc"
        DBCGenerator.generate(doc, output_path)

        content = output_path.read_text(encoding="latin-1")

        # Unsigned should have +, signed should have -
        assert "@1+" in content  # Unsigned
        assert "@1-" in content  # Signed


class TestRoundTrip:
    """Tests for DBC generation → parsing round-trip."""

    def test_roundtrip_basic(self, tmp_path: Path, simple_discovery_doc: DiscoveryDocument):
        """Test that generated DBC can be parsed back."""
        from oscura.automotive.dbc.parser import DBCParser

        output_path = tmp_path / "roundtrip.dbc"

        # Generate DBC without comments to avoid encoding issues
        DBCGenerator.generate(
            simple_discovery_doc, output_path, min_confidence=0.0, include_comments=False
        )

        # Parse it back
        parser = DBCParser(output_path)

        # Verify messages
        message_ids = parser.get_message_ids()
        assert 0x280 in message_ids

        # Verify message name
        assert parser.get_message_name(0x280) == "Engine_Data"

    def test_roundtrip_decode_message(
        self, tmp_path: Path, simple_discovery_doc: DiscoveryDocument
    ):
        """Test that generated DBC can decode messages correctly."""
        from oscura.automotive.dbc.parser import DBCParser

        output_path = tmp_path / "roundtrip_decode.dbc"

        # Generate DBC with include_comments=False to avoid encoding issues
        DBCGenerator.generate(
            simple_discovery_doc, output_path, min_confidence=0.0, include_comments=False
        )

        # Parse it back
        parser = DBCParser(output_path)

        # Create test message with known values
        # RPM = 2000 (raw = 8000), little-endian at bit 16 (bytes 2-3)
        # Little-endian: 0x1F40 stored as 0x40 0x1F
        data = bytes([0x00, 0x00, 0x40, 0x1F, 0x00, 0x00, 0x00, 0x00])
        msg = CANMessage(arbitration_id=0x280, timestamp=1.0, data=data)

        # Decode
        signals = parser.decode_message(msg)

        assert "Engine_RPM" in signals
        assert abs(signals["Engine_RPM"].value - 2000.0) < 0.1
        assert signals["Engine_RPM"].unit == "rpm"

    def test_roundtrip_preserves_scale_offset(self, tmp_path: Path):
        """Test that scale and offset are preserved in round-trip."""
        from oscura.automotive.dbc.parser import DBCParser

        # Create discovery with specific scale/offset
        doc = DiscoveryDocument()
        signal = SignalDiscovery(
            name="Test_Signal",
            start_bit=0,
            length=16,
            byte_order="little_endian",
            value_type="unsigned",
            scale=0.5,
            offset=-100.0,
            unit="test",
            confidence=1.0,
            evidence=[],
        )
        msg = MessageDiscovery(
            id=0x100,
            name="Test_Msg",
            length=8,
            cycle_time_ms=10.0,
            confidence=1.0,
            signals=[signal],
        )
        doc.add_message(msg)

        output_path = tmp_path / "scale_offset.dbc"

        # Generate and parse
        DBCGenerator.generate(doc, output_path)
        parser = DBCParser(output_path)

        # Create message with raw value 200
        # Expected: 200 * 0.5 - 100 = 0
        data = bytes([200, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
        msg = CANMessage(arbitration_id=0x100, timestamp=1.0, data=data)

        signals = parser.decode_message(msg)
        assert abs(signals["Test_Signal"].value - 0.0) < 0.1


class TestEdgeCases:
    """Tests for edge cases in DBC generation."""

    def test_empty_discovery_document(self, tmp_path: Path):
        """Test generating DBC from empty discovery document."""
        doc = DiscoveryDocument()
        output_path = tmp_path / "empty.dbc"

        DBCGenerator.generate(doc, output_path)

        content = output_path.read_text(encoding="latin-1")
        # Should have headers but no messages
        assert 'VERSION ""' in content
        assert "BO_" not in content

    def test_message_with_no_signals_after_filtering(self, tmp_path: Path):
        """Test that messages with no high-confidence signals are excluded."""
        doc = DiscoveryDocument()

        # All signals have low confidence
        low_conf = SignalDiscovery(
            name="Low",
            start_bit=0,
            length=8,
            byte_order="little_endian",
            value_type="unsigned",
            scale=1.0,
            offset=0.0,
            unit="",
            confidence=0.3,
            evidence=[],
        )

        msg = MessageDiscovery(
            id=0x600,
            name="Low_Conf_Msg",
            length=8,
            cycle_time_ms=10.0,
            confidence=1.0,
            signals=[low_conf],
        )
        doc.add_message(msg)

        output_path = tmp_path / "filtered_out.dbc"

        # Generate with high threshold
        DBCGenerator.generate(doc, output_path, min_confidence=0.8)

        content = output_path.read_text(encoding="latin-1")
        # Message should not appear
        assert "Low_Conf_Msg" not in content

    def test_special_characters_in_names(self, tmp_path: Path):
        """Test handling of special characters in signal/message names."""
        doc = DiscoveryDocument()

        signal = SignalDiscovery(
            name="Signal_With_Underscores_123",
            start_bit=0,
            length=8,
            byte_order="little_endian",
            value_type="unsigned",
            scale=1.0,
            offset=0.0,
            unit="",
            confidence=1.0,
            evidence=[],
        )

        msg = MessageDiscovery(
            id=0x700,
            name="Message_With_Underscores",
            length=8,
            cycle_time_ms=10.0,
            confidence=1.0,
            signals=[signal],
        )
        doc.add_message(msg)

        output_path = tmp_path / "special_chars.dbc"
        DBCGenerator.generate(doc, output_path)

        content = output_path.read_text(encoding="latin-1")
        assert "Message_With_Underscores" in content
        assert "Signal_With_Underscores_123" in content
