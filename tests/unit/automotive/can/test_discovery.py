"""Comprehensive test suite for CAN discovery documentation (.tkcan format).

Tests cover SignalDiscovery, MessageDiscovery, Hypothesis tracking,
and full discovery document save/load workflows.
"""

from __future__ import annotations

from datetime import datetime

import pytest

# Module under test
try:
    from oscura.automotive.can.discovery import (
        DiscoveryDocument,
        Hypothesis,
        MessageDiscovery,
        SignalDiscovery,
        VehicleInfo,
    )
    from oscura.automotive.can.models import SignalDefinition

    HAS_CAN = True
except ImportError:
    HAS_CAN = False

pytestmark = pytest.mark.skipif(not HAS_CAN, reason="CAN modules not available")


# ============================================================================
# Signal Discovery Tests
# ============================================================================


def test_signal_discovery_creation():
    """Test creating SignalDiscovery instance."""
    signal = SignalDiscovery(
        name="EngineRPM",
        start_bit=0,
        length=16,
        byte_order="big_endian",
        value_type="unsigned",
        scale=0.25,
        offset=0.0,
        unit="rpm",
        min_value=0.0,
        max_value=8000.0,
        confidence=0.95,
        evidence=["Confirmed by bench test", "Matches expected range"],
        comment="Engine speed signal",
    )

    assert signal.name == "EngineRPM"
    assert signal.length == 16
    assert signal.confidence == 0.95
    assert len(signal.evidence) == 2


def test_signal_discovery_from_definition():
    """Test creating SignalDiscovery from SignalDefinition."""
    definition = SignalDefinition(
        name="VehicleSpeed",
        start_bit=8,
        length=8,
        byte_order="big_endian",
        value_type="unsigned",
        scale=1.0,
        offset=0.0,
        unit="km/h",
    )

    signal = SignalDiscovery.from_definition(definition, confidence=0.9, evidence=["Test evidence"])

    assert signal.name == "VehicleSpeed"
    assert signal.start_bit == 8
    assert signal.confidence == 0.9
    assert signal.evidence == ["Test evidence"]


def test_signal_discovery_to_dict():
    """Test SignalDiscovery serialization to dictionary."""
    signal = SignalDiscovery(
        name="TestSignal",
        start_bit=0,
        length=8,
        confidence=0.8,
        evidence=["Evidence 1"],
    )

    data = signal.to_dict()

    assert data["name"] == "TestSignal"
    assert data["start_bit"] == 0
    assert data["length"] == 8
    assert data["confidence"] == 0.8


# ============================================================================
# Message Discovery Tests
# ============================================================================


def test_message_discovery_creation():
    """Test creating MessageDiscovery instance."""
    message = MessageDiscovery(
        id=0x100,
        name="EngineData",
        length=8,
        transmitter="ECU",
        cycle_time_ms=10.0,
        confidence=0.95,
        evidence=["Periodic at 100Hz"],
    )

    assert message.id == 0x100
    assert message.name == "EngineData"
    assert message.cycle_time_ms == 10.0


def test_message_discovery_with_signals():
    """Test MessageDiscovery with attached signals."""
    signal1 = SignalDiscovery(name="RPM", start_bit=0, length=16, confidence=0.9)
    signal2 = SignalDiscovery(name="Temp", start_bit=16, length=8, confidence=0.8)

    message = MessageDiscovery(
        id=0x200,
        name="Diagnostics",
        length=8,
        signals=[signal1, signal2],
    )

    assert len(message.signals) == 2
    assert message.signals[0].name == "RPM"


def test_message_discovery_to_dict():
    """Test MessageDiscovery serialization."""
    message = MessageDiscovery(
        id=0x180,
        name="TestMsg",
        length=8,
        confidence=0.85,
    )

    data = message.to_dict()

    assert data["id"] == "0x180"
    assert data["name"] == "TestMsg"
    assert data["length"] == 8


# ============================================================================
# Hypothesis Tests
# ============================================================================


def test_hypothesis_creation():
    """Test creating Hypothesis instance."""
    hyp = Hypothesis(
        message_id=0x100,
        signal="Unknown_Signal_1",
        hypothesis="This signal represents throttle position",
        status="testing",
        test_plan="Monitor during acceleration",
    )

    assert hyp.message_id == 0x100
    assert hyp.status == "testing"
    assert hyp.hypothesis == "This signal represents throttle position"


def test_hypothesis_to_dict():
    """Test Hypothesis serialization."""
    hyp = Hypothesis(
        message_id=0x200,
        signal="Signal_X",
        hypothesis="Test hypothesis",
        status="confirmed",
    )

    data = hyp.to_dict()

    assert data["message_id"] == "0x200"
    assert data["signal"] == "Signal_X"
    assert data["status"] == "confirmed"


def test_hypothesis_timestamps():
    """Test that hypothesis has creation timestamps."""
    hyp = Hypothesis(
        message_id=0x100,
        signal="TestSignal",
        hypothesis="Test",
    )

    # Should have ISO format timestamps
    assert hyp.created is not None
    assert hyp.updated is not None
    # Should be parseable
    datetime.fromisoformat(hyp.created)
    datetime.fromisoformat(hyp.updated)


# ============================================================================
# Vehicle Info Tests
# ============================================================================


def test_vehicle_info_creation():
    """Test creating VehicleInfo instance."""
    info = VehicleInfo(
        make="Toyota",
        model="Camry",
        year="2023",
        vin="1234567890ABCDEFG",
        notes="Test vehicle",
    )

    assert info.make == "Toyota"
    assert info.model == "Camry"
    assert info.year == "2023"


def test_vehicle_info_defaults():
    """Test VehicleInfo with default values."""
    info = VehicleInfo()

    assert info.make == "Unknown"
    assert info.model == "Unknown"
    assert info.year is None


def test_vehicle_info_to_dict():
    """Test VehicleInfo serialization."""
    info = VehicleInfo(make="Honda", model="Civic")

    data = info.to_dict()

    assert data["make"] == "Honda"
    assert data["model"] == "Civic"
    # Should not include None values
    assert "year" not in data or data["year"] is None


# ============================================================================
# Discovery Document Tests
# ============================================================================


def test_discovery_document_creation():
    """Test creating empty DiscoveryDocument."""
    doc = DiscoveryDocument()

    assert doc.format_version == "1.0"
    assert len(doc.messages) == 0
    assert len(doc.hypotheses) == 0


def test_discovery_document_add_message():
    """Test adding messages to discovery document."""
    doc = DiscoveryDocument()

    msg1 = MessageDiscovery(id=0x100, name="Msg1", length=8)
    msg2 = MessageDiscovery(id=0x200, name="Msg2", length=8)

    doc.add_message(msg1)
    doc.add_message(msg2)

    assert len(doc.messages) == 2
    assert 0x100 in doc.messages
    assert 0x200 in doc.messages


def test_discovery_document_add_hypothesis():
    """Test adding hypotheses to discovery document."""
    doc = DiscoveryDocument()

    hyp = Hypothesis(message_id=0x100, signal="TestSig", hypothesis="Test")
    doc.add_hypothesis(hyp)

    assert len(doc.hypotheses) == 1


def test_discovery_document_save_load(tmp_path):
    """Test saving and loading discovery document."""
    # Create document
    doc = DiscoveryDocument()
    doc.vehicle = VehicleInfo(make="Toyota", model="Camry", year="2023")

    signal = SignalDiscovery(
        name="RPM",
        start_bit=0,
        length=16,
        confidence=0.95,
        evidence=["Confirmed"],
    )

    message = MessageDiscovery(
        id=0x280,
        name="EngineData",
        length=8,
        cycle_time_ms=10.0,
        signals=[signal],
    )

    doc.add_message(message)

    hyp = Hypothesis(message_id=0x280, signal="RPM", hypothesis="Engine speed", status="confirmed")
    doc.add_hypothesis(hyp)

    # Save
    file_path = tmp_path / "test.tkcan"
    doc.save(file_path)

    assert file_path.exists()

    # Load
    loaded_doc = DiscoveryDocument.load(file_path)

    assert loaded_doc.vehicle.make == "Toyota"
    assert len(loaded_doc.messages) == 1
    assert 0x280 in loaded_doc.messages
    assert loaded_doc.messages[0x280].name == "EngineData"
    assert len(loaded_doc.messages[0x280].signals) == 1
    assert loaded_doc.messages[0x280].signals[0].name == "RPM"
    assert len(loaded_doc.hypotheses) == 1


def test_discovery_document_save_complex(tmp_path):
    """Test saving complex discovery document with multiple elements."""
    doc = DiscoveryDocument()

    # Add multiple messages
    for i in range(5):
        msg_id = 0x100 + i
        msg = MessageDiscovery(id=msg_id, name=f"Message_{i}", length=8, confidence=0.9)

        # Add signals to each message
        for j in range(3):
            signal = SignalDiscovery(
                name=f"Sig_{i}_{j}",
                start_bit=j * 8,
                length=8,
                confidence=0.8,
            )
            msg.signals.append(signal)

        doc.add_message(msg)

    # Save and load
    file_path = tmp_path / "complex.tkcan"
    doc.save(file_path)

    loaded = DiscoveryDocument.load(file_path)

    assert len(loaded.messages) == 5
    # Check signal count
    total_signals = sum(len(msg.signals) for msg in loaded.messages.values())
    assert total_signals == 15


def test_discovery_document_load_hex_id_formats(tmp_path):
    """Test loading different CAN ID hex formats."""
    import yaml

    # Create YAML with various ID formats
    data = {
        "format_version": "1.0",
        "vehicle": {"make": "Test"},
        "messages": [
            {"id": "0x100", "name": "Msg1", "length": 8},
            {"id": "0x200", "name": "Msg2", "length": 8},
        ],
    }

    file_path = tmp_path / "test_ids.tkcan"
    with open(file_path, "w") as f:
        yaml.safe_dump(data, f)

    doc = DiscoveryDocument.load(file_path)

    assert 0x100 in doc.messages
    assert 0x200 in doc.messages


def test_discovery_document_repr():
    """Test DiscoveryDocument __repr__ method."""
    doc = DiscoveryDocument()

    msg1 = MessageDiscovery(id=0x100, name="Msg1", length=8)
    signal1 = SignalDiscovery(name="Sig1", start_bit=0, length=8)
    msg1.signals.append(signal1)
    doc.add_message(msg1)

    hyp = Hypothesis(message_id=0x100, signal="Sig1", hypothesis="Test")
    doc.add_hypothesis(hyp)

    repr_str = repr(doc)

    assert "1 messages" in repr_str
    assert "1 signals" in repr_str
    assert "1 hypotheses" in repr_str


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================


def test_discovery_document_empty_save_load(tmp_path):
    """Test saving and loading empty discovery document."""
    doc = DiscoveryDocument()
    file_path = tmp_path / "empty.tkcan"

    doc.save(file_path)
    loaded = DiscoveryDocument.load(file_path)

    assert len(loaded.messages) == 0
    assert len(loaded.hypotheses) == 0


def test_discovery_document_overwrite_message():
    """Test that adding message with same ID overwrites."""
    doc = DiscoveryDocument()

    msg1 = MessageDiscovery(id=0x100, name="First", length=8)
    msg2 = MessageDiscovery(id=0x100, name="Second", length=8)

    doc.add_message(msg1)
    doc.add_message(msg2)

    assert len(doc.messages) == 1
    assert doc.messages[0x100].name == "Second"


def test_signal_discovery_minimal():
    """Test SignalDiscovery with minimal required fields."""
    signal = SignalDiscovery(name="MinimalSignal", start_bit=0, length=8)

    assert signal.name == "MinimalSignal"
    assert signal.confidence == 0.0
    assert signal.evidence == []


def test_message_discovery_minimal():
    """Test MessageDiscovery with minimal required fields."""
    msg = MessageDiscovery(id=0x123, name="MinimalMsg", length=8)

    assert msg.id == 0x123
    assert msg.confidence == 0.0
    assert msg.signals == []


def test_discovery_document_load_missing_fields(tmp_path):
    """Test loading document with missing optional fields."""
    import yaml

    # Minimal valid document
    data = {
        "format_version": "1.0",
        "messages": [{"id": "0x100", "name": "TestMsg", "length": 8}],
    }

    file_path = tmp_path / "minimal.tkcan"
    with open(file_path, "w") as f:
        yaml.safe_dump(data, f)

    doc = DiscoveryDocument.load(file_path)

    assert len(doc.messages) == 1
    # Should have default vehicle info
    assert doc.vehicle.make == "Unknown"
