"""Tests for CAN discovery documentation."""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.automotive]

from oscura.automotive.can.discovery import (
    DiscoveryDocument,
    Hypothesis,
    MessageDiscovery,
    SignalDiscovery,
)


class TestSignalDiscovery:
    """Tests for SignalDiscovery class."""

    def test_create_signal_discovery(self):
        """Test creating signal discovery."""
        sig = SignalDiscovery(
            name="rpm",
            start_bit=16,
            length=16,
            byte_order="big_endian",
            scale=0.25,
            unit="rpm",
            min_value=800.0,
            max_value=6000.0,
            confidence=0.95,
            evidence=["Correlated with OBD-II PID 0x0C"],
            comment="Engine RPM",
        )

        assert sig.name == "rpm"
        assert sig.start_bit == 16
        assert sig.confidence == 0.95

    def test_signal_to_dict(self):
        """Test converting signal to dictionary."""
        sig = SignalDiscovery(
            name="rpm",
            start_bit=16,
            length=16,
        )

        d = sig.to_dict()
        assert "name" in d
        assert "start_bit" in d
        assert "length" in d


class TestMessageDiscovery:
    """Tests for MessageDiscovery class."""

    def test_create_message_discovery(self):
        """Test creating message discovery."""
        sig1 = SignalDiscovery(name="rpm", start_bit=16, length=16)
        sig2 = SignalDiscovery(name="throttle", start_bit=32, length=8)

        msg = MessageDiscovery(
            id=0x280,
            name="Engine_Status",
            length=8,
            cycle_time_ms=10.0,
            confidence=0.9,
            evidence=["Periodic at 100 Hz"],
            signals=[sig1, sig2],
        )

        assert msg.id == 0x280
        assert msg.name == "Engine_Status"
        assert len(msg.signals) == 2

    def test_message_to_dict(self):
        """Test converting message to dictionary."""
        msg = MessageDiscovery(
            id=0x280,
            name="Engine_Status",
            length=8,
        )

        d = msg.to_dict()
        assert d["id"] == "0x280"
        assert "name" in d
        assert "length" in d


class TestDiscoveryDocument:
    """Tests for DiscoveryDocument class."""

    def test_create_empty_document(self):
        """Test creating empty discovery document."""
        doc = DiscoveryDocument()

        assert doc.format_version == "1.0"
        assert len(doc.messages) == 0
        assert len(doc.hypotheses) == 0

    def test_add_message(self):
        """Test adding message discovery."""
        doc = DiscoveryDocument()

        msg = MessageDiscovery(
            id=0x280,
            name="Engine_Status",
            length=8,
        )

        doc.add_message(msg)

        assert 0x280 in doc.messages
        assert doc.messages[0x280].name == "Engine_Status"

    def test_add_hypothesis(self):
        """Test adding hypothesis."""
        doc = DiscoveryDocument()

        hyp = Hypothesis(
            message_id=0x280,
            signal="unknown_byte4",
            hypothesis="Coolant temperature",
            status="testing",
            test_plan="Compare with OBD-II coolant temp",
        )

        doc.add_hypothesis(hyp)

        assert len(doc.hypotheses) == 1
        assert doc.hypotheses[0].signal == "unknown_byte4"

    def test_save_and_load(self, temp_dir):
        """Test saving and loading discovery document."""
        doc = DiscoveryDocument()

        # Set vehicle info
        doc.vehicle.make = "Toyota"
        doc.vehicle.model = "Camry"
        doc.vehicle.year = "2020"

        # Add message with signals
        sig = SignalDiscovery(
            name="rpm",
            start_bit=16,
            length=16,
            scale=0.25,
            unit="rpm",
            confidence=0.95,
            evidence=["OBD-II correlation"],
        )

        msg = MessageDiscovery(
            id=0x280,
            name="Engine_Status",
            length=8,
            signals=[sig],
        )

        doc.add_message(msg)

        # Add hypothesis
        hyp = Hypothesis(
            message_id=0x300,
            signal="byte2",
            hypothesis="Vehicle speed",
            status="testing",
        )

        doc.add_hypothesis(hyp)

        # Save
        file_path = temp_dir / "discoveries.tkcan"
        doc.save(file_path)

        # Load
        loaded_doc = DiscoveryDocument.load(file_path)

        # Verify
        assert loaded_doc.vehicle.make == "Toyota"
        assert loaded_doc.vehicle.model == "Camry"
        assert 0x280 in loaded_doc.messages
        assert loaded_doc.messages[0x280].name == "Engine_Status"
        assert len(loaded_doc.messages[0x280].signals) == 1
        assert loaded_doc.messages[0x280].signals[0].name == "rpm"
        assert len(loaded_doc.hypotheses) == 1
        assert loaded_doc.hypotheses[0].message_id == 0x300

    def test_document_repr(self):
        """Test document string representation."""
        doc = DiscoveryDocument()

        msg = MessageDiscovery(id=0x280, name="Engine", length=8)
        doc.add_message(msg)

        hyp = Hypothesis(message_id=0x280, signal="test", hypothesis="test")
        doc.add_hypothesis(hyp)

        repr_str = repr(doc)
        assert "1 messages" in repr_str
        assert "1 hypotheses" in repr_str
