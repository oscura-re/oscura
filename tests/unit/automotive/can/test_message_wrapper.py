"""Comprehensive tests for CANMessageWrapper class.

This module tests CAN message analysis, hypothesis testing, and signal documentation.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.automotive]

from oscura.automotive.can.message_wrapper import (
    CANMessageWrapper,
    HypothesisResult,
)
from oscura.automotive.can.session import CANSession


class TestHypothesisResult:
    """Tests for HypothesisResult class."""

    def test_hypothesis_result_creation(self):
        """Test creating hypothesis result."""
        from oscura.automotive.can.models import SignalDefinition

        sig_def = SignalDefinition(
            name="test_signal",
            start_bit=16,
            length=16,
            byte_order="big_endian",
            value_type="unsigned",
            scale=0.25,
            offset=0.0,
            unit="rpm",
        )

        result = HypothesisResult(
            signal_name="test_signal",
            definition=sig_def,
            values=[100.0, 110.0, 120.0],
            min_value=100.0,
            max_value=120.0,
            mean=110.0,
            std=10.0,
            is_valid=True,
            confidence=0.95,
            feedback="Test feedback",
        )

        assert result.signal_name == "test_signal"
        assert result.min_value == 100.0
        assert result.max_value == 120.0
        assert result.mean == 110.0
        assert result.is_valid is True
        assert result.confidence == 0.95

    def test_hypothesis_result_repr(self):
        """Test HypothesisResult string representation."""
        from oscura.automotive.can.models import SignalDefinition

        sig_def = SignalDefinition(
            name="rpm",
            start_bit=16,
            length=16,
            byte_order="big_endian",
            value_type="unsigned",
            scale=1.0,
            offset=0.0,
            unit="rpm",
        )

        result = HypothesisResult(
            signal_name="rpm",
            definition=sig_def,
            values=[1000.0, 2000.0],
            min_value=1000.0,
            max_value=2000.0,
            mean=1500.0,
            std=500.0,
            is_valid=True,
            confidence=0.95,
            feedback="Values in expected range",
        )

        repr_str = repr(result)
        assert "VALID" in repr_str
        assert "0.95" in repr_str
        assert "rpm" in repr_str

    def test_hypothesis_result_summary(self):
        """Test hypothesis result summary generation."""
        from oscura.automotive.can.models import SignalDefinition

        sig_def = SignalDefinition(
            name="speed",
            start_bit=0,
            length=16,
            byte_order="big_endian",
            value_type="unsigned",
            scale=0.01,
            offset=0.0,
            unit="km/h",
        )

        result = HypothesisResult(
            signal_name="speed",
            definition=sig_def,
            values=[50.0, 60.0, 70.0],
            min_value=50.0,
            max_value=70.0,
            mean=60.0,
            std=10.0,
            is_valid=True,
            confidence=0.90,
            feedback="Signal shows variation",
        )

        summary = result.summary()
        assert "speed" in summary
        assert "VALID" in summary
        assert "0.90" in summary
        assert "50.00 km/h" in summary
        assert "70.00 km/h" in summary


class TestCANMessageWrapper:
    """Tests for CANMessageWrapper class."""

    def test_wrapper_creation(self, sample_can_messages):
        """Test creating message wrapper."""
        session = CANSession(name="Test")
        session._messages = sample_can_messages

        wrapper = CANMessageWrapper(session, 0x280)

        assert wrapper.arbitration_id == 0x280
        assert len(wrapper.get_documented_signals()) == 0

    def test_wrapper_repr(self, sample_can_messages):
        """Test wrapper string representation."""
        session = CANSession(name="Test")
        session._messages = sample_can_messages

        wrapper = CANMessageWrapper(session, 0x280)
        repr_str = repr(wrapper)

        assert "0x280" in repr_str
        assert "documented_signals=0" in repr_str

    def test_analyze_message(self, sample_can_messages):
        """Test analyzing message through wrapper."""
        session = CANSession(name="Test")
        session._messages = sample_can_messages

        wrapper = CANMessageWrapper(session, 0x280)
        analysis = wrapper.analyze()

        assert analysis is not None
        assert analysis.arbitration_id == 0x280
        assert analysis.message_count > 0

    def test_analyze_message_force_refresh(self, sample_can_messages):
        """Test forcing analysis refresh."""
        session = CANSession(name="Test")
        session._messages = sample_can_messages

        wrapper = CANMessageWrapper(session, 0x280)

        # First analysis
        analysis1 = wrapper.analyze()
        # Second analysis (cached)
        analysis2 = wrapper.analyze()
        # Should be same object
        assert analysis1 is analysis2

        # Force refresh
        analysis3 = wrapper.analyze(force_refresh=True)
        # Should be different object
        assert analysis3 is not analysis1

    def test_test_hypothesis_valid(self, sample_can_messages):
        """Test hypothesis testing with valid signal."""
        session = CANSession(name="Test")
        session._messages = sample_can_messages

        wrapper = CANMessageWrapper(session, 0x280)

        # Test RPM signal (bytes 2-3, big-endian, scale 0.25)
        result = wrapper.test_hypothesis(
            signal_name="rpm",
            start_byte=2,
            bit_length=16,
            byte_order="big_endian",
            value_type="unsigned",
            scale=0.25,
            unit="rpm",
            expected_min=0,
            expected_max=8000,
        )

        assert result.signal_name == "rpm"
        assert result.is_valid is True
        assert len(result.values) > 0
        assert 0 <= result.min_value <= 8000
        assert 0 <= result.max_value <= 8000

    def test_test_hypothesis_invalid_range(self, sample_can_messages):
        """Test hypothesis with values outside expected range."""
        session = CANSession(name="Test")
        session._messages = sample_can_messages

        wrapper = CANMessageWrapper(session, 0x280)

        # Test with unrealistic min/max
        result = wrapper.test_hypothesis(
            signal_name="rpm",
            start_byte=2,
            bit_length=16,
            byte_order="big_endian",
            value_type="unsigned",
            scale=0.25,
            unit="rpm",
            expected_min=10000,  # Unrealistic minimum
            expected_max=20000,
        )

        # Should fail validation
        assert result.is_valid is False
        assert result.confidence < 1.0

    def test_test_hypothesis_little_endian(self, sample_can_messages):
        """Test hypothesis with little-endian byte order."""
        session = CANSession(name="Test")
        session._messages = sample_can_messages

        wrapper = CANMessageWrapper(session, 0x400)

        # Test speed signal (bytes 0-1, big-endian in fixture)
        result = wrapper.test_hypothesis(
            signal_name="speed",
            start_byte=0,
            bit_length=16,
            byte_order="little_endian",
            value_type="unsigned",
            scale=0.01,
            unit="km/h",
        )

        assert result.signal_name == "speed"
        assert len(result.values) > 0

    def test_test_hypothesis_signed_value(self, sample_can_messages):
        """Test hypothesis with signed value type."""
        session = CANSession(name="Test")
        session._messages = sample_can_messages

        wrapper = CANMessageWrapper(session, 0x280)

        result = wrapper.test_hypothesis(
            signal_name="test_signed",
            start_byte=0,
            bit_length=8,
            byte_order="big_endian",
            value_type="signed",
            scale=1.0,
            unit="",
        )

        assert result.signal_name == "test_signed"
        assert len(result.values) > 0

    def test_document_signal(self, sample_can_messages):
        """Test documenting a signal."""
        session = CANSession(name="Test")
        session._messages = sample_can_messages

        wrapper = CANMessageWrapper(session, 0x280)

        wrapper.document_signal(
            name="rpm",
            start_bit=16,
            length=16,
            byte_order="big_endian",
            value_type="unsigned",
            scale=0.25,
            unit="rpm",
            comment="Engine RPM",
        )

        signals = wrapper.get_documented_signals()
        assert "rpm" in signals
        assert signals["rpm"].name == "rpm"
        assert signals["rpm"].scale == 0.25
        assert signals["rpm"].unit == "rpm"
        assert signals["rpm"].comment == "Engine RPM"

    def test_document_multiple_signals(self, sample_can_messages):
        """Test documenting multiple signals."""
        session = CANSession(name="Test")
        session._messages = sample_can_messages

        wrapper = CANMessageWrapper(session, 0x400)

        wrapper.document_signal(
            name="speed",
            start_bit=0,
            length=16,
            byte_order="big_endian",
            value_type="unsigned",
            scale=0.01,
            unit="km/h",
        )

        wrapper.document_signal(
            name="throttle",
            start_bit=16,
            length=8,
            byte_order="big_endian",
            value_type="unsigned",
            scale=0.4,
            unit="%",
        )

        signals = wrapper.get_documented_signals()
        assert len(signals) == 2
        assert "speed" in signals
        assert "throttle" in signals

    def test_decode_signals(self, sample_can_messages):
        """Test decoding documented signals."""
        session = CANSession(name="Test")
        session._messages = sample_can_messages

        wrapper = CANMessageWrapper(session, 0x280)

        # Document a signal
        wrapper.document_signal(
            name="rpm",
            start_bit=16,
            length=16,
            byte_order="big_endian",
            value_type="unsigned",
            scale=0.25,
            unit="rpm",
        )

        # Decode signals
        decoded = wrapper.decode_signals()

        assert len(decoded) > 0
        # Each decoded signal should have required fields
        for sig in decoded:
            assert sig.name == "rpm"
            assert sig.unit == "rpm"
            assert sig.timestamp >= 0

    def test_decode_signals_multiple(self, sample_can_messages):
        """Test decoding multiple signals."""
        session = CANSession(name="Test")
        session._messages = sample_can_messages

        wrapper = CANMessageWrapper(session, 0x400)

        # Document multiple signals
        wrapper.document_signal(
            name="speed",
            start_bit=0,
            length=16,
            byte_order="big_endian",
            value_type="unsigned",
            scale=0.01,
            unit="km/h",
        )

        wrapper.document_signal(
            name="throttle",
            start_bit=16,
            length=8,
            byte_order="big_endian",
            value_type="unsigned",
            scale=0.4,
            unit="%",
        )

        # Decode signals
        decoded = wrapper.decode_signals()

        # Should have 2 signals per message
        filtered = session._messages.filter_by_id(0x400)
        expected_count = len(filtered.messages) * 2

        assert len(decoded) == expected_count

    def test_decode_signals_empty(self, sample_can_messages):
        """Test decoding with no documented signals."""
        session = CANSession(name="Test")
        session._messages = sample_can_messages

        wrapper = CANMessageWrapper(session, 0x280)

        # No signals documented
        decoded = wrapper.decode_signals()

        assert len(decoded) == 0

    def test_hypothesis_no_values(self, sample_can_messages):
        """Test hypothesis when no values can be decoded."""
        session = CANSession(name="Test")
        session._messages = sample_can_messages

        wrapper = CANMessageWrapper(session, 0x123)

        # Test with parameters that won't decode properly
        result = wrapper.test_hypothesis(
            signal_name="invalid",
            start_byte=10,  # Beyond message length
            bit_length=16,
            byte_order="big_endian",
            value_type="unsigned",
            scale=1.0,
        )

        assert result.is_valid is False
        assert result.confidence == 0.0
        assert len(result.values) == 0

    def test_hypothesis_constant_values(self, sample_can_messages):
        """Test hypothesis with constant (non-varying) values."""
        session = CANSession(name="Test")
        session._messages = sample_can_messages

        wrapper = CANMessageWrapper(session, 0x280)

        # Test constant byte (byte 0 = 0xAA in fixture)
        result = wrapper.test_hypothesis(
            signal_name="constant",
            start_byte=0,
            bit_length=8,
            byte_order="big_endian",
            value_type="unsigned",
            scale=1.0,
        )

        # Should have reduced confidence due to no variation
        assert result.confidence < 1.0
        assert "identical" in result.feedback.lower() or "constant" in result.feedback.lower()

    def test_hypothesis_large_range(self, sample_can_messages):
        """Test hypothesis with very large value range."""
        session = CANSession(name="Test")
        session._messages = sample_can_messages

        wrapper = CANMessageWrapper(session, 0x280)

        # Test with very large scale to create large range
        result = wrapper.test_hypothesis(
            signal_name="large_range",
            start_byte=2,
            bit_length=16,
            byte_order="big_endian",
            value_type="unsigned",
            scale=10000.0,  # Very large scale
            unit="test",
        )

        # Should have reduced confidence due to large range
        assert result.confidence < 1.0
