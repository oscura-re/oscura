"""Comprehensive tests for CANSession class.

This module tests CAN session management, analysis, comparison, and pattern detection.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.automotive]

from oscura.automotive.can.models import CANMessage, CANMessageList
from oscura.automotive.can.session import CANSession


class TestCANSessionBasic:
    """Tests for basic CANSession functionality."""

    def test_session_creation(self):
        """Test creating a CAN session."""
        session = CANSession(name="Test Session")

        assert session.name == "Test Session"
        assert len(session) == 0
        assert len(session.unique_ids()) == 0

    def test_session_creation_with_crc_options(self):
        """Test creating session with CRC options."""
        session = CANSession(
            name="Test",
            auto_crc=False,
            crc_validate=False,
            crc_min_messages=20,
        )

        assert session.auto_crc is False
        assert session.crc_validate is False
        assert session.crc_min_messages == 20

    def test_session_length(self, sample_can_messages):
        """Test session length."""
        session = CANSession(name="Test")
        session._messages = sample_can_messages

        assert len(session) == len(sample_can_messages)

    def test_session_unique_ids(self, sample_can_messages):
        """Test getting unique CAN IDs."""
        session = CANSession(name="Test")
        session._messages = sample_can_messages

        unique_ids = session.unique_ids()
        expected_ids = sample_can_messages.unique_ids()

        assert unique_ids == expected_ids

    def test_session_time_range(self, sample_can_messages):
        """Test getting time range."""
        session = CANSession(name="Test")
        session._messages = sample_can_messages

        start, end = session.time_range()
        assert start >= 0.0
        assert end > start

    def test_session_time_range_empty(self):
        """Test time range with empty session."""
        session = CANSession(name="Empty")

        start, end = session.time_range()
        assert start == 0.0
        assert end == 0.0

    def test_session_repr(self, sample_can_messages):
        """Test session string representation."""
        session = CANSession(name="Test Session")
        session._messages = sample_can_messages

        repr_str = repr(session)
        assert "Test Session" in repr_str
        assert "messages" in repr_str
        assert "unique IDs" in repr_str

    def test_session_repr_empty(self):
        """Test session repr when empty."""
        session = CANSession(name="Empty")

        repr_str = repr(session)
        assert "Empty" in repr_str
        assert "recordings=0" in repr_str


class TestCANSessionInventory:
    """Tests for CAN session inventory generation."""

    def test_inventory_generation(self, sample_can_messages):
        """Test generating message inventory."""
        session = CANSession(name="Test")
        session._messages = sample_can_messages

        inventory = session.inventory()

        # Should have one row per unique ID
        assert len(inventory) == len(sample_can_messages.unique_ids())

        # Check required columns
        required_columns = [
            "arbitration_id",
            "count",
            "frequency_hz",
            "period_ms",
            "first_seen",
            "last_seen",
            "dlc",
        ]
        for col in required_columns:
            assert col in inventory.columns

    def test_inventory_arbitration_ids(self, sample_can_messages):
        """Test inventory contains correct arbitration IDs."""
        session = CANSession(name="Test")
        session._messages = sample_can_messages

        inventory = session.inventory()

        # Extract IDs from inventory
        inv_ids = set()
        for arb_id_str in inventory["arbitration_id"]:
            # Format is '0xXXX'
            inv_ids.add(int(arb_id_str, 16))

        # Should match unique IDs from messages
        assert inv_ids == sample_can_messages.unique_ids()

    def test_inventory_message_counts(self, sample_can_messages):
        """Test inventory message counts."""
        session = CANSession(name="Test")
        session._messages = sample_can_messages

        inventory = session.inventory()

        # Verify counts for known IDs
        for _, row in inventory.iterrows():
            arb_id = int(row["arbitration_id"], 16)
            expected_count = len(sample_can_messages.filter_by_id(arb_id))
            assert int(row["count"]) == expected_count

    def test_inventory_frequency_calculation(self, sample_can_messages):
        """Test inventory frequency calculation."""
        session = CANSession(name="Test")
        session._messages = sample_can_messages

        inventory = session.inventory()

        # All frequencies should be non-negative
        for _, row in inventory.iterrows():
            freq = float(row["frequency_hz"])
            assert freq >= 0.0


class TestCANSessionMessageWrapper:
    """Tests for message wrapper access."""

    def test_get_message_wrapper(self, sample_can_messages):
        """Test getting message wrapper."""
        session = CANSession(name="Test")
        session._messages = sample_can_messages

        wrapper = session.message(0x280)

        assert wrapper.arbitration_id == 0x280

    def test_get_message_wrapper_invalid_id(self, sample_can_messages):
        """Test getting wrapper for non-existent ID."""
        session = CANSession(name="Test")
        session._messages = sample_can_messages

        with pytest.raises(ValueError, match="No messages found"):
            session.message(0xFFF)


class TestCANSessionAnalysis:
    """Tests for message analysis functionality."""

    def test_analyze_message_basic(self, sample_can_messages):
        """Test analyzing a message."""
        session = CANSession(name="Test")
        session._messages = sample_can_messages

        analysis = session.analyze_message(0x280)

        assert analysis is not None
        assert analysis.arbitration_id == 0x280
        assert analysis.message_count > 0

    def test_analyze_message_caching(self, sample_can_messages):
        """Test that analysis is cached."""
        session = CANSession(name="Test")
        session._messages = sample_can_messages

        # First analysis
        analysis1 = session.analyze_message(0x280)

        # Second analysis (should use cache)
        analysis2 = session.analyze_message(0x280)

        # Should be same object
        assert analysis1 is analysis2

    def test_analyze_message_force_refresh(self, sample_can_messages):
        """Test forcing analysis refresh."""
        session = CANSession(name="Test")
        session._messages = sample_can_messages

        # First analysis
        analysis1 = session.analyze_message(0x280)

        # Force refresh
        analysis2 = session.analyze_message(0x280, force_refresh=True)

        # Should be different objects
        assert analysis2 is not analysis1

    def test_analyze_session_complete(self, sample_can_messages):
        """Test complete session analysis."""
        session = CANSession(name="Test")
        session._messages = sample_can_messages

        results = session.analyze()

        # Check required keys
        assert "inventory" in results
        assert "num_messages" in results
        assert "num_unique_ids" in results
        assert "time_range" in results
        assert "message_analyses" in results
        assert "patterns" in results

        # Check values
        assert results["num_messages"] == len(sample_can_messages)
        assert results["num_unique_ids"] == len(sample_can_messages.unique_ids())


class TestCANSessionFiltering:
    """Tests for message filtering."""

    def test_filter_by_arbitration_ids(self, sample_can_messages):
        """Test filtering by CAN IDs."""
        session = CANSession(name="Test")
        session._messages = sample_can_messages

        filtered = session.filter(arbitration_ids=[0x280, 0x300])

        assert len(filtered.unique_ids()) == 2
        assert 0x280 in filtered.unique_ids()
        assert 0x300 in filtered.unique_ids()
        assert 0x123 not in filtered.unique_ids()

    def test_filter_by_time_range(self, sample_can_messages):
        """Test filtering by time range."""
        session = CANSession(name="Test")
        session._messages = sample_can_messages

        filtered = session.filter(time_range=(0.2, 0.5))

        start, end = filtered.time_range()
        assert start >= 0.2
        assert end <= 0.5

    def test_filter_by_frequency(self, sample_can_messages):
        """Test filtering by frequency."""
        session = CANSession(name="Test")
        session._messages = sample_can_messages

        # Filter for high-frequency messages
        filtered = session.filter(min_frequency=50.0)

        # All remaining messages should have high frequency
        for arb_id in filtered.unique_ids():
            msgs = filtered._messages.filter_by_id(arb_id)
            if len(msgs) > 1:
                timestamps = [m.timestamp for m in msgs.messages]
                duration = max(timestamps) - min(timestamps)
                if duration > 0:
                    freq = (len(msgs) - 1) / duration
                    assert freq >= 50.0

    def test_filter_combined(self, sample_can_messages):
        """Test combining multiple filters."""
        session = CANSession(name="Test")
        session._messages = sample_can_messages

        filtered = session.filter(
            arbitration_ids=[0x280, 0x300],
            time_range=(0.1, 0.8),
        )

        # Check ID filtering
        assert len(filtered.unique_ids()) <= 2

        # Check time filtering
        start, end = filtered.time_range()
        assert start >= 0.1
        assert end <= 0.8


class TestCANSessionPatterns:
    """Tests for pattern detection integration."""

    def test_find_message_pairs(self, sample_can_messages):
        """Test finding message pairs through session."""
        session = CANSession(name="Test")
        session._messages = sample_can_messages

        pairs = session.find_message_pairs(time_window_ms=100, min_occurrence=2)

        # Should return list (may be empty)
        assert isinstance(pairs, list)

    def test_find_message_sequences(self, sample_can_messages):
        """Test finding message sequences through session."""
        session = CANSession(name="Test")
        session._messages = sample_can_messages

        sequences = session.find_message_sequences(
            max_sequence_length=3,
            time_window_ms=100,
            min_support=0.5,
        )

        # Should return list (may be empty)
        assert isinstance(sequences, list)

    def test_find_temporal_correlations(self, sample_can_messages):
        """Test finding temporal correlations through session."""
        session = CANSession(name="Test")
        session._messages = sample_can_messages

        correlations = session.find_temporal_correlations(max_delay_ms=100)

        # Should return dict
        assert isinstance(correlations, dict)


class TestCANSessionComparison:
    """Tests for session comparison and stimulus-response analysis."""

    def test_compare_to_basic(self, sample_can_messages):
        """Test comparing two sessions."""
        session1 = CANSession(name="Baseline")
        session1._messages = sample_can_messages

        # Create modified session
        messages2 = CANMessageList()
        for msg in sample_can_messages.messages:
            # Modify data slightly for 0x280
            if msg.arbitration_id == 0x280:
                data = bytearray(msg.data)
                data[0] = (data[0] + 1) % 256
                new_msg = CANMessage(
                    arbitration_id=msg.arbitration_id,
                    timestamp=msg.timestamp,
                    data=bytes(data),
                    is_extended=msg.is_extended,
                )
                messages2.append(new_msg)
            else:
                messages2.append(msg)

        session2 = CANSession(name="Stimulus")
        session2._messages = messages2

        report = session1.compare_to(session2)

        # Should have detected changes
        assert report is not None

    def test_compare_to_identical_sessions(self, sample_can_messages):
        """Test comparing identical sessions."""
        session1 = CANSession(name="Session1")
        session1._messages = sample_can_messages

        session2 = CANSession(name="Session2")
        session2._messages = sample_can_messages

        report = session1.compare_to(session2)

        # Should have minimal changes
        assert len(report.changed_messages) == 0 or len(report.changed_messages) < len(
            sample_can_messages.unique_ids()
        )


class TestCANSessionCRC:
    """Tests for CRC recovery and validation."""

    def test_crc_info_empty(self):
        """Test CRC info when no CRCs recovered."""
        session = CANSession(name="Test")

        crc_info = session.crc_info
        assert isinstance(crc_info, dict)
        assert len(crc_info) == 0

    def test_auto_crc_disabled(self, sample_can_messages):
        """Test that auto CRC is skipped when disabled."""
        session = CANSession(name="Test", auto_crc=False)
        session._messages = sample_can_messages

        # Analyze should not trigger CRC recovery
        session.analyze()

        # No CRCs should be recovered
        assert len(session._crc_params) == 0

    def test_validate_crc_no_params(self, sample_can_messages):
        """Test CRC validation with no parameters."""
        session = CANSession(name="Test")
        session._messages = sample_can_messages

        # Should return True (no validation without params)
        msg = sample_can_messages.messages[0]
        result = session._validate_crc(msg)
        assert result is True


class TestCANSessionStateMachine:
    """Tests for state machine learning integration."""

    def test_learn_state_machine_basic(self):
        """Test state machine learning."""
        # Create session with trigger messages
        messages = CANMessageList()
        timestamp = 0.0

        for i in range(5):
            # Sequence: 0x100 → 0x200 → 0x300 (trigger)
            msg1 = CANMessage(
                arbitration_id=0x100,
                timestamp=timestamp,
                data=bytes([0] * 8),
                is_extended=False,
            )
            messages.append(msg1)

            msg2 = CANMessage(
                arbitration_id=0x200,
                timestamp=timestamp + 0.1,
                data=bytes([0] * 8),
                is_extended=False,
            )
            messages.append(msg2)

            msg3 = CANMessage(
                arbitration_id=0x300,
                timestamp=timestamp + 0.2,
                data=bytes([0] * 8),
                is_extended=False,
            )
            messages.append(msg3)

            timestamp += 1.0

        session = CANSession(name="State Machine Test")
        session._messages = messages

        # Learn state machine
        automaton = session.learn_state_machine(
            trigger_ids=[0x300],
            context_window_ms=300,
        )

        assert automaton is not None
        assert len(automaton.states) > 0

    def test_learn_state_machine_no_triggers(self, sample_can_messages):
        """Test state machine learning with no trigger messages."""
        session = CANSession(name="Test")
        session._messages = sample_can_messages

        # Use non-existent trigger ID
        with pytest.raises(ValueError, match="No sequences found"):
            session.learn_state_machine(
                trigger_ids=[0xFFF],
                context_window_ms=100,
            )


class TestCANSessionFrequencyCalculation:
    """Tests for frequency calculation helpers."""

    def test_calculate_message_frequency_basic(self):
        """Test frequency calculation."""
        messages = []
        for i in range(10):
            msg = CANMessage(
                arbitration_id=0x100,
                timestamp=i * 0.1,  # 10 Hz
                data=bytes([0] * 8),
                is_extended=False,
            )
            messages.append(msg)

        session = CANSession(name="Test")
        freq = session._calculate_message_frequency(messages)

        # Should be approximately 10 Hz
        assert 9.0 < freq < 11.0

    def test_calculate_message_frequency_single_message(self):
        """Test frequency with single message."""
        messages = [
            CANMessage(
                arbitration_id=0x100,
                timestamp=0.0,
                data=bytes([0] * 8),
                is_extended=False,
            )
        ]

        session = CANSession(name="Test")
        freq = session._calculate_message_frequency(messages)

        # Should return None
        assert freq is None

    def test_calculate_message_frequency_zero_duration(self):
        """Test frequency with zero duration."""
        messages = [
            CANMessage(
                arbitration_id=0x100,
                timestamp=1.0,
                data=bytes([0] * 8),
                is_extended=False,
            ),
            CANMessage(
                arbitration_id=0x100,
                timestamp=1.0,  # Same timestamp
                data=bytes([0] * 8),
                is_extended=False,
            ),
        ]

        session = CANSession(name="Test")
        freq = session._calculate_message_frequency(messages)

        # Should return None
        assert freq is None
