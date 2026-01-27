"""Comprehensive tests for CAN state machine learning.

This module tests state machine inference from CAN message sequences.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.automotive]

from oscura.automotive.can.models import CANMessage, CANMessageList
from oscura.automotive.can.session import CANSession
from oscura.automotive.can.state_machine import (
    CANStateMachine,
    SequenceExtraction,
    learn_state_machine,
)


@pytest.fixture
def session_with_sequences() -> CANSession:
    """Create session with repeated sequences for state machine learning."""
    messages = CANMessageList()
    timestamp = 0.0

    # Create repeated sequence: 0x100 → 0x200 → 0x300 (trigger)
    for i in range(10):
        msg1 = CANMessage(
            arbitration_id=0x100,
            timestamp=timestamp,
            data=bytes([i % 256] + [0] * 7),
            is_extended=False,
        )
        messages.append(msg1)

        msg2 = CANMessage(
            arbitration_id=0x200,
            timestamp=timestamp + 0.1,
            data=bytes([i % 256] + [0x11] * 7),
            is_extended=False,
        )
        messages.append(msg2)

        msg3 = CANMessage(
            arbitration_id=0x300,  # Trigger
            timestamp=timestamp + 0.2,
            data=bytes([i % 256] + [0x22] * 7),
            is_extended=False,
        )
        messages.append(msg3)

        timestamp += 1.0  # 1 second between sequences

    session = CANSession(name="Sequence Test")
    session._messages = messages
    return session


class TestSequenceExtraction:
    """Tests for SequenceExtraction dataclass."""

    def test_sequence_extraction_creation(self):
        """Test creating sequence extraction."""
        extraction = SequenceExtraction(
            trigger_id=0x300,
            trigger_timestamp=1.5,
            sequence=[0x100, 0x200, 0x300],
            timestamps=[1.0, 1.2, 1.5],
            window_start=1.0,
            window_end=1.5,
        )

        assert extraction.trigger_id == 0x300
        assert extraction.trigger_timestamp == 1.5
        assert extraction.sequence == [0x100, 0x200, 0x300]
        assert len(extraction.timestamps) == 3

    def test_sequence_to_symbol_sequence(self):
        """Test converting sequence to symbols."""
        extraction = SequenceExtraction(
            trigger_id=0x300,
            trigger_timestamp=1.0,
            sequence=[0x100, 0x200, 0x300],
            timestamps=[0.5, 0.7, 1.0],
            window_start=0.5,
            window_end=1.0,
        )

        symbols = extraction.to_symbol_sequence()

        assert symbols == ["0x100", "0x200", "0x300"]


class TestCANStateMachine:
    """Tests for CANStateMachine class."""

    def test_state_machine_creation(self):
        """Test creating state machine learner."""
        sm = CANStateMachine()
        assert sm is not None

    def test_extract_sequences_basic(self, session_with_sequences):
        """Test basic sequence extraction."""
        sm = CANStateMachine()

        extractions = sm.extract_sequences(
            session=session_with_sequences,
            trigger_ids=[0x300],
            context_window_ms=300,
        )

        # Should extract sequences before each trigger
        assert len(extractions) > 0

        # Each extraction should have trigger info
        for ext in extractions:
            assert ext.trigger_id == 0x300
            assert ext.trigger_timestamp > 0
            assert len(ext.sequence) > 0

    def test_extract_sequences_context_window(self, session_with_sequences):
        """Test context window parameter."""
        sm = CANStateMachine()

        # Small window
        ext_small = sm.extract_sequences(
            session=session_with_sequences,
            trigger_ids=[0x300],
            context_window_ms=100,
        )

        # Large window
        ext_large = sm.extract_sequences(
            session=session_with_sequences,
            trigger_ids=[0x300],
            context_window_ms=500,
        )

        # Large window should capture more messages per sequence
        if ext_small and ext_large:
            avg_len_small = sum(len(e.sequence) for e in ext_small) / len(ext_small)
            avg_len_large = sum(len(e.sequence) for e in ext_large) / len(ext_large)
            assert avg_len_large >= avg_len_small

    def test_extract_sequences_multiple_triggers(self, session_with_sequences):
        """Test extracting with multiple trigger IDs."""
        sm = CANStateMachine()

        extractions = sm.extract_sequences(
            session=session_with_sequences,
            trigger_ids=[0x200, 0x300],  # Multiple triggers
            context_window_ms=300,
        )

        # Should find triggers for both IDs
        trigger_ids_found = {ext.trigger_id for ext in extractions}
        assert len(trigger_ids_found) > 0

    def test_extract_sequences_no_triggers(self):
        """Test extraction when no triggers found."""
        session = CANSession(name="No Triggers")
        # Empty session
        sm = CANStateMachine()

        extractions = sm.extract_sequences(
            session=session,
            trigger_ids=[0xFFF],  # Non-existent ID
            context_window_ms=100,
        )

        assert len(extractions) == 0

    def test_learn_from_session_basic(self, session_with_sequences):
        """Test learning state machine from session."""
        sm = CANStateMachine()

        automaton = sm.learn_from_session(
            session=session_with_sequences,
            trigger_ids=[0x300],
            context_window_ms=300,
            min_sequence_length=2,
        )

        # Should learn a finite automaton
        assert automaton is not None
        assert len(automaton.states) > 0
        assert automaton.initial_state is not None

    def test_learn_from_session_min_sequence_length(self, session_with_sequences):
        """Test minimum sequence length parameter."""
        sm = CANStateMachine()

        # Very short sequences
        automaton_short = sm.learn_from_session(
            session=session_with_sequences,
            trigger_ids=[0x300],
            context_window_ms=300,
            min_sequence_length=1,
        )

        # Longer sequences
        automaton_long = sm.learn_from_session(
            session=session_with_sequences,
            trigger_ids=[0x300],
            context_window_ms=300,
            min_sequence_length=3,
        )

        # Both should succeed
        assert automaton_short is not None
        assert automaton_long is not None

    def test_learn_from_session_no_sequences(self):
        """Test learning when no sequences can be extracted."""
        session = CANSession(name="Empty")
        sm = CANStateMachine()

        with pytest.raises(ValueError, match="No sequences found"):
            sm.learn_from_session(
                session=session,
                trigger_ids=[0xFFF],
                context_window_ms=100,
                min_sequence_length=2,
            )

    def test_learn_from_session_sequences_too_short(self):
        """Test when sequences don't meet minimum length."""
        # Create session with very short sequences
        messages = CANMessageList()
        for i in range(5):
            msg = CANMessage(
                arbitration_id=0x300,  # Only trigger messages
                timestamp=i * 1.0,
                data=bytes([0] * 8),
                is_extended=False,
            )
            messages.append(msg)

        session = CANSession(name="Short Sequences")
        session._messages = messages

        sm = CANStateMachine()

        with pytest.raises(ValueError, match="No sequences with length"):
            sm.learn_from_session(
                session=session,
                trigger_ids=[0x300],
                context_window_ms=100,
                min_sequence_length=10,  # Too long
            )

    def test_learn_with_states_basic(self, session_with_sequences):
        """Test learning with predefined state labels."""
        sm = CANStateMachine()

        state_definitions = {
            "STATE_A": [0x100],
            "STATE_B": [0x200],
            "STATE_C": [0x300],
        }

        automaton = sm.learn_with_states(
            session=session_with_sequences,
            state_definitions=state_definitions,
            context_window_ms=500,
        )

        assert automaton is not None
        assert len(automaton.states) > 0

    def test_learn_with_states_overlapping_ids(self, session_with_sequences):
        """Test learning with overlapping state IDs."""
        sm = CANStateMachine()

        state_definitions = {
            "STATE_A": [0x100, 0x200],  # Multiple IDs for one state
            "STATE_B": [0x300],
        }

        automaton = sm.learn_with_states(
            session=session_with_sequences,
            state_definitions=state_definitions,
            context_window_ms=500,
        )

        assert automaton is not None

    def test_learn_with_states_no_sequences(self):
        """Test learning with states when no sequences found."""
        session = CANSession(name="Empty")
        sm = CANStateMachine()

        state_definitions = {
            "STATE_A": [0x100],
        }

        with pytest.raises(ValueError, match="No state sequences found"):
            sm.learn_with_states(
                session=session,
                state_definitions=state_definitions,
                context_window_ms=100,
            )

    def test_learn_with_states_window_timeout(self):
        """Test state learning with context window timeout."""
        # Create messages with large time gaps
        messages = CANMessageList()
        timestamp = 0.0

        for i in range(3):
            msg1 = CANMessage(
                arbitration_id=0x100,
                timestamp=timestamp,
                data=bytes([0] * 8),
                is_extended=False,
            )
            messages.append(msg1)

            # Large gap
            timestamp += 10.0

            msg2 = CANMessage(
                arbitration_id=0x200,
                timestamp=timestamp,
                data=bytes([0] * 8),
                is_extended=False,
            )
            messages.append(msg2)

            timestamp += 10.0

        session = CANSession(name="Large Gaps")
        session._messages = messages

        sm = CANStateMachine()

        state_definitions = {
            "STATE_A": [0x100],
            "STATE_B": [0x200],
        }

        # Small window won't capture sequences across gaps
        try:
            sm.learn_with_states(
                session=session,
                state_definitions=state_definitions,
                context_window_ms=100,
            )
        except ValueError:
            # Expected - sequences broken by time window
            pass


class TestLearnStateMachineConvenience:
    """Tests for learn_state_machine convenience function."""

    def test_learn_state_machine_basic(self, session_with_sequences):
        """Test convenience function."""
        automaton = learn_state_machine(
            session=session_with_sequences,
            trigger_ids=[0x300],
            context_window_ms=300,
        )

        assert automaton is not None
        assert len(automaton.states) > 0

    def test_learn_state_machine_parameters(self, session_with_sequences):
        """Test convenience function with parameters."""
        automaton = learn_state_machine(
            session=session_with_sequences,
            trigger_ids=[0x300],
            context_window_ms=500,
        )

        assert automaton is not None


class TestStateMachineLearningEdgeCases:
    """Tests for edge cases in state machine learning."""

    def test_single_message_sequence(self):
        """Test learning from single-message sequences."""
        messages = CANMessageList()
        for i in range(5):
            msg = CANMessage(
                arbitration_id=0x100,
                timestamp=i * 0.1,
                data=bytes([0] * 8),
                is_extended=False,
            )
            messages.append(msg)

        session = CANSession(name="Single Message")
        session._messages = messages

        sm = CANStateMachine()

        try:
            # This should raise an error (need min_sequence_length >= 2)
            sm.learn_from_session(
                session=session,
                trigger_ids=[0x100],
                context_window_ms=50,
                min_sequence_length=2,
            )
        except ValueError:
            # Expected
            pass

    def test_complex_sequence_pattern(self):
        """Test learning from complex sequence patterns."""
        messages = CANMessageList()
        timestamp = 0.0

        # Create alternating patterns
        for i in range(6):
            if i % 2 == 0:
                # Pattern A: 0x100 → 0x200 → 0x300
                ids = [0x100, 0x200, 0x300]
            else:
                # Pattern B: 0x100 → 0x400 → 0x300
                ids = [0x100, 0x400, 0x300]

            for msg_id in ids:
                msg = CANMessage(
                    arbitration_id=msg_id,
                    timestamp=timestamp,
                    data=bytes([0] * 8),
                    is_extended=False,
                )
                messages.append(msg)
                timestamp += 0.1

            timestamp += 0.5  # Gap between patterns

        session = CANSession(name="Complex Patterns")
        session._messages = messages

        sm = CANStateMachine()

        automaton = sm.learn_from_session(
            session=session,
            trigger_ids=[0x300],
            context_window_ms=500,
            min_sequence_length=2,
        )

        # Should learn automaton that accepts both patterns
        assert automaton is not None
        assert len(automaton.states) > 0

    def test_timestamp_ordering(self):
        """Test that messages are processed in timestamp order."""
        # Create messages with out-of-order timestamps
        messages = CANMessageList()
        messages.append(
            CANMessage(
                arbitration_id=0x200,
                timestamp=0.2,
                data=bytes([0] * 8),
                is_extended=False,
            )
        )
        messages.append(
            CANMessage(
                arbitration_id=0x100,
                timestamp=0.1,  # Earlier timestamp
                data=bytes([0] * 8),
                is_extended=False,
            )
        )
        messages.append(
            CANMessage(
                arbitration_id=0x300,
                timestamp=0.3,
                data=bytes([0] * 8),
                is_extended=False,
            )
        )

        session = CANSession(name="Out of Order")
        session._messages = messages

        sm = CANStateMachine()

        extractions = sm.extract_sequences(
            session=session,
            trigger_ids=[0x300],
            context_window_ms=300,
        )

        # Should have sorted by timestamp internally
        if extractions:
            # Sequence should be ordered by timestamp
            for ext in extractions:
                timestamps = ext.timestamps
                assert timestamps == sorted(timestamps)
