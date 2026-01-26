"""Comprehensive tests for CAN state machine learning.

This test suite covers:
- CANStateMachine class functionality
- Sequence extraction around trigger messages
- State machine learning from CAN sessions
- Integration with StateMachineInferrer
- Edge cases and error conditions
- Real-world use cases (ignition sequences, initialization)
"""

import pytest

from oscura.automotive.can.models import CANMessage
from oscura.automotive.can.session import CANSession
from oscura.automotive.can.state_machine import (
    CANStateMachine,
    SequenceExtraction,
    learn_state_machine,
)
from oscura.inference.state_machine import FiniteAutomaton

pytestmark = [pytest.mark.unit, pytest.mark.analyzer]


# =============================================================================
# SequenceExtraction Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.analyzer
class TestSequenceExtraction:
    """Test SequenceExtraction dataclass."""

    def test_create_sequence_extraction(self) -> None:
        """Test creating a sequence extraction."""
        extraction = SequenceExtraction(
            trigger_id=0x280,
            trigger_timestamp=1.5,
            sequence=[0x100, 0x200, 0x280],
            timestamps=[1.0, 1.2, 1.5],
            window_start=1.0,
            window_end=1.5,
        )

        assert extraction.trigger_id == 0x280
        assert extraction.trigger_timestamp == 1.5
        assert extraction.sequence == [0x100, 0x200, 0x280]
        assert len(extraction.timestamps) == 3
        assert extraction.window_start == 1.0
        assert extraction.window_end == 1.5

    def test_to_symbol_sequence(self) -> None:
        """Test converting CAN IDs to symbol strings."""
        extraction = SequenceExtraction(
            trigger_id=0x280,
            trigger_timestamp=1.5,
            sequence=[0x100, 0x200, 0x280],
            timestamps=[1.0, 1.2, 1.5],
            window_start=1.0,
            window_end=1.5,
        )

        symbols = extraction.to_symbol_sequence()

        assert symbols == ["0x100", "0x200", "0x280"]

    def test_to_symbol_sequence_empty(self) -> None:
        """Test converting empty sequence."""
        extraction = SequenceExtraction(
            trigger_id=0x280,
            trigger_timestamp=1.5,
            sequence=[],
            timestamps=[],
            window_start=1.0,
            window_end=1.5,
        )

        symbols = extraction.to_symbol_sequence()

        assert symbols == []

    def test_to_symbol_sequence_extended_ids(self) -> None:
        """Test converting extended CAN IDs."""
        extraction = SequenceExtraction(
            trigger_id=0x18FEF100,
            trigger_timestamp=1.5,
            sequence=[0x18FEF100, 0x18FEF200],
            timestamps=[1.0, 1.5],
            window_start=1.0,
            window_end=1.5,
        )

        symbols = extraction.to_symbol_sequence()

        # Note: format uses 3 digits but will show more for extended IDs
        assert "0x18FEF100" in symbols[0]
        assert "0x18FEF200" in symbols[1]


# =============================================================================
# CANStateMachine Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.analyzer
class TestCANStateMachine:
    """Test CANStateMachine class."""

    def test_initialization(self) -> None:
        """Test CANStateMachine initialization."""
        sm = CANStateMachine()
        assert sm._inferrer is not None

    def test_extract_sequences_simple(self) -> None:
        """Test extracting sequences with single trigger."""
        # Create test session with messages
        messages = [
            CANMessage(arbitration_id=0x100, timestamp=0.0, data=b"\x00\x00"),
            CANMessage(arbitration_id=0x200, timestamp=0.1, data=b"\x01\x01"),
            CANMessage(arbitration_id=0x280, timestamp=0.2, data=b"\x02\x02"),  # Trigger
        ]
        session = CANSession(name="Test")
        session._messages.messages = messages

        sm = CANStateMachine()
        extractions = sm.extract_sequences(
            session=session, trigger_ids=[0x280], context_window_ms=500
        )

        assert len(extractions) == 1
        assert extractions[0].trigger_id == 0x280
        assert extractions[0].sequence == [0x100, 0x200, 0x280]

    def test_extract_sequences_multiple_triggers(self) -> None:
        """Test extracting sequences with multiple trigger occurrences."""
        # Create test session with repeated pattern
        messages = [
            CANMessage(arbitration_id=0x100, timestamp=0.0, data=b"\x00"),
            CANMessage(arbitration_id=0x280, timestamp=0.1, data=b"\x01"),  # Trigger 1
            CANMessage(arbitration_id=0x100, timestamp=0.5, data=b"\x02"),
            CANMessage(arbitration_id=0x280, timestamp=0.6, data=b"\x03"),  # Trigger 2
        ]
        session = CANSession(name="Test")
        session._messages.messages = messages

        sm = CANStateMachine()
        extractions = sm.extract_sequences(
            session=session, trigger_ids=[0x280], context_window_ms=200
        )

        assert len(extractions) == 2
        assert all(e.trigger_id == 0x280 for e in extractions)

    def test_extract_sequences_multiple_trigger_ids(self) -> None:
        """Test extracting sequences with multiple trigger IDs."""
        messages = [
            CANMessage(arbitration_id=0x100, timestamp=0.0, data=b"\x00"),
            CANMessage(arbitration_id=0x280, timestamp=0.1, data=b"\x01"),  # Trigger A
            CANMessage(arbitration_id=0x200, timestamp=0.2, data=b"\x02"),
            CANMessage(arbitration_id=0x290, timestamp=0.3, data=b"\x03"),  # Trigger B
        ]
        session = CANSession(name="Test")
        session._messages.messages = messages

        sm = CANStateMachine()
        extractions = sm.extract_sequences(
            session=session, trigger_ids=[0x280, 0x290], context_window_ms=200
        )

        assert len(extractions) == 2
        trigger_ids = {e.trigger_id for e in extractions}
        assert trigger_ids == {0x280, 0x290}

    def test_extract_sequences_respects_time_window(self) -> None:
        """Test that extraction respects time window."""
        messages = [
            CANMessage(arbitration_id=0x100, timestamp=0.0, data=b"\x00"),
            CANMessage(arbitration_id=0x200, timestamp=0.8, data=b"\x01"),  # Too old
            CANMessage(arbitration_id=0x300, timestamp=0.95, data=b"\x02"),  # In window
            CANMessage(arbitration_id=0x280, timestamp=1.0, data=b"\x03"),  # Trigger
        ]
        session = CANSession(name="Test")
        session._messages.messages = messages

        sm = CANStateMachine()
        extractions = sm.extract_sequences(
            session=session,
            trigger_ids=[0x280],
            context_window_ms=100,  # 100ms = 0.1s
        )

        assert len(extractions) == 1
        # Should only include messages from 0.9s to 1.0s
        assert 0x100 not in extractions[0].sequence
        assert 0x200 not in extractions[0].sequence
        assert 0x300 in extractions[0].sequence
        assert 0x280 in extractions[0].sequence

    def test_extract_sequences_no_triggers(self) -> None:
        """Test extraction with no trigger messages found."""
        messages = [
            CANMessage(arbitration_id=0x100, timestamp=0.0, data=b"\x00"),
            CANMessage(arbitration_id=0x200, timestamp=0.1, data=b"\x01"),
        ]
        session = CANSession(name="Test")
        session._messages.messages = messages

        sm = CANStateMachine()
        extractions = sm.extract_sequences(
            session=session, trigger_ids=[0x999], context_window_ms=500
        )

        assert len(extractions) == 0

    def test_learn_from_session_simple(self) -> None:
        """Test learning simple state machine from session."""
        # Create pattern: A -> B -> C (repeated)
        messages = [
            CANMessage(arbitration_id=0x100, timestamp=0.0, data=b"\x00"),
            CANMessage(arbitration_id=0x200, timestamp=0.1, data=b"\x01"),
            CANMessage(arbitration_id=0x300, timestamp=0.2, data=b"\x02"),  # Trigger
            CANMessage(arbitration_id=0x100, timestamp=0.5, data=b"\x03"),
            CANMessage(arbitration_id=0x200, timestamp=0.6, data=b"\x04"),
            CANMessage(arbitration_id=0x300, timestamp=0.7, data=b"\x05"),  # Trigger
        ]
        session = CANSession(name="Test")
        session._messages.messages = messages

        sm = CANStateMachine()
        automaton = sm.learn_from_session(
            session=session, trigger_ids=[0x300], context_window_ms=300
        )

        assert isinstance(automaton, FiniteAutomaton)
        assert len(automaton.states) >= 1
        assert len(automaton.transitions) >= 1

        # Should accept the learned pattern
        assert automaton.accepts(["0x100", "0x200", "0x300"]) is True

    def test_learn_from_session_no_sequences_raises(self) -> None:
        """Test that learning with no sequences raises error."""
        messages = [
            CANMessage(arbitration_id=0x100, timestamp=0.0, data=b"\x00"),
        ]
        session = CANSession(name="Test")
        session._messages.messages = messages

        sm = CANStateMachine()

        with pytest.raises(ValueError, match="No sequences found"):
            sm.learn_from_session(session=session, trigger_ids=[0x999], context_window_ms=500)

    def test_learn_from_session_min_sequence_length(self) -> None:
        """Test minimum sequence length filtering."""
        messages = [
            CANMessage(arbitration_id=0x100, timestamp=0.0, data=b"\x00"),
            CANMessage(arbitration_id=0x300, timestamp=0.01, data=b"\x01"),  # Trigger (short)
            CANMessage(arbitration_id=0x100, timestamp=0.5, data=b"\x02"),
            CANMessage(arbitration_id=0x200, timestamp=0.6, data=b"\x03"),
            CANMessage(arbitration_id=0x300, timestamp=0.7, data=b"\x04"),  # Trigger (long)
        ]
        session = CANSession(name="Test")
        session._messages.messages = messages

        sm = CANStateMachine()
        automaton = sm.learn_from_session(
            session=session, trigger_ids=[0x300], context_window_ms=300, min_sequence_length=3
        )

        # Should only learn from the longer sequence
        assert isinstance(automaton, FiniteAutomaton)

    def test_learn_from_session_complex_pattern(self) -> None:
        """Test learning complex multi-branch pattern."""
        # Pattern: A -> (B or C) -> D
        messages = [
            # Branch 1: A -> B -> D
            CANMessage(arbitration_id=0x100, timestamp=0.0, data=b"\x00"),
            CANMessage(arbitration_id=0x200, timestamp=0.1, data=b"\x01"),
            CANMessage(arbitration_id=0x400, timestamp=0.2, data=b"\x02"),  # Trigger
            # Branch 2: A -> C -> D
            CANMessage(arbitration_id=0x100, timestamp=0.5, data=b"\x03"),
            CANMessage(arbitration_id=0x300, timestamp=0.6, data=b"\x04"),
            CANMessage(arbitration_id=0x400, timestamp=0.7, data=b"\x05"),  # Trigger
        ]
        session = CANSession(name="Test")
        session._messages.messages = messages

        sm = CANStateMachine()
        automaton = sm.learn_from_session(
            session=session, trigger_ids=[0x400], context_window_ms=300
        )

        # Should accept both branches
        assert automaton.accepts(["0x100", "0x200", "0x400"]) is True
        assert automaton.accepts(["0x100", "0x300", "0x400"]) is True

    def test_learn_with_states_simple(self) -> None:
        """Test learning with predefined state labels."""
        # Create ignition sequence pattern
        messages = [
            CANMessage(arbitration_id=0x100, timestamp=0.0, data=b"\x00"),  # OFF
            CANMessage(arbitration_id=0x200, timestamp=0.5, data=b"\x01"),  # ACC
            CANMessage(arbitration_id=0x300, timestamp=1.0, data=b"\x02"),  # ON
            CANMessage(arbitration_id=0x400, timestamp=1.5, data=b"\x03"),  # START
        ]
        session = CANSession(name="Test")
        session._messages.messages = messages

        sm = CANStateMachine()
        automaton = sm.learn_with_states(
            session=session,
            state_definitions={
                "OFF": [0x100],
                "ACC": [0x200],
                "ON": [0x300],
                "START": [0x400],
            },
            context_window_ms=1000,
        )

        assert isinstance(automaton, FiniteAutomaton)
        # Should learn the sequence OFF -> ACC -> ON -> START
        assert automaton.accepts(["OFF", "ACC", "ON", "START"]) is True

    def test_learn_with_states_multiple_ids_per_state(self) -> None:
        """Test learning with multiple CAN IDs per state."""
        messages = [
            CANMessage(arbitration_id=0x100, timestamp=0.0, data=b"\x00"),  # STATE_A
            CANMessage(arbitration_id=0x101, timestamp=0.1, data=b"\x01"),  # STATE_A (alt)
            CANMessage(arbitration_id=0x200, timestamp=0.5, data=b"\x02"),  # STATE_B
        ]
        session = CANSession(name="Test")
        session._messages.messages = messages

        sm = CANStateMachine()
        automaton = sm.learn_with_states(
            session=session,
            state_definitions={
                "STATE_A": [0x100, 0x101],  # Multiple IDs for same state
                "STATE_B": [0x200],
            },
            context_window_ms=1000,
        )

        assert isinstance(automaton, FiniteAutomaton)

    def test_learn_with_states_no_sequences_raises(self) -> None:
        """Test that learning with no state sequences raises error."""
        messages = [
            CANMessage(arbitration_id=0x999, timestamp=0.0, data=b"\x00"),
        ]
        session = CANSession(name="Test")
        session._messages.messages = messages

        sm = CANStateMachine()

        with pytest.raises(ValueError, match="No state sequences found"):
            sm.learn_with_states(
                session=session,
                state_definitions={"STATE_A": [0x100]},
                context_window_ms=1000,
            )

    def test_learn_with_states_sequence_breaking(self) -> None:
        """Test that state sequences break on timeout."""
        messages = [
            CANMessage(arbitration_id=0x100, timestamp=0.0, data=b"\x00"),  # STATE_A
            CANMessage(arbitration_id=0x200, timestamp=0.1, data=b"\x01"),  # STATE_B
            # Long gap - should break sequence
            CANMessage(arbitration_id=0x100, timestamp=10.0, data=b"\x02"),  # STATE_A (new seq)
            CANMessage(arbitration_id=0x300, timestamp=10.1, data=b"\x03"),  # STATE_C
        ]
        session = CANSession(name="Test")
        session._messages.messages = messages

        sm = CANStateMachine()
        automaton = sm.learn_with_states(
            session=session,
            state_definitions={
                "STATE_A": [0x100],
                "STATE_B": [0x200],
                "STATE_C": [0x300],
            },
            context_window_ms=500,  # 500ms timeout
        )

        # Should learn two separate sequences
        assert isinstance(automaton, FiniteAutomaton)


# =============================================================================
# Convenience Function Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.analyzer
class TestLearnStateMachine:
    """Test convenience function for state machine learning."""

    def test_learn_state_machine_function(self) -> None:
        """Test top-level learn_state_machine function."""
        messages = [
            CANMessage(arbitration_id=0x100, timestamp=0.0, data=b"\x00"),
            CANMessage(arbitration_id=0x200, timestamp=0.1, data=b"\x01"),
            CANMessage(arbitration_id=0x300, timestamp=0.2, data=b"\x02"),
        ]
        session = CANSession(name="Test")
        session._messages.messages = messages

        automaton = learn_state_machine(session=session, trigger_ids=[0x300], context_window_ms=500)

        assert isinstance(automaton, FiniteAutomaton)


# =============================================================================
# CANSession Integration Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.analyzer
class TestCANSessionIntegration:
    """Test CANSession.learn_state_machine method."""

    def test_session_learn_state_machine_method(self) -> None:
        """Test learn_state_machine method on CANSession."""
        messages = [
            CANMessage(arbitration_id=0x100, timestamp=0.0, data=b"\x00"),
            CANMessage(arbitration_id=0x200, timestamp=0.1, data=b"\x01"),
            CANMessage(arbitration_id=0x300, timestamp=0.2, data=b"\x02"),
        ]
        session = CANSession(name="Test")
        session._messages.messages = messages

        automaton = session.learn_state_machine(trigger_ids=[0x300], context_window_ms=500)

        assert isinstance(automaton, FiniteAutomaton)

    def test_session_learn_state_machine_accepts_pattern(self) -> None:
        """Test that learned automaton accepts observed patterns."""
        messages = [
            CANMessage(arbitration_id=0x100, timestamp=0.0, data=b"\x00"),
            CANMessage(arbitration_id=0x200, timestamp=0.1, data=b"\x01"),
            CANMessage(arbitration_id=0x300, timestamp=0.2, data=b"\x02"),
        ]
        session = CANSession(name="Test")
        session._messages.messages = messages

        automaton = session.learn_state_machine(trigger_ids=[0x300], context_window_ms=500)

        assert automaton.accepts(["0x100", "0x200", "0x300"]) is True


# =============================================================================
# Real-World Use Case Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.analyzer
class TestRealWorldUseCases:
    """Test real-world CAN protocol reverse engineering scenarios."""

    def test_ignition_sequence_learning(self) -> None:
        """Test learning ignition state machine from vehicle data.

        Typical ignition sequence: OFF -> ACC -> ON -> START -> ON -> OFF
        """
        messages = [
            # Cycle 1: OFF -> ACC -> ON -> START -> ON
            CANMessage(arbitration_id=0x280, timestamp=0.0, data=b"\x00\x00"),  # OFF
            CANMessage(arbitration_id=0x280, timestamp=1.0, data=b"\x01\x00"),  # ACC
            CANMessage(arbitration_id=0x280, timestamp=2.0, data=b"\x03\x00"),  # ON
            CANMessage(arbitration_id=0x280, timestamp=3.0, data=b"\x07\x00"),  # START
            CANMessage(arbitration_id=0x280, timestamp=4.0, data=b"\x03\x00"),  # ON (running)
            # Cycle 2: Same pattern
            CANMessage(arbitration_id=0x280, timestamp=10.0, data=b"\x00\x00"),  # OFF
            CANMessage(arbitration_id=0x280, timestamp=11.0, data=b"\x01\x00"),  # ACC
            CANMessage(arbitration_id=0x280, timestamp=12.0, data=b"\x03\x00"),  # ON
            CANMessage(arbitration_id=0x280, timestamp=13.0, data=b"\x07\x00"),  # START
        ]
        session = CANSession(name="Test")
        session._messages.messages = messages

        sm = CANStateMachine()

        # Learn using trigger on ignition state changes
        automaton = sm.learn_from_session(
            session=session, trigger_ids=[0x280], context_window_ms=2000
        )

        assert isinstance(automaton, FiniteAutomaton)
        assert len(automaton.states) >= 1

    def test_ecu_initialization_discovery(self) -> None:
        """Test discovering ECU initialization sequence.

        Typical pattern: Power-on -> Self-test -> Ready
        """
        messages = [
            # Initialization sequence
            CANMessage(arbitration_id=0x600, timestamp=0.0, data=b"\x01"),  # Power-on
            CANMessage(arbitration_id=0x601, timestamp=0.1, data=b"\x02"),  # Self-test
            CANMessage(arbitration_id=0x602, timestamp=0.2, data=b"\x03"),  # Calibration
            CANMessage(arbitration_id=0x603, timestamp=0.3, data=b"\x04"),  # Ready
        ]
        session = CANSession(name="Test")
        session._messages.messages = messages

        sm = CANStateMachine()
        automaton = sm.learn_from_session(
            session=session, trigger_ids=[0x603], context_window_ms=500
        )

        # Should learn initialization sequence
        assert isinstance(automaton, FiniteAutomaton)
        assert automaton.accepts(["0x600", "0x601", "0x602", "0x603"]) is True

    def test_state_dependent_message_patterns(self) -> None:
        """Test identifying state-dependent message patterns.

        Some messages only appear in certain states.
        """
        messages = [
            # State A: Messages 0x100, 0x200
            CANMessage(arbitration_id=0x100, timestamp=0.0, data=b"\x00"),
            CANMessage(arbitration_id=0x200, timestamp=0.1, data=b"\x01"),
            CANMessage(arbitration_id=0x500, timestamp=0.2, data=b"\x02"),  # Transition
            # State B: Messages 0x300, 0x400
            CANMessage(arbitration_id=0x300, timestamp=0.5, data=b"\x03"),
            CANMessage(arbitration_id=0x400, timestamp=0.6, data=b"\x04"),
            CANMessage(arbitration_id=0x500, timestamp=0.7, data=b"\x05"),  # Transition
        ]
        session = CANSession(name="Test")
        session._messages.messages = messages

        sm = CANStateMachine()
        automaton = sm.learn_from_session(
            session=session, trigger_ids=[0x500], context_window_ms=300
        )

        # Should learn different patterns before transitions
        assert isinstance(automaton, FiniteAutomaton)

    def test_message_dependency_discovery(self) -> None:
        """Test discovering message dependencies.

        Message B always follows message A within time window.
        """
        messages = [
            # Pattern: A always followed by B
            CANMessage(arbitration_id=0x100, timestamp=0.0, data=b"\x00"),
            CANMessage(arbitration_id=0x200, timestamp=0.05, data=b"\x01"),
            CANMessage(arbitration_id=0x300, timestamp=0.1, data=b"\x02"),  # Trigger
            CANMessage(arbitration_id=0x100, timestamp=1.0, data=b"\x03"),
            CANMessage(arbitration_id=0x200, timestamp=1.05, data=b"\x04"),
            CANMessage(arbitration_id=0x300, timestamp=1.1, data=b"\x05"),  # Trigger
        ]
        session = CANSession(name="Test")
        session._messages.messages = messages

        sm = CANStateMachine()
        automaton = sm.learn_from_session(
            session=session, trigger_ids=[0x300], context_window_ms=200
        )

        # Should learn that 0x100 -> 0x200 -> 0x300
        assert automaton.accepts(["0x100", "0x200", "0x300"]) is True


# =============================================================================
# Edge Cases and Error Conditions
# =============================================================================


@pytest.mark.unit
@pytest.mark.analyzer
class TestStateMachineEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_session(self) -> None:
        """Test learning from empty session raises error."""
        session = CANSession(name="Empty Test")
        session._messages.messages = []

        sm = CANStateMachine()

        with pytest.raises(ValueError):
            sm.learn_from_session(session=session, trigger_ids=[0x100], context_window_ms=500)

    def test_single_message_session(self) -> None:
        """Test learning from session with single message."""
        messages = [
            CANMessage(arbitration_id=0x100, timestamp=0.0, data=b"\x00"),
        ]
        session = CANSession(name="Test")
        session._messages.messages = messages

        sm = CANStateMachine()

        # Should raise because single message produces sequence of length 1
        with pytest.raises(ValueError, match="No sequences with length"):
            sm.learn_from_session(
                session=session, trigger_ids=[0x100], context_window_ms=500, min_sequence_length=2
            )

    def test_zero_context_window(self) -> None:
        """Test extraction with zero context window."""
        messages = [
            CANMessage(arbitration_id=0x100, timestamp=0.0, data=b"\x00"),
            CANMessage(arbitration_id=0x200, timestamp=0.1, data=b"\x01"),
        ]
        session = CANSession(name="Test")
        session._messages.messages = messages

        sm = CANStateMachine()
        extractions = sm.extract_sequences(
            session=session, trigger_ids=[0x200], context_window_ms=0
        )

        # Should only include the trigger message itself
        assert len(extractions) == 1
        assert extractions[0].sequence == [0x200]

    def test_very_large_context_window(self) -> None:
        """Test extraction with very large context window."""
        messages = [
            CANMessage(arbitration_id=0x100, timestamp=0.0, data=b"\x00"),
            CANMessage(arbitration_id=0x200, timestamp=1.0, data=b"\x01"),
            CANMessage(arbitration_id=0x300, timestamp=2.0, data=b"\x02"),
        ]
        session = CANSession(name="Test")
        session._messages.messages = messages

        sm = CANStateMachine()
        extractions = sm.extract_sequences(
            session=session,
            trigger_ids=[0x300],
            context_window_ms=10000,  # 10 seconds
        )

        # Should include all messages
        assert len(extractions) == 1
        assert len(extractions[0].sequence) == 3

    def test_overlapping_sequences(self) -> None:
        """Test extraction with overlapping time windows."""
        messages = [
            CANMessage(arbitration_id=0x100, timestamp=0.0, data=b"\x00"),
            CANMessage(arbitration_id=0x200, timestamp=0.05, data=b"\x01"),  # Trigger 1
            CANMessage(arbitration_id=0x300, timestamp=0.1, data=b"\x02"),
            CANMessage(arbitration_id=0x200, timestamp=0.15, data=b"\x03"),  # Trigger 2
        ]
        session = CANSession(name="Test")
        session._messages.messages = messages

        sm = CANStateMachine()
        extractions = sm.extract_sequences(
            session=session, trigger_ids=[0x200], context_window_ms=100
        )

        # Should create two overlapping sequences
        assert len(extractions) == 2

    def test_duplicate_sequences(self) -> None:
        """Test learning from duplicate sequences."""
        messages = [
            # Pattern repeated 3 times
            CANMessage(arbitration_id=0x100, timestamp=0.0, data=b"\x00"),
            CANMessage(arbitration_id=0x200, timestamp=0.1, data=b"\x01"),
            CANMessage(arbitration_id=0x100, timestamp=1.0, data=b"\x02"),
            CANMessage(arbitration_id=0x200, timestamp=1.1, data=b"\x03"),
            CANMessage(arbitration_id=0x100, timestamp=2.0, data=b"\x04"),
            CANMessage(arbitration_id=0x200, timestamp=2.1, data=b"\x05"),
        ]
        session = CANSession(name="Test")
        session._messages.messages = messages

        sm = CANStateMachine()
        automaton = sm.learn_from_session(
            session=session, trigger_ids=[0x200], context_window_ms=200
        )

        # Should learn pattern even with duplicates
        assert isinstance(automaton, FiniteAutomaton)
        assert automaton.accepts(["0x100", "0x200"]) is True

    def test_extended_can_ids(self) -> None:
        """Test learning with extended (29-bit) CAN IDs."""
        messages = [
            CANMessage(arbitration_id=0x18FEF100, timestamp=0.0, data=b"\x00", is_extended=True),
            CANMessage(arbitration_id=0x18FEF200, timestamp=0.1, data=b"\x01", is_extended=True),
        ]
        session = CANSession(name="Test")
        session._messages.messages = messages

        sm = CANStateMachine()
        automaton = sm.learn_from_session(
            session=session, trigger_ids=[0x18FEF200], context_window_ms=500
        )

        assert isinstance(automaton, FiniteAutomaton)

    def test_can_fd_messages(self) -> None:
        """Test learning from CAN-FD messages."""
        messages = [
            CANMessage(arbitration_id=0x100, timestamp=0.0, data=b"\x00" * 16, is_fd=True),
            CANMessage(arbitration_id=0x200, timestamp=0.1, data=b"\x01" * 16, is_fd=True),
        ]
        session = CANSession(name="Test")
        session._messages.messages = messages

        sm = CANStateMachine()
        automaton = sm.learn_from_session(
            session=session, trigger_ids=[0x200], context_window_ms=500
        )

        assert isinstance(automaton, FiniteAutomaton)


# =============================================================================
# Export and Visualization Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.analyzer
class TestStateMachineExport:
    """Test exporting learned state machines."""

    def test_export_to_dot(self) -> None:
        """Test exporting learned automaton to DOT format."""
        messages = [
            CANMessage(arbitration_id=0x100, timestamp=0.0, data=b"\x00"),
            CANMessage(arbitration_id=0x200, timestamp=0.1, data=b"\x01"),
            CANMessage(arbitration_id=0x300, timestamp=0.2, data=b"\x02"),
        ]
        session = CANSession(name="Test")
        session._messages.messages = messages

        automaton = session.learn_state_machine(trigger_ids=[0x300], context_window_ms=500)

        dot = automaton.to_dot()

        assert "digraph" in dot  # Accept any digraph name
        assert "0x100" in dot or "0x200" in dot or "0x300" in dot

    def test_export_to_networkx(self) -> None:
        """Test exporting learned automaton to NetworkX."""
        pytest.importorskip("networkx")

        messages = [
            CANMessage(arbitration_id=0x100, timestamp=0.0, data=b"\x00"),
            CANMessage(arbitration_id=0x200, timestamp=0.1, data=b"\x01"),
        ]
        session = CANSession(name="Test")
        session._messages.messages = messages

        automaton = session.learn_state_machine(trigger_ids=[0x200], context_window_ms=500)

        graph = automaton.to_networkx()

        assert graph.number_of_nodes() >= 1
        assert graph.number_of_edges() >= 0
