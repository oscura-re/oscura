"""State machine learning for CAN protocol reverse engineering.

This module integrates Oscura's state machine inference capabilities with
CAN bus analysis to learn protocol state machines from message sequences.

Key capabilities:
- Learn state machines from CAN message sequences
- Extract sequences around trigger messages
- Map CAN message patterns to states
- Discover protocol initialization sequences
- Identify state-dependent message patterns

Use cases:
- Learn ignition sequence state machines (key off → acc → on → start)
- Discover ECU initialization sequences
- Identify state-dependent message patterns
- Find message dependencies and ordering constraints
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from oscura.inference.state_machine import FiniteAutomaton, StateMachineInferrer

if TYPE_CHECKING:
    from oscura.automotive.can.session import CANSession

__all__ = ["CANStateMachine", "SequenceExtraction"]


@dataclass
class SequenceExtraction:
    """A sequence of CAN messages extracted around a trigger.

    Attributes:
        trigger_id: CAN ID that triggered extraction.
        trigger_timestamp: Timestamp of trigger message.
        sequence: List of CAN IDs in the sequence.
        timestamps: Corresponding timestamps for each message.
        window_start: Start of time window.
        window_end: End of time window.
    """

    trigger_id: int
    trigger_timestamp: float
    sequence: list[int]
    timestamps: list[float]
    window_start: float
    window_end: float

    def to_symbol_sequence(self) -> list[str]:
        """Convert CAN IDs to symbol strings for state machine learning.

        Returns:
            List of symbols (CAN IDs as hex strings).
        """
        return [f"0x{can_id:03X}" for can_id in self.sequence]


class CANStateMachine:
    """Learn CAN protocol state machines from message sequences.

    This class wraps StateMachineInferrer for CAN-specific use cases,
    handling the extraction of message sequences from CAN sessions and
    converting them to the format needed for state machine learning.

    The learned state machines reveal:
    - Protocol initialization sequences
    - State transitions triggered by specific messages
    - Message ordering constraints
    - State-dependent message patterns

    Example - Learn ignition sequence:
        >>> from oscura.automotive.sources import FileSource
        >>> session = CANSession(name="Ignition Analysis")
        >>> session.add_recording("cycles", FileSource("ignition_cycles.blf"))
        >>> sm = CANStateMachine()
        >>> # Use ignition-related CAN IDs as triggers
        >>> automaton = sm.learn_from_session(
        ...     session,
        ...     trigger_ids=[0x280],  # Engine status message
        ...     context_window_ms=500
        ... )
        >>> # Export for visualization
        >>> print(automaton.to_dot())

    Example - Discover initialization sequence:
        >>> session = CANSession(name="ECU Startup")
        >>> session.add_recording("startup", FileSource("ecu_startup.blf"))
        >>> sm = CANStateMachine()
        >>> # Use diagnostic messages as triggers
        >>> automaton = sm.learn_from_session(
        ...     session,
        ...     trigger_ids=[0x7E0, 0x7E8],
        ...     context_window_ms=1000
        ... )
    """

    def __init__(self) -> None:
        """Initialize CAN state machine learner."""
        self._inferrer = StateMachineInferrer()

    def learn_from_session(
        self,
        session: CANSession,
        trigger_ids: list[int],
        context_window_ms: float = 500,
        min_sequence_length: int = 2,
    ) -> FiniteAutomaton:
        """Learn state machine from CAN session.

        Extracts sequences of messages around trigger messages and learns
        a finite automaton that captures the observed patterns.

        Args:
            session: CAN session with messages to analyze.
            trigger_ids: CAN IDs that trigger sequence extraction.
            context_window_ms: Time window (ms) before trigger to capture.
            min_sequence_length: Minimum sequence length to include.

        Returns:
            Learned finite automaton representing state machine.

        Raises:
            ValueError: If no sequences could be extracted.
        """
        # Extract sequences around triggers
        extractions = self.extract_sequences(
            session=session,
            trigger_ids=trigger_ids,
            context_window_ms=context_window_ms,
        )

        if not extractions:
            raise ValueError(
                f"No sequences found for trigger IDs {[f'0x{tid:03X}' for tid in trigger_ids]}"
            )

        # Convert to symbol sequences for learning
        traces = []
        for extraction in extractions:
            symbol_seq = extraction.to_symbol_sequence()
            if len(symbol_seq) >= min_sequence_length:
                traces.append(symbol_seq)

        if not traces:
            raise ValueError(
                f"No sequences with length >= {min_sequence_length} found. "
                f"Try increasing context_window_ms or reducing min_sequence_length."
            )

        # Learn state machine using RPNI
        # Cast: list[list[str]] is compatible with list[list[str | int]] at runtime
        automaton = self._inferrer.infer_rpni(positive_traces=cast("list[list[str | int]]", traces))

        return automaton

    def extract_sequences(
        self,
        session: CANSession,
        trigger_ids: list[int],
        context_window_ms: float = 500,
    ) -> list[SequenceExtraction]:
        """Extract message sequences around trigger messages.

        For each occurrence of a trigger ID, extract all messages within
        the specified time window before the trigger.

        Args:
            session: CAN session to extract from.
            trigger_ids: CAN IDs that trigger extraction.
            context_window_ms: Time window (ms) before trigger.

        Returns:
            List of extracted sequences.
        """
        context_window_s = context_window_ms / 1000.0
        extractions = []

        # Get all messages sorted by timestamp
        all_messages = sorted(session._messages.messages, key=lambda m: m.timestamp)

        # Find trigger messages
        trigger_messages = [msg for msg in all_messages if msg.arbitration_id in trigger_ids]

        # Extract context around each trigger
        for trigger_msg in trigger_messages:
            window_start = trigger_msg.timestamp - context_window_s
            window_end = trigger_msg.timestamp

            # Find messages in window
            sequence_msgs = [
                msg for msg in all_messages if window_start <= msg.timestamp <= window_end
            ]

            if not sequence_msgs:
                continue

            # Create extraction
            extraction = SequenceExtraction(
                trigger_id=trigger_msg.arbitration_id,
                trigger_timestamp=trigger_msg.timestamp,
                sequence=[msg.arbitration_id for msg in sequence_msgs],
                timestamps=[msg.timestamp for msg in sequence_msgs],
                window_start=window_start,
                window_end=window_end,
            )
            extractions.append(extraction)

        return extractions

    def learn_with_states(
        self,
        session: CANSession,
        state_definitions: dict[str, list[int]],
        context_window_ms: float = 500,
    ) -> FiniteAutomaton:
        """Learn state machine with predefined state labels.

        Instead of using trigger IDs, this method allows you to define
        states explicitly and learn transitions between them.

        Args:
            session: CAN session to analyze.
            state_definitions: Mapping of state names to CAN IDs that
                indicate that state (e.g., {"IGNITION_OFF": [0x123],
                "IGNITION_ACC": [0x124], "IGNITION_ON": [0x125]}).
            context_window_ms: Time window for state detection.

        Returns:
            Learned finite automaton with state labels.

        Raises:
            ValueError: If no state sequences could be extracted.
        """
        context_window_s = context_window_ms / 1000.0

        # Build reverse mapping: CAN ID -> state name
        id_to_state: dict[int, str] = {}
        for state_name, can_ids in state_definitions.items():
            for can_id in can_ids:
                id_to_state[can_id] = state_name

        # Get all messages sorted by timestamp
        all_messages = sorted(session._messages.messages, key=lambda m: m.timestamp)

        # Detect state transitions
        state_sequences = []
        current_sequence = []
        last_state_time = None

        for msg in all_messages:
            if msg.arbitration_id in id_to_state:
                state_name = id_to_state[msg.arbitration_id]

                # Check if this is part of current sequence
                if last_state_time is None or (msg.timestamp - last_state_time) <= context_window_s:
                    current_sequence.append(state_name)
                else:
                    # New sequence
                    if len(current_sequence) >= 2:
                        state_sequences.append(current_sequence)
                    current_sequence = [state_name]

                last_state_time = msg.timestamp

        # Add final sequence
        if len(current_sequence) >= 2:
            state_sequences.append(current_sequence)

        if not state_sequences:
            raise ValueError(
                f"No state sequences found. Check state_definitions: {state_definitions}"
            )

        # Learn state machine
        # Cast: list[list[str]] is compatible with list[list[str | int]] at runtime
        state_sequences_union: list[list[str | int]] = state_sequences  # type: ignore[assignment]
        automaton = self._inferrer.infer_rpni(positive_traces=state_sequences_union)

        return automaton


def learn_state_machine(
    session: CANSession,
    trigger_ids: list[int],
    context_window_ms: float = 500,
) -> FiniteAutomaton:
    """Convenience function to learn state machine from session.

    Args:
        session: CAN session to analyze.
        trigger_ids: CAN IDs that trigger sequence extraction.
        context_window_ms: Time window (ms) before trigger.

    Returns:
        Learned finite automaton.
    """
    learner = CANStateMachine()
    return learner.learn_from_session(
        session=session,
        trigger_ids=trigger_ids,
        context_window_ms=context_window_ms,
    )
