"""CAN reverse engineering session.

This module provides the main user-facing API for CAN bus reverse engineering,
centered around the CANSession class which manages message collections and
provides discovery-oriented analysis workflows.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from oscura.automotive.can.analysis import MessageAnalyzer
from oscura.automotive.can.models import (
    CANMessage,
    CANMessageList,
    MessageAnalysis,
)
from oscura.automotive.can.patterns import (
    MessagePair,
    MessageSequence,
    PatternAnalyzer,
    TemporalCorrelation,
)

if TYPE_CHECKING:
    from oscura.automotive.can.message_wrapper import CANMessageWrapper
    from oscura.automotive.can.stimulus_response import StimulusResponseReport
    from oscura.inference.state_machine import FiniteAutomaton

__all__ = ["CANSession"]


class CANSession:
    """CAN bus reverse engineering session.

    This is the primary API for discovering and analyzing unknown CAN bus
    protocols. It provides:
    - Message inventory and filtering
    - Per-message statistical analysis
    - Discovery-oriented workflows
    - Hypothesis testing
    - Documentation generation

    Example - Discovery workflow:
        >>> session = CANSession.from_log("capture.blf")
        >>> inventory = session.inventory()
        >>> print(inventory)
        >>>
        >>> # Focus on a specific message
        >>> msg = session.message(0x280)
        >>> analysis = msg.analyze()
        >>> print(analysis.summary())
        >>>
        >>> # Test hypothesis
        >>> hypothesis = msg.test_hypothesis(
        ...     signal_name="rpm",
        ...     start_byte=2,
        ...     bit_length=16,
        ...     scale=0.25
        ... )

    Example - Known protocol decoding:
        >>> session = CANSession.from_log("capture.blf")
        >>> from oscura.automotive.dbc import load_dbc
        >>> dbc = load_dbc("vehicle.dbc")
        >>> decoded = session.decode(dbc)
    """

    def __init__(self, messages: CANMessageList | None = None):
        """Initialize CAN session.

        Args:
            messages: Initial message collection (optional).
        """
        self._messages = messages or CANMessageList()
        self._analyses_cache: dict[int, MessageAnalysis] = {}

    @classmethod
    def from_log(cls, file_path: Path | str) -> CANSession:
        """Create session from automotive log file.

        Automatically detects file format (BLF, ASC, MDF, CSV) and loads.

        Args:
            file_path: Path to log file.

        Returns:
            New CANSession with loaded messages.
        """
        from oscura.automotive.loaders import load_automotive_log

        messages = load_automotive_log(file_path)
        return cls(messages=messages)

    @classmethod
    def from_messages(cls, messages: list[CANMessage]) -> CANSession:
        """Create session from list of CAN messages.

        Args:
            messages: List of CAN messages.

        Returns:
            New CANSession.
        """
        msg_list = CANMessageList(messages=messages)
        return cls(messages=msg_list)

    def inventory(self) -> pd.DataFrame:
        """Generate message inventory.

        Returns a pandas DataFrame with one row per unique CAN ID, showing:
        - arbitration_id: CAN ID
        - count: Number of messages
        - frequency_hz: Average frequency in Hz
        - period_ms: Average period in milliseconds
        - first_seen: Timestamp of first message
        - last_seen: Timestamp of last message
        - dlc: Data length (bytes)

        Returns:
            DataFrame with message inventory.
        """
        unique_ids = sorted(self._messages.unique_ids())

        inventory_data = []
        for arb_id in unique_ids:
            filtered = self._messages.filter_by_id(arb_id)
            timestamps = [msg.timestamp for msg in filtered.messages]

            count = len(filtered)
            first_seen = min(timestamps)
            last_seen = max(timestamps)
            duration = last_seen - first_seen

            if duration > 0 and count > 1:
                frequency_hz = (count - 1) / duration
                period_ms = (duration / (count - 1)) * 1000
            else:
                frequency_hz = 0.0
                period_ms = 0.0

            # Get DLC from first message
            dlc = filtered.messages[0].dlc

            inventory_data.append(
                {
                    "arbitration_id": f"0x{arb_id:03X}",
                    "count": count,
                    "frequency_hz": f"{frequency_hz:.1f}",
                    "period_ms": f"{period_ms:.1f}",
                    "first_seen": f"{first_seen:.6f}",
                    "last_seen": f"{last_seen:.6f}",
                    "dlc": dlc,
                }
            )

        return pd.DataFrame(inventory_data)

    def message(self, arbitration_id: int) -> CANMessageWrapper:
        """Get message wrapper for analysis of a specific CAN ID.

        Args:
            arbitration_id: CAN ID to analyze.

        Returns:
            CANMessageWrapper for this message ID.

        Raises:
            ValueError: If no messages with this ID exist.
        """
        from oscura.automotive.can.message_wrapper import CANMessageWrapper

        filtered = self._messages.filter_by_id(arbitration_id)
        if not filtered.messages:
            raise ValueError(f"No messages found for ID 0x{arbitration_id:03X}")

        return CANMessageWrapper(self, arbitration_id)

    def analyze_message(self, arbitration_id: int, force_refresh: bool = False) -> MessageAnalysis:
        """Analyze a specific message ID.

        Args:
            arbitration_id: CAN ID to analyze.
            force_refresh: Force re-analysis even if cached.

        Returns:
            MessageAnalysis with complete analysis.
        """
        # Check cache
        if not force_refresh and arbitration_id in self._analyses_cache:
            return self._analyses_cache[arbitration_id]

        # Perform analysis
        analysis = MessageAnalyzer.analyze_message_id(self._messages, arbitration_id)

        # Cache result
        self._analyses_cache[arbitration_id] = analysis

        return analysis

    def filter(
        self,
        min_frequency: float | None = None,
        max_frequency: float | None = None,
        arbitration_ids: list[int] | None = None,
        time_range: tuple[float, float] | None = None,
    ) -> CANSession:
        """Filter messages and return new session.

        Args:
            min_frequency: Minimum message frequency in Hz.
            max_frequency: Maximum message frequency in Hz.
            arbitration_ids: List of CAN IDs to include.
            time_range: Tuple of (start_time, end_time) in seconds.

        Returns:
            New CANSession with filtered messages.
        """
        filtered_messages = []

        # First, filter by time range if specified
        if time_range:
            start_time, end_time = time_range
            for msg in self._messages:
                if start_time <= msg.timestamp <= end_time:
                    filtered_messages.append(msg)
        else:
            filtered_messages = list(self._messages)

        # Filter by CAN IDs if specified
        if arbitration_ids:
            filtered_messages = [
                msg for msg in filtered_messages if msg.arbitration_id in arbitration_ids
            ]

        # Filter by frequency if specified
        if min_frequency or max_frequency:
            # Group by ID and calculate frequencies
            from collections import defaultdict

            id_messages: dict[int, list[CANMessage]] = defaultdict(list)
            for msg in filtered_messages:
                id_messages[msg.arbitration_id].append(msg)

            # Filter IDs by frequency
            valid_ids = set()
            for arb_id, msgs in id_messages.items():
                if len(msgs) > 1:
                    timestamps = [m.timestamp for m in msgs]
                    duration = max(timestamps) - min(timestamps)
                    if duration > 0:
                        freq = (len(msgs) - 1) / duration

                        if min_frequency and freq < min_frequency:
                            continue
                        if max_frequency and freq > max_frequency:
                            continue

                valid_ids.add(arb_id)

            filtered_messages = [
                msg for msg in filtered_messages if msg.arbitration_id in valid_ids
            ]

        return CANSession.from_messages(filtered_messages)

    def unique_ids(self) -> set[int]:
        """Get set of unique CAN IDs in this session.

        Returns:
            Set of unique arbitration IDs.
        """
        return self._messages.unique_ids()

    def time_range(self) -> tuple[float, float]:
        """Get time range of all messages.

        Returns:
            Tuple of (first_timestamp, last_timestamp).
        """
        return self._messages.time_range()

    def __len__(self) -> int:
        """Return total number of messages."""
        return len(self._messages)

    def compare_to(self, other_session: CANSession) -> StimulusResponseReport:
        """Compare this session to another to detect changes.

        This method is useful for stimulus-response analysis where you compare
        a baseline capture (no user action) against a stimulus capture (with
        user action) to identify which messages and signals respond.

        Args:
            other_session: Session to compare against (treated as stimulus).

        Returns:
            StimulusResponseReport with detected changes.

        Example - Brake pedal analysis:
            >>> baseline = CANSession.from_log("no_brake.blf")
            >>> stimulus = CANSession.from_log("brake_pressed.blf")
            >>> report = baseline.compare_to(stimulus)
            >>> print(report.summary())
            >>> # Show which messages changed
            >>> for msg_id in report.changed_messages:
            ...     print(f"0x{msg_id:03X} responded to brake press")

        Example - Throttle position analysis:
            >>> idle = CANSession.from_log("idle.blf")
            >>> throttle = CANSession.from_log("throttle_50pct.blf")
            >>> report = idle.compare_to(throttle)
            >>> # Examine byte-level changes
            >>> for msg_id, changes in report.byte_changes.items():
            ...     print(f"Message 0x{msg_id:03X}:")
            ...     for change in changes:
            ...         print(f"  Byte {change.byte_position}: {change.change_magnitude:.2f}")
        """
        from oscura.automotive.can.stimulus_response import (
            StimulusResponseAnalyzer,
        )

        analyzer = StimulusResponseAnalyzer()
        return analyzer.detect_responses(self, other_session)

    def find_message_pairs(
        self,
        time_window_ms: float = 100,
        min_occurrence: int = 3,
    ) -> list[MessagePair]:
        """Find message pairs that frequently occur together.

        Discovers request-response patterns and coordinated message transmissions
        by detecting messages that consistently appear within a short time window.

        Args:
            time_window_ms: Maximum time window in milliseconds.
            min_occurrence: Minimum number of occurrences to report.

        Returns:
            List of MessagePair objects, sorted by occurrence count.

        Example:
            >>> session = CANSession.from_log("capture.blf")
            >>> pairs = session.find_message_pairs(time_window_ms=50)
            >>> for pair in pairs[:5]:
            ...     print(pair)
        """
        return PatternAnalyzer.find_message_pairs(
            self, time_window_ms=time_window_ms, min_occurrence=min_occurrence
        )

    def find_message_sequences(
        self,
        max_sequence_length: int = 5,
        time_window_ms: float = 500,
        min_support: float = 0.7,
    ) -> list[MessageSequence]:
        """Find message sequences (A → B → C patterns).

        Discovers multi-step control sequences or protocol handshakes by
        mining sequential patterns in the message stream.

        Args:
            max_sequence_length: Maximum length of sequences (2-10).
            time_window_ms: Maximum time window for entire sequence.
            min_support: Minimum support score (0.0-1.0).

        Returns:
            List of MessageSequence objects, sorted by support.

        Example:
            >>> session = CANSession.from_log("startup.blf")
            >>> sequences = session.find_message_sequences(
            ...     max_sequence_length=3,
            ...     time_window_ms=1000
            ... )
            >>> for seq in sequences[:5]:
            ...     print(seq)
        """
        return PatternAnalyzer.find_message_sequences(
            self,
            max_sequence_length=max_sequence_length,
            time_window_ms=time_window_ms,
            min_support=min_support,
        )

    def find_temporal_correlations(
        self,
        max_delay_ms: float = 100,
    ) -> dict[tuple[int, int], TemporalCorrelation]:
        """Find temporal correlations between messages.

        Analyzes timing relationships to determine which messages consistently
        follow others with predictable delays.

        Args:
            max_delay_ms: Maximum delay to consider for correlations.

        Returns:
            Dictionary mapping (leader_id, follower_id) to correlation info.

        Example:
            >>> session = CANSession.from_log("capture.blf")
            >>> correlations = session.find_temporal_correlations(max_delay_ms=50)
            >>> for (leader, follower), corr in correlations.items():
            ...     print(f"0x{leader:03X} → 0x{follower:03X}: {corr.avg_delay_ms:.2f}ms")
        """
        return PatternAnalyzer.find_temporal_correlations(self, max_delay_ms=max_delay_ms)

    def learn_state_machine(
        self, trigger_ids: list[int], context_window_ms: float = 500
    ) -> FiniteAutomaton:
        """Learn state machine from message sequences.

        This method integrates Oscura's state machine inference to learn
        protocol state machines from CAN message sequences around trigger messages.

        Args:
            trigger_ids: CAN IDs that trigger sequence extraction.
            context_window_ms: Time window (ms) before trigger to capture sequences.

        Returns:
            Learned finite automaton representing the state machine.

        Example:
            >>> session = CANSession.from_log("ignition_cycles.blf")
            >>> automaton = session.learn_state_machine(
            ...     trigger_ids=[0x280],
            ...     context_window_ms=500
            ... )
            >>> print(automaton.to_dot())
        """
        from oscura.automotive.can.state_machine import learn_state_machine

        return learn_state_machine(
            session=self, trigger_ids=trigger_ids, context_window_ms=context_window_ms
        )

    def __repr__(self) -> str:
        """Human-readable representation."""
        num_messages = len(self._messages)
        num_ids = len(self.unique_ids())
        time_start, time_end = self.time_range()
        duration = time_end - time_start

        return (
            f"CANSession({num_messages} messages, {num_ids} unique IDs, duration={duration:.2f}s)"
        )
