"""Multi-message pattern learning for CAN bus analysis.

This module provides algorithms for discovering relationships between CAN messages,
including message pairs, sequences, and temporal correlations.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from oscura.automotive.can.session import CANSession

__all__ = [
    "MessagePair",
    "MessageSequence",
    "PatternAnalyzer",
    "TemporalCorrelation",
]


@dataclass
class MessagePair:
    """A pair of CAN messages that frequently occur together.

    Attributes:
        id_a: First message arbitration ID.
        id_b: Second message arbitration ID.
        occurrences: Number of times this pair occurred within time window.
        avg_delay_ms: Average time delay from id_a to id_b in milliseconds.
        confidence: Confidence score (0.0-1.0) based on consistency.
    """

    id_a: int
    id_b: int
    occurrences: int
    avg_delay_ms: float
    confidence: float

    def __repr__(self) -> str:
        """Human-readable representation."""
        return (
            f"MessagePair(0x{self.id_a:03X} → 0x{self.id_b:03X}, "
            f"occurrences={self.occurrences}, "
            f"delay={self.avg_delay_ms:.2f}ms, "
            f"confidence={self.confidence:.2f})"
        )


@dataclass
class MessageSequence:
    """A sequence of CAN messages that occur in order.

    Attributes:
        ids: List of message arbitration IDs in sequence order.
        occurrences: Number of times this sequence occurred.
        avg_timing: Average delays between consecutive messages in milliseconds.
        support: Support score (0.0-1.0) - fraction of time sequence appears.
    """

    ids: list[int]
    occurrences: int
    avg_timing: list[float]
    support: float

    def __repr__(self) -> str:
        """Human-readable representation."""
        id_str = " → ".join(f"0x{id_:03X}" for id_ in self.ids)
        timing_str = " → ".join(f"{t:.1f}ms" for t in self.avg_timing)
        return (
            f"MessageSequence({id_str}, "
            f"occurrences={self.occurrences}, "
            f"timing=[{timing_str}], "
            f"support={self.support:.2f})"
        )


@dataclass
class TemporalCorrelation:
    """Temporal correlation between two CAN messages.

    Attributes:
        leader_id: Message ID that typically appears first.
        follower_id: Message ID that typically appears after leader.
        avg_delay_ms: Average delay from leader to follower in milliseconds.
        std_delay_ms: Standard deviation of delay in milliseconds.
        occurrences: Number of times this correlation was observed.
    """

    leader_id: int
    follower_id: int
    avg_delay_ms: float
    std_delay_ms: float
    occurrences: int

    def __repr__(self) -> str:
        """Human-readable representation."""
        return (
            f"TemporalCorrelation(0x{self.leader_id:03X} → 0x{self.follower_id:03X}, "
            f"delay={self.avg_delay_ms:.2f}±{self.std_delay_ms:.2f}ms, "
            f"occurrences={self.occurrences})"
        )


class PatternAnalyzer:
    """Detect patterns in CAN message sequences.

    This class provides static methods for discovering relationships between
    CAN messages, useful for understanding message dependencies and control flows.

    Example - Find message pairs:
        >>> from oscura.automotive.sources import FileSource
        >>> session = CANSession(name="Analysis")
        >>> session.add_recording("main", FileSource("capture.blf"))
        >>> pairs = PatternAnalyzer.find_message_pairs(session, time_window_ms=100)
        >>> for pair in pairs:
        ...     print(pair)

    Example - Find message sequences:
        >>> sequences = PatternAnalyzer.find_message_sequences(
        ...     session,
        ...     max_sequence_length=3,
        ...     time_window_ms=500
        ... )

    Example - Find temporal correlations:
        >>> correlations = PatternAnalyzer.find_temporal_correlations(
        ...     session,
        ...     max_delay_ms=100
        ... )
    """

    @staticmethod
    def find_message_pairs(
        session: CANSession,
        time_window_ms: float = 100,
        min_occurrence: int = 3,
    ) -> list[MessagePair]:
        """Find message pairs that frequently occur together.

        Uses a sliding time window to detect messages that consistently appear
        within a short time of each other. This can reveal request-response
        patterns or coordinated control messages.

        Args:
            session: CAN session to analyze.
            time_window_ms: Maximum time window in milliseconds to consider
                messages as a pair.
            min_occurrence: Minimum number of occurrences to report a pair.

        Returns:
            List of MessagePair objects, sorted by occurrence count (descending).
        """
        time_window_s = time_window_ms / 1000.0

        # Get all messages sorted by timestamp
        all_messages = sorted(session._messages.messages, key=lambda m: m.timestamp)

        # Track pairs: (id_a, id_b) -> list of delays
        pair_delays: dict[tuple[int, int], list[float]] = defaultdict(list)

        # Use sliding window approach
        for i, msg_a in enumerate(all_messages):
            # Look forward in time window
            for msg_b in all_messages[i + 1 :]:
                delay = msg_b.timestamp - msg_a.timestamp
                if delay > time_window_s:
                    break  # Outside window, stop searching

                # Don't pair message with itself (same ID)
                if msg_a.arbitration_id != msg_b.arbitration_id:
                    pair_key = (msg_a.arbitration_id, msg_b.arbitration_id)
                    pair_delays[pair_key].append(delay * 1000)  # Convert to ms

        # Build MessagePair objects
        pairs = []
        for (id_a, id_b), delays in pair_delays.items():
            occurrences = len(delays)
            if occurrences >= min_occurrence:
                avg_delay = np.mean(delays)
                std_delay = np.std(delays)

                # Confidence based on consistency (lower std = higher confidence)
                # Normalize std by average delay
                if avg_delay > 0:
                    cv = std_delay / avg_delay  # Coefficient of variation
                    confidence = float(1.0 / (1.0 + cv))  # Higher confidence for lower CV
                else:
                    confidence = 1.0

                pairs.append(
                    MessagePair(
                        id_a=id_a,
                        id_b=id_b,
                        occurrences=occurrences,
                        avg_delay_ms=float(avg_delay),
                        confidence=float(confidence),
                    )
                )

        # Sort by occurrence count (descending)
        pairs.sort(key=lambda p: p.occurrences, reverse=True)

        return pairs

    @staticmethod
    def find_message_sequences(
        session: CANSession,
        max_sequence_length: int = 5,
        time_window_ms: float = 500,
        min_support: float = 0.7,
    ) -> list[MessageSequence]:
        """Find message sequences (A → B → C patterns).

        Uses sequential pattern mining to discover message sequences that
        frequently occur in order. This can reveal multi-step control
        sequences or protocol handshakes.

        Args:
            session: CAN session to analyze.
            max_sequence_length: Maximum length of sequences to find (2-10).
            time_window_ms: Maximum time window for entire sequence.
            min_support: Minimum support (0.0-1.0) - fraction of occurrences
                relative to most frequent message.

        Returns:
            List of MessageSequence objects, sorted by support (descending).

        Raises:
            ValueError: If max_sequence_length is invalid (<2 or >10) or
                min_support is not in range [0.0, 1.0].
        """
        if max_sequence_length < 2:
            raise ValueError("max_sequence_length must be at least 2")
        if max_sequence_length > 10:
            raise ValueError("max_sequence_length cannot exceed 10")
        if not 0.0 <= min_support <= 1.0:
            raise ValueError("min_support must be between 0.0 and 1.0")

        time_window_s = time_window_ms / 1000.0

        # Get all messages sorted by timestamp
        all_messages = sorted(session._messages.messages, key=lambda m: m.timestamp)

        if not all_messages:
            return []

        # Find maximum message frequency (for support calculation)
        id_counts: dict[int, int] = defaultdict(int)
        for msg in all_messages:
            id_counts[msg.arbitration_id] += 1
        max_count = max(id_counts.values()) if id_counts else 1

        # Mine sequences of increasing length
        sequences: dict[tuple[int, ...], list[list[float]]] = defaultdict(list)

        # Start with 2-message sequences
        for i, msg_a in enumerate(all_messages):
            # Look forward in time window
            sequence_start = msg_a.timestamp
            current_sequence = [msg_a.arbitration_id]
            timing = []

            for msg_b in all_messages[i + 1 :]:
                if msg_b.timestamp - sequence_start > time_window_s:
                    break

                # Extend sequence
                current_sequence.append(msg_b.arbitration_id)
                timing.append((msg_b.timestamp - msg_a.timestamp) * 1000)  # ms

                # Record sequences of desired length
                if 2 <= len(current_sequence) <= max_sequence_length:
                    seq_key = tuple(current_sequence)
                    sequences[seq_key].append(timing.copy())

                # Update for next iteration
                msg_a = msg_b
                if len(current_sequence) >= max_sequence_length:
                    break

        # Build MessageSequence objects
        result = []
        for seq_ids, timing_lists in sequences.items():
            occurrences = len(timing_lists)
            support = occurrences / max_count

            if support >= min_support:
                # Calculate average timing between consecutive messages
                avg_timing = []
                if timing_lists:
                    # timing_lists[i] has len(seq_ids) - 1 elements
                    num_gaps = len(seq_ids) - 1
                    for gap_idx in range(num_gaps):
                        gap_times = [t[gap_idx] for t in timing_lists if gap_idx < len(t)]
                        if gap_times:
                            avg_timing.append(float(np.mean(gap_times)))

                result.append(
                    MessageSequence(
                        ids=list(seq_ids),
                        occurrences=occurrences,
                        avg_timing=avg_timing,
                        support=float(support),
                    )
                )

        # Sort by support (descending), then by occurrences
        result.sort(key=lambda s: (s.support, s.occurrences), reverse=True)

        return result

    @staticmethod
    def find_temporal_correlations(
        session: CANSession,
        max_delay_ms: float = 100,
    ) -> dict[tuple[int, int], TemporalCorrelation]:
        """Find temporal correlations between messages.

        Analyzes timing relationships to determine which messages consistently
        follow others with predictable delays. Unlike message pairs, this focuses
        on statistical correlation rather than simple co-occurrence.

        Args:
            session: CAN session to analyze.
            max_delay_ms: Maximum delay to consider for correlations.

        Returns:
            Dictionary mapping (leader_id, follower_id) to TemporalCorrelation.
        """
        max_delay_s = max_delay_ms / 1000.0

        # Get all messages sorted by timestamp
        all_messages = sorted(session._messages.messages, key=lambda m: m.timestamp)

        # Track correlations: (leader_id, follower_id) -> list of delays
        correlation_delays: dict[tuple[int, int], list[float]] = defaultdict(list)

        # For each message, find the next occurrence of each other message ID
        # within the time window
        for i, leader_msg in enumerate(all_messages):
            leader_id = leader_msg.arbitration_id

            # Track which follower IDs we've seen (only count first occurrence)
            seen_followers = set()

            # Look forward for followers
            for follower_msg in all_messages[i + 1 :]:
                delay = follower_msg.timestamp - leader_msg.timestamp
                if delay > max_delay_s:
                    break  # Outside window

                follower_id = follower_msg.arbitration_id

                # Skip same ID
                if follower_id == leader_id:
                    continue

                # Only record first occurrence of each follower ID
                if follower_id not in seen_followers:
                    correlation_delays[(leader_id, follower_id)].append(delay * 1000)
                    seen_followers.add(follower_id)

        # Build TemporalCorrelation objects
        correlations = {}
        for (leader_id, follower_id), delays in correlation_delays.items():
            if len(delays) >= 2:  # Need at least 2 samples for std
                avg_delay = np.mean(delays)
                std_delay = np.std(delays)

                correlations[(leader_id, follower_id)] = TemporalCorrelation(
                    leader_id=leader_id,
                    follower_id=follower_id,
                    avg_delay_ms=float(avg_delay),
                    std_delay_ms=float(std_delay),
                    occurrences=len(delays),
                )

        return correlations
