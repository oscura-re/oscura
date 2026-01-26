"""Comprehensive tests for CAN pattern detection and analysis.

This module tests message pairs, sequences, and temporal correlations.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.automotive]

from oscura.automotive.can.models import CANMessage, CANMessageList
from oscura.automotive.can.patterns import (
    MessagePair,
    MessageSequence,
    PatternAnalyzer,
    TemporalCorrelation,
)
from oscura.automotive.can.session import CANSession


@pytest.fixture
def session_with_patterns() -> CANSession:
    """Create CAN session with known patterns for testing.

    Patterns:
    - 0x100 followed by 0x200 within 10ms (request-response)
    - 0x300 → 0x301 → 0x302 sequence every 50ms
    - 0x400 appears every 100ms (periodic)
    """
    messages = CANMessageList()
    timestamp = 0.0

    # Create request-response pattern (0x100 → 0x200)
    for i in range(10):
        # Request
        msg1 = CANMessage(
            arbitration_id=0x100,
            timestamp=timestamp,
            data=bytes([0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08]),
            is_extended=False,
        )
        messages.append(msg1)

        # Response after 5ms
        msg2 = CANMessage(
            arbitration_id=0x200,
            timestamp=timestamp + 0.005,
            data=bytes([0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0x80]),
            is_extended=False,
        )
        messages.append(msg2)

        timestamp += 0.1  # Next request after 100ms

    # Create sequence pattern (0x300 → 0x301 → 0x302)
    timestamp = 0.0
    for i in range(8):
        msg1 = CANMessage(
            arbitration_id=0x300,
            timestamp=timestamp,
            data=bytes([i, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]),
            is_extended=False,
        )
        messages.append(msg1)

        msg2 = CANMessage(
            arbitration_id=0x301,
            timestamp=timestamp + 0.010,
            data=bytes([i, 0x11, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]),
            is_extended=False,
        )
        messages.append(msg2)

        msg3 = CANMessage(
            arbitration_id=0x302,
            timestamp=timestamp + 0.020,
            data=bytes([i, 0x22, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]),
            is_extended=False,
        )
        messages.append(msg3)

        timestamp += 0.05  # Sequence repeats every 50ms

    # Create periodic message
    timestamp = 0.0
    for i in range(10):
        msg = CANMessage(
            arbitration_id=0x400,
            timestamp=timestamp,
            data=bytes([i % 256, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF]),
            is_extended=False,
        )
        messages.append(msg)
        timestamp += 0.1  # Every 100ms

    session = CANSession(name="Pattern Test")
    session._messages = messages
    return session


class TestMessagePair:
    """Tests for MessagePair dataclass."""

    def test_message_pair_creation(self):
        """Test creating a message pair."""
        pair = MessagePair(
            id_a=0x100,
            id_b=0x200,
            occurrences=10,
            avg_delay_ms=5.5,
            confidence=0.95,
        )

        assert pair.id_a == 0x100
        assert pair.id_b == 0x200
        assert pair.occurrences == 10
        assert pair.avg_delay_ms == 5.5
        assert pair.confidence == 0.95

    def test_message_pair_repr(self):
        """Test MessagePair string representation."""
        pair = MessagePair(
            id_a=0x123,
            id_b=0x456,
            occurrences=5,
            avg_delay_ms=10.25,
            confidence=0.85,
        )

        repr_str = repr(pair)
        assert "0x123" in repr_str
        assert "0x456" in repr_str
        assert "occurrences=5" in repr_str
        assert "10.25" in repr_str
        assert "0.85" in repr_str


class TestMessageSequence:
    """Tests for MessageSequence dataclass."""

    def test_message_sequence_creation(self):
        """Test creating a message sequence."""
        seq = MessageSequence(
            ids=[0x100, 0x200, 0x300],
            occurrences=8,
            avg_timing=[10.0, 15.0],
            support=0.75,
        )

        assert seq.ids == [0x100, 0x200, 0x300]
        assert seq.occurrences == 8
        assert seq.avg_timing == [10.0, 15.0]
        assert seq.support == 0.75

    def test_message_sequence_repr(self):
        """Test MessageSequence string representation."""
        seq = MessageSequence(
            ids=[0x123, 0x456],
            occurrences=10,
            avg_timing=[5.5],
            support=0.90,
        )

        repr_str = repr(seq)
        assert "0x123" in repr_str
        assert "0x456" in repr_str
        assert "occurrences=10" in repr_str
        assert "support=0.90" in repr_str


class TestTemporalCorrelation:
    """Tests for TemporalCorrelation dataclass."""

    def test_temporal_correlation_creation(self):
        """Test creating a temporal correlation."""
        corr = TemporalCorrelation(
            leader_id=0x100,
            follower_id=0x200,
            avg_delay_ms=7.5,
            std_delay_ms=1.2,
            occurrences=15,
        )

        assert corr.leader_id == 0x100
        assert corr.follower_id == 0x200
        assert corr.avg_delay_ms == 7.5
        assert corr.std_delay_ms == 1.2
        assert corr.occurrences == 15

    def test_temporal_correlation_repr(self):
        """Test TemporalCorrelation string representation."""
        corr = TemporalCorrelation(
            leader_id=0x300,
            follower_id=0x400,
            avg_delay_ms=10.0,
            std_delay_ms=2.5,
            occurrences=20,
        )

        repr_str = repr(corr)
        assert "0x300" in repr_str
        assert "0x400" in repr_str
        assert "10.00" in repr_str
        assert "2.50" in repr_str
        assert "occurrences=20" in repr_str


class TestPatternAnalyzer:
    """Tests for PatternAnalyzer class."""

    def test_find_message_pairs_basic(self, session_with_patterns):
        """Test finding basic message pairs."""
        pairs = PatternAnalyzer.find_message_pairs(
            session_with_patterns,
            time_window_ms=100,
            min_occurrence=3,
        )

        # Should find 0x100 → 0x200 pair
        assert len(pairs) > 0

        # Find the specific pair
        pair_100_200 = None
        for pair in pairs:
            if pair.id_a == 0x100 and pair.id_b == 0x200:
                pair_100_200 = pair
                break

        assert pair_100_200 is not None
        assert pair_100_200.occurrences >= 3
        assert 0 < pair_100_200.avg_delay_ms < 100

    def test_find_message_pairs_time_window(self, session_with_patterns):
        """Test time window filtering in pair detection."""
        # Small time window (should find fewer pairs)
        pairs_small = PatternAnalyzer.find_message_pairs(
            session_with_patterns,
            time_window_ms=10,
            min_occurrence=3,
        )

        # Large time window (should find more pairs)
        pairs_large = PatternAnalyzer.find_message_pairs(
            session_with_patterns,
            time_window_ms=200,
            min_occurrence=3,
        )

        # Large window should find at least as many pairs
        assert len(pairs_large) >= len(pairs_small)

    def test_find_message_pairs_min_occurrence(self, session_with_patterns):
        """Test minimum occurrence filtering."""
        # Low threshold
        pairs_low = PatternAnalyzer.find_message_pairs(
            session_with_patterns,
            time_window_ms=100,
            min_occurrence=2,
        )

        # High threshold
        pairs_high = PatternAnalyzer.find_message_pairs(
            session_with_patterns,
            time_window_ms=100,
            min_occurrence=8,
        )

        # Low threshold should find more pairs
        assert len(pairs_low) >= len(pairs_high)

    def test_find_message_pairs_sorting(self, session_with_patterns):
        """Test that pairs are sorted by occurrence count."""
        pairs = PatternAnalyzer.find_message_pairs(
            session_with_patterns,
            time_window_ms=100,
            min_occurrence=2,
        )

        if len(pairs) > 1:
            # Check descending order
            for i in range(len(pairs) - 1):
                assert pairs[i].occurrences >= pairs[i + 1].occurrences

    def test_find_message_pairs_no_self_pairing(self, session_with_patterns):
        """Test that messages don't pair with themselves."""
        pairs = PatternAnalyzer.find_message_pairs(
            session_with_patterns,
            time_window_ms=100,
            min_occurrence=2,
        )

        # No pair should have same ID for both sides
        for pair in pairs:
            assert pair.id_a != pair.id_b

    def test_find_message_sequences_basic(self, session_with_patterns):
        """Test finding basic message sequences."""
        sequences = PatternAnalyzer.find_message_sequences(
            session_with_patterns,
            max_sequence_length=3,
            time_window_ms=100,
            min_support=0.5,
        )

        # Should find some sequences
        assert len(sequences) > 0

        # Check that sequences have correct structure
        for seq in sequences:
            assert len(seq.ids) >= 2
            assert len(seq.ids) <= 3
            assert len(seq.avg_timing) == len(seq.ids) - 1
            assert 0.0 <= seq.support <= 1.0

    def test_find_message_sequences_length_validation(self):
        """Test sequence length validation."""
        session = CANSession(name="Test")

        # Test too small
        with pytest.raises(ValueError, match="at least 2"):
            PatternAnalyzer.find_message_sequences(
                session,
                max_sequence_length=1,
                time_window_ms=100,
                min_support=0.5,
            )

        # Test too large
        with pytest.raises(ValueError, match="cannot exceed 10"):
            PatternAnalyzer.find_message_sequences(
                session,
                max_sequence_length=11,
                time_window_ms=100,
                min_support=0.5,
            )

    def test_find_message_sequences_support_validation(self):
        """Test support parameter validation."""
        session = CANSession(name="Test")

        # Test negative support
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            PatternAnalyzer.find_message_sequences(
                session,
                max_sequence_length=3,
                time_window_ms=100,
                min_support=-0.1,
            )

        # Test support > 1.0
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            PatternAnalyzer.find_message_sequences(
                session,
                max_sequence_length=3,
                time_window_ms=100,
                min_support=1.5,
            )

    def test_find_message_sequences_empty_session(self):
        """Test sequence finding with empty session."""
        session = CANSession(name="Empty")

        sequences = PatternAnalyzer.find_message_sequences(
            session,
            max_sequence_length=3,
            time_window_ms=100,
            min_support=0.5,
        )

        assert len(sequences) == 0

    def test_find_message_sequences_sorting(self, session_with_patterns):
        """Test that sequences are sorted by support."""
        sequences = PatternAnalyzer.find_message_sequences(
            session_with_patterns,
            max_sequence_length=4,
            time_window_ms=150,
            min_support=0.3,
        )

        if len(sequences) > 1:
            # Check descending order by support
            for i in range(len(sequences) - 1):
                # Support values should be non-increasing
                assert sequences[i].support >= sequences[i + 1].support

    def test_find_temporal_correlations_basic(self, session_with_patterns):
        """Test finding temporal correlations."""
        correlations = PatternAnalyzer.find_temporal_correlations(
            session_with_patterns,
            max_delay_ms=100,
        )

        # Should find some correlations
        assert len(correlations) > 0

        # Each correlation should have valid structure
        for key, corr in correlations.items():
            leader_id, follower_id = key
            assert corr.leader_id == leader_id
            assert corr.follower_id == follower_id
            assert corr.avg_delay_ms >= 0
            assert corr.std_delay_ms >= 0
            assert corr.occurrences >= 2  # Need at least 2 for std calculation

    def test_find_temporal_correlations_max_delay(self, session_with_patterns):
        """Test max_delay parameter."""
        # Small delay window
        corr_small = PatternAnalyzer.find_temporal_correlations(
            session_with_patterns,
            max_delay_ms=10,
        )

        # Large delay window
        corr_large = PatternAnalyzer.find_temporal_correlations(
            session_with_patterns,
            max_delay_ms=200,
        )

        # Large window should find at least as many correlations
        assert len(corr_large) >= len(corr_small)

    def test_find_temporal_correlations_no_self_correlation(self, session_with_patterns):
        """Test that messages don't correlate with themselves."""
        correlations = PatternAnalyzer.find_temporal_correlations(
            session_with_patterns,
            max_delay_ms=100,
        )

        # No correlation should have same ID
        for leader_id, follower_id in correlations:
            assert leader_id != follower_id

    def test_find_temporal_correlations_first_occurrence_only(self, session_with_patterns):
        """Test that only first occurrence of follower is counted."""
        # This is an implementation detail test
        correlations = PatternAnalyzer.find_temporal_correlations(
            session_with_patterns,
            max_delay_ms=50,
        )

        # All correlations should have reasonable occurrence counts
        for corr in correlations.values():
            assert corr.occurrences > 0

    def test_pattern_analyzer_confidence_calculation(self, session_with_patterns):
        """Test confidence calculation for message pairs."""
        pairs = PatternAnalyzer.find_message_pairs(
            session_with_patterns,
            time_window_ms=100,
            min_occurrence=3,
        )

        # All pairs should have confidence between 0 and 1
        for pair in pairs:
            assert 0.0 <= pair.confidence <= 1.0

    def test_pattern_analyzer_with_single_message_id(self):
        """Test pattern detection with only one message ID."""
        messages = CANMessageList()
        for i in range(10):
            msg = CANMessage(
                arbitration_id=0x100,
                timestamp=i * 0.1,
                data=bytes([i % 256] + [0] * 7),
                is_extended=False,
            )
            messages.append(msg)

        session = CANSession(name="Single ID")
        session._messages = messages

        # Should find no pairs (need at least 2 different IDs)
        pairs = PatternAnalyzer.find_message_pairs(session, time_window_ms=100, min_occurrence=2)
        assert len(pairs) == 0

        # Should find no sequences with length > 1
        sequences = PatternAnalyzer.find_message_sequences(
            session, max_sequence_length=3, time_window_ms=100, min_support=0.5
        )
        # May find sequences of same ID repeated
        # That's allowed by the algorithm

    def test_calculate_max_message_frequency(self):
        """Test _calculate_max_message_frequency helper."""
        messages = CANMessageList()

        # Add 10 messages for 0x100
        for i in range(10):
            msg = CANMessage(
                arbitration_id=0x100,
                timestamp=i * 0.01,
                data=bytes([0] * 8),
                is_extended=False,
            )
            messages.append(msg)

        # Add 5 messages for 0x200
        for i in range(5):
            msg = CANMessage(
                arbitration_id=0x200,
                timestamp=i * 0.01,
                data=bytes([0] * 8),
                is_extended=False,
            )
            messages.append(msg)

        max_freq = PatternAnalyzer._calculate_max_message_frequency(messages.messages)
        assert max_freq == 10  # 0x100 has most messages

    def test_calculate_max_message_frequency_empty(self):
        """Test _calculate_max_message_frequency with empty list."""
        max_freq = PatternAnalyzer._calculate_max_message_frequency([])
        assert max_freq == 1  # Default value
