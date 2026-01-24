"""Tests for multi-message pattern learning.

This module tests the PatternAnalyzer class for discovering relationships
between CAN messages, including pairs, sequences, and temporal correlations.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.automotive]

import pytest

from oscura.automotive.can.models import CANMessage, CANMessageList
from oscura.automotive.can.patterns import (
    MessagePair,
    MessageSequence,
    PatternAnalyzer,
    TemporalCorrelation,
)
from oscura.automotive.can.session import CANSession


class TestMessagePairs:
    """Test message pair detection."""

    def test_find_simple_pair(self):
        """Test detection of a simple message pair."""
        messages = CANMessageList()

        # Create 10 occurrences of 0x100 followed by 0x200 within 50ms
        for i in range(10):
            base_time = i * 1.0  # 1 second apart
            messages.append(CANMessage(arbitration_id=0x100, timestamp=base_time, data=b"\x00" * 8))
            messages.append(
                CANMessage(
                    arbitration_id=0x200, timestamp=base_time + 0.03, data=b"\x00" * 8
                )  # 30ms delay
            )

        session = CANSession(name="Pattern Test")
        session._messages = messages
        pairs = PatternAnalyzer.find_message_pairs(session, time_window_ms=100, min_occurrence=3)

        # Should find the 0x100 -> 0x200 pair
        assert len(pairs) > 0
        pair = pairs[0]
        assert pair.id_a == 0x100
        assert pair.id_b == 0x200
        assert pair.occurrences == 10
        assert 25 < pair.avg_delay_ms < 35  # Around 30ms
        assert pair.confidence > 0.9  # High confidence due to consistency

    def test_no_pairs_found(self):
        """Test when no pairs exist."""
        messages = CANMessageList()

        # Create messages that never occur together
        for i in range(10):
            messages.append(CANMessage(arbitration_id=0x100, timestamp=i * 1.0, data=b"\x00" * 8))

        session = CANSession(name="Pattern Test")
        session._messages = messages
        pairs = PatternAnalyzer.find_message_pairs(session, time_window_ms=10, min_occurrence=3)

        assert len(pairs) == 0

    def test_multiple_pairs(self):
        """Test detection of multiple pairs."""
        messages = CANMessageList()

        # Pair 1: 0x100 -> 0x200 (20ms delay)
        # Pair 2: 0x100 -> 0x300 (40ms delay)
        for i in range(10):
            base_time = i * 1.0
            messages.append(CANMessage(arbitration_id=0x100, timestamp=base_time, data=b"\x00" * 8))
            messages.append(
                CANMessage(arbitration_id=0x200, timestamp=base_time + 0.02, data=b"\x00" * 8)
            )
            messages.append(
                CANMessage(arbitration_id=0x300, timestamp=base_time + 0.04, data=b"\x00" * 8)
            )

        session = CANSession(name="Pattern Test")
        session._messages = messages
        pairs = PatternAnalyzer.find_message_pairs(session, time_window_ms=100, min_occurrence=3)

        # Should find at least 2 pairs (100->200, 100->300)
        assert len(pairs) >= 2

        # Check that 0x100 -> 0x200 is found
        pair_100_200 = next((p for p in pairs if p.id_a == 0x100 and p.id_b == 0x200), None)
        assert pair_100_200 is not None
        assert pair_100_200.occurrences == 10
        assert 15 < pair_100_200.avg_delay_ms < 25

    def test_min_occurrence_threshold(self):
        """Test that min_occurrence threshold is respected."""
        messages = CANMessageList()

        # Create only 2 occurrences
        for i in range(2):
            base_time = i * 1.0
            messages.append(CANMessage(arbitration_id=0x100, timestamp=base_time, data=b"\x00" * 8))
            messages.append(
                CANMessage(arbitration_id=0x200, timestamp=base_time + 0.03, data=b"\x00" * 8)
            )

        session = CANSession(name="Pattern Test")
        session._messages = messages

        # With min_occurrence=3, should find nothing
        pairs = PatternAnalyzer.find_message_pairs(session, time_window_ms=100, min_occurrence=3)
        assert len(pairs) == 0

        # With min_occurrence=2, should find the pair
        pairs = PatternAnalyzer.find_message_pairs(session, time_window_ms=100, min_occurrence=2)
        assert len(pairs) > 0

    def test_time_window_filtering(self):
        """Test that time window is properly applied."""
        messages = CANMessageList()

        # Create pairs with 150ms delay
        for i in range(10):
            base_time = i * 1.0
            messages.append(CANMessage(arbitration_id=0x100, timestamp=base_time, data=b"\x00" * 8))
            messages.append(
                CANMessage(arbitration_id=0x200, timestamp=base_time + 0.15, data=b"\x00" * 8)
            )

        session = CANSession(name="Pattern Test")
        session._messages = messages

        # With 100ms window, should not find the pair
        pairs = PatternAnalyzer.find_message_pairs(session, time_window_ms=100, min_occurrence=3)
        assert len(pairs) == 0

        # With 200ms window, should find the pair
        pairs = PatternAnalyzer.find_message_pairs(session, time_window_ms=200, min_occurrence=3)
        assert len(pairs) > 0

    def test_confidence_calculation(self):
        """Test confidence score calculation."""
        messages = CANMessageList()

        # Create consistent pairs (low variance)
        for i in range(10):
            base_time = i * 1.0
            messages.append(CANMessage(arbitration_id=0x100, timestamp=base_time, data=b"\x00" * 8))
            messages.append(
                CANMessage(arbitration_id=0x200, timestamp=base_time + 0.03, data=b"\x00" * 8)
            )

        session = CANSession(name="Pattern Test")
        session._messages = messages
        pairs = PatternAnalyzer.find_message_pairs(session, time_window_ms=100, min_occurrence=3)

        assert len(pairs) > 0
        # Consistent timing should have high confidence
        assert pairs[0].confidence > 0.8

    def test_same_id_not_paired(self):
        """Test that messages with same ID are not paired with themselves."""
        messages = CANMessageList()

        # Create multiple messages with same ID close together
        for i in range(10):
            base_time = i * 1.0
            messages.append(CANMessage(arbitration_id=0x100, timestamp=base_time, data=b"\x00" * 8))
            messages.append(
                CANMessage(arbitration_id=0x100, timestamp=base_time + 0.01, data=b"\x01" * 8)
            )

        session = CANSession(name="Pattern Test")
        session._messages = messages
        pairs = PatternAnalyzer.find_message_pairs(session, time_window_ms=100, min_occurrence=3)

        # Should not find 0x100 -> 0x100 pairs
        assert len(pairs) == 0


class TestMessageSequences:
    """Test message sequence detection."""

    def test_find_2_message_sequence(self):
        """Test detection of 2-message sequences."""
        messages = CANMessageList()

        # Create sequence: 0x100 -> 0x200
        for i in range(10):
            base_time = i * 1.0
            messages.append(CANMessage(arbitration_id=0x100, timestamp=base_time, data=b"\x00" * 8))
            messages.append(
                CANMessage(arbitration_id=0x200, timestamp=base_time + 0.05, data=b"\x00" * 8)
            )

        session = CANSession(name="Pattern Test")
        session._messages = messages
        sequences = PatternAnalyzer.find_message_sequences(
            session, max_sequence_length=2, time_window_ms=200, min_support=0.5
        )

        # Should find the sequence
        assert len(sequences) > 0
        seq = sequences[0]
        assert len(seq.ids) == 2
        assert seq.ids[0] == 0x100
        assert seq.ids[1] == 0x200
        assert seq.occurrences == 10

    def test_find_3_message_sequence(self):
        """Test detection of 3-message sequences."""
        messages = CANMessageList()

        # Create sequence: 0x100 -> 0x200 -> 0x300
        for i in range(10):
            base_time = i * 1.0
            messages.append(CANMessage(arbitration_id=0x100, timestamp=base_time, data=b"\x00" * 8))
            messages.append(
                CANMessage(arbitration_id=0x200, timestamp=base_time + 0.05, data=b"\x00" * 8)
            )
            messages.append(
                CANMessage(arbitration_id=0x300, timestamp=base_time + 0.10, data=b"\x00" * 8)
            )

        session = CANSession(name="Pattern Test")
        session._messages = messages
        sequences = PatternAnalyzer.find_message_sequences(
            session, max_sequence_length=3, time_window_ms=300, min_support=0.5
        )

        # Find the 3-message sequence
        seq_3 = [s for s in sequences if len(s.ids) == 3]
        assert len(seq_3) > 0

        seq = seq_3[0]
        assert seq.ids == [0x100, 0x200, 0x300]
        assert seq.occurrences == 10
        assert len(seq.avg_timing) == 2  # 2 gaps for 3 messages

    def test_find_4_message_sequence(self):
        """Test detection of 4-message sequences."""
        messages = CANMessageList()

        # Create sequence: 0x100 -> 0x200 -> 0x300 -> 0x400
        for i in range(10):
            base_time = i * 1.0
            messages.append(CANMessage(arbitration_id=0x100, timestamp=base_time, data=b"\x00" * 8))
            messages.append(
                CANMessage(arbitration_id=0x200, timestamp=base_time + 0.02, data=b"\x00" * 8)
            )
            messages.append(
                CANMessage(arbitration_id=0x300, timestamp=base_time + 0.04, data=b"\x00" * 8)
            )
            messages.append(
                CANMessage(arbitration_id=0x400, timestamp=base_time + 0.06, data=b"\x00" * 8)
            )

        session = CANSession(name="Pattern Test")
        session._messages = messages
        sequences = PatternAnalyzer.find_message_sequences(
            session, max_sequence_length=4, time_window_ms=200, min_support=0.5
        )

        # Find the 4-message sequence
        seq_4 = [s for s in sequences if len(s.ids) == 4]
        assert len(seq_4) > 0

        seq = seq_4[0]
        assert seq.ids == [0x100, 0x200, 0x300, 0x400]
        assert seq.occurrences == 10

    def test_min_support_threshold(self):
        """Test that min_support threshold is respected."""
        messages = CANMessageList()

        # Create frequent message 0x100 (20 times)
        for i in range(20):
            messages.append(CANMessage(arbitration_id=0x100, timestamp=i * 0.1, data=b"\x00" * 8))

        # Create rare sequence 0x200 -> 0x300 (only 3 times)
        for i in range(3):
            base_time = i * 2.0
            messages.append(CANMessage(arbitration_id=0x200, timestamp=base_time, data=b"\x00" * 8))
            messages.append(
                CANMessage(arbitration_id=0x300, timestamp=base_time + 0.05, data=b"\x00" * 8)
            )

        session = CANSession(name="Pattern Test")
        session._messages = messages

        # High support threshold should exclude rare sequence
        sequences = PatternAnalyzer.find_message_sequences(
            session, max_sequence_length=2, time_window_ms=200, min_support=0.3
        )
        # Support = 3/20 = 0.15, which is < 0.3
        seq_200_300 = [s for s in sequences if s.ids == [0x200, 0x300]]
        assert len(seq_200_300) == 0

        # Low support threshold should include rare sequence
        sequences = PatternAnalyzer.find_message_sequences(
            session, max_sequence_length=2, time_window_ms=200, min_support=0.1
        )
        seq_200_300 = [s for s in sequences if s.ids == [0x200, 0x300]]
        assert len(seq_200_300) > 0

    def test_time_window_constraint(self):
        """Test that time window constraint is applied."""
        messages = CANMessageList()

        # Create sequences with 600ms total duration
        # Use different periods to avoid cross-cycle sequences
        for i in range(10):
            base_time = i * 2.0  # 2 seconds apart to avoid overlap
            messages.append(CANMessage(arbitration_id=0x100, timestamp=base_time, data=b"\x00" * 8))
            messages.append(
                CANMessage(arbitration_id=0x200, timestamp=base_time + 0.6, data=b"\x00" * 8)
            )

        session = CANSession(name="Pattern Test")
        session._messages = messages

        # 500ms window should not capture the 0x100 -> 0x200 sequence
        sequences = PatternAnalyzer.find_message_sequences(
            session, max_sequence_length=2, time_window_ms=500, min_support=0.5
        )
        # Filter to only sequences starting with 0x100
        seq_100_200 = [s for s in sequences if s.ids == [0x100, 0x200]]
        assert len(seq_100_200) == 0

        # 700ms window should capture the sequence
        sequences = PatternAnalyzer.find_message_sequences(
            session, max_sequence_length=2, time_window_ms=700, min_support=0.5
        )
        seq_100_200 = [s for s in sequences if s.ids == [0x100, 0x200]]
        assert len(seq_100_200) > 0

    def test_max_sequence_length_validation(self):
        """Test validation of max_sequence_length parameter."""
        session = CANSession(name="Pattern Test")
        session._messages = CANMessageList()

        # Too small
        with pytest.raises(ValueError, match="at least 2"):
            PatternAnalyzer.find_message_sequences(
                session, max_sequence_length=1, time_window_ms=500
            )

        # Too large
        with pytest.raises(ValueError, match="cannot exceed 10"):
            PatternAnalyzer.find_message_sequences(
                session, max_sequence_length=11, time_window_ms=500
            )

    def test_min_support_validation(self):
        """Test validation of min_support parameter."""
        session = CANSession(name="Pattern Test")
        session._messages = CANMessageList()

        # Out of range
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            PatternAnalyzer.find_message_sequences(session, min_support=1.5)

        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            PatternAnalyzer.find_message_sequences(session, min_support=-0.1)

    def test_empty_session(self):
        """Test with empty session."""
        session = CANSession(name="Pattern Test")
        session._messages = CANMessageList()
        sequences = PatternAnalyzer.find_message_sequences(session)
        assert len(sequences) == 0

    def test_avg_timing_calculation(self):
        """Test average timing calculation in sequences."""
        messages = CANMessageList()

        # Create consistent 3-message sequence
        for i in range(10):
            base_time = i * 1.0
            messages.append(CANMessage(arbitration_id=0x100, timestamp=base_time, data=b"\x00" * 8))
            messages.append(
                CANMessage(arbitration_id=0x200, timestamp=base_time + 0.03, data=b"\x00" * 8)
            )
            messages.append(
                CANMessage(arbitration_id=0x300, timestamp=base_time + 0.08, data=b"\x00" * 8)
            )

        session = CANSession(name="Pattern Test")
        session._messages = messages
        sequences = PatternAnalyzer.find_message_sequences(
            session, max_sequence_length=3, time_window_ms=200, min_support=0.5
        )

        seq_3 = [s for s in sequences if len(s.ids) == 3]
        assert len(seq_3) > 0

        seq = seq_3[0]
        # Should have timing for 2 gaps
        assert len(seq.avg_timing) == 2
        # First gap: ~30ms, second gap: ~50ms (80 - 30)
        assert 25 < seq.avg_timing[0] < 35
        assert 45 < seq.avg_timing[1] < 55


class TestTemporalCorrelations:
    """Test temporal correlation detection."""

    def test_find_simple_correlation(self):
        """Test detection of simple temporal correlation."""
        messages = CANMessageList()

        # 0x100 always followed by 0x200 after ~30ms
        for i in range(20):
            base_time = i * 0.5
            messages.append(CANMessage(arbitration_id=0x100, timestamp=base_time, data=b"\x00" * 8))
            messages.append(
                CANMessage(arbitration_id=0x200, timestamp=base_time + 0.03, data=b"\x00" * 8)
            )

        session = CANSession(name="Pattern Test")
        session._messages = messages
        correlations = PatternAnalyzer.find_temporal_correlations(session, max_delay_ms=100)

        # Should find correlation
        assert (0x100, 0x200) in correlations

        corr = correlations[(0x100, 0x200)]
        assert corr.leader_id == 0x100
        assert corr.follower_id == 0x200
        assert corr.occurrences == 20
        assert 25 < corr.avg_delay_ms < 35
        assert corr.std_delay_ms < 5  # Low variance

    def test_multiple_correlations(self):
        """Test detection of multiple correlations."""
        messages = CANMessageList()

        # 0x100 -> 0x200 (20ms)
        # 0x100 -> 0x300 (40ms)
        # 0x200 -> 0x400 (30ms)
        for i in range(15):
            base_time = i * 0.5
            messages.append(CANMessage(arbitration_id=0x100, timestamp=base_time, data=b"\x00" * 8))
            messages.append(
                CANMessage(arbitration_id=0x200, timestamp=base_time + 0.02, data=b"\x00" * 8)
            )
            messages.append(
                CANMessage(arbitration_id=0x300, timestamp=base_time + 0.04, data=b"\x00" * 8)
            )
            messages.append(
                CANMessage(arbitration_id=0x400, timestamp=base_time + 0.05, data=b"\x00" * 8)
            )

        session = CANSession(name="Pattern Test")
        session._messages = messages
        correlations = PatternAnalyzer.find_temporal_correlations(session, max_delay_ms=100)

        # Should find multiple correlations
        assert len(correlations) >= 3
        assert (0x100, 0x200) in correlations
        assert (0x100, 0x300) in correlations
        assert (0x200, 0x400) in correlations

    def test_max_delay_filtering(self):
        """Test that max_delay filter is applied."""
        messages = CANMessageList()

        # Create correlation with 150ms delay
        for i in range(20):
            base_time = i * 0.5
            messages.append(CANMessage(arbitration_id=0x100, timestamp=base_time, data=b"\x00" * 8))
            messages.append(
                CANMessage(arbitration_id=0x200, timestamp=base_time + 0.15, data=b"\x00" * 8)
            )

        session = CANSession(name="Pattern Test")
        session._messages = messages

        # With 100ms limit, should not find correlation
        correlations = PatternAnalyzer.find_temporal_correlations(session, max_delay_ms=100)
        assert (0x100, 0x200) not in correlations

        # With 200ms limit, should find correlation
        correlations = PatternAnalyzer.find_temporal_correlations(session, max_delay_ms=200)
        assert (0x100, 0x200) in correlations

    def test_first_occurrence_only(self):
        """Test that only first occurrence of each follower is counted."""
        messages = CANMessageList()

        # 0x100 followed by two 0x200 messages
        for i in range(10):
            base_time = i * 1.0
            messages.append(CANMessage(arbitration_id=0x100, timestamp=base_time, data=b"\x00" * 8))
            messages.append(
                CANMessage(arbitration_id=0x200, timestamp=base_time + 0.02, data=b"\x00" * 8)
            )
            messages.append(
                CANMessage(arbitration_id=0x200, timestamp=base_time + 0.04, data=b"\x00" * 8)
            )

        session = CANSession(name="Pattern Test")
        session._messages = messages
        correlations = PatternAnalyzer.find_temporal_correlations(session, max_delay_ms=100)

        # Should count only first occurrence
        corr = correlations[(0x100, 0x200)]
        assert corr.occurrences == 10
        # Delay should be ~20ms (first occurrence)
        assert 15 < corr.avg_delay_ms < 25

    def test_no_correlations_found(self):
        """Test when no correlations exist."""
        messages = CANMessageList()

        # Messages that are far apart
        for i in range(10):
            messages.append(CANMessage(arbitration_id=0x100, timestamp=i * 1.0, data=b"\x00" * 8))

        session = CANSession(name="Pattern Test")
        session._messages = messages
        correlations = PatternAnalyzer.find_temporal_correlations(session, max_delay_ms=10)

        assert len(correlations) == 0

    def test_same_id_not_correlated(self):
        """Test that same ID messages are not correlated."""
        messages = CANMessageList()

        # Multiple 0x100 messages close together
        for i in range(10):
            base_time = i * 0.5
            messages.append(CANMessage(arbitration_id=0x100, timestamp=base_time, data=b"\x00" * 8))
            messages.append(
                CANMessage(arbitration_id=0x100, timestamp=base_time + 0.01, data=b"\x01" * 8)
            )

        session = CANSession(name="Pattern Test")
        session._messages = messages
        correlations = PatternAnalyzer.find_temporal_correlations(session, max_delay_ms=100)

        # Should not find 0x100 -> 0x100
        assert (0x100, 0x100) not in correlations

    def test_std_deviation_calculation(self):
        """Test standard deviation calculation."""
        messages = CANMessageList()

        # Variable timing: alternating 20ms and 40ms delays
        for i in range(10):
            base_time = i * 0.5
            delay = 0.02 if i % 2 == 0 else 0.04
            messages.append(CANMessage(arbitration_id=0x100, timestamp=base_time, data=b"\x00" * 8))
            messages.append(
                CANMessage(arbitration_id=0x200, timestamp=base_time + delay, data=b"\x00" * 8)
            )

        session = CANSession(name="Pattern Test")
        session._messages = messages
        correlations = PatternAnalyzer.find_temporal_correlations(session, max_delay_ms=100)

        corr = correlations[(0x100, 0x200)]
        # Average should be ~30ms
        assert 25 < corr.avg_delay_ms < 35
        # Should have non-zero std deviation
        assert corr.std_delay_ms > 5

    def test_minimum_samples_requirement(self):
        """Test that at least 2 samples are needed."""
        messages = CANMessageList()

        # Only one occurrence
        messages.append(CANMessage(arbitration_id=0x100, timestamp=0.0, data=b"\x00" * 8))
        messages.append(CANMessage(arbitration_id=0x200, timestamp=0.03, data=b"\x00" * 8))

        session = CANSession(name="Pattern Test")
        session._messages = messages
        correlations = PatternAnalyzer.find_temporal_correlations(session, max_delay_ms=100)

        # Should not create correlation with only 1 sample
        assert (0x100, 0x200) not in correlations


class TestIntegrationWithCANSession:
    """Test integration of pattern analysis with CANSession."""

    def test_session_find_message_pairs(self):
        """Test CANSession.find_message_pairs() integration."""
        messages = CANMessageList()

        for i in range(10):
            base_time = i * 1.0
            messages.append(CANMessage(arbitration_id=0x100, timestamp=base_time, data=b"\x00" * 8))
            messages.append(
                CANMessage(arbitration_id=0x200, timestamp=base_time + 0.03, data=b"\x00" * 8)
            )

        session = CANSession(name="Pattern Test")
        session._messages = messages
        pairs = session.find_message_pairs(time_window_ms=100, min_occurrence=3)

        assert len(pairs) > 0
        assert isinstance(pairs[0], MessagePair)

    def test_session_find_message_sequences(self):
        """Test CANSession.find_message_sequences() integration."""
        messages = CANMessageList()

        for i in range(10):
            base_time = i * 1.0
            messages.append(CANMessage(arbitration_id=0x100, timestamp=base_time, data=b"\x00" * 8))
            messages.append(
                CANMessage(arbitration_id=0x200, timestamp=base_time + 0.05, data=b"\x00" * 8)
            )

        session = CANSession(name="Pattern Test")
        session._messages = messages
        sequences = session.find_message_sequences(
            max_sequence_length=2, time_window_ms=200, min_support=0.5
        )

        assert len(sequences) > 0
        assert isinstance(sequences[0], MessageSequence)

    def test_session_find_temporal_correlations(self):
        """Test CANSession.find_temporal_correlations() integration."""
        messages = CANMessageList()

        for i in range(20):
            base_time = i * 0.5
            messages.append(CANMessage(arbitration_id=0x100, timestamp=base_time, data=b"\x00" * 8))
            messages.append(
                CANMessage(arbitration_id=0x200, timestamp=base_time + 0.03, data=b"\x00" * 8)
            )

        session = CANSession(name="Pattern Test")
        session._messages = messages
        correlations = session.find_temporal_correlations(max_delay_ms=100)

        assert len(correlations) > 0
        assert (0x100, 0x200) in correlations
        assert isinstance(correlations[(0x100, 0x200)], TemporalCorrelation)


class TestDataModels:
    """Test data model representations."""

    def test_message_pair_repr(self):
        """Test MessagePair string representation."""
        pair = MessagePair(
            id_a=0x100, id_b=0x200, occurrences=10, avg_delay_ms=25.5, confidence=0.95
        )

        repr_str = repr(pair)
        assert "0x100" in repr_str
        assert "0x200" in repr_str
        assert "occurrences=10" in repr_str
        assert "25.5" in repr_str
        assert "0.95" in repr_str

    def test_message_sequence_repr(self):
        """Test MessageSequence string representation."""
        seq = MessageSequence(
            ids=[0x100, 0x200, 0x300],
            occurrences=5,
            avg_timing=[10.0, 20.0],
            support=0.8,
        )

        repr_str = repr(seq)
        assert "0x100" in repr_str
        assert "0x200" in repr_str
        assert "0x300" in repr_str
        assert "occurrences=5" in repr_str
        assert "support=0.80" in repr_str

    def test_temporal_correlation_repr(self):
        """Test TemporalCorrelation string representation."""
        corr = TemporalCorrelation(
            leader_id=0x100,
            follower_id=0x200,
            avg_delay_ms=30.0,
            std_delay_ms=5.0,
            occurrences=15,
        )

        repr_str = repr(corr)
        assert "0x100" in repr_str
        assert "0x200" in repr_str
        assert "30.00" in repr_str
        assert "5.00" in repr_str
        assert "occurrences=15" in repr_str
