"""Comprehensive unit tests for sequence pattern detection and correlation.

    - RE-SEQ-002: Sequence Pattern Detection
    - RE-SEQ-003: Request-Response Correlation

This module tests all public functions and classes in the inference.sequences module,
including pattern detection, request-response correlation, flow extraction, and
convenience functions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from oscura.inference.sequences import (
    CommunicationFlow,
    RequestResponseCorrelator,
    RequestResponsePair,
    SequencePattern,
    SequencePatternDetector,
    calculate_latency_stats,
    correlate_requests,
    detect_sequence_patterns,
    find_message_dependencies,
)

pytestmark = [pytest.mark.unit, pytest.mark.inference]


# =============================================================================
# Test Data Classes
# =============================================================================


@pytest.mark.unit
@pytest.mark.inference
class TestSequencePattern:
    """Test SequencePattern dataclass."""

    def test_create_sequence_pattern(self) -> None:
        """Test creating a SequencePattern instance."""
        pattern = SequencePattern(
            pattern=["A", "B", "C"],
            frequency=5,
            positions=[0, 5, 10, 15, 20],
            confidence=0.85,
            avg_gap=5.0,
            gap_variance=0.5,
        )

        assert pattern.pattern == ["A", "B", "C"]
        assert pattern.frequency == 5
        assert pattern.positions == [0, 5, 10, 15, 20]
        assert pattern.confidence == 0.85
        assert pattern.avg_gap == 5.0
        assert pattern.gap_variance == 0.5

    def test_sequence_pattern_default_values(self) -> None:
        """Test SequencePattern with default values."""
        pattern = SequencePattern(pattern=["X", "Y"], frequency=3)

        assert pattern.pattern == ["X", "Y"]
        assert pattern.frequency == 3
        assert pattern.positions == []
        assert pattern.confidence == 0.0
        assert pattern.avg_gap == 0.0
        assert pattern.gap_variance == 0.0

    def test_sequence_pattern_with_various_types(self) -> None:
        """Test SequencePattern can hold various data types."""
        # Integer patterns
        int_pattern = SequencePattern(pattern=[1, 2, 3], frequency=2)
        assert int_pattern.pattern == [1, 2, 3]

        # Byte patterns
        byte_pattern = SequencePattern(pattern=[b"REQ", b"RSP"], frequency=4)
        assert byte_pattern.pattern == [b"REQ", b"RSP"]

        # Dict-based patterns
        dict_pattern = SequencePattern(pattern=[{"type": "A"}, {"type": "B"}], frequency=2)
        assert len(dict_pattern.pattern) == 2


@pytest.mark.unit
@pytest.mark.inference
class TestRequestResponsePair:
    """Test RequestResponsePair dataclass."""

    def test_create_request_response_pair(self) -> None:
        """Test creating a RequestResponsePair instance."""
        pair = RequestResponsePair(
            request_index=0,
            response_index=1,
            request={"type": "GET", "id": 1},
            response={"type": "OK", "id": 1},
            latency=0.5,
            correlation_id=1,
            confidence=0.95,
        )

        assert pair.request_index == 0
        assert pair.response_index == 1
        assert pair.request["type"] == "GET"
        assert pair.response["type"] == "OK"
        assert pair.latency == 0.5
        assert pair.correlation_id == 1
        assert pair.confidence == 0.95

    def test_request_response_pair_default_values(self) -> None:
        """Test RequestResponsePair with default values."""
        pair = RequestResponsePair(
            request_index=5,
            response_index=6,
            request="req",
            response="rsp",
            latency=1.0,
        )

        assert pair.correlation_id is None
        assert pair.confidence == 0.0

    def test_request_response_pair_with_bytes_correlation(self) -> None:
        """Test RequestResponsePair with bytes correlation ID."""
        pair = RequestResponsePair(
            request_index=0,
            response_index=1,
            request=b"\x01\x02",
            response=b"\x03\x04",
            latency=0.1,
            correlation_id=b"\xab\xcd",
        )

        assert pair.correlation_id == b"\xab\xcd"


@pytest.mark.unit
@pytest.mark.inference
class TestCommunicationFlow:
    """Test CommunicationFlow dataclass."""

    def test_create_communication_flow(self) -> None:
        """Test creating a CommunicationFlow instance."""
        pair = RequestResponsePair(
            request_index=0,
            response_index=1,
            request="REQ",
            response="RSP",
            latency=0.5,
        )
        flow = CommunicationFlow(
            flow_id=1,
            messages=["REQ", "RSP", "REQ", "RSP"],
            pairs=[pair],
            direction="request_first",
            participants=["client", "server"],
            duration=2.0,
        )

        assert flow.flow_id == 1
        assert len(flow.messages) == 4
        assert len(flow.pairs) == 1
        assert flow.direction == "request_first"
        assert "client" in flow.participants
        assert flow.duration == 2.0

    def test_communication_flow_empty(self) -> None:
        """Test CommunicationFlow with empty pairs."""
        flow = CommunicationFlow(
            flow_id=0,
            messages=[],
            pairs=[],
            direction="request_first",
            participants=[],
            duration=0.0,
        )

        assert flow.flow_id == 0
        assert flow.messages == []
        assert flow.pairs == []


# =============================================================================
# Test SequencePatternDetector
# =============================================================================


@pytest.mark.unit
@pytest.mark.inference
class TestSequencePatternDetector:
    """Test SequencePatternDetector class."""

    def test_initialization_defaults(self) -> None:
        """Test detector initialization with default values."""
        detector = SequencePatternDetector()

        assert detector.min_pattern_length == 2
        assert detector.max_pattern_length == 10
        assert detector.min_frequency == 2
        assert detector.max_gap is None

    def test_initialization_custom(self) -> None:
        """Test detector initialization with custom values."""
        detector = SequencePatternDetector(
            min_pattern_length=3,
            max_pattern_length=20,
            min_frequency=5,
            max_gap=1.0,
        )

        assert detector.min_pattern_length == 3
        assert detector.max_pattern_length == 20
        assert detector.min_frequency == 5
        assert detector.max_gap == 1.0

    def test_detect_patterns_empty_messages(self) -> None:
        """Test detect_patterns with empty message list."""
        detector = SequencePatternDetector()
        patterns = detector.detect_patterns([])

        assert patterns == []

    def test_detect_patterns_basic(self) -> None:
        """Test basic pattern detection."""
        detector = SequencePatternDetector(min_pattern_length=2, min_frequency=2)
        messages = ["A", "B", "C", "A", "B", "C", "A", "B", "C"]
        patterns = detector.detect_patterns(messages)

        assert len(patterns) > 0
        # Should find "A", "B" pattern
        ab_found = any(p.pattern == ["A", "B"] for p in patterns)
        assert ab_found

    def test_detect_patterns_with_key(self) -> None:
        """Test pattern detection with key function."""
        detector = SequencePatternDetector(min_pattern_length=2, min_frequency=2)

        @dataclass
        class Message:
            msg_type: str
            value: int

        messages = [
            Message("REQ", 1),
            Message("RSP", 2),
            Message("REQ", 3),
            Message("RSP", 4),
            Message("REQ", 5),
            Message("RSP", 6),
        ]

        patterns = detector.detect_patterns(messages, key=lambda m: m.msg_type)

        assert len(patterns) > 0
        req_rsp_found = any(p.pattern == ["REQ", "RSP"] for p in patterns)
        assert req_rsp_found

    def test_detect_patterns_with_timestamps(self) -> None:
        """Test pattern detection with timestamps."""
        detector = SequencePatternDetector(min_pattern_length=2, min_frequency=2)
        messages = [
            {"type": "A", "ts": 0.0},
            {"type": "B", "ts": 1.0},
            {"type": "A", "ts": 2.0},
            {"type": "B", "ts": 3.0},
            {"type": "A", "ts": 4.0},
            {"type": "B", "ts": 5.0},
        ]

        patterns = detector.detect_patterns(
            messages,
            key=lambda m: m["type"],
            timestamp_key=lambda m: m["ts"],
        )

        assert len(patterns) > 0
        # Check avg_gap is calculated
        for pattern in patterns:
            if pattern.frequency > 1:
                assert pattern.avg_gap >= 0.0

    def test_detect_patterns_with_max_gap(self) -> None:
        """Test pattern detection respects max_gap constraint."""
        detector = SequencePatternDetector(
            min_pattern_length=2,
            min_frequency=2,
            max_gap=0.5,
        )
        messages = [
            {"type": "A", "ts": 0.0},
            {"type": "B", "ts": 0.3},  # Within gap
            {"type": "A", "ts": 1.0},
            {"type": "B", "ts": 2.0},  # Gap too large (1.0 > 0.5)
            {"type": "A", "ts": 3.0},
            {"type": "B", "ts": 3.3},  # Within gap
        ]

        patterns = detector.detect_patterns(
            messages,
            key=lambda m: m["type"],
            timestamp_key=lambda m: m["ts"],
        )

        # Should find patterns only where gaps are within limit
        assert isinstance(patterns, list)

    def test_detect_patterns_sorted_by_confidence(self) -> None:
        """Test that patterns are sorted by confidence (descending)."""
        detector = SequencePatternDetector(min_pattern_length=2, min_frequency=2)
        messages = ["A", "B"] * 10 + ["C", "D"] * 3

        patterns = detector.detect_patterns(messages)

        # Check sorted by confidence
        for i in range(len(patterns) - 1):
            assert patterns[i].confidence >= patterns[i + 1].confidence or (
                patterns[i].confidence == patterns[i + 1].confidence
                and patterns[i].frequency >= patterns[i + 1].frequency
            )

    def test_find_repeating_sequences_empty(self) -> None:
        """Test find_repeating_sequences with empty messages."""
        detector = SequencePatternDetector()
        result = detector.find_repeating_sequences([])

        assert result == []

    def test_find_repeating_sequences_basic(self) -> None:
        """Test basic repeating sequence detection."""
        detector = SequencePatternDetector(
            min_pattern_length=2, max_pattern_length=4, min_frequency=2
        )
        messages = ["X", "Y", "Z", "X", "Y", "Z"]

        result = detector.find_repeating_sequences(messages)

        assert len(result) > 0
        # Should find (sequence, count, positions)
        xyz_found = any(seq == ["X", "Y", "Z"] for seq, _, _ in result)
        assert xyz_found

    def test_find_repeating_sequences_with_key(self) -> None:
        """Test repeating sequence detection with key function."""
        detector = SequencePatternDetector(min_pattern_length=2, min_frequency=2)
        messages = [
            {"id": 1},
            {"id": 2},
            {"id": 1},
            {"id": 2},
        ]

        result = detector.find_repeating_sequences(messages, key=lambda m: m["id"])

        assert len(result) > 0
        id_seq_found = any(seq == [1, 2] for seq, _, _ in result)
        assert id_seq_found

    def test_find_repeating_sequences_positions_correct(self) -> None:
        """Test that positions are correctly returned."""
        detector = SequencePatternDetector(
            min_pattern_length=2, max_pattern_length=2, min_frequency=2
        )
        messages = ["A", "B", "C", "A", "B", "D"]

        result = detector.find_repeating_sequences(messages)

        ab_result = next((r for r in result if r[0] == ["A", "B"]), None)
        assert ab_result is not None
        seq, count, positions = ab_result
        assert count == 2
        assert positions == [0, 3]

    def test_find_repeating_sequences_sorted_by_frequency(self) -> None:
        """Test results are sorted by frequency."""
        detector = SequencePatternDetector(
            min_pattern_length=2, max_pattern_length=2, min_frequency=2
        )
        messages = ["A", "A"] * 5 + ["B", "B"] * 2

        result = detector.find_repeating_sequences(messages)

        for i in range(len(result) - 1):
            assert result[i][1] >= result[i + 1][1]

    def test_detect_periodic_patterns_empty(self) -> None:
        """Test periodic pattern detection with empty messages."""
        detector = SequencePatternDetector()
        result = detector.detect_periodic_patterns([])

        assert result == []

    def test_detect_periodic_patterns_regular_interval(self) -> None:
        """Test periodic pattern detection with regular intervals."""
        detector = SequencePatternDetector(min_pattern_length=2, min_frequency=3)
        messages = []
        # Create pattern with regular intervals
        for i in range(9):
            messages.append({"type": "A" if i % 2 == 0 else "B", "ts": float(i)})

        result = detector.detect_periodic_patterns(
            messages,
            key=lambda m: m["type"],
            timestamp_key=lambda m: m["ts"],
        )

        # Should find periodic patterns with low variance
        assert isinstance(result, list)

    def test_detect_periodic_patterns_irregular_interval(self) -> None:
        """Test periodic detection excludes irregular intervals."""
        detector = SequencePatternDetector(min_pattern_length=2, min_frequency=3)
        # Irregular intervals should have high variance
        messages = [
            {"type": "A", "ts": 0.0},
            {"type": "B", "ts": 1.0},
            {"type": "A", "ts": 10.0},  # Large gap
            {"type": "B", "ts": 11.0},
            {"type": "A", "ts": 12.0},
            {"type": "B", "ts": 13.0},
        ]

        result = detector.detect_periodic_patterns(
            messages,
            key=lambda m: m["type"],
            timestamp_key=lambda m: m["ts"],
        )

        # High variance patterns should be filtered
        for pattern in result:
            if pattern.avg_gap > 0 and pattern.gap_variance > 0:
                cv = (pattern.gap_variance**0.5) / pattern.avg_gap
                assert cv < 0.2


# =============================================================================
# Test RequestResponseCorrelator
# =============================================================================


@pytest.mark.unit
@pytest.mark.inference
class TestRequestResponseCorrelator:
    """Test RequestResponseCorrelator class."""

    def test_initialization_defaults(self) -> None:
        """Test correlator initialization with defaults."""
        correlator = RequestResponseCorrelator()

        assert correlator.max_latency == 10.0
        assert correlator.correlation_key is None

    def test_initialization_custom(self) -> None:
        """Test correlator initialization with custom values."""
        correlator = RequestResponseCorrelator(
            max_latency=5.0,
            correlation_key=lambda m: m.get("id"),
        )

        assert correlator.max_latency == 5.0
        assert correlator.correlation_key is not None

    def test_correlate_basic(self) -> None:
        """Test basic correlation without filters."""
        correlator = RequestResponseCorrelator(max_latency=10.0)
        messages = ["REQ1", "RSP1", "REQ2", "RSP2"]

        # Without filters, all messages are considered both requests and responses
        pairs = correlator.correlate(messages)

        assert len(pairs) > 0

    def test_correlate_with_filters(self) -> None:
        """Test correlation with request/response filters."""
        correlator = RequestResponseCorrelator(max_latency=10.0)
        messages = [
            {"type": "REQ", "id": 1},
            {"type": "RSP", "id": 1},
            {"type": "REQ", "id": 2},
            {"type": "RSP", "id": 2},
        ]

        pairs = correlator.correlate(
            messages,
            request_filter=lambda m: m["type"] == "REQ",
            response_filter=lambda m: m["type"] == "RSP",
        )

        assert len(pairs) == 2
        assert all(p.request["type"] == "REQ" for p in pairs)
        assert all(p.response["type"] == "RSP" for p in pairs)

    def test_correlate_with_timestamps(self) -> None:
        """Test correlation with timestamp function."""
        correlator = RequestResponseCorrelator(max_latency=5.0)
        messages = [
            {"type": "REQ", "ts": 0.0},
            {"type": "RSP", "ts": 1.0},
            {"type": "REQ", "ts": 10.0},
            {"type": "RSP", "ts": 11.0},
        ]

        pairs = correlator.correlate(
            messages,
            request_filter=lambda m: m["type"] == "REQ",
            response_filter=lambda m: m["type"] == "RSP",
            timestamp_key=lambda m: m["ts"],
        )

        assert len(pairs) == 2
        # Check latencies
        for pair in pairs:
            assert pair.latency == pytest.approx(1.0)

    def test_correlate_with_correlation_key(self) -> None:
        """Test correlation using correlation ID."""
        correlator = RequestResponseCorrelator(
            max_latency=10.0,
            correlation_key=lambda m: m["id"],
        )
        messages = [
            {"type": "REQ", "id": 100},
            {"type": "RSP", "id": 200},  # Different ID
            {"type": "RSP", "id": 100},  # Matching ID
        ]

        pairs = correlator.correlate(
            messages,
            request_filter=lambda m: m["type"] == "REQ",
            response_filter=lambda m: m["type"] == "RSP",
        )

        # Should match by ID
        assert len(pairs) == 1
        assert pairs[0].request["id"] == 100
        assert pairs[0].response["id"] == 100

    def test_correlate_respects_max_latency(self) -> None:
        """Test that max_latency constraint is respected."""
        correlator = RequestResponseCorrelator(max_latency=1.0)
        messages = [
            {"type": "REQ", "ts": 0.0},
            {"type": "RSP", "ts": 2.0},  # Latency > max_latency
        ]

        pairs = correlator.correlate(
            messages,
            request_filter=lambda m: m["type"] == "REQ",
            response_filter=lambda m: m["type"] == "RSP",
            timestamp_key=lambda m: m["ts"],
        )

        assert len(pairs) == 0

    def test_correlate_response_before_request_ignored(self) -> None:
        """Test that responses before requests are not matched."""
        correlator = RequestResponseCorrelator(max_latency=10.0)
        messages = [
            {"type": "RSP", "ts": 0.0},  # Response first
            {"type": "REQ", "ts": 1.0},
        ]

        pairs = correlator.correlate(
            messages,
            request_filter=lambda m: m["type"] == "REQ",
            response_filter=lambda m: m["type"] == "RSP",
            timestamp_key=lambda m: m["ts"],
        )

        # RSP at t=0 cannot match REQ at t=1 (negative latency)
        assert len(pairs) == 0

    def test_correlate_correlation_key_exception_handled(self) -> None:
        """Test that exceptions in correlation_key are handled."""
        correlator = RequestResponseCorrelator(
            max_latency=10.0,
            correlation_key=lambda m: m["id"],  # Will raise KeyError for some
        )
        messages = [
            {"type": "REQ"},  # No 'id' key
            {"type": "RSP"},
        ]

        # Should not raise, handles KeyError
        pairs = correlator.correlate(
            messages,
            request_filter=lambda m: m["type"] == "REQ",
            response_filter=lambda m: m["type"] == "RSP",
        )

        assert isinstance(pairs, list)

    def test_correlate_by_content_basic(self) -> None:
        """Test content-based correlation."""
        correlator = RequestResponseCorrelator(max_latency=10.0)
        messages = [
            {"data": b"ABC", "ts": 0.0},
            {"data": b"ABD", "ts": 1.0},  # Similar content
            {"data": b"XYZ", "ts": 2.0},  # Different content
        ]

        pairs = correlator.correlate_by_content(
            messages,
            content_key=lambda m: m["data"],
            timestamp_key=lambda m: m["ts"],
        )

        # Should find pair with similar content
        assert len(pairs) > 0

    def test_correlate_by_content_empty_content(self) -> None:
        """Test content correlation with empty content."""
        correlator = RequestResponseCorrelator(max_latency=10.0)
        messages = [
            {"data": b"", "ts": 0.0},
            {"data": b"ABC", "ts": 1.0},
        ]

        pairs = correlator.correlate_by_content(
            messages,
            content_key=lambda m: m["data"],
            timestamp_key=lambda m: m["ts"],
        )

        # Empty content should not match well
        assert isinstance(pairs, list)

    def test_correlate_by_content_threshold(self) -> None:
        """Test that content correlation respects similarity threshold."""
        correlator = RequestResponseCorrelator(max_latency=10.0)
        messages = [
            {"data": b"ABCD", "ts": 0.0},
            {"data": b"WXYZ", "ts": 1.0},  # Very different
        ]

        pairs = correlator.correlate_by_content(
            messages,
            content_key=lambda m: m["data"],
            timestamp_key=lambda m: m["ts"],
        )

        # Different content should not match (score < 0.3)
        assert len(pairs) == 0

    def test_correlate_by_content_without_timestamps(self) -> None:
        """Test content correlation using index as timestamp."""
        correlator = RequestResponseCorrelator(max_latency=10.0)
        messages = [
            {"data": b"ABC"},
            {"data": b"ABD"},
        ]

        pairs = correlator.correlate_by_content(
            messages,
            content_key=lambda m: m["data"],
        )

        assert isinstance(pairs, list)

    def test_extract_flows_no_flow_key(self) -> None:
        """Test flow extraction without flow key (single flow)."""
        correlator = RequestResponseCorrelator()
        pairs = [
            RequestResponsePair(0, 1, "REQ1", "RSP1", 0.5),
            RequestResponsePair(2, 3, "REQ2", "RSP2", 0.6),
        ]
        messages = ["REQ1", "RSP1", "REQ2", "RSP2"]

        flows = correlator.extract_flows(pairs, messages)

        assert len(flows) == 1
        assert flows[0].flow_id == 0
        assert len(flows[0].messages) == 4
        assert len(flows[0].pairs) == 2

    def test_extract_flows_with_flow_key(self) -> None:
        """Test flow extraction with flow key (multiple flows)."""
        correlator = RequestResponseCorrelator()
        pairs = [
            RequestResponsePair(0, 1, {"flow": "A"}, {"flow": "A"}, 0.5),
            RequestResponsePair(2, 3, {"flow": "B"}, {"flow": "B"}, 0.6),
        ]
        messages = [
            {"flow": "A"},
            {"flow": "A"},
            {"flow": "B"},
            {"flow": "B"},
        ]

        flows = correlator.extract_flows(
            pairs,
            messages,
            flow_key=lambda m: m["flow"],
        )

        assert len(flows) == 2
        # Each flow should have its own pairs
        flow_ids = {f.flow_id for f in flows}
        assert len(flow_ids) == 2

    def test_extract_flows_empty_pairs(self) -> None:
        """Test flow extraction with empty pairs."""
        correlator = RequestResponseCorrelator()
        flows = correlator.extract_flows([], [], flow_key=lambda m: m.get("id"))

        assert flows == []

    def test_extract_flows_duration_calculated(self) -> None:
        """Test that flow duration is calculated."""
        correlator = RequestResponseCorrelator()
        pairs = [
            RequestResponsePair(0, 1, {"flow": "A"}, {}, 1.0),
            RequestResponsePair(2, 3, {"flow": "A"}, {}, 2.0),
        ]
        messages = [{"flow": "A"}] * 4

        flows = correlator.extract_flows(
            pairs,
            messages,
            flow_key=lambda m: m["flow"],
        )

        assert len(flows) == 1
        assert flows[0].duration == 2.0  # Max latency

    def test_content_similarity_identical(self) -> None:
        """Test content similarity with identical bytes."""
        correlator = RequestResponseCorrelator()
        score = correlator._content_similarity(b"ABCD", b"ABCD")

        assert score == 1.0

    def test_content_similarity_different(self) -> None:
        """Test content similarity with different bytes."""
        correlator = RequestResponseCorrelator()
        score = correlator._content_similarity(b"ABCD", b"WXYZ")

        assert score < 1.0

    def test_content_similarity_empty(self) -> None:
        """Test content similarity with empty bytes."""
        correlator = RequestResponseCorrelator()

        assert correlator._content_similarity(b"", b"ABC") == 0.0
        assert correlator._content_similarity(b"ABC", b"") == 0.0
        assert correlator._content_similarity(b"", b"") == 0.0


# =============================================================================
# Test Convenience Functions
# =============================================================================


@pytest.mark.unit
@pytest.mark.inference
class TestConvenienceFunctions:
    """Test module-level convenience functions."""

    def test_detect_sequence_patterns_basic(self) -> None:
        """Test detect_sequence_patterns function."""
        messages = ["A", "B", "C", "A", "B", "C", "A", "B", "C"]
        patterns = detect_sequence_patterns(messages)

        assert len(patterns) > 0

    def test_detect_sequence_patterns_with_key(self) -> None:
        """Test detect_sequence_patterns with key function."""
        messages = [{"t": "X"}, {"t": "Y"}, {"t": "X"}, {"t": "Y"}]
        patterns = detect_sequence_patterns(messages, key=lambda m: m["t"])

        assert len(patterns) > 0

    def test_detect_sequence_patterns_custom_params(self) -> None:
        """Test detect_sequence_patterns with custom parameters."""
        messages = ["A"] * 20
        patterns = detect_sequence_patterns(
            messages,
            min_length=3,
            max_length=5,
            min_frequency=10,
        )

        for pattern in patterns:
            assert len(pattern.pattern) >= 3
            assert len(pattern.pattern) <= 5
            assert pattern.frequency >= 10

    def test_detect_sequence_patterns_empty(self) -> None:
        """Test detect_sequence_patterns with empty messages."""
        patterns = detect_sequence_patterns([])

        assert patterns == []

    def test_correlate_requests_basic(self) -> None:
        """Test correlate_requests function."""
        messages = [
            {"dir": "out", "ts": 0.0},
            {"dir": "in", "ts": 1.0},
        ]

        pairs = correlate_requests(
            messages,
            request_filter=lambda m: m["dir"] == "out",
            response_filter=lambda m: m["dir"] == "in",
            timestamp_key=lambda m: m["ts"],
        )

        assert len(pairs) == 1
        assert pairs[0].latency == pytest.approx(1.0)

    def test_correlate_requests_custom_max_latency(self) -> None:
        """Test correlate_requests with custom max_latency."""
        messages = [
            {"dir": "out", "ts": 0.0},
            {"dir": "in", "ts": 20.0},  # Latency > default 10.0
        ]

        pairs_default = correlate_requests(
            messages,
            request_filter=lambda m: m["dir"] == "out",
            response_filter=lambda m: m["dir"] == "in",
            timestamp_key=lambda m: m["ts"],
            max_latency=10.0,
        )

        pairs_extended = correlate_requests(
            messages,
            request_filter=lambda m: m["dir"] == "out",
            response_filter=lambda m: m["dir"] == "in",
            timestamp_key=lambda m: m["ts"],
            max_latency=25.0,
        )

        assert len(pairs_default) == 0
        assert len(pairs_extended) == 1

    def test_find_message_dependencies_basic(self) -> None:
        """Test find_message_dependencies function."""
        messages = [
            {"type": "REQ"},
            {"type": "RSP"},
            {"type": "REQ"},
            {"type": "ACK"},
        ]

        deps = find_message_dependencies(messages, key=lambda m: m["type"])

        assert "REQ" in deps
        assert "RSP" in deps["REQ"] or "ACK" in deps["REQ"]

    def test_find_message_dependencies_unique_successors(self) -> None:
        """Test that dependencies contain unique successors."""
        messages = [
            {"type": "A"},
            {"type": "B"},
            {"type": "A"},
            {"type": "B"},
            {"type": "A"},
            {"type": "B"},
        ]

        deps = find_message_dependencies(messages, key=lambda m: m["type"])

        # "B" should appear only once in A's successors
        assert deps["A"].count("B") == 1

    def test_find_message_dependencies_chain(self) -> None:
        """Test message dependency chain."""
        messages = [
            {"type": "START"},
            {"type": "PROCESS"},
            {"type": "END"},
        ]

        deps = find_message_dependencies(messages, key=lambda m: m["type"])

        assert "PROCESS" in deps["START"]
        assert "END" in deps["PROCESS"]

    def test_find_message_dependencies_empty(self) -> None:
        """Test find_message_dependencies with empty messages."""
        deps = find_message_dependencies([], key=lambda m: m)

        assert deps == {}

    def test_find_message_dependencies_single_message(self) -> None:
        """Test find_message_dependencies with single message."""
        deps = find_message_dependencies([{"type": "ONLY"}], key=lambda m: m["type"])

        assert deps == {}

    def test_calculate_latency_stats_basic(self) -> None:
        """Test calculate_latency_stats function."""
        pairs = [
            RequestResponsePair(0, 1, None, None, 1.0),
            RequestResponsePair(2, 3, None, None, 2.0),
            RequestResponsePair(4, 5, None, None, 3.0),
            RequestResponsePair(6, 7, None, None, 4.0),
            RequestResponsePair(8, 9, None, None, 5.0),
        ]

        stats = calculate_latency_stats(pairs)

        assert stats["min"] == 1.0
        assert stats["max"] == 5.0
        assert stats["mean"] == 3.0
        assert stats["median"] == 3.0
        assert stats["std"] == pytest.approx(np.std([1, 2, 3, 4, 5]))

    def test_calculate_latency_stats_empty(self) -> None:
        """Test calculate_latency_stats with empty pairs."""
        stats = calculate_latency_stats([])

        assert stats["min"] == 0.0
        assert stats["max"] == 0.0
        assert stats["mean"] == 0.0
        assert stats["median"] == 0.0
        assert stats["std"] == 0.0

    def test_calculate_latency_stats_single_pair(self) -> None:
        """Test calculate_latency_stats with single pair."""
        pairs = [RequestResponsePair(0, 1, None, None, 5.0)]

        stats = calculate_latency_stats(pairs)

        assert stats["min"] == 5.0
        assert stats["max"] == 5.0
        assert stats["mean"] == 5.0
        assert stats["median"] == 5.0
        assert stats["std"] == 0.0


# =============================================================================
# Edge Cases and Integration Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.inference
class TestInferenceSequencesEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_pattern_detection_with_single_element_message(self) -> None:
        """Test pattern detection when messages have single elements."""
        detector = SequencePatternDetector(min_pattern_length=1, min_frequency=2)
        messages = ["A", "A", "A"]

        patterns = detector.detect_patterns(messages)

        # Should handle gracefully
        assert isinstance(patterns, list)

    def test_pattern_detection_all_unique(self) -> None:
        """Test pattern detection with all unique messages."""
        detector = SequencePatternDetector(min_pattern_length=2, min_frequency=2)
        messages = ["A", "B", "C", "D", "E"]

        patterns = detector.detect_patterns(messages)

        # No repeating patterns
        assert len(patterns) == 0

    def test_correlation_all_requests(self) -> None:
        """Test correlation when all messages are requests."""
        correlator = RequestResponseCorrelator()
        messages = [{"type": "REQ"}, {"type": "REQ"}, {"type": "REQ"}]

        pairs = correlator.correlate(
            messages,
            request_filter=lambda m: m["type"] == "REQ",
            response_filter=lambda m: m["type"] == "RSP",
        )

        assert len(pairs) == 0

    def test_correlation_all_responses(self) -> None:
        """Test correlation when all messages are responses."""
        correlator = RequestResponseCorrelator()
        messages = [{"type": "RSP"}, {"type": "RSP"}, {"type": "RSP"}]

        pairs = correlator.correlate(
            messages,
            request_filter=lambda m: m["type"] == "REQ",
            response_filter=lambda m: m["type"] == "RSP",
        )

        assert len(pairs) == 0

    def test_correlation_interleaved_without_match(self) -> None:
        """Test correlation with interleaved but unmatched messages."""
        correlator = RequestResponseCorrelator(
            max_latency=10.0,
            correlation_key=lambda m: m.get("id"),
        )
        messages = [
            {"type": "REQ", "id": 1, "ts": 0.0},
            {"type": "RSP", "id": 2, "ts": 1.0},  # Different ID
            {"type": "REQ", "id": 3, "ts": 2.0},
            {"type": "RSP", "id": 4, "ts": 3.0},  # Different ID
        ]

        pairs = correlator.correlate(
            messages,
            request_filter=lambda m: m["type"] == "REQ",
            response_filter=lambda m: m["type"] == "RSP",
            timestamp_key=lambda m: m["ts"],
        )

        # No matching correlation IDs
        assert len(pairs) == 0

    def test_large_message_stream(self) -> None:
        """Test with large message stream."""
        detector = SequencePatternDetector(min_pattern_length=2, min_frequency=10)
        messages = ["A", "B"] * 500  # 1000 messages

        patterns = detector.detect_patterns(messages)

        assert len(patterns) > 0
        ab_pattern = next((p for p in patterns if p.pattern == ["A", "B"]), None)
        assert ab_pattern is not None
        assert ab_pattern.frequency >= 10

    def test_timestamps_with_zero_gaps(self) -> None:
        """Test pattern detection with zero timestamp gaps."""
        detector = SequencePatternDetector(min_pattern_length=2, min_frequency=2)
        messages = [
            {"type": "A", "ts": 0.0},
            {"type": "B", "ts": 0.0},  # Same timestamp
            {"type": "A", "ts": 0.0},
            {"type": "B", "ts": 0.0},
        ]

        patterns = detector.detect_patterns(
            messages,
            key=lambda m: m["type"],
            timestamp_key=lambda m: m["ts"],
        )

        assert isinstance(patterns, list)

    def test_pattern_confidence_calculation(self) -> None:
        """Test that confidence is calculated correctly."""
        detector = SequencePatternDetector(min_pattern_length=2, min_frequency=2)
        messages = [
            {"type": "A", "ts": 0.0},
            {"type": "B", "ts": 1.0},
            {"type": "A", "ts": 2.0},
            {"type": "B", "ts": 3.0},
            {"type": "A", "ts": 4.0},
            {"type": "B", "ts": 5.0},
        ]

        patterns = detector.detect_patterns(
            messages,
            key=lambda m: m["type"],
            timestamp_key=lambda m: m["ts"],
        )

        for pattern in patterns:
            # Confidence should be between 0 and 1
            assert 0.0 <= pattern.confidence <= 1.0

    def test_gap_variance_zero_when_no_variance(self) -> None:
        """Test gap variance is zero with consistent gaps."""
        detector = SequencePatternDetector(min_pattern_length=2, min_frequency=3)
        messages = [
            {"type": "A", "ts": 0.0},
            {"type": "B", "ts": 1.0},
            {"type": "A", "ts": 2.0},
            {"type": "B", "ts": 3.0},
            {"type": "A", "ts": 4.0},
            {"type": "B", "ts": 5.0},
        ]

        patterns = detector.detect_patterns(
            messages,
            key=lambda m: m["type"],
            timestamp_key=lambda m: m["ts"],
        )

        # Find AB pattern which should have consistent gap
        ab_pattern = next((p for p in patterns if p.pattern == ["A", "B"]), None)
        if ab_pattern and ab_pattern.frequency >= 2:
            # Gap variance should be close to zero
            assert ab_pattern.gap_variance == pytest.approx(0.0, abs=0.1)

    def test_multiple_correlation_matches(self) -> None:
        """Test correlation when multiple responses could match."""
        correlator = RequestResponseCorrelator(max_latency=10.0)
        messages = [
            {"type": "REQ", "ts": 0.0},
            {"type": "RSP", "ts": 1.0},
            {"type": "RSP", "ts": 2.0},  # Both could match
        ]

        pairs = correlator.correlate(
            messages,
            request_filter=lambda m: m["type"] == "REQ",
            response_filter=lambda m: m["type"] == "RSP",
            timestamp_key=lambda m: m["ts"],
        )

        # Should match with closest/best response
        assert len(pairs) == 1

    def test_response_used_only_once(self) -> None:
        """Test that each response is used only once."""
        correlator = RequestResponseCorrelator(max_latency=10.0)
        messages = [
            {"type": "REQ", "ts": 0.0},
            {"type": "REQ", "ts": 0.5},
            {"type": "RSP", "ts": 1.0},  # Can only match one REQ
        ]

        pairs = correlator.correlate(
            messages,
            request_filter=lambda m: m["type"] == "REQ",
            response_filter=lambda m: m["type"] == "RSP",
            timestamp_key=lambda m: m["ts"],
        )

        # Response should be used only once
        response_indices = [p.response_index for p in pairs]
        assert len(response_indices) == len(set(response_indices))


@pytest.mark.unit
@pytest.mark.inference
class TestScorePatterns:
    """Test internal scoring functionality."""

    def test_score_patterns_frequency_score(self) -> None:
        """Test frequency component of pattern scoring."""
        detector = SequencePatternDetector(min_pattern_length=2, min_frequency=2)

        # Create message stream with varying frequencies
        messages = ["A", "B"] * 15 + ["C", "D"] * 2

        patterns = detector.detect_patterns(messages)

        # Higher frequency patterns should have higher confidence
        ab_pattern = next((p for p in patterns if p.pattern == ["A", "B"]), None)
        cd_pattern = next((p for p in patterns if p.pattern == ["C", "D"]), None)

        if ab_pattern and cd_pattern:
            assert ab_pattern.confidence > cd_pattern.confidence

    def test_score_patterns_length_score(self) -> None:
        """Test length component of pattern scoring."""
        detector = SequencePatternDetector(
            min_pattern_length=2,
            max_pattern_length=6,
            min_frequency=2,
        )

        # Create patterns of different lengths with same frequency
        messages = ["A", "B", "C", "D", "A", "B", "C", "D", "A", "B"]

        patterns = detector.detect_patterns(messages)

        # Longer patterns should have length score contribution
        for pattern in patterns:
            assert 0.0 <= pattern.confidence <= 1.0


@pytest.mark.unit
@pytest.mark.inference
class TestMatchPairs:
    """Test internal pair matching functionality."""

    def test_match_pairs_timing_proximity(self) -> None:
        """Test that timing proximity affects matching score."""
        correlator = RequestResponseCorrelator(max_latency=10.0)
        messages = [
            {"type": "REQ", "ts": 0.0},
            {"type": "RSP", "ts": 0.1},  # Very close
        ]

        pairs = correlator.correlate(
            messages,
            request_filter=lambda m: m["type"] == "REQ",
            response_filter=lambda m: m["type"] == "RSP",
            timestamp_key=lambda m: m["ts"],
        )

        # Close timing should give high confidence
        assert len(pairs) == 1
        assert pairs[0].confidence > 0.9

    def test_match_pairs_far_timing(self) -> None:
        """Test that far timing reduces matching score."""
        correlator = RequestResponseCorrelator(max_latency=10.0)
        messages = [
            {"type": "REQ", "ts": 0.0},
            {"type": "RSP", "ts": 9.0},  # Near max latency
        ]

        pairs = correlator.correlate(
            messages,
            request_filter=lambda m: m["type"] == "REQ",
            response_filter=lambda m: m["type"] == "RSP",
            timestamp_key=lambda m: m["ts"],
        )

        # Far timing should give lower confidence
        assert len(pairs) == 1
        assert pairs[0].confidence < 0.2


@pytest.mark.unit
@pytest.mark.inference
class TestContentSimilarity:
    """Test content similarity calculation."""

    def test_content_similarity_partial_overlap(self) -> None:
        """Test content similarity with partial byte overlap."""
        correlator = RequestResponseCorrelator()

        # 50% overlap: ABCD vs ABEF shares A, B (2 out of 4 unique)
        score = correlator._content_similarity(b"ABCD", b"ABEF")

        assert 0.0 < score < 1.0

    def test_content_similarity_superset(self) -> None:
        """Test content similarity when one is superset."""
        correlator = RequestResponseCorrelator()

        # ABC vs ABCDE: ABC is subset
        score = correlator._content_similarity(b"ABC", b"ABCDE")

        # Should be intersection/union = 3/5 = 0.6
        assert score == pytest.approx(0.6, abs=0.01)

    def test_content_similarity_disjoint(self) -> None:
        """Test content similarity with no overlap."""
        correlator = RequestResponseCorrelator()

        score = correlator._content_similarity(b"ABC", b"XYZ")

        assert score == 0.0

    def test_content_similarity_single_byte(self) -> None:
        """Test content similarity with single byte."""
        correlator = RequestResponseCorrelator()

        # Same single byte
        assert correlator._content_similarity(b"A", b"A") == 1.0

        # Different single byte
        assert correlator._content_similarity(b"A", b"B") == 0.0
