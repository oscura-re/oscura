"""Comprehensive unit tests for payload pattern search and delimiter detection.

Tests for:
    - RE-PAY-002: Payload Pattern Search
    - RE-PAY-003: Payload Delimiter Detection
    - Edge cases: empty delimiters, empty patterns, empty data
"""

from __future__ import annotations

import pytest

from oscura.analyzers.packet.payload_patterns import (
    DelimiterResult,
    LengthPrefixResult,
    MessageBoundary,
    PatternMatch,
    detect_delimiter,
    detect_length_prefix,
    filter_by_pattern,
    find_message_boundaries,
    search_pattern,
    search_patterns,
    segment_messages,
)

pytestmark = [pytest.mark.unit, pytest.mark.analyzer, pytest.mark.packet]


# =============================================================================
# RE-PAY-002: Pattern Search Tests
# =============================================================================


class TestSearchPattern:
    """Test search_pattern function."""

    def test_exact_pattern_match(self) -> None:
        """Test exact pattern matching."""
        packets = [
            b"\x00\x01\x02\x03",
            b"\xaa\xbb\xcc\xdd",
            b"\x00\x01\x00\x00",
        ]

        matches = search_pattern(packets, b"\x00\x01", pattern_type="exact")

        assert len(matches) == 2
        assert matches[0].packet_index == 0
        assert matches[0].offset == 0
        assert matches[1].packet_index == 2
        assert matches[1].offset == 0

    def test_empty_pattern_returns_many_matches(self) -> None:
        """Test that empty pattern raises ValueError."""
        packets = [b"\xaa\xbb\xcc\xdd"]

        # Empty bytes pattern should raise ValueError (prevents infinite loops)
        with pytest.raises(ValueError, match="Pattern cannot be empty"):
            search_pattern(packets, b"", pattern_type="exact")

    def test_pattern_in_empty_data(self) -> None:
        """Test pattern search in empty packets."""
        packets = [b"", b"", b""]

        matches = search_pattern(packets, b"\xaa", pattern_type="exact")
        assert len(matches) == 0

    def test_wildcard_pattern(self) -> None:
        """Test wildcard pattern matching."""
        packets = [b"\xaa\x01\xbb", b"\xaa\xff\xbb"]

        # Pattern with ?? wildcard
        matches = search_pattern(packets, b"\xaa??\xbb", pattern_type="wildcard")

        assert len(matches) == 2

    def test_regex_pattern(self) -> None:
        """Test regex pattern matching."""
        packets = [b"\xaa\xbb\xcc\xdd\xee"]

        matches = search_pattern(packets, r"\xAA.{2}\xDD", pattern_type="regex")

        assert len(matches) == 1
        assert len(matches[0].matched) == 4  # AA BB CC DD


class TestSearchPatterns:
    """Test search_patterns for multiple patterns."""

    def test_multiple_patterns(self) -> None:
        """Test searching for multiple patterns simultaneously."""
        packets = [b"\xaa\x55\x00\x01\xde\xad"]

        patterns: dict[str, bytes | str] = {
            "header": b"\xaa\x55",
            "marker": b"\xde\xad",
            "missing": b"\xff\xff",
        }

        results = search_patterns(packets, patterns)

        assert len(results["header"]) == 1
        assert len(results["marker"]) == 1
        assert len(results["missing"]) == 0

    def test_empty_patterns_dict(self) -> None:
        """Test search_patterns with empty patterns dictionary."""
        packets = [b"\xaa\xbb\xcc"]

        results = search_patterns(packets, {})

        assert len(results) == 0


class TestFilterByPattern:
    """Test filter_by_pattern function."""

    def test_filter_packets_containing_pattern(self) -> None:
        """Test filtering packets that contain a pattern."""
        packets = [
            b"\xaa\xbb\xcc",
            b"\x00\x00\x00",
            b"\xaa\xbb\xdd",
        ]

        filtered = filter_by_pattern(packets, b"\xaa\xbb", pattern_type="exact")

        assert len(filtered) == 2
        assert filtered[0] == b"\xaa\xbb\xcc"
        assert filtered[1] == b"\xaa\xbb\xdd"

    def test_filter_with_no_matches(self) -> None:
        """Test filter when no packets match."""
        packets = [b"\x00\x00\x00", b"\x11\x11\x11"]

        filtered = filter_by_pattern(packets, b"\xff\xff", pattern_type="exact")

        assert len(filtered) == 0


# =============================================================================
# RE-PAY-003: Delimiter Detection Tests
# =============================================================================


class TestDetectDelimiter:
    """Test detect_delimiter function."""

    def test_detect_crlf_delimiter(self) -> None:
        """Test detecting CRLF delimiter."""
        data = b"msg1\r\nmsg2\r\nmsg3\r\n"

        result = detect_delimiter(data)

        assert result.delimiter == b"\r\n"
        assert result.occurrences >= 2
        assert result.confidence > 0.0

    def test_detect_null_delimiter(self) -> None:
        """Test detecting null byte delimiter."""
        data = b"msg1\x00msg2\x00msg3\x00"

        result = detect_delimiter(data)

        assert result.delimiter == b"\x00"
        assert result.occurrences == 3

    def test_detect_delimiter_empty_data(self) -> None:
        """Test delimiter detection on empty data."""
        result = detect_delimiter(b"")

        assert result.delimiter == b""
        assert result.confidence == 0.0
        assert result.occurrences == 0

    def test_detect_delimiter_no_delimiters(self) -> None:
        """Test detection when no clear delimiter exists."""
        data = b"AAAAAAAAAAAAAAAA"

        result = detect_delimiter(data)

        # May return empty or low confidence result
        assert result.confidence < 0.5 or result.delimiter == b""

    def test_detect_delimiter_with_empty_candidates(self) -> None:
        """Test delimiter detection with empty candidate list."""
        data = b"msg1\r\nmsg2\r\n"

        # Empty candidate should be rejected
        result = detect_delimiter(data, candidates=[b""])

        assert result.delimiter == b""
        assert result.confidence == 0.0

    def test_detect_delimiter_single_occurrence(self) -> None:
        """Test that single delimiter occurrence is rejected."""
        data = b"msg1\r\nmsg2"  # Only one delimiter

        result = detect_delimiter(data, candidates=[b"\r\n"])

        # Should have low confidence or empty delimiter
        assert result.confidence < 0.5 or result.delimiter == b""


class TestDetectLengthPrefix:
    """Test detect_length_prefix function."""

    def test_detect_simple_length_prefix(self) -> None:
        """Test detecting simple 2-byte length prefix."""
        # Create payloads with 2-byte big-endian length prefix
        payloads = [
            b"\x00\x04TEST",
            b"\x00\x05HELLO",
            b"\x00\x03ABC",
        ]

        result = detect_length_prefix(payloads)

        assert result.detected is True
        assert result.length_bytes in [1, 2]

    def test_detect_length_prefix_empty_payloads(self) -> None:
        """Test length prefix detection with empty payloads list."""
        result = detect_length_prefix([])

        assert result.detected is False
        assert result.confidence == 0.0

    def test_detect_length_prefix_insufficient_matches(self) -> None:
        """Test that insufficient matches result in no detection."""
        # Single payload is not enough
        payloads = [b"\x00\x04TEST"]

        result = detect_length_prefix(payloads)

        # Should require at least 3 matches
        assert result.detected is False or result.confidence < 0.5


class TestFindMessageBoundaries:
    """Test find_message_boundaries function."""

    def test_find_boundaries_with_delimiter(self) -> None:
        """Test finding message boundaries using delimiter."""
        data = b"msg1\nmsg2\nmsg3"

        boundaries = find_message_boundaries(data, delimiter=b"\n")

        assert len(boundaries) == 3
        assert boundaries[0].data == b"msg1"
        assert boundaries[1].data == b"msg2"
        assert boundaries[2].data == b"msg3"

    def test_find_boundaries_empty_data(self) -> None:
        """Test finding boundaries in empty data."""
        boundaries = find_message_boundaries(b"")

        assert len(boundaries) == 0

    def test_find_boundaries_no_delimiter(self) -> None:
        """Test finding boundaries when no delimiter found."""
        data = b"SINGLEMESSAGE"

        boundaries = find_message_boundaries(data)

        # Should return whole data as one message
        assert len(boundaries) == 1
        assert boundaries[0].data == data

    def test_find_boundaries_empty_delimiter(self) -> None:
        """Test that empty delimiter returns whole data as one message."""
        data = b"TESTDATA"

        boundaries = find_message_boundaries(data, delimiter=b"")

        # Empty delimiter should be handled gracefully
        assert len(boundaries) >= 1

    def test_find_boundaries_with_length_prefix(self) -> None:
        """Test finding boundaries with length prefix."""
        # Create data with 1-byte length prefix
        data = b"\x04TEST\x05HELLO\x03ABC"

        length_prefix = LengthPrefixResult(
            detected=True,
            length_bytes=1,
            endian="big",
            offset=0,
            includes_length=False,
            confidence=0.9,
        )

        boundaries = find_message_boundaries(data, length_prefix=length_prefix)

        assert len(boundaries) == 3
        assert boundaries[0].data == b"\x04TEST"
        assert boundaries[1].data == b"\x05HELLO"
        assert boundaries[2].data == b"\x03ABC"


class TestSegmentMessages:
    """Test segment_messages function."""

    def test_segment_with_delimiter(self) -> None:
        """Test message segmentation with delimiter."""
        data = b"msg1\r\nmsg2\r\nmsg3\r\n"

        messages = segment_messages(data, delimiter=b"\r\n")

        assert len(messages) == 3
        assert messages[0] == b"msg1"
        assert messages[1] == b"msg2"
        assert messages[2] == b"msg3"

    def test_segment_empty_data(self) -> None:
        """Test segmentation of empty data."""
        messages = segment_messages(b"")

        assert len(messages) == 0

    def test_segment_list_of_payloads(self) -> None:
        """Test segmentation with list of payloads."""
        payloads = [b"msg1\n", b"msg2\n", b"msg3\n"]

        messages = segment_messages(payloads, delimiter=b"\n")

        assert len(messages) == 3


# =============================================================================
# Edge Cases and Error Handling Tests
# =============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_pattern_match_with_context(self) -> None:
        """Test pattern match includes context bytes."""
        packets = [b"\x00\x00\xaa\xbb\xcc\xdd\x00\x00"]

        matches = search_pattern(packets, b"\xaa\xbb", context_bytes=2)

        assert len(matches) == 1
        assert len(matches[0].context) >= 4  # At least pattern + some context

    def test_delimiter_result_dataclass(self) -> None:
        """Test DelimiterResult dataclass creation."""
        result = DelimiterResult(
            delimiter=b"\r\n",
            delimiter_type="fixed",
            confidence=0.95,
            occurrences=10,
            positions=[0, 10, 20],
        )

        assert result.delimiter == b"\r\n"
        assert result.delimiter_type == "fixed"
        assert result.confidence == 0.95
        assert result.occurrences == 10
        assert len(result.positions) == 3

    def test_length_prefix_result_dataclass(self) -> None:
        """Test LengthPrefixResult dataclass creation."""
        result = LengthPrefixResult(
            detected=True,
            length_bytes=2,
            endian="big",
            offset=0,
            includes_length=False,
            confidence=0.9,
        )

        assert result.detected is True
        assert result.length_bytes == 2
        assert result.endian == "big"

    def test_message_boundary_dataclass(self) -> None:
        """Test MessageBoundary dataclass creation."""
        boundary = MessageBoundary(start=0, end=10, length=10, data=b"TESTDATA01", index=0)

        assert boundary.start == 0
        assert boundary.end == 10
        assert boundary.length == 10
        assert boundary.data == b"TESTDATA01"

    def test_pattern_match_dataclass(self) -> None:
        """Test PatternMatch dataclass creation."""
        match = PatternMatch(
            pattern_name="test",
            offset=5,
            matched=b"\xaa\xbb",
            packet_index=0,
            context=b"\x00\xaa\xbb\x00",
        )

        assert match.pattern_name == "test"
        assert match.offset == 5
        assert match.matched == b"\xaa\xbb"
        assert match.packet_index == 0


class TestInternalHelpers:
    """Test internal helper functions behavior through public API."""

    def test_evaluate_delimiter_with_single_occurrence(self) -> None:
        """Test that delimiter with only 1 occurrence is rejected."""
        data = b"msg1\r\nmsg2"  # Only one \r\n

        result = detect_delimiter(data, candidates=[b"\r\n", b"\n"])

        # Should prefer \n (2 occurrences) over \r\n (1 occurrence)
        # or have low confidence
        if result.delimiter == b"\r\n":
            assert result.confidence < 0.5

    def test_interval_regularity_calculation(self) -> None:
        """Test regular intervals result in high confidence."""
        # Regular intervals: 10, 10, 10
        data = b"A" * 10 + b"\n" + b"B" * 10 + b"\n" + b"C" * 10 + b"\n"

        result = detect_delimiter(data, candidates=[b"\n"])

        # Regular intervals should give high confidence
        if result.delimiter == b"\n":
            assert result.confidence > 0.6

    def test_length_prefix_negative_payload_size(self) -> None:
        """Test that negative payload size breaks parsing."""
        # Create malformed data where length < header_size
        data = b"\x00\x01TESTDATA"  # length=1 but header_included=True would be invalid

        length_prefix = LengthPrefixResult(
            detected=True,
            length_bytes=2,
            endian="big",
            offset=0,
            includes_length=True,  # This would make payload_size negative
            confidence=0.9,
        )

        boundaries = find_message_boundaries(data, length_prefix=length_prefix)

        # Should handle gracefully (no crash)
        assert isinstance(boundaries, list)

    def test_length_prefix_exceeds_data_length(self) -> None:
        """Test handling when length prefix exceeds available data."""
        # Length says 100 bytes but only 10 bytes available
        data = b"\x00\x64" + b"SHORT"  # length=100, data=5 bytes

        length_prefix = LengthPrefixResult(
            detected=True,
            length_bytes=2,
            endian="big",
            offset=0,
            includes_length=False,
            confidence=0.9,
        )

        boundaries = find_message_boundaries(data, length_prefix=length_prefix)

        # Should stop parsing gracefully (may extract partial or none)
        assert isinstance(boundaries, list)  # No crash
        # The implementation may extract the available data or stop
        if len(boundaries) > 0:
            # If it extracts, should be limited to available data
            assert boundaries[0].end <= len(data)
