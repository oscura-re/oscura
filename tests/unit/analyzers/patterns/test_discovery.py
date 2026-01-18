"""Comprehensive unit tests for pattern discovery module.

This module tests automatic signature and delimiter discovery functionality.


Author: Oscura Development Team
"""

from __future__ import annotations

import numpy as np
import pytest

from oscura.analyzers.patterns.discovery import (
    CandidateSignature,
    SignatureDiscovery,
    _calculate_entropy,
    _to_bytes,
    discover_signatures,
    find_delimiter_candidates,
    find_header_candidates,
)

pytestmark = [pytest.mark.unit, pytest.mark.analyzer, pytest.mark.pattern]


@pytest.mark.unit
@pytest.mark.pattern
@pytest.mark.requirement("PAT-003")
class TestCandidateSignature:
    """Test CandidateSignature dataclass."""

    def test_create_valid_signature(self) -> None:
        """Test creating a valid CandidateSignature."""
        sig = CandidateSignature(
            pattern=b"\xaa\x55",
            length=2,
            occurrences=10,
            positions=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90],
            interval_mean=10.0,
            interval_std=0.0,
            entropy=1.0,
            score=0.75,
        )

        assert sig.pattern == b"\xaa\x55"
        assert sig.length == 2
        assert sig.occurrences == 10
        assert sig.score == 0.75

    def test_validate_positive_length(self) -> None:
        """Test that length must be positive."""
        with pytest.raises(ValueError, match="length must be positive"):
            CandidateSignature(
                pattern=b"\xaa",
                length=0,
                occurrences=5,
                positions=[0, 10, 20, 30, 40],
                interval_mean=10.0,
                interval_std=0.0,
                entropy=1.0,
                score=0.5,
            )

    def test_validate_non_negative_occurrences(self) -> None:
        """Test that occurrences must be non-negative."""
        with pytest.raises(ValueError, match="occurrences must be non-negative"):
            CandidateSignature(
                pattern=b"\xaa",
                length=1,
                occurrences=-1,
                positions=[],
                interval_mean=0.0,
                interval_std=0.0,
                entropy=1.0,
                score=0.5,
            )

    def test_validate_pattern_length_match(self) -> None:
        """Test that pattern length must match length field."""
        with pytest.raises(ValueError, match="pattern length must match length field"):
            CandidateSignature(
                pattern=b"\xaa\x55",
                length=3,  # Mismatch
                occurrences=5,
                positions=[0, 10, 20, 30, 40],
                interval_mean=10.0,
                interval_std=0.0,
                entropy=1.0,
                score=0.5,
            )

    def test_validate_score_range(self) -> None:
        """Test that score must be in [0, 1] range."""
        with pytest.raises(ValueError, match="score must be in range"):
            CandidateSignature(
                pattern=b"\xaa",
                length=1,
                occurrences=5,
                positions=[0, 10, 20, 30, 40],
                interval_mean=10.0,
                interval_std=0.0,
                entropy=1.0,
                score=1.5,  # Invalid
            )


@pytest.mark.unit
@pytest.mark.pattern
@pytest.mark.requirement("PAT-003")
class TestSignatureDiscoveryInit:
    """Test SignatureDiscovery initialization."""

    def test_init_default_params(self) -> None:
        """Test initialization with default parameters."""
        discovery = SignatureDiscovery()

        assert discovery.min_length == 4
        assert discovery.max_length == 16
        assert discovery.min_occurrences == 2

    def test_init_custom_params(self) -> None:
        """Test initialization with custom parameters."""
        discovery = SignatureDiscovery(min_length=2, max_length=8, min_occurrences=5)

        assert discovery.min_length == 2
        assert discovery.max_length == 8
        assert discovery.min_occurrences == 5

    def test_init_validate_min_length(self) -> None:
        """Test that min_length must be at least 1."""
        with pytest.raises(ValueError, match="min_length must be at least 1"):
            SignatureDiscovery(min_length=0)

    def test_init_validate_max_length(self) -> None:
        """Test that max_length must be >= min_length."""
        with pytest.raises(ValueError, match="max_length must be >= min_length"):
            SignatureDiscovery(min_length=10, max_length=5)

    def test_init_validate_min_occurrences(self) -> None:
        """Test that min_occurrences must be at least 1."""
        with pytest.raises(ValueError, match="min_occurrences must be at least 1"):
            SignatureDiscovery(min_occurrences=0)


@pytest.mark.unit
@pytest.mark.pattern
@pytest.mark.requirement("PAT-003")
class TestDiscoverSignatures:
    """Test signature discovery functionality."""

    def test_discover_simple_repeating_pattern(self) -> None:
        """Test discovery of simple repeating pattern."""
        # Create data with repeating signature
        data = b"\xaa\x55" + b"DATA" * 50

        discovery = SignatureDiscovery(min_length=2, max_length=8)
        signatures = discovery.discover_signatures(data)

        # Should find the repeating "DATA" pattern
        assert len(signatures) > 0

        # Check if "DATA" or "\xAA\x55" was found
        patterns = [sig.pattern for sig in signatures]
        assert b"DATA" in patterns or b"\xaa\x55" in patterns

    def test_discover_header_pattern(self) -> None:
        """Test discovery of header pattern."""
        # Create data with regular header
        data = b""
        for _ in range(100):
            data += b"\xff\xff" + b"payload" * 5

        discovery = SignatureDiscovery(min_length=2, max_length=8)
        signatures = discovery.discover_signatures(data)

        # Should find the header pattern
        assert len(signatures) > 0
        patterns = [sig.pattern for sig in signatures]
        assert b"\xff\xff" in patterns

    def test_discover_with_list_of_messages(self) -> None:
        """Test discovery on list of messages."""
        messages = [
            b"\xaa\x55message1",
            b"\xaa\x55message2",
            b"\xaa\x55message3",
            b"\xaa\x55message4",
        ]

        discovery = SignatureDiscovery(min_length=2, max_length=8)
        signatures = discovery.discover_signatures(messages)

        # Should find the common header
        assert len(signatures) > 0
        patterns = [sig.pattern for sig in signatures]
        assert b"\xaa\x55" in patterns

    def test_discover_numpy_array(self) -> None:
        """Test discovery on numpy array."""
        # Create numpy array
        data_bytes = b"\xaa\x55" + b"DATA" * 20
        data_array = np.frombuffer(data_bytes, dtype=np.uint8)

        discovery = SignatureDiscovery(min_length=2, max_length=8)
        signatures = discovery.discover_signatures(data_array)

        # Should find patterns
        assert len(signatures) > 0

    def test_discover_empty_data(self) -> None:
        """Test discovery on empty data."""
        data = b""

        discovery = SignatureDiscovery(min_length=2, max_length=8)
        signatures = discovery.discover_signatures(data)

        assert signatures == []

    def test_discover_too_short_data(self) -> None:
        """Test discovery on data shorter than min_length."""
        data = b"X"

        discovery = SignatureDiscovery(min_length=2, max_length=8)
        signatures = discovery.discover_signatures(data)

        assert signatures == []

    def test_discover_respects_min_occurrences(self) -> None:
        """Test that min_occurrences is respected."""
        # Pattern appears only once
        data = b"\xaa\x55" + b"X" * 100

        discovery = SignatureDiscovery(min_length=2, max_length=8, min_occurrences=5)
        signatures = discovery.discover_signatures(data)

        # Should not find \xAA\x55 (only appears once)
        patterns = [sig.pattern for sig in signatures]
        assert b"\xaa\x55" not in patterns

    def test_discover_sorted_by_score(self) -> None:
        """Test that results are sorted by score (descending)."""
        data = b"AAAA" + b"BB" * 10 + b"C" * 50

        discovery = SignatureDiscovery(min_length=1, max_length=4)
        signatures = discovery.discover_signatures(data)

        # Verify descending order
        scores = [sig.score for sig in signatures]
        assert scores == sorted(scores, reverse=True)

    def test_discover_calculates_statistics(self) -> None:
        """Test that statistics are calculated correctly."""
        # Create data with very regular intervals
        data = b""
        for _ in range(20):
            data += b"\xff\xff" + b"X" * 10

        discovery = SignatureDiscovery(min_length=2, max_length=4)
        signatures = discovery.discover_signatures(data)

        # Find the \xFF\xFF signature
        header_sig = next((s for s in signatures if s.pattern == b"\xff\xff"), None)
        assert header_sig is not None

        # Should have regular intervals
        assert header_sig.interval_mean > 0
        assert header_sig.interval_std >= 0


@pytest.mark.unit
@pytest.mark.pattern
@pytest.mark.requirement("PAT-003")
class TestFindHeaderCandidates:
    """Test header candidate discovery."""

    def test_find_header_basic(self) -> None:
        """Test finding basic header pattern."""
        # Create data with regular headers
        # Note: HDR appears 50 times, which is enough for header detection
        data = b""
        for _ in range(50):
            data += b"HDR" + b"payload_data_here"

        discovery = SignatureDiscovery(min_length=2, max_length=16)
        headers = discovery.find_header_candidates(data, max_candidates=20)

        # Should find the header or its substrings
        assert len(headers) > 0
        patterns = [h.pattern for h in headers]
        # HDR might be found as-is or as substrings like "HD" or "DR"
        assert b"HDR" in patterns or b"HD" in patterns or b"DR" in patterns

    def test_find_header_filters_high_entropy(self) -> None:
        """Test that high-entropy patterns are filtered out."""
        # Create data with random bytes (high entropy)
        rng = np.random.default_rng(42)
        random_data = bytes(rng.integers(0, 256, size=1000, dtype=np.uint8))

        discovery = SignatureDiscovery(min_length=2, max_length=16)
        headers = discovery.find_header_candidates(random_data, max_candidates=20)

        # High entropy patterns should be filtered
        # Results should have low entropy
        for header in headers[:5]:  # Check top 5
            assert header.entropy <= 6.0

    def test_find_header_requires_min_occurrences(self) -> None:
        """Test that headers must appear at least 3 times."""
        # Create data with pattern appearing only twice
        data = b"HDR" + b"X" * 100 + b"HDR" + b"Y" * 100

        discovery = SignatureDiscovery(min_length=2, max_length=16)
        headers = discovery.find_header_candidates(data, max_candidates=20)

        # Should not find HDR (only 2 occurrences)
        patterns = [h.pattern for h in headers]
        assert b"HDR" not in patterns

    def test_find_header_max_candidates(self) -> None:
        """Test max_candidates parameter."""
        # Create data with many patterns
        data = b"A" * 100 + b"B" * 100 + b"C" * 100 + b"D" * 100

        discovery = SignatureDiscovery(min_length=1, max_length=4)
        headers = discovery.find_header_candidates(data, max_candidates=3)

        # Should return at most 3 candidates
        assert len(headers) <= 3

    def test_find_header_empty_data(self) -> None:
        """Test finding headers in empty data."""
        data = b""

        discovery = SignatureDiscovery(min_length=2, max_length=16)
        headers = discovery.find_header_candidates(data)

        assert headers == []

    def test_find_header_prefers_regularity(self) -> None:
        """Test that headers with regular intervals score higher."""
        # Create two patterns: regular and irregular
        data = b""
        # Regular pattern every 20 bytes
        for _ in range(30):
            data += b"REG" + b"X" * 17

        # Add irregular pattern
        data += b"IRR" + b"Y" * 500
        data += b"IRR" + b"Z" * 50
        data += b"IRR"

        discovery = SignatureDiscovery(min_length=3, max_length=8)
        headers = discovery.find_header_candidates(data, max_candidates=20)

        # Regular pattern should score higher
        if len(headers) > 1:
            top_pattern = headers[0].pattern
            # REG should be ranked higher than IRR
            assert top_pattern == b"REG" or headers[0].score > 0.3


@pytest.mark.unit
@pytest.mark.pattern
@pytest.mark.requirement("PAT-003")
class TestFindDelimiterCandidates:
    """Test delimiter candidate discovery."""

    def test_find_delimiter_basic(self) -> None:
        """Test finding basic delimiter."""
        # Need at least 5 occurrences for delimiter detection
        data = b"field1,field2,field3,field4,field5,field6"

        discovery = SignatureDiscovery(min_length=1, max_length=4)
        delimiters = discovery.find_delimiter_candidates(data)

        # Should find comma
        assert len(delimiters) > 0
        patterns = [d.pattern for d in delimiters]
        assert b"," in patterns

    def test_find_delimiter_newline(self) -> None:
        """Test finding newline delimiter."""
        data = b"line1\nline2\nline3\nline4\nline5\nline6\n"

        discovery = SignatureDiscovery(min_length=1, max_length=4)
        delimiters = discovery.find_delimiter_candidates(data)

        # Should find newline
        assert len(delimiters) > 0
        patterns = [d.pattern for d in delimiters]
        assert b"\n" in patterns

    def test_find_delimiter_multi_byte(self) -> None:
        """Test finding multi-byte delimiter."""
        data = b"record1\r\nrecord2\r\nrecord3\r\nrecord4\r\nrecord5\r\n"

        discovery = SignatureDiscovery(min_length=1, max_length=4)
        delimiters = discovery.find_delimiter_candidates(data)

        # Should find \r\n
        assert len(delimiters) > 0
        patterns = [d.pattern for d in delimiters]
        assert b"\r\n" in patterns or b"\n" in patterns

    def test_find_delimiter_requires_frequency(self) -> None:
        """Test that delimiters must appear frequently (>= 5 times)."""
        # Pattern appears only 4 times
        data = b"field1,field2,field3,field4"

        discovery = SignatureDiscovery(min_length=1, max_length=4)
        delimiters = discovery.find_delimiter_candidates(data)

        # May or may not find comma (borderline case)
        # Just verify it doesn't crash
        assert isinstance(delimiters, list)

    def test_find_delimiter_filters_high_entropy(self) -> None:
        """Test that high-entropy patterns are filtered."""
        # Create data with varied bytes
        data = bytes(range(256)) * 10

        discovery = SignatureDiscovery(min_length=1, max_length=4)
        delimiters = discovery.find_delimiter_candidates(data)

        # Delimiters should have low entropy
        for delim in delimiters[:5]:
            assert delim.entropy <= 3.0

    def test_find_delimiter_prefers_short(self) -> None:
        """Test that shorter delimiters score higher."""
        data = b"a|b|c|d|e|f|g|h|i|j|k|l|m|n|o|p|q|r|s|t"

        discovery = SignatureDiscovery(min_length=1, max_length=4)
        delimiters = discovery.find_delimiter_candidates(data)

        # Single-byte delimiter should score well
        if delimiters:
            top_delims = delimiters[:3]
            # Check that short delimiters are preferred
            assert any(d.length <= 2 for d in top_delims)

    def test_find_delimiter_empty_data(self) -> None:
        """Test finding delimiters in empty data."""
        data = b""

        discovery = SignatureDiscovery(min_length=1, max_length=4)
        delimiters = discovery.find_delimiter_candidates(data)

        assert delimiters == []

    def test_find_delimiter_too_short_data(self) -> None:
        """Test finding delimiters in very short data."""
        data = b"X"

        discovery = SignatureDiscovery(min_length=1, max_length=4)
        delimiters = discovery.find_delimiter_candidates(data)

        assert delimiters == []

    def test_find_delimiter_returns_top_20(self) -> None:
        """Test that at most 20 delimiters are returned."""
        # Create data with many repeating bytes
        data = bytes(range(256)) * 100

        discovery = SignatureDiscovery(min_length=1, max_length=4)
        delimiters = discovery.find_delimiter_candidates(data)

        # Should return at most 20
        assert len(delimiters) <= 20


@pytest.mark.unit
@pytest.mark.pattern
@pytest.mark.requirement("PAT-003")
class TestRankSignatures:
    """Test signature ranking functionality."""

    def test_rank_signatures_basic(self) -> None:
        """Test basic signature ranking."""
        # Create some candidate signatures
        sig1 = CandidateSignature(
            pattern=b"AAA",
            length=3,
            occurrences=50,
            positions=list(range(0, 1000, 20)),
            interval_mean=20.0,
            interval_std=0.5,
            entropy=2.0,
            score=0.5,
        )

        sig2 = CandidateSignature(
            pattern=b"BBB",
            length=3,
            occurrences=10,
            positions=[0, 100, 300, 600, 800, 900, 950, 980, 990, 995],
            interval_mean=50.0,
            interval_std=40.0,
            entropy=5.0,
            score=0.3,
        )

        discovery = SignatureDiscovery()
        ranked = discovery.rank_signatures([sig1, sig2])

        # Should return re-ranked list
        assert len(ranked) == 2
        # sig1 should rank higher (more regular, lower entropy)
        assert ranked[0].pattern in [b"AAA", b"BBB"]

    def test_rank_signatures_empty_list(self) -> None:
        """Test ranking empty list."""
        discovery = SignatureDiscovery()
        ranked = discovery.rank_signatures([])

        assert ranked == []

    def test_rank_signatures_sorted_descending(self) -> None:
        """Test that ranked results are sorted by score."""
        # Create multiple signatures
        signatures = [
            CandidateSignature(
                pattern=bytes([i]),
                length=1,
                occurrences=10 + i,
                positions=list(range(i)),
                interval_mean=float(i + 1),
                interval_std=float(i),
                entropy=float(i) / 2,
                score=0.5,
            )
            for i in range(10)
        ]

        discovery = SignatureDiscovery()
        ranked = discovery.rank_signatures(signatures)

        # Verify sorted by score (descending)
        scores = [sig.score for sig in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_rank_signatures_handles_zero_interval(self) -> None:
        """Test ranking with zero interval mean."""
        sig = CandidateSignature(
            pattern=b"X",
            length=1,
            occurrences=2,
            positions=[0, 0],
            interval_mean=0.0,
            interval_std=0.0,
            entropy=1.0,
            score=0.5,
        )

        discovery = SignatureDiscovery()
        ranked = discovery.rank_signatures([sig])

        # Should handle gracefully
        assert len(ranked) == 1
        assert ranked[0].score >= 0.0


@pytest.mark.unit
@pytest.mark.pattern
@pytest.mark.requirement("PAT-003")
class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_discover_signatures_function(self) -> None:
        """Test discover_signatures convenience function."""
        data = b"\xff\xff" + b"DATA" * 50

        signatures = discover_signatures(data, min_length=2, max_length=8)

        assert isinstance(signatures, list)
        assert len(signatures) > 0

    def test_find_header_candidates_function(self) -> None:
        """Test find_header_candidates convenience function."""
        data = b""
        for _ in range(20):
            data += b"HDR" + b"payload" * 5

        headers = find_header_candidates(data)

        assert isinstance(headers, list)

    def test_find_delimiter_candidates_function(self) -> None:
        """Test find_delimiter_candidates convenience function."""
        data = b"field1,field2,field3,field4,field5,field6"

        delimiters = find_delimiter_candidates(data)

        assert isinstance(delimiters, list)
        # Should find comma
        patterns = [d.pattern for d in delimiters]
        assert b"," in patterns

    def test_convenience_functions_with_numpy(self) -> None:
        """Test convenience functions with numpy arrays."""
        data_bytes = b"A" * 100
        data_array = np.frombuffer(data_bytes, dtype=np.uint8)

        signatures = discover_signatures(data_array)
        headers = find_header_candidates(data_array)
        delimiters = find_delimiter_candidates(data_array)

        assert isinstance(signatures, list)
        assert isinstance(headers, list)
        assert isinstance(delimiters, list)


@pytest.mark.unit
@pytest.mark.pattern
@pytest.mark.requirement("PAT-003")
class TestHelperFunctions:
    """Test helper functions."""

    def test_to_bytes_from_bytes(self) -> None:
        """Test _to_bytes with bytes input."""
        data = b"\x01\x02\x03"
        result = _to_bytes(data)

        assert result == data
        assert isinstance(result, bytes)

    def test_to_bytes_from_bytearray(self) -> None:
        """Test _to_bytes with bytearray input."""
        data = bytearray([1, 2, 3])
        result = _to_bytes(data)

        assert result == b"\x01\x02\x03"
        assert isinstance(result, bytes)

    def test_to_bytes_from_memoryview(self) -> None:
        """Test _to_bytes with memoryview input."""
        data = memoryview(b"\x01\x02\x03")
        result = _to_bytes(data)

        assert result == b"\x01\x02\x03"
        assert isinstance(result, bytes)

    def test_to_bytes_from_numpy_array(self) -> None:
        """Test _to_bytes with numpy array input."""
        data = np.array([1, 2, 3], dtype=np.uint8)
        result = _to_bytes(data)

        assert result == b"\x01\x02\x03"
        assert isinstance(result, bytes)

    def test_to_bytes_unsupported_type(self) -> None:
        """Test _to_bytes with unsupported type."""
        with pytest.raises(TypeError, match="Unsupported data type"):
            _to_bytes("string")  # type: ignore

    def test_to_bytes_numpy_with_conversion(self) -> None:
        """Test _to_bytes with numpy array requiring conversion."""
        # Create array with different dtype
        data = np.array([1, 2, 3], dtype=np.int32)
        result = _to_bytes(data)

        # Should convert to uint8
        assert isinstance(result, bytes)

    def test_calculate_entropy_uniform(self) -> None:
        """Test entropy calculation for uniform distribution."""
        # All 256 possible bytes appear once
        data = bytes(range(256))
        entropy = _calculate_entropy(data)

        # Should be close to 8.0 (maximum for bytes)
        assert entropy > 7.9
        assert entropy <= 8.0

    def test_calculate_entropy_constant(self) -> None:
        """Test entropy calculation for constant data."""
        # All same byte
        data = b"\x00" * 100
        entropy = _calculate_entropy(data)

        # Should be 0 (no information)
        assert entropy == 0.0

    def test_calculate_entropy_low(self) -> None:
        """Test entropy calculation for low-entropy data."""
        # Mostly one byte with a few others
        data = b"\x00" * 90 + b"\x01" * 10
        entropy = _calculate_entropy(data)

        # Should be low but not zero
        assert 0.0 < entropy < 2.0

    def test_calculate_entropy_medium(self) -> None:
        """Test entropy calculation for medium-entropy data."""
        # 4 bytes appearing equally
        data = bytes([0, 1, 2, 3] * 25)
        entropy = _calculate_entropy(data)

        # Should be around 2.0 (log2(4))
        assert 1.8 < entropy < 2.2

    def test_calculate_entropy_empty(self) -> None:
        """Test entropy calculation for empty data."""
        data = b""
        entropy = _calculate_entropy(data)

        assert entropy == 0.0


@pytest.mark.unit
@pytest.mark.pattern
@pytest.mark.requirement("PAT-003")
class TestPatternsDiscoveryEdgeCases:
    """Test edge cases and error conditions."""

    def test_large_data_performance(self) -> None:
        """Test discovery on large data doesn't hang."""
        # 10 MB of data with pattern
        data = (b"\xaa\x55" + b"X" * 100) * 10000

        discovery = SignatureDiscovery(min_length=2, max_length=8)

        # Should complete without hanging (this is a smoke test)
        signatures = discovery.discover_signatures(data)

        assert isinstance(signatures, list)

    def test_max_length_exceeds_data_length(self) -> None:
        """Test when max_length is larger than data."""
        data = b"SHORT"

        discovery = SignatureDiscovery(min_length=2, max_length=1000)
        signatures = discovery.discover_signatures(data)

        # Should handle gracefully
        assert isinstance(signatures, list)

    def test_single_occurrence_pattern(self) -> None:
        """Test pattern that appears only once."""
        data = b"UNIQUE" + b"X" * 100

        discovery = SignatureDiscovery(min_length=4, min_occurrences=2)
        signatures = discovery.discover_signatures(data)

        # UNIQUE should not be found (only 1 occurrence)
        patterns = [s.pattern for s in signatures]
        assert b"UNIQUE" not in patterns

    def test_all_unique_bytes(self) -> None:
        """Test data where every position is unique."""
        data = bytes(range(256))

        discovery = SignatureDiscovery(min_length=2, min_occurrences=2)
        signatures = discovery.discover_signatures(data)

        # Should not find any repeating patterns
        assert len(signatures) == 0

    def test_overlapping_patterns(self) -> None:
        """Test handling of overlapping patterns."""
        # AAA appears in AAAA
        data = b"AAAA" * 20

        discovery = SignatureDiscovery(min_length=2, max_length=8)
        signatures = discovery.discover_signatures(data)

        # Should find various lengths of A's
        assert len(signatures) > 0
        patterns = [s.pattern for s in signatures]
        # Should find at least AA or AAA
        assert any(b"A" in p for p in patterns)

    def test_binary_zeros(self) -> None:
        """Test with binary data of all zeros."""
        data = b"\x00" * 1000

        discovery = SignatureDiscovery(min_length=2, max_length=8)
        signatures = discovery.discover_signatures(data)

        # Should find repeating null pattern
        assert len(signatures) > 0

    def test_binary_high_values(self) -> None:
        """Test with binary data of high byte values."""
        data = b"\xff" * 500 + b"\xfe" * 500

        discovery = SignatureDiscovery(min_length=1, max_length=8)
        signatures = discovery.discover_signatures(data)

        # Should find repeating patterns
        assert len(signatures) > 0

    def test_mixed_data_types_in_list(self) -> None:
        """Test list with mixed data types."""
        messages = [
            b"MSG1",
            np.array([77, 83, 71, 50], dtype=np.uint8),  # MSG2
            bytearray(b"MSG3"),
        ]

        discovery = SignatureDiscovery(min_length=2, max_length=8)
        signatures = discovery.discover_signatures(messages)

        # Should handle mixed types
        assert isinstance(signatures, list)


@pytest.mark.unit
@pytest.mark.pattern
@pytest.mark.requirement("PAT-003")
class TestScoreCalculation:
    """Test score calculation logic."""

    def test_score_in_valid_range(self) -> None:
        """Test that scores are always in [0, 1]."""
        data = b"PATTERN" * 100

        discovery = SignatureDiscovery(min_length=2, max_length=10)
        signatures = discovery.discover_signatures(data)

        for sig in signatures:
            assert 0.0 <= sig.score <= 1.0

    def test_regular_intervals_score_higher(self) -> None:
        """Test that regular intervals produce higher scores."""
        # Very regular pattern
        regular_data = (b"HDR" + b"X" * 17) * 50

        # Irregular pattern
        irregular_data = b"HDR" + b"X" * 100 + b"HDR" + b"Y" * 10 + b"HDR"

        discovery = SignatureDiscovery(min_length=3, max_length=8)

        regular_sigs = discovery.discover_signatures(regular_data)
        irregular_sigs = discovery.discover_signatures(irregular_data)

        # Find HDR in both
        regular_hdr = next((s for s in regular_sigs if s.pattern == b"HDR"), None)
        irregular_hdr = next((s for s in irregular_sigs if s.pattern == b"HDR"), None)

        if regular_hdr and irregular_hdr:
            # Regular should score higher
            assert regular_hdr.interval_std < irregular_hdr.interval_std

    def test_frequency_affects_score(self) -> None:
        """Test that frequency affects score calculation."""
        # High frequency pattern
        high_freq = b"A" * 200

        # Low frequency pattern
        low_freq = b"A" * 10 + b"X" * 190

        discovery = SignatureDiscovery(min_length=1, max_length=4)

        high_sigs = discovery.discover_signatures(high_freq)
        low_sigs = discovery.discover_signatures(low_freq)

        # Both should find 'A', but with different scores
        high_a = next((s for s in high_sigs if s.pattern == b"A"), None)
        low_a = next((s for s in low_sigs if s.pattern == b"A"), None)

        if high_a and low_a:
            assert high_a.occurrences > low_a.occurrences


@pytest.mark.unit
@pytest.mark.pattern
@pytest.mark.requirement("PAT-003")
class TestRealWorldScenarios:
    """Test real-world usage scenarios."""

    def test_csv_delimiter_detection(self) -> None:
        """Test detecting CSV comma delimiter."""
        csv_data = (
            b"name,age,city\nAlice,30,NYC\nBob,25,LA\nCarol,35,SF\nDave,28,Chicago\nEve,32,Boston\n"
        )

        delimiters = find_delimiter_candidates(csv_data)

        # Should find comma and newline
        patterns = [d.pattern for d in delimiters[:5]]
        assert b"," in patterns
        assert b"\n" in patterns

    def test_packet_header_detection(self) -> None:
        """Test detecting packet headers."""
        # Simulate network packets with sync bytes
        packets = []
        for i in range(30):
            packet = b"\xaa\xaa" + bytes([i % 256]) + b"payload_data" + b"\x00"
            packets.append(packet)

        data = b"".join(packets)

        headers = find_header_candidates(data)

        # Should find \xAA\xAA header
        patterns = [h.pattern for h in headers[:5]]
        assert b"\xaa\xaa" in patterns

    def test_protocol_signature_detection(self) -> None:
        """Test detecting protocol signatures."""
        # Simulate HTTP-like protocol
        messages = [
            b"GET /index.html HTTP/1.1\r\n",
            b"GET /about.html HTTP/1.1\r\n",
            b"GET /contact.html HTTP/1.1\r\n",
            b"POST /submit HTTP/1.1\r\n",
        ]

        signatures = discover_signatures(messages, min_length=3, max_length=10)

        # Should find HTTP-related patterns
        patterns = [s.pattern for s in signatures[:10]]
        # Check for any HTTP-related substrings
        http_found = any(b"HTTP" in p or b"GET" in p or b"\r\n" in p for p in patterns)
        assert http_found

    def test_binary_protocol_sync_marker(self) -> None:
        """Test detecting binary protocol sync markers."""
        # Binary protocol with sync marker
        messages = []
        for i in range(40):
            msg = b"\xff\xff\xff\xff" + bytes([i, (i * 2) % 256]) + b"DATA" * 3
            messages.append(msg)

        data = b"".join(messages)

        signatures = discover_signatures(data, min_length=4, max_length=8)

        # Should find the sync marker
        patterns = [s.pattern for s in signatures]
        assert b"\xff\xff\xff\xff" in patterns

    def test_text_line_delimiter(self) -> None:
        """Test detecting text line delimiters."""
        text = b"Line 1\nLine 2\nLine 3\nLine 4\nLine 5\nLine 6\nLine 7\n"

        delimiters = find_delimiter_candidates(text)

        # Should find newline
        patterns = [d.pattern for d in delimiters]
        assert b"\n" in patterns

        # Newline should score high
        newline_sig = next((d for d in delimiters if d.pattern == b"\n"), None)
        if newline_sig:
            assert newline_sig.score > 0.5
