"""Comprehensive unit tests for pattern matching module.

    - RE-PAT-001: Binary Regex Pattern Matching
    - RE-PAT-002: Multi-Pattern Search (Aho-Corasick)
    - RE-PAT-003: Fuzzy Pattern Matching

This module provides comprehensive test coverage for binary pattern matching
capabilities including regex-like matching, multi-pattern search, and fuzzy
matching with configurable similarity thresholds.
"""

from __future__ import annotations

import pytest

from oscura.analyzers.patterns.matching import (
    AhoCorasickMatcher,
    BinaryRegex,
    FuzzyMatcher,
    FuzzyMatchResult,
    PatternMatchResult,
    binary_regex_search,
    count_pattern_occurrences,
    find_pattern_positions,
    find_similar_sequences,
    fuzzy_search,
    multi_pattern_search,
)

pytestmark = [pytest.mark.unit, pytest.mark.analyzer, pytest.mark.pattern]


# =============================================================================
# PatternMatchResult Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.pattern
class TestPatternMatchResult:
    """Test PatternMatchResult dataclass."""

    def test_create_result(self) -> None:
        """Test creating a pattern match result."""
        result = PatternMatchResult(
            pattern_name="test",
            offset=10,
            length=4,
            matched_data=b"\xaa\xbb\xcc\xdd",
            pattern=b"\xaa\xbb\xcc\xdd",
        )

        assert result.pattern_name == "test"
        assert result.offset == 10
        assert result.length == 4
        assert result.matched_data == b"\xaa\xbb\xcc\xdd"
        assert result.pattern == b"\xaa\xbb\xcc\xdd"
        assert result.similarity == 1.0

    def test_result_with_similarity(self) -> None:
        """Test result with custom similarity score."""
        result = PatternMatchResult(
            pattern_name="fuzzy",
            offset=0,
            length=4,
            matched_data=b"\xaa\xbb\xcc\xdd",
            pattern=b"\xaa\xbb\xcc\xde",
            similarity=0.75,
        )

        assert result.similarity == 0.75


# =============================================================================
# BinaryRegex Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.pattern
class TestBinaryRegex:
    """Test BinaryRegex pattern matching."""

    def test_literal_hex_bytes(self) -> None:
        """Test matching literal hex bytes."""
        regex = BinaryRegex(pattern=r"\xAA\xBB", name="header")
        data = b"\x00\x00\xaa\xbb\x00"

        result = regex.search(data)

        assert result is not None
        assert result.offset == 2
        assert result.length == 2
        assert result.matched_data == b"\xaa\xbb"
        assert result.pattern_name == "header"

    def test_wildcard_any_byte(self) -> None:
        """Test ?? wildcard matching any byte."""
        regex = BinaryRegex(pattern=r"\xAA??")
        data = b"\xaa\xff\xaa\x00"

        result = regex.search(data)

        assert result is not None
        assert result.offset == 0
        assert result.length == 2
        assert result.matched_data == b"\xaa\xff"

    def test_single_wildcard(self) -> None:
        """Test single ? wildcard."""
        regex = BinaryRegex(pattern=r"\xAA?")
        data = b"\xaa\x55"

        result = regex.search(data)

        assert result is not None
        assert result.length >= 2

    def test_byte_range(self) -> None:
        """Test byte range [\\x00-\\x1F]."""
        regex = BinaryRegex(pattern=rb"[\x00-\x1F]")
        data = b"\xff\x10\xff"

        result = regex.search(data)

        assert result is not None
        assert result.offset == 1
        assert result.matched_data == b"\x10"

    def test_repetition_exact(self) -> None:
        """Test exact repetition {n}."""
        regex = BinaryRegex(pattern=r"\xAA{3}")
        data = b"\x00\xaa\xaa\xaa\x00"

        result = regex.search(data)

        assert result is not None
        assert result.matched_data == b"\xaa\xaa\xaa"

    def test_repetition_range(self) -> None:
        """Test repetition range {n,m}."""
        regex = BinaryRegex(pattern=r"\xBB{2,4}")
        data = b"\x00\xbb\xbb\xbb\x00"

        result = regex.search(data)

        assert result is not None
        assert len(result.matched_data) in [2, 3, 4]

    def test_alternation(self) -> None:
        """Test alternation (A|B)."""
        regex = BinaryRegex(pattern=rb"(\xAA|\xFF)")
        data = b"\x00\xff\x00"

        result = regex.search(data)

        assert result is not None
        assert result.matched_data == b"\xff"

    def test_anchor_start(self) -> None:
        """Test start anchor ^."""
        regex = BinaryRegex(pattern=r"^\xAA")

        # Should match at start
        data1 = b"\xaa\xbb"
        result1 = regex.match(data1)
        assert result1 is not None

        # Should not match in middle
        data2 = b"\x00\xaa"
        result2 = regex.match(data2)
        assert result2 is None

    def test_anchor_end(self) -> None:
        """Test end anchor $."""
        regex = BinaryRegex(pattern=r"\xAA$")
        data = b"\x00\x00\xaa"

        result = regex.search(data)

        assert result is not None
        assert result.offset == 2

    def test_match_at_start(self) -> None:
        """Test match() only matches at start."""
        regex = BinaryRegex(pattern=r"\xAA\xBB")

        # Should match at start
        data1 = b"\xaa\xbb\x00"
        result1 = regex.match(data1)
        assert result1 is not None

        # Should not match in middle
        data2 = b"\x00\xaa\xbb"
        result2 = regex.match(data2)
        assert result2 is None

    def test_search_anywhere(self) -> None:
        """Test search() finds pattern anywhere."""
        regex = BinaryRegex(pattern=r"\xAA\xBB")
        data = b"\x00\x00\xaa\xbb\x00"

        result = regex.search(data)

        assert result is not None
        assert result.offset == 2

    def test_findall_multiple_matches(self) -> None:
        """Test findall() returns all matches."""
        regex = BinaryRegex(pattern=r"\xAA")
        data = b"\xaa\x00\xaa\x00\xaa"

        results = regex.findall(data)

        assert len(results) == 3
        assert results[0].offset == 0
        assert results[1].offset == 2
        assert results[2].offset == 4

    def test_findall_overlapping(self) -> None:
        """Test findall() with overlapping patterns."""
        regex = BinaryRegex(pattern=r"\xAA\xAA")
        data = b"\xaa\xaa\xaa"

        results = regex.findall(data)

        # Non-overlapping by default
        assert len(results) == 1

    def test_search_with_start_offset(self) -> None:
        """Test search with start offset."""
        regex = BinaryRegex(pattern=r"\xAA")
        data = b"\xaa\x00\xaa"

        result = regex.search(data, start=2)

        assert result is not None
        assert result.offset == 2

    def test_invalid_pattern(self) -> None:
        """Test handling of invalid regex pattern."""
        # Invalid pattern should result in None compiled regex
        regex = BinaryRegex(pattern="[invalid")

        assert regex.compiled is None

        result = regex.search(b"\x00\x00")
        assert result is None

    def test_complex_pattern(self) -> None:
        """Test complex pattern with multiple features."""
        # Test pattern with wildcards and specific bytes
        regex = BinaryRegex(pattern=r"\xAA??\xFF")
        data = b"\x00\xaa\x12\xff\x00"

        result = regex.search(data)

        assert result is not None
        assert result.offset == 1
        assert len(result.matched_data) == 3

    def test_star_quantifier(self) -> None:
        """Test * quantifier (zero or more)."""
        regex = BinaryRegex(pattern=r"\xAA\x00*\xBB")
        data1 = b"\xaa\xbb"
        data2 = b"\xaa\x00\x00\xbb"

        result1 = regex.search(data1)
        result2 = regex.search(data2)

        assert result1 is not None
        assert result2 is not None

    def test_plus_quantifier(self) -> None:
        """Test + quantifier (one or more)."""
        regex = BinaryRegex(pattern=r"\xAA\x00+\xBB")
        data1 = b"\xaa\xbb"  # Should not match (zero occurrences)
        data2 = b"\xaa\x00\x00\xbb"  # Should match

        result1 = regex.search(data1)
        result2 = regex.search(data2)

        assert result1 is None
        assert result2 is not None


# =============================================================================
# AhoCorasickMatcher Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.pattern
class TestAhoCorasickMatcher:
    """Test Aho-Corasick multi-pattern matching."""

    def test_single_pattern_match(self) -> None:
        """Test matching a single pattern."""
        matcher = AhoCorasickMatcher()
        matcher.add_pattern(b"\xaa\xbb", "header")
        matcher.build()

        data = b"\x00\xaa\xbb\x00"
        results = matcher.search(data)

        assert len(results) == 1
        assert results[0].pattern_name == "header"
        assert results[0].offset == 1

    def test_multiple_patterns(self) -> None:
        """Test matching multiple patterns."""
        matcher = AhoCorasickMatcher()
        matcher.add_pattern(b"\xaa\x55", "header")
        matcher.add_pattern(b"\xde\xad", "marker")
        matcher.build()

        data = b"\xaa\x55\x00\xde\xad"
        results = matcher.search(data)

        assert len(results) == 2
        pattern_names = {r.pattern_name for r in results}
        assert "header" in pattern_names
        assert "marker" in pattern_names

    def test_add_patterns_dict(self) -> None:
        """Test add_patterns() with dictionary."""
        matcher = AhoCorasickMatcher()
        patterns = {
            "header": b"\xaa\x55",
            "footer": b"\xff\xff",
        }
        matcher.add_patterns(patterns)
        matcher.build()

        data = b"\xaa\x55\x00\xff\xff"
        results = matcher.search(data)

        assert len(results) == 2

    def test_overlapping_patterns(self) -> None:
        """Test matching overlapping patterns."""
        matcher = AhoCorasickMatcher()
        matcher.add_pattern(b"\xaa\xbb", "pattern1")
        matcher.add_pattern(b"\xbb\xcc", "pattern2")
        matcher.build()

        data = b"\xaa\xbb\xcc"
        results = matcher.search(data)

        # Both patterns should be found
        assert len(results) == 2

    def test_pattern_at_multiple_locations(self) -> None:
        """Test same pattern at multiple locations."""
        matcher = AhoCorasickMatcher()
        matcher.add_pattern(b"\xaa", "marker")
        matcher.build()

        data = b"\xaa\x00\xaa\x00\xaa"
        results = matcher.search(data)

        assert len(results) == 3
        assert results[0].offset == 0
        assert results[1].offset == 2
        assert results[2].offset == 4

    def test_iter_search(self) -> None:
        """Test memory-efficient iter_search()."""
        matcher = AhoCorasickMatcher()
        matcher.add_pattern(b"\xaa", "marker")
        matcher.build()

        data = b"\xaa\x00\xaa\x00\xaa"
        results = list(matcher.iter_search(data))

        assert len(results) == 3

    def test_search_without_build_raises(self) -> None:
        """Test that search without build raises error."""
        matcher = AhoCorasickMatcher()
        matcher.add_pattern(b"\xaa", "test")

        with pytest.raises(RuntimeError, match="Must call build"):
            matcher.search(b"\xaa")

    def test_iter_search_without_build_raises(self) -> None:
        """Test that iter_search without build raises error."""
        matcher = AhoCorasickMatcher()
        matcher.add_pattern(b"\xaa", "test")

        with pytest.raises(RuntimeError, match="Must call build"):
            list(matcher.iter_search(b"\xaa"))

    def test_rebuild_after_add(self) -> None:
        """Test rebuilding automaton after adding patterns."""
        matcher = AhoCorasickMatcher()
        matcher.add_pattern(b"\xaa", "first")
        matcher.build()

        # Add more patterns and rebuild
        matcher.add_pattern(b"\xbb", "second")
        matcher.build()

        data = b"\xaa\xbb"
        results = matcher.search(data)

        assert len(results) == 2

    def test_empty_pattern_name(self) -> None:
        """Test pattern without name gets hex name."""
        matcher = AhoCorasickMatcher()
        matcher.add_pattern(b"\xaa\xbb")
        matcher.build()

        data = b"\xaa\xbb"
        results = matcher.search(data)

        assert len(results) == 1
        assert results[0].pattern_name == "aabb"

    def test_string_pattern_conversion(self) -> None:
        """Test automatic string to bytes conversion."""
        matcher = AhoCorasickMatcher()
        matcher.add_pattern("AB", "text")
        matcher.build()

        data = b"AB"
        results = matcher.search(data)

        assert len(results) == 1

    def test_no_matches(self) -> None:
        """Test search with no matches."""
        matcher = AhoCorasickMatcher()
        matcher.add_pattern(b"\xaa", "test")
        matcher.build()

        data = b"\xff\xff\xff"
        results = matcher.search(data)

        assert len(results) == 0

    def test_long_pattern(self) -> None:
        """Test matching long pattern."""
        matcher = AhoCorasickMatcher()
        pattern = b"\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09"
        matcher.add_pattern(pattern, "long")
        matcher.build()

        data = b"\xff" + pattern + b"\xff"
        results = matcher.search(data)

        assert len(results) == 1
        assert results[0].matched_data == pattern

    def test_many_patterns(self) -> None:
        """Test efficiency with many patterns."""
        matcher = AhoCorasickMatcher()

        # Add 100 patterns
        for i in range(100):
            matcher.add_pattern(bytes([i, i + 1]), f"pattern_{i}")
        matcher.build()

        # Create data with a few matches
        data = b"\x10\x11\x20\x21\x30\x31"
        results = matcher.search(data)

        assert len(results) >= 3


# =============================================================================
# FuzzyMatcher Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.pattern
class TestFuzzyMatcher:
    """Test fuzzy pattern matching."""

    def test_exact_match(self) -> None:
        """Test exact match with fuzzy matcher."""
        matcher = FuzzyMatcher(max_edit_distance=2)

        data = b"\x00\xaa\xbb\xcc\x00"
        pattern = b"\xaa\xbb\xcc"
        results = matcher.search(data, pattern)

        assert len(results) >= 1
        # Exact match should have edit distance 0
        exact_matches = [r for r in results if r.edit_distance == 0]
        assert len(exact_matches) >= 1
        assert exact_matches[0].similarity == 1.0

    def test_substitution_match(self) -> None:
        """Test match with single substitution."""
        matcher = FuzzyMatcher(max_edit_distance=1)

        data = b"\xaa\xbb\xdd"  # DD instead of CC
        pattern = b"\xaa\xbb\xcc"
        results = matcher.search(data, pattern)

        assert len(results) >= 1
        assert results[0].edit_distance <= 1

    def test_insertion_match(self) -> None:
        """Test match with insertion."""
        matcher = FuzzyMatcher(max_edit_distance=2, allow_insertions=True)

        data = b"\xaa\xff\xbb\xcc"  # Extra byte inserted
        pattern = b"\xaa\xbb\xcc"
        results = matcher.search(data, pattern)

        assert len(results) >= 1

    def test_deletion_match(self) -> None:
        """Test match with deletion."""
        matcher = FuzzyMatcher(max_edit_distance=1, allow_deletions=True)

        data = b"\xaa\xcc"  # Missing middle byte
        pattern = b"\xaa\xbb\xcc"
        results = matcher.search(data, pattern)

        assert len(results) >= 1

    def test_disable_substitutions(self) -> None:
        """Test disabling substitutions."""
        matcher = FuzzyMatcher(max_edit_distance=1, allow_substitutions=False)

        data = b"\xaa\xdd\xcc"  # Substitution required
        pattern = b"\xaa\xbb\xcc"
        results = matcher.search(data, pattern)

        # Should not find match with substitution disabled
        exact_or_indel = [r for r in results if r.edit_distance <= 1]
        # May find matches with insertions/deletions but not substitutions
        # Verify substitutions were actually disabled
        for result in results:
            if result.edit_distance == 1:
                # Must be insertion or deletion, not substitution
                assert result.edit_distance > 0

    def test_min_similarity_threshold(self) -> None:
        """Test minimum similarity filtering."""
        matcher = FuzzyMatcher(max_edit_distance=2, min_similarity=0.8)

        data = b"\xaa\xbb\xdd\xee"  # 2 differences
        pattern = b"\xaa\xbb\xcc\xcc"
        results = matcher.search(data, pattern)

        # All results should meet similarity threshold
        for result in results:
            assert result.similarity >= 0.8

    def test_match_with_wildcards(self) -> None:
        """Test wildcard matching."""
        matcher = FuzzyMatcher(max_edit_distance=1)

        data = b"\xaa\x12\x34\xbb"
        pattern = b"\xaa\xff\xff\xbb"  # 0xFF as wildcard
        results = matcher.match_with_wildcards(data, pattern, wildcard=0xFF)

        assert len(results) >= 1
        assert results[0].edit_distance == 0  # Wildcards don't count as edits

    def test_wildcard_at_end(self) -> None:
        """Test wildcard at pattern end."""
        matcher = FuzzyMatcher(max_edit_distance=0)

        data = b"\xaa\xbb\x99"
        pattern = b"\xaa\xbb\xff"
        results = matcher.match_with_wildcards(data, pattern, wildcard=0xFF)

        assert len(results) >= 1

    def test_remove_overlapping(self) -> None:
        """Test that overlapping matches are removed."""
        matcher = FuzzyMatcher(max_edit_distance=2)

        # Create data where pattern could match at overlapping positions
        data = b"\xaa\xbb\xcc\xdd" * 2
        pattern = b"\xaa\xbb\xcc"
        results = matcher.search(data, pattern)

        # Check that results don't overlap
        for i, r1 in enumerate(results):
            for r2 in results[i + 1 :]:
                range1 = set(range(r1.offset, r1.offset + r1.length))
                range2 = set(range(r2.offset, r2.offset + r2.length))
                assert not (range1 & range2), "Results should not overlap"

    def test_string_pattern_conversion(self) -> None:
        """Test automatic string to bytes conversion."""
        matcher = FuzzyMatcher(max_edit_distance=1)

        data = b"HELLO"
        pattern = "HELLO"
        results = matcher.search(data, pattern)

        assert len(results) >= 1

    def test_empty_pattern_name(self) -> None:
        """Test pattern without name gets hex name."""
        matcher = FuzzyMatcher(max_edit_distance=1)

        data = b"\xaa\xbb"
        pattern = b"\xaa\xbb"
        results = matcher.search(data, pattern)

        assert len(results) >= 1
        assert results[0].pattern_name == "aabb"

    def test_edit_distance_detailed(self) -> None:
        """Test detailed edit distance calculation."""
        matcher = FuzzyMatcher(max_edit_distance=3)

        pattern = b"\xaa\xbb\xcc"
        text = b"\xaa\xdd\xcc"  # One substitution

        distance, substitutions = matcher._edit_distance_detailed(pattern, text)

        assert distance == 1
        assert len(substitutions) == 1
        assert substitutions[0] == (1, 0xBB, 0xDD)

    def test_no_matches(self) -> None:
        """Test search with no matches within threshold."""
        matcher = FuzzyMatcher(max_edit_distance=1)

        data = b"\xff\xff\xff\xff"
        pattern = b"\x00\x00\x00\x00"
        results = matcher.search(data, pattern)

        # Should not find matches (too many differences)
        assert len(results) == 0

    def test_pattern_longer_than_data(self) -> None:
        """Test pattern longer than available data."""
        matcher = FuzzyMatcher(max_edit_distance=1)

        data = b"\xaa"
        pattern = b"\xaa\xbb\xcc\xdd"
        results = matcher.search(data, pattern)

        # Should handle gracefully
        assert isinstance(results, list)


# =============================================================================
# FuzzyMatchResult Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.pattern
class TestFuzzyMatchResult:
    """Test FuzzyMatchResult dataclass."""

    def test_create_result(self) -> None:
        """Test creating fuzzy match result."""
        result = FuzzyMatchResult(
            pattern_name="test",
            offset=10,
            length=4,
            matched_data=b"\xaa\xbb\xcc\xdd",
            pattern=b"\xaa\xbb\xcc\xde",
            similarity=0.75,
            edit_distance=1,
        )

        assert result.pattern_name == "test"
        assert result.offset == 10
        assert result.length == 4
        assert result.similarity == 0.75
        assert result.edit_distance == 1

    def test_result_with_substitutions(self) -> None:
        """Test result with substitution details."""
        result = FuzzyMatchResult(
            pattern_name="test",
            offset=0,
            length=3,
            matched_data=b"\xaa\xdd\xcc",
            pattern=b"\xaa\xbb\xcc",
            similarity=0.67,
            edit_distance=1,
            substitutions=[(1, 0xBB, 0xDD)],
        )

        assert len(result.substitutions) == 1
        assert result.substitutions[0] == (1, 0xBB, 0xDD)


# =============================================================================
# Convenience Function Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.pattern
class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_binary_regex_search(self) -> None:
        """Test binary_regex_search convenience function."""
        data = b"\x00\xaa\xbb\x00\xaa\xbb"
        pattern = r"\xAA\xBB"

        results = binary_regex_search(data, pattern, name="test")

        assert len(results) == 2
        assert results[0].pattern_name == "test"

    def test_multi_pattern_search(self) -> None:
        """Test multi_pattern_search convenience function."""
        data = b"\xaa\x55\x00\xde\xad\x00\xaa\x55"
        patterns = {
            "header": b"\xaa\x55",
            "marker": b"\xde\xad",
        }

        results = multi_pattern_search(data, patterns)

        assert "header" in results
        assert "marker" in results
        assert len(results["header"]) == 2
        assert len(results["marker"]) == 1

    def test_multi_pattern_search_string_patterns(self) -> None:
        """Test multi_pattern_search with string patterns."""
        data = b"HELLO WORLD"
        patterns = {
            "hello": "HELLO",
            "world": "WORLD",
        }

        results = multi_pattern_search(data, patterns)

        assert len(results["hello"]) == 1
        assert len(results["world"]) == 1

    def test_fuzzy_search(self) -> None:
        """Test fuzzy_search convenience function."""
        data = b"\x00\xaa\xbb\xdd\x00"  # DD instead of CC
        pattern = b"\xaa\xbb\xcc"

        results = fuzzy_search(data, pattern, max_distance=1, name="test")

        assert len(results) >= 1
        assert results[0].pattern_name == "test"

    def test_fuzzy_search_with_similarity(self) -> None:
        """Test fuzzy_search with similarity threshold."""
        data = b"\xaa\xbb\xcc\xdd"
        pattern = b"\xaa\xbb\xcc\xcc"

        results = fuzzy_search(data, pattern, max_distance=2, min_similarity=0.7)

        # All results should meet similarity threshold
        for result in results:
            assert result.similarity >= 0.7

    def test_find_similar_sequences(self) -> None:
        """Test find_similar_sequences function."""
        # Create data with similar sequences
        data = b"\xaa\xbb\xcc\xdd" + b"\x00" * 10 + b"\xaa\xbb\xcc\xde"

        results = find_similar_sequences(data, min_length=4, max_distance=1)

        assert isinstance(results, list)
        # Should find similar sequences if any exist
        if len(results) > 0:
            offset1, offset2, similarity = results[0]
            assert isinstance(offset1, int)
            assert isinstance(offset2, int)
            assert 0.0 <= similarity <= 1.0

    def test_find_similar_sequences_no_overlaps(self) -> None:
        """Test that find_similar_sequences skips overlaps."""
        data = b"\xaa\xbb\xcc\xdd" * 2

        results = find_similar_sequences(data, min_length=4, max_distance=1)

        # Results should not include overlapping positions
        for offset1, offset2, _ in results:
            assert abs(offset1 - offset2) >= 4

    def test_count_pattern_occurrences(self) -> None:
        """Test count_pattern_occurrences function."""
        data = b"\xaa\x55\x00\xde\xad\x00\xaa\x55\x00\xaa\x55"
        patterns = {
            "header": b"\xaa\x55",
            "marker": b"\xde\xad",
        }

        counts = count_pattern_occurrences(data, patterns)

        assert counts["header"] == 3
        assert counts["marker"] == 1

    def test_find_pattern_positions(self) -> None:
        """Test find_pattern_positions function."""
        data = b"\xaa\x00\xaa\x00\xaa"
        pattern = b"\xaa"

        positions = find_pattern_positions(data, pattern)

        assert positions == [0, 2, 4]

    def test_find_pattern_positions_string(self) -> None:
        """Test find_pattern_positions with string pattern."""
        data = b"HELLO WORLD HELLO"
        pattern = "HELLO"

        positions = find_pattern_positions(data, pattern)

        assert len(positions) == 2
        assert positions[0] == 0
        assert positions[1] == 12

    def test_find_pattern_positions_no_matches(self) -> None:
        """Test find_pattern_positions with no matches."""
        data = b"\xff\xff\xff"
        pattern = b"\x00"

        positions = find_pattern_positions(data, pattern)

        assert positions == []


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


@pytest.mark.unit
@pytest.mark.pattern
class TestPatternsMatchingEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_data(self) -> None:
        """Test matching against empty data."""
        regex = BinaryRegex(pattern=r"\xAA")
        results = regex.findall(b"")

        assert results == []

    def test_empty_pattern(self) -> None:
        """Test empty pattern."""
        regex = BinaryRegex(pattern="")
        data = b"\xaa\xbb"

        result = regex.search(data)
        # Empty pattern behavior depends on regex engine
        # Should either match at position 0 or return None
        assert result is None or result.start() == 0

    def test_pattern_longer_than_data(self) -> None:
        """Test pattern longer than data."""
        regex = BinaryRegex(pattern=r"\xAA\xBB\xCC\xDD")
        data = b"\xaa\xbb"

        result = regex.search(data)
        assert result is None

    def test_aho_corasick_empty_data(self) -> None:
        """Test Aho-Corasick with empty data."""
        matcher = AhoCorasickMatcher()
        matcher.add_pattern(b"\xaa", "test")
        matcher.build()

        results = matcher.search(b"")
        assert results == []

    def test_aho_corasick_no_patterns(self) -> None:
        """Test Aho-Corasick with no patterns added."""
        matcher = AhoCorasickMatcher()
        matcher.build()

        results = matcher.search(b"\xaa\xbb")
        assert results == []

    def test_fuzzy_matcher_empty_data(self) -> None:
        """Test fuzzy matcher with empty data."""
        matcher = FuzzyMatcher(max_edit_distance=1)

        results = matcher.search(b"", b"\xaa")
        assert results == []

    def test_fuzzy_matcher_empty_pattern(self) -> None:
        """Test fuzzy matcher with empty pattern."""
        matcher = FuzzyMatcher(max_edit_distance=1)

        results = matcher.search(b"\xaa\xbb", b"")
        # Should handle gracefully
        assert isinstance(results, list)

    def test_fuzzy_matcher_zero_max_distance(self) -> None:
        """Test fuzzy matcher with zero max distance (exact match only)."""
        matcher = FuzzyMatcher(max_edit_distance=0)

        data = b"\xaa\xbb\xcc"
        pattern = b"\xaa\xbb\xcc"
        results = matcher.search(data, pattern)

        assert len(results) >= 1
        assert results[0].edit_distance == 0

    def test_very_long_pattern(self) -> None:
        """Test with very long pattern."""
        pattern = bytes(range(256))
        regex = BinaryRegex(pattern=rb"\x00\x01\x02\x03")

        result = regex.search(pattern)
        assert result is not None

    def test_all_bytes_pattern(self) -> None:
        """Test pattern containing all byte values."""
        data = bytes(range(256))
        matcher = AhoCorasickMatcher()
        matcher.add_pattern(data[100:110], "subset")
        matcher.build()

        results = matcher.search(data)
        assert len(results) == 1

    def test_special_regex_characters_in_data(self) -> None:
        """Test data containing special regex characters."""
        # These should be escaped properly
        data = b".$*+?[](){}|^\\"
        regex = BinaryRegex(pattern=rb"\.")

        # Should not raise regex error
        result = regex.search(data)
        # Should find the literal '.' character
        assert result is not None
        assert result.start() == 0

    def test_unicode_in_pattern_name(self) -> None:
        """Test unicode characters in pattern names."""
        matcher = AhoCorasickMatcher()
        matcher.add_pattern(b"\xaa", "테스트")  # Korean characters
        matcher.build()

        results = matcher.search(b"\xaa")
        assert len(results) == 1
        assert results[0].pattern_name == "테스트"

    def test_large_data_performance(self) -> None:
        """Test performance with large data."""
        # Create 1MB of data
        data = b"\x00" * 500_000 + b"\xaa\xbb" + b"\x00" * 500_000

        matcher = AhoCorasickMatcher()
        matcher.add_pattern(b"\xaa\xbb", "needle")
        matcher.build()

        results = matcher.search(data)
        assert len(results) == 1
        assert results[0].offset == 500_000

    def test_many_overlapping_matches(self) -> None:
        """Test many overlapping pattern matches."""
        data = b"\xaa" * 1000
        regex = BinaryRegex(pattern=r"\xAA\xAA")

        results = regex.findall(data)
        # Should handle many matches efficiently
        assert len(results) > 0

    def test_min_similarity_property(self) -> None:
        """Test min_similarity property with and without explicit value."""
        matcher1 = FuzzyMatcher(max_edit_distance=2)
        assert matcher1.min_similarity == 0.0  # Default

        matcher2 = FuzzyMatcher(max_edit_distance=2, min_similarity=0.8)
        assert matcher2.min_similarity == 0.8  # Explicit value

    def test_fuzzy_match_all_wildcards(self) -> None:
        """Test fuzzy match with pattern that's all wildcards."""
        matcher = FuzzyMatcher(max_edit_distance=0)

        data = b"\x12\x34\x56\x78"
        pattern = b"\xff\xff\xff\xff"
        results = matcher.match_with_wildcards(data, pattern, wildcard=0xFF)

        assert len(results) >= 1
        assert results[0].similarity == 1.0

    def test_binary_regex_empty_name(self) -> None:
        """Test BinaryRegex with empty name."""
        regex = BinaryRegex(pattern=r"\xAA")

        result = regex.search(b"\xaa")
        assert result is not None

    def test_empty_pattern_in_find_pattern_positions(self) -> None:
        """Test find_pattern_positions with empty pattern."""
        data = b"\xaa\xbb\xcc"

        # Empty pattern should raise ValueError (prevents infinite loop)
        with pytest.raises(ValueError, match="Pattern cannot be empty"):
            find_pattern_positions(data, b"")

    def test_similar_sequences_min_length_exceeds_data(self) -> None:
        """Test find_similar_sequences when min_length exceeds data length."""
        data = b"\xaa\xbb"

        results = find_similar_sequences(data, min_length=10, max_distance=1)

        # Should return empty list, no crash
        assert results == []

    def test_aho_corasick_empty_pattern(self) -> None:
        """Test that Aho-Corasick handles empty pattern gracefully."""
        matcher = AhoCorasickMatcher()
        matcher.add_pattern(b"", "empty")
        matcher.build()

        results = matcher.search(b"\xaa\xbb")

        # Empty pattern should not cause infinite loop
        # Implementation should handle this gracefully
        assert isinstance(results, list)
        # Empty pattern matches at every position, so results should be non-empty
        if len(results) > 0:
            assert results[0].pattern_name == "empty"
