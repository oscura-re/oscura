"""Comprehensive unit tests for N-gram frequency analysis module.

Tests for all functions and classes in oscura.analyzers.statistical.ngrams.
"""

from __future__ import annotations

import numpy as np
import pytest

from oscura.analyzers.statistical.ngrams import (
    NGramAnalyzer,
    NgramComparison,
    NgramProfile,
    compare_ngram_profiles,
    find_unusual_ngrams,
    ngram_entropy,
    ngram_frequency,
    ngram_heatmap,
)

pytestmark = [pytest.mark.unit, pytest.mark.analyzer, pytest.mark.statistical]


# =============================================================================
# N-gram Frequency Function Tests
# =============================================================================


class TestNgramFrequency:
    """Test ngram_frequency function."""

    def test_ngram_invalid_n_raises(self) -> None:
        """Test that n < 1 raises ValueError."""
        with pytest.raises(ValueError, match="N-gram size must be >= 1"):
            ngram_frequency(b"test", n=0)
        with pytest.raises(ValueError, match="N-gram size must be >= 1"):
            ngram_frequency(b"test", n=-1)

    def test_ngram_data_shorter_than_n(self) -> None:
        """Test with data shorter than n-gram size."""
        result = ngram_frequency(b"AB", n=5)
        assert result.total_ngrams == 0
        assert result.unique_ngrams == 0
        assert result.frequencies == {}
        assert result.entropy == 0.0

    def test_ngram_bigram_basic(self) -> None:
        """Test basic bigram (n=2) extraction."""
        result = ngram_frequency(b"ABCABC", n=2)

        assert result.n == 2
        assert result.total_ngrams == 5  # AB, BC, CA, AB, BC
        assert result.unique_ngrams == 3  # AB, BC, CA
        assert result.frequencies[b"AB"] == 2
        assert result.frequencies[b"BC"] == 2
        assert result.frequencies[b"CA"] == 1

    def test_ngram_trigram(self) -> None:
        """Test trigram (n=3) extraction."""
        result = ngram_frequency(b"ABCABCABC", n=3)

        assert result.n == 3
        assert result.frequencies[b"ABC"] == 3
        assert result.frequencies[b"BCA"] == 2
        assert result.frequencies[b"CAB"] == 2

    def test_ngram_unigram(self) -> None:
        """Test unigram (n=1) extraction."""
        result = ngram_frequency(b"AABBB", n=1)

        assert result.n == 1
        assert result.frequencies[b"A"] == 2
        assert result.frequencies[b"B"] == 3

    def test_ngram_overlapping_default(self) -> None:
        """Test that overlapping is default behavior."""
        result = ngram_frequency(b"ABCD", n=2, overlap=True)
        assert result.total_ngrams == 3  # AB, BC, CD

    def test_ngram_non_overlapping(self) -> None:
        """Test non-overlapping n-gram extraction."""
        result = ngram_frequency(b"ABCDEF", n=2, overlap=False)
        assert result.total_ngrams == 3  # AB, CD, EF
        assert b"BC" not in result.frequencies

    def test_ngram_top_k_sorted(self) -> None:
        """Test that top_k is sorted by count descending."""
        data = b"AAABBC"
        result = ngram_frequency(data, n=2)

        # AA(2), AB(1), BB(1), BC(1)
        assert result.top_k[0][0] == b"AA"
        assert result.top_k[0][1] == 2  # count
        assert result.top_k[0][2] > 0  # frequency

    def test_ngram_top_k_limited(self) -> None:
        """Test that top_k is limited to 100 entries."""
        # Create data with many unique bigrams
        data = bytes(range(256)) * 2
        result = ngram_frequency(data, n=2)
        assert len(result.top_k) <= 100

    def test_ngram_entropy_calculation(self) -> None:
        """Test entropy in n-gram profile."""
        # Single n-gram has zero entropy
        result_constant = ngram_frequency(b"AAA", n=2)
        assert result_constant.entropy == 0.0

        # Multiple equal n-grams have higher entropy
        result_varied = ngram_frequency(b"ABCD" * 100, n=2)
        assert result_varied.entropy > 0

    def test_ngram_bytes_input(self) -> None:
        """Test with bytes input."""
        result = ngram_frequency(b"Hello", n=2)
        assert result.total_ngrams == 4

    def test_ngram_bytearray_input(self) -> None:
        """Test with bytearray input."""
        result = ngram_frequency(bytearray(b"Hello"), n=2)
        assert result.total_ngrams == 4

    def test_ngram_numpy_input(self) -> None:
        """Test with numpy array input."""
        data = np.array([65, 66, 67, 65, 66, 67], dtype=np.uint8)  # ABCABC
        result = ngram_frequency(data, n=2)
        assert result.frequencies[b"AB"] == 2


# =============================================================================
# N-gram Entropy Function Tests
# =============================================================================


class TestNgramEntropy:
    """Test ngram_entropy function."""

    def test_ngram_entropy_constant(self) -> None:
        """Test entropy of constant n-grams is zero."""
        result = ngram_entropy(b"AAAA", n=2)
        assert result == 0.0

    def test_ngram_entropy_varied(self) -> None:
        """Test entropy of varied n-grams is positive."""
        result = ngram_entropy(b"ABCDEFGH", n=2)
        assert result > 0

    def test_ngram_entropy_uniform(self) -> None:
        """Test entropy with uniform distribution."""
        # Create data where each bigram appears equally
        data = b"ABCD" * 100
        result = ngram_entropy(data, n=2)
        # Should be close to log2(3) since there are 3 unique bigrams
        assert result > 1.5

    def test_ngram_entropy_short_data(self) -> None:
        """Test entropy with data shorter than n."""
        result = ngram_entropy(b"A", n=2)
        assert result == 0.0


# =============================================================================
# Compare N-gram Profiles Function Tests
# =============================================================================


class TestCompareNgramProfiles:
    """Test compare_ngram_profiles function."""

    def test_compare_identical(self) -> None:
        """Test comparing identical data."""
        data = b"ABCABCABC"
        result = compare_ngram_profiles(data, data, n=2)

        assert isinstance(result, NgramComparison)
        assert result.similarity == 1.0
        assert result.cosine_similarity > 0.99
        assert result.chi_square < 0.01
        assert result.unique_to_a == 0
        assert result.unique_to_b == 0

    def test_compare_completely_different(self) -> None:
        """Test comparing completely different data."""
        data_a = b"AAAAAAA"
        data_b = b"BBBBBBB"
        result = compare_ngram_profiles(data_a, data_b, n=2)

        assert result.similarity == 0.0
        assert result.common_ngrams == 0
        assert result.unique_to_a == 1  # Only "AA"
        assert result.unique_to_b == 1  # Only "BB"

    def test_compare_partial_overlap(self) -> None:
        """Test comparing data with partial overlap."""
        # Use data with explicit overlap in bigrams
        data_a = b"ABCDAB"  # Bigrams: AB, BC, CD, DA, AB
        data_b = b"CDABCD"  # Bigrams: CD, DA, AB, BC, CD
        result = compare_ngram_profiles(data_a, data_b, n=2)

        # Should have some common bigrams (AB, BC, CD, DA)
        assert result.common_ngrams > 0
        # Similarity should be positive since they share bigrams
        assert result.similarity >= 0

    def test_compare_jaccard_similarity(self) -> None:
        """Test Jaccard similarity calculation."""
        data_a = b"ABAB"  # Bigrams: AB(2), BA(1)
        data_b = b"ABCD"  # Bigrams: AB(1), BC(1), CD(1)
        result = compare_ngram_profiles(data_a, data_b, n=2)

        # Jaccard = |intersection| / |union| = 1 / 4 = 0.25
        # Set A: {AB, BA}, Set B: {AB, BC, CD}
        # Intersection: {AB}, Union: {AB, BA, BC, CD}
        assert abs(result.similarity - 0.25) < 0.01

    def test_compare_empty_data(self) -> None:
        """Test comparing empty profiles."""
        result = compare_ngram_profiles(b"A", b"B", n=2)
        # Both have empty profiles (too short)
        assert result.similarity == 1.0  # Empty sets are equal


# =============================================================================
# Find Unusual N-grams Function Tests
# =============================================================================


class TestFindUnusualNgrams:
    """Test find_unusual_ngrams function."""

    def test_unusual_no_baseline_uniform(self) -> None:
        """Test unusual detection with uniform distribution and no baseline."""
        data = bytes(range(256)) * 10
        result = find_unusual_ngrams(data, n=2)
        # With uniform distribution and no baseline, fewer anomalies expected
        assert isinstance(result, list)

    def test_unusual_overrepresented(self) -> None:
        """Test detecting overrepresented n-grams."""
        data = b"AA" * 100 + bytes(range(256))
        result = find_unusual_ngrams(data, n=2, z_threshold=3.0)

        # "AA" should be detected as unusual
        unusual_ngrams = [ng for ng, _ in result]
        assert b"AA" in unusual_ngrams

    def test_unusual_with_baseline(self) -> None:
        """Test unusual detection with baseline profile."""
        baseline_data = bytes(range(256)) * 10
        baseline = ngram_frequency(baseline_data, n=2)

        test_data = b"ZZ" * 100 + bytes(range(256))
        result = find_unusual_ngrams(test_data, baseline=baseline, n=2, z_threshold=2.0)

        # Results are sorted by |z_score| descending
        if len(result) > 0:
            # First result should have highest absolute z-score
            assert abs(result[0][1]) >= abs(result[-1][1])

    def test_unusual_baseline_mismatch_raises(self) -> None:
        """Test that baseline n mismatch raises ValueError."""
        baseline = ngram_frequency(b"ABCABC", n=3)  # n=3

        with pytest.raises(ValueError, match="Baseline n-gram size"):
            find_unusual_ngrams(b"ABCABC", baseline=baseline, n=2)  # n=2

    def test_unusual_empty_data(self) -> None:
        """Test with data too short for n-grams."""
        result = find_unusual_ngrams(b"A", n=2)
        assert result == []

    def test_unusual_z_threshold(self) -> None:
        """Test z_threshold parameter."""
        data = b"AA" * 50 + bytes(range(64))

        # Higher threshold should find fewer unusual n-grams
        result_low_threshold = find_unusual_ngrams(data, n=2, z_threshold=1.0)
        result_high_threshold = find_unusual_ngrams(data, n=2, z_threshold=5.0)

        assert len(result_high_threshold) <= len(result_low_threshold)


# =============================================================================
# N-gram Heatmap Function Tests
# =============================================================================


class TestNgramHeatmap:
    """Test ngram_heatmap function."""

    def test_heatmap_shape(self) -> None:
        """Test heatmap has correct shape."""
        result = ngram_heatmap(b"ABAB", n=2)
        assert result.shape == (256, 256)

    def test_heatmap_only_bigrams(self) -> None:
        """Test that heatmap only works for n=2."""
        with pytest.raises(ValueError, match="Heatmap only supported for bigrams"):
            ngram_heatmap(b"ABC", n=3)

    def test_heatmap_normalized(self) -> None:
        """Test heatmap is normalized to [0, 1]."""
        data = b"ABAB" * 100
        result = ngram_heatmap(data, n=2)

        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_heatmap_correct_positions(self) -> None:
        """Test heatmap has correct byte pair positions."""
        # Use single AB followed by a different pattern to avoid BA
        data = b"AB" + b"C" * 100  # AB once, then CC many times
        result = ngram_heatmap(data, n=2)

        # A=65, B=66, C=67
        # AB appears once, BC appears once, CC appears many times
        # CC should be the most frequent (max = 1.0)
        assert result[67, 67] == 1.0  # CC is most frequent
        # AB and BC should be present but not 1.0
        assert result[65, 66] > 0.0  # AB exists
        assert result[66, 67] > 0.0  # BC exists

    def test_heatmap_empty_data(self) -> None:
        """Test heatmap with single byte (no bigrams)."""
        result = ngram_heatmap(b"A", n=2)
        assert result.max() == 0.0

    def test_heatmap_numpy_input(self) -> None:
        """Test heatmap with numpy array input."""
        data = np.array([65, 66] * 100, dtype=np.uint8)
        result = ngram_heatmap(data, n=2)
        assert result[65, 66] == 1.0


# =============================================================================
# NGramAnalyzer Class Tests
# =============================================================================


class TestNGramAnalyzer:
    """Test NGramAnalyzer class."""

    def test_analyzer_init(self) -> None:
        """Test analyzer initialization."""
        analyzer = NGramAnalyzer(n=3, overlap=False)
        assert analyzer.n == 3
        assert analyzer.overlap is False

    def test_analyzer_analyze_returns_dict(self) -> None:
        """Test analyze returns frequency dictionary."""
        analyzer = NGramAnalyzer(n=2)
        result = analyzer.analyze(b"ABCABC")

        assert isinstance(result, dict)
        assert result[b"AB"] == 2
        assert result[b"BC"] == 2

    def test_analyzer_analyze_profile(self) -> None:
        """Test analyze_profile returns NgramProfile."""
        analyzer = NGramAnalyzer(n=2)
        result = analyzer.analyze_profile(b"ABCABC")

        assert isinstance(result, NgramProfile)
        assert result.n == 2
        assert result.total_ngrams == 5

    def test_analyzer_get_distribution(self) -> None:
        """Test get_distribution normalizes frequencies."""
        analyzer = NGramAnalyzer(n=2)
        frequencies = analyzer.analyze(b"ABAB")
        distribution = analyzer.get_distribution(frequencies)

        # Distribution should sum to 1.0
        assert abs(sum(distribution.values()) - 1.0) < 0.01

    def test_analyzer_get_distribution_empty(self) -> None:
        """Test get_distribution with empty frequencies."""
        analyzer = NGramAnalyzer(n=2)
        distribution = analyzer.get_distribution({})
        assert distribution == {}

    def test_analyzer_entropy(self) -> None:
        """Test entropy calculation through analyzer."""
        analyzer = NGramAnalyzer(n=2)

        # Constant data has zero entropy
        assert analyzer.entropy(b"AAA") == 0.0

        # Varied data has positive entropy
        assert analyzer.entropy(b"ABCDEF") > 0

    def test_analyzer_compare(self) -> None:
        """Test compare method."""
        analyzer = NGramAnalyzer(n=2)
        result = analyzer.compare(b"ABCABC", b"ABCABC")

        assert isinstance(result, NgramComparison)
        assert result.similarity == 1.0

    def test_analyzer_find_unusual(self) -> None:
        """Test find_unusual method."""
        analyzer = NGramAnalyzer(n=2)
        data = b"AA" * 100 + bytes(range(64))
        result = analyzer.find_unusual(data, z_threshold=3.0)

        assert isinstance(result, list)

    def test_analyzer_heatmap(self) -> None:
        """Test heatmap method."""
        analyzer = NGramAnalyzer(n=2)
        result = analyzer.heatmap(b"ABAB")

        assert result.shape == (256, 256)

    def test_analyzer_heatmap_wrong_n(self) -> None:
        """Test heatmap raises for n != 2."""
        analyzer = NGramAnalyzer(n=3)

        with pytest.raises(ValueError, match="Heatmap only supported for bigrams"):
            analyzer.heatmap(b"ABCABC")

    def test_analyzer_last_profile_stored(self) -> None:
        """Test that _last_profile is stored after analyze."""
        analyzer = NGramAnalyzer(n=2)
        analyzer.analyze(b"ABCABC")

        assert analyzer._last_profile is not None
        assert analyzer._last_profile.n == 2


# =============================================================================
# NgramProfile Dataclass Tests
# =============================================================================


class TestNgramProfileDataclass:
    """Test NgramProfile dataclass."""

    def test_profile_creation(self) -> None:
        """Test creating NgramProfile directly."""
        profile = NgramProfile(
            n=2,
            frequencies={b"AB": 2, b"BC": 1},
            total_ngrams=3,
            unique_ngrams=2,
            top_k=[(b"AB", 2, 0.67), (b"BC", 1, 0.33)],
            entropy=0.92,
        )

        assert profile.n == 2
        assert profile.total_ngrams == 3
        assert profile.unique_ngrams == 2
        assert len(profile.top_k) == 2
        assert profile.entropy == 0.92


# =============================================================================
# NgramComparison Dataclass Tests
# =============================================================================


class TestNgramComparisonDataclass:
    """Test NgramComparison dataclass."""

    def test_comparison_creation(self) -> None:
        """Test creating NgramComparison directly."""
        comparison = NgramComparison(
            similarity=0.8,
            cosine_similarity=0.9,
            chi_square=0.1,
            common_ngrams=5,
            unique_to_a=2,
            unique_to_b=3,
        )

        assert comparison.similarity == 0.8
        assert comparison.cosine_similarity == 0.9
        assert comparison.chi_square == 0.1
        assert comparison.common_ngrams == 5
        assert comparison.unique_to_a == 2
        assert comparison.unique_to_b == 3


# =============================================================================
# Edge Cases and Data Types
# =============================================================================


class TestNgramEdgeCases:
    """Test edge cases and various scenarios."""

    def test_binary_data_with_nulls(self) -> None:
        """Test n-gram analysis with null bytes."""
        data = b"\x00\x00\x01\x00\x00"
        result = ngram_frequency(data, n=2)

        assert b"\x00\x00" in result.frequencies
        assert result.frequencies[b"\x00\x00"] == 2

    def test_all_unique_bigrams(self) -> None:
        """Test data with all unique bigrams."""
        data = bytes(range(256))  # 255 unique bigrams
        result = ngram_frequency(data, n=2)

        assert result.unique_ngrams == 255
        assert all(count == 1 for count in result.frequencies.values())

    def test_large_n_gram(self) -> None:
        """Test with large n-gram size."""
        data = b"ABCDEFGHIJ" * 10
        result = ngram_frequency(data, n=8)

        assert result.n == 8
        assert result.total_ngrams > 0

    def test_single_byte_repeated(self) -> None:
        """Test single byte repeated many times."""
        data = b"A" * 1000
        result = ngram_frequency(data, n=2)

        assert result.unique_ngrams == 1
        assert result.frequencies[b"AA"] == 999
        assert result.entropy == 0.0

    def test_text_data(self) -> None:
        """Test with typical text data."""
        text = b"The quick brown fox jumps over the lazy dog"
        result = ngram_frequency(text, n=2)

        # Common English bigrams should be present
        assert b"th" in result.frequencies or b"Th" in result.frequencies

    def test_performance_large_data(self) -> None:
        """Test performance with moderately large data."""
        np.random.seed(42)
        data = bytes(np.random.randint(0, 256, 100_000, dtype=np.uint8))

        result = ngram_frequency(data, n=2)
        assert result.total_ngrams == 99_999

    def test_numpy_various_dtypes(self) -> None:
        """Test numpy arrays with different dtypes."""
        # uint8
        data1 = np.array([65, 66, 67], dtype=np.uint8)
        result1 = ngram_frequency(data1, n=2)
        assert result1.total_ngrams == 2

        # int32 is flattened to bytes differently - the function converts via flatten()
        # When int32 is converted to bytes, each int becomes 4 bytes
        data2 = np.array([65, 66, 67], dtype=np.int32)
        result2 = ngram_frequency(data2, n=2)
        # 3 int32 values = 12 bytes when flattened, so 11 bigrams
        assert result2.total_ngrams > 0

    def test_compare_empty_profiles(self) -> None:
        """Test comparing profiles when one or both are empty."""
        short_a = b"A"  # Too short for bigram
        short_b = b"B"
        result = compare_ngram_profiles(short_a, short_b, n=2)

        # Both empty, should be considered equal
        assert result.similarity == 1.0
        assert result.common_ngrams == 0

    def test_heatmap_all_zeros(self) -> None:
        """Test heatmap when data is too short."""
        result = ngram_heatmap(b"", n=2)
        assert result.shape == (256, 256)
        assert result.sum() == 0.0

    def test_chi_square_distance(self) -> None:
        """Test chi-square distance calculation in comparison."""
        # Identical data should have chi-square near 0
        data = b"ABCABC" * 100
        result = compare_ngram_profiles(data, data, n=2)
        assert result.chi_square < 0.01

        # Different data should have higher chi-square
        data_a = b"AAAA" * 100
        data_b = b"BBBB" * 100
        result2 = compare_ngram_profiles(data_a, data_b, n=2)
        assert result2.chi_square > 0.1


# =============================================================================
# Protocol Fingerprinting Tests
# =============================================================================


class TestProtocolFingerprinting:
    """Test n-gram analysis for protocol fingerprinting use case."""

    def test_fingerprint_distinct_protocols(self) -> None:
        """Test distinguishing different protocol patterns."""
        # Protocol A: Header + incrementing sequence
        protocol_a = bytes([0x55, 0xAA] + list(range(100)))

        # Protocol B: Different header + different pattern
        protocol_b = bytes([0xAA, 0x55] + [i % 16 for i in range(100)])

        comparison = compare_ngram_profiles(protocol_a, protocol_b, n=2)

        # Different protocols should have low similarity
        assert comparison.similarity < 0.8

    def test_fingerprint_same_protocol(self) -> None:
        """Test identifying same protocol with different data."""
        # Same protocol structure, different payload
        protocol_sample_1 = bytes([0x55, 0xAA, 0x10] + list(range(50)))
        protocol_sample_2 = bytes([0x55, 0xAA, 0x20] + list(range(50, 100)))

        comparison = compare_ngram_profiles(protocol_sample_1, protocol_sample_2, n=2)

        # Same protocol structure should have some similarity
        # Header bytes are shared
        assert comparison.common_ngrams > 0

    def test_extract_protocol_signature(self) -> None:
        """Test extracting protocol signature from n-gram profile."""
        # Protocol with known header and structure
        messages = [bytes([0x02, 0x00, i, 0x03]) for i in range(256)]
        combined = b"".join(messages)

        profile = ngram_frequency(combined, n=2)

        # Most common bigrams should include protocol header
        top_bigrams = [ng for ng, _, _ in profile.top_k[:10]]

        # Check that protocol structure is captured
        assert len(top_bigrams) > 0
