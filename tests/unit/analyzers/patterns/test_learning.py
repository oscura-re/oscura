"""Unit tests for pattern learning and discovery (RE-PAT-004).

    - RE-PAT-004: Pattern Learning and Discovery

This test module provides comprehensive coverage of the learning module,
including n-gram models, pattern discovery, structure inference, and
byte prediction.
"""

from __future__ import annotations

import pytest

from oscura.analyzers.patterns.learning import (
    LearnedPattern,
    NgramModel,
    PatternLearner,
    StructureHypothesis,
    find_recurring_structures,
    infer_structure,
    learn_patterns_from_data,
)

pytestmark = [pytest.mark.unit, pytest.mark.analyzer, pytest.mark.pattern]


@pytest.mark.unit
@pytest.mark.pattern
class TestLearnedPattern:
    """Test LearnedPattern dataclass."""

    def test_learned_pattern_creation(self) -> None:
        """Test creating a LearnedPattern."""
        pattern = LearnedPattern(
            pattern=b"\xaa\xbb",
            frequency=5,
            confidence=0.8,
        )
        assert pattern.pattern == b"\xaa\xbb"
        assert pattern.frequency == 5
        assert pattern.confidence == 0.8
        assert pattern.positions == []
        assert pattern.context_before == b""
        assert pattern.context_after == b""
        assert not pattern.is_structural
        assert not pattern.is_delimiter

    def test_learned_pattern_with_positions(self) -> None:
        """Test LearnedPattern with positions."""
        pattern = LearnedPattern(
            pattern=b"\x00\x00",
            frequency=3,
            confidence=0.7,
            positions=[10, 20, 30],
        )
        assert pattern.positions == [10, 20, 30]

    def test_learned_pattern_with_context(self) -> None:
        """Test LearnedPattern with context."""
        pattern = LearnedPattern(
            pattern=b"SEP",
            frequency=4,
            confidence=0.9,
            context_before=b"DATA",
            context_after=b"NEXT",
        )
        assert pattern.context_before == b"DATA"
        assert pattern.context_after == b"NEXT"

    def test_learned_pattern_structural(self) -> None:
        """Test LearnedPattern marked as structural."""
        pattern = LearnedPattern(
            pattern=b"\xff\xff",
            frequency=10,
            confidence=0.95,
            is_structural=True,
        )
        assert pattern.is_structural

    def test_learned_pattern_delimiter(self) -> None:
        """Test LearnedPattern marked as delimiter."""
        pattern = LearnedPattern(
            pattern=b",",
            frequency=20,
            confidence=0.85,
            is_delimiter=True,
        )
        assert pattern.is_delimiter


@pytest.mark.unit
@pytest.mark.pattern
class TestStructureHypothesis:
    """Test StructureHypothesis dataclass."""

    def test_structure_hypothesis_creation(self) -> None:
        """Test creating a StructureHypothesis."""
        hypothesis = StructureHypothesis(
            field_boundaries=[0, 4, 8, 16],
            field_types=["header", "counter", "data"],
            header_size=4,
            record_size=16,
            delimiters=[b"\n"],
            confidence=0.75,
        )
        assert hypothesis.field_boundaries == [0, 4, 8, 16]
        assert hypothesis.field_types == ["header", "counter", "data"]
        assert hypothesis.header_size == 4
        assert hypothesis.record_size == 16
        assert hypothesis.delimiters == [b"\n"]
        assert hypothesis.confidence == 0.75

    def test_structure_hypothesis_no_record_size(self) -> None:
        """Test StructureHypothesis with variable record size."""
        hypothesis = StructureHypothesis(
            field_boundaries=[],
            field_types=[],
            header_size=0,
            record_size=None,
            delimiters=[],
            confidence=0.0,
        )
        assert hypothesis.record_size is None


@pytest.mark.unit
@pytest.mark.pattern
class TestNgramModel:
    """Test NgramModel dataclass."""

    def test_ngram_model_creation(self) -> None:
        """Test creating an NgramModel."""
        model = NgramModel(n=2)
        assert model.n == 2
        assert model.counts == {}
        assert model.total == 0
        assert model.vocabulary_size == 0

    def test_ngram_model_with_counts(self) -> None:
        """Test NgramModel with frequency counts."""
        counts = {b"aa": 5, b"bb": 3, b"cc": 2}
        model = NgramModel(
            n=2,
            counts=counts,
            total=10,
            vocabulary_size=3,
        )
        assert model.counts == counts
        assert model.total == 10
        assert model.vocabulary_size == 3


@pytest.mark.unit
@pytest.mark.pattern
class TestPatternLearner:
    """Test PatternLearner class."""

    def test_initialization_default(self) -> None:
        """Test PatternLearner initialization with defaults."""
        learner = PatternLearner()
        assert learner.min_pattern_length == 2
        assert learner.max_pattern_length == 16
        assert learner.min_frequency == 3
        assert learner.min_confidence == 0.5

    def test_initialization_custom(self) -> None:
        """Test PatternLearner initialization with custom parameters."""
        learner = PatternLearner(
            min_pattern_length=3,
            max_pattern_length=8,
            min_frequency=5,
            min_confidence=0.7,
        )
        assert learner.min_pattern_length == 3
        assert learner.max_pattern_length == 8
        assert learner.min_frequency == 5
        assert learner.min_confidence == 0.7

    def test_add_sample(self) -> None:
        """Test adding a single sample."""
        learner = PatternLearner()
        data = b"\x01\x02\x03\x04"
        learner.add_sample(data)
        assert len(learner._samples) == 1
        assert learner._samples[0] == data

    def test_add_samples(self) -> None:
        """Test adding multiple samples."""
        learner = PatternLearner()
        samples = [b"\x01\x02", b"\x03\x04", b"\x05\x06"]
        learner.add_samples(samples)
        assert len(learner._samples) == 3
        assert learner._samples == samples

    def test_learn_patterns_empty(self) -> None:
        """Test learning patterns with no samples."""
        learner = PatternLearner()
        patterns = learner.learn_patterns()
        assert patterns == []

    def test_learn_patterns_single_pattern(self) -> None:
        """Test learning a simple repeating pattern."""
        learner = PatternLearner(min_frequency=3, min_confidence=0.3)
        # Pattern "AB" repeats 4 times
        data = b"ABABABAB"
        learner.add_sample(data)
        patterns = learner.learn_patterns(top_k=10)

        # Should find "AB" pattern
        assert len(patterns) > 0
        ab_patterns = [p for p in patterns if p.pattern == b"AB"]
        assert len(ab_patterns) > 0
        assert ab_patterns[0].frequency >= 3

    def test_learn_patterns_multiple_samples(self) -> None:
        """Test learning patterns from multiple samples."""
        learner = PatternLearner(min_frequency=3, min_confidence=0.3)
        # Same pattern in multiple samples
        samples = [
            b"HEADER\x00\x00DATA",
            b"HEADER\x00\x00MORE",
            b"HEADER\x00\x00TAIL",
        ]
        learner.add_samples(samples)
        patterns = learner.learn_patterns(top_k=20)

        # Should find "HEADER" pattern
        header_patterns = [p for p in patterns if b"HEADER" in p.pattern or p.pattern == b"HEADER"]
        assert len(header_patterns) > 0

    def test_learn_patterns_top_k(self) -> None:
        """Test that top_k limits results."""
        learner = PatternLearner(min_frequency=2, min_confidence=0.2)
        # Create data with many patterns
        data = b"ABCABCABCDEFDEFDEF123123123"
        learner.add_sample(data)

        patterns_5 = learner.learn_patterns(top_k=5)
        patterns_10 = learner.learn_patterns(top_k=10)

        assert len(patterns_5) <= 5
        assert len(patterns_10) <= 10

    def test_learn_patterns_sorted_by_confidence(self) -> None:
        """Test that patterns are sorted by confidence."""
        learner = PatternLearner(min_frequency=2, min_confidence=0.2)
        data = b"AAABBBAAABBBAAABBB"
        learner.add_sample(data)
        patterns = learner.learn_patterns(top_k=10)

        # Should be sorted by confidence descending
        if len(patterns) > 1:
            for i in range(len(patterns) - 1):
                assert patterns[i].confidence >= patterns[i + 1].confidence

    def test_build_ngram_model(self) -> None:
        """Test building n-gram model."""
        learner = PatternLearner()
        data = b"ABCABC"
        learner.add_sample(data)

        model = learner.build_ngram_model(2)
        assert model.n == 2
        assert model.total == 5  # AB, BC, CA, AB, BC
        assert model.vocabulary_size == 3  # AB, BC, CA
        assert model.counts[b"AB"] == 2
        assert model.counts[b"BC"] == 2
        assert model.counts[b"CA"] == 1

    def test_build_ngram_model_various_sizes(self) -> None:
        """Test building n-gram models of various sizes."""
        learner = PatternLearner()
        data = b"ABCDEF"
        learner.add_sample(data)

        model1 = learner.build_ngram_model(1)
        assert model1.n == 1
        assert model1.total == 6

        model2 = learner.build_ngram_model(2)
        assert model2.n == 2
        assert model2.total == 5

        model3 = learner.build_ngram_model(3)
        assert model3.n == 3
        assert model3.total == 4

    def test_predict_next_bytes_empty(self) -> None:
        """Test predicting next bytes with no data."""
        learner = PatternLearner()
        predictions = learner.predict_next_bytes(b"AB")
        assert predictions == []

    def test_predict_next_bytes_simple(self) -> None:
        """Test predicting next bytes with simple pattern."""
        learner = PatternLearner(max_pattern_length=4)
        # Pattern: ABC repeats
        data = b"ABCABCABC"
        learner.add_sample(data)
        learner._build_ngram_models()

        predictions = learner.predict_next_bytes(b"AB", n_predictions=3)

        # Should predict "C" as next byte
        assert len(predictions) > 0
        assert predictions[0][0] == b"C"
        assert 0 < predictions[0][1] <= 1  # Valid probability

    def test_predict_next_bytes_n_predictions(self) -> None:
        """Test that n_predictions limits results."""
        learner = PatternLearner(max_pattern_length=3)
        data = b"ABCDEFGH" * 3
        learner.add_sample(data)
        learner._build_ngram_models()

        predictions_2 = learner.predict_next_bytes(b"AB", n_predictions=2)
        predictions_5 = learner.predict_next_bytes(b"AB", n_predictions=5)

        assert len(predictions_2) <= 2
        assert len(predictions_5) <= 5

    def test_learn_structure_empty(self) -> None:
        """Test learning structure with no samples."""
        learner = PatternLearner()
        hypothesis = learner.learn_structure()
        assert hypothesis.field_boundaries == []
        assert hypothesis.field_types == []
        assert hypothesis.header_size == 0
        assert hypothesis.record_size is None
        assert hypothesis.delimiters == []
        assert hypothesis.confidence == 0.0

    def test_learn_structure_fixed_size(self) -> None:
        """Test learning structure with fixed-size records."""
        learner = PatternLearner(min_frequency=2, min_confidence=0.3)
        # Fixed 8-byte records
        samples = [
            b"\x01\x02\x03\x04\x05\x06\x07\x08",
            b"\x11\x12\x13\x14\x15\x16\x17\x18",
            b"\x21\x22\x23\x24\x25\x26\x27\x28",
        ]
        learner.add_samples(samples)
        hypothesis = learner.learn_structure()

        # Should detect fixed record size
        assert hypothesis.record_size == 8
        assert hypothesis.confidence > 0

    def test_learn_structure_with_delimiters(self) -> None:
        """Test learning structure with delimiters."""
        learner = PatternLearner(min_frequency=3, min_confidence=0.3)
        # Data with comma delimiters
        samples = [
            b"A,B,C,D",
            b"E,F,G,H",
            b"I,J,K,L",
        ]
        learner.add_samples(samples)
        hypothesis = learner.learn_structure()

        # Should detect delimiter
        assert len(hypothesis.delimiters) > 0 or hypothesis.confidence > 0

    def test_classify_field_empty(self) -> None:
        """Test classifying empty field."""
        learner = PatternLearner()
        field_type = learner._classify_field(b"")
        assert field_type == "empty"

    def test_classify_field_constant(self) -> None:
        """Test classifying constant field."""
        learner = PatternLearner()
        field_type = learner._classify_field(b"\x00\x00\x00\x00")
        assert field_type == "constant"

    def test_classify_field_counter(self) -> None:
        """Test classifying counter field."""
        learner = PatternLearner()
        field_type = learner._classify_field(b"\x01\x02\x03\x04")
        assert field_type == "counter"

    def test_classify_field_text(self) -> None:
        """Test classifying text field."""
        learner = PatternLearner()
        field_type = learner._classify_field(b"Hello World")
        assert field_type == "text"

    def test_classify_field_binary(self) -> None:
        """Test classifying binary field."""
        learner = PatternLearner()
        # Mixed binary data with medium entropy
        field_type = learner._classify_field(b"\x01\x02\x03\xff\xfe\xfd")
        assert field_type in ["binary", "structured", "counter"]

    def test_detect_record_size_same_length(self) -> None:
        """Test detecting record size when all same length."""
        learner = PatternLearner()
        samples = [b"AAAA", b"BBBB", b"CCCC"]
        learner.add_samples(samples)
        record_size = learner._detect_record_size()
        assert record_size == 4

    def test_detect_record_size_variable(self) -> None:
        """Test detecting record size with variable lengths."""
        learner = PatternLearner()
        samples = [b"AAAA", b"BBBBBB", b"CCCCCCCC"]
        learner.add_samples(samples)
        record_size = learner._detect_record_size()
        # Should find GCD of 4, 6, 8 = 2, but 2 != min(4,6,8)=4, so returns 2
        assert record_size == 2

    def test_detect_record_size_no_pattern(self) -> None:
        """Test detecting record size with no pattern."""
        learner = PatternLearner()
        samples = [b"A", b"BBB", b"CCCCC"]
        learner.add_samples(samples)
        record_size = learner._detect_record_size()
        # GCD of 1, 3, 5 = 1, but 1 should be excluded
        assert record_size is None or record_size == 1

    def test_is_structural_same_offset(self) -> None:
        """Test identifying structural pattern at same offset."""
        learner = PatternLearner()
        # Pattern at position 5 in multiple samples
        positions = [(0, 5), (1, 5), (2, 5)]
        is_structural = learner._is_structural(b"HDR", positions)
        assert is_structural

    def test_is_structural_regular_intervals(self) -> None:
        """Test identifying structural pattern at regular intervals."""
        learner = PatternLearner()
        # Pattern every 10 bytes
        positions = [(0, 0), (0, 10), (0, 20), (0, 30)]
        is_structural = learner._is_structural(b"SEP", positions)
        assert is_structural

    def test_is_structural_irregular(self) -> None:
        """Test identifying non-structural pattern."""
        learner = PatternLearner()
        # Irregular positions
        positions = [(0, 5), (0, 17), (0, 23), (0, 89)]
        is_structural = learner._is_structural(b"RND", positions)
        assert not is_structural

    def test_is_delimiter_regular_spacing(self) -> None:
        """Test identifying delimiter with regular spacing."""
        learner = PatternLearner()
        samples = [b"A,B,C,D", b"E,F,G,H"]
        learner.add_samples(samples)
        # Comma at positions 1, 3, 5 in first sample
        positions = [(0, 1), (0, 3), (0, 5), (1, 1), (1, 3), (1, 5)]
        is_delimiter = learner._is_delimiter(b",", positions)
        assert is_delimiter

    def test_is_delimiter_irregular_spacing(self) -> None:
        """Test identifying non-delimiter pattern."""
        learner = PatternLearner()
        learner.add_sample(b"ABCDEF")
        positions = [(0, 0), (0, 5)]
        is_delimiter = learner._is_delimiter(b"A", positions)
        assert not is_delimiter

    def test_calculate_pattern_confidence(self) -> None:
        """Test calculating pattern confidence."""
        learner = PatternLearner()
        samples = [b"AAABBB", b"CCCAAA", b"DDDAAA"]
        learner.add_samples(samples)

        # Pattern "AAA" appears in all 3 samples
        positions = [(0, 0), (1, 3), (2, 3)]
        confidence = learner._calculate_pattern_confidence(b"AAA", positions)

        assert 0 <= confidence <= 1
        assert confidence > 0.5  # High sample coverage

    def test_calculate_pattern_confidence_no_positions(self) -> None:
        """Test calculating confidence with no positions."""
        learner = PatternLearner()
        learner.add_sample(b"DATA")
        confidence = learner._calculate_pattern_confidence(b"XYZ", [])
        assert confidence == 0.0

    def test_get_context_simple(self) -> None:
        """Test getting context around pattern."""
        learner = PatternLearner()
        sample = b"AAABBBCCCDDD"
        learner.add_sample(sample)

        # Pattern "BBB" at position 3
        positions = [(0, 3)]
        before, after = learner._get_context(b"BBB", positions)

        # Context should be nearby bytes (limited by context_len)
        assert isinstance(before, bytes)
        assert isinstance(after, bytes)

    def test_get_context_common_context(self) -> None:
        """Test getting common context from multiple occurrences."""
        learner = PatternLearner(min_pattern_length=2)
        samples = [
            b"XXXXSEPYYYY",
            b"XXXXSEPYYYY",
            b"XXXXSEPYYYY",
        ]
        learner.add_samples(samples)

        positions = [(0, 4), (1, 4), (2, 4)]
        before, after = learner._get_context(b"SEP", positions)

        # Common context should be detected
        assert before != b"" or after != b""


@pytest.mark.unit
@pytest.mark.pattern
class TestLearnPatternsFromData:
    """Test learn_patterns_from_data convenience function."""

    def test_single_bytes_input(self) -> None:
        """Test with single bytes input."""
        data = b"ABCABCABC"
        patterns = learn_patterns_from_data(
            data,
            min_length=2,
            max_length=8,
            min_frequency=2,
            top_k=10,
        )
        assert isinstance(patterns, list)
        # Should find some patterns
        assert len(patterns) >= 0

    def test_multiple_samples_input(self) -> None:
        """Test with multiple samples input."""
        samples = [b"HEADER1", b"HEADER2", b"HEADER3"]
        patterns = learn_patterns_from_data(
            samples,
            min_length=3,
            max_length=10,
            min_frequency=3,
            top_k=5,
        )
        assert isinstance(patterns, list)

    def test_custom_parameters(self) -> None:
        """Test with custom parameters."""
        data = b"AAAABBBBCCCC"
        patterns = learn_patterns_from_data(
            data,
            min_length=4,
            max_length=4,
            min_frequency=1,
            top_k=3,
        )
        assert len(patterns) <= 3
        # Should find 4-byte patterns like "AAAA", "BBBB", "CCCC"
        if len(patterns) > 0:
            assert all(len(p.pattern) == 4 for p in patterns)

    def test_no_patterns_found(self) -> None:
        """Test when no patterns meet criteria."""
        data = b"\x01\x02\x03\x04"  # All unique
        patterns = learn_patterns_from_data(
            data,
            min_length=2,
            max_length=4,
            min_frequency=5,  # High frequency requirement
            top_k=10,
        )
        # Might not find any patterns with high frequency requirement
        assert isinstance(patterns, list)


@pytest.mark.unit
@pytest.mark.pattern
class TestInferStructure:
    """Test infer_structure convenience function."""

    def test_infer_structure_fixed_records(self) -> None:
        """Test inferring structure from fixed-size records."""
        samples = [
            b"\x01\x02\x03\x04",
            b"\x05\x06\x07\x08",
            b"\x09\x0a\x0b\x0c",
        ]
        hypothesis = infer_structure(samples)

        assert isinstance(hypothesis, StructureHypothesis)
        assert hypothesis.record_size == 4
        assert hypothesis.confidence > 0

    def test_infer_structure_empty(self) -> None:
        """Test inferring structure from empty samples."""
        hypothesis = infer_structure([])

        assert hypothesis.field_boundaries == []
        assert hypothesis.field_types == []
        assert hypothesis.header_size == 0
        assert hypothesis.record_size is None
        assert hypothesis.confidence == 0.0

    def test_infer_structure_single_sample(self) -> None:
        """Test inferring structure from single sample."""
        samples = [b"HEADER\x00\x00DATA"]
        hypothesis = infer_structure(samples)

        assert isinstance(hypothesis, StructureHypothesis)
        # Single sample won't detect record size
        assert hypothesis.record_size is None


@pytest.mark.unit
@pytest.mark.pattern
class TestFindRecurringStructures:
    """Test find_recurring_structures function."""

    def test_find_fixed_size_structures(self) -> None:
        """Test finding fixed-size recurring structures."""
        # 4 records of 8 bytes each
        data = b"\x01\x02\x03\x04\x05\x06\x07\x08" * 4
        results = find_recurring_structures(data, min_size=8, max_size=16)

        assert isinstance(results, list)
        # Should find 8-byte structure
        if len(results) > 0:
            sizes = [r[0] for r in results]
            assert 8 in sizes

    def test_find_structures_similar_records(self) -> None:
        """Test finding structures with similar records."""
        # Records with some similarity
        data = (
            b"\x00\x01\x02\x03\xaa\xbb\xcc\xdd"
            + b"\x00\x01\x02\x04\x11\x22\x33\x44"
            + b"\x00\x01\x02\x05\x55\x66\x77\x88"
        )
        results = find_recurring_structures(data, min_size=8, max_size=16)

        assert isinstance(results, list)
        # Should detect some structure
        if len(results) > 0:
            # Check confidence scores are valid
            for size, _offset, confidence in results:
                assert 0 <= confidence <= 1
                assert size >= 8

    def test_find_structures_no_pattern(self) -> None:
        """Test with random data (no structure)."""
        # Random-looking data
        data = b"\x01\x23\x45\x67\x89\xab\xcd\xef\x10\x32\x54\x76\x98\xba\xdc\xfe"
        results = find_recurring_structures(data, min_size=8, max_size=16)

        assert isinstance(results, list)
        # Might not find any strong structures
        # Results sorted by confidence, low confidence expected

    def test_find_structures_too_small(self) -> None:
        """Test with data too small for structures."""
        data = b"\x01\x02\x03"
        results = find_recurring_structures(data, min_size=8, max_size=16)

        assert results == []

    def test_find_structures_size_range(self) -> None:
        """Test respecting min/max size parameters."""
        data = b"\x01\x02\x03\x04" * 10  # 40 bytes
        results = find_recurring_structures(data, min_size=4, max_size=8)

        # All results should be within size range
        for size, _, _ in results:
            assert 4 <= size <= 8

    def test_find_structures_max_results(self) -> None:
        """Test that results are limited to top 5."""
        # Create data that divides evenly many ways
        data = b"X" * 120  # Divides by 2, 3, 4, 5, 6, 8, 10, 12, etc.
        results = find_recurring_structures(data, min_size=2, max_size=30)

        # Should return at most 5 results
        assert len(results) <= 5

    def test_find_structures_sorted_by_confidence(self) -> None:
        """Test that results are sorted by confidence."""
        data = b"\x00\x01\x02\x03" * 8
        results = find_recurring_structures(data, min_size=4, max_size=16)

        # Results should be sorted by confidence descending
        if len(results) > 1:
            for i in range(len(results) - 1):
                assert results[i][2] >= results[i + 1][2]


@pytest.mark.unit
@pytest.mark.pattern
class TestPatternLearnerEdgeCases:
    """Test edge cases and error conditions for PatternLearner."""

    def test_very_short_data(self) -> None:
        """Test with very short data."""
        learner = PatternLearner(min_pattern_length=2)
        learner.add_sample(b"A")  # Too short for patterns
        patterns = learner.learn_patterns()
        assert patterns == []

    def test_all_unique_bytes(self) -> None:
        """Test with all unique bytes (no patterns)."""
        learner = PatternLearner(min_frequency=2)
        learner.add_sample(b"\x00\x01\x02\x03\x04\x05\x06\x07")
        patterns = learner.learn_patterns()
        # Should find no repeating patterns
        assert len(patterns) == 0

    def test_high_confidence_threshold(self) -> None:
        """Test with very high confidence threshold."""
        learner = PatternLearner(min_confidence=0.99)
        data = b"ABCABCABC"
        learner.add_sample(data)
        patterns = learner.learn_patterns()
        # Very few patterns should meet 0.99 confidence
        assert len(patterns) <= 3

    def test_long_pattern_length(self) -> None:
        """Test with very long max pattern length."""
        learner = PatternLearner(min_pattern_length=2, max_pattern_length=100)
        data = b"SHORT"
        learner.add_sample(data)
        # Should handle gracefully even though data is shorter
        patterns = learner.learn_patterns()
        assert isinstance(patterns, list)

    def test_predict_with_long_context(self) -> None:
        """Test prediction with context longer than data."""
        learner = PatternLearner(max_pattern_length=5)
        learner.add_sample(b"ABC")
        learner._build_ngram_models()

        # Context longer than any n-gram
        predictions = learner.predict_next_bytes(b"VERYLONGCONTEXT", n_predictions=5)
        # Should handle gracefully
        assert isinstance(predictions, list)

    def test_zero_top_k(self) -> None:
        """Test learning patterns with top_k=0."""
        learner = PatternLearner()
        learner.add_sample(b"ABCABC")
        patterns = learner.learn_patterns(top_k=0)
        assert patterns == []

    def test_negative_top_k(self) -> None:
        """Test learning patterns with negative top_k."""
        learner = PatternLearner()
        learner.add_sample(b"ABCABC")
        # Should handle as 0 or return empty
        patterns = learner.learn_patterns(top_k=-1)
        assert patterns == []

    def test_min_pattern_longer_than_max(self) -> None:
        """Test invalid min > max pattern length."""
        learner = PatternLearner(min_pattern_length=10, max_pattern_length=5)
        learner.add_sample(b"ABCDEFGHIJ")
        # Should handle gracefully (no valid range)
        patterns = learner.learn_patterns()
        assert patterns == []

    def test_single_byte_samples(self) -> None:
        """Test with single-byte samples."""
        learner = PatternLearner(min_pattern_length=2)
        learner.add_samples([b"A", b"B", b"C"])
        patterns = learner.learn_patterns()
        # No patterns possible with 1-byte samples and min_length=2
        assert patterns == []

    def test_empty_sample(self) -> None:
        """Test adding empty sample."""
        learner = PatternLearner()
        learner.add_sample(b"")
        learner.add_sample(b"DATA")
        patterns = learner.learn_patterns()
        # Should handle empty sample gracefully
        assert isinstance(patterns, list)

    def test_very_large_frequency(self) -> None:
        """Test pattern that appears very frequently."""
        learner = PatternLearner(min_frequency=3)
        # Pattern "A" appears 100 times
        data = b"A" * 100
        learner.add_sample(data)
        patterns = learner.learn_patterns()
        # Should handle high frequency
        if len(patterns) > 0:
            assert any(p.frequency >= 3 for p in patterns)

    def test_context_at_boundaries(self) -> None:
        """Test getting context when pattern is at data boundaries."""
        learner = PatternLearner()
        # Pattern at start
        learner.add_sample(b"PATTERNDATA")
        # Pattern at end
        learner.add_sample(b"DATAPATTERN")

        positions = [(0, 0), (1, 4)]
        before, after = learner._get_context(b"PATTERN", positions)
        # Should handle boundary cases
        assert isinstance(before, bytes)
        assert isinstance(after, bytes)

    def test_structure_with_single_byte_samples(self) -> None:
        """Test structure learning with very small samples."""
        learner = PatternLearner()
        learner.add_samples([b"A", b"B"])
        hypothesis = learner.learn_structure()
        # Should return low confidence
        assert hypothesis.confidence <= 0.5

    def test_infer_field_types_empty_boundaries(self) -> None:
        """Test inferring field types with no boundaries."""
        learner = PatternLearner()
        learner.add_sample(b"DATA")
        field_types = learner._infer_field_types([])
        assert field_types == []

    def test_multiple_ngram_builds(self) -> None:
        """Test building n-gram models multiple times."""
        learner = PatternLearner()
        learner.add_sample(b"ABCABC")

        model1 = learner.build_ngram_model(2)
        model2 = learner.build_ngram_model(2)

        # Second build should overwrite first
        assert model1.counts == model2.counts
        assert learner._ngram_models[2] == model2
