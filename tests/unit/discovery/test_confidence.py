"""Tests for confidence scoring.

Requirements tested:
"""

import pytest

from oscura.core.confidence import ConfidenceScore, calculate_confidence

pytestmark = pytest.mark.unit


class TestConfidenceScore:
    """Tests for ConfidenceScore class."""

    def test_valid_confidence_creation(self):
        """Test creating valid confidence score."""
        score = ConfidenceScore(value=0.85)
        assert score.value == 0.85
        assert score.level == "medium"
        assert score.interpretation == "likely"

    def test_confidence_rounding(self):
        """Test confidence values are rounded to 2 decimals."""
        score = ConfidenceScore(value=0.8567)
        assert score.value == 0.86

    def test_invalid_confidence_value(self):
        """Test validation of confidence value range."""
        with pytest.raises(ValueError, match="Confidence value must be in"):
            ConfidenceScore(value=1.5)

        with pytest.raises(ValueError, match="Confidence value must be in"):
            ConfidenceScore(value=-0.1)

    def test_confidence_levels(self):
        """Test confidence level classification."""
        assert ConfidenceScore(0.95).level == "high"
        assert ConfidenceScore(0.85).level == "medium"
        assert ConfidenceScore(0.65).level == "low"
        assert ConfidenceScore(0.45).level == "unreliable"

    def test_confidence_interpretation(self):
        """Test human-readable interpretation."""
        assert ConfidenceScore(0.96).interpretation == "almost certain"
        assert ConfidenceScore(0.87).interpretation == "likely"
        assert ConfidenceScore(0.77).interpretation == "possible"
        assert ConfidenceScore(0.57).interpretation == "uncertain"
        assert ConfidenceScore(0.30).interpretation == "unlikely"

    def test_confidence_with_factors(self):
        """Test confidence with contributing factors."""
        score = ConfidenceScore(
            value=0.85,
            factors={"quality": 0.9, "timing": 0.8},
            explanation="Test explanation",
        )
        assert score.value == 0.85
        assert "quality" in score.factors
        assert score.explanation == "Test explanation"

    def test_invalid_factor_value(self):
        """Test validation of factor values."""
        with pytest.raises(ValueError, match=r"Factor .* must be in"):
            ConfidenceScore(value=0.85, factors={"bad": 1.5})

    def test_combine_scores_equal_weights(self):
        """Test combining scores with equal weights."""
        scores = [0.8, 0.9, 0.7]
        combined = ConfidenceScore.combine(scores)
        assert combined == 0.8  # Average

    def test_combine_scores_custom_weights(self):
        """Test combining scores with custom weights."""
        scores = [0.8, 0.9, 0.7]
        weights = [0.5, 0.3, 0.2]
        combined = ConfidenceScore.combine(scores, weights)
        expected = 0.8 * 0.5 + 0.9 * 0.3 + 0.7 * 0.2
        assert abs(combined - expected) < 0.01

    def test_combine_empty_scores(self):
        """Test combining empty score list raises error."""
        with pytest.raises(ValueError, match="Cannot combine empty"):
            ConfidenceScore.combine([])

    def test_combine_invalid_score(self):
        """Test combining with invalid score raises error."""
        with pytest.raises(ValueError, match="Score must be in"):
            ConfidenceScore.combine([0.8, 1.5, 0.7])

    def test_combine_mismatched_lengths(self):
        """Test combining with mismatched score/weight lengths."""
        with pytest.raises(ValueError, match="length mismatch"):
            ConfidenceScore.combine([0.8, 0.9], [0.5, 0.3, 0.2])

    def test_to_dict(self):
        """Test conversion to dictionary."""
        score = ConfidenceScore(0.85, factors={"test": 0.8})
        d = score.to_dict()
        assert d["value"] == 0.85
        assert d["level"] == "medium"
        assert d["interpretation"] == "likely"
        assert "test" in d["factors"]

    def test_float_conversion(self):
        """Test conversion to float."""
        score = ConfidenceScore(0.85)
        assert float(score) == 0.85


class TestCalculateConfidence:
    """Tests for calculate_confidence function."""

    def test_calculate_from_factors(self):
        """Test calculating confidence from multiple factors."""
        factors = {"quality": 0.9, "timing": 0.8, "pattern": 0.85}
        score = calculate_confidence(factors)
        assert isinstance(score, ConfidenceScore)
        assert 0.8 <= score.value <= 0.9

    def test_calculate_with_weights(self):
        """Test calculating confidence with weights."""
        factors = {"quality": 0.9, "timing": 0.8}
        weights = {"quality": 0.7, "timing": 0.3}
        score = calculate_confidence(factors, weights)
        expected = 0.9 * 0.7 + 0.8 * 0.3
        assert abs(score.value - expected) < 0.01

    def test_calculate_with_explanation(self):
        """Test calculating confidence with explanation."""
        factors = {"quality": 0.9}
        score = calculate_confidence(factors, explanation="Test explanation")
        assert score.explanation == "Test explanation"

    def test_calculate_empty_factors(self):
        """Test calculating with empty factors raises error."""
        with pytest.raises(ValueError, match="Cannot calculate confidence from empty"):
            calculate_confidence({})

    def test_calculate_missing_weight(self):
        """Test calculating with missing weight raises error."""
        factors = {"quality": 0.9, "timing": 0.8}
        weights = {"quality": 0.7}  # Missing 'timing'
        with pytest.raises(ValueError, match="Missing weight for factor"):
            calculate_confidence(factors, weights)
