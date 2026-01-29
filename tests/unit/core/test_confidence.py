"""Comprehensive unit tests for src/oscura/core/confidence.py

Tests coverage for:

Covers all public classes, functions, properties, and methods with
edge cases, error handling, and validation.
"""

import pytest

from oscura.core.confidence import ConfidenceScore, calculate_confidence

pytestmark = [pytest.mark.unit, pytest.mark.core]


# ==============================================================================
# ConfidenceScore Creation and Validation Tests
# ==============================================================================


class TestConfidenceScoreCreation:
    """Test ConfidenceScore dataclass creation and validation."""

    def test_create_with_valid_value(self) -> None:
        """Test creating ConfidenceScore with valid value."""
        score = ConfidenceScore(value=0.85)
        assert score.value == 0.85
        assert score.factors == {}
        assert score.explanation is None

    def test_create_with_factors(self) -> None:
        """Test creating ConfidenceScore with factors."""
        factors = {"signal_quality": 0.9, "pattern_match": 0.8}
        score = ConfidenceScore(value=0.85, factors=factors)
        assert score.value == 0.85
        assert score.factors == factors

    def test_create_with_explanation(self) -> None:
        """Test creating ConfidenceScore with explanation."""
        score = ConfidenceScore(value=0.85, explanation="Based on SNR and timing")
        assert score.explanation == "Based on SNR and timing"

    def test_value_rounded_to_2_decimals(self) -> None:
        """Test that value is rounded to 2 decimal places."""
        score = ConfidenceScore(value=0.123456)
        assert score.value == 0.12

    def test_value_below_zero_raises_error(self) -> None:
        """Test that value < 0.0 raises ValueError."""
        with pytest.raises(ValueError, match="must be in .0.0, 1.0."):
            ConfidenceScore(value=-0.1)

    def test_value_above_one_raises_error(self) -> None:
        """Test that value > 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="must be in .0.0, 1.0."):
            ConfidenceScore(value=1.5)

    def test_factor_below_zero_raises_error(self) -> None:
        """Test that factor < 0.0 raises ValueError."""
        with pytest.raises(ValueError, match="Factor.*must be in .0.0, 1.0."):
            ConfidenceScore(value=0.5, factors={"bad": -0.1})

    def test_factor_above_one_raises_error(self) -> None:
        """Test that factor > 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="Factor.*must be in .0.0, 1.0."):
            ConfidenceScore(value=0.5, factors={"bad": 1.5})

    def test_boundary_value_zero(self) -> None:
        """Test boundary value 0.0."""
        score = ConfidenceScore(value=0.0)
        assert score.value == 0.0

    def test_boundary_value_one(self) -> None:
        """Test boundary value 1.0."""
        score = ConfidenceScore(value=1.0)
        assert score.value == 1.0


# ==============================================================================
# ConfidenceScore Level Property Tests
# ==============================================================================


class TestConfidenceLevel:
    """Test ConfidenceScore.level property."""

    def test_level_high_at_0_90(self) -> None:
        """Test level is 'high' at 0.90."""
        score = ConfidenceScore(value=0.90)
        assert score.level == "high"

    def test_level_high_at_1_00(self) -> None:
        """Test level is 'high' at 1.00."""
        score = ConfidenceScore(value=1.00)
        assert score.level == "high"

    def test_level_medium_at_0_70(self) -> None:
        """Test level is 'medium' at 0.70."""
        score = ConfidenceScore(value=0.70)
        assert score.level == "medium"

    def test_level_medium_at_0_89(self) -> None:
        """Test level is 'medium' at 0.89."""
        score = ConfidenceScore(value=0.89)
        assert score.level == "medium"

    def test_level_low_at_0_50(self) -> None:
        """Test level is 'low' at 0.50."""
        score = ConfidenceScore(value=0.50)
        assert score.level == "low"

    def test_level_low_at_0_69(self) -> None:
        """Test level is 'low' at 0.69."""
        score = ConfidenceScore(value=0.69)
        assert score.level == "low"

    def test_level_unreliable_at_0_00(self) -> None:
        """Test level is 'unreliable' at 0.00."""
        score = ConfidenceScore(value=0.00)
        assert score.level == "unreliable"

    def test_level_unreliable_at_0_49(self) -> None:
        """Test level is 'unreliable' at 0.49."""
        score = ConfidenceScore(value=0.49)
        assert score.level == "unreliable"


# ==============================================================================
# ConfidenceScore Interpretation Property Tests
# ==============================================================================


class TestConfidenceInterpretation:
    """Test ConfidenceScore.interpretation property."""

    def test_interpretation_almost_certain_at_0_95(self) -> None:
        """Test interpretation is 'almost certain' at 0.95."""
        score = ConfidenceScore(value=0.95)
        assert score.interpretation == "almost certain"

    def test_interpretation_almost_certain_at_1_00(self) -> None:
        """Test interpretation is 'almost certain' at 1.00."""
        score = ConfidenceScore(value=1.00)
        assert score.interpretation == "almost certain"

    def test_interpretation_likely_at_0_85(self) -> None:
        """Test interpretation is 'likely' at 0.85."""
        score = ConfidenceScore(value=0.85)
        assert score.interpretation == "likely"

    def test_interpretation_likely_at_0_94(self) -> None:
        """Test interpretation is 'likely' at 0.94."""
        score = ConfidenceScore(value=0.94)
        assert score.interpretation == "likely"

    def test_interpretation_possible_at_0_75(self) -> None:
        """Test interpretation is 'possible' at 0.75."""
        score = ConfidenceScore(value=0.75)
        assert score.interpretation == "possible"

    def test_interpretation_possible_at_0_84(self) -> None:
        """Test interpretation is 'possible' at 0.84."""
        score = ConfidenceScore(value=0.84)
        assert score.interpretation == "possible"

    def test_interpretation_uncertain_at_0_55(self) -> None:
        """Test interpretation is 'uncertain' at 0.55."""
        score = ConfidenceScore(value=0.55)
        assert score.interpretation == "uncertain"

    def test_interpretation_uncertain_at_0_74(self) -> None:
        """Test interpretation is 'uncertain' at 0.74."""
        score = ConfidenceScore(value=0.74)
        assert score.interpretation == "uncertain"

    def test_interpretation_unlikely_at_0_00(self) -> None:
        """Test interpretation is 'unlikely' at 0.00."""
        score = ConfidenceScore(value=0.00)
        assert score.interpretation == "unlikely"

    def test_interpretation_unlikely_at_0_54(self) -> None:
        """Test interpretation is 'unlikely' at 0.54."""
        score = ConfidenceScore(value=0.54)
        assert score.interpretation == "unlikely"


# ==============================================================================
# ConfidenceScore.combine() Static Method Tests
# ==============================================================================


class TestConfidenceCombine:
    """Test ConfidenceScore.combine() static method."""

    def test_combine_equal_weights_two_scores(self) -> None:
        """Test combining two scores with equal weights."""
        result = ConfidenceScore.combine([0.8, 0.6])
        assert result == 0.7  # (0.8 + 0.6) / 2 = 0.7

    def test_combine_equal_weights_three_scores(self) -> None:
        """Test combining three scores with equal weights."""
        result = ConfidenceScore.combine([0.9, 0.8, 0.7])
        assert result == 0.8  # (0.9 + 0.8 + 0.7) / 3 = 0.8

    def test_combine_custom_weights(self) -> None:
        """Test combining with custom weights."""
        result = ConfidenceScore.combine([0.9, 0.8, 0.7], weights=[0.5, 0.3, 0.2])
        # 0.9*0.5 + 0.8*0.3 + 0.7*0.2 = 0.45 + 0.24 + 0.14 = 0.83
        assert result == 0.83

    def test_combine_weights_normalized(self) -> None:
        """Test that weights are normalized if they don't sum to 1.0."""
        result = ConfidenceScore.combine([0.8, 0.6], weights=[2.0, 1.0])
        # Normalized: [2/3, 1/3], result = 0.8*(2/3) + 0.6*(1/3) = 0.73
        assert result == 0.73

    def test_combine_single_score(self) -> None:
        """Test combining a single score."""
        result = ConfidenceScore.combine([0.85])
        assert result == 0.85

    def test_combine_empty_list_raises_error(self) -> None:
        """Test that empty score list raises ValueError."""
        with pytest.raises(ValueError, match="Cannot combine empty score list"):
            ConfidenceScore.combine([])

    def test_combine_score_below_zero_raises_error(self) -> None:
        """Test that score < 0.0 raises ValueError."""
        with pytest.raises(ValueError, match="Score must be in .0.0, 1.0."):
            ConfidenceScore.combine([0.5, -0.1])

    def test_combine_score_above_one_raises_error(self) -> None:
        """Test that score > 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="Score must be in .0.0, 1.0."):
            ConfidenceScore.combine([0.5, 1.5])

    def test_combine_mismatched_lengths_raises_error(self) -> None:
        """Test that mismatched scores/weights raises ValueError."""
        with pytest.raises(ValueError, match="length mismatch"):
            ConfidenceScore.combine([0.8, 0.6], weights=[0.5])

    def test_combine_zero_weight_sum_raises_error(self) -> None:
        """Test that weights summing to zero raises ValueError."""
        with pytest.raises(ValueError, match="Weights must sum to non-zero"):
            ConfidenceScore.combine([0.8, 0.6], weights=[0.0, 0.0])


# ==============================================================================
# ConfidenceScore Method Tests
# ==============================================================================


class TestConfidenceMethods:
    """Test ConfidenceScore methods."""

    def test_to_dict_minimal(self) -> None:
        """Test to_dict() with minimal ConfidenceScore."""
        score = ConfidenceScore(value=0.85)
        result = score.to_dict()
        assert result == {
            "value": 0.85,
            "level": "medium",
            "interpretation": "likely",
            "factors": {},
            "explanation": None,
        }

    def test_to_dict_with_factors(self) -> None:
        """Test to_dict() with factors."""
        factors = {"snr": 0.9, "timing": 0.8}
        score = ConfidenceScore(value=0.85, factors=factors)
        result = score.to_dict()
        assert result["factors"] == factors

    def test_to_dict_with_explanation(self) -> None:
        """Test to_dict() with explanation."""
        score = ConfidenceScore(value=0.85, explanation="Test explanation")
        result = score.to_dict()
        assert result["explanation"] == "Test explanation"

    def test_repr(self) -> None:
        """Test __repr__() method."""
        score = ConfidenceScore(value=0.85)
        assert repr(score) == "ConfidenceScore(0.85, level='medium')"

    def test_repr_high_confidence(self) -> None:
        """Test __repr__() for high confidence."""
        score = ConfidenceScore(value=0.95)
        assert repr(score) == "ConfidenceScore(0.95, level='high')"

    def test_float_conversion(self) -> None:
        """Test __float__() conversion."""
        score = ConfidenceScore(value=0.85)
        assert float(score) == 0.85

    def test_float_conversion_in_arithmetic(self) -> None:
        """Test that float conversion works in arithmetic."""
        score = ConfidenceScore(value=0.85)
        result = float(score) * 2
        assert result == 1.7


# ==============================================================================
# calculate_confidence() Function Tests
# ==============================================================================


class TestCalculateConfidence:
    """Test calculate_confidence() function."""

    def test_calculate_equal_weights(self) -> None:
        """Test calculating confidence with equal weights."""
        factors = {"signal_quality": 0.9, "pattern_match": 0.8}
        score = calculate_confidence(factors)
        assert score.value == 0.85  # (0.9 + 0.8) / 2
        assert score.factors == factors

    def test_calculate_custom_weights(self) -> None:
        """Test calculating confidence with custom weights."""
        factors = {"signal_quality": 0.9, "pattern_match": 0.8, "timing": 0.7}
        weights = {"signal_quality": 0.5, "pattern_match": 0.3, "timing": 0.2}
        score = calculate_confidence(factors, weights)
        # 0.9*0.5 + 0.8*0.3 + 0.7*0.2 = 0.83
        assert score.value == 0.83

    def test_calculate_with_explanation(self) -> None:
        """Test calculating confidence with explanation."""
        factors = {"snr": 0.9}
        score = calculate_confidence(factors, explanation="Based on SNR")
        assert score.explanation == "Based on SNR"

    def test_calculate_empty_factors_raises_error(self) -> None:
        """Test that empty factors raises ValueError."""
        with pytest.raises(ValueError, match="Cannot calculate confidence from empty factors"):
            calculate_confidence({})

    def test_calculate_missing_weight_raises_error(self) -> None:
        """Test that missing weight for factor raises ValueError."""
        factors = {"signal_quality": 0.9, "pattern_match": 0.8}
        weights = {"signal_quality": 0.6}  # Missing pattern_match
        with pytest.raises(ValueError, match="Missing weight for factor"):
            calculate_confidence(factors, weights)

    def test_calculate_single_factor(self) -> None:
        """Test calculating confidence with single factor."""
        factors = {"quality": 0.85}
        score = calculate_confidence(factors)
        assert score.value == 0.85

    def test_calculate_result_is_confidence_score(self) -> None:
        """Test that result is ConfidenceScore instance."""
        factors = {"a": 0.9, "b": 0.8}
        score = calculate_confidence(factors)
        assert isinstance(score, ConfidenceScore)
        assert hasattr(score, "level")
        assert hasattr(score, "interpretation")


# ==============================================================================
# Edge Cases and Integration Tests
# ==============================================================================


class TestCoreConfidenceEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_small_positive_value(self) -> None:
        """Test very small positive value."""
        score = ConfidenceScore(value=0.001)
        assert score.value == 0.0  # Rounded to 2 decimals

    def test_value_just_below_threshold(self) -> None:
        """Test values just below thresholds."""
        score = ConfidenceScore(value=0.899)
        assert score.level == "high"  # 0.899 rounds to 0.90

    def test_multiple_factors_all_zero(self) -> None:
        """Test factors all at zero."""
        factors = {"a": 0.0, "b": 0.0, "c": 0.0}
        score = calculate_confidence(factors)
        assert score.value == 0.0

    def test_multiple_factors_all_one(self) -> None:
        """Test factors all at one."""
        factors = {"a": 1.0, "b": 1.0, "c": 1.0}
        score = calculate_confidence(factors)
        assert score.value == 1.0

    def test_factors_preserved_in_score(self) -> None:
        """Test that factors are preserved in resulting score."""
        factors = {"snr": 0.95, "timing": 0.89}
        score = calculate_confidence(factors)
        assert score.factors == factors


class TestCoreConfidenceIntegration:
    """Integration tests combining multiple features."""

    def test_full_workflow(self) -> None:
        """Test complete workflow from factors to dict output."""
        factors = {"signal_quality": 0.9, "pattern_match": 0.85, "timing": 0.8}
        weights = {"signal_quality": 0.4, "pattern_match": 0.4, "timing": 0.2}

        score = calculate_confidence(factors, weights, explanation="Multi-factor analysis")

        assert score.value == 0.86
        assert score.level == "medium"
        assert score.interpretation == "likely"

        result_dict = score.to_dict()
        assert result_dict["value"] == 0.86
        assert result_dict["explanation"] == "Multi-factor analysis"

    def test_combine_then_create_score(self) -> None:
        """Test combining scores then creating ConfidenceScore."""
        combined = ConfidenceScore.combine([0.9, 0.8, 0.7])
        score = ConfidenceScore(
            value=combined,
            factors={"method1": 0.9, "method2": 0.8, "method3": 0.7},
        )
        assert score.value == 0.8
        assert score.level == "medium"
