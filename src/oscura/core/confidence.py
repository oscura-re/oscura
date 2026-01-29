"""Universal confidence scoring for auto-detection.

This module provides confidence scoring infrastructure used across
all auto-discovery and analysis functions.


Example:
    >>> from oscura.core.confidence import ConfidenceScore
    >>> score = ConfidenceScore(value=0.85, factors={'signal_quality': 0.9, 'pattern_match': 0.8})
    >>> print(f"Confidence: {score.value:.2f} ({score.interpretation})")
    Confidence: 0.85 (likely)

References:
    Oscura Auto-Discovery Specification
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ConfidenceScore:
    """Confidence score for auto-detection results.

    Represents reliability of automated analysis results with
    standardized 0.0-1.0 scale and human-readable interpretation.

    Confidence scale:
        - 0.9-1.0: High - "almost certain", trust the result
        - 0.7-0.9: Medium - "likely", verify if critical
        - 0.5-0.7: Low - "possible", check alternatives
        - 0.0-0.5: Unreliable - "uncertain", manual analysis recommended

    Attributes:
        value: Confidence value (0.0-1.0, 2 decimal precision).
        factors: Dictionary of contributing factors and their scores.
        explanation: Optional explanation of confidence calculation.

    Example:
        >>> score = ConfidenceScore(0.92, factors={'snr': 0.95, 'timing': 0.89})
        >>> print(f"{score.value:.2f} - {score.interpretation}")
        0.92 - almost certain
    """

    value: float
    factors: dict[str, float] = field(default_factory=dict)
    explanation: str | None = None

    def __post_init__(self) -> None:
        """Validate confidence score after initialization."""
        if not 0.0 <= self.value <= 1.0:
            raise ValueError(f"Confidence value must be in [0.0, 1.0], got {self.value}")
        # Round to 2 decimal places
        self.value = round(self.value, 2)

        # Validate factors
        for name, factor_value in self.factors.items():
            if not 0.0 <= factor_value <= 1.0:
                raise ValueError(f"Factor '{name}' must be in [0.0, 1.0], got {factor_value}")

    @property
    def level(self) -> str:
        """Confidence level classification.

        Returns:
            str: One of "high", "medium", "low", "unreliable".
        """
        if self.value >= 0.9:
            return "high"
        elif self.value >= 0.7:
            return "medium"
        elif self.value >= 0.5:
            return "low"
        else:
            return "unreliable"

    @property
    def interpretation(self) -> str:
        """Human-readable interpretation.

        Returns:
            Descriptive interpretation string.
        """
        if self.value >= 0.95:
            return "almost certain"
        elif self.value >= 0.85:
            return "likely"
        elif self.value >= 0.75:
            return "possible"
        elif self.value >= 0.55:
            return "uncertain"
        else:
            return "unlikely"

    @staticmethod
    def combine(
        scores: list[float],
        weights: list[float] | None = None,
    ) -> float:
        """Combine multiple confidence scores into one.

        Uses weighted average to combine scores. Equal weights if not specified.

        Args:
            scores: List of confidence values (0.0-1.0).
            weights: Optional weight for each score (must sum to 1.0).

        Returns:
            Combined confidence score (0.0-1.0).

        Raises:
            ValueError: If scores/weights are invalid or don't match.

        Example:
            >>> scores = [0.9, 0.8, 0.7]
            >>> combined = ConfidenceScore.combine(scores, weights=[0.5, 0.3, 0.2])
            >>> print(f"{combined:.2f}")
            0.83
        """
        if not scores:
            raise ValueError("Cannot combine empty score list")

        for score in scores:
            if not 0.0 <= score <= 1.0:
                raise ValueError(f"Score must be in [0.0, 1.0], got {score}")

        if weights is None:
            # Equal weights
            weights = [1.0 / len(scores)] * len(scores)

        if len(scores) != len(weights):
            raise ValueError(f"Scores ({len(scores)}) and weights ({len(weights)}) length mismatch")

        # Normalize weights to sum to 1.0
        weight_sum = sum(weights)
        if weight_sum == 0:
            raise ValueError("Weights must sum to non-zero value")

        normalized_weights = [w / weight_sum for w in weights]

        # Weighted average
        combined = sum(s * w for s, w in zip(scores, normalized_weights, strict=False))
        return round(combined, 2)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary with confidence details.
        """
        return {
            "value": self.value,
            "level": self.level,
            "interpretation": self.interpretation,
            "factors": self.factors,
            "explanation": self.explanation,
        }

    def __repr__(self) -> str:
        """String representation."""
        return f"ConfidenceScore({self.value:.2f}, level='{self.level}')"

    def __float__(self) -> float:
        """Convert to float (returns value)."""
        return self.value


def calculate_confidence(
    factors: dict[str, float],
    weights: dict[str, float] | None = None,
    *,
    explanation: str | None = None,
) -> ConfidenceScore:
    """Calculate confidence score from multiple factors.

    Args:
        factors: Dictionary of factor names to values (0.0-1.0).
        weights: Optional weights for each factor (must sum to 1.0).
        explanation: Optional explanation of calculation.

    Returns:
        ConfidenceScore object with combined value.

    Raises:
        ValueError: If factors is empty or missing weight for a factor.

    Example:
        >>> factors = {'signal_quality': 0.9, 'pattern_match': 0.85, 'timing': 0.8}
        >>> weights = {'signal_quality': 0.4, 'pattern_match': 0.4, 'timing': 0.2}
        >>> score = calculate_confidence(factors, weights)
        >>> print(f"Confidence: {score.value:.2f}")
        Confidence: 0.86
    """
    if not factors:
        raise ValueError("Cannot calculate confidence from empty factors")

    if weights is None:
        # Equal weights
        score_values = list(factors.values())
        weight_values = None
    else:
        # Use provided weights
        score_values = []
        weight_values = []
        for name, value in factors.items():
            score_values.append(value)
            if name not in weights:
                raise ValueError(f"Missing weight for factor '{name}'")
            weight_values.append(weights[name])

    combined_value = ConfidenceScore.combine(score_values, weight_values)

    return ConfidenceScore(
        value=combined_value,
        factors=factors,
        explanation=explanation,
    )


__all__ = [
    "ConfidenceScore",
    "calculate_confidence",
]
