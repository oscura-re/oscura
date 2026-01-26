"""Analysis quality scoring for Oscura.

This module provides quality scoring and reliability categorization for
analysis results, enabling users to assess confidence in automated findings.


Example:
    >>> from oscura.validation.quality.scoring import AnalysisQualityScore, ReliabilityCategory
    >>> score = AnalysisQualityScore(
    ...     confidence=0.85,
    ...     category=ReliabilityCategory.HIGH,
    ...     data_quality_factor=0.9,
    ...     sample_sufficiency=0.8,
    ...     method_reliability=0.85,
    ... )
    >>> print(score.explain())
    >>> recommendations = score.get_recommendations()

References:
    - Quality scoring for automated analysis results
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class ReliabilityCategory(Enum):
    """Reliability categories for analysis results.

    Attributes:
        HIGH: Result is highly reliable (confidence >= 0.8)
        MEDIUM: Result has moderate reliability (0.6 <= confidence < 0.8)
        LOW: Result has low reliability (0.4 <= confidence < 0.6)
        UNRELIABLE: Result is unreliable (confidence < 0.4)
    """

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNRELIABLE = "unreliable"

    @classmethod
    def from_confidence(cls, confidence: float) -> ReliabilityCategory:
        """Get category from confidence score.

        Args:
            confidence: Confidence value in range [0, 1]

        Returns:
            Appropriate ReliabilityCategory
        """
        if confidence >= 0.8:
            return cls.HIGH
        elif confidence >= 0.6:
            return cls.MEDIUM
        elif confidence >= 0.4:
            return cls.LOW
        else:
            return cls.UNRELIABLE


@dataclass
class AnalysisQualityScore:
    """Quality score for an analysis result.

    Attributes:
        confidence: Overall confidence in result (0-1)
        category: Reliability category
        data_quality_factor: Quality of input data (0-1)
        sample_sufficiency: Sufficiency of sample count (0-1)
        method_reliability: Inherent reliability of method (0-1)
        factors: Additional contributing factors
        warnings: Quality warnings
        metadata: Additional metadata

    Example:
        >>> score = AnalysisQualityScore(
        ...     confidence=0.85,
        ...     category=ReliabilityCategory.HIGH,
        ...     data_quality_factor=0.9,
        ...     sample_sufficiency=0.8,
        ...     method_reliability=0.85,
        ... )
        >>> if score.is_reliable:
        ...     print("Result is reliable")
    """

    confidence: float
    category: ReliabilityCategory
    data_quality_factor: float
    sample_sufficiency: float
    method_reliability: float
    factors: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate score values."""
        # Ensure confidence is in valid range
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")

        # Validate factors
        for name, value in [
            ("data_quality_factor", self.data_quality_factor),
            ("sample_sufficiency", self.sample_sufficiency),
            ("method_reliability", self.method_reliability),
        ]:
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1], got {value}")

    @property
    def is_reliable(self) -> bool:
        """Check if result is reliable (medium confidence or higher).

        Returns:
            True if category is HIGH or MEDIUM
        """
        return self.category in (ReliabilityCategory.HIGH, ReliabilityCategory.MEDIUM)

    def explain(self, include_factors: bool = True) -> str:
        """Generate human-readable explanation of the quality score.

        Args:
            include_factors: Whether to include factor breakdown

        Returns:
            Human-readable explanation string

        Example:
            >>> print(score.explain())
            ✓ High confidence result (85.0%)

            Contributing factors:
              ✓ Data Quality Factor: 90.0%
              ✓ Sample Sufficiency: 80.0%
              ✓ Method Reliability: 85.0%
        """
        lines = []

        # Overall assessment
        if self.category == ReliabilityCategory.HIGH:
            lines.append(f"✓ High confidence result ({self.confidence:.1%})")
        elif self.category == ReliabilityCategory.MEDIUM:
            lines.append(f"◐ Medium confidence result ({self.confidence:.1%})")
        elif self.category == ReliabilityCategory.LOW:
            lines.append(f"◯ Low confidence result ({self.confidence:.1%})")
        else:
            lines.append(f"✗ Unreliable result ({self.confidence:.1%})")

        # Factor breakdown
        if include_factors and self.factors:
            lines.append("\nContributing factors:")
            for factor_name, factor_value in sorted(self.factors.items()):
                status = "✓" if factor_value >= 0.7 else "◐" if factor_value >= 0.4 else "✗"
                lines.append(
                    f"  {status} {factor_name.replace('_', ' ').title()}: {factor_value:.1%}"
                )

        # Warnings
        if self.warnings:
            lines.append("\nWarnings:")
            for warning in self.warnings:
                lines.append(f"  ⚠ {warning}")

        return "\n".join(lines)

    def get_recommendations(self) -> list[str]:
        """Get actionable recommendations to improve result quality.

        Returns:
            List of recommendation strings

        Example:
            >>> recommendations = score.get_recommendations()
            >>> for rec in recommendations:
            ...     print(rec)
            Consider improving input signal quality (filtering, averaging)
        """
        recommendations = []

        if self.data_quality_factor < 0.5:
            recommendations.append("Consider improving input signal quality (filtering, averaging)")

        if self.sample_sufficiency < 0.5:
            recommendations.append("Capture more data points for reliable analysis")

        if "snr" in str(self.warnings).lower():
            recommendations.append("Use a bandpass filter to improve SNR")

        if "clipping" in str(self.warnings).lower():
            recommendations.append("Adjust input gain to avoid signal clipping")

        if not recommendations:
            recommendations.append("Result quality is acceptable")

        return recommendations

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of quality score.
        """
        return {
            "confidence": self.confidence,
            "category": self.category.value,
            "is_reliable": self.is_reliable,
            "data_quality_factor": self.data_quality_factor,
            "method_reliability": self.method_reliability,
            "sample_sufficiency": self.sample_sufficiency,
            "factors": self.factors,
            "warnings": self.warnings,
            "metadata": self.metadata,
        }


def calculate_quality_score(
    data_quality_factor: float,
    sample_sufficiency: float,
    method_reliability: float,
    *,
    weights: tuple[float, float, float] | None = None,
    warnings: list[str] | None = None,
    factors: dict[str, float] | None = None,
    metadata: dict[str, Any] | None = None,
) -> AnalysisQualityScore:
    """Calculate overall quality score from component factors.

    Args:
        data_quality_factor: Quality of input data (0-1)
        sample_sufficiency: Sufficiency of sample count (0-1)
        method_reliability: Inherent reliability of method (0-1)
        weights: Optional custom weights (data, sample, method), defaults to (0.4, 0.3, 0.3)
        warnings: Optional quality warnings
        factors: Optional additional factors
        metadata: Optional metadata

    Returns:
        AnalysisQualityScore with computed confidence

    Example:
        >>> score = calculate_quality_score(
        ...     data_quality_factor=0.9,
        ...     sample_sufficiency=0.8,
        ...     method_reliability=0.85,
        ... )
        >>> print(f"Confidence: {score.confidence:.1%}")
    """
    if weights is None:
        weights = (0.4, 0.3, 0.3)

    w_data, w_sample, w_method = weights

    # Calculate weighted confidence
    confidence = (
        w_data * data_quality_factor + w_sample * sample_sufficiency + w_method * method_reliability
    )

    # Determine category
    category = ReliabilityCategory.from_confidence(confidence)

    # Build factors dictionary
    all_factors = {
        "data_quality_factor": data_quality_factor,
        "sample_sufficiency": sample_sufficiency,
        "method_reliability": method_reliability,
    }
    if factors:
        all_factors.update(factors)

    return AnalysisQualityScore(
        confidence=confidence,
        category=category,
        data_quality_factor=data_quality_factor,
        sample_sufficiency=sample_sufficiency,
        method_reliability=method_reliability,
        factors=all_factors,
        warnings=warnings or [],
        metadata=metadata or {},
    )


@dataclass
class DataQualityMetrics:
    """Metrics describing input data quality.

        QUAL-002: Data quality assessment

    Attributes:
        snr_db: Signal-to-noise ratio in decibels
        sample_count: Number of samples in the data
        has_clipping: Whether the signal shows clipping
        has_saturation: Whether the signal shows saturation
        noise_floor: Estimated noise floor level
        completeness: Fraction of non-NaN values (0-1)
    """

    snr_db: float | None = None
    sample_count: int = 0
    has_clipping: bool = False
    has_saturation: bool = False
    noise_floor: float | None = None
    completeness: float = 1.0  # Fraction of non-NaN values

    def to_factor(self) -> float:
        """Convert metrics to single quality factor (0-1).

        Returns:
            Quality factor between 0 and 1.
        """
        factors = []

        # SNR contribution
        if self.snr_db is not None:
            snr_factor = min(1.0, max(0.0, self.snr_db / 40.0))
            factors.append(snr_factor)

        # Sample count contribution (diminishing returns after 1000)
        sample_factor = min(1.0, np.log10(max(1, self.sample_count)) / 4.0)
        factors.append(sample_factor)

        # Clipping/saturation penalties
        if self.has_clipping:
            factors.append(0.7)
        if self.has_saturation:
            factors.append(0.6)

        # Completeness
        factors.append(self.completeness)

        return float(np.mean(factors)) if factors else 0.5


# Method reliability scores (based on algorithm characteristics)
#: Method reliability tracking
METHOD_RELIABILITY: dict[str, float] = {
    # High reliability methods
    "fft": 0.95,
    "welch": 0.90,
    "autocorrelation": 0.85,
    "histogram": 0.95,
    "statistics": 0.95,
    # Medium reliability methods
    "edge_detection": 0.80,
    "zero_crossing": 0.75,
    "peak_detection": 0.70,
    "pattern_matching": 0.75,
    # Lower reliability methods (heuristic-based)
    "protocol_inference": 0.60,
    "signal_classification": 0.65,
    "anomaly_detection": 0.60,
}


def assess_data_quality(
    data: NDArray[np.float64], sample_rate: float | None = None
) -> DataQualityMetrics:
    """Assess quality of input data.

        QUAL-002: Data quality assessment

    Args:
        data: Input data array
        sample_rate: Sample rate in Hz (optional)

    Returns:
        DataQualityMetrics with quality assessment
    """
    metrics = DataQualityMetrics()

    try:
        # Sample count
        metrics.sample_count = len(data)

        # Check for NaN/Inf
        valid_mask = np.isfinite(data)
        metrics.completeness = float(np.mean(valid_mask))

        if metrics.completeness < 0.01:
            return metrics

        valid_data = data[valid_mask]

        # Check for clipping (values at min/max bounds)
        data_range = np.ptp(valid_data)
        if data_range > 0:
            min_count = np.sum(valid_data == np.min(valid_data))
            max_count = np.sum(valid_data == np.max(valid_data))
            clip_threshold = 0.01 * len(valid_data)
            metrics.has_clipping = min_count > clip_threshold or max_count > clip_threshold

        # Estimate SNR using signal variance vs noise floor
        # Use median absolute deviation for robust noise estimation
        median = np.median(valid_data)
        mad = np.median(np.abs(valid_data - median)) * 1.4826
        metrics.noise_floor = float(mad)

        signal_power = float(np.var(valid_data))
        noise_power = mad**2

        if noise_power > 0:
            snr_linear = signal_power / noise_power
            metrics.snr_db = float(10 * np.log10(max(1e-10, snr_linear)))

    except Exception as e:
        logger.debug(f"Error assessing data quality: {e}")

    return metrics


def score_analysis_result(
    result: Any,
    method_name: str,
    data: NDArray[np.float64] | None = None,
    data_quality: DataQualityMetrics | None = None,
    min_samples: int = 10,
) -> AnalysisQualityScore:
    """Score the quality of an analysis result.

        QUAL-001: Quality scoring foundation

    Args:
        result: The analysis result to score
        method_name: Name of the analysis method
        data: Input data (for quality assessment)
        data_quality: Pre-computed data quality metrics
        min_samples: Minimum samples for reliable result

    Returns:
        AnalysisQualityScore with confidence and factors
    """
    factors = {}
    warnings = []

    # Get data quality
    if data_quality is None and data is not None:
        data_quality = assess_data_quality(data)

    # Data quality factor
    if data_quality is not None:
        data_factor = data_quality.to_factor()
        factors["data_quality"] = data_factor

        if data_quality.has_clipping:
            warnings.append("Input data shows clipping")
        if data_quality.snr_db is not None and data_quality.snr_db < 20:
            warnings.append(f"Low SNR ({data_quality.snr_db:.1f} dB)")
    else:
        data_factor = 0.5
        factors["data_quality"] = data_factor

    # Method reliability
    method_key = method_name.lower().split(".")[-1].replace("_", "")
    method_reliability = METHOD_RELIABILITY.get(method_key, 0.7)

    # Check for partial matches
    for key, reliability in METHOD_RELIABILITY.items():
        if key in method_name.lower():
            method_reliability = reliability
            break

    factors["method_reliability"] = method_reliability

    # Sample sufficiency
    if data_quality is not None:
        sample_sufficiency = min(1.0, data_quality.sample_count / (min_samples * 10))
        if data_quality.sample_count < min_samples:
            warnings.append(f"Insufficient samples ({data_quality.sample_count} < {min_samples})")
    else:
        sample_sufficiency = 0.5
    factors["sample_sufficiency"] = sample_sufficiency

    # Result-specific scoring
    result_factor = _score_result_value(result)
    factors["result_validity"] = result_factor

    # Combine factors
    confidence = (
        data_factor * 0.3
        + method_reliability * 0.25
        + sample_sufficiency * 0.25
        + result_factor * 0.2
    )

    # Determine category from confidence
    category = ReliabilityCategory.from_confidence(confidence)

    return AnalysisQualityScore(
        confidence=confidence,
        category=category,
        data_quality_factor=data_factor,
        method_reliability=method_reliability,
        sample_sufficiency=sample_sufficiency,
        factors=factors,
        warnings=warnings,
    )


def _score_result_value(result: Any) -> float:
    """Score result validity based on value characteristics.

    Args:
        result: Analysis result to score.

    Returns:
        Validity score between 0 and 1.
    """
    if result is None:
        return 0.0

    # Handle numeric results
    if isinstance(result, int | float):
        if np.isnan(result) or np.isinf(result):
            return 0.0
        return 1.0

    # Handle array results
    if isinstance(result, np.ndarray):
        valid_ratio = np.mean(np.isfinite(result))
        return float(valid_ratio)

    # Handle dict results
    if isinstance(result, dict):
        if not result:
            return 0.3
        return 1.0

    # Handle list results
    if isinstance(result, list):
        if not result:
            return 0.3
        return 1.0

    return 0.7  # Default for other types


def combine_quality_scores(
    scores: list[AnalysisQualityScore],
    weights: list[float] | None = None,
) -> AnalysisQualityScore:
    """Combine multiple quality scores into one.

    Args:
        scores: List of quality scores to combine
        weights: Optional weights for each score

    Returns:
        Combined quality score
    """
    if not scores:
        return AnalysisQualityScore(
            confidence=0.0,
            category=ReliabilityCategory.UNRELIABLE,
            data_quality_factor=0.0,
            method_reliability=0.0,
            sample_sufficiency=0.0,
        )

    if weights is None:
        weights = [1.0] * len(scores)

    total_weight = sum(weights)

    combined_confidence = (
        sum(s.confidence * w for s, w in zip(scores, weights, strict=True)) / total_weight
    )
    combined_data = (
        sum(s.data_quality_factor * w for s, w in zip(scores, weights, strict=True)) / total_weight
    )
    combined_method = (
        sum(s.method_reliability * w for s, w in zip(scores, weights, strict=True)) / total_weight
    )
    combined_samples = (
        sum(s.sample_sufficiency * w for s, w in zip(scores, weights, strict=True)) / total_weight
    )

    # Aggregate warnings
    all_warnings = []
    for score in scores:
        all_warnings.extend(score.warnings)

    # Determine category
    category = ReliabilityCategory.from_confidence(combined_confidence)

    return AnalysisQualityScore(
        confidence=combined_confidence,
        category=category,
        data_quality_factor=combined_data,
        method_reliability=combined_method,
        sample_sufficiency=combined_samples,
        warnings=list(set(all_warnings)),
    )


__all__ = [
    "METHOD_RELIABILITY",
    "AnalysisQualityScore",
    "DataQualityMetrics",
    "ReliabilityCategory",
    "assess_data_quality",
    "calculate_quality_score",
    "combine_quality_scores",
    "score_analysis_result",
]
