"""Result explainability for analysis outputs.

Generates human-readable explanations for why analysis results
are reliable or unreliable.

Example:
    >>> from oscura.validation.quality.explainer import explain_result
    >>> from oscura.validation.quality.scoring import calculate_quality_score
    >>> score = calculate_quality_score(0.9, 0.8, 0.85)
    >>> explanation = explain_result("frequency", 10.5e6, score, "fft")
    >>> print(explanation)

References:
    - Result explainability for analysis outputs
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from oscura.validation.quality.scoring import AnalysisQualityScore


@dataclass
class ResultExplanation:
    """Detailed explanation for an analysis result.

    Attributes:
        result_name: Name of the result
        result_value: Value of the result
        quality: Quality score for the result
        why_reliable: Reasons why result is reliable
        why_unreliable: Reasons why result may be unreliable
        recommendations: Actionable recommendations
        alternative_methods: Alternative analysis methods

    Example:
        >>> explanation = ResultExplanation(
        ...     result_name="frequency",
        ...     result_value=10.5e6,
        ...     quality=score,
        ... )
        >>> print(explanation.to_narrative())
    """

    result_name: str
    result_value: Any
    quality: AnalysisQualityScore

    # Detailed reasoning
    why_reliable: list[str] = field(default_factory=list)
    why_unreliable: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    alternative_methods: list[str] = field(default_factory=list)

    def to_narrative(self) -> str:
        """Generate narrative explanation.

        Returns:
            Human-readable narrative explanation string

        Example:
            >>> narrative = explanation.to_narrative()
            >>> print(narrative)
            Result: frequency = 1.050e+07
            Confidence: 85.0% (high)

            Why this result is reliable:
              ✓ FFT is mathematically exact for periodic signals
              ✓ High quality input data
        """
        lines = []

        # Header with result
        lines.append(f"Result: {self.result_name} = {self._format_value(self.result_value)}")
        lines.append(f"Confidence: {self.quality.confidence:.1%} ({self.quality.category.value})")
        lines.append("")

        # Why reliable
        if self.why_reliable:
            lines.append("Why this result is reliable:")
            for reason in self.why_reliable:
                lines.append(f"  ✓ {reason}")
            lines.append("")

        # Why unreliable
        if self.why_unreliable:
            lines.append("Concerns affecting reliability:")
            for reason in self.why_unreliable:
                lines.append(f"  ⚠ {reason}")
            lines.append("")

        # Recommendations
        if self.recommendations:
            lines.append("Recommendations:")
            for rec in self.recommendations:
                lines.append(f"  → {rec}")
            lines.append("")

        # Alternatives
        if self.alternative_methods:
            lines.append("Alternative approaches:")
            for alt in self.alternative_methods:
                lines.append(f"  • {alt}")

        return "\n".join(lines)

    def _format_value(self, value: Any) -> str:
        """Format result value for display.

        Args:
            value: Value to format

        Returns:
            Formatted string representation
        """
        if isinstance(value, float):
            if abs(value) < 0.001 or abs(value) > 10000:
                return f"{value:.3e}"
            return f"{value:.4f}"
        return str(value)


class ResultExplainer:
    """Generate explanations for analysis results.

    This class provides method-specific explanations for common
    analysis techniques used in signal processing.

    Example:
        >>> explainer = ResultExplainer()
        >>> explanation = explainer.explain(
        ...     result_name="frequency",
        ...     result_value=10.5e6,
        ...     quality=score,
        ...     method_name="fft",
        ... )
        >>> print(explanation.to_narrative())
    """

    # Method-specific explanation templates
    METHOD_EXPLANATIONS: ClassVar[dict[str, Any]] = {
        "fft": {
            "reliable": [
                "FFT is mathematically exact for periodic signals",
                "Sufficient frequency resolution for analysis",
            ],
            "concerns": {
                "low_snr": "Low SNR may cause spurious frequency peaks",
                "short_duration": "Short capture limits frequency resolution",
                "windowing": "Spectral leakage may affect amplitude accuracy",
            },
            "alternatives": ["Welch method for noisy signals", "Zero-padding for interpolation"],
        },
        "edge_detection": {
            "reliable": [
                "Multiple consistent edges detected",
                "Clear transition between logic levels",
            ],
            "concerns": {
                "noise": "Noise may cause false edge detections",
                "ringing": "Signal ringing may create multiple edge crossings",
            },
            "alternatives": ["Interpolated edge timing", "Histogram-based threshold"],
        },
        "statistics": {
            "reliable": [
                "Statistical methods are mathematically well-defined",
                "Large sample size provides reliable estimates",
            ],
            "concerns": {
                "outliers": "Outliers may skew mean and variance",
                "non_normal": "Non-normal distribution affects some statistics",
            },
            "alternatives": ["Robust statistics (median, MAD)", "Trimmed mean"],
        },
    }

    def explain(
        self,
        result_name: str,
        result_value: Any,
        quality: AnalysisQualityScore,
        method_name: str | None = None,
    ) -> ResultExplanation:
        """Generate explanation for an analysis result.

        Args:
            result_name: Name of the result
            result_value: The result value
            quality: Quality score for the result
            method_name: Optional method name for specific explanations

        Returns:
            ResultExplanation with detailed reasoning

        Example:
            >>> explanation = explainer.explain(
            ...     "frequency",
            ...     10.5e6,
            ...     score,
            ...     "fft"
            ... )
            >>> print(explanation.to_narrative())
        """
        explanation = ResultExplanation(
            result_name=result_name,
            result_value=result_value,
            quality=quality,
        )

        # Get method-specific templates
        method_key = self._get_method_key(method_name or result_name)
        templates = self.METHOD_EXPLANATIONS.get(method_key, {})

        # Generate reliability reasons
        if quality.is_reliable:
            explanation.why_reliable = self._generate_reliable_reasons(quality, templates)

        explanation.why_unreliable = self._generate_unreliable_reasons(quality, templates)
        explanation.recommendations = quality.get_recommendations()
        explanation.alternative_methods = list(templates.get("alternatives", []))

        return explanation

    def _get_method_key(self, name: str) -> str:
        """Extract method key from name.

        Args:
            name: Method or result name

        Returns:
            Standardized method key
        """
        name_lower = name.lower()
        if "fft" in name_lower or "spectral" in name_lower:
            return "fft"
        if "edge" in name_lower:
            return "edge_detection"
        if "stat" in name_lower or "mean" in name_lower or "std" in name_lower:
            return "statistics"
        return "generic"

    def _generate_reliable_reasons(
        self,
        quality: AnalysisQualityScore,
        templates: dict[str, Any],
    ) -> list[str]:
        """Generate reasons why result is reliable.

        Args:
            quality: Quality score
            templates: Method-specific explanation templates

        Returns:
            List of reasons
        """
        reasons = []

        # Add template reasons
        reasons.extend(templates.get("reliable", []))

        # Add factor-based reasons
        if quality.data_quality_factor >= 0.8:
            reasons.append("High quality input data")
        if quality.sample_sufficiency >= 0.9:
            reasons.append("Sufficient sample count for analysis")
        if quality.method_reliability >= 0.85:
            reasons.append("Analysis method has high inherent reliability")

        return reasons[:4]  # Limit to top 4

    def _generate_unreliable_reasons(
        self,
        quality: AnalysisQualityScore,
        templates: dict[str, Any],
    ) -> list[str]:
        """Generate reasons why result may be unreliable.

        Args:
            quality: Quality score
            templates: Method-specific explanation templates

        Returns:
            List of concerns
        """
        reasons = []

        # Add warnings
        reasons.extend(quality.warnings)

        # Add factor-based concerns
        if quality.data_quality_factor < 0.5:
            reasons.append("Input data quality is poor")
        if quality.sample_sufficiency < 0.5:
            reasons.append("Insufficient samples for reliable estimate")
        if quality.method_reliability < 0.6:
            reasons.append("Analysis method has lower reliability")

        return reasons


def explain_result(
    result_name: str,
    result_value: Any,
    quality: AnalysisQualityScore,
    method_name: str | None = None,
) -> str:
    """Convenience function to get result explanation as text.

    Args:
        result_name: Name of the result
        result_value: The result value
        quality: Quality score
        method_name: Optional method name

    Returns:
        Human-readable explanation string

    Example:
        >>> explanation = explain_result("frequency", 10.5e6, score, "fft")
        >>> print(explanation)
        Result: frequency = 1.050e+07
        Confidence: 85.0% (high)
        ...
    """
    explainer = ResultExplainer()
    explanation = explainer.explain(result_name, result_value, quality, method_name)
    return explanation.to_narrative()


__all__ = [
    "ResultExplainer",
    "ResultExplanation",
    "explain_result",
]
