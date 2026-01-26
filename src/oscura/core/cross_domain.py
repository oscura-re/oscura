"""Cross-domain correlation for analysis results.

Enables results from different analysis domains to inform and validate
each other, improving overall confidence and detecting inconsistencies.


Example:
    >>> from oscura.core.cross_domain import correlate_results
    >>> from oscura.reporting.config import AnalysisDomain
    >>> results = {
    ...     AnalysisDomain.SPECTRAL: {'dominant_frequency': 1000.0},
    ...     AnalysisDomain.TIMING: {'period': 0.001}
    ... }
    >>> correlation = correlate_results(results)
    >>> print(f"Coherence: {correlation.overall_coherence:.2f}")
    >>> print(f"Agreements: {correlation.agreements_detected}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from oscura.reporting.config import AnalysisDomain

logger = logging.getLogger(__name__)


@dataclass
class CrossDomainInsight:
    """An insight derived from cross-domain correlation.

    Attributes:
        insight_type: Type of insight ("agreement", "conflict", "implication").
        source_domains: Analysis domains that contributed to this insight.
        description: Human-readable description of the insight.
        confidence_impact: How much this affects confidence (-1.0 to +1.0).
        details: Additional details specific to this insight.
    """

    insight_type: str  # "agreement", "conflict", "implication"
    source_domains: list[AnalysisDomain]
    description: str
    confidence_impact: float  # How much this affects confidence (-1 to +1)
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate confidence impact after initialization."""
        if not -1.0 <= self.confidence_impact <= 1.0:
            raise ValueError(
                f"Confidence impact must be in [-1.0, 1.0], got {self.confidence_impact}"
            )


@dataclass
class CorrelationResult:
    """Result of cross-domain correlation analysis.

    Attributes:
        insights: List of discovered cross-domain insights.
        confidence_adjustments: Per-domain confidence adjustments.
        conflicts_detected: Number of conflicts found.
        agreements_detected: Number of agreements found.
    """

    insights: list[CrossDomainInsight] = field(default_factory=list)
    confidence_adjustments: dict[str, float] = field(default_factory=dict)
    conflicts_detected: int = 0
    agreements_detected: int = 0

    @property
    def overall_coherence(self) -> float:
        """Calculate overall coherence score (0-1).

        Returns:
            Coherence score based on agreement/conflict ratio.
        """
        total = self.agreements_detected + self.conflicts_detected
        if total == 0:
            return 0.5
        return self.agreements_detected / total


# Define which domains are semantically related
DOMAIN_AFFINITY: dict[AnalysisDomain, list[AnalysisDomain]] = {
    AnalysisDomain.DIGITAL: [AnalysisDomain.TIMING, AnalysisDomain.PROTOCOLS],
    AnalysisDomain.TIMING: [AnalysisDomain.DIGITAL, AnalysisDomain.JITTER, AnalysisDomain.SPECTRAL],
    AnalysisDomain.SPECTRAL: [
        AnalysisDomain.JITTER,
        AnalysisDomain.STATISTICS,
        AnalysisDomain.TIMING,
    ],
    AnalysisDomain.WAVEFORM: [AnalysisDomain.STATISTICS],
    AnalysisDomain.STATISTICS: [AnalysisDomain.SPECTRAL, AnalysisDomain.WAVEFORM],
    AnalysisDomain.JITTER: [AnalysisDomain.TIMING, AnalysisDomain.EYE, AnalysisDomain.SPECTRAL],
    AnalysisDomain.EYE: [AnalysisDomain.JITTER, AnalysisDomain.SIGNAL_INTEGRITY],
    AnalysisDomain.PATTERNS: [AnalysisDomain.PROTOCOLS, AnalysisDomain.INFERENCE],
    AnalysisDomain.INFERENCE: [AnalysisDomain.PATTERNS, AnalysisDomain.PROTOCOLS],
    AnalysisDomain.PROTOCOLS: [AnalysisDomain.DIGITAL, AnalysisDomain.PATTERNS],
}


class CrossDomainCorrelator:
    """Correlate results across analysis domains."""

    def __init__(self, tolerance: float = 0.1):
        """Initialize correlator.

        Args:
            tolerance: Tolerance for value comparisons (fraction).
        """
        self.tolerance = tolerance
        self._correlation_rules = self._build_correlation_rules()

    def correlate(
        self,
        results: dict[AnalysisDomain, dict[str, Any]],
    ) -> CorrelationResult:
        """Find correlations between domain results.

        Args:
            results: Dictionary mapping domains to their results.

        Returns:
            CorrelationResult with insights and adjustments.
        """
        correlation_result = CorrelationResult()

        # Only correlate domains that have results
        active_domains = [d for d, r in results.items() if r]

        # Track checked pairs to avoid duplicates
        checked_pairs: set[tuple[AnalysisDomain, AnalysisDomain]] = set()

        # Check each pair of related domains
        for domain in active_domains:
            related = DOMAIN_AFFINITY.get(domain, [])
            for related_domain in related:
                if related_domain in active_domains:
                    # Create canonical pair ordering to avoid duplicates
                    pair_tuple: tuple[AnalysisDomain, AnalysisDomain] = tuple(
                        sorted([domain, related_domain], key=lambda d: d.value)
                    )  # type: ignore[assignment]
                    if pair_tuple not in checked_pairs:
                        checked_pairs.add(pair_tuple)
                        insights = self._correlate_pair(
                            domain, results[domain], related_domain, results[related_domain]
                        )
                        correlation_result.insights.extend(insights)

        # Count agreements and conflicts
        for insight in correlation_result.insights:
            if insight.insight_type == "agreement":
                correlation_result.agreements_detected += 1
            elif insight.insight_type == "conflict":
                correlation_result.conflicts_detected += 1

        # Calculate confidence adjustments
        correlation_result.confidence_adjustments = self._calculate_adjustments(
            correlation_result.insights
        )

        return correlation_result

    def _correlate_pair(
        self,
        domain1: AnalysisDomain,
        results1: dict[str, Any],
        domain2: AnalysisDomain,
        results2: dict[str, Any],
    ) -> list[CrossDomainInsight]:
        """Correlate a pair of domains."""
        insights = []

        # Apply correlation rules
        for rule in self._correlation_rules:
            if rule["domains"] == {domain1, domain2} or rule["domains"] == {domain2, domain1}:
                try:
                    insight = rule["check"](results1, results2, domain1, domain2)
                    if insight:
                        insights.append(insight)
                except Exception as e:
                    logger.debug(f"Correlation rule failed: {e}")

        return insights

    def _build_correlation_rules(self) -> list[dict[str, Any]]:
        """Build correlation rules for domain pairs."""
        return [
            {
                "domains": {AnalysisDomain.SPECTRAL, AnalysisDomain.TIMING},
                "check": self._check_frequency_timing_agreement,
            },
            {
                "domains": {AnalysisDomain.DIGITAL, AnalysisDomain.TIMING},
                "check": self._check_digital_timing_consistency,
            },
            {
                "domains": {AnalysisDomain.JITTER, AnalysisDomain.EYE},
                "check": self._check_jitter_eye_correlation,
            },
            {
                "domains": {AnalysisDomain.WAVEFORM, AnalysisDomain.STATISTICS},
                "check": self._check_waveform_stats_consistency,
            },
        ]

    def _check_frequency_timing_agreement(
        self,
        results1: dict[str, Any],
        results2: dict[str, Any],
        domain1: AnalysisDomain,
        domain2: AnalysisDomain,
    ) -> CrossDomainInsight | None:
        """Check if spectral frequency matches timing period."""
        # Extract frequency from spectral results
        spectral_freq = self._extract_value(
            results1 if domain1 == AnalysisDomain.SPECTRAL else results2,
            ["dominant_frequency", "peak_frequency", "fundamental"],
        )

        # Extract period from timing results
        timing_period = self._extract_value(
            results2 if domain2 == AnalysisDomain.TIMING else results1,
            ["period", "avg_period", "mean_period"],
        )

        if spectral_freq and timing_period and spectral_freq > 0 and timing_period > 0:
            expected_period = 1.0 / spectral_freq
            ratio = timing_period / expected_period

            if 0.9 < ratio < 1.1:  # Within 10%
                return CrossDomainInsight(
                    insight_type="agreement",
                    source_domains=[AnalysisDomain.SPECTRAL, AnalysisDomain.TIMING],
                    description=(
                        f"Spectral frequency ({spectral_freq:.1f} Hz) matches "
                        f"timing period ({timing_period:.3e} s)"
                    ),
                    confidence_impact=0.15,
                    details={
                        "spectral_freq": spectral_freq,
                        "timing_period": timing_period,
                        "ratio": ratio,
                    },
                )
            elif ratio < 0.5 or ratio > 2.0:
                return CrossDomainInsight(
                    insight_type="conflict",
                    source_domains=[AnalysisDomain.SPECTRAL, AnalysisDomain.TIMING],
                    description=(
                        f"Spectral frequency ({spectral_freq:.1f} Hz) conflicts with "
                        f"timing period ({timing_period:.3e} s)"
                    ),
                    confidence_impact=-0.2,
                    details={
                        "spectral_freq": spectral_freq,
                        "timing_period": timing_period,
                        "ratio": ratio,
                    },
                )

        return None

    def _check_digital_timing_consistency(
        self,
        results1: dict[str, Any],
        results2: dict[str, Any],
        domain1: AnalysisDomain,
        domain2: AnalysisDomain,
    ) -> CrossDomainInsight | None:
        """Check if digital edge count matches timing analysis."""
        digital_results = results1 if domain1 == AnalysisDomain.DIGITAL else results2
        timing_results = results2 if domain2 == AnalysisDomain.TIMING else results1

        edge_count = self._extract_value(
            digital_results, ["edge_count", "num_edges", "transitions"]
        )
        timing_edges = self._extract_value(timing_results, ["edge_count", "transitions_detected"])

        if edge_count and timing_edges:
            if abs(edge_count - timing_edges) <= 2:
                return CrossDomainInsight(
                    insight_type="agreement",
                    source_domains=[AnalysisDomain.DIGITAL, AnalysisDomain.TIMING],
                    description=(f"Edge counts agree: Digital={edge_count}, Timing={timing_edges}"),
                    confidence_impact=0.1,
                )

        return None

    def _check_jitter_eye_correlation(
        self,
        results1: dict[str, Any],
        results2: dict[str, Any],
        domain1: AnalysisDomain,
        domain2: AnalysisDomain,
    ) -> CrossDomainInsight | None:
        """Check jitter vs eye diagram correlation."""
        jitter_results = results1 if domain1 == AnalysisDomain.JITTER else results2
        eye_results = results2 if domain2 == AnalysisDomain.EYE else results1

        total_jitter = self._extract_value(jitter_results, ["total_jitter", "tj", "jitter_pp"])
        eye_width = self._extract_value(eye_results, ["eye_width", "horizontal_opening"])

        # High jitter should correlate with narrow eye
        if total_jitter is not None and eye_width is not None:
            return CrossDomainInsight(
                insight_type="implication",
                source_domains=[AnalysisDomain.JITTER, AnalysisDomain.EYE],
                description=(f"Jitter ({total_jitter:.2e}) affects eye width ({eye_width:.2e})"),
                confidence_impact=0.05,
                details={"jitter": total_jitter, "eye_width": eye_width},
            )

        return None

    def _check_waveform_stats_consistency(
        self,
        results1: dict[str, Any],
        results2: dict[str, Any],
        domain1: AnalysisDomain,
        domain2: AnalysisDomain,
    ) -> CrossDomainInsight | None:
        """Check waveform measurements vs statistical analysis."""
        waveform_results = results1 if domain1 == AnalysisDomain.WAVEFORM else results2
        stats_results = results2 if domain2 == AnalysisDomain.STATISTICS else results1

        wf_amplitude = self._extract_value(waveform_results, ["amplitude", "vpp", "peak_to_peak"])
        stats_std = self._extract_value(stats_results, ["std", "standard_deviation", "stdev"])

        if wf_amplitude and stats_std:
            # For periodic signals, amplitude ~ 2.83 * std (for sine wave)
            expected_ratio = wf_amplitude / (2.83 * stats_std) if stats_std > 0 else 0

            if 0.8 < expected_ratio < 1.2:
                return CrossDomainInsight(
                    insight_type="agreement",
                    source_domains=[AnalysisDomain.WAVEFORM, AnalysisDomain.STATISTICS],
                    description="Waveform amplitude consistent with statistical std dev",
                    confidence_impact=0.1,
                )

        return None

    def _extract_value(
        self,
        results: dict[str, Any],
        keys: list[str],
    ) -> float | None:
        """Extract a value from results using multiple possible keys."""
        for key in keys:
            # Try direct key
            if key in results:
                val = results[key]
                if isinstance(val, int | float) and not np.isnan(val):
                    return float(val)

            # Try nested keys
            for result_val in results.values():
                if isinstance(result_val, dict) and key in result_val:
                    val = result_val[key]
                    if isinstance(val, int | float) and not np.isnan(val):
                        return float(val)

        return None

    def _calculate_adjustments(
        self,
        insights: list[CrossDomainInsight],
    ) -> dict[str, float]:
        """Calculate confidence adjustments based on insights."""
        adjustments: dict[str, float] = {}

        for insight in insights:
            for domain in insight.source_domains:
                domain_key = domain.value
                current = adjustments.get(domain_key, 0.0)
                adjustments[domain_key] = current + insight.confidence_impact

        # Clamp adjustments to [-0.3, +0.3]
        return {k: max(-0.3, min(0.3, v)) for k, v in adjustments.items()}


def correlate_results(
    results: dict[AnalysisDomain, dict[str, Any]],
    tolerance: float = 0.1,
) -> CorrelationResult:
    """Convenience function to correlate domain results.

    Args:
        results: Dictionary mapping domains to their results.
        tolerance: Tolerance for value comparisons.

    Returns:
        CorrelationResult with insights.

    Example:
        >>> from oscura.reporting.config import AnalysisDomain
        >>> results = {
        ...     AnalysisDomain.SPECTRAL: {'dominant_frequency': 1000.0},
        ...     AnalysisDomain.TIMING: {'period': 0.001}
        ... }
        >>> correlation = correlate_results(results)
        >>> print(f"Insights: {len(correlation.insights)}")
    """
    correlator = CrossDomainCorrelator(tolerance)
    return correlator.correlate(results)


__all__ = [
    "DOMAIN_AFFINITY",
    "CorrelationResult",
    "CrossDomainCorrelator",
    "CrossDomainInsight",
    "correlate_results",
]
