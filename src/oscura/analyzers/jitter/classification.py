"""Jitter classification result types and utilities.

This module provides result types for jitter classification analysis,
allowing unified representation of jitter component estimates with
confidence metrics and classification methods.

The classification results support IEEE 2414-2020 compliant jitter
decomposition workflows by providing structured outputs for RJ, DJ,
PJ, and TJ estimates with associated confidence levels.

Example:
    >>> from oscura.analyzers.jitter.classification import (
    ...     JitterComponentEstimate,
    ...     JitterClassificationResult,
    ... )
    >>> rj_est = JitterComponentEstimate(value=0.05, confidence=0.92, unit="UI")
    >>> dj_est = JitterComponentEstimate(value=0.12, confidence=0.88, unit="UI")
    >>> result = JitterClassificationResult(
    ...     rj_estimate=rj_est,
    ...     dj_estimate=dj_est,
    ...     tj_estimate=0.17,
    ...     classification_method="dual_dirac",
    ...     ber_target=1e-12,
    ... )
    >>> print(f"Total Jitter at BER={result.ber_target}: {result.tj_estimate} UI")

References:
    IEEE 2414-2020: Standard for Jitter and Phase Noise
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class JitterComponentEstimate:
    """Estimate of a single jitter component with confidence metrics.

    Represents the estimated magnitude of a jitter component (RJ, DJ, PJ, etc.)
    along with a confidence score indicating the reliability of the estimate
    and the unit of measurement.

    Attributes:
        value: Estimated magnitude of the jitter component
        confidence: Confidence score for this estimate (0.0-1.0, where 1.0 is
            highest confidence). Based on fit quality, sample size, and
            statistical significance.
        unit: Unit of measurement (e.g., "UI" for unit intervals, "s" for
            seconds, "ps" for picoseconds)

    Example:
        >>> rj = JitterComponentEstimate(value=5.2e-12, confidence=0.94, unit="s")
        >>> print(f"RJ: {rj.value*1e12:.2f} ps (confidence: {rj.confidence:.1%})")
        RJ: 5.20 ps (confidence: 94.0%)
    """

    value: float
    confidence: float
    unit: str

    def __post_init__(self) -> None:
        """Validate component estimate fields."""
        if self.confidence < 0 or self.confidence > 1:
            raise ValueError(f"Confidence must be in [0,1], got {self.confidence}")
        if not isinstance(self.unit, str) or not self.unit:
            raise ValueError(f"Unit must be non-empty string, got {self.unit!r}")


@dataclass(frozen=True)
class JitterClassificationResult:
    """Complete jitter classification result with all components.

    Represents the outcome of a comprehensive jitter analysis, including
    estimates for random jitter (RJ), deterministic jitter (DJ), and
    total jitter (TJ) at a specified BER target. Includes metadata about
    the classification method used.

    This result type is useful for timing closure analysis, link budget
    calculations, and system-level jitter characterization.

    Attributes:
        rj_estimate: Random jitter component estimate with confidence
        dj_estimate: Deterministic jitter component estimate with confidence
        tj_estimate: Total jitter at the specified BER (RJ + DJ convolution)
        classification_method: Method used for classification (e.g.,
            "dual_dirac", "spectral_fit", "tail_fit"). Indicates the
            algorithm used for RJ/DJ separation.
        ber_target: Target bit error rate for TJ calculation (e.g., 1e-12)
        pj_estimate: Optional periodic jitter component estimate
        ddj_estimate: Optional data-dependent jitter component estimate

    Example:
        >>> result = JitterClassificationResult(
        ...     rj_estimate=JitterComponentEstimate(0.05, 0.92, "UI"),
        ...     dj_estimate=JitterComponentEstimate(0.12, 0.88, "UI"),
        ...     tj_estimate=0.17,
        ...     classification_method="dual_dirac",
        ...     ber_target=1e-12,
        ... )
        >>> print(f"TJ@{result.ber_target}: {result.tj_estimate} {result.rj_estimate.unit}")
        TJ@1e-12: 0.17 UI

    Notes:
        For IEEE 2414-2020 compliance, TJ should be calculated using:
        TJ = DJ_pp + n*RJ_rms, where n = Q(BER/2) and Q is the inverse
        Q-function (Gaussian tail probability).
    """

    rj_estimate: JitterComponentEstimate
    dj_estimate: JitterComponentEstimate
    tj_estimate: float
    classification_method: str
    ber_target: float
    pj_estimate: JitterComponentEstimate | None = None
    ddj_estimate: JitterComponentEstimate | None = None

    def __post_init__(self) -> None:
        """Validate classification result fields."""
        if self.ber_target <= 0 or self.ber_target >= 1:
            raise ValueError(f"BER target must be in (0,1), got {self.ber_target}")
        if not isinstance(self.classification_method, str) or not self.classification_method:
            raise ValueError(
                f"Classification method must be non-empty string, got "
                f"{self.classification_method!r}"
            )
        if self.tj_estimate < 0:
            raise ValueError(f"TJ estimate cannot be negative, got {self.tj_estimate}")

    @property
    def rj_confidence(self) -> float:
        """Convenience accessor for RJ confidence score."""
        return self.rj_estimate.confidence

    @property
    def dj_confidence(self) -> float:
        """Convenience accessor for DJ confidence score."""
        return self.dj_estimate.confidence

    @property
    def overall_confidence(self) -> float:
        """Overall confidence as minimum of RJ and DJ confidence scores.

        Returns the more conservative (lower) of the two confidence values,
        as the TJ estimate depends on both RJ and DJ being accurate.

        Returns:
            Minimum confidence score between RJ and DJ estimates
        """
        return min(self.rj_estimate.confidence, self.dj_estimate.confidence)


__all__ = [
    "JitterClassificationResult",
    "JitterComponentEstimate",
]
