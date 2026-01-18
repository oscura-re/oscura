"""Ensemble methods for combining multiple analysis algorithms.

This module provides robust analysis by combining results from multiple algorithms
using various aggregation strategies. Ensemble methods reduce individual algorithm
bias, handle outliers, and provide confidence bounds for more reliable measurements.


Example:
    >>> from oscura.quality.ensemble import EnsembleAggregator, AggregationMethod
    >>> from oscura.quality.ensemble import create_frequency_ensemble
    >>> # Combine multiple frequency measurements
    >>> result = create_frequency_ensemble(signal, sample_rate=1e9)
    >>> print(f"Frequency: {result.value:.2f} Hz ± {result.confidence*100:.1f}%")
    >>> print(f"Methods agree: {result.method_agreement*100:.1f}%")
    >>> # Use custom ensemble
    >>> aggregator = EnsembleAggregator(method=AggregationMethod.WEIGHTED_AVERAGE)
    >>> results = [
    ...     {"value": 1000.0, "confidence": 0.9, "method": "fft"},
    ...     {"value": 1005.0, "confidence": 0.8, "method": "autocorr"},
    ...     {"value": 995.0, "confidence": 0.85, "method": "zero_crossing"},
    ... ]
    >>> ensemble_result = aggregator.aggregate(results)

References:
    - Kuncheva, L.I.: "Combining Pattern Classifiers" (2nd Ed), Wiley, 2014
    - Polikar, R.: "Ensemble Learning", Scholarpedia, 2009
    - Dietterich, T.G.: "Ensemble Methods in Machine Learning", 2000
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import stats

from oscura.quality.scoring import AnalysisQualityScore, combine_quality_scores

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class AggregationMethod(Enum):
    """Strategy for combining multiple analysis results.

    Attributes:
        WEIGHTED_AVERAGE: Weight results by confidence (best for numeric values).
        VOTING: Majority voting (best for categorical results).
        MEDIAN: Robust to outliers (best when outliers expected).
        BAYESIAN: Bayesian combination with prior (best when prior knowledge available).
    """

    WEIGHTED_AVERAGE = "weighted_average"
    VOTING = "voting"
    MEDIAN = "median"
    BAYESIAN = "bayesian"


@dataclass
class EnsembleResult:
    """Combined result from multiple analysis methods.

    Attributes:
        value: Aggregated value (numeric or categorical).
        confidence: Overall confidence in combined result (0-1).
        lower_bound: Lower confidence bound (None for categorical).
        upper_bound: Upper confidence bound (None for categorical).
        method_agreement: Agreement between methods (0-1, higher is better).
        individual_results: List of individual method results.
        aggregation_method: Method used for aggregation.
        quality_score: Optional quality score for the ensemble result.
        outlier_methods: Indices of methods producing outlier results.

    Example:
        >>> if result.method_agreement > 0.8:
        ...     print(f"High agreement: {result.value}")
        >>> else:
        ...     print(f"Methods disagree, confidence: {result.confidence}")
    """

    value: Any
    confidence: float
    lower_bound: float | None = None
    upper_bound: float | None = None
    method_agreement: float = 1.0
    individual_results: list[dict[str, Any]] = field(default_factory=list)
    aggregation_method: AggregationMethod = AggregationMethod.WEIGHTED_AVERAGE
    quality_score: AnalysisQualityScore | None = None
    outlier_methods: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate confidence and agreement values."""
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")
        if not 0 <= self.method_agreement <= 1:
            raise ValueError(f"Method agreement must be in [0, 1], got {self.method_agreement}")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of ensemble result.
        """
        return {
            "value": self.value,
            "confidence": self.confidence,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "method_agreement": self.method_agreement,
            "aggregation_method": self.aggregation_method.value,
            "individual_results": self.individual_results,
            "outlier_methods": self.outlier_methods,
            "quality_score": self.quality_score.to_dict() if self.quality_score else None,
        }


class EnsembleAggregator:
    """Combines multiple analysis results for robust estimation.

    Supports various aggregation strategies optimized for different data types
    and analysis scenarios. Automatically detects and handles outliers, computes
    confidence bounds, and measures inter-method agreement.

        QUAL-004: Ensemble Methods for Robust Analysis
        QUAL-005: Disagreement Detection and Handling
        QUAL-006: Confidence Bound Estimation

    Example:
        >>> aggregator = EnsembleAggregator(method=AggregationMethod.WEIGHTED_AVERAGE)
        >>> results = [
        ...     {"value": 100.0, "confidence": 0.9},
        ...     {"value": 102.0, "confidence": 0.85},
        ...     {"value": 98.0, "confidence": 0.8},
        ... ]
        >>> ensemble = aggregator.aggregate(results)
        >>> print(f"Result: {ensemble.value:.2f} ± {ensemble.confidence*100:.1f}%")
    """

    def __init__(
        self,
        method: AggregationMethod = AggregationMethod.WEIGHTED_AVERAGE,
        outlier_threshold: float = 3.0,
        min_agreement: float = 0.5,
    ):
        """Initialize ensemble aggregator.

        Args:
            method: Aggregation strategy to use.
            outlier_threshold: Z-score threshold for outlier detection (default 3.0).
            min_agreement: Minimum agreement threshold to warn (default 0.5).
        """
        self.method = method
        self.outlier_threshold = outlier_threshold
        self.min_agreement = min_agreement

    def aggregate(self, results: list[dict[str, Any]]) -> EnsembleResult:
        """Combine multiple results into one robust estimate.

        Args:
            results: List of result dictionaries with keys:
                - value: Measured value (numeric or categorical)
                - confidence: Confidence score (0-1)
                - method: Optional method name
                - quality_score: Optional AnalysisQualityScore

        Returns:
            EnsembleResult with combined value and metadata.

        Raises:
            ValueError: If results list is empty or invalid.

        Example:
            >>> results = [
            ...     {"value": 1000, "confidence": 0.9, "method": "fft"},
            ...     {"value": 1005, "confidence": 0.85, "method": "autocorr"},
            ... ]
            >>> ensemble = aggregator.aggregate(results)
        """
        if not results:
            raise ValueError("Cannot aggregate empty results list")

        # Extract values and confidences
        values = [r["value"] for r in results]
        confidences = [r.get("confidence", 1.0) for r in results]

        # Determine if values are numeric or categorical
        is_numeric = all(isinstance(v, int | float | np.number) for v in values)

        if is_numeric:
            return self.aggregate_numeric(
                [float(v) for v in values],
                confidences,
                original_results=results,
            )
        else:
            return self.aggregate_categorical(
                [str(v) for v in values],
                confidences,
                original_results=results,
            )

    def aggregate_numeric(
        self,
        values: list[float],
        confidences: list[float],
        original_results: list[dict[str, Any]] | None = None,
    ) -> EnsembleResult:
        """Combine numeric values with confidence weighting.

        Args:
            values: List of numeric values to combine.
            confidences: Confidence scores for each value (0-1).
            original_results: Optional original result dictionaries.

        Returns:
            EnsembleResult with aggregated numeric value.

        Raises:
            ValueError: If values list is empty.

        Example:
            >>> values = [100.0, 102.0, 98.0, 150.0]  # 150 is outlier
            >>> confidences = [0.9, 0.85, 0.8, 0.7]
            >>> result = aggregator.aggregate_numeric(values, confidences)
            >>> # Outlier detected and handled
        """
        if not values:
            raise ValueError("Cannot aggregate empty values list")

        if original_results is None:
            original_results = [
                {"value": v, "confidence": c} for v, c in zip(values, confidences, strict=False)
            ]

        values_arr = np.array(values, dtype=np.float64)
        confidences_arr = np.array(confidences, dtype=np.float64)

        # Detect outliers
        outlier_indices = self.detect_outlier_methods(original_results)

        # Create mask for non-outlier values
        valid_mask = np.ones(len(values), dtype=bool)
        valid_mask[outlier_indices] = False

        # Use only non-outliers for aggregation
        valid_values = values_arr[valid_mask]
        valid_confidences = confidences_arr[valid_mask]

        if len(valid_values) == 0:
            # All values are outliers, use all with warning
            logger.warning("All methods detected as outliers, using all values")
            valid_values = values_arr
            valid_confidences = confidences_arr
            outlier_indices = []

        # Compute aggregated value based on method
        if self.method == AggregationMethod.WEIGHTED_AVERAGE:
            # Normalize weights
            weights = valid_confidences / np.sum(valid_confidences)
            aggregated_value = float(np.sum(valid_values * weights))
            # Weighted variance
            variance = float(np.sum(weights * (valid_values - aggregated_value) ** 2))
            std_dev = np.sqrt(variance)

        elif self.method == AggregationMethod.MEDIAN:
            aggregated_value = float(np.median(valid_values))
            # Use MAD (Median Absolute Deviation) for robust std estimate
            mad = float(np.median(np.abs(valid_values - aggregated_value)))
            std_dev = mad * 1.4826  # Scale factor for normal distribution

        elif self.method == AggregationMethod.BAYESIAN:
            # Bayesian combination with Gaussian likelihood
            # Prior: uniform over range
            # Likelihood: Gaussian with confidence-based variance
            precisions = valid_confidences**2  # Higher confidence = lower variance
            total_precision = np.sum(precisions)
            aggregated_value = float(np.sum(valid_values * precisions) / total_precision)
            # Posterior variance
            variance = 1.0 / total_precision
            std_dev = float(np.sqrt(variance))

        else:
            # Fallback to simple average
            aggregated_value = float(np.mean(valid_values))
            std_dev = float(np.std(valid_values))

        # Compute confidence bounds (95% confidence interval)
        if len(valid_values) > 1:
            # Use t-distribution for small samples
            dof = len(valid_values) - 1
            t_value = stats.t.ppf(0.975, dof)  # 95% CI
            margin = t_value * std_dev / np.sqrt(len(valid_values))
            lower_bound = aggregated_value - margin
            upper_bound = aggregated_value + margin
        else:
            lower_bound = aggregated_value
            upper_bound = aggregated_value

        # Compute method agreement (inverse of coefficient of variation)
        if len(valid_values) > 1 and aggregated_value != 0:
            cv = std_dev / abs(aggregated_value)
            method_agreement = float(np.clip(1.0 - cv, 0.0, 1.0))
        else:
            method_agreement = 1.0

        # Overall confidence (weighted average of individual confidences)
        overall_confidence = float(np.mean(valid_confidences))

        # Penalize confidence if agreement is low
        if method_agreement < self.min_agreement:
            overall_confidence *= method_agreement
            logger.warning(
                f"Low method agreement ({method_agreement:.2f}), "
                f"reduced confidence to {overall_confidence:.2f}"
            )

        # Combine quality scores if available
        quality_scores_raw = [
            r.get("quality_score") for r in original_results if "quality_score" in r
        ]
        ensemble_quality = None
        if quality_scores_raw and all(
            isinstance(q, AnalysisQualityScore) for q in quality_scores_raw
        ):
            # Type narrowing - we know all are AnalysisQualityScore at this point
            quality_scores: list[AnalysisQualityScore] = quality_scores_raw  # type: ignore[assignment]
            ensemble_quality = combine_quality_scores(
                quality_scores, weights=confidences[: len(quality_scores)]
            )

        return EnsembleResult(
            value=aggregated_value,
            confidence=overall_confidence,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            method_agreement=method_agreement,
            individual_results=original_results,
            aggregation_method=self.method,
            quality_score=ensemble_quality,
            outlier_methods=outlier_indices,
        )

    def aggregate_categorical(
        self,
        values: list[str],
        confidences: list[float],
        original_results: list[dict[str, Any]] | None = None,
    ) -> EnsembleResult:
        """Combine categorical values via weighted voting.

        Args:
            values: List of categorical values to combine.
            confidences: Confidence scores for each value (0-1).
            original_results: Optional original result dictionaries.

        Returns:
            EnsembleResult with majority vote value.

        Raises:
            ValueError: If values list is empty.

        Example:
            >>> values = ["rising", "rising", "falling", "rising"]
            >>> confidences = [0.9, 0.85, 0.6, 0.8]
            >>> result = aggregator.aggregate_categorical(values, confidences)
            >>> # "rising" wins by weighted vote
        """
        if not values:
            raise ValueError("Cannot aggregate empty values list")

        if original_results is None:
            original_results = [
                {"value": v, "confidence": c} for v, c in zip(values, confidences, strict=False)
            ]

        # Weighted voting
        vote_weights: dict[str, float] = {}
        for value, confidence in zip(values, confidences, strict=False):
            vote_weights[value] = vote_weights.get(value, 0.0) + confidence

        # Get winner
        winner = max(vote_weights.items(), key=lambda x: x[1])
        aggregated_value = winner[0]
        total_weight = sum(vote_weights.values())

        # Confidence is the fraction of votes for winner
        overall_confidence = winner[1] / total_weight if total_weight > 0 else 0.0

        # Agreement is measured by vote concentration
        # Higher agreement when votes are concentrated on one option
        vote_counts = Counter(values)
        total_votes = len(values)
        max_count = vote_counts.most_common(1)[0][1]
        method_agreement = max_count / total_votes

        # Combine quality scores if available
        quality_scores_raw = [
            r.get("quality_score") for r in original_results if "quality_score" in r
        ]
        ensemble_quality = None
        if quality_scores_raw and all(
            isinstance(q, AnalysisQualityScore) for q in quality_scores_raw
        ):
            # Type narrowing - we know all are AnalysisQualityScore at this point
            quality_scores: list[AnalysisQualityScore] = quality_scores_raw  # type: ignore[assignment]
            ensemble_quality = combine_quality_scores(
                quality_scores, weights=confidences[: len(quality_scores)]
            )

        return EnsembleResult(
            value=aggregated_value,
            confidence=overall_confidence,
            lower_bound=None,
            upper_bound=None,
            method_agreement=method_agreement,
            individual_results=original_results,
            aggregation_method=self.method,
            quality_score=ensemble_quality,
            outlier_methods=[],  # No outlier detection for categorical
        )

    def detect_outlier_methods(self, results: list[dict[str, Any]]) -> list[int]:
        """Identify methods producing outlier results.

        Uses modified Z-score (based on MAD) for robust outlier detection.

        Args:
            results: List of result dictionaries with "value" key.

        Returns:
            List of indices corresponding to outlier methods.

        Example:
            >>> results = [
            ...     {"value": 100}, {"value": 102}, {"value": 98}, {"value": 500}
            ... ]
            >>> outliers = aggregator.detect_outlier_methods(results)
            >>> # Returns [3] - the 500 value is an outlier
        """
        values = [r["value"] for r in results]

        # Only works for numeric values
        if not all(isinstance(v, int | float | np.number) for v in values):
            return []

        if len(values) < 3:
            # Need at least 3 values for outlier detection
            return []

        values_arr = np.array(values, dtype=np.float64)

        # Use modified Z-score based on MAD (robust to outliers)
        median = np.median(values_arr)
        mad = np.median(np.abs(values_arr - median))

        if mad == 0:
            # All values are identical
            return []

        # Modified Z-score
        modified_z_scores = 0.6745 * (values_arr - median) / mad

        # Identify outliers
        outlier_mask = np.abs(modified_z_scores) > self.outlier_threshold
        outlier_indices: list[int] = np.where(outlier_mask)[0].tolist()

        if outlier_indices:
            logger.info(f"Detected {len(outlier_indices)} outlier method(s): {outlier_indices}")

        return outlier_indices


# Pre-configured ensembles for common analysis types
# Weights represent the relative reliability of each method

FREQUENCY_ENSEMBLE: list[tuple[str, float]] = [
    ("fft_peak", 0.4),  # FFT peak is generally most reliable
    ("zero_crossing", 0.3),  # Zero crossing is robust but can be noisy
    ("autocorrelation", 0.3),  # Autocorrelation handles noise well
]

EDGE_DETECTION_ENSEMBLE: list[tuple[str, float]] = [
    ("threshold_crossing", 0.5),  # Most direct method
    ("derivative", 0.3),  # Good for clean signals
    ("schmitt_trigger", 0.2),  # Noise immunity but less precise
]

AMPLITUDE_ENSEMBLE: list[tuple[str, float]] = [
    ("peak_to_peak", 0.4),  # Direct measurement
    ("rms", 0.3),  # Robust to noise
    ("percentile_99", 0.3),  # Outlier resistant
]


def create_frequency_ensemble(
    signal: NDArray[np.float64],
    sample_rate: float,
    method_weights: list[tuple[str, float]] | None = None,
) -> EnsembleResult:
    """Run multiple frequency detection methods and combine results.

    Applies FFT peak detection, zero-crossing rate, and autocorrelation-based
    frequency estimation, then combines using weighted averaging.

    Args:
        signal: Input signal array.
        sample_rate: Sample rate in Hz.
        method_weights: Optional custom method weights. Defaults to FREQUENCY_ENSEMBLE.

    Returns:
        EnsembleResult with combined frequency estimate.

    Raises:
        ValueError: If all frequency detection methods fail.

    Example:
        >>> import numpy as np
        >>> t = np.linspace(0, 1, 1000)
        >>> signal = np.sin(2 * np.pi * 10 * t)  # 10 Hz sine
        >>> result = create_frequency_ensemble(signal, sample_rate=1000)
        >>> print(f"Frequency: {result.value:.2f} Hz")
        >>> print(f"Confidence: {result.confidence:.2%}")
    """
    if method_weights is None:
        method_weights = FREQUENCY_ENSEMBLE

    results = []

    # Method 1: FFT peak detection
    try:
        fft_result = np.fft.rfft(signal)
        freqs = np.fft.rfftfreq(len(signal), d=1.0 / sample_rate)
        peak_idx = np.argmax(np.abs(fft_result[1:])) + 1  # Skip DC
        freq_fft = float(freqs[peak_idx])
        # Confidence based on peak prominence
        peak_magnitude = np.abs(fft_result[peak_idx])
        mean_magnitude = np.mean(np.abs(fft_result[1:]))
        confidence_fft = min(1.0, peak_magnitude / (mean_magnitude * 10))
        results.append(
            {
                "value": freq_fft,
                "confidence": confidence_fft * method_weights[0][1],
                "method": "fft_peak",
            }
        )
    except Exception as e:
        logger.debug(f"FFT peak detection failed: {e}")

    # Method 2: Zero crossing rate
    try:
        zero_crossings = np.where(np.diff(np.sign(signal)))[0]
        if len(zero_crossings) > 1:
            # Average time between zero crossings (half period)
            avg_half_period = np.mean(np.diff(zero_crossings)) / sample_rate
            freq_zc = 1.0 / (2.0 * avg_half_period)
            # Confidence based on regularity of crossings
            std_half_period = np.std(np.diff(zero_crossings)) / sample_rate
            confidence_zc = max(0.0, 1.0 - std_half_period / avg_half_period)
            results.append(
                {
                    "value": float(freq_zc),
                    "confidence": confidence_zc * method_weights[1][1],
                    "method": "zero_crossing",
                }
            )
    except Exception as e:
        logger.debug(f"Zero crossing detection failed: {e}")

    # Method 3: Autocorrelation
    try:
        # Compute autocorrelation
        autocorr = np.correlate(signal, signal, mode="full")
        autocorr = autocorr[len(autocorr) // 2 :]
        # Find first peak after zero lag (skip DC)
        peaks = []
        for i in range(1, min(len(autocorr) - 1, len(signal) // 2)):
            if autocorr[i] > autocorr[i - 1] and autocorr[i] > autocorr[i + 1]:
                peaks.append(i)
        if peaks:
            first_peak = peaks[0]
            period_samples = first_peak
            freq_ac = sample_rate / period_samples
            # Confidence based on peak strength
            peak_strength = autocorr[first_peak] / autocorr[0]
            confidence_ac = float(np.clip(peak_strength, 0.0, 1.0))
            results.append(
                {
                    "value": float(freq_ac),
                    "confidence": confidence_ac * method_weights[2][1],
                    "method": "autocorrelation",
                }
            )
    except Exception as e:
        logger.debug(f"Autocorrelation detection failed: {e}")

    if not results:
        raise ValueError("All frequency detection methods failed")

    # Aggregate results
    aggregator = EnsembleAggregator(method=AggregationMethod.WEIGHTED_AVERAGE)
    return aggregator.aggregate(results)


def create_edge_ensemble(
    signal: NDArray[np.float64],
    sample_rate: float,
    threshold: float | None = None,
    method_weights: list[tuple[str, float]] | None = None,
) -> EnsembleResult:
    """Run multiple edge detection methods and combine results.

    Applies threshold crossing, derivative-based, and Schmitt trigger edge
    detection, then combines results using weighted voting or averaging.

    Args:
        signal: Input signal array.
        sample_rate: Sample rate in Hz.
        threshold: Detection threshold. If None, uses signal midpoint.
        method_weights: Optional custom method weights. Defaults to EDGE_DETECTION_ENSEMBLE.

    Returns:
        EnsembleResult with combined edge detection results.

    Raises:
        ValueError: If all edge detection methods fail.

    Example:
        >>> signal = np.array([0, 0, 1, 1, 0, 0, 1, 1])
        >>> result = create_edge_ensemble(signal, sample_rate=1000)
        >>> print(f"Edge count: {result.value}")
        >>> print(f"Agreement: {result.method_agreement:.2%}")
    """
    if method_weights is None:
        method_weights = EDGE_DETECTION_ENSEMBLE

    if threshold is None:
        threshold = float((np.max(signal) + np.min(signal)) / 2.0)

    results = []

    # Method 1: Threshold crossing
    try:
        crossings = np.where(np.diff(np.sign(signal - threshold)))[0]
        edge_count_tc = len(crossings)
        # Confidence based on signal quality (SNR proxy)
        signal_range = np.ptp(signal)
        noise_estimate = np.std(np.diff(signal))
        confidence_tc = (
            min(1.0, signal_range / (noise_estimate * 10)) if noise_estimate > 0 else 0.5
        )
        results.append(
            {
                "value": edge_count_tc,
                "confidence": confidence_tc * method_weights[0][1],
                "method": "threshold_crossing",
            }
        )
    except Exception as e:
        logger.debug(f"Threshold crossing detection failed: {e}")

    # Method 2: Derivative-based
    try:
        derivative = np.diff(signal)
        # Find peaks in absolute derivative
        deriv_std = np.std(derivative)
        deriv_threshold = deriv_std * 2
        edge_indices = np.where(np.abs(derivative) > deriv_threshold)[0]
        # Remove consecutive detections (within 2 samples)
        filtered_edges = []
        for i, idx in enumerate(edge_indices):
            if i == 0 or idx - edge_indices[i - 1] > 2:
                filtered_edges.append(idx)
        edge_count_deriv = len(filtered_edges)
        # Confidence based on peak derivative prominence above threshold
        # Higher max derivative relative to threshold means clearer edges
        max_deriv = np.max(np.abs(derivative)) if len(derivative) > 0 else 0.0
        prominence_ratio = (max_deriv / deriv_threshold) if deriv_threshold > 0 else 0.0
        confidence_deriv = float(
            np.clip(prominence_ratio / 3.0, 0.0, 1.0)
        )  # Normalize: 3x threshold = 100%
        results.append(
            {
                "value": edge_count_deriv,
                "confidence": confidence_deriv * method_weights[1][1],
                "method": "derivative",
            }
        )
    except Exception as e:
        logger.debug(f"Derivative edge detection failed: {e}")

    # Method 3: Schmitt trigger (hysteresis)
    try:
        hysteresis = float(np.std(signal) * 0.1)
        thresh_high = threshold + hysteresis
        thresh_low = threshold - hysteresis
        state = signal[0] > threshold
        edge_count_schmitt = 0
        for val in signal:
            if not state and val > thresh_high:
                edge_count_schmitt += 1
                state = True
            elif state and val < thresh_low:
                edge_count_schmitt += 1
                state = False
        # Confidence based on hysteresis effectiveness
        confidence_schmitt = 0.7  # Lower base confidence due to hysteresis delay
        results.append(
            {
                "value": edge_count_schmitt,
                "confidence": confidence_schmitt * method_weights[2][1],
                "method": "schmitt_trigger",
            }
        )
    except Exception as e:
        logger.debug(f"Schmitt trigger detection failed: {e}")

    if not results:
        raise ValueError("All edge detection methods failed")

    # Aggregate results (use median for integer counts)
    aggregator = EnsembleAggregator(method=AggregationMethod.MEDIAN)
    return aggregator.aggregate(results)


__all__ = [
    "AMPLITUDE_ENSEMBLE",
    "EDGE_DETECTION_ENSEMBLE",
    "FREQUENCY_ENSEMBLE",
    "AggregationMethod",
    "EnsembleAggregator",
    "EnsembleResult",
    "create_edge_ensemble",
    "create_frequency_ensemble",
]
