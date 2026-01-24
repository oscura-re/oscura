"""Timing side-channel analysis.

This module implements timing attack analysis for extracting secrets from
execution time variations in cryptographic implementations.

Example:
    >>> from oscura.analyzers.side_channel.timing import TimingAnalyzer
    >>> analyzer = TimingAnalyzer(confidence_level=0.95)
    >>> result = analyzer.analyze(timings, inputs)
    >>> if result.has_leak:
    ...     print(f"Timing leak detected with {result.confidence:.2%} confidence")

References:
    Kocher "Timing Attacks on Implementations of Diffie-Hellman, RSA, DSS" (CRYPTO 1996)
    Brumley & Boneh "Remote Timing Attacks are Practical" (USENIX Security 2003)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import stats

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = [
    "TimingAnalyzer",
    "TimingAttackResult",
    "TimingLeak",
]


@dataclass
class TimingLeak:
    """Detected timing leak information.

    Attributes:
        input_bit: Bit position causing leak (if bit-level analysis).
        input_byte: Byte position causing leak (if byte-level analysis).
        mean_difference: Mean timing difference between groups.
        t_statistic: T-test statistic.
        p_value: Statistical p-value.
        confidence: Confidence level (1 - p_value).
        effect_size: Cohen's d effect size.
    """

    input_bit: int | None
    input_byte: int | None
    mean_difference: float
    t_statistic: float
    p_value: float
    confidence: float
    effect_size: float

    @property
    def is_significant(self) -> bool:
        """Check if leak is statistically significant (p < 0.05)."""
        return self.p_value < 0.05


@dataclass
class TimingAttackResult:
    """Result of timing attack analysis.

    Attributes:
        has_leak: Whether timing leak detected.
        leaks: List of detected leaks.
        confidence: Overall confidence in leak detection.
        timing_statistics: Summary statistics of timings.
    """

    has_leak: bool
    leaks: list[TimingLeak]
    confidence: float
    timing_statistics: dict[str, float]


class TimingAnalyzer:
    """Timing side-channel attack analyzer.

    Detects timing leaks in cryptographic implementations by analyzing
    execution time variations correlated with input data.

    Args:
        confidence_level: Confidence level for leak detection (default 0.95).
        min_samples: Minimum samples required for analysis (default 100).

    Example:
        >>> analyzer = TimingAnalyzer(confidence_level=0.99)
        >>> # Measure RSA decryption times
        >>> timings = measure_decryption_times(ciphertexts)
        >>> result = analyzer.analyze(timings, ciphertexts)
        >>> for leak in result.leaks:
        ...     print(f"Leak at bit {leak.input_bit}: p={leak.p_value:.6f}")

    References:
        Kocher "Timing Attacks on Implementations of Diffie-Hellman, RSA, DSS"
    """

    def __init__(self, confidence_level: float = 0.95, min_samples: int = 100) -> None:
        """Initialize timing analyzer.

        Args:
            confidence_level: Confidence level (0.0-1.0).
            min_samples: Minimum number of samples.

        Raises:
            ValueError: If parameters out of range.
        """
        if not 0.0 < confidence_level < 1.0:
            raise ValueError(f"confidence_level must be in (0, 1), got {confidence_level}")
        if min_samples < 10:
            raise ValueError(f"min_samples must be >= 10, got {min_samples}")

        self.confidence_level = confidence_level
        self.min_samples = min_samples

    def analyze(
        self,
        timings: NDArray[np.floating[Any]],
        inputs: NDArray[np.integer[Any]],
    ) -> TimingAttackResult:
        """Analyze timing measurements for leaks.

        Performs statistical analysis (t-tests) to detect correlations
        between input bits/bytes and execution time.

        Args:
            timings: Execution times (n_samples,).
            inputs: Input values (n_samples, input_size) or (n_samples,).

        Returns:
            TimingAttackResult with detected leaks.

        Raises:
            ValueError: If inputs invalid or insufficient samples.

        Example:
            >>> timings = np.array([...])  # Measured execution times
            >>> inputs = np.random.randint(0, 256, (1000, 16), dtype=np.uint8)
            >>> result = analyzer.analyze(timings, inputs)
            >>> print(f"Leaks detected: {len(result.leaks)}")
        """
        if len(timings) < self.min_samples:
            raise ValueError(f"Insufficient samples: {len(timings)} < {self.min_samples}")

        if len(timings) != len(inputs):
            raise ValueError(f"Timings ({len(timings)}) and inputs ({len(inputs)}) length mismatch")

        # Compute timing statistics
        timing_stats = {
            "mean": float(np.mean(timings)),
            "std": float(np.std(timings)),
            "min": float(np.min(timings)),
            "max": float(np.max(timings)),
            "median": float(np.median(timings)),
        }

        # Detect leaks
        leaks: list[TimingLeak] = []

        if inputs.ndim == 1:
            # Single byte per input
            byte_leaks = self._analyze_byte_leaks(timings, inputs, byte_pos=0)
            leaks.extend(byte_leaks)
        else:
            # Multiple bytes per input
            for byte_pos in range(inputs.shape[1]):
                byte_leaks = self._analyze_byte_leaks(timings, inputs[:, byte_pos], byte_pos)
                leaks.extend(byte_leaks)

        # Overall confidence is max of individual leak confidences
        overall_confidence = max([leak.confidence for leak in leaks], default=0.0)

        return TimingAttackResult(
            has_leak=len(leaks) > 0,
            leaks=leaks,
            confidence=overall_confidence,
            timing_statistics=timing_stats,
        )

    def _analyze_byte_leaks(
        self,
        timings: NDArray[np.floating[Any]],
        byte_values: NDArray[np.integer[Any]],
        byte_pos: int,
    ) -> list[TimingLeak]:
        """Analyze timing leaks for specific byte position.

        Args:
            timings: Execution times.
            byte_values: Byte values at this position.
            byte_pos: Byte position index.

        Returns:
            List of detected leaks for this byte.
        """
        leaks: list[TimingLeak] = []

        # Test each bit position in the byte
        for bit_pos in range(8):
            # Partition timings by bit value
            bit_mask = 1 << bit_pos
            bit_set = (byte_values & bit_mask) != 0

            timings_0 = timings[~bit_set]
            timings_1 = timings[bit_set]

            # Need sufficient samples in each group
            if len(timings_0) < 10 or len(timings_1) < 10:
                continue

            # Welch's t-test (unequal variances)
            t_stat, p_value = stats.ttest_ind(timings_0, timings_1, equal_var=False)

            # Check for significance
            alpha = 1.0 - self.confidence_level
            if p_value < alpha:
                # Calculate effect size (Cohen's d)
                mean_diff = float(np.mean(timings_1) - np.mean(timings_0))
                pooled_std = np.sqrt((np.var(timings_0) + np.var(timings_1)) / 2)
                effect_size = mean_diff / pooled_std if pooled_std > 0 else 0.0

                leak = TimingLeak(
                    input_bit=bit_pos,
                    input_byte=byte_pos,
                    mean_difference=mean_diff,
                    t_statistic=float(t_stat),
                    p_value=float(p_value),
                    confidence=1.0 - float(p_value),
                    effect_size=float(effect_size),
                )
                leaks.append(leak)

        return leaks

    def analyze_with_partitioning(
        self,
        timings: NDArray[np.floating[Any]],
        inputs: NDArray[np.integer[Any]],
        partition_func: Callable[[NDArray[np.integer[Any]]], NDArray[np.bool_]],
    ) -> tuple[float, float, float]:
        """Analyze timing with custom partitioning function.

        Allows custom grouping criteria beyond bit values.

        Args:
            timings: Execution times.
            inputs: Input values.
            partition_func: Function mapping inputs to boolean partition.

        Returns:
            Tuple of (mean_difference, t_statistic, p_value).

        Example:
            >>> # Partition by high/low byte value
            >>> def partition(x):
            ...     return x >= 128
            >>> diff, t_stat, p_val = analyzer.analyze_with_partitioning(
            ...     timings, inputs, partition
            ... )
        """
        partition = partition_func(inputs)

        timings_0 = timings[~partition]
        timings_1 = timings[partition]

        if len(timings_0) < 10 or len(timings_1) < 10:
            return 0.0, 0.0, 1.0

        t_stat, p_value = stats.ttest_ind(timings_0, timings_1, equal_var=False)
        mean_diff = float(np.mean(timings_1) - np.mean(timings_0))

        return mean_diff, float(t_stat), float(p_value)

    def detect_outliers(
        self,
        timings: NDArray[np.floating[Any]],
        threshold: float = 3.0,
    ) -> NDArray[np.bool_]:
        """Detect outlier measurements using modified Z-score.

        Outliers may indicate measurement errors or specific attack scenarios.

        Args:
            timings: Execution times.
            threshold: Z-score threshold for outliers (default 3.0).

        Returns:
            Boolean array marking outliers.

        Example:
            >>> outliers = analyzer.detect_outliers(timings)
            >>> clean_timings = timings[~outliers]
        """
        median = np.median(timings)
        mad = np.median(np.abs(timings - median))

        # Modified Z-score (more robust than standard Z-score)
        if mad == 0:
            # All values identical
            return np.zeros(len(timings), dtype=bool)

        modified_z_scores = 0.6745 * (timings - median) / mad
        outliers: NDArray[np.bool_] = np.abs(modified_z_scores) > threshold

        return outliers

    def compute_mutual_information(
        self,
        timings: NDArray[np.floating[Any]],
        inputs: NDArray[np.integer[Any]],
        n_bins: int = 10,
    ) -> float:
        """Compute mutual information between timings and inputs.

        Measures information leakage in bits.

        Args:
            timings: Execution times.
            inputs: Input values.
            n_bins: Number of bins for discretization.

        Returns:
            Mutual information in bits.

        Example:
            >>> mi = analyzer.compute_mutual_information(timings, inputs)
            >>> print(f"Information leakage: {mi:.4f} bits")
        """
        # Discretize timings into bins
        timing_bins = np.digitize(
            timings,
            bins=np.linspace(np.min(timings), np.max(timings), n_bins),
        )

        # Discretize inputs
        if inputs.ndim == 1:
            input_bins = inputs
        else:
            # Hash multi-dimensional inputs
            input_bins = np.sum(inputs * np.arange(inputs.shape[1]), axis=1)

        # Compute joint histogram
        joint_hist, _, _ = np.histogram2d(
            timing_bins,
            input_bins,
            bins=[n_bins, len(np.unique(input_bins))],
        )
        joint_prob = joint_hist / np.sum(joint_hist)

        # Marginal distributions
        timing_prob = np.sum(joint_prob, axis=1)
        input_prob = np.sum(joint_prob, axis=0)

        # Mutual information: I(X; Y) = sum p(x,y) * log(p(x,y) / (p(x) * p(y)))
        mi = 0.0
        for i in range(len(timing_prob)):
            for j in range(len(input_prob)):
                if joint_prob[i, j] > 0:
                    mi += joint_prob[i, j] * np.log2(
                        joint_prob[i, j] / (timing_prob[i] * input_prob[j])
                    )

        return float(mi)
