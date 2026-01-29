"""Streaming statistics computation for large datasets.

This module provides online/streaming algorithms for computing statistics
incrementally without loading entire datasets into memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class StreamingStatsResult:
    """Result from streaming statistics computation."""

    mean: float
    variance: float
    std: float
    min: float
    max: float
    count: int


class StreamingStats:
    """Compute statistics incrementally using Welford's online algorithm.

    This class allows computing mean, variance, and other statistics
    without storing all data points in memory.

    Example:
        >>> stats = StreamingStats()
        >>> stats.update(np.array([1, 2, 3]))
        >>> stats.update(np.array([4, 5, 6]))
        >>> result = stats.finalize()
        >>> print(result.mean, result.std)
    """

    def __init__(self) -> None:
        """Initialize streaming statistics tracker."""
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0  # Sum of squared differences from mean
        self.min_val = float("inf")
        self.max_val = float("-inf")
        self.has_pos_inf = False
        self.has_neg_inf = False

    def update(self, data: NDArray[np.floating[Any]]) -> None:
        """Update statistics with new data chunk.

        Uses Welford's online algorithm for numerical stability.

        Args:
            data: New data chunk to incorporate.
        """
        data = np.asarray(data, dtype=np.float64).ravel()

        for value in data:
            self.count += 1

            # Track infinities for proper mean calculation
            if np.isposinf(value):
                self.has_pos_inf = True
            elif np.isneginf(value):
                self.has_neg_inf = True

            # Only update Welford's algorithm for finite values when no inf present
            if not (self.has_pos_inf or self.has_neg_inf):
                delta = value - self.mean
                self.mean += delta / self.count
                delta2 = value - self.mean
                self.m2 += delta * delta2

            # Update min/max
            if value < self.min_val:
                self.min_val = value
            if value > self.max_val:
                self.max_val = value

    def finalize(self) -> StreamingStatsResult:
        """Finalize and return computed statistics.

        Returns:
            StreamingStatsResult with mean, variance, std, min, max, count.
        """
        # Handle infinities in mean calculation
        if self.has_pos_inf and self.has_neg_inf:
            # Both +inf and -inf present: mean is undefined (NaN)
            mean = float("nan")
        elif self.has_pos_inf:
            # Only +inf present: mean is +inf
            mean = float("inf")
        elif self.has_neg_inf:
            # Only -inf present: mean is -inf
            mean = float("-inf")
        else:
            # No infinities: use Welford's result
            mean = self.mean

        if self.count < 2:
            variance = 0.0
            std = 0.0
        else:
            variance = self.m2 / (self.count - 1)  # Sample variance
            std = np.sqrt(variance)

        return StreamingStatsResult(
            mean=mean,
            variance=variance,
            std=std,
            min=self.min_val if self.min_val != float("inf") else 0.0,
            max=self.max_val if self.max_val != float("-inf") else 0.0,
            count=self.count,
        )
