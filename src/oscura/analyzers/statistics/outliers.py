"""Outlier detection for signal analysis.

This module provides multiple outlier detection methods suitable for
different data distributions and contamination levels.


Example:
    >>> from oscura.analyzers.statistics.outliers import (
    ...     zscore_outliers, modified_zscore_outliers, iqr_outliers
    ... )
    >>> outliers = zscore_outliers(trace, threshold=3.0)
    >>> robust_outliers = modified_zscore_outliers(trace, threshold=3.5)
    >>> iqr_result = iqr_outliers(trace, multiplier=1.5)

References:
    Iglewicz, B. & Hoaglin, D. (1993). How to Detect and Handle Outliers
    NIST Engineering Statistics Handbook
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from oscura.core.types import WaveformTrace

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class OutlierResult:
    """Result of outlier detection.

    Attributes:
        indices: Array of indices where outliers were detected.
        values: Array of outlier values.
        scores: Array of outlier scores (z-scores or similar).
        mask: Boolean mask (True = outlier).
        count: Number of outliers detected.
        method: Detection method used.
        threshold: Threshold used for detection.
    """

    indices: NDArray[np.intp]
    values: NDArray[np.float64]
    scores: NDArray[np.float64]
    mask: NDArray[np.bool_]
    count: int
    method: str
    threshold: float


def zscore_outliers(
    trace: WaveformTrace | NDArray[np.floating[Any]],
    *,
    threshold: float = 3.0,
    return_scores: bool = False,
) -> OutlierResult | tuple[OutlierResult, NDArray[np.float64]]:
    """Detect outliers using Z-score method.

    Identifies points where |z-score| exceeds the threshold.
    Best for normally distributed data without heavy contamination.

    Args:
        trace: Input trace or numpy array.
        threshold: Z-score threshold for outlier detection (default 3.0).
        return_scores: If True, also return full z-score array.

    Returns:
        OutlierResult containing outlier information.
        If return_scores=True, also returns full z-score array.

    Example:
        >>> result = zscore_outliers(trace, threshold=3.0)
        >>> print(f"Found {result.count} outliers")
        >>> print(f"Outlier indices: {result.indices}")

    References:
        NIST Engineering Statistics Handbook Section 1.3.5.17
    """
    data = trace.data if isinstance(trace, WaveformTrace) else trace

    n = len(data)
    if n < 3:
        empty_result = OutlierResult(
            indices=np.array([], dtype=np.intp),
            values=np.array([], dtype=np.float64),
            scores=np.array([], dtype=np.float64),
            mask=np.zeros(n, dtype=np.bool_),
            count=0,
            method="zscore",
            threshold=threshold,
        )
        if return_scores:
            return empty_result, np.zeros(n, dtype=np.float64)
        return empty_result

    # Compute z-scores
    mean = np.mean(data)
    std = np.std(data, ddof=1)

    if std < 1e-12:
        # No variation, no outliers
        zscores = np.zeros(n, dtype=np.float64)
    else:
        zscores = (data - mean) / std

    # Find outliers
    mask = np.abs(zscores) > threshold
    indices = np.where(mask)[0]
    outlier_values = data[mask].astype(np.float64)
    outlier_scores = zscores[mask]

    result = OutlierResult(
        indices=indices,
        values=outlier_values,
        scores=outlier_scores,
        mask=mask,
        count=int(np.sum(mask)),
        method="zscore",
        threshold=threshold,
    )

    if return_scores:
        return result, zscores
    return result


def modified_zscore_outliers(
    trace: WaveformTrace | NDArray[np.floating[Any]],
    *,
    threshold: float = 3.5,
    return_scores: bool = False,
) -> OutlierResult | tuple[OutlierResult, NDArray[np.float64]]:
    """Detect outliers using Modified Z-score (MAD-based) method.

    Uses Median Absolute Deviation (MAD) for robust outlier detection.
    More resistant to contaminated data than standard z-score.

    Args:
        trace: Input trace or numpy array.
        threshold: Modified z-score threshold (default 3.5, per Iglewicz & Hoaglin).
        return_scores: If True, also return full modified z-score array.

    Returns:
        OutlierResult containing outlier information.
        If return_scores=True, also returns full modified z-score array.

    Example:
        >>> result = modified_zscore_outliers(trace, threshold=3.5)
        >>> print(f"Found {result.count} outliers")
        >>> # Robust to up to ~50% contamination

    References:
        Iglewicz, B. & Hoaglin, D. (1993). How to Detect and Handle Outliers
    """
    data = trace.data if isinstance(trace, WaveformTrace) else trace

    n = len(data)
    if n < 3:
        empty_result = OutlierResult(
            indices=np.array([], dtype=np.intp),
            values=np.array([], dtype=np.float64),
            scores=np.array([], dtype=np.float64),
            mask=np.zeros(n, dtype=np.bool_),
            count=0,
            method="modified_zscore",
            threshold=threshold,
        )
        if return_scores:
            return empty_result, np.zeros(n, dtype=np.float64)
        return empty_result

    # Compute median and MAD
    median = np.median(data)
    mad = np.median(np.abs(data - median))

    # Consistency constant for normal distribution
    # MAD = 0.6745 * sigma for normal distribution
    k = 0.6745

    if mad < 1e-12:
        # Very low spread - use fallback
        # Check if there are any points far from median
        deviations = np.abs(data - median)
        max_dev = np.max(deviations)
        if max_dev < 1e-12:
            # All points identical, no outliers
            modified_zscores = np.zeros(n, dtype=np.float64)
        else:
            # Scale deviations so that max_dev gets a high score
            # This ensures outliers in nearly-constant data are detected
            # Use a scale factor that makes max_dev map to a large z-score
            scale = max_dev / (threshold * 2)  # Conservative scaling
            modified_zscores = deviations / scale
    else:
        modified_zscores = k * (data - median) / mad

    # Find outliers
    mask = np.abs(modified_zscores) > threshold
    indices = np.where(mask)[0]
    outlier_values = data[mask].astype(np.float64)
    outlier_scores = modified_zscores[mask]

    result = OutlierResult(
        indices=indices,
        values=outlier_values,
        scores=outlier_scores,
        mask=mask,
        count=int(np.sum(mask)),
        method="modified_zscore",
        threshold=threshold,
    )

    if return_scores:
        return result, modified_zscores.astype(np.float64)
    return result


def iqr_outliers(
    trace: WaveformTrace | NDArray[np.floating[Any]],
    *,
    multiplier: float = 1.5,
    return_fences: bool = False,
) -> OutlierResult | tuple[OutlierResult, dict[str, float]]:
    """Detect outliers using Interquartile Range (IQR) method.

    Flags points outside the fences: [Q1 - k*IQR, Q3 + k*IQR].
    Good for skewed distributions.

    Args:
        trace: Input trace or numpy array.
        multiplier: IQR multiplier for fence calculation (default 1.5).
            Use 3.0 for "extreme" outliers.
        return_fences: If True, also return fence values.

    Returns:
        OutlierResult containing outlier information.
        If return_fences=True, also returns dict with Q1, Q3, IQR, fences.

    Example:
        >>> result = iqr_outliers(trace, multiplier=1.5)
        >>> print(f"Found {result.count} outliers")

        >>> # Get fence values
        >>> result, fences = iqr_outliers(trace, return_fences=True)
        >>> print(f"Lower fence: {fences['lower_fence']}")

    References:
        Tukey, J. W. (1977). Exploratory Data Analysis
    """
    data = trace.data if isinstance(trace, WaveformTrace) else trace

    n = len(data)
    if n < 4:
        empty_result = OutlierResult(
            indices=np.array([], dtype=np.intp),
            values=np.array([], dtype=np.float64),
            scores=np.array([], dtype=np.float64),
            mask=np.zeros(n, dtype=np.bool_),
            count=0,
            method="iqr",
            threshold=multiplier,
        )
        if return_fences:
            return empty_result, {
                "q1": np.nan,
                "q3": np.nan,
                "iqr": np.nan,
                "lower_fence": np.nan,
                "upper_fence": np.nan,
            }
        return empty_result

    # Compute quartiles
    q1 = float(np.percentile(data, 25))
    q3 = float(np.percentile(data, 75))
    iqr = q3 - q1

    # Calculate fences
    lower_fence = q1 - multiplier * iqr
    upper_fence = q3 + multiplier * iqr

    # Find outliers
    mask = (data < lower_fence) | (data > upper_fence)
    indices = np.where(mask)[0]
    outlier_values = data[mask].astype(np.float64)

    # Calculate "scores" as distance from nearest fence normalized by IQR
    if iqr > 0:
        scores = np.zeros(n, dtype=np.float64)
        below = data < lower_fence
        above = data > upper_fence
        scores[below] = (lower_fence - data[below]) / iqr
        scores[above] = (data[above] - upper_fence) / iqr
        outlier_scores = scores[mask]
    else:
        outlier_scores = np.zeros(len(indices), dtype=np.float64)

    result = OutlierResult(
        indices=indices,
        values=outlier_values,
        scores=outlier_scores,
        mask=mask,
        count=int(np.sum(mask)),
        method="iqr",
        threshold=multiplier,
    )

    if return_fences:
        fences = {
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "lower_fence": lower_fence,
            "upper_fence": upper_fence,
        }
        return result, fences
    return result


def detect_outliers(
    trace: WaveformTrace | NDArray[np.floating[Any]],
    *,
    method: str = "modified_zscore",
    **kwargs: Any,
) -> OutlierResult:
    """Detect outliers using specified method.

    Convenience function that dispatches to the appropriate outlier
    detection method based on the method parameter.

    Args:
        trace: Input trace or numpy array.
        method: Detection method. One of:
            - "zscore": Standard z-score method
            - "modified_zscore": MAD-based robust method (default)
            - "iqr": Interquartile range method
        **kwargs: Additional arguments passed to the detection method.

    Returns:
        OutlierResult containing outlier information.

    Raises:
        ValueError: If method is not one of the supported types.

    Example:
        >>> result = detect_outliers(trace, method="iqr", multiplier=2.0)
        >>> print(f"Method: {result.method}, Count: {result.count}")
    """
    methods = {
        "zscore": zscore_outliers,  # type: ignore[dict-item]
        "modified_zscore": modified_zscore_outliers,  # type: ignore[dict-item]
        "iqr": iqr_outliers,
    }

    if method not in methods:
        available = ", ".join(methods.keys())
        raise ValueError(f"Unknown method: {method}. Available: {available}")

    result = methods[method](trace, **kwargs)

    # Handle tuple returns
    if isinstance(result, tuple):
        return result[0]
    return result


def remove_outliers(  # type: ignore[no-untyped-def]
    trace: WaveformTrace | NDArray[np.floating[Any]],
    *,
    method: str = "modified_zscore",
    replacement: str = "nan",
    **kwargs,
) -> NDArray[np.float64]:
    """Remove or replace outliers in data.

    Args:
        trace: Input trace or numpy array.
        method: Detection method (see detect_outliers).
        replacement: How to handle outliers:
            - "nan": Replace with NaN
            - "clip": Clip to nearest fence/threshold
            - "interpolate": Linear interpolation from neighbors
        **kwargs: Additional arguments for detection method.

    Returns:
        Array with outliers handled according to replacement method.

    Raises:
        ValueError: If replacement method is not one of the supported types.

    Example:
        >>> cleaned = remove_outliers(trace, method="iqr", replacement="nan")
        >>> # Use for analysis that can handle NaN values
    """
    if isinstance(trace, WaveformTrace):
        data = trace.data.copy()
    else:
        data = np.array(trace, dtype=np.float64)

    result = detect_outliers(trace, method=method, **kwargs)

    if result.count == 0:
        return data

    if replacement == "nan":
        data[result.mask] = np.nan

    elif replacement == "clip":
        if method == "iqr":
            # Get fence values
            _, fences = iqr_outliers(trace, return_fences=True, **kwargs)  # type: ignore[misc]
            data = np.clip(data, fences["lower_fence"], fences["upper_fence"])
        else:
            # For z-score methods, clip to mean +/- threshold * std
            mean = np.mean(data[~result.mask]) if np.any(~result.mask) else np.mean(data)
            std = (
                np.std(data[~result.mask], ddof=1) if np.any(~result.mask) else np.std(data, ddof=1)
            )
            threshold = result.threshold
            data = np.clip(data, mean - threshold * std, mean + threshold * std)

    elif replacement == "interpolate":
        # Linear interpolation from non-outlier neighbors
        outlier_indices = result.indices
        valid_indices = np.where(~result.mask)[0]

        if len(valid_indices) > 0:
            for idx in outlier_indices:
                # Find nearest valid neighbors
                left_valid = valid_indices[valid_indices < idx]
                right_valid = valid_indices[valid_indices > idx]

                if len(left_valid) > 0 and len(right_valid) > 0:
                    # Interpolate between neighbors
                    left_idx = left_valid[-1]
                    right_idx = right_valid[0]
                    weight = (idx - left_idx) / (right_idx - left_idx)
                    data[idx] = data[left_idx] + weight * (data[right_idx] - data[left_idx])
                elif len(left_valid) > 0:
                    data[idx] = data[left_valid[-1]]
                elif len(right_valid) > 0:
                    data[idx] = data[right_valid[0]]
                # else: leave unchanged

    else:
        raise ValueError(f"Unknown replacement method: {replacement}")

    return data


__all__ = [
    "OutlierResult",
    "detect_outliers",
    "iqr_outliers",
    "modified_zscore_outliers",
    "remove_outliers",
    "zscore_outliers",
]
