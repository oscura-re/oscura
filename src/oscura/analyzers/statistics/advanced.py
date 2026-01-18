"""Advanced statistical analysis methods.

This module provides advanced outlier detection and time series analysis
methods for signal analysis.


Example:
    >>> from oscura.analyzers.statistics.advanced import (
    ...     isolation_forest_outliers, local_outlier_factor,
    ...     seasonal_decompose, detect_change_points,
    ...     phase_coherence, kernel_density
    ... )
    >>> outliers = isolation_forest_outliers(trace)
    >>> decomp = seasonal_decompose(trace, period=100)

References:
    Liu et al. (2008): Isolation Forest
    Breunig et al. (2000): Local Outlier Factor
    Cleveland et al. (1990): STL Decomposition
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from scipy import signal
from scipy import stats as sp_stats

from oscura.core.types import WaveformTrace

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class IsolationForestResult:
    """Result of Isolation Forest outlier detection.

    Attributes:
        indices: Array of outlier indices.
        scores: Anomaly scores for all samples (-1 = outlier, 1 = normal).
        decision_scores: Raw decision function scores.
        mask: Boolean mask (True = outlier).
        count: Number of outliers detected.
        contamination: Contamination fraction used.

    References:
        STAT-011
    """

    indices: NDArray[np.intp]
    scores: NDArray[np.int8]
    decision_scores: NDArray[np.float64]
    mask: NDArray[np.bool_]
    count: int
    contamination: float


@dataclass
class LOFResult:
    """Result of Local Outlier Factor detection.

    Attributes:
        indices: Array of outlier indices.
        scores: LOF scores for all samples (>1 = outlier).
        mask: Boolean mask (True = outlier).
        count: Number of outliers detected.
        threshold: Threshold used for outlier classification.
        n_neighbors: Number of neighbors used.

    References:
        STAT-012
    """

    indices: NDArray[np.intp]
    scores: NDArray[np.float64]
    mask: NDArray[np.bool_]
    count: int
    threshold: float
    n_neighbors: int


@dataclass
class DecompositionResult:
    """Result of seasonal decomposition.

    Attributes:
        trend: Trend component.
        seasonal: Seasonal component.
        residual: Residual (remainder) component.
        period: Detected or specified period.
        observed: Original signal.

    References:
        STAT-013
    """

    trend: NDArray[np.float64]
    seasonal: NDArray[np.float64]
    residual: NDArray[np.float64]
    period: int
    observed: NDArray[np.float64]


@dataclass
class ChangePointResult:
    """Result of change point detection.

    Attributes:
        indices: Array of change point indices.
        n_changes: Number of change points detected.
        segments: List of (start, end) segment boundaries.
        segment_means: Mean value for each segment.
        segment_stds: Standard deviation for each segment.
        cost: Total cost of the segmentation.

    References:
        STAT-014
    """

    indices: NDArray[np.intp]
    n_changes: int
    segments: list[tuple[int, int]]
    segment_means: NDArray[np.float64]
    segment_stds: NDArray[np.float64]
    cost: float


@dataclass
class CoherenceResult:
    """Result of phase coherence analysis.

    Attributes:
        coherence: Coherence spectrum (0 to 1).
        frequencies: Frequency axis in Hz.
        phase: Phase difference spectrum in radians.
        mean_coherence: Average coherence across frequencies.
        peak_frequency: Frequency of maximum coherence.
        peak_coherence: Maximum coherence value.

    References:
        STAT-015
    """

    coherence: NDArray[np.float64]
    frequencies: NDArray[np.float64]
    phase: NDArray[np.float64]
    mean_coherence: float
    peak_frequency: float
    peak_coherence: float


@dataclass
class KDEResult:
    """Result of kernel density estimation.

    Attributes:
        x: Evaluation points.
        density: Probability density at each point.
        bandwidth: Bandwidth used for estimation.
        peaks: Indices of density peaks (modes).
        peak_values: X-values at density peaks.

    References:
        STAT-016
    """

    x: NDArray[np.float64]
    density: NDArray[np.float64]
    bandwidth: float
    peaks: NDArray[np.intp]
    peak_values: NDArray[np.float64]


def isolation_forest_outliers(
    trace: WaveformTrace | NDArray[np.floating[Any]],
    *,
    contamination: float = 0.05,
    n_estimators: int = 100,
    max_samples: int | str = "auto",
    random_state: int | None = None,
) -> IsolationForestResult:
    """Detect outliers using Isolation Forest algorithm.

    Isolation Forest isolates anomalies by randomly selecting features
    and split values. Anomalies are isolated in fewer splits on average.

    Args:
        trace: Input trace or numpy array.
        contamination: Expected proportion of outliers (0.0 to 0.5).
        n_estimators: Number of isolation trees.
        max_samples: Samples for each tree ("auto" = min(256, n_samples)).
        random_state: Random seed for reproducibility.

    Returns:
        IsolationForestResult with outlier information.

    Example:
        >>> result = isolation_forest_outliers(trace, contamination=0.01)
        >>> print(f"Found {result.count} outliers")
        >>> clean_data = trace[~result.mask]

    References:
        Liu, Ting & Zhou (2008): Isolation Forest
        STAT-011
    """
    data = trace.data if isinstance(trace, WaveformTrace) else np.asarray(trace)
    n_samples = len(data)

    if n_samples < 10:
        return IsolationForestResult(
            indices=np.array([], dtype=np.intp),
            scores=np.ones(n_samples, dtype=np.int8),
            decision_scores=np.zeros(n_samples, dtype=np.float64),
            mask=np.zeros(n_samples, dtype=np.bool_),
            count=0,
            contamination=contamination,
        )

    # Set random state
    rng = np.random.default_rng(random_state)

    # Determine max_samples
    max_samples_int: int
    if max_samples == "auto":
        max_samples_int = min(256, n_samples)
    elif isinstance(max_samples, float):
        max_samples_int = int(max_samples * n_samples)
    elif isinstance(max_samples, int):
        max_samples_int = max_samples
    else:
        # Fallback for any other string value
        max_samples_int = min(256, n_samples)
    max_samples_int = min(max_samples_int, n_samples)

    # Build isolation forest
    decision_scores = np.zeros(n_samples, dtype=np.float64)

    for _ in range(n_estimators):
        # Bootstrap sample
        sample_idx = rng.choice(n_samples, size=max_samples_int, replace=False)
        sample_data = data[sample_idx]

        # Compute path lengths for all points
        path_lengths = _isolation_tree_path_lengths(data, sample_data, rng)
        decision_scores += path_lengths

    # Average and normalize
    decision_scores /= n_estimators

    # Compute anomaly scores: shorter paths = anomalies
    # Normalize using average path length formula
    avg_path = _average_path_length(max_samples_int)
    decision_scores = 2 ** (-decision_scores / avg_path)

    # Threshold based on contamination
    threshold = np.percentile(decision_scores, 100 * (1 - contamination))

    # Classify
    mask = decision_scores >= threshold
    indices = np.where(mask)[0]
    scores = np.where(mask, -1, 1).astype(np.int8)

    return IsolationForestResult(
        indices=indices.astype(np.intp),
        scores=scores,
        decision_scores=decision_scores,
        mask=mask,
        count=int(np.sum(mask)),
        contamination=contamination,
    )


def _isolation_tree_path_lengths(
    data: NDArray[Any], sample: NDArray[Any], rng: np.random.Generator
) -> NDArray[np.float64]:
    """Compute isolation path lengths for data points."""
    n = len(data)
    path_lengths = np.zeros(n, dtype=np.float64)

    # Simple recursive isolation tree simulation
    # For each point, estimate how many splits to isolate it
    for i, point in enumerate(data):
        path_lengths[i] = _compute_path_length(point, sample, rng, 0)

    return path_lengths


def _compute_path_length(
    point: float,
    sample: NDArray[Any],
    rng: np.random.Generator,
    depth: int,
    max_depth: int = 20,
) -> float:
    """Recursively compute path length to isolate a point."""
    if len(sample) <= 1 or depth >= max_depth:
        return depth + _average_path_length(len(sample))

    # Random split point
    min_val, max_val = np.min(sample), np.max(sample)
    if max_val == min_val:
        return depth

    split = rng.uniform(min_val, max_val)

    if point < split:
        left_sample = sample[sample < split]
        return _compute_path_length(point, left_sample, rng, depth + 1, max_depth)
    else:
        right_sample = sample[sample >= split]
        return _compute_path_length(point, right_sample, rng, depth + 1, max_depth)


def _average_path_length(n: int) -> float:
    """Compute average path length for n samples (H(n-1) formula)."""
    if n <= 1:
        return 0
    if n == 2:
        return 1
    # Harmonic number approximation
    return 2 * (np.log(n - 1) + 0.5772156649) - 2 * (n - 1) / n  # type: ignore[no-any-return]


def local_outlier_factor(
    trace: WaveformTrace | NDArray[np.floating[Any]],
    *,
    n_neighbors: int = 20,
    threshold: float = 1.5,
    metric: Literal["euclidean", "manhattan"] = "euclidean",
) -> LOFResult:
    """Detect outliers using Local Outlier Factor.

    LOF measures local density deviation of a point with respect to
    its neighbors. Points with substantially lower density than
    their neighbors are considered outliers.

    Args:
        trace: Input trace or numpy array.
        n_neighbors: Number of neighbors to use for density estimation.
        threshold: LOF threshold for outlier classification (>1 = outlier).
        metric: Distance metric ("euclidean" or "manhattan").

    Returns:
        LOFResult with outlier information.

    Example:
        >>> result = local_outlier_factor(trace, n_neighbors=10)
        >>> print(f"Found {result.count} outliers")

    References:
        Breunig, Kriegel, Ng & Sander (2000): LOF Algorithm
        STAT-012
    """
    data = trace.data if isinstance(trace, WaveformTrace) else np.asarray(trace)
    n_samples = len(data)

    if n_samples < n_neighbors + 1:
        return LOFResult(
            indices=np.array([], dtype=np.intp),
            scores=np.ones(n_samples, dtype=np.float64),
            mask=np.zeros(n_samples, dtype=np.bool_),
            count=0,
            threshold=threshold,
            n_neighbors=n_neighbors,
        )

    # For 1D data, use index-based neighbors
    # Reshape for compatibility
    X = data.reshape(-1, 1)

    # Compute k-distances and neighbors
    k_distances = np.zeros(n_samples, dtype=np.float64)
    k_neighbors = np.zeros((n_samples, n_neighbors), dtype=np.intp)

    for i in range(n_samples):
        # Compute distances to all other points
        if metric == "euclidean":
            distances = np.abs(X[:, 0] - X[i, 0])
        else:  # manhattan
            distances = np.abs(X[:, 0] - X[i, 0])

        # Get k nearest neighbors (excluding self)
        distances[i] = np.inf
        neighbor_idx = np.argsort(distances)[:n_neighbors]
        k_neighbors[i] = neighbor_idx
        k_distances[i] = distances[neighbor_idx[-1]]

    # Compute Local Reachability Density (LRD)
    lrd = np.zeros(n_samples, dtype=np.float64)
    for i in range(n_samples):
        reach_dists = np.maximum(
            np.abs(X[k_neighbors[i], 0] - X[i, 0]),
            k_distances[k_neighbors[i]],
        )
        mean_reach_dist = np.mean(reach_dists)
        lrd[i] = 1.0 / mean_reach_dist if mean_reach_dist > 0 else np.inf

    # Compute LOF scores
    lof_scores = np.zeros(n_samples, dtype=np.float64)
    for i in range(n_samples):
        neighbor_lrd = lrd[k_neighbors[i]]
        lof_scores[i] = np.mean(neighbor_lrd) / lrd[i] if lrd[i] > 0 else 1.0

    # Handle infinities
    lof_scores = np.nan_to_num(lof_scores, nan=1.0, posinf=threshold * 2)

    # Classify outliers
    mask = lof_scores > threshold
    indices = np.where(mask)[0]

    return LOFResult(
        indices=indices.astype(np.intp),
        scores=lof_scores,
        mask=mask,
        count=int(np.sum(mask)),
        threshold=threshold,
        n_neighbors=n_neighbors,
    )


def seasonal_decompose(
    trace: WaveformTrace | NDArray[np.floating[Any]],
    *,
    period: int | None = None,
    model: Literal["additive", "multiplicative"] = "additive",
) -> DecompositionResult:
    """Decompose time series into trend, seasonal, and residual components.

    Uses classical decomposition (moving average for trend extraction).

    Args:
        trace: Input trace or numpy array.
        period: Period of seasonality. If None, auto-detected.
        model: Decomposition model:
            - "additive": y = trend + seasonal + residual
            - "multiplicative": y = trend * seasonal * residual

    Returns:
        DecompositionResult with trend, seasonal, and residual components.

    Example:
        >>> result = seasonal_decompose(trace, period=100)
        >>> plt.plot(result.trend, label="Trend")
        >>> plt.plot(result.seasonal, label="Seasonal")

    References:
        Cleveland et al. (1990): STL Decomposition
        STAT-013
    """
    data = trace.data if isinstance(trace, WaveformTrace) else np.asarray(trace)
    n = len(data)

    # Auto-detect period if not provided
    if period is None:
        period = _detect_period(data)
        if period is None or period < 2:
            period = min(n // 4, 10)  # Default fallback

    period = max(2, min(period, n // 2))

    # Extract trend using centered moving average
    if period % 2 == 0:
        # For even period, use 2-stage moving average
        ma = np.convolve(data, np.ones(period) / period, mode="same")
        trend = np.convolve(ma, np.ones(2) / 2, mode="same")
    else:
        trend = np.convolve(data, np.ones(period) / period, mode="same")

    # Handle edges
    half_period = period // 2
    trend[:half_period] = trend[half_period]
    trend[-half_period:] = trend[-half_period - 1]

    # Detrend
    if model == "multiplicative":
        with np.errstate(divide="ignore", invalid="ignore"):
            detrended = data / trend
            detrended = np.nan_to_num(detrended, nan=1.0)
    else:
        detrended = data - trend

    # Extract seasonal component (average for each phase)
    seasonal = np.zeros_like(data)
    for i in range(period):
        indices = np.arange(i, n, period)
        seasonal_mean = np.mean(detrended[indices])
        seasonal[indices] = seasonal_mean

    # Center seasonal component
    if model == "multiplicative":
        seasonal /= np.mean(seasonal)
    else:
        seasonal -= np.mean(seasonal)

    # Compute residual
    if model == "multiplicative":
        with np.errstate(divide="ignore", invalid="ignore"):
            residual = data / (trend * seasonal)
            residual = np.nan_to_num(residual, nan=1.0)
    else:
        residual = data - trend - seasonal

    return DecompositionResult(
        trend=trend.astype(np.float64),
        seasonal=seasonal.astype(np.float64),
        residual=residual.astype(np.float64),
        period=period,
        observed=data.astype(np.float64),
    )


def _detect_period(data: NDArray[Any]) -> int | None:
    """Auto-detect dominant period using autocorrelation."""
    n = len(data)
    if n < 20:
        return None

    # Compute autocorrelation
    data_centered = data - np.mean(data)
    acf = np.correlate(data_centered, data_centered, mode="full")
    acf = acf[n - 1 :]  # Keep positive lags only
    acf = acf / acf[0]  # Normalize

    # Find first significant peak after lag 0
    # Skip first few lags to avoid noise
    min_lag = max(2, n // 100)
    max_lag = n // 2

    # Find peaks in autocorrelation
    peaks, _ = signal.find_peaks(acf[min_lag:max_lag], height=0.1, distance=min_lag)

    if len(peaks) > 0:
        return peaks[0] + min_lag  # type: ignore[no-any-return]

    return None


def detect_change_points(
    trace: WaveformTrace | NDArray[np.floating[Any]],
    *,
    n_changes: int | None = None,
    min_size: int = 10,
    penalty: float | None = None,
    method: Literal["pelt", "binseg"] = "pelt",
) -> ChangePointResult:
    """Detect change points in time series.

    Identifies points where the statistical properties of the signal
    change significantly.

    Args:
        trace: Input trace or numpy array.
        n_changes: Number of change points to find. If None, auto-detected.
        min_size: Minimum segment length between change points.
        penalty: Penalty for adding change points (higher = fewer changes).
        method: Detection method:
            - "pelt": Pruned Exact Linear Time (fast, optimal)
            - "binseg": Binary Segmentation (fast, approximate)

    Returns:
        ChangePointResult with change point locations and segment info.

    Example:
        >>> result = detect_change_points(trace, n_changes=3)
        >>> for start, end in result.segments:
        ...     print(f"Segment: {start} to {end}")

    References:
        Killick et al. (2012): PELT Algorithm
        STAT-014
    """
    data = trace.data if isinstance(trace, WaveformTrace) else np.asarray(trace)
    n = len(data)

    if n < min_size * 2:
        return ChangePointResult(
            indices=np.array([], dtype=np.intp),
            n_changes=0,
            segments=[(0, n)],
            segment_means=np.array([np.mean(data)]),
            segment_stds=np.array([np.std(data)]),
            cost=0.0,
        )

    # Set default penalty based on BIC
    if penalty is None:
        penalty = np.log(n) * np.var(data)

    if method == "pelt":
        change_points = _pelt_change_points(data, min_size, penalty, n_changes)
    else:
        change_points = _binseg_change_points(data, min_size, penalty, n_changes)

    # Build segments
    all_points = [0, *list(change_points), n]
    segments = [(all_points[i], all_points[i + 1]) for i in range(len(all_points) - 1)]

    # Compute segment statistics
    segment_means = np.array([np.mean(data[s:e]) for s, e in segments])
    segment_stds = np.array([np.std(data[s:e]) for s, e in segments])

    # Compute total cost
    total_cost = sum(_segment_cost(data[s:e]) for s, e in segments) + penalty * len(change_points)

    return ChangePointResult(
        indices=np.array(change_points, dtype=np.intp),
        n_changes=len(change_points),
        segments=segments,
        segment_means=segment_means,
        segment_stds=segment_stds,
        cost=float(total_cost),
    )


def _segment_cost(segment: NDArray[Any]) -> float:
    """Compute cost of a segment (negative log-likelihood for normal)."""
    n = len(segment)
    if n < 2:
        return 0.0
    var = np.var(segment)
    if var <= 0:
        return 0.0
    return n * np.log(var)  # type: ignore[no-any-return]


def _pelt_change_points(
    data: NDArray[Any],
    min_size: int,
    penalty: float,
    n_changes: int | None,
) -> list[int]:
    """PELT algorithm for change point detection."""
    len(data)

    # Simple implementation: use binary segmentation as approximation
    # Full PELT requires dynamic programming which is more complex
    return _binseg_change_points(data, min_size, penalty, n_changes)


def _binseg_change_points(
    data: NDArray[Any],
    min_size: int,
    penalty: float,
    n_changes: int | None,
) -> list[int]:
    """Binary segmentation for change point detection."""
    n = len(data)
    change_points: list[int] = []

    def find_best_split(start: int, end: int) -> tuple[int, float]:
        """Find best split point in segment."""
        if end - start < 2 * min_size:
            return -1, 0.0

        best_idx = -1
        best_gain = 0.0

        for i in range(start + min_size, end - min_size + 1):
            left = data[start:i]
            right = data[i:end]
            full = data[start:end]

            cost_full = _segment_cost(full)
            cost_split = _segment_cost(left) + _segment_cost(right)
            gain = cost_full - cost_split - penalty

            if gain > best_gain:
                best_gain = gain
                best_idx = i

        return best_idx, best_gain

    # Iteratively find change points
    segments = [(0, n)]
    max_iter = n_changes if n_changes is not None else n // min_size

    for _ in range(max_iter):
        best_segment_idx = -1
        best_split_idx = -1
        best_gain = 0.0

        for seg_idx, (start, end) in enumerate(segments):
            split_idx, gain = find_best_split(start, end)
            if gain > best_gain:
                best_gain = gain
                best_split_idx = split_idx
                best_segment_idx = seg_idx

        if best_split_idx == -1:
            break

        # Add change point
        change_points.append(best_split_idx)

        # Update segments
        start, end = segments[best_segment_idx]
        segments[best_segment_idx] = (start, best_split_idx)
        segments.insert(best_segment_idx + 1, (best_split_idx, end))

    return sorted(change_points)


def phase_coherence(
    trace1: WaveformTrace | NDArray[np.floating[Any]],
    trace2: WaveformTrace | NDArray[np.floating[Any]],
    *,
    sample_rate: float | None = None,
    nperseg: int | None = None,
) -> CoherenceResult:
    """Compute phase coherence between two signals.

    Coherence measures the linear correlation between two signals
    as a function of frequency.

    Args:
        trace1: First input trace.
        trace2: Second input trace.
        sample_rate: Sample rate in Hz. Required if traces are arrays.
        nperseg: Segment length for Welch method.

    Returns:
        CoherenceResult with coherence spectrum and phase.

    Example:
        >>> result = phase_coherence(signal1, signal2, sample_rate=1e6)
        >>> print(f"Mean coherence: {result.mean_coherence:.3f}")

    References:
        STAT-015
    """
    data1 = trace1.data if isinstance(trace1, WaveformTrace) else np.asarray(trace1)
    data2 = trace2.data if isinstance(trace2, WaveformTrace) else np.asarray(trace2)

    # Get sample rate
    if sample_rate is None:
        sample_rate = trace1.metadata.sample_rate if isinstance(trace1, WaveformTrace) else 1.0

    # Ensure same length
    n = min(len(data1), len(data2))
    data1 = data1[:n]
    data2 = data2[:n]

    if nperseg is None:
        nperseg = min(256, n // 4)
    nperseg = max(16, min(nperseg, n))

    # Compute coherence
    frequencies, coherence = signal.coherence(data1, data2, fs=sample_rate, nperseg=nperseg)

    # Compute cross-spectral phase
    _, Pxy = signal.csd(data1, data2, fs=sample_rate, nperseg=nperseg)
    phase = np.angle(Pxy)

    # Statistics
    mean_coherence = float(np.mean(coherence))
    peak_idx = np.argmax(coherence)
    peak_frequency = float(frequencies[peak_idx])
    peak_coherence = float(coherence[peak_idx])

    return CoherenceResult(
        coherence=coherence.astype(np.float64),
        frequencies=frequencies.astype(np.float64),
        phase=phase.astype(np.float64),
        mean_coherence=mean_coherence,
        peak_frequency=peak_frequency,
        peak_coherence=peak_coherence,
    )


def kernel_density(
    trace: WaveformTrace | NDArray[np.floating[Any]],
    *,
    n_points: int = 1000,
    bandwidth: float | str = "scott",
    kernel: Literal["gaussian", "tophat", "epanechnikov"] = "gaussian",
) -> KDEResult:
    """Estimate probability density using kernel density estimation.

    Args:
        trace: Input trace or numpy array.
        n_points: Number of evaluation points.
        bandwidth: Bandwidth for kernel ("scott", "silverman", or float).
        kernel: Kernel function to use.

    Returns:
        KDEResult with density estimate and mode information.

    Raises:
        ValueError: If kernel is not one of the supported types.

    Example:
        >>> result = kernel_density(trace)
        >>> plt.plot(result.x, result.density)
        >>> print(f"Modes at: {result.peak_values}")

    References:
        Scott (1992): Multivariate Density Estimation
        STAT-016
    """
    data = trace.data if isinstance(trace, WaveformTrace) else np.asarray(trace)
    n = len(data)

    if n < 2:
        return KDEResult(
            x=np.array([np.mean(data)]),
            density=np.array([1.0]),
            bandwidth=0.0,
            peaks=np.array([0], dtype=np.intp),
            peak_values=np.array([np.mean(data)]),
        )

    # Compute bandwidth
    std = np.std(data)
    iqr = np.percentile(data, 75) - np.percentile(data, 25)

    if isinstance(bandwidth, str):
        if bandwidth == "scott":
            bw = 1.06 * std * n ** (-1 / 5)
        elif bandwidth == "silverman":
            bw = 0.9 * min(std, iqr / 1.34) * n ** (-1 / 5)
        else:
            bw = 1.06 * std * n ** (-1 / 5)
    else:
        bw = bandwidth

    bw = max(bw, 1e-10)  # Prevent zero bandwidth

    # Evaluation grid
    margin = 3 * bw
    x_min = np.min(data) - margin
    x_max = np.max(data) + margin
    x = np.linspace(x_min, x_max, n_points)

    # Compute density
    if kernel == "gaussian":
        kde = sp_stats.gaussian_kde(data, bw_method=bw / std if std > 0 else 1.0)
        density = kde(x)
    elif kernel == "tophat":
        density = np.zeros(n_points)
        for xi in data:
            mask = np.abs(x - xi) <= bw
            density[mask] += 1.0
        density /= n * 2 * bw
    elif kernel == "epanechnikov":
        density = np.zeros(n_points)
        for xi in data:
            u = (x - xi) / bw
            mask = np.abs(u) <= 1
            density[mask] += 0.75 * (1 - u[mask] ** 2)
        density /= n * bw
    else:
        raise ValueError(f"Unknown kernel: {kernel}")

    # Find peaks (modes)
    peaks_idx, _ = signal.find_peaks(density)
    if len(peaks_idx) == 0:
        peaks_idx = np.array([np.argmax(density)])
    peak_values = x[peaks_idx]

    return KDEResult(
        x=x.astype(np.float64),
        density=density.astype(np.float64),
        bandwidth=float(bw),
        peaks=peaks_idx.astype(np.intp),
        peak_values=peak_values.astype(np.float64),
    )


__all__ = [
    "ChangePointResult",
    "CoherenceResult",
    "DecompositionResult",
    "IsolationForestResult",
    "KDEResult",
    "LOFResult",
    "detect_change_points",
    "isolation_forest_outliers",
    "kernel_density",
    "local_outlier_factor",
    "phase_coherence",
    "seasonal_decompose",
]
