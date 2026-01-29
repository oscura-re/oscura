"""Distribution analysis functions.

This module provides histogram generation and distribution
shape metrics for signal analysis.


Example:
    >>> from oscura.analyzers.statistics.distribution import histogram, distribution_metrics
    >>> counts, edges = histogram(trace, bins=50)
    >>> metrics = distribution_metrics(trace)
    >>> print(f"Skewness: {metrics['skewness']}")

References:
    scipy.stats for distribution analysis
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from scipy import stats as sp_stats

from oscura.core.types import WaveformTrace

if TYPE_CHECKING:
    from numpy.typing import NDArray


def histogram(
    trace: WaveformTrace | NDArray[np.floating[Any]],
    bins: int | str | NDArray[np.floating[Any]] = "auto",
    *,
    density: bool = False,
    range: tuple[float, float] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Generate amplitude histogram.

    Args:
        trace: Input trace or numpy array.
        bins: Number of bins, binning strategy ("auto", "fd", "sturges"),
              or explicit bin edges.
        density: If True, normalize to probability density.
        range: (min, max) range for histogram.

    Returns:
        (counts, bin_edges) - Histogram counts and bin edges.

    Example:
        >>> counts, edges = histogram(trace, bins=50)
        >>> centers = (edges[:-1] + edges[1:]) / 2
        >>> plt.bar(centers, counts, width=np.diff(edges))
    """
    data = trace.data if isinstance(trace, WaveformTrace) else trace

    counts, edges = np.histogram(data, bins=bins, density=density, range=range)

    return counts.astype(np.float64), edges.astype(np.float64)


def distribution_metrics(
    trace: WaveformTrace | NDArray[np.floating[Any]],
) -> dict[str, float]:
    """Compute distribution shape metrics.

    Calculates skewness, kurtosis, and crest factor.

    Args:
        trace: Input trace or numpy array.

    Returns:
        Dictionary with distribution metrics:
            - skewness: Distribution asymmetry (-=left, +=right tail)
            - kurtosis: Tail weight (>3=heavy, <3=light tails)
            - excess_kurtosis: Kurtosis - 3 (0 for normal)
            - crest_factor: Peak / RMS ratio
            - crest_factor_db: Crest factor in dB

    Example:
        >>> metrics = distribution_metrics(trace)
        >>> print(f"Skewness: {metrics['skewness']:.3f}")
        >>> print(f"Kurtosis: {metrics['kurtosis']:.3f}")

    References:
        Fisher's definitions for skewness and kurtosis
    """
    data = trace.data if isinstance(trace, WaveformTrace) else trace

    # Skewness (Fisher's definition)
    skewness = float(sp_stats.skew(data))

    # Kurtosis (Fisher's definition gives excess kurtosis)
    excess_kurtosis = float(sp_stats.kurtosis(data, fisher=True))
    kurtosis = excess_kurtosis + 3  # Pearson's definition

    # Crest factor
    rms = np.sqrt(np.mean(data**2))
    peak = np.max(np.abs(data))

    if rms > 0:
        crest_factor = peak / rms
        crest_factor_db = 20 * np.log10(crest_factor)
    else:
        crest_factor = float("inf")
        crest_factor_db = float("inf")

    return {
        "skewness": skewness,
        "kurtosis": kurtosis,
        "excess_kurtosis": excess_kurtosis,
        "crest_factor": float(crest_factor),
        "crest_factor_db": float(crest_factor_db),
    }


def moment(
    trace: WaveformTrace | NDArray[np.floating[Any]],
    order: int,
    *,
    central: bool = True,
) -> float:
    """Compute statistical moment.

    Args:
        trace: Input trace or numpy array.
        order: Moment order (1=mean, 2=variance, 3=skewness, 4=kurtosis).
        central: If True, compute central moment (about mean).

    Returns:
        Moment value.

    Example:
        >>> m2 = moment(trace, 2)  # Variance
        >>> m3 = moment(trace, 3)  # Third central moment
    """
    data = trace.data if isinstance(trace, WaveformTrace) else trace

    if central:
        return float(sp_stats.moment(data, moment=order))
    else:
        return float(np.mean(data**order))


def fit_distribution(
    trace: WaveformTrace | NDArray[np.floating[Any]],
    distribution: Literal["normal", "lognormal", "exponential", "uniform"] = "normal",
) -> dict[str, float]:
    """Fit distribution to data and compute goodness of fit.

    Args:
        trace: Input trace or numpy array.
        distribution: Distribution to fit.

    Returns:
        Dictionary with fitted parameters and fit quality:
            - params: Distribution parameters
            - ks_statistic: Kolmogorov-Smirnov test statistic
            - p_value: KS test p-value

    Raises:
        ValueError: If distribution is not one of the supported types.

    Example:
        >>> fit = fit_distribution(trace, "normal")
        >>> print(f"Mean: {fit['loc']}, Std: {fit['scale']}")
        >>> print(f"Normality p-value: {fit['p_value']}")
    """
    data = trace.data if isinstance(trace, WaveformTrace) else trace

    result: dict[str, float] = {}

    if distribution == "normal":
        loc, scale = sp_stats.norm.fit(data)
        result["loc"] = float(loc)
        result["scale"] = float(scale)
        ks_stat, p_value = sp_stats.kstest(data, "norm", args=(loc, scale))

    elif distribution == "lognormal":
        shape, loc, scale = sp_stats.lognorm.fit(data, floc=0)
        result["shape"] = float(shape)
        result["loc"] = float(loc)
        result["scale"] = float(scale)
        ks_stat, p_value = sp_stats.kstest(data, "lognorm", args=(shape, loc, scale))

    elif distribution == "exponential":
        loc, scale = sp_stats.expon.fit(data)
        result["loc"] = float(loc)
        result["scale"] = float(scale)
        ks_stat, p_value = sp_stats.kstest(data, "expon", args=(loc, scale))

    elif distribution == "uniform":
        loc, scale = sp_stats.uniform.fit(data)
        result["loc"] = float(loc)
        result["scale"] = float(scale)
        ks_stat, p_value = sp_stats.kstest(data, "uniform", args=(loc, scale))

    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    result["ks_statistic"] = float(ks_stat)
    result["p_value"] = float(p_value)

    return result


def normality_test(
    trace: WaveformTrace | NDArray[np.floating[Any]],
    method: Literal["shapiro", "dagostino", "anderson"] = "shapiro",
) -> dict[str, float]:
    """Test for normality.

    Args:
        trace: Input trace or numpy array.
        method: Test method:
            - "shapiro": Shapiro-Wilk test (best for small samples)
            - "dagostino": D'Agostino-Pearson test
            - "anderson": Anderson-Darling test

    Returns:
        Dictionary with test results:
            - statistic: Test statistic
            - p_value: P-value (probability data is normal)
            - is_normal: True if p_value > 0.05

    Raises:
        ValueError: If method is not one of the supported types.

    Example:
        >>> result = normality_test(trace)
        >>> if result['is_normal']:
        ...     print("Data appears normally distributed")
    """
    data = trace.data if isinstance(trace, WaveformTrace) else trace

    if method == "shapiro":
        # Shapiro-Wilk limited to 5000 samples
        if len(data) > 5000:
            data = np.random.choice(data, 5000, replace=False)
        stat, p_value = sp_stats.shapiro(data)

    elif method == "dagostino":
        stat, p_value = sp_stats.normaltest(data)

    elif method == "anderson":
        # Use new SciPy 1.17+ API with method parameter
        result = sp_stats.anderson(data, dist="norm", method="interpolate")
        stat = result.statistic
        # SciPy 1.17+ provides p-value directly
        p_value = result.pvalue

    else:
        raise ValueError(f"Unknown method: {method}")

    return {
        "statistic": float(stat),
        "p_value": float(p_value),
        "is_normal": p_value > 0.05,
    }


def bimodality_coefficient(
    trace: WaveformTrace | NDArray[np.floating[Any]],
) -> float:
    """Compute bimodality coefficient.

    Values > 0.555 suggest bimodal distribution.

    Args:
        trace: Input trace or numpy array.

    Returns:
        Bimodality coefficient (0-1).

    Example:
        >>> bc = bimodality_coefficient(trace)
        >>> if bc > 0.555:
        ...     print("Distribution appears bimodal")
    """
    data = trace.data if isinstance(trace, WaveformTrace) else trace

    n = len(data)
    skewness = sp_stats.skew(data)
    kurtosis = sp_stats.kurtosis(data, fisher=True)  # Excess kurtosis

    # Bimodality coefficient formula
    bc = (skewness**2 + 1) / (kurtosis + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))

    return float(bc)


__all__ = [
    "bimodality_coefficient",
    "distribution_metrics",
    "fit_distribution",
    "histogram",
    "moment",
    "normality_test",
]
