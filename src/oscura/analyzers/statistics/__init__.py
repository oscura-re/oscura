"""Statistical analysis package.

Provides statistical measures, outlier detection, correlation analysis,
trend detection, and advanced statistical methods for signal data.
"""

from oscura.analyzers.statistics.advanced import (
    ChangePointResult,
    CoherenceResult,
    DecompositionResult,
    IsolationForestResult,
    KDEResult,
    LOFResult,
    detect_change_points,
    isolation_forest_outliers,
    kernel_density,
    local_outlier_factor,
    phase_coherence,
    seasonal_decompose,
)
from oscura.analyzers.statistics.basic import (
    basic_stats,
    percentiles,
    quartiles,
    running_stats,
    summary_stats,
    weighted_mean,
)
from oscura.analyzers.statistics.correlation import (
    CrossCorrelationResult,
    autocorrelation,
    coherence,
    correlation_coefficient,
    cross_correlation,
    find_periodicity,
)
from oscura.analyzers.statistics.distribution import (
    bimodality_coefficient,
    distribution_metrics,
    fit_distribution,
    histogram,
    moment,
    normality_test,
)
from oscura.analyzers.statistics.outliers import (
    OutlierResult,
    detect_outliers,
    iqr_outliers,
    modified_zscore_outliers,
    remove_outliers,
    zscore_outliers,
)
from oscura.analyzers.statistics.trend import (
    TrendResult,
    change_point_detection,
    detect_drift_segments,
    detect_trend,
    detrend,
    moving_average,
    piecewise_linear_fit,
)

__all__ = [
    # Result types - Advanced (STAT-011 to STAT-016)
    "ChangePointResult",
    "CoherenceResult",
    "CrossCorrelationResult",
    "DecompositionResult",
    "IsolationForestResult",
    "KDEResult",
    "LOFResult",
    "OutlierResult",
    "TrendResult",
    # Correlation
    "autocorrelation",
    # Basic statistics
    "basic_stats",
    "bimodality_coefficient",
    "change_point_detection",
    "coherence",
    "correlation_coefficient",
    "cross_correlation",
    # Advanced (STAT-014)
    "detect_change_points",
    "detect_drift_segments",
    "detect_outliers",
    # Trend
    "detect_trend",
    "detrend",
    # Distribution
    "distribution_metrics",
    "find_periodicity",
    "fit_distribution",
    "histogram",
    "iqr_outliers",
    # Advanced (STAT-011)
    "isolation_forest_outliers",
    # Advanced (STAT-016)
    "kernel_density",
    # Advanced (STAT-012)
    "local_outlier_factor",
    "modified_zscore_outliers",
    "moment",
    "moving_average",
    "normality_test",
    "percentiles",
    # Advanced (STAT-015)
    "phase_coherence",
    "piecewise_linear_fit",
    "quartiles",
    "remove_outliers",
    "running_stats",
    # Advanced (STAT-013)
    "seasonal_decompose",
    "summary_stats",
    "weighted_mean",
    # Outlier detection
    "zscore_outliers",
]
