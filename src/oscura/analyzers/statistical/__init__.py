"""Statistical analysis module for reverse engineering.

This module combines general statistical analysis (from oscura.analyzers.statistics)
with additional entropy-based and binary data analysis functions for protocol
reverse engineering.

Use cases:
- Binary data analysis: shannon_entropy, byte_frequency_distribution
- Checksum detection: detect_checksum_fields, verify_checksums
- Data classification: classify_data_type, detect_encrypted_regions
- Plus all functions from oscura.analyzers.statistics

For general signal statistics without reverse engineering features, use
oscura.analyzers.statistics instead. See IMPORT-PATHS.md for details.

Requirements:
- RE-ENT-002: Byte Frequency Distribution
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

from oscura.analyzers.statistics import (
    ChangePointResult,
    CoherenceResult,
    CrossCorrelationResult,
    DecompositionResult,
    IsolationForestResult,
    KDEResult,
    LOFResult,
    OutlierResult,
    TrendResult,
    autocorrelation,
    basic_stats,
    change_point_detection,
    coherence,
    correlation_coefficient,
    cross_correlation,
    detect_change_points,
    detect_drift_segments,
    detect_outliers,
    detect_trend,
    detrend,
    find_periodicity,
    iqr_outliers,
    isolation_forest_outliers,
    kernel_density,
    local_outlier_factor,
    modified_zscore_outliers,
    moving_average,
    percentiles,
    phase_coherence,
    piecewise_linear_fit,
    quartiles,
    remove_outliers,
    running_stats,
    seasonal_decompose,
    summary_stats,
    weighted_mean,
    zscore_outliers,
)

from .checksum import (
    ChecksumCandidate,
    ChecksumDetectionResult,
    ChecksumDetector,
    ChecksumMatch,
    compute_checksum,
    crc8,
    crc16_ccitt,
    crc16_ibm,
    crc32,
    detect_checksum_fields,
    identify_checksum_algorithm,
    sum8,
    sum16,
    verify_checksums,
    xor_checksum,
)
from .classification import (
    ClassificationResult,
    DataClassifier,
    RegionClassification,
    classify_data_type,
    detect_compressed_regions,
    detect_encrypted_regions,
    detect_padding_regions,
    detect_text_regions,
    segment_by_type,
)
from .entropy import (
    ByteFrequencyResult,
    CompressionIndicator,
    EntropyAnalyzer,
    EntropyResult,
    EntropyTransition,
    FrequencyAnomalyResult,
    bit_entropy,
    byte_frequency_distribution,
    classify_by_entropy,
    compare_byte_distributions,
    detect_compression_indicators,
    detect_entropy_transitions,
    detect_frequency_anomalies,
    entropy_histogram,
    entropy_profile,
    shannon_entropy,
    sliding_byte_frequency,
    sliding_entropy,
)
from .ngrams import (
    NGramAnalyzer,
    NgramComparison,
    NgramProfile,
    compare_ngram_profiles,
    extract_ngrams,
    find_common_ngrams,
    find_unusual_ngrams,
    ngram_entropy,
    ngram_frequencies,
    ngram_frequency,
    ngram_heatmap,
)

# Function alias for test compatibility
calculate_entropy = shannon_entropy
entropy = shannon_entropy

# Type alias for input data (matching entropy.py)
DataType = Union[bytes, bytearray, "NDArray[np.uint8]"]


def entropy_windowed(data: DataType, window_size: int = 256, step: int = 1) -> NDArray[np.float64]:
    """Windowed entropy calculation (alias for sliding_entropy)."""
    return sliding_entropy(data, window_size=window_size, step=step)


__all__ = [
    # RE-ENT-002: Byte Frequency Distribution
    "ByteFrequencyResult",
    # Result types
    "ChangePointResult",
    "ChecksumCandidate",
    "ChecksumDetectionResult",
    "ChecksumDetector",
    "ChecksumMatch",
    "ClassificationResult",
    "CoherenceResult",
    "CompressionIndicator",
    "CrossCorrelationResult",
    "DataClassifier",
    "DecompositionResult",
    "EntropyAnalyzer",
    "EntropyResult",
    "EntropyTransition",
    "FrequencyAnomalyResult",
    "IsolationForestResult",
    "KDEResult",
    "LOFResult",
    "NGramAnalyzer",
    "NgramComparison",
    "NgramProfile",
    "OutlierResult",
    "RegionClassification",
    "TrendResult",
    # Correlation
    "autocorrelation",
    # Basic statistics
    "basic_stats",
    "bit_entropy",
    "byte_frequency_distribution",
    "calculate_entropy",
    "change_point_detection",
    "classify_by_entropy",
    "classify_data_type",
    "coherence",
    "compare_byte_distributions",
    "compare_ngram_profiles",
    "compute_checksum",
    "correlation_coefficient",
    "crc8",
    "crc16_ccitt",
    "crc16_ibm",
    "crc32",
    "cross_correlation",
    # Advanced (STAT-014)
    "detect_change_points",
    "detect_checksum_fields",
    "detect_compressed_regions",
    "detect_compression_indicators",
    "detect_drift_segments",
    "detect_encrypted_regions",
    "detect_entropy_transitions",
    "detect_frequency_anomalies",
    "detect_outliers",
    "detect_padding_regions",
    "detect_text_regions",
    # Trend
    "detect_trend",
    "detrend",
    "entropy",
    "entropy_histogram",
    "entropy_profile",
    "entropy_windowed",
    "extract_ngrams",
    "find_common_ngrams",
    "find_periodicity",
    "find_unusual_ngrams",
    "identify_checksum_algorithm",
    "iqr_outliers",
    # Advanced (STAT-011)
    "isolation_forest_outliers",
    # Advanced (STAT-016)
    "kernel_density",
    # Advanced (STAT-012)
    "local_outlier_factor",
    "modified_zscore_outliers",
    "moving_average",
    "ngram_entropy",
    "ngram_frequencies",
    "ngram_frequency",
    "ngram_heatmap",
    "percentiles",
    # Advanced (STAT-015)
    "phase_coherence",
    "piecewise_linear_fit",
    "quartiles",
    "remove_outliers",
    "running_stats",
    # Advanced (STAT-013)
    "seasonal_decompose",
    "segment_by_type",
    "shannon_entropy",
    "sliding_byte_frequency",
    "sliding_entropy",
    "sum8",
    "sum16",
    "summary_stats",
    "verify_checksums",
    "weighted_mean",
    "xor_checksum",
    # Outlier detection
    "zscore_outliers",
]
