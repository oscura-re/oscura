"""Pattern Detection & Analysis module for Oscura.

This module provides comprehensive pattern detection and analysis capabilities
for digital signals and binary data, including:

- Periodic pattern detection (autocorrelation, FFT, suffix array)
- Repeating sequence detection (n-grams, LRS, approximate matching)
- Automatic signature discovery (headers, delimiters, magic bytes)
- Pattern clustering by similarity (Hamming, edit distance, hierarchical)
- Binary regex pattern matching
- Multi-pattern search (Aho-Corasick)
- Fuzzy/approximate pattern matching
- Pattern learning and discovery

    - RE-PAT-001: Binary Regex Pattern Matching
    - RE-PAT-002: Multi-Pattern Search (Aho-Corasick)
    - RE-PAT-003: Fuzzy Pattern Matching
    - RE-PAT-004: Pattern Learning and Discovery

Author: Oscura Development Team
"""

# Periodic pattern detection (PAT-001)
# Pattern clustering (PAT-004)
from __future__ import annotations

from .clustering import (
    ClusteringResult,
    ClusterResult,
    analyze_cluster,
    cluster_by_edit_distance,
    cluster_by_hamming,
    cluster_hierarchical,
    compute_distance_matrix,
)

# Signature discovery (PAT-003)
from .discovery import (
    CandidateSignature,
    SignatureDiscovery,
    discover_signatures,
    find_delimiter_candidates,
    find_header_candidates,
)

# RE-PAT-004: Pattern Learning and Discovery
from .learning import (
    LearnedPattern,
    NgramModel,
    PatternLearner,
    StructureHypothesis,
    find_recurring_structures,
    infer_structure,
    learn_patterns_from_data,
)

# RE-PAT-001, RE-PAT-002, RE-PAT-003: Advanced pattern matching
from .matching import (
    AhoCorasickMatcher,
    # Classes
    BinaryRegex,
    FuzzyMatcher,
    FuzzyMatchResult,
    # Data classes
    PatternMatchResult,
    # RE-PAT-001: Binary Regex
    binary_regex_search,
    count_pattern_occurrences,
    # Utilities
    find_pattern_positions,
    find_similar_sequences,
    # RE-PAT-003: Fuzzy Matching
    fuzzy_search,
    # RE-PAT-002: Multi-Pattern Search
    multi_pattern_search,
)
from .periodic import (
    PeriodicPatternDetector,
    PeriodResult,
    detect_period,
    detect_periods_autocorr,
    detect_periods_fft,
    validate_period,
)

# Alias for backward compatibility
detect_period_autocorr = detect_periods_autocorr
detect_period_fft = detect_periods_fft

# Repeating sequence detection (PAT-002)
# Motif detection functions (aliases for test compatibility)
from typing import TYPE_CHECKING, Any, cast

from .sequences import (
    NgramResult,
    RepeatingSequence,
    find_approximate_repeats,
    find_frequent_ngrams,
    find_longest_repeat,
    find_repeating_sequences,
)

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


def find_motifs(
    data: Any, motif_length: int = 8, max_distance: float = 0.1
) -> list[RepeatingSequence]:
    """Find motifs (repeating patterns) in data.

    This is an alias for find_repeating_sequences for test compatibility.

    Args:
        data: Input data array.
        motif_length: Length of motifs to find.
        max_distance: Maximum distance for fuzzy matching (unused).

    Returns:
        List of RepeatingSequence objects.
    """
    import numpy as np

    data = np.asarray(data)
    results = find_repeating_sequences(data, min_length=motif_length, min_count=2)
    return results


def extract_motif(data: Any, start: int = 0, length: int | None = None) -> NDArray[np.generic]:
    """Extract a motif from data.

    Args:
        data: Input data array. If start and length not provided, attempts to detect and extract
            the first repeating motif automatically.
        start: Start index for extraction (default: 0).
        length: Length to extract. If None, attempts to detect motif length automatically.

    Returns:
        Extracted motif as numpy array.

    Raises:
        ValueError: If automatic detection fails and no length specified.
    """
    import numpy as np

    data_arr = np.asarray(data)

    # If length not specified, try to detect motif automatically
    if length is None:
        # Try to find repeating pattern using period detection
        try:
            period_result = detect_period(data_arr)
            if period_result is not None and hasattr(period_result, "period"):
                length = int(period_result.period)
            else:
                # Default to 8 samples if detection fails
                length = min(8, len(data_arr))
        except Exception:
            # Fallback to reasonable default
            length = min(8, len(data_arr))

    # Ensure we don't exceed array bounds
    end = min(start + length, len(data_arr))
    result = data_arr[start:end]
    return result


def detect_anomalies(data: Any, threshold: float = 3.0) -> list[int]:
    """Detect anomalies in data using z-score.

    Args:
        data: Input data array.
        threshold: Z-score threshold for anomaly detection.

    Returns:
        List of anomaly indices.
    """
    import numpy as np

    data_arr = np.asarray(data)
    mean = np.mean(data_arr)
    std = np.std(data_arr)
    if std == 0:
        return []

    z_scores = np.abs((data_arr - mean) / std)
    indices = np.where(z_scores > threshold)[0].tolist()
    return cast("list[int]", indices)


def cluster_patterns(patterns: Any, method: str = "hamming") -> ClusteringResult:
    """Cluster patterns by similarity.

    Args:
        patterns: List of patterns to cluster.
        method: Clustering method ('hamming' or 'edit').

    Returns:
        ClusteringResult with cluster assignments.
    """
    if method == "hamming":
        return cluster_by_hamming(patterns)
    else:
        return cluster_by_edit_distance(patterns)


def pattern_similarity(pattern1: Any, pattern2: Any) -> float:
    """Calculate similarity between two patterns.

    Args:
        pattern1: First pattern.
        pattern2: Second pattern.

    Returns:
        Similarity score (0-1, 1 = identical).
    """
    import numpy as np

    p1 = np.asarray(pattern1)
    p2 = np.asarray(pattern2)

    if len(p1) != len(p2):
        return 0.0

    if len(p1) == 0:
        return 1.0

    matches = int(np.sum(p1 == p2))
    return float(matches / len(p1))


__all__ = [
    # RE-PAT-002: Multi-Pattern Search
    "AhoCorasickMatcher",
    # RE-PAT-001: Binary Regex Pattern Matching
    "BinaryRegex",
    "CandidateSignature",
    "ClusterResult",
    "ClusteringResult",
    "FuzzyMatchResult",
    # RE-PAT-003: Fuzzy Pattern Matching
    "FuzzyMatcher",
    # RE-PAT-004: Pattern Learning and Discovery
    "LearnedPattern",
    "NgramModel",
    "NgramResult",
    "PatternLearner",
    "PatternMatchResult",
    "PeriodResult",
    "PeriodicPatternDetector",
    "RepeatingSequence",
    "SignatureDiscovery",
    "StructureHypothesis",
    "analyze_cluster",
    "binary_regex_search",
    "cluster_by_edit_distance",
    "cluster_by_hamming",
    "cluster_hierarchical",
    "cluster_patterns",
    "compute_distance_matrix",
    "count_pattern_occurrences",
    # Motif detection (compatibility)
    "detect_anomalies",
    "detect_period",
    "detect_period_autocorr",
    "detect_period_fft",
    "detect_periods_autocorr",
    "detect_periods_fft",
    "discover_signatures",
    "extract_motif",
    "find_approximate_repeats",
    "find_delimiter_candidates",
    "find_frequent_ngrams",
    "find_header_candidates",
    "find_longest_repeat",
    "find_motifs",
    "find_pattern_positions",
    "find_recurring_structures",
    "find_repeating_sequences",
    "find_similar_sequences",
    "fuzzy_search",
    "infer_structure",
    "learn_patterns_from_data",
    "multi_pattern_search",
    "pattern_similarity",
    "validate_period",
]
