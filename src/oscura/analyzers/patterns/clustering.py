"""Pattern clustering by similarity.

This module implements algorithms for clustering similar patterns/messages
using various distance metrics and clustering approaches.


Author: Oscura Development Team
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray


def cluster_messages(
    data: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    n_clusters: int = 3,
    method: str = "kmeans",
    random_state: int | None = None,
) -> np.ndarray[tuple[int], np.dtype[np.int_]]:
    """Cluster data points using K-means algorithm.

    Groups data points into n_clusters clusters using K-means clustering.
    Supports deterministic clustering with random_state.

    Args:
        data: Data points as (n_points, dimensions) array
        n_clusters: Number of clusters to create
        method: Clustering method (default: 'kmeans')
        random_state: Random seed for deterministic results

    Returns:
        Array of cluster labels (one per data point), in range [0, n_clusters)

    Raises:
        ValueError: If n_clusters is invalid or data shape is incorrect

    Examples:
        >>> data = np.random.randn(20, 2)
        >>> labels = cluster_messages(data, n_clusters=3, random_state=42)
        >>> assert len(labels) == 20
        >>> assert np.all((labels >= 0) & (labels < 3))
    """
    if data.ndim != 2:
        raise ValueError(f"Expected 2D data array, got shape {data.shape}")

    if n_clusters < 1:
        raise ValueError(f"n_clusters must be >= 1, got {n_clusters}")

    n_points = data.shape[0]
    if n_clusters > n_points:
        raise ValueError(f"n_clusters ({n_clusters}) cannot exceed n_points ({n_points})")

    # Use K-means clustering
    return _kmeans_clustering(data, n_clusters=n_clusters, random_state=random_state)


def _kmeans_clustering(
    data: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    n_clusters: int,
    random_state: int | None = None,
    max_iterations: int = 100,
    tolerance: float = 1e-4,
) -> np.ndarray[tuple[int], np.dtype[np.int_]]:
    """K-means clustering implementation.

    Args:
        data: Input data points (n_points, dimensions)
        n_clusters: Number of clusters
        random_state: Random seed
        max_iterations: Maximum iterations
        tolerance: Convergence tolerance

    Returns:
        Cluster labels for each point
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_points = data.shape[0]

    # Initialize centroids randomly from data points
    initial_indices = np.random.choice(n_points, size=n_clusters, replace=False)
    centroids = data[initial_indices].copy()

    labels = np.zeros(n_points, dtype=int)

    for _iteration in range(max_iterations):
        # Assign points to nearest centroid
        distances = np.zeros((n_points, n_clusters))
        for k in range(n_clusters):
            distances[:, k] = np.linalg.norm(data - centroids[k], axis=1)

        new_labels = np.argmin(distances, axis=1)

        # Check for convergence
        if np.array_equal(new_labels, labels):
            break

        labels = new_labels

        # Update centroids
        for k in range(n_clusters):
            cluster_points = data[labels == k]
            if len(cluster_points) > 0:
                centroids[k] = np.mean(cluster_points, axis=0)

    return labels


@dataclass
class ClusterResult:
    """Result of pattern clustering.

    Attributes:
        cluster_id: Unique cluster identifier
        patterns: List of patterns in this cluster
        centroid: Representative pattern (centroid)
        size: Number of patterns in cluster
        variance: Within-cluster variance
        common_bytes: Byte positions that are constant across all patterns
        variable_bytes: Byte positions that vary across patterns
    """

    cluster_id: int
    patterns: list[bytes | np.ndarray[tuple[int], np.dtype[np.uint8]]]
    centroid: bytes | np.ndarray[tuple[int], np.dtype[np.uint8]]
    size: int
    variance: float
    common_bytes: list[int]
    variable_bytes: list[int]

    def __post_init__(self) -> None:
        """Validate cluster result."""
        if self.cluster_id < 0:
            raise ValueError("cluster_id must be non-negative")
        if self.size < 0:
            raise ValueError("size must be non-negative")
        if len(self.patterns) != self.size:
            raise ValueError("patterns length must match size")


@dataclass
class ClusteringResult:
    """Complete clustering result.

    Attributes:
        clusters: List of ClusterResult objects
        labels: Cluster assignment for each input pattern
        num_clusters: Total number of clusters
        silhouette_score: Clustering quality metric (-1 to 1, higher = better)
    """

    clusters: list[ClusterResult]
    labels: np.ndarray[tuple[int], np.dtype[np.int_]]
    num_clusters: int
    silhouette_score: float

    def __post_init__(self) -> None:
        """Validate clustering result."""
        if self.num_clusters != len(self.clusters):
            raise ValueError("num_clusters must match clusters length")


def cluster_by_hamming(
    patterns: list[bytes | np.ndarray[tuple[int], np.dtype[np.uint8]]],
    threshold: float = 0.2,
    min_cluster_size: int = 2,
) -> ClusteringResult:
    """Cluster fixed-length patterns by Hamming distance.

    : Hamming distance clustering

    Groups patterns that differ by at most threshold * pattern_length bits.
    Efficient for fixed-length binary patterns.

    Args:
        patterns: List of patterns (all must have same length)
        threshold: Maximum normalized Hamming distance within cluster (0-1)
        min_cluster_size: Minimum patterns per cluster

    Returns:
        ClusteringResult with cluster assignments

    Raises:
        ValueError: If patterns have different lengths or invalid parameters

    Examples:
        >>> patterns = [b"ABCD", b"ABCE", b"ABCF", b"XYZA"]
        >>> result = cluster_by_hamming(patterns, threshold=0.3)
        >>> assert result.num_clusters >= 1
    """
    if not patterns:
        return ClusteringResult(
            clusters=[], labels=np.array([]), num_clusters=0, silhouette_score=0.0
        )

    # Validate all patterns have same length
    pattern_length = len(patterns[0])
    for i, p in enumerate(patterns):
        if len(p) != pattern_length:
            raise ValueError(f"Pattern {i} has length {len(p)}, expected {pattern_length}")

    # Convert to numpy arrays for efficient computation
    pattern_arrays = [_to_array(p) for p in patterns]
    n = len(pattern_arrays)

    # Compute distance matrix
    dist_matrix = compute_distance_matrix(patterns, metric="hamming")

    # Perform clustering using simple threshold-based approach
    labels, num_clusters = _perform_threshold_clustering(
        dist_matrix, n, threshold, min_cluster_size
    )

    # Build cluster results
    clusters = _build_cluster_results(num_clusters, labels, patterns, pattern_arrays, dist_matrix)

    # Compute silhouette score
    silhouette = _compute_silhouette_score(dist_matrix, labels) if num_clusters > 1 else 0.0

    return ClusteringResult(
        clusters=clusters, labels=labels, num_clusters=num_clusters, silhouette_score=silhouette
    )


def cluster_by_edit_distance(
    patterns: list[bytes | np.ndarray[tuple[int], np.dtype[np.uint8]]],
    threshold: float = 0.3,
    min_cluster_size: int = 2,
) -> ClusteringResult:
    """Cluster variable-length patterns by edit distance.

    : Edit distance (Levenshtein) clustering

    Groups patterns with normalized edit distance <= threshold.
    Works with variable-length patterns.

    Args:
        patterns: List of patterns (can have different lengths)
        threshold: Maximum normalized edit distance (0-1)
        min_cluster_size: Minimum patterns per cluster

    Returns:
        ClusteringResult with cluster assignments

    Examples:
        >>> patterns = [b"ABCD", b"ABCDE", b"ABCDF", b"XYZ"]
        >>> result = cluster_by_edit_distance(patterns, threshold=0.4)
    """
    if not patterns:
        return ClusteringResult(
            clusters=[], labels=np.array([]), num_clusters=0, silhouette_score=0.0
        )

    dist_matrix = compute_distance_matrix(patterns, metric="levenshtein")
    labels, num_clusters = _cluster_by_threshold(
        len(patterns), dist_matrix, threshold, min_cluster_size
    )

    clusters = _build_edit_clusters(patterns, labels, num_clusters, dist_matrix)
    silhouette = _compute_silhouette_score(dist_matrix, labels) if num_clusters > 1 else 0.0

    return ClusteringResult(
        clusters=clusters, labels=labels, num_clusters=num_clusters, silhouette_score=silhouette
    )


def _cluster_by_threshold(
    n: int, dist_matrix: NDArray[np.float64], threshold: float, min_cluster_size: int
) -> tuple[NDArray[np.int_], int]:
    """Perform threshold-based clustering."""
    labels = np.full(n, -1, dtype=int)
    cluster_id = 0

    for i in range(n):
        if labels[i] != -1:
            continue

        cluster_members = [i]
        labels[i] = cluster_id

        # Find similar patterns
        for j in range(i + 1, n):
            if labels[j] == -1 and max(dist_matrix[j, m] for m in cluster_members) <= threshold:
                cluster_members.append(j)
                labels[j] = cluster_id

        # Keep cluster if large enough
        if len(cluster_members) < min_cluster_size:
            for m in cluster_members:
                labels[m] = -1
        else:
            cluster_id += 1

    return labels, cluster_id


def _build_edit_clusters(
    patterns: list[bytes | np.ndarray[tuple[int], np.dtype[np.uint8]]],
    labels: NDArray[np.int_],
    num_clusters: int,
    dist_matrix: NDArray[np.float64],
) -> list[ClusterResult]:
    """Build cluster results from labels."""
    clusters = []
    for cid in range(num_clusters):
        cluster_indices = np.where(labels == cid)[0]
        cluster_patterns = [patterns[i] for i in cluster_indices]

        centroid = _compute_centroid_edit(cluster_patterns)

        # Pad and analyze variance
        max_len = max(len(p) for p in cluster_patterns)
        padded = [_to_array(p, target_length=max_len) for p in cluster_patterns]
        common, variable = _analyze_pattern_variance(padded)

        variance = (
            np.mean([dist_matrix[i, j] for i in cluster_indices for j in cluster_indices if i < j])
            if len(cluster_indices) > 1
            else 0.0
        )

        clusters.append(
            ClusterResult(
                cluster_id=cid,
                patterns=cluster_patterns,
                centroid=centroid,
                size=len(cluster_patterns),
                variance=float(variance),
                common_bytes=common,
                variable_bytes=variable,
            )
        )

    return clusters


def cluster_hierarchical(
    patterns: list[bytes | np.ndarray[tuple[int], np.dtype[np.uint8]]],
    method: Literal["single", "complete", "average", "upgma"] = "upgma",
    num_clusters: int | None = None,
    distance_threshold: float | None = None,
) -> ClusteringResult:
    """Hierarchical clustering of patterns.

    : Hierarchical clustering (UPGMA, etc.)

    Uses agglomerative hierarchical clustering with various linkage methods.

    Args:
        patterns: List of patterns
        method: Linkage method ('single', 'complete', 'average', 'upgma')
        num_clusters: Desired number of clusters (if None, use distance_threshold)
        distance_threshold: Distance threshold for cutting dendrogram

    Returns:
        ClusteringResult with cluster assignments

    Raises:
        ValueError: If neither num_clusters nor distance_threshold is specified

    Examples:
        >>> patterns = [b"AAA", b"AAB", b"BBB", b"BBC"]
        >>> result = cluster_hierarchical(patterns, method='average', num_clusters=2)
    """
    if num_clusters is None and distance_threshold is None:
        raise ValueError("Must specify either num_clusters or distance_threshold")

    if not patterns:
        return ClusteringResult(
            clusters=[], labels=np.array([]), num_clusters=0, silhouette_score=0.0
        )

    # Normalize method and compute distance matrix
    method = "average" if method == "upgma" else method
    dist_matrix = compute_distance_matrix(patterns, metric="hamming")

    # Perform clustering
    labels = _hierarchical_clustering(
        dist_matrix, method=method, num_clusters=num_clusters, distance_threshold=distance_threshold
    )

    # Build clusters
    unique_labels = set(labels[labels >= 0])
    clusters = _build_hierarchical_clusters(patterns, labels, unique_labels, dist_matrix)

    # Compute silhouette
    silhouette = _compute_silhouette_score(dist_matrix, labels) if len(unique_labels) > 1 else 0.0

    return ClusteringResult(
        clusters=clusters,
        labels=labels,
        num_clusters=len(unique_labels),
        silhouette_score=silhouette,
    )


def _build_hierarchical_clusters(
    patterns: list[bytes | np.ndarray[tuple[int], np.dtype[np.uint8]]],
    labels: NDArray[np.int_],
    unique_labels: set[int],
    dist_matrix: NDArray[np.float64],
) -> list[ClusterResult]:
    """Build cluster results from hierarchical clustering labels."""
    clusters = []
    for cid in sorted(unique_labels):
        cluster_indices = np.where(labels == cid)[0]
        cluster_patterns = [patterns[i] for i in cluster_indices]

        # Compute centroid based on pattern type
        pattern_arrays = [_to_array(p) for p in cluster_patterns]
        if len({len(p) for p in pattern_arrays}) == 1:
            centroid_array = _compute_centroid_hamming(pattern_arrays)
            centroid = bytes(centroid_array) if isinstance(patterns[0], bytes) else centroid_array
        else:
            centroid = _compute_centroid_edit(cluster_patterns)

        # Analyze variance
        max_len = max(len(p) for p in pattern_arrays)
        padded = [_to_array(p, target_length=max_len) for p in pattern_arrays]
        common, variable = _analyze_pattern_variance(padded)

        variance = (
            np.mean([dist_matrix[i, j] for i in cluster_indices for j in cluster_indices if i < j])
            if len(cluster_indices) > 1
            else 0.0
        )

        clusters.append(
            ClusterResult(
                cluster_id=cid,
                patterns=cluster_patterns,
                centroid=centroid,
                size=len(cluster_patterns),
                variance=float(variance),
                common_bytes=common,
                variable_bytes=variable,
            )
        )

    return clusters


def analyze_cluster(cluster: ClusterResult) -> dict[str, list[int] | list[float] | bytes]:
    """Analyze cluster to find common vs variable regions.

    : Cluster analysis

    Performs detailed analysis of a cluster to identify byte positions
    that are constant vs. those that vary.

    Args:
        cluster: ClusterResult to analyze

    Returns:
        Dictionary with analysis results including:
        - common_bytes: List of byte positions that are constant
        - variable_bytes: List of byte positions that vary
        - entropy_per_byte: Entropy at each byte position
        - consensus: Consensus pattern with variable bytes marked

    Examples:
        >>> # Assume we have a cluster
        >>> analysis = analyze_cluster(cluster)
        >>> print(f"Common positions: {analysis['common_bytes']}")
    """
    if cluster.size == 0:
        return {"common_bytes": [], "variable_bytes": [], "entropy_per_byte": [], "consensus": b""}

    # Convert patterns to arrays
    pattern_arrays = [_to_array(p) for p in cluster.patterns]

    # Pad to same length
    max_len = max(len(p) for p in pattern_arrays)
    padded = [_to_array(p, target_length=max_len) for p in pattern_arrays]

    # Compute entropy per byte position
    entropy_per_byte = []
    for pos in range(max_len):
        byte_values = [p[pos] for p in padded]
        entropy = _compute_byte_entropy(byte_values)
        entropy_per_byte.append(entropy)

    # Threshold for "common" (low entropy)
    common_threshold = 0.1
    common_bytes = [i for i, e in enumerate(entropy_per_byte) if e < common_threshold]
    variable_bytes = [i for i, e in enumerate(entropy_per_byte) if e >= common_threshold]

    # Build consensus pattern
    consensus = np.zeros(max_len, dtype=np.uint8)
    for pos in range(max_len):
        byte_values = [p[pos] for p in padded]
        # Use most common byte
        consensus[pos] = max(set(byte_values), key=byte_values.count)

    return {
        "common_bytes": common_bytes,
        "variable_bytes": variable_bytes,
        "entropy_per_byte": entropy_per_byte,
        "consensus": bytes(consensus),
    }


def compute_distance_matrix(
    patterns: list[bytes | np.ndarray[tuple[int], np.dtype[np.uint8]]],
    metric: Literal["hamming", "levenshtein", "jaccard"] = "hamming",
) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
    """Compute pairwise distance matrix.

    : Distance matrix computation

    Computes all pairwise distances between patterns using the specified metric.

    Args:
        patterns: List of patterns
        metric: Distance metric ('hamming', 'levenshtein', 'jaccard')

    Returns:
        Symmetric distance matrix (n x n)

    Raises:
        ValueError: If unknown metric is specified

    Examples:
        >>> patterns = [b"ABC", b"ABD", b"XYZ"]
        >>> dist = compute_distance_matrix(patterns, metric='hamming')
        >>> assert dist.shape == (3, 3)
    """
    n = len(patterns)
    dist_matrix = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(i + 1, n):
            if metric == "hamming":
                dist = _hamming_distance(patterns[i], patterns[j])
            elif metric == "levenshtein":
                dist = _edit_distance(patterns[i], patterns[j])
            elif metric == "jaccard":
                dist = _jaccard_distance(patterns[i], patterns[j])
            else:
                raise ValueError(f"Unknown metric: {metric}")

            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist

    return dist_matrix


# Helper functions


def _to_array(
    data: bytes | np.ndarray[tuple[int], np.dtype[np.uint8]] | memoryview | bytearray,
    target_length: int | None = None,
) -> np.ndarray[tuple[int], np.dtype[np.uint8]]:
    """Convert to numpy array, optionally padding to target length.

    Args:
        data: Input data (bytes, bytearray, memoryview, or numpy array)
        target_length: If specified, pad to this length

    Returns:
        Numpy array of uint8

    Raises:
        TypeError: If data type is not supported
    """
    if isinstance(data, bytes):
        arr = np.frombuffer(data, dtype=np.uint8)
    elif isinstance(data, bytearray | memoryview):
        arr = np.frombuffer(bytes(data), dtype=np.uint8)
    elif isinstance(data, np.ndarray):
        arr = data.astype(np.uint8)
    else:
        raise TypeError(f"Unsupported type: {type(data)}")

    if target_length is not None and len(arr) < target_length:
        # Pad with zeros
        padded = np.zeros(target_length, dtype=np.uint8)
        padded[: len(arr)] = arr
        return padded

    return arr


def _hamming_distance(
    a: bytes | np.ndarray[tuple[int], np.dtype[np.uint8]],
    b: bytes | np.ndarray[tuple[int], np.dtype[np.uint8]],
) -> float:
    """Compute normalized Hamming distance."""
    arr_a = _to_array(a)
    arr_b = _to_array(b)

    if len(arr_a) != len(arr_b):
        # Pad shorter to match longer
        max_len = max(len(arr_a), len(arr_b))
        arr_a = _to_array(a, target_length=max_len)
        arr_b = _to_array(b, target_length=max_len)

    # Count differences
    differences = np.sum(arr_a != arr_b)
    return float(differences) / len(arr_a)


def _edit_distance(
    a: bytes | np.ndarray[tuple[int], np.dtype[np.uint8]],
    b: bytes | np.ndarray[tuple[int], np.dtype[np.uint8]],
) -> float:
    """Compute normalized Levenshtein edit distance."""
    bytes_a = bytes(a) if isinstance(a, np.ndarray) else a
    bytes_b = bytes(b) if isinstance(b, np.ndarray) else b

    m, n = len(bytes_a), len(bytes_b)

    if m == 0 and n == 0:
        return 0.0
    if m == 0:
        return 1.0
    if n == 0:
        return 1.0

    # DP table
    prev_row = list(range(n + 1))
    curr_row = [0] * (n + 1)

    for i in range(1, m + 1):
        curr_row[0] = i
        for j in range(1, n + 1):
            if bytes_a[i - 1] == bytes_b[j - 1]:
                curr_row[j] = prev_row[j - 1]
            else:
                curr_row[j] = 1 + min(prev_row[j], curr_row[j - 1], prev_row[j - 1])
        prev_row, curr_row = curr_row, prev_row

    # Normalize by max length
    return prev_row[n] / max(m, n)


def _jaccard_distance(
    a: bytes | np.ndarray[tuple[int], np.dtype[np.uint8]],
    b: bytes | np.ndarray[tuple[int], np.dtype[np.uint8]],
) -> float:
    """Compute Jaccard distance based on byte sets."""
    set_a = set(_to_array(a))
    set_b = set(_to_array(b))

    if len(set_a) == 0 and len(set_b) == 0:
        return 0.0

    intersection = len(set_a & set_b)
    union = len(set_a | set_b)

    if union == 0:
        return 0.0

    # Jaccard distance = 1 - Jaccard similarity
    return 1.0 - (intersection / union)


def _perform_threshold_clustering(
    dist_matrix: NDArray[np.float64],
    n: int,
    threshold: float,
    min_cluster_size: int,
) -> tuple[NDArray[np.int_], int]:
    """Perform threshold-based clustering on distance matrix.

    Args:
        dist_matrix: Pairwise distance matrix.
        n: Number of patterns.
        threshold: Maximum distance within cluster.
        min_cluster_size: Minimum patterns per cluster.

    Returns:
        Tuple of (labels, num_clusters).
    """
    labels = np.full(n, -1, dtype=int)
    cluster_id = 0

    for i in range(n):
        if labels[i] != -1:
            continue  # Already assigned

        # Start new cluster
        cluster_members = [i]
        labels[i] = cluster_id

        # Find all patterns within threshold
        for j in range(i + 1, n):
            if labels[j] != -1:
                continue

            # Check if j is close to all members of current cluster
            max_dist = max(dist_matrix[j, m] for m in cluster_members)
            if max_dist <= threshold:
                cluster_members.append(j)
                labels[j] = cluster_id

        # Only keep cluster if large enough
        if len(cluster_members) < min_cluster_size:
            for m in cluster_members:
                labels[m] = -1
        else:
            cluster_id += 1

    return labels, cluster_id


def _build_cluster_results(
    num_clusters: int,
    labels: NDArray[np.int_],
    patterns: list[bytes | NDArray[Any]],
    pattern_arrays: list[NDArray[Any]],
    dist_matrix: NDArray[np.float64],
) -> list[ClusterResult]:
    """Build ClusterResult objects from clustering labels.

    Args:
        num_clusters: Number of clusters found.
        labels: Cluster labels for each pattern.
        patterns: Original patterns (bytes or arrays).
        pattern_arrays: Patterns as numpy arrays.
        dist_matrix: Pairwise distance matrix.

    Returns:
        List of ClusterResult objects.
    """
    clusters = []
    for cid in range(num_clusters):
        cluster_indices = np.where(labels == cid)[0]
        cluster_patterns = [patterns[i] for i in cluster_indices]

        # Compute centroid (majority vote per byte)
        centroid = _compute_centroid_hamming([pattern_arrays[i] for i in cluster_indices])

        # Analyze common vs variable bytes
        common, variable = _analyze_pattern_variance([pattern_arrays[i] for i in cluster_indices])

        # Compute within-cluster variance
        variance = (
            np.mean([dist_matrix[i, j] for i in cluster_indices for j in cluster_indices if i < j])
            if len(cluster_indices) > 1
            else 0.0
        )

        clusters.append(
            ClusterResult(
                cluster_id=cid,
                patterns=cluster_patterns,
                centroid=bytes(centroid) if isinstance(patterns[0], bytes) else centroid,
                size=len(cluster_patterns),
                variance=float(variance),
                common_bytes=common,
                variable_bytes=variable,
            )
        )

    return clusters


def _compute_centroid_hamming(
    patterns: list[np.ndarray[tuple[int], np.dtype[np.uint8]]],
) -> np.ndarray[tuple[int], np.dtype[np.uint8]]:
    """Compute centroid using majority vote (for fixed-length patterns)."""
    if not patterns:
        return np.array([], dtype=np.uint8)

    _n = len(patterns)
    length = len(patterns[0])

    centroid = np.zeros(length, dtype=np.uint8)
    for pos in range(length):
        bytes_at_pos = [p[pos] for p in patterns]
        # Most common byte
        centroid[pos] = max(set(bytes_at_pos), key=bytes_at_pos.count)

    return centroid


def _compute_centroid_edit(
    patterns: list[bytes | np.ndarray[tuple[int], np.dtype[np.uint8]]],
) -> bytes | np.ndarray[tuple[int], np.dtype[np.uint8]]:
    """Compute centroid for variable-length patterns (most central pattern)."""
    if not patterns:
        return b"" if isinstance(patterns[0], bytes) else np.array([])

    # Use most common pattern as centroid
    from collections import Counter

    pattern_counts = Counter(bytes(p) if isinstance(p, np.ndarray) else p for p in patterns)
    most_common = pattern_counts.most_common(1)[0][0]

    # Return in original type
    if isinstance(patterns[0], bytes):
        return most_common
    else:
        return np.frombuffer(most_common, dtype=np.uint8)


def _analyze_pattern_variance(
    patterns: list[np.ndarray[tuple[int], np.dtype[np.uint8]]],
) -> tuple[list[int], list[int]]:
    """Analyze which byte positions are common vs variable."""
    if not patterns or len(patterns) == 0:
        return [], []

    length = len(patterns[0])
    common_bytes = []
    variable_bytes = []

    for pos in range(length):
        bytes_at_pos = [p[pos] for p in patterns]
        unique_values = len(set(bytes_at_pos))

        if unique_values == 1:
            common_bytes.append(pos)
        else:
            variable_bytes.append(pos)

    return common_bytes, variable_bytes


def _compute_byte_entropy(byte_values: list[int]) -> float:
    """Compute Shannon entropy of byte values."""
    if not byte_values:
        return 0.0

    from collections import Counter

    counts = Counter(byte_values)
    n = len(byte_values)

    entropy = 0.0
    for count in counts.values():
        if count > 0:
            prob = count / n
            entropy -= prob * np.log2(prob)

    return entropy


def _compute_silhouette_score(
    dist_matrix: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    labels: np.ndarray[tuple[int], np.dtype[np.int_]],
) -> float:
    """Compute average silhouette score for clustering quality."""
    n = len(labels)
    if n <= 1:
        return 0.0

    # Filter out noise points (-1 labels)
    valid_mask = labels >= 0
    if np.sum(valid_mask) <= 1:
        return 0.0

    unique_labels = set(labels[valid_mask])
    if len(unique_labels) <= 1:
        return 0.0

    silhouette_scores = []

    for i in range(n):
        if labels[i] == -1:
            continue

        # a(i): average distance to points in same cluster
        same_cluster = (labels == labels[i]) & (np.arange(n) != i)
        if np.sum(same_cluster) == 0:
            continue

        a_i = np.mean(dist_matrix[i, same_cluster])

        # b(i): minimum average distance to points in other clusters
        b_i = float("inf")
        for other_label in unique_labels:
            if other_label == labels[i]:
                continue

            other_cluster = labels == other_label
            if np.sum(other_cluster) > 0:
                avg_dist = np.mean(dist_matrix[i, other_cluster])
                b_i = min(b_i, avg_dist)

        # Silhouette coefficient
        if b_i == float("inf"):
            s_i = 0.0
        else:
            s_i = (b_i - a_i) / max(a_i, b_i)

        silhouette_scores.append(s_i)

    return float(np.mean(silhouette_scores)) if silhouette_scores else 0.0


def _hierarchical_clustering(
    dist_matrix: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    method: str,
    num_clusters: int | None,
    distance_threshold: float | None,
) -> np.ndarray[tuple[int], np.dtype[np.int_]]:
    """Perform agglomerative hierarchical clustering."""
    MAX_ITERATIONS = 10000  # Prevent infinite loops in malformed distance matrices

    n = dist_matrix.shape[0]

    # Initialize: each point is its own cluster
    clusters = [[i] for i in range(n)]
    _cluster_distances = dist_matrix.copy()

    # Merge until desired number of clusters
    iteration_count = 0
    while len(clusters) > 1:
        iteration_count += 1
        if iteration_count > MAX_ITERATIONS:
            raise RuntimeError(
                f"Hierarchical clustering exceeded maximum iterations ({MAX_ITERATIONS}). "
                "This may indicate a malformed distance matrix or insufficient convergence criteria."
            )

        if num_clusters is not None and len(clusters) <= num_clusters:
            break

        # Find closest pair of clusters
        min_dist = float("inf")
        merge_i, merge_j = -1, -1

        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                # Compute inter-cluster distance
                dist = _linkage_distance(clusters[i], clusters[j], dist_matrix, method)

                if dist < min_dist:
                    min_dist = dist
                    merge_i, merge_j = i, j

        # Check distance threshold
        if distance_threshold is not None and min_dist > distance_threshold:
            break

        # Merge clusters
        if merge_i >= 0 and merge_j >= 0:
            clusters[merge_i].extend(clusters[merge_j])
            del clusters[merge_j]

    # Assign labels
    labels = np.full(n, -1, dtype=int)
    for cid, cluster in enumerate(clusters):
        for idx in cluster:
            labels[idx] = cid

    return labels


def _linkage_distance(
    cluster_a: list[int],
    cluster_b: list[int],
    dist_matrix: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    method: str,
) -> float:
    """Compute distance between two clusters using linkage method."""
    distances = [dist_matrix[i, j] for i in cluster_a for j in cluster_b]

    if not distances:
        return 0.0

    if method == "single":
        return float(min(distances))
    elif method == "complete":
        return float(max(distances))
    elif method == "average":
        return float(np.mean(distances))
    else:
        return float(np.mean(distances))  # Default to average


class PatternClusterer:
    """Object-oriented wrapper for pattern clustering functionality.

    Provides a class-based interface for clustering operations,
    wrapping the functional API for consistency with test expectations.



    Example:
        >>> clusterer = PatternClusterer(n_clusters=3)
        >>> labels = clusterer.cluster(messages)
    """

    def __init__(
        self,
        n_clusters: int = 3,
        method: Literal["hamming", "edit", "hierarchical"] = "hamming",
        distance_metric: Literal["hamming", "levenshtein", "jaccard"] = "hamming",
        threshold: float = 0.3,
        min_cluster_size: int = 2,
    ):
        """Initialize pattern clusterer.

        Args:
            n_clusters: Desired number of clusters.
            method: Clustering method ('hamming', 'edit', or 'hierarchical').
            distance_metric: Distance metric to use.
            threshold: Distance threshold for clustering.
            min_cluster_size: Minimum patterns per cluster.
        """
        self.n_clusters = n_clusters
        self.method = method
        self.distance_metric = distance_metric
        self.threshold = threshold
        self.min_cluster_size = min_cluster_size
        self.result_: ClusteringResult | None = None

    def cluster(
        self, patterns: list[bytes | np.ndarray[tuple[int], np.dtype[np.uint8]]]
    ) -> np.ndarray[tuple[int], np.dtype[np.int_]]:
        """Cluster patterns and return labels.

        Args:
            patterns: List of patterns to cluster.

        Returns:
            Array of cluster labels (one per pattern).

        Example:
            >>> clusterer = PatternClusterer(n_clusters=3)
            >>> labels = clusterer.cluster(messages)
        """
        if self.method == "hamming":
            self.result_ = cluster_by_hamming(
                patterns, threshold=self.threshold, min_cluster_size=self.min_cluster_size
            )
        elif self.method == "edit":
            self.result_ = cluster_by_edit_distance(
                patterns, threshold=self.threshold, min_cluster_size=self.min_cluster_size
            )
        else:  # hierarchical or default
            self.result_ = cluster_hierarchical(
                patterns, method="average", num_clusters=self.n_clusters
            )

        return self.result_.labels

    def fit(
        self, patterns: list[bytes | np.ndarray[tuple[int], np.dtype[np.uint8]]]
    ) -> PatternClusterer:
        """Fit the clusterer to patterns (sklearn-style interface).

        Args:
            patterns: List of patterns to cluster.

        Returns:
            Self (for method chaining).
        """
        self.cluster(patterns)
        return self

    def fit_predict(
        self, patterns: list[bytes | np.ndarray[tuple[int], np.dtype[np.uint8]]]
    ) -> np.ndarray[tuple[int], np.dtype[np.int_]]:
        """Fit and return cluster labels (sklearn-style interface).

        Args:
            patterns: List of patterns to cluster.

        Returns:
            Array of cluster labels.
        """
        return self.cluster(patterns)

    def get_clusters(self) -> list[ClusterResult]:
        """Get detailed cluster results.

        Returns:
            List of ClusterResult objects with full cluster analysis.

        Raises:
            ValueError: If cluster() hasn't been called yet.
        """
        if self.result_ is None:
            raise ValueError("Must call cluster() before get_clusters()")
        return self.result_.clusters

    def get_silhouette_score(self) -> float:
        """Get silhouette score for clustering quality.

        Returns:
            Silhouette score (-1 to 1, higher is better).

        Raises:
            ValueError: If cluster() hasn't been called yet.
        """
        if self.result_ is None:
            raise ValueError("Must call cluster() before get_silhouette_score()")
        return self.result_.silhouette_score
